import os
import warnings
import re
import pandas as pd
import torch
import torch.distributed.tensor  # noqa: F401 — PEFT uses torch.distributed.tensor.DTensor before lazy bind
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

# ----------------------------
# (Optional) silence HF hub FutureWarning about resume_download
# ----------------------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*resume_download.*deprecated.*"
)

# ----------------------------
# Paths
# ----------------------------
FASTA_PATH_TRAIN = "/home/alanwu/Documents/iggen_model/data/training data/h3_split_incremental/sabdab_plus_oas_4813/train.fasta"
FASTA_PATH_VAL = "/home/alanwu/Documents/iggen_model/data/training data/h3_split_incremental/sabdab_plus_oas_4813/val.fasta"

SPANS_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"
RMSD_XLSX = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"

# OAS H3 span on full Fv: exact CDRH3 AA from these cdr_loops.fasta files, unique substring in Fv.
OAS_CDR_FASTAS = [
    "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1279065_1/cdr_loops.fasta",
    "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1279073_1/cdr_loops.fasta",
    "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1287155_1/cdr_loops.fasta",
]

OUT_DIR = "/home/alanwu/Documents/iggen_model/piggen_lora_cdr_masked_oas_4813"
repo_id = "ollieturnbull/p-IgGen"

# ----------------------------
# Hyperparams
# ----------------------------
LEARNING_RATE = 1e-4
EVAL_STEPS = 40
PATIENCE = 3
MAX_EPOCHS = 80

SABDAB_H3_RMSD_MAX = 1.0
OAS_PLDDT_THRESHOLD = 80.0

# ----------------------------
# Loop span cols + RMSD cols (SAbDab)
# ----------------------------
LOOPS = ["H1", "H2", "H3", "L1", "L2", "L3"]
SPAN_COLS = {loop: (f"{loop}_start", f"{loop}_end") for loop in LOOPS}
RMSD_COLS = {loop: f"{loop}_RMSD" for loop in LOOPS}


def normalize_id(x: str) -> str:
    x = str(x).strip()
    x = x.replace(".pdb", "")
    x = x.split("|", 1)[0]
    return x.lower()


def normalize_header_id(x: str) -> str:
    return str(x).strip().split("|", 1)[0]


def read_full_fv_fasta(path: str) -> dict[str, str]:
    """
    Parses FASTA files where each record contains a full Fv sequence.

    Some records may still contain a VH:VL separator ':' in the sequence.
    We remove ':' here so sequence length matches token length for AA tokenization.

    Returns:
      dict: header_id -> full_fv_sequence_without_separator
    """
    out = {}
    cur_id, cur_seq = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seq = "".join(cur_seq).upper().replace(":", "")
                    out[cur_id] = seq
                cur_id = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        seq = "".join(cur_seq).upper().replace(":", "")
        out[cur_id] = seq
    return out


def canonical_sabdab_id(x: str) -> str:
    """
    Canonical key for SAbDab header matching across sources.
    Supports both '9ds1' and '9ds1.pdb|VH:VL' style IDs.
    """
    base = normalize_header_id(x).strip().lower()
    if base.endswith(".pdb"):
        base = base[:-4]
    return base


def is_sabdab_header_id(header_id: str, sabdab_df_by_header: dict[str, pd.Series]) -> bool:
    return canonical_sabdab_id(header_id) in sabdab_df_by_header


def canonicalize_oas_key(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("__", "_")
    s = re.sub(r"_+", "_", s)
    return s


def parse_two_contigs(key: str):
    k = canonicalize_oas_key(key)
    m = re.match(r"^(.*?_contig_\d+)_(.*?_contig_\d+)$", k)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def read_oas_h3_sequences_from_cdr_fastas(paths: list[str]) -> dict[str, str]:
    """
    Reads FASTA records >pair|CDRH3 and returns canonicalize_oas_key(pair) -> H3 AA.
    Also stores reversed contig-order alias when parseable.
    """
    out = {}

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"OAS CDR fasta not found: {path}")

        cur_header = None
        cur_seq = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    if cur_header is not None:
                        _store_oas_h3_record(cur_header, "".join(cur_seq), out)
                    cur_header = line[1:].strip()
                    cur_seq = []
                else:
                    cur_seq.append(line)

        if cur_header is not None:
            _store_oas_h3_record(cur_header, "".join(cur_seq), out)

    print(f"Loaded OAS exact H3 loop sequences for {len(out)} canonical/reversed keys")
    return out


def _store_oas_h3_record(header: str, seq: str, out: dict[str, str]):
    if "|" not in header:
        return

    pair_id, loop_name = header.split("|", 1)
    loop_name = loop_name.strip().upper()
    if loop_name != "CDRH3":
        return

    h3_seq = seq.strip().upper().replace(":", "")
    if not h3_seq:
        return

    key = canonicalize_oas_key(pair_id)
    out[key] = h3_seq

    a, b = parse_two_contigs(key)
    if a is not None:
        out[f"{b}_{a}"] = h3_seq


def find_unique_subsequence_span(full_seq: str, sub_seq: str):
    """Returns (start, end_exclusive) if sub_seq occurs exactly once; else None."""
    full_seq = full_seq.upper().replace(":", "")
    sub_seq = sub_seq.upper().replace(":", "")

    if not sub_seq or len(sub_seq) > len(full_seq):
        return None

    first = full_seq.find(sub_seq)
    if first == -1:
        return None

    second = full_seq.find(sub_seq, first + 1)
    if second != -1:
        return None

    return first, first + len(sub_seq)


# ----------------------------
# Mask builders
# ----------------------------

def build_loss_mask_from_spans_h3_only(fv_len: int, row: pd.Series) -> torch.Tensor:
    """
    SAbDab H3-only:
      - mask=1 only on H3 span
    """
    m = torch.zeros(fv_len, dtype=torch.float32)

    s_col, e_col = SPAN_COLS["H3"]
    s = row.get(s_col, None)
    e = row.get(e_col, None)

    if pd.isna(s) or pd.isna(e):
        return m

    s = max(0, min(int(s), fv_len))
    e = max(0, min(int(e), fv_len))
    if e > s:
        m[s:e] = 1.0

    return m


def build_loss_mask_from_exact_span(fv_len: int, start: int, end: int) -> torch.Tensor:
    m = torch.zeros(fv_len, dtype=torch.float32)
    start = max(0, min(int(start), fv_len))
    end = max(0, min(int(end), fv_len))
    if end > start:
        m[start:end] = 1.0
    return m


# ----------------------------
# Dataset
# ----------------------------

class FvMaskedDataset(Dataset):
    """
    Uses already-filtered train/val FASTA files directly.

    For each example:
      - input = full Fv sequence
      - loss mask = H3 span only
      - SAbDab H3 spans come from loop_spans_from_pdb.csv + RMSD merge
      - OAS H3 spans: exact CDRH3 from cdr_loops.fasta, located once in the Fv sequence
    """
    def __init__(
        self,
        fasta_map: dict[str, str],
        sabdab_df_by_header: dict[str, pd.Series],
        oas_h3_lookup: dict[str, str],
        tokenizer,
        max_len=512,
        verbose=True,
    ):
        self.fasta_map = fasta_map
        self.sabdab_df_by_header = sabdab_df_by_header
        self.oas_h3_lookup = oas_h3_lookup
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.items = []
        self._oas_h3_span_by_header: dict[str, tuple[int, int]] = {}

        sab_total = sab_ok = sab_missing = 0
        oas_total = 0
        oas_ok = 0
        oas_missing = 0
        oas_no_h3 = oas_not_found = oas_ambig = 0

        for hid in self.fasta_map.keys():
            hid_str = str(hid)

            if is_sabdab_header_id(hid_str, self.sabdab_df_by_header):
                sab_total += 1
                base = canonical_sabdab_id(hid_str)
                row = self.sabdab_df_by_header.get(base, None)
                if row is None:
                    sab_missing += 1
                    continue

                s = row.get(SPAN_COLS["H3"][0], None)
                e = row.get(SPAN_COLS["H3"][1], None)
                if pd.isna(s) or pd.isna(e) or int(e) <= int(s):
                    sab_missing += 1
                    continue

                sab_ok += 1
                self.items.append(("sabdab", hid_str))

            else:
                oas_total += 1
                base = normalize_header_id(hid_str)
                key = canonicalize_oas_key(base)

                h3_seq = self.oas_h3_lookup.get(key, None)
                if h3_seq is None:
                    a, b = parse_two_contigs(key)
                    if a is not None:
                        h3_seq = self.oas_h3_lookup.get(f"{b}_{a}", None)

                if h3_seq is None:
                    oas_no_h3 += 1
                    oas_missing += 1
                    continue

                full_seq = self.fasta_map[hid_str].upper().replace(":", "")
                span = find_unique_subsequence_span(full_seq, h3_seq)

                if span is None:
                    if full_seq.count(h3_seq) == 0:
                        oas_not_found += 1
                    else:
                        oas_ambig += 1
                    oas_missing += 1
                    continue

                s, e = span
                self._oas_h3_span_by_header[hid_str] = (s, e)
                oas_ok += 1
                self.items.append(("oas", hid_str))

        if verbose:
            print("\n[FvMaskedDataset] Using prefiltered FASTA directly")
            print("  ---- SAbDab ----")
            print(f"  FASTA entries:              {sab_total}")
            print(f"  with usable H3 spans:       {sab_ok}")
            print(f"  missing metadata/spans:     {sab_missing}")
            print("  ---- OAS (CDR FASTA substring) ----")
            print(f"  FASTA entries:              {oas_total}")
            print(f"  with exact H3 matched:      {oas_ok}")
            print(f"  missing H3 loop record:     {oas_no_h3}")
            print(f"  H3 exact seq not found:     {oas_not_found}")
            print(f"  H3 exact seq ambiguous:     {oas_ambig}")
            print(f"  total OAS skipped:          {oas_missing}")
            print(f"  Final dataset items:        {len(self.items)}\n")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        source, header_id = self.items[idx]

        seq = self.fasta_map[header_id].upper()
        fv_len = len(seq)

        enc = self.tokenizer(
            seq,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        enc.pop("token_type_ids", None)

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        if source == "sabdab":
            base = canonical_sabdab_id(header_id)
            row = self.sabdab_df_by_header[base]

            fv_len_csv = int(row["Fv_len"])
            if fv_len_csv != fv_len:
                raise ValueError(
                    f"Length mismatch (SAbDab) {header_id}: len(sequence)={fv_len} vs Fv_len={fv_len_csv}"
                )

            loss_mask = build_loss_mask_from_spans_h3_only(
                fv_len=fv_len,
                row=row,
            )
        else:
            if header_id not in self._oas_h3_span_by_header:
                raise ValueError(f"No cached OAS H3 span for {header_id}")

            s, e = self._oas_h3_span_by_header[header_id]
            loss_mask = build_loss_mask_from_exact_span(fv_len=fv_len, start=s, end=e)

        if loss_mask.numel() != input_ids.numel():
            raise ValueError(
                f"Tokenizer/token mismatch for {header_id}: "
                f"seq_len={len(seq)} mask_len={loss_mask.numel()} token_len={input_ids.numel()} "
                f"example_seq_prefix={seq[:80]}"
            )

        if loss_mask.sum().item() < 1:
            raise ValueError(f"No supervised H3 tokens for {header_id}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
        }


# ----------------------------
# Collate
# ----------------------------

def collate(batch, pad_id: int):
    max_len = max(x["input_ids"].numel() for x in batch)

    def pad_1d(x, value, dtype=None):
        if dtype is None:
            dtype = x.dtype
        pad_len = max_len - x.numel()
        if pad_len <= 0:
            return x
        return torch.cat([x, torch.full((pad_len,), value, dtype=dtype)], dim=0)

    input_ids = torch.stack([pad_1d(x["input_ids"], pad_id) for x in batch])
    attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch])
    labels = torch.stack([pad_1d(x["labels"], -100) for x in batch])
    loss_mask = torch.stack([pad_1d(x["loss_mask"], 0.0, dtype=torch.float32) for x in batch])

    labels = labels.masked_fill(attention_mask == 0, -100)
    loss_mask = loss_mask.masked_fill(attention_mask == 0, 0.0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }


# ----------------------------
# Trainer with masked loss
# ----------------------------

class MaskedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_mask = inputs.pop("loss_mask")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Logits already NaN/Inf in forward pass.")

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        denom = shift_mask.sum()
        if denom.item() < 1:
            loss = token_loss.sum() * 0.0
        else:
            loss = (token_loss * shift_mask).sum() / denom

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss became NaN/Inf.")

        return (loss, outputs) if return_outputs else loss


def guess_lora_targets(model):
    present = set(name for name, _ in model.named_modules())
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "c_attn", "c_proj",
        "query_key_value", "dense",
        "Wqkv", "out_proj",
    ]
    leaf_names = set(n.split(".")[-1] for n in present)
    found = [c for c in candidates if c in leaf_names]
    return found if found else ["c_attn", "c_proj"]


# ----------------------------
# Plotting
# ----------------------------

def plot_loss_curves(trainer: Trainer, out_png: str, *, sabdab_thr: float, oas_thr: float, lr: float, eval_steps: int):
    hist = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps_list, eval_losses = [], []

    for entry in hist:
        if "loss" in entry and "step" in entry:
            if isinstance(entry["loss"], (int, float)):
                train_steps.append(int(entry["step"]))
                train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry and "step" in entry:
            eval_steps_list.append(int(entry["step"]))
            eval_losses.append(float(entry["eval_loss"]))

    if len(train_steps) == 0 and len(eval_steps_list) == 0:
        print("WARNING: No loss history found to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if train_steps:
        ax.plot(train_steps, train_losses, label="train_loss")
    if eval_steps_list:
        ax.plot(eval_steps_list, eval_losses, marker="o", label="eval_loss")

    ax.set_xlabel("steps (optimizer updates)")
    ax.set_ylabel("loss")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    epochs_ran = trainer.state.epoch if trainer.state.epoch is not None else float("nan")
    info = (
        f"Prefiltered split used directly\n"
        f"SAbDab H3 RMSD max (upstream): {sabdab_thr:.2f} Å\n"
        f"OAS H3 mean pLDDT thr (upstream): {oas_thr:.1f}\n"
        f"LR: {lr:g}\n"
        f"Eval interval: every {eval_steps} steps\n"
        f"Epochs run: {epochs_ran:.2f}"
    )

    ax.text(
        1.02, 0.5, info,
        transform=ax.transAxes,
        va="center", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("Saved loss curve:", out_png)


# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----------------------------
    # Load SAbDab spans + RMSD
    # ----------------------------
    spans = pd.read_csv(SPANS_CSV)
    spans["Antibody_ID"] = spans["Antibody_ID"].map(normalize_id)

    rmsd = pd.read_excel(RMSD_XLSX)
    rmsd["Antibody_ID"] = rmsd["Antibody_ID"].map(normalize_id)

    keep_cols = ["Antibody_ID"] + [RMSD_COLS[l] for l in LOOPS]
    rmsd = rmsd[keep_cols].copy()

    df_sabdab = spans.merge(rmsd, on="Antibody_ID", how="inner")
    if len(df_sabdab) == 0:
        raise RuntimeError("After merging spans with RMSD table, zero rows remain. Check Antibody_ID normalization.")

    sabdab_df_by_header = {}
    for _, row in df_sabdab.iterrows():
        hid = canonical_sabdab_id(row["fasta_header_id"])
        if hid not in sabdab_df_by_header:
            sabdab_df_by_header[hid] = row

    # ----------------------------
    # OAS: CDR FASTA -> exact H3 AA -> unique substring span in Fv
    # ----------------------------
    oas_h3_lookup = read_oas_h3_sequences_from_cdr_fastas(OAS_CDR_FASTAS)

    # ----------------------------
    # Load prefiltered split FASTA maps
    # ----------------------------
    fasta_train = read_full_fv_fasta(FASTA_PATH_TRAIN)
    fasta_val = read_full_fv_fasta(FASTA_PATH_VAL)

    train_ids = set(fasta_train.keys())
    val_ids = set(fasta_val.keys())

    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"Train/val overlap detected in prefiltered FASTA files: {len(overlap)} headers")

    # Report composition
    n_train_sab = sum(1 for k in train_ids if is_sabdab_header_id(k, sabdab_df_by_header))
    n_val_sab = sum(1 for k in val_ids if is_sabdab_header_id(k, sabdab_df_by_header))
    print(f"Train FASTA entries: {len(train_ids)} (SAbDab: {n_train_sab}, OAS: {len(train_ids)-n_train_sab})")
    print(f"Val FASTA entries:   {len(val_ids)} (SAbDab: {n_val_sab}, OAS: {len(val_ids)-n_val_sab})")

    # ----------------------------
    # Model + tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA
    targets = guess_lora_targets(model)
    print("LoRA target_modules found (FYI):", targets)

    lora_cfg = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----------------------------
    # Datasets
    # ----------------------------
    ds_train = FvMaskedDataset(
        fasta_map=fasta_train,
        sabdab_df_by_header=sabdab_df_by_header,
        oas_h3_lookup=oas_h3_lookup,
        tokenizer=tokenizer,
        max_len=512,
        verbose=True,
    )

    ds_val = FvMaskedDataset(
        fasta_map=fasta_val,
        sabdab_df_by_header=sabdab_df_by_header,
        oas_h3_lookup=oas_h3_lookup,
        tokenizer=tokenizer,
        max_len=512,
        verbose=True,
    )

    print(f"Train dataset items: {len(ds_train)} | Val dataset items: {len(ds_val)}")

    # ----------------------------
    # Training with validation + early stopping
    # ----------------------------
    args = TrainingArguments(
        output_dir=OUT_DIR,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,

        learning_rate=LEARNING_RATE,
        num_train_epochs=MAX_EPOCHS,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        warmup_ratio=0.05,
        report_to="none",

        remove_unused_columns=False,
    )

    trainer = MaskedLossTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=lambda b: collate(b, pad_id=tokenizer.pad_token_id),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=PATIENCE,
                early_stopping_threshold=0.0,
            )
        ],
    )

    trainer.train()

    best_dir = os.path.join(OUT_DIR, "best_or_final")
    trainer.save_model(best_dir)
    print("Saved model:", best_dir)
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best eval_loss:", trainer.state.best_metric)

    out_png = os.path.join(OUT_DIR, "loss_curve_train_vs_val.png")
    plot_loss_curves(
        trainer,
        out_png,
        sabdab_thr=SABDAB_H3_RMSD_MAX,
        oas_thr=OAS_PLDDT_THRESHOLD,
        lr=LEARNING_RATE,
        eval_steps=EVAL_STEPS,
    )


if __name__ == "__main__":
    main()