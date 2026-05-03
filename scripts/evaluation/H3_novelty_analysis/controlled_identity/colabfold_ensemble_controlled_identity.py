#!/usr/bin/env python3
"""
Controlled-Identity Analysis: Training Identity vs Mean Pairwise H3 RMSD
=========================================================================
Investigates whether structural quality of generated CDR-H3 loops is a genuine
model capability or merely a reflection of sequence memorisation from training
data.

The core idea ("controlled identity") is to stratify generated antibodies by
how similar their H3 loop is to the nearest example in the SAbDab training set,
then examine whether structural quality (measured as mean pairwise H3 Cα RMSD
across multiple AlphaFold2 seed predictions) depends on that identity level.

If a finetuned model is truly learning to generate good H3 structures — not
just copying training examples — its structural quality should remain good even
for sequences with low identity to the training set.

Pipeline
--------
1. Build the reference H3 pool from SAbDab:
   - Load Fv FASTA sequences and loop-span coordinates.
   - Keep only antibodies whose AlphaFold2 structural prediction achieved
     H3 RMSD ≤ SABDAB_H3_RMSD_MAX (1.5 Å), ensuring the reference pool
     consists of structurally reliable H3 examples.
   - Extract H3 subsequences using the loop-span coordinates.

2. Load generated sequence + structural quality data:
   - Each model's CSV contains one row per generated antibody with:
     * h3_seq: the CDR-H3 amino-acid sequence.
     * mean pairwise H3 RMSD: average Cα RMSD between all pairs of the
       five AlphaFold2 multimer seed predictions.  Low RMSD = high
       structural consensus = confident, well-defined loop.

3. For each generated H3 sequence, compute nearest-neighbour identity to
   the SAbDab pool via global pairwise alignment.

4. Bin sequences by NN identity (5% wide bins from 0–100%) and summarise
   the mean ± SE pairwise RMSD per bin.

5. Plot a bubble chart: x = identity bin centre, y = mean RMSD, bubble size
   ∝ number of sequences in the bin.  Error bars show standard error.

Inputs
------
- BASELINE_CSV   : CSV with h3_seq + mean pairwise H3 RMSD for the baseline model.
- FINETUNED_CSV  : Same for the finetuned model.
- SABDAB_TRAIN_FASTA    : FASTA of SAbDab Fv sequences used during training.
- SABDAB_TRAIN_LOOP_CSV : CSV with per-antibody H3 loop start/end coordinates.
- SABDAB_RMSD_XLSX      : Excel file of per-antibody AlphaFold2 H3 RMSD values
                          (used to filter to structurally reliable training examples).

Outputs (written to OUTPUT_DIR)
--------------------------------
- per_entry_identity_mean_pairwise_h3_rmsd.tsv : Per-generated-antibody table.
- binned_mean_pairwise_h3_rmsd.tsv             : Per-bin summary statistics.
- mean_pairwise_h3_rmsd_vs_training_identity_baseline_vs_finetuned.png
- run_log.txt

"""

import os
import re
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

plt.rcParams.update({'font.size': 14})
from Bio.Align import PairwiseAligner


# =========================================================
# CONFIG
# =========================================================
BASELINE_CSV = "/home/alanwu/Documents/iggen_model/evaluation_metrics/multiseed_alphafold/iggen/h3_mean_pairwise_rmsd_CA_complete_link_1.50A.csv"
FINETUNED_CSV = "/home/alanwu/Documents/iggen_model/evaluation_metrics/multiseed_alphafold/oas_v5/h3_mean_pairwise_rmsd_CA_complete_link_1.50A.csv"

SABDAB_TRAIN_FASTA = "/home/alanwu/Documents/iggen_model/data/single_fv_pdb.fasta"
SABDAB_TRAIN_LOOP_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"
SABDAB_RMSD_XLSX = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"

OUTPUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/multiseed_alphafold/oas_v5_identity_vs_mean_pairwise_h3_rmsd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_PATH = os.path.join(OUTPUT_DIR, "mean_pairwise_h3_rmsd_vs_training_identity_baseline_vs_finetuned.png")
RAW_TABLE_PATH = os.path.join(OUTPUT_DIR, "per_entry_identity_mean_pairwise_h3_rmsd.tsv")
BIN_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "binned_mean_pairwise_h3_rmsd.tsv")
LOG_PATH = os.path.join(OUTPUT_DIR, "run_log.txt")

# Only include training-set antibodies where AlphaFold2 predicted the H3 loop
# within 1.5 Å of the experimentally-determined structure, i.e. the reference
# pool is structurally reliable
SABDAB_H3_RMSD_MAX = 1.5

# Bins with fewer than this many sequences are excluded from the plot to avoid
# misleading statistics from sparse data
MIN_COUNT_PER_BIN = 3

# 5% wide identity bins spanning the full [0, 1] range
BIN_EDGES = np.arange(0.0, 1.0001, 0.05)

# Strict 20 canonical amino acids only; non-standard residues are stripped
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

# Alignment scoring: high open-gap penalty discourages fragmentary gapped
# alignments on short CDR loops (~10–20 residues)
ALIGN_MATCH = 2
ALIGN_MISMATCH = -1
ALIGN_OPEN_GAP = -10
ALIGN_EXTEND_GAP = -0.5
EPS = 1e-12  # Tie-breaking epsilon for identity comparisons


# =========================================================
# LOGGING
# =========================================================
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# =========================================================
# FASTA HELPERS
# =========================================================
def read_fasta_as_dict(path: str):
    d = {}
    for rec in SeqIO.parse(path, "fasta"):
        d[rec.id] = str(rec.seq).strip()
    return d


def clean_to_20aa(seq: str) -> str:
    if seq is None:
        return ""
    seq = str(seq).upper().strip()
    return "".join([c for c in seq if c in AA_SET])


def split_vh_vl(fv: str):
    # SAbDab FASTA stores Fv as "VH_sequence:VL_sequence"; split on ':'
    fv = fv.strip().upper()
    if ":" in fv:
        vh_raw, vl_raw = fv.split(":", 1)
    else:
        vh_raw, vl_raw = fv, ""
    vh = clean_to_20aa(vh_raw)
    vl = clean_to_20aa(vl_raw)
    return vh, vl


# =========================================================
# TRAINING H3 EXTRACTION
# =========================================================
def infer_loop_csv_cols(df: pd.DataFrame):
    """
    Robustly map logical column roles to actual column names by trying
    exact matches first, then regex search — guards against minor naming
    differences between CSV versions.
    """
    cols = {c.lower(): c for c in df.columns}

    def need(label: str, patterns):
        for pat in patterns:
            for lc, orig in cols.items():
                if lc == pat:
                    return orig
        for pat in patterns:
            for lc, orig in cols.items():
                if re.search(pat, lc):
                    return orig
        raise ValueError(f"Could not infer column for {label}. Available columns: {list(df.columns)}")

    ab_id = need("Antibody_ID", [r"^antibody_id$"])
    fasta_id = need("fasta_header_id", [r"^fasta_header_id$"])
    vh_len = need("VH_len", [r"^vh_len$"])
    h3s = need("H3_start", [r"^h3_start$"])
    h3e = need("H3_end", [r"^h3_end$"])

    return {
        "antibody_id": ab_id,
        "fasta_header_id": fasta_id,
        "vh_len": vh_len,
        "h3_start": h3s,
        "h3_end": h3e,
    }


def slice_inclusive(seq: str, start: int, end: int, one_based: bool):
    # Converts 1-based coordinates to 0-based when required, then slices
    if one_based:
        start -= 1
        end -= 1
    if start < 0 or end < start or end >= len(seq):
        return None
    return seq[start:end + 1]


def load_keep_ids_from_h3_rmsd(xlsx_path: str, threshold: float = 1.5):
    df = pd.read_excel(xlsx_path)

    id_col = None
    for c in df.columns:
        if str(c).lower() in ("antibody_id", "pdb", "pdb_id", "id", "name", "fasta_header_id"):
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    rmsd_candidates = []
    for c in df.columns:
        lc = str(c).lower()
        if "rmsd" in lc and ("h3" in lc or "cdrh3" in lc):
            rmsd_candidates.append(c)

    if not rmsd_candidates:
        raise ValueError(
            "Could not find an H3 RMSD column in the RMSD Excel. "
            "Please ensure the column name includes 'RMSD' and 'H3' (or 'CDRH3')."
        )

    # Prefer the shorter column name when multiple candidates exist (e.g.
    # "H3_RMSD" vs "CDR_H3_RMSD_mean") — shortest is most likely the primary column
    rmsd_candidates.sort(key=lambda x: len(str(x)))
    rmsd_col = rmsd_candidates[0]

    keep = set()
    for _, row in df.iterrows():
        ab = str(row[id_col]).strip()
        try:
            v = float(row[rmsd_col])
        except Exception:
            continue
        if np.isfinite(v) and v <= threshold:
            keep.add(ab)

    return keep


def extract_sabdab_h3s_filtered(train_fa, loop_df, cols, keep_ids):
    out = {}

    for _, row in loop_df.iterrows():
        ab = str(row[cols["antibody_id"]]).strip()
        if ab not in keep_ids:
            continue

        fasta_id = str(row[cols["fasta_header_id"]]).strip()

        # The loop CSV and FASTA may use slightly different ID formats;
        # fall back to fuzzy matching to reconcile mismatches
        key = None
        if fasta_id in train_fa:
            key = fasta_id
        else:
            for k in train_fa:
                if k == ab or k.startswith(ab) or ab in k or fasta_id in k:
                    key = k
                    break

        if key is None:
            continue

        fv = train_fa[key]
        vh, _vl = split_vh_vl(fv)

        try:
            vh_len = int(row[cols["vh_len"]])
        except Exception:
            vh_len = 0

        # The FASTA concatenates VH+VL; trim to VH length if recorded so that
        # H3 coordinates (which are VH-relative) index correctly
        if vh_len > 0 and vh_len <= len(vh):
            vh = vh[:vh_len]

        try:
            s = int(row[cols["h3_start"]])
            e = int(row[cols["h3_end"]])
        except Exception:
            continue

        # Try both 0-based and 1-based indexing; pick whichever yields a
        # biologically plausible H3 length (5–35 aa) when both are feasible
        h3_0 = slice_inclusive(vh, s, e, one_based=False)
        h3_1 = slice_inclusive(vh, s, e, one_based=True)

        if h3_0 is None and h3_1 is None:
            continue
        if h3_0 is not None and h3_1 is None:
            one_based = False
        elif h3_0 is None and h3_1 is not None:
            one_based = True
        else:
            if 5 <= len(h3_1) <= 35 and not (5 <= len(h3_0) <= 35):
                one_based = True
            else:
                one_based = False

        h3 = slice_inclusive(vh, s, e, one_based=one_based)
        h3 = clean_to_20aa(h3)

        # Guard against truncated or garbage entries; CDR-H3 is typically 5–30 aa
        if h3 and 0 < len(h3) < 60:
            out[ab] = h3

    return out


def get_training_h3_sequences():
    log(f"Loading SAbDab training FASTA: {SABDAB_TRAIN_FASTA}")
    sabdab_fa = read_fasta_as_dict(SABDAB_TRAIN_FASTA)
    log(f"Loaded {len(sabdab_fa)} FASTA records")

    log(f"Loading keep IDs from RMSD Excel with H3 RMSD <= {SABDAB_H3_RMSD_MAX}")
    keep_ids = load_keep_ids_from_h3_rmsd(SABDAB_RMSD_XLSX, threshold=SABDAB_H3_RMSD_MAX)
    log(f"Keep IDs: {len(keep_ids)}")

    log(f"Loading loop spans CSV: {SABDAB_TRAIN_LOOP_CSV}")
    loop_df = pd.read_csv(SABDAB_TRAIN_LOOP_CSV, sep=None, engine="python")
    cols = infer_loop_csv_cols(loop_df)

    h3_map = extract_sabdab_h3s_filtered(sabdab_fa, loop_df, cols, keep_ids)
    log(f"Extracted filtered SAbDab H3 loops: {len(h3_map)}")

    train_h3s = list(h3_map.values())
    if not train_h3s:
        raise RuntimeError("SAbDab H3 pool is empty after filtering. Check RMSD Excel + ID matching.")

    # Deduplicate — identical sequences should not be counted multiple times
    # when computing nearest-neighbour identity
    return sorted(set(train_h3s))


# =========================================================
# ALIGNMENT / IDENTITY
# =========================================================
def build_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = ALIGN_MATCH
    aligner.mismatch_score = ALIGN_MISMATCH
    aligner.open_gap_score = ALIGN_OPEN_GAP
    aligner.extend_gap_score = ALIGN_EXTEND_GAP
    return aligner


def _gapped_strings_from_blocks(a: str, b: str, aligned_blocks):
    """
    Reconstruct gapped alignment strings from BioPython's aligned-blocks
    representation.  BioPython ≥ 1.80 returns alignment coordinates as a
    pair of block-lists rather than directly yielding aligned strings, so
    we manually insert gap characters to build the comparable strings.
    """
    a_blocks, b_blocks = aligned_blocks
    i = 0
    j = 0
    out_a = []
    out_b = []
    for (a0, a1), (b0, b1) in zip(a_blocks, b_blocks):
        if i < a0:
            out_a.append(a[i:a0])
            out_b.append("-" * (a0 - i))
            i = a0
        if j < b0:
            out_a.append("-" * (b0 - j))
            out_b.append(b[j:b0])
            j = b0
        block_a = a[a0:a1]
        block_b = b[b0:b1]
        L = min(len(block_a), len(block_b))
        out_a.append(block_a[:L])
        out_b.append(block_b[:L])
        i = a0 + L
        j = b0 + L
    return "".join(out_a), "".join(out_b)


def alignment_global_identity(aligner: PairwiseAligner, a: str, b: str):
    if not a or not b:
        return 0.0, float("-inf")

    alns = aligner.align(a, b)
    if len(alns) == 0:
        return 0.0, float("-inf")

    aln = alns[0]
    score = float(aln.score)
    a_aln, b_aln = _gapped_strings_from_blocks(a, b, aln.aligned)

    matches = 0
    for ca, cb in zip(a_aln, b_aln):
        if ca == "-" or cb == "-":
            continue
        if ca == cb:
            matches += 1

    # Normalise by the longer sequence so length differences are penalised
    denom = max(len(a), len(b))
    ident = matches / denom if denom > 0 else 0.0
    return ident, score


def nearest_training_identity(query_h3, training_h3s, aligner):
    best_ident = -1.0
    best_score = float("-inf")
    best_train_h3 = None

    for train_h3 in training_h3s:
        ident, score = alignment_global_identity(aligner, query_h3, train_h3)
        # EPS tie-breaking: when two training sequences share identical identity,
        # prefer the one with the better alignment score (fewer / smaller gaps)
        if (ident > best_ident + EPS) or (abs(ident - best_ident) <= EPS and score > best_score):
            best_ident = ident
            best_score = score
            best_train_h3 = train_h3

    return best_ident, best_train_h3


# =========================================================
# LOAD MULTISEED CSV
# =========================================================
def load_multiseed_csv(path):
    """
    Load the pre-computed per-antibody mean pairwise H3 RMSD CSV.
    Column names are discovered dynamically so the function tolerates minor
    naming variations across experiment runs.
    """
    df = pd.read_csv(path)

    cols_lower = {str(c).lower(): c for c in df.columns}

    seq_col = None
    rmsd_col = None
    entry_col = None

    for c in df.columns:
        lc = str(c).lower()
        if lc == "h3_seq":
            seq_col = c
        elif "mean_pairwise" in lc and "h3" in lc and "rmsd" in lc:
            rmsd_col = c
        elif lc in ("antibody", "entry", "id", "name"):
            entry_col = c

    if seq_col is None:
        raise ValueError(f"Could not find h3_seq column in {path}")
    if rmsd_col is None:
        raise ValueError(f"Could not find mean pairwise H3 RMSD column in {path}")
    if entry_col is None:
        entry_col = df.columns[0]

    rows = []
    for _, row in df.iterrows():
        h3_seq = clean_to_20aa(row[seq_col])
        if not h3_seq:
            continue

        try:
            mean_rmsd = float(row[rmsd_col])
        except Exception:
            continue
        if not np.isfinite(mean_rmsd):
            continue

        rows.append({
            "entry": str(row[entry_col]),
            "h3_seq": h3_seq,
            "mean_pairwise_h3_rmsd": mean_rmsd,
        })

    return rows


# =========================================================
# BUILD DATASET
# =========================================================
def build_dataset(group_label, csv_path, training_h3s, aligner):
    log(f"\n=== Processing {group_label} ===")
    log(f"CSV: {csv_path}")

    raw_rows = load_multiseed_csv(csv_path)
    log(f"Loaded multiseed rows: {len(raw_rows)}")

    rows = []
    skipped = []

    for r in raw_rows:
        h3_seq = r["h3_seq"]
        if not h3_seq:
            skipped.append((r["entry"], "Empty H3 sequence"))
            continue

        # This is the computationally expensive step: O(|training_h3s|) alignments
        # per generated sequence.  No approximation is used here to ensure exact NN.
        nn_ident, nn_train_h3 = nearest_training_identity(h3_seq, training_h3s, aligner)

        rows.append({
            "entry": r["entry"],
            "group": group_label,
            "h3_seq": h3_seq,
            "h3_len": len(h3_seq),
            "mean_pairwise_h3_rmsd": float(r["mean_pairwise_h3_rmsd"]),
            "nearest_train_identity": float(nn_ident),
            "nearest_train_h3": nn_train_h3,
        })

    log(f"Kept: {len(rows)}")
    log(f"Skipped: {len(skipped)}")

    if skipped:
        log("Skipped examples:")
        for entry_id, reason in skipped[:10]:
            log(f"  {entry_id}: {reason}")

    return rows


# =========================================================
# BINNING / PLOTTING
# =========================================================
def summarise_by_identity_bin(rows, bin_edges):
    """
    Aggregate mean pairwise RMSD statistics within each identity bin.
    The final bin is closed on the right ([left, right]) while all others
    are half-open ([left, right)) to avoid double-counting the boundary.
    """
    summaries = []
    groups = sorted(set(r["group"] for r in rows))

    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        identities = np.array([r["nearest_train_identity"] for r in group_rows], dtype=float)
        rmsds = np.array([r["mean_pairwise_h3_rmsd"] for r in group_rows], dtype=float)

        for i in range(len(bin_edges) - 1):
            left = float(bin_edges[i])
            right = float(bin_edges[i + 1])

            if i == len(bin_edges) - 2:
                mask = (identities >= left) & (identities <= right)
            else:
                mask = (identities >= left) & (identities < right)

            vals = rmsds[mask]
            n = int(np.sum(mask))

            if n < MIN_COUNT_PER_BIN:
                continue

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals))
            se_val = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

            summaries.append({
                "group": group,
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "n": n,
                "mean_pairwise_h3_rmsd": mean_val,
                "sd_pairwise_h3_rmsd": sd_val,
                "se_pairwise_h3_rmsd": se_val,
            })

    return summaries


def plot_bubble_identity_rmsd(summaries, out_path):
    plt.figure(figsize=(8, 5.8))

    label_map = {
        "baseline": "Baseline",
        "finetuned": "Finetuned",
    }
    color_map = {
        "baseline": "tab:orange",
        "finetuned": "tab:blue",
    }

    plotted_any = False

    for group in ["baseline", "finetuned"]:
        grp = [x for x in summaries if x["group"] == group]
        grp = sorted(grp, key=lambda d: d["bin_center"])
        if not grp:
            continue

        x = np.array([d["bin_center"] for d in grp], dtype=float)
        y = np.array([d["mean_pairwise_h3_rmsd"] for d in grp], dtype=float)
        n = np.array([d["n"] for d in grp], dtype=float)
        se = np.array([d["se_pairwise_h3_rmsd"] for d in grp], dtype=float)

        # Bubble size encodes sample count per bin so sparse bins are visually
        # distinguishable from well-supported ones
        sizes = 40 + 25 * n
        color = color_map[group]

        plt.plot(x, y, linewidth=1.5, alpha=0.6, label=label_map[group], color=color)
        plt.scatter(x, y, s=sizes, alpha=0.55, color=color)
        plt.errorbar(x, y, yerr=se, fmt="none", elinewidth=1, capsize=3, alpha=0.7, color=color)

        plotted_any = True

    if not plotted_any:
        raise ValueError("No bins had enough points to plot.")

    plt.xlabel("Nearest neighbour identity")
    plt.ylabel("Mean pairwise H3 RMSD (Å)")
    if summaries:
        x_min = min(d["bin_left"] for d in summaries)
        x_max = max(d["bin_right"] for d in summaries)
        pad = (x_max - x_min) * 0.04
        plt.xlim(x_min - pad, x_max + pad)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# OUTPUT TABLES
# =========================================================
def write_rows_tsv(rows, path):
    cols = [
        "entry",
        "group",
        "h3_seq",
        "h3_len",
        "mean_pairwise_h3_rmsd",
        "nearest_train_identity",
        "nearest_train_h3",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def write_bin_summary_tsv(summaries, path):
    cols = [
        "group",
        "bin_left",
        "bin_right",
        "bin_center",
        "n",
        "mean_pairwise_h3_rmsd",
        "sd_pairwise_h3_rmsd",
        "se_pairwise_h3_rmsd",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summaries:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


# =========================================================
# MAIN
# =========================================================
def main():
    training_h3s = get_training_h3_sequences()
    log(f"Unique filtered training H3 loops: {len(training_h3s)}")

    aligner = build_aligner()

    baseline_rows = build_dataset(
        group_label="baseline",
        csv_path=BASELINE_CSV,
        training_h3s=training_h3s,
        aligner=aligner,
    )

    finetuned_rows = build_dataset(
        group_label="finetuned",
        csv_path=FINETUNED_CSV,
        training_h3s=training_h3s,
        aligner=aligner,
    )

    all_rows = baseline_rows + finetuned_rows
    if not all_rows:
        raise ValueError("No valid rows produced for either group.")

    write_rows_tsv(all_rows, RAW_TABLE_PATH)
    log(f"\nPer-entry table written to: {RAW_TABLE_PATH}")

    summaries = summarise_by_identity_bin(all_rows, BIN_EDGES)
    if not summaries:
        raise ValueError("No identity bins had enough points to summarise.")

    write_bin_summary_tsv(summaries, BIN_SUMMARY_PATH)
    log(f"Binned summary written to: {BIN_SUMMARY_PATH}")

    plot_bubble_identity_rmsd(summaries, PLOT_PATH)
    log(f"Plot written to: {PLOT_PATH}")

    for group in ["baseline", "finetuned"]:
        grp = [r for r in all_rows if r["group"] == group]
        if grp:
            mean_rmsd = np.mean([r["mean_pairwise_h3_rmsd"] for r in grp])
            mean_ident = np.mean([r["nearest_train_identity"] for r in grp])
            log(f"{group}: n={len(grp)}, mean pairwise H3 RMSD={mean_rmsd:.3f}, mean nearest-training identity={mean_ident:.3f}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    log(f"Log written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
