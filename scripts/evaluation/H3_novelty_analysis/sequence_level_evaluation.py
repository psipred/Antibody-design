"""
H3 Sequence-Level Novelty Analysis
====================================
Evaluates how novel the generated CDR-H3 loops are relative to the SAbDab
training set at the sequence level.  Three complementary views are produced:

1. **Length distribution** — Do generated H3 loops match the length profile of
   the training set?  Divergence here could indicate mode collapse or systematic
   length bias introduced by fine-tuning.

2. **Amino-acid frequency** — Do generated H3 loops reproduce the amino-acid
   composition of training H3 loops?  Large deviations may indicate the model
   is generating biologically implausible sequences.

3. **Nearest-neighbour (NN) identity to training set** — For each generated H3
   sequence, find the most similar training-set H3 via global pairwise
   alignment.  The distribution of these NN identity scores is the primary
   novelty metric: a lower mean indicates more novel (less memorised) sequences.

Unlike the controlled-identity scripts, this script compares against the SAbDab
training set rather than within the generated set, making it an *extrinsic*
novelty measure.

SAbDab Quality Filter
---------------------
Training-set antibodies are pre-filtered to those where AlphaFold2 predicted
the H3 loop within SABDAB_H3_RMSD_MAX (1 Å) of the experimentally-determined
structure.  This ensures the reference pool consists only of structurally
reliable H3 examples — removing poor-quality structures that might inflate
apparent novelty by acting as very distant nearest neighbours.

NN Search Optimisation
----------------------
A length-bucketed candidate search is used: for each generated H3 of length L,
only training H3s within ±LEN_WINDOW residues of L are aligned in the first
pass.  If no candidates fall in the window (e.g. for unusual lengths), the
full training set is used as a fallback.  This avoids aligning every generated
sequence against every training sequence while still guaranteeing the true NN
is found for typical H3 lengths.

Inputs
------
- SABDAB_TRAIN_FASTA    : FASTA of SAbDab Fv sequences used during training.
- SABDAB_TRAIN_LOOP_CSV : CSV with per-antibody H3 start/end coordinates.
- SABDAB_RMSD_XLSX      : Excel file of per-antibody H3 structural RMSD values.
- DATASETS              : Dict mapping group names to generated H3 FASTA files
                          (Chothia-numbered, produced by ANARCI).

Outputs (written to OUT_DIR)
-----------------------------
- generated_best_match_sabdab_only.csv : Per-generated-sequence NN table.
- length_hist_{group}.png              : Length histogram vs training set.
- best_identity_hist_{group}.png       : NN-identity histogram per group.
- aa_freq.png                          : Grouped amino-acid frequency bar chart.
- length_hist_combined.png             : Length histogram, all groups overlaid.
- identity_hist_combined.png           : NN-identity, all groups overlaid.

"""

import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Align import PairwiseAligner

plt.rcParams.update({'font.size': 14})

# ----------------------------
# Paths
# ----------------------------
SABDAB_TRAIN_FASTA    = "/home/alanwu/Documents/iggen_model/data/single_fv_pdb.fasta"
SABDAB_TRAIN_LOOP_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"
SABDAB_RMSD_XLSX      = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"

DATASETS = {
    "Baseline": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_h3_chothia.fasta",
    "Finetuned": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v6/generated_h3_chothia.fasta",
}

# Consistent colour palette shared across evaluation scripts
GROUP_COLORS = {
    "Baseline": "#f5c87a",
    "Finetuned": "#a8c8e8",
}

OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/sequence_level/combined/h3_similarity_analysis_sabdab_only"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Config
# ----------------------------
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET   = set(AA_ORDER)

# Window (in residues) used for length-bucketed NN candidate pruning;
# keeps full alignment cost manageable without missing true neighbours
LEN_WINDOW       = 10

# Stricter than the controlled-identity scripts (1 Å vs 1.5 Å) to keep only
# the most confident training examples as the novelty reference
SABDAB_H3_RMSD_MAX = 1

ALIGN_MATCH      =  2
ALIGN_MISMATCH   = -1
ALIGN_OPEN_GAP   = -10
ALIGN_EXTEND_GAP = -0.5
EPS              = 1e-12


# ----------------------------
# FASTA helpers
# ----------------------------
def read_fasta_as_dict(path: str) -> Dict[str, str]:
    d = {}
    for rec in SeqIO.parse(path, "fasta"):
        d[rec.id] = str(rec.seq).strip()
    return d


def clean_to_20aa(seq: str) -> str:
    if seq is None:
        return ""
    seq = str(seq).upper().strip()
    return "".join([c for c in seq if c in AA_SET])


def split_vh_vl(fv: str) -> Tuple[str, str]:
    # SAbDab FASTA encodes Fv as "VH:VL"; split and clean each chain separately
    fv = fv.strip().upper()
    if ":" in fv:
        vh_raw, vl_raw = fv.split(":", 1)
    else:
        vh_raw, vl_raw = fv, ""
    return clean_to_20aa(vh_raw), clean_to_20aa(vl_raw)


# ----------------------------
# SAbDab helpers
# ----------------------------
def infer_loop_csv_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Robustly map logical column roles to actual DataFrame column names using
    exact match first, then regex fallback.  Tolerates minor naming differences
    between CSV file versions.
    """
    cols = {c.lower(): c for c in df.columns}

    def need(label, patterns):
        for pat in patterns:
            for lc, orig in cols.items():
                if lc == pat:
                    return orig
        for pat in patterns:
            for lc, orig in cols.items():
                if re.search(pat, lc):
                    return orig
        raise ValueError(f"Could not infer column for {label}. Available: {list(df.columns)}")

    return {
        "antibody_id":   need("Antibody_ID",   [r"^antibody_id$"]),
        "fasta_header_id": need("fasta_header_id", [r"^fasta_header_id$"]),
        "vh_len":        need("VH_len",        [r"^vh_len$"]),
        "h3_start":      need("H3_start",      [r"^h3_start$"]),
        "h3_end":        need("H3_end",        [r"^h3_end$"]),
    }


def slice_inclusive(seq: str, start: int, end: int, one_based: bool) -> Optional[str]:
    if one_based:
        start -= 1
        end -= 1
    if start < 0 or end < start or end >= len(seq):
        return None
    return seq[start:end + 1]


def load_keep_ids_from_h3_rmsd(xlsx_path: str, threshold: float = 1.5) -> set:
    df = pd.read_excel(xlsx_path)
    id_col = next(
        (c for c in df.columns if str(c).lower() in ("antibody_id", "pdb", "pdb_id", "id", "name", "fasta_header_id")),
        df.columns[0]
    )
    rmsd_candidates = [c for c in df.columns if "rmsd" in str(c).lower() and ("h3" in str(c).lower() or "cdrh3" in str(c).lower())]
    if not rmsd_candidates:
        raise ValueError("Could not find an H3 RMSD column in the RMSD Excel.")
    # Prefer the shortest matching column name as the primary RMSD column
    rmsd_col = sorted(rmsd_candidates, key=lambda x: len(str(x)))[0]
    keep = set()
    for _, row in df.iterrows():
        try:
            v = float(row[rmsd_col])
        except Exception:
            continue
        if np.isfinite(v) and v <= threshold:
            keep.add(str(row[id_col]))
    return keep


def extract_sabdab_h3s_filtered(train_fa, loop_df, cols, keep_ids) -> Dict[str, str]:
    out = {}
    for _, row in loop_df.iterrows():
        ab = str(row[cols["antibody_id"]])
        if ab not in keep_ids:
            continue
        fasta_id = str(row[cols["fasta_header_id"]])
        # Attempt direct key lookup; fall back to fuzzy matching for ID mismatches
        key = fasta_id if fasta_id in train_fa else next(
            (k for k in train_fa if k == ab or k.startswith(ab) or ab in k or fasta_id in k), None
        )
        if key is None:
            continue
        fv = train_fa[key]
        vh, _ = split_vh_vl(fv)
        try:
            vh_len = int(row[cols["vh_len"]])
            # Trim to VH length so H3 coordinates index into the VH only
            if 0 < vh_len <= len(vh):
                vh = vh[:vh_len]
        except Exception:
            pass
        try:
            s, e = int(row[cols["h3_start"]]), int(row[cols["h3_end"]])
        except Exception:
            continue
        h3_0 = slice_inclusive(vh, s, e, one_based=False)
        h3_1 = slice_inclusive(vh, s, e, one_based=True)
        if h3_0 is None and h3_1 is None:
            continue
        if h3_0 is not None and h3_1 is None:
            one_based = False
        elif h3_0 is None:
            one_based = True
        else:
            # Prefer 1-based when it uniquely yields a biologically reasonable
            # H3 length and 0-based does not
            one_based = 5 <= len(h3_1) <= 35 and not (5 <= len(h3_0) <= 35)
        h3 = clean_to_20aa(slice_inclusive(vh, s, e, one_based=one_based))
        if h3 and 0 < len(h3) < 60:
            out[ab] = h3
    return out


# ----------------------------
# AA frequency
# ----------------------------
def pooled_aa_freq(seqs: List[str]) -> np.ndarray:
    """
    Compute pooled amino-acid frequency across all sequences.
    Pooling (rather than averaging per-sequence frequencies) means longer
    sequences contribute proportionally more, matching biological intuition.
    """
    counts = Counter()
    for s in seqs:
        counts.update(aa for aa in s if aa in AA_SET)
    total = sum(counts[a] for a in AA_ORDER)
    return np.array([counts[a] / total if total > 0 else 0.0 for a in AA_ORDER], dtype=float)


# ----------------------------
# Alignment
# ----------------------------
def build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode           = "global"
    aligner.match_score    = ALIGN_MATCH
    aligner.mismatch_score = ALIGN_MISMATCH
    aligner.open_gap_score = ALIGN_OPEN_GAP
    aligner.extend_gap_score = ALIGN_EXTEND_GAP
    return aligner


def _gapped_strings_from_blocks(a, b, aligned_blocks):
    """
    Reconstruct gapped alignment strings from BioPython's block-coordinate
    representation (required for BioPython ≥ 1.80, which no longer yields
    aligned strings directly).
    """
    a_blocks, b_blocks = aligned_blocks
    i, j, out_a, out_b = 0, 0, [], []
    for (a0, a1), (b0, b1) in zip(a_blocks, b_blocks):
        if i < a0:
            out_a.append(a[i:a0]); out_b.append("-" * (a0 - i)); i = a0
        if j < b0:
            out_a.append("-" * (b0 - j)); out_b.append(b[j:b0]); j = b0
        L = min(a1 - a0, b1 - b0)
        out_a.append(a[a0:a0 + L]); out_b.append(b[b0:b0 + L])
        i, j = a0 + L, b0 + L
    return "".join(out_a), "".join(out_b)


def alignment_global_identity(aligner, a, b):
    if not a or not b:
        return 0.0, float("-inf")
    alns = aligner.align(a, b)
    if not alns:
        return 0.0, float("-inf")
    aln = alns[0]
    a_aln, b_aln = _gapped_strings_from_blocks(a, b, aln.aligned)
    matches = sum(ca == cb for ca, cb in zip(a_aln, b_aln) if ca != "-" and cb != "-")
    denom = max(len(a), len(b))
    return matches / denom if denom > 0 else 0.0, float(aln.score)


def nn_search(gen_h3_map, sabdab_h3_map, aligner) -> List[float]:
    """
    For each generated H3, find the nearest-neighbour in the SAbDab training
    set using length-bucketed candidate pruning.

    Returns
    -------
    best_identities : list of float, one per generated sequence
    rows            : list of dicts with detailed per-sequence results
    """
    # Pre-bucket training H3s by length for O(1) candidate retrieval
    bucket = defaultdict(list)
    for ab, h3 in sabdab_h3_map.items():
        bucket[len(h3)].append((ab, h3))

    best_identities = []
    rows = []
    for gid, gh3 in gen_h3_map.items():
        L = len(gh3)
        # Limit alignment to training sequences within ±LEN_WINDOW residues to
        # avoid O(|gen| × |train|) full alignments; the window is wide enough to
        # capture the true NN for all but pathologically short or long H3s
        candidates = []
        for l2 in range(L - LEN_WINDOW, L + LEN_WINDOW + 1):
            candidates.extend(bucket.get(l2, []))
        if not candidates:
            # Fallback: use the full training set when the generated sequence
            # length is outside the range of all training H3 lengths
            candidates = list(sabdab_h3_map.items())

        best_ab, best_th3, best_ident, best_score = None, None, -1.0, float("-inf")
        for ab, th3 in candidates:
            ident, score = alignment_global_identity(aligner, gh3, th3)
            if (ident > best_ident + EPS) or (abs(ident - best_ident) <= EPS and score > best_score):
                best_ident, best_score, best_ab, best_th3 = ident, score, ab, th3

        best_identities.append(best_ident)
        rows.append({
            "gen_id": gid, "gen_h3": gh3, "gen_len": L,
            "best_sabdab_antibody_id": best_ab, "best_sabdab_h3": best_th3,
            "best_sabdab_len": len(best_th3) if best_th3 else np.nan,
            "best_global_score": best_score, "best_identity": best_ident,
            "n_candidates": len(candidates),
        })
    return best_identities, rows


# ----------------------------
# Per-group plots
# ----------------------------
def plot_length_hist_single(train_lens: List[int], gen_lens: List[int],
                            group_name: str, color: str, out_path: str) -> None:
    fig, ax = plt.subplots()
    all_lens = train_lens + gen_lens
    bins = range(0, max(all_lens, default=0) + 2)

    ax.hist(train_lens, bins=bins, alpha=0.45, color="gray", label="Training set")
    ax.hist(gen_lens,   bins=bins, alpha=0.7,  color=color,  label=group_name)

    ytop = ax.get_ylim()[1]
    train_mean = np.mean(train_lens) if train_lens else 0
    gen_mean   = np.mean(gen_lens)   if gen_lens   else 0

    ax.axvline(train_mean, linestyle="--", linewidth=1.5, color="gray")
    ax.text(train_mean + 0.2, ytop * 0.88, f"{train_mean:.3g}", fontsize=14, color="gray")
    ax.axvline(gen_mean, linestyle="--", linewidth=1.5, color=color)
    ax.text(gen_mean + 0.2, ytop * 0.76, f"{gen_mean:.3g}", fontsize=14, color=color)

    ax.set_xlabel("H3 length (aa)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_identity_hist_single(identities: List[float], group_name: str,
                              color: str, out_path: str) -> None:
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 31)
    mean_val = float(np.mean(identities)) if identities else 0.0

    ax.hist(identities, bins=bins, alpha=0.85, color=color, edgecolor="none")
    ax.axvline(mean_val, linestyle="--", linewidth=1.5, color=color)

    ytop = ax.get_ylim()[1]
    ax.text(mean_val + 0.02, ytop * 0.9, f"Mean = {mean_val:.3g}", fontsize=14)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Nearest-neighbour identity to training set")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Combined plots
# ----------------------------
def plot_length_hist_combined(train_lens, group_lens: Dict[str, List[int]], out_path: str) -> None:
    fig, ax = plt.subplots()
    all_lens = train_lens + [l for ls in group_lens.values() for l in ls]
    bins = range(0, max(all_lens, default=0) + 2)

    ax.hist(train_lens, bins=bins, alpha=0.45, color="gray", label="SAbDab training")
    for group_name, lens in group_lens.items():
        ax.hist(lens, bins=bins, alpha=0.6, color=GROUP_COLORS[group_name], label=group_name)

    # draw mean vlines after histograms so ylim is settled
    ytop = ax.get_ylim()[1]
    train_mean = np.mean(train_lens) if train_lens else 0
    ax.axvline(train_mean, linestyle="--", linewidth=1.5, color="gray")
    ax.text(train_mean + 0.2, ytop * 0.88, f"{train_mean:.3g}", fontsize=14, color="gray")

    # Stagger vertical text labels so they do not overlap when means are close
    offsets = [0.78, 0.68]
    for (group_name, lens), offset in zip(group_lens.items(), offsets):
        color = GROUP_COLORS[group_name]
        mean_val = np.mean(lens) if lens else 0
        ax.axvline(mean_val, linestyle="--", linewidth=1.5, color=color)
        ax.text(mean_val + 0.2, ytop * offset, f"{mean_val:.3g}", fontsize=14, color=color)

    ax.set_xlabel("H3 length (aa)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_aa_freq_combined(train_freq, group_freqs: Dict[str, np.ndarray], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(AA_ORDER))
    n_groups = 1 + len(group_freqs)
    width = 0.8 / n_groups
    # Centre the cluster of bars over each amino-acid tick position
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

    ax.bar(x + offsets[0], train_freq, width=width, color="gray", alpha=0.7, label="SAbDab training")
    for i, (group_name, freq) in enumerate(group_freqs.items(), start=1):
        ax.bar(x + offsets[i], freq, width=width, color=GROUP_COLORS[group_name], alpha=0.85, label=group_name)

    ax.set_xticks(x)
    ax.set_xticklabels(AA_ORDER)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_identity_hist_combined(group_identities: Dict[str, List[float]], out_path: str) -> None:
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 31)

    for group_name, identities in group_identities.items():
        color = GROUP_COLORS[group_name]
        mean_val = float(np.mean(identities)) if identities else 0.0
        ax.hist(identities, bins=bins, alpha=0.6, color=color,
                edgecolor="none", label=f"{group_name} (mean = {mean_val:.3g})")
        ax.axvline(mean_val, linestyle="--", linewidth=1.5, color=color)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Nearest-neighbour identity to training set")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    print("[1/6] Load SAbDab Fv FASTA...")
    sabdab_fa = read_fasta_as_dict(SABDAB_TRAIN_FASTA)

    print(f"[2/6] Load SAbDab keep IDs (H3 RMSD <= {SABDAB_H3_RMSD_MAX})...")
    sabdab_keep_ids = load_keep_ids_from_h3_rmsd(SABDAB_RMSD_XLSX, threshold=SABDAB_H3_RMSD_MAX)
    print(f"  Keep IDs: {len(sabdab_keep_ids)}")

    print("[3/6] Extract SAbDab H3 loops...")
    loop_df = pd.read_csv(SABDAB_TRAIN_LOOP_CSV, sep=None, engine="python")
    cols = infer_loop_csv_cols(loop_df)
    sabdab_h3_map = extract_sabdab_h3s_filtered(sabdab_fa, loop_df, cols, sabdab_keep_ids)
    print(f"  SAbDab H3 kept: {len(sabdab_h3_map)}")
    train_h3s = list(sabdab_h3_map.values())
    if not train_h3s:
        raise RuntimeError("SAbDab H3 pool is empty after filtering.")

    print("[4/6] Load generated H3s for all datasets...")
    gen_h3_maps = {}
    for group_name, fasta_path in DATASETS.items():
        raw = read_fasta_as_dict(fasta_path)
        # Walrus operator (:=) avoids recomputing clean_to_20aa for the filter
        # and the dict value simultaneously
        gen_h3_maps[group_name] = {gid: h3c for gid, h3 in raw.items() if (h3c := clean_to_20aa(h3))}
        print(f"  {group_name}: {len(gen_h3_maps[group_name])} H3 sequences")

    print("[5/6] Nearest-neighbour search (each dataset vs SAbDab)...")
    aligner = build_aligner()
    group_identities = {}
    all_rows = []
    for group_name, gen_h3_map in gen_h3_maps.items():
        print(f"  Processing {group_name}...")
        identities, rows = nn_search(gen_h3_map, sabdab_h3_map, aligner)
        group_identities[group_name] = identities
        for r in rows:
            r["group"] = group_name
        all_rows.extend(rows)
        mean_id = np.mean(identities) if identities else 0.0
        print(f"  {group_name}: mean NN identity = {mean_id:.3f}")

    pd.DataFrame(all_rows).sort_values(["group", "best_identity"], ascending=[True, False]).to_csv(
        os.path.join(OUT_DIR, "generated_best_match_sabdab_only.csv"), index=False
    )

    print("[6/6] Generating plots...")
    train_lens  = [len(x) for x in train_h3s]
    train_freq  = pooled_aa_freq(train_h3s)
    group_lens  = {g: [len(h3) for h3 in gen_h3_maps[g].values()] for g in gen_h3_maps}
    group_freqs = {g: pooled_aa_freq(list(gen_h3_maps[g].values())) for g in gen_h3_maps}

    # Per-group: length histogram and identity histogram
    for group_name in gen_h3_maps:
        color = GROUP_COLORS[group_name]
        tag   = group_name.lower()
        plot_length_hist_single(
            train_lens, group_lens[group_name], group_name, color,
            os.path.join(OUT_DIR, f"length_hist_{tag}.png"),
        )
        plot_identity_hist_single(
            group_identities[group_name], group_name, color,
            os.path.join(OUT_DIR, f"best_identity_hist_{tag}.png"),
        )

    # Combined: AA frequency
    plot_aa_freq_combined(
        train_freq,
        group_freqs,
        os.path.join(OUT_DIR, "aa_freq.png"),
    )

    print("Done. Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
