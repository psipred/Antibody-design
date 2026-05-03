"""
Sequence-level evaluation of generated CDR H3 loops vs SAbDab training set
============================================================================
Compares H3 loop sequences produced by the fine-tuned antibody model against
H3 loops extracted from the SAbDab structural training set.  Only high-quality
SAbDab structures are included: antibodies are kept only if their H3 loop has a
backbone RMSD ≤ SABDAB_H3_RMSD_MAX Å relative to the experimental structure
(verified by ColabFold prediction quality).

Analyses performed
------------------
1. H3 loop length distribution — histogram comparing training vs generated.
2. Amino-acid composition — per-AA frequency bar chart.
3. Nearest-neighbour sequence identity — for each generated H3, find the
   closest SAbDab H3 by global pairwise alignment identity.

Identity is defined as:
    matches / max(len(query), len(target))
where *matches* counts aligned, non-gap, identical residue pairs.
Using max(len) rather than alignment length penalises indels and avoids
inflating identity scores for partial overlaps.

Search is accelerated by a length-bucketing heuristic: only SAbDab H3s within
±LEN_WINDOW residues of the query length are considered as alignment targets.
This is safe because global alignment identity drops sharply when lengths
differ significantly.

Inputs
------
  SABDAB_TRAIN_FASTA      : FASTA of full Fv sequences from SAbDab (VH:VL format)
  SABDAB_TRAIN_LOOP_CSV   : CSV with antibody_id, fasta_header_id, VH_len,
                            H3_start, H3_end columns
  SABDAB_RMSD_XLSX        : Excel table with antibody_id and H3 RMSD columns
  GEN_H3_FASTA            : FASTA of H3 sequences extracted by ANARCI from
                            model-generated antibodies

Outputs (written to OUT_DIR)
-----------------------------
  length_hist.png                        — H3 length distribution
  aa_freq.png                            — amino-acid composition bar chart
  best_identity_hist.png                 — histogram of nearest-neighbour identities
  generated_best_match_sabdab_only.csv   — per-generated-H3 best-match details

"""

import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

plt.rcParams.update({'font.size': 14})
from Bio.Align import PairwiseAligner


# ----------------------------
# Paths
# ----------------------------
SABDAB_TRAIN_FASTA = "/home/alanwu/Documents/iggen_model/data/single_fv_pdb.fasta"
SABDAB_TRAIN_LOOP_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"

# This Excel must contain an H3 RMSD column (name includes 'RMSD' and 'H3' or 'CDRH3')
SABDAB_RMSD_XLSX = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"

# Generated H3 fasta produced from terminal ANARCI extraction
GEN_H3_FASTA = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v6/generated_h3_chothia.fasta"

OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/sequence_level/oas_v6/h3_similarity_analysis_sabdab_only"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------
# Config
# ----------------------------
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

# Only compare against SAbDab H3s within this length window of the query.
# Speeds up nearest-neighbour search; rarely misses the true best match since
# global alignment identity drops sharply for large length differences.
LEN_WINDOW = 10

# Structural quality filter: only include SAbDab antibodies whose H3 loop
# was modelled with RMSD ≤ this threshold (in Å).  Keeps the reference set
# to well-predicted, high-confidence loops.
SABDAB_H3_RMSD_MAX = 1  # include SAbDab H3 only if RMSD <= 1.5

# Alignment parameters (GLOBAL alignment)
ALIGN_MATCH = 2
ALIGN_MISMATCH = -1
# Heavy gap-open penalty discourages introducing gaps, consistent with the
# expectation that true H3 sequences don't frequently have internal deletions.
ALIGN_OPEN_GAP = -10
ALIGN_EXTEND_GAP = -0.5

# Floating-point tolerance when comparing identity scores; avoids incorrect
# tie-breaking due to rounding errors in alignment score accumulation.
EPS = 1e-12


# ----------------------------
# FASTA helpers
# ----------------------------
def read_fasta_as_dict(path: str) -> Dict[str, str]:
    d = {}
    for rec in SeqIO.parse(path, "fasta"):
        d[rec.id] = str(rec.seq).strip()
    return d


def clean_to_20aa(seq: str) -> str:
    """Keep only 20 standard amino acids, uppercase (drops X, B, Z, ., *, etc)."""
    if seq is None:
        return ""
    seq = str(seq).upper().strip()
    return "".join([c for c in seq if c in AA_SET])


def split_vh_vl(fv: str) -> Tuple[str, str]:
    """
    Split FIRST on ':' then clean each chain.
    If ':' missing, treat whole record as VH.
    """
    fv = fv.strip().upper()
    if ":" in fv:
        vh_raw, vl_raw = fv.split(":", 1)
    else:
        vh_raw, vl_raw = fv, ""
    vh = clean_to_20aa(vh_raw)
    vl = clean_to_20aa(vl_raw)
    return vh, vl


# ----------------------------
# SAbDab: parse loop CSV columns and slice H3 from VH
# ----------------------------
def infer_loop_csv_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}

    def need(label: str, patterns: List[str]) -> str:
        # First try exact match, then regex search — handles minor naming variants
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


def slice_inclusive(seq: str, start: int, end: int, one_based: bool) -> Optional[str]:
    if one_based:
        start -= 1
        end -= 1
    if start < 0 or end < start or end >= len(seq):
        return None
    return seq[start:end + 1]


def load_keep_ids_from_h3_rmsd(xlsx_path: str, threshold: float = 1.5) -> set:
    """
    Reads an Excel table that contains an antibody id column and an H3 RMSD column.
    Keeps antibody ids where H3 RMSD <= threshold.
    """
    df = pd.read_excel(xlsx_path)

    # Heuristic column selection: check several common id-column names
    id_col = None
    for c in df.columns:
        if str(c).lower() in ("antibody_id", "pdb", "pdb_id", "id", "name", "fasta_header_id"):
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    # Select the shortest column name that matches "rmsd + (h3 or cdrh3)" to
    # prefer a specific H3 RMSD column over composite metric names.
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
    rmsd_candidates.sort(key=lambda x: len(str(x)))
    rmsd_col = rmsd_candidates[0]

    keep = set()
    for _, row in df.iterrows():
        ab = str(row[id_col])
        try:
            v = float(row[rmsd_col])
        except Exception:
            continue
        if np.isfinite(v) and v <= threshold:
            keep.add(ab)

    return keep


def extract_sabdab_h3s_filtered(
    train_fa: Dict[str, str],
    loop_df: pd.DataFrame,
    cols: Dict[str, str],
    keep_ids: set,
) -> Dict[str, str]:
    """
    Extract SAbDab H3 loops only for antibody_ids in keep_ids (RMSD<=threshold).
    """
    out = {}

    for _, row in loop_df.iterrows():
        ab = str(row[cols["antibody_id"]])
        if ab not in keep_ids:
            continue

        fasta_id = str(row[cols["fasta_header_id"]])

        # Try multiple key matching strategies: exact fasta_id, exact ab id, and
        # partial substring matches to handle prefix mismatches between the loop CSV
        # and FASTA headers (e.g. "1abc_H" vs "1abc").
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
        # Trim VH to the annotated length to avoid including the VL portion
        # if the FASTA record contains the concatenated Fv without a ':' separator.
        if vh_len > 0 and vh_len <= len(vh):
            vh = vh[:vh_len]

        try:
            s = int(row[cols["h3_start"]])
            e = int(row[cols["h3_end"]])
        except Exception:
            continue

        # Automatically detect whether the CSV uses 0-based or 1-based indexing
        # by checking which interpretation yields a plausible H3 slice.
        h3_0 = slice_inclusive(vh, s, e, one_based=False)
        h3_1 = slice_inclusive(vh, s, e, one_based=True)

        if h3_0 is None and h3_1 is None:
            continue
        if h3_0 is not None and h3_1 is None:
            one_based = False
        elif h3_0 is None and h3_1 is not None:
            one_based = True
        else:
            # Both slices valid; choose 1-based if it produces a CDR-plausible
            # length (5–35 aa), which is the biologically expected range for H3.
            if 5 <= len(h3_1) <= 35 and not (5 <= len(h3_0) <= 35):
                one_based = True
            else:
                one_based = False

        h3 = slice_inclusive(vh, s, e, one_based=one_based)
        h3 = clean_to_20aa(h3)
        if h3 and 0 < len(h3) < 60:
            out[ab] = h3

    return out


# ----------------------------
# Plots
# ----------------------------
def pooled_aa_freq(seqs: List[str]) -> np.ndarray:
    counts = Counter()
    for s in seqs:
        counts.update([aa for aa in s if aa in AA_SET])
    total = sum(counts[a] for a in AA_ORDER)
    return np.array([counts[a] / total if total > 0 else 0.0 for a in AA_ORDER], dtype=float)


def plot_length_hist(train_lens: List[int], gen_lens: List[int], out_path: str) -> None:
    plt.figure()

    bins = range(0, max(max(train_lens, default=0), max(gen_lens, default=0)) + 2)

    # Histograms
    plt.hist(train_lens, bins=bins, alpha=0.6, label="Training set")
    plt.hist(gen_lens, bins=bins, alpha=0.6, label="Baseline generated")

    # Means
    train_mean = np.mean(train_lens) if train_lens else 0
    gen_mean   = np.mean(gen_lens) if gen_lens else 0

    plt.axvline(train_mean, linestyle="--", linewidth=2)
    plt.axvline(gen_mean, linestyle="--", linewidth=2)

    # Annotate means (3 significant figures)
    ytop = plt.ylim()[1]

    plt.text(
        train_mean + 0.2,
        ytop * 0.9,
        f"{train_mean:.3g}",
        fontsize=14
    )

    plt.text(
        gen_mean + 0.2,
        ytop * 0.8,
        f"{gen_mean:.3g}",
        fontsize=14
    )

    plt.xlabel("H3 length (aa)")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_aa_freq(train_freq: np.ndarray, gen_freq: np.ndarray, out_path: str, train_label: str) -> None:
    plt.figure()
    x = np.arange(len(AA_ORDER))
    plt.bar(x - 0.2, train_freq, width=0.4, label=train_label)
    plt.bar(x + 0.2, gen_freq, width=0.4, label="Generated H3")
    plt.xticks(x, AA_ORDER)
    plt.xlabel("Amino acid")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_identity_hist(best_identities: List[float], out_path: str) -> None:
    plt.figure()

    finetuned_color = "#a8c8e8"

    bins = np.linspace(0, 1, 31)
    plt.hist(best_identities, bins=bins, alpha=0.85, color=finetuned_color)

    plt.xlim(0, 1.0)

    mean_identity = float(np.mean(best_identities)) if best_identities else 0.0
    plt.axvline(mean_identity, linestyle="--", linewidth=2, color=finetuned_color)

    ytop = plt.ylim()[1]
    plt.text(
        mean_identity + 0.02,
        ytop * 0.9 if ytop > 0 else 0.0,
        f"Mean = {mean_identity:.3g}",
    )

    plt.xlabel("identity")
    plt.ylabel("count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Alignment + identity
# ----------------------------
def build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = ALIGN_MATCH
    aligner.mismatch_score = ALIGN_MISMATCH
    aligner.open_gap_score = ALIGN_OPEN_GAP
    aligner.extend_gap_score = ALIGN_EXTEND_GAP
    return aligner


def _gapped_strings_from_blocks(a: str, b: str, aligned_blocks) -> Tuple[str, str]:
    """
    Reconstructs the gapped alignment strings from Biopython's aligned block
    representation, which gives contiguous matched segment coordinates rather
    than an explicit character-by-character alignment string.  This is needed
    because Biopython ≥1.80 no longer exposes alignment strings directly.
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


def alignment_global_identity(aligner: PairwiseAligner, a: str, b: str) -> Tuple[float, float]:
    """
    Global alignment score + FULL-LENGTH identity:
      identity = matches / max(len(a), len(b))
    where matches counts positions where both residues are non-gap and equal.
    """
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

    # Denominator is max length, not alignment length; this penalises
    # length mismatches and avoids inflating identity for short matches.
    denom = max(len(a), len(b))
    ident = matches / denom if denom > 0 else 0.0
    return ident, score


# ----------------------------
# Main
# ----------------------------
def main():
    print("[1/7] Load SAbDab Fv FASTA...")
    sabdab_fa = read_fasta_as_dict(SABDAB_TRAIN_FASTA)

    print(f"[2/7] Load SAbDab keep IDs where H3 RMSD <= {SABDAB_H3_RMSD_MAX}...")
    sabdab_keep_ids = load_keep_ids_from_h3_rmsd(SABDAB_RMSD_XLSX, threshold=SABDAB_H3_RMSD_MAX)
    print(f"  Keep IDs: {len(sabdab_keep_ids)}")

    print("[3/7] Extract SAbDab H3 loops (filtered by RMSD)...")
    loop_df = pd.read_csv(SABDAB_TRAIN_LOOP_CSV, sep=None, engine="python")
    cols = infer_loop_csv_cols(loop_df)
    sabdab_h3_map = extract_sabdab_h3s_filtered(sabdab_fa, loop_df, cols, sabdab_keep_ids)
    print(f"  SAbDab H3 kept after extraction: {len(sabdab_h3_map)}")

    train_h3s = list(sabdab_h3_map.values())
    if not train_h3s:
        raise RuntimeError("SAbDab H3 pool is empty after filtering. Check RMSD Excel + ID matching.")

    print("[4/7] Load generated H3 from FASTA (ANARCI extracted)...")
    gen_h3_map_raw = read_fasta_as_dict(GEN_H3_FASTA)
    gen_h3_map = {}
    for gid, h3 in gen_h3_map_raw.items():
        h3c = clean_to_20aa(h3)
        if h3c:
            gen_h3_map[gid] = h3c

    gen_h3s = list(gen_h3_map.values())
    print(f"  Generated H3 loaded: {len(gen_h3s)}")
    if not gen_h3s:
        raise RuntimeError(f"No generated H3 found in {GEN_H3_FASTA} (after cleaning).")

    print("[5/7] Length + AA composition plots...")
    plot_length_hist(
        [len(x) for x in train_h3s],
        [len(x) for x in gen_h3s],
        os.path.join(OUT_DIR, "length_hist.png"),
    )
    plot_aa_freq(
        pooled_aa_freq(train_h3s),
        pooled_aa_freq(gen_h3s),
        os.path.join(OUT_DIR, "aa_freq.png"),
        train_label="SAbDab H3 (filtered)",
    )

    print("[6/7] Nearest-neighbour search (generated -> SAbDab only, choose neighbour by IDENTITY)...")
    aligner = build_aligner()

    # Bucket SAbDab H3s by exact length for O(1) candidate lookup;
    # only sequences within ±LEN_WINDOW are aligned, avoiding O(N²) comparisons.
    bucket = defaultdict(list)  # length -> list[(antibody_id, h3)]
    for ab, h3 in sabdab_h3_map.items():
        bucket[len(h3)].append((ab, h3))

    rows = []
    best_identities = []

    for gid, gh3 in gen_h3_map.items():
        L = len(gh3)

        candidates = []
        for l2 in range(L - LEN_WINDOW, L + LEN_WINDOW + 1):
            candidates.extend(bucket.get(l2, []))
        # Safety fallback: if no candidates found within the length window
        # (can happen for very rare H3 lengths), search the full pool.
        if not candidates:
            candidates = list(sabdab_h3_map.items())

        best_ab = None
        best_th3 = None
        best_ident = -1.0
        best_score = float("-inf")

        for ab, th3 in candidates:
            ident, score = alignment_global_identity(aligner, gh3, th3)
            # Primary sort key: identity; secondary: alignment score (breaks ties
            # from floating-point equality at the EPS tolerance).
            if (ident > best_ident + EPS) or (abs(ident - best_ident) <= EPS and score > best_score):
                best_ident = ident
                best_score = score
                best_ab = ab
                best_th3 = th3

        best_identities.append(best_ident)
        rows.append({
            "gen_id": gid,
            "gen_h3": gh3,
            "gen_len": len(gh3),
            "best_sabdab_antibody_id": best_ab,
            "best_sabdab_h3": best_th3,
            "best_sabdab_len": len(best_th3) if best_th3 else np.nan,
            "best_global_score": best_score,
            "best_identity": best_ident,
            "n_candidates": len(candidates),
        })

    out_csv = os.path.join(OUT_DIR, "generated_best_match_sabdab_only.csv")
    pd.DataFrame(rows).sort_values("best_identity", ascending=False).to_csv(out_csv, index=False)
    plot_identity_hist(best_identities, os.path.join(OUT_DIR, "best_identity_hist.png"))

    print("[7/7] Done.")
    print("Outputs:", OUT_DIR)
    print("  - length_hist.png")
    print("  - aa_freq.png")
    print("  - best_identity_hist.png")
    print("  - generated_best_match_sabdab_only.csv")


if __name__ == "__main__":
    main()
