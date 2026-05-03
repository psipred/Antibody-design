#!/usr/bin/env python3
"""
Controlled-Identity Analysis: Training Identity vs CDR-H3 pLDDT
================================================================
Examines whether AlphaFold2 structural confidence (pLDDT) for generated CDR-H3
loops correlates with how similar those loops are to the training data — the
"controlled identity" question.

This is the pLDDT-based counterpart to colabfold_ensemble_controlled_identity.py,
which uses mean pairwise RMSD as the structural quality metric.  Using both
metrics together provides a more complete picture: RMSD measures inter-seed
consistency while pLDDT measures per-residue model confidence.

The central question: does the finetuned model produce high-confidence H3
structures even for loops that are novel (low identity to training data), or
does structural confidence drop off once sequences diverge from the training
distribution?

Pipeline overview
-----------------
1. Build the SAbDab reference H3 pool (same as the RMSD script):
   - Load Fv FASTA + loop-span coordinates.
   - Retain only antibodies with AlphaFold2 H3 RMSD ≤ 1.5 Å from experiment,
     ensuring the reference pool is structurally well-predicted.
   - Extract H3 subsequences.

2. For each generated antibody (baseline and finetuned):
   a. Parse the ANARCI Chothia-numbered output to get the H3 residue positions
      (Chothia positions 95–102) and their indices in the full heavy-chain
      sequence.
   b. Load the AlphaFold2 JSON score files (up to 5 seeds per antibody) and
      average the pLDDT arrays across seeds to get a per-position mean pLDDT.
   c. Index into the mean-pLDDT array using the H3 residue indices to obtain
      the H3-specific mean pLDDT and the fraction of residues above the
      "good confidence" threshold (80.0).
   d. Run nearest-neighbour identity search against the SAbDab reference pool.

3. Bin by NN identity (5% bins) and plot mean H3 pLDDT per bin as a bubble
   chart.

Key design choices
------------------
- ANARCI parsing is used instead of just reading sequences from FASTA because
  ANARCI provides Chothia position numbers, which are needed to precisely
  identify H3 residues without relying on fragile substring heuristics.
- pLDDT is averaged *across seeds first* (per-position mean), then the H3
  region is indexed.  This treats the ensemble as a single pooled prediction
  rather than treating each seed independently.
- The SCORES_PATTERN glob targets ranks 1–5 of the AlphaFold2 multimer v2
  model with seed 000; this matches the ColabFold output naming convention.

Inputs
------
- BASELINE_RUN_DIR   : Directory of ColabFold JSON score files for the baseline model.
- BASELINE_ANARCI    : ANARCI Chothia output for the baseline generated sequences.
- FINETUNED_RUN_DIR  : ColabFold JSON score files for the finetuned model.
- FINETUNED_ANARCI   : ANARCI Chothia output for the finetuned model.
- SABDAB_TRAIN_FASTA    : SAbDab Fv FASTA used during training.
- SABDAB_TRAIN_LOOP_CSV : CSV of per-antibody H3 start/end coordinates.
- SABDAB_RMSD_XLSX      : Excel file of per-antibody AlphaFold2 H3 RMSD values.

Outputs (written to OUTPUT_DIR)
--------------------------------
- per_entry_h3_identity_plddt.tsv : Per-antibody table.
- binned_mean_h3_plddt.tsv        : Per-bin summary statistics.
- mean_h3_plddt_vs_training_identity_baseline_vs_finetuned.png
- run_log.txt

"""

import os
import re
import glob
import json
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
BASELINE_RUN_DIR = "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/iggen"
BASELINE_ANARCI = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_anarci_chothia.txt"

FINETUNED_RUN_DIR = "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/oas_v5"
FINETUNED_ANARCI = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v5/generated_anarci_chothia.txt"

SABDAB_TRAIN_FASTA = "/home/alanwu/Documents/iggen_model/data/single_fv_pdb.fasta"
SABDAB_TRAIN_LOOP_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"
SABDAB_RMSD_XLSX = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"

OUTPUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/plddt/oas_v5_identity_vs_plddt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_PATH = os.path.join(OUTPUT_DIR, "mean_h3_plddt_vs_training_identity_baseline_vs_finetuned.png")
RAW_TABLE_PATH = os.path.join(OUTPUT_DIR, "per_entry_h3_identity_plddt.tsv")
BIN_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "binned_mean_h3_plddt.tsv")
LOG_PATH = os.path.join(OUTPUT_DIR, "run_log.txt")

# pLDDT ≥ 80 is the standard AlphaFold2 threshold for a "confident" prediction
GOOD_THRESHOLD = 80.0
MIN_COUNT_PER_BIN = 3
SABDAB_H3_RMSD_MAX = 1.5

# 5% wide identity bins spanning [0, 1]
BIN_EDGES = np.arange(0.0, 1.0001, 0.05)

# ColabFold names score files with ranks 001–005 for the five seeds
SCORES_PATTERN = "*_scores_rank_00[1-5]_alphafold2_multimer_v2_model_*_seed_000.json"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

ALIGN_MATCH = 2
ALIGN_MISMATCH = -1
ALIGN_OPEN_GAP = -10
ALIGN_EXTEND_GAP = -0.5
EPS = 1e-12


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
    """
    Split on ':' then clean each chain.
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


# =========================================================
# LOOP CSV / RMSD FILTER HELPERS
# =========================================================
def infer_loop_csv_cols(df: pd.DataFrame):
    """
    Robustly locate required columns by exact match first, regex fallback.
    Handles minor naming differences across dataset versions.
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
    if one_based:
        start -= 1
        end -= 1
    if start < 0 or end < start or end >= len(seq):
        return None
    return seq[start:end + 1]


def load_keep_ids_from_h3_rmsd(xlsx_path: str, threshold: float = 1.5):
    """
    Keep antibody ids where H3 RMSD <= threshold.
    """
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

    # Shortest column name heuristic to pick the primary RMSD column
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
    """
    Extract SAbDab H3 loops only for antibody_ids in keep_ids.
    Uses the same logic as your reference script.
    """
    out = {}

    for _, row in loop_df.iterrows():
        ab = str(row[cols["antibody_id"]]).strip()
        if ab not in keep_ids:
            continue

        fasta_id = str(row[cols["fasta_header_id"]]).strip()

        # Attempt direct FASTA key lookup; fall back to fuzzy matching to
        # handle ID format discrepancies between the CSV and FASTA files
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

        # Trim to VH length so that H3 coordinates (which are VH-relative)
        # index correctly into the isolated VH sequence
        if vh_len > 0 and vh_len <= len(vh):
            vh = vh[:vh_len]

        try:
            s = int(row[cols["h3_start"]])
            e = int(row[cols["h3_end"]])
        except Exception:
            continue

        # Attempt both indexing conventions and choose the one yielding a
        # biologically plausible H3 length (5–35 residues)
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

    return sorted(set(train_h3s))


# =========================================================
# ANARCI PARSING FOR GENERATED SETS
# =========================================================
def is_entry_header(line: str):
    """
    Identify ANARCI output lines that introduce a new sequence entry.
    ANARCI header lines begin with "# " but many are internal metadata
    lines (version, domain info, scheme); we filter those out and return
    the entry ID string only for genuine sequence-entry headers.
    """
    if not line.startswith("# "):
        return None

    rest = line[2:].strip()

    # Skip ANARCI's own metadata comment lines
    if rest.startswith("ANARCI"):
        return None
    if rest.startswith("Domain"):
        return None
    if rest.startswith("Most"):
        return None
    if rest.startswith("Scheme"):
        return None
    if rest.startswith("|") or rest.startswith("-") or rest.startswith("species"):
        return None

    entry_id = rest.split()[0]
    if re.match(r"^[A-Za-z0-9_.:/+-]+$", entry_id):
        return entry_id
    return None


def parse_anarci_chothia_with_h3(filepath):
    """
    Returns:
        entry_id -> {
            "heavy_seq": str,
            "h3_indices": list of 0-based indices in heavy-chain sequence,
            "h3_seq": str
        }

    Chothia H3 = heavy positions 95–102 inclusive, insertions supported.
    """
    results = {}

    current_id = None
    seq_index = 0
    heavy_seq_chars = []
    h3_indices = []
    h3_chars = []

    def flush():
        """Save the completed entry and reset per-entry state."""
        nonlocal current_id, seq_index, heavy_seq_chars, h3_indices, h3_chars
        if current_id is not None:
            results[current_id] = {
                "heavy_seq": "".join(heavy_seq_chars),
                "h3_indices": h3_indices.copy(),
                "h3_seq": "".join(h3_chars),
            }
        current_id = None
        seq_index = 0
        heavy_seq_chars = []
        h3_indices = []
        h3_chars = []

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            eid = is_entry_header(line)
            if eid is not None:
                flush()
                current_id = eid
                continue

            if current_id is None:
                continue

            # Only process heavy-chain residue lines (start with "H ")
            if not line.startswith("H "):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] != "H":
                continue

            try:
                pos = int(parts[1])
            except ValueError:
                continue

            # Last token is the amino acid; earlier tokens are chain/position/insertion
            aa = parts[2] if len(parts) == 3 else parts[-1]
            # "-" and "." represent gaps/deletions; skip to avoid inflating sequence
            if aa in ("-", ".", "X"):
                continue

            heavy_seq_chars.append(aa)

            # Chothia numbering: H3 loop spans positions 95 through 102 (inclusive),
            # with insertion codes handled transparently since we read the integer
            # part only
            if 95 <= pos <= 102:
                h3_indices.append(seq_index)
                h3_chars.append(aa)

            seq_index += 1

    flush()
    results.pop(None, None)
    return results


# =========================================================
# GENERATED SET pLDDT
# =========================================================
def load_entry_mean_plddt_from_run(run_dir):
    """
    For each antibody entry, average the per-residue pLDDT arrays across
    all available seed JSON files to get a single mean pLDDT profile.

    Truncates all arrays to the shortest length before stacking so that
    occasional off-by-one length differences between seeds do not cause
    errors — such differences arise when ColabFold crops the padding
    differently across seeds.
    """
    score_glob = os.path.join(run_dir, SCORES_PATTERN)
    json_files = sorted(glob.glob(score_glob))
    if not json_files:
        raise FileNotFoundError(f"No score JSONs found under: {run_dir}")

    entry_to_files = defaultdict(list)
    # Extract the entry name from the ColabFold filename convention:
    # "<entry>_scores_rank_00X_alphafold2_multimer_v2_model_Y_seed_000.json"
    entry_regex = re.compile(r"^(?P<entry>.+?)_scores_rank_00[1-5]_")

    for fpath in json_files:
        base = os.path.basename(fpath)
        m = entry_regex.match(base)
        if m:
            entry_to_files[m.group("entry")].append(fpath)

    entry_to_mean_plddt = {}

    for entry_id, files in sorted(entry_to_files.items()):
        arrs = []
        for fp in sorted(files):
            with open(fp, "r") as f:
                data = json.load(f)
            if "plddt" in data:
                arrs.append(np.array(data["plddt"], dtype=float))

        if not arrs:
            continue

        # Align lengths before stacking: minor length variation can occur at
        # sequence ends due to padding removal
        min_len = min(len(arr) for arr in arrs)
        mat = np.vstack([arr[:min_len] for arr in arrs])
        # axis=0 averages across seeds; result is a single per-position vector
        entry_to_mean_plddt[entry_id] = np.mean(mat, axis=0)

    return entry_to_mean_plddt


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
    Reconstruct explicit gapped alignment strings from BioPython's block
    coordinate representation (required for BioPython ≥ 1.80).
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
    """
    identity = matches / max(len(a), len(b))
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

    denom = max(len(a), len(b))
    ident = matches / denom if denom > 0 else 0.0
    return ident, score


def nearest_training_identity(query_h3, training_h3s, aligner):
    best_ident = -1.0
    best_score = float("-inf")
    best_train_h3 = None

    for train_h3 in training_h3s:
        ident, score = alignment_global_identity(aligner, query_h3, train_h3)
        # EPS prevents floating-point near-ties from picking an arbitrary winner
        if (ident > best_ident + EPS) or (abs(ident - best_ident) <= EPS and score > best_score):
            best_ident = ident
            best_score = score
            best_train_h3 = train_h3

    return best_ident, best_train_h3


# =========================================================
# BUILD DATASET
# =========================================================
def build_dataset(group_label, run_dir, anarci_file, training_h3s, aligner):
    log(f"\n=== Processing {group_label} ===")
    log(f"Run dir: {run_dir}")
    log(f"ANARCI:  {anarci_file}")

    anarci_data = parse_anarci_chothia_with_h3(anarci_file)
    log(f"Parsed ANARCI entries: {len(anarci_data)}")

    entry_to_mean_plddt = load_entry_mean_plddt_from_run(run_dir)
    log(f"Parsed JSON entries:   {len(entry_to_mean_plddt)}")

    rows = []
    skipped = []

    for entry_id, mean_plddt_vec in sorted(entry_to_mean_plddt.items()):
        if entry_id not in anarci_data:
            skipped.append((entry_id, "No ANARCI entry"))
            continue

        h3_indices = anarci_data[entry_id]["h3_indices"]
        h3_seq = clean_to_20aa(anarci_data[entry_id]["h3_seq"])

        if not h3_indices:
            skipped.append((entry_id, "No H3 residues found"))
            continue

        if not h3_seq:
            skipped.append((entry_id, "Empty H3 sequence"))
            continue

        # Guard against index out of bounds: the ANARCI-derived indices must
        # fall within the pLDDT vector length (rare mismatches can occur if
        # ColabFold processes a truncated or differently-padded sequence)
        if max(h3_indices) >= len(mean_plddt_vec):
            skipped.append((entry_id, f"Index mismatch: max(H3 idx)={max(h3_indices)} >= pLDDT len={len(mean_plddt_vec)}"))
            continue

        # Index the mean-pLDDT vector with the H3 residue positions to extract
        # only the CDR-H3 confidence scores
        h3_plddt = mean_plddt_vec[h3_indices]
        h3_mean = float(np.mean(h3_plddt))
        h3_good_pct = float(100.0 * np.mean(h3_plddt > GOOD_THRESHOLD))

        nn_ident, nn_train_h3 = nearest_training_identity(h3_seq, training_h3s, aligner)

        rows.append({
            "entry": entry_id,
            "group": group_label,
            "h3_seq": h3_seq,
            "h3_len": len(h3_seq),
            "h3_mean_plddt": h3_mean,
            "h3_good_pct": h3_good_pct,
            "nearest_train_identity": float(nn_ident),
            "nearest_train_h3": nn_train_h3,
        })

    log(f"Kept:    {len(rows)}")
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
    Aggregate per-entry pLDDT values within each identity bin.
    The last bin is closed on the right to include identity = 1.0.
    """
    summaries = []
    groups = sorted(set(r["group"] for r in rows))

    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        identities = np.array([r["nearest_train_identity"] for r in group_rows], dtype=float)
        plddts = np.array([r["h3_mean_plddt"] for r in group_rows], dtype=float)

        for i in range(len(bin_edges) - 1):
            left = float(bin_edges[i])
            right = float(bin_edges[i + 1])

            if i == len(bin_edges) - 2:
                mask = (identities >= left) & (identities <= right)
            else:
                mask = (identities >= left) & (identities < right)

            vals = plddts[mask]
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
                "mean_h3_plddt": mean_val,
                "sd_h3_plddt": sd_val,
                "se_h3_plddt": se_val,
            })

    return summaries


def plot_bubble_identity_plddt(summaries, out_path):
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
        y = np.array([d["mean_h3_plddt"] for d in grp], dtype=float)
        n = np.array([d["n"] for d in grp], dtype=float)
        se = np.array([d["se_h3_plddt"] for d in grp], dtype=float)

        # Bubble area ∝ sample count: visually communicates statistical weight
        sizes = 40 + 25 * n
        color = color_map[group]

        plt.plot(x, y, linewidth=1.5, alpha=0.6, label=label_map[group], color=color)
        plt.scatter(x, y, s=sizes, alpha=0.55, color=color)
        plt.errorbar(x, y, yerr=se, fmt="none", elinewidth=1, capsize=3, alpha=0.7, color=color)

        plotted_any = True

    if not plotted_any:
        raise ValueError("No bins had enough points to plot.")

    plt.xlabel("Nearest neighbour identity")
    plt.ylabel("Mean H3 pLDDT")
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
        "h3_mean_plddt",
        "h3_good_pct",
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
        "mean_h3_plddt",
        "sd_h3_plddt",
        "se_h3_plddt",
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
        run_dir=BASELINE_RUN_DIR,
        anarci_file=BASELINE_ANARCI,
        training_h3s=training_h3s,
        aligner=aligner,
    )

    finetuned_rows = build_dataset(
        group_label="finetuned",
        run_dir=FINETUNED_RUN_DIR,
        anarci_file=FINETUNED_ANARCI,
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

    plot_bubble_identity_plddt(summaries, PLOT_PATH)
    log(f"Plot written to: {PLOT_PATH}")

    for group in ["baseline", "finetuned"]:
        grp = [r for r in all_rows if r["group"] == group]
        if grp:
            mean_plddt = np.mean([r["h3_mean_plddt"] for r in grp])
            mean_ident = np.mean([r["nearest_train_identity"] for r in grp])
            log(f"{group}: n={len(grp)}, mean H3 pLDDT={mean_plddt:.2f}, mean nearest-training identity={mean_ident:.3f}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    log(f"Log written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
