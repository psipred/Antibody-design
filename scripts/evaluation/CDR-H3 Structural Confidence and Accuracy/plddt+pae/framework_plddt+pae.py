#!/usr/bin/env python3
"""
plddt+pae.py
============
Computes two structural confidence metrics from ColabFold outputs for
CDR-H3 and its flanking framework regions, then compares Baseline vs
Finetuned antibody models via violin plots and a summary CSV.

Overview
--------
ColabFold (AlphaFold2 Multimer) produces per-residue pLDDT scores and
pairwise Predicted Aligned Error (PAE) matrices saved as JSON files.
This script extracts two complementary signals for each generated antibody:

  1. Combined flank mean pLDDT (H90–H94 + H103–H107)
       The Chothia framework residues immediately flanking CDR-H3 act as
       anchor points that determine loop geometry. High pLDDT here suggests
       the model is confident about the structural context into which H3 is
       embedded, even when H3 itself is uncertain.

  2. H3 mean PAE to all residues
       PAE[i, j] is AlphaFold's estimated positional error for residue i
       when residue j is used as the alignment reference frame. Averaging
       PAE over H3 rows and all columns measures how well the model places
       the H3 loop relative to the rest of the structure — a lower value
       indicates higher inter-domain confidence.

ANARCI Chothia numbering is used to map sequence positions to the canonical
H3 and flanking-residue index ranges. Rather than hardcoding sequence offsets
(which vary by antibody length), the per-entry ANARCI files provide the exact
sequence indices for each named region.

Inputs
------
  Per-dataset ColabFold output directories (JSON score files and PAE files).
  Per-dataset ANARCI files with Chothia-numbered heavy-chain annotations.
  Configured via DATASETS dict.

Outputs
-------
  OUTPUT_DIR/per_entry_metrics.csv        — per-entry table of all metrics
  OUTPUT_DIR/flank_plddt_h3_pae_summary.txt — printed summary statistics
  OUTPUT_DIR/violin_flank_mean_plddt.png  — violin: flank pLDDT Baseline vs Finetuned
  OUTPUT_DIR/violin_h3_mean_pae_to_all_residues.png — violin: H3 PAE Baseline vs Finetuned


Update DATASETS and OUTPUT_DIR to point to your local directories.
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({'font.size': 14})
from scipy.stats import mannwhitneyu

# =========================
# CONFIG
# =========================
DATASETS = {
    "iggen": {
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/iggen",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_anarci_chothia.txt",
    },
    "finetuned": {
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/oas_v6",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v6/generated_anarci_chothia.txt",
    },
}

OUTPUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/plddt_pae/oas_v6_flank_plddt_h3_pae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_TXT = os.path.join(OUTPUT_DIR, "flank_plddt_h3_pae_summary.txt")
PER_ENTRY_CSV = os.path.join(OUTPUT_DIR, "per_entry_metrics.csv")

VIOLIN_FLANK_PLOT = os.path.join(OUTPUT_DIR, "violin_flank_mean_plddt.png")
VIOLIN_H3_PAE_PLOT = os.path.join(OUTPUT_DIR, "violin_h3_mean_pae_to_all_residues.png")

# Glob patterns that match ColabFold's output file naming convention.
# Ranks 1–5 are selected; rank 000 would be the unrelaxed best, but ColabFold
# writes individual model scores for each of its 5 model instances.
SCORES_PATTERN = "*_scores_rank_00[1-5]_alphafold2_multimer_v2_model_*_seed_000.json"
PAE_PATTERN = "*_predicted_aligned_error*.json"

# Chothia heavy-chain regions used in the analysis.
# Left flank (H90–H94) and right flank (H103–H107) are the framework residues
# directly adjacent to the H3 loop on either side.
LEFT_FLANK_START = 90
LEFT_FLANK_END   = 94

H3_START = 95
H3_END   = 102

RIGHT_FLANK_START = 103
RIGHT_FLANK_END   = 107

# =========================
# Logging
# =========================
# All output is both printed to stdout and accumulated for writing to the
# summary text file, so the run can be reproduced from the saved log.
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# =========================
# STEP 1 — Parse ANARCI file
# =========================
def is_entry_header(line: str):
    """
    Returns entry_id if this line is a true entry header like '# ab_1',
    otherwise returns None.

    ANARCI output files contain many lines starting with '# ' that are
    metadata (version info, table headers, species annotations). This
    function distinguishes genuine entry-name headers from those metadata
    lines by checking for known prefixes and validating the identifier format.
    """
    if not line.startswith("# "):
        return None

    rest = line[2:].strip()

    # Ignore ANARCI metadata/comment lines
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
    # Require the identifier to be alphanumeric (plus underscores, dots,
    # hyphens). This guards against partially-parsed comment lines.
    if re.match(r"^[A-Za-z0-9_.-]+$", entry_id):
        return entry_id
    return None


def parse_anarci_chothia(filepath):
    """
    Parse an ANARCI Chothia-numbered output file and extract the 0-based
    sequence indices for the H3 loop and its flanking framework residues.

    ANARCI assigns Chothia position numbers to each residue in the input
    sequence. Positions with '-' (insertion gaps) or 'X' (unknown) are
    ignored when computing sequence indices so that the index stored here
    directly indexes into the pLDDT array returned by ColabFold (which
    contains only real residues, not gaps).

    Parameters
    ----------
    filepath : str
        Path to the ANARCI output text file for the heavy chain.

    Returns
    -------
    dict
        Maps entry_id -> {
            "left_flank_indices":    list of 0-based indices for H90–H94,
            "h3_indices":            list of 0-based indices for H95–H102,
            "right_flank_indices":   list of 0-based indices for H103–H107,
            "combined_flank_indices": left + right flank indices combined,
        }
    """
    results = {}

    current_id = None
    seq_index = 0

    left_flank_indices = []
    h3_indices = []
    right_flank_indices = []

    def flush():
        """Commit the current entry's accumulated indices to results and reset state."""
        nonlocal current_id, seq_index, left_flank_indices, h3_indices, right_flank_indices
        if current_id is not None:
            results[current_id] = {
                "left_flank_indices": left_flank_indices.copy(),
                "h3_indices": h3_indices.copy(),
                "right_flank_indices": right_flank_indices.copy(),
                # Combined flank is the union of left and right, in order,
                # which is used when computing a single pLDDT value for the
                # entire H3-flanking context.
                "combined_flank_indices": left_flank_indices.copy() + right_flank_indices.copy(),
            }
        current_id = None
        seq_index = 0
        left_flank_indices = []
        h3_indices = []
        right_flank_indices = []

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            eid = is_entry_header(line)
            if eid is not None:
                # Starting a new entry: flush the previous one first.
                flush()
                current_id = eid
                continue

            if current_id is None:
                continue

            # Only process heavy-chain lines (lines starting with "H ").
            # Light chain (L) lines are present in the file but not needed
            # because all regions of interest are heavy-chain positions.
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

            aa = parts[-1]
            # Skip gap characters — they represent alignment padding that has
            # no corresponding residue in the actual sequence.
            if aa in ("-", ".", "X"):
                continue

            if LEFT_FLANK_START <= pos <= LEFT_FLANK_END:
                left_flank_indices.append(seq_index)

            if H3_START <= pos <= H3_END:
                h3_indices.append(seq_index)

            if RIGHT_FLANK_START <= pos <= RIGHT_FLANK_END:
                right_flank_indices.append(seq_index)

            # Increment only for real (non-gap) residues so that seq_index
            # stays aligned with the pLDDT array from ColabFold.
            seq_index += 1

    flush()
    # If a file has no valid entry headers, the None key may have been set;
    # remove it to avoid keying downstream dicts with None.
    results.pop(None, None)
    return results

# =========================
# STEP 2 — Group files by entry
# =========================
def group_score_files(run_dir):
    """
    Discover ColabFold score JSON files in run_dir and group them by entry ID.

    ColabFold writes one score file per model run per entry, e.g.:
        ab_1_scores_rank_001_alphafold2_multimer_v2_model_1_seed_000.json

    The regex strips the rank/model suffix to recover the entry name, then
    collects all files belonging to the same entry.

    Returns
    -------
    dict : entry_id -> list of file paths
    """
    pattern = os.path.join(run_dir, SCORES_PATTERN)
    files = sorted(glob.glob(pattern))
    entry_to_files = defaultdict(list)

    entry_regex = re.compile(r"^(?P<entry>.+?)_scores_rank_00[1-5]_")

    for fpath in files:
        base = os.path.basename(fpath)
        m = entry_regex.match(base)
        if m:
            entry_to_files[m.group("entry")].append(fpath)

    return entry_to_files


def group_pae_files(run_dir):
    """
    Discover ColabFold PAE JSON files in run_dir and group them by entry ID.

    Returns
    -------
    dict : entry_id -> list of file paths
    """
    pattern = os.path.join(run_dir, PAE_PATTERN)
    files = sorted(glob.glob(pattern))
    entry_to_files = defaultdict(list)

    entry_regex = re.compile(r"^(?P<entry>.+?)_predicted_aligned_error")

    for fpath in files:
        base = os.path.basename(fpath)
        m = entry_regex.match(base)
        if m:
            entry_to_files[m.group("entry")].append(fpath)

    return entry_to_files

# =========================
# STEP 3 — Load JSON metrics
# =========================
def load_mean_plddt(score_files):
    """
    Load pLDDT arrays from multiple ColabFold score JSONs and return their
    element-wise mean across model runs.

    Averaging ranks 1–5 rather than using the best rank alone reduces
    sensitivity to which particular model instance happened to score highest,
    giving a more representative confidence estimate for the sequence.

    Parameters
    ----------
    score_files : list of str
        Paths to ColabFold score JSON files for one entry.

    Returns
    -------
    numpy.ndarray or None
        1-D array of mean per-residue pLDDT, or None if no valid arrays found.
    """
    plddt_arrays = []

    for fp in sorted(score_files):
        with open(fp) as f:
            data = json.load(f)

        if "plddt" not in data:
            continue

        arr = np.array(data["plddt"], dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            continue

        plddt_arrays.append(arr)

    if not plddt_arrays:
        return None

    # Trim to the shortest array before stacking in case models produce
    # slightly different lengths (can happen with padding).
    min_len = min(arr.shape[0] for arr in plddt_arrays)
    mat = np.vstack([arr[:min_len] for arr in plddt_arrays])
    return np.mean(mat, axis=0)


def load_mean_pae_matrix(pae_files):
    """
    Load PAE matrices from multiple ColabFold PAE JSONs and return their
    element-wise mean across model runs.

    Parameters
    ----------
    pae_files : list of str
        Paths to ColabFold PAE JSON files for one entry.

    Returns
    -------
    numpy.ndarray or None
        2-D array (n_residues x n_residues) of mean PAE in Angstroms,
        or None if no valid matrices found.
    """
    pae_mats = []

    for fp in sorted(pae_files):
        with open(fp) as f:
            data = json.load(f)

        if "predicted_aligned_error" not in data:
            continue

        mat = np.array(data["predicted_aligned_error"], dtype=float)
        if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
            continue

        pae_mats.append(mat)

    if not pae_mats:
        return None

    # Trim to the minimum shared dimensions before stacking, for the same
    # reason as in load_mean_plddt.
    min_rows = min(m.shape[0] for m in pae_mats)
    min_cols = min(m.shape[1] for m in pae_mats)
    stacked = np.stack([m[:min_rows, :min_cols] for m in pae_mats], axis=0)
    return np.mean(stacked, axis=0)

# =========================
# STEP 4 — Metric helpers
# =========================
def safe_mean_from_indices(arr, indices):
    """
    Return the mean of arr at the given indices, or NaN if indices are empty
    or out of bounds. NaN propagates gracefully into pandas and the summary
    stats, avoiding silent zero-biasing of results.
    """
    if arr is None or len(indices) == 0:
        return np.nan
    if max(indices) >= len(arr):
        return np.nan
    return float(np.mean(arr[indices]))


def safe_mean_pae_h3_to_all(pae_mat, h3_indices):
    """
    Mean PAE from H3 residues to all residues:
        mean( PAE[h3_indices, :] )

    This measures, on average, how uncertain AlphaFold is about the position
    of each H3 residue relative to every other residue in the structure.
    A lower value means the H3 loop is placed with higher inter-domain
    confidence relative to the rest of the antibody.

    Parameters
    ----------
    pae_mat : numpy.ndarray or None
        2-D PAE matrix (n_residues x n_residues).
    h3_indices : list of int
        0-based row indices corresponding to H3 residues.

    Returns
    -------
    float or NaN
    """
    if pae_mat is None or len(h3_indices) == 0:
        return np.nan

    n_rows, _ = pae_mat.shape
    if max(h3_indices) >= n_rows:
        return np.nan

    sub = pae_mat[h3_indices, :]
    if sub.size == 0:
        return np.nan

    return float(np.mean(sub))

# =========================
# STEP 5 — Plotting
# =========================
def plot_violin_custom(baseline_vals, finetuned_vals, outpath,
                       ylabel, title, unit=None, add_significance=False):
    """
    Render a two-group violin plot (Baseline vs Finetuned) with annotated
    medians and an optional Mann-Whitney U significance bracket.

    Parameters
    ----------
    baseline_vals : array-like
        Per-entry metric values for the baseline model.
    finetuned_vals : array-like
        Per-entry metric values for the finetuned model.
    outpath : str
        Output file path for the PNG.
    ylabel : str
        Y-axis label.
    title : str
        Plot title (not rendered on the axes, kept for reference).
    unit : str or None
        If "A", the median annotation is suffixed with " Å" (Angstroms).
    add_significance : bool
        If True, compute and annotate a Mann-Whitney U significance bracket.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    def format_sigfig(x, sig=3):
        """Format number to N significant figures."""
        if x == 0 or np.isnan(x):
            return "0"
        return f"{x:.{sig}g}"

    fig, ax = plt.subplots(figsize=(5, 6))

    data   = [baseline_vals, finetuned_vals]
    labels = ["Baseline", "Finetuned"]
    colors = ["#f5c87a", "#a8c8e8"]

    parts = ax.violinplot(
        data,
        positions=[1, 2],
        widths=0.6,
        showmedians=False,  # Drawn manually so we can annotate with value
        showextrema=False   # Hide min/max lines for a cleaner appearance
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)

    # Median annotation (3 significant figures)
    for vals, pos in zip(data, [1, 2]):
        if len(vals) == 0:
            continue

        median = np.median(vals)
        text = format_sigfig(median, 3)

        if unit == "A":
            text += " Å"

        # Adaptive vertical offset: scale with the spread of the data so the
        # annotation doesn't overlap the violin body at different value scales.
        offset = max(np.std(vals) * 0.1, 0.02)

        ax.text(
            pos, median + offset, text,
            ha="center", va="bottom", fontsize=14
        )

    if add_significance and len(baseline_vals) > 0 and len(finetuned_vals) > 0:
        _, pvalue = mannwhitneyu(baseline_vals, finetuned_vals, alternative="two-sided")
        if pvalue < 0.001:
            sig_label = "***"
        elif pvalue < 0.01:
            sig_label = "**"
        elif pvalue < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        # Position the bracket just above the tallest data point, scaling the
        # gap and bracket height proportionally to the y-axis range so the
        # annotation looks correct at different metric scales (pLDDT 0–100
        # vs PAE 0–30 Å use very different absolute values).
        y_data_max = max(np.max(np.asarray(vals, dtype=float)) for vals in data if len(vals) > 0)
        y_range = y_data_max if y_data_max > 0 else 1.0
        bracket_y = y_data_max + 0.06 * y_range
        bracket_h = 0.03 * y_range
        text_y = bracket_y + bracket_h + 0.01 * y_range

        ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
        ax.text(1.5, text_y, sig_label, ha="center", va="bottom", fontsize=14)
        log(f"Mann-Whitney U (Baseline vs Finetuned): p={pvalue:.6g} ({sig_label})")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

# =========================
# STEP 6 — Dataset processing
# =========================
def process_dataset(dataset_name, run_dir, anarci_file):
    """
    End-to-end processing for one dataset: parse ANARCI, load ColabFold
    JSON files, extract pLDDT and PAE metrics for H3 and flanking residues,
    and return a tidy DataFrame.

    Parameters
    ----------
    dataset_name : str
        Identifier used to label rows in the output DataFrame (e.g. "iggen").
    run_dir : str
        Directory containing ColabFold JSON output files.
    anarci_file : str
        Path to the ANARCI Chothia-numbered output file for this dataset.

    Returns
    -------
    pandas.DataFrame
        One row per successfully processed entry with columns for each metric.
    """
    log(f"\n=== Processing dataset: {dataset_name} ===")
    log(f"Run dir:     {run_dir}")
    log(f"ANARCI file: {anarci_file}")

    anarci_data = parse_anarci_chothia(anarci_file)
    log(f"✅ Parsed ANARCI data for {len(anarci_data)} entries")

    if len(anarci_data) == 0:
        raise ValueError(f"❌ No ANARCI entries parsed for {dataset_name}")

    score_groups = group_score_files(run_dir)
    pae_groups = group_pae_files(run_dir)

    log(f"✅ Parsed {len(score_groups)} entries from score JSONs")
    log(f"✅ Parsed {len(pae_groups)} entries from PAE JSONs")

    # Union of entries across both file types: some entries may have pLDDT but
    # no PAE file, or vice versa; we still want to capture what we can.
    all_entries = sorted(set(score_groups.keys()) | set(pae_groups.keys()))

    results = []
    skipped = []

    for entry_id in all_entries:
        if entry_id not in anarci_data:
            # If ANARCI has no annotation for this entry we cannot determine
            # which sequence positions correspond to H3, so skip it.
            skipped.append((entry_id, "No ANARCI data"))
            continue

        region_data = anarci_data[entry_id]
        left_flank_indices = region_data["left_flank_indices"]
        h3_indices = region_data["h3_indices"]
        right_flank_indices = region_data["right_flank_indices"]
        combined_flank_indices = region_data["combined_flank_indices"]

        if not h3_indices:
            skipped.append((entry_id, "No H3 residues found"))
            continue

        mean_plddt = None
        if entry_id in score_groups:
            mean_plddt = load_mean_plddt(score_groups[entry_id])

        mean_pae = None
        if entry_id in pae_groups:
            mean_pae = load_mean_pae_matrix(pae_groups[entry_id])

        if mean_plddt is None and mean_pae is None:
            skipped.append((entry_id, "No usable pLDDT or PAE data"))
            continue

        # Extract per-region metrics using the sequence indices from ANARCI.
        # safe_mean_from_indices returns NaN when a region has no residues
        # (e.g. very short H3), so downstream stats handle missing data cleanly.
        left_flank_mean_plddt = safe_mean_from_indices(mean_plddt, left_flank_indices)
        right_flank_mean_plddt = safe_mean_from_indices(mean_plddt, right_flank_indices)
        combined_flank_mean_plddt = safe_mean_from_indices(mean_plddt, combined_flank_indices)
        h3_mean_plddt = safe_mean_from_indices(mean_plddt, h3_indices)

        h3_mean_pae_to_all_residues = safe_mean_pae_h3_to_all(mean_pae, h3_indices)

        results.append({
            "dataset": dataset_name,
            "entry": entry_id,
            "n_left_flank_residues": len(left_flank_indices),
            "n_h3_residues": len(h3_indices),
            "n_right_flank_residues": len(right_flank_indices),
            "left_flank_mean_plddt": left_flank_mean_plddt,
            "right_flank_mean_plddt": right_flank_mean_plddt,
            "combined_flank_mean_plddt": combined_flank_mean_plddt,
            "h3_mean_plddt": h3_mean_plddt,
            "h3_mean_pae_to_all_residues": h3_mean_pae_to_all_residues,
        })

    log(f"Kept:   {len(results)} entries")
    log(f"Skipped:{len(skipped)} entries")

    if skipped:
        log("\nSkipped examples:")
        for e, r in skipped[:10]:
            log(f"  {e}: {r}")

    return pd.DataFrame(results)

# =========================
# STEP 7 — Summary helper
# =========================
def summarise_metric(df, metric_col, pretty_name):
    """
    Print and log mean, standard deviation, and median for a metric column,
    split by dataset label.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined results DataFrame from process_dataset.
    metric_col : str
        Column name to summarise.
    pretty_name : str
        Human-readable metric name for the log output.
    """
    log(f"\n--- {pretty_name} ---")
    for dataset_name, label in [("iggen", "Baseline"), ("finetuned", "Finetuned")]:
        vals = df.loc[df["dataset"] == dataset_name, metric_col].dropna().values
        if len(vals) == 0:
            log(f"{label}: no data")
        else:
            log(
                f"{label}: n={len(vals)}, "
                f"mean={np.mean(vals):.3f}, "
                f"sd={np.std(vals):.3f}, "
                f"median={np.median(vals):.3f}"
            )

# =========================
# STEP 8 — Run
# =========================
all_results = []

for dataset_name, cfg in DATASETS.items():
    df = process_dataset(
        dataset_name=dataset_name,
        run_dir=cfg["run_dir"],
        anarci_file=cfg["anarci_file"],
    )
    all_results.append(df)

if len(all_results) == 0 or all(df.empty for df in all_results):
    raise ValueError("❌ No valid results produced for any dataset.")

all_results = pd.concat(all_results, ignore_index=True)

# Save per-entry CSV
all_results.to_csv(PER_ENTRY_CSV, index=False)
log(f"\n✅ Per-entry CSV written to: {PER_ENTRY_CSV}")

# Summaries
summarise_metric(
    all_results,
    "combined_flank_mean_plddt",
    "Combined flank mean pLDDT (H90–H94 + H103–H107)"
)
summarise_metric(
    all_results,
    "h3_mean_pae_to_all_residues",
    "H3 mean PAE to all residues"
)

# ---- Plot 1: flank mean pLDDT ----
# The flank regions are used rather than H3 itself because H3 pLDDT alone can
# be misleading — a loop confidently placed in the wrong geometry still scores
# high. Flank confidence reflects whether the structural scaffold surrounding
# H3 is well-resolved, which is a prerequisite for the loop to be meaningful.
baseline_flank = all_results.loc[
    all_results["dataset"] == "iggen",
    "combined_flank_mean_plddt"
].dropna().values

finetuned_flank = all_results.loc[
    all_results["dataset"] == "finetuned",
    "combined_flank_mean_plddt"
].dropna().values

plot_violin_custom(
    baseline_vals=baseline_flank,
    finetuned_vals=finetuned_flank,
    outpath=VIOLIN_FLANK_PLOT,
    ylabel="Mean H3 Flanking Region pLDDT",
    title="H3 Anchoring Framework Confidence",
    unit="plddt"
)

# ---- Plot 2: H3 mean PAE to all residues ----
# Significance testing is included here because the PAE metric is the primary
# measure of improvement: lower PAE means H3 is placed more confidently
# relative to the rest of the antibody, which is the goal of fine-tuning.
baseline_h3_pae = all_results.loc[
    all_results["dataset"] == "iggen",
    "h3_mean_pae_to_all_residues"
].dropna().values

finetuned_h3_pae = all_results.loc[
    all_results["dataset"] == "finetuned",
    "h3_mean_pae_to_all_residues"
].dropna().values

plot_violin_custom(
    baseline_vals=baseline_h3_pae,
    finetuned_vals=finetuned_h3_pae,
    outpath=VIOLIN_H3_PAE_PLOT,
    ylabel="Mean H3 PAE to All Residues [Å]",
    title="H3 loop orientation",
    unit="A",
    add_significance=True
)

# Write summary log
with open(SUMMARY_TXT, "w") as f:
    f.write("\n".join(log_lines) + "\n")

log(f"\n✅ Summary written to: {SUMMARY_TXT}")
log("✅ Done.")
