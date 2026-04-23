#!/usr/bin/env python3

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
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

SCORES_PATTERN = "*_scores_rank_00[1-5]_alphafold2_multimer_v2_model_*_seed_000.json"
PAE_PATTERN = "*_predicted_aligned_error*.json"

# Chothia heavy-chain regions
LEFT_FLANK_START = 90
LEFT_FLANK_END   = 94

H3_START = 95
H3_END   = 102

RIGHT_FLANK_START = 103
RIGHT_FLANK_END   = 107

# =========================
# Logging
# =========================
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
    if re.match(r"^[A-Za-z0-9_.-]+$", entry_id):
        return entry_id
    return None


def parse_anarci_chothia(filepath):
    """
    Returns:
        dict:
            entry_id -> {
                "left_flank_indices": list of 0-based indices for H90-H94
                "h3_indices": list of 0-based indices for H95-H102
                "right_flank_indices": list of 0-based indices for H103-H107
                "combined_flank_indices": left + right
            }
    """
    results = {}

    current_id = None
    seq_index = 0

    left_flank_indices = []
    h3_indices = []
    right_flank_indices = []

    def flush():
        nonlocal current_id, seq_index, left_flank_indices, h3_indices, right_flank_indices
        if current_id is not None:
            results[current_id] = {
                "left_flank_indices": left_flank_indices.copy(),
                "h3_indices": h3_indices.copy(),
                "right_flank_indices": right_flank_indices.copy(),
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
                flush()
                current_id = eid
                continue

            if current_id is None:
                continue

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
            if aa in ("-", ".", "X"):
                continue

            if LEFT_FLANK_START <= pos <= LEFT_FLANK_END:
                left_flank_indices.append(seq_index)

            if H3_START <= pos <= H3_END:
                h3_indices.append(seq_index)

            if RIGHT_FLANK_START <= pos <= RIGHT_FLANK_END:
                right_flank_indices.append(seq_index)

            seq_index += 1

    flush()
    results.pop(None, None)
    return results

# =========================
# STEP 2 — Group files by entry
# =========================
def group_score_files(run_dir):
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
    Average pLDDT across all provided score JSONs for one entry.
    Returns a 1D numpy array.
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

    min_len = min(arr.shape[0] for arr in plddt_arrays)
    mat = np.vstack([arr[:min_len] for arr in plddt_arrays])
    return np.mean(mat, axis=0)


def load_mean_pae_matrix(pae_files):
    """
    Average PAE matrices across all provided PAE JSONs for one entry.
    Returns a 2D numpy array.
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

    min_rows = min(m.shape[0] for m in pae_mats)
    min_cols = min(m.shape[1] for m in pae_mats)
    stacked = np.stack([m[:min_rows, :min_cols] for m in pae_mats], axis=0)
    return np.mean(stacked, axis=0)

# =========================
# STEP 4 — Metric helpers
# =========================
def safe_mean_from_indices(arr, indices):
    if arr is None or len(indices) == 0:
        return np.nan
    if max(indices) >= len(arr):
        return np.nan
    return float(np.mean(arr[indices]))


def safe_mean_pae_h3_to_all(pae_mat, h3_indices):
    """
    Mean PAE from H3 residues to all residues:
        mean( PAE[h3_indices, :] )
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
        showmedians=False,
        showextrema=False
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

        # smarter offset based on scale
        offset = max(np.std(vals) * 0.1, 0.02)

        ax.text(
            pos, median + offset, text,
            ha="center", va="bottom", fontsize=11
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

        y_data_max = max(np.max(np.asarray(vals, dtype=float)) for vals in data if len(vals) > 0)
        y_range = y_data_max if y_data_max > 0 else 1.0
        bracket_y = y_data_max + 0.06 * y_range
        bracket_h = 0.03 * y_range
        text_y = bracket_y + bracket_h + 0.01 * y_range

        ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
        ax.text(1.5, text_y, sig_label, ha="center", va="bottom", fontsize=12)
        log(f"Mann-Whitney U (Baseline vs Finetuned): p={pvalue:.6g} ({sig_label})")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

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

    all_entries = sorted(set(score_groups.keys()) | set(pae_groups.keys()))

    results = []
    skipped = []

    for entry_id in all_entries:
        if entry_id not in anarci_data:
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

# Plot 1 — flank mean pLDDT
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

# Plot 2 — H3 mean PAE to all residues
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