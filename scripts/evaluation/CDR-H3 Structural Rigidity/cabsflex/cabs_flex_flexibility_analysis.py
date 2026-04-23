#!/usr/bin/env python3
"""
CDR H3 Flexibility Analysis
============================
Extracts CDR H3 RMSF from CABSflex outputs and produces:
  1. Violin plot of mean CDR H3 RMSF (baseline vs finetuned)
  2. Per-residue mean RMSF line plot across H3 positions (baseline vs finetuned)

RMSF.csv format expected:
    A1    1.350
    A2    0.734
    ...
    B1    0.500
    ...
Where A = heavy chain, B = light chain.

CDR H3 is identified from ANARCI chothia numbering (positions 95-102 on heavy chain).
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------

BASELINE_DIR  = "/home/alanwu/Documents/iggen_model/data/cabsflex/iggen"
FINETUNED_DIR = "/home/alanwu/Documents/iggen_model/data/cabsflex/oas_v6"

BASELINE_ANARCI  = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_anarci_chothia.txt"
FINETUNED_ANARCI = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v6/generated_anarci_chothia.txt"

# CDR H3 Chothia positions on heavy chain (inclusive)
H3_START = 95
H3_END   = 102

# Heavy chain label in RMSF.csv (ColabFold default)
HEAVY_CHAIN = "A"

PLOT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/cabsflex/plots"

# ---------------------------------------------------------------------------
# Parse RMSF.csv
# Format: "A1\t1.350" — chain+residue_num in col 1, rmsf in col 2
# Returns dict: (chain, res_num) -> rmsf
# e.g. ("A", 1) -> 1.350
# ---------------------------------------------------------------------------

def parse_rmsf(rmsf_path):
    rmsf = {}
    with open(rmsf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]   # e.g. "A1", "A95", "B1"
            val = parts[1]
            # Split chain letter from residue number
            match = re.match(r"([A-Za-z]+)(\d+)", key)
            if match:
                chain   = match.group(1)
                res_num = int(match.group(2))
                try:
                    rmsf[(chain, res_num)] = float(val)
                except ValueError:
                    continue
    return rmsf


def find_rmsf_file(job_dir):
    path = os.path.join(job_dir, "plots", "RMSF.csv")
    if os.path.exists(path):
        return path
    # Fallback: search recursively
    candidates = glob.glob(os.path.join(job_dir, "**", "RMSF.csv"), recursive=True) + \
                 glob.glob(os.path.join(job_dir, "**", "rmsf.csv"), recursive=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Parse ANARCI chothia file
# Returns dict: ab_id -> list of (chothia_num, aa, seq_index_1based)
# seq_index_1based matches the residue numbering in RMSF.csv for chain A
# ---------------------------------------------------------------------------

def parse_anarci(filepath):
    ab_data = {}
    current_ab = None
    current_residues = []
    seq_index = 0  # 0-based, will be converted to 1-based on append

    with open(filepath) as f:
        for line in f:
            line = line.strip()

            # New antibody block
            if line.startswith("# ") and not any(x in line for x in
                    ["ANARCI", "Domain", "Most", "species", "Scheme", "e-value",
                     "numbered", "significant", "HMM", "|"]):
                if current_ab is not None:
                    ab_data[current_ab] = current_residues
                current_ab = line[2:].strip()
                current_residues = []
                seq_index = 0
                continue

            if line.startswith("#") or line == "":
                continue

            if line == "//":
                if current_ab is not None:
                    ab_data[current_ab] = current_residues
                current_ab = None
                current_residues = []
                seq_index = 0
                continue

            # Residue line: "H 95      C" or "H 95A     -"
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "H":
                num_str = parts[1]
                aa      = parts[2]
                num_match = re.match(r"(\d+)", num_str)
                if num_match:
                    chothia_num = int(num_match.group(1))
                    if aa != "-":
                        seq_index += 1
                        current_residues.append((chothia_num, num_str, aa, seq_index))

    if current_ab is not None:
        ab_data[current_ab] = current_residues

    return ab_data


def get_h3_seq_indices(anarci_data, ab_id):
    """
    Returns list of 1-based sequence indices for H3 residues (Chothia 95-102)
    These correspond to the residue numbers in RMSF.csv under chain A.
    """
    if ab_id not in anarci_data:
        return []
    return [r[3] for r in anarci_data[ab_id] if H3_START <= r[0] <= H3_END]


def extract_ab_id(job_folder_name):
    """
    Extracts ab_id from job folder name.
    e.g. "ab_1_unrelaxed_rank_001_..." -> "ab_1"
    """
    match = re.match(r"(ab_\d+)", job_folder_name)
    return match.group(1) if match else job_folder_name


# ---------------------------------------------------------------------------
# Load all data for one group
# ---------------------------------------------------------------------------

def load_group(cabsflex_dir, anarci_file, group_name):
    anarci_data = parse_anarci(anarci_file)
    mean_h3_rmsf = []
    per_position = defaultdict(list)  # relative H3 position -> list of RMSF values
    skipped = []

    job_dirs = sorted([
        d for d in glob.glob(os.path.join(cabsflex_dir, "*"))
        if os.path.isdir(d)
    ])

    for job_dir in job_dirs:
        folder_name = os.path.basename(job_dir)
        ab_id = extract_ab_id(folder_name)

        if ab_id not in anarci_data:
            skipped.append("{} (no ANARCI match)".format(ab_id))
            continue

        rmsf_file = find_rmsf_file(job_dir)
        if rmsf_file is None:
            skipped.append("{} (no RMSF.csv)".format(ab_id))
            continue

        rmsf = parse_rmsf(rmsf_file)
        if not rmsf:
            skipped.append("{} (empty RMSF)".format(ab_id))
            continue

        h3_indices = get_h3_seq_indices(anarci_data, ab_id)
        if not h3_indices:
            skipped.append("{} (no H3 residues in Chothia 95-102)".format(ab_id))
            continue

        # Look up RMSF values for H3 residues on heavy chain A
        h3_rmsf_vals = []
        for rel_pos, seq_idx in enumerate(h3_indices, start=1):
            key = (HEAVY_CHAIN, seq_idx)
            if key in rmsf:
                h3_rmsf_vals.append(rmsf[key])
                per_position[rel_pos].append(rmsf[key])

        if not h3_rmsf_vals:
            skipped.append("{} (H3 indices not found in RMSF)".format(ab_id))
            continue

        mean_h3_rmsf.append(np.mean(h3_rmsf_vals))

    if skipped:
        print("[{}] Skipped {} entries:".format(group_name, len(skipped)))
        for s in skipped[:5]:
            print("  - {}".format(s))
        if len(skipped) > 5:
            print("  ... and {} more".format(len(skipped) - 5))

    print("[{}] Successfully loaded {} antibodies".format(group_name, len(mean_h3_rmsf)))
    return mean_h3_rmsf, per_position


# ---------------------------------------------------------------------------
# Plot 1: Violin plot
# ---------------------------------------------------------------------------

def plot_violin(baseline_vals, finetuned_vals, outpath):
    fig, ax = plt.subplots(figsize=(5, 6))

    data   = [baseline_vals, finetuned_vals]
    labels = ["Baseline", "Finetuned"]
    colors = ["#f5c87a", "#a8c8e8"]

    parts = ax.violinplot(data, positions=[1, 2], widths=0.6,
                          showmedians=False, showextrema=False)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)

    # Median annotation
    for vals, pos in zip(data, [1, 2]):
        median = np.median(vals)
        ax.text(pos, median + 0.02, "{:.2f} \u00c5".format(median),
                ha="center", va="bottom", fontsize=11)

    # Mann-Whitney U significance test (Baseline vs Finetuned)
    _, pvalue = mannwhitneyu(baseline_vals, finetuned_vals, alternative="two-sided")
    if pvalue < 0.001:
        sig_label = "***"
    elif pvalue < 0.01:
        sig_label = "**"
    elif pvalue < 0.05:
        sig_label = "*"
    else:
        sig_label = "ns"

    y_data_max = max(np.max(baseline_vals), np.max(finetuned_vals))
    bracket_y = y_data_max + 0.08
    bracket_h = 0.04
    text_y = bracket_y + bracket_h + 0.01
    ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
    ax.text(1.5, text_y, sig_label, ha="center", va="bottom", fontsize=12)
    print("Mann-Whitney U (Baseline vs Finetuned): p={:.6g} ({})".format(pvalue, sig_label))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean CDR H3 RMSF [\u00c5]", fontsize=12)
    ax.set_title("CDR H3 Loop Rigidity", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(outpath))


# ---------------------------------------------------------------------------
# Plot 2: Per-residue mean RMSF dotted line plot
# ---------------------------------------------------------------------------

def plot_per_residue(baseline_pos, finetuned_pos, outpath):
    all_positions = sorted(set(list(baseline_pos.keys()) + list(finetuned_pos.keys())))

    baseline_means  = [np.mean(baseline_pos[p])  if p in baseline_pos  else np.nan for p in all_positions]
    finetuned_means = [np.mean(finetuned_pos[p]) if p in finetuned_pos else np.nan for p in all_positions]

    x      = np.arange(len(all_positions))
    labels = ["H3-{}".format(p) for p in all_positions]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x, baseline_means,  color="grey",    linestyle="--", marker="o",
            markersize=3, linewidth=1.0, label="Baseline")
    ax.plot(x, finetuned_means, color="#4a90d9", linestyle="--", marker="o",
            markersize=3, linewidth=1.0, label="Finetuned")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSF", fontsize=11)
    ax.set_xlabel("CDR H3 Residue Position", fontsize=11)
    ax.set_title("Mean CDR H3 RMSF per Residue", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(outpath))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading baseline (oas_v6)...")
    baseline_means, baseline_pos = load_group(BASELINE_DIR, BASELINE_ANARCI, "Baseline")

    print("\nLoading finetuned (iggen)...")
    finetuned_means, finetuned_pos = load_group(FINETUNED_DIR, FINETUNED_ANARCI, "Finetuned")

    if not baseline_means or not finetuned_means:
        print("ERROR: No data loaded for one or both groups. Check paths.")
        return

    plot_violin(baseline_means, finetuned_means,
                os.path.join(PLOT_DIR, "cdr_h3_rmsf_violin.png"))

    plot_per_residue(baseline_pos, finetuned_pos,
                     os.path.join(PLOT_DIR, "cdr_h3_rmsf_per_residue.png"))

    # Save raw summary CSV
    summary = pd.DataFrame({
        "group":        ["baseline"] * len(baseline_means) + ["finetuned"] * len(finetuned_means),
        "mean_h3_rmsf": baseline_means + finetuned_means
    })
    csv_path = os.path.join(PLOT_DIR, "cdr_h3_rmsf_summary.csv")
    summary.to_csv(csv_path, index=False)
    print("Saved summary CSV: {}".format(csv_path))

    print("\nBaseline  median H3 RMSF: {:.3f} A".format(np.median(baseline_means)))
    print("Finetuned median H3 RMSF: {:.3f} A".format(np.median(finetuned_means)))


if __name__ == "__main__":
    main()