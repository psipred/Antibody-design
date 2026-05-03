#!/usr/bin/env python3
"""
CDR H3 RMSF vs Loop Length — controlled comparison
====================================================
Investigates whether differences in mean CDR H3 RMSF between baseline and
fine-tuned model outputs are confounded by H3 loop length.  CABSflex RMSF
generally scales with loop length (longer loops have more degrees of freedom),
so a naïve group comparison could reflect length distributions rather than
true flexibility differences.

This script produces a length-stratified bubble chart where:
  * x-axis  = H3 loop length (number of residues in Chothia 95–102)
  * y-axis  = mean H3 RMSF averaged over all antibodies at that length
  * bubble size ∝ sample count  (larger bubbles = more antibodies at that length)
  * error bars = standard error of the mean at each length bin

Only length bins with at least MIN_COUNT_PER_LEN antibodies are plotted; sparse
bins would give unreliable means and inflate visual differences.

Inputs
------
  BASELINE_DIR / FINETUNED_DIR : CABSflex output trees (one subdir per antibody)
  BASELINE_ANARCI / FINETUNED_ANARCI : ANARCI Chothia-numbered output files

Outputs (written to OUTPUT_DIR)
--------------------------------
  mean_h3_rmsf_vs_loop_length_baseline_vs_finetuned.png
      Bubble chart with error bars.
  per_entry_h3_length_rmsf.tsv
      Flat table: (entry, group, h3_seq, h3_len, h3_mean_rmsf) per antibody.
  binned_mean_h3_rmsf.tsv
      Summary table: (group, h3_len, n, mean, sd, se) per length bin.
  run_log.txt
      All log() output for reproducibility.

"""

import os
import re
import glob
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


# =========================================================
# CONFIG
# =========================================================
BASELINE_DIR  = "/home/alanwu/Documents/iggen_model/data/cabsflex/iggen"
FINETUNED_DIR = "/home/alanwu/Documents/iggen_model/data/cabsflex/oas_v6"

BASELINE_ANARCI  = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_anarci_chothia.txt"
FINETUNED_ANARCI = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v6/generated_anarci_chothia.txt"

H3_START = 95
H3_END   = 102
HEAVY_CHAIN = "A"

OUTPUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/cabsflex/oas_v6_length_vs_rmsf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_PATH = os.path.join(OUTPUT_DIR, "mean_h3_rmsf_vs_loop_length_baseline_vs_finetuned.png")
RAW_TABLE_PATH = os.path.join(OUTPUT_DIR, "per_entry_h3_length_rmsf.tsv")
BIN_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "binned_mean_h3_rmsf.tsv")
LOG_PATH = os.path.join(OUTPUT_DIR, "run_log.txt")

# Minimum antibodies per length to include a bin in the plot;
# bins with fewer samples give unreliable mean estimates.
MIN_COUNT_PER_LEN = 3


# =========================================================
# LOGGING
# =========================================================
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# =========================================================
# RMSF PARSING
# =========================================================
def parse_rmsf(rmsf_path):
    """
    RMSF.csv format:
        A1    1.350
        A2    0.734
        ...
        B1    0.500

    Returns:
        dict[(chain, res_num)] = rmsf
    """
    rmsf = {}
    with open(rmsf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            key = parts[0]
            val = parts[1]

            match = re.match(r"([A-Za-z]+)(\d+)", key)
            if not match:
                continue

            chain = match.group(1)
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

    # Fallback for non-standard CABSflex output layouts
    candidates = (
        glob.glob(os.path.join(job_dir, "**", "RMSF.csv"), recursive=True) +
        glob.glob(os.path.join(job_dir, "**", "rmsf.csv"), recursive=True)
    )
    return candidates[0] if candidates else None


# =========================================================
# ANARCI PARSING
# =========================================================
def parse_anarci(filepath):
    """
    Returns:
        ab_data[ab_id] = list of tuples:
            (chothia_num_int, original_num_str, aa, seq_index_1based)

    seq_index_1based matches the residue numbering in RMSF.csv
    for heavy chain A.
    """
    ab_data = {}
    current_ab = None
    current_residues = []
    seq_index = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()

            # Distinguish antibody ID header lines from ANARCI metadata lines;
            # the negative-list blocks known metadata patterns shared by both.
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

            # "//" marks end-of-record
            if line == "//":
                if current_ab is not None:
                    ab_data[current_ab] = current_residues
                current_ab = None
                current_residues = []
                seq_index = 0
                continue

            parts = line.split()
            if len(parts) >= 3 and parts[0] == "H":
                num_str = parts[1]
                aa = parts[2]

                num_match = re.match(r"(\d+)", num_str)
                if num_match:
                    chothia_num = int(num_match.group(1))
                    if aa != "-":
                        # Gap positions don't correspond to a physical residue
                        # and have no RMSF entry, so skip seq_index increment.
                        seq_index += 1
                        current_residues.append((chothia_num, num_str, aa, seq_index))

    if current_ab is not None:
        ab_data[current_ab] = current_residues

    return ab_data


def get_h3_residues(anarci_data, ab_id):
    """
    Returns list of H3 residues:
        [(chothia_num, original_num_str, aa, seq_index_1based), ...]
    """
    if ab_id not in anarci_data:
        return []
    return [r for r in anarci_data[ab_id] if H3_START <= r[0] <= H3_END]


def get_h3_seq_indices(anarci_data, ab_id):
    """
    Returns list of 1-based seq indices for H3 residues.
    """
    return [r[3] for r in get_h3_residues(anarci_data, ab_id)]


def get_h3_sequence(anarci_data, ab_id):
    """
    Returns H3 amino-acid sequence string.
    """
    residues = get_h3_residues(anarci_data, ab_id)
    return "".join(r[2] for r in residues)


def extract_ab_id(job_folder_name):
    """
    e.g. "ab_1_unrelaxed_rank_001_..." -> "ab_1"
    """
    match = re.match(r"(ab_\d+)", job_folder_name)
    return match.group(1) if match else job_folder_name


# =========================================================
# BUILD DATASET
# =========================================================
def build_dataset(cabsflex_dir, anarci_file, group_label):
    log(f"\n=== Processing {group_label} ===")
    log(f"CABS-flex dir: {cabsflex_dir}")
    log(f"ANARCI file:   {anarci_file}")

    anarci_data = parse_anarci(anarci_file)
    log(f"Parsed ANARCI entries: {len(anarci_data)}")

    rows = []
    skipped = []

    job_dirs = sorted([
        d for d in glob.glob(os.path.join(cabsflex_dir, "*"))
        if os.path.isdir(d)
    ])
    log(f"Found job dirs: {len(job_dirs)}")

    for job_dir in job_dirs:
        folder_name = os.path.basename(job_dir)
        ab_id = extract_ab_id(folder_name)

        if ab_id not in anarci_data:
            skipped.append((ab_id, "no ANARCI match"))
            continue

        rmsf_file = find_rmsf_file(job_dir)
        if rmsf_file is None:
            skipped.append((ab_id, "no RMSF.csv"))
            continue

        rmsf = parse_rmsf(rmsf_file)
        if not rmsf:
            skipped.append((ab_id, "empty RMSF"))
            continue

        h3_indices = get_h3_seq_indices(anarci_data, ab_id)
        h3_seq = get_h3_sequence(anarci_data, ab_id)

        if not h3_indices:
            skipped.append((ab_id, "no H3 residues in Chothia 95-102"))
            continue

        if not h3_seq:
            skipped.append((ab_id, "empty H3 sequence"))
            continue

        h3_rmsf_vals = []
        for seq_idx in h3_indices:
            key = (HEAVY_CHAIN, seq_idx)
            if key in rmsf:
                h3_rmsf_vals.append(rmsf[key])

        if not h3_rmsf_vals:
            skipped.append((ab_id, "H3 indices not found in RMSF"))
            continue

        rows.append({
            "entry": ab_id,
            "group": group_label,
            "h3_seq": h3_seq,
            "h3_len": len(h3_seq),
            "h3_mean_rmsf": float(np.mean(h3_rmsf_vals)),
        })

    log(f"Kept:    {len(rows)}")
    log(f"Skipped: {len(skipped)}")

    if skipped:
        log("Skipped examples:")
        for ab_id, reason in skipped[:10]:
            log(f"  {ab_id}: {reason}")

    return rows


# =========================================================
# SUMMARISE BY LOOP LENGTH
# =========================================================
def summarise_by_loop_length(rows):
    summaries = []
    groups = sorted(set(r["group"] for r in rows))

    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        lengths = sorted(set(r["h3_len"] for r in group_rows))

        for L in lengths:
            vals = [r["h3_mean_rmsf"] for r in group_rows if r["h3_len"] == L]
            n = len(vals)

            # Skip sparse bins to avoid misleading data points
            if n < MIN_COUNT_PER_LEN:
                continue

            vals = np.array(vals, dtype=float)

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals))
            # SE (standard error of the mean) is the relevant quantity for
            # error bars because we care about precision of the mean, not
            # the spread of individual observations (which SD would show).
            se_val = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

            summaries.append({
                "group": group,
                "h3_len": int(L),
                "n": int(n),
                "mean_h3_rmsf": mean_val,
                "sd_h3_rmsf": sd_val,
                "se_h3_rmsf": se_val,
            })

    return summaries


# =========================================================
# PLOTTING
# =========================================================
def plot_bubble_length_rmsf(summaries, out_path):
    plt.figure(figsize=(8, 5.8))

    label_map = {
        "baseline": "Baseline",
        "finetuned": "Finetuned",
    }

    color_map = {
        "baseline": "#f5c87a",
        "finetuned": "#a8c8e8",
    }

    plotted_any = False

    for group in ["baseline", "finetuned"]:
        grp = [x for x in summaries if x["group"] == group]
        grp = sorted(grp, key=lambda d: d["h3_len"])
        if not grp:
            continue

        x = np.array([d["h3_len"] for d in grp], dtype=float)
        y = np.array([d["mean_h3_rmsf"] for d in grp], dtype=float)
        n = np.array([d["n"] for d in grp], dtype=float)
        se = np.array([d["se_h3_rmsf"] for d in grp], dtype=float)

        # Bubble area encodes sample count: base size 40 prevents dots from
        # disappearing at n=1, and the 25*n term scales linearly with count.
        sizes = 40 + 25 * n
        color = color_map[group]

        plt.plot(x, y, linewidth=1.5, alpha=0.6, color=color, label=label_map[group])
        plt.scatter(x, y, s=sizes, alpha=0.55, color=color)
        plt.errorbar(x, y, yerr=se, fmt="none", elinewidth=1, capsize=3, alpha=0.7, color=color)

        plotted_any = True

    if not plotted_any:
        raise ValueError("No loop-length groups had enough points to plot.")

    plt.xlabel("H3 loop length")
    plt.ylabel("Mean H3 RMSF (Å)")
    if summaries:
        all_lens = [d["h3_len"] for d in summaries]
        # Small proportional padding so the outermost bubbles aren't clipped
        pad = (max(all_lens) - min(all_lens)) * 0.04 + 0.5
        plt.xlim(min(all_lens) - pad, max(all_lens) + pad)
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
        "h3_mean_rmsf",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def write_bin_summary_tsv(summaries, path):
    cols = [
        "group",
        "h3_len",
        "n",
        "mean_h3_rmsf",
        "sd_h3_rmsf",
        "se_h3_rmsf",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summaries:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


# =========================================================
# MAIN
# =========================================================
def main():
    baseline_rows = build_dataset(
        cabsflex_dir=BASELINE_DIR,
        anarci_file=BASELINE_ANARCI,
        group_label="baseline",
    )

    finetuned_rows = build_dataset(
        cabsflex_dir=FINETUNED_DIR,
        anarci_file=FINETUNED_ANARCI,
        group_label="finetuned",
    )

    all_rows = baseline_rows + finetuned_rows
    if not all_rows:
        raise ValueError("No valid rows produced for either group.")

    write_rows_tsv(all_rows, RAW_TABLE_PATH)
    log(f"\nPer-entry table written to: {RAW_TABLE_PATH}")

    summaries = summarise_by_loop_length(all_rows)
    if not summaries:
        raise ValueError("No loop-length groups had enough points to summarise.")

    write_bin_summary_tsv(summaries, BIN_SUMMARY_PATH)
    log(f"Binned summary written to: {BIN_SUMMARY_PATH}")

    plot_bubble_length_rmsf(summaries, PLOT_PATH)
    log(f"Plot written to: {PLOT_PATH}")

    for group in ["baseline", "finetuned"]:
        grp = [r for r in all_rows if r["group"] == group]
        if grp:
            mean_rmsf = np.mean([r["h3_mean_rmsf"] for r in grp])
            mean_len = np.mean([r["h3_len"] for r in grp])
            log(f"{group}: n={len(grp)}, mean H3 RMSF={mean_rmsf:.3f} Å, mean H3 length={mean_len:.2f}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    log(f"Log written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
