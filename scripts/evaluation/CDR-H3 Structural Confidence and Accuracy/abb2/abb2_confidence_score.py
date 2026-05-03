"""
abb2_confidence_score.py
========================
Extracts ABodyBuilder2 (ImmuneBuilder) predicted confidence scores for the
CDR-H3 loop and produces a violin plot comparing scores across multiple model
runs (Baseline, Random finetune, SAbDab finetune, Finetuned).

Overview
--------
ImmuneBuilder stores per-residue confidence scores in the B-factor column of
its output PDB files. Unlike crystallographic B-factors (thermal displacement),
here the value encodes the model's own predicted error for each residue — a
lower value means lower predicted error (i.e. higher confidence), analogous to
how pLDDT works in AlphaFold. This script reads those values exclusively from
CDR-H3 residues (Chothia positions 95–102) on the heavy chain, computes one
mean score per structure, and visualises the distribution as a violin plot.

Inputs
------
  Per-run directories of .pdb files produced by ABodyBuilder2. Each PDB must
  have a chain labelled "H" (heavy chain) with Chothia-numbered residues.
  Configured via RUN_CONFIGS.

Outputs
-------
  OUT_DIR/abb2_h3_violin.png  — violin plot comparing confidence distributions

"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
from Bio.PDB import PDBParser
from scipy.stats import mannwhitneyu

# =========================
# CONFIG
# =========================
RUN_CONFIGS = [
    {
        "label": "Baseline",
        "input_dir": "/home/alanwu/Documents/iggen_model/immunebuilder_output/iggen",
    },
    {
        "label": "Random finetune",
        "input_dir": "/home/alanwu/Documents/iggen_model/immunebuilder_output/random",
    },
    {
        "label": "SAbDab finetune",
        "input_dir": "/home/alanwu/Documents/iggen_model/immunebuilder_output/1A",
    },
    {
        "label": "Finetuned",
        "input_dir": "/home/alanwu/Documents/iggen_model/immunebuilder_output/oas_v6",
    },
]

OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/abb2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "abb2_h3_violin.png")

# These labels are computed and logged but excluded from the final violin plot
# to keep the plot focused on the primary Baseline vs Finetuned comparison.
PLOT_EXCLUDE_LABELS = {"Random finetune", "SAbDab finetune"}

# Chothia CDR-H3 residue range on the heavy chain.
# Positions 95–102 are the canonical H3 loop definition under Chothia numbering.
CDR_H3_START = 95
CDR_H3_END   = 102

parser = PDBParser(QUIET=True)


def collect_h3_means(input_dir):
    """
    For each PDB in input_dir, extract CA b-factors from heavy-chain
    Chothia H3 residues 95-102 and compute one mean value per entry.

    Parameters
    ----------
    input_dir : str
        Directory containing .pdb files output by ABodyBuilder2.

    Returns
    -------
    list of float
        One mean confidence score per successfully parsed PDB.
        Entries where no H3 residues are found on chain H are skipped
        with a warning rather than raising an error.
    """
    pdb_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pdb")])
    print("Found {} PDB files in {}\n".format(len(pdb_files), input_dir))

    per_entry_means = []

    for pdb_file in pdb_files:
        name = pdb_file.replace(".pdb", "")
        pdb_path = os.path.join(input_dir, pdb_file)

        structure = parser.get_structure(name, pdb_path)

        scores = []
        for model in structure:
            for chain in model:
                # Only process the heavy chain; the light chain (L) is ignored
                # because CDR-H3 is a heavy-chain-only loop.
                if chain.id != "H":
                    continue
                for residue in chain:
                    if CDR_H3_START <= residue.id[1] <= CDR_H3_END:
                        for atom in residue:
                            if atom.name == "CA":
                                # The CA B-factor in ImmuneBuilder PDBs encodes
                                # the model's predicted error for this residue.
                                scores.append(atom.bfactor)
                                # Stop after the first CA to avoid double-counting
                                # alternate conformers, if present.
                                break

        if scores:
            per_entry_means.append(float(np.mean(scores)))
        else:
            print("  [WARNING] No CDR H3 residues found in {}, skipping.".format(name))

    if len(per_entry_means) == 0:
        raise RuntimeError("No valid H3 scores found in {}".format(input_dir))

    return per_entry_means


def plot_violin(data, labels, outpath):
    """
    Render a violin plot of per-entry mean H3 confidence scores across runs,
    annotating median values and a Mann-Whitney significance bracket between
    "Baseline" and "Finetuned" groups.

    Parameters
    ----------
    data : list of list of float
        Parallel to labels; each inner list is per-entry means for one run.
    labels : list of str
        Display labels for each violin.
    outpath : str
        File path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    color_by_label = {
        "Baseline": "#f5c87a",
        "Random finetune": "#d9c27a",
        "SAbDab finetune": "#b8d39a",
        "Finetuned": "#a8c8e8",
    }
    colors = [color_by_label.get(label, "#a8c8e8") for label in labels]
    positions = list(range(1, len(labels) + 1))

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.6,
        showmedians=False,  # Medians drawn manually below so we can annotate them
        showextrema=False   # Hide min/max whiskers for a cleaner look
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)

    # Median annotation — placed just above the median value so it doesn't
    # overlap the violin body, using a fixed offset of 0.02 score units.
    for vals, pos in zip(data, positions):
        median = np.median(vals)
        ax.text(
            pos,
            median + 0.02,
            "{:.3g}".format(median),
            ha="center",
            va="bottom",
            fontsize=14
        )

    # Mann-Whitney U test: Baseline vs Finetuned
    # Two-sided test; significance indicated by standard star notation.
    baseline_label = "Baseline"
    target_label = "Finetuned"
    if baseline_label in labels and target_label in labels:
        idx_a = labels.index(baseline_label)
        idx_b = labels.index(target_label)
        group_a = np.asarray(data[idx_a], dtype=float)
        group_b = np.asarray(data[idx_b], dtype=float)

        if len(group_a) > 0 and len(group_b) > 0:
            _, pvalue = mannwhitneyu(group_a, group_b, alternative="two-sided")
            if pvalue < 0.001:
                sig_label = "***"
            elif pvalue < 0.01:
                sig_label = "**"
            elif pvalue < 0.05:
                sig_label = "*"
            else:
                sig_label = "ns"

            # Draw a bracket from Baseline to Finetuned positions, positioned
            # slightly above the maximum data point in either group.
            x1, x2 = positions[idx_a], positions[idx_b]
            y_data_max = max(np.max(np.asarray(vals, dtype=float)) for vals in data if len(vals) > 0)
            bracket_y = y_data_max + 0.05
            bracket_h = 0.02
            text_y = bracket_y + bracket_h + 0.01

            ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
            ax.text((x1 + x2) / 2, text_y, sig_label, ha="center", va="bottom", fontsize=14)

            print("Mann-Whitney U (Baseline vs Finetuned): p={:.6g} ({})".format(pvalue, sig_label))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=14, rotation=0, ha="center")
    ax.set_ylabel("Mean CDR H3 predicted error", fontsize=14)

    # Remove top and right spines for a cleaner publication-style appearance.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0.3, len(labels) + 0.7)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: {}".format(outpath))


# =========================
# RUN
# =========================
# Collect scores from every configured run directory, including the excluded
# runs so their summary statistics still appear in the terminal output.
all_data = []
all_labels = []

for cfg in RUN_CONFIGS:
    vals = collect_h3_means(cfg["input_dir"])
    all_data.append(vals)
    all_labels.append(cfg["label"])

    print(
        "{}: n={} median={:.3g} mean={:.3g}".format(
            cfg["label"],
            len(vals),
            np.median(vals),
            np.mean(vals)
        )
    )

# Filter down to only the runs that should appear in the final plot.
plot_data = []
plot_labels = []
for vals, label in zip(all_data, all_labels):
    if label in PLOT_EXCLUDE_LABELS:
        continue
    plot_data.append(vals)
    plot_labels.append(label)

plot_violin(plot_data, plot_labels, OUT_PNG)
