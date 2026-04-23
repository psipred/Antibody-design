import os
import numpy as np
import matplotlib.pyplot as plt
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
PLOT_EXCLUDE_LABELS = {"Random finetune", "SAbDab finetune"}

CDR_H3_START = 95
CDR_H3_END   = 102

parser = PDBParser(QUIET=True)


def collect_h3_means(input_dir):
    """
    For each PDB in input_dir, extract CA b-factors from heavy-chain
    Chothia H3 residues 95-102 and compute one mean value per entry.
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
                if chain.id != "H":
                    continue
                for residue in chain:
                    if CDR_H3_START <= residue.id[1] <= CDR_H3_END:
                        for atom in residue:
                            if atom.name == "CA":
                                scores.append(atom.bfactor)
                                break

        if scores:
            per_entry_means.append(float(np.mean(scores)))
        else:
            print("  [WARNING] No CDR H3 residues found in {}, skipping.".format(name))

    if len(per_entry_means) == 0:
        raise RuntimeError("No valid H3 scores found in {}".format(input_dir))

    return per_entry_means


def plot_violin(data, labels, outpath):
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
        showmedians=False,
        showextrema=False
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)

    # Median annotation
    for vals, pos in zip(data, positions):
        median = np.median(vals)
        ax.text(
            pos,
            median + 0.02,
            "{:.3g}".format(median),
            ha="center",
            va="bottom",
            fontsize=11
        )

    # Mann-Whitney U test: Baseline vs Finetuned
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

            x1, x2 = positions[idx_a], positions[idx_b]
            y_data_max = max(np.max(np.asarray(vals, dtype=float)) for vals in data if len(vals) > 0)
            bracket_y = y_data_max + 0.05
            bracket_h = 0.02
            text_y = bracket_y + bracket_h + 0.01

            ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
            ax.text((x1 + x2) / 2, text_y, sig_label, ha="center", va="bottom", fontsize=12)

            print("Mann-Whitney U (Baseline vs Finetuned): p={:.6g} ({})".format(pvalue, sig_label))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, rotation=20, ha="right")
    ax.set_ylabel("Mean CDR H3 predicted error", fontsize=12)
    ax.set_title("CDR H3 Structure Confidence (ABB2)", fontsize=13)

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

plot_data = []
plot_labels = []
for vals, label in zip(all_data, all_labels):
    if label in PLOT_EXCLUDE_LABELS:
        continue
    plot_data.append(vals)
    plot_labels.append(label)

plot_violin(plot_data, plot_labels, OUT_PNG)