import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# =========================
# CONFIG
# =========================
IGGEN_CSV = "/home/alanwu/Documents/iggen_model/evaluation_metrics/multiseed_alphafold/iggen/h3_mean_pairwise_rmsd_CA_complete_link_1.50A.csv"
OAS_CSV   = "/home/alanwu/Documents/iggen_model/evaluation_metrics/multiseed_alphafold/oas_v5/h3_mean_pairwise_rmsd_CA_complete_link_1.50A.csv"

OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/multiseed_alphafold"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_VIOLIN = os.path.join(OUT_DIR, "h3_rmsd_iggen_vs_oas_violin.png")
OUT_BAR    = os.path.join(OUT_DIR, "h3_rmsd_percentage_bar.png")

RMSD_COL = "mean_pairwise_h3_rmsd_CA_A"
THRESHOLD = 1.5  # Å


def pvalue_to_sig_label(pvalue):
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def load_rmsd_values(csv_path):
    df = pd.read_csv(csv_path)

    if RMSD_COL not in df.columns:
        raise RuntimeError(f"Column '{RMSD_COL}' not found in {csv_path}")

    vals = pd.to_numeric(df[RMSD_COL], errors="coerce")
    vals = vals[np.isfinite(vals)].to_numpy(dtype=float)

    if len(vals) == 0:
        raise RuntimeError(f"No valid RMSD values found in {csv_path}")

    return vals


def plot_violin(iggen_vals, oas_vals, outpath):
    fig, ax = plt.subplots(figsize=(5, 6))

    data = [iggen_vals, oas_vals]
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
        median = np.median(vals)
        ax.text(
            pos,
            median + 0.02,
            "{:.3g} Å".format(median),
            ha="center",
            va="bottom",
            fontsize=11
        )

    # Mann-Whitney U test on mean pairwise RMSD distributions
    _, pvalue = mannwhitneyu(iggen_vals, oas_vals, alternative="two-sided")
    sig_label = pvalue_to_sig_label(pvalue)
    y_data_max = max(np.max(iggen_vals), np.max(oas_vals))
    bracket_y = y_data_max + 0.08
    bracket_h = 0.04
    text_y = bracket_y + bracket_h + 0.01
    ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
    ax.text(1.5, text_y, sig_label, ha="center", va="bottom", fontsize=12)
    print("Mann-Whitney U (Baseline vs Finetuned): p={:.6g} ({})".format(pvalue, sig_label))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean pairwise CDR H3 RMSD [Å]", fontsize=12)
    ax.set_title("CDR H3 conformation variability", fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", outpath)


def compute_percentage_below(vals, threshold):
    return 100.0 * np.sum(vals < threshold) / len(vals)


def plot_percentage_bar(iggen_vals, oas_vals, outpath):
    fig, ax = plt.subplots(figsize=(5, 6))

    labels = ["Baseline", "Finetuned"]
    colors = ["#f5c87a", "#a8c8e8"]

    iggen_pct = compute_percentage_below(iggen_vals, THRESHOLD)
    oas_pct   = compute_percentage_below(oas_vals, THRESHOLD)

    values = [iggen_pct, oas_pct]

    bars = ax.bar([1, 2], values, color=colors, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            "{:.3g}%".format(val),
            ha="center",
            va="bottom",
            fontsize=11
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"% with only one H3 cluster (<{THRESHOLD:.3g} Å)", fontsize=12)
    ax.set_title("CDR H3 conformation variability", fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", outpath)


# =========================
# RUN
# =========================
iggen_vals = load_rmsd_values(IGGEN_CSV)
oas_vals   = load_rmsd_values(OAS_CSV)

print("IgGen:      n={} median={:.3g} Å mean={:.3g} Å".format(
    len(iggen_vals), np.median(iggen_vals), np.mean(iggen_vals)
))
print("SAbDab+OAS: n={} median={:.3g} Å mean={:.3g} Å".format(
    len(oas_vals), np.median(oas_vals), np.mean(oas_vals)
))

print("IgGen:      {:.3g}% below {:.3g} Å".format(
    compute_percentage_below(iggen_vals, THRESHOLD), THRESHOLD
))
print("SAbDab+OAS: {:.3g}% below {:.3g} Å".format(
    compute_percentage_below(oas_vals, THRESHOLD), THRESHOLD
))

plot_violin(iggen_vals, oas_vals, OUT_VIOLIN)
plot_percentage_bar(iggen_vals, oas_vals, OUT_BAR)