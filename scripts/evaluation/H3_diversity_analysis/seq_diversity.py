#!/usr/bin/env python3
"""
H3 Sequence Diversity Analysis
================================
Measures the within-set sequence diversity of generated CDR-H3 loops for two
model variants — a Baseline (IgGen) and a Finetuned (OAS) model — and produces
summary statistics and publication-ready plots.

Two complementary diversity metrics are computed:

1. **Nearest-Neighbour (NN) Identity**
   For every sequence in a set, the closest other sequence (by global pairwise
   alignment identity) within the *same* set is found.  The distribution of these
   NN identity scores characterises within-set redundancy: a lower mean means
   sequences are more diverse relative to each other.

2. **Unique Fraction**
   The fraction of sequences whose exact string is unique within the set.
   A value of 1.0 means every generated sequence is distinct.

Inputs
------
- IGGEN_FASTA  : FASTA of IgGen-generated CDR-H3 sequences (Chothia-numbered).
- OAS_FASTA    : FASTA of OAS-finetuned model CDR-H3 sequences.

Outputs (written to IGGEN_OUT_DIR, OAS_OUT_DIR, and COMBINED_OUT_DIR)
----------------------------------------------------------------------
- *_nn_identity_per_sequence.tsv : Per-sequence NN identity table.
- *_diversity_summary.tsv        : Single-row summary (total, unique count, fractions).
- *_nn_identity_distribution.png : Histogram of NN identity for one group.
- combined_nn_identity_distribution.png : Overlapping histograms for both groups.
- combined_unique_fraction_bar.png      : Side-by-side bar chart of unique fractions.

Algorithm note
--------------
Global Needleman-Wunsch alignment is used with a simple match/mismatch/gap
scheme.  Identity is defined as (exact matches) / max(len(seq1), len(seq2)),
which penalises length differences — important for CDR-H3 loops whose length
varies considerably.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO

plt.rcParams.update({'font.size': 14})
from Bio.Align import PairwiseAligner


# ============================================
# INPUTS
# ============================================
IGGEN_FASTA = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_h3_chothia.fasta"
OAS_FASTA   = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/oas/oas_v6/generated_h3_chothia.fasta"

IGGEN_OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/sequence_level/iggen"
OAS_OUT_DIR   = "/home/alanwu/Documents/iggen_model/evaluation_metrics/sequence_level/oas_v6"
COMBINED_OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/sequence_level/diversity"

os.makedirs(IGGEN_OUT_DIR, exist_ok=True)
os.makedirs(OAS_OUT_DIR, exist_ok=True)
os.makedirs(COMBINED_OUT_DIR, exist_ok=True)

# Consistent colour scheme shared with other evaluation scripts in this project
GROUP_COLORS = {
    "Baseline": "#f5c87a",
    "Finetuned": "#a8c8e8",
}


# ============================================
# HELPERS
# ============================================
def read_fasta_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).strip().upper()
        if seq:
            records.append((rec.id, seq))
    return records


def sequence_identity(seq1: str, seq2: str, aligner: PairwiseAligner) -> float:
    """
    Global alignment identity:
    matches / max(len(seq1), len(seq2))
    """
    alignment = aligner.align(seq1, seq2)[0]
    aln1 = alignment[0]
    aln2 = alignment[1]

    matches = sum(a == b for a, b in zip(aln1, aln2) if a != "-" and b != "-")
    # Normalise by the longer sequence so that a perfect subsequence match does
    # not yield 100% identity — a meaningful penalty for length mismatches in H3
    denom = max(len(seq1), len(seq2))
    return matches / denom if denom > 0 else 0.0


def compute_nn_identity(records: List[Tuple[str, str]]) -> List[Dict]:
    """
    For each sequence, compute nearest-neighbour identity within the same set.
    """
    if len(records) < 2:
        raise ValueError("Need at least 2 sequences to compute nearest-neighbour identity.")

    # Gap penalties chosen to discourage spurious gapped alignments on short H3
    # loops; open gap is heavier than extend so single large gaps are preferred
    # over many small ones, which is more biologically sensible for CDR loops
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5

    results = []

    for i, (seq_id, seq) in enumerate(records):
        best_identity = -1.0
        best_neighbor_id = None

        for j, (other_id, other_seq) in enumerate(records):
            if i == j:
                continue
            ident = sequence_identity(seq, other_seq, aligner)
            if ident > best_identity:
                best_identity = ident
                best_neighbor_id = other_id

        results.append({
            "id": seq_id,
            "sequence": seq,
            "length": len(seq),
            "nn_id": best_neighbor_id,
            "nn_identity": best_identity
        })

    return results


def compute_unique_fraction(records: List[Tuple[str, str]]) -> Tuple[int, int, float]:
    seqs = [seq for _, seq in records]
    total = len(seqs)
    unique = len(set(seqs))
    frac = unique / total if total > 0 else np.nan
    return unique, total, frac


def save_per_sequence_nn_tsv(nn_results: List[Dict], group_name: str, outpath: str) -> None:
    with open(outpath, "w") as f:
        f.write("group\tid\tsequence\tlength\tnearest_neighbor_id\tnearest_neighbor_identity\n")
        for row in nn_results:
            f.write(
                f"{group_name}\t{row['id']}\t{row['sequence']}\t{row['length']}\t"
                f"{row['nn_id']}\t{row['nn_identity']:.6f}\n"
            )


def save_summary_tsv(
    group_name: str,
    n_total: int,
    n_unique: int,
    unique_fraction: float,
    mean_nn_identity: float,
    outpath: str
) -> None:
    with open(outpath, "w") as f:
        f.write("group\tn_total\tn_unique\tunique_fraction\tmean_nn_identity\n")
        f.write(
            f"{group_name}\t{n_total}\t{n_unique}\t"
            f"{unique_fraction:.6f}\t{mean_nn_identity:.6f}\n"
        )


def plot_single_group_nn_distribution(vals: List[float], group_name: str, outpath: str) -> None:
    vals = np.array(vals, dtype=float)
    color = GROUP_COLORS[group_name]

    fig, ax = plt.subplots(figsize=(6, 5))
    # 30 bins over [0, 1] gives 0.033-wide bins, appropriate resolution for
    # identity scores that typically span ~0.2–1.0 for H3 loops
    bins = np.linspace(0, 1, 31)

    ax.hist(vals, bins=bins, alpha=0.8, color=color, edgecolor="black", linewidth=0.5)

    mean_val = np.mean(vals)
    ax.axvline(mean_val, linestyle="--", linewidth=2, color="black")

    # Annotate the mean line; placed near the top with slight left offset so
    # it does not overlap the bar that the line passes through
    ymax = ax.get_ylim()[1]
    ax.text(
        mean_val,
        ymax * 0.92,
        f"Mean = {mean_val:.3f}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    )

    ax.set_xlabel("Nearest-neighbour identity")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_combined_nn_distribution(group_to_nn: Dict[str, List[float]], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bins = np.linspace(0, 1, 31)

    # alpha=0.45 keeps both histograms visible when they overlap
    for group_name, vals in group_to_nn.items():
        vals = np.array(vals, dtype=float)
        color = GROUP_COLORS[group_name]
        mean_val = np.mean(vals)

        ax.hist(
            vals,
            bins=bins,
            alpha=0.45,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{group_name} (mean = {mean_val:.3f})"
        )

    ax.set_xlabel("Nearest-neighbour identity")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_combined_unique_fraction(unique_summary: Dict[str, Dict[str, float]], outpath: str) -> None:
    groups = list(unique_summary.keys())
    fracs = [unique_summary[g]["unique_fraction"] for g in groups]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(groups, fracs, color=colors, edgecolor="black", linewidth=0.8)

    # Label each bar with its exact value to avoid needing the y-axis for
    # precise reading
    for bar, val in zip(bars, fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=14
        )

    ax.set_ylabel("Unique fraction")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ============================================
# MAIN
# ============================================
def main():
    # "Baseline" = raw IgGen output, "Finetuned" = OAS-finetuned variant
    datasets = {
        "Baseline": {
            "fasta": IGGEN_FASTA,
            "out_dir": IGGEN_OUT_DIR,
            "tag": "iggen",
        },
        "Finetuned": {
            "fasta": OAS_FASTA,
            "out_dir": OAS_OUT_DIR,
            "tag": "oas_v6",
        },
    }

    # Accumulate across both groups so combined plots can be made at the end
    group_to_nn = {}
    unique_summary = {}

    for group_name, info in datasets.items():
        fasta_path = info["fasta"]
        out_dir = info["out_dir"]
        tag = info["tag"]

        records = read_fasta_sequences(fasta_path)
        if len(records) == 0:
            raise ValueError(f"No sequences found in {fasta_path}")

        # O(n^2) all-vs-all within the set — acceptable for typical generated
        # set sizes of a few hundred to low thousands of sequences
        nn_results = compute_nn_identity(records)
        nn_values = [x["nn_identity"] for x in nn_results]
        mean_nn_identity = float(np.mean(nn_values))

        n_unique, n_total, unique_fraction = compute_unique_fraction(records)

        group_to_nn[group_name] = nn_values
        unique_summary[group_name] = {
            "n_unique": n_unique,
            "n_total": n_total,
            "unique_fraction": unique_fraction,
        }

        save_per_sequence_nn_tsv(
            nn_results,
            group_name,
            os.path.join(out_dir, f"{tag}_nn_identity_per_sequence.tsv")
        )

        save_summary_tsv(
            group_name,
            n_total,
            n_unique,
            unique_fraction,
            mean_nn_identity,
            os.path.join(out_dir, f"{tag}_diversity_summary.tsv")
        )

        plot_single_group_nn_distribution(
            nn_values,
            group_name,
            os.path.join(out_dir, f"{tag}_nn_identity_distribution.png")
        )

        print(f"\n{group_name}")
        print(f"  Total sequences:  {n_total}")
        print(f"  Unique sequences: {n_unique}")
        print(f"  Unique fraction:  {unique_fraction:.3f}")
        print(f"  Mean NN identity: {mean_nn_identity:.3f}")

    plot_combined_nn_distribution(
        group_to_nn,
        os.path.join(COMBINED_OUT_DIR, "combined_nn_identity_distribution.png")
    )

    plot_combined_unique_fraction(
        unique_summary,
        os.path.join(COMBINED_OUT_DIR, "combined_unique_fraction_bar.png")
    )

    print("\nSaved combined plots:")
    print(os.path.join(COMBINED_OUT_DIR, "combined_nn_identity_distribution.png"))
    print(os.path.join(COMBINED_OUT_DIR, "combined_unique_fraction_bar.png"))


if __name__ == "__main__":
    main()
