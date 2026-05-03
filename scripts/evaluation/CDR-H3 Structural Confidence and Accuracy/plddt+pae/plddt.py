"""
plot_plddt.py
=============
Extracts per-entry mean CDR-H3 pLDDT scores from ColabFold JSON outputs and
produces a violin plot comparing multiple model configurations (Baseline,
Random finetune, Finetuned).

Overview
--------
ColabFold (AlphaFold2 Multimer) stores per-residue pLDDT confidence scores
(0–100, higher = more confident) in JSON files alongside its PDB outputs.
This script:
  1. Parses ANARCI Chothia-numbered output files to identify which 0-based
     sequence indices correspond to CDR-H3 (Chothia positions 95–102) in
     each generated antibody.
  2. Loads the ColabFold score JSON files (ranks 1–5) for each entry and
     averages the pLDDT arrays across model runs to get a stable estimate.
  3. Extracts the H3-specific pLDDT values using the ANARCI indices and
     computes one mean per entry.
  4. Renders a violin plot comparing distributions across model runs, with
     a Mann-Whitney U significance bracket between Baseline and Finetuned.


Inputs
------
  Per-run ColabFold output directories (JSON score files matching the
  SCORES_PATTERN glob). Configured via RUN_CONFIGS.
  Per-run ANARCI Chothia-numbered output files.

Outputs
-------
  OUT_DIR/h3_plddt_violin_all.png  — violin plot of per-entry mean H3 pLDDT


Update RUN_CONFIGS and OUT_DIR to point to your local directories.
"""

import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({'font.size': 14})
from scipy.stats import mannwhitneyu

# =========================
# CONFIG
# =========================
RUN_CONFIGS = [
    {
        "label": "Baseline",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/iggen",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_anarci_chothia.txt",
    },
    {
        "label": "Random finetune",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/random",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/random/generated_anarci_chothia.txt",
    },
    {
        "label": "Finetuned",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/3000",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/3000/generated_anarci_chothia.txt",
    },
]

OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/plddt"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "h3_plddt_violin_all.png")


def is_entry_header(line: str):
    """
    Return the entry ID if this ANARCI line is a genuine entry header
    (e.g. '# ab_1'), otherwise return None.

    ANARCI files contain many comment-style lines beginning with '# ' that
    are metadata (version, scheme, species table). This function filters them
    out by checking for known prefixes and validating the resulting token
    against an alphanumeric pattern.
    """
    if not line.startswith("# "):
        return None

    rest = line[2:].strip()

    # Known ANARCI metadata prefixes to ignore
    if rest.startswith(("ANARCI", "Domain", "Most", "Scheme")):
        return None
    if rest.startswith("|") or rest.startswith("-") or rest.startswith("species"):
        return None

    entry_id = rest.split()[0]
    if re.match(r"^[A-Za-z0-9_.-]+$", entry_id):
        return entry_id
    return None


def parse_anarci_chothia(filepath):
    """
    Parse an ANARCI Chothia-numbered file and return the 0-based sequence
    indices of CDR-H3 (Chothia positions 95–102) for each entry.

    The 0-based index counts only real (non-gap) residues on the heavy chain,
    so the resulting indices directly address the pLDDT array returned by
    ColabFold (which also contains only real residues).

    Parameters
    ----------
    filepath : str
        Path to the ANARCI output file.

    Returns
    -------
    dict
        Maps entry_id -> {"h3_indices": list of int}
    """
    results = {}

    current_id = None
    seq_index = 0
    h3_indices = []

    def flush():
        """Commit the current entry's H3 indices to results and reset state."""
        nonlocal current_id, seq_index, h3_indices
        if current_id is not None:
            results[current_id] = {"h3_indices": h3_indices.copy()}
        current_id = None
        seq_index = 0
        h3_indices = []

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

            # Only process heavy-chain lines; light chain not needed here.
            if current_id is not None and line.startswith("H "):
                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    pos = int(parts[1])
                except ValueError:
                    continue

                aa = parts[-1]

                # Skip ANARCI gap characters — they are alignment padding with
                # no corresponding position in the actual sequence.
                if aa in ("-", ".", "X"):
                    continue

                if 95 <= pos <= 102:
                    h3_indices.append(seq_index)

                # Increment for every real residue so seq_index stays aligned
                # with the ColabFold pLDDT array.
                seq_index += 1

    flush()
    results.pop(None, None)
    return results


def collect_h3_means(run_dir, anarci_file):
    """
    For one model run, extract per-entry mean CDR-H3 pLDDT scores.

    Workflow:
      1. Discover all ColabFold score JSON files in run_dir.
      2. Group them by entry, then deduplicate by rank (first file wins).
      3. Load pLDDT arrays from each rank file, average them.
      4. Slice out the H3 residues using ANARCI-derived indices.
      5. Return one mean pLDDT value per entry.

    Parameters
    ----------
    run_dir : str
        Directory containing ColabFold JSON score files.
    anarci_file : str
        Path to the ANARCI Chothia output file for this run's sequences.

    Returns
    -------
    list of float
        Per-entry mean H3 pLDDT values. Entries without ANARCI data, H3
        residues, or valid JSON files are silently skipped.
    """
    scores_glob = os.path.join(
        run_dir,
        "*_scores_rank_00[1-5]_alphafold2_multimer_v2_model_*_seed_000.json"
    )

    anarci_data = parse_anarci_chothia(anarci_file)

    json_files = sorted(glob.glob(scores_glob))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {run_dir}")

    # Group all rank files for each entry together so we can average across them.
    entry_to_files = defaultdict(list)
    entry_regex = re.compile(r"^(?P<entry>.+?)_scores_rank_00[1-5]_")
    rank_regex = re.compile(r"_scores_rank_(00[1-5])_")

    for fpath in json_files:
        base = os.path.basename(fpath)
        m = entry_regex.match(base)
        if m:
            entry_to_files[m.group("entry")].append(fpath)

    h3_means = []

    for entry_id, files in sorted(entry_to_files.items()):
        if entry_id not in anarci_data:
            continue

        h3_indices = sorted(set(anarci_data[entry_id]["h3_indices"]))
        if not h3_indices:
            continue

        # Keep one deterministic file per rank to avoid run-to-run drift when
        # duplicate rank files exist for an entry.
        rank_to_file = {}
        for fp in sorted(files):
            base = os.path.basename(fp)
            rank_match = rank_regex.search(base)
            if not rank_match:
                continue
            rank = int(rank_match.group(1))
            rank_to_file.setdefault(rank, fp)  # setdefault keeps the first seen
        selected_files = [rank_to_file[r] for r in sorted(rank_to_file)]

        plddt_arrays = []
        for fp in selected_files:
            with open(fp, "r") as f:
                data = json.load(f)
            if "plddt" in data:
                plddt_arrays.append(np.array(data["plddt"], dtype=float))

        if not plddt_arrays:
            continue

        # Trim to the shortest array before averaging in case models differ
        # in padding (rare but possible in multimer mode).
        min_len = min(len(arr) for arr in plddt_arrays)
        matrix = np.vstack([arr[:min_len] for arr in plddt_arrays])
        mean_plddt = np.mean(matrix, axis=0)

        # Guard against ANARCI indices that exceed the pLDDT array length,
        # which can happen when the ColabFold sequence is shorter than expected.
        if max(h3_indices) >= len(mean_plddt):
            continue

        h3_vals = mean_plddt[h3_indices]
        h3_means.append(float(np.mean(h3_vals)))

    print(f"{os.path.basename(run_dir)}: n={len(h3_means)}")
    return h3_means


def pvalue_to_sig_label(pvalue):
    """
    Convert a p-value to the conventional star-notation significance label.

    Returns
    -------
    str
        "***" (p < 0.001), "**" (p < 0.01), "*" (p < 0.05), or "ns".
    """
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def plot_violin(data, labels, outpath):
    """
    Render a violin plot of per-entry mean H3 pLDDT across model configurations,
    with mean value annotations and a significance bracket between "Baseline"
    and "Finetuned" groups.

    Parameters
    ----------
    data : list of list of float
        Parallel to labels; each inner list contains per-entry H3 pLDDT means.
    labels : list of str
        Display labels for each violin (must be parallel to data).
    outpath : str
        File path to save the PNG.
    """
    # Width scales with the number of violins so annotations don't crowd.
    fig_width = max(10, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    colors = [
        "#f5c87a",  # Baseline
        "#d9c27a",  # Random finetune
        "#b7dbe8",  # Finetuned (SAbDab + 3000 OAS)
    ]

    positions = list(range(1, len(labels) + 1))

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.6,
        showmedians=False,  # Drawn manually below
        showextrema=False   # Hide min/max lines for a cleaner look
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.85)

    # Annotate each violin with the group mean (not median) — mean is used here
    # because pLDDT is bounded and approximately symmetric within a group.
    for vals, pos in zip(data, positions):
        mean_val = np.mean(vals)
        ax.text(
            pos,
            mean_val + 0.8,  # Fixed 0.8-point offset above mean on 0–100 scale
            f"{mean_val:.3g}",
            ha="center",
            va="bottom",
            fontsize=14
        )

    # Mann-Whitney U test: Baseline vs Finetuned
    # Two-sided test is used because we have no a priori directional hypothesis —
    # finetuning could in principle increase or decrease H3 pLDDT.
    baseline_label = "Baseline"
    target_label = "Finetuned"
    if baseline_label in labels and target_label in labels:
        idx_a = labels.index(baseline_label)
        idx_b = labels.index(target_label)
        group_a = np.asarray(data[idx_a], dtype=float)
        group_b = np.asarray(data[idx_b], dtype=float)

        if len(group_a) > 0 and len(group_b) > 0:
            _, pvalue = mannwhitneyu(group_a, group_b, alternative="two-sided")
            sig_label = pvalue_to_sig_label(pvalue)

            # Draw significance bracket between the two comparison groups.
            # Bracket is placed 2 pLDDT units above the highest data point.
            x1, x2 = positions[idx_a], positions[idx_b]
            y_data_max = max(np.max(np.asarray(vals, dtype=float)) for vals in data if len(vals) > 0)
            bracket_y = y_data_max + 2.0
            bracket_h = 0.8
            text_y = bracket_y + bracket_h + 0.2

            ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
            ax.text((x1 + x2) / 2, text_y, sig_label, ha="center", va="bottom", fontsize=14)

            print(f"Mann-Whitney U (Baseline vs Finetuned): p={pvalue:.6g} ({sig_label})")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=14)
    ax.set_ylabel("Mean H3 pLDDT")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.3, len(labels) + 0.7)
    # Enforce a y-axis ceiling of at least 100 (the pLDDT upper bound) so the
    # significance bracket always has headroom and the scale is consistent.
    current_ymax = ax.get_ylim()[1]
    ax.set_ylim(0, max(current_ymax, 100))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", outpath)


# =========================
# RUN
# =========================
all_data = []
all_labels = []

for cfg in RUN_CONFIGS:
    vals = collect_h3_means(cfg["run_dir"], cfg["anarci_file"])
    all_data.append(vals)
    all_labels.append(cfg["label"])

    print(
        f"{cfg['label']}: n={len(vals)} median={np.median(vals):.3g} mean={np.mean(vals):.3g}"
    )

plot_violin(all_data, all_labels, OUT_PNG)
