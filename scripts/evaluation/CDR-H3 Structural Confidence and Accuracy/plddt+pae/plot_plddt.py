import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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
        "label": "SAbDab finetune",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/1A",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/1A/generated_anarci_chothia.txt",
    },
    {
        "label": "SAbDab + 1000 OAS",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/1000",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/1000/generated_anarci_chothia.txt",
    },
    {
        "label": "SAbDab + 2000 OAS",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/2000",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/2000/generated_anarci_chothia.txt",
    },
    {
        "label": "SAbDab + 3000 OAS",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/3000",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/3000/generated_anarci_chothia.txt",
    },
    {
        "label": "SAbDab + 4000 OAS",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/4000",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4000/generated_anarci_chothia.txt",
    },
    {
        "label": "SAbDab + 4813 OAS",
        "run_dir": "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/4813",
        "anarci_file": "/home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_anarci_chothia.txt",
    },
]

OUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/plddt"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "h3_plddt_violin_all.png")


def is_entry_header(line: str):
    if not line.startswith("# "):
        return None

    rest = line[2:].strip()

    if rest.startswith(("ANARCI", "Domain", "Most", "Scheme")):
        return None
    if rest.startswith("|") or rest.startswith("-") or rest.startswith("species"):
        return None

    entry_id = rest.split()[0]
    if re.match(r"^[A-Za-z0-9_.-]+$", entry_id):
        return entry_id
    return None


def parse_anarci_chothia(filepath):
    results = {}

    current_id = None
    seq_index = 0
    h3_indices = []

    def flush():
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

            if current_id is not None and line.startswith("H "):
                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    pos = int(parts[1])
                except ValueError:
                    continue

                aa = parts[-1]

                if aa in ("-", ".", "X"):
                    continue

                if 95 <= pos <= 102:
                    h3_indices.append(seq_index)

                seq_index += 1

    flush()
    results.pop(None, None)
    return results


def collect_h3_means(run_dir, anarci_file):
    scores_glob = os.path.join(
        run_dir,
        "*_scores_rank_00[1-5]_alphafold2_multimer_v2_model_*_seed_000.json"
    )

    anarci_data = parse_anarci_chothia(anarci_file)

    json_files = sorted(glob.glob(scores_glob))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {run_dir}")

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
            rank_to_file.setdefault(rank, fp)
        selected_files = [rank_to_file[r] for r in sorted(rank_to_file)]

        plddt_arrays = []
        for fp in selected_files:
            with open(fp, "r") as f:
                data = json.load(f)
            if "plddt" in data:
                plddt_arrays.append(np.array(data["plddt"], dtype=float))

        if not plddt_arrays:
            continue

        min_len = min(len(arr) for arr in plddt_arrays)
        matrix = np.vstack([arr[:min_len] for arr in plddt_arrays])
        mean_plddt = np.mean(matrix, axis=0)

        if max(h3_indices) >= len(mean_plddt):
            continue

        h3_vals = mean_plddt[h3_indices]
        h3_means.append(float(np.mean(h3_vals)))

    print(f"{os.path.basename(run_dir)}: n={len(h3_means)}")
    return h3_means


def pvalue_to_sig_label(pvalue):
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def plot_violin(data, labels, outpath):
    fig_width = max(10, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # removed SAbDab+OAS group, and reused its blue for 4813
    colors = [
        "#f5c87a",  # Baseline
        "#d9c27a",  # Random finetune
        "#b8d39a",  # SAbDab finetune
        "#d6b3e6",  # SAbDab + 1000 OAS
        "#f2b5c4",  # SAbDab + 2000 OAS
        "#b7dbe8",  # SAbDab + 3000 OAS
        "#c7d89b",  # SAbDab + 4000 OAS
        "#a8c8e8",  # SAbDab + 4813 OAS (old SAbDab+OAS blue)
    ]

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

    for vals, pos in zip(data, positions):
        mean_val = np.mean(vals)
        ax.text(
            pos,
            mean_val + 0.8,
            f"{mean_val:.3g}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Mann-Whitney U test: Baseline vs SAbDab + 3000 OAS
    baseline_label = "Baseline"
    target_label = "SAbDab + 3000 OAS"
    if baseline_label in labels and target_label in labels:
        idx_a = labels.index(baseline_label)
        idx_b = labels.index(target_label)
        group_a = np.asarray(data[idx_a], dtype=float)
        group_b = np.asarray(data[idx_b], dtype=float)

        if len(group_a) > 0 and len(group_b) > 0:
            _, pvalue = mannwhitneyu(group_a, group_b, alternative="two-sided")
            sig_label = pvalue_to_sig_label(pvalue)

            x1, x2 = positions[idx_a], positions[idx_b]
            y_data_max = max(np.max(np.asarray(vals, dtype=float)) for vals in data if len(vals) > 0)
            bracket_y = y_data_max + 2.0
            bracket_h = 0.8
            text_y = bracket_y + bracket_h + 0.2

            ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], c="black", lw=1.2)
            ax.text((x1 + x2) / 2, text_y, sig_label, ha="center", va="bottom", fontsize=12)

            print(f"Mann-Whitney U (Baseline vs SAbDab + 3000 OAS): p={pvalue:.6g} ({sig_label})")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Mean H3 pLDDT")
    ax.set_title("Mean per H3 loop pLDDT")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.3, len(labels) + 0.7)
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