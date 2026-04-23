#!/usr/bin/env python3

import shutil
import subprocess
from pathlib import Path
from Bio import SeqIO

# ================= CONFIG =================
DATASETS = [
    {
        "tag": "1279065_1",
        "fv_fasta": "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/native_healthy/1279065_1/paired_fv_trimmed.fasta",
        "cdr_fasta": "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1279065_1/cdr_loops.fasta",
    },
    {
        "tag": "1279073_1",
        "fv_fasta": "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/native_healthy/1279073_1/paired_fv_trimmed.fasta",
        "cdr_fasta": "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1279073_1/cdr_loops.fasta",
    },
    {
        "tag": "1287155_1",
        "fv_fasta": "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/native_healthy/1287155_1/paired_fv_trimmed.fasta",
        "cdr_fasta": "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1287155_1/cdr_loops.fasta",
    },
]

MIN_SEQ_ID = 0.99
COVERAGE   = 0.90
COV_MODE   = 0

# Final output AFTER:
# 1) pooling all datasets
# 2) clustering on CDRH3 only
# 3) choosing one representative per cluster
# 4) removing representatives/members from 1287155_1 by preferring a non-1287155 member
OUT_FASTA = Path(
    "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/native_healthy/paired_fv_trimmed_h3mmseq99_cov0.9_rep_combined_no1287155.fasta"
)

# no spaces for mmseqs temp work
SAFE_TMP_ROOT = Path("/home/alanwu/Documents/iggen_model/tmp_mmseqs_h3_cluster_combined")
EXCLUDE_TAG = "1287155_1"
# =========================================


def run(cmd: list[str]):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalize_base_id(s: str) -> str:
    """
    Normalize IDs so FV and CDR files can match.

    Examples:
      AAAC...__...|VH:VL  -> AAAC...__...
      AAAC...__...|CDRH3  -> AAAC...__...
    """
    s = str(s).strip()
    s = s.split()[0]
    s = s.split("|", 1)[0]
    return s.strip()


def write_fasta_wrapped(path: Path, seq_map: dict[str, str], id_list: list[str], wrap: int = 80):
    with path.open("w") as f:
        for rid in id_list:
            seq = seq_map[rid]
            f.write(f">{rid}\n")
            for i in range(0, len(seq), wrap):
                f.write(seq[i:i + wrap] + "\n")


def load_fv_fasta(path: Path) -> dict[str, str]:
    """
    Load paired_fv_trimmed.fasta as:
      {base_id: full_vh:vl_sequence}
    """
    seqs = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        base_id = normalize_base_id(rec.id)
        seq = str(rec.seq).strip().upper()
        if base_id and seq:
            seqs[base_id] = seq
    return seqs


def load_cdrh3_only(path: Path) -> dict[str, str]:
    """
    From cdr_loops.fasta, keep only entries ending with |CDRH3
    Return:
      {base_id: h3_seq}
    """
    h3_map = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        raw_id = str(rec.id).strip()
        if not raw_id.endswith("|CDRH3"):
            continue
        base_id = normalize_base_id(raw_id)
        seq = str(rec.seq).strip().upper()
        if base_id and seq:
            h3_map[base_id] = seq
    return h3_map


def unique_id(rid: str, used: set[str], tag: str) -> str:
    """
    Preserve original ID if unique.
    If duplicated across datasets, append |tag, then |tag|2, etc.
    """
    if rid not in used:
        return rid
    rid2 = f"{rid}|{tag}"
    k = 2
    while rid2 in used:
        rid2 = f"{rid}|{tag}|{k}"
        k += 1
    return rid2


def main():
    if SAFE_TMP_ROOT.exists():
        shutil.rmtree(SAFE_TMP_ROOT)
    SAFE_TMP_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        run(["mmseqs", "version"])
    except Exception as e:
        raise RuntimeError(
            "mmseqs not found or not runnable. Try `which mmseqs` and `mmseqs version`."
        ) from e

    pooled_fv_map: dict[str, str] = {}
    pooled_h3_map: dict[str, str] = {}
    id_source_map: dict[str, str] = {}
    used_ids: set[str] = set()

    total_fv_loaded = 0
    total_h3_loaded = 0
    total_shared = 0

    print("\n=== Loading and pooling datasets ===")
    for ds in DATASETS:
        tag = ds["tag"]
        fv_fasta = Path(ds["fv_fasta"])
        cdr_fasta = Path(ds["cdr_fasta"])

        if not fv_fasta.exists():
            print(f"[warning] FV FASTA missing, skipping: {fv_fasta}")
            continue
        if not cdr_fasta.exists():
            print(f"[warning] CDR FASTA missing, skipping: {cdr_fasta}")
            continue

        fv_map_raw = load_fv_fasta(fv_fasta)
        h3_map_raw = load_cdrh3_only(cdr_fasta)

        total_fv_loaded += len(fv_map_raw)
        total_h3_loaded += len(h3_map_raw)

        fv_ids = set(fv_map_raw.keys())
        h3_ids = set(h3_map_raw.keys())
        shared_ids = sorted(fv_ids & h3_ids)
        total_shared += len(shared_ids)

        print(f"\n[{tag}]")
        print(f"  FV loaded:      {len(fv_map_raw)}")
        print(f"  CDRH3 loaded:   {len(h3_map_raw)}")
        print(f"  Shared IDs:     {len(shared_ids)}")

        print("  Example FV IDs:")
        for x in list(sorted(fv_ids))[:5]:
            print(f"    {x}")

        print("  Example CDRH3 IDs:")
        for x in list(sorted(h3_ids))[:5]:
            print(f"    {x}")

        if len(shared_ids) == 0:
            print("  Example FV-only IDs:")
            for x in list(sorted(fv_ids - h3_ids))[:5]:
                print(f"    {x}")
            print("  Example H3-only IDs:")
            for x in list(sorted(h3_ids - fv_ids))[:5]:
                print(f"    {x}")
            continue

        for rid in shared_ids:
            new_id = unique_id(rid, used_ids, tag)
            used_ids.add(new_id)

            pooled_fv_map[new_id] = fv_map_raw[rid]
            pooled_h3_map[new_id] = h3_map_raw[rid]
            id_source_map[new_id] = tag

    if not pooled_fv_map or not pooled_h3_map:
        raise RuntimeError(
            "No overlapping FV/CDRH3 sequences were found across the datasets.\n"
            "Check the printed example FV IDs and CDRH3 IDs above to see how they differ."
        )

    all_ids = list(pooled_h3_map.keys())

    print("\n=== Combined pool summary ===")
    print(f"Total FV loaded:         {total_fv_loaded}")
    print(f"Total CDRH3 loaded:      {total_h3_loaded}")
    print(f"Total shared raw IDs:    {total_shared}")
    print(f"Total pooled sequences:  {len(all_ids)}")

    # write pooled H3 FASTA to a no-space temp path for mmseqs
    safe_h3_fasta = SAFE_TMP_ROOT / "combined_h3_input.fasta"
    write_fasta_wrapped(safe_h3_fasta, pooled_h3_map, all_ids)

    cluster_prefix = SAFE_TMP_ROOT / "clusters"
    tmp_dir = SAFE_TMP_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTemporary combined H3 FASTA: {safe_h3_fasta}")

    run([
        "mmseqs", "easy-cluster",
        str(safe_h3_fasta),
        str(cluster_prefix),
        str(tmp_dir),
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c", str(COVERAGE),
        "--cov-mode", str(COV_MODE),
    ])

    tsv_path = Path(str(cluster_prefix) + "_cluster.tsv")
    if not tsv_path.exists():
        raise RuntimeError(f"Expected cluster TSV not found: {tsv_path}")

    # representative -> members
    clusters: dict[str, list[str]] = {}
    with tsv_path.open("r") as f:
        for line in f:
            rep, mem = line.rstrip("\n").split("\t")
            clusters.setdefault(rep, []).append(mem)

    # Pick one sequence per cluster, but exclude 1287155_1 from the final population.
    # If a cluster has at least one non-1287155_1 member, keep the first such member.
    # If every member in the cluster is from 1287155_1, drop the whole cluster.
    final_ids = []
    dropped_clusters_all_excluded = 0

    for rep, members in clusters.items():
        chosen = None
        for m in members:
            if m in pooled_fv_map and id_source_map.get(m) != EXCLUDE_TAG:
                chosen = m
                break

        if chosen is None:
            dropped_clusters_all_excluded += 1
            continue

        final_ids.append(chosen)

    if not final_ids:
        raise RuntimeError(
            f"All clusters were removed after excluding sequences from {EXCLUDE_TAG}."
        )

    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    write_fasta_wrapped(OUT_FASTA, pooled_fv_map, final_ids)

    cluster_sizes = [len(members) for members in clusters.values()]
    total_members = sum(cluster_sizes)

    n_clusters = len(clusters)
    n_before_filter = n_clusters
    n_after_filter = len(final_ids)
    n_removed_by_filter = n_before_filter - n_after_filter

    print("\n=== Clustering summary ===")
    print(f"Total pooled input H3s:  {len(all_ids)}")
    print(f"Clusters formed:         {n_clusters}")
    print(f"Total H3 members:        {total_members}")
    print(f"Largest cluster size:    {max(cluster_sizes)}")
    print(f"Smallest cluster size:   {min(cluster_sizes)}")
    print(f"Mean cluster size:       {sum(cluster_sizes)/len(cluster_sizes):.3f}")
    print(f"MMseqs params:           min_seq_id={MIN_SEQ_ID}, coverage={COVERAGE}, cov_mode={COV_MODE}")

    print(f"\n=== Post-filtering (exclude {EXCLUDE_TAG}) ===")
    print(f"One-per-cluster before filter:   {n_before_filter}")
    print(f"Clusters removed entirely:       {dropped_clusters_all_excluded}")
    print(f"Final sequences written:         {n_after_filter}")
    print(f"Total removed by filter:         {n_removed_by_filter}")

    # Optional: show source breakdown of final output
    source_counts = {}
    for rid in final_ids:
        src = id_source_map.get(rid, "UNKNOWN")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\n=== Final source breakdown ===")
    for src in sorted(source_counts):
        print(f"{src}: {source_counts[src]}")

    print("\nDone.")
    print(f"Combined output FASTA:   {OUT_FASTA}")


if __name__ == "__main__":
    main()