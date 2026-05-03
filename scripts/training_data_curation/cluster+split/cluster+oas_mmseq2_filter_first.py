"""
Filter-then-cluster pipeline for antibody Fv training data.

Overview
--------
This script builds the train/val FASTA split used to train the IgGen model.
It draws sequences from two sources:
  - SAbDab: crystallographically resolved antibody Fv structures (high-confidence
    ground-truth structures), filtered by CDRH3 RMSD (ColabFold prediction vs PDB).
  - OAS: paired heavy/light sequences from the Observed Antibody Space, filtered
    by CDRH3 mean pLDDT (AlphaFold2/ColabFold confidence score).

Pipeline stages
---------------
1. Quality filtering FIRST (before clustering):
     - SAbDab entries: kept only if H3_RMSD ≤ SABDAB_H3_RMSD_MAX (2.0 Å).
       RMSD measures how accurately ColabFold can reconstruct the CDR-H3 loop,
       ensuring we only train on structures whose loop geometry can be predicted.
     - OAS entries: kept only if CDRH3_mean_pLDDT ≥ OAS_PLDDT_THRESHOLD (80.0).
       pLDDT ≥ 80 indicates the model is confident about the predicted loop
       structure.
2. CDRH3-based clustering with MMseqs2 (easy-cluster):
     - Clustering is done on CDRH3 sequences alone, not the full Fv.
     - Parameters: 50% sequence identity, 80% coverage of the shorter sequence.
3. Train/val split by cluster (not by individual sequence):
     - Clusters are shuffled and greedily assigned to train until ~90% is reached.
     - Splitting at the cluster level ensures no two sequences from the same cluster
       appear in both train and val (preventing data leakage).
4. Full Fv FASTA output: train.fasta / val.fasta.

Inputs
------
  - FV_FASTA_*      : Paired VH:VL Fv sequences (SAbDab + three OAS batches).
  - SPANS_CSV       : CSV with Chothia-based CDRH3 start/end positions in the
                      concatenated VH+VL sequence for each SAbDab entry.
  - RMSD_XLSX       : Per-antibody H3_RMSD values from ColabFold vs ground truth.
  - OAS_CDR_DIR_*   : FASTA files with CDRH3 sequences for each OAS entry.
  - OAS_PLDDT_DIRS  : Excel tables with per-CDR pLDDT statistics for OAS entries.

Outputs
-------
  - OUT_TRAIN / OUT_VAL : Paired Fv FASTA files for training and validation.

"""

import os
import re
import random
import subprocess
import glob
from pathlib import Path

import pandas as pd
from Bio import SeqIO

# ================= CONFIG =================
# Full Fv FASTA inputs
FV_FASTA_1 = "/home/alanwu/Documents/iggen_model/data/single_fv_pdb.fasta"
FV_FASTA_2 = "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/1279052/paired_fv_trimmed.fasta"
FV_FASTA_3 = "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/1279060/paired_fv_trimmed.fasta"
FV_FASTA_4 = "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/1287155/paired_fv_trimmed.fasta"

# SAbDab metadata
SPANS_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"
RMSD_XLSX = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"

# OAS CDR FASTA dirs/files (CDRH3 directly available here)
OAS_CDR_DIR_1 = "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/1279052/cdr_loops.fasta"
OAS_CDR_DIR_2 = "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/1279060/cdr_loops.fasta"
OAS_CDR_DIR_3 = "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/1287155/cdr_loops.fasta"

# OAS pLDDT tables
OAS_PLDDT_DIRS = [
    "/home/alanwu/Documents/iggen_model/data/training_data/oas data/plddt_table/1279052",
    "/home/alanwu/Documents/iggen_model/data/training_data/oas data/plddt_table/1279060",
    "/home/alanwu/Documents/iggen_model/data/training_data/oas data/plddt_table/1287155",
]

# Outputs
OUT_TRAIN = "/home/alanwu/Documents/iggen_model/data/train.fasta"
OUT_VAL   = "/home/alanwu/Documents/iggen_model/data/val.fasta"

# MMseqs2 clustering params (on CDRH3)
MIN_SEQ_ID = 0.5
COVERAGE   = 0.80
COV_MODE   = 1   # coverage of shorter sequence

TRAIN_RATIO = 0.9
SEED = 42

WORKDIR = "/home/alanwu/Documents/iggen_model/data/mmseqs_h3_split_work"

# Filtering thresholds
SABDAB_H3_RMSD_MAX = 2.0   # Angstroms; entries above this are likely poorly predicted
OAS_PLDDT_THRESHOLD = 80.0  # pLDDT; below this the predicted structure is low-confidence
# =========================================

random.seed(SEED)


# ----------------------------
# Helpers
# ----------------------------
def run(cmd: list[str]):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_fv_fasta(path: str) -> list[SeqIO.SeqRecord]:
    return list(SeqIO.parse(path, "fasta"))


def unique_id(rid: str, used: set[str], tag: str) -> str:
    """
    Avoid ID collisions when pooling sequences from multiple source files.
    If the original ID already exists, append a dataset tag and an incrementing
    counter until uniqueness is achieved.
    """
    if rid not in used:
        return rid
    rid2 = f"{rid}|{tag}"
    k = 2
    while rid2 in used:
        rid2 = f"{rid}|{tag}|{k}"
        k += 1
    return rid2


def write_fasta_wrapped(path: str, seq_map: dict[str, str], id_list: list[str], wrap: int = 80):
    with open(path, "w") as f:
        for rid in id_list:
            f.write(f">{rid}\n")
            s = seq_map[rid]
            # Standard 80-character line wrap for FASTA compatibility
            for k in range(0, len(s), wrap):
                f.write(s[k:k+wrap] + "\n")


def normalize_spans_key(x: str) -> str:
    """
    Extract the bare PDB/antibody ID from a FASTA header that may contain
    pipe-separated fields (e.g. '7w55.pdb|VH:VL' → '7w55.pdb').
    """
    return str(x).strip().split("|", 1)[0]


def normalize_id_for_merge(x: str) -> str:
    """
    Create a canonical lowercase key for joining SAbDab IDs across different
    tables (spans CSV vs RMSD Excel) that may have inconsistent suffixes
    like '.pdb' or pipe annotations.
    """
    x = str(x).strip()
    x = x.replace(".pdb", "")
    x = x.split("|", 1)[0]
    return x.lower()


def canonicalize_oas_key(s: str) -> str:
    """
    Normalize OAS antibody IDs for robust dictionary lookup.
    OAS IDs frequently contain double-underscores or repeated underscores
    from how paired heavy/light chain IDs are concatenated; collapsing them
    prevents spurious lookup misses.
    """
    s = str(s).strip().lower()
    s = s.replace("__", "_")
    s = re.sub(r"_+", "_", s)
    return s


def parse_two_contigs(key: str):
    """
    OAS paired IDs are formed by joining two contig IDs, e.g.:
      'seqA_contig_1_seqB_contig_2'
    This extracts both halves so we can try both orderings when looking up
    a key in the pLDDT table (the two halves may appear in either order
    depending on which FASTA was read first).
    """
    k = canonicalize_oas_key(key)
    m = re.match(r"^(.*?_contig_\d+)_(.*?_contig_\d+)$", k)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def list_fasta_files(path_like: str) -> list[Path]:
    """
    Accept either a direct FASTA file path or a directory.
    When a directory is given, collect all files with recognised FASTA
    extensions; fall back to every file in the directory if none match.
    """
    p = Path(path_like)
    if p.is_dir():
        files = []
        for pat in ("*.fa", "*.fasta", "*.fna", "*.faa", "*.fas", "*.txt"):
            files.extend(sorted(p.glob(pat)))
        if not files:
            files = sorted([x for x in p.iterdir() if x.is_file()])
        return files
    return [p]


def load_and_merge_oas_plddt_tables(oas_dirs: list[str]) -> pd.DataFrame:
    """
    Each OAS batch generates one Excel file of per-antibody CDR pLDDT statistics.
    This concatenates them all and deduplicates by canonical antibody ID so that
    the same entry appearing in multiple batches is counted only once.
    """
    xlsx_files = []
    for d in oas_dirs:
        xlsx_files.extend(glob.glob(os.path.join(d, "**", "*.xlsx"), recursive=True))

    if not xlsx_files:
        raise FileNotFoundError(
            "No .xlsx files found under OAS_PLDDT_DIRS:\n" + "\n".join(oas_dirs)
        )

    dfs = []
    for fp in sorted(xlsx_files):
        dfs.append(pd.read_excel(fp))

    merged = pd.concat(dfs, ignore_index=True)

    if "antibody_id" not in merged.columns:
        raise RuntimeError("Merged OAS table missing column 'antibody_id'.")

    # Deduplicate after canonicalizing IDs; keep first occurrence
    merged["_canon_key"] = merged["antibody_id"].astype(str).map(canonicalize_oas_key)
    merged = merged.drop_duplicates(subset=["_canon_key"], keep="first").drop(columns=["_canon_key"])

    print(f"Loaded OAS pLDDT tables: {len(xlsx_files)} files -> merged rows: {len(merged)}")
    return merged


def build_oas_lookup(oas_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Build a fast lookup dict for OAS pLDDT rows.
    Because paired IDs consist of two contig halves that can appear in either
    order, we register both orderings (A_B and B_A) so that lookups succeed
    regardless of which order the calling code uses.
    """
    lut = {}
    for _, row in oas_df.iterrows():
        ab = canonicalize_oas_key(row["antibody_id"])
        lut[ab] = row
        a, b = parse_two_contigs(ab)
        if a is not None:
            # Register reversed-order key as an alias
            lut[f"{b}_{a}"] = row
    return lut


# ----------------------------
# SAbDab loading/filtering
# ----------------------------
def load_sabdab_metadata(spans_csv: str, rmsd_xlsx: str) -> dict[str, dict]:
    """
    Join the loop-span table (CDRH3 start/end positions within the Fv sequence)
    with the RMSD table (ColabFold prediction quality vs PDB ground truth).
    The join is performed on a normalised antibody ID to handle format
    inconsistencies between the two files.

    Returns a dict keyed by the FASTA header ID of each SAbDab entry.
    """
    spans = pd.read_csv(spans_csv)
    rmsd = pd.read_excel(rmsd_xlsx)

    required_spans = {"Antibody_ID", "fasta_header_id", "H3_start", "H3_end", "Fv_len"}
    required_rmsd = {"Antibody_ID", "H3_RMSD"}

    miss_spans = required_spans - set(spans.columns)
    miss_rmsd = required_rmsd - set(rmsd.columns)
    if miss_spans:
        raise RuntimeError(f"SPANS_CSV missing required columns: {sorted(miss_spans)}")
    if miss_rmsd:
        raise RuntimeError(f"RMSD_XLSX missing required columns: {sorted(miss_rmsd)}")

    spans = spans.copy()
    rmsd = rmsd.copy()

    spans["Antibody_ID_norm"] = spans["Antibody_ID"].map(normalize_id_for_merge)
    rmsd["Antibody_ID_norm"] = rmsd["Antibody_ID"].map(normalize_id_for_merge)

    rmsd = rmsd[["Antibody_ID_norm", "H3_RMSD"]].copy()
    merged = spans.merge(rmsd, on="Antibody_ID_norm", how="inner")

    out = {}
    for _, row in merged.iterrows():
        hid = normalize_spans_key(row["fasta_header_id"])
        if hid in out:
            continue  # take only the first match if the same ID appears multiple times
        out[hid] = {
            "H3_start": row["H3_start"],
            "H3_end": row["H3_end"],
            "Fv_len": row["Fv_len"],
            "H3_RMSD": row["H3_RMSD"],
        }

    print(f"SAbDab metadata rows after spans/RMSD merge: {len(out)}")
    print("Example SAbDab metadata keys:", list(out.keys())[:5])
    return out


def extract_and_filter_sabdab_h3(
    fv_records: list[SeqIO.SeqRecord],
    sabdab_meta: dict[str, dict],
    rmsd_thr: float,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    For each SAbDab Fv record, apply quality filters and extract the CDRH3
    subsequence using precomputed span positions.

    Returns:
      selected_map: normalized SAbDab id -> H3 sequence
      stats dict
    """
    selected = {}

    stats = {
        "fv_total": len(fv_records),
        "kept": 0,
        "drop_no_meta": 0,
        "drop_missing_values": 0,
        "drop_bad_span": 0,
        "drop_rmsd_fail": 0,
        "warn_fv_len_mismatch": 0,
    }

    print("Example SAbDab FASTA ids:", [r.id for r in fv_records[:5]])

    for rec in fv_records:
        hid = normalize_spans_key(rec.id)
        meta = sabdab_meta.get(hid)

        if meta is None:
            stats["drop_no_meta"] += 1
            continue

        s = meta.get("H3_start", None)
        e = meta.get("H3_end", None)
        fv_len_meta = meta.get("Fv_len", None)
        h3_rmsd = meta.get("H3_RMSD", None)

        # Any NaN in span coordinates or RMSD makes the entry unusable
        if pd.isna(s) or pd.isna(e) or pd.isna(fv_len_meta) or pd.isna(h3_rmsd):
            stats["drop_missing_values"] += 1
            continue

        try:
            s = int(s)
            e = int(e)
            fv_len_meta = int(fv_len_meta)
            h3_rmsd = float(h3_rmsd)
        except Exception:
            stats["drop_missing_values"] += 1
            continue

        # Primary quality gate: exclude entries where ColabFold could not
        # accurately reconstruct the CDRH3 loop geometry
        if h3_rmsd > rmsd_thr:
            stats["drop_rmsd_fail"] += 1
            continue

        seq = str(rec.seq).strip().upper()

        # Tolerate minor mismatches between the stored Fv length and the
        # actual FASTA sequence length (e.g. trailing residue differences)
        if len(seq) != fv_len_meta:
            stats["warn_fv_len_mismatch"] += 1
            fv_len_meta = len(seq)

        # Clamp span to valid range before slicing
        s = max(0, min(s, fv_len_meta))
        e = max(0, min(e, fv_len_meta))
        if e <= s:
            stats["drop_bad_span"] += 1
            continue

        h3 = seq[s:e]
        if not h3:
            stats["drop_bad_span"] += 1
            continue

        selected[hid] = h3
        stats["kept"] += 1

    return selected, stats


# ----------------------------
# OAS loading/filtering
# ----------------------------
def read_oas_cdrh3_from_dirs(*paths: str) -> dict[str, str]:
    """
    Scan one or more FASTA files/directories for CDRH3 entries.
    OAS CDR FASTA files contain all six loops per antibody in a single file,
    tagged with '|CDRH3', '|CDRL1', etc. in the header. We extract only CDRH3.
    The base ID (before the first '|') is used as the lookup key so it can be
    matched against the paired Fv FASTA records.
    """
    out = {}
    n_files = 0
    n_records = 0
    n_h3 = 0

    for path_like in paths:
        for fp in list_fasta_files(path_like):
            n_files += 1
            for rec in SeqIO.parse(str(fp), "fasta"):
                n_records += 1
                rid = rec.id
                desc = rec.description

                # Support both formats: tag in ID field or in description field
                if "|CDRH3" in rid:
                    base = rid.split("|", 1)[0]
                elif "|CDRH3" in desc:
                    base = desc.split("|", 1)[0].lstrip(">")
                else:
                    continue

                if base not in out:
                    seq = str(rec.seq).strip().upper()
                    if seq:
                        out[base] = seq
                        n_h3 += 1

    print(f"OAS CDR files scanned: {n_files}")
    print(f"OAS CDR records scanned: {n_records}")
    print(f"OAS CDRH3 loaded: {n_h3}")
    return out


def filter_oas_fv_records(
    fv_records: list[SeqIO.SeqRecord],
    oas_h3_by_base: dict[str, str],
    oas_lookup: dict[str, pd.Series],
    plddt_thr: float,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Filter OAS Fv records by requiring:
      1. A matching CDRH3 sequence exists in the CDR FASTA files.
      2. A pLDDT row exists in the merged Excel tables.
      3. The CDRH3 mean pLDDT exceeds the threshold.

    The pLDDT is used as a proxy for structural confidence: sequences whose
    AlphaFold predictions have high pLDDT are more likely to have a well-defined
    loop geometry that a model can learn from.

    Returns:
      selected_map: original OAS rec.id -> H3 sequence
      stats dict
    """
    selected = {}

    stats = {
        "fv_total": len(fv_records),
        "kept": 0,
        "drop_no_h3": 0,
        "drop_no_plddt_row": 0,
        "drop_missing_plddt": 0,
        "drop_plddt_fail": 0,
    }

    for rec in fv_records:
        rid = rec.id
        base = rid.split("|", 1)[0]

        h3 = oas_h3_by_base.get(base)
        if h3 is None or len(h3) == 0:
            stats["drop_no_h3"] += 1
            continue

        key = canonicalize_oas_key(base)
        row = oas_lookup.get(key, None)

        # If the direct key lookup fails, try the reversed-contig ordering
        if row is None:
            a, b = parse_two_contigs(key)
            if a is not None:
                row = oas_lookup.get(f"{b}_{a}", None)

        if row is None:
            stats["drop_no_plddt_row"] += 1
            continue

        mp = row.get("CDRH3_mean_plddt", None)
        if pd.isna(mp):
            stats["drop_missing_plddt"] += 1
            continue

        try:
            mp = float(mp)
        except Exception:
            stats["drop_missing_plddt"] += 1
            continue

        # Primary quality gate for OAS entries
        if mp < plddt_thr:
            stats["drop_plddt_fail"] += 1
            continue

        selected[rid] = h3
        stats["kept"] += 1

    return selected, stats


# ----------------------------
# Main
# ----------------------------
def main():
    workdir = Path(WORKDIR)
    workdir.mkdir(parents=True, exist_ok=True)

    # ---- Load raw FV sequences by source
    fv_records_sabdab = load_fv_fasta(FV_FASTA_1)
    fv_records_oas_1 = load_fv_fasta(FV_FASTA_2)
    fv_records_oas_2 = load_fv_fasta(FV_FASTA_3)
    fv_records_oas_3 = load_fv_fasta(FV_FASTA_4)

    print("\n=== Raw FV counts ===")
    print(f"SAbDab raw FV:          {len(fv_records_sabdab)}")
    print(f"OAS 1279052+1279060:    {len(fv_records_oas_1)}")
    print(f"OAS 1287155 part1:      {len(fv_records_oas_2)}")
    print(f"OAS 1287155 part2:      {len(fv_records_oas_3)}")
    print(
        f"Total raw FV:           "
        f"{len(fv_records_sabdab) + len(fv_records_oas_1) + len(fv_records_oas_2) + len(fv_records_oas_3)}"
    )

    # ---- Load metadata for filtering
    sabdab_meta = load_sabdab_metadata(SPANS_CSV, RMSD_XLSX)
    oas_df = load_and_merge_oas_plddt_tables(OAS_PLDDT_DIRS)
    oas_lookup = build_oas_lookup(oas_df)
    oas_h3_by_base = read_oas_cdrh3_from_dirs(OAS_CDR_DIR_1, OAS_CDR_DIR_2, OAS_CDR_DIR_3)

    # ---- FILTER FIRST
    # Quality filtering precedes clustering so that MMseqs2 sees only
    # sequences we actually want to keep; this avoids low-quality sequences
    # acting as cluster representatives.
    sabdab_selected_h3, sab_stats = extract_and_filter_sabdab_h3(
        fv_records=fv_records_sabdab,
        sabdab_meta=sabdab_meta,
        rmsd_thr=SABDAB_H3_RMSD_MAX,
    )

    oas_selected_h3_1, oas_stats_1 = filter_oas_fv_records(
        fv_records=fv_records_oas_1,
        oas_h3_by_base=oas_h3_by_base,
        oas_lookup=oas_lookup,
        plddt_thr=OAS_PLDDT_THRESHOLD,
    )
    oas_selected_h3_2, oas_stats_2 = filter_oas_fv_records(
        fv_records=fv_records_oas_2,
        oas_h3_by_base=oas_h3_by_base,
        oas_lookup=oas_lookup,
        plddt_thr=OAS_PLDDT_THRESHOLD,
    )
    oas_selected_h3_3, oas_stats_3 = filter_oas_fv_records(
        fv_records=fv_records_oas_3,
        oas_h3_by_base=oas_h3_by_base,
        oas_lookup=oas_lookup,
        plddt_thr=OAS_PLDDT_THRESHOLD,
    )

    # ---- Build final selected FV pool ONLY from filtered sequences
    # We rebuild from the original records (not the H3-only map) so the
    # output FASTAs contain the full paired Fv sequence.
    fv_seqs: dict[str, str] = {}
    h3_map: dict[str, str] = {}
    used_ids: set[str] = set()

    def add_selected_records(records, selected_h3_map, tag, key_func=None):
        kept = 0
        for r in records:
            # key_func translates from the record ID to the key used in selected_h3_map
            lookup_id = key_func(r.id) if key_func is not None else r.id
            if lookup_id not in selected_h3_map:
                continue

            rid = unique_id(r.id, used_ids, tag)
            used_ids.add(rid)
            fv_seqs[rid] = str(r.seq).strip().upper()
            h3_map[rid] = selected_h3_map[lookup_id]
            kept += 1
        return kept

    kept_sab = add_selected_records(
        fv_records_sabdab,
        sabdab_selected_h3,
        "single_fv_pdb",
        key_func=normalize_spans_key,
    )

    kept_oas_1 = add_selected_records(
        fv_records_oas_1,
        oas_selected_h3_1,
        "oas_1279052_1279060",
    )

    kept_oas_2 = add_selected_records(
        fv_records_oas_2,
        oas_selected_h3_2,
        "oas_1287155_part1",
    )

    kept_oas_3 = add_selected_records(
        fv_records_oas_3,
        oas_selected_h3_3,
        "oas_1287155_part2",
    )

    kept_oas_total = kept_oas_1 + kept_oas_2 + kept_oas_3
    kept_total = kept_sab + kept_oas_total

    print("\n=== Filtering summary ===")
    print(f"SAbDab threshold: H3_RMSD <= {SABDAB_H3_RMSD_MAX}")
    print(f"OAS threshold:    CDRH3_mean_plddt >= {OAS_PLDDT_THRESHOLD}")

    print("\n[SAbDab]")
    print(f"  raw FV:                 {sab_stats['fv_total']}")
    print(f"  kept:                   {sab_stats['kept']}")
    print(f"  drop_no_meta:           {sab_stats['drop_no_meta']}")
    print(f"  drop_missing_values:    {sab_stats['drop_missing_values']}")
    print(f"  drop_bad_span:          {sab_stats['drop_bad_span']}")
    print(f"  drop_rmsd_fail:         {sab_stats['drop_rmsd_fail']}")
    print(f"  warn_fv_len_mismatch:   {sab_stats['warn_fv_len_mismatch']}")

    print("\n[OAS 1279052+1279060]")
    for k, v in oas_stats_1.items():
        print(f"  {k:22s} {v}")

    print("\n[OAS 1287155 part1]")
    for k, v in oas_stats_2.items():
        print(f"  {k:22s} {v}")

    print("\n[OAS 1287155 part2]")
    for k, v in oas_stats_3.items():
        print(f"  {k:22s} {v}")

    print("\n=== Selected pool before split ===")
    print(f"SAbDab selected:         {kept_sab}")
    print(f"OAS selected total:      {kept_oas_total}")
    print(f"  - OAS dir1 selected:   {kept_oas_1}")
    print(f"  - OAS dir2 selected:   {kept_oas_2}")
    print(f"  - OAS dir3 selected:   {kept_oas_3}")
    print(f"Total selected FV:       {kept_total}")

    if kept_total == 0:
        raise RuntimeError("No sequences passed filtering. Nothing to cluster/split.")

    # ---- Write selected CDRH3 FASTA for MMseqs2 clustering
    # We cluster on CDRH3 only (not the full Fv) because CDRH3 is the most
    # variable region and the primary driver of antigen specificity. Clustering
    # on the full Fv would be slower and would over-cluster based on conserved
    # framework regions.
    h3_fasta = workdir / "cdrh3_selected.fasta"
    with open(h3_fasta, "w") as f:
        for rid, h3 in h3_map.items():
            f.write(f">{rid}\n{h3}\n")
    print(f"\nWrote selected CDRH3 FASTA for clustering: {h3_fasta}")

    # ---- Run MMseqs2 clustering on SELECTED CDRH3 only
    cluster_prefix = workdir / "clusters_h3_selected"
    tmp_dir = workdir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    try:
        run(["mmseqs", "version"])
    except Exception as e:
        raise RuntimeError(
            "mmseqs not found or not runnable. Try `which mmseqs` and `mmseqs version`."
        ) from e

    run([
        "mmseqs", "easy-cluster",
        str(h3_fasta),
        str(cluster_prefix),
        str(tmp_dir),
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c", str(COVERAGE),
        "--cov-mode", str(COV_MODE),
    ])

    # Parse the TSV output: each line is 'representative<TAB>member'
    tsv_path = Path(str(cluster_prefix) + "_cluster.tsv")
    if not tsv_path.exists():
        raise RuntimeError(f"Expected cluster TSV not found: {tsv_path}")

    clusters: dict[str, list[str]] = {}
    with open(tsv_path, "r") as f:
        for line in f:
            rep, mem = line.rstrip("\n").split("\t")
            clusters.setdefault(rep, []).append(mem)

    cluster_list = list(clusters.values())
    cluster_sizes = [len(c) for c in cluster_list]
    n_clusters = len(cluster_list)

    print("\n=== Clustering summary ===")
    print(f"Selected sequences clustered:   {kept_total}")
    print(f"Clusters formed:               {n_clusters}")
    if cluster_sizes:
        print(f"Largest cluster size:          {max(cluster_sizes)}")
        print(f"Smallest cluster size:         {min(cluster_sizes)}")
        print(f"Mean cluster size:             {sum(cluster_sizes)/len(cluster_sizes):.3f}")
    print(f"MMseqs params: min_seq_id={MIN_SEQ_ID}, coverage={COVERAGE}, cov_mode={COV_MODE}")

    # ---- Split by cluster
    # Shuffling clusters before greedy assignment ensures the train/val
    # boundary falls roughly at the target ratio regardless of cluster order,
    # and prevents systematic bias (e.g. all SAbDab clusters at the end).
    random.shuffle(cluster_list)
    total = sum(len(c) for c in cluster_list)
    target_train = int(TRAIN_RATIO * total)

    train_ids: list[str] = []
    val_ids: list[str] = []
    count = 0

    # Greedy cluster-level assignment: assign entire clusters to train until
    # the target count is reached, then the remainder goes to val.
    for c in cluster_list:
        if count < target_train:
            train_ids.extend(c)
            count += len(c)
        else:
            val_ids.extend(c)

    # Sanity check: cluster-level splitting guarantees no overlap
    assert set(train_ids).isdisjoint(val_ids)

    # ---- Source breakdown after split
    def count_source(ids):
        # SAbDab entries originate from PDB files and retain the .pdb suffix
        sab = sum(1 for x in ids if x.split("|", 1)[0].endswith(".pdb"))
        oas = len(ids) - sab
        return sab, oas

    train_sab, train_oas = count_source(train_ids)
    val_sab, val_oas = count_source(val_ids)

    print("\n=== Final split ===")
    print(f"Train: {len(train_ids)} sequences ({len(train_ids)/total:.3f})")
    print(f"  SAbDab: {train_sab}")
    print(f"  OAS:    {train_oas}")
    print(f"Val:   {len(val_ids)} sequences ({len(val_ids)/total:.3f})")
    print(f"  SAbDab: {val_sab}")
    print(f"  OAS:    {val_oas}")

    # ---- Write FULL FV FASTAs for selected+split IDs
    write_fasta_wrapped(OUT_TRAIN, fv_seqs, train_ids)
    write_fasta_wrapped(OUT_VAL, fv_seqs, val_ids)

    print("\nDone.")
    print("Train FASTA:", OUT_TRAIN)
    print("Val FASTA:  ", OUT_VAL)
    print("(These files are overwritten each run.)")


if __name__ == "__main__":
    main()
