#!/usr/bin/env python3
"""
OAS paired table extractor + FV trimming (includes FR4) + APPEND mode.

Overview
--------
This script processes raw OAS (Observed Antibody Space) paired antibody
sequence tables (compressed CSV) and extracts two FASTA outputs per batch:
  1. paired_fv_trimmed.fasta  — full variable domain (VH:VL), including FR4.
  2. cdr_loops.fasta          — all six CDR loops (CDRH1/2/3, CDRL1/2/3).

These FASTAs are the primary inputs to the downstream pLDDT computation
(oas_plddt.py) and clustering (oas_cluster.py) steps.

Inputs
------
  INFILES: list of Path objects pointing to OAS paired CSV.gz files.
           Each filename must match the pattern {study_id}_Paired_All.csv.gz.

Outputs
-------
  For each input file, creates one subdirectory named after the study ID stem
  (e.g. 1279065_1) under both VHVL_ROOT and CDR_ROOT:
    {VHVL_ROOT}/{study_id}/paired_fv_trimmed.fasta
    {CDR_ROOT}/{study_id}/cdr_loops.fasta

"""

from pathlib import Path
import gzip
import pandas as pd


# --------------------
# INPUT FILES
# --------------------
INFILES = [
    Path("/home/alanwu/Documents/iggen_model/data/oas data/csv table/native_healthy/1279065_1_Paired_All.csv.gz"),
    Path("/home/alanwu/Documents/iggen_model/data/oas data/csv table/native_healthy/1279073_1_Paired_All.csv.gz"),
    Path("/home/alanwu/Documents/iggen_model/data/oas data/csv table/native_healthy/1287155_1_Paired_All.csv.gz"),
]

# --------------------
# OUTPUT ROOTS
# --------------------
VHVL_ROOT = Path("/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/native_healthy")
CDR_ROOT  = Path("/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy")

# --------------------
# SETTINGS
# --------------------
CHUNKSIZE = 200_000
SKIPROWS = 1  # skip metadata line (force header on 2nd line)

DEDUP_EXISTING = True  # set False if you don't care about duplicates
DEDUP_ID_ONLY = True   # True: dedup by record id; False: dedup by full header line


def infer_batch_name(infile: Path) -> str:
    """
    Extract the study batch ID from the filename.
    Example:
      1279065_1_Paired_All.csv.gz -> 1279065_1
    """
    name = infile.name
    suffix = "_Paired_All.csv.gz"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return infile.stem.replace(".csv", "")


def detect_sep_from_second_line(path: Path) -> str:
    """
    OAS files may be tab- or comma-delimited. Auto-detect by counting
    delimiter occurrences on the header line (second line, since the first
    line is metadata). Whichever delimiter appears more often wins.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        _ = f.readline()      # metadata
        header = f.readline() # header
        if not header:
            raise RuntimeError("File ended before the header line (second line).")
        header = header.strip()
    return "\t" if header.count("\t") >= header.count(",") else ","


def _norm(s: str) -> str:
    """
    Normalize a column name by removing BOM characters, spaces, tabs, and
    newlines. Used to make column name matching robust to encoding artifacts
    commonly found in OAS files.
    """
    return (
        str(s)
        .strip()
        .replace("\ufeff", "")
        .replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("\r", "")
    )


def pick_col(cols, candidates):
    """
    Find the actual column name in 'cols' that matches any of the 'candidates'.

    Two match modes:
      - Exact (no '*'): normalized column name must equal the normalized candidate.
      - Suffix ('*suffix'): normalized column name must end with the normalized suffix.
        This handles OAS files where column names have dataset-specific prefixes.

    Exact matches are preferred over suffix matches.
    """
    norm_map = {_norm(c): c for c in cols}

    for cand in candidates:
        if cand.startswith("*"):
            continue
        nc = _norm(cand)
        if nc in norm_map:
            return norm_map[nc]

    for cand in candidates:
        if not cand.startswith("*"):
            continue
        suffix = _norm(cand[1:])
        for ncol, orig in norm_map.items():
            if ncol.endswith(suffix):
                return orig

    return None


def clean_seq_letters_only(x):
    """Strip non-alphabetic characters; uppercase. Returns None for empty/NaN."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = "".join(ch for ch in s if "A" <= ch <= "Z")
    return s or None


def clean_seq_aa(x):
    """
    Like clean_seq_letters_only but also removes stop-codon asterisks ('*').
    OAS amino acid columns can contain '*' for stop codons at the end of
    truncated sequences; these must be stripped before writing FASTA.
    """
    s = clean_seq_letters_only(x)
    if not s:
        return None
    return s.replace("*", "") or None


def _to_int(x):
    """Safe int conversion; returns None for NaN or non-numeric values."""
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def trim_1based(seq: str, start_1based: int, end_1based: int):
    """
    Trim a sequence using 1-based inclusive coordinates (as stored in OAS).
    Returns None if coordinates are invalid or out of range.
    """
    if seq is None or start_1based is None or end_1based is None:
        return None
    if start_1based < 1 or end_1based < 1 or end_1based < start_1based:
        return None
    if end_1based > len(seq):
        return None
    return seq[start_1based - 1 : end_1based] or None


def trim_0based(seq: str, start_0based: int, end_0based_inclusive: int):
    """
    Trim using 0-based coordinates as a fallback if 1-based trimming fails.
    Some OAS files store coordinates in 0-based format; trying both avoids
    losing sequences due to off-by-one uncertainty.
    """
    if seq is None or start_0based is None or end_0based_inclusive is None:
        return None
    if start_0based < 0 or end_0based_inclusive < 0 or end_0based_inclusive < start_0based:
        return None
    if end_0based_inclusive >= len(seq):
        return None
    return seq[start_0based : end_0based_inclusive + 1] or None


def build_fv_from_regions(row, cols_map):
    """
    Build FV AA by concatenating region AA strings INCLUDING FR4.

    The order is: FWR1 + CDR1 + FWR2 + CDR2 + FWR3 + CDR3 + FWR4 for
    both heavy and light chains. Including FR4 is important because it
    contains residues that are part of the variable domain fold (WGxG motif)
    and are needed as a context signal by the antibody model.

    Returns (vh_fv, vl_fv) or (None, None) if any region is missing.
    """
    (
        h_fwr1, h_cdr1, h_fwr2, h_cdr2, h_fwr3, h_cdr3, h_fwr4,
        l_fwr1, l_cdr1, l_fwr2, l_cdr2, l_fwr3, l_cdr3, l_fwr4,
    ) = cols_map

    parts_h = [clean_seq_aa(row.get(c)) for c in [h_fwr1, h_cdr1, h_fwr2, h_cdr2, h_fwr3, h_cdr3, h_fwr4]]
    parts_l = [clean_seq_aa(row.get(c)) for c in [l_fwr1, l_cdr1, l_fwr2, l_cdr2, l_fwr3, l_cdr3, l_fwr4]]

    # Any missing region makes the full Fv unusable
    if any(p is None for p in parts_h) or any(p is None for p in parts_l):
        return None, None

    vh_fv = "".join(parts_h)
    vl_fv = "".join(parts_l)
    return vh_fv or None, vl_fv or None


def load_existing_ids_vhvl(fasta_path: Path) -> set[str]:
    """
    For >{rid}|VH:VL headers: store rid only if DEDUP_ID_ONLY else full header.
    Used to avoid re-writing entries that are already present in the output FASTA.
    """
    if not fasta_path.exists():
        return set()
    seen = set()
    with fasta_path.open("r") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            hdr = line[1:].strip()
            if not hdr:
                continue
            if DEDUP_ID_ONLY:
                rid = hdr.split("|", 1)[0].strip()
                if rid:
                    seen.add(rid)
            else:
                seen.add(hdr)
    return seen


def load_existing_loop_headers(loop_fasta: Path) -> set[str]:
    """
    For loop FASTA headers: >{rid}|CDRH1 etc.
    If DEDUP_ID_ONLY=True, dedup by {rid} only (skip writing any loops for that rid if seen),
    else dedup by full header line.
    """
    if not loop_fasta.exists():
        return set()
    seen = set()
    with loop_fasta.open("r") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            hdr = line[1:].strip()
            if not hdr:
                continue
            if DEDUP_ID_ONLY:
                rid = hdr.split("|", 1)[0].strip()
                if rid:
                    seen.add(rid)
            else:
                seen.add(hdr)
    return seen


def process_one_file(infile: Path):
    batch_name = infer_batch_name(infile)

    out_dir_vhvl = VHVL_ROOT / batch_name
    out_dir_cdr  = CDR_ROOT / batch_name

    out_fasta = out_dir_vhvl / "paired_fv_trimmed.fasta"
    out_loops = out_dir_cdr / "cdr_loops.fasta"

    out_dir_vhvl.mkdir(parents=True, exist_ok=True)
    out_dir_cdr.mkdir(parents=True, exist_ok=True)

    if DEDUP_EXISTING:
        existing_vhvl = load_existing_ids_vhvl(out_fasta)
        existing_loops = load_existing_loop_headers(out_loops)
    else:
        existing_vhvl = set()
        existing_loops = set()

    sep = detect_sep_from_second_line(infile)
    sep_name = "TAB" if sep == "\t" else "COMMA"

    print(f"\n=== Processing: {infile} ===")
    print(f"[setup] batch={batch_name}, skiprows={SKIPROWS}, sep={sep_name}")
    if DEDUP_EXISTING:
        print(f"[dedup] existing VHVL records: {len(existing_vhvl)}")
        print(f"[dedup] existing loop records: {len(existing_loops)} (dedup key={'rid' if DEDUP_ID_ONLY else 'full header'})")

    # Read only the header row first to detect available columns before
    # committing to a full chunked read with usecols
    header_df = pd.read_csv(
        infile,
        compression="gzip",
        sep=sep,
        skiprows=SKIPROWS,
        nrows=0,
        low_memory=False,
    )
    cols = list(header_df.columns)

    # IDs
    col_id_h = pick_col(cols, ["sequence_id_heavy", "*sequence_id_heavy"])
    col_id_l = pick_col(cols, ["sequence_id_light", "*sequence_id_light"])

    if col_id_h is None:
        print("\n[debug] Parsed columns (first 200):")
        for c in cols[:200]:
            print(" ", repr(c))
        raise SystemExit(f"Could not find 'sequence_id_heavy' column in {infile}")

    # Region AA columns (preferred; includes FR4)
    h_fwr1_aa = pick_col(cols, ["fwr1_aa_heavy", "*fwr1_aa_heavy"])
    h_cdr1_aa = pick_col(cols, ["cdr1_aa_heavy", "*cdr1_aa_heavy"])
    h_fwr2_aa = pick_col(cols, ["fwr2_aa_heavy", "*fwr2_aa_heavy"])
    h_cdr2_aa = pick_col(cols, ["cdr2_aa_heavy", "*cdr2_aa_heavy"])
    h_fwr3_aa = pick_col(cols, ["fwr3_aa_heavy", "*fwr3_aa_heavy"])
    h_cdr3_aa = pick_col(cols, ["cdr3_aa_heavy", "*cdr3_aa_heavy"])
    h_fwr4_aa = pick_col(cols, ["fwr4_aa_heavy", "*fwr4_aa_heavy"])

    l_fwr1_aa = pick_col(cols, ["fwr1_aa_light", "*fwr1_aa_light"])
    l_cdr1_aa = pick_col(cols, ["cdr1_aa_light", "*cdr1_aa_light"])
    l_fwr2_aa = pick_col(cols, ["fwr2_aa_light", "*fwr2_aa_light"])
    l_cdr2_aa = pick_col(cols, ["cdr2_aa_light", "*cdr2_aa_light"])
    l_fwr3_aa = pick_col(cols, ["fwr3_aa_light", "*fwr3_aa_light"])
    l_cdr3_aa = pick_col(cols, ["cdr3_aa_light", "*cdr3_aa_light"])
    l_fwr4_aa = pick_col(cols, ["fwr4_aa_light", "*fwr4_aa_light"])

    have_region_aa = all(x is not None for x in [
        h_fwr1_aa, h_cdr1_aa, h_fwr2_aa, h_cdr2_aa, h_fwr3_aa, h_cdr3_aa, h_fwr4_aa,
        l_fwr1_aa, l_cdr1_aa, l_fwr2_aa, l_cdr2_aa, l_fwr3_aa, l_cdr3_aa, l_fwr4_aa,
    ])

    # Fallback columns (used when pre-split region columns are absent)
    col_seq_h = pick_col(cols, ["sequence_heavy", "*sequence_heavy"])
    col_seq_l = pick_col(cols, ["sequence_light", "*sequence_light", "sequence"])

    col_h_fwr1_s = pick_col(cols, ["fwr1_start_heavy", "*fwr1_start_heavy"])
    col_h_fwr4_e = pick_col(cols, ["fwr4_end_heavy", "*fwr4_end_heavy"])
    col_l_fwr1_s = pick_col(cols, ["fwr1_start_light", "*fwr1_start_light"])
    col_l_fwr4_e = pick_col(cols, ["fwr4_end_light", "*fwr4_end_light"])

    col_h_cdr1_s = pick_col(cols, ["cdr1_start_heavy", "*cdr1_start_heavy"])
    col_h_cdr1_e = pick_col(cols, ["cdr1_end_heavy", "*cdr1_end_heavy"])
    col_h_cdr2_s = pick_col(cols, ["cdr2_start_heavy", "*cdr2_start_heavy"])
    col_h_cdr2_e = pick_col(cols, ["cdr2_end_heavy", "*cdr2_end_heavy"])
    col_h_cdr3_s = pick_col(cols, ["cdr3_start_heavy", "*cdr3_start_heavy"])
    col_h_cdr3_e = pick_col(cols, ["cdr3_end_heavy", "*cdr3_end_heavy"])

    col_l_cdr1_s = pick_col(cols, ["cdr1_start_light", "*cdr1_start_light"])
    col_l_cdr1_e = pick_col(cols, ["cdr1_end_light", "*cdr1_end_light"])
    col_l_cdr2_s = pick_col(cols, ["cdr2_start_light", "*cdr2_start_light"])
    col_l_cdr2_e = pick_col(cols, ["cdr2_end_light", "*cdr2_end_light"])
    col_l_cdr3_s = pick_col(cols, ["cdr3_start_light", "*cdr3_start_light"])
    col_l_cdr3_e = pick_col(cols, ["cdr3_end_light", "*cdr3_end_light"])

    # Prefer region AA assembly (concatenate pre-annotated FWR/CDR AA segments)
    # because it is unambiguous.  Fall back to coordinate-based trimming of the
    # full sequence when region columns are absent (older OAS releases).
    if not have_region_aa:
        needed_fallback = [col_seq_h, col_seq_l, col_h_fwr1_s, col_h_fwr4_e, col_l_fwr1_s, col_l_fwr4_e]
        if any(x is None for x in needed_fallback):
            print("\n[debug] Parsed columns (first 220):")
            for c in cols[:220]:
                print(" ", repr(c))
            raise SystemExit(
                f"Missing AA-region columns AND fallback columns in {infile}.\n"
                "Need either region AA columns (fwr1_aa..fwr4_aa + cdr1_aa..cdr3_aa)\n"
                "OR sequence + fwr1_start + fwr4_end."
            )

    # Only load the columns we actually need to minimise memory usage
    usecols = {col_id_h}
    if col_id_l is not None:
        usecols.add(col_id_l)

    if have_region_aa:
        usecols.update([
            h_fwr1_aa, h_cdr1_aa, h_fwr2_aa, h_cdr2_aa, h_fwr3_aa, h_cdr3_aa, h_fwr4_aa,
            l_fwr1_aa, l_cdr1_aa, l_fwr2_aa, l_cdr2_aa, l_fwr3_aa, l_cdr3_aa, l_fwr4_aa,
        ])
    else:
        usecols.update([
            col_seq_h, col_seq_l,
            col_h_fwr1_s, col_h_fwr4_e, col_l_fwr1_s, col_l_fwr4_e,
            col_h_cdr1_s, col_h_cdr1_e, col_h_cdr2_s, col_h_cdr2_e, col_h_cdr3_s, col_h_cdr3_e,
            col_l_cdr1_s, col_l_cdr1_e, col_l_cdr2_s, col_l_cdr2_e, col_l_cdr3_s, col_l_cdr3_e,
        ])
        for c in [h_cdr1_aa, h_cdr2_aa, h_cdr3_aa, l_cdr1_aa, l_cdr2_aa, l_cdr3_aa]:
            if c is not None:
                usecols.add(c)

    usecols = [c for c in usecols if c is not None]

    region_map = (
        h_fwr1_aa, h_cdr1_aa, h_fwr2_aa, h_cdr2_aa, h_fwr3_aa, h_cdr3_aa, h_fwr4_aa,
        l_fwr1_aa, l_cdr1_aa, l_fwr2_aa, l_cdr2_aa, l_fwr3_aa, l_cdr3_aa, l_fwr4_aa,
    )

    n_written = 0
    n_skipped = 0
    n_bad_trim = 0
    used_region_aa = 0
    used_fallback = 0
    n_dupe_skip = 0

    with out_fasta.open("a") as f_fv, out_loops.open("a") as f_loops:
        for chunk in pd.read_csv(
            infile,
            compression="gzip",
            sep=sep,
            skiprows=SKIPROWS,
            usecols=usecols,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            for _, row in chunk.iterrows():
                hid = row.get(col_id_h)
                lid = row.get(col_id_l) if col_id_l is not None else None

                if pd.isna(hid):
                    n_skipped += 1
                    continue

                base_id = str(hid).strip()
                rid = base_id
                # Combine heavy and light IDs with '__' when both are present,
                # matching the format used by canonicalize_id in oas_plddt.py
                if lid is not None and not pd.isna(lid):
                    lid_s = str(lid).strip()
                    if lid_s and lid_s != rid:
                        rid = f"{rid}__{lid_s}"

                dedup_key_vhvl = rid if DEDUP_ID_ONLY else f"{rid}|VH:VL"

                if DEDUP_EXISTING:
                    if dedup_key_vhvl in existing_vhvl:
                        n_dupe_skip += 1
                        continue
                    if DEDUP_ID_ONLY and rid in existing_loops:
                        n_dupe_skip += 1
                        continue

                vh_fv = vl_fv = None
                cdrh1 = cdrh2 = cdrh3 = None
                cdrl1 = cdrl2 = cdrl3 = None

                if have_region_aa:
                    vh_fv, vl_fv = build_fv_from_regions(row, region_map)
                    if vh_fv and vl_fv:
                        used_region_aa += 1

                    cdrh1 = clean_seq_aa(row.get(h_cdr1_aa))
                    cdrh2 = clean_seq_aa(row.get(h_cdr2_aa))
                    cdrh3 = clean_seq_aa(row.get(h_cdr3_aa))
                    cdrl1 = clean_seq_aa(row.get(l_cdr1_aa))
                    cdrl2 = clean_seq_aa(row.get(l_cdr2_aa))
                    cdrl3 = clean_seq_aa(row.get(l_cdr3_aa))
                else:
                    vh_full = clean_seq_letters_only(row.get(col_seq_h)) if col_seq_h else None
                    vl_full = clean_seq_letters_only(row.get(col_seq_l)) if col_seq_l else None
                    if not vh_full or not vl_full:
                        n_skipped += 1
                        continue

                    h_start = _to_int(row.get(col_h_fwr1_s))
                    h_end = _to_int(row.get(col_h_fwr4_e))
                    l_start = _to_int(row.get(col_l_fwr1_s))
                    l_end = _to_int(row.get(col_l_fwr4_e))

                    # Try 1-based trimming first; fall back to 0-based if that fails
                    vh_try = trim_1based(vh_full, h_start, h_end)
                    vl_try = trim_1based(vl_full, l_start, l_end)
                    if vh_try is None or vl_try is None:
                        vh_try = trim_0based(vh_full, h_start, h_end)
                        vl_try = trim_0based(vl_full, l_start, l_end)

                    vh_fv, vl_fv = vh_try, vl_try
                    if vh_fv and vl_fv:
                        used_fallback += 1

                    cdrh1 = clean_seq_aa(row.get(h_cdr1_aa)) if h_cdr1_aa else None
                    cdrh2 = clean_seq_aa(row.get(h_cdr2_aa)) if h_cdr2_aa else None
                    cdrh3 = clean_seq_aa(row.get(h_cdr3_aa)) if h_cdr3_aa else None
                    cdrl1 = clean_seq_aa(row.get(l_cdr1_aa)) if l_cdr1_aa else None
                    cdrl2 = clean_seq_aa(row.get(l_cdr2_aa)) if l_cdr2_aa else None
                    cdrl3 = clean_seq_aa(row.get(l_cdr3_aa)) if l_cdr3_aa else None

                    def slice_loop(full_seq, s_col, e_col):
                        s = _to_int(row.get(s_col)) if s_col else None
                        e = _to_int(row.get(e_col)) if e_col else None
                        if not full_seq or s is None or e is None:
                            return None
                        out = trim_1based(full_seq, s, e)
                        if out is None:
                            out = trim_0based(full_seq, s, e)
                        return out

                    # If AA columns for CDR loops are missing, extract from the
                    # full sequence using coordinate columns as a fallback
                    if cdrh1 is None:
                        cdrh1 = slice_loop(vh_full, col_h_cdr1_s, col_h_cdr1_e)
                    if cdrh2 is None:
                        cdrh2 = slice_loop(vh_full, col_h_cdr2_s, col_h_cdr2_e)
                    if cdrh3 is None:
                        cdrh3 = slice_loop(vh_full, col_h_cdr3_s, col_h_cdr3_e)
                    if cdrl1 is None:
                        cdrl1 = slice_loop(vl_full, col_l_cdr1_s, col_l_cdr1_e)
                    if cdrl2 is None:
                        cdrl2 = slice_loop(vl_full, col_l_cdr2_s, col_l_cdr2_e)
                    if cdrl3 is None:
                        cdrl3 = slice_loop(vl_full, col_l_cdr3_s, col_l_cdr3_e)

                    cdrh1 = clean_seq_aa(cdrh1)
                    cdrh2 = clean_seq_aa(cdrh2)
                    cdrh3 = clean_seq_aa(cdrh3)
                    cdrl1 = clean_seq_aa(cdrl1)
                    cdrl2 = clean_seq_aa(cdrl2)
                    cdrl3 = clean_seq_aa(cdrl3)

                if not vh_fv or not vl_fv:
                    n_bad_trim += 1
                    continue

                # VH:VL separated by ':' — this delimiter is used consistently
                # across the pipeline to split paired sequences back apart
                f_fv.write(f">{rid}|VH:VL\n{vh_fv}:{vl_fv}\n")

                loops = [
                    ("CDRH1", cdrh1),
                    ("CDRH2", cdrh2),
                    ("CDRH3", cdrh3),
                    ("CDRL1", cdrl1),
                    ("CDRL2", cdrl2),
                    ("CDRL3", cdrl3),
                ]

                if DEDUP_EXISTING and not DEDUP_ID_ONLY:
                    for loop_name, seq in loops:
                        if not seq:
                            continue
                        hdr = f"{rid}|{loop_name}"
                        if hdr in existing_loops:
                            continue
                        f_loops.write(f">{hdr}\n{seq}\n")
                        existing_loops.add(hdr)
                else:
                    for loop_name, seq in loops:
                        if not seq:
                            continue
                        f_loops.write(f">{rid}|{loop_name}\n{seq}\n")

                n_written += 1

                # Update in-memory dedup sets so later chunks in the same file
                # are also protected against writing duplicates
                if DEDUP_EXISTING:
                    existing_vhvl.add(dedup_key_vhvl)
                    if DEDUP_ID_ONLY:
                        existing_loops.add(rid)

    print("\nDone (APPEND mode, single output per input file).")
    print(f"Appended {n_written} new paired FV records.")
    if DEDUP_EXISTING:
        print(f"Skipped {n_dupe_skip} records already present in output(s).")
    print(f"Used region-AA assembly for {used_region_aa} records.")
    print(f"Used coordinate fallback for {used_fallback} records.")
    print(f"Skipped {n_skipped} rows (missing id/seq).")
    print(f"Dropped {n_bad_trim} rows (could not trim to FV).")

    print("\nOutputs:")
    print(f"  Paired FV FASTA: {out_fasta}")
    print(f"  Loop FASTA:      {out_loops}")


def main():
    VHVL_ROOT.mkdir(parents=True, exist_ok=True)
    CDR_ROOT.mkdir(parents=True, exist_ok=True)

    for infile in INFILES:
        if not infile.exists():
            print(f"[warning] Input file not found, skipping: {infile}")
            continue
        process_one_file(infile)


if __name__ == "__main__":
    main()
