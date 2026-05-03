"""
Per-CDR pLDDT extraction from ColabFold structure prediction outputs.

Overview
--------
This script reads ColabFold JSON score files for OAS antibody sequences and
computes per-CDR-loop pLDDT statistics. The resulting table is used downstream
to filter OAS sequences: only those with sufficiently confident CDRH3 structure
predictions (mean pLDDT ≥ 80) are retained for training (see
cluster+oas_mmseq2_filter_first.py).

Pipeline
--------
1. Load paired Fv FASTA and CDR loop FASTA for the OAS batch (1287155).
   Antibody IDs are canonicalized (sorted halves joined by '__') to allow
   robust joining between files that may use different orderings of the two
   paired sequences.
2. Load only the top-ranked (rank_001) ColabFold score JSON for each entry.
   Multiple model outputs exist per entry; only the best-ranked one is used.
3. For each entry with a score JSON, locate each CDR loop within the full
   concatenated Fv sequence via exact substring search.
   - If the loop appears exactly once: extract pLDDT values for those residues.
   - If ambiguous (appears > 1 time): mark as 'ambiguous_match' and skip.
   - If not found: mark as 'not_found_in_seq' and skip.
4. Compute per-loop statistics: mean pLDDT, standard deviation, and percentage
   of residues with pLDDT > 80 (the 'good' threshold).
5. Write output to an Excel file (for downstream filtering) and a text log.

Inputs
------
  run_dir           : Directory containing ColabFold .json score files.
  paired_fv_fastas  : Full paired VH:VL FASTA (used to map pLDDT to residue positions).
  cdr_loops_fastas  : CDR loop FASTA with all six loops per antibody.

Outputs
-------
  output_xlsx : Excel table with one row per antibody; columns include per-loop
                mean/sd pLDDT, residue positions, and loop sequences.
  output_txt  : Plain-text log of the run.

"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# =========================
# CONFIG
# =========================
run_dir = "/home/alanwu/Documents/iggen_model/data/oas data/colabfold_output/1287155"

paired_fv_fastas = [
    "/home/alanwu/Documents/iggen_model/data/oas data/vh_vl/native_healthy/1287155_1/paired_fv_trimmed.fasta"
]

cdr_loops_fastas = [
    "/home/alanwu/Documents/iggen_model/data/oas data/cdr_sequence/native_healthy/1287155_1/cdr_loops.fasta"
]

output_dir = "/home/alanwu/Documents/iggen_model/data/oas data/plddt_table/1287155"
os.makedirs(output_dir, exist_ok=True)

output_txt = os.path.join(output_dir, "cdr_plddt_summary.txt")
output_xlsx = os.path.join(output_dir, "cdr_plddt_table.xlsx")

GOOD_THRESHOLD = 80.0
LOOP_NAMES = ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"]

# =========================
# Logging
# =========================
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# =========================
# FASTA utils
# =========================
def load_fasta_simple(path):
    """
    Minimal FASTA parser. BioPython is intentionally avoided here for speed
    when loading large OAS files.
    """
    seqs = {}
    cur_id, cur = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
    if cur_id:
        seqs[cur_id] = "".join(cur)
    return seqs


def clean_letters_only(s):
    """Strip any non-uppercase-letter characters from a sequence string."""
    if not s:
        return None
    s = re.sub(r"[^A-Z]", "", s.upper())
    return s if s else None


def canonicalize_id(raw_id):
    """
    Produce a canonical antibody ID that is stable regardless of which order
    the two paired contig IDs appear in the FASTA header.

    OAS paired antibody IDs are formed by joining heavy and light chain IDs,
    e.g. 'seqA__seqB' or 'seqA_contig_1_seqB_contig_1'. The two halves may
    appear in different orders in the FV FASTA vs the CDR FASTA or the ColabFold
    output filename. Sorting the two halves and rejoining with '__' gives a stable
    canonical form that matches across all three sources.
    """
    if not raw_id:
        return ""
    s = raw_id.split("|")[0]
    # Remove chain-suffix annotations that may be present in some file formats
    s = re.sub(r"_VH_VL|VH_VL|_VH|_VL", "", s)

    # Handle the '__' separator style (most common in OAS)
    if "__" in s:
        a, b = s.split("__", 1)
        return "__".join(sorted([a, b]))

    # Handle '_contig_N' separator style
    m = re.match(r"(.+?_contig_\d)_(.+?_contig_\d)$", s)
    if m:
        return "__".join(sorted([m.group(1), m.group(2)]))

    return s


def parse_loop_header(h):
    """
    Split a CDR loop FASTA header into (canonical_antibody_id, loop_name).
    Example: 'AAAC__|CDRH3' → ('canonicalized_AAAC__', 'CDRH3')
    """
    if "|" not in h:
        return canonicalize_id(h), None
    rid, loop = h.split("|", 1)
    return canonicalize_id(rid), loop.strip().upper()

# =========================
# Load FASTAs
# =========================
def load_all_sequences():
    """
    Returns:
      seq_map   : {canonical_id: full_fv_sequence}
      loops_map : {canonical_id: {loop_name: loop_sequence}}
    """
    seq_map = {}
    loops_map = defaultdict(dict)

    for path in paired_fv_fastas:
        raw = load_fasta_simple(path)
        added = 0
        for rid, seq in raw.items():
            cid = canonicalize_id(rid)
            seq = clean_letters_only(seq)
            if cid and seq and cid not in seq_map:
                seq_map[cid] = seq
                added += 1
        log(f"✅ Loaded paired Fv fasta: {path}  raw={len(raw)}  kept_new={added}")

    for path in cdr_loops_fastas:
        raw = load_fasta_simple(path)
        added = 0
        for hdr, seq in raw.items():
            rid, loop = parse_loop_header(hdr)
            seq = clean_letters_only(seq)
            if rid and loop in LOOP_NAMES and seq:
                if loop not in loops_map[rid]:
                    loops_map[rid][loop] = seq
                    added += 1
        log(f"✅ Loaded loop fasta: {path}  raw={len(raw)}  kept_new_loops={added}")

    log(f"✅ sequences loaded: {len(seq_map)}")
    log(f"✅ loop entries loaded: {len(loops_map)}")

    return seq_map, loops_map

# =========================
# Load only top-ranked pLDDT JSONs
# =========================
def load_top_rank_score_jsons():
    """
    ColabFold outputs multiple JSON score files per antibody (one per model and
    recycle count). We only want the rank_001 (best) model's scores.

    File naming convention: {entry_id}_scores_rank_001_{model_suffix}.json
    The entry ID is extracted by splitting on '_scores_' and canonicalizing.

    Returns a dict: {canonical_entry_id: path_to_rank001_json}
    """
    all_jsons = glob.glob(os.path.join(run_dir, "*.json"))

    entry_to_file = {}
    skipped_template = 0
    skipped_non_rank1 = 0
    skipped_no_plddt = 0
    skipped_bad_json = 0
    duplicate_rank1 = 0

    for fp in all_jsons:
        base = os.path.basename(fp)

        # ColabFold also writes template domain name JSONs; skip them
        if base.endswith("template_domain_names.json"):
            skipped_template += 1
            continue

        if "_scores_" not in base:
            skipped_non_rank1 += 1
            continue

        if "_scores_rank_001_" not in base:
            skipped_non_rank1 += 1
            continue

        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            skipped_bad_json += 1
            continue

        if "plddt" not in data:
            skipped_no_plddt += 1
            continue

        # Recover the entry ID from the filename prefix before '_scores_'
        entry = base.split("_scores_", 1)[0]
        cid = canonicalize_id(entry)

        # If two rank_001 files map to the same canonical ID (shouldn't happen),
        # keep the first one encountered
        if cid in entry_to_file:
            duplicate_rank1 += 1
            continue

        entry_to_file[cid] = fp

    log(f"✅ top-rank score JSON entries: {len(entry_to_file)}")
    log(f"⚠️ skipped template JSON files: {skipped_template}")
    log(f"⚠️ skipped non-rank001 / non-score JSON files: {skipped_non_rank1}")
    log(f"⚠️ skipped rank001 JSONs without plddt: {skipped_no_plddt}")
    log(f"⚠️ skipped unreadable rank001 JSONs: {skipped_bad_json}")
    log(f"⚠️ duplicate rank001 JSONs skipped: {duplicate_rank1}")

    return entry_to_file

# =========================
# Substring matching
# =========================
def find_unique_substring(seq, sub):
    """
    Locate a CDR loop within the full Fv sequence by exact substring search.

    Returns:
      (start, end)   if the loop appears exactly once (0-based, end-exclusive)
      'AMBIGUOUS'    if the loop sequence appears more than once in the Fv
      None           if the loop sequence is not found

    Ambiguous matches (CDRH1/2/3 sharing subsequences) are rare but can occur
    for very short loops; they are skipped to avoid assigning pLDDT scores to
    the wrong residues.
    """
    hits = [m.start() for m in re.finditer(re.escape(sub), seq)]
    if len(hits) == 1:
        return hits[0], hits[0] + len(sub)
    if len(hits) > 1:
        return "AMBIGUOUS"
    return None

# =========================
# MAIN
# =========================
seq_map, loops_map = load_all_sequences()
entry_to_file = load_top_rank_score_jsons()

rows = []
n_skip_no_seq = 0
n_skip_no_loops = 0
n_skip_no_valid_loop = 0

for entry_id, fp in entry_to_file.items():
    # Skip entries with no corresponding Fv sequence
    if entry_id not in seq_map:
        n_skip_no_seq += 1
        continue

    if entry_id not in loops_map:
        n_skip_no_loops += 1
        continue

    seq = seq_map[entry_id]
    loops = loops_map[entry_id]

    with open(fp) as f:
        data = json.load(f)

    plddt = np.array(data["plddt"], dtype=float)

    # Truncate to the shorter of (Fv sequence length, pLDDT array length).
    # Small discrepancies can arise from linker residues added by ColabFold.
    min_len = min(len(seq), len(plddt))
    if min_len <= 0:
        n_skip_no_valid_loop += 1
        continue

    seq = seq[:min_len]
    plddt = plddt[:min_len]

    row = {
        "antibody_id": entry_id,
        "n_ranks_used": 1,
        "input_seq_len_used": int(min_len),
    }

    valid = False  # becomes True if at least one loop is successfully mapped

    for loop_name in LOOP_NAMES:
        loop_seq = loops.get(loop_name)
        if not loop_seq:
            row[f"{loop_name}_status"] = "missing_loop_seq"
            continue

        span = find_unique_substring(seq, loop_seq)
        if span == "AMBIGUOUS":
            row[f"{loop_name}_status"] = "ambiguous_match"
            continue
        if span is None:
            row[f"{loop_name}_status"] = "not_found_in_seq"
            continue

        s, e = span
        vals = plddt[s:e]
        if len(vals) == 0:
            row[f"{loop_name}_status"] = "empty_slice"
            continue

        valid = True
        row[f"{loop_name}_status"] = "ok"
        row[f"{loop_name}_aa"] = loop_seq
        row[f"{loop_name}_len"] = int(e - s)
        row[f"{loop_name}_start_idx_0based"] = int(s)
        row[f"{loop_name}_end_idx_excl_0based"] = int(e)
        row[f"{loop_name}_mean_plddt"] = float(np.mean(vals))
        row[f"{loop_name}_sd_plddt"] = float(np.std(vals))
        # Fraction of loop residues with pLDDT above the 'good' threshold
        row[f"{loop_name}_good_pct_res_gt80"] = float(100.0 * np.mean(vals > GOOD_THRESHOLD))

    if valid:
        rows.append(row)
    else:
        n_skip_no_valid_loop += 1

# =========================
# OUTPUT
# =========================
df = pd.DataFrame(rows)
df.to_excel(output_xlsx, index=False)

log("\n=== Entry-level accounting ===")
log(f"Kept rows: {len(df)}")
log(f"Skipped: no matching paired sequence = {n_skip_no_seq}")
log(f"Skipped: no matching loop record    = {n_skip_no_loops}")
log(f"Skipped: no valid mapped loop       = {n_skip_no_valid_loop}")

with open(output_txt, "w") as f:
    f.write("\n".join(log_lines) + "\n")

log(f"\n✅ Excel written to: {output_xlsx}")
log(f"✅ Text summary written to: {output_txt}")

if df.empty:
    log("⚠️ No entries matched — check ID consistency")
