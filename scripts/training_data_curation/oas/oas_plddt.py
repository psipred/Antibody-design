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
    if not s:
        return None
    s = re.sub(r"[^A-Z]", "", s.upper())
    return s if s else None


def canonicalize_id(raw_id):
    if not raw_id:
        return ""
    s = raw_id.split("|")[0]
    s = re.sub(r"_VH_VL|VH_VL|_VH|_VL", "", s)

    if "__" in s:
        a, b = s.split("__", 1)
        return "__".join(sorted([a, b]))

    m = re.match(r"(.+?_contig_\d)_(.+?_contig_\d)$", s)
    if m:
        return "__".join(sorted([m.group(1), m.group(2)]))

    return s


def parse_loop_header(h):
    if "|" not in h:
        return canonicalize_id(h), None
    rid, loop = h.split("|", 1)
    return canonicalize_id(rid), loop.strip().upper()

# =========================
# Load FASTAs
# =========================
def load_all_sequences():
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
    all_jsons = glob.glob(os.path.join(run_dir, "*.json"))

    entry_to_file = {}
    skipped_template = 0
    skipped_non_rank1 = 0
    skipped_no_plddt = 0
    skipped_bad_json = 0
    duplicate_rank1 = 0

    for fp in all_jsons:
        base = os.path.basename(fp)

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

        entry = base.split("_scores_", 1)[0]
        cid = canonicalize_id(entry)

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

    valid = False

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


