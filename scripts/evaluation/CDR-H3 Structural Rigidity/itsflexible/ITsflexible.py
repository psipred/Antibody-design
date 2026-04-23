#!/usr/bin/env python3

import os
import csv
import glob
from collections import OrderedDict
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

PDB_DIR = "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/iggen"
H3_TSV = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/generated_h3_loop107_116_imgt.tsv"
OUT_CSV = "/home/alanwu/Documents/iggen_model/model_output/anarci_files/iggen/itsflexible_loop_input_iggen.csv"

CHAIN_HEAVY = "A"
AB_CHAINS = "AB"


def resname_to_aa(resname: str) -> str:
    # Biopython mapping keys are like "Glu", "Tyr", not "GLU"
    return protein_letters_3to1.get(resname.capitalize(), "X").upper()


def extract_chain_sequence_and_resnums(pdb_path: str, chain_id: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    seq = []
    resnums = []

    for res in chain:
        hetflag, resseq, icode = res.id
        if hetflag.strip():
            continue
        if "CA" not in res:
            continue

        aa = resname_to_aa(res.resname)
        if aa == "X":
            continue

        seq.append(aa)
        resnums.append(resseq)

    return "".join(seq), resnums


def load_h3_map(tsv_path: str):
    h3_map = OrderedDict()
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            seq_id = parts[0].strip()
            h3 = parts[1].strip().upper().rstrip(",")
            if h3:
                h3_map[seq_id] = h3
    return h3_map


def list_all_pdbs(root_dir: str):
    return sorted(glob.glob(os.path.join(root_dir, "*.pdb")))


def find_rank001_pdb(seq_id: str, pdb_paths):
    target_prefix = f"{seq_id}_"
    matches = []

    for p in pdb_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith(target_prefix) and "rank_001" in stem:
            matches.append(p)

    if not matches:
        return None

    return sorted(matches)[0]


def find_unique_substring(seq: str, motif: str):
    hits = []
    start = seq.find(motif)
    while start != -1:
        hits.append(start)
        start = seq.find(motif, start + 1)

    if len(hits) == 0:
        raise ValueError("motif_not_found")
    if len(hits) > 1:
        raise ValueError(f"motif_not_unique:{hits}")
    return hits[0]


def main():
    h3_map = load_h3_map(H3_TSV)
    pdb_paths = list_all_pdbs(PDB_DIR)

    print(f"Found {len(pdb_paths)} pdb files")

    rows = []
    failures = []

    for seq_id, h3 in h3_map.items():
        pdb_path = find_rank001_pdb(seq_id, pdb_paths)
        if pdb_path is None:
            failures.append((seq_id, "rank_001_pdb_not_found"))
            continue

        try:
            chain_seq, resnums = extract_chain_sequence_and_resnums(pdb_path, CHAIN_HEAVY)

            start0 = find_unique_substring(chain_seq, h3)
            end0 = start0 + len(h3) - 1

            rows.append({
                "index": len(rows),
                "pdb": pdb_path,
                "ab_chains": AB_CHAINS,
                "chain": CHAIN_HEAVY,
                "resi_start": resnums[start0],
                "resi_end": resnums[end0],
            })

        except Exception as e:
            failures.append((seq_id, f"{os.path.basename(pdb_path)}\t{str(e)}"))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "pdb", "ab_chains", "chain", "resi_start", "resi_end"]
        )
        writer.writeheader()
        writer.writerows(rows)

    fail_path = OUT_CSV.replace(".csv", "_failures.tsv")
    with open(fail_path, "w") as f:
        for seq_id, reason in failures:
            f.write(f"{seq_id}\t{reason}\n")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote failures: {fail_path}")
    print(f"Success: {len(rows)}")
    print(f"Failed: {len(failures)}")


if __name__ == "__main__":
    main()