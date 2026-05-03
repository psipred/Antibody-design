"""
Compute Chothia CDR loop spans from SAbDab PDB structures.

Overview
--------
This script is a preprocessing step for the SAbDab data pipeline. It reads
crystallographic PDB files from the SAbDab database, identifies the heavy and
light chain for each antibody, and computes the 0-based start/end residue
positions of all six CDR loops within the concatenated VH+VL sequence.

These span coordinates are stored in a CSV file (loop_spans_from_pdb.csv) and
later used by cluster+oas_mmseq2_filter_first.py to slice the CDRH3 subsequence
from the full Fv FASTA for MMseqs2 clustering.

Background on Chothia numbering
---------------------------------
Chothia numbering is a residue-numbering scheme for antibody variable domains
that aligns structurally equivalent positions across all antibodies. The CDR
loop boundaries are defined as fixed residue number ranges in this scheme
(e.g., CDRH3 = residues 95–102 in the heavy chain). PDB files from SAbDab use
Chothia numbering, so CDR positions can be recovered directly by looking up
residues with resSeq in the canonical ranges, without performing sequence
alignment.

Coordinate space
----------------
Loop spans are reported in the concatenated VH+VL coordinate space (0-based,
end-exclusive), NOT in individual chain coordinates. This is because the model
operates on the full paired Fv sequence, where VH residues occupy positions
0..len(VH)-1 and VL residues occupy positions len(VH)..len(VH)+len(VL)-1.
For heavy chain loops, the span is taken directly from the chain's residue index.
For light chain loops, len(VH) is added as an offset.

Inputs
------
  PDB_DIR    : Directory of SAbDab single-Fv PDB files (Chothia-numbered).
  FASTA_PATH : Paired VH:VL FASTA produced by 'pdb to fasta converted using chothia.py'.
               Provides the reference VH/VL sequences used to identify which PDB
               chain is VH and which is VL.
  EXCEL_PATH : Excel table of antibody IDs to process (produced by rmsd comparison).

Outputs
-------
  OUT_CSV : CSV with one row per antibody containing:
              Antibody_ID, pdb_path, fasta_header_id,
              heavy_chain, light_chain, VH_len, VL_len, Fv_len,
              H1_start, H1_end, H2_start, H2_end, H3_start, H3_end,
              L1_start, L1_end, L2_start, L2_end, L3_start, L3_end
            All start/end values are in concatenated-Fv 0-based coordinates,
            end-exclusive.

"""

import os
import glob
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
PDB_DIR = "/home/alanwu/Documents/colabfold pipeline/sabdab data/single_fv_pdbs"
FASTA_PATH = "/home/alanwu/Documents/iggen_model/data/single_fv_pdb.fasta"
EXCEL_PATH = "/home/alanwu/Documents/colabfold pipeline/ground truth comparison result/fv_human_v5.xlsx"
OUT_CSV = "/home/alanwu/Documents/iggen_model/data/loop_spans_from_pdb.csv"

EXCEL_ID_COL = "Antibody_ID"
print("SCRIPT STARTED", flush=True)

# Chothia CDR numeric ranges (inclusive).
# These are fixed constants defined by the Chothia numbering scheme and are
# the same for every antibody — no alignment is needed to find CDR residues.
CHOTHIA_CDR_RANGES = {
    "H1": (26, 32),
    "H2": (52, 56),
    "H3": (95, 102),
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}

# 3-letter to 1-letter AA mapping (standard + uncommon residues)
AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    # common alternatives
    "SEC":"U","PYL":"O","ASX":"B","GLX":"Z","XLE":"J"
}

def normalize_id(x: str) -> str:
    """
    Create a canonical lowercase ID for cross-table joining.
    Strips '.pdb' suffixes and pipe annotations that may appear in some filenames.
    """
    x = str(x).strip()
    x = x.split("|", 1)[0]
    x = x.replace(".pdb", "")
    return x.lower()

def read_vh_vl_fasta(path: str) -> pd.DataFrame:
    """
    Parses:
    >7w55.pdb|VH:VL
    VHSEQ:VLSEQ
    and returns join_id, vh, vl
    """
    records = []
    cur_id, cur_seq = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    full = "".join(cur_seq)
                    vh, vl = full.split(":", 1)  # heavy first, light second
                    records.append({"id": cur_id, "vh": vh, "vl": vl})
                header = line[1:]
                cur_id = header.split("|", 1)[0]  # e.g. 7w55.pdb
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        full = "".join(cur_seq)
        vh, vl = full.split(":", 1)
        records.append({"id": cur_id, "vh": vh, "vl": vl})

    df = pd.DataFrame(records)
    df["join_id"] = df["id"].map(normalize_id)
    return df

def find_pdb_file(pdb_dir: str, antibody_id: str) -> str:
    """
    Find a PDB file in pdb_dir whose filename contains antibody_id (case-insensitive).
    Prefers an exact '{id}.pdb' match; among equal matches, prefers shortest filename.
    """
    aid = normalize_id(antibody_id)
    candidates = []
    for p in glob.glob(os.path.join(pdb_dir, "*.pdb")):
        fn = os.path.basename(p).lower()
        if aid in fn:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No PDB found for {antibody_id} in {pdb_dir}")
    # prefer exact match id.pdb then shortest filename
    def score(p):
        fn = os.path.basename(p).lower()
        return (0 if fn == f"{aid}.pdb" else 1, len(fn))
    candidates.sort(key=score)
    return candidates[0]

def parse_all_chains_from_pdb(pdb_path: str):
    """
    Parse ATOM records from a PDB file and extract per-chain residue lists.

    Only standard amino acid residues are retained (HETATM records and
    non-protein residues are filtered out). Insertion codes (e.g. 32A, 32B)
    are preserved as part of the residue identity — this is critical for
    Chothia-numbered PDBs where CDRH3 often has insertion-coded residues.

    Returns dict: chain_id -> {
        'residues': [(resSeq:int, iCode:str, resname3:str), ...] (unique, ordered),
        'seq': one-letter sequence
    }
    """
    chain_res = {}   # chain -> list of residues
    chain_seen = {}  # chain -> set of (resSeq, iCode)

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            chain = line[21].strip()
            if not chain:
                continue

            resname = line[17:20].strip()
            if resname not in AA3_TO_1:
                continue

            resseq_str = line[22:26].strip()
            if not resseq_str:
                continue
            resseq = int(resseq_str)

            icode = line[26].strip()  # insertion code (A/B/...) or ""
            key = (resseq, icode)

            if chain not in chain_res:
                chain_res[chain] = []
                chain_seen[chain] = set()

            if key in chain_seen[chain]:
                continue  # skip duplicate ATOM records for the same residue

            chain_seen[chain].add(key)
            chain_res[chain].append((resseq, icode, resname))

    out = {}
    for chain, residues in chain_res.items():
        seq = "".join(AA3_TO_1[rn] for (_n, _i, rn) in residues)
        out[chain] = {"residues": residues, "seq": seq}
    return out

def build_index_map(residues):
    """
    Map each (resSeq, iCode) tuple to its 0-based position in the residue list.
    This allows converting from PDB residue numbers to sequence indices needed
    for slicing the Fv string.
    """
    return {(n, i): idx for idx, (n, i, _rn) in enumerate(residues)}

def span_from_numeric_range(idx_map, start_num: int, end_num: int):
    """
    Find the 0-based index span (start_inclusive, end_exclusive) of all residues
    with resSeq in [start_num, end_num], regardless of insertion code.

    Including all insertion codes within the numeric range is necessary because
    Chothia CDRH3 loops often contain insertion-coded residues (e.g. 100A, 100B)
    that fall within the 95–102 range but would be missed if insertion codes
    were required to be empty.
    """
    indices = [idx for (resnum, _ins), idx in idx_map.items() if start_num <= resnum <= end_num]
    if not indices:
        raise ValueError(f"No residues found for range {start_num}-{end_num}")
    return min(indices), max(indices) + 1

def choose_chain_by_sequence_match(chains_dict, target_seq: str):
    """
    Pick the chain whose sequence best matches target_seq.
    We first try exact match, else substring containment, else best overlap by length of LCS-ish proxy.
    For your data, exact/containment should usually work.

    This function is used to identify which PDB chain corresponds to VH and which
    to VL, using the FASTA sequences (which were already extracted and trimmed)
    as the reference.
    """
    # exact
    for c, d in chains_dict.items():
        if d["seq"] == target_seq:
            return c

    # containment (sometimes PDB has extra residues; usually not, but safe)
    for c, d in chains_dict.items():
        if target_seq in d["seq"] or d["seq"] in target_seq:
            return c

    # fallback: best character overlap in order (very rough, but avoids total failure)
    def rough_score(a, b):
        # count matching characters at same positions for min length
        m = min(len(a), len(b))
        return sum(1 for i in range(m) if a[i] == b[i])

    best_c, best_s = None, -1
    for c, d in chains_dict.items():
        s = rough_score(d["seq"], target_seq)
        if s > best_s:
            best_s = s
            best_c = c
    return best_c

def compute_loop_spans_for_entry(pdb_path: str, vh_seq: str, vl_seq: str):
    """
    Uses FASTA (VH first, VL second) to identify heavy vs light chains in the PDB.
    Then computes loop spans in CONCATENATED VH+VL index space.

    The concatenated-Fv coordinate space is necessary because the model and the
    pLDDT arrays both treat the Fv as a single sequence (VH immediately followed
    by VL), so loop spans must be expressed in that same coordinate system.
    """
    chains = parse_all_chains_from_pdb(pdb_path)
    if len(chains) < 2:
        raise ValueError(f"Expected >=2 chains in {pdb_path}, found {list(chains.keys())}")

    heavy_chain = choose_chain_by_sequence_match(chains, vh_seq)
    # Exclude the already-chosen heavy chain when searching for the light chain
    light_chain = choose_chain_by_sequence_match({k:v for k,v in chains.items() if k != heavy_chain}, vl_seq)

    if heavy_chain is None or light_chain is None:
        raise ValueError(f"Could not identify heavy/light chains in {pdb_path}. Chains: {list(chains.keys())}")

    h = chains[heavy_chain]
    l = chains[light_chain]

    # Sanity: ensure sequences match exactly, or at least contain
    if not (h["seq"] == vh_seq or vh_seq in h["seq"] or h["seq"] in vh_seq):
        raise ValueError(f"Heavy chain sequence mismatch for {pdb_path} (chain {heavy_chain})")
    if not (l["seq"] == vl_seq or vl_seq in l["seq"] or l["seq"] in vl_seq):
        raise ValueError(f"Light chain sequence mismatch for {pdb_path} (chain {light_chain})")

    # Use the PDB chain residue ordering to build numbering->index map
    h_map = build_index_map(h["residues"])
    l_map = build_index_map(l["residues"])

    # VH length in FASTA defines the offset for converting light-chain indices
    # to concatenated-Fv indices
    vh_len = len(vh_seq)

    spans = {
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "VH_len": vh_len,
        "VL_len": len(vl_seq),
        "Fv_len": vh_len + len(vl_seq),
    }

    for loop, (a, b) in CHOTHIA_CDR_RANGES.items():
        if loop.startswith("H"):
            s, e = span_from_numeric_range(h_map, a, b)
            spans[f"{loop}_start"] = s
            spans[f"{loop}_end"] = e
        else:
            s, e = span_from_numeric_range(l_map, a, b)
            # Add VH length offset to convert light-chain indices to Fv-space
            spans[f"{loop}_start"] = vh_len + s
            spans[f"{loop}_end"] = vh_len + e

    return spans

def main():
    # Load FASTA sequences (gives VH first, VL second)
    df_fasta = read_vh_vl_fasta(FASTA_PATH)

    # Load Excel IDs (source of truth for which entries we care about)
    df_excel = pd.read_excel(EXCEL_PATH)
    df_excel["join_id"] = df_excel[EXCEL_ID_COL].map(normalize_id)

    # Inner join: only process entries that appear in both the FASTA and the Excel
    # (the Excel filters to entries for which RMSD data exists)
    df = df_excel[["join_id"]].drop_duplicates().merge(
        df_fasta[["join_id", "vh", "vl", "id"]],
        on="join_id",
        how="inner"
    )
    print(f"Excel unique IDs: {df_excel['join_id'].nunique()} | FASTA entries: {len(df_fasta)} | matched: {len(df)}")

    rows = []
    failed = []

    for _, r in df.iterrows():
        aid = r["join_id"]
        try:
            pdb_path = find_pdb_file(PDB_DIR, aid)
            spans = compute_loop_spans_for_entry(pdb_path, r["vh"], r["vl"])
            rows.append({"Antibody_ID": aid, "pdb_path": pdb_path, "fasta_header_id": r["id"], **spans})
        except Exception as e:
            failed.append((aid, str(e)))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Success:", len(out), "| Failed:", len(failed))
    if failed:
        print("First 10 failures:")
        for x in failed[:10]:
            print(" ", x)

if __name__ == "__main__":
    main()
