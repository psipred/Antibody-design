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

# Chothia CDR numeric ranges (inclusive)
CHOTHIA_CDR_RANGES = {
    "H1": (26, 32),
    "H2": (52, 56),
    "H3": (95, 102),
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}

# 3-letter to 1-letter AA mapping
AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    # common alternatives
    "SEC":"U","PYL":"O","ASX":"B","GLX":"Z","XLE":"J"
}

def normalize_id(x: str) -> str:
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
                continue

            chain_seen[chain].add(key)
            chain_res[chain].append((resseq, icode, resname))

    out = {}
    for chain, residues in chain_res.items():
        seq = "".join(AA3_TO_1[rn] for (_n, _i, rn) in residues)
        out[chain] = {"residues": residues, "seq": seq}
    return out

def build_index_map(residues):
    return {(n, i): idx for idx, (n, i, _rn) in enumerate(residues)}

def span_from_numeric_range(idx_map, start_num: int, end_num: int):
    """
    Include all residues with resSeq within [start_num, end_num], regardless of insertion code.
    Returns (start_idx, end_idx_exclusive) in that chain's 0-based indices.
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
    """
    chains = parse_all_chains_from_pdb(pdb_path)
    if len(chains) < 2:
        raise ValueError(f"Expected >=2 chains in {pdb_path}, found {list(chains.keys())}")

    heavy_chain = choose_chain_by_sequence_match(chains, vh_seq)
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

    vh_len = len(vh_seq)  # FASTA VH length defines concatenation boundary

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
            spans[f"{loop}_start"] = vh_len + s
            spans[f"{loop}_end"] = vh_len + e

    return spans

def main():
    # Load FASTA sequences (gives VH first, VL second)
    df_fasta = read_vh_vl_fasta(FASTA_PATH)

    # Load Excel IDs (source of truth for which entries we care about)
    df_excel = pd.read_excel(EXCEL_PATH)
    df_excel["join_id"] = df_excel[EXCEL_ID_COL].map(normalize_id)

    # Merge to ensure we have VH/VL for each Excel entry
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
