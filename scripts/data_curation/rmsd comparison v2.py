#!/usr/bin/env python3
"""
Full antibody RMSD + per-loop pLDDT evaluation pipeline
(upgraded Feb 2025)

Features:
 - Reverse matching (ColabFold → SAbDab)
 - Modern sequence alignment (PairwiseAligner)
 - Per-loop RMSD (H1/H2/H3, L1/L2/L3)
 - Per-loop identity %
 - Per-loop pLDDT
 - Global RMSD + global pLDDT
 - Robust file matching (handles chothia, uppercase, weird prefixes)
"""

import os, re, math, json
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer, PPBuilder
from Bio.Data.IUPACData import protein_letters_3to1
from Bio import pairwise2


# ================================
# CONFIG
# ================================
SABDAB_DIR = "/home/alanwu/Documents/iggen_model/data/post_alphafold_cutoff/post_cutoff/20260305_0449097_single_fv_pdbs"
COLABFOLD_DIR = "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/post_cutoff"
OUTPUT_EXCEL = "/home/alanwu/Documents/iggen_model/data/post_alphafold_cutoff/post_cutoff_rmsd_comparison.xlsx"

# Correct Chothia CDR definitions
cdr_ranges = {
    "H": [(26, 32), (52, 56), (95, 102)],
    "L": [(24, 34), (50, 56), (89, 97)],
}


def classify_vh_vs_vl(seq):
    """
    Classifies VH vs VL based on conserved cysteine spacing.
    VL: Cys at ~23 and ~88 (distance ~65)
    VH: Cys at ~22 and ~92-104 (distance ~70–85)
    """

    cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]

    if len(cys_positions) < 2:
        # fallback: VH is usually longer
        return "H" if len(seq) > 110 else "L"

    dist = cys_positions[1] - cys_positions[0]

    if 60 <= dist <= 70:
        return "L"
    else:
        return "H"


def resolve_hl_fallback_by_cysteines(pdb_path, fallback_letter):
    """
    For cases where HCHAIN == LCHAIN in REMARK.
    Search for both uppercase and lowercase chain IDs (A, a),
    extract sequences, and classify VH/VL by cysteine spacing.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", pdb_path)

    chain_ids = {fallback_letter.upper(), fallback_letter.lower()}
    sequences = {}

    ppb = PPBuilder()

    for model in structure:
        for chain in model:
            if chain.id in chain_ids:
                peptides = ppb.build_peptides(chain)
                if peptides:
                    seq = "".join(str(pp.get_sequence()) for pp in peptides)
                    sequences[chain.id] = seq

    if len(sequences) < 2:
        return None, None  # cannot resolve

    types = {cid: classify_vh_vs_vl(seq) for cid, seq in sequences.items()}

    vh = [cid for cid, t in types.items() if t == "H"]
    vl = [cid for cid, t in types.items() if t == "L"]

    if len(vh) == 1 and len(vl) == 1:
        return vh[0], vl[0]

    return None, None


def align_cdr(seq_model, ref_seq):
    """
    Local alignment via pairwise2. Returns (start_idx, end_idx, identity_pct)
    where start/end refer to positions in seq_model.
    """

    # gap open = -2, gap extend = -0.5, match = 1, mismatch = -1
    alns = pairwise2.align.localms(seq_model, ref_seq,
                                   1, -1, -2, -0.5)

    if len(alns) == 0:
        return None, None, None

    best = alns[0]
    aln_model, aln_ref, score, begin, end = best

    # reconstruct mapping: index in seq_model → index in aligned model
    model_pos = -1
    ref_pos   = -1

    aligned_positions = []   # list of tuples: (model_pos, ref_pos)

    for i, (c_m, c_r) in enumerate(zip(aln_model, aln_ref)):
        if c_m != "-":
            model_pos += 1
        if c_r != "-":
            ref_pos += 1

        # Only keep aligned positions where both are not gaps
        if c_m != "-" and c_r != "-":
            aligned_positions.append(model_pos)

    if not aligned_positions:
        return None, None, None

    start_idx = aligned_positions[0]
    end_idx   = aligned_positions[-1] + 1  # Python slicing

    identity_pct = (score / len(ref_seq)) * 100

    return start_idx, end_idx, identity_pct



# ================================
# UTILITIES
# ================================

def three_to_one(resname):
    resname = resname.capitalize().strip()
    return protein_letters_3to1.get(resname, "X")

def parse_first_hl_pair(pdb_path):
    """
    Retrieve H/L chain ID from SAbDab REMARK.
    If both assigned to same letter (e.g., H=A, L=A),
    attempt fallback HL resolution using cysteine spacing.
    """

    h_chain, l_chain = None, None

    with open(pdb_path, "r") as f:
        for line in f:
            if "PAIRED_HL" in line:
                m = re.search(r"HCHAIN=(\w)\s+LCHAIN=(\w)", line)
                if m:
                    h_chain, l_chain = m.group(1), m.group(2)
                    break

    # If REMARK missing, default old behavior
    if not h_chain or not l_chain:
        return "H", "L"

    # If VH and VL are assigned to the same chain ID → fallback
    if h_chain == l_chain:
        vh, vl = resolve_hl_fallback_by_cysteines(pdb_path, h_chain)

        if vh is not None and vl is not None:
            print(f"🔧 Fallback VH/VL resolved for {os.path.basename(pdb_path)} → VH={vh}, VL={vl}")
            return vh, vl

        print(f"⚠ Fallback resolution failed for {os.path.basename(pdb_path)}, keeping original H=L={h_chain}")

    return h_chain, l_chain



def get_ca_atoms(structure, chain_id=None, residue_range=None):
    model = next(structure.get_models())
    atoms = []
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        for res in chain:
            het, resseq, icode = res.id
            if het != " ": continue
            if residue_range is None or resseq in residue_range:
                if "CA" in res:
                    atoms.append(res["CA"])
    return atoms


def trim_ca_to_variable_domain(structure, chain_id, limit):
    model = next(structure.get_models())
    atoms = []
    for chain in model:
        if chain.id != chain_id: continue
        for res in chain:
            het, resseq, ic = res.id
            if het != " ": continue
            if 1 <= resseq <= limit and "CA" in res:
                atoms.append(res["CA"])
    return atoms


def calc_rmsd(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    sup = Superimposer()
    sup.set_atoms(a[:n], b[:n])
    return sup.rms


def sumsq_from_aligned_pairs(a, b):
    n = min(len(a), len(b))
    if n == 0: return 0.0, 0
    ss = 0.0
    for i in range(n):
        d = a[i].coord - b[i].coord
        ss += np.dot(d, d)
    return ss, n


def find_rank1_model(ab_id):
    for f in os.listdir(COLABFOLD_DIR):
        if f.startswith(ab_id) and "_rank_001_" in f and f.endswith(".pdb"):
            return os.path.join(COLABFOLD_DIR, f)
    return None


def find_rank1_json(ab_id):
    for f in os.listdir(COLABFOLD_DIR):
        if f.startswith(ab_id) and "_scores_rank_001_" in f and f.endswith(".json"):
            return os.path.join(COLABFOLD_DIR, f)
    return None


# ================================
# CDR EXTRACTION
# ================================

def get_cdr_ref_atoms_and_seq(ref_struct, chain_id, start, end):
    model = next(ref_struct.get_models())
    atoms, seq = [], []
    for chain in model:
        if chain.id != chain_id: continue
        for res in chain:
            het, resseq, icode = res.id
            if het != " ": continue
            if start <= resseq <= end and "CA" in res:
                atoms.append(res["CA"])
                seq.append(three_to_one(res.get_resname()))
    return atoms, "".join(seq)


# ================================
# CDR LOOP PROCESSOR
# ================================

def compute_cdr_stats_for_chain(
        ref_struct, ref_chain_id,
        model_chain, seq_model,
        cdr_list, plddt, offset, prefix):

    results = {}
    ca_list = [res["CA"] for res in model_chain if "CA" in res]
    loop_idx = 1

    for start, end in cdr_list:
        key_rmsd = f"{prefix}{loop_idx}_RMSD"
        key_id   = f"{prefix}{loop_idx}_Identity(%)"
        key_pld  = f"{prefix}{loop_idx}_pLDDT"

        ref_atoms, ref_seq = get_cdr_ref_atoms_and_seq(ref_struct, ref_chain_id, start, end)

        if not ref_atoms:
            results[key_rmsd] = None
            results[key_id] = None
            results[key_pld] = None
            loop_idx += 1
            continue

        s, e, ident = align_cdr(seq_model, ref_seq)
        if s is None:
            results[key_rmsd] = None
            results[key_id] = None
            results[key_pld] = None
            loop_idx += 1
            continue

        mod_atoms = ca_list[s:e]
        rmsd = calc_rmsd(ref_atoms, mod_atoms)

        if plddt is not None:
            abs_s = offset + s
            abs_e = min(offset + e, len(plddt))
            vals = plddt[abs_s:abs_e]
            mean_plddt = float(np.mean(vals)) if len(vals) else None
        else:
            mean_plddt = None

        results[key_rmsd] = rmsd
        results[key_id] = ident
        results[key_pld] = mean_plddt

        loop_idx += 1

    return results


# ================================
# MAIN ANTIBODY COMPARISON
# ================================

def compare_antibody(ab_id):

    base = {
        "Antibody_ID": ab_id,
        "Status": "Missing file",

        "Heavy_RMSD": None, "Light_RMSD": None, "Global_RMSD": None,
        "H1_RMSD": None, "H2_RMSD": None, "H3_RMSD": None,
        "L1_RMSD": None, "L2_RMSD": None, "L3_RMSD": None,
        "H1_Identity(%)": None, "H2_Identity(%)": None, "H3_Identity(%)": None,
        "L1_Identity(%)": None, "L2_Identity(%)": None, "L3_Identity(%)": None,
        "H1_pLDDT": None, "H2_pLDDT": None, "H3_pLDDT": None,
        "L1_pLDDT": None, "L2_pLDDT": None, "L3_pLDDT": None,
        "Heavy_pLDDT": None,
        "Global_pLDDT": None,
    }

    # Try several candidate file names
    candidates = [
        f"{ab_id}.pdb",
        f"{ab_id}_chothia.pdb",
        f"{ab_id.lower()}.pdb",
        f"{ab_id.lower()}_chothia.pdb",
        f"{ab_id.upper()}.pdb",
        f"{ab_id.upper()}_chothia.pdb",
    ]

    ref_path = None
    for fname in candidates:
        p = os.path.join(SABDAB_DIR, fname)
        if os.path.exists(p):
            ref_path = p
            break

    model_path = find_rank1_model(ab_id)

    if ref_path is None or model_path is None:
        return base

    # Load structures
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", ref_path)
    model_struct = parser.get_structure("model", model_path)

    # Chain IDs
    h_chain, l_chain = parse_first_hl_pair(ref_path)

    # Extract CA atoms
    ref_h = trim_ca_to_variable_domain(ref_struct, h_chain, 113)
    ref_l = trim_ca_to_variable_domain(ref_struct, l_chain, 107)

    m0 = next(model_struct.get_models())
    chainA = m0["A"]
    chainB = m0["B"]

    mod_h = [res["CA"] for res in chainA if "CA" in res]
    mod_l = [res["CA"] for res in chainB if "CA" in res]

    if len(ref_h) == 0 or len(mod_h) == 0:
        return base

    # Align VH
    # Build matched CA lists for alignment
    ref_all = ref_h + ref_l
    mod_all = mod_h + mod_l

    n = min(len(ref_all), len(mod_all))

    # RMSDs
    heavy_rmsd = calc_rmsd(ref_h, mod_h)
    light_rmsd = calc_rmsd(ref_l, mod_l)

    Lh = min(len(ref_h), len(mod_h))
    Ll = min(len(ref_l), len(mod_l))
    #breakpoint()

    global_rmsd=calc_rmsd(ref_all, mod_all)


    # Model sequences
    ppb = PPBuilder()
    seq_H = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chainA))
    seq_L = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chainB))

    # pLDDT
    plddt = None
    jpath = find_rank1_json(ab_id)
    if jpath and os.path.exists(jpath):
        with open(jpath) as fp:
            data = json.load(fp)
            if "plddt" in data:
                plddt = np.array(data["plddt"])

    global_plddt = float(np.mean(plddt)) if plddt is not None else None

    offset_H = 0
    offset_L = len(seq_H) if plddt is not None else 0

    heavy_plddt = None
    if plddt is not None:
        heavy_vals = plddt[offset_H:offset_H + len(seq_H)]
        heavy_plddt = float(np.mean(heavy_vals)) if len(heavy_vals) else None

    # CDR metrics
    h_results = compute_cdr_stats_for_chain(
        ref_struct, h_chain, chainA, seq_H,
        cdr_ranges["H"], plddt, offset_H, "H"
    )
    l_results = compute_cdr_stats_for_chain(
        ref_struct, l_chain, chainB, seq_L,
        cdr_ranges["L"], plddt, offset_L, "L"
    )

    out = base.copy()
    out.update({
        "Status": "OK",
        "Heavy_RMSD": heavy_rmsd,
        "Light_RMSD": light_rmsd,
        "Global_RMSD": global_rmsd,
        "Heavy_pLDDT": heavy_plddt,
        "Global_pLDDT": global_plddt,
    })

    out.update(h_results)
    out.update(l_results)

    return out


# ================================
# REVERSED MATCHING BATCH MODE
# ================================

def main():
    results = []

    # Look for rank_001 models
    pdbs = sorted([
        f for f in os.listdir(COLABFOLD_DIR)
        if f.endswith(".pdb") and "_rank_001_" in f
    ])

    print(f"Found {len(pdbs)} predicted Fv models.\n")

    for f in pdbs:
        # Extract ID from ColabFold filename
        # Example: "1x9q.pdb_VH_VL_unrelaxed_rank_001_..."
        prefix = f.split("_rank_001_")[0]               # → "1x9q.pdb_VH_VL_unrelaxed"
        ab_id = prefix.split(".pdb")[0].lower()         # → "1x9q"

        print(f"→ Processing {ab_id} ...")
        results.append(compare_antibody(ab_id))

    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✔ Results saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()