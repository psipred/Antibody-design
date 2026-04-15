from Bio.PDB import PDBParser, PDBIO, Select, PPBuilder
import os
import re

# === CONFIG ===
template_dir = "/home/alanwu/Documents/colabfold pipeline/templates/templates(untrimmed)_v4"
output_dir = os.path.join("/home/alanwu/Documents/colabfold pipeline/templates", "templates(trimmed)_v4")
os.makedirs(output_dir, exist_ok=True)

# CDR definitions (Chothia)
cdr_ranges = {
    "H": [(26, 32), (52, 56), (95, 102)],
    "L": [(24, 34), (50, 56), (89, 97)],
}

parser = PDBParser(QUIET=True)
ppb = PPBuilder()


# ------------------------------------------------------
# STEP 1 — Safe cysteine-based classifier
# ------------------------------------------------------
def classify_vh_vs_vl(seq):
    """Robust classifier using cysteine spacing + conservative fallback."""

    cys = [i for i, aa in enumerate(seq) if aa == "C"]

    if len(cys) < 2:
        # Heavy chains are usually longer and more often missing second cysteine in structures
        return "H"

    dist = cys[1] - cys[0]

    # Light chains have Cys23–Cys88 (~65)
    if 60 <= dist <= 72:
        return "L"

    # Heavy chains have Cys22–Cys92-104 (~70–85)
    if 70 <= dist <= 90:
        return "H"

    # If ambiguous, fallback to heavy
    return "H"


# ------------------------------------------------------
# STEP 2 — Extract sequence from chain
# ------------------------------------------------------
def get_chain_seq(structure, chain_id):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                peptides = ppb.build_peptides(chain)
                if peptides:
                    return "".join(str(pp.get_sequence()) for pp in peptides)
    return None


# ------------------------------------------------------
# STEP 3 — Read REMARK 5 PAIRED_HL if present
# ------------------------------------------------------
def parse_remark_pairing(pdb_file):
    """Return (H_chain, L_chain) or (None, None) if absent."""
    with open(pdb_file, "r") as f:
        for line in f:
            if "PAIRED_HL" in line:
                h = re.search(r"HCHAIN=([A-Za-z0-9])", line)
                l = re.search(r"LCHAIN=([A-Za-z0-9])", line)
                if h and l:
                    return h.group(1), l.group(1)
    return None, None


# ------------------------------------------------------
# STEP 4 — VH/VL identification pipeline
# ------------------------------------------------------
def identify_vh_vl_chains(structure, pdb_path):
    """
    Returns mapping: {chainID: 'H'/'L'}
    Priority:
    1. REMARK annotation
    2. Cysteine-based classifier
    """

    # Attempt 1 — REMARK pairing
    rH, rL = parse_remark_pairing(pdb_path)
    if rH and rL and rH != rL:
        print(f"✔ REMARK pairing found: VH={rH}, VL={rL}")
        return {rH: "H", rL: "L"}

    chains = list(structure.get_chains())
    seqs = {}

    for c in chains:
        seq = get_chain_seq(structure, c.id)
        if seq:
            seqs[c.id] = seq

    # Debug output
    print(f"\nDEBUG: sequences found in {os.path.basename(pdb_path)}")
    for cid, seq in seqs.items():
        print(f"  Chain {cid}: length={len(seq)}, first Cys at {[i for i,aa in enumerate(seq) if aa=='C'][:2]}")

    if len(seqs) < 2:
        return {}

    # Step 2 — classify all chains
    types = {cid: classify_vh_vs_vl(seq) for cid, seq in seqs.items()}

    print("DEBUG: preliminary VH/VL classification:", types)

    vh = [cid for cid, t in types.items() if t == "H"]
    vl = [cid for cid, t in types.items() if t == "L"]

    # Ensure exactly one heavy and one light
    if len(vh) == 1 and len(vl) == 1:
        print(f"✔ Cysteine classifier: VH={vh[0]}, VL={vl[0]}")
        return {vh[0]: "H", vl[0]: "L"}

    # Fallback — longest chain is VH, second longest is VL
    sorted_chains = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)

    H_id = sorted_chains[0][0]
    L_id = sorted_chains[1][0]

    print(f"⚠ Fallback length-based mapping: VH={H_id}, VL={L_id}")
    return {H_id: "H", L_id: "L"}


# ------------------------------------------------------
# STEP 5 — Select class removing CDRs
# ------------------------------------------------------
class FrameworkSelect(Select):
    def __init__(self, chain_map):
        self.chain_map = chain_map

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        res_id = residue.get_id()[1]

        chain_type = self.chain_map.get(chain_id)

        # Keep antigen / unknown chains
        if chain_type is None:
            return 1

        # Remove CDR residues
        for start, end in cdr_ranges[chain_type]:
            if start - 1 <= res_id <= end + 1:
                return 0

        return 1


# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------
io = PDBIO()

for pdb_file in os.listdir(template_dir):
    if not pdb_file.endswith(".pdb"):
        continue

    path = os.path.join(template_dir, pdb_file)
    structure = parser.get_structure("antibody", path)

    chain_map = identify_vh_vl_chains(structure, path)

    output_path = os.path.join(output_dir, pdb_file)
    io.set_structure(structure)
    io.save(output_path, select=FrameworkSelect(chain_map))

    print(f"Trimmed CDRs from {pdb_file} using {chain_map} → {output_path}\n")

