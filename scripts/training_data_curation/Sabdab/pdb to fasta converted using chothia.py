from Bio.PDB import PDBParser, PPBuilder
import glob, os, re

# ---------------------------------------------------------------
# VH/VL classifier based on conserved cysteine spacing
# ---------------------------------------------------------------
def classify_vh_vs_vl(seq):
    """
    Classify VH vs VL using spacing between conserved cysteines.
    VL: Cys23–Cys88 (~65 distance)
    VH: Cys22–Cys92–104 (~70–85 distance)
    """

    cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]

    # If cysteines missing → fallback on length heuristic
    if len(cys_positions) < 2:
        return "H" if len(seq) > 110 else "L"

    dist = cys_positions[1] - cys_positions[0]

    if 60 <= dist <= 70:
        return "L"
    else:
        return "H"


def extract_sequence_from_chain(chothia_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antibody", chothia_path)

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                ppb = PPBuilder()
                peptides = ppb.build_peptides(chain)
                if not peptides:
                    return None
                return "".join(str(pp.get_sequence()) for pp in peptides)
    return None


def get_chothia_cutoff(chothia_path, chain_id, limit):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antibody", chothia_path)

    count = 0
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue

            for residue in chain:
                het, num, ins = residue.id
                if het != " ":
                    continue

                m = re.match(r"(\d+)([A-Za-z]*)", str(num))
                if not m:
                    continue

                num_int = int(m.group(1))

                if num_int < limit:
                    count += 1
                elif num_int == limit:
                    count += 1
                    return count
                else:
                    return count
    return count


# ---------------------------------------------------------------
# NEW: Resolve VH/VL fallback when HCHAIN == LCHAIN
# ---------------------------------------------------------------
def resolve_fallback_chain_ids(chothia_path, letter):
    """
    If REMARK says H=A L=A, check uppercase and lowercase (A/a),
    extract sequences, classify VH/VL by cysteine spacing.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antibody", chothia_path)

    ppb = PPBuilder()

    candidate_ids = {letter.upper(), letter.lower()}
    chain_seqs = {}

    # Extract sequences for both versions (A and a)
    for model in structure:
        for chain in model:
            if chain.id in candidate_ids:
                peptides = ppb.build_peptides(chain)
                if peptides:
                    seq = "".join(str(pp.get_sequence()) for pp in peptides)
                    chain_seqs[chain.id] = seq

    if len(chain_seqs) < 2:
        return None, None   # cannot resolve

    # classify
    assignments = {cid: classify_vh_vs_vl(seq) for cid, seq in chain_seqs.items()}
    vh = [cid for cid, t in assignments.items() if t == "H"]
    vl = [cid for cid, t in assignments.items() if t == "L"]

    if len(vh) == 1 and len(vl) == 1:
        return vh[0], vl[0]

    return None, None


# ---------------------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------------------
def extract_Fv_from_chothia(chothia_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    output_fasta = os.path.join(output_dir, "single_fv_pdb.fasta")

    chothia_files = glob.glob(os.path.join(chothia_dir, "*.pdb"))
    print(f"Found {len(chothia_files)} Chothia PDBs.\n")

    written = 0

    with open(output_fasta, "w") as out_handle:
        for chothia_path in chothia_files:

            pdb_id = os.path.basename(chothia_path).replace("_chothia.pdb", "").lower()

            # --- Step 1: read H/L from REMARK ---
            h_chain = None
            l_chain = None

            with open(chothia_path, "r") as f:
                for line in f:
                    if "PAIRED_HL" in line:
                        m = re.search(r"HCHAIN=(\S)\s+LCHAIN=(\S)", line)
                        if m:
                            h_chain, l_chain = m.groups()
                            break

            if not h_chain or not l_chain:
                print(f"⚠ No HL pairing in {pdb_id}, skipping.")
                continue

            # -------------------------------------------------------
            # NEW: Fallback when HCHAIN == LCHAIN (e.g. A/A case)
            # -------------------------------------------------------
            if h_chain == l_chain:

                fallback_letter = h_chain
                vh_id, vl_id = resolve_fallback_chain_ids(chothia_path, fallback_letter)

                if vh_id is None or vl_id is None:
                    print(f"⚠ {pdb_id}: Could not resolve VH/VL fallback. Skipping.")
                    continue

                print(f"🔧 {pdb_id}: REMARK ambiguous → Assigned VH={vh_id}, VL={vl_id} via cysteine spacing.")
                h_chain, l_chain = vh_id, vl_id

            # --- Step 2: extract sequences ---
            vh_seq = extract_sequence_from_chain(chothia_path, h_chain)
            vl_seq = extract_sequence_from_chain(chothia_path, l_chain)

            if not vh_seq or not vl_seq:
                print(f"⚠ Missing VH/VL for {pdb_id}, skipping.")
                continue

            # --- Step 3: trim ---
            vh_trim = vh_seq[:get_chothia_cutoff(chothia_path, h_chain, 113)]
            vl_trim = vl_seq[:get_chothia_cutoff(chothia_path, l_chain, 107)]

            if len(vh_trim) < 90 or len(vl_trim) < 90:
                print(f"⚠ {pdb_id} trimmed sequences too short. Skipping.")
                continue

            # --- Step 4: write output ---
            out_handle.write(f">{pdb_id}|VH:VL\n{vh_trim}:{vl_trim}\n\n")
            written += 1

            print(f"✅ {pdb_id} written (VH={len(vh_trim)}, VL={len(vl_trim)}).")

    print(f"\n🎯 Done. {written} Fv sequences written to {output_fasta}")


# ---------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":

    chothia_dir = "/home/alanwu/Documents/colabfold pipeline/sabdab data/single_fv_pdbs"
    output_dir  = "/home/alanwu/Documents/iggen_model/data"

    extract_Fv_from_chothia(chothia_dir, output_dir)


