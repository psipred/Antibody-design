"""
Extract paired Fv sequences from Chothia-numbered SAbDab PDB files.

Overview
--------
This script converts SAbDab antibody PDB structures into a paired FASTA file
suitable for ColabFold structure prediction and downstream training data
curation. For each PDB file, it:
  1. Reads the REMARK record to identify the heavy (H) and light (L) chain IDs.
  2. Extracts the amino acid sequence for each chain using BioPython's PPBuilder.
  3. Trims both chains to the variable domain boundary using Chothia numbering:
       VH: up to and including residue 113
       VL: up to and including residue 107
  4. Writes the trimmed VH and VL sequences in paired FASTA format (VH:VL).

Why trim at Chothia 113 / 107?
-------------------------------
SAbDab PDB structures from the Chothia-renumbered set may include the constant
domain (CH1/CL) residues beyond the variable domain. Trimming at position 113
(VH) and 107 (VL) retains only the variable domain, which is the region used
for antigen binding and the target of the IgGen model. This also makes all
input sequences structurally comparable regardless of whether the crystal
structure includes part of the constant domain.

Chain disambiguation
--------------------
SAbDab REMARK records contain a PAIRED_HL annotation, e.g.:
  REMARK 950 PAIRED_HL HCHAIN=H LCHAIN=L
In rare cases, the REMARK assigns both chains to the same letter (e.g., A/A),
which can happen when the two chains use case-sensitive chain IDs (A and a).
In those cases, `resolve_fallback_chain_ids` tries both the uppercase and
lowercase variant of the chain letter, extracts the sequence for each, and
classifies VH vs VL using the spacing between the two conserved cysteines.

VH vs VL classification by cysteine spacing
--------------------------------------------
The variable domain fold contains two conserved cysteines (forming the
intra-domain disulfide bond). Their spacing differs between VH and VL:
  - VL: ~65 residues between cysteine 1 (~pos 23) and cysteine 2 (~pos 88)
  - VH: > 70 residues between cysteine 1 (~pos 22) and cysteine 2 (~pos 92–104)
A distance of 60–70 is classified as VL; anything else as VH.
If fewer than 2 cysteines are found, length is used as a proxy (>110 → VH).

Inputs
------
  chothia_dir : Directory of *_chothia.pdb files from SAbDab.

Outputs
-------
  single_fv_pdb.fasta : Paired VH:VL FASTA; one entry per successfully processed
                        antibody. Format:
                          >{pdb_id}|VH:VL
                          {VH_trimmed}:{VL_trimmed}

"""

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
    """
    Extract the amino acid sequence for a single chain from a PDB file.
    PPBuilder builds polypeptide fragments; joining them handles chain breaks
    (missing residues) without losing the sequence context.
    """
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
    """
    Count the number of ATOM residues in 'chain_id' with Chothia resSeq ≤ limit.

    This produces the correct trim length even when the Chothia numbering has
    gaps (deleted residues) or insertions before the cutoff position. Counting
    actual residues — rather than using the residue number directly as an index
    — is essential because Chothia resSeq values are not guaranteed to be
    contiguous (e.g. a chain might jump from 52 to 54, with no residue 53).

    Returns the count of residues up to and including the limit position.
    """
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
                    continue  # skip HETATM residues (ligands, waters)

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
                    return count  # past the cutoff — stop counting
    return count


# ---------------------------------------------------------------
# NEW: Resolve VH/VL fallback when HCHAIN == LCHAIN
# ---------------------------------------------------------------
def resolve_fallback_chain_ids(chothia_path, letter):
    """
    If REMARK says H=A L=A, check uppercase and lowercase (A/a),
    extract sequences, classify VH/VL by cysteine spacing.

    This edge case occurs in some SAbDab structures where BioPython preserves
    case-sensitive chain IDs (e.g. 'A' for heavy, 'a' for light) that are
    collapsed to the same letter in the REMARK annotation.
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
    """
    Main extraction loop: processes all *.pdb files in chothia_dir
    and writes a single paired FASTA to output_dir/single_fv_pdb.fasta.
    """

    os.makedirs(output_dir, exist_ok=True)

    output_fasta = os.path.join(output_dir, "single_fv_pdb.fasta")

    chothia_files = glob.glob(os.path.join(chothia_dir, "*.pdb"))
    print(f"Found {len(chothia_files)} Chothia PDBs.\n")

    written = 0

    with open(output_fasta, "w") as out_handle:
        for chothia_path in chothia_files:

            pdb_id = os.path.basename(chothia_path).replace("_chothia.pdb", "").lower()

            # --- Step 1: read H/L from REMARK ---
            # SAbDab encodes H/L chain assignments in a REMARK 950 PAIRED_HL line;
            # this is more reliable than guessing by chain letter conventions
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

            # --- Step 3: trim to variable domain boundary ---
            # get_chothia_cutoff returns the number of residues up to and
            # including the Chothia limit position; slicing to that count
            # removes the constant domain tail while respecting Chothia gaps
            vh_trim = vh_seq[:get_chothia_cutoff(chothia_path, h_chain, 113)]
            vl_trim = vl_seq[:get_chothia_cutoff(chothia_path, l_chain, 107)]

            # Sequences shorter than 90 residues are almost certainly incomplete
            # or misidentified chains
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
