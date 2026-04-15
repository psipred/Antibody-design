import os

input_dir = "/home/alanwu/Documents/colabfold pipeline/templates/templates(trimmed)_v4"
output_dir = "/home/alanwu/Documents/colabfold pipeline/templates/templates(trimmed)_v4_no_insertion_code"

os.makedirs(output_dir, exist_ok=True)

def has_insertion_code(line):
    """Returns True if ATOM/HETATM line has a residue insertion code."""
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return False

    ins_code = line[26]   # column 27 (0-based)
    return ins_code.strip() != ""   # insertion code exists

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".pdb"):
        continue

    inp = os.path.join(input_dir, fname)
    outp = os.path.join(output_dir, fname)

    with open(inp, "r") as f_in, open(outp, "w") as f_out:
        for line in f_in:
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                if has_insertion_code(line):
                    # skip the residue with insertion code
                    continue

            f_out.write(line)

    print(f"Processed: {fname}")

print("Finished! Cleaned PDBs saved to:", output_dir)
