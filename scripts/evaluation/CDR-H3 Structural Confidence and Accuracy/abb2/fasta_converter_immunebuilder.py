import os

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "/home/alanwu/Documents/iggen_model/model_output/fasta/1A/pairs_split.fasta"   # path to your input fasta file
OUTPUT_DIR  = "/home/alanwu/Documents/iggen_model/model_output/fasta/1A/abb2_pairs_split.fasta"      # folder where converted files will be saved
# ──────────────────────────────────────────────────────────────────────────────

def convert_fasta(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    errors    = 0

    with open(input_file) as f:
        name = None
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                name = line[1:]

            elif name and ":" in line:
                parts = line.split(":")
                if len(parts) != 2:
                    print(f"  [WARNING] Skipping {name} — expected exactly one ':' separator")
                    errors += 1
                    continue

                h_seq, l_seq = parts
                out_path = os.path.join(output_dir, f"{name}.fasta")

                with open(out_path, "w") as out:
                    out.write(f">H\n{h_seq}\n>L\n{l_seq}\n")

                print(f"  [OK] Created {out_path}")
                converted += 1
                name = None

            elif name:
                print(f"  [WARNING] Skipping {name} — no ':' separator found in sequence line")
                errors += 1
                name = None

    print(f"\nDone! {converted} file(s) converted, {errors} skipped.")
    print(f"Output folder: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    print(f"Reading: {INPUT_FILE}")
    print(f"Output:  {OUTPUT_DIR}\n")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        print("Make sure INPUT_FILE points to the correct path.")
    else:
        convert_fasta(INPUT_FILE, OUTPUT_DIR)