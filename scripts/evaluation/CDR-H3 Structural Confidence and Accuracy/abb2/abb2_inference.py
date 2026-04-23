import os
import subprocess

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_DIR   = "/home/alanwu/Documents/iggen_model/model_output/fasta/1A/abb2_pairs_split.fasta"
OUTPUT_DIR  = "/home/alanwu/Documents/iggen_model/immunebuilder_output/1A"
NUMBERING   = "chothia"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

fasta_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".fasta")]

if not fasta_files:
    print(f"[ERROR] No .fasta files found in {INPUT_DIR}")
    exit(1)

print(f"Found {len(fasta_files)} fasta files to process\n")

success = 0
failed  = []

for fasta_file in fasta_files:
    name      = fasta_file.replace(".fasta", "")
    input_path  = os.path.join(INPUT_DIR, fasta_file)
    output_path = os.path.join(OUTPUT_DIR, f"{name}.pdb")

    print(f"Predicting {name}...")

    cmd = [
        "ABodyBuilder2",
        "-f", input_path,
        "-o", output_path,
        "-n", NUMBERING,
        "-v"
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"  [OK] Saved {output_path}\n")
        success += 1
    else:
        print(f"  [FAILED] {name}\n")
        failed.append(name)

print(f"\nDone! {success}/{len(fasta_files)} structures predicted successfully.")
if failed:
    print(f"Failed: {', '.join(failed)}")