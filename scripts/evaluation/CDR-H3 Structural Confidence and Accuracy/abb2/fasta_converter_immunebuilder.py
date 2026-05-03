"""
fasta_converter_immunebuilder.py
================================
Converts a multi-entry paired FASTA file (one heavy+light pair per entry,
encoded as "H_sequence:L_sequence" on a single sequence line) into a directory
of individual per-antibody FASTA files formatted for ImmuneBuilder/ABodyBuilder2.

Overview
--------
p-IgGen outputs paired Fv sequences as a FASTA file where each entry's
sequence line encodes both chains separated by a colon:

    >antibody_name
    EVQLVESGG...FCAS:DIQMTQSPS...HFGQGTKL

ImmuneBuilder (ABodyBuilder2) expects each antibody in its own file with
separate headers for the heavy and light chains:

    >H
    EVQLVESGG...FCAS
    >L
    DIQMTQSPS...HFGQGTKL

This script reads the combined input format and writes one .fasta file per
antibody into the output directory, ready for batch inference by
abb2_inference.py.

Inputs
------
  INPUT_FILE : path to the multi-entry combined FASTA file.

Outputs
-------
  OUTPUT_DIR/<name>.fasta for each successfully parsed antibody pair.

"""

import os

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "/home/alanwu/Documents/iggen_model/model_output/fasta/1A/pairs_split.fasta"   # path to your input fasta file
OUTPUT_DIR  = "/home/alanwu/Documents/iggen_model/model_output/fasta/1A/abb2_pairs_split.fasta"      # folder where converted files will be saved
# ──────────────────────────────────────────────────────────────────────────────

def convert_fasta(input_file, output_dir):
    """
    Parse a combined-format FASTA and write one ImmuneBuilder-format .fasta
    per entry into output_dir.

    The combined format expects:
        >entry_name
        HEAVY_SEQ:LIGHT_SEQ

    Each output file is named <entry_name>.fasta and contains:
        >H
        HEAVY_SEQ
        >L
        LIGHT_SEQ

    Parameters
    ----------
    input_file : str
        Path to the input multi-entry combined FASTA file.
    output_dir : str
        Directory to write individual per-antibody FASTA files into.
        Created if it does not exist.
    """
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
                # Header line: capture the entry name (everything after '>').
                # We don't immediately write anything — we wait for the
                # sequence line that follows.
                name = line[1:]

            elif name and ":" in line:
                # Sequence line for a named entry: validate the colon-separator
                # and split into heavy and light chains.
                parts = line.split(":")
                if len(parts) != 2:
                    # More than one ':' means the format is unexpected — skip
                    # rather than guessing which colon is the chain boundary.
                    print(f"  [WARNING] Skipping {name} — expected exactly one ':' separator")
                    errors += 1
                    continue

                h_seq, l_seq = parts
                out_path = os.path.join(output_dir, f"{name}.fasta")

                with open(out_path, "w") as out:
                    out.write(f">H\n{h_seq}\n>L\n{l_seq}\n")

                print(f"  [OK] Created {out_path}")
                converted += 1
                # Reset name so that a bare sequence line after this entry
                # does not get mistakenly associated with the next header.
                name = None

            elif name:
                # A sequence line was found for this entry but contains no ':'
                # so we cannot split it into two chains. Log and skip.
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
