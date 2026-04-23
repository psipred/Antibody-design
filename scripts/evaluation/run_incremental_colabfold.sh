#!/usr/bin/env bash
set -euo pipefail

COLABFOLD_BIN="$HOME/localcolabfold/colabfold-conda/bin/colabfold_batch"
TEMPLATE_PATH="/home/alanwu/Documents/colabfold pipeline/templates/templates(trimmed)_v4_no_insertion_code"
OUT_BASE="/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental"
IN_BASE="/home/alanwu/Documents/iggen_model/model_output/fasta/incremental"

run_one () {
  local in_fasta="$1"
  local out_dir="$2"

  echo "Starting: $in_fasta"
  mkdir -p "$out_dir"

  "$COLABFOLD_BIN" \
    --model-type alphafold2_multimer_v2 \
    --msa-mode single_sequence \
    --templates \
    --custom-template-path "$TEMPLATE_PATH" \
    --num-models 3 \
    --num-recycle 2 \
    --random-seed 0 \
    --overwrite-existing-results \
    "$in_fasta" \
    "$out_dir"

  echo "Finished: $in_fasta"
}

run_one "$IN_BASE/1000.fasta" "$OUT_BASE/1000"
run_one "$IN_BASE/2000.fasta" "$OUT_BASE/2000"
run_one "$IN_BASE/3000.fasta" "$OUT_BASE/3000"
run_one "$IN_BASE/4000.fasta" "$OUT_BASE/4000"

echo "All jobs finished."