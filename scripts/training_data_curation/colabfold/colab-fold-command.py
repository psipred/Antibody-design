#!/usr/bin/env bash
# Shell snippet (not Python). Run: bash colab-fold-command.py   or: chmod +x && ./colab-fold-command.py
set -euo pipefail

colabfold_batch \
  --model-type alphafold2_multimer_v2 \
  --msa-mode single_sequence \
  --templates \
  --custom-template-path "/home/alanwu/Documents/colabfold pipeline/templates/templates(trimmed)_v4_no_insertion_code" \
  --num-models 3 \
  --num-recycle 2 \
  --random-seed 0 \
  --overwrite-existing-results \
  "/home/alanwu/Documents/iggen_model/model_output/fasta/incremental/4813.fasta" \
  "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/incremental/4813"
