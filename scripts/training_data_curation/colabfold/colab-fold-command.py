#!/usr/bin/env bash
# =============================================================================
# ColabFold batch prediction command — single FASTA file.
#
# Despite the .py extension this is a Bash shell script (not Python).
#
# What this does
# --------------
# Runs `colabfold_batch` to predict the 3-D structure of a single paired
# antibody Fv sequence (FASTA file 4813.fasta) using AlphaFold2-Multimer v2.
#
# Input / Output
# --------------
# Input : .../incremental/4813.fasta       — paired Fv FASTA for antibody 4813
# Output: .../colabfold outputs folder/incremental/4813/  — PDB + JSON files
# =============================================================================
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
