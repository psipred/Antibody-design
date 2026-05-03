# colabfold_ensemble.py
# =====================
# NOTE: Despite the .py extension this is a Bash shell script.
#
# PURPOSE
# -------
# Runs ColabFold (AlphaFold2 Multimer v2) 8 times over the same input FASTA,
# each time with a different random seed (0–7), to build a multi-seed structural
# ensemble.  The ensemble is then used downstream to assess CDR-H3 "structural
# determinism": sequences whose CDR-H3 loop folds identically across all seeds
# are considered conformationally well-defined, while sequences that scatter
# across different conformations are considered ambiguous or disordered.
#
# INPUT
#   pairs_split.fasta  –  paired heavy/light chain sequences in ColabFold
#                         multimer format (colon-separated chains per entry).
#
# OUTPUT
#   One output directory per seed:  colabfold_outputs/iggen/seed_<N>/
#   Each directory contains the 5 ranked PDB structures (num-models 5) for
#   every sequence in the FASTA, along with per-model pLDDT/PAE JSON files.
#
# KEY DESIGN CHOICES
# ------------------
# --model-type alphafold2_multimer_v2
#     Multimer model is required for paired heavy+light chain inputs.
#     v2 is used (rather than v3) for consistency with prior evaluations.
#
# --msa-mode single_sequence
#     Disables the multiple-sequence alignment step.  This isolates the
#     model's ability to predict structure purely from its learned sequence
#     representations, without MSA co-evolutionary signal that would
#     dominate and homogenise predictions across seeds.
#
# --use-dropout
#     Keeps MC-dropout active at inference time, making each forward pass
#     stochastic.  Combined with different random seeds this is the primary
#     source of ensemble diversity; without it, different seeds would produce
#     nearly identical outputs despite the seed change.
#
# --templates / --custom-template-path
#     Provides structural templates from a curated local database, rather
#     than querying PDB online.  Keeps runs reproducible and offline.
#
# --num-models 5
#     Generates 5 independently initialised models per seed, giving
#     8 seeds × 5 models = 40 structures per sequence for RMSD analysis.
#
# --num-recycle 2
#     Two recycling iterations balances prediction accuracy with compute
#     cost for a large-scale ensemble run; full accuracy (20 recycles) is
#     not needed here because the goal is conformational diversity sampling,
#     not single-structure quality.
#
# --random-seed ${SEED}
#     Controls the JAX/NumPy PRNG state.  Sweeping seeds 0–7 ensures that
#     observed structural variation is not an artefact of a single seed.
#
# --overwrite-existing-results
#     Allows re-running individual seeds without manually clearing output dirs.
#
# HOW TO RUN
# ----------
#   bash colabfold_ensemble.py
#   (or:  chmod +x colabfold_ensemble.py && ./colabfold_ensemble.py)
#
# REQUIREMENTS
#   colabfold_batch must be available on PATH (LocalColabFold installation).

for SEED in 0 1 2 3 4 5 6 7; do
  echo "Running seed ${SEED}"
  colabfold_batch \
    --model-type alphafold2_multimer_v2 \
    --msa-mode single_sequence \
    --templates \
    --use-dropout \
    --custom-template-path /cs/student/project_msc/2025/alanwu/colabfold_templates \
    --num-models 5 \
    --num-recycle 2 \
    --random-seed ${SEED} \
    --overwrite-existing-results \
    /cs/student/project_msc/2025/alanwu/colabfold_inputs/iggen/pairs_split.fasta \
    /cs/student/project_msc/2025/alanwu/colabfold_outputs/iggen/seed_${SEED}
done
