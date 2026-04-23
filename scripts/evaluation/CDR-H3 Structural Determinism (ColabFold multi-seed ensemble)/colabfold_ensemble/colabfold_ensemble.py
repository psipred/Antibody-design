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