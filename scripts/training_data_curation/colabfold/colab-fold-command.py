colabfold_batch \
  --model-type alphafold2_multimer_v2 \
  --msa-mode single_sequence \
  --templates \
  --custom-template-path "/home/alanwu/Documents/colabfold pipeline/templates/templates(trimmed)_v4_no_insertion_code" \
  --num-models 3 \
  --num-recycle 2 \
  --random-seed 0 \
  --overwrite-existing-results \
   "/home/alanwu/Documents/iggen_model/model_output/fasta/1A/pairs_split.fasta" \
  "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/1A" \
