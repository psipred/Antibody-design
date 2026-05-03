"""
inference.py — p-IgGen Sequence Generation Command
====================================================

PURPOSE
-------
Generates novel antibody Fv sequences using the fine-tuned p-IgGen model.
This file records the exact shell invocation used so the run is reproducible.

WHAT IT DOES
------------
Calls `piggen_generate`, the CLI entry point bundled with the p-IgGen package,
to sample new VH+VL sequences from the merged (base + LoRA) model checkpoint.

The model was fine-tuned on OAS-filtered CDR-masked sequences; see merge.py
for how the LoRA adapter was fused into the base model before this step.

KEY FLAGS
---------
  HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
      Force Hugging Face libraries to use only the local cache — prevents
      accidental downloads and ensures the merged checkpoint in `inference/`
      is used, not the original upstream weights.

  --cache_dir
      Points to the directory containing the merged model weights (produced
      by merge.py). p-IgGen's CLI searches this directory for the tokenizer
      and model config.

  --n_sequences 200
      Number of antibody sequences to sample. Downstream evaluation uses this
      batch for structural prediction and CDR-H3 analysis.

  --temp 0.5
      Sampling temperature. Lower values (< 1.0) make the distribution sharper,
      favouring higher-probability (more "canonical") sequences. 0.5 balances
      diversity against sequence quality — high enough for diversity, low enough
      to avoid nonsensical outputs.

OUTPUT
------
  model_output/raw_sequences.txt
    Plain text; one concatenated VH+VL sequence per line (no header, no
    delimiter between chains). Feed directly into split_sequence.py.
"""

# Shell command — execute in bash, not via `python inference.py`:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 piggen_generate   --cache_dir /home/alanwu/Documents/iggen_model/inference   --output_file /home/alanwu/Documents/iggen_model/model_output/raw_sequences.txt   --n_sequences 200 --temp 0.5
