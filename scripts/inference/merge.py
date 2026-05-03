"""
merge.py — LoRA Adapter Fusion for p-IgGen Fine-Tuned Model
=============================================================

PURPOSE
-------
Fuses a LoRA (Low-Rank Adaptation) adapter — trained on OAS-filtered,
CDR-masked antibody sequences — into the base p-IgGen weights, producing a
single self-contained model checkpoint that can be served by the standard
`piggen_generate` CLI without needing PEFT at inference time.

WHY MERGE RATHER THAN SERVE WITH PEFT?
---------------------------------------
LoRA adapters add a small set of low-rank matrices on top of frozen base
weights (ΔW = A·B, where rank(A) = rank(B) = r ≪ d). Merging performs
W_merged = W_base + ΔW in-place and discards the adapter wrappers, resulting
in a model identical in architecture to the original — no PEFT dependency at
runtime, simpler deployment, and no per-forward-pass adapter overhead.

PEFT VERSION COMPATIBILITY
---------------------------
The `lora_config_from_adapter_dir` helper works around a forward-compatibility
issue: adapters saved with newer PEFT versions include config keys that older
installed versions do not recognise, causing `LoraConfig(**raw)` to raise.
By filtering to only the keys that the installed LoraConfig.__init__ accepts,
the helper allows loading adapters across minor PEFT version mismatches.

INPUT
-----
  BASE_REPO    — HuggingFace model ID for the pre-trained p-IgGen base model.
  ADAPTER_DIR  — Directory containing the LoRA adapter weights and
                 `adapter_config.json` (output of the fine-tuning script).

OUTPUT
------
  MERGED_DIR   — Directory containing the merged model in safetensors format
                 plus the tokenizer files. This directory is passed as
                 `--cache_dir` to `piggen_generate` in inference.py.

HOW TO RUN
----------
  python merge.py
  (Requires: torch, transformers, peft; GPU optional but speeds up loading.)

NOTE ON fp32
------------
Merging is done in fp32 even if the adapter was trained in bf16/fp16. This
prevents small numerical errors from accumulating during the W += A·B addition
across many layers, which could otherwise shift generation statistics subtly.
The merged checkpoint is saved in fp32; quantise at load time if needed.
"""

import inspect
import json
from pathlib import Path

import torch
import torch.distributed.tensor  # noqa: F401 — PEFT may touch torch.distributed.tensor.DTensor before lazy bind
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel


def lora_config_from_adapter_dir(adapter_dir: str) -> LoraConfig:
    """
    Adapters saved with newer PEFT add config keys older PEFT does not accept.
    Build LoraConfig only from kwargs supported by the installed peft version.
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Introspect the constructor signature at runtime so this works regardless
    # of which PEFT minor version is installed.
    params = inspect.signature(LoraConfig.__init__).parameters
    kwargs = {k: v for k, v in raw.items() if k in params}
    return LoraConfig(**kwargs)

BASE_REPO = "ollieturnbull/p-IgGen"
ADAPTER_DIR = "/home/alanwu/Documents/iggen_model/piggen_lora_cdr_masked_oas_4813/best_or_final"  # your LoRA adapter folder
MERGED_DIR  = "/home/alanwu/Documents/iggen_model/inference/models--ollieturnbull--p-IgGen1"                    # output folder

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE_REPO,
    trust_remote_code=True,
    torch_dtype=torch.float32,   # merge in fp32 for safety
).to(device)

# pad safety
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Some causal LM configs leave pad_token_id unset, which causes warnings or
# errors during generation — mirror the tokenizer setting into the model config.
if base.config.pad_token_id is None:
    base.config.pad_token_id = tokenizer.pad_token_id

# load adapter (pass filtered config so old PEFT can load adapters saved with PEFT 0.19+)
model = PeftModel.from_pretrained(
    base,
    ADAPTER_DIR,
    config=lora_config_from_adapter_dir(ADAPTER_DIR),
)

# merge LoRA weights into base weights and drop adapter wrappers
merged = model.merge_and_unload()

# save merged model + tokenizer
# safe_serialization=True writes .safetensors instead of legacy .bin pickles,
# which is safer to load and faster to memory-map at inference time.
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("Merged model saved to:", MERGED_DIR)
