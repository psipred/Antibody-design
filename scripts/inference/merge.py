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
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("Merged model saved to:", MERGED_DIR)
