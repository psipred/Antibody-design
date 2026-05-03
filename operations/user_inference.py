#!/usr/bin/env python3

import logging
import math
import click
import torch
import transformers
from transformers import GenerationConfig
from piggen import utils
from anarci.anarci import anarci
print(anarci)

log_format = "%(asctime)s - %(levelname)s - %(message)s"
date_format = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
logger = logging.getLogger(__name__)

# Motif-based VH/VL splitting adapted from scripts/evaluation/pre_evaluation_processing/split_sequence.py
VL_START_MOTIFS = [
    "DIQMTQ", "DIVMTQ", "DVVMTQ",
    "EIVLTQ", "EIVMTQ",
    "QSVLTQ", "QSALTQ", "QITLK", "QVTLR",
    "ENVLTQ", "SYELTQ", "SSELTQ", "AIQLTQ",
    "MTQSP", "VLTQSP", "LTQSP",
]

HEAVY_END_MOTIFS = [
    "WGQGTLVTVSS", "WGQGTTVTVSS", "WGQGSLVTVSS", "WGQGILVTVSS", "WGQGTSVTVSS",
    "WGRGTLVTVSS", "WGHGTTVTVSS", "WGKGTTVTVSS", "WGHGTLVTVSS",
    "WGQGTLVTVSA", "WGQGTTVTVSA", "WGRGTLVTVSA", "WGKGTTVTVSA",
]

MIN_VH_LEN = 90
MIN_VL_LEN = 80


def _split_by_vl_start(seq: str):
    safe_offset = 80
    candidates = []
    for motif in VL_START_MOTIFS:
        idx = seq.find(motif, safe_offset)
        if idx != -1:
            candidates.append(idx)

    if not candidates:
        return None

    cut = min(candidates)
    vh, vl = seq[:cut], seq[cut:]
    if len(vh) < MIN_VH_LEN or len(vl) < MIN_VL_LEN:
        return None
    return vh, vl


def _split_by_heavy_end(seq: str):
    best_cut = None
    for motif in HEAVY_END_MOTIFS:
        idx = seq.rfind(motif)
        if idx != -1:
            cut = idx + len(motif)
            if best_cut is None or cut > best_cut:
                best_cut = cut

    if best_cut is None:
        return None

    vh, vl = seq[:best_cut], seq[best_cut:]
    if len(vh) < MIN_VH_LEN or len(vl) < MIN_VL_LEN:
        return None
    return vh, vl


def split_vh_vl_motif(seq: str):
    seq = seq.strip().replace(" ", "")
    if not seq:
        return None
    res = _split_by_vl_start(seq)
    if res is not None:
        return res
    return _split_by_heavy_end(seq)


class CustomPIgGen:
    def __init__(self, model_name, tokenizer_name="ollieturnbull/p-IgGen", device=None, cache_dir=None):
        logger.info("Initializing custom p-IgGen model...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir
        )

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"No device specified, automatically selected {self.device}.")
        else:
            self.device = device

        self.model.to(self.device)
        logger.info(f"Custom p-IgGen model initialized on {self.device}.")

    def _generate(self, num_return_sequences, prompt_sequence, eos_token_id, temp=1.25, top_p=0.95, batch_size=1):
        input_ids = self.tokenizer.encode(prompt_sequence, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.pad_token_id

        generation_config = GenerationConfig(
            do_sample=True,
            top_p=top_p,
            pad_token_id=pad_token_id,
            max_new_tokens=400,
            temperature=temp,
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_token_id,
        )

        generated_token_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )

        decoded_sequences = self.tokenizer.batch_decode(
            generated_token_ids, skip_special_tokens=True
        )
        return decoded_sequences

    def generate(self, num_return_sequences, backwards=False, top_p=0.95, temp=1.25,
                 batch_size=1, prompt=None, discard_bottom_n_percent=None,
                 separated_output=False):
        target_output = num_return_sequences
        MAX_ATTEMPTS = 10

        if backwards:
            prompt = prompt or "2"
            eos_token_id = self.tokenizer.encode("1")[0]
        else:
            prompt = prompt or "1"
            eos_token_id = self.tokenizer.encode("2")[0]

        if discard_bottom_n_percent is not None and target_output < 100:
            logger.warning(
                "Cannot discard bottom n percent with less than 100 sequences. Ignoring discard_bottom_n_percent."
            )
            discard_bottom_n_percent = None

        # How many valid sequences we need before the optional discard step.
        if discard_bottom_n_percent is not None:
            n_to_collect = math.ceil(target_output / (1 - discard_bottom_n_percent / 100))
        else:
            n_to_collect = target_output

        # --- Phase 1: collect exactly n_to_collect valid sequences ---
        decoded_sequences = []
        for attempt in range(MAX_ATTEMPTS):
            if len(decoded_sequences) >= n_to_collect:
                break
            need = n_to_collect - len(decoded_sequences)
            raw = self._generate(need, prompt, eos_token_id=eos_token_id,
                                 temp=temp, top_p=top_p, batch_size=batch_size)
            new_valid = [utils.format_and_validate_output(s) for s in raw]
            new_valid = [s for s in new_valid if s is not None]
            decoded_sequences.extend(new_valid)
            if len(decoded_sequences) < n_to_collect:
                logger.info(
                    "Attempt %d/%d: %d/%d valid sequences so far; retrying.",
                    attempt + 1, MAX_ATTEMPTS, len(decoded_sequences), n_to_collect,
                )
        decoded_sequences = decoded_sequences[:n_to_collect]

        if len(decoded_sequences) < n_to_collect:
            logger.warning(
                "Could only collect %d/%d valid sequences after %d attempts.",
                len(decoded_sequences), n_to_collect, MAX_ATTEMPTS,
            )

        # --- Phase 2: discard bottom n percent by likelihood ---
        if discard_bottom_n_percent is not None:
            likelihoods = self.get_batch_log_likelihoods(decoded_sequences, batch_size=batch_size)
            paired = sorted(zip(decoded_sequences, likelihoods), key=lambda x: x[1], reverse=True)
            decoded_sequences = [x[0] for x in paired[:target_output]]

        logger.info("Generated %d sequences with temp %s.", len(decoded_sequences), temp)

        # --- Phase 3: split into VH/VL if requested ---
        if separated_output:
            vh_list, vl_list = [], []
            pending = list(decoded_sequences)

            for attempt in range(MAX_ATTEMPTS):
                still_pending = []
                for seq in pending:
                    split = split_vh_vl_motif(seq)
                    if split is None:
                        still_pending.append(seq)
                    else:
                        vh_list.append(split[0])
                        vl_list.append(split[1])
                pending = still_pending

                if len(vh_list) >= target_output:
                    break

                if not pending:
                    # Need more sequences to fill the gap.
                    need = target_output - len(vh_list)
                    raw = self._generate(need, prompt, eos_token_id=eos_token_id,
                                         temp=temp, top_p=top_p, batch_size=batch_size)
                    new_valid = [utils.format_and_validate_output(s) for s in raw]
                    pending = [s for s in new_valid if s is not None]
                    if not pending:
                        logger.warning(
                            "Attempt %d/%d: no new valid sequences generated for splitting.",
                            attempt + 1, MAX_ATTEMPTS,
                        )
                        break

            if not vh_list:
                logger.warning(
                    "Motif-based split failed for all sequences; falling back to ANARCI."
                )
                VH, VL = utils.get_separate_VH_VL(decoded_sequences)
                if VH is None or VL is None:
                    logger.warning("ANARCI split also failed; returning empty chain outputs.")
                    return [], []
                return VH[:target_output], VL[:target_output]

            if len(vh_list) < target_output:
                logger.warning(
                    "Could only produce %d/%d split pairs after %d attempts.",
                    len(vh_list), target_output, MAX_ATTEMPTS,
                )
            return vh_list[:target_output], vl_list[:target_output]

        return decoded_sequences

    def generate_heavy_chain(self, light_chain, num_return_sequences, top_p=0.95, temp=1.25, batch_size=1):
        prompt = f"2{light_chain[::-1]}"
        return self.generate(
            num_return_sequences,
            top_p=top_p,
            temp=temp,
            batch_size=batch_size,
            backwards=True,
            prompt=prompt,
        )

    def generate_light_chain(self, heavy_chain, num_return_sequences, top_p=0.95, temp=1.25, batch_size=1):
        prompt = f"1{heavy_chain}"
        return self.generate(
            num_return_sequences,
            backwards=False,
            top_p=top_p,
            temp=temp,
            batch_size=batch_size,
            prompt=prompt,
        )

    def get_batch_log_likelihoods(self, sequences, batch_size=32):
        likelihoods = []
        bos_token_id = self.tokenizer.encode("1")[0]
        eos_token_id = self.tokenizer.encode("2")[0]
        pad_token_id = self.tokenizer.pad_token_id
        special_token_ids = [bos_token_id, eos_token_id, pad_token_id]

        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            for input_id, logit in zip(input_ids, logits):
                shift_logits = logit[:-1, :].contiguous()
                shift_labels = input_id[1:].contiguous().long()

                mask = torch.ones(shift_labels.shape, dtype=torch.bool).to(self.device)
                for token_id in special_token_ids:
                    mask = mask & (shift_labels != token_id)

                nll = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1))[mask],
                    shift_labels.view(-1)[mask],
                    reduction="mean",
                )
                likelihoods.append(-nll)

        return torch.stack(likelihoods, dim=0).cpu().numpy()


@click.command()
@click.option("--model_name", default="Wu1234sdsd/piggen-merged-finetuned", show_default=True)
@click.option("--tokenizer_name", default="ollieturnbull/p-IgGen", show_default=True)
@click.option("--heavy_chain_file", default=None)
@click.option("--light_chain_file", default=None)
@click.option("--initial_sequence", default=None)
@click.option("--n_sequences", default=1, show_default=True, type=int)
@click.option("--top_p", default=0.95, show_default=True, type=float)
@click.option("--temp", default=1.2, show_default=True, type=float)
@click.option("--bottom_n_percent", default=5, show_default=True, type=int)
@click.option("--backwards", is_flag=True)
@click.option("--output_file", required=True)
@click.option("--separate_chains", is_flag=True)
@click.option("--cache_dir", default=None)
@click.option("--device", default=None)
def main(model_name, tokenizer_name, heavy_chain_file, light_chain_file, initial_sequence,
         n_sequences, top_p, temp, bottom_n_percent, backwards, output_file,
         separate_chains, cache_dir, device):

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using tokenizer: {tokenizer_name}")
    model = CustomPIgGen(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        device=device,
    )

    if heavy_chain_file and light_chain_file:
        raise click.UsageError("Specify only one of --heavy_chain_file or --light_chain_file.")

    sequences = []

    if heavy_chain_file:
        with open(heavy_chain_file) as f:
            heavy_chains = [line.strip() for line in f if line.strip()]

        for i, heavy_chain in enumerate(heavy_chains):
            generated = model.generate_light_chain(
                heavy_chain=heavy_chain,
                num_return_sequences=n_sequences,
                top_p=top_p,
                temp=temp,
                batch_size=1,
            )
            generated_light_chains = [seq[len(heavy_chain):] for seq in generated]
            sequences.extend((i, light_seq) for light_seq in generated_light_chains)

    elif light_chain_file:
        with open(light_chain_file) as f:
            light_chains = [line.strip() for line in f if line.strip()]

        for i, light_chain in enumerate(light_chains):
            generated = model.generate_heavy_chain(
                light_chain=light_chain,
                num_return_sequences=n_sequences,
                top_p=top_p,
                temp=temp,
                batch_size=1,
            )
            generated_heavy_chains = [seq[:-len(light_chain)] for seq in generated]
            sequences.extend((i, heavy_seq) for heavy_seq in generated_heavy_chains)

    else:
        sequences = model.generate(
            num_return_sequences=n_sequences,
            top_p=top_p,
            temp=temp,
            batch_size=1,
            prompt=initial_sequence,
            discard_bottom_n_percent=bottom_n_percent,
            separated_output=separate_chains,
            backwards=backwards,
        )

    with open(output_file, "w") as f:
        if heavy_chain_file or light_chain_file:
            for idx, seq in sequences:
                f.write(f"{idx}, {seq}\n")
        else:
            if separate_chains:
                if isinstance(sequences, tuple) and len(sequences) == 2:
                    vh_list, vl_list = sequences
                    for vh, vl in zip(vh_list, vl_list):
                        f.write(f"{vh}, {vl}\n")
                else:
                    for seq in sequences:
                        f.write(f"{seq[0]}, {seq[1]}\n")
            else:
                for seq in sequences:
                    f.write(f"{seq}\n")

    logger.info(f"Wrote output to {output_file}")
    


if __name__ == "__main__":
    main()