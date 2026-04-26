# **Antibody Fv Generative Model for high accuracy prediction of CDR-H3 Loops**
---
## Overview

This repository provides a local inference workflow for our finetuned antibody language model based on p-IgGen. Our Finetuned model is able to generate paried Fv sequences that has high CDR H3 loop prediction accuracy

---

## Installation

### Prerequisites

- Conda
- Git
- Internet access (for downloading model and dependencies)

### 1. Create environment

```bash
conda create -n piggen_infer python=3.11 pip -y
conda activate piggen_infer
```

### 2. Install p-IgGen

```bash
pip install git+https://github.com/OliverT1/p-IgGen.git
```

### 3. Install required dependencies

```bash
pip install torch transformers click
```

### 4. Install ANARCI 

```bash
conda install -c bioconda anarci
```

### 5. Install HMMER 

```bash
conda install -c bioconda hmmer
```

---

## Inference Script

Download the inference script directly from this repository:

```bash
wget https://raw.githubusercontent.com/psipred/Antibody-design/main/operations/user_inference.py -O github_inference.py
```

Or with curl:

```bash
curl -o github_inference.py https://raw.githubusercontent.com/psipred/Antibody-design/main/operations/user_inference.py
```

---

## Generation Modes

### 1. Unconditional generation

Generate complete paired antibody sequences from scratch, with no input conditioning. The model samples freely from its learned distribution.

```bash
python github_inference.py \
  --n_sequences 10 \
  --output_file output_sequences.txt
```

---

### 2. Conditional generation from a heavy chain

Provide a file of known heavy chain sequences (one per line). For each heavy chain, the model generates a paired light chain. The heavy chain is used as a prefix prompt (`1{heavy_chain}`) and the model generates the remainder of the paired sequence.

```bash
python github_inference.py \
  --heavy_chain_file heavy_chains.txt \
  --n_sequences 5 \
  --output_file output_sequences.txt
```

Output format: `index, generated_light_chain`

---

### 3. Conditional generation from a light chain

The reverse of the above — provide known light chain sequences and the model generates a paired heavy chain for each. Internally the model runs in backwards mode, reversing the light chain and generating the heavy chain in the reverse direction.

```bash
python github_inference.py \
  --light_chain_file light_chains.txt \
  --n_sequences 5 \
  --output_file output_sequences.txt
```

Output format: `index, generated_heavy_chain`

> **Note:** `--heavy_chain_file` and `--light_chain_file` are mutually exclusive — passing both will raise an error.

---

### 4. Prompted generation from an initial sequence

Provide the beginning of a sequence and the model will continue generating from it. This is open-ended generation — unlike the chain file modes, no pairing logic is applied. The model treats your input as a raw sequence prefix and samples forward from that point.

The example below uses a partial heavy chain as the prompt:

```bash
python github_inference.py \
  --initial_sequence QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKG \
  --n_sequences 20 \
  --top_p 0.9 \
  --temp 1.1 \
  --output_file output_sequences.txt
```

This generates 20 sequences that all begin with `QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKG`, with slightly tighter sampling than the defaults (`--top_p 0.9` vs `0.95`, `--temp 1.1` vs `1.2`), producing more conservative completions while still maintaining diversity.

Output format: one complete sequence per line.

---

## Parameters

### Model and tokenizer

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model_name` | TEXT | `Wu1234sdsd/piggen-merged-finetuned` | Hugging Face model repo or local path to load the finetuned model from |
| `--tokenizer_name` | TEXT | `ollieturnbull/p-IgGen` | Hugging Face tokenizer repo or local path. Should generally be left as default unless using a custom tokenizer |

### Input conditioning

Exactly one of the following may be provided. If none are given, the model generates unconditionally.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--heavy_chain_file` | TEXT | None | Path to a file of heavy chain sequences, one per line. For each sequence, the model generates `--n_sequences` paired light chains |
| `--light_chain_file` | TEXT | None | Path to a file of light chain sequences, one per line. For each sequence, the model generates `--n_sequences` paired heavy chains using backwards generation |
| `--initial_sequence` | TEXT | None | A partial sequence to use as a generation prompt. The model will continue the sequence from this prefix. Cannot be combined with `--heavy_chain_file` or `--light_chain_file` |

### Sampling

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_sequences` | INTEGER | `1` | Number of sequences to generate. When using `--heavy_chain_file` or `--light_chain_file`, this is the number of sequences generated **per input chain** |
| `--top_p` | FLOAT | `0.95` | Nucleus sampling threshold. At each generation step, only tokens whose cumulative probability reaches `top_p` are considered. Lower values (e.g. `0.8`) make outputs more focused and conservative; higher values allow more diversity |
| `--temp` | FLOAT | `1.2` | Sampling temperature. Higher values (e.g. `1.4`) increase randomness and sequence diversity; lower values (e.g. `0.9`) make the model more deterministic and likely to produce higher-probability sequences |
| `--bottom_n_percent` | INTEGER | `5` | Percentage of generated sequences to discard based on model log-likelihood. Only applied when `--n_sequences >= 100`. For example, `--bottom_n_percent 10` discards the lowest-scoring 10% of outputs, returning the top 90% |

### Generation control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backwards` | FLAG | False | Generate sequences in the reverse direction. Used internally by `--light_chain_file` mode. Can also be set manually for custom backwards generation workflows |
| `--separate_chains` | FLAG | False | After generation, use ANARCI to split outputs into separate VH and VL chains. Requires ANARCI and HMMER to be installed. Only applies to unconditional or prompted generation — not compatible with `--heavy_chain_file` or `--light_chain_file` modes |

### Output

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output_file` | TEXT | — | Path to write output sequences. **Required.** |

### Runtime

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache_dir` | TEXT | None | Directory for caching Hugging Face model and tokenizer downloads. Useful if you want to store model weights in a specific location or reuse a previously downloaded model |
| `--device` | TEXT | None | Device to run inference on (`cuda`, `mps`, or `cpu`). If not specified, the script automatically selects the best available device in order: `cuda` → `mps` → `cpu` |

---

## Output Format

### Unconditional / prompted generation

One complete sequence per line:

```
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVSYISSSGSTIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAREDYYGMDVWGQGTTVTVSSQSALTQPASVSGSPGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIYEG
QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVSYISSSGSTIYVADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAREDYYGMDVWGQGTTVTVSSQSALTQPASVSGSAGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIY
```

### Conditional generation (heavy or light chain file)

Index of the input chain followed by the generated partner chain:

```
0, SYELTQPPSVSVSPGQTARITCSGDALPKQYAYWYQQKSGQAPVLVIYKDSERPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQSADSSGTYVFGTGTKVTVL
1, SYELTQPPSVSVSPGQTARITCSGDALPKQYAYWYQQKSGQAPVLVIYKDSERPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQSADSSGTYVFGTGTKVTVL
```

### Separate chain output (`--separate_chains`)

VH and VL sequences separated by a comma:

```
EVQLVESGGGLVQPGG..., SYELTQPPSVSVSPGQ...
```

---

## Quick Test

```bash
python -c "import torch; import transformers; print('Core dependencies OK')"
python -c "from piggen import utils; print('p-IgGen import OK')"
python github_inference.py --help
```

---

## Troubleshooting

**ANARCI not found**
```bash
conda install -c bioconda anarci
```

**`hmmscan` not found**
```bash
conda install -c bioconda hmmer
```

**`--separate_chains` not working**

Ensure both ANARCI and HMMER are installed and available in `PATH`.

---

## About

This repository provides local inference for a finetuned version of p-IgGen.

Upstream package: [https://github.com/OliverT1/p-IgGen](https://github.com/OliverT1/p-IgGen)
