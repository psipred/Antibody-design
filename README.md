# **Antibody Fv Generative Model for high accuracy prediction of CDR-H3 Loops**
---
# Finetuned p-IgGen Inference

This repository provides a local inference workflow for our finetuned antibody language model based on p-IgGen.

The inference script supports:

- Generation of full-length paired antibody sequences
- Conditional generation from a provided heavy or light chain
- Prompted generation from an initial sequence
- Optional likelihood-based filtering of outputs
- Optional separation of VH and VL chains

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

### 4. Install HMMER (required by ANARCI)

```bash
conda install -n piggen_infer -c bioconda hmmer -y
```

### 5. Install ANARCI

```bash
conda install -n piggen_infer -c bioconda anarci -y
```

If the conda ANARCI install fails with an `UnsatisfiableError`, use:

```bash
conda run -n piggen_infer pip install anarci
```

> Note: This inference script imports ANARCI at startup, so ANARCI must be installed
> even if you are not using `--separate_chains`.

---

## Inference Script

Save the inference script as `github_inference.py`.

---

## Usage

Run inference:

```bash
python github_inference.py --output_file output_sequences.txt
```

> **Notes**
> - Generation uses sampling (`do_sample=True`)
> - `max_new_tokens` is fixed at 400
> - `bottom_n_percent` only applies when `n_sequences >= 100`
> - GPU is used automatically if available
> - Some generated samples can be dropped by post-generation validation, so written
>   output count may be lower than `--n_sequences`

---

## Parameters

### Model and tokenizer

| Flag | Description | Default |
|------|-------------|---------|
| `--model_name TEXT` | Hugging Face model or local path | `Wu1234sdsd/piggen-merged-finetuned` |
| `--tokenizer_name TEXT` | Tokenizer repository or path | `ollieturnbull/p-IgGen` |

### Input conditioning

| Flag | Description |
|------|-------------|
| `--heavy_chain_file TEXT` | File containing heavy chain sequences (one per line) |
| `--light_chain_file TEXT` | File containing light chain sequences (one per line) |
| `--initial_sequence TEXT` | Initial sequence prompt |

For p-IgGen forward prompting, prefix the prompt with `1`.
Example: `--initial_sequence 1QVQLVES...`

### Sampling

| Flag | Description | Default |
|------|-------------|---------|
| `--n_sequences INTEGER` | Number of sequences to generate | `1` |
| `--top_p FLOAT` | Top-p nucleus sampling | `0.95` |
| `--temp FLOAT` | Sampling temperature | `1.2` |
| `--bottom_n_percent INTEGER` | Percentage of lowest-likelihood sequences to discard (only used if `n_sequences >= 100`) | `5` |

### Generation control

| Flag | Description |
|------|-------------|
| `--backwards` | Generate sequences in reverse direction |
| `--separate_chains` | Output VH and VL separately (requires ANARCI) |

### Output

| Flag | Description |
|------|-------------|
| `--output_file TEXT` | Output file path (required) |

### Runtime

| Flag | Description |
|------|-------------|
| `--cache_dir TEXT` | Hugging Face cache directory |
| `--device TEXT` | Device to run inference on |

Automatically selects: `cuda` → `mps` → `cpu`

If CUDA is detected but unsupported by your GPU/PyTorch build, run with:
`--device cpu`

---

## Output Format

### Default output

```
SEQUENCE_1
SEQUENCE_2
SEQUENCE_3
```

### Conditional generation output

```
index, generated_sequence
```

### Separate chain output

```
VH_SEQUENCE, VL_SEQUENCE
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
conda install -n piggen_infer -c bioconda anarci -y
# If that fails:
conda run -n piggen_infer pip install anarci
```

**`hmmscan` not found**
```bash
conda install -n piggen_infer -c bioconda hmmer -y
```

**`--separate_chains` not working**

Ensure both ANARCI and HMMER are installed and available in `PATH`.

**Prompted generation returns 0 sequences**

Use `--initial_sequence` with a leading `1` token (for example,
`--initial_sequence 1QVQLVES...`).

---

## About

This repository provides local inference for a finetuned version of p-IgGen.

Upstream package: [https://github.com/OliverT1/p-IgGen](https://github.com/OliverT1/p-IgGen)
