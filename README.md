# **Antibody Fv Generative Model for Structurally Reliable CDR-H3 Loops**
---
# Finetuned p-IgGen Inference

This repository provides a local inference workflow for our finetuned antibody language model based on **p-IgGen**. The inference script supports generation of full-length paired antibody sequences, conditional generation from a provided heavy or light chain, prompted generation from an initial sequence, optional likelihood-based filtering of outputs, and optional separation of VH and VL chains. The original p-IgGen package also provides utility functions for generation and scoring of antibody sequences, with model weights hosted on Hugging Face. :contentReference[oaicite:1]{index=1}

## Features

- Generate full-length antibody sequences.
- Generate a light chain given a heavy chain.
- Generate a heavy chain given a light chain.
- Generate full-length antibody sequences from an initial sequence prompt.
- Filter generated sequences by model log-likelihood.
- Optionally output VH and VL separately.

## Installation

We recommend installing and running inference inside a fresh conda environment.

### Prerequisites

- Conda
- Git
- Internet access for downloading the p-IgGen package and model weights

### Step-by-Step Setup

1. Create and activate a new conda environment:

```bash
conda create -n piggen_infer python=3.11 pip -y
conda activate piggen_infer
```bash
## Overview
---

This project presents a fine-tuned antibody Fv sequence generative model designed to produce sequences with CDR-H3 loops that are more structurally predictable and reliable. By training on sequences with accurately modelled H3 conformations, the model learns sequence patterns associated with increased structural determinism while preserving sequence diversity, enabling exploration of a broad antibody sequence space.

---

## Data Curation
---

### ColabFold
ColabFold is run without MSA. Ten human paired antibody PDB structures were selected from SAbDab as templates. These were trimmed to Fv regions, insertion codes were removed, and CDR-H3 residues were masked.

---

### SAbDab Sequences
Human paired antibody–antigen complex structures were downloaded from SAbDab and filtered to retain only monomeric antibodies. Antibody sequences were extracted in FASTA format and trimmed to Fv regions.

Sequences were modelled using ColabFold and compared to ground-truth bound conformations. CDR-H3 RMSD was calculated by aligning on the H3 loop and computing RMSD over Cα atoms.

Sequences with H3 RMSD ≤ 1 Å were included in the training set.

---

### OAS Sequences
Human paired antibody Fv sequences from healthy donor B cells were obtained from OAS. Raw `.csv.gz` files were processed to extract full Fv and CDR-H3 sequences.

Sequences were clustered at 99% H3 sequence identity using MMseqs2, retaining one sequence per cluster. These were modelled using ColabFold, and sequences with mean H3 pLDDT ≥ 80 were included in the training set.

---

### Clustering
Combined SAbDab and OAS sequences were clustered at 50% H3 sequence identity using MMseqs2. Data were split into training and validation sets (9:1) at the cluster level.

---

## Model Training
---

The dataset was used to fine-tune the p-IgGen model using LoRA.

- Loss computed only on CDR-H3 residues  
- H3 residue indices obtained from `loop.csv`  
- Early stopping applied after three consecutive non-improving evaluations  

---

## Inference
---

- 200 unpaired sequences generated (temperature = 0.5)  
- Sequences paired during preprocessing  

---

## Evaluation
---

### Structural Confidence and Accuracy
- Mean H3 pLDDT (ColabFold)  
- ABB2 per-residue error  
- Framework-region pLDDT flanking H3  
- H3–Fv Predicted Aligned Error (PAE)  

### Structural Determinism
- Mean pairwise H3 RMSD (multi-seed ColabFold)  
- Structural clustering of H3 conformations  

### Structural Rigidity
- H3 loop length distribution  
- ITsFlexible scores  
- Fraction of rigid-like H3 loops  
- CABS-flex RMSF  

### Sequence Diversity
- Internal nearest-neighbor identity  
- Fraction of unique H3 sequences  

### Novelty Analysis
- Nearest-neighbor identity to training set  
- Metrics evaluated at matched identity (pLDDT, RMSD)  

---

## Results
---

### Improved Structural Confidence and Accuracy
- H3 pLDDT increased (65.2 → 74.4)  
- ABB2 error decreased  
- Framework pLDDT unchanged  
- H3–Fv PAE decreased  

---

### Increased Structural Determinism
- Mean pairwise H3 RMSD decreased (1.43 Å → 0.749 Å)  
- Single-cluster fraction increased (51% → 79.5%)  

---

### Increased Structural Rigidity
- Mean H3 length reduced (14.8 → 12.1 aa)  
- ITsFlexible scores decreased (0.107 → 0.0516)  
- Rigid-like fraction increased (9.0% → 27.5%)  
- Mean H3 RMSF decreased (0.79 Å → 0.65 Å)  

---

### Preserved Sequence Diversity
- Internal NN identity unchanged (0.781 vs 0.776)  
- Nearly all sequences remained unique (100% vs 99.0%)  

---

### Improvements Beyond Sequence Similarity
- NN identity to training set increased (0.436 → 0.505)  

At matched identity:
- H3 pLDDT remained higher  
- H3 RMSD remained lower  

---
