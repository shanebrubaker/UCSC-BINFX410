# CNN-Based Copy Number Variation (CNV) Caller

**BINFX 410 — Chapter 5 Solution**

A complete pipeline for detecting genomic copy number variations (CNVs) from sequencing read-depth data using a 1D Convolutional Neural Network built in PyTorch.

---

## Overview

Copy number variations — deletions and duplications of genomic segments — are a major class of structural variant with clinical significance in cancer, rare disease, and population genetics. This project simulates realistic whole-genome sequencing read-depth data, applies a multi-step normalization pipeline to remove technical biases, trains a 1D-CNN to classify genomic windows as deletion / normal / duplication, and reports results with genome-wide visualizations and a CNV segment table. An advanced pseudogene analysis module demonstrates how read mismapping from high-homology paralogs generates false-positive CNV calls and how MAPQ-based filtering recovers accuracy.

---

## Repository Structure

```
solution-cnn-cnv/
├── cnn_cnv_caller.ipynb   # Main notebook (all sections)
├── environment.yml        # Conda environment definition
├── requirements.txt       # Pip requirements (for non-conda setups)
├── plan.md                # Implementation plan and design rationale
└── README.md              # This file
```

Figures are written to the working directory when the notebook is run:

```
fig1_raw_data.png              Raw depth + GC wave + mappability track
fig2_normalization.png         All 5 normalization stages overlaid
fig3_norm_distributions.png    Depth distributions at each stage
fig4_training_curves.png       Loss and accuracy vs. epoch
fig5_confusion_matrix.png      Normalised confusion matrix (test set)
fig6_roc_curves.png            One-vs-rest ROC + AUC for all classes
fig7_genome_wide_cnv.png       Genome-wide prediction probability stack
fig8_cnv_segments.png          Segment size distribution + event counts
fig9_pseudogene_analysis.png   Mismapping effect on calls (5-panel)
fig10_pseudogene_metrics.png   Precision/Recall/F1: clean vs. filtered
```

---

## Environment Setup

### Recommended: create a dedicated conda environment

```bash
conda env create -f environment.yml   # only needed once
conda activate cnn-cnv
jupyter notebook cnn_cnv_caller.ipynb
```

This creates an isolated `cnn-cnv` environment with all required packages at tested versions.

### Alternative: install into an existing environment

```bash
pip install -r requirements.txt
```

> **Why `numpy<2`?**
> PyTorch builds distributed via PyPI pin their NumPy ABI at compile time.
> PyTorch ≤ 2.1 was compiled against NumPy 1.x; running it with NumPy 2.x
> produces the error `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`.
> Pinning `numpy<2` in the environment eliminates this crash entirely.

---

## Pipeline Summary

### 1. Synthetic Data Generation

Simulates 30 samples × 5,000 genomic bins (1 kb resolution, ~5 Mb total) with realistic sequencing noise:

- **Copy number events** — 3–7 per sample; deletions (CN 0–1) and duplications (CN 3–6) placed at random positions and sizes
- **GC-content wave** — sinusoidal bias (±15%) with ~300 kb period
- **Mappability dips** — 2% of bins set to 0.1–0.4× mappability
- **Negative-binomial overdispersion** — per-bin count noise (r = 10)
- **Batch offset** — per-sample ±15% depth scale

### 2. Normalization (5 Methods)

| Step | Method | Bias removed |
|------|--------|--------------|
| 1 | Median normalization | Per-sample depth scale |
| 2 | GC-content correction | GC-bias via linear regression |
| 3 | Mappability correction | Low-complexity region artefacts |
| 4 | Z-score (bin-wise) | Between-sample variance per bin → **CNN input** |
| 5 | Quantile normalization | Distribution shape differences across samples |

### 3. CNN Architecture

```
Input  (1 channel, 50 bins)
  Conv1d(1→64,  k=5, pad=2)  → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
  Conv1d(64→128,k=3, pad=1)  → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
  Conv1d(128→256,k=3,pad=1)  → BatchNorm → ReLU → AdaptiveAvgPool(1)
  Flatten → Linear(256→128)  → ReLU → Dropout(0.4)
  Linear(128→3)               → class logits (Del / Normal / Dup)
```

- Class-weighted cross-entropy loss (handles class imbalance)
- Adam optimizer, `ReduceLROnPlateau` scheduler, early stopping (patience = 8)
- Runs on CPU by default

### 4. Evaluation

- Held-out test set: stratified 15% split
- Per-class precision, recall, F1
- Normalised confusion matrix
- One-vs-rest ROC curves and AUC for all three classes
- Genome-wide sliding-window prediction with probability stack plot

### 5. CNV Segment Report

Run-length encoding of per-bin predictions → segment table with:
- Genomic coordinates (start/end bin, kb)
- Segment length
- CNV class and class ID

### 6. Pseudogene Analysis (Advanced)

Four simulated gene–pseudogene pairs with 87–95% sequence identity:

| Gene pair | Sequence identity | Mismapping fraction |
|-----------|-------------------|---------------------|
| GENE1/PGENE1 | 92% | 60% |
| GENE2/PGENE2 | 88% | 40% |
| GENE3/PGENE3 | 95% | 75% |
| GENE4/PGENE4 | 87% | 35% |

**Effect:** reads from the parent locus are redistributed to the pseudogene locus, creating phantom duplications and suppressing the true signal. MAPQ drops proportionally to mismapping fraction.

**Mitigation:** bins with MAPQ < 30 are masked to Normal before reporting. Precision, recall, and F1 are compared across three conditions: clean data, with mismapping, and after MAPQ filtering.

---

## CNV Classes

| Label | Class | Copy number | Depth fold-change |
|-------|-------|-------------|-------------------|
| 0 | Deletion | 0–1 | 0–0.5× |
| 1 | Normal | 2 | 1.0× |
| 2 | Duplication | 3–6 | 1.5–3.0× |
