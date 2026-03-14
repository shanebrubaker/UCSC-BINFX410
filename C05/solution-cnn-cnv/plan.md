# Implementation Plan: CNN CNV Caller

**BINFX 410 — Chapter 5**

---

## Objective

Build a complete copy number variation (CNV) detection pipeline that:
1. Generates synthetic sequencing read-depth data with CNVs introduced
2. Applies 3–5 normalization methods to remove technical biases
3. Trains a CNN to classify genomic windows as deletion / normal / duplication
4. Reports and visualizes CNV calls genome-wide
5. *(Advanced)* Demonstrates pseudogene-driven false positives and MAPQ-based mitigation

---

## Design Decisions

### Why synthetic data?

Real WGS data requires large files and reference genome access. Synthetic data lets us:
- Control the ground truth precisely (known CNV locations, sizes, copy numbers)
- Tune noise parameters to match realistic sequencing characteristics
- Evaluate all metrics against a known answer

Noise model chosen: **negative-binomial** overdispersion (r = 10) — standard for count-based sequencing depth, more realistic than Poisson for WGS.

### Why 1D-CNN over other approaches?

| Approach | Pros | Cons |
|----------|------|------|
| HMM (CBS-style) | Interpretable, no training data | Assumes Gaussian noise, no spatial features |
| Random Forest | Fast, robust | Requires hand-crafted features per bin |
| LSTM/RNN | Long-range dependencies | Slow, harder to train |
| **1D-CNN** | Learns local depth patterns, fast inference, parallelizable | Fixed window size |

A 1D-CNN is a natural fit: CNVs manifest as consistent depth shifts over contiguous windows, which convolution filters can detect directly from the normalized signal.

### Why sliding windows?

Converting the per-bin depth signal to fixed-size windows:
- Produces a large training set from relatively few samples
- Captures local context (neighboring bins inform the call)
- Enables genome-wide inference via a single forward pass per window
- Window size (50 bins = 50 kb) chosen to span typical CNV event lengths while staying well above the receptive field of the filters

### Normalization strategy

Five methods were chosen to address the four primary sources of bias in WGS read depth:

```
Raw depth
    ↓  Median normalize     → remove sample-level depth scale
    ↓  GC correction        → remove GC-content wave
    ↓  Mappability correct  → remove low-complexity artefacts
    ↓  Z-score (bin-wise)   → centre and scale for CNN input   ← used as CNN input
    ↓  Quantile normalize   → (shown for comparison)
```

Z-score normalization was selected as the CNN input because it:
- Produces zero-mean, unit-variance signal that stabilizes training
- Makes deviations from the cohort mean directly interpretable as copy-number changes
- Is computed across the sample cohort (bin-wise), so a duplicated bin shows as a positive z-score relative to diploid samples

### Class imbalance handling

CNV events cover ~20–40% of bins per sample on average in this simulation, but in real data deletions and duplications are far less common than diploid regions. Class-weighted cross-entropy loss is used to prevent the model from defaulting to "Normal" for all predictions.

---

## Implementation Phases

### Phase 1 — Data generation
- [x] Define GC-wave and mappability tracks
- [x] Implement `place_cnv_segments()` — random CNV placement with configurable length/type distributions
- [x] Implement `simulate_sample()` — negative-binomial depth generation with batch offset
- [x] Generate 30 samples × 5,000 bins

### Phase 2 — Normalization pipeline
- [x] Median normalization
- [x] GC-content correction (per-sample linear regression)
- [x] Mappability correction (divide + interpolate masked bins)
- [x] Z-score normalization (bin-wise across cohort)
- [x] Quantile normalization (rank-mean replacement)
- [x] Visualization: all stages, distribution comparisons

### Phase 3 — Dataset construction
- [x] Sliding window extraction (`WINDOW_SIZE=50`, `STEP_SIZE=25`)
- [x] Majority-vote label per window
- [x] Stratified train / val / test split (70 / 15 / 15)
- [x] Tensor conversion using `torch.as_tensor(..., dtype=...)` for safe numpy interop
- [x] Class-weight computation

### Phase 4 — CNN model
- [x] `CNVNet`: 3× Conv1d blocks + AdaptiveAvgPool + classifier head
- [x] Training loop with early stopping (patience = 8) and `ReduceLROnPlateau`
- [x] Best-weights checkpoint via CPU state dict copy

### Phase 5 — Evaluation
- [x] Classification report (precision / recall / F1 / support)
- [x] Normalised confusion matrix
- [x] One-vs-rest ROC curves + AUC
- [x] Genome-wide sliding-window prediction (`predict_genome_wide()`)
- [x] Per-bin probability stack plot

### Phase 6 — CNV segment report
- [x] Run-length encoding of per-bin predictions → segment DataFrame
- [x] Minimum segment length filter (5 bins = 5 kb)
- [x] Segment size distribution histogram
- [x] Event count bar chart

### Phase 7 — Pseudogene analysis (advanced)
- [x] Define 4 gene–pseudogene pairs with varying identity (87–95%)
- [x] `simulate_mismapping()` — redistribute reads proportional to identity
- [x] Per-bin MAPQ track computed from mismapping fraction
- [x] MAPQ < 30 filter applied to suppress false calls
- [x] TP / FP / FN / Precision / Recall / F1 comparison across three conditions
- [x] 5-panel visualization (depth, true, mismapped calls, filtered calls, MAPQ)
- [x] Metric bar chart (clean vs. mismapped vs. filtered)

---

## Technical Notes

### NumPy / PyTorch ABI compatibility

PyTorch packages distributed via PyPI pin their NumPy ABI at compile time.
PyTorch ≤ 2.1 was built against NumPy 1.x. Running those builds with NumPy ≥ 2.0
installed produces a hard crash:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x as it may crash.
```

**Fix:** `environment.yml` pins `numpy<2`, which is fully compatible with any PyTorch
2.0+ release. This also allows the standard `tensor.numpy()` conversion to work without
any workarounds.

### Device

The notebook targets CPU by default (`DEVICE = torch.device('cpu')`). This avoids
platform-specific issues with MPS (Apple Silicon) or CUDA detection and ensures the
notebook runs identically on any machine. Training on the synthetic dataset (5,000 bins ×
30 samples) typically completes in a few minutes on CPU.

### Tensor construction

Arrays passed to `torch.as_tensor(..., dtype=torch.float32)` are kept as `float64`
(NumPy default) until the call site; the dtype cast to `float32` happens inside PyTorch.
This avoids any ambiguity with `torch.tensor()` which can infer types inconsistently
across NumPy versions.

### Window label strategy

Majority-vote over window bins was chosen over center-bin label or fractional threshold because:
- More robust to CNV boundary effects
- Naturally handles small CNVs that may not dominate a window (they get labeled Normal, reducing FPs at boundaries)
- Consistent with how segment callers aggregate per-position evidence

---

## Parameter Reference

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `N_BINS` | 5,000 | ~5 Mb at 1 kb resolution |
| `N_SAMPLES` | 30 | Sufficient cohort for bin-wise z-score |
| `MEAN_DEPTH` | 50× | Typical WGS coverage |
| `WINDOW_SIZE` | 50 bins (50 kb) | Captures most CNV events; ≥ filter length |
| `STEP_SIZE` | 25 bins | 50% overlap — smooths boundary calls |
| `BATCH` | 256 | Comfortable for CPU training |
| `EPOCHS` | 60 (max) | Early stopping fires well before this |
| `PATIENCE` | 8 epochs | ~2 LR reductions before stopping |
| `LR` | 1e-3 | Adam default; halved every 4 epochs without improvement |
| `MAPQ_THRESH` | 30 | Standard mapping quality cutoff in variant calling |
| `MIN_SEGMENT_BINS` | 5 (5 kb) | Suppress single-bin noise calls |

---

## Possible Extensions

- **Multi-sample calling** — joint segmentation across the cohort (reduce FP rate via cohort z-score)
- **Regression head** — predict continuous copy number instead of 3-class labels
- **Attention layer** — replace GlobalAvgPool with self-attention for longer context
- **Real data** — apply to public WGS cohorts (e.g., 1000 Genomes) with GC/mappability tracks from ENCODE
- **Breakpoint refinement** — use gradient-weighted class activation maps (Grad-CAM) to localize CNV boundaries within the window
- **Somatic CNV** — pair tumour/normal samples; use log2 ratio signal instead of raw depth
