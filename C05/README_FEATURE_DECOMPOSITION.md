# Feature Decomposition Methods for RNA-Seq Data

## Overview

This notebook provides a comprehensive demonstration of four major dimensionality reduction techniques used in bioinformatics and computational biology:

1. **PCA (Principal Component Analysis)**
2. **ICA (Independent Component Analysis)**
3. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
4. **UMAP (Uniform Manifold Approximation and Projection)**

## What's Included

### 1. Simulated RNA-Seq Data Generation
- **4 cell types**: T-cells, B-cells, Neurons, Hepatocytes
- **2000 genes** with biologically realistic expression patterns
- **200 samples** (50 per cell type)
- Intentional structure based on cell-type specific gene programs

### 2. Data Preprocessing Pipeline
- Log transformation (log2(counts + 1))
- Highly variable gene selection
- Z-score standardization
- Best practices for RNA-Seq analysis

### 3. Detailed Method Implementations

#### PCA
- Explained variance analysis
- Scree plots and cumulative variance
- Gene loading visualization
- Multiple PC comparisons

#### ICA
- Statistical independence analysis
- Component kurtosis (non-Gaussianity)
- Gene weight interpretation
- Comparison with PCA

#### t-SNE
- Parameter exploration (perplexity)
- KL divergence tracking
- Multiple visualizations
- Comparison with PCA

#### UMAP
- Parameter exploration (n_neighbors, min_dist)
- Fast computation
- Global+local structure preservation
- Parameter effect demonstrations

### 4. Comprehensive Visualizations
- Side-by-side method comparisons
- Parameter sensitivity analysis
- Gene contribution plots
- Publication-quality figures

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_feature_decomposition.txt

# Launch Jupyter
jupyter notebook feature_decomposition_demo.ipynb
```

## Quick Start

```python
# The notebook is self-contained and runs top-to-bottom
# Simply run all cells sequentially

# Expected runtime: 3-5 minutes (depending on hardware)
```

## Key Features

### Educational Content
- **Detailed explanations** of when to use each method
- **Advantages and disadvantages** clearly stated
- **Parameter guidance** for practical use
- **Interpretation tips** for biological data

### Code Quality
- Well-commented functions
- Reproducible (fixed random seeds)
- Modular design for easy modification
- Clear variable naming

### Visualizations
- **30+ plots** showing different aspects
- Publication-ready figure quality
- Color-coded by cell type
- Multiple comparison views

## Learning Outcomes

After completing this notebook, you will understand:

1. **When to use each method**
   - PCA for quick exploration
   - ICA for independent processes
   - t-SNE for visualization
   - UMAP for large datasets

2. **How parameters affect results**
   - Perplexity in t-SNE
   - n_neighbors and min_dist in UMAP
   - Number of components in PCA/ICA

3. **How to interpret results**
   - Explained variance (PCA)
   - Component independence (ICA)
   - Cluster structure (t-SNE, UMAP)
   - Gene contributions (loadings/weights)

4. **Best practices**
   - Proper preprocessing
   - Parameter tuning
   - Method selection
   - Biological validation

## Comparison Table

| Method | Speed | Preserves | Deterministic | New Data | Best For |
|--------|-------|-----------|---------------|----------|----------|
| **PCA** | ⚡⚡⚡ | Global | ✅ | ✅ | Quick exploration |
| **ICA** | ⚡⚡ | Independence | ⚠️ | ✅ | Gene modules |
| **t-SNE** | ⚡ | Local | ❌ | ❌ | Visualization |
| **UMAP** | ⚡⚡⚡ | Both | ⚠️ | ✅ | General use |

## Customization

The notebook is designed to be easily modified:

```python
# Change cell types and sample sizes
counts_df, metadata_df = generate_rnaseq_data(
    n_samples_per_type=100,  # Increase samples
    n_genes=5000,            # More genes
    noise_level=0.5          # More noise
)

# Try different preprocessing
data_scaled, log_counts_hvg = preprocess_rnaseq(
    counts_df,
    n_top_genes=1000  # Select more genes
)

# Experiment with parameters
tsne = TSNE(perplexity=50)  # Try different values
umap = UMAP(n_neighbors=30, min_dist=0.5)
```

## Common Use Cases

### 1. Single-Cell RNA-Seq Analysis
```python
# UMAP is most popular for scRNA-seq
# Fast, preserves structure, works with large datasets
umap = UMAP(n_neighbors=15, min_dist=0.1)
embedding = umap.fit_transform(data_scaled)
```

### 2. Bulk RNA-Seq Sample Clustering
```python
# PCA for quick overview, then t-SNE for final viz
pca = PCA(n_components=50)
pca_result = pca.fit_transform(data_scaled)
tsne = TSNE(perplexity=30)
final_viz = tsne.fit_transform(pca_result)
```

### 3. Gene Module Discovery
```python
# ICA identifies independent gene programs
ica = FastICA(n_components=20)
components = ica.fit_transform(data_scaled)
gene_weights = ica.components_  # Analyze these
```

## Troubleshooting

### Issue: t-SNE is very slow
**Solution**:
- Reduce dataset size
- Use PCA preprocessing (reduce to 50 PCs first)
- Try UMAP instead

### Issue: UMAP results vary between runs
**Solution**:
- Set `random_state=42` for reproducibility
- UMAP has some stochastic elements but is more stable than t-SNE

### Issue: Clusters overlap
**Solution**:
- Try different perplexity (t-SNE) or n_neighbors (UMAP)
- Select more highly variable genes
- Check if biological separation truly exists

### Issue: Results look different from PCA
**Solution**:
- This is expected! Nonlinear methods capture different structure
- PCA is linear, t-SNE/UMAP are nonlinear
- Both can be "correct" - they show different aspects

## Further Reading

### Papers
- **PCA**: Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space"
- **ICA**: Hyvärinen, A., & Oja, E. (2000). "Independent component analysis: algorithms and applications"
- **t-SNE**: van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE"
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform manifold approximation and projection"

### Applications in Bioinformatics
- Luecken, M. D., & Theis, F. J. (2019). "Current best practices in single-cell RNA-seq analysis"
- Becht, E., et al. (2019). "Dimensionality reduction for visualizing single-cell data using UMAP"

## Contributing

This notebook is designed for educational purposes. Feel free to:
- Modify the simulation parameters
- Add new cell types
- Try different distance metrics
- Create 3D visualizations
- Add batch effects to test robustness

## License

Educational use - free to modify and distribute with attribution.

---

**Created for**: BINFX410 - Machine Learning for Bioinformatics
**Last Updated**: December 2025
**Estimated Runtime**: 3-5 minutes
