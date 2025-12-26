"""
Generate synthetic gene expression dataset with intentional data quality issues.

This script creates a messy dataset for teaching data cleaning, labeling, and
normalization in machine learning.
"""

import numpy as np
import pandas as pd
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_gene_expression_data(n_samples=100, n_genes=50):
    """
    Generate synthetic gene expression data with intentional issues.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_genes : int
        Number of genes to measure

    Returns:
    --------
    tuple : (expression_df, metadata_df)
    """

    # Generate base gene expression data
    # Cancer samples have higher expression on average
    n_cancer = 60
    n_normal = 40

    # Generate expression values with different scales per gene
    gene_names = [f"GENE_{i+1:03d}" for i in range(n_genes)]

    # Normal tissue (lower expression)
    normal_expr = np.random.lognormal(mean=2, sigma=1.5, size=(n_normal, n_genes))

    # Cancer tissue (higher expression with more variance)
    cancer_expr = np.random.lognormal(mean=3, sigma=1.8, size=(n_cancer, n_genes))

    # Combine
    expression_data = np.vstack([normal_expr, cancer_expr])

    # Apply different scales to different genes to make normalization important
    scale_factors = np.random.uniform(0.01, 100, size=n_genes)
    expression_data = expression_data * scale_factors

    # Add extreme outliers to 3 genes
    outlier_genes = random.sample(range(n_genes), 3)
    for gene_idx in outlier_genes:
        outlier_samples = random.sample(range(n_samples), 2)
        for sample_idx in outlier_samples:
            # Make outliers 5-10 standard deviations from mean
            mean_val = np.mean(expression_data[:, gene_idx])
            std_val = np.std(expression_data[:, gene_idx])
            expression_data[sample_idx, gene_idx] = mean_val + random.choice([1, -1]) * random.uniform(5, 10) * std_val

    # Create sample IDs (will be made inconsistent later)
    base_sample_ids = [f"SAMPLE_{i+1:03d}" for i in range(n_samples)]

    # Create DataFrame
    expr_df = pd.DataFrame(expression_data, columns=gene_names)
    expr_df.insert(0, 'sample_id', base_sample_ids)

    # Add duplicate samples (3-4 duplicates)
    duplicate_indices = random.sample(range(n_samples), 4)
    duplicate_rows = expr_df.iloc[duplicate_indices].copy()
    expr_df = pd.concat([expr_df, duplicate_rows], ignore_index=True)

    # Introduce missing values (5-10% of data)
    n_missing = int(0.07 * expr_df.shape[0] * expr_df.shape[1])
    missing_positions = random.sample(
        [(i, j) for i in range(len(expr_df)) for j in range(1, len(expr_df.columns))],
        n_missing
    )
    for row, col in missing_positions:
        expr_df.iloc[row, col] = np.nan

    # Make some sample IDs inconsistent
    for idx in random.sample(range(len(expr_df)), 20):
        original_id = expr_df.loc[idx, 'sample_id']
        # Different inconsistent formats
        format_choice = random.choice(['dash', 'underscore_lower', 'patient_prefix'])
        if format_choice == 'dash':
            expr_df.loc[idx, 'sample_id'] = original_id.replace('_', '-')
        elif format_choice == 'underscore_lower':
            expr_df.loc[idx, 'sample_id'] = original_id.lower()
        elif format_choice == 'patient_prefix':
            num = original_id.split('_')[1]
            expr_df.loc[idx, 'sample_id'] = f"Patient_{num}"

    return expr_df, base_sample_ids, n_normal

def generate_metadata(base_sample_ids, n_normal):
    """
    Generate sample metadata with inconsistent labels and IDs.

    Parameters:
    -----------
    base_sample_ids : list
        List of base sample IDs
    n_normal : int
        Number of normal samples

    Returns:
    --------
    pd.DataFrame : Metadata with labels and demographics
    """
    n_samples = len(base_sample_ids)

    # Generate labels with inconsistent terminology
    labels = []

    # Normal samples (indices 0 to n_normal-1)
    normal_terms = ['normal', 'healthy', 'control', 'Normal']
    for i in range(n_normal):
        labels.append(random.choice(normal_terms))

    # Cancer samples
    cancer_terms = ['cancer', 'tumor', 'malignant', 'Cancer']
    for i in range(n_normal, n_samples):
        labels.append(random.choice(cancer_terms))

    # Replace 3 labels with ambiguous ones
    ambiguous_indices = random.sample(range(n_samples), 3)
    ambiguous_terms = ['borderline', 'unclear', 'suspicious']
    for idx, term in zip(ambiguous_indices, ambiguous_terms):
        labels[idx] = term

    # Generate demographics
    ages = np.random.randint(25, 85, size=n_samples)
    genders = np.random.choice(['M', 'F'], size=n_samples)

    # Create DataFrame
    metadata_df = pd.DataFrame({
        'sample_id': base_sample_ids,
        'diagnosis': labels,
        'age': ages,
        'gender': genders
    })

    # Make sample IDs inconsistent to match expression data
    for idx in random.sample(range(len(metadata_df)), 20):
        original_id = metadata_df.loc[idx, 'sample_id']
        format_choice = random.choice(['dash', 'caps', 'patient_alt'])
        if format_choice == 'dash':
            metadata_df.loc[idx, 'sample_id'] = original_id.replace('_', '-')
        elif format_choice == 'caps':
            metadata_df.loc[idx, 'sample_id'] = original_id.upper()
        elif format_choice == 'patient_alt':
            num = original_id.split('_')[1]
            metadata_df.loc[idx, 'sample_id'] = f"patient-{int(num)}"

    return metadata_df

def main():
    """Generate and save messy datasets."""

    print("Generating synthetic gene expression dataset...")
    print("=" * 60)

    # Generate data
    expr_df, base_sample_ids, n_normal = generate_gene_expression_data(
        n_samples=100,
        n_genes=50
    )

    metadata_df = generate_metadata(base_sample_ids, n_normal)

    # Save to CSV
    expr_df.to_csv('gene_expression_data.csv', index=False)
    metadata_df.to_csv('sample_metadata.csv', index=False)

    print(f"✓ Generated gene_expression_data.csv")
    print(f"  - Shape: {expr_df.shape}")
    print(f"  - Missing values: {expr_df.isna().sum().sum()} ({100*expr_df.isna().sum().sum()/(expr_df.shape[0]*expr_df.shape[1]):.1f}%)")
    print(f"  - Duplicate rows: {expr_df.duplicated().sum()}")
    print()

    print(f"✓ Generated sample_metadata.csv")
    print(f"  - Shape: {metadata_df.shape}")
    print(f"  - Unique diagnoses: {metadata_df['diagnosis'].unique()}")
    print(f"  - Class distribution:")
    print(f"    {metadata_df['diagnosis'].value_counts().to_dict()}")
    print()

    print("=" * 60)
    print("Data generation complete! Ready for ML exercise.")
    print()
    print("Intentional issues included:")
    print("  ✗ ~7% missing values in expression data")
    print("  ✗ 4 duplicate sample rows")
    print("  ✗ Extreme outliers in 3 genes")
    print("  ✗ Inconsistent sample ID formatting")
    print("  ✗ Different gene expression scales (0.01 to 10,000)")
    print("  ✗ Inconsistent diagnosis labels (cancer/tumor/malignant/Cancer)")
    print("  ✗ Inconsistent normal labels (normal/healthy/control/Normal)")
    print("  ✗ Ambiguous labels (borderline/unclear/suspicious)")
    print()

if __name__ == "__main__":
    main()
