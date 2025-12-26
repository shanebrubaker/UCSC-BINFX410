# Machine Learning Fundamentals: Data Cleaning, Labeling, and Normalization

## Overview

This hands-on exercise teaches essential data preprocessing skills for machine learning using a synthetic gene expression dataset for cancer vs. normal tissue classification. Students will learn to identify and fix real-world data quality issues before building predictive models.

## Learning Objectives

By completing this exercise, you will:

1. **Data Cleaning**
   - Identify and visualize missing data patterns
   - Implement strategies for handling missing values
   - Detect and remove duplicate samples
   - Identify and handle outliers appropriately
   - Standardize inconsistent identifiers across datasets

2. **Data Labeling**
   - Recognize and fix inconsistent label terminology
   - Map categorical labels to binary classification targets
   - Handle ambiguous or uncertain labels
   - Create stratified train/validation/test splits
   - Assess and address class imbalance

3. **Normalization & Feature Scaling**
   - Understand why normalization matters for ML models
   - Apply and compare multiple normalization techniques:
     - Min-Max scaling
     - Standardization (z-score)
     - Log transformation
   - Avoid data leakage when normalizing
   - Build proper sklearn pipelines

4. **Model Evaluation**
   - Compare model performance with and without preprocessing
   - Interpret the impact of data quality on predictions
   - Use appropriate metrics for binary classification

## Dataset Description

### Files Generated

- `gene_expression_data.csv`: Expression levels for 50 genes across 104 samples
- `sample_metadata.csv`: Diagnosis labels and patient demographics for 100 samples

### Intentional Data Issues

This dataset contains realistic data quality problems:

- **Missing values**: ~7% of expression measurements are missing
- **Duplicates**: Same samples measured multiple times
- **Outliers**: Extreme values in some gene measurements
- **Inconsistent IDs**: Sample identifiers don't match between files
- **Scale differences**: Gene expression ranges from 0.01 to 10,000
- **Inconsistent labels**: Multiple terms for same diagnosis (cancer/tumor/malignant)
- **Ambiguous cases**: Some samples have unclear diagnoses

These issues mirror real-world biological datasets and clinical data.

## Prerequisites

- Basic Python programming
- Familiarity with pandas and numpy
- Basic understanding of machine learning concepts
- Jupyter notebook environment

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Generate the messy dataset
python generate_messy_data.py
```

## Exercise Structure

### Part 1: Data Cleaning (Tasks 1.1-1.5)
- Visualize and handle missing data
- Detect and remove duplicates
- Identify and address outliers
- Standardize sample identifiers

### Part 2: Data Labeling (Tasks 2.1-2.6)
- Map inconsistent labels to binary targets
- Handle ambiguous labels
- Create stratified data splits
- Verify class balance

### Part 3: Normalization (Tasks 3.1-3.4)
- Train baseline model without normalization
- Apply Min-Max scaling
- Apply standardization (z-score)
- Apply log transformation
- Compare all approaches

### Part 4: Pipeline Integration (Tasks 4.1-4.4)
- Build sklearn Pipeline
- Demonstrate data leakage
- Implement proper preprocessing
- Final model comparison

## How to Use

1. **Start with the exercise notebook**: `ml_fundamentals_exercise.ipynb`
   - Contains instructions, scaffolding code, and TODO comments
   - Estimated completion time: 60 minutes
   - Includes discussion prompts and checkpoints

2. **Check your work**: Each section has validation cells
   - Compare your results to expected outputs
   - Ensure your code runs without errors

3. **Consult the solution**: `ml_fundamentals_solution.ipynb`
   - Reference if you get stuck
   - Compare your approach to the complete solution
   - Review explanations and best practices

## Key Concepts

### Why Data Cleaning Matters

- **Missing data** can bias models or cause errors
- **Duplicates** violate independence assumptions
- **Outliers** can dominate model training
- **Inconsistent labels** prevent proper learning

### Why Normalization Matters

- Many ML algorithms are sensitive to feature scale
- Features with large ranges dominate distance calculations
- Gradient descent converges faster with normalized features
- Model coefficients become interpretable

### Avoiding Data Leakage

- **Wrong**: Normalize entire dataset, then split
- **Right**: Split first, fit normalizer on training data only
- Pipelines help prevent this common mistake

## Expected Outcomes

Students should observe:

- **Cleaning improves data quality**: Fewer errors, clearer patterns
- **Normalization improves model performance**: Higher accuracy, more stable coefficients
- **Proper pipelines prevent leakage**: Realistic performance estimates
- **Different normalization methods** work better for different data distributions

## Extension Ideas

For advanced students:

1. Try different ML models (Random Forest, SVM, Neural Networks)
2. Implement additional normalization methods (RobustScaler, QuantileTransformer)
3. Use cross-validation instead of single train/test split
4. Feature selection: identify most important genes
5. Handle class imbalance with SMOTE or class weights
6. Create visualizations: PCA plots, t-SNE, heatmaps

## Common Pitfalls to Avoid

1. Normalizing before train/test split (data leakage)
2. Dropping too many rows with missing data
3. Not documenting cleaning decisions
4. Ignoring class imbalance
5. Using accuracy alone for imbalanced datasets
6. Fitting normalizers on test data

## Assessment Criteria

Students should demonstrate:

- Proper use of pandas for data manipulation
- Thoughtful decision-making about cleaning strategies
- Understanding of train/test split importance
- Correct implementation of normalization
- Ability to interpret model results
- Clear code documentation

## Additional Resources

- [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Missing Data Tutorial](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [ML Pipelines Documentation](https://scikit-learn.org/stable/modules/compose.html)

## Support

If you encounter issues:

1. Check that all requirements are installed
2. Verify datasets were generated correctly
3. Read error messages carefully
4. Review the solution notebook
5. Consult documentation links above

## License

This educational material is provided for teaching purposes. Feel free to adapt and modify for your courses.

## Acknowledgments

Dataset is synthetic but inspired by real gene expression studies in cancer biology, including The Cancer Genome Atlas (TCGA) project.
