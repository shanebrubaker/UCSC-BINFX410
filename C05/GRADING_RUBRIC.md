# Machine Learning Fundamentals Exercise - Grading Rubric

**Course**: BINFX410 - Machine Learning for Bioinformatics
**Assignment**: Data Cleaning, Labeling, and Normalization
**Total Points**: 100

---

## Overview

This rubric evaluates student competency in:
- Data quality assessment and cleaning
- Label standardization and handling ambiguity
- Feature scaling and normalization techniques
- Proper ML workflow (avoiding data leakage)
- Code quality and documentation
- Critical thinking and interpretation

---

## Part 1: Data Cleaning (25 points)

### Task 1.1: Visualize Missing Data (3 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Heatmap created** | 2 | Successfully creates seaborn heatmap showing missing values |
| **Interpretation** | 1 | Provides thoughtful answer to "Think About It" questions |

**Deductions:**
- Missing heatmap: -2 points
- Incorrect visualization (wrong data shown): -1 point

---

### Task 1.2: Handle Missing Values (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Implementation** | 3 | Correctly implements median imputation using SimpleImputer or equivalent |
| **Verification** | 2 | Assertion passes (no NaN values remain) |
| **Documentation** | 1 | Comments explain approach and reasoning |

**Common Issues:**
- Used mean instead of median: Full credit if justified
- Dropped rows: -1 point (reduces dataset size unnecessarily)
- Manual loop instead of vectorized: No deduction if correct
- Still has NaN values: -5 points (breaks downstream analysis)

**Acceptable Approaches:**
- `SimpleImputer(strategy='median')`
- `fillna(df.median())`
- `df.apply(lambda x: x.fillna(x.median()))`

---

### Task 1.3: Detect and Remove Duplicates (5 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Detection** | 2 | Correctly identifies duplicate samples |
| **Removal** | 2 | Properly removes duplicates keeping first occurrence |
| **Verification** | 1 | Assertion passes (no duplicates remain) |

**Deductions:**
- Removes all duplicates including original: -2 points
- Checks wrong columns (includes sample_id): -1 point
- Doesn't verify removal: -1 point

---

### Task 1.4: Identify Outliers (3 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Visualization** | 2 | Creates boxplots for genes |
| **Discussion** | 1 | Thoughtful response about handling outliers vs normalization |

**Note:** No points deducted for not removing outliers (by design)

---

### Task 1.5: Standardize Sample IDs (8 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Function implementation** | 4 | Correctly standardizes various ID formats to SAMPLE_XXX |
| **Handles edge cases** | 2 | Works with different input formats (dash, underscore, etc.) |
| **Verification** | 2 | Achieves ~100 matching IDs between datasets |

**Deductions:**
- Function doesn't handle multiple formats: -2 points
- Hardcoded solution (not generalizable): -2 points
- Final match count < 95: -1 to -2 points depending on severity

**Test Cases:**
- "patient-1" → "SAMPLE_001" ✓
- "Patient_01" → "SAMPLE_001" ✓
- "SAMPLE-002" → "SAMPLE_002" ✓
- "sample_005" → "SAMPLE_005" ✓

---

## Part 2: Data Labeling (20 points)

### Task 2.1: Inspect Label Inconsistencies (2 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Execution** | 1 | Displays unique diagnosis values |
| **Observation** | 1 | Notes inconsistencies in comments or discussion |

---

### Task 2.2: Map Labels to Binary (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Mapping logic** | 4 | Correctly maps all label variants to 0/1/None |
| **Handles ambiguous** | 1 | Returns None for borderline/unclear/suspicious |
| **Verification** | 1 | Label distribution shows proper mapping |

**Required Mappings:**
- Normal class (0): normal, healthy, control (case-insensitive)
- Cancer class (1): cancer, tumor, malignant (case-insensitive)
- Ambiguous (None): borderline, unclear, suspicious

**Deductions:**
- Missing label variants: -1 point per variant
- Incorrect handling of ambiguous: -2 points
- Case-sensitive implementation: -1 point

---

### Task 2.3: Handle Ambiguous Labels (3 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Removal** | 2 | Drops rows with ambiguous labels |
| **Verification** | 1 | No NaN values remain in label column |

**Acceptable:** Discussion of alternative approaches (bonus: +1)

---

### Task 2.4: Merge Expression Data with Labels (3 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Merge execution** | 2 | Inner join on sample_id |
| **Result validation** | 1 | Shape is reasonable (~97 samples with all features) |

**Deductions:**
- Wrong join type (outer, left, right): -1 point if creates NaN
- Merge on wrong column: -2 points

---

### Task 2.5: Visualize Class Distribution (2 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Bar plot** | 1 | Creates clear visualization of class counts |
| **Calculation** | 1 | Correctly calculates percentages |

---

### Task 2.6: Create Train/Validation/Test Splits (4 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Split ratios** | 2 | Achieves approximately 60/20/20 split |
| **Stratification** | 2 | Uses stratify parameter, maintains class balance |

**Deductions:**
- Wrong split ratios: -1 point
- No stratification: -2 points (serious issue)
- Doesn't set random_state: -0.5 points (reproducibility)

**Verification:** Class distribution should be similar across all splits (within 5%)

---

## Part 3: Normalization (30 points)

### Task 3.1: Baseline Model (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Model training** | 2 | Trains LogisticRegression without normalization |
| **Evaluation** | 2 | Calculates accuracy and confusion matrix |
| **Visualization** | 2 | Plots model coefficients |

**Note:** Model performance itself is not graded, only correct implementation

---

### Task 3.2: Min-Max Scaling (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Proper fit/transform** | 3 | Fits on training only, transforms all sets |
| **Model training** | 2 | Trains on scaled data |
| **Comparison** | 1 | Compares to baseline |

**Critical:** Fit on training data ONLY
- Fits on all data: -3 points (data leakage)
- Fits on val/test: -3 points (data leakage)

---

### Task 3.3: Standardization (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Proper fit/transform** | 3 | Fits on training only, transforms all sets |
| **Model training** | 2 | Trains on scaled data |
| **Comparison** | 1 | Compares to baseline |

**Same data leakage rules as 3.2**

---

### Task 3.4: Log Transformation (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Log transform** | 2 | Uses np.log1p (handles zeros) |
| **Standardization** | 2 | Applies StandardScaler after log |
| **Evaluation** | 2 | Trains model and evaluates |

**Deductions:**
- Uses np.log instead of np.log1p: -1 point (potential errors)
- Doesn't standardize after log: -1 point

---

### Task 3.5: Compare Normalization Methods (6 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Comparison table** | 3 | Creates DataFrame comparing all methods |
| **Visualization** | 2 | Bar plot showing accuracy differences |
| **Interpretation** | 1 | Thoughtful discussion of results |

**Look For:**
- Student identifies which method performed best
- Explains why log transform might be appropriate for gene expression
- Notes improvement magnitude

---

## Part 4: Pipeline Integration (15 points)

### Task 4.1: Build Pipeline (5 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Pipeline creation** | 3 | Creates Pipeline with scaler and classifier |
| **Execution** | 2 | Fits and evaluates pipeline correctly |

**Required Components:**
- StandardScaler (or other scaler)
- LogisticRegression (or other classifier)

---

### Task 4.2: Demonstrate Data Leakage (5 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Executes leaky code** | 2 | Runs provided code (or creates own example) |
| **Understanding** | 3 | Explains why leakage is problematic in comments/discussion |

**Look For:**
- Student recognizes why fitting on all data is wrong
- Understands impact on performance estimates
- Notes this won't generalize to new data

---

### Task 4.3: Final Test Evaluation (5 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Model selection** | 2 | Chooses best model from validation results |
| **Test evaluation** | 2 | Evaluates on held-out test set |
| **Interpretation** | 1 | Compares test vs validation performance |

**Note:** Don't penalize if test performance is worse than validation (that's realistic!)

---

## Code Quality (10 points)

| Criteria | Points | Description |
|----------|--------|-------------|
| **Runs without errors** | 3 | Notebook executes from top to bottom |
| **Code readability** | 2 | Clear variable names, proper formatting |
| **Comments** | 2 | Explains non-obvious code sections |
| **No warnings** | 1 | Addresses RuntimeWarnings (NaN, etc.) |
| **Efficiency** | 2 | Uses vectorized operations where appropriate |

**Deductions:**
- Cells out of order: -1 point
- Multiple errors requiring fixes: -2 points
- Hardcoded values that should be variables: -1 point
- Inefficient loops where vectorization possible: -1 point

---

## Summary and Reflection (10 points)

### Required Reflections (6 points)

| Question | Points | Description |
|----------|--------|-------------|
| **Biggest impact step** | 2 | Identifies and justifies which preprocessing step mattered most |
| **Surprises** | 1 | Thoughtful discussion of unexpected findings |
| **Real-world application** | 2 | Discusses how to apply in research context |
| **Additional checks** | 1 | Suggests meaningful quality checks beyond exercise |

**Grading:**
- Superficial answers (1-2 sentences): 50% credit
- Thoughtful answers with justification: Full credit
- Exceptional insight: Bonus +1 point

---

### Bonus Challenges (up to +5 extra credit)

| Challenge | Points | Description |
|----------|--------|-------------|
| **Different ML models** | +2 | Tries RandomForest, SVM, etc. with comparison |
| **Cross-validation** | +2 | Implements proper k-fold CV |
| **Feature selection** | +1 | Identifies most important genes |
| **Dimensionality reduction** | +2 | Applies PCA or t-SNE |
| **Class imbalance handling** | +2 | Implements SMOTE or class weights |
| **Error analysis** | +1 | Analyzes misclassified samples |

**Note:** Maximum +5 bonus points total (not cumulative of all challenges)

---

## Common Mistakes and Deductions

### Critical Errors (Major Deductions)

| Error | Deduction | Impact |
|-------|-----------|--------|
| **Data leakage** (fitting scaler on all data) | -10 points | Fundamentally breaks ML workflow |
| **No imputation** (NaN values remain) | -10 points | Breaks all downstream analysis |
| **No stratification** in splits | -5 points | Biased evaluation |
| **Wrong label mapping** | -8 points | Invalidates all results |

### Moderate Errors

| Error | Deduction | Impact |
|-------|-----------|--------|
| Missing imports | -2 points | Code doesn't run |
| Incorrect split ratios | -2 points | Suboptimal evaluation |
| No documentation | -5 points | Poor scientific practice |
| Inefficient implementation | -2 points | Works but suboptimal |

### Minor Issues

| Error | Deduction | Impact |
|-------|-----------|--------|
| Typos in comments | -0.5 points | Minor professionalism issue |
| Inconsistent variable naming | -1 point | Readability |
| Missing plot labels | -1 point | Poor visualization practice |
| No random seed | -0.5 points | Not reproducible |

---

## Grade Distribution Guidelines

| Grade Range | Criteria |
|-------------|----------|
| **A (90-100)** | All tasks completed correctly, good code quality, thoughtful reflections |
| **B (80-89)** | Most tasks correct, minor errors in implementation or understanding |
| **C (70-79)** | Major tasks completed but with errors, some misunderstanding of concepts |
| **D (60-69)** | Incomplete work, significant conceptual errors, poor code quality |
| **F (<60)** | Critical failures (data leakage, no imputation), incomplete work |

---

## Grading Workflow for Instructors

### Step 1: Quick Validation (5 minutes)
1. Run notebook top-to-bottom
2. Check for errors
3. Verify final shapes and outputs

### Step 2: Detailed Review (15 minutes)
1. Review Part 1 (Data Cleaning) - verify imputation, deduplication, ID standardization
2. Review Part 2 (Labeling) - check mapping function and split implementation
3. Review Part 3 (Normalization) - **Critical:** verify no data leakage
4. Review Part 4 (Pipelines) - check understanding of leakage

### Step 3: Code Quality (5 minutes)
1. Check comments and documentation
2. Verify code readability
3. Look for inefficiencies

### Step 4: Reflections (5 minutes)
1. Read reflection answers
2. Assess depth of understanding
3. Award bonus points if applicable

**Total grading time per submission: ~30 minutes**

---

## Automated Testing (Optional)

For large classes, consider automated tests:

```python
# Example test: Check imputation
def test_imputation():
    assert expr_cleaned.isnull().sum().sum() == 0, "Missing values remain"

# Example test: Check label mapping
def test_label_mapping():
    labels = metadata_df['label'].unique()
    assert set(labels).issubset({0, 1}), "Labels not binary"

# Example test: Check data leakage
def test_no_leakage():
    # Verify scaler was fit on training only
    # This is harder to test automatically
    pass
```

---

## Feedback Templates

### Excellent Work (90-100)
```
Excellent work! Your implementation demonstrates strong understanding of:
- Data quality assessment and cleaning techniques
- Proper ML workflow (especially avoiding data leakage)
- Feature scaling and its impact on model performance

Your reflection shows critical thinking about real-world applications.

Suggestions for further learning: [specific to student's work]
```

### Good Work with Minor Issues (80-89)
```
Good work overall! Your implementation is mostly correct.

Areas to improve:
- [Specific issue 1]
- [Specific issue 2]

Your understanding of the core concepts is solid. Review [specific topic]
to strengthen your implementation.
```

### Needs Improvement (70-79)
```
Your submission shows effort, but there are significant issues:

Critical problems:
- [Issue 1 with explanation]
- [Issue 2 with explanation]

Please review the solution notebook and focus on:
- [Concept 1]
- [Concept 2]

Office hours recommended to discuss these concepts.
```

### Incomplete (< 70)
```
Your submission is incomplete or has critical errors that prevent proper evaluation.

Required corrections:
- [Critical issue 1]
- [Critical issue 2]

Please resubmit after addressing these issues. See me in office hours.
```

---

## Academic Integrity

### Acceptable Collaboration
- Discussing concepts and approaches
- Helping with debugging syntax errors
- Sharing resources and documentation links

### Not Acceptable
- Copying code from other students
- Sharing completed notebooks
- Using external solutions without attribution
- Having someone else write code for you

### Plagiarism Check
- Compare function names and variable naming patterns
- Check for identical comments (especially typos)
- Look for identical random_state values (if not specified in assignment)
- Review git commit history if using version control

---

## Appendix: Quick Reference Checklist

### Data Cleaning
- [ ] Missing value heatmap created
- [ ] Median imputation implemented (no NaN remain)
- [ ] Duplicates removed
- [ ] Outliers identified
- [ ] Sample IDs standardized (~100 matches)

### Data Labeling
- [ ] Label inconsistencies noted
- [ ] Binary mapping function correct
- [ ] Ambiguous labels handled
- [ ] Data merged correctly
- [ ] Class distribution visualized
- [ ] 60/20/20 split with stratification

### Normalization
- [ ] Baseline model trained
- [ ] Min-Max scaling (fit on train only!)
- [ ] Standardization (fit on train only!)
- [ ] Log transformation + standardization
- [ ] Comparison table and visualization

### Pipelines
- [ ] Pipeline created correctly
- [ ] Data leakage demonstrated
- [ ] Test set evaluation
- [ ] Understanding of leakage shown

### Quality
- [ ] Runs without errors
- [ ] Well-commented
- [ ] Good variable names
- [ ] Thoughtful reflections

---

**Version**: 1.0
**Last Updated**: December 2025
**Recommended Time**: 60 minutes for completion, 30 minutes for grading
