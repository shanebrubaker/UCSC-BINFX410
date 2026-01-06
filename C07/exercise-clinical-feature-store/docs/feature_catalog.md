# Feature Catalog

Complete documentation of all features in the clinical feature store.

## Table of Contents

1. [Imputation Features](#imputation-features)
2. [Encoding Features](#encoding-features)
3. [Scaling Features](#scaling-features)
4. [Derived Features](#derived-features)
5. [Binning Features](#binning-features)
6. [Feature Dependencies](#feature-dependencies)

---

## Imputation Features

Features that handle missing laboratory values using statistical imputation.

### wbc_imputed

**Category**: Lab Values
**Type**: Continuous
**Source**: wbc_count
**Version**: 1.0

**Description**:
White blood cell count with missing values imputed using median strategy.

**Medical Rationale**:
WBC count is frequently missing in clinical datasets (~10-15%) because tests aren't ordered for all patients or sample quality issues prevent measurement. Median imputation is preferred over mean because:
- Lab value distributions are often skewed
- Median is robust to outliers (e.g., very high WBC in leukemia)
- Preserves realistic value ranges

**Calculation**:
```python
# Training: fit imputer on non-missing values
median_wbc = wbc_count[~wbc_count.isna()].median()

# Inference: apply learned median
wbc_imputed = wbc_count.fillna(median_wbc)
```

**Expected Range**: 0.5 - 50 K/uL
**Normal Range**: 4 - 11 K/uL

**Use Cases**:
- Predictive models requiring complete data
- Risk stratification (low WBC = infection risk)
- Treatment eligibility screening

**Quality Checks**:
- No missing values in output
- Values within valid range (0.5-50)
- Distribution similar to non-missing subset

---

### hemoglobin_imputed

**Category**: Lab Values
**Type**: Continuous
**Source**: hemoglobin
**Version**: 1.0

**Description**:
Hemoglobin level with missing values imputed using median strategy.

**Medical Rationale**:
Hemoglobin measures oxygen-carrying capacity and is often low (anemia) in cancer patients. Missing data (~10-15%) occurs for similar reasons as WBC. Anemia is:
- Common in cancer patients (poor nutrition, bone marrow involvement)
- Prognostic indicator (associated with worse outcomes)
- Treatment decision factor (may require transfusion)

**Calculation**:
```python
median_hgb = hemoglobin[~hemoglobin.isna()].median()
hemoglobin_imputed = hemoglobin.fillna(median_hgb)
```

**Expected Range**: 4 - 20 g/dL
**Normal Range**: 12 - 17 g/dL (varies by sex)

**Use Cases**:
- Survival prediction models
- Treatment tolerance assessment
- Supportive care planning

**Quality Checks**:
- No missing values in output
- Values within valid range (4-20)
- Mean within expected range (10-13 for cancer cohort)

---

### platelet_imputed

**Category**: Lab Values
**Type**: Continuous
**Source**: platelet_count
**Version**: 1.0

**Description**:
Platelet count with missing values imputed using median strategy.

**Medical Rationale**:
Platelets are essential for blood clotting. Low platelet count (thrombocytopenia):
- Increases bleeding risk
- May be caused by chemotherapy (bone marrow suppression)
- Can delay or prevent treatment
- Requires monitoring and possible transfusion

**Calculation**:
```python
median_plt = platelet_count[~platelet_count.isna()].median()
platelet_imputed = platelet_count.fillna(median_plt)
```

**Expected Range**: 10 - 800 K/uL
**Normal Range**: 150 - 400 K/uL

**Use Cases**:
- Chemotherapy safety assessment
- Bleeding risk prediction
- Treatment modification decisions

**Quality Checks**:
- No missing values in output
- Values within valid range (10-800)
- Distribution consistent with chemotherapy-exposed population

---

## Encoding Features

Features that convert categorical variables to numeric representations for ML algorithms.

### sex_encoded

**Category**: Demographics
**Type**: Categorical (One-Hot)
**Source**: sex
**Version**: 1.0

**Description**:
One-hot encoded sex variable with normalization of inconsistent formatting.

**Medical Rationale**:
Sex is an important predictor in oncology because:
- Cancer incidence varies by sex (e.g., breast, prostate)
- Treatment response may differ
- Survival outcomes can vary
- Hormone-related factors affect some cancers

**Calculation**:
```python
# Step 1: Normalize inconsistent values
sex_normalized = sex.map({
    'Male': 'Male', 'M': 'Male', 'male': 'Male',
    'Female': 'Female', 'F': 'Female', 'female': 'Female'
})

# Step 2: One-hot encode (drop first to avoid multicollinearity)
sex_encoded_Female = (sex_normalized == 'Female').astype(int)
# Reference category: Male (encoded as 0)
```

**Values**:
- sex_encoded_Female: 1 if Female, 0 if Male

**Use Cases**:
- All clinical prediction models
- Risk stratification
- Treatment recommendation systems

**Quality Checks**:
- All values are 0 or 1
- No missing values after encoding
- Distribution matches original data

---

### ethnicity_encoded

**Category**: Demographics
**Type**: Categorical (One-Hot)
**Source**: ethnicity
**Version**: 1.0

**Description**:
One-hot encoded ethnicity variable.

**Medical Rationale**:
Ethnicity is relevant in precision oncology because:
- Cancer incidence varies by ethnicity
- Genetic variants differ across populations
- Treatment response may vary (pharmacogenomics)
- Health disparities affect outcomes

**Calculation**:
```python
# One-hot encode with drop_first=True
ethnicity_dummies = pd.get_dummies(ethnicity, drop_first=True)
# Creates: ethnicity_encoded_Asian, ethnicity_encoded_Hispanic, etc.
# Reference: First category (alphabetically)
```

**Values**:
- ethnicity_encoded_African_American: 1 if African American, 0 otherwise
- ethnicity_encoded_Asian: 1 if Asian, 0 otherwise
- ethnicity_encoded_Hispanic: 1 if Hispanic, 0 otherwise
- ethnicity_encoded_Other: 1 if Other, 0 otherwise
- Reference: Caucasian (all zeros)

**Use Cases**:
- Health disparity research
- Personalized treatment models
- Population health studies

**Quality Checks**:
- All values are 0 or 1
- Exactly one category is 1 per patient (before drop_first)
- No missing values

---

## Scaling Features

Features standardized to have mean ≈ 0 and standard deviation ≈ 1.

### age_scaled

**Category**: Demographics
**Type**: Continuous
**Source**: age
**Version**: 1.0

**Description**:
Age standardized using z-score normalization.

**Medical Rationale**:
Age is one of the strongest predictors in oncology:
- Cancer incidence increases with age
- Treatment tolerance decreases with age
- Comorbidity burden correlates with age
- Survival outcomes worsen with age

Standardization is important because:
- Makes age comparable in magnitude to other features
- Improves convergence in gradient-based algorithms
- Enables proper regularization (L1/L2)

**Calculation**:
```python
# Training: compute statistics
mean_age = age.mean()
std_age = age.std()

# Transform: apply z-score
age_scaled = (age - mean_age) / std_age
```

**Expected Statistics**:
- Mean: ≈ 0
- Std: ≈ 1
- Range: Approximately -3 to +3 (covers 99.7% of data)

**Use Cases**:
- All predictive models (universal feature)
- Risk stratification
- Treatment eligibility assessment

**Quality Checks**:
- Mean within [-0.1, 0.1]
- Std within [0.9, 1.1]
- No extreme outliers (|z| > 4)

---

### tmb_score_scaled

**Category**: Genomic
**Type**: Continuous
**Source**: tmb_score
**Version**: 1.0

**Description**:
Tumor mutational burden (TMB) score standardized using z-score normalization.

**Medical Rationale**:
TMB measures the number of mutations per megabase in tumor DNA:
- High TMB predicts immunotherapy response (more neoantigens)
- Used as biomarker for checkpoint inhibitor therapy
- Varies widely across cancer types
- Important for precision medicine decisions

Standardization allows TMB to be combined with other features in models.

**Calculation**:
```python
mean_tmb = tmb_score.mean()
std_tmb = tmb_score.std()
tmb_score_scaled = (tmb_score - mean_tmb) / std_tmb
```

**Expected Statistics**:
- Mean: ≈ 0
- Std: ≈ 1
- Original range: 0-100 mutations/Mb

**Use Cases**:
- Immunotherapy response prediction
- Treatment selection models
- Clinical trial enrollment criteria

**Quality Checks**:
- Mean within [-0.1, 0.1]
- Std within [0.9, 1.1]
- All values from valid TMB range (0-100 before scaling)

---

## Derived Features

New features created by combining or transforming existing features.

### mutation_burden

**Category**: Genomic
**Type**: Discrete (0-3)
**Source**: tp53_mutation, kras_mutation, egfr_mutation
**Version**: 1.0

**Description**:
Count of key oncogenic driver mutations present in the tumor.

**Medical Rationale**:
Driver mutations are genetic changes that promote cancer growth:
- TP53: "Guardian of the genome", most common cancer mutation (~50%)
- KRAS: Drives cell proliferation, common in lung/colon/pancreatic (~25%)
- EGFR: Receptor tyrosine kinase, targetable in lung cancer (~15%)

The total count may predict:
- Tumor aggressiveness
- Treatment options (targeted therapies)
- Prognosis

**Calculation**:
```python
mutation_burden = (
    tp53_mutation.astype(int) +
    kras_mutation.astype(int) +
    egfr_mutation.astype(int)
)
```

**Possible Values**: 0, 1, 2, 3

**Distribution** (typical):
- 0 mutations: ~25%
- 1 mutation: ~45%
- 2 mutations: ~25%
- 3 mutations: ~5%

**Use Cases**:
- Targeted therapy selection
- Prognosis prediction
- Clinical trial matching

**Quality Checks**:
- Values in range [0, 3]
- Integer values only
- No missing values

---

### clinical_risk_score

**Category**: Clinical
**Type**: Continuous
**Source**: age, comorbidity_count
**Version**: 1.0

**Description**:
Composite risk score combining age and comorbidity burden.

**Medical Rationale**:
Age and comorbidities together predict:
- Treatment tolerance (can patient handle chemotherapy?)
- Perioperative risk (for surgical candidates)
- Overall survival (independent of cancer)
- Quality of life impacts

The formula weights:
- Age: 1 point per decade (age/10)
- Comorbidities: 5 points each (significant impact)

**Calculation**:
```python
clinical_risk_score = (age / 10) + (comorbidity_count * 5)
```

**Score Interpretation**:
- 0-10: Low risk (young, healthy)
- 10-20: Moderate risk (elderly or comorbidities)
- >20: High risk (elderly with comorbidities)

**Expected Range**: 0-100 (typical: 5-35)

**Use Cases**:
- Treatment intensity decisions
- Surgical candidacy assessment
- Palliative vs curative intent
- Risk-adjusted survival models

**Quality Checks**:
- Values in range [0, 100]
- Mean around 15-20 for typical cancer cohort
- Correlates with age and comorbidity_count

---

### high_risk_patient

**Category**: Clinical
**Type**: Binary
**Source**: age, comorbidity_count, tmb_score
**Version**: 1.0

**Description**:
Binary flag identifying patients meeting multiple high-risk criteria.

**Medical Rationale**:
Identifies patients who may need:
- More aggressive monitoring
- Modified treatment approaches
- Supportive care services
- Clinical trial enrollment

**Calculation**:
```python
# Define risk criteria
age_risk = age > 70  # Elderly
comorbidity_risk = comorbidity_count >= 3  # Multiple comorbidities
tmb_risk = tmb_score > 20  # High mutation burden

# Count risk factors
risk_count = age_risk + comorbidity_risk + tmb_risk

# Flag if 2+ risk factors present
high_risk_patient = (risk_count >= 2).astype(int)
```

**Values**:
- 0: Not high-risk (<2 risk factors)
- 1: High-risk (≥2 risk factors)

**Expected Prevalence**: 15-25% of cohort

**Use Cases**:
- Triage for intensive case management
- Clinical trial enrollment
- Treatment modification triggers
- Resource allocation

**Quality Checks**:
- Values are 0 or 1
- No missing values
- Prevalence between 10-30%

---

## Binning Features

Features that discretize continuous variables into clinically meaningful categories.

### age_group

**Category**: Demographics
**Type**: Categorical
**Source**: age
**Version**: 1.0

**Description**:
Age binned into clinically relevant groups.

**Medical Rationale**:
Age groups are used in oncology because:
- Treatment protocols may differ by age group
- Some effects are non-linear (e.g., very old vs old)
- Clinical guidelines use age cutoffs
- Easier interpretation than continuous age

**Calculation**:
```python
bins = [0, 40, 60, 75, 120]
labels = ['18-40', '41-60', '61-75', '76+']
age_group = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
```

**Categories**:
- **18-40**: Young adults (different tumor biology, better tolerance)
- **41-60**: Middle age (peak working years, moderate risk)
- **61-75**: Older adults (typical cancer age, elevated risk)
- **76+**: Elderly (treatment tolerance concerns, high risk)

**Use Cases**:
- Stratified analysis by age
- Treatment protocol selection
- Risk group assignment
- Population health reporting

**Quality Checks**:
- All ages assigned to a category
- No missing values
- Categories mutually exclusive

---

### wbc_category

**Category**: Lab Values
**Type**: Categorical
**Source**: wbc_count (or wbc_imputed)
**Source**: wbc_count
**Version**: 1.0

**Description**:
White blood cell count categorized into clinical ranges.

**Medical Rationale**:
WBC categories guide clinical decisions:
- **Low (<4)**: Leukopenia
  - Infection risk
  - May require growth factors
  - Chemotherapy dose reduction

- **Normal (4-11)**: Healthy immune function
  - Standard treatment approaches
  - Normal infection risk

- **High (>11)**: Leukocytosis
  - Possible infection
  - Hematologic malignancy
  - Inflammatory response

**Calculation**:
```python
def categorize_wbc(value):
    if pd.isna(value):
        return 'Unknown'
    elif value < 4:
        return 'Low'
    elif value <= 11:
        return 'Normal'
    else:
        return 'High'

wbc_category = wbc_count.apply(categorize_wbc)
```

**Categories**:
- **Low**: <4 K/uL (leukopenia)
- **Normal**: 4-11 K/uL
- **High**: >11 K/uL (leukocytosis)
- **Unknown**: Missing value (if using raw wbc_count)

**Use Cases**:
- Infection risk prediction
- Treatment modification decisions
- Chemotherapy safety screening
- Growth factor administration

**Quality Checks**:
- All values in valid categories
- Distribution matches clinical expectations
- No "Unknown" if using imputed values

---

## Feature Dependencies

### Dependency Graph

```
Raw Data
├── age
│   ├── age_scaled
│   ├── age_group
│   └── clinical_risk_score (+ comorbidity_count)
│       └── high_risk_patient (+ tmb_score)
│
├── sex
│   └── sex_encoded
│
├── ethnicity
│   └── ethnicity_encoded
│
├── comorbidity_count
│   ├── clinical_risk_score (+ age)
│   └── high_risk_patient (+ age, tmb_score)
│
├── tmb_score
│   ├── tmb_score_scaled
│   └── high_risk_patient (+ age, comorbidity_count)
│
├── tp53_mutation
│   └── mutation_burden (+ kras_mutation, egfr_mutation)
│
├── kras_mutation
│   └── mutation_burden (+ tp53_mutation, egfr_mutation)
│
├── egfr_mutation
│   └── mutation_burden (+ tp53_mutation, kras_mutation)
│
├── wbc_count
│   ├── wbc_imputed
│   └── wbc_category
│
├── hemoglobin
│   └── hemoglobin_imputed
│
└── platelet_count
    └── platelet_imputed
```

### Feature Computation Order

Features must be computed in this order due to dependencies:

1. **Imputation**: wbc_imputed, hemoglobin_imputed, platelet_imputed
2. **Encoding & Scaling**: sex_encoded, ethnicity_encoded, age_scaled, tmb_score_scaled
3. **Derived**: mutation_burden, clinical_risk_score
4. **High-Level Derived**: high_risk_patient
5. **Binning**: age_group, wbc_category

### Reproducibility Requirements

To reproduce features exactly:
- Save imputer statistics (median values)
- Save scaler statistics (mean, std)
- Save encoder mappings (category values)
- Version all transformation code
- Track feature computation timestamps

---

## Feature Quality Metrics

### Validation Thresholds

| Feature | Missing Rate | Distribution Check | Value Range |
|---------|--------------|-------------------|-------------|
| age_scaled | 0% | mean ∈ [-0.5, 0.5], std ∈ [0.8, 1.2] | Any |
| mutation_burden | 0% | - | [0, 3] |
| clinical_risk_score | 0% | - | [0, 100] |
| wbc_imputed | 0% | - | [0.5, 50] |
| All features | <20% | - | Type-specific |

### Monitoring Metrics

Track over time:
- Missing data rates
- Distribution statistics (mean, std, quantiles)
- Categorical value frequencies
- Feature correlations
- Outlier counts

---

## Version History

### Version 1.0 (Current)
- Initial feature set
- 12 features across 5 categories
- Supports treatment response prediction

### Planned Version 2.0
- Add temporal features (time since diagnosis)
- Add interaction features (mutation × treatment)
- Add more sophisticated risk scores
- Point-in-time correctness for temporal joins

---

## Usage Examples

### Get Feature Metadata

```python
from src.feature_store import FeatureStore

fs = FeatureStore()
metadata = fs.get_feature_metadata('mutation_burden')
print(f"Source columns: {metadata['source_columns']}")
print(f"Description: {metadata['description']}")
```

### Get Feature Lineage

```python
lineage = fs.get_feature_lineage('clinical_risk_score')
print(f"Depends on: {lineage['source_columns']}")
```

### Create Training Dataset

```python
features = [
    'age_scaled',
    'mutation_burden',
    'clinical_risk_score',
    'tmb_score_scaled'
]

training_data = fs.create_training_dataset(
    feature_list=features,
    target='response_status'
)
```

---

## References

- Feature engineering patterns from clinical ML best practices
- TMB scoring from Foundation Medicine TMB assay
- Risk scoring inspired by ECOG performance status
- Age groups aligned with NCCN guidelines
- Lab value ranges from standard clinical chemistry references
