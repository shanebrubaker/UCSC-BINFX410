# Clinical Feature Store for ML in Bioinformatics

A lightweight, educational feature store system for clinical and genomic data that demonstrates MLOps best practices using free, local tools.

## Overview

This project implements a production-grade feature store tailored for precision medicine applications. It demonstrates key MLOps concepts including:

- **Data Validation**: Automated quality checks using Pandera
- **Feature Engineering**: Reproducible transformations with clear medical rationale
- **Feature Versioning**: Track feature evolution over time
- **Lineage Tracking**: Understand feature dependencies
- **Data Quality Monitoring**: Detect drift and quality issues
- **Local-First**: Runs entirely on your machine with DuckDB

## Project Structure

```
clinical-feature-store/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config/
│   ├── features.yaml          # Feature definitions and metadata
│   └── validation.yaml        # Data validation rules
├── data/
│   ├── raw/                   # Raw synthetic patient data
│   └── feature_store.duckdb   # DuckDB feature store (created on first run)
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic data generation
│   ├── data_validators.py     # Pandera validation schemas
│   ├── features.py            # Feature engineering functions
│   ├── feature_store.py       # Core FeatureStore class
│   ├── monitoring.py          # Data quality monitoring
│   └── utils.py               # Utility functions
├── notebooks/
│   ├── 01_generate_data.ipynb            # Generate synthetic dataset
│   ├── 02_feature_engineering.ipynb      # Feature store setup
│   ├── 03_model_training_demo.ipynb      # ML model training
│   └── 04_monitoring_report.ipynb        # Data quality monitoring
├── tests/
│   ├── test_data_generator.py
│   ├── test_data_validators.py
│   ├── test_features.py
│   └── test_feature_store.py
└── docs/
    └── feature_catalog.md     # Complete feature documentation
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# Run the data generator
python -c "
from src.data_generator import ClinicalDataGenerator
gen = ClinicalDataGenerator(seed=42)
data = gen.generate_dataset(n_patients=1000)
gen.save_dataset(data, 'data/raw/synthetic_patients.csv')
"
```

Or use the Jupyter notebook: `notebooks/01_generate_data.ipynb`

### 3. Set Up Feature Store

```python
from src.feature_store import FeatureStore
import pandas as pd

# Initialize feature store
fs = FeatureStore()

# Load and ingest data
data = pd.read_csv('data/raw/synthetic_patients.csv')
fs.ingest_raw_data(data, validate=True)

# Register features
fs.register_features_from_config()

# Compute features
features = fs.compute_features(feature_version=1)
```

### 4. Create Training Dataset

```python
# Define features for your model
feature_list = [
    'age_scaled',
    'mutation_burden',
    'clinical_risk_score',
    'tmb_score_scaled'
]

# Create training dataset
training_data = fs.create_training_dataset(
    feature_list=feature_list,
    target='response_status'
)

# Ready for ML!
```

## Tutorials

Follow the notebooks in order to learn the complete workflow:

1. **01_generate_data.ipynb**: Generate synthetic clinical/genomic data
2. **02_feature_engineering.ipynb**: Set up feature store and compute features
3. **03_model_training_demo.ipynb**: Train ML models using the feature store
4. **04_monitoring_report.ipynb**: Monitor data quality and detect drift

## Features

### Implemented Features

The feature store includes 12 engineered features across 5 categories:

**1. Imputation** (Handle missing lab values)
- `wbc_imputed`: White blood cell count with median imputation
- `hemoglobin_imputed`: Hemoglobin level with median imputation
- `platelet_imputed`: Platelet count with median imputation

**2. Encoding** (Categorical variables)
- `sex_encoded`: One-hot encoded sex
- `ethnicity_encoded`: One-hot encoded ethnicity

**3. Scaling** (Standardization)
- `age_scaled`: Age standardized (z-score)
- `tmb_score_scaled`: TMB score standardized

**4. Derived Features** (New features from combinations)
- `mutation_burden`: Count of driver mutations (0-3)
- `clinical_risk_score`: Age + comorbidity composite score
- `high_risk_patient`: Binary flag for high-risk patients

**5. Binning** (Discretization)
- `age_group`: Age categories (18-40, 41-60, 61-75, 76+)
- `wbc_category`: WBC categories (Low, Normal, High)

See `docs/feature_catalog.md` for complete documentation.

## Dataset

The synthetic dataset includes 1000 patients with:

**Demographics**
- Age (18-95)
- Sex (with intentional formatting inconsistencies)
- Ethnicity

**Clinical Features**
- Cancer diagnosis
- Comorbidity count
- Treatment response
- Survival time

**Genomic Features**
- TP53, KRAS, EGFR mutation status
- Tumor mutational burden (TMB) score
- Microsatellite instability (MSI) status

**Lab Values** (with 10-15% realistic missing data)
- WBC count
- Hemoglobin
- Platelet count

**Intentional Quality Issues** (for teaching validation)
- Invalid age values (~1%)
- Out-of-range lab values (~0.5%)
- Inconsistent categorical formatting

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_store.py -v
```

Expected: >80% test coverage

## Architecture

```
┌─────────────┐
│  Raw Data   │
│  (CSV)      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Validation     │◄── validation.yaml
│  (Pandera)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Feature Store  │◄── features.yaml
│  (DuckDB)       │
│                 │
│  ┌───────────┐  │
│  │ Registry  │  │  (Metadata)
│  ├───────────┤  │
│  │ Raw Data  │  │  (Source data)
│  ├───────────┤  │
│  │ Features  │  │  (Computed features)
│  └───────────┘  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Training       │
│  Dataset        │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  ML Model       │
│  Training       │
└─────────────────┘
```

## Key Concepts

### Why Use a Feature Store?

1. **Consistency**: Same features in training and production
2. **Reusability**: Define features once, use in multiple models
3. **Collaboration**: Shared feature repository for teams
4. **Lineage**: Track which raw data created which features
5. **Quality**: Automated validation ensures data integrity
6. **Versioning**: Reproduce exact feature sets used for any model

### Feature Engineering Pipeline

```python
Raw Data
  └─> Validation (catch errors early)
      └─> Imputation (handle missing values)
          └─> Encoding (categoricals to numeric)
              └─> Scaling (standardization)
                  └─> Derived Features (combinations)
                      └─> Binning (discretization)
                          └─> Feature Store
```

### Data Quality Monitoring

The system tracks:
- Missing data rates over time
- Distribution shifts (mean, std, quantiles)
- Validation failures
- Feature correlations
- Statistical anomalies

## Production Considerations

This is an educational implementation. For production use, consider:

### Scalability
- **Replace DuckDB** with distributed storage (e.g., Delta Lake, Iceberg)
- **Add caching** for frequently accessed features
- **Implement batch processing** for large datasets
- **Use Feast/Tecton** for enterprise feature stores

### Feature Serving
- **Add REST API** (FastAPI) for online serving
- **Implement caching** (Redis) for low-latency access
- **Point-in-time correctness** for temporal features
- **Feature versioning** in production

### Monitoring
- **Real-time alerting** (e.g., PagerDuty, Slack)
- **Drift detection** algorithms (KS test, PSI)
- **Model performance monitoring**
- **Integration with observability tools** (Prometheus, Grafana)

### Security
- **Data encryption** at rest and in transit
- **Access controls** for sensitive features
- **Audit logging** for compliance
- **PII/PHI handling** for healthcare data

## Learning Resources

### Key MLOps Concepts Demonstrated

- **Data Validation**: Pandera schemas prevent bad data from entering the pipeline
- **Feature Engineering**: Separation of concerns between feature creation and model training
- **Feature Store**: Centralized feature repository with versioning and lineage
- **Monitoring**: Proactive detection of data quality issues and drift
- **Testing**: Comprehensive test coverage ensures reliability

### Further Reading

- [Feast Feature Store](https://feast.dev/)
- [MLOps Principles](https://ml-ops.org/)
- [Data Validation in Production ML](https://www.tensorflow.org/tfx/guide/tfdv)
- [Feature Store for ML](https://www.featurestore.org/)

## Contributing

This is an educational project. Suggestions for improvements:

1. Add point-in-time correctness for temporal features
2. Implement feature serving API
3. Add A/B testing framework for features
4. Integrate with MLflow for experiment tracking
5. Add more sophisticated drift detection
6. Implement feature recommendations based on target correlation

## License

MIT License - Free for educational and commercial use

## Contact

For questions or feedback about this educational project, please open an issue on GitHub.

## Acknowledgments

- Synthetic data generation inspired by real-world precision oncology datasets
- Feature engineering patterns from clinical ML best practices
- MLOps principles from industry leaders (Uber, Airbnb, Netflix)
