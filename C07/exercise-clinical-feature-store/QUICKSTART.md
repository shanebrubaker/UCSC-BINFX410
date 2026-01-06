# Quick Start Guide

Get up and running with the Clinical Feature Store in 5 minutes!

## 1. Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Generate Synthetic Data

### Option A: Python Script

```bash
python -c "
from src.data_generator import ClinicalDataGenerator
gen = ClinicalDataGenerator(seed=42)
data = gen.generate_dataset(n_patients=1000)
gen.save_dataset(data, 'data/raw/synthetic_patients.csv')
"
```

### Option B: Jupyter Notebook

```bash
jupyter notebook notebooks/01_generate_data.ipynb
```

## 3. Set Up Feature Store

```python
from src.feature_store import FeatureStore
import pandas as pd

# Initialize
fs = FeatureStore()

# Load and validate data
data = pd.read_csv('data/raw/synthetic_patients.csv')

# Clean data (remove intentional errors)
clean_data = data[
    (data['age'] >= 0) & (data['age'] <= 120) &
    (data['tmb_score'] <= 100) &
    (data['comorbidity_count'] <= 10)
]

# Ingest
fs.ingest_raw_data(clean_data, validate=True)

# Register features
fs.register_features_from_config()

# Compute features
features = fs.compute_features(feature_version=1)

print(f"Feature store ready with {len(features)} patients!")
```

## 4. Create Training Dataset

```python
# Define features for your model
feature_list = [
    'age_scaled',
    'mutation_burden',
    'clinical_risk_score',
    'tmb_score_scaled',
    'wbc_imputed',
    'hemoglobin_imputed'
]

# Create training dataset
training_data = fs.create_training_dataset(
    feature_list=feature_list,
    target='response_status',
    include_metadata=False
)

print(f"Training dataset: {training_data.shape}")
```

## 5. Train a Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Split data
X = training_data[feature_list]
y = training_data['response_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## 6. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On Mac/Linux
# start htmlcov/index.html  # On Windows
```

## 7. Generate Data Quality Report

```python
from src.monitoring import DataQualityMonitor

# Initialize monitor
monitor = DataQualityMonitor(feature_store=fs)

# Generate report
report_path = monitor.generate_quality_report(
    df=clean_data,
    report_name="data_quality",
    output_dir="reports"
)

print(f"Report saved to: {report_path}")
# Open the HTML file in your browser to view
```

## Next Steps

1. **Explore the notebooks** in order:
   - `01_generate_data.ipynb` - Data generation and exploration
   - `02_feature_engineering.ipynb` - Feature store setup
   - `03_model_training_demo.ipynb` - ML model training
   - `04_monitoring_report.ipynb` - Data quality monitoring

2. **Read the documentation**:
   - `README.md` - Complete project overview
   - `docs/feature_catalog.md` - Detailed feature documentation

3. **Experiment**:
   - Try different feature combinations
   - Add your own features
   - Test different models
   - Implement feature serving API

## Common Issues

### Issue: DuckDB database locked

**Solution**: Close all feature store connections
```python
fs.close()
```

### Issue: Missing data directory

**Solution**: Create directories
```bash
mkdir -p data/raw reports
```

### Issue: Import errors

**Solution**: Install in development mode
```bash
pip install -e .
```

## Resources

- **Documentation**: See `README.md` and `docs/feature_catalog.md`
- **Tests**: See `tests/` for usage examples
- **Configuration**: See `config/` for feature and validation configs

## Getting Help

- Check the documentation in `docs/`
- Review the test files for usage examples
- Run the notebooks for interactive tutorials

Happy feature engineering! 🚀
