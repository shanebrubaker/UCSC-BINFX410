"""
Tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import ClinicalFeatureEngineer, get_feature_lineage, get_feature_descriptions
from data_generator import ClinicalDataGenerator


class TestClinicalFeatureEngineer:
    """Test suite for ClinicalFeatureEngineer."""

    @pytest.fixture
    def engineer(self):
        """Create engineer instance."""
        return ClinicalFeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        generator = ClinicalDataGenerator(seed=42)
        return generator.generate_dataset(n_patients=100, introduce_errors=False)

    def test_impute_lab_values(self, engineer, sample_data):
        """Test lab value imputation."""
        df_imputed = engineer.impute_lab_values(sample_data, strategy='median', fit=True)

        # Should create imputed columns
        assert 'wbc_imputed' in df_imputed.columns
        assert 'hemoglobin_imputed' in df_imputed.columns
        assert 'platelet_imputed' in df_imputed.columns

        # Should have no missing values in imputed columns
        assert df_imputed['wbc_imputed'].isnull().sum() == 0
        assert df_imputed['hemoglobin_imputed'].isnull().sum() == 0
        assert df_imputed['platelet_imputed'].isnull().sum() == 0

    def test_encode_categorical(self, engineer, sample_data):
        """Test categorical encoding."""
        df_encoded = engineer.encode_categorical(sample_data, fit=True)

        # Should create encoded columns
        sex_encoded_cols = [col for col in df_encoded.columns if 'sex_encoded' in col]
        assert len(sex_encoded_cols) > 0

        # Normalized sex should be in standardized format
        assert 'sex' in df_encoded.columns
        unique_sex = df_encoded['sex'].unique()
        assert all(sex in ['Male', 'Female'] for sex in unique_sex)

    def test_scale_continuous(self, engineer, sample_data):
        """Test continuous variable scaling."""
        df_scaled = engineer.scale_continuous(sample_data, fit=True)

        # Should create scaled columns
        assert 'age_scaled' in df_scaled.columns
        assert 'tmb_score_scaled' in df_scaled.columns

        # Scaled values should be approximately mean=0, std=1
        assert abs(df_scaled['age_scaled'].mean()) < 0.1
        assert abs(df_scaled['age_scaled'].std() - 1.0) < 0.1

    def test_create_mutation_burden(self, engineer, sample_data):
        """Test mutation burden calculation."""
        df_burden = engineer.create_mutation_burden(sample_data)

        assert 'mutation_burden' in df_burden.columns

        # Should be between 0 and 3
        assert df_burden['mutation_burden'].min() >= 0
        assert df_burden['mutation_burden'].max() <= 3

        # Check calculation is correct
        for idx, row in sample_data.head(10).iterrows():
            expected = int(row['tp53_mutation']) + int(row['kras_mutation']) + int(row['egfr_mutation'])
            actual = df_burden.loc[idx, 'mutation_burden']
            assert actual == expected

    def test_create_clinical_risk_score(self, engineer, sample_data):
        """Test clinical risk score calculation."""
        df_risk = engineer.create_clinical_risk_score(sample_data)

        assert 'clinical_risk_score' in df_risk.columns

        # Check calculation for a few rows
        for idx, row in sample_data.head(10).iterrows():
            expected = (row['age'] / 10) + (row['comorbidity_count'] * 5)
            actual = df_risk.loc[idx, 'clinical_risk_score']
            assert abs(actual - expected) < 0.01

    def test_create_high_risk_flag(self, engineer, sample_data):
        """Test high risk patient flagging."""
        df_risk = engineer.create_high_risk_flag(sample_data)

        assert 'high_risk_patient' in df_risk.columns

        # Should be binary
        assert df_risk['high_risk_patient'].isin([0, 1]).all()

    def test_create_age_groups(self, engineer, sample_data):
        """Test age group binning."""
        df_bins = engineer.create_age_groups(sample_data)

        assert 'age_group' in df_bins.columns

        # Should be one of the defined categories
        valid_groups = ['18-40', '41-60', '61-75', '76+']
        assert all(group in valid_groups for group in df_bins['age_group'].dropna().unique())

    def test_create_wbc_categories(self, engineer, sample_data):
        """Test WBC categorization."""
        # First impute
        df = engineer.impute_lab_values(sample_data, strategy='median', fit=True)
        df_cat = engineer.create_wbc_categories(df)

        assert 'wbc_category' in df_cat.columns

        # Should be one of the defined categories
        valid_cats = ['Low', 'Normal', 'High', 'Unknown']
        assert all(cat in valid_cats for cat in df_cat['wbc_category'].unique())

    def test_transform_all(self, engineer, sample_data):
        """Test complete transformation pipeline."""
        df_transformed = engineer.transform_all(sample_data, fit=True)

        # Check that engineer is marked as fitted
        assert engineer.fitted

        # Check that all expected features are created
        expected_features = [
            'wbc_imputed', 'hemoglobin_imputed', 'platelet_imputed',
            'age_scaled', 'tmb_score_scaled',
            'mutation_burden', 'clinical_risk_score', 'high_risk_patient',
            'age_group', 'wbc_category'
        ]

        for feature in expected_features:
            assert feature in df_transformed.columns, f"Missing feature: {feature}"

    def test_transform_inference_mode(self, engineer, sample_data):
        """Test that inference mode works after fitting."""
        # Fit on first half
        train_data = sample_data.head(50)
        engineer.transform_all(train_data, fit=True)

        # Transform on second half without fitting
        test_data = sample_data.tail(50)
        df_transformed = engineer.transform_all(test_data, fit=False)

        # Should succeed and produce same features
        assert len(df_transformed) == 50

    def test_missing_column_raises_error(self, engineer):
        """Test that missing required columns raise errors."""
        df = pd.DataFrame({'patient_id': ['PT00001']})

        with pytest.raises(ValueError):
            engineer.create_mutation_burden(df)


class TestFeatureMetadata:
    """Test feature metadata functions."""

    def test_get_feature_lineage(self):
        """Test feature lineage retrieval."""
        lineage = get_feature_lineage()

        assert isinstance(lineage, dict)
        assert 'mutation_burden' in lineage
        assert lineage['mutation_burden'] == ['tp53_mutation', 'kras_mutation', 'egfr_mutation']

    def test_get_feature_descriptions(self):
        """Test feature descriptions retrieval."""
        descriptions = get_feature_descriptions()

        assert isinstance(descriptions, dict)
        assert 'mutation_burden' in descriptions
        assert isinstance(descriptions['mutation_burden'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
