"""
Tests for data validators.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_validators import RawDataValidator, FeatureValidator, ValidationReport
from data_generator import ClinicalDataGenerator


class TestRawDataValidator:
    """Test suite for RawDataValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RawDataValidator()

    @pytest.fixture
    def clean_data(self):
        """Generate clean test data."""
        generator = ClinicalDataGenerator(seed=42)
        return generator.generate_dataset(n_patients=50, introduce_errors=False)

    @pytest.fixture
    def dirty_data(self):
        """Generate data with errors."""
        generator = ClinicalDataGenerator(seed=42)
        return generator.generate_dataset(n_patients=50, introduce_errors=True)

    def test_validate_clean_data(self, validator, clean_data):
        """Test validation of clean data passes."""
        report = validator.validate(clean_data)

        assert isinstance(report, ValidationReport)
        assert report.is_valid
        # May have warnings but should have no errors
        assert len(report.errors) == 0

    def test_validate_dirty_data(self, validator, dirty_data):
        """Test validation of dirty data fails."""
        report = validator.validate(dirty_data)

        # Should fail due to intentional errors
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_validate_missing_required_column(self, validator, clean_data):
        """Test validation fails when required column is missing."""
        df = clean_data.drop(columns=['age'])

        report = validator.validate(df)
        assert not report.is_valid

    def test_validate_invalid_age_range(self, validator, clean_data):
        """Test validation catches invalid age values."""
        df = clean_data.copy()
        df.loc[0, 'age'] = 150  # Invalid age

        report = validator.validate(df)
        assert not report.is_valid
        assert any('age' in str(error).lower() for error in report.errors)

    def test_validate_invalid_categorical(self, validator, clean_data):
        """Test validation catches invalid categorical values."""
        df = clean_data.copy()
        df.loc[0, 'sex'] = 'Unknown'  # Invalid value

        report = validator.validate(df)
        assert not report.is_valid

    def test_validation_report_print(self, validator, clean_data):
        """Test that ValidationReport.print_report() works."""
        report = validator.validate(clean_data)

        # Should not raise exception
        report.print_report()


class TestFeatureValidator:
    """Test suite for FeatureValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FeatureValidator()

    @pytest.fixture
    def features_df(self):
        """Create sample features DataFrame."""
        return pd.DataFrame({
            'patient_id': [f'PT{i:05d}' for i in range(100)],
            'age_scaled': np.random.normal(0, 1, 100),
            'tmb_score_scaled': np.random.normal(0, 1, 100),
            'mutation_burden': np.random.randint(0, 4, 100),
            'clinical_risk_score': np.random.uniform(0, 30, 100)
        })

    def test_validate_features_success(self, validator, features_df):
        """Test feature validation passes for valid features."""
        feature_list = ['age_scaled', 'mutation_burden', 'clinical_risk_score']
        report = validator.validate_features(features_df, feature_list)

        assert isinstance(report, ValidationReport)
        assert report.is_valid

    def test_validate_features_missing_feature(self, validator, features_df):
        """Test validation fails when requested feature is missing."""
        feature_list = ['nonexistent_feature']
        report = validator.validate_features(features_df, feature_list)

        assert not report.is_valid
        assert len(report.errors) > 0

    def test_validate_high_missing_rate(self, validator):
        """Test validation warns about high missing rates."""
        df = pd.DataFrame({
            'feature1': [np.nan] * 80 + [1.0] * 20,  # 80% missing
            'feature2': [1.0] * 100
        })

        report = validator.validate_features(df, ['feature1', 'feature2'])

        # Should have warnings about high missing rate
        assert len(report.warnings) > 0
        assert any('missing' in str(w).lower() for w in report.warnings)

    def test_validate_correlations(self, validator):
        """Test correlation validation."""
        # Create highly correlated features
        x = np.random.normal(0, 1, 100)
        df = pd.DataFrame({
            'feature1': x,
            'feature2': x + np.random.normal(0, 0.01, 100),  # Almost identical
            'feature3': np.random.normal(0, 1, 100)
        })

        report = validator.validate_features(df, ['feature1', 'feature2', 'feature3'])

        # Should warn about high correlation
        assert len(report.warnings) > 0
        assert any('correlation' in str(w).lower() for w in report.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
