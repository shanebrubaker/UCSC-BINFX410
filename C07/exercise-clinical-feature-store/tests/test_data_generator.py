"""
Tests for synthetic data generator.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import ClinicalDataGenerator


class TestClinicalDataGenerator:
    """Test suite for ClinicalDataGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return ClinicalDataGenerator(seed=42)

    def test_generate_demographics(self, generator):
        """Test demographic feature generation."""
        n = 100
        df = generator.generate_demographics(n)

        assert len(df) == n
        assert 'age' in df.columns
        assert 'sex' in df.columns
        assert 'ethnicity' in df.columns

        # Check age range
        assert df['age'].min() >= 18
        assert df['age'].max() <= 95

        # Check sex values
        assert df['sex'].isin(['Male', 'Female', 'M', 'F', 'male', 'female']).all()

    def test_generate_clinical_features(self, generator):
        """Test clinical feature generation."""
        n = 100
        ages = np.random.randint(30, 80, size=n)
        df = generator.generate_clinical_features(n, ages)

        assert len(df) == n
        assert 'diagnosis' in df.columns
        assert 'comorbidity_count' in df.columns
        assert 'treatment_response' in df.columns
        assert 'response_status' in df.columns

        # Check comorbidity range
        assert df['comorbidity_count'].min() >= 0
        assert df['comorbidity_count'].max() <= 10

        # Check response status is binary
        assert df['response_status'].isin([0, 1]).all()

    def test_generate_genomic_features(self, generator):
        """Test genomic feature generation."""
        n = 100
        df = generator.generate_genomic_features(n)

        assert len(df) == n
        assert 'tp53_mutation' in df.columns
        assert 'kras_mutation' in df.columns
        assert 'egfr_mutation' in df.columns
        assert 'tmb_score' in df.columns
        assert 'msi_status' in df.columns

        # Check mutation types
        assert df['tp53_mutation'].dtype == bool
        assert df['kras_mutation'].dtype == bool
        assert df['egfr_mutation'].dtype == bool

        # Check TMB range
        assert df['tmb_score'].min() >= 0
        assert df['tmb_score'].max() <= 100

    def test_generate_lab_values(self, generator):
        """Test lab value generation with missing data."""
        n = 100
        df = generator.generate_lab_values(n)

        assert len(df) == n
        assert 'wbc_count' in df.columns
        assert 'hemoglobin' in df.columns
        assert 'platelet_count' in df.columns

        # Check that there is some missing data (but not too much)
        for col in ['wbc_count', 'hemoglobin', 'platelet_count']:
            missing_pct = df[col].isnull().sum() / len(df)
            assert 0 < missing_pct < 0.25  # Should be around 10-15%

    def test_generate_outcomes(self, generator):
        """Test outcome generation."""
        n = 100
        ages = np.random.randint(30, 80, size=n)
        comorbidities = np.random.randint(0, 5, size=n)
        responses = np.random.choice(
            ["Complete Response", "Partial Response", "Stable Disease", "Progressive Disease"],
            size=n
        )

        df = generator.generate_outcomes(n, ages, comorbidities, responses)

        assert len(df) == n
        assert 'survival_months' in df.columns
        assert df['survival_months'].min() > 0
        assert df['survival_months'].max() <= 180

    def test_generate_dataset(self, generator):
        """Test complete dataset generation."""
        n = 50
        df = generator.generate_dataset(n_patients=n, introduce_errors=False)

        assert len(df) == n
        assert 'patient_id' in df.columns

        # Check patient ID format
        assert all(pid.startswith('PT') for pid in df['patient_id'])

        # Check all expected columns exist
        expected_cols = [
            'patient_id', 'age', 'sex', 'ethnicity', 'diagnosis',
            'comorbidity_count', 'tp53_mutation', 'kras_mutation',
            'egfr_mutation', 'tmb_score', 'msi_status', 'wbc_count',
            'hemoglobin', 'platelet_count', 'treatment_response',
            'survival_months', 'response_status'
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_introduce_errors(self, generator):
        """Test that intentional errors are introduced."""
        n = 100
        df = generator.generate_dataset(n_patients=n, introduce_errors=True)

        # Should have some invalid ages
        invalid_ages = ((df['age'] < 0) | (df['age'] > 120)).sum()
        assert invalid_ages > 0  # With 100 patients and 1% error rate, likely to have some

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        gen1 = ClinicalDataGenerator(seed=42)
        gen2 = ClinicalDataGenerator(seed=42)

        df1 = gen1.generate_dataset(n_patients=50, introduce_errors=False)
        df2 = gen2.generate_dataset(n_patients=50, introduce_errors=False)

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
