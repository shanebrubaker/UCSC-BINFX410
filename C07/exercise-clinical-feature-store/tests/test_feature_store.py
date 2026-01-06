"""
Tests for feature store.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_store import FeatureStore
from data_generator import ClinicalDataGenerator


class TestFeatureStore:
    """Test suite for FeatureStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def feature_store(self, temp_dir):
        """Create feature store instance with temporary database."""
        db_path = Path(temp_dir) / "test_feature_store.duckdb"
        config_dir = Path(__file__).parent.parent / "config"
        fs = FeatureStore(db_path=str(db_path), config_dir=str(config_dir))
        yield fs
        fs.close()

    @pytest.fixture
    def sample_data(self):
        """Generate sample clinical data."""
        generator = ClinicalDataGenerator(seed=42)
        return generator.generate_dataset(n_patients=50, introduce_errors=False)

    def test_initialization(self, feature_store):
        """Test feature store initializes correctly."""
        assert feature_store.conn is not None
        assert feature_store.raw_validator is not None
        assert feature_store.feature_validator is not None
        assert feature_store.engineer is not None

    def test_register_feature(self, feature_store):
        """Test feature registration."""
        feature_store.register_feature(
            name='test_feature',
            feature_type='continuous',
            description='Test feature',
            source_columns=['age'],
            category='test'
        )

        # Check it was registered
        metadata = feature_store.get_feature_metadata('test_feature')
        assert metadata is not None
        assert metadata['feature_name'] == 'test_feature'
        assert metadata['version'] == 1

    def test_register_features_from_config(self, feature_store):
        """Test bulk registration from config file."""
        feature_store.register_features_from_config()

        # Check that features were registered
        features_df = feature_store.list_features()
        assert len(features_df) > 0

    def test_ingest_raw_data(self, feature_store, sample_data):
        """Test raw data ingestion."""
        feature_store.ingest_raw_data(sample_data, validate=True, data_version=1)

        # Retrieve and verify
        retrieved_data = feature_store.get_raw_data()
        assert len(retrieved_data) == len(sample_data)
        assert 'patient_id' in retrieved_data.columns

    def test_ingest_invalid_data_raises_error(self, feature_store):
        """Test that invalid data raises validation error."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'patient_id': ['PT00001'],
            'age': [200],  # Invalid age
            'sex': ['Male'],
            'ethnicity': ['Caucasian'],
            'diagnosis': ['Lung Cancer'],
            'comorbidity_count': [2],
            'tp53_mutation': [True],
            'kras_mutation': [False],
            'egfr_mutation': [False],
            'tmb_score': [10.0],
            'msi_status': ['MSS'],
            'wbc_count': [8.0],
            'hemoglobin': [12.0],
            'platelet_count': [250.0],
            'treatment_response': ['Partial Response'],
            'survival_months': [24.0],
            'response_status': [1]
        })

        with pytest.raises(ValueError):
            feature_store.ingest_raw_data(invalid_data, validate=True)

    def test_compute_features(self, feature_store, sample_data):
        """Test feature computation."""
        # First ingest raw data
        feature_store.ingest_raw_data(sample_data, validate=True, data_version=1)

        # Compute features
        features_df = feature_store.compute_features(
            feature_version=1,
            validate=True
        )

        assert len(features_df) > 0
        assert 'mutation_burden' in features_df.columns
        assert 'age_scaled' in features_df.columns

    def test_get_features(self, feature_store, sample_data):
        """Test feature retrieval."""
        # Ingest and compute
        feature_store.ingest_raw_data(sample_data, validate=False)
        feature_store.compute_features(feature_version=1, validate=False)

        # Get features
        features = feature_store.get_features(feature_version=1)

        assert len(features) > 0
        assert 'patient_id' in features.columns

    def test_get_features_filtered(self, feature_store, sample_data):
        """Test filtered feature retrieval."""
        # Ingest and compute
        feature_store.ingest_raw_data(sample_data, validate=False)
        feature_store.compute_features(feature_version=1, validate=False)

        # Get specific patient
        patient_ids = [sample_data['patient_id'].iloc[0]]
        features = feature_store.get_features(
            patient_ids=patient_ids,
            feature_version=1
        )

        assert len(features) == 1
        assert features['patient_id'].iloc[0] == patient_ids[0]

    def test_create_training_dataset(self, feature_store, sample_data):
        """Test training dataset creation."""
        # Ingest and compute
        feature_store.ingest_raw_data(sample_data, validate=False)
        feature_store.compute_features(feature_version=1, validate=False)

        # Create training dataset
        feature_list = ['mutation_burden', 'clinical_risk_score', 'age_scaled']
        training_df = feature_store.create_training_dataset(
            feature_list=feature_list,
            target='response_status',
            include_metadata=False
        )

        assert len(training_df) > 0
        assert 'response_status' in training_df.columns

        # Should have all requested features
        for feature in feature_list:
            assert feature in training_df.columns

        # Should not have patient_id when include_metadata=False
        assert 'patient_id' not in training_df.columns

    def test_get_feature_lineage(self, feature_store):
        """Test feature lineage retrieval."""
        feature_store.register_feature(
            name='test_lineage',
            feature_type='continuous',
            description='Test',
            source_columns=['age', 'comorbidity_count'],
            category='test'
        )

        lineage = feature_store.get_feature_lineage('test_lineage')

        assert 'source_columns' in lineage
        assert lineage['source_columns'] == ['age', 'comorbidity_count']

    def test_list_features(self, feature_store):
        """Test feature listing."""
        # Register some features
        feature_store.register_feature(
            name='feature1',
            feature_type='continuous',
            description='Test 1',
            source_columns=['age'],
            category='demographics'
        )

        feature_store.register_feature(
            name='feature2',
            feature_type='continuous',
            description='Test 2',
            source_columns=['tmb_score'],
            category='genomic'
        )

        # List all features
        all_features = feature_store.list_features()
        assert len(all_features) >= 2

        # List by category
        demo_features = feature_store.list_features(category='demographics')
        assert len(demo_features) >= 1

    def test_data_quality_logging(self, feature_store, sample_data):
        """Test data quality check logging."""
        # Ingest data (which logs quality check)
        feature_store.ingest_raw_data(sample_data, validate=True)

        # Get quality history
        history = feature_store.get_data_quality_history(limit=10)

        assert len(history) > 0
        assert 'check_type' in history.columns
        assert 'passed' in history.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
