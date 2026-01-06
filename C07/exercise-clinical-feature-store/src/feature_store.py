"""
Feature Store Implementation using DuckDB

A lightweight feature store for clinical/genomic data that provides:
- Feature registration and metadata management
- Feature computation and storage
- Training dataset creation
- Feature versioning and lineage tracking
- Data quality validation integration

This demonstrates production MLOps patterns in an educational context.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import yaml

from features import ClinicalFeatureEngineer, get_feature_lineage, get_feature_descriptions
from data_validators import RawDataValidator, FeatureValidator


class FeatureStore:
    """
    Lightweight feature store using DuckDB for storage.

    Key concepts:
    - Feature Registry: Metadata about each feature (definition, lineage, version)
    - Feature Tables: Actual feature values indexed by patient_id
    - Versioning: Track feature evolution over time
    - Lineage: Know which raw columns created which features
    - Validation: Ensure data quality throughout

    Why use a feature store?
    1. Centralize feature definitions (reuse across models)
    2. Ensure consistency (same features in training/serving)
    3. Enable collaboration (shared feature repository)
    4. Track lineage (debug and audit)
    5. Version features (reproducibility)
    """

    def __init__(self, db_path: str = "data/feature_store.duckdb",
                 config_dir: str = "config"):
        """
        Initialize feature store.

        Args:
            db_path: Path to DuckDB database file
            config_dir: Directory containing configuration files
        """
        self.db_path = db_path
        self.config_dir = Path(config_dir)

        # Create database connection
        self.conn = duckdb.connect(db_path)

        # Initialize validators
        self.raw_validator = RawDataValidator()
        self.feature_validator = FeatureValidator()

        # Initialize feature engineer
        self.engineer = ClinicalFeatureEngineer()

        # Create tables if they don't exist
        self._initialize_database()

        print(f"Feature store initialized: {db_path}")

    def _initialize_database(self, reset: bool = False) -> None:
        """
        Create feature store schema.

        Args:
            reset: If True, drop existing tables and recreate

        Tables:
        - feature_registry: Metadata about features
        - raw_data: Raw patient data
        - feature_values: Computed feature values
        - data_quality_log: Quality check results
        """
        if reset:
            # Drop existing tables
            self.conn.execute("DROP TABLE IF EXISTS feature_registry")
            self.conn.execute("DROP TABLE IF EXISTS raw_data")
            self.conn.execute("DROP TABLE IF EXISTS feature_values")
            self.conn.execute("DROP TABLE IF EXISTS data_quality_log")
            self.conn.commit()

        # Feature registry table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_name VARCHAR PRIMARY KEY,
                feature_type VARCHAR,
                description TEXT,
                source_columns VARCHAR,  -- JSON array
                category VARCHAR,
                version INTEGER,
                created_date TIMESTAMP,
                updated_date TIMESTAMP,
                validation_schema TEXT,  -- JSON
                tags VARCHAR  -- JSON array
            )
        """)

        # Raw data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_data (
                patient_id VARCHAR PRIMARY KEY,
                age INTEGER,
                sex VARCHAR,
                ethnicity VARCHAR,
                diagnosis VARCHAR,
                comorbidity_count INTEGER,
                tp53_mutation BOOLEAN,
                kras_mutation BOOLEAN,
                egfr_mutation BOOLEAN,
                tmb_score DOUBLE,
                msi_status VARCHAR,
                wbc_count DOUBLE,
                hemoglobin DOUBLE,
                platelet_count DOUBLE,
                treatment_response VARCHAR,
                survival_months DOUBLE,
                response_status INTEGER,
                ingestion_timestamp TIMESTAMP,
                data_version INTEGER
            )
        """)

        # Feature values table (wide format with patient_id + all features)
        # Note: In production, might use narrow format (patient_id, feature_name, value)
        # but wide format is simpler for ML training dataset creation
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                patient_id VARCHAR,
                feature_version INTEGER,
                computed_timestamp TIMESTAMP,
                -- Imputed features
                wbc_imputed DOUBLE,
                hemoglobin_imputed DOUBLE,
                platelet_imputed DOUBLE,
                -- Scaled features
                age_scaled DOUBLE,
                tmb_score_scaled DOUBLE,
                -- Derived features
                mutation_burden INTEGER,
                clinical_risk_score DOUBLE,
                high_risk_patient INTEGER,
                -- Categorical features (stored as strings, can be one-hot encoded on read)
                age_group VARCHAR,
                wbc_category VARCHAR,
                sex_normalized VARCHAR,
                PRIMARY KEY (patient_id, feature_version)
            )
        """)

        # Data quality log
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_log (
                check_id INTEGER PRIMARY KEY,
                check_timestamp TIMESTAMP,
                check_type VARCHAR,
                passed BOOLEAN,
                error_count INTEGER,
                warning_count INTEGER,
                details TEXT  -- JSON
            )
        """)

        self.conn.commit()

    # ============================================================
    # FEATURE REGISTRATION
    # ============================================================

    def register_feature(self, name: str, feature_type: str,
                        description: str, source_columns: List[str],
                        category: str, validation_schema: Optional[Dict] = None,
                        tags: Optional[List[str]] = None) -> None:
        """
        Register a new feature in the feature registry.

        This creates metadata that documents what the feature is,
        where it comes from, and how it should be validated.

        Args:
            name: Unique feature name
            feature_type: Type (continuous, categorical, binary)
            description: Human-readable description
            source_columns: Raw columns used to compute this feature
            category: Feature category (demographics, clinical, genomic, etc.)
            validation_schema: Optional validation rules
            tags: Optional tags for organization
        """
        # Check if feature already exists
        existing = self.conn.execute(
            "SELECT version FROM feature_registry WHERE feature_name = ?",
            [name]
        ).fetchone()

        if existing:
            # Increment version
            new_version = existing[0] + 1
            timestamp = datetime.now()

            self.conn.execute("""
                UPDATE feature_registry
                SET version = ?,
                    updated_date = ?,
                    description = ?,
                    source_columns = ?,
                    category = ?,
                    validation_schema = ?,
                    tags = ?
                WHERE feature_name = ?
            """, [
                new_version, timestamp, description,
                json.dumps(source_columns), category,
                json.dumps(validation_schema) if validation_schema else None,
                json.dumps(tags) if tags else None,
                name
            ])
            print(f"Updated feature '{name}' to version {new_version}")
        else:
            # Create new feature
            timestamp = datetime.now()
            self.conn.execute("""
                INSERT INTO feature_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                name, feature_type, description,
                json.dumps(source_columns), category,
                1, timestamp, timestamp,
                json.dumps(validation_schema) if validation_schema else None,
                json.dumps(tags) if tags else None
            ])
            print(f"Registered new feature '{name}' (version 1)")

        self.conn.commit()

    def register_features_from_config(self) -> None:
        """
        Register all features defined in config/features.yaml.

        This is a convenient way to bulk-register features.
        """
        config_path = self.config_dir / "features.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for feature_name, feature_config in config['features'].items():
            self.register_feature(
                name=feature_name,
                feature_type=feature_config['type'],
                description=feature_config['description'],
                source_columns=feature_config['source_columns'],
                category=feature_config['category']
            )

        print(f"Registered {len(config['features'])} features from config")

    def get_feature_metadata(self, feature_name: str) -> Optional[Dict]:
        """
        Get metadata for a specific feature.

        Args:
            feature_name: Name of feature

        Returns:
            Dictionary with feature metadata or None if not found
        """
        result = self.conn.execute("""
            SELECT * FROM feature_registry WHERE feature_name = ?
        """, [feature_name]).fetchone()

        if result is None:
            return None

        columns = [desc[0] for desc in self.conn.description]
        metadata = dict(zip(columns, result))

        # Parse JSON fields
        metadata['source_columns'] = json.loads(metadata['source_columns'])
        if metadata['validation_schema']:
            metadata['validation_schema'] = json.loads(metadata['validation_schema'])
        if metadata['tags']:
            metadata['tags'] = json.loads(metadata['tags'])

        return metadata

    def list_features(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        List all registered features.

        Args:
            category: Optional category filter

        Returns:
            DataFrame with feature metadata
        """
        if category:
            query = "SELECT * FROM feature_registry WHERE category = ?"
            result = self.conn.execute(query, [category]).df()
        else:
            result = self.conn.execute("SELECT * FROM feature_registry").df()

        return result

    # ============================================================
    # DATA INGESTION
    # ============================================================

    def ingest_raw_data(self, df: pd.DataFrame,
                       validate: bool = True,
                       data_version: int = 1) -> None:
        """
        Ingest raw clinical data into the feature store.

        This is the entry point for new data. Validation is performed
        to catch issues early before they propagate to features.

        Args:
            df: Raw patient data
            validate: Whether to validate data quality
            data_version: Version number for this data batch

        Raises:
            ValueError: If validation fails
        """
        if validate:
            print("Validating raw data...")
            report = self.raw_validator.validate(df)
            report.print_report()

            if not report.is_valid:
                raise ValueError("Raw data validation failed. Fix errors before ingesting.")

            # Log validation results
            self._log_data_quality_check(
                check_type="raw_data_validation",
                passed=True,
                error_count=0,
                warning_count=len(report.warnings),
                details={"warnings": report.warnings}
            )

        # Add metadata columns
        df_to_insert = df.copy()
        df_to_insert['ingestion_timestamp'] = datetime.now()
        df_to_insert['data_version'] = data_version

        # Reorder columns to match table schema
        column_order = [
            'patient_id', 'age', 'sex', 'ethnicity', 'diagnosis',
            'comorbidity_count', 'tp53_mutation', 'kras_mutation', 'egfr_mutation',
            'tmb_score', 'msi_status', 'wbc_count', 'hemoglobin', 'platelet_count',
            'treatment_response', 'survival_months', 'response_status',
            'ingestion_timestamp', 'data_version'
        ]
        df_to_insert = df_to_insert[column_order]

        # Insert into database (replace if patient_id exists)
        self.conn.execute("DELETE FROM raw_data WHERE patient_id IN (SELECT patient_id FROM df_to_insert)")
        self.conn.execute("INSERT INTO raw_data SELECT * FROM df_to_insert")
        self.conn.commit()

        print(f"Ingested {len(df)} patient records (version {data_version})")

    def get_raw_data(self, patient_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve raw patient data.

        Args:
            patient_ids: Optional list of patient IDs to filter

        Returns:
            DataFrame with raw data
        """
        if patient_ids:
            placeholders = ','.join(['?' for _ in patient_ids])
            query = f"SELECT * FROM raw_data WHERE patient_id IN ({placeholders})"
            return self.conn.execute(query, patient_ids).df()
        else:
            return self.conn.execute("SELECT * FROM raw_data").df()

    # ============================================================
    # FEATURE COMPUTATION
    # ============================================================

    def compute_features(self, patient_ids: Optional[List[str]] = None,
                        feature_version: int = 1,
                        validate: bool = True) -> pd.DataFrame:
        """
        Compute features for specified patients and store in feature table.

        This is the core feature engineering pipeline:
        1. Retrieve raw data
        2. Apply transformations
        3. Validate computed features
        4. Store in feature_values table

        Args:
            patient_ids: Optional list of patient IDs (None = all patients)
            feature_version: Version number for these features
            validate: Whether to validate computed features

        Returns:
            DataFrame with computed features
        """
        # Get raw data
        print("Retrieving raw data...")
        raw_df = self.get_raw_data(patient_ids)

        if len(raw_df) == 0:
            print("No patients found")
            return pd.DataFrame()

        print(f"Computing features for {len(raw_df)} patients...")

        # Apply feature engineering
        # For first computation, fit transformers; otherwise use existing
        fit_transformers = not self.engineer.fitted
        features_df = self.engineer.transform_all(raw_df, fit=fit_transformers)

        # Select columns to store in feature_values table
        feature_cols_to_store = [
            'patient_id',
            'wbc_imputed', 'hemoglobin_imputed', 'platelet_imputed',
            'age_scaled', 'tmb_score_scaled',
            'mutation_burden', 'clinical_risk_score', 'high_risk_patient',
            'age_group', 'wbc_category'
        ]

        # Add normalized sex
        features_df['sex_normalized'] = features_df['sex'].map({
            'Male': 'Male', 'M': 'Male', 'male': 'Male',
            'Female': 'Female', 'F': 'Female', 'female': 'Female'
        })

        # Select relevant columns
        store_df = features_df[feature_cols_to_store + ['sex_normalized']].copy()

        # Add metadata
        store_df['feature_version'] = feature_version
        store_df['computed_timestamp'] = datetime.now()

        # Validate features
        if validate:
            print("Validating computed features...")
            validation_features = [col for col in store_df.columns
                                  if col not in ['patient_id', 'feature_version',
                                               'computed_timestamp', 'age_group',
                                               'wbc_category', 'sex_normalized']]

            report = self.feature_validator.validate_features(store_df, validation_features)
            report.print_report()

            self._log_data_quality_check(
                check_type="feature_validation",
                passed=report.is_valid,
                error_count=len(report.errors),
                warning_count=len(report.warnings),
                details={"errors": report.errors, "warnings": report.warnings}
            )

        # Reorder columns to match table schema
        column_order = [
            'patient_id', 'feature_version', 'computed_timestamp',
            'wbc_imputed', 'hemoglobin_imputed', 'platelet_imputed',
            'age_scaled', 'tmb_score_scaled',
            'mutation_burden', 'clinical_risk_score', 'high_risk_patient',
            'age_group', 'wbc_category', 'sex_normalized'
        ]
        store_df = store_df[column_order]

        # Store in database
        self.conn.execute(
            "DELETE FROM feature_values WHERE patient_id IN (SELECT patient_id FROM store_df)"
        )
        self.conn.execute("INSERT INTO feature_values SELECT * FROM store_df")
        self.conn.commit()

        print(f"Stored features for {len(store_df)} patients (version {feature_version})")

        return features_df

    def get_features(self, patient_ids: Optional[List[str]] = None,
                    feature_list: Optional[List[str]] = None,
                    feature_version: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve computed features from the feature store.

        Args:
            patient_ids: Optional list of patient IDs
            feature_list: Optional list of specific features to retrieve
            feature_version: Optional feature version (None = latest)

        Returns:
            DataFrame with requested features
        """
        # Build query
        select_cols = "patient_id"
        if feature_list:
            select_cols += ", " + ", ".join(feature_list)
        else:
            select_cols = "*"

        query = f"SELECT {select_cols} FROM feature_values WHERE 1=1"
        params = []

        if patient_ids:
            placeholders = ','.join(['?' for _ in patient_ids])
            query += f" AND patient_id IN ({placeholders})"
            params.extend(patient_ids)

        if feature_version is not None:
            query += " AND feature_version = ?"
            params.append(feature_version)

        if params:
            return self.conn.execute(query, params).df()
        else:
            return self.conn.execute(query).df()

    # ============================================================
    # TRAINING DATASET CREATION
    # ============================================================

    def create_training_dataset(self, feature_list: List[str],
                               target: str,
                               patient_ids: Optional[List[str]] = None,
                               include_metadata: bool = False) -> pd.DataFrame:
        """
        Create a training dataset by joining features with target variable.

        This is the primary interface for ML model training:
        - Select specific features
        - Join with target variable from raw data
        - Optionally filter to specific patients

        Args:
            feature_list: List of feature names to include
            target: Target variable name (from raw_data)
            patient_ids: Optional patient ID filter
            include_metadata: Whether to include patient_id and timestamps

        Returns:
            DataFrame ready for ML training (features + target)
        """
        # Get features
        features_df = self.get_features(patient_ids=patient_ids, feature_list=feature_list)

        # Get target variable from raw data
        if patient_ids:
            placeholders = ','.join(['?' for _ in patient_ids])
            query = f"SELECT patient_id, {target} FROM raw_data WHERE patient_id IN ({placeholders})"
            target_df = self.conn.execute(query, patient_ids).df()
        else:
            target_df = self.conn.execute(f"SELECT patient_id, {target} FROM raw_data").df()

        # Merge
        training_df = features_df.merge(target_df, on='patient_id', how='inner')

        # Remove metadata columns unless requested
        if not include_metadata:
            metadata_cols = ['feature_version', 'computed_timestamp']
            training_df = training_df.drop(columns=[c for c in metadata_cols if c in training_df.columns])
            training_df = training_df.drop(columns=['patient_id'])

        print(f"Created training dataset: {len(training_df)} samples, {len(training_df.columns)} columns")

        return training_df

    # ============================================================
    # LINEAGE AND METADATA
    # ============================================================

    def get_feature_lineage(self, feature_name: str) -> Dict:
        """
        Get lineage information for a feature.

        Shows:
        - Source columns used to compute the feature
        - Feature version history
        - Transformation applied

        Args:
            feature_name: Name of feature

        Returns:
            Dictionary with lineage information
        """
        metadata = self.get_feature_metadata(feature_name)

        if metadata is None:
            return {"error": f"Feature '{feature_name}' not found"}

        lineage = {
            "feature_name": feature_name,
            "source_columns": metadata['source_columns'],
            "category": metadata['category'],
            "version": metadata['version'],
            "created": metadata['created_date'],
            "updated": metadata['updated_date']
        }

        return lineage

    # ============================================================
    # DATA QUALITY LOGGING
    # ============================================================

    def _log_data_quality_check(self, check_type: str, passed: bool,
                                error_count: int, warning_count: int,
                                details: Dict) -> None:
        """
        Log data quality check results.

        Args:
            check_type: Type of check performed
            passed: Whether check passed
            error_count: Number of errors
            warning_count: Number of warnings
            details: Additional details (JSON serializable)
        """
        # Get next check_id
        max_id = self.conn.execute("SELECT COALESCE(MAX(check_id), 0) FROM data_quality_log").fetchone()[0]
        next_id = max_id + 1

        self.conn.execute("""
            INSERT INTO data_quality_log VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id,
            datetime.now(),
            check_type,
            passed,
            error_count,
            warning_count,
            json.dumps(details)
        ])
        self.conn.commit()

    def get_data_quality_history(self, limit: int = 10) -> pd.DataFrame:
        """
        Get recent data quality check history.

        Args:
            limit: Number of recent checks to return

        Returns:
            DataFrame with quality check history
        """
        return self.conn.execute(f"""
            SELECT * FROM data_quality_log
            ORDER BY check_timestamp DESC
            LIMIT {limit}
        """).df()

    # ============================================================
    # CLEANUP
    # ============================================================

    def reset_database(self) -> None:
        """
        Drop and recreate all database tables.

        WARNING: This will delete all existing data!
        Use this to fix schema issues or start fresh.
        """
        print("Resetting database schema...")
        self._initialize_database(reset=True)
        print("Database schema reset complete")

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        print("Feature store connection closed")


if __name__ == "__main__":
    # Example usage
    fs = FeatureStore()
    print("\nFeature store ready!")
    print(f"Database: {fs.db_path}")
