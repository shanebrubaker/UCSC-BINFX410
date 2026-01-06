"""
Data Validation Schemas using Pandera

Defines validation rules for both raw clinical data and computed features.
Validation prevents silent failures and ensures data quality throughout the pipeline.
"""

import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class ValidationReport:
    """Container for validation results."""

    def __init__(self, is_valid: bool, errors: List[str], warnings: List[str]):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings

    def __repr__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return f"ValidationReport(status={status}, errors={len(self.errors)}, warnings={len(self.warnings)})"

    def print_report(self) -> None:
        """Print formatted validation report."""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {'PASSED' if self.is_valid else 'FAILED'}")
        print(f"{'='*60}")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if self.is_valid and not self.warnings:
            print("\nAll validation checks passed!")
        print(f"{'='*60}\n")


class RawDataValidator:
    """
    Validates raw clinical data before it enters the feature store.

    This is the first line of defense against data quality issues.
    Catches problems like:
    - Missing required fields
    - Invalid data types
    - Out-of-range values
    - Invalid categorical values
    - Statistical anomalies
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize validator with configuration.

        Args:
            config_path: Path to validation.yaml config file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "validation.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.schema = self._build_schema()

    def _build_schema(self) -> DataFrameSchema:
        """
        Build Pandera schema from configuration.

        Returns:
            Pandera DataFrameSchema for validation
        """
        raw_config = self.config['raw_data_validation']
        columns = {}

        # patient_id
        columns['patient_id'] = Column(
            str,
            nullable=False,
            unique=True,
            description="Unique patient identifier"
        )

        # age
        columns['age'] = Column(
            int,
            checks=[
                Check.in_range(0, 120, include_min=True, include_max=True)
            ],
            nullable=False,
            description="Patient age in years"
        )

        # sex (normalize inconsistent values during validation)
        columns['sex'] = Column(
            str,
            checks=[
                Check.isin(["Male", "Female", "M", "F", "male", "female"])
            ],
            nullable=False,
            description="Patient sex"
        )

        # ethnicity
        columns['ethnicity'] = Column(
            str,
            checks=[
                Check.isin(["Caucasian", "African American", "Asian", "Hispanic", "Other"])
            ],
            nullable=False,
            description="Patient ethnicity"
        )

        # diagnosis
        columns['diagnosis'] = Column(
            str,
            nullable=False,
            description="Primary cancer diagnosis"
        )

        # comorbidity_count
        columns['comorbidity_count'] = Column(
            int,
            checks=[
                Check.in_range(0, 10, include_min=True, include_max=True)
            ],
            nullable=False,
            description="Number of comorbid conditions"
        )

        # Mutation flags
        columns['tp53_mutation'] = Column(
            bool,
            nullable=False,
            description="TP53 mutation status"
        )

        columns['kras_mutation'] = Column(
            bool,
            nullable=False,
            description="KRAS mutation status"
        )

        columns['egfr_mutation'] = Column(
            bool,
            nullable=False,
            description="EGFR mutation status"
        )

        # tmb_score
        columns['tmb_score'] = Column(
            float,
            checks=[
                Check.in_range(0, 100, include_min=True, include_max=True)
            ],
            nullable=False,
            description="Tumor mutational burden score"
        )

        # msi_status
        columns['msi_status'] = Column(
            str,
            checks=[
                Check.isin(["MSI-H", "MSI-L", "MSS"])
            ],
            nullable=False,
            description="Microsatellite instability status"
        )

        # Lab values (nullable)
        columns['wbc_count'] = Column(
            float,
            checks=[
                Check.in_range(0, 50, include_min=False, include_max=True)
            ],
            nullable=True,
            description="White blood cell count (K/uL)"
        )

        columns['hemoglobin'] = Column(
            float,
            checks=[
                Check.in_range(0, 25, include_min=False, include_max=True)
            ],
            nullable=True,
            description="Hemoglobin level (g/dL)"
        )

        columns['platelet_count'] = Column(
            float,
            checks=[
                Check.in_range(0, 1000, include_min=False, include_max=True)
            ],
            nullable=True,
            description="Platelet count (K/uL)"
        )

        # treatment_response
        columns['treatment_response'] = Column(
            str,
            checks=[
                Check.isin([
                    "Complete Response", "Partial Response",
                    "Stable Disease", "Progressive Disease"
                ])
            ],
            nullable=False,
            description="Treatment response category"
        )

        # survival_months
        columns['survival_months'] = Column(
            float,
            checks=[
                Check.in_range(0, 200, include_min=False, include_max=True)
            ],
            nullable=False,
            description="Survival time in months"
        )

        # response_status
        columns['response_status'] = Column(
            int,
            checks=[
                Check.isin([0, 1])
            ],
            nullable=False,
            description="Binary response indicator"
        )

        return DataFrameSchema(columns, strict=False)

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate raw clinical data.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationReport with results
        """
        errors = []
        warnings = []

        try:
            # Run Pandera validation
            validated_df = self.schema.validate(df, lazy=True)

            # Additional statistical checks
            stat_warnings = self._check_statistical_properties(df)
            warnings.extend(stat_warnings)

            # Check missing data patterns
            missing_warnings = self._check_missing_patterns(df)
            warnings.extend(missing_warnings)

            return ValidationReport(is_valid=True, errors=errors, warnings=warnings)

        except pa.errors.SchemaErrors as e:
            # Parse Pandera errors
            for error in e.failure_cases.itertuples():
                error_msg = f"Column '{error.column}': {error.check} failed for {error.failure_case}"
                errors.append(error_msg)

            return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

    def _check_statistical_properties(self, df: pd.DataFrame) -> List[str]:
        """
        Check for statistical anomalies that might indicate data issues.

        Args:
            df: DataFrame to check

        Returns:
            List of warning messages
        """
        warnings = []

        # Check age distribution (should be roughly normal/gamma)
        if 'age' in df.columns:
            age_mean = df['age'].mean()
            if age_mean < 40 or age_mean > 75:
                warnings.append(
                    f"Age distribution unusual: mean={age_mean:.1f} "
                    "(expected ~55-65 for cancer cohort)"
                )

        # Check lab value distributions
        if 'wbc_count' in df.columns:
            wbc_values = df['wbc_count'].dropna()
            if len(wbc_values) > 0:
                outlier_pct = (wbc_values > 30).sum() / len(wbc_values) * 100
                if outlier_pct > 5:
                    warnings.append(
                        f"WBC count has {outlier_pct:.1f}% outliers (>30), "
                        "may indicate data quality issue"
                    )

        # Check mutation rates are reasonable
        if 'tp53_mutation' in df.columns:
            tp53_rate = df['tp53_mutation'].mean()
            if tp53_rate < 0.3 or tp53_rate > 0.7:
                warnings.append(
                    f"TP53 mutation rate {tp53_rate*100:.1f}% outside expected range (40-60%)"
                )

        return warnings

    def _check_missing_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Check for unusual missing data patterns.

        Args:
            df: DataFrame to check

        Returns:
            List of warning messages
        """
        warnings = []

        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 20]

        if len(high_missing) > 0:
            for col in high_missing.index:
                warnings.append(
                    f"Column '{col}' has {high_missing[col]:.1f}% missing data "
                    "(threshold: 20%)"
                )

        return warnings


class FeatureValidator:
    """
    Validates computed features to ensure they meet quality standards.

    This prevents propagating errors from feature engineering and
    catches issues like:
    - Features with too much missing data
    - Incorrect scaling (z-scores should have mean~0, std~1)
    - Invalid derived values
    - High feature correlations (potential multicollinearity)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature validator.

        Args:
            config_path: Path to validation.yaml config file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "validation.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def validate_features(self, df: pd.DataFrame,
                         feature_list: List[str]) -> ValidationReport:
        """
        Validate computed features.

        Args:
            df: DataFrame with computed features
            feature_list: List of feature names to validate

        Returns:
            ValidationReport with results
        """
        errors = []
        warnings = []

        # Check all requested features exist
        missing_features = set(feature_list) - set(df.columns)
        if missing_features:
            errors.append(f"Missing features: {missing_features}")

        if errors:
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

        # Validate distributional properties for scaled features
        dist_warnings = self._validate_distributions(df, feature_list)
        warnings.extend(dist_warnings)

        # Check missing data rates
        missing_warnings = self._validate_missing_rates(df, feature_list)
        warnings.extend(missing_warnings)

        # Check for high correlations
        corr_warnings = self._validate_correlations(df, feature_list)
        warnings.extend(corr_warnings)

        return ValidationReport(is_valid=True, errors=errors, warnings=warnings)

    def _validate_distributions(self, df: pd.DataFrame,
                               feature_list: List[str]) -> List[str]:
        """
        Validate distributional properties of features.

        Scaled features should have mean~0, std~1.
        Derived features should be in expected ranges.

        Args:
            df: DataFrame with features
            feature_list: Features to check

        Returns:
            List of warnings
        """
        warnings = []
        dist_config = self.config.get('feature_validation', {}).get('distributional_checks', {})

        for feature in feature_list:
            if feature not in df.columns:
                continue

            if feature in dist_config:
                feature_config = dist_config[feature]
                values = df[feature].dropna()

                if len(values) == 0:
                    warnings.append(f"Feature '{feature}' has no non-null values")
                    continue

                # Check mean range for scaled features
                if 'mean_range' in feature_config:
                    mean_val = values.mean()
                    min_mean, max_mean = feature_config['mean_range']
                    if not (min_mean <= mean_val <= max_mean):
                        warnings.append(
                            f"Feature '{feature}' mean {mean_val:.3f} outside "
                            f"expected range [{min_mean}, {max_mean}]"
                        )

                # Check std range for scaled features
                if 'std_range' in feature_config:
                    std_val = values.std()
                    min_std, max_std = feature_config['std_range']
                    if not (min_std <= std_val <= max_std):
                        warnings.append(
                            f"Feature '{feature}' std {std_val:.3f} outside "
                            f"expected range [{min_std}, {max_std}]"
                        )

                # Check value ranges
                if 'min_value' in feature_config:
                    min_val = values.min()
                    expected_min = feature_config['min_value']
                    if min_val < expected_min:
                        warnings.append(
                            f"Feature '{feature}' has minimum {min_val:.3f} "
                            f"below expected {expected_min}"
                        )

                if 'max_value' in feature_config:
                    max_val = values.max()
                    expected_max = feature_config['max_value']
                    if max_val > expected_max:
                        warnings.append(
                            f"Feature '{feature}' has maximum {max_val:.3f} "
                            f"above expected {expected_max}"
                        )

        return warnings

    def _validate_missing_rates(self, df: pd.DataFrame,
                                feature_list: List[str]) -> List[str]:
        """
        Check that features don't have excessive missing data.

        Args:
            df: DataFrame with features
            feature_list: Features to check

        Returns:
            List of warnings
        """
        warnings = []
        max_missing = self.config['feature_validation']['missing_data_thresholds']['max_missing_rate']

        for feature in feature_list:
            if feature not in df.columns:
                continue

            missing_rate = df[feature].isnull().sum() / len(df)
            if missing_rate > max_missing:
                warnings.append(
                    f"Feature '{feature}' has {missing_rate*100:.1f}% missing data "
                    f"(threshold: {max_missing*100:.0f}%)"
                )

        return warnings

    def _validate_correlations(self, df: pd.DataFrame,
                              feature_list: List[str]) -> List[str]:
        """
        Check for high feature correlations that might indicate redundancy.

        Args:
            df: DataFrame with features
            feature_list: Features to check

        Returns:
            List of warnings
        """
        warnings = []

        corr_config = self.config['feature_validation']['correlation_monitoring']
        if not corr_config['enabled']:
            return warnings

        # Get numeric features only
        numeric_features = [f for f in feature_list
                          if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

        if len(numeric_features) < 2:
            return warnings

        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr().abs()

        # Find high correlations (excluding diagonal)
        threshold = corr_config['high_correlation_threshold']
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        for feat1, feat2, corr_val in high_corr:
            warnings.append(
                f"High correlation ({corr_val:.3f}) between '{feat1}' and '{feat2}' "
                f"(threshold: {threshold})"
            )

        return warnings


if __name__ == "__main__":
    # Example usage
    print("Testing validators...")

    # Test raw data validator
    validator = RawDataValidator()
    print("\nRaw data schema:")
    print(validator.schema)
