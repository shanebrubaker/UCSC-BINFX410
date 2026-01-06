"""
Feature Engineering Functions for Clinical/Genomic Data

Each function transforms raw clinical data into ML-ready features with:
- Clear medical/biological rationale
- Input validation
- Documentation of expected inputs/outputs
- Handling of edge cases

Feature engineering is separated from model training to enable:
- Feature reuse across multiple models
- Consistent transformations
- Easy testing and validation
- Feature versioning and lineage tracking
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ClinicalFeatureEngineer:
    """
    Feature engineering pipeline for clinical and genomic data.

    This class implements the five core transformation types:
    1. Imputation - Handle missing lab values
    2. Encoding - Convert categorical variables to numeric
    3. Scaling - Standardize continuous variables
    4. Derived features - Create new features from combinations
    5. Binning - Discretize continuous variables into categories
    """

    def __init__(self):
        """Initialize feature engineering components."""
        self.imputers: Dict[str, SimpleImputer] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.fitted = False

    # ============================================================
    # 1. IMPUTATION - Handle Missing Lab Values
    # ============================================================

    def impute_lab_values(self, df: pd.DataFrame,
                         strategy: str = 'median',
                         fit: bool = False) -> pd.DataFrame:
        """
        Impute missing laboratory values using statistical strategies.

        Medical Rationale:
        Lab values are frequently missing in clinical datasets because:
        - Tests aren't ordered for all patients
        - Sample quality issues prevent measurement
        - Results are pending or lost

        For lab values, median imputation is often preferred over mean because:
        - Lab distributions are often skewed
        - Median is robust to outliers
        - Preserves realistic value ranges

        Args:
            df: DataFrame with lab value columns
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')
            fit: Whether to fit imputer (True for training, False for inference)

        Returns:
            DataFrame with imputed values
        """
        lab_columns = ['wbc_count', 'hemoglobin', 'platelet_count']
        result_df = df.copy()

        for col in lab_columns:
            if col not in df.columns:
                continue

            # Create imputer for this column if needed
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy=strategy)

            # Get values
            values = df[[col]].values

            if fit:
                # Fit and transform
                imputed_values = self.imputers[col].fit_transform(values)
            else:
                # Transform only (requires prior fitting)
                if not hasattr(self.imputers[col], 'statistics_'):
                    raise ValueError(f"Imputer for '{col}' not fitted. Set fit=True first.")
                imputed_values = self.imputers[col].transform(values)

            # Create feature name
            feature_name = f"{col.replace('_count', '')}_imputed"
            if col == 'wbc_count':
                feature_name = 'wbc_imputed'

            result_df[feature_name] = imputed_values

        return result_df

    # ============================================================
    # 2. ENCODING - One-Hot Encode Categorical Variables
    # ============================================================

    def encode_categorical(self, df: pd.DataFrame,
                          fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical variables.

        Medical Rationale:
        Categorical variables like sex and ethnicity have no ordinal relationship,
        so they must be encoded before use in ML models. One-hot encoding:
        - Preserves all information
        - Creates binary features that most algorithms can use
        - Avoids imposing artificial ordering

        Note: Sex is first normalized to handle inconsistent formatting
        (Male/M/male -> Male, Female/F/female -> Female)

        Args:
            df: DataFrame with categorical columns
            fit: Whether to fit encoder (True for training, False for inference)

        Returns:
            DataFrame with encoded features
        """
        result_df = df.copy()

        # Normalize sex values first (handles inconsistent formatting)
        if 'sex' in df.columns:
            result_df = self._normalize_sex(result_df)

        categorical_features = {
            'sex': 'sex_encoded',
            'ethnicity': 'ethnicity_encoded'
        }

        for col, feature_name in categorical_features.items():
            if col not in df.columns:
                continue

            # Create encoder for this column if needed
            if col not in self.encoders:
                self.encoders[col] = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore',  # Important for new categories in production
                    drop='first'  # Drop first category to avoid multicollinearity
                )

            # Get values
            values = result_df[[col]].values

            if fit:
                # Fit and transform
                encoded_values = self.encoders[col].fit_transform(values)
            else:
                # Transform only
                if not hasattr(self.encoders[col], 'categories_'):
                    raise ValueError(f"Encoder for '{col}' not fitted. Set fit=True first.")
                encoded_values = self.encoders[col].transform(values)

            # Create column names
            categories = self.encoders[col].categories_[0][1:]  # Skip first (dropped)
            encoded_cols = [f"{feature_name}_{cat}" for cat in categories]

            # Add to dataframe
            for i, col_name in enumerate(encoded_cols):
                result_df[col_name] = encoded_values[:, i]

        return result_df

    def _normalize_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sex values to consistent format.

        Handles common inconsistencies in clinical data entry:
        Male/M/male -> Male
        Female/F/female -> Female

        Args:
            df: DataFrame with 'sex' column

        Returns:
            DataFrame with normalized sex values
        """
        result_df = df.copy()

        if 'sex' in df.columns:
            sex_mapping = {
                'Male': 'Male', 'M': 'Male', 'male': 'Male',
                'Female': 'Female', 'F': 'Female', 'female': 'Female'
            }
            result_df['sex'] = result_df['sex'].map(sex_mapping)

        return result_df

    # ============================================================
    # 3. SCALING - Standardize Continuous Variables
    # ============================================================

    def scale_continuous(self, df: pd.DataFrame,
                        fit: bool = False) -> pd.DataFrame:
        """
        Standardize continuous variables using z-score normalization.

        Medical Rationale:
        Features like age and TMB score have different scales and units.
        Standardization:
        - Transforms to mean=0, std=1
        - Makes features comparable in magnitude
        - Improves convergence for gradient-based ML algorithms
        - Prevents features with large scales from dominating

        Formula: z = (x - mean) / std

        Args:
            df: DataFrame with continuous columns
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            DataFrame with scaled features
        """
        result_df = df.copy()

        continuous_features = {
            'age': 'age_scaled',
            'tmb_score': 'tmb_score_scaled'
        }

        for col, feature_name in continuous_features.items():
            if col not in df.columns:
                continue

            # Create scaler for this column if needed
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()

            # Get values
            values = df[[col]].values

            if fit:
                # Fit and transform
                scaled_values = self.scalers[col].fit_transform(values)
            else:
                # Transform only
                if not hasattr(self.scalers[col], 'mean_'):
                    raise ValueError(f"Scaler for '{col}' not fitted. Set fit=True first.")
                scaled_values = self.scalers[col].transform(values)

            result_df[feature_name] = scaled_values

        return result_df

    # ============================================================
    # 4. DERIVED FEATURES - Create New Features from Combinations
    # ============================================================

    def create_mutation_burden(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total mutation burden from individual mutation flags.

        Medical Rationale:
        Individual mutations (TP53, KRAS, EGFR) are important drivers, but
        the total number of mutations may also predict:
        - Tumor aggressiveness
        - Treatment response
        - Prognosis

        This creates a simple count feature (0-3) representing how many
        of the key driver mutations are present.

        Args:
            df: DataFrame with mutation columns

        Returns:
            DataFrame with mutation_burden feature
        """
        result_df = df.copy()

        mutation_cols = ['tp53_mutation', 'kras_mutation', 'egfr_mutation']

        # Check all columns exist
        if not all(col in df.columns for col in mutation_cols):
            missing = [col for col in mutation_cols if col not in df.columns]
            raise ValueError(f"Missing mutation columns: {missing}")

        # Count mutations (True = 1, False = 0)
        result_df['mutation_burden'] = (
            df[mutation_cols].astype(int).sum(axis=1)
        )

        return result_df

    def create_clinical_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite clinical risk score from age and comorbidities.

        Medical Rationale:
        Age and comorbidity burden are two of the strongest predictors of
        cancer outcomes. Combining them creates a risk stratification score:
        - Older patients with more comorbidities have worse outcomes
        - Used to adjust treatment intensity
        - Helps predict surgical/treatment tolerance

        Formula: risk = (age/10) + (comorbidities * 5)
        This gives roughly equal weight to decades of age and comorbidity count.

        Score interpretation:
        - 0-10: Low risk
        - 10-20: Moderate risk
        - >20: High risk

        Args:
            df: DataFrame with age and comorbidity_count

        Returns:
            DataFrame with clinical_risk_score feature
        """
        result_df = df.copy()

        required_cols = ['age', 'comorbidity_count']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        result_df['clinical_risk_score'] = (
            (df['age'] / 10) + (df['comorbidity_count'] * 5)
        )

        return result_df

    def create_high_risk_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag patients with multiple high-risk factors.

        Medical Rationale:
        Identifies patients who meet multiple high-risk criteria:
        - Age > 70 (elderly patients have worse outcomes)
        - Comorbidity count >= 3 (multiple comorbidities reduce treatment tolerance)
        - TMB score > 20 (high mutation burden, may indicate aggressive tumor)

        Patients meeting 2+ criteria are flagged as high-risk.
        This can be used for:
        - Treatment stratification
        - Closer monitoring
        - Clinical trial eligibility

        Args:
            df: DataFrame with age, comorbidity_count, tmb_score

        Returns:
            DataFrame with high_risk_patient binary flag
        """
        result_df = df.copy()

        required_cols = ['age', 'comorbidity_count', 'tmb_score']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # Define risk criteria
        age_risk = df['age'] > 70
        comorbidity_risk = df['comorbidity_count'] >= 3
        tmb_risk = df['tmb_score'] > 20

        # Count risk factors
        risk_count = age_risk.astype(int) + comorbidity_risk.astype(int) + tmb_risk.astype(int)

        # Flag if 2+ risk factors
        result_df['high_risk_patient'] = (risk_count >= 2).astype(int)

        return result_df

    # ============================================================
    # 5. BINNING - Discretize Continuous Variables
    # ============================================================

    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin age into clinically meaningful groups.

        Medical Rationale:
        Age groups are often used in oncology because:
        - Treatment protocols may differ by age group
        - Risk stratification uses age cutoffs
        - Some effects are non-linear (e.g., very old vs old)

        Standard age groups for cancer:
        - Young adult: 18-40 (different tumor biology)
        - Middle age: 41-60 (peak working years)
        - Older adult: 61-75 (typical cancer age)
        - Elderly: 76+ (treatment tolerance concerns)

        Args:
            df: DataFrame with age column

        Returns:
            DataFrame with age_group categorical feature
        """
        result_df = df.copy()

        if 'age' not in df.columns:
            raise ValueError("Missing 'age' column")

        # Define age bins
        bins = [0, 40, 60, 75, 120]
        labels = ['18-40', '41-60', '61-75', '76+']

        result_df['age_group'] = pd.cut(
            df['age'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        return result_df

    def create_wbc_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize WBC count into clinical categories.

        Medical Rationale:
        White blood cell (WBC) count is categorized because:
        - Clinical decisions often use categorical cutoffs
        - Low WBC (leukopenia) increases infection risk
        - High WBC may indicate infection or leukemia
        - Normal range is well-established: 4-11 K/uL

        Categories:
        - Low: <4 K/uL (leukopenia - infection risk)
        - Normal: 4-11 K/uL
        - High: >11 K/uL (possible infection/hematologic malignancy)

        Args:
            df: DataFrame with wbc_count or wbc_imputed column

        Returns:
            DataFrame with wbc_category feature
        """
        result_df = df.copy()

        # Use imputed values if available, otherwise raw
        wbc_col = 'wbc_imputed' if 'wbc_imputed' in df.columns else 'wbc_count'

        if wbc_col not in df.columns:
            raise ValueError(f"Missing '{wbc_col}' column")

        # Define clinical cutoffs
        def categorize_wbc(value):
            if pd.isna(value):
                return 'Unknown'
            elif value < 4:
                return 'Low'
            elif value <= 11:
                return 'Normal'
            else:
                return 'High'

        result_df['wbc_category'] = df[wbc_col].apply(categorize_wbc)

        return result_df

    # ============================================================
    # PIPELINE - Apply All Transformations
    # ============================================================

    def transform_all(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Apply all feature transformations in correct order.

        Order matters:
        1. Imputation (creates imputed features needed by later steps)
        2. Scaling and encoding (independent transformations)
        3. Derived features (may use raw or transformed features)
        4. Binning (creates categorical versions)

        Args:
            df: Raw DataFrame
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            DataFrame with all engineered features
        """
        result_df = df.copy()

        # 1. Imputation
        result_df = self.impute_lab_values(result_df, strategy='median', fit=fit)

        # 2. Encoding and Scaling (independent, can be done in any order)
        result_df = self.encode_categorical(result_df, fit=fit)
        result_df = self.scale_continuous(result_df, fit=fit)

        # 3. Derived features
        result_df = self.create_mutation_burden(result_df)
        result_df = self.create_clinical_risk_score(result_df)
        result_df = self.create_high_risk_flag(result_df)

        # 4. Binning
        result_df = self.create_age_groups(result_df)
        result_df = self.create_wbc_categories(result_df)

        if fit:
            self.fitted = True

        return result_df


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_feature_lineage() -> Dict[str, List[str]]:
    """
    Get lineage information showing which raw columns created which features.

    This is critical for:
    - Understanding feature dependencies
    - Debugging feature issues
    - Impact analysis when raw data changes
    - Documentation and reproducibility

    Returns:
        Dictionary mapping feature name to source column(s)
    """
    return {
        # Imputed features
        'wbc_imputed': ['wbc_count'],
        'hemoglobin_imputed': ['hemoglobin'],
        'platelet_imputed': ['platelet_count'],

        # Encoded features
        'sex_encoded': ['sex'],
        'ethnicity_encoded': ['ethnicity'],

        # Scaled features
        'age_scaled': ['age'],
        'tmb_score_scaled': ['tmb_score'],

        # Derived features
        'mutation_burden': ['tp53_mutation', 'kras_mutation', 'egfr_mutation'],
        'clinical_risk_score': ['age', 'comorbidity_count'],
        'high_risk_patient': ['age', 'comorbidity_count', 'tmb_score'],

        # Binned features
        'age_group': ['age'],
        'wbc_category': ['wbc_count']  # or wbc_imputed
    }


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get human-readable descriptions of all features.

    Returns:
        Dictionary mapping feature name to description
    """
    return {
        'wbc_imputed': 'White blood cell count with median imputation for missing values',
        'hemoglobin_imputed': 'Hemoglobin level with median imputation for missing values',
        'platelet_imputed': 'Platelet count with median imputation for missing values',
        'sex_encoded': 'One-hot encoded sex (reference: Male)',
        'ethnicity_encoded': 'One-hot encoded ethnicity (reference: first category)',
        'age_scaled': 'Age standardized to mean=0, std=1',
        'tmb_score_scaled': 'Tumor mutational burden score standardized to mean=0, std=1',
        'mutation_burden': 'Count of driver mutations (TP53, KRAS, EGFR): range 0-3',
        'clinical_risk_score': 'Composite risk score: (age/10) + (comorbidities*5)',
        'high_risk_patient': 'Binary flag: 1 if patient meets 2+ high-risk criteria',
        'age_group': 'Age binned into clinical categories: 18-40, 41-60, 61-75, 76+',
        'wbc_category': 'WBC count categorized as Low (<4), Normal (4-11), High (>11)'
    }


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module Loaded")
    print(f"\nAvailable features: {len(get_feature_lineage())}")
    print("\nFeature lineage:")
    for feature, sources in get_feature_lineage().items():
        print(f"  {feature}: {sources}")
