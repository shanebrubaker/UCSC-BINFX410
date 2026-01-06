"""
Synthetic Clinical Data Generator

Creates realistic clinical/genomic datasets for educational purposes.
Includes intentional data quality issues to demonstrate validation.
"""

import pandas as pd
import numpy as np
from typing import Optional
import random


class ClinicalDataGenerator:
    """
    Generate synthetic patient data with clinical and genomic features.

    This generator creates realistic datasets that include:
    - Demographics (age, sex, ethnicity)
    - Clinical features (diagnosis, comorbidities, treatment response)
    - Genomic features (mutations, TMB, MSI status)
    - Lab values (WBC, hemoglobin, platelets) with realistic missing patterns
    - Outcomes (survival, response status)

    Data quality issues are intentionally introduced to demonstrate validation:
    - Missing values in lab data (~10-15%)
    - Inconsistent formatting (e.g., "Male"/"M"/"male")
    - Occasional outliers
    - Some invalid values that should be caught by validation
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_demographics(self, n: int) -> pd.DataFrame:
        """
        Generate demographic features.

        Args:
            n: Number of patients to generate

        Returns:
            DataFrame with demographic columns
        """
        # Age distribution skewed toward older patients (cancer is age-related)
        ages = np.random.gamma(shape=8, scale=8, size=n).astype(int)
        ages = np.clip(ages, 18, 95)  # Adult population

        # Sex with intentional formatting inconsistencies
        sex_options = ["Male", "Female", "M", "F", "male", "female"]
        sex = np.random.choice(sex_options, size=n, p=[0.35, 0.35, 0.1, 0.1, 0.05, 0.05])

        # Ethnicity distribution
        ethnicity_options = ["Caucasian", "African American", "Asian", "Hispanic", "Other"]
        ethnicity = np.random.choice(
            ethnicity_options,
            size=n,
            p=[0.6, 0.15, 0.12, 0.1, 0.03]
        )

        return pd.DataFrame({
            'age': ages,
            'sex': sex,
            'ethnicity': ethnicity
        })

    def generate_clinical_features(self, n: int, ages: np.ndarray) -> pd.DataFrame:
        """
        Generate clinical features including diagnosis and comorbidities.

        Args:
            n: Number of patients
            ages: Age array to correlate comorbidities

        Returns:
            DataFrame with clinical columns
        """
        # Cancer diagnoses (common solid tumors)
        diagnosis_options = [
            "Lung Cancer", "Breast Cancer", "Colorectal Cancer",
            "Prostate Cancer", "Melanoma", "Pancreatic Cancer",
            "Gastric Cancer", "Ovarian Cancer"
        ]
        diagnosis = np.random.choice(diagnosis_options, size=n)

        # Comorbidity count increases with age
        base_comorbidities = np.random.poisson(lam=1.5, size=n)
        age_effect = ((ages - 40) / 20).clip(0, 3).astype(int)
        comorbidity_count = (base_comorbidities + age_effect).clip(0, 10)

        # Treatment response (realistic distribution)
        response_options = [
            "Complete Response", "Partial Response",
            "Stable Disease", "Progressive Disease"
        ]
        treatment_response = np.random.choice(
            response_options,
            size=n,
            p=[0.15, 0.35, 0.30, 0.20]
        )

        # Response status (binary outcome)
        response_status = np.where(
            np.isin(treatment_response, ["Complete Response", "Partial Response"]),
            1, 0
        )

        return pd.DataFrame({
            'diagnosis': diagnosis,
            'comorbidity_count': comorbidity_count,
            'treatment_response': treatment_response,
            'response_status': response_status
        })

    def generate_genomic_features(self, n: int) -> pd.DataFrame:
        """
        Generate genomic features including mutations and biomarkers.

        These represent common oncogenic drivers and biomarkers used in
        precision oncology to guide treatment decisions.

        Args:
            n: Number of patients

        Returns:
            DataFrame with genomic columns
        """
        # TP53 mutation - most common cancer mutation (~50% across cancers)
        tp53_mutation = np.random.choice([True, False], size=n, p=[0.5, 0.5])

        # KRAS mutation - common in lung, colon, pancreatic (~25%)
        kras_mutation = np.random.choice([True, False], size=n, p=[0.25, 0.75])

        # EGFR mutation - important in lung cancer (~15%)
        egfr_mutation = np.random.choice([True, False], size=n, p=[0.15, 0.85])

        # Tumor Mutational Burden (TMB) - mutations per megabase
        # Higher TMB may predict response to immunotherapy
        # Log-normal distribution with mean ~10 mutations/Mb
        tmb_score = np.random.lognormal(mean=2, sigma=0.8, size=n)
        tmb_score = np.clip(tmb_score, 0, 100)

        # Microsatellite Instability (MSI) status
        # MSI-H (~15%) predicts immunotherapy response
        msi_options = ["MSI-H", "MSI-L", "MSS"]
        msi_status = np.random.choice(msi_options, size=n, p=[0.15, 0.10, 0.75])

        return pd.DataFrame({
            'tp53_mutation': tp53_mutation,
            'kras_mutation': kras_mutation,
            'egfr_mutation': egfr_mutation,
            'tmb_score': tmb_score,
            'msi_status': msi_status
        })

    def generate_lab_values(self, n: int) -> pd.DataFrame:
        """
        Generate lab values with realistic missing data patterns.

        Lab values often have missing data in real clinical datasets due to:
        - Tests not ordered for all patients
        - Sample quality issues
        - Equipment failures

        Args:
            n: Number of patients

        Returns:
            DataFrame with lab value columns
        """
        # WBC count (normal: 4-11 K/uL, but cancer patients often abnormal)
        wbc_mean = 8.5
        wbc_std = 3.5
        wbc_count = np.random.normal(loc=wbc_mean, scale=wbc_std, size=n)
        wbc_count = np.clip(wbc_count, 0.5, 50)

        # Hemoglobin (normal: 12-17 g/dL, anemia common in cancer)
        hgb_mean = 11.5
        hgb_std = 2.5
        hemoglobin = np.random.normal(loc=hgb_mean, scale=hgb_std, size=n)
        hemoglobin = np.clip(hemoglobin, 4, 20)

        # Platelet count (normal: 150-400 K/uL)
        plt_mean = 250
        plt_std = 80
        platelet_count = np.random.normal(loc=plt_mean, scale=plt_std, size=n)
        platelet_count = np.clip(platelet_count, 10, 800)

        # Introduce realistic missing data patterns (10-15% missing)
        # Missing not completely at random - slightly more missing in older/sicker patients
        missing_prob_base = 0.10

        wbc_missing_mask = np.random.random(n) < (missing_prob_base + 0.03)
        hgb_missing_mask = np.random.random(n) < (missing_prob_base + 0.02)
        plt_missing_mask = np.random.random(n) < (missing_prob_base + 0.04)

        wbc_count[wbc_missing_mask] = np.nan
        hemoglobin[hgb_missing_mask] = np.nan
        platelet_count[plt_missing_mask] = np.nan

        # Add occasional outliers (1-2%)
        outlier_mask = np.random.random(n) < 0.015
        wbc_count[outlier_mask] = np.random.uniform(30, 45, size=outlier_mask.sum())

        return pd.DataFrame({
            'wbc_count': wbc_count,
            'hemoglobin': hemoglobin,
            'platelet_count': platelet_count
        })

    def generate_outcomes(self, n: int, ages: np.ndarray,
                         comorbidities: np.ndarray,
                         treatment_response: np.ndarray) -> pd.DataFrame:
        """
        Generate outcome variables (survival time).

        Survival is correlated with age, comorbidities, and treatment response
        to create realistic relationships.

        Args:
            n: Number of patients
            ages: Patient ages
            comorbidities: Comorbidity counts
            treatment_response: Treatment response categories

        Returns:
            DataFrame with outcome columns
        """
        # Base survival time from exponential distribution
        base_survival = np.random.exponential(scale=24, size=n)

        # Adjust for age (older patients have worse outcomes)
        age_effect = -0.2 * (ages - 60) / 10

        # Adjust for comorbidities
        comorbidity_effect = -2 * comorbidities

        # Adjust for treatment response (strong effect)
        response_effect = np.zeros(n)
        response_effect[treatment_response == "Complete Response"] = 18
        response_effect[treatment_response == "Partial Response"] = 8
        response_effect[treatment_response == "Stable Disease"] = 0
        response_effect[treatment_response == "Progressive Disease"] = -10

        # Combine effects
        survival_months = base_survival + age_effect + comorbidity_effect + response_effect
        survival_months = np.clip(survival_months, 1, 180)

        return pd.DataFrame({
            'survival_months': survival_months
        })

    def generate_dataset(self, n_patients: int = 1000,
                        introduce_errors: bool = True) -> pd.DataFrame:
        """
        Generate complete synthetic clinical dataset.

        Args:
            n_patients: Number of patients to generate
            introduce_errors: Whether to add intentional data quality issues

        Returns:
            Complete DataFrame with all features
        """
        # Generate patient IDs
        patient_ids = [f"PT{i:05d}" for i in range(1, n_patients + 1)]

        # Generate each feature group
        demographics = self.generate_demographics(n_patients)
        clinical = self.generate_clinical_features(n_patients, demographics['age'].values)
        genomic = self.generate_genomic_features(n_patients)
        labs = self.generate_lab_values(n_patients)
        outcomes = self.generate_outcomes(
            n_patients,
            demographics['age'].values,
            clinical['comorbidity_count'].values,
            clinical['treatment_response'].values
        )

        # Combine all features
        df = pd.DataFrame({'patient_id': patient_ids})
        df = pd.concat([df, demographics, clinical, genomic, labs, outcomes], axis=1)

        if introduce_errors:
            df = self._introduce_data_quality_issues(df)

        return df

    def _introduce_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intentionally introduce data quality issues for teaching purposes.

        These issues should be caught by validation:
        - Invalid age values
        - Out-of-range lab values
        - Inconsistent categorical values

        Args:
            df: Clean dataframe

        Returns:
            DataFrame with intentional errors
        """
        df = df.copy()

        # Invalid age (1% of patients)
        invalid_age_mask = np.random.random(len(df)) < 0.01
        df.loc[invalid_age_mask, 'age'] = np.random.choice(
            [150, -5, 999],
            size=invalid_age_mask.sum()
        )

        # Out-of-range TMB score (0.5% of patients)
        invalid_tmb_mask = np.random.random(len(df)) < 0.005
        df.loc[invalid_tmb_mask, 'tmb_score'] = np.random.uniform(150, 200,
                                                                   size=invalid_tmb_mask.sum())

        # Invalid comorbidity count (0.5% of patients)
        invalid_comorb_mask = np.random.random(len(df)) < 0.005
        df.loc[invalid_comorb_mask, 'comorbidity_count'] = 15

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save dataset to CSV file.

        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Missing data summary:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


if __name__ == "__main__":
    # Example usage
    generator = ClinicalDataGenerator(seed=42)
    data = generator.generate_dataset(n_patients=1000)
    generator.save_dataset(data, "data/raw/synthetic_patients.csv")
