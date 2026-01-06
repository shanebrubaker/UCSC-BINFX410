"""
Utility functions for the feature store.

Helper functions that don't belong to a specific class but are
used across the project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import yaml
import json
from pathlib import Path


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_metadata(metadata: Dict, filepath: str) -> None:
    """
    Save metadata to JSON file.

    Args:
        metadata: Dictionary to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(filepath: str) -> Dict:
    """
    Load metadata from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with metadata
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def split_train_test(df: pd.DataFrame,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify_col: Optional[str] = None) -> tuple:
    """
    Split DataFrame into train and test sets.

    Args:
        df: DataFrame to split
        test_size: Proportion for test set
        random_state: Random seed
        stratify_col: Optional column for stratified split

    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col]
    else:
        stratify = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return train_df, test_df


def calculate_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Works with scikit-learn models that have feature_importances_ or coef_ attributes.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for all features in a DataFrame.

    Args:
        df: DataFrame to summarize

    Returns:
        DataFrame with summary statistics
    """
    summary_data = []

    for col in df.columns:
        col_data = {
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isnull().sum()),
            'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
            'unique_count': int(df[col].nunique())
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_data['mean'] = float(df[col].mean())
            col_data['std'] = float(df[col].std())
            col_data['min'] = float(df[col].min())
            col_data['max'] = float(df[col].max())
        else:
            col_data['mean'] = None
            col_data['std'] = None
            col_data['min'] = None
            col_data['max'] = None

        summary_data.append(col_data)

    return pd.DataFrame(summary_data)


def validate_patient_ids(patient_ids: List[str],
                        valid_ids: List[str]) -> tuple:
    """
    Validate that patient IDs exist in the dataset.

    Args:
        patient_ids: Patient IDs to validate
        valid_ids: List of valid patient IDs

    Returns:
        Tuple of (valid_ids, invalid_ids)
    """
    valid_set = set(valid_ids)
    input_set = set(patient_ids)

    valid = list(input_set & valid_set)
    invalid = list(input_set - valid_set)

    return valid, invalid


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    print("Utilities module loaded")
