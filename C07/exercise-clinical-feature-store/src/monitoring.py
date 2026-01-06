"""
Data Quality Monitoring and Reporting

Tracks data quality metrics over time including:
- Missing data rates
- Distribution shifts
- Validation failures
- Feature correlations

Generates HTML reports for visualization and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json


class DataQualityMonitor:
    """
    Monitor and report on data quality metrics.

    Why monitor data quality?
    - Detect data drift (distribution changes over time)
    - Catch pipeline failures early
    - Track data quality trends
    - Support debugging and root cause analysis
    """

    def __init__(self, feature_store=None):
        """
        Initialize monitor.

        Args:
            feature_store: Optional FeatureStore instance for accessing data
        """
        self.feature_store = feature_store
        self.metrics_history = []

    def compute_missing_data_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute missing data statistics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing data metrics
        """
        total_rows = len(df)
        missing_stats = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_rate = missing_count / total_rows if total_rows > 0 else 0

            missing_stats[col] = {
                'missing_count': int(missing_count),
                'missing_rate': float(missing_rate),
                'total_rows': total_rows
            }

        return missing_stats

    def compute_distribution_metrics(self, df: pd.DataFrame,
                                    numeric_cols: Optional[List[str]] = None) -> Dict:
        """
        Compute distribution statistics for numeric columns.

        These metrics help detect distribution drift:
        - Mean/median shift indicates central tendency change
        - Std change indicates spread change
        - Quantiles show tail behavior

        Args:
            df: DataFrame to analyze
            numeric_cols: Optional list of numeric columns (None = auto-detect)

        Returns:
            Dictionary with distribution metrics
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        dist_stats = {}

        for col in numeric_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna()

            if len(values) == 0:
                dist_stats[col] = {'error': 'No non-null values'}
                continue

            dist_stats[col] = {
                'count': int(len(values)),
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75)),
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis())
            }

        return dist_stats

    def compute_categorical_metrics(self, df: pd.DataFrame,
                                   categorical_cols: Optional[List[str]] = None) -> Dict:
        """
        Compute statistics for categorical columns.

        Args:
            df: DataFrame to analyze
            categorical_cols: Optional list of categorical columns

        Returns:
            Dictionary with categorical metrics
        """
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        cat_stats = {}

        for col in categorical_cols:
            if col not in df.columns:
                continue

            value_counts = df[col].value_counts()
            total = len(df[col].dropna())

            cat_stats[col] = {
                'unique_values': int(df[col].nunique()),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'most_common_pct': float(value_counts.iloc[0] / total * 100) if total > 0 else 0,
                'distribution': value_counts.to_dict()
            }

        return cat_stats

    def compute_correlation_metrics(self, df: pd.DataFrame,
                                   numeric_cols: Optional[List[str]] = None,
                                   threshold: float = 0.7) -> Dict:
        """
        Compute feature correlations and flag high correlations.

        High correlations can indicate:
        - Redundant features (multicollinearity)
        - Data leakage
        - Derived features that are too similar to source

        Args:
            df: DataFrame to analyze
            numeric_cols: Optional list of numeric columns
            threshold: Correlation threshold for flagging

        Returns:
            Dictionary with correlation metrics
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to available columns
        available_cols = [col for col in numeric_cols if col in df.columns]

        if len(available_cols) < 2:
            return {'error': 'Need at least 2 numeric columns'}

        # Compute correlation matrix
        corr_matrix = df[available_cols].corr()

        # Find high correlations (excluding diagonal)
        high_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })

        return {
            'high_correlations': high_correlations,
            'correlation_matrix': corr_matrix.to_dict()
        }

    def generate_quality_report(self, df: pd.DataFrame,
                               report_name: str = "data_quality_report",
                               output_dir: str = "reports") -> str:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze
            report_name: Name for report file
            output_dir: Directory to save report

        Returns:
            Path to generated report
        """
        # Compute all metrics
        print("Computing data quality metrics...")

        missing_metrics = self.compute_missing_data_metrics(df)
        distribution_metrics = self.compute_distribution_metrics(df)
        categorical_metrics = self.compute_categorical_metrics(df)
        correlation_metrics = self.compute_correlation_metrics(df)

        # Generate HTML report
        html = self._generate_html_report(
            df=df,
            missing_metrics=missing_metrics,
            distribution_metrics=distribution_metrics,
            categorical_metrics=categorical_metrics,
            correlation_metrics=correlation_metrics,
            report_name=report_name
        )

        # Save report
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        report_file = output_path / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(report_file, 'w') as f:
            f.write(html)

        print(f"Report saved to: {report_file}")

        # Save metrics as JSON for programmatic access
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'missing_data': missing_metrics,
            'distributions': distribution_metrics,
            'categorical': categorical_metrics,
            'correlations': correlation_metrics
        }

        json_file = output_path / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return str(report_file)

    def _generate_html_report(self, df: pd.DataFrame,
                             missing_metrics: Dict,
                             distribution_metrics: Dict,
                             categorical_metrics: Dict,
                             correlation_metrics: Dict,
                             report_name: str) -> str:
        """
        Generate HTML report content.

        Args:
            df: DataFrame being analyzed
            missing_metrics: Missing data metrics
            distribution_metrics: Distribution metrics
            categorical_metrics: Categorical metrics
            correlation_metrics: Correlation metrics
            report_name: Report name

        Returns:
            HTML string
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric {{
            background-color: #e7f3e7;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .error {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .stat-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }}
        .stat-label {{
            font-weight: bold;
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            color: #333;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Quality Report: {report_name}</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Dataset Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>

        <h2>1. Overview Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Rows</div>
                <div class="stat-value">{df.shape[0]:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Columns</div>
                <div class="stat-value">{df.shape[1]}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Numeric Columns</div>
                <div class="stat-value">{len(df.select_dtypes(include=[np.number]).columns)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Categorical Columns</div>
                <div class="stat-value">{len(df.select_dtypes(include=['object']).columns)}</div>
            </div>
        </div>

        <h2>2. Missing Data Analysis</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Missing Count</th>
                <th>Missing Rate (%)</th>
                <th>Status</th>
            </tr>
        """

        # Missing data table
        for col, metrics in missing_metrics.items():
            missing_rate = metrics['missing_rate'] * 100
            status_class = ""
            status = "OK"

            if missing_rate > 20:
                status_class = "error"
                status = "HIGH"
            elif missing_rate > 10:
                status_class = "warning"
                status = "MODERATE"

            html += f"""
            <tr>
                <td>{col}</td>
                <td>{metrics['missing_count']}</td>
                <td>{missing_rate:.2f}%</td>
                <td class="{status_class}">{status}</td>
            </tr>
            """

        html += """
        </table>

        <h2>3. Distribution Statistics (Numeric Features)</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
        """

        # Distribution table
        for col, metrics in distribution_metrics.items():
            if 'error' in metrics:
                continue

            html += f"""
            <tr>
                <td>{col}</td>
                <td>{metrics['count']}</td>
                <td>{metrics['mean']:.3f}</td>
                <td>{metrics['median']:.3f}</td>
                <td>{metrics['std']:.3f}</td>
                <td>{metrics['min']:.3f}</td>
                <td>{metrics['max']:.3f}</td>
            </tr>
            """

        html += """
        </table>

        <h2>4. Categorical Feature Distribution</h2>
        """

        # Categorical summaries
        for col, metrics in categorical_metrics.items():
            html += f"""
            <div class="metric">
                <strong>{col}</strong><br>
                Unique Values: {metrics['unique_values']}<br>
                Most Common: {metrics['most_common']} ({metrics['most_common_pct']:.1f}%)
            </div>
            """

        html += """
        <h2>5. Feature Correlations</h2>
        """

        # High correlations
        if correlation_metrics.get('high_correlations'):
            html += """
            <div class="metric warning">
                <strong>High Correlations Detected</strong>
            </div>
            <table>
                <tr>
                    <th>Feature 1</th>
                    <th>Feature 2</th>
                    <th>Correlation</th>
                </tr>
            """

            for corr in correlation_metrics['high_correlations']:
                html += f"""
                <tr>
                    <td>{corr['feature1']}</td>
                    <td>{corr['feature2']}</td>
                    <td>{corr['correlation']:.3f}</td>
                </tr>
                """

            html += "</table>"
        else:
            html += """
            <div class="metric">
                <strong>No high correlations detected (threshold: 0.7)</strong>
            </div>
            """

        html += """
    </div>
</body>
</html>
        """

        return html

    def compare_distributions(self, df1: pd.DataFrame, df2: pd.DataFrame,
                            label1: str = "Dataset 1",
                            label2: str = "Dataset 2") -> Dict:
        """
        Compare distributions between two datasets to detect drift.

        Args:
            df1: First dataset
            df2: Second dataset
            label1: Label for first dataset
            label2: Label for second dataset

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'label1': label1,
            'label2': label2,
            'timestamp': datetime.now().isoformat(),
            'differences': {}
        }

        # Get common numeric columns
        numeric_cols = set(df1.select_dtypes(include=[np.number]).columns) & \
                      set(df2.select_dtypes(include=[np.number]).columns)

        for col in numeric_cols:
            vals1 = df1[col].dropna()
            vals2 = df2[col].dropna()

            if len(vals1) == 0 or len(vals2) == 0:
                continue

            mean_diff = abs(vals1.mean() - vals2.mean())
            std_diff = abs(vals1.std() - vals2.std())

            # Flag as drift if mean changes by >20% or std changes by >30%
            mean_drift = mean_diff / vals1.mean() if vals1.mean() != 0 else 0
            std_drift = std_diff / vals1.std() if vals1.std() != 0 else 0

            comparison['differences'][col] = {
                f'{label1}_mean': float(vals1.mean()),
                f'{label2}_mean': float(vals2.mean()),
                'mean_difference': float(mean_diff),
                'mean_drift_pct': float(mean_drift * 100),
                f'{label1}_std': float(vals1.std()),
                f'{label2}_std': float(vals2.std()),
                'std_difference': float(std_diff),
                'std_drift_pct': float(std_drift * 100),
                'drift_detected': mean_drift > 0.2 or std_drift > 0.3
            }

        return comparison


if __name__ == "__main__":
    # Example usage
    print("Data Quality Monitoring Module Loaded")
