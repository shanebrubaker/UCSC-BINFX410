"""Plotting helpers for genomics data visualization in Spark exercises.

Provides consistent, publication-quality plots for variant data analysis
including chromosome distributions, quality histograms, and performance
comparisons.

Usage:
    from viz_utils import plot_chromosome_distribution, plot_quality_histogram
"""

from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Consistent chromosome order for all plots
CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def _sort_by_chrom(df: pd.DataFrame, chrom_col: str = "CHROM") -> pd.DataFrame:
    """Sort a DataFrame by chromosome in biological order."""
    chrom_cat = pd.CategoricalDtype(categories=CHROM_ORDER, ordered=True)
    df = df.copy()
    df[chrom_col] = df[chrom_col].astype(chrom_cat)
    return df.sort_values(chrom_col)


def plot_chromosome_distribution(
    df: pd.DataFrame,
    value_col: str,
    chrom_col: str = "CHROM",
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Bar plot of values by chromosome, ordered chr1-chr22, chrX, chrY.

    Args:
        df: pandas DataFrame with chromosome and value columns.
        value_col: Column name for y-axis values.
        chrom_col: Column name for chromosome (default 'CHROM').
        title: Plot title (auto-generated if None).
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    sorted_df = _sort_by_chrom(df, chrom_col)

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", n_colors=len(sorted_df))
    ax.bar(range(len(sorted_df)), sorted_df[value_col], color=colors)
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df[chrom_col], rotation=45, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} by Chromosome")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_quality_histogram(
    df: pd.DataFrame,
    qual_col: str = "QUAL",
    bins: int = 50,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Distribution of quality scores with vertical line at QUAL=30 threshold.

    Args:
        df: pandas DataFrame with quality column.
        qual_col: Column name for quality scores.
        bins: Number of histogram bins.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[qual_col].dropna(), bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=30, color="red", linestyle="--", linewidth=2, label="QUAL=30 threshold")
    ax.set_xlabel("Quality Score (QUAL)")
    ax.set_ylabel("Count")
    ax.set_title("Variant Quality Score Distribution")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_af_distribution(
    df: pd.DataFrame,
    af_col: str = "AF",
    log_scale: bool = True,
    bins: int = 50,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Allele frequency spectrum plot.

    Args:
        df: pandas DataFrame with allele frequency column.
        af_col: Column name for allele frequency.
        log_scale: Whether to use log scale on y-axis.
        bins: Number of histogram bins.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[af_col].dropna(), bins=bins, color="darkorange", edgecolor="white", alpha=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Allele Frequency")
    ax.set_ylabel("Count" + (" (log scale)" if log_scale else ""))
    ax.set_title("Allele Frequency Spectrum")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_performance_comparison(
    timing_results: pd.DataFrame,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Bar chart comparing operation times.

    Args:
        timing_results: DataFrame with columns 'operation' and 'mean_time'.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("Set2", n_colors=len(timing_results))
    bars = ax.barh(
        timing_results["operation"],
        timing_results["mean_time"],
        color=colors,
        edgecolor="white",
    )

    for bar, val in zip(bars, timing_results["mean_time"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center")

    ax.set_xlabel("Time (seconds)")
    ax.set_title("Performance Comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_filter_impact(
    before_counts: Dict[str, int],
    after_counts: Dict[str, int],
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Side-by-side bar chart showing variant counts before and after filtering.

    Args:
        before_counts: Dict of category -> count before filtering.
        after_counts: Dict of category -> count after filtering.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    categories = list(before_counts.keys())
    before_vals = [before_counts[c] for c in categories]
    after_vals = [after_counts.get(c, 0) for c in categories]

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(categories))
    width = 0.35
    ax.bar([i - width / 2 for i in x], before_vals, width, label="Before filtering", color="steelblue")
    ax.bar([i + width / 2 for i in x], after_vals, width, label="After filtering", color="darkorange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Variant Count")
    ax.set_title("Impact of Quality Filtering")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_consequence_distribution(
    df: pd.DataFrame,
    consequence_col: str = "CONSEQUENCE",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Horizontal bar chart of variant consequence types.

    Args:
        df: pandas DataFrame with consequence column.
        consequence_col: Column name for consequence type.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    counts = df[consequence_col].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", n_colors=len(counts))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Variant Consequence Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("viz_utils - Genomics visualization utilities")
    print("Import this module in your notebook:")
    print("  from viz_utils import plot_chromosome_distribution, plot_quality_histogram")
