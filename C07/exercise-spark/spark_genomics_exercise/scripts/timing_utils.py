"""Reusable timing and performance utilities for Spark exercises.

Provides helper functions for measuring Spark operation performance,
comparing multiple approaches, and inspecting execution plans.

Usage:
    from timing_utils import time_operation, compare_operations, partition_info
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd


def time_operation(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Time a Spark operation and return (result, duration_seconds).

    If the result is a Spark DataFrame, forces execution by calling .count()
    to ensure actual computation time is measured, not just plan creation.

    Args:
        func: Callable to time.
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        Tuple of (result, duration_in_seconds).
    """
    start = time.time()
    result = func(*args, **kwargs)

    # Force Spark action if result is a DataFrame
    try:
        from pyspark.sql import DataFrame
        if isinstance(result, DataFrame):
            result.count()
    except ImportError:
        pass

    duration = time.time() - start
    return result, duration


def compare_operations(
    operations: Dict[str, Callable], num_runs: int = 3
) -> pd.DataFrame:
    """Time multiple operations and return a comparison DataFrame.

    Runs each operation num_runs times and reports min, max, and mean times.

    Args:
        operations: Dict mapping operation names to callables.
        num_runs: Number of times to run each operation (default 3).

    Returns:
        pandas DataFrame with columns: operation, min_time, max_time, mean_time.
    """
    results = []
    for name, func in operations.items():
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = func()
            # Force Spark action if result is a DataFrame
            try:
                from pyspark.sql import DataFrame
                if isinstance(result, DataFrame):
                    result.count()
            except ImportError:
                pass
            times.append(time.time() - start)

        results.append({
            "operation": name,
            "min_time": min(times),
            "max_time": max(times),
            "mean_time": sum(times) / len(times),
        })
        print(f"  {name}: {sum(times)/len(times):.3f}s (mean of {num_runs} runs)")

    return pd.DataFrame(results)


def spark_ui_link(spark: Any) -> str:
    """Return and print the clickable link to the Spark UI.

    Args:
        spark: SparkSession instance.

    Returns:
        URL string for the Spark UI.
    """
    ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
    print(f"Spark UI: {ui_url}")
    return ui_url


def partition_info(df: Any) -> Dict[str, Any]:
    """Display and return partition count and estimated size info.

    Args:
        df: Spark DataFrame to inspect.

    Returns:
        Dict with num_partitions, total_rows, approx_rows_per_partition.
    """
    num_partitions = df.rdd.getNumPartitions()
    total_rows = df.count()
    rows_per_partition = total_rows / max(num_partitions, 1)

    info = {
        "num_partitions": num_partitions,
        "total_rows": total_rows,
        "approx_rows_per_partition": int(rows_per_partition),
    }

    print(f"Partitions:          {num_partitions}")
    print(f"Total rows:          {total_rows:,}")
    print(f"~Rows per partition: {int(rows_per_partition):,}")

    return info


def explain_plan(df: Any, mode: str = "simple") -> None:
    """Pretty-print the Spark execution plan.

    Args:
        df: Spark DataFrame to explain.
        mode: One of 'simple', 'extended', 'codegen', 'cost', 'formatted'.
    """
    valid_modes = {"simple", "extended", "codegen", "cost", "formatted"}
    if mode not in valid_modes:
        print(f"Invalid mode '{mode}'. Choose from: {valid_modes}")
        return

    if mode == "simple":
        df.explain(False)
    elif mode == "extended":
        df.explain(True)
    else:
        df.explain(mode=mode)


def memory_usage(df: Any) -> str:
    """Estimate memory usage of a Spark DataFrame as a human-readable string.

    Note: This is an approximation based on sampling.

    Args:
        df: Spark DataFrame to estimate.

    Returns:
        Human-readable size string.
    """
    # Sample-based estimation
    sample_fraction = min(1.0, 1000 / max(df.count(), 1))
    sample_size = df.sample(False, sample_fraction).toPandas().memory_usage(deep=True).sum()
    estimated_total = sample_size / max(sample_fraction, 0.001)

    if estimated_total < 1024:
        size_str = f"{estimated_total:.0f} B"
    elif estimated_total < 1024 ** 2:
        size_str = f"{estimated_total / 1024:.1f} KB"
    elif estimated_total < 1024 ** 3:
        size_str = f"{estimated_total / 1024**2:.1f} MB"
    else:
        size_str = f"{estimated_total / 1024**3:.1f} GB"

    print(f"Estimated memory usage: {size_str}")
    return size_str


if __name__ == "__main__":
    print("timing_utils - Spark performance measurement utilities")
    print("Import this module in your notebook:")
    print("  from timing_utils import time_operation, compare_operations")
