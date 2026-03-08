# Spark Genomics Exercise Module

## Overview

This module teaches distributed data processing with Apache Spark applied to genomics
data. Students analyze synthetic variant data (VCF-like format) using PySpark, learning
core concepts of distributed computing through hands-on exercises.

Everything runs locally via Docker - no cloud resources needed.

### Learning Objectives

1. Understand distributed data processing concepts (partitioning, shuffling, lazy evaluation)
2. Use Spark DataFrame API for genomics data analysis
3. Apply quality control filters to variant data
4. Perform aggregations and joins on large-scale genomic datasets
5. Optimize Spark queries and understand performance trade-offs
6. Compare Spark vs. pandas for different data scales

## Prerequisites

- Basic Python programming (functions, loops, pandas familiarity)
- Basic genomics knowledge (variants, chromosomes, VCF format concept)
- Docker Desktop installed ([download](https://www.docker.com/products/docker-desktop/))
- 8 GB RAM minimum (16 GB recommended)
- 10 GB free disk space

## Quick Start

```bash
# 1. Navigate to the exercise directory
cd spark_genomics_exercise

# 2. Start the Docker environment
cd docker
docker compose up --build

# 3. Open Jupyter Lab in your browser
#    Visit: http://localhost:8888

# 4. In Jupyter, open a terminal and generate data:
cd /workspace/data
python generate_data.py --size small
python generate_data.py --size medium

# 5. Start with notebook 00_setup_and_test.ipynb
```

To stop: press `Ctrl+C` in the terminal running docker-compose, or run `docker-compose down`.

## Directory Structure

```
spark_genomics_exercise/
├── README.md                          # This file
├── docker/
│   ├── Dockerfile                     # Container with PySpark + dependencies
│   └── docker-compose.yml             # One-command startup
├── data/
│   ├── generate_data.py               # Script to create synthetic datasets
│   ├── README.md                      # Data schema documentation
│   └── .gitkeep
├── notebooks/
│   ├── 00_setup_and_test.ipynb        # Verify installation works
│   ├── 01_spark_basics.ipynb          # Introduction to Spark concepts
│   ├── 02_variant_qc_exercise.ipynb   # Main exercise (student version)
│   ├── 02_variant_qc_solutions.ipynb  # Solutions (instructor only)
│   └── 03_advanced_concepts.ipynb     # Bonus material
├── scripts/
│   ├── timing_utils.py                # Performance measurement helpers
│   └── viz_utils.py                   # Plotting helpers
├── exercises/
│   ├── exercise_1_filtering.md        # Written exercises: filtering
│   ├── exercise_2_aggregation.md      # Written exercises: aggregation
│   └── exercise_3_optimization.md     # Written exercises: optimization
└── instructor/
    ├── teaching_notes.md              # Lecture outline and tips
    ├── grading_rubric.md              # Assessment criteria
    └── common_issues.md               # Troubleshooting FAQ
```

## Exercise Structure

### Notebook 00: Setup and Test (~15 min)
Verify your environment works: test imports, create a SparkSession, run basic
operations, check the Spark UI.

### Notebook 01: Spark Fundamentals (~90 min)
Learn core concepts through guided examples:
- Lazy evaluation and query plans
- Partitions and data distribution
- Core operations (filter, select, groupBy, join)
- Understanding shuffles
- Caching strategies

### Notebook 02: Variant QC Exercise (~2.5 hours)
**This is the main graded exercise.** 8 tasks covering:
- Data loading and exploration
- Quality filtering (GATK hard filters)
- Per-chromosome statistics
- Allele frequency stratification
- Gene annotation joins
- Spark vs. pandas performance comparison
- Query optimization challenge
- Parquet output

### Notebook 03: Advanced Concepts (bonus, ~2 hours)
Optional material: window functions, UDFs, partitioning strategies, ML preview.

### Written Exercises
Accompanying markdown files with conceptual questions about filtering, aggregation,
and optimization.

## Time Estimates

| Component | Estimated Time |
|-----------|---------------|
| Setup & Environment | 15-30 min |
| Spark Fundamentals (NB 01) | 90 min |
| Variant QC Exercise (NB 02) | 2.5 hours |
| Written Exercises | 1 hour |
| Advanced Concepts (NB 03) | 2 hours (optional) |
| **Total (required)** | **~4.5 hours** |

## Recommended Learning Path

1. Complete Notebook 00 to verify your setup
2. Work through Notebook 01 to learn Spark concepts
3. Complete Notebook 02 (the main exercise)
4. Answer the written exercises alongside Notebook 02
5. Attempt Notebook 03 if time permits

## Spark UI

While a SparkSession is active, the Spark UI is available at **http://localhost:4040**:

- **Jobs tab**: All completed and running jobs
- **Stages tab**: Individual stages and shuffle details
- **Storage tab**: Cached DataFrames
- **SQL tab**: Query plans for DataFrame operations

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker won't start | Ensure Docker Desktop is running with 4+ GB memory |
| Port 8888 in use | `lsof -i :8888` to find the process, or change port in docker-compose.yml |
| Jupyter not loading | Wait 30 sec after `docker-compose up`; check `docker-compose logs` |
| Out of memory | Use small dataset; increase Docker memory; close other notebooks |
| Spark errors | Restart kernel; only one SparkSession per notebook |

See `instructor/common_issues.md` for detailed troubleshooting.

## Data

The exercise uses synthetic genomics data generated by `data/generate_data.py`:

- **Variants**: VCF-like CSV (CHROM, POS, REF, ALT, QUAL, FILTER, DP, AF, GQ, GENE, CONSEQUENCE)
- **Gene annotations**: BED format with 500 gene regions
- **Sample metadata**: 200 samples with population, sex, phenotype
- **Known variants**: 1,000 annotated variants for join exercises

Three sizes: small (1K rows), medium (100K), large (1M).
See `data/README.md` for full schema documentation.

## Additional Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Hail - Genomics on Spark](https://hail.is/)
- [ADAM - Genomics Formats on Spark](https://github.com/bigdatagenomics/adam)

## License

Created for BINFX410. For educational use only.
