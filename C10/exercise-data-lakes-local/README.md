# BINFX410 — Chapter 10: Data Storage Architectures

**Data Warehouses, Lakes, and Lakehouses — Local Edition**

A five-notebook series that builds all three major data storage architectures from scratch using only local, open-source tools. No cloud account, no cluster, no server required.

---

## What You Build

| Notebook | Topic | Output |
|----------|-------|--------|
| `01_introduction_and_setup.ipynb` | Architecture concepts + genomics dataset generation | `./raw_data/` (CSV files) |
| `02_data_lake.ipynb` | Medallion lake (Bronze/Silver/Gold) | `./data_lake/` (Parquet files) |
| `03_data_warehouse.ipynb` | Star schema warehouse | `./warehouse/genomics.duckdb` |
| `04_lakehouse.ipynb` | Delta Lake with ACID + time travel | `./lakehouse/` (Delta tables) |
| `05_comparison_and_capstone.ipynb` | Benchmarks + capstone project | Charts + genomics analysis |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start JupyterLab
jupyter lab

# 3. Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05
```

Notebook 01 must run first — it generates the `./raw_data/` CSV files that all subsequent notebooks depend on.

---

## Dataset

A synthetic cancer genomics dataset generated with Faker (seed=42, fully reproducible):

| Table | Rows | Description |
|-------|------|-------------|
| `samples.csv` | 500 | Sequencing samples: tissue type, platform, library prep, coverage metrics, project |
| `genes.csv` | 100 | Gene annotations: real cancer genes + synthetic names, chromosomal positions, biotype |
| `variant_calls.csv` | 2,000 | Variant calling runs linked to samples, with caller and filter status |
| `variants.csv` | 5,000 | Individual variants linked to calls and genes, with allele frequency and quality |

The schema mirrors a real cancer genomics pipeline:
- Multiple samples per patient (tumor + normal pairs)
- Variant calls run through standard callers (GATK, DeepVariant, Strelka2, Mutect2)
- Variants annotated with consequence and allele frequency in tumor

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `duckdb >= 0.10` | SQL engine for warehouse and Parquet/Delta queries |
| `pandas >= 2.0` | DataFrame transformations |
| `pyarrow >= 14.0` | Parquet read/write, Arrow schema inspection |
| `deltalake >= 0.15` | Delta Lake tables (ACID, time travel, MERGE) |
| `faker >= 20.0` | Synthetic dataset generation |
| `matplotlib >= 3.7` | Benchmark visualizations |
| `seaborn >= 0.12` | Statistical charts |

---

## Architecture Overview

```
1990s: DATA WAREHOUSE        2010s: DATA LAKE           2020s: LAKEHOUSE
--------------------         ------------------         ----------------
Schema-on-write              Schema-on-read             Schema-on-write + evolution
ACID (full)                  No ACID                    ACID via transaction log
High storage cost            Low storage cost           Low storage cost
SQL/BI only                  Any data type              BI + ML + streaming
DuckDB (local)               PyArrow + Parquet          Delta Lake + DuckDB
```

---

## Key Concepts by Notebook

**Notebook 02 — Data Lake**
- Medallion architecture: Bronze (raw copy) -> Silver (cleaned/typed) -> Gold (pre-aggregated)
- Parquet columnar format and compression (CSV vs Parquet size comparison)
- Querying Parquet directly with DuckDB `read_parquet()`
- Schema drift demo: incompatible variant batch files silently produce NULLs
- Data swamp demo: governance failure and minimal catalog solution

**Notebook 03 — Data Warehouse**
- OLTP vs OLAP workloads
- Star schema: `dim_date`, `dim_samples`, `dim_genes`, `fact_variant_calls`, `fact_variants`
- ETL pipeline loading from CSV into DuckDB
- Window functions: `PERCENT_RANK`, `LAG`, running totals, moving averages
- `EXPLAIN` / `EXPLAIN ANALYZE` for query plan inspection

**Notebook 04 — Lakehouse**
- Delta Lake transaction log (`_delta_log/`) — ACID on object storage
- Schema enforcement (bad writes blocked) vs schema evolution (`schema_mode='merge'`)
- Time travel: read any historical version with `DeltaTable(path, version=N)`
- MERGE (upsert): re-evaluate filter status + insert new calls atomically
- OPTIMIZE (file compaction) and VACUUM (remove old unreferenced files)
- DuckDB `delta_scan()` for SQL on Delta tables

**Notebook 05 — Comparison & Capstone**
- Benchmark: same query across CSV, Parquet, DuckDB warehouse, Delta Lake
- Decision framework: choosing the right architecture for a workload
- Capstone: 5 genomics questions from Dr. Kim answered with SQL + visualizations

---

## Exercises Summary

Each notebook contains exercises. A brief guide:

| Exercise | Topic |
|----------|-------|
| 1.1 | Explore raw DataFrames (avg depth, unique PASS samples, non-PASS fraction, top biotype by quality) |
| 1.2 | Design a star schema for the genomics dataset |
| 1.3 | Pick the right architecture for 3 clinical/research scenarios |
| 2.1 | DuckDB SQL: top 3 tissue types by PASS variant count |
| 2.2 | Build a new Gold table: consequence x month variant counts |
| 2.3 | Schema compatibility checker function |
| 3.1 | Cohort analysis with window functions |
| 3.2 | Running total and 3-month moving average of variant counts |
| 3.3 | Gene co-occurrence analysis (genes mutated in the same call) |
| 4.1 | Time travel audit: row counts and changed records by version |
| 4.2 | Bulk MERGE: mark old LowQuality calls as Reanalysis_Needed |
| 4.3 | Cohort retention table via DuckDB on Delta tables |
| 5.1 | Highest-confidence somatic calls: scatter plot af_tumor vs quality_score |
| 5.2 | Scale test: 10x dataset, re-run benchmarks |
| 5.3 | Lakehouse vs. warehouse for clinical genomics architecture design |
