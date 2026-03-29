# BINFX410 — Chapter 10: Data Lakes, Warehouses, and Lakehouses
## Bioinformatics Data Engineering Exercises

This module teaches the core concepts of modern data architecture — data lakes, data warehouses, and lakehouses — using real-world genomics and biotechnology datasets hosted freely on the AWS Open Data Registry.

---

## Overview

| # | Exercise | Key Concept | Dataset | Duration |
|---|---|---|---|---|
| 1 | Genomics Data Lake Anatomy | Bronze/Silver/Gold zones, schema-on-read | GATK Test Data | 45 min |
| 2 | File Format Showdown | VCF vs Parquet, columnar storage | ClinVar | 45 min |
| 3 | Glue Data Catalog | Schema discovery, metadata management | TCGA open access | 60 min |
| 4 | ELT Pipeline — RNA-seq | Medallion architecture, TPM normalization | GTEx v10 | 75 min |
| 5 | Star Schema for SRA | Dimension modeling, dbt, Athena | SRA Metadata | 75 min |
| 6 | Iceberg Time Travel | ACID transactions, ClinVar reclassification | ClinVar | 60 min |
| 7 | Partitioning Strategies | Partition pruning, small files problem | 1000 Genomes chr22 | 60 min |
| 8 | Lake Formation Governance | Column/row-level security, HIPAA patterns | Synthetic PHI data | 60 min |
| 9 | Streaming Ingestion | Kinesis Firehose, sequencer data simulation | Synthetic | 75 min |
| 10 | Capstone: TCGA Lakehouse | End-to-end integration | TCGA BRCA | 3–4 hours |

**Total estimated time (exercises 1–9):** ~9 hours
**Capstone:** 3–4 hours additional

---

## Prerequisites

### Knowledge
- Python 3.9+ (pandas, boto3 familiarity)
- Basic SQL (SELECT, JOIN, GROUP BY)
- Basic understanding of RNA-seq (raw counts, normalization)
- Basic understanding of genomic variant files (VCF format)

### AWS Account Setup
You need an AWS account with the following services accessible:
- Amazon S3
- AWS Glue (Data Catalog + ETL)
- Amazon Athena (with Athena engine v3)
- AWS Lake Formation
- Amazon Kinesis Data Firehose
- AWS Lambda
- AWS IAM

> **Cost Note:** All exercises are designed to run within or very near the AWS Free Tier. Estimated total cost for all 10 exercises: **$2–5 USD per student** (primarily Athena query costs). See [Cost Management](#cost-management) below.

---

## Repository Structure

```
exercise-data-lakes/
├── README.md                    ← You are here
├── plan.md                      ← Instructor implementation guide
├── notebooks/
│   ├── 01_genomics_data_lake_anatomy.ipynb
│   ├── 02_file_format_showdown.ipynb
│   ├── 03_glue_data_catalog.ipynb
│   ├── 04_elt_pipeline_rnaseq.ipynb
│   ├── 05_star_schema_sra.ipynb
│   ├── 06_iceberg_time_travel.ipynb
│   ├── 07_partitioning_strategies.ipynb
│   ├── 08_lake_formation_governance.ipynb
│   ├── 09_streaming_ingestion.ipynb
│   └── 10_capstone_tcga_lakehouse.ipynb
└── dbt/
    ├── dbt_project.yml
    └── models/
        ├── dim_sample.sql
        ├── dim_study.sql
        ├── dim_platform.sql
        └── fact_sequencing_run.sql
```

---

## Quick Start

### Step 1 — Clone and Configure AWS

```bash
# Clone the repository
git clone <repo-url>
cd exercise-data-lakes

# Configure AWS credentials (use your student IAM user)
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

### Step 2 — Install Python Dependencies

```bash
pip install \
  boto3 \
  awswrangler \
  pandas \
  numpy \
  pyarrow \
  matplotlib \
  seaborn \
  faker \
  jupyter
```

> **Note:** `awswrangler` (AWS SDK for pandas) is the primary tool for interacting with Athena, Glue, and S3 from Python. It wraps boto3 with a pandas-friendly API.

### Step 3 — Create Your Student S3 Bucket

Replace `YOUR_STUDENT_ID` with your university ID or initials.

```bash
# Create your working bucket
aws s3 mb s3://binfx410-datalake-YOUR_STUDENT_ID --region us-east-1

# Verify you can write to it
echo "test" | aws s3 cp - s3://binfx410-datalake-YOUR_STUDENT_ID/test.txt
aws s3 rm s3://binfx410-datalake-YOUR_STUDENT_ID/test.txt
```

### Step 4 — Configure Athena Output Location

```bash
# Create an Athena results bucket
aws s3 mb s3://binfx410-athena-results-YOUR_STUDENT_ID --region us-east-1

# Set this as your Athena workgroup output (in the AWS Console or via CLI)
aws athena update-work-group --work-group primary  --configuration-updates ‘ResultConfigurationUpdates={OutputLocation=s3://binfx410-athena-results-YOUR-STUDENT-ID/}’
```

### Step 5 — Open the First Notebook

```bash
jupyter notebook notebooks/01_genomics_data_lake_anatomy.ipynb
```

Set the `BUCKET_NAME` variable in the first code cell to `binfx410-datalake-YOUR_STUDENT_ID` and proceed.

---

## Datasets

All public datasets are accessed with `--no-sign-request` (no AWS credentials required to read them).

### GATK Test Data
- **Bucket:** `s3://gatk-test-data`
- **Use in:** Exercise 1
- **Preview:**
  ```bash
  aws s3 ls s3://gatk-test-data/ --no-sign-request
  ```

### ClinVar Summary Variants (Parquet)
- **Bucket:** `s3://aws-roda-hcls-datalake/clinvar_summary_variants/`
- **Use in:** Exercises 2, 6
- **Preview:**
  ```bash
  aws s3 ls s3://aws-roda-hcls-datalake/clinvar_summary_variants/ --no-sign-request
  ```

### GTEx v10 (Gene Expression)
- **Bucket:** `s3://gtex-resources/`
- **Use in:** Exercise 4
- **Preview:**
  ```bash
  aws s3 ls s3://gtex-resources/ --no-sign-request
  ```

### SRA Run Metadata
- **Bucket:** `s3://sra-pub-metadata-us-east-1`
- **Use in:** Exercise 5
- **Note:** Query directly with Athena — no need to copy data

### 1000 Genomes Project (chr22)
- **Bucket:** `s3://1000genomes/release/20130502/`
- **Use in:** Exercise 7
- **Preview:**
  ```bash
  aws s3 ls "s3://1000genomes/release/20130502/ALL.chr22*" --no-sign-request
  ```

### TCGA Open Access
- **Bucket:** `s3://tcga-2-open`
- **Use in:** Exercises 3, 8, 10
- **Preview:**
  ```bash
  aws s3 ls s3://tcga-2-open/ --no-sign-request --recursive | head -50
  ```

---

## Exercise Dependencies

```
Exercise 1 (Data Lake Structure)
    └── Exercise 2 (File Formats)
    └── Exercise 3 (Glue Catalog)
            └── Exercise 4 (ELT Pipeline)
                    └── Exercise 5 (Star Schema)
Exercise 6 (Iceberg)          ← can be done independently after Ex. 2
Exercise 7 (Partitioning)     ← can be done independently after Ex. 1
Exercise 8 (Governance)       ← can be done independently after Ex. 3
Exercise 9 (Streaming)        ← can be done independently
Exercise 10 (Capstone)        ← requires completion of Ex. 1–8
```

---

## AWS Services Used

### Always Used
| Service | Purpose | Free Tier |
|---|---|---|
| S3 | Storage for all data zones | 5 GB free |
| Glue Data Catalog | Table metadata | 1M objects free |
| Athena | SQL queries | ~$0.000005/small query |

### Used in Specific Exercises
| Service | Exercise | Free Tier |
|---|---|---|
| Glue ETL | 4 | 1M DPU-seconds/month |
| Lake Formation | 8, 10 | No charge beyond underlying |
| Kinesis Firehose | 9 | First 5 GB/month free |
| Lambda | 9 | 1M requests/month free |
| IAM | 8 | Always free |

---

## Cost Management

### Keeping Costs Low
- **Use small data subsets.** Each notebook specifies recommended subset sizes. Don't copy full chromosomes or entire TCGA when the exercise only needs 50 samples.
- **Delete intermediate data** after each exercise if storage accumulates.
- **Use Athena partition projection** where configured — it avoids scanning extra data.
- **Monitor Athena costs** in the console: Services → Athena → Workgroup → Data scanned.

### Estimated Costs per Exercise
| Exercise | Primary Cost Driver | Estimated Cost |
|---|---|---|
| 1–3 | S3 PUT requests + Athena queries | < $0.10 |
| 4 | Glue ETL DPU-seconds | < $0.25 |
| 5 | Athena queries on SRA | < $0.10 |
| 6 | Athena Iceberg queries | < $0.25 |
| 7 | Athena benchmark queries | < $0.50 |
| 8 | Lake Formation + IAM | < $0.05 |
| 9 | Kinesis Firehose | < $0.10 |
| 10 | All services | $1.00–2.00 |
| **Total** | | **~$2.35–3.35** |

### Teardown After Each Session
```bash
# Empty and delete your working bucket when done with the module
aws s3 rm s3://binfx410-datalake-YOUR_STUDENT_ID --recursive
aws s3 rb s3://binfx410-datalake-YOUR_STUDENT_ID

aws s3 rm s3://binfx410-athena-results-YOUR_STUDENT_ID --recursive
aws s3 rb s3://binfx410-athena-results-YOUR_STUDENT_ID

# Delete Glue databases created during exercises
aws glue delete-database --name binfx410_bronze
aws glue delete-database --name binfx410_silver
aws glue delete-database --name binfx410_gold
```

---

## Troubleshooting

### "Access Denied" reading public buckets
Make sure you're using `--no-sign-request` for public buckets:
```python
import boto3
s3 = boto3.client('s3', region_name='us-east-1')
# For public buckets, use unsigned config:
from botocore import UNSIGNED
from botocore.config import Config
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
```

### Athena "Query failed: HIVE_BAD_DATA"
Usually caused by a schema mismatch. Check:
1. Your Glue table column types match the actual file
2. VCF files have `#` headers that need to be skipped (`skip.header.line.count` = number of header lines)
3. The SerDe is set correctly for your file format

### Glue Crawler Produces Wrong Schema for VCF/MAF
Expected — VCF and MAF formats use non-standard headers that Glue doesn't natively understand. Exercise 3 walks through manual schema correction. For VCF, use `OpenCSVSerDe` with tab delimiter and set `skip.header.line.count`.

### Lake Formation Denying Access Despite Correct Policy
Check that:
1. The S3 location is registered with Lake Formation (`register_resource`)
2. The IAM role has `lakeformation:GetDataAccess` permission
3. The Glue database is also granted to the role (not just the table)
4. You are **not** also using S3 bucket policies that deny access — Lake Formation and S3 policies are evaluated together

### Iceberg Table Queries Failing in Athena
Ensure your Athena workgroup uses **engine version 3** (Athena v3 is required for Iceberg DML):
```bash
aws athena update-work-group \
  --work-group primary \
  --configuration-updates "EngineVersion={SelectedEngineVersion=Athena engine version 3}"
```

### Small Files Problem Symptoms
Signs you have too many small files:
- Athena queries are slow despite small total data size
- `list_objects_v2` returns thousands of files < 1 MB each
- Glue crawlers take a very long time

Fix: use Athena `OPTIMIZE table REWRITE DATA USING BIN_PACK` (Iceberg) or rewrite Parquet files with a compaction step.

---

## Key Concepts Reference

| Term | Definition |
|---|---|
| **Data Lake** | Low-cost object storage (S3) holding data in any format; schema applied at query time |
| **Data Warehouse** | Structured, schema-on-write storage optimized for analytical SQL (e.g., Redshift) |
| **Lakehouse** | Layer over a data lake that adds warehouse features: ACID, schema enforcement, time travel |
| **Medallion Architecture** | Bronze (raw) → Silver (cleaned) → Gold (aggregated) zone pattern |
| **Partition Pruning** | Query planner skips partitions not matching the WHERE clause — dramatically reduces bytes scanned |
| **Iceberg** | Open table format providing ACID transactions, time travel, and schema evolution on S3 |
| **TPM** | Transcripts Per Million — RNA-seq normalization metric accounting for gene length and sequencing depth |
| **TMB** | Tumor Mutation Burden — count of somatic mutations per megabase; correlates with immunotherapy response |
| **VCF** | Variant Call Format — standard text format for genomic variants |
| **MAF** | Mutation Annotation Format — TCGA's format for annotated somatic mutations |
| **Star Schema** | Dimensional model with one central fact table surrounded by dimension tables |
| **Lake Formation** | AWS service adding fine-grained access control (column/row-level) to data lake tables |

---

## Academic Integrity

These exercises use real public datasets but require original analysis and written responses. The "Reflection Questions" at the end of each notebook must be answered in your own words and will be evaluated for depth of understanding, not just technical correctness.

---

## Additional Resources

- [AWS Open Data Registry — Genomics](https://registry.opendata.aws/tag/genomic/)
- [AWS Glue Documentation](https://docs.aws.amazon.com/glue/)
- [Amazon Athena User Guide](https://docs.aws.amazon.com/athena/)
- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/)
- [AWS SDK for pandas (awswrangler)](https://aws-sdk-pandas.readthedocs.io/)
- [GTEx Portal](https://gtexportal.org/)
- [NCBI SRA](https://www.ncbi.nlm.nih.gov/sra)
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)
- [1000 Genomes Project](https://www.internationalgenome.org/)
