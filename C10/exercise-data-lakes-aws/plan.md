# BINFX410 Chapter 10 — Implementation Plan
## Data Lakes, Warehouses, and Lakehouses in Bioinformatics

**Last Updated:** 2026-03-15
**Course:** BINFX410 — Bioinformatics Data Engineering
**Module:** Chapter 10 — Modern Data Architecture

---

## 1. Module Overview

### Learning Objectives

By the end of this module, students will be able to:

1. Explain the architectural differences between data lakes, data warehouses, and lakehouses
2. Organize genomics data using the Bronze/Silver/Gold (medallion) architecture on Amazon S3
3. Choose appropriate file formats (VCF, Parquet, ORC) for different genomics analytics use cases
4. Build and query a Glue Data Catalog over heterogeneous multi-omics data
5. Design and implement ELT pipelines transforming raw RNA-seq count matrices to analysis-ready aggregations
6. Model genomics metadata as a dimensional star schema queryable via Athena
7. Apply Apache Iceberg for ACID transactions and time travel on mutable genomics data (ClinVar reclassifications)
8. Choose and implement partitioning strategies optimized for genomic query patterns
9. Enforce column-level and row-level access control using Lake Formation for HIPAA-relevant governance scenarios
10. Design a micro-batching streaming ingestion pipeline for sequencing instrument telemetry

### Prerequisites

| Prerequisite | Level Required |
|---|---|
| Python (pandas, boto3) | Intermediate |
| SQL (SELECT, JOIN, GROUP BY, window functions) | Intermediate |
| AWS Console navigation | Basic |
| RNA-seq concepts (raw counts, TPM, normalization) | Basic |
| Genomic variant formats (VCF structure, INFO field) | Basic |
| Command line / terminal | Basic |

### Module Structure

| Exercise | Title | Estimated Time | Difficulty |
|---|---|---|---|
| 1 | Genomics Data Lake Anatomy | 45 min | Beginner |
| 2 | File Format Showdown | 45 min | Beginner |
| 3 | Glue Data Catalog | 60 min | Intermediate |
| 4 | ELT Pipeline — RNA-seq | 75 min | Intermediate |
| 5 | Star Schema for SRA Metadata | 75 min | Intermediate |
| 6 | Iceberg Time Travel | 60 min | Intermediate |
| 7 | Partitioning Strategies | 60 min | Intermediate |
| 8 | Lake Formation Governance | 60 min | Advanced |
| 9 | Streaming Ingestion | 75 min | Advanced |
| 10 | Capstone: TCGA Lakehouse | 3–4 hours | Advanced |

**Total time (Ex. 1–9):** ~9.25 hours
**Capstone (Ex. 10):** 3–4 hours

---

## 2. Infrastructure Setup

### 2.1 Required AWS Services

| Service | Purpose | Notes |
|---|---|---|
| S3 | Data lake storage (all zones) | Two buckets per student: data + Athena results |
| AWS Glue Data Catalog | Table metadata, schema management | One database per zone per student |
| AWS Glue ETL | PySpark transformation jobs | Exercise 4 only |
| Amazon Athena (v3) | SQL query engine over S3 | Must use engine version 3 for Iceberg DML |
| AWS Lake Formation | Fine-grained access control | Exercise 8, 10 |
| Amazon Kinesis Firehose | Streaming micro-batch ingestion | Exercise 9 |
| AWS Lambda | Event emitter simulation | Exercise 9 |
| AWS IAM | Roles for governance exercises | Exercises 8, 10 |

### 2.2 IAM Permissions for Student Role

The student IAM user/role needs the following policies attached:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3DataLake",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
        "s3:ListBucket", "s3:CreateBucket", "s3:DeleteBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::binfx410-datalake-*",
        "arn:aws:s3:::binfx410-datalake-*/*",
        "arn:aws:s3:::binfx410-athena-results-*",
        "arn:aws:s3:::binfx410-athena-results-*/*"
      ]
    },
    {
      "Sid": "PublicDatasets",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::gatk-test-data/*",
        "arn:aws:s3:::aws-roda-hcls-datalake/*",
        "arn:aws:s3:::gtex-resources/*",
        "arn:aws:s3:::sra-pub-metadata-us-east-1/*",
        "arn:aws:s3:::1000genomes/*",
        "arn:aws:s3:::tcga-2-open/*"
      ]
    },
    {
      "Sid": "GlueFullAccess",
      "Effect": "Allow",
      "Action": ["glue:*"],
      "Resource": ["*"]
    },
    {
      "Sid": "AthenaFullAccess",
      "Effect": "Allow",
      "Action": ["athena:*"],
      "Resource": ["*"]
    },
    {
      "Sid": "LakeFormation",
      "Effect": "Allow",
      "Action": [
        "lakeformation:GetDataAccess",
        "lakeformation:GrantPermissions",
        "lakeformation:RevokePermissions",
        "lakeformation:RegisterResource",
        "lakeformation:ListPermissions",
        "lakeformation:GetDataLakeSettings",
        "lakeformation:PutDataLakeSettings"
      ],
      "Resource": ["*"]
    },
    {
      "Sid": "FirehoseLambda",
      "Effect": "Allow",
      "Action": [
        "firehose:CreateDeliveryStream",
        "firehose:PutRecord",
        "firehose:PutRecordBatch",
        "firehose:DeleteDeliveryStream",
        "lambda:CreateFunction",
        "lambda:InvokeFunction",
        "lambda:DeleteFunction",
        "iam:PassRole"
      ],
      "Resource": ["*"]
    }
  ]
}
```

### 2.3 S3 Bucket Naming Convention

Each student creates two buckets:

| Bucket | Purpose | Example |
|---|---|---|
| `binfx410-datalake-{student_id}` | Data lake (bronze/silver/gold) | `binfx410-datalake-jsmith` |
| `binfx410-athena-results-{student_id}` | Athena query output | `binfx410-athena-results-jsmith` |

**S3 Prefix Structure (within data lake bucket):**

```
binfx410-datalake-{student_id}/
├── bronze/
│   ├── raw-reads/sample={sample_id}/reference={ref}/
│   ├── variants/source={source}/
│   └── clinical/study={study_id}/
├── silver/
│   ├── expression/cancer_type={cancer}/sample_type={type}/
│   ├── variants/chromosome=chr{n}/
│   └── clinical/
└── gold/
    ├── expression_summary/tissue_type={tissue}/
    ├── tmb_per_sample/
    └── star_schema/
        ├── fact_sequencing_run/
        ├── dim_sample/
        ├── dim_study/
        └── dim_platform/
```

### 2.4 Athena Workgroup Setup

Each student must configure their Athena workgroup to use:
- **Engine version:** Athena engine version 3 (required for Iceberg)
- **Output location:** `s3://binfx410-athena-results-{student_id}/`
- **Enforce workgroup settings:** Yes

```bash
aws athena update-work-group \
  --work-group primary \
  --configuration-updates \
    "ResultConfigurationUpdates={OutputLocation=s3://binfx410-athena-results-STUDENT_ID/},EnforceWorkGroupConfiguration=true,EngineVersion={SelectedEngineVersion='Athena engine version 3'}"
```

---

## 3. Dataset Acquisition Plan

All datasets are on the AWS Open Data Registry and are publicly accessible without credentials.

### 3.1 GATK Test Data (Exercise 1)

```bash
# List available content
aws s3 ls s3://gatk-test-data/ --no-sign-request --recursive | head -50

# Recommended files to copy for Exercise 1 (~500 MB total)
aws s3 cp s3://gatk-test-data/wgs_bam/NA12878_20k_b37/ \
  s3://binfx410-datalake-STUDENT_ID/bronze/raw-reads/sample=NA12878/reference=hg19/ \
  --recursive --no-sign-request

aws s3 cp s3://gatk-test-data/wgs_vcf/ \
  s3://binfx410-datalake-STUDENT_ID/bronze/variants/source=gatk/ \
  --recursive --no-sign-request
```

**Instructor note:** GATK test data is purpose-built for exercises; files are small and well-annotated.

### 3.2 ClinVar Summary Variants (Exercises 2, 6)

```bash
# Already in Parquet format — no preprocessing needed
aws s3 ls s3://aws-roda-hcls-datalake/clinvar_summary_variants/ --no-sign-request

# Copy to student bucket for Exercise 2 (~2 GB)
aws s3 cp s3://aws-roda-hcls-datalake/clinvar_summary_variants/ \
  s3://binfx410-datalake-STUDENT_ID/bronze/variants/source=clinvar/ \
  --recursive --no-sign-request
```

**Instructor note:** ClinVar is updated weekly. The RODA version is pre-partitioned Parquet — students can query it in place for Exercise 2 to save on S3 costs.

### 3.3 GTEx v10 (Exercise 4)

```bash
# List available files
aws s3 ls s3://gtex-resources/GTEx_Analysis_v10/ --no-sign-request

# The gene TPM matrix (gzipped GCT format)
# Recommended: download a pre-filtered subset for teaching (~50 samples)
aws s3 cp \
  "s3://gtex-resources/GTEx_Analysis_v10/GTEx_Analysis_v10_RNASeQCv2.0.0_gene_tpm.gct.gz" \
  /tmp/gtex_tpm.gct.gz --no-sign-request

# Then upload to bronze zone
aws s3 cp /tmp/gtex_tpm.gct.gz \
  s3://binfx410-datalake-STUDENT_ID/bronze/expression/gtex_tpm_raw.gct.gz
```

**Instructor note:** The full GTEx TPM matrix is ~1 GB compressed. The notebook filters to the first 1,000 genes × 50 samples as a working subset. For production exercises, pre-filter to a single tissue before distributing.

### 3.4 SRA Metadata (Exercise 5)

```bash
# This dataset is designed to be queried in-place via Athena
# No copy needed — create an Athena data source pointing directly to:
# s3://sra-pub-metadata-us-east-1/sra/metadata/

# Register as external Glue table (done in the notebook via awswrangler)
# Approximate query cost: < $0.01 per query on typical filter
```

**Instructor note:** The SRA metadata catalog is ~500 GB but students only scan relevant partitions. Filter queries cost fractions of a cent.

### 3.5 1000 Genomes chr22 (Exercise 7)

```bash
# Single chromosome VCF (all samples, chr22 only) — ~1 GB compressed
aws s3 ls "s3://1000genomes/release/20130502/" --no-sign-request | grep chr22

# Copy chr22 VCF to student bucket
aws s3 cp \
  "s3://1000genomes/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
  s3://binfx410-datalake-STUDENT_ID/bronze/variants/source=1000genomes/ \
  --no-sign-request
```

**Instructor note:** The notebook samples ~10,000 variants from chr22 for the partitioning exercise to keep Athena costs minimal.

### 3.6 TCGA Open Access (Exercises 3, 8, 10)

```bash
# List TCGA open access content
aws s3 ls s3://tcga-2-open/ --no-sign-request --recursive | head -100

# BRCA (breast cancer) files for capstone — copy a representative subset
# RNA-seq HTSeq counts: look for *.htseq.counts.gz files
# Clinical: look for *.clinical.tsv files
# Somatic mutations: look for *.maf.gz files

# Recommended: use the GDC Data Portal to identify ~20 BRCA sample UUIDs
# then bulk copy via manifest for Exercise 10
```

**Instructor note:** TCGA files are organized by UUID — the notebook includes code to list and filter to BRCA samples specifically.

---

## 4. Exercise Sequence and Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Recommended Sequence                          │
│                                                                  │
│  Ex 1: Data Lake Anatomy          (no dependencies)             │
│    ├──► Ex 2: File Formats        (needs Ex 1 S3 bucket)        │
│    └──► Ex 7: Partitioning        (independent after Ex 1)      │
│                                                                  │
│  Ex 3: Glue Data Catalog          (needs Ex 1 S3 bucket)        │
│    └──► Ex 4: ELT Pipeline        (needs Ex 3 Glue DB)          │
│              └──► Ex 5: Star Schema (needs Ex 4 Gold data)      │
│                                                                  │
│  Ex 6: Iceberg Time Travel        (independent — needs Athena)  │
│  Ex 8: Lake Formation             (independent — needs IAM)     │
│  Ex 9: Streaming Ingestion        (independent — needs Firehose)│
│                                                                  │
│  Ex 10: Capstone                  (requires Ex 1–8 complete)    │
└─────────────────────────────────────────────────────────────────┘
```

**Recommended session groupings for a 3-session course:**

| Session | Exercises | Focus |
|---|---|---|
| Session 1 (3 hrs) | Ex 1, 2, 3 | Data lake fundamentals, file formats, catalog |
| Session 2 (3 hrs) | Ex 4, 5, 6, 7 | ELT, modeling, Iceberg, partitioning |
| Session 3 (3 hrs) | Ex 8, 9, + start 10 | Governance, streaming, begin capstone |
| Take-home | Ex 10 | Capstone completion |

---

## 5. AWS Resource Provisioning Checklist

### Before Students Arrive (Instructor Setup)

#### S3
- [ ] Verify `gatk-test-data`, `aws-roda-hcls-datalake`, `gtex-resources`, `sra-pub-metadata-us-east-1`, `1000genomes`, and `tcga-2-open` are accessible with `--no-sign-request`
- [ ] Create a shared staging bucket (optional) with pre-downloaded subsets to avoid repeated large downloads

#### IAM
- [ ] Create `binfx410-student-policy` with the JSON policy from Section 2.2
- [ ] Create `binfx410-student-role` (for cross-account or assumed-role scenarios)
- [ ] Create `binfx410-researcher-role` — for Exercise 8 (no PHI columns, tissue-restricted rows)
- [ ] Create `binfx410-clinical-role` — for Exercise 8 (full access)
- [ ] For each: attach trust policy allowing student IAM user to assume the role

#### Athena
- [ ] Confirm Athena engine version 3 is available in the account region
- [ ] Set up a shared workgroup `binfx410` with spending limits ($5/query max)

#### Lake Formation
- [ ] Enable Lake Formation as data lake administrator for instructor account
- [ ] Pre-register the shared S3 location (if using shared data bucket)
- [ ] Confirm Lake Formation is enabled (not in legacy IAM mode) before Exercise 8

#### Kinesis Firehose (Exercise 9)
- [ ] Create IAM role `binfx410-firehose-role` with S3 write + Glue schema access
- [ ] Students will create their own Firehose streams in the notebook

#### Lambda (Exercise 9)
- [ ] Create IAM role `binfx410-lambda-role` with Firehose PutRecord + CloudWatch Logs permissions

### Per-Student Checklist

- [ ] IAM user created with `binfx410-student-policy` attached
- [ ] Student has created `binfx410-datalake-{student_id}` bucket
- [ ] Student has created `binfx410-athena-results-{student_id}` bucket
- [ ] Athena workgroup configured with engine v3 and output location
- [ ] Python environment with required packages installed (`pip install boto3 awswrangler pandas numpy pyarrow matplotlib seaborn faker jupyter`)
- [ ] `aws configure` completed with student credentials

---

## 6. Exercise Summary Table

| # | Title | Services | Dataset | Duration | Key Concept |
|---|---|---|---|---|---|
| 1 | Genomics Data Lake Anatomy | S3, Athena | GATK Test Data | 45 min | Bronze/Silver/Gold, schema-on-read, Hive partitioning |
| 2 | File Format Showdown | S3, Athena, Glue | ClinVar | 45 min | Columnar vs row storage, predicate pushdown, cost |
| 3 | Glue Data Catalog | Glue, S3, Athena | TCGA open | 60 min | Schema discovery, catalog management, MAF/VCF limitations |
| 4 | ELT Pipeline — RNA-seq | Glue ETL, S3, Athena | GTEx v10 | 75 min | Medallion arch, TPM normalization, wide-to-long transform |
| 5 | Star Schema for SRA | Athena, Glue, (dbt) | SRA metadata | 75 min | Dimension modeling, fact tables, CTAS |
| 6 | Iceberg Time Travel | Athena (v3), S3 | ClinVar | 60 min | ACID, time travel, schema evolution, reclassification |
| 7 | Partitioning Strategies | S3, Athena, Glue | 1000 Genomes chr22 | 60 min | Partition pruning, small files, chr as partition key |
| 8 | Lake Formation Governance | Lake Formation, IAM, Athena | Synthetic PHI | 60 min | Column masking, row filtering, HIPAA patterns |
| 9 | Streaming Ingestion | Kinesis Firehose, Lambda, S3, Athena | Synthetic | 75 min | Micro-batching, hot/cold path, format conversion |
| 10 | Capstone: TCGA Lakehouse | All above | TCGA BRCA | 3–4 hrs | End-to-end integration, multi-omics, design decisions |

---

## 7. Grading Rubrics

### Exercise 1 — Genomics Data Lake Anatomy (20 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Three-zone S3 structure created with correct prefix conventions | 5 | Screenshot of S3 console or `aws s3 ls` output |
| Hive-style partitioning applied (sample=X/reference=Y) | 5 | S3 prefix structure in notebook output |
| Athena query runs successfully against staged VCF | 5 | Query result cell in notebook |
| Reflection questions answered with biological context | 5 | Written cell responses |

### Exercise 2 — File Format Showdown (20 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Both VCF and Parquet tables registered in Athena | 4 | `awswrangler.catalog.get_table_types()` output |
| Same query run against both; bytes scanned captured | 6 | DataFrame with `DataScannedInMegaBytes` for each format |
| Matplotlib comparison chart rendered | 4 | Chart cell output |
| Correct cost calculation shown | 3 | Cost formula cell with numeric result |
| Reflection explains columnar advantage for VCF INFO fields | 3 | Written response |

### Exercise 3 — Glue Data Catalog (20 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Glue crawler runs and completes successfully | 5 | Crawler status in notebook output |
| Three table types discovered (TSV, JSON, MAF) | 4 | `glue.get_tables()` output showing all three |
| MAF schema correctly patched with proper column names | 6 | `get_table()` after manual fix showing Hugo_Symbol etc. |
| Cross-table Athena join executes | 5 | Query result joining clinical to expression |

### Exercise 4 — ELT Pipeline (25 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Bronze data written to S3 in original format | 4 | S3 listing of bronze zone |
| Silver: TPM normalization implemented correctly | 8 | TPM sum per sample ≈ 1,000,000 (verify in notebook) |
| Silver: Long format Parquet written, partitioned by tissue | 6 | S3 listing shows partition structure; schema in Glue |
| Gold: HVG flagging logic correct (CV-based) | 4 | Gold table `is_hvg` column count reasonable |
| Glue PySpark script shown | 3 | Code cell with PySpark equivalent |

### Exercise 5 — Star Schema (20 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Three dimension tables created via CTAS | 6 | `awswrangler.catalog.get_tables()` shows dim_* tables |
| Fact table with correct foreign keys | 6 | Schema output showing FK columns |
| At least two analytical queries produce non-empty results | 5 | Query result cells |
| Reflection connects star schema to bioinformatics metadata needs | 3 | Written response |

### Exercise 6 — Iceberg Time Travel (25 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Iceberg table created with `table_type='ICEBERG'` | 4 | Table properties in Athena output |
| UPDATE and DELETE statements executed successfully | 6 | Row counts before/after |
| Schema evolution (ADD COLUMN) demonstrated | 5 | Table schema shows new column; old rows queryable |
| Time travel query returns pre-update classification | 7 | Query result showing historical value |
| Reflection addresses clinical implications | 3 | Written response |

### Exercise 7 — Partitioning (20 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Three partitioning strategies implemented | 6 | S3 listings showing different structures |
| Benchmark queries run against all three | 6 | DataFrame with bytes scanned per strategy per query |
| Visualization rendered | 4 | Chart output |
| Small files analysis with file count and size statistics | 4 | `list_objects_v2` summary statistics |

### Exercise 8 — Lake Formation (25 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Synthetic dataset with PHI columns generated and staged | 4 | S3 listing + schema output |
| Lake Formation column exclusion applied | 6 | `lakeformation.list_permissions()` output |
| Row filter applied for researcher role | 5 | Filter expression in LF permissions |
| Assumed-role Athena query omits PHI columns | 7 | Query result cell showing truncated schema |
| Clinical role query shows full schema | 3 | Second query result cell |

### Exercise 9 — Streaming Ingestion (20 pts)

| Criterion | Points | Evidence |
|---|---|---|
| Firehose delivery stream created | 4 | `describe_delivery_stream()` output |
| Events sent to Firehose (at least 3 batches) | 5 | `put_record_batch()` response cells |
| Parquet files appear in S3 after flush | 4 | S3 listing post-flush |
| Athena query on streamed data succeeds | 4 | Query result cell |
| Hot/cold path design documented | 3 | Markdown cell with architecture |

### Exercise 10 — Capstone (100 pts)

| Criterion | Points |
|---|---|
| Bronze zone: all three TCGA file types staged with crawler | 10 |
| Silver zone: RNA-seq correctly normalized to TPM, partitioned | 20 |
| Silver zone: MAF parsed, partitioned by chromosome | 10 |
| Gold zone: TMB calculation correct (mutations / Mb) | 15 |
| Iceberg Gold tables created with at least one UPDATE | 10 |
| Star schema CTAS queries run successfully | 10 |
| Two analytical queries produce meaningful results | 10 |
| Lake Formation governance applied | 5 |
| Design document addresses all six prompts | 10 |

---

## 8. Troubleshooting Guide

### Athena Permissions Errors

**Symptom:** `Access denied` when running Athena queries
**Causes and fixes:**
1. Athena output location not set → run `update-work-group` command from README
2. Student IAM policy missing `athena:StartQueryExecution` → add to policy
3. Lake Formation denying Glue catalog access → grant `DESCRIBE` on database

### Glue Crawler Failing on VCF/MAF

**Symptom:** Crawler completes but produces wrong schema or marks table `DEPRECATED`
**Root cause:** VCF `##` headers and MAF `#` comment lines are not standard delimiters
**Fix:** Exercise 3 walks through manual `update_table()`. For instructors pre-staging data:
- Use `LazySimpleSerDe` with `skip.header.line.count` property
- Set delimiter to `\t` explicitly
- For MAF: the `#version` line needs to be counted as a header

### Parquet Schema Mismatches in Athena

**Symptom:** `HIVE_BAD_DATA: Error opening Hive split` or column type errors
**Common causes:**
- Mixed integer/float in a column (pandas infers differently per partition)
- `None` in a string column causing type inference failure
- Glue schema not updated after adding a new column

**Fix:**
```python
# Force schema when writing Parquet
import pyarrow as pa
schema = pa.schema([
    ('gene_id', pa.string()),
    ('sample_id', pa.string()),
    ('tpm', pa.float64())
])
df.to_parquet('output.parquet', schema=schema)
```

### Lake Formation Denying Access Unexpectedly

**Symptom:** `Access denied` even though LF grant is confirmed
**Checklist:**
1. Is the S3 location registered with Lake Formation? (`lakeformation.list_resources()`)
2. Does the IAM role have `lakeformation:GetDataAccess`?
3. Is Lake Formation enabled on the Glue database (not legacy IAM mode)?
4. Did you grant both the database AND the table? (LF requires both)
5. Is there a conflicting S3 bucket policy denying access?

### Small Files Problem Symptoms

**Symptoms:**
- Athena queries slow on < 100 MB of data
- Query plan shows `SplitSize` very small
- `list_objects_v2` returns > 1000 files for a simple dataset

**Diagnosis:**
```python
response = s3.list_objects_v2(Bucket=BUCKET, Prefix='silver/variants/')
sizes = [obj['Size'] for obj in response.get('Contents', [])]
print(f"Files: {len(sizes)}, Avg size: {sum(sizes)/len(sizes)/1024:.1f} KB")
```

**Fix (Iceberg):**
```sql
OPTIMIZE your_table REWRITE DATA USING BIN_PACK
  WHERE dt > '2026-01-01';
```

**Fix (regular Parquet):** Re-run the Glue ETL job with `coalesce(8)` before writing.

### Iceberg DML Not Supported

**Symptom:** `UPDATE/DELETE/MERGE not supported in this version`
**Fix:** Workgroup must use Athena engine version 3:
```bash
aws athena update-work-group \
  --work-group primary \
  --configuration-updates "EngineVersion={SelectedEngineVersion='Athena engine version 3'}"
```

### Kinesis Firehose Parquet Conversion Failures

**Symptom:** Files arrive in S3 as JSON instead of Parquet
**Causes:**
1. Glue table schema doesn't match JSON record exactly (extra/missing fields)
2. Firehose IAM role missing `glue:GetTableVersions` permission
3. Athena v3 not enabled in workgroup (format conversion requirement)

---

## 9. Cost Management

### Services That Accrue Real Costs

| Service | Billing Model | Expected Cost per Student |
|---|---|---|
| Athena | $5.00 per TB scanned | $0.50–1.50 total |
| S3 Storage | $0.023 per GB-month | $0.05–0.25 |
| S3 Requests | $0.0004 per 1K PUT | < $0.10 |
| Glue ETL | $0.44 per DPU-hour (2 DPU min) | $0.10–0.50 |
| Kinesis Firehose | $0.029 per GB ingested | < $0.05 |
| **Total** | | **$0.80–2.40** |

### Services That Are Free

- Lambda (1M requests/month free tier)
- Glue Data Catalog (1M objects/month free tier)
- Lake Formation (no charge beyond underlying)
- IAM (always free)
- S3 within free tier (5 GB storage, 20K GET, 2K PUT)

### Cost Monitoring Commands

```bash
# Check Athena data scanned this month (approximate)
aws cloudwatch get-metric-statistics \
  --namespace AWS/Athena \
  --metric-name DataScannedInBytes \
  --start-time $(date -u -v-30d +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 2592000 \
  --statistics Sum

# Set a billing alert at $5
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget '{"BudgetName":"binfx410-limit","BudgetLimit":{"Amount":"5","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}'
```

### End-of-Module Teardown Order

```bash
# 1. Revoke Lake Formation permissions
aws lakeformation revoke-permissions --principal DataLakePrincipalIdentifier=...

# 2. Deregister Lake Formation resource
aws lakeformation deregister-resource --resource-arn arn:aws:s3:::binfx410-datalake-STUDENT_ID

# 3. Delete Kinesis Firehose streams
aws firehose delete-delivery-stream --delivery-stream-name binfx410-sequencer-feed

# 4. Delete Lambda functions
aws lambda delete-function --function-name binfx410-sequencer-emitter

# 5. Delete Glue databases (cascade deletes tables)
for db in binfx410_bronze binfx410_silver binfx410_gold binfx410_sra; do
  aws glue delete-database --name $db
done

# 6. Empty and delete S3 buckets
aws s3 rm s3://binfx410-datalake-STUDENT_ID --recursive
aws s3 rb s3://binfx410-datalake-STUDENT_ID
aws s3 rm s3://binfx410-athena-results-STUDENT_ID --recursive
aws s3 rb s3://binfx410-athena-results-STUDENT_ID

# 7. Delete IAM roles created during exercises
aws iam delete-role --role-name binfx410-researcher-role
aws iam delete-role --role-name binfx410-clinical-role
```

---

## 10. Extension Ideas

For students who complete all exercises early:

### Extension A: Delta Lake vs Iceberg
Compare Delta Lake (via Apache Spark on EMR Serverless) and Apache Iceberg (via Athena) using the same ClinVar dataset. Benchmark: CREATE, INSERT, UPDATE, time travel query performance and cost.

### Extension B: dbt Full Implementation
Set up a real dbt project with `dbt-athena-community` adapter. Implement the full star schema from Exercise 5 as a dbt project with tests, documentation, and lineage DAG visualization.

### Extension C: Multi-Study RNA-seq Meta-Analysis
Pull 5–10 RNA-seq studies from SRA (different cancer types) into the Bronze zone. Build a Silver layer that harmonizes gene IDs across studies (Ensembl vs gene symbol). Gold layer: cross-study differential expression analysis.

### Extension D: Variant Annotation Pipeline
Build an ELT pipeline that takes raw 1000 Genomes VCF variants, annotates them with ClinVar pathogenicity (join on CHROM + POS + REF + ALT), and produces a Gold layer table of clinically relevant variants per population.

### Extension E: Real-Time Dashboard
Use AWS QuickSight (free tier) connected to Athena to build a dashboard over the Exercise 9 streaming data: instrument utilization by lab, yield trends, Q30% distribution by flow cell type.

### Extension F: Cost Optimization Lab
Given a fixed Athena budget of $1.00 for the month, optimize the Exercise 7 queries to use the minimum data scanned while still returning correct results. Techniques: partition projection, column pruning, result caching, materialized views.
