# BINFX410 Apache Spark Exercise Module - Implementation Plan

## Project Overview
Create a complete, self-contained exercise module teaching Apache Spark concepts using synthetic genomics data. Students will run everything locally on their laptops without requiring cloud resources.

## Learning Objectives
1. Understand distributed data processing concepts (partitioning, shuffling, lazy evaluation)
2. Use Spark DataFrame API for genomics data analysis
3. Apply quality control filters to variant data
4. Perform aggregations and joins on large-scale genomic datasets
5. Optimize Spark queries and understand performance tradeoffs
6. Compare Spark vs. pandas for different data scales

## Directory Structure to Create

```
spark_genomics_exercise/
├── README.md                          # Overview and setup instructions
├── docker/
│   ├── Dockerfile                     # Container with PySpark + dependencies
│   └── docker-compose.yml             # Easy startup for students
├── data/
│   ├── generate_data.py               # Script to create synthetic datasets
│   ├── README.md                      # Data documentation
│   └── .gitkeep                       # Ensure directory exists
├── notebooks/
│   ├── 00_setup_and_test.ipynb       # Verify installation works
│   ├── 01_spark_basics.ipynb         # Introduction to Spark concepts
│   ├── 02_variant_qc_exercise.ipynb  # Main exercise (student version)
│   ├── 02_variant_qc_solutions.ipynb # Solutions (instructor version)
│   └── 03_advanced_concepts.ipynb    # Bonus material
├── scripts/
│   ├── timing_utils.py                # Helper functions for performance measurement
│   └── viz_utils.py                   # Plotting helpers for results
├── exercises/
│   ├── exercise_1_filtering.md        # Written exercises with questions
│   ├── exercise_2_aggregation.md
│   └── exercise_3_optimization.md
└── instructor/
    ├── teaching_notes.md              # Tips for teaching this module
    ├── grading_rubric.md              # How to assess student work
    └── common_issues.md               # Troubleshooting FAQ
```

## File Specifications

### 1. data/generate_data.py

**Purpose**: Generate realistic synthetic variant data at multiple scales

**Features**:
- Command-line interface: `python generate_data.py --size small|medium|large`
- Creates VCF-like CSV files with realistic distributions
- Generates companion files (gene annotations, sample metadata)
- Reproducible (seeded random generation)
- Well-documented data schema

**Data Schema for Variants**:
```python
{
    'CHROM': 'chr1-chr22, chrX, chrY',
    'POS': 'int, position on chromosome',
    'ID': 'rs12345678 or . for novel',
    'REF': 'A, C, G, T',
    'ALT': 'A, C, G, T',
    'QUAL': 'float, 0-100 (Phred-scaled)',
    'FILTER': 'PASS, LowQual, etc.',
    'DP': 'int, read depth 1-100',
    'AF': 'float, allele frequency 0-1',
    'GQ': 'int, genotype quality 0-99',
    'GENE': 'string, associated gene name or .',
    'CONSEQUENCE': 'missense_variant, synonymous_variant, etc.',
}
```

**Dataset Sizes**:
- Small: 1,000 variants (~200 KB) - for quick testing
- Medium: 100,000 variants (~20 MB) - main exercise dataset
- Large: 1,000,000 variants (~200 MB) - bonus performance comparisons

**Companion Files**:
- `genes.bed`: Gene annotations (chr, start, end, name, strand)
- `samples.csv`: Sample metadata (sample_id, population, sex, phenotype)
- `known_variants.csv`: Database of "known" pathogenic variants for joins

**Implementation Notes**:
- Use numpy for efficient random generation
- Make chromosome distribution realistic (chr1 is longer than chr22)
- Include realistic quality score distributions (bimodal: high-quality and low-quality)
- Add correlated features (e.g., high DP correlates with high GQ)
- Generate some variants in genes, some intergenic

### 2. docker/Dockerfile

**Purpose**: Provide consistent environment for all students

**Base Image**: python:3.10-slim

**Installed Packages**:
```
pyspark==3.5.0
jupyter==1.0.0
jupyterlab==4.0.0
pandas==2.1.0
matplotlib==3.8.0
seaborn==0.13.0
numpy==1.26.0
```

**Configuration**:
- Expose port 8888 for Jupyter
- Expose port 4040 for Spark UI
- Set working directory to /workspace
- Copy in helper scripts
- Create user 'student' (not root)
- Set appropriate permissions

**Entry Point**: Launch Jupyter Lab with token displayed

### 3. docker/docker-compose.yml

**Purpose**: One-command startup

**Configuration**:
```yaml
version: '3.8'
services:
  spark-jupyter:
    build: .
    ports:
      - "8888:8888"  # Jupyter
      - "4040:4040"  # Spark UI
    volumes:
      - ../data:/workspace/data
      - ../notebooks:/workspace/notebooks
      - ../scripts:/workspace/scripts
    environment:
      - SPARK_LOCAL_DIRS=/tmp/spark
```

**Usage**: `docker-compose up` starts everything

### 4. notebooks/00_setup_and_test.ipynb

**Purpose**: Verify installation and introduce Spark

**Contents**:
1. Import test (PySpark, pandas, etc.)
2. SparkSession creation with explanation
3. Simple "Hello World" DataFrame operations
4. Check Spark UI is accessible (http://localhost:4040)
5. Generate small test dataset and verify loading
6. Basic operations: filter, select, show, count
7. Success confirmation message

**Pedagogical Notes**:
- Very hand-holdy for first notebook
- Lots of explanatory text
- Screenshots of what Spark UI should look like
- Troubleshooting section for common issues

### 5. notebooks/01_spark_basics.ipynb

**Purpose**: Teach core Spark concepts before main exercise

**Sections**:

**Section 1: Lazy Evaluation** (15 min)
- Transformations vs. Actions
- Build a query plan without executing
- Use .explain() to see physical plan
- Demonstrate with timing comparisons

**Section 2: Partitions** (20 min)
- What are partitions and why do they matter?
- Check number of partitions: `df.rdd.getNumPartitions()`
- Repartition examples
- Show partition contents
- Explain data locality concept

**Section 3: Common Operations** (25 min)
- Filtering: `filter()`, `where()`
- Selecting: `select()`, `selectExpr()`
- Aggregating: `groupBy()`, `agg()`
- Sorting: `orderBy()`
- Joining: `join()` different types
- Each with genomics examples

**Section 4: Understanding Shuffles** (20 min)
- What is a shuffle?
- Which operations trigger shuffles?
- Use .explain() to identify shuffles
- Measure shuffle cost with timing
- Strategies to minimize shuffles

**Section 5: Caching** (10 min)
- When to cache
- Different storage levels
- Performance comparison: cached vs. uncached
- How to uncache

### 6. notebooks/02_variant_qc_exercise.ipynb (Student Version)

**Purpose**: Main hands-on exercise applying learned concepts

**Exercise Flow**:

**Introduction** (5 min reading)
- Scenario: You're analyzing variants from 200 whole-genome samples
- Goal: Perform quality control and generate summary statistics
- Data description and schema explanation

**Task 1: Load and Explore Data** (10 min)
```python
# TODO: Load the medium variant dataset (100K variants)
# Hint: Use spark.read.csv() with header=True, inferSchema=True

variants = # YOUR CODE HERE

# TODO: Display the schema
# TODO: Show the first 10 rows
# TODO: Count total variants

# Expected output: 100,000 variants
```

**Task 2: Basic Quality Filtering** (15 min)
```python
# TODO: Apply GATK-recommended hard filters:
# - QUAL >= 30
# - DP >= 10
# - GQ >= 20

# Count how many variants pass all filters
# What percentage of variants are filtered out?

# QUESTION: Look at the Spark UI. Did this operation cause a shuffle? Why or why not?
```

**Task 3: Per-Chromosome Statistics** (20 min)
```python
# TODO: For each chromosome, calculate:
# - Total variant count
# - Mean QUAL score
# - Mean depth (DP)
# - Number of PASS variants
# - Pass rate (PASS / total)

# Sort by chromosome (chr1, chr2, ..., chrX, chrY)

# QUESTION: Which chromosome has the highest variant density?
# QUESTION: Look at the Spark UI - how many stages did this operation have?
```

**Task 4: Allele Frequency Stratification** (20 min)
```python
# TODO: Categorize variants by allele frequency:
# - Singleton: AF < 0.001
# - Rare: 0.001 <= AF < 0.01
# - Low frequency: 0.01 <= AF < 0.05
# - Common: AF >= 0.05

# Create a new column called "frequency_class"

# TODO: Count variants in each frequency class per chromosome

# QUESTION: Are rare variants uniformly distributed across chromosomes?
```

**Task 5: Join with Gene Annotations** (25 min)
```python
# TODO: Load the gene annotations file (genes.bed)
# Schema: chrom, start, end, gene_name, strand

# TODO: Join variants with genes based on position
# A variant at position POS is in a gene if: start <= POS <= end

# HINT: You'll need to join on CHROM and add a condition for position

# TODO: Count how many variants fall within genes
# TODO: Find the top 10 genes with the most variants

# QUESTION: What percentage of variants are intergenic?
# QUESTION: Did this join cause a shuffle? Check the Spark UI.
```

**Task 6: Performance Comparison** (20 min)
```python
# Compare Spark vs. pandas on the same task

# TODO: Using pandas, load the medium dataset and calculate
# per-chromosome variant counts

# TODO: Using Spark (cached), do the same calculation

# TODO: Time both approaches and compare

# QUESTIONS:
# - Which is faster on this dataset?
# - At what data size would Spark become necessary?
# - What's the overhead of using Spark?
```

**Task 7: Optimization Challenge** (30 min)
```python
# The query below finds rare coding variants (in genes, AF < 0.01):

result = variants \
    .join(genes, on="CHROM") \
    .filter((col("POS") >= col("start")) & (col("POS") <= col("end"))) \
    .filter(col("AF") < 0.01) \
    .select("CHROM", "POS", "gene_name", "AF")

# TODO: Measure execution time for this query

# TODO: Optimize this query. Try:
# 1. Filter before join
# 2. Broadcast the smaller table (genes)
# 3. Repartition by CHROM
# 4. Cache intermediate results

# Which optimization(s) helped the most?

# TODO: Use .explain() to see the physical plan before and after optimization
```

**Task 8: Write Results** (10 min)
```python
# TODO: Save the filtered, high-quality variants to Parquet format

# TODO: Compare file sizes:
# - Original CSV
# - Parquet (uncompressed)
# - Parquet (compressed - default)

# QUESTION: Why is Parquet preferred for big data?
```

**Reflection Questions** (Written responses)
1. When would you use Spark instead of pandas for genomics analysis?
2. What are the main costs in distributed computing? (Hint: not just money)
3. How would you design a Spark job to analyze 100,000 whole genomes?
4. What did you learn from the Spark UI about how your queries executed?

### 7. notebooks/02_variant_qc_solutions.ipynb (Instructor Version)

**Purpose**: Complete solutions with additional explanations

**Contents**:
- All tasks completed with well-commented code
- Multiple solution approaches shown where applicable
- Performance measurements and comparisons
- Spark UI screenshots showing shuffle stages
- Extended explanations of why certain approaches are better
- Common mistakes and how to avoid them
- Additional challenge problems for advanced students

### 8. notebooks/03_advanced_concepts.ipynb (Bonus)

**Purpose**: Optional material for advanced students

**Topics**:

**Window Functions** (30 min)
- Calculating running statistics
- Ranking variants within genes
- Finding top N variants per chromosome

**UDFs (User-Defined Functions)** (20 min)
- When to use UDFs
- Performance implications
- Vectorized UDFs (pandas_udf)
- Example: Custom variant consequence predictor

**Partitioning Strategies** (25 min)
- Hash partitioning
- Range partitioning
- Custom partitioners
- When each is appropriate

**Machine Learning Preview** (30 min)
- Feature engineering for variant classification
- VectorAssembler
- Simple RandomForest classifier
- Train/test split
- Evaluation metrics

### 9. scripts/timing_utils.py

**Purpose**: Reusable timing and performance utilities

**Functions**:
```python
def time_operation(func, *args, **kwargs):
    """Time a Spark operation and return result + duration"""
    
def compare_operations(operations_dict):
    """Time multiple operations and return comparison DataFrame"""
    
def spark_ui_link(spark_session):
    """Print clickable link to Spark UI"""
    
def partition_info(df):
    """Display partition count and estimated size per partition"""
    
def explain_plan(df, mode='simple'):
    """Pretty-print Spark execution plan"""
```

### 10. scripts/viz_utils.py

**Purpose**: Plotting helpers for genomics data

**Functions**:
```python
def plot_chromosome_distribution(df, value_col):
    """Bar plot of values by chromosome"""
    
def plot_quality_histogram(df):
    """Distribution of QUAL scores"""
    
def plot_af_distribution(df):
    """Allele frequency spectrum"""
    
def plot_performance_comparison(timing_results):
    """Bar chart comparing operation times"""
    
def plot_shuffle_analysis(spark_ui_metrics):
    """Visualize shuffle read/write volumes"""
```

### 11. README.md (Root)

**Sections**:
1. **Overview**: What this module teaches
2. **Prerequisites**: Python basics, genomics concepts
3. **Setup Instructions**: 
   - Docker installation
   - Starting the environment
   - Accessing Jupyter
   - Generating data
4. **Exercise Structure**: What's in each notebook
5. **Time Estimates**: How long each part takes
6. **Learning Path**: Recommended order
7. **Getting Help**: Common issues, where to ask questions
8. **Additional Resources**: Links to Spark docs, tutorials

### 12. instructor/teaching_notes.md

**Contents**:
- Suggested lecture outline before assigning exercises
- Key points to emphasize in each section
- Common student misconceptions about distributed computing
- Live demo suggestions
- Discussion prompts for class
- Connections to real-world bioinformatics pipelines
- How this connects to cloud computing (AWS, GCP)

### 13. instructor/grading_rubric.md

**Assessment Categories**:

**Code Functionality (40%)**
- Does code run without errors?
- Are results correct?
- Proper use of Spark API?

**Understanding (30%)**
- Written answers demonstrate comprehension?
- Can explain when Spark is appropriate?
- Understands shuffles and partitioning?

**Optimization (20%)**
- Successfully optimized queries?
- Used appropriate techniques (caching, filtering, broadcasting)?
- Can explain performance tradeoffs?

**Code Quality (10%)**
- Well-commented code?
- Good variable names?
- Organized and readable?

### 14. instructor/common_issues.md

**Troubleshooting Guide**:

**"Spark UI not accessible"**
- Check port 4040 is exposed
- Verify Spark session is active
- Multiple sessions on same port

**"Out of memory errors"**
- Reduce dataset size
- Increase executor memory
- Check for unnecessary caching

**"Operations very slow"**
- Too many partitions
- Not enough memory
- Data skew issues

**"Import errors"**
- Python version mismatch
- Missing dependencies
- Environment activation

## Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
1. ✓ Create directory structure
2. ✓ Build Docker environment
3. ✓ Write data generation script
4. ✓ Test environment end-to-end

### Phase 2: Educational Content (Week 2)
5. ✓ Write setup notebook (00)
6. ✓ Write basics notebook (01)
7. ✓ Create helper scripts
8. ✓ Generate test datasets

### Phase 3: Main Exercise (Week 3)
9. ✓ Write student exercise notebook (02)
10. ✓ Create solutions notebook
11. ✓ Test all exercises with fresh eyes
12. ✓ Write reflection questions

### Phase 4: Documentation (Week 4)
13. ✓ Complete README with setup instructions
14. ✓ Write teaching notes
15. ✓ Create grading rubric
16. ✓ Document common issues

### Phase 5: Testing & Refinement (Week 5)
17. ✓ Have colleague test-run exercises
18. ✓ Gather feedback and iterate
19. ✓ Add bonus material (03)
20. ✓ Final polish and release

## Technical Requirements

**Student Machines**:
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Docker Desktop installed
- Web browser (for Jupyter)

**Software Versions**:
- Python 3.10
- PySpark 3.5.0
- Jupyter Lab 4.0+
- Docker 20.10+

**Performance Expectations**:
- Small dataset: < 5 seconds for most operations
- Medium dataset: 10-30 seconds for complex queries
- Large dataset: 1-3 minutes for full pipeline

## Success Metrics

**Students should be able to**:
1. ✓ Explain when distributed computing is necessary
2. ✓ Write basic Spark DataFrame queries
3. ✓ Identify operations that cause shuffles
4. ✓ Use Spark UI to diagnose performance issues
5. ✓ Optimize queries using caching and filtering
6. ✓ Compare Spark vs. single-machine approaches

**Instructors should observe**:
1. ✓ High engagement during exercises
2. ✓ Conceptual understanding in discussions
3. ✓ Successful completion of main tasks
4. ✓ Creative approaches to optimization challenges

## Future Enhancements

**Possible Additions**:
- MLlib exercise for variant classification
- Streaming exercise simulating real-time sequencing
- Integration with Nextflow for pipeline orchestration
- AWS EMR deployment guide (optional cloud module)
- Comparison with Dask as alternative framework
- Advanced optimization with Catalyst optimizer details

## Questions for Course Director

Before implementing, confirm:
1. Is 5 weeks reasonable timeline for development?
2. Should this be graded or pass/fail?
3. Include in Docker image or separate setup?
4. Real genomics data vs. synthetic only?
5. Individual work or group project?
6. Integration with existing course Nextflow content?

## Notes for Claude Code

When implementing:
- Follow PEP 8 style for all Python code
- Include comprehensive docstrings
- Add type hints where appropriate
- Use meaningful variable names
- Comment complex logic thoroughly
- Test on both Mac and Linux if possible
- Ensure all paths use os.path for cross-platform compatibility
- Make data generation deterministic (seeded random)
- Include progress bars for long operations
- Provide informative error messages

Focus on pedagogical clarity over code brevity. Students are learning both Spark AND distributed computing concepts, so explanations matter more than conciseness.
