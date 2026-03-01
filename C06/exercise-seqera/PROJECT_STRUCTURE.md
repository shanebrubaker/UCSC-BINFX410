# Project Structure

This document describes the files in this Sequence QC Pipeline project.

## Core Workflow Files

### `main.nf` (16 KB)
The main NextFlow workflow definition file. Contains 7 processes:
1. `SPLIT_REFERENCE` - Splits FASTA by chromosome
2. `ALIGN_TO_CHROMOSOME` - Aligns reads to each chromosome
3. `QC_CHROMOSOME` - Initial QC per chromosome
4. `TRIM_AND_CLEAN` - Adapter trimming and quality filtering
5. `COMBINE_CLEANED` - Combines all cleaned chromosomes
6. `GENERATE_QC_REPORT` - Creates final QC report with visualizations
7. `CHROMOSOME_QC_SUMMARY` - Generates per-chromosome pass/fail summary

### `nextflow.config` (3 KB)
NextFlow configuration file with:
- Pipeline parameters and defaults
- Process resource allocations
- Execution profiles (standard, docker, conda, cluster)
- Trace and reporting settings

## Data Generation Scripts

### `generate_reference.py` (2.4 KB)
Generates a faux reference genome with 10 chromosomes.

**Features:**
- Chromosomes range from 50kb to 200kb
- Variable GC content per chromosome
- Reproducible with seed parameter

**Usage:**
```bash
python3 generate_reference.py -o reference.fasta -s 42
```

### `generate_fastq.py` (7.9 KB)
Generates faux FASTQ data with various quality issues.

**Quality Issues Introduced:**
- 5% overrepresented sequences (PCR duplicates)
- 10% low quality reads (Q10-Q25)
- 10% adapter contamination
- 5% reads with N bases
- 20% reads with end quality degradation
- 50% normal quality reads

**Usage:**
```bash
python3 generate_fastq.py -o reads.fastq -n 100000 -r reference.fasta
```

## Analysis and Reporting Scripts

### `qc_analysis.py` (13 KB)
QC analysis and visualization script that generates:

**5 Visualizations:**
1. Quality distribution histogram
2. Per-position quality scores (line plot with quartiles)
3. Read length distribution after trimming
4. GC content distribution
5. Per-chromosome read counts (before vs after cleaning)

**Outputs:**
- HTML report with embedded plots
- JSON summary with statistics
- Individual PNG plot files

**Usage:**
```bash
qc_analysis.py --fastq combined.fastq --initial-qc "*.json" \
    --trim-stats "*.json" --output-html report.html \
    --output-json summary.json --plot-dir plots/
```

## Helper Scripts

### `run_test.sh` (3.1 KB)
Automated test script that:
1. Checks NextFlow installation
2. Checks Python and dependencies
3. Makes scripts executable
4. Generates test data
5. Runs complete pipeline
6. Displays results location

**Usage:**
```bash
./run_test.sh
```

### `.gitignore` (342 bytes)
Git ignore file to exclude:
- Generated data files (*.fasta, *.fastq)
- Results directories
- NextFlow work directory
- Python cache
- IDE and OS files

## Documentation

### `README.md` (11 KB)
Comprehensive documentation including:
- Overview and features
- Installation instructions for all dependencies
- Testing procedures
- Usage examples and parameters
- Troubleshooting guide
- Advanced usage (conda, cluster)

### `QUICK_START.md` (3.4 KB)
Quick start guide for getting running in 5 minutes:
- Minimal installation steps
- Automated and manual test procedures
- Common commands
- Quick troubleshooting

### `PROJECT_STRUCTURE.md` (This file)
Describes all files in the project and their purposes.

## Workflow Data Flow

```
Input Files:
  reference.fasta (generated)
  reads.fastq (generated)
         ↓
    [main.nf workflow]
         ↓
Output Structure:
  results/
  ├── chromosomes/          # 10 chromosome FASTA files
  ├── aligned/              # 10 chromosome subset FASTQ files
  ├── qc_initial/          # 10 initial QC JSON files
  ├── cleaned/             # 10 cleaned FASTQ files + stats
  ├── combined_cleaned.fastq
  ├── qc_report/
  │   ├── qc_report.html
  │   ├── qc_summary.json
  │   ├── chromosome_qc_summary.txt
  │   ├── chromosome_qc_summary.json
  │   └── qc_plots/
  │       ├── quality_distribution.png
  │       ├── quality_per_position.png
  │       ├── read_length_distribution.png
  │       ├── gc_content.png
  │       └── chromosome_comparison.png
  └── trace/
      ├── execution_trace.txt
      ├── execution_timeline.html
      ├── execution_report.html
      └── pipeline_dag.html
```

## File Dependencies

```
Main Workflow:
  main.nf
  ├── requires: nextflow.config
  ├── requires: qc_analysis.py
  ├── input: reference.fasta
  └── input: reads.fastq

Test Data Generation:
  generate_reference.py → reference.fasta
  generate_fastq.py
  ├── requires: reference.fasta
  └── output: reads.fastq

Automated Testing:
  run_test.sh
  ├── runs: generate_reference.py
  ├── runs: generate_fastq.py
  └── runs: nextflow main.nf
```

## Size Information

Total repository size (without generated data): ~60 KB

Expected sizes with test data:
- `reference.fasta`: ~1.2 MB (10 chromosomes)
- `reads.fastq`: ~20-60 MB (depending on read count)
- `results/`: ~40-100 MB (full pipeline output)
- `work/`: ~100-200 MB (NextFlow cache)

## Technologies Used

- **NextFlow**: Workflow management (DSL2)
- **Python 3**: Data generation and analysis
- **matplotlib**: Visualization library
- **numpy**: Numerical computing
- **Bash**: Testing and automation

## Key Features

1. **Parallel Processing**: Each chromosome processed independently
2. **Quality Control**: Multi-level QC at chromosome and final stages
3. **Comprehensive Reporting**: HTML reports with interactive visualizations
4. **Reproducible**: Seed-based random data generation
5. **Flexible**: Configurable parameters and execution profiles
6. **Well-Documented**: Multiple documentation files for different needs

## Getting Started

1. **Quick Start**: See [QUICK_START.md](QUICK_START.md)
2. **Full Documentation**: See [README.md](README.md)
3. **Run Automated Test**: Execute `./run_test.sh`

## License

MIT License - See LICENSE file for details
