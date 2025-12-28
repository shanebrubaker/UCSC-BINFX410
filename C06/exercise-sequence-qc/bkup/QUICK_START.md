# Quick Start Guide

This guide will get you up and running with the Sequence QC Pipeline in 5 minutes.

## Prerequisites

- Java 8+ installed
- Python 3.7+ installed
- 2 GB free disk space

## Installation (2 minutes)

### 1. Install NextFlow

```bash
# Quick install
curl -s https://get.nextflow.io | bash

# Move to PATH (optional but recommended)
sudo mv nextflow /usr/local/bin/
```

### 2. Install Python Dependencies

```bash
pip install matplotlib numpy
```

## Run the Test (3 minutes)

### Option A: Automated Test (Recommended)

```bash
./run_test.sh
```

This script will:
- Check all dependencies
- Generate test data
- Run the complete pipeline
- Display results location

### Option B: Manual Test

```bash
# 1. Generate test data
python3 generate_reference.py -o reference.fasta
python3 generate_fastq.py -o reads.fastq -n 10000 -r reference.fasta

# 2. Run pipeline
nextflow run main.nf

# 3. View results
open results/qc_report/qc_report.html  # macOS
xdg-open results/qc_report/qc_report.html  # Linux
```

## View Results

After the pipeline completes:

```bash
# View QC report in browser
open results/qc_report/qc_report.html

# View chromosome summary
cat results/qc_report/chromosome_qc_summary.txt

# View execution timeline
open results/trace/execution_timeline.html
```

## Common Commands

### Generate Custom Data

```bash
# Different number of reads
python3 generate_fastq.py -o reads.fastq -n 50000

# Different read length
python3 generate_fastq.py -o reads.fastq -l 100
```

### Run with Custom Parameters

```bash
nextflow run main.nf \
    --min_quality 30 \
    --min_length 75 \
    --outdir my_results
```

### Resume Failed Run

```bash
nextflow run main.nf -resume
```

### Clean Up

```bash
# Remove results
rm -rf results/

# Remove test data
rm reference.fasta reads.fastq

# Remove NextFlow work directory
rm -rf work/ .nextflow*
```

## Expected Output

```
results/
├── combined_cleaned.fastq          ← Final cleaned reads
├── qc_report/
│   ├── qc_report.html             ← Main QC report (open this!)
│   ├── chromosome_qc_summary.txt  ← Per-chromosome summary
│   └── qc_plots/                  ← 5 visualization plots
└── trace/
    ├── execution_timeline.html    ← Execution timeline
    └── execution_report.html      ← Resource usage report
```

## Troubleshooting

### NextFlow not found

```bash
# Add NextFlow to PATH
export PATH=$PATH:$(pwd)
```

### Permission denied

```bash
chmod +x generate_reference.py generate_fastq.py qc_analysis.py run_test.sh
```

### Module not found

```bash
pip install matplotlib numpy
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize parameters in `nextflow.config`
- Run on your own data:
  ```bash
  nextflow run main.nf --reference your_ref.fasta --reads your_reads.fastq
  ```

## Need Help?

- Check the full documentation: [README.md](README.md)
- View the troubleshooting section in README.md
- Check NextFlow documentation: https://www.nextflow.io/docs/latest/

## Quick Reference

| Command | Description |
|---------|-------------|
| `./run_test.sh` | Run complete automated test |
| `nextflow run main.nf` | Run pipeline |
| `nextflow run main.nf -resume` | Resume failed run |
| `nextflow run main.nf -preview` | Preview workflow |
| `nextflow log` | View execution history |
| `nextflow clean -f` | Clean work directory |

---

For complete documentation, see [README.md](README.md)
