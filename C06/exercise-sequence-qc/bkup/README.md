# Sequence QC Pipeline with Chromosome-Level Processing

A comprehensive NextFlow workflow for performing quality control on sequencing data with chromosome-level processing, adapter trimming, and detailed QC reporting.

## Overview

This pipeline performs the following steps:

1. **Split Reference**: Divides a reference FASTA file into separate chromosomes
2. **Align Reads**: Aligns FASTQ reads to each chromosome separately
3. **Initial QC**: Performs quality control analysis on each chromosome's subset
4. **Trim & Clean**: Removes adapters and filters low-quality reads
5. **Combine**: Merges all cleaned chromosome subsets back together
6. **Final QC Report**: Generates visualizations and comprehensive QC metrics
7. **Chromosome Summary**: Produces per-chromosome QC pass/fail summary

## Features

- **Chromosome-level processing**: Processes each chromosome independently for parallel execution
- **Quality issue detection**: Identifies low-quality reads, adapter contamination, and N bases
- **Adapter trimming**: Removes Illumina adapter sequences
- **Quality filtering**: Filters reads based on quality scores and length
- **Comprehensive reporting**: Generates 5 detailed visualizations and HTML report
- **Per-chromosome QC**: Shows which chromosomes passed QC criteria
- **Test data generation**: Includes scripts to generate faux reference genome and FASTQ data

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Memory**: At least 4 GB RAM
- **Disk Space**: At least 2 GB free space
- **Python**: Version 3.7 or higher
- **Java**: Version 8 or higher (for NextFlow)

### Software Dependencies

- NextFlow (>= 21.04.0)
- Python 3.7+
- Python packages: matplotlib, numpy

## Installation

### 1. Install Java

NextFlow requires Java 8 or later.

**macOS** (using Homebrew):
```bash
brew install openjdk@11
```

**Linux** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install openjdk-11-jdk
```

**Verify Java installation**:
```bash
java -version
```

### 2. Install NextFlow

**Option A: Quick install** (recommended):
```bash
curl -s https://get.nextflow.io | bash
```

This downloads the `nextflow` executable to your current directory.

**Option B: Using Conda**:
```bash
conda install -c bioconda nextflow
```

**Make NextFlow executable and move to PATH**:
```bash
chmod +x nextflow
sudo mv nextflow /usr/local/bin/
```

Or add to your current directory to PATH:
```bash
export PATH=$PATH:$(pwd)
```

### 3. Verify NextFlow Installation

```bash
nextflow -version
```

You should see output like:
```
nextflow version 23.10.0.5889
```

### 4. Install Python Dependencies

```bash
pip install matplotlib numpy
```

Or using conda:
```bash
conda install matplotlib numpy
```

### 5. Make Scripts Executable

```bash
chmod +x generate_reference.py generate_fastq.py qc_analysis.py
```

## Testing the Installation

### Test 1: Check NextFlow

Run the NextFlow test command:
```bash
nextflow info
```

This should display NextFlow system information without errors.

### Test 2: Generate Test Data

Generate a small reference genome and FASTQ file for testing:

```bash
# Generate reference genome (10 chromosomes, total ~1.2 Mb)
python3 generate_reference.py -o reference.fasta

# Generate FASTQ data (10,000 reads with quality issues)
python3 generate_fastq.py -o reads.fastq -n 10000 -r reference.fasta
```

Verify the files were created:
```bash
ls -lh reference.fasta reads.fastq
```

### Test 3: Run a Dry Run

Test the workflow without executing:
```bash
nextflow run main.nf -preview
```

This validates the workflow syntax without running the processes.

### Test 4: Run the Complete Pipeline

Run the full pipeline on the test data:
```bash
nextflow run main.nf
```

This will:
- Process the reference genome
- Align and QC each chromosome
- Generate the final report

Expected output directory structure:
```
results/
├── chromosomes/          # Individual chromosome FASTA files
├── aligned/              # Aligned reads per chromosome
├── qc_initial/          # Initial QC metrics per chromosome
├── cleaned/             # Cleaned FASTQ files per chromosome
├── combined_cleaned.fastq  # Final combined cleaned reads
├── qc_report/           # Final QC report and visualizations
│   ├── qc_report.html
│   ├── qc_plots/
│   │   ├── quality_distribution.png
│   │   ├── quality_per_position.png
│   │   ├── read_length_distribution.png
│   │   ├── gc_content.png
│   │   └── chromosome_comparison.png
│   ├── qc_summary.json
│   ├── chromosome_qc_summary.txt
│   └── chromosome_qc_summary.json
└── trace/               # Execution traces and reports
    ├── execution_trace.txt
    ├── execution_timeline.html
    ├── execution_report.html
    └── pipeline_dag.html
```

## Usage

### Basic Usage

```bash
nextflow run main.nf
```

### Custom Parameters

```bash
nextflow run main.nf \
    --reference my_reference.fasta \
    --reads my_reads.fastq \
    --outdir my_results \
    --min_quality 30 \
    --min_length 75
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reference` | `reference.fasta` | Input reference genome FASTA file |
| `--reads` | `reads.fastq` | Input FASTQ file with reads |
| `--outdir` | `results` | Output directory for results |
| `--adapter` | `AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC` | Illumina adapter sequence to trim |
| `--min_quality` | `25` | Minimum average quality score for reads |
| `--min_length` | `50` | Minimum read length after trimming |

### Resume Failed Runs

NextFlow supports resuming from the last successful step:

```bash
nextflow run main.nf -resume
```

## Generating Custom Test Data

### Custom Reference Genome

```bash
python3 generate_reference.py \
    --output my_reference.fasta \
    --seed 12345  # For reproducibility
```

### Custom FASTQ Data

```bash
python3 generate_fastq.py \
    --output my_reads.fastq \
    --num-reads 100000 \
    --read-length 150 \
    --reference my_reference.fasta \
    --seed 12345
```

The FASTQ generator creates reads with various quality issues:
- 5% overrepresented sequences (PCR duplicates)
- 10% low quality reads
- 10% adapter contamination
- 5% reads with N bases
- 20% reads with quality degradation toward the end
- 50% normal quality reads

## Understanding the Results

### QC Report (`qc_report.html`)

Open the HTML report in a browser to view:

1. **Summary Statistics**: Total reads, bases, average quality, GC content
2. **Quality Distribution**: Histogram of read quality scores
3. **Per-Position Quality**: Quality scores across read positions
4. **Read Length Distribution**: Distribution of read lengths after trimming
5. **GC Content**: GC percentage distribution
6. **Chromosome Comparison**: Read counts before/after cleaning per chromosome

### Chromosome QC Summary (`chromosome_qc_summary.txt`)

Shows per-chromosome QC status with pass/fail criteria:

- **Pass rate** ≥ 50% of reads pass filtering
- **Average quality** ≥ Q25
- **Adapter contamination** < 20%

Example:
```
Chromosome: chr1
  Status: PASS
  Input reads: 12,543
  Cleaned reads: 9,234
  Pass rate: 73.6%
  Avg quality: 31.2
  Adapter contamination: 8.3%
```

### Execution Reports

NextFlow generates several execution reports in `results/trace/`:

- **execution_trace.txt**: Detailed process execution information
- **execution_timeline.html**: Visual timeline of process execution
- **execution_report.html**: Resource usage and performance metrics
- **pipeline_dag.html**: Directed acyclic graph of workflow

## Troubleshooting

### Issue: "Command not found: nextflow"

**Solution**: Ensure NextFlow is in your PATH:
```bash
export PATH=$PATH:/usr/local/bin
```

### Issue: "Unable to find image 'python:3.9-slim'"

**Solution**: If using Docker profile, ensure Docker is installed and running. Otherwise, use the standard profile:
```bash
nextflow run main.nf -profile standard
```

### Issue: Python module not found (matplotlib, numpy)

**Solution**: Install required Python packages:
```bash
pip install matplotlib numpy
```

### Issue: Java version too old

**Solution**: Update Java to version 8 or higher:
```bash
# macOS
brew install openjdk@11

# Linux
sudo apt-get install openjdk-11-jdk
```

### Issue: Out of memory errors

**Solution**: Increase memory allocation in `nextflow.config`:
```groovy
process {
    memory = '4 GB'  // Increase as needed
}
```

### Issue: Permission denied on scripts

**Solution**: Make scripts executable:
```bash
chmod +x generate_reference.py generate_fastq.py qc_analysis.py
```

## Advanced Usage

### Running with Conda Environment

Create a conda environment:
```bash
conda create -n seqqc python=3.9 matplotlib numpy nextflow
conda activate seqqc
```

Run the pipeline:
```bash
nextflow run main.nf -profile conda
```

### Running on a Cluster

Edit `nextflow.config` to configure cluster settings:
```groovy
profiles {
    cluster {
        process.executor = 'slurm'
        process.queue = 'normal'
        process.clusterOptions = '--account=your_account'
    }
}
```

Run with cluster profile:
```bash
nextflow run main.nf -profile cluster
```

### Customizing QC Thresholds

Edit the QC pass criteria in `main.nf` (Process: CHROMOSOME_QC_SUMMARY):
```python
# Adjust these values as needed
pass_rate >= 50      # Minimum 50% reads pass
avg_quality >= 25    # Minimum Q25
adapter_contamination < 20  # Maximum 20% contamination
```

## Pipeline Architecture

```
Input FASTA → Split by Chromosome → Align Reads → Initial QC
                                          ↓
                                    Trim & Clean
                                          ↓
                                    [Per Chromosome]
                                          ↓
                                    Combine All
                                          ↓
                              Final QC + Visualizations
                                          ↓
                              Chromosome Summary Report
```

## File Descriptions

- **main.nf**: Main NextFlow workflow definition
- **nextflow.config**: Configuration file for the pipeline
- **generate_reference.py**: Script to generate faux reference genome
- **generate_fastq.py**: Script to generate faux FASTQ data with quality issues
- **qc_analysis.py**: Script for QC analysis and visualization generation
- **README.md**: This file

## Citation

If you use this pipeline in your research, please cite:
```
BINFX410 Sequence QC Pipeline (2024)
https://github.com/your-repo/sequence-qc-pipeline
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the course instructor.

## Acknowledgments

This pipeline was developed as part of the BINFX410 course exercise on NextFlow workflows and sequence quality control.
