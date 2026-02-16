# Bioinformatics Docker Exercise

A Docker container with common bioinformatics tools for sequence analysis.

## Prerequisites

Install Docker for your platform:

- **macOS:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
- **Linux:** `sudo apt-get install docker.io` (Ubuntu/Debian) or follow the [official docs](https://docs.docker.com/engine/install/)

## Building the Image

```bash
docker build -t binfx-tools .
```

## Running the Container

Start an interactive shell with your local `data/` directory mounted:

```bash
docker run -it -v $(pwd)/data:/data binfx-tools
```

This drops you into a bash shell inside the container at `/data`, where any files in your host's `data/` directory are accessible.

To run the container in the background (visible in Docker Desktop):

```bash
docker run -itd --name binfx-tools -v $(pwd)/data:/data binfx-tools
```

Then open a shell into it:

```bash
docker exec -it binfx-tools bash
```

Stop the container when done:

```bash
docker stop binfx-tools
```

## Verifying Tools

Confirm each tool is installed:

```bash
samtools --version
bwa
fastqc --version
bedtools --version
blastn -version
```

Or verify from outside the container:

```bash
docker run --rm binfx-tools samtools --version
docker run --rm binfx-tools bwa
docker run --rm binfx-tools fastqc --version
docker run --rm binfx-tools bedtools --version
docker run --rm binfx-tools blastn -version
```

## Sample Data

The `data/` directory contains small example files for trying out each tool:

| File | Description |
|---|---|
| `reads.fastq` | 6 synthetic 51bp reads with varying quality scores |
| `input.fasta` | 2 query sequences for BLAST searches |
| `regions.bed` | 5 genomic regions across chr1–chr3 |
| `features.bed` | 5 features that partially overlap regions.bed |
| `sample.sam` | 5 aligned reads in SAM format |
| `reference.fa` | Small 2-chromosome reference genome |

## Example Usage

All examples below are run **inside the container** from the `/data` directory.

### samtools — Convert SAM to BAM and view it

```bash
samtools view -bS sample.sam > sample.bam
samtools index sample.bam
samtools view -h sample.bam | head
```

### FastQC — Quality control on FASTQ reads

```bash
fastqc reads.fastq
```

This produces `reads_fastqc.html` and `reads_fastqc.zip` in the current directory.

### BWA — Align reads to a reference

```bash
bwa index reference.fa
bwa mem reference.fa reads.fastq > aligned.sam
```

### bedtools — Intersect two BED files

```bash
bedtools intersect -a regions.bed -b features.bed
```

### BLAST — Search query sequences against a local database

```bash
makeblastdb -in reference.fa -dbtype nucl -out refdb
blastn -query input.fasta -db refdb -out results.txt -outfmt 6
cat results.txt
```
