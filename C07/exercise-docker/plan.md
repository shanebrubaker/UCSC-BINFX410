# Plan: Bioinformatics Docker Exercise

## Context
This is a BINFX410 student exercise in an empty directory (`exercise-docker/`). The goal is to teach students how to build and run a Docker container with common bioinformatics tools. The deliverables are a Dockerfile and a README with installation/usage instructions.

## Files to Create

### 1. `Dockerfile`
- **Base image:** `ubuntu:22.04`
- **Tools to install via `apt-get`:**
  - `samtools` — manipulate SAM/BAM alignments
  - `bwa` — Burrows-Wheeler Aligner for short reads
  - `fastqc` — quality control for sequencing data
  - `bedtools` — genome arithmetic (intersect, merge, etc.)
  - `ncbi-blast+` — sequence similarity search
- **Best practices:**
  - Single `RUN` layer to minimize image size (chain `apt-get update && apt-get install && rm -rf /var/lib/apt/lists/*`)
  - Set `WORKDIR /data` as a mount point for user data
  - Add `LABEL` metadata (maintainer, description)
  - Use `CMD ["/bin/bash"]` for an interactive default

### 2. `README.md`
Structured guide covering:
1. **Prerequisites** — brief Docker install instructions per platform (macOS via Docker Desktop, Windows via Docker Desktop/WSL2, Linux via apt)
2. **Building the image** — `docker build -t binfx-tools .`
3. **Running the container** — interactive shell with volume mount: `docker run -it -v $(pwd)/data:/data binfx-tools`
4. **Verifying tools** — quick commands to confirm each tool works (`samtools --version`, `bwa`, `fastqc --version`, `bedtools --version`, `blastn -version`)
5. **Example usage** — a short samtools example (e.g., `samtools view -h sample.bam | head`)

## Verification
After implementation, confirm the exercise works end-to-end:
1. `docker build -t binfx-tools .` — image builds without errors
2. `docker run --rm binfx-tools samtools --version` — prints samtools version
3. `docker run --rm binfx-tools bwa` — prints bwa usage
4. `docker run --rm binfx-tools fastqc --version` — prints FastQC version
5. `docker run --rm binfx-tools bedtools --version` — prints bedtools version
6. `docker run --rm binfx-tools blastn -version` — prints BLAST version
