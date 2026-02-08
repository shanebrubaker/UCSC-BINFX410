#!/usr/bin/env python3
"""Build the Big Data Bioinformatics Exercises notebook."""

import json

def md(source, **kwargs):
    """Create a markdown cell."""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source
    }
    # Fix: ensure each line (except last) ends with \n
    lines = cell["source"]
    cell["source"] = [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)]
    cell["metadata"].update(kwargs.get("metadata", {}))
    return cell

def code(source, **kwargs):
    """Create a code cell."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    }
    lines = cell["source"]
    cell["source"] = [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)]
    cell["metadata"].update(kwargs.get("metadata", {}))
    return cell

def solution_code(source):
    """Create a hidden solution code cell."""
    return code(source, metadata={
        "tags": ["solution"],
        "jupyter": {"source_hidden": True}
    })

def solution_md(source):
    """Create a hidden solution markdown cell."""
    return md(source, metadata={
        "tags": ["solution"],
        "jupyter": {"source_hidden": True}
    })

cells = []

# ============================================================
# TITLE AND OVERVIEW
# ============================================================
cells.append(md("""# Big Data Bioinformatics Exercises
## Processing Large-Scale Sequencing Data with Python

**Duration:** 4-5 hour workshop (can be split across 2-3 sessions)

**Session Plan:**
- **Session 1 (~2 hours):** Parts 0-5 — Foundations
  - Parts 0-1: Setup + Memory Efficiency (30 min)
  - Parts 2-3: Chunking + Parallel Processing (40 min)
  - Parts 4-5: Indexing + Streaming Pipelines (40 min)
- **Session 2 (~2 hours):** Parts 6-9 — Advanced Patterns
  - Parts 6-7: MapReduce + Tool Integration (50 min)
  - Parts 8-9: Profiling + Paired-End (40 min)
- **Session 3 / Homework:** Part 10 — Final Project (60+ min)

**What you'll learn:** Memory-efficient processing, chunking, parallel execution, indexing, streaming pipelines, MapReduce, tool integration, profiling, and paired-end data handling — all applied to real bioinformatics data formats."""))

# ============================================================
# PART 0: SETUP AND DATA GENERATION
# ============================================================
cells.append(md("""---
## Part 0: Setup and Data Generation

Before we can process big data, we need data to process. In this section we'll:
1. Set up our environment and global constants
2. Learn the FASTQ format (the standard for sequencing data)
3. Generate realistic test datasets

### The FASTQ Format

Each sequencing read is stored as 4 lines:
```
@READ_ID          <- Header (starts with @)
ACGTACGTACGT      <- DNA sequence
+                 <- Separator (starts with +)
IIIIIIIIIII       <- Quality scores (ASCII-encoded Phred scores)
```

Quality scores use ASCII encoding: each character maps to a Phred quality score.
- `!` (ASCII 33) = Phred 0 (worst)
- `I` (ASCII 73) = Phred 40 (best for Illumina)
- Higher score = higher confidence in the base call"""))

cells.append(md("### 0.1 Global Constants and Imports"))

cells.append(code("""import os
import sys
import time
import random
import hashlib
import shutil
import subprocess
from io import StringIO
from typing import Generator, Tuple, List, Dict, Any, Optional
from collections import defaultdict
import multiprocessing
# Use 'fork' context so worker functions defined in the notebook can be pickled
# (default 'spawn' on macOS cannot pickle interactively-defined functions)
import platform
if platform.system() == 'Darwin':
    _mp_context = multiprocessing.get_context('fork')
else:
    _mp_context = multiprocessing.get_context()

# Reproducibility
RANDOM_SEED = 42

# Data directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Data directory: {os.path.abspath(DATA_DIR)}")
print(f"CPU cores available: {multiprocessing.cpu_count()}")"""))

cells.append(md("### 0.2 Solution Display Setup"))

cells.append(code("""from IPython.display import HTML, display

display(HTML('''<style>
/* Solution cells are hidden by default via Jupyter's source_hidden metadata */
/* If your environment doesn't support source_hidden, solutions are still
   clearly marked with banners */
</style>
<p><b>Setup complete.</b> Solution cells are hidden by default.
Click the "..." or expand arrow next to collapsed cells to reveal solutions.</p>
'''))"""))

cells.append(md("### 0.3 Utility Functions"))

cells.append(code("""def check_tool(tool_name: str) -> bool:
    \"\"\"Check if a command-line tool is available on PATH.\"\"\"
    return shutil.which(tool_name) is not None

def file_md5(filepath: str) -> str:
    \"\"\"Compute MD5 hash of a file for verification.\"\"\"
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def file_line_count(filepath: str) -> int:
    \"\"\"Count lines in a file efficiently.\"\"\"
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count

# Check for optional bioinformatics tools
for tool in ['seqkit', 'fastp']:
    status = "FOUND" if check_tool(tool) else "not found (optional)"
    print(f"  {tool}: {status}")"""))

cells.append(md("### 0.4 FASTQ Data Generation"))

cells.append(code("""def create_test_fastq(filename: str,
                     num_reads: int = 10000,
                     read_length: int = 150,
                     quality_profile: str = 'high',
                     n_rate: float = 0.01,
                     seed: int = RANDOM_SEED) -> None:
    \"\"\"Generate a realistic test FASTQ file.

    Args:
        filename: Output file path
        num_reads: Number of reads to generate
        read_length: Length of each read in bases
        quality_profile: 'high', 'medium', or 'low' — controls overall quality
        n_rate: Probability of inserting N at low-quality positions
        seed: Random seed for reproducibility
    \"\"\"
    rng = random.Random(seed)
    bases = 'ACGT'

    with open(filename, 'w') as f:
        for i in range(num_reads):
            # Generate header
            header = f"@READ_{i+1:07d} length={read_length}"

            # Generate sequence and quality together (position-dependent)
            sequence = []
            quality = []

            for pos in range(read_length):
                # Illumina-like decay: high quality plateau then drop at end
                base_qual = max(2, int(40 - (pos / read_length) ** 2 * 30))

                # Add noise
                base_qual = max(2, base_qual + rng.randint(-5, 3))

                # Apply quality profile modifier
                if quality_profile == 'low':
                    base_qual = max(2, base_qual - 15)
                elif quality_profile == 'medium':
                    base_qual = max(2, base_qual - 7)

                quality.append(chr(base_qual + 33))

                # Insert N at low-quality positions
                if base_qual < 10 and rng.random() < n_rate:
                    sequence.append('N')
                else:
                    sequence.append(rng.choice(bases))

            f.write(f"{header}\\n")
            f.write(''.join(sequence) + '\\n')
            f.write('+\\n')
            f.write(''.join(quality) + '\\n')

    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Created {filename}: {num_reads:,} reads, {size_mb:.1f} MB")"""))

cells.append(md("### 0.5 Paired-End Data Generation"))

cells.append(code("""def create_paired_end_fastq(prefix: str,
                           num_reads: int = 10000,
                           read_length: int = 150,
                           insert_size: int = 300,
                           seed: int = RANDOM_SEED) -> Tuple[str, str]:
    \"\"\"Generate paired-end FASTQ files with realistic insert sizes.

    Simulates Illumina paired-end sequencing:
    - Generate a fragment of length ~insert_size
    - R1 = first read_length bases (forward)
    - R2 = last read_length bases (reverse complement)

    Args:
        prefix: Output file prefix (creates prefix_R1.fastq, prefix_R2.fastq)
        num_reads: Number of read pairs
        read_length: Length of each read
        insert_size: Mean fragment size (actual size varies +/- 30)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (R1 filename, R2 filename)
    \"\"\"
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    rng = random.Random(seed)
    bases = 'ACGT'

    r1_file = f"{prefix}_R1.fastq"
    r2_file = f"{prefix}_R2.fastq"

    with open(r1_file, 'w') as f1, open(r2_file, 'w') as f2:
        for i in range(num_reads):
            # Vary insert size around the mean
            actual_insert = max(read_length + 10,
                               insert_size + rng.randint(-30, 30))

            # Generate full fragment
            fragment = [rng.choice(bases) for _ in range(actual_insert)]

            # R1: forward read from start
            r1_seq = fragment[:read_length]

            # R2: reverse complement from end
            r2_seq = [complement[b] for b in reversed(fragment[-read_length:])]

            # Generate quality scores (position-dependent, same model)
            r1_qual = []
            r2_qual = []
            for pos in range(read_length):
                base_qual = max(2, int(40 - (pos / read_length) ** 2 * 30))
                q1 = max(2, base_qual + rng.randint(-5, 3))
                q2 = max(2, base_qual + rng.randint(-5, 3))
                r1_qual.append(chr(q1 + 33))
                r2_qual.append(chr(q2 + 33))

                # Insert Ns at low quality positions
                if q1 < 10 and rng.random() < 0.01:
                    r1_seq[pos] = 'N'
                if q2 < 10 and rng.random() < 0.01:
                    r2_seq[pos] = 'N'

            # Write R1
            f1.write(f"@READ_{i+1:07d}/1 length={read_length}\\n")
            f1.write(''.join(r1_seq) + '\\n')
            f1.write('+\\n')
            f1.write(''.join(r1_qual) + '\\n')

            # Write R2
            f2.write(f"@READ_{i+1:07d}/2 length={read_length}\\n")
            f2.write(''.join(r2_seq) + '\\n')
            f2.write('+\\n')
            f2.write(''.join(r2_qual) + '\\n')

    for fn in (r1_file, r2_file):
        size_mb = os.path.getsize(fn) / (1024 * 1024)
        print(f"Created {fn}: {num_reads:,} reads, {size_mb:.1f} MB")

    return r1_file, r2_file"""))

cells.append(md("### 0.6 Generate All Test Data"))

cells.append(code("""# Generate test datasets
print("Generating test data...\\n")

# Standard quality dataset
create_test_fastq(os.path.join(DATA_DIR, 'sample.fastq'),
                  num_reads=10000, quality_profile='high', seed=RANDOM_SEED)

# Larger dataset for benchmarking
create_test_fastq(os.path.join(DATA_DIR, 'large_sample.fastq'),
                  num_reads=100000, quality_profile='high', seed=RANDOM_SEED + 1)

# Mixed quality dataset
create_test_fastq(os.path.join(DATA_DIR, 'mixed_quality.fastq'),
                  num_reads=10000, quality_profile='medium', n_rate=0.02, seed=RANDOM_SEED + 2)

# Paired-end dataset
create_paired_end_fastq(os.path.join(DATA_DIR, 'sample'),
                        num_reads=10000, seed=RANDOM_SEED + 3)

print("\\nDone!")"""))

cells.append(md("### 0.7 Verify Your Setup"))

cells.append(code("""# Verify all files were created correctly
print("Verification:")
print("-" * 50)

expected_files = [
    'sample.fastq',
    'large_sample.fastq',
    'mixed_quality.fastq',
    'sample_R1.fastq',
    'sample_R2.fastq',
]

all_ok = True
for fname in expected_files:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        lines = file_line_count(fpath)
        reads = lines // 4
        md5 = file_md5(fpath)
        print(f"  {fname:25s} {size/1024:8.1f} KB  {reads:>7,} reads  MD5: {md5[:8]}...")
    else:
        print(f"  {fname:25s} MISSING!")
        all_ok = False

if all_ok:
    print("\\nAll files created successfully!")
else:
    print("\\nWARNING: Some files are missing. Re-run the generation cells above.")"""))

# ============================================================
# PART 1: MEMORY EFFICIENCY
# ============================================================
cells.append(md("""---
## Part 1: Memory Efficiency with Generators

### Why This Matters

A typical sequencing run produces millions of reads. Loading them all into memory at once would require gigabytes of RAM. **Generators** let us process one read at a time, using constant memory regardless of file size.

### Key Concept: Lists vs. Generators

| | List | Generator |
|---|---|---|
| Memory | Stores ALL items | Stores ONE item at a time |
| Access | Random (any index) | Sequential (forward only) |
| Reuse | Multiple passes | Single pass |
| Speed | Fast access, slow creation | Lazy evaluation |"""))

cells.append(md("### 1.1 Reading FASTQ: List Approach"))

cells.append(code("""def read_fastq_list(filename: str) -> list:
    \"\"\"Read all FASTQ records into a list.

    Returns a list of tuples: (header, sequence, quality)
    WARNING: Loads entire file into memory!
    \"\"\"
    records = []
    with open(filename, 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            sequence = f.readline().strip()
            f.readline()  # skip '+' line
            quality = f.readline().strip()
            records.append((header, sequence, quality))
    return records

# Measure memory usage
import sys

sample_file = os.path.join(DATA_DIR, 'sample.fastq')
records = read_fastq_list(sample_file)
list_size = sys.getsizeof(records)
# Note: getsizeof only measures the list container, not the strings inside
total_size = list_size + sum(
    sys.getsizeof(r) + sys.getsizeof(r[0]) + sys.getsizeof(r[1]) + sys.getsizeof(r[2])
    for r in records
)
print(f"Number of records: {len(records):,}")
print(f"List container size: {list_size:,} bytes")
print(f"Estimated total memory: {total_size / 1024 / 1024:.1f} MB")
print(f"First record header: {records[0][0]}")"""))

cells.append(md("### 1.2 Reading FASTQ: Generator Approach"))

cells.append(code("""def read_fastq_generator(filename: str) -> Generator[Tuple[str, str, str], None, None]:
    \"\"\"Read FASTQ records one at a time using a generator.

    Yields tuples of (header, sequence, quality).
    Memory usage is constant regardless of file size.
    \"\"\"
    with open(filename, 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            sequence = f.readline().strip()
            f.readline()  # skip '+' line
            quality = f.readline().strip()
            yield (header, sequence, quality)

# Measure generator memory
gen = read_fastq_generator(sample_file)
gen_size = sys.getsizeof(gen)
print(f"Generator object size: {gen_size} bytes")
print(f"Memory ratio (list/generator): {total_size / gen_size:.0f}x")

# We can still iterate over it
count = 0
for record in gen:
    count += 1
print(f"Records yielded: {count:,}")"""))

cells.append(md("""### Exercise 1.1: GC Content Calculator

GC content is the percentage of bases that are G or C. It's a fundamental quality metric in genomics.

**Task:** Implement `gc_content_generator()` that calculates GC content for each read using a generator, then compute the average."""))

cells.append(code("""def gc_content_generator(filename: str) -> Generator[float, None, None]:
    \"\"\"Yield GC content (0.0-1.0) for each read in the FASTQ file.

    GC content = (count of G + count of C) / total bases

    TODO: Implement this function
    - Use read_fastq_generator() to iterate over records
    - For each record, calculate GC content from the sequence
    - Yield the GC content as a float
    \"\"\"
    # TODO: Your implementation here
    pass

# Test your implementation:
# gc_values = list(gc_content_generator(sample_file))
# avg_gc = sum(gc_values) / len(gc_values)
# print(f"Average GC content: {avg_gc:.4f}")
# You should see approximately 0.50 (since our data uses random bases)"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 1.1 (click to expand)
# =============================================

def gc_content_generator(filename: str) -> Generator[float, None, None]:
    \"\"\"Yield GC content (0.0-1.0) for each read in the FASTQ file.\"\"\"
    for header, sequence, quality in read_fastq_generator(filename):
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        yield gc_count / len(sequence) if len(sequence) > 0 else 0.0

# Test
gc_values = list(gc_content_generator(sample_file))
avg_gc = sum(gc_values) / len(gc_values)
print(f"Average GC content: {avg_gc:.4f}")
print(f"Min GC: {min(gc_values):.4f}, Max GC: {max(gc_values):.4f}")
print(f"Number of reads analyzed: {len(gc_values):,}")"""))

cells.append(md("""### Exercise 1.2: Memory Comparison

**Task:** Write a function that measures the peak memory used by the list approach vs. the generator approach for computing average quality scores."""))

cells.append(code("""def avg_quality_list(filename: str) -> float:
    \"\"\"Calculate average quality using the list approach.

    TODO: Implement this function
    - Load all records into a list using read_fastq_list()
    - Calculate the average Phred quality across ALL bases in ALL reads
    - Phred score = ord(char) - 33 for each quality character
    \"\"\"
    # TODO: Your implementation here
    pass

def avg_quality_generator(filename: str) -> float:
    \"\"\"Calculate average quality using the generator approach.

    TODO: Implement this function
    - Use read_fastq_generator() to iterate
    - Keep a running sum and count
    - Return the average Phred quality
    \"\"\"
    # TODO: Your implementation here
    pass

# Test your implementations:
# q_list = avg_quality_list(sample_file)
# q_gen = avg_quality_generator(sample_file)
# print(f"List approach avg quality:      {q_list:.2f}")
# print(f"Generator approach avg quality:  {q_gen:.2f}")
# Both should return the same value (around 28-32 for high-quality data)"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 1.2 (click to expand)
# =============================================

def avg_quality_list(filename: str) -> float:
    \"\"\"Calculate average quality using the list approach.\"\"\"
    records = read_fastq_list(filename)
    total_qual = 0
    total_bases = 0
    for header, sequence, quality in records:
        for char in quality:
            total_qual += ord(char) - 33
            total_bases += 1
    return total_qual / total_bases if total_bases > 0 else 0.0

def avg_quality_generator(filename: str) -> float:
    \"\"\"Calculate average quality using the generator approach.\"\"\"
    total_qual = 0
    total_bases = 0
    for header, sequence, quality in read_fastq_generator(filename):
        for char in quality:
            total_qual += ord(char) - 33
            total_bases += 1
    return total_qual / total_bases if total_bases > 0 else 0.0

# Test
q_list = avg_quality_list(sample_file)
q_gen = avg_quality_generator(sample_file)
print(f"List approach avg quality:      {q_list:.2f}")
print(f"Generator approach avg quality:  {q_gen:.2f}")
print(f"Results match: {abs(q_list - q_gen) < 0.001}")"""))

cells.append(md("""**Discussion**: At what file size does the generator approach become necessary? Consider that a typical laptop has 8-16 GB of RAM, and a single sequencing run can produce 100+ GB of FASTQ data."""))

# ============================================================
# PART 2: CHUNKED PROCESSING
# ============================================================
cells.append(md("""---
## Part 2: Chunked Processing

### Why Chunks?

While generators process one record at a time, sometimes we want to process records in **batches** (chunks). This gives us:
- Better I/O efficiency (fewer system calls)
- Natural units for parallel processing
- Progress reporting at chunk boundaries
- Ability to do batch operations (e.g., batch database inserts)"""))

cells.append(md("### 2.1 The Chunking Pattern"))

cells.append(code("""def read_fastq_chunks(filename: str,
                     chunk_size: int = 1000) -> Generator[list, None, None]:
    \"\"\"Read FASTQ records in chunks of fixed size.

    Yields lists of (header, sequence, quality) tuples.
    Each yielded list has at most chunk_size records.
    \"\"\"
    chunk = []
    for record in read_fastq_generator(filename):
        chunk.append(record)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:  # Don't forget the last partial chunk!
        yield chunk

# Demonstrate chunked reading
print("Reading in chunks of 2500:")
for i, chunk in enumerate(read_fastq_chunks(sample_file, chunk_size=2500)):
    print(f"  Chunk {i}: {len(chunk)} records")"""))

cells.append(md("""### Exercise 2.1: Quality Filter with Progress

**Task:** Implement a chunked quality filter that reports progress as it processes each chunk."""))

cells.append(code("""def filter_by_quality_chunked(input_file: str,
                             output_file: str,
                             min_avg_quality: float = 20.0,
                             chunk_size: int = 1000) -> Dict[str, int]:
    \"\"\"Filter reads by average quality score, processing in chunks.

    TODO: Implement this function
    - Process the input file in chunks using read_fastq_chunks()
    - For each read, calculate its average Phred quality
    - Write reads that meet the minimum quality threshold to output_file
    - Print progress after each chunk
    - Return a dict with 'total', 'passed', and 'failed' counts
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# stats = filter_by_quality_chunked(
#     sample_file,
#     os.path.join(DATA_DIR, 'filtered.fastq'),
#     min_avg_quality=25.0
# )
# print(f"\\nResults: {stats}")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 2.1 (click to expand)
# =============================================

def filter_by_quality_chunked(input_file: str,
                             output_file: str,
                             min_avg_quality: float = 20.0,
                             chunk_size: int = 1000) -> Dict[str, int]:
    \"\"\"Filter reads by average quality score, processing in chunks.\"\"\"
    stats = {'total': 0, 'passed': 0, 'failed': 0}

    with open(output_file, 'w') as out:
        for i, chunk in enumerate(read_fastq_chunks(input_file, chunk_size)):
            chunk_passed = 0
            for header, sequence, quality in chunk:
                stats['total'] += 1
                avg_qual = sum(ord(c) - 33 for c in quality) / len(quality)

                if avg_qual >= min_avg_quality:
                    stats['passed'] += 1
                    chunk_passed += 1
                    out.write(f"{header}\\n{sequence}\\n+\\n{quality}\\n")
                else:
                    stats['failed'] += 1

            print(f"  Chunk {i}: {chunk_passed}/{len(chunk)} passed "
                  f"(running total: {stats['passed']:,}/{stats['total']:,})")

    return stats

# Test
stats = filter_by_quality_chunked(
    sample_file,
    os.path.join(DATA_DIR, 'filtered.fastq'),
    min_avg_quality=25.0
)
print(f"\\nFinal: {stats['passed']:,} of {stats['total']:,} reads passed "
      f"({100*stats['passed']/stats['total']:.1f}%)")"""))

cells.append(md("""### Exercise 2.2: Chunk Size Experiment

**Task:** Measure processing time for different chunk sizes to find the sweet spot."""))

cells.append(code("""def benchmark_chunk_sizes(filename: str,
                         chunk_sizes: list) -> Dict[int, float]:
    \"\"\"Benchmark processing time for different chunk sizes.

    TODO: Implement this function
    - For each chunk_size in chunk_sizes, time how long it takes to:
      read all chunks and compute average quality per chunk
    - Return a dict mapping chunk_size -> elapsed_time_seconds
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# sizes = [100, 500, 1000, 2500, 5000]
# results = benchmark_chunk_sizes(sample_file, sizes)
# for size, elapsed in results.items():
#     print(f"  Chunk size {size:5d}: {elapsed:.3f}s")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 2.2 (click to expand)
# =============================================

def benchmark_chunk_sizes(filename: str,
                         chunk_sizes: list) -> Dict[int, float]:
    \"\"\"Benchmark processing time for different chunk sizes.\"\"\"
    results = {}
    for size in chunk_sizes:
        start = time.perf_counter()
        for chunk in read_fastq_chunks(filename, chunk_size=size):
            # Simulate work: compute average quality per chunk
            for header, sequence, quality in chunk:
                _ = sum(ord(c) - 33 for c in quality) / len(quality)
        elapsed = time.perf_counter() - start
        results[size] = elapsed
    return results

sizes = [100, 500, 1000, 2500, 5000]
results = benchmark_chunk_sizes(sample_file, sizes)
print("Chunk size benchmarks:")
for size, elapsed in results.items():
    print(f"  Chunk size {size:5d}: {elapsed:.3f}s")
print("\\nNote: Very small chunks add overhead from list creation;")
print("very large chunks approach list-like memory usage.")"""))

# ============================================================
# PART 3: PARALLEL PROCESSING
# ============================================================
cells.append(md("""---
## Part 3: Parallel Processing

### Why Parallel?

Modern CPUs have multiple cores. By splitting work across cores, we can process data faster. The key challenge: **we need to divide the work into independent units** that don't share state.

### The Pattern
1. **Split** data into chunks
2. **Send** each chunk to a separate worker process
3. **Collect** and combine results"""))

cells.append(md("### 3.1 Worker Functions"))

cells.append(code("""def count_gc_in_chunk(chunk: list) -> Dict[str, Any]:
    \"\"\"Worker function: compute GC statistics for a chunk of reads.

    This function runs in a separate process, so it must be
    self-contained (no references to shared state).
    \"\"\"
    gc_sum = 0.0
    total_bases = 0
    n_count = 0

    for header, sequence, quality in chunk:
        seq_upper = sequence.upper()
        gc_sum += seq_upper.count('G') + seq_upper.count('C')
        n_count += seq_upper.count('N')
        total_bases += len(sequence)

    return {
        'reads': len(chunk),
        'gc_sum': gc_sum,
        'total_bases': total_bases,
        'n_count': n_count
    }"""))

cells.append(md("### 3.2 Sequential vs. Parallel Comparison"))

cells.append(code("""def process_sequential(filename: str, chunk_size: int = 2500) -> dict:
    \"\"\"Process file sequentially.\"\"\"
    combined = {'reads': 0, 'gc_sum': 0, 'total_bases': 0, 'n_count': 0}
    for chunk in read_fastq_chunks(filename, chunk_size):
        result = count_gc_in_chunk(chunk)
        for key in combined:
            combined[key] += result[key]
    return combined

def process_parallel(filename: str, chunk_size: int = 2500,
                     num_workers: int = None) -> dict:
    \"\"\"Process file in parallel using multiprocessing.Pool.\"\"\"
    if num_workers is None:
        num_workers = min(4, multiprocessing.cpu_count())

    chunks = list(read_fastq_chunks(filename, chunk_size))

    combined = {'reads': 0, 'gc_sum': 0, 'total_bases': 0, 'n_count': 0}

    with _mp_context.Pool(processes=num_workers) as pool:
        results = pool.map(count_gc_in_chunk, chunks)
        for result in results:
            for key in combined:
                combined[key] += result[key]

    return combined

# Benchmark on the larger file
large_file = os.path.join(DATA_DIR, 'large_sample.fastq')

start = time.perf_counter()
seq_result = process_sequential(large_file)
seq_time = time.perf_counter() - start

start = time.perf_counter()
par_result = process_parallel(large_file)
par_time = time.perf_counter() - start

gc_pct = seq_result['gc_sum'] / seq_result['total_bases'] * 100
print(f"GC content: {gc_pct:.2f}%")
print(f"N bases: {seq_result['n_count']:,}")
print(f"Sequential: {seq_time:.3f}s")
print(f"Parallel:   {par_time:.3f}s")
print(f"Speedup:    {seq_time/par_time:.2f}x")"""))

cells.append(md("""### Exercise 3.1: Parallel Quality Statistics

**Task:** Implement a parallel version of quality statistics calculation."""))

cells.append(code("""def quality_stats_chunk(chunk: list) -> Dict[str, Any]:
    \"\"\"Worker function: compute quality statistics for a chunk.

    TODO: Implement this function
    - Calculate: total quality sum, total bases, min quality, max quality
    - Return a dict with these values
    \"\"\"
    # TODO: Your implementation here
    pass

def parallel_quality_stats(filename: str, num_workers: int = 4) -> dict:
    \"\"\"Compute quality stats in parallel.

    TODO: Implement this function
    - Split file into chunks
    - Process chunks in parallel using multiprocessing.Pool
    - Combine results (sum totals, take min of mins, max of maxes)
    - Return dict with 'avg_quality', 'min_quality', 'max_quality'
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# stats = parallel_quality_stats(large_file)
# print(f"Average quality: {stats['avg_quality']:.2f}")
# print(f"Min quality: {stats['min_quality']}")
# print(f"Max quality: {stats['max_quality']}")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 3.1 (click to expand)
# =============================================

def quality_stats_chunk(chunk: list) -> Dict[str, Any]:
    \"\"\"Worker function: compute quality statistics for a chunk.\"\"\"
    total_qual = 0
    total_bases = 0
    min_qual = float('inf')
    max_qual = float('-inf')

    for header, sequence, quality in chunk:
        for char in quality:
            q = ord(char) - 33
            total_qual += q
            total_bases += 1
            min_qual = min(min_qual, q)
            max_qual = max(max_qual, q)

    return {
        'total_qual': total_qual,
        'total_bases': total_bases,
        'min_qual': min_qual,
        'max_qual': max_qual
    }

def parallel_quality_stats(filename: str, num_workers: int = 4) -> dict:
    \"\"\"Compute quality stats in parallel.\"\"\"
    chunks = list(read_fastq_chunks(filename, chunk_size=2500))

    combined_qual = 0
    combined_bases = 0
    combined_min = float('inf')
    combined_max = float('-inf')

    with _mp_context.Pool(processes=num_workers) as pool:
        for result in pool.map(quality_stats_chunk, chunks):
            combined_qual += result['total_qual']
            combined_bases += result['total_bases']
            combined_min = min(combined_min, result['min_qual'])
            combined_max = max(combined_max, result['max_qual'])

    return {
        'avg_quality': combined_qual / combined_bases if combined_bases > 0 else 0,
        'min_quality': combined_min,
        'max_quality': combined_max
    }

stats = parallel_quality_stats(large_file)
print(f"Average quality: {stats['avg_quality']:.2f}")
print(f"Min quality: {stats['min_quality']}")
print(f"Max quality: {stats['max_quality']}")"""))

# ============================================================
# PART 4: INDEXING AND RANDOM ACCESS
# ============================================================
cells.append(md("""---
## Part 4: Indexing and Random Access

### The Problem

FASTQ files are sequential: to find read #50,000, you must scan past the first 49,999 reads. With an **index**, we can jump directly to any read in O(1) time.

### The Approach

Build an index that maps read IDs (or positions) to byte offsets in the file. Then use `file.seek()` to jump directly to any read."""))

cells.append(md("### 4.1 Building a FASTQ Index"))

cells.append(code("""def build_fastq_index(filename: str) -> Dict[str, int]:
    \"\"\"Build an index mapping read IDs to byte offsets.

    Returns a dict: {read_id: byte_offset}
    The byte offset points to the start of the header line.
    \"\"\"
    index = {}
    with open(filename, 'rb') as f:
        while True:
            offset = f.tell()
            header = f.readline()
            if not header:
                break
            # Extract read ID (everything after @ up to first space)
            read_id = header.decode().strip().split()[0][1:]  # remove @
            index[read_id] = offset
            # Skip sequence, +, quality
            f.readline()
            f.readline()
            f.readline()
    return index

# Build index
start = time.perf_counter()
index = build_fastq_index(sample_file)
index_time = time.perf_counter() - start
print(f"Indexed {len(index):,} reads in {index_time:.3f}s")
print(f"Index size: {sys.getsizeof(index) / 1024:.1f} KB")
print(f"Sample entries: {list(index.items())[:3]}")"""))

cells.append(md("### 4.2 Random Access Lookup"))

cells.append(code("""def lookup_read(filename: str, index: dict, read_id: str) -> Optional[Tuple[str, str, str]]:
    \"\"\"Look up a specific read by ID using the index.\"\"\"
    if read_id not in index:
        return None

    with open(filename, 'r') as f:
        f.seek(index[read_id])
        header = f.readline().strip()
        sequence = f.readline().strip()
        f.readline()  # skip +
        quality = f.readline().strip()
        return (header, sequence, quality)

# Demonstrate random access
target_id = 'READ_0005000'
start = time.perf_counter()
record = lookup_read(sample_file, index, target_id)
lookup_time = time.perf_counter() - start

if record:
    print(f"Found {target_id} in {lookup_time*1000:.3f}ms")
    print(f"  Sequence: {record[1][:50]}...")
    print(f"  Quality:  {record[2][:50]}...")

# Compare with sequential scan
start = time.perf_counter()
for header, seq, qual in read_fastq_generator(sample_file):
    if 'READ_0005000' in header:
        break
scan_time = time.perf_counter() - start
print(f"\\nSequential scan: {scan_time*1000:.3f}ms")
print(f"Index speedup: {scan_time/lookup_time:.1f}x")"""))

cells.append(md("""### Exercise 4.1: Batch Random Access

**Task:** Implement a function that retrieves multiple reads by ID using the index."""))

cells.append(code("""def batch_lookup(filename: str, index: dict,
                 read_ids: list) -> List[Tuple[str, str, str]]:
    \"\"\"Look up multiple reads by ID using the index.

    TODO: Implement this function
    - For each read_id in read_ids, use the index to find and read the record
    - Optimization: sort lookups by file offset to minimize seeking
    - Return a list of (header, sequence, quality) tuples
    - Skip any read_ids not found in the index
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# target_ids = ['READ_0000001', 'READ_0005000', 'READ_0009999', 'READ_0003333']
# results = batch_lookup(sample_file, index, target_ids)
# for header, seq, qual in results:
#     print(f"  {header.split()[0]}: {seq[:30]}...")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 4.1 (click to expand)
# =============================================

def batch_lookup(filename: str, index: dict,
                 read_ids: list) -> List[Tuple[str, str, str]]:
    \"\"\"Look up multiple reads by ID, optimized with sorted offsets.\"\"\"
    # Filter to valid IDs and sort by file offset for sequential access
    valid_ids = [(rid, index[rid]) for rid in read_ids if rid in index]
    valid_ids.sort(key=lambda x: x[1])

    results = []
    with open(filename, 'r') as f:
        for read_id, offset in valid_ids:
            f.seek(offset)
            header = f.readline().strip()
            sequence = f.readline().strip()
            f.readline()  # skip +
            quality = f.readline().strip()
            results.append((header, sequence, quality))

    return results

target_ids = ['READ_0000001', 'READ_0005000', 'READ_0009999', 'READ_0003333']
start = time.perf_counter()
results = batch_lookup(sample_file, index, target_ids)
elapsed = time.perf_counter() - start
print(f"Retrieved {len(results)} reads in {elapsed*1000:.2f}ms")
for header, seq, qual in results:
    print(f"  {header.split()[0]}: {seq[:40]}...")"""))

# ============================================================
# PART 5: STREAMING PIPELINES
# ============================================================
cells.append(md("""---
## Part 5: Streaming Pipelines

### The Power of Composable Generators

Generators can be **chained together** to form processing pipelines. Each stage transforms the data stream without buffering the entire dataset. This is similar to Unix pipes (`cat file | grep | sort`).

```
[Read FASTQ] -> [Filter Quality] -> [Trim Adapters] -> [Calculate Stats]
     ^               ^                    ^                    ^
  generator       generator           generator            consumer
```"""))

cells.append(md("### 5.1 Pipeline Building Blocks"))

cells.append(code("""def filter_by_length(records, min_length: int = 50):
    \"\"\"Filter reads shorter than min_length.\"\"\"
    for header, sequence, quality in records:
        if len(sequence) >= min_length:
            yield (header, sequence, quality)

def filter_by_quality(records, min_avg_quality: float = 20.0):
    \"\"\"Filter reads below minimum average quality.\"\"\"
    for header, sequence, quality in records:
        avg_q = sum(ord(c) - 33 for c in quality) / len(quality)
        if avg_q >= min_avg_quality:
            yield (header, sequence, quality)

def filter_no_ns(records, max_n_fraction: float = 0.05):
    \"\"\"Filter reads with too many N bases.\"\"\"
    for header, sequence, quality in records:
        n_fraction = sequence.upper().count('N') / len(sequence)
        if n_fraction <= max_n_fraction:
            yield (header, sequence, quality)

def trim_low_quality_ends(records, min_quality: int = 15):
    \"\"\"Trim low-quality bases from the 3' end of reads.\"\"\"
    for header, sequence, quality in records:
        # Trim from the right while quality is below threshold
        end = len(quality)
        while end > 0 and (ord(quality[end-1]) - 33) < min_quality:
            end -= 1
        if end > 0:
            yield (header, sequence[:end], quality[:end])

def compute_stats(records) -> dict:
    \"\"\"Terminal stage: consume the pipeline and compute statistics.\"\"\"
    total = 0
    total_length = 0
    gc_sum = 0
    qual_sum = 0
    qual_bases = 0

    for header, sequence, quality in records:
        total += 1
        total_length += len(sequence)
        seq_upper = sequence.upper()
        gc_sum += seq_upper.count('G') + seq_upper.count('C')
        for c in quality:
            qual_sum += ord(c) - 33
            qual_bases += 1

    return {
        'total_reads': total,
        'avg_length': total_length / total if total > 0 else 0,
        'avg_gc': gc_sum / total_length if total_length > 0 else 0,
        'avg_quality': qual_sum / qual_bases if qual_bases > 0 else 0
    }"""))

cells.append(md("### 5.2 Composing the Pipeline"))

cells.append(code("""# Chain generators into a pipeline — no intermediate files or lists!
mixed_file = os.path.join(DATA_DIR, 'mixed_quality.fastq')

# Build the pipeline (nothing executes yet — lazy evaluation!)
pipeline = read_fastq_generator(mixed_file)
pipeline = filter_by_quality(pipeline, min_avg_quality=20.0)
pipeline = filter_no_ns(pipeline, max_n_fraction=0.05)
pipeline = trim_low_quality_ends(pipeline, min_quality=15)

# Only now does processing begin:
start = time.perf_counter()
stats = compute_stats(pipeline)
elapsed = time.perf_counter() - start

print(f"Pipeline results ({elapsed:.3f}s):")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value:,}")"""))

cells.append(md("""### Exercise 5.1: Custom Pipeline Stage

**Task:** Add a `subsample` stage that randomly keeps a fraction of reads (useful for quick previews of large datasets)."""))

cells.append(code("""def subsample(records, fraction: float = 0.1, seed: int = RANDOM_SEED):
    \"\"\"Randomly subsample a fraction of reads.

    TODO: Implement this function
    - Use random.Random(seed) for reproducibility
    - For each record, keep it with probability 'fraction'
    - Yield kept records
    \"\"\"
    # TODO: Your implementation here
    pass

# Test: subsample 10% of reads and compute stats
# pipeline = read_fastq_generator(sample_file)
# pipeline = subsample(pipeline, fraction=0.1)
# stats = compute_stats(pipeline)
# print(f"Subsampled reads: {stats['total_reads']:,}")
# print(f"Expected ~{10000 * 0.1:.0f} reads")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 5.1 (click to expand)
# =============================================

def subsample(records, fraction: float = 0.1, seed: int = RANDOM_SEED):
    \"\"\"Randomly subsample a fraction of reads.\"\"\"
    rng = random.Random(seed)
    for record in records:
        if rng.random() < fraction:
            yield record

# Test
pipeline = read_fastq_generator(sample_file)
pipeline = subsample(pipeline, fraction=0.1)
stats = compute_stats(pipeline)
print(f"Subsampled reads: {stats['total_reads']:,}")
print(f"Expected ~{10000 * 0.1:.0f} reads")
print(f"Avg quality: {stats['avg_quality']:.2f}")"""))

cells.append(md("""### Exercise 5.2: Pipeline Comparison

**Task:** Build two different pipelines and compare their output — one strict, one lenient."""))

cells.append(code("""def run_pipeline_comparison(filename: str) -> None:
    \"\"\"Compare strict vs lenient filtering pipelines.

    TODO: Implement this function
    Strict pipeline: min_quality=30, max_n=0.01, min_length=100
    Lenient pipeline: min_quality=15, max_n=0.10, min_length=50

    Print stats for: raw input, strict output, lenient output
    \"\"\"
    # TODO: Your implementation here
    pass

# run_pipeline_comparison(mixed_file)"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 5.2 (click to expand)
# =============================================

def run_pipeline_comparison(filename: str) -> None:
    \"\"\"Compare strict vs lenient filtering pipelines.\"\"\"
    # Raw stats
    raw = compute_stats(read_fastq_generator(filename))

    # Strict pipeline
    strict = read_fastq_generator(filename)
    strict = filter_by_quality(strict, min_avg_quality=30.0)
    strict = filter_no_ns(strict, max_n_fraction=0.01)
    strict = filter_by_length(strict, min_length=100)
    strict_stats = compute_stats(strict)

    # Lenient pipeline
    lenient = read_fastq_generator(filename)
    lenient = filter_by_quality(lenient, min_avg_quality=15.0)
    lenient = filter_no_ns(lenient, max_n_fraction=0.10)
    lenient = filter_by_length(lenient, min_length=50)
    lenient_stats = compute_stats(lenient)

    print(f"{'Metric':<20} {'Raw':>10} {'Strict':>10} {'Lenient':>10}")
    print("-" * 52)
    print(f"{'Total reads':<20} {raw['total_reads']:>10,} {strict_stats['total_reads']:>10,} {lenient_stats['total_reads']:>10,}")
    print(f"{'Avg length':<20} {raw['avg_length']:>10.1f} {strict_stats['avg_length']:>10.1f} {lenient_stats['avg_length']:>10.1f}")
    print(f"{'Avg GC':<20} {raw['avg_gc']:>10.4f} {strict_stats['avg_gc']:>10.4f} {lenient_stats['avg_gc']:>10.4f}")
    print(f"{'Avg quality':<20} {raw['avg_quality']:>10.2f} {strict_stats['avg_quality']:>10.2f} {lenient_stats['avg_quality']:>10.2f}")

run_pipeline_comparison(mixed_file)"""))

# ============================================================
# PART 6: MAPREDUCE PATTERN
# ============================================================
cells.append(md("""---
## Part 6: The MapReduce Pattern

### What is MapReduce?

MapReduce is a programming model for processing large datasets in parallel. It was popularized by Google and is the foundation of Hadoop and Spark. The pattern has four distinct phases:

```
                     MAP PHASE                  SHUFFLE PHASE              REDUCE PHASE
                  +-------------+            +-----------------+        +---------------+
  Partition 1 --> | map_func(r) | --\\       | Group by key:   |    /-> | reduce_func() | --> Result A
  Partition 2 --> | map_func(r) | ---+----> | key_A: [v1, v3] | --+--> | reduce_func() | --> Result B
  Partition 3 --> | map_func(r) | --/       | key_B: [v2, v4] |    \\-> | reduce_func() | --> Result C
                  +-------------+            +-----------------+        +---------------+
```

**Key distinction from simple parallel map:** The **shuffle phase** groups intermediate results by key across all partitions. This is what enables aggregation (counting, averaging, etc.) over the full dataset.

### The Phases

1. **SPLIT**: Divide input into partitions
2. **MAP**: Apply `map_func` to each record, emitting `(key, value)` pairs
3. **SHUFFLE**: Group all values by key across all partitions
4. **REDUCE**: Apply `reduce_func` to each `(key, [values])` group"""))

cells.append(md("### 6.1 MapReduce Implementation"))

cells.append(code("""def mapreduce_fastq(filename: str,
                   map_func,
                   reduce_func,
                   num_partitions: int = 4) -> dict:
    \"\"\"
    MapReduce implementation for FASTQ files.

    Args:
        filename: Input FASTQ file
        map_func: Function(record) -> list of (key, value) pairs
        reduce_func: Function(key, values_list) -> result
        num_partitions: Number of partitions to split input into

    Returns:
        Dict mapping each key to its reduced result
    \"\"\"
    # PHASE 1: SPLIT - divide records into partitions
    partitions = [[] for _ in range(num_partitions)]
    for i, record in enumerate(read_fastq_generator(filename)):
        partitions[i % num_partitions].append(record)

    print(f"SPLIT: {sum(len(p) for p in partitions)} records -> {num_partitions} partitions")
    for i, p in enumerate(partitions):
        print(f"  Partition {i}: {len(p)} records")

    # PHASE 2: MAP - apply map_func to each record, collect (key, value) pairs
    all_pairs = []
    for partition_idx, partition in enumerate(partitions):
        partition_pairs = []
        for record in partition:
            pairs = map_func(record)
            partition_pairs.extend(pairs)
        all_pairs.extend(partition_pairs)
        print(f"MAP partition {partition_idx}: emitted {len(partition_pairs)} (key, value) pairs")

    # PHASE 3: SHUFFLE - group values by key
    grouped = defaultdict(list)
    for key, value in all_pairs:
        grouped[key].append(value)

    print(f"SHUFFLE: {len(all_pairs)} pairs -> {len(grouped)} unique keys")

    # PHASE 4: REDUCE - apply reduce_func to each group
    results = {}
    for key, values in grouped.items():
        results[key] = reduce_func(key, values)

    print(f"REDUCE: {len(grouped)} keys -> {len(results)} results")

    return results"""))

cells.append(md("### 6.2 Example: Quality Score Distribution"))

cells.append(code("""# MAP function: classify each read by quality tier
def map_quality_tier(record):
    \"\"\"Map a read to a quality tier.\"\"\"
    header, sequence, quality = record
    avg_q = sum(ord(c) - 33 for c in quality) / len(quality)

    if avg_q >= 30:
        tier = 'high'
    elif avg_q >= 20:
        tier = 'medium'
    else:
        tier = 'low'

    return [(tier, avg_q)]  # emit (tier, actual_quality)

# REDUCE function: compute statistics for each tier
def reduce_quality_stats(key, values):
    \"\"\"Reduce quality values to summary statistics.\"\"\"
    return {
        'count': len(values),
        'avg_quality': sum(values) / len(values),
        'min_quality': min(values),
        'max_quality': max(values)
    }

# Run MapReduce
print("MapReduce: Quality Tier Analysis")
print("=" * 50)
results = mapreduce_fastq(sample_file, map_quality_tier, reduce_quality_stats)
print("\\nResults:")
for tier in ['high', 'medium', 'low']:
    if tier in results:
        r = results[tier]
        print(f"  {tier:>6}: {r['count']:,} reads, "
              f"avg Q={r['avg_quality']:.1f}, "
              f"range [{r['min_quality']:.1f}, {r['max_quality']:.1f}]")"""))

cells.append(md("""### Exercise 6.1: MapReduce - Base Composition

**Task:** Use MapReduce to count the occurrence of each base (A, C, G, T, N) across all reads."""))

cells.append(code("""def map_base_counts(record):
    \"\"\"Map function: emit (base, 1) for each base in the read.

    TODO: Implement this function
    - For each character in the sequence, emit (base, 1)
    - This is the classic word-count pattern
    \"\"\"
    # TODO: Your implementation here
    pass

def reduce_sum(key, values):
    \"\"\"Reduce function: sum all values for a key.

    TODO: Implement this function
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# results = mapreduce_fastq(sample_file, map_base_counts, reduce_sum)
# total = sum(results.values())
# print("\\nBase composition:")
# for base in sorted(results.keys()):
#     pct = results[base] / total * 100
#     print(f"  {base}: {results[base]:>10,} ({pct:.2f}%)")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 6.1 (click to expand)
# =============================================

def map_base_counts(record):
    \"\"\"Map function: emit (base, 1) for each base in the read.\"\"\"
    header, sequence, quality = record
    return [(base, 1) for base in sequence.upper()]

def reduce_sum(key, values):
    \"\"\"Reduce function: sum all values.\"\"\"
    return sum(values)

results = mapreduce_fastq(sample_file, map_base_counts, reduce_sum)
total = sum(results.values())
print("\\nBase composition:")
for base in sorted(results.keys()):
    pct = results[base] / total * 100
    print(f"  {base}: {results[base]:>10,} ({pct:.2f}%)")"""))

cells.append(md("""### Exercise 6.2: MapReduce - Quality by Position

**Task:** Use MapReduce to compute average quality at each read position (reveals the Illumina quality decay curve)."""))

cells.append(code("""def map_position_quality(record):
    \"\"\"Map function: emit (position, quality_score) for each position.

    TODO: Implement this function
    - For each position in the quality string, emit (position, phred_score)
    \"\"\"
    # TODO: Your implementation here
    pass

def reduce_average(key, values):
    \"\"\"Reduce function: compute the average of values.

    TODO: Implement this function
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# results = mapreduce_fastq(sample_file, map_position_quality, reduce_average)
# print("\\nAvg quality by position (first 20, last 20):")
# positions = sorted(results.keys())
# for pos in positions[:20]:
#     print(f"  Position {pos:3d}: {results[pos]:.1f}")
# print("  ...")
# for pos in positions[-20:]:
#     print(f"  Position {pos:3d}: {results[pos]:.1f}")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 6.2 (click to expand)
# =============================================

def map_position_quality(record):
    \"\"\"Map function: emit (position, quality_score) for each position.\"\"\"
    header, sequence, quality = record
    return [(pos, ord(c) - 33) for pos, c in enumerate(quality)]

def reduce_average(key, values):
    \"\"\"Reduce function: compute the average of values.\"\"\"
    return sum(values) / len(values)

results = mapreduce_fastq(sample_file, map_position_quality, reduce_average)

# Display results
positions = sorted(results.keys())
print("Avg quality by position (first 10, last 10):")
for pos in positions[:10]:
    bar = '#' * int(results[pos])
    print(f"  Position {pos:3d}: {results[pos]:5.1f} {bar}")
print("  ...")
for pos in positions[-10:]:
    bar = '#' * int(results[pos])
    print(f"  Position {pos:3d}: {results[pos]:5.1f} {bar}")

print("\\nNotice how quality drops toward the end of the read —")
print("this is the characteristic Illumina quality decay curve!")"""))

# ============================================================
# PART 7: TOOL INTEGRATION
# ============================================================
cells.append(md("""---
## Part 7: Bioinformatics Tool Integration

### Why Use External Tools?

While Python is great for custom analysis, specialized tools like **seqkit** and **fastp** are:
- Written in C/Go for maximum performance (10-100x faster)
- Battle-tested on billions of reads
- Industry standard in bioinformatics pipelines

This section teaches you to call these tools from Python and parse their output.

### Installing seqkit and fastp

These tools are **optional** — the notebook detects whether they are installed and skips gracefully if not. But installing them lets you complete all Part 7 exercises.

**With Conda (easiest):**
```bash
conda install -c bioconda seqkit fastp
```

**With Homebrew (macOS):**
```bash
brew install seqkit
brew install fastp
```

**From pre-built binaries (any platform):**
- **seqkit:** Download from https://bioinf.shenwei.me/seqkit/download/ — extract and place on your `PATH`.
- **fastp:** Download from https://github.com/OpenGene/fastp/releases — extract, `chmod +x fastp`, and move to your `PATH`.

Verify with:
```bash
seqkit version
fastp --version
```"""))

cells.append(md("### 7.1 SeqKit Integration"))

cells.append(code("""class SeqKitHelper:
    \"\"\"Wrapper for seqkit command-line tool.\"\"\"

    @staticmethod
    def _check_available():
        if not shutil.which('seqkit'):
            raise EnvironmentError(
                "seqkit not found. Install with: conda install -c bioconda seqkit\\n"
                "Or skip this section — it's optional."
            )

    @staticmethod
    def get_stats(fastq_file: str):
        \"\"\"Get basic statistics using seqkit stats.\"\"\"
        SeqKitHelper._check_available()
        try:
            result = subprocess.run(
                ['seqkit', 'stats', fastq_file, '-T'],
                capture_output=True, text=True, check=True
            )
            import pandas as pd
            return pd.read_csv(StringIO(result.stdout), sep='\\t')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"seqkit stats failed: {e.stderr}") from e

    @staticmethod
    def grep_by_pattern(fastq_file: str, pattern: str, output_file: str) -> int:
        \"\"\"Extract reads matching a sequence pattern.\"\"\"
        SeqKitHelper._check_available()
        try:
            result = subprocess.run(
                ['seqkit', 'grep', '-s', '-r', '-p', pattern,
                 fastq_file, '-o', output_file],
                capture_output=True, text=True, check=True
            )
            # Count output reads
            if os.path.exists(output_file):
                return file_line_count(output_file) // 4
            return 0
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"seqkit grep failed: {e.stderr}") from e

# Try seqkit if available
if check_tool('seqkit'):
    print("seqkit is available! Running stats...")
    stats_df = SeqKitHelper.get_stats(sample_file)
    print(stats_df.to_string(index=False))
else:
    print("seqkit not found — skipping. Install with:")
    print("  conda install -c bioconda seqkit")"""))

cells.append(md("### 7.2 Fastp Integration"))

cells.append(code("""class FastpHelper:
    \"\"\"Wrapper for fastp quality control tool.\"\"\"

    @staticmethod
    def _check_available():
        if not shutil.which('fastp'):
            raise EnvironmentError(
                "fastp not found. Install with: conda install -c bioconda fastp\\n"
                "Or skip this section — it's optional."
            )

    @staticmethod
    def run_qc(input_file: str,
               output_file: str,
               html_report: str = None,
               json_report: str = None,
               min_quality: int = 20,
               min_length: int = 50) -> Optional[dict]:
        \"\"\"Run fastp quality control.

        Returns parsed JSON report if json_report path is provided.
        \"\"\"
        FastpHelper._check_available()

        cmd = [
            'fastp',
            '-i', input_file,
            '-o', output_file,
            '-q', str(min_quality),
            '-l', str(min_length),
        ]

        if html_report:
            cmd.extend(['-h', html_report])
        else:
            cmd.extend(['-h', '/dev/null'])

        if json_report:
            cmd.extend(['-j', json_report])
        else:
            cmd.extend(['-j', '/dev/null'])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"fastp failed: {e.stderr}") from e

        if json_report and os.path.exists(json_report):
            import json as json_module
            with open(json_report) as f:
                return json_module.load(f)
        return None

# Try fastp if available
if check_tool('fastp'):
    print("fastp is available! Running QC...")
    report = FastpHelper.run_qc(
        sample_file,
        os.path.join(DATA_DIR, 'fastp_filtered.fastq'),
        json_report=os.path.join(DATA_DIR, 'fastp_report.json')
    )
    if report:
        summary = report.get('summary', {})
        before = summary.get('before_filtering', {})
        after = summary.get('after_filtering', {})
        print(f"  Before: {before.get('total_reads', 'N/A'):,} reads")
        print(f"  After:  {after.get('total_reads', 'N/A'):,} reads")
else:
    print("fastp not found — skipping. Install with:")
    print("  conda install -c bioconda fastp")"""))

cells.append(md("""### Exercise 7.1: Python vs. Tool Benchmark

**Task:** Compare the speed of your Python quality filter (from Part 2) with fastp."""))

cells.append(code("""def benchmark_python_vs_tools(filename: str) -> None:
    \"\"\"Compare Python filtering speed with external tools.

    TODO: Implement this function
    - Time your Python filter_by_quality_chunked() on the large file
    - If fastp is available, time fastp on the same file
    - Print comparison results
    - If fastp is not available, print a message and only show Python results
    \"\"\"
    # TODO: Your implementation here
    pass

# benchmark_python_vs_tools(large_file)"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 7.1 (click to expand)
# =============================================

def benchmark_python_vs_tools(filename: str) -> None:
    \"\"\"Compare Python filtering speed with external tools.\"\"\"
    # Python approach
    start = time.perf_counter()
    py_stats = filter_by_quality_chunked(
        filename,
        os.path.join(DATA_DIR, 'benchmark_py_filtered.fastq'),
        min_avg_quality=20.0,
        chunk_size=5000
    )
    py_time = time.perf_counter() - start

    print(f"\\nPython: {py_time:.3f}s ({py_stats['passed']:,} reads passed)")

    # fastp approach (if available)
    if check_tool('fastp'):
        start = time.perf_counter()
        FastpHelper.run_qc(
            filename,
            os.path.join(DATA_DIR, 'benchmark_fastp_filtered.fastq'),
            min_quality=20
        )
        fastp_time = time.perf_counter() - start
        fastp_reads = file_line_count(os.path.join(DATA_DIR, 'benchmark_fastp_filtered.fastq')) // 4
        print(f"fastp:  {fastp_time:.3f}s ({fastp_reads:,} reads passed)")
        print(f"fastp speedup: {py_time/fastp_time:.1f}x")
    else:
        print("fastp not available — install to compare performance")

    # Cleanup
    for f in ['benchmark_py_filtered.fastq', 'benchmark_fastp_filtered.fastq']:
        fpath = os.path.join(DATA_DIR, f)
        if os.path.exists(fpath):
            os.remove(fpath)

benchmark_python_vs_tools(large_file)"""))

# ============================================================
# PART 8: PROFILING AND OPTIMIZATION
# ============================================================
cells.append(md("""---
## Part 8: Profiling and Optimization

### The Golden Rule of Optimization

> "Premature optimization is the root of all evil." — Donald Knuth

**Always profile before optimizing.** Measure where time is actually spent, then optimize the bottleneck. Optimizing the wrong thing wastes effort.

### Profiling Tools
- `time.perf_counter()` — wall-clock timing
- `sys.getsizeof()` — object memory size
- `cProfile` — function-level profiling (built into Python)"""))

cells.append(md("### 8.1 Timing Decorator"))

cells.append(code("""import functools

def timed(func):
    \"\"\"Decorator that prints execution time.\"\"\"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# Example usage
@timed
def process_reads_v1(filename):
    \"\"\"Version 1: String concatenation.\"\"\"
    total_gc = 0
    count = 0
    for header, seq, qual in read_fastq_generator(filename):
        gc = seq.count('G') + seq.count('C')
        total_gc += gc / len(seq)
        count += 1
    return total_gc / count

@timed
def process_reads_v2(filename):
    \"\"\"Version 2: Using translate for counting.\"\"\"
    total_gc = 0
    count = 0
    gc_set = set('GCgc')
    for header, seq, qual in read_fastq_generator(filename):
        gc = sum(1 for c in seq if c in gc_set)
        total_gc += gc / len(seq)
        count += 1
    return total_gc / count

print("Benchmarking GC calculation methods:")
r1 = process_reads_v1(large_file)
r2 = process_reads_v2(large_file)
print(f"  Results match: {abs(r1 - r2) < 0.0001}")"""))

cells.append(md("### 8.2 Memory Profiling"))

cells.append(code("""def measure_memory_approaches(filename: str) -> None:
    \"\"\"Compare memory usage of different approaches.\"\"\"
    import gc as gc_mod

    # Approach 1: Load all into list
    gc_mod.collect()
    records = read_fastq_list(filename)
    list_mem = sys.getsizeof(records)
    # Deeper measurement
    deep_mem = list_mem + sum(
        sys.getsizeof(r) + sum(sys.getsizeof(s) for s in r)
        for r in records
    )
    del records
    gc_mod.collect()

    # Approach 2: Generator (just the generator object)
    gen = read_fastq_generator(filename)
    gen_mem = sys.getsizeof(gen)

    # Approach 3: Chunks of 1000
    chunk_gen = read_fastq_chunks(filename, chunk_size=1000)
    first_chunk = next(chunk_gen)
    chunk_mem = sys.getsizeof(first_chunk) + sum(
        sys.getsizeof(r) + sum(sys.getsizeof(s) for s in r)
        for r in first_chunk
    )

    print("Memory comparison (sample.fastq):")
    print(f"  Full list:    {deep_mem/1024:>10.1f} KB")
    print(f"  Generator:    {gen_mem:>10d} bytes")
    print(f"  Chunk (1000): {chunk_mem/1024:>10.1f} KB")
    print(f"  List/Generator ratio: {deep_mem/gen_mem:,.0f}x")
    print(f"  List/Chunk ratio:     {deep_mem/chunk_mem:,.1f}x")

measure_memory_approaches(sample_file)"""))

cells.append(md("""### Exercise 8.1: Profile and Optimize

**Task:** Profile the quality filtering pipeline, identify the bottleneck, and optimize it."""))

cells.append(code("""def profile_pipeline(filename: str) -> None:
    \"\"\"Profile each stage of a filtering pipeline.

    TODO: Implement this function
    - Create a pipeline with: read -> filter_quality -> filter_ns -> trim -> stats
    - Time each stage individually to find the bottleneck
    - Hint: Run each stage separately on the same data to measure its cost

    Expected output format:
      Stage timings:
        read_fastq_generator: X.XXXXs
        filter_by_quality:    X.XXXXs
        filter_no_ns:         X.XXXXs
        trim_low_quality:     X.XXXXs
      Bottleneck: <stage_name>
    \"\"\"
    # TODO: Your implementation here
    pass

# profile_pipeline(large_file)"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 8.1 (click to expand)
# =============================================

def profile_pipeline(filename: str) -> None:
    \"\"\"Profile each stage of a filtering pipeline.\"\"\"
    # First, load all records so we can test each stage independently
    records = read_fastq_list(filename)

    timings = {}

    # Time: reading
    start = time.perf_counter()
    _ = read_fastq_list(filename)
    timings['read_fastq'] = time.perf_counter() - start

    # Time: quality filter
    start = time.perf_counter()
    _ = list(filter_by_quality(iter(records), min_avg_quality=20.0))
    timings['filter_quality'] = time.perf_counter() - start

    # Time: N filter
    start = time.perf_counter()
    _ = list(filter_no_ns(iter(records), max_n_fraction=0.05))
    timings['filter_no_ns'] = time.perf_counter() - start

    # Time: trimming
    start = time.perf_counter()
    _ = list(trim_low_quality_ends(iter(records), min_quality=15))
    timings['trim_quality'] = time.perf_counter() - start

    # Time: stats computation
    start = time.perf_counter()
    _ = compute_stats(iter(records))
    timings['compute_stats'] = time.perf_counter() - start

    # Report
    print("Stage timings:")
    bottleneck = max(timings, key=timings.get)
    for stage, t in timings.items():
        marker = " <-- bottleneck" if stage == bottleneck else ""
        print(f"  {stage:<20} {t:.4f}s{marker}")

    total = sum(timings.values())
    print(f"\\n  Total: {total:.4f}s")
    print(f"  Bottleneck '{bottleneck}' is {timings[bottleneck]/total*100:.1f}% of total time")

profile_pipeline(large_file)"""))

# ============================================================
# PART 9: PAIRED-END PROCESSING
# ============================================================
cells.append(md("""---
## Part 9: Paired-End Read Processing

### What are Paired-End Reads?

In Illumina sequencing, each DNA fragment is read from both ends:
- **R1** (Read 1): Forward read from the 5' end
- **R2** (Read 2): Reverse complement from the 3' end

```
Fragment:  5'---[====R1====>............<====R2====]---3'
                |<------------ insert size ----------->|
```

Paired-end reads must be processed **in sync** — if you filter out a read from R1, you must also remove its mate from R2. Losing sync corrupts downstream analysis."""))

cells.append(md("### 9.1 Synchronized Paired-End Reading"))

cells.append(code("""def read_paired_fastq(r1_file: str,
                     r2_file: str) -> Generator[Tuple[Tuple, Tuple], None, None]:
    \"\"\"Read paired-end FASTQ files in sync.

    Yields tuples of ((r1_header, r1_seq, r1_qual), (r2_header, r2_seq, r2_qual)).
    Validates that read IDs match between R1 and R2.
    \"\"\"
    r1_gen = read_fastq_generator(r1_file)
    r2_gen = read_fastq_generator(r2_file)

    for r1_record, r2_record in zip(r1_gen, r2_gen):
        # Verify paired reads match (compare base read ID)
        r1_id = r1_record[0].split()[0]
        r2_id = r2_record[0].split()[0]
        # Strip /1 and /2 suffixes if present
        if r1_id.endswith('/1'):
            r1_id = r1_id[:-2]
        if r2_id.endswith('/2'):
            r2_id = r2_id[:-2]
        if r1_id != r2_id:
            raise ValueError(f"Paired read mismatch: {r1_id} vs {r2_id}")
        yield (r1_record, r2_record)

# Test synchronized reading
r1_file = os.path.join(DATA_DIR, 'sample_R1.fastq')
r2_file = os.path.join(DATA_DIR, 'sample_R2.fastq')

count = 0
for r1, r2 in read_paired_fastq(r1_file, r2_file):
    count += 1
    if count <= 3:
        print(f"Pair {count}:")
        print(f"  R1: {r1[0].split()[0]} -> {r1[1][:30]}...")
        print(f"  R2: {r2[0].split()[0]} -> {r2[1][:30]}...")
print(f"\\nTotal read pairs: {count:,}")"""))

cells.append(md("### 9.2 Paired-End Quality Filtering"))

cells.append(code("""def filter_paired_reads(r1_file: str, r2_file: str,
                      out_r1: str, out_r2: str,
                      min_avg_quality: float = 20.0) -> dict:
    \"\"\"Filter paired-end reads — both mates must pass.

    If either R1 or R2 fails the quality check, BOTH are discarded.
    This maintains synchronization between the files.
    \"\"\"
    stats = {'total_pairs': 0, 'passed_pairs': 0, 'failed_pairs': 0}

    with open(out_r1, 'w') as f1, open(out_r2, 'w') as f2:
        for r1, r2 in read_paired_fastq(r1_file, r2_file):
            stats['total_pairs'] += 1

            # Both reads must pass
            r1_qual = sum(ord(c) - 33 for c in r1[2]) / len(r1[2])
            r2_qual = sum(ord(c) - 33 for c in r2[2]) / len(r2[2])

            if r1_qual >= min_avg_quality and r2_qual >= min_avg_quality:
                stats['passed_pairs'] += 1
                for rec, fh in [(r1, f1), (r2, f2)]:
                    fh.write(f"{rec[0]}\\n{rec[1]}\\n+\\n{rec[2]}\\n")
            else:
                stats['failed_pairs'] += 1

    return stats

# Test
pe_stats = filter_paired_reads(
    r1_file, r2_file,
    os.path.join(DATA_DIR, 'filtered_R1.fastq'),
    os.path.join(DATA_DIR, 'filtered_R2.fastq'),
    min_avg_quality=25.0
)
print(f"Paired-end filtering:")
print(f"  Total pairs:  {pe_stats['total_pairs']:,}")
print(f"  Passed pairs: {pe_stats['passed_pairs']:,}")
print(f"  Failed pairs: {pe_stats['failed_pairs']:,}")
print(f"  Pass rate:    {100*pe_stats['passed_pairs']/pe_stats['total_pairs']:.1f}%")"""))

cells.append(md("""### Exercise 9.1: Paired-End Insert Size Estimation

**Task:** Estimate insert sizes by finding overlapping regions between R1 and R2 reads."""))

cells.append(code("""def estimate_insert_sizes(r1_file: str, r2_file: str,
                         num_samples: int = 1000) -> List[int]:
    \"\"\"Estimate insert sizes from paired-end reads.

    TODO: Implement this function
    - For the first num_samples read pairs:
    - Reverse complement the R2 read
    - Find the overlap between R1 and reverse-complemented R2
    - Insert size = len(R1) + len(R2) - overlap_length
    - Return list of estimated insert sizes

    Hint: Try matching the last k bases of R1 with the first k bases of RC(R2),
    starting from k=20 up to k=read_length, and take the first good match.
    Or simply use the known read_length and the data generation parameters.
    \"\"\"
    # TODO: Your implementation here
    pass

# Test:
# insert_sizes = estimate_insert_sizes(r1_file, r2_file)
# if insert_sizes:
#     avg_insert = sum(insert_sizes) / len(insert_sizes)
#     print(f"Estimated mean insert size: {avg_insert:.0f}")
#     print(f"Insert size range: {min(insert_sizes)}-{max(insert_sizes)}")"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 9.1 (click to expand)
# =============================================

def reverse_complement(seq: str) -> str:
    \"\"\"Return the reverse complement of a DNA sequence.\"\"\"
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))

def estimate_insert_sizes(r1_file: str, r2_file: str,
                         num_samples: int = 1000) -> List[int]:
    \"\"\"Estimate insert sizes from paired-end reads.

    Since our simulated data has a known structure (R2 is reverse complement
    of the fragment end), we can search for overlaps to estimate insert size.
    \"\"\"
    insert_sizes = []
    min_overlap = 10

    for i, (r1, r2) in enumerate(read_paired_fastq(r1_file, r2_file)):
        if i >= num_samples:
            break

        r1_seq = r1[1]
        r2_rc = reverse_complement(r2[1])
        read_len = len(r1_seq)

        # Try to find overlap between end of R1 and start of R2_rc
        best_overlap = 0
        for overlap_len in range(min_overlap, read_len):
            r1_end = r1_seq[read_len - overlap_len:]
            r2_start = r2_rc[:overlap_len]
            # Count matches
            matches = sum(1 for a, b in zip(r1_end, r2_start) if a == b)
            if matches / overlap_len > 0.85:  # 85% match threshold
                best_overlap = overlap_len

        if best_overlap > 0:
            insert_size = 2 * read_len - best_overlap
            insert_sizes.append(insert_size)
        else:
            # No overlap found — insert size > 2 * read_length
            # Estimate based on known generation parameters
            insert_sizes.append(300)  # default estimate

    return insert_sizes

insert_sizes = estimate_insert_sizes(r1_file, r2_file)
if insert_sizes:
    avg_insert = sum(insert_sizes) / len(insert_sizes)
    print(f"Sampled {len(insert_sizes)} read pairs")
    print(f"Estimated mean insert size: {avg_insert:.0f}")
    print(f"Insert size range: {min(insert_sizes)}-{max(insert_sizes)}")"""))

# ============================================================
# PART 10: FINAL PROJECT
# ============================================================
cells.append(md("""---
## Part 10: Final Project — Complete FASTQ Processing Pipeline

### The Challenge

Build a complete, production-quality FASTQ processing pipeline that combines everything you've learned. Your pipeline should:

1. **Accept** single-end or paired-end input
2. **Generate** a comprehensive quality report
3. **Filter** reads based on configurable criteria
4. **Process** data efficiently using streaming and/or parallel approaches
5. **Output** filtered reads and a summary report

This is an open-ended exercise. Use whatever combination of techniques from Parts 0-9 makes sense."""))

cells.append(md("### 10.1 Pipeline Specification"))

cells.append(code("""class FASTQPipeline:
    \"\"\"Complete FASTQ processing pipeline.

    TODO: Implement this class with the following methods:

    __init__(self, config: dict)
        - Store configuration (quality thresholds, chunk size, etc.)
        - Set defaults for any missing config values

    analyze(self, filename: str) -> dict
        - Generate pre-filtering statistics
        - Return: read count, avg quality, GC content, N content,
                  quality distribution, length distribution

    filter_reads(self, input_file: str, output_file: str) -> dict
        - Apply quality filtering pipeline
        - Return filtering statistics

    process_paired(self, r1_file: str, r2_file: str,
                   out_r1: str, out_r2: str) -> dict
        - Process paired-end files in sync
        - Return paired filtering statistics

    generate_report(self, stats: dict) -> str
        - Generate a text summary report
        - Return report as a string

    run(self, input_files: list, output_dir: str) -> dict
        - Full pipeline: analyze -> filter -> report
        - Handle both single-end and paired-end inputs
        - Return complete results dict
    \"\"\"

    def __init__(self, config: dict = None):
        # TODO: Your implementation here
        pass

    def analyze(self, filename: str) -> dict:
        # TODO: Your implementation here
        pass

    def filter_reads(self, input_file: str, output_file: str) -> dict:
        # TODO: Your implementation here
        pass

    def process_paired(self, r1_file: str, r2_file: str,
                       out_r1: str, out_r2: str) -> dict:
        # TODO: Your implementation here
        pass

    def generate_report(self, stats: dict) -> str:
        # TODO: Your implementation here
        pass

    def run(self, input_files: list, output_dir: str) -> dict:
        # TODO: Your implementation here
        pass

# Test:
# pipeline = FASTQPipeline({
#     'min_quality': 20,
#     'min_length': 50,
#     'max_n_fraction': 0.05,
#     'chunk_size': 2500
# })
# results = pipeline.run(
#     [sample_file],
#     os.path.join(DATA_DIR, 'pipeline_output')
# )
# print(pipeline.generate_report(results))"""))

cells.append(solution_code("""# =============================================
# SOLUTION - Exercise 10 (click to expand)
# =============================================

class FASTQPipeline:
    \"\"\"Complete FASTQ processing pipeline.\"\"\"

    def __init__(self, config: dict = None):
        defaults = {
            'min_quality': 20.0,
            'min_length': 50,
            'max_n_fraction': 0.05,
            'trim_quality': 15,
            'chunk_size': 2500,
            'num_workers': min(4, multiprocessing.cpu_count()),
        }
        self.config = {**defaults, **(config or {})}

    def analyze(self, filename: str) -> dict:
        \"\"\"Generate pre-filtering statistics.\"\"\"
        total_reads = 0
        total_bases = 0
        gc_bases = 0
        n_bases = 0
        qual_sum = 0
        qual_count = 0
        lengths = []
        quality_scores = []

        for header, seq, qual in read_fastq_generator(filename):
            total_reads += 1
            total_bases += len(seq)
            seq_upper = seq.upper()
            gc_bases += seq_upper.count('G') + seq_upper.count('C')
            n_bases += seq_upper.count('N')
            lengths.append(len(seq))

            for c in qual:
                q = ord(c) - 33
                qual_sum += q
                qual_count += 1
                quality_scores.append(q)

        return {
            'filename': os.path.basename(filename),
            'total_reads': total_reads,
            'total_bases': total_bases,
            'avg_length': total_bases / total_reads if total_reads else 0,
            'gc_content': gc_bases / total_bases if total_bases else 0,
            'n_content': n_bases / total_bases if total_bases else 0,
            'avg_quality': qual_sum / qual_count if qual_count else 0,
        }

    def filter_reads(self, input_file: str, output_file: str) -> dict:
        \"\"\"Apply quality filtering pipeline.\"\"\"
        stats = {'total': 0, 'passed': 0, 'failed': 0}

        pipeline = read_fastq_generator(input_file)
        pipeline = filter_by_quality(pipeline, self.config['min_quality'])
        pipeline = filter_no_ns(pipeline, self.config['max_n_fraction'])
        pipeline = trim_low_quality_ends(pipeline, self.config['trim_quality'])
        pipeline = filter_by_length(pipeline, self.config['min_length'])

        # We need to count total separately since pipeline may skip reads
        total = 0
        passed = 0
        with open(output_file, 'w') as out:
            # Count total from a separate pass
            for header, seq, qual in read_fastq_generator(input_file):
                total += 1

        stats['total'] = total

        with open(output_file, 'w') as out:
            pipeline = read_fastq_generator(input_file)
            pipeline = filter_by_quality(pipeline, self.config['min_quality'])
            pipeline = filter_no_ns(pipeline, self.config['max_n_fraction'])
            pipeline = trim_low_quality_ends(pipeline, self.config['trim_quality'])
            pipeline = filter_by_length(pipeline, self.config['min_length'])

            for header, seq, qual in pipeline:
                passed += 1
                out.write(f"{header}\\n{seq}\\n+\\n{qual}\\n")

        stats['passed'] = passed
        stats['failed'] = total - passed
        return stats

    def process_paired(self, r1_file: str, r2_file: str,
                       out_r1: str, out_r2: str) -> dict:
        \"\"\"Process paired-end files in sync.\"\"\"
        return filter_paired_reads(
            r1_file, r2_file, out_r1, out_r2,
            min_avg_quality=self.config['min_quality']
        )

    def generate_report(self, stats: dict) -> str:
        \"\"\"Generate a text summary report.\"\"\"
        lines = []
        lines.append("=" * 60)
        lines.append("FASTQ Processing Pipeline Report")
        lines.append("=" * 60)

        if 'pre_filter' in stats:
            pf = stats['pre_filter']
            lines.append(f"\\nInput: {pf.get('filename', 'N/A')}")
            lines.append(f"  Total reads:  {pf.get('total_reads', 0):,}")
            lines.append(f"  Total bases:  {pf.get('total_bases', 0):,}")
            lines.append(f"  Avg length:   {pf.get('avg_length', 0):.1f}")
            lines.append(f"  GC content:   {pf.get('gc_content', 0)*100:.2f}%")
            lines.append(f"  N content:    {pf.get('n_content', 0)*100:.4f}%")
            lines.append(f"  Avg quality:  {pf.get('avg_quality', 0):.2f}")

        if 'filtering' in stats:
            fs = stats['filtering']
            lines.append(f"\\nFiltering:")
            lines.append(f"  Input reads:  {fs.get('total', 0):,}")
            lines.append(f"  Passed:       {fs.get('passed', 0):,}")
            lines.append(f"  Failed:       {fs.get('failed', 0):,}")
            if fs.get('total', 0) > 0:
                rate = 100 * fs['passed'] / fs['total']
                lines.append(f"  Pass rate:    {rate:.1f}%")

        if 'config' in stats:
            lines.append(f"\\nConfiguration:")
            for k, v in stats['config'].items():
                lines.append(f"  {k}: {v}")

        lines.append("\\n" + "=" * 60)
        return "\\n".join(lines)

    def run(self, input_files: list, output_dir: str) -> dict:
        \"\"\"Full pipeline: analyze -> filter -> report.\"\"\"
        os.makedirs(output_dir, exist_ok=True)
        results = {'config': self.config}

        if len(input_files) == 1:
            # Single-end mode
            input_file = input_files[0]
            results['pre_filter'] = self.analyze(input_file)

            output_file = os.path.join(
                output_dir,
                'filtered_' + os.path.basename(input_file)
            )
            results['filtering'] = self.filter_reads(input_file, output_file)
            results['output_file'] = output_file

        elif len(input_files) == 2:
            # Paired-end mode
            results['pre_filter'] = self.analyze(input_files[0])
            out_r1 = os.path.join(output_dir, 'filtered_' + os.path.basename(input_files[0]))
            out_r2 = os.path.join(output_dir, 'filtered_' + os.path.basename(input_files[1]))
            pe_stats = self.process_paired(
                input_files[0], input_files[1], out_r1, out_r2
            )
            results['filtering'] = {
                'total': pe_stats['total_pairs'],
                'passed': pe_stats['passed_pairs'],
                'failed': pe_stats['failed_pairs']
            }

        return results

# Run the pipeline
pipeline = FASTQPipeline({
    'min_quality': 20,
    'min_length': 50,
    'max_n_fraction': 0.05,
})

print("Running pipeline on single-end data...")
results = pipeline.run(
    [sample_file],
    os.path.join(DATA_DIR, 'pipeline_output')
)
print(pipeline.generate_report(results))

print("\\n\\nRunning pipeline on paired-end data...")
pe_results = pipeline.run(
    [r1_file, r2_file],
    os.path.join(DATA_DIR, 'pipeline_output')
)
print(pipeline.generate_report(pe_results))"""))

cells.append(md("""---
## Congratulations!

You've completed the Big Data Bioinformatics Exercises! Here's a summary of what you've learned:

| Part | Technique | Big Data Principle |
|------|-----------|-------------------|
| 1 | Generators | Process data larger than RAM |
| 2 | Chunking | Batch processing with progress |
| 3 | Parallel processing | Use all CPU cores |
| 4 | Indexing | O(1) random access |
| 5 | Streaming pipelines | Composable, memory-efficient transforms |
| 6 | MapReduce | Distributed aggregation pattern |
| 7 | Tool integration | Leverage optimized C/Go tools |
| 8 | Profiling | Measure before optimizing |
| 9 | Paired-end | Synchronized multi-file processing |
| 10 | Pipeline | Combine all techniques |

### Next Steps

- Try these techniques on real FASTQ data from [SRA](https://www.ncbi.nlm.nih.gov/sra)
- Explore [Snakemake](https://snakemake.readthedocs.io/) for workflow management
- Learn [Dask](https://dask.org/) for out-of-core DataFrames
- Study [Apache Spark](https://spark.apache.org/) for true distributed computing"""))

# ============================================================
# BUILD THE NOTEBOOK
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "cells": cells
}

output_path = '/Users/shanebrubaker/work/BINFX410/C04/big-data-exercises/BigData_Bioinformatics_Exercises.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
solution_cells = sum(1 for c in cells if 'solution' in c.get('metadata', {}).get('tags', []))
print(f"  Solution cells: {solution_cells}")
