# Big Data Bioinformatics Exercises - Implementation Plan

## Project Overview

Create a comprehensive Jupyter notebook teaching big data concepts for bioinformatics using small example files. The notebook should demonstrate memory-efficient parsing, parallel processing, streaming, indexing, and MapReduce patterns while integrating industry-standard tools (seqkit, fastp) alongside custom Python implementations.

**Target Audience**: BINFX410 students (computational biology/bioinformatics)  
**Cost**: $0 (runs entirely locally)  
**Duration**: 2-3 hour workshop  
**Platform**: Jupyter Notebook

## Learning Objectives

Students will:
1. Understand memory-efficient file parsing (generators vs loading all data)
2. Implement streaming and chunking patterns for large files
3. Compare serial vs parallel processing approaches
4. Build and use indexes for random access
5. Apply MapReduce thinking to genomics problems
6. Use industry tools (seqkit, fastp) effectively
7. Profile and optimize bioinformatics code
8. Bridge between custom Python and production tools

## Prerequisites and Setup

### Required Software
- Python 3.8+
- Jupyter Notebook
- seqkit (conda install -c bioconda seqkit)
- fastp (conda install -c bioconda fastp)
- Standard Python libraries: subprocess, multiprocessing, time, sys, tracemalloc

### Installation Commands
```bash
# Create conda environment
conda create -n bigdata_bioinfo python=3.9
conda activate bigdata_bioinfo

# Install tools
conda install -c bioconda seqkit fastp
conda install jupyter pandas matplotlib

# Launch notebook
jupyter notebook
```

## File Structure

```
bigdata_bioinformatics_exercises/
├── BigData_Bioinformatics_Exercises.ipynb  # Main notebook
├── data/                                    # Generated test data
│   ├── test_reads.fastq
│   ├── large_test.fastq
│   ├── paired_R1.fastq
│   └── paired_R2.fastq
├── outputs/                                 # Student outputs
├── solutions/                               # Instructor solutions (optional)
└── README.md                                # Quick start guide
```

## Notebook Structure

### Part 0: Setup and Data Generation

**Objective**: Create synthetic test datasets

**Implementation**:
- Function to generate realistic FASTQ files with variable quality
- Create small (10K reads), medium (50K reads), and large (100K reads) datasets
- Create paired-end read files
- Generate files with adapters for fastp testing

**Code sections**:
1. `create_test_fastq(filename, num_reads, read_length, quality_profile)`
2. `create_paired_end_fastq(prefix, num_reads)`
3. `add_adapters_to_fastq(input, output, adapter_seq)`

### Part 1: Memory Efficiency - Generators vs Loading

**Objective**: Understand memory implications of different parsing approaches

**Sections**:

#### 1.1 The Problem
- Show memory explosion when loading entire FASTQ into memory
- Demonstrate tracemalloc for memory profiling

#### 1.2 Generator-Based Parsing
- Implement `parse_fastq_generator()` 
- Compare memory usage: loading vs streaming
- Visualize memory usage with matplotlib

#### 1.3 Compare with seqkit
- Use `seqkit stats` to get file statistics instantly
- Use `seqkit sample` to create subsets without loading entire file
- Time comparison: Python loading vs Python generator vs seqkit

**Student Exercise 1.1**: 
Calculate GC content both ways (loading vs streaming) and measure memory difference

**Student Exercise 1.2**: 
Use seqkit to get stats on multiple files simultaneously and compare speed to custom Python

**Code deliverable**: 
- `calculate_gc_streaming(filename) -> float`
- Comparison table showing memory/time for different approaches

### Part 2: Chunking and Batching

**Objective**: Process data in manageable chunks for efficiency

**Sections**:

#### 2.1 Chunked Reading
- Implement `parse_fastq_chunks(filename, chunk_size)`
- Demonstrate batch quality score calculation
- Show effect of different chunk sizes on performance

#### 2.2 Quality Filtering with Chunks
- Filter reads by average quality using chunked processing
- Write filtered output efficiently

#### 2.3 seqkit Filtering
- Use `seqkit seq --min-qual` for quality filtering
- Use `seqkit seq --min-len --max-len` for length filtering
- Compare performance: chunked Python vs seqkit

**Student Exercise 2.1**: 
Implement quality filtering with different chunk sizes (100, 1000, 10000) and measure performance

**Student Exercise 2.2**: 
Use seqkit to filter by multiple criteria simultaneously and compare to sequential Python filters

**Code deliverable**: 
- `filter_by_quality_chunked(input, output, min_qual, chunk_size)`
- Performance comparison chart

### Part 3: Parallel Processing

**Objective**: Speed up computation with multiprocessing

**Sections**:

#### 3.1 Serial K-mer Counting
- Implement `count_kmers_serial(filename, k)`
- Baseline performance measurement

#### 3.2 Parallel K-mer Counting
- Implement `count_kmers_parallel(filename, k, num_processes)`
- Use multiprocessing.Pool for parallelization
- Map-Reduce pattern for aggregating results

#### 3.3 Speedup Analysis
- Test with 1, 2, 4, 8 processes
- Plot speedup vs number of cores
- Discuss Amdahl's Law and parallel overhead

#### 3.4 seqkit Parallel Processing
- Demonstrate `seqkit --threads` option
- Compare seqkit multi-threading to Python multiprocessing

**Student Exercise 3.1**: 
Implement parallel quality filtering and measure speedup

**Student Exercise 3.2**: 
Use seqkit to process multiple FASTQ files in parallel using shell scripting

**Code deliverable**: 
- `filter_quality_parallel(input, output, min_qual, num_processes)`
- Speedup graph (cores vs time)

### Part 4: Indexing for Random Access

**Objective**: Build indexes for O(1) read access

**Sections**:

#### 4.1 Building a FASTQ Index
- Implement `FASTQIndex` class
- `build_index()` method using file offsets
- `load_index()` and `get_read(read_id)` methods

#### 4.2 Sequential vs Indexed Access
- Benchmark: find 10 specific reads via scan vs index
- Demonstrate dramatic speedup for random access

#### 4.3 Range Queries
- Extend index to support range queries
- Implement "sample by stride" (every Nth read)

#### 4.4 Real-World Indexing
- Explain BAM/CRAM indexing (.bai, .crai files)
- Show how samtools uses indexes: `samtools view -b file.bam chr1:1000-2000`
- Discuss when indexing is worth the overhead

**Student Exercise 4.1**: 
Implement `get_read_range(start_id, end_id)` using the index

**Student Exercise 4.2**: 
Create an index that supports queries by quality score range

**Code deliverable**: 
- Complete `FASTQIndex` class with range query support
- Performance comparison table (scan vs index for different access patterns)

### Part 5: Streaming Analysis Pipelines

**Objective**: Build composable, memory-efficient analysis pipelines

**Sections**:

#### 5.1 Pipeline Design Pattern
- Implement `StreamingPipeline` class
- Support for chainable filters and analyzers
- Single-pass processing

#### 5.2 Reusable Filters
- `filter_min_length(min_len)`
- `filter_min_quality(min_qual)`
- `filter_no_ns()`
- `filter_gc_range(min_gc, max_gc)`
- `filter_contains_motif(motif)`

#### 5.3 Reusable Analyzers
- `analyze_composition()` - track GC%, length distribution
- `analyze_quality()` - quality metrics
- `analyze_kmer_diversity()` - unique k-mers

#### 5.4 fastp Integration
- Run fastp for comprehensive QC
- Parse fastp JSON output
- Compare custom pipeline metrics to fastp results

**Student Exercise 5.1**: 
Build a pipeline with custom filters and analyzers, output summary statistics

**Student Exercise 5.2**: 
Compare your pipeline results to fastp output - do they match?

**Code deliverable**: 
- Custom pipeline that filters and analyzes reads
- Comparison table: custom pipeline vs fastp

### Part 6: MapReduce Pattern for Genomics

**Objective**: Understand distributed computing patterns

**Sections**:

#### 6.1 MapReduce Fundamentals
- Explain Map, Shuffle, Reduce phases
- Implement `mapreduce_fastq(filename, map_func, reduce_func, partitions)`

#### 6.2 K-mer Counting with MapReduce
- `map_kmers(read, k)` - emit (kmer, 1) pairs
- `reduce_kmers(kmer, counts)` - sum counts
- Demonstrate partitioning and aggregation

#### 6.3 Quality Category Analysis
- Map reads to quality categories (low/medium/high)
- Calculate average GC% per category
- Find most common motifs in each category

#### 6.4 Connection to Real Distributed Systems
- Discuss how this relates to Spark, BigSeqKit
- Explain when distributed processing is necessary
- Show theoretical speedup calculation

**Student Exercise 6.1**: 
Implement MapReduce to count reads by quality category and calculate stats per category

**Student Exercise 6.2**: 
Use MapReduce pattern to find the 100 most common 10-mers in high-quality reads

**Code deliverable**: 
- `map_quality_category(read)` and `reduce_quality_stats(category, infos)`
- Top motifs per quality category

### Part 7: Tool Integration and Comparison

**Objective**: Compare custom code to industry-standard tools

**Sections**:

#### 7.1 seqkit Comprehensive Operations
- Statistics: `seqkit stats`
- Sampling: `seqkit sample`
- Filtering: `seqkit seq`
- Searching: `seqkit grep`
- Format conversion: `seqkit fq2fa`

#### 7.2 fastp Quality Control
- Adapter trimming
- Quality filtering
- Length filtering
- HTML report generation
- JSON output parsing

#### 7.3 Python Wrapper Classes
- Implement `SeqKitHelper` class for calling seqkit from Python
- Implement `FastpHelper` class for running fastp
- Parse and visualize tool outputs

#### 7.4 Benchmark Comparison
- Task: Filter 50K reads by quality and length
- Compare: Pure Python, seqkit, fastp
- Metrics: Time, memory, output size

**Student Exercise 7.1**: 
Create a wrapper function that runs fastp and extracts key metrics from JSON output

**Student Exercise 7.2**: 
Build a comparison table showing Python vs seqkit vs fastp for common tasks

**Code deliverable**: 
- `SeqKitHelper` and `FastpHelper` classes
- Comprehensive benchmark table

### Part 8: Performance Profiling and Optimization

**Objective**: Learn to identify and fix performance bottlenecks

**Sections**:

#### 8.1 Profiling with cProfile
- Profile three different GC% calculation implementations
- Identify bottlenecks
- Understand cumulative time vs total time

#### 8.2 Memory Profiling
- Use tracemalloc to track memory allocation
- Find memory leaks and inefficiencies

#### 8.3 Optimization Case Study
- Start with deliberately slow reverse-complement function
- Profile to find bottlenecks
- Optimize step by step
- Measure speedup at each step

#### 8.4 Best Practices
- When to optimize vs when to use existing tools
- Profile before optimizing
- Big-O analysis for bioinformatics operations

**Student Exercise 8.1**: 
Profile the provided slow_reverse_complement() function, identify bottlenecks, and create optimized version

**Student Exercise 8.2**: 
Compare your optimized reverse-complement to `seqkit seq -r -p` - which is faster?

**Code deliverable**: 
- Optimized `fast_reverse_complement()` function
- Profiling report showing improvements

### Part 9: Paired-End Read Handling

**Objective**: Handle paired-end sequencing data

**Sections**:

#### 9.1 Paired-End Concepts
- Explain R1/R2 read pairs
- Importance of maintaining pairing
- Common paired-end operations

#### 9.2 Synchronized Processing
- Parse R1 and R2 files simultaneously
- Filter both files with same criteria
- Maintain read pairing

#### 9.3 seqkit Paired-End Operations
- Sample paired files: `seqkit sample -p 0.1 -s 100` (same seed)
- Process R1 and R2 with identical operations

#### 9.4 fastp Paired-End QC
- Adapter trimming for paired reads
- Merged output generation
- Per-read pair statistics

**Student Exercise 9.1**: 
Implement synchronized filtering for paired-end reads

**Student Exercise 9.2**: 
Use fastp to QC paired-end reads and extract insert size distribution

**Code deliverable**: 
- `filter_paired_end(r1_in, r2_in, r1_out, r2_out, filters)`
- Insert size histogram

### Part 10: Final Integrated Project

**Objective**: Combine all concepts into production-quality QC pipeline

**Sections**:

#### 10.1 Project Specification
Design and implement a complete QC pipeline that:
1. Accepts FASTQ input (single or paired-end, gzipped or not)
2. Generates comprehensive statistics
3. Filters by quality, length, and complexity
4. Removes adapters if present
5. Counts k-mers in parallel
6. Builds index of clean reads
7. Generates detailed report (markdown + plots)
8. Compares results to fastp baseline

#### 10.2 Pipeline Architecture
- Modular design with separate filter/analysis stages
- Configuration via parameters or config file
- Progress reporting
- Error handling

#### 10.3 Performance Optimization
- Use multiprocessing where beneficial
- Stream processing throughout
- Efficient I/O (chunked writes)
- Memory profiling

#### 10.4 Validation
- Compare outputs to fastp and seqkit
- Verify all reads accounted for
- Check output file integrity

**Student Deliverables**:
1. Complete pipeline implementation
2. Test with provided datasets
3. Generate report comparing to fastp/seqkit
4. Document design decisions
5. Performance analysis showing optimizations

## Code Implementation Details

### Cell 1: Environment Setup and Imports
```python
import os
import sys
import time
import gzip
import pickle
import subprocess
import tracemalloc
import cProfile
import pstats
from io import StringIO
from pathlib import Path
from typing import Generator, List, Dict, Callable, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
import json

# Visualization
import matplotlib.pyplot as plt
import pandas as pd

# Check for required tools
def check_tool(tool_name):
    """Check if a command-line tool is available"""
    try:
        subprocess.run([tool_name, '--help'], 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

print("Checking required tools...")
print(f"seqkit: {'✓' if check_tool('seqkit') else '✗ NOT FOUND'}")
print(f"fastp: {'✓' if check_tool('fastp') else '✗ NOT FOUND'}")
```

### Cell 2: Data Generation Functions
```python
def create_test_fastq(filename: str, 
                      num_reads: int = 10000,
                      read_length: int = 150,
                      quality_profile: str = 'high') -> None:
    """
    Generate synthetic FASTQ file with realistic quality scores.
    
    Args:
        filename: Output FASTQ filename
        num_reads: Number of reads to generate
        read_length: Length of each read
        quality_profile: 'high', 'medium', or 'low' quality
    """
    import random
    
    bases = ['A', 'C', 'G', 'T']
    
    # Quality score ranges (Phred+33)
    quality_ranges = {
        'high': 'IIIIIIIIII',      # Q=40
        'medium': 'EEEEEE',         # Q=36
        'low': '(((((('             # Q=7
    }
    
    quality_chars = quality_ranges.get(quality_profile, quality_ranges['high'])
    
    with open(filename, 'w') as f:
        for i in range(num_reads):
            read_id = f"@READ_{i:06d}"
            sequence = ''.join(random.choices(bases, k=read_length))
            
            # Variable quality - higher at beginning, lower at end
            quality = ''
            for pos in range(read_length):
                if pos < read_length * 0.7:
                    quality += random.choice(quality_chars)
                else:
                    # Lower quality towards end
                    quality += random.choice('###(((')
            
            f.write(f"{read_id}\n{sequence}\n+\n{quality}\n")
    
    print(f"Created {filename} with {num_reads} reads ({quality_profile} quality)")

# Additional helper functions...
```

### Cell 3-N: Exercise Implementations

Each exercise should be implemented as described in the sections above, with:
- Clear markdown cell explaining the concept
- Code cell with implementation
- Code cell for demonstration/testing
- Student exercise cell (with TODO markers)
- Solution cell (initially collapsed or in separate notebook)

### Tool Integration Pattern

For each tool (seqkit, fastp), provide wrapper classes:

```python
class SeqKitHelper:
    """Python wrapper for seqkit operations"""
    
    @staticmethod
    def get_stats(fastq_file: str) -> pd.DataFrame:
        """Get statistics using seqkit stats"""
        result = subprocess.run(
            ['seqkit', 'stats', fastq_file, '-T'],
            capture_output=True,
            text=True,
            check=True
        )
        return pd.read_csv(StringIO(result.stdout), sep='\t')
    
    @staticmethod
    def filter_quality(input_file: str, output_file: str, 
                       min_qual: int = 20) -> None:
        """Filter by quality using seqkit"""
        subprocess.run([
            'seqkit', 'seq',
            '--min-qual', str(min_qual),
            input_file,
            '-o', output_file
        ], check=True)
    
    # Additional methods...

class FastpHelper:
    """Python wrapper for fastp operations"""
    
    @staticmethod
    def run_qc(input_file: str, output_file: str,
               json_file: str = 'fastp.json',
               html_file: str = 'fastp.html',
               **kwargs) -> Dict:
        """Run fastp and return metrics"""
        cmd = [
            'fastp',
            '-i', input_file,
            '-o', output_file,
            '-j', json_file,
            '-h', html_file
        ]
        
        # Add optional parameters
        if 'min_qual' in kwargs:
            cmd.extend(['-q', str(kwargs['min_qual'])])
        if 'min_len' in kwargs:
            cmd.extend(['-l', str(kwargs['min_len'])])
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Parse JSON output
        with open(json_file) as f:
            return json.load(f)
    
    # Additional methods...
```

## Student Exercise Format

Each student exercise should follow this template:

```python
# ============================================================================
# STUDENT EXERCISE X.Y: [Title]
# ============================================================================

print("""
OBJECTIVE:
[Clear statement of what student should accomplish]

REQUIREMENTS:
1. [Requirement 1]
2. [Requirement 2]
...

HINTS:
- [Helpful hint 1]
- [Helpful hint 2]

EXPECTED OUTPUT:
[Description of what the output should look like]

DELIVERABLES:
- Function: function_name(params) -> return_type
- Analysis: [What analysis to perform]
- Comparison: [What to compare]
""")

def student_function_template(param1, param2):
    """
    YOUR CODE HERE
    
    Implement the function according to requirements above.
    """
    # TODO: Implement this
    pass

# Test your implementation
# [Provide test cases]
```

## Solution Format

Solutions should be in a separate cell, initially collapsed:

```python
# ============================================================================
# SOLUTION for Exercise X.Y
# ============================================================================
# Run this cell to see the solution (try implementing yourself first!)

def solution_function(param1, param2):
    """
    Solution implementation with detailed comments.
    """
    # Step 1: [Explanation]
    result = something
    
    # Step 2: [Explanation]
    processed = process(result)
    
    return processed

# Demonstration
result = solution_function(test_input1, test_input2)
print(f"Result: {result}")

# Performance analysis
# [Include timing, memory usage, comparison to alternatives]
```

## Assessment Rubric

Each exercise should be assessed on:

1. **Correctness** (40%): Does it produce correct output?
2. **Efficiency** (30%): Is it memory/time efficient?
3. **Code Quality** (20%): Clear, documented, well-structured?
4. **Understanding** (10%): Comments demonstrate comprehension?

## Discussion Questions

Include throughout the notebook:

- 💭 **Think About It**: At what file size does streaming become necessary?
- 💭 **Think About It**: Why is seqkit faster than Python for the same task?
- 💭 **Think About It**: When would you build a custom tool vs use existing tools?
- 💭 **Think About It**: How do these patterns apply to other large file types (BAM, VCF)?

## Extension Activities

For advanced students:

1. **Compressed File Handling**: Modify all functions to handle .gz files transparently
2. **Progress Bars**: Add tqdm progress bars to long-running operations
3. **Workflow Integration**: Create a Nextflow or Snakemake workflow
4. **Cloud Processing**: Adapt pipeline to process files from S3
5. **GPU Acceleration**: Explore CuPy for k-mer counting
6. **Format Conversion**: Extend to handle FASTA, BAM, CRAM files

## Instructor Notes

### Timing Suggestions
- Part 0-1: 20 minutes
- Part 2-3: 30 minutes  
- Part 4-5: 30 minutes
- Part 6-7: 40 minutes
- Part 8-9: 30 minutes
- Part 10: 60+ minutes (can be homework)

### Common Pitfalls
- Students forgetting to use generators → memory issues
- Not closing file handles → resource leaks
- Comparing functions without warm-up runs → inaccurate timing
- Forgetting to verify paired-end reads stay synchronized

### Key Takeaways
1. Generators are essential for large file processing
2. Parallel processing has overhead - profile first
3. Right tool for the job: custom code vs existing tools
4. Understanding algorithms matters more than memorizing syntax
5. Real-world bioinformatics is about data engineering, not just analysis

## Files to Generate

The implementation should create:

1. **BigData_Bioinformatics_Exercises.ipynb** - Main student notebook
2. **BigData_Bioinformatics_Solutions.ipynb** - Instructor solutions
3. **README.md** - Quick start guide
4. **requirements.txt** - Python dependencies
5. **environment.yml** - Conda environment spec
6. **test_data/** - Generated FASTQ files
7. **utils.py** - Helper functions (optional, can be in notebook)

## Validation Checklist

Before finalizing:
- [ ] All code cells run without errors
- [ ] Test data is generated correctly
- [ ] seqkit and fastp commands work
- [ ] Student exercises have clear instructions
- [ ] Solutions are provided for all exercises
- [ ] Performance comparisons are fair (same inputs)
- [ ] Memory measurements are accurate
- [ ] Visualizations are clear and labeled
- [ ] Discussion questions promote critical thinking
- [ ] Extension activities are challenging but achievable

## Example Output Structure

Student final report should include:

```markdown
# Quality Control Analysis Report

## Input Files
- File: large_test.fastq
- Reads: 50,000
- Total bases: 7,500,000

## QC Metrics

### Before Filtering
| Metric | Value |
|--------|-------|
| Total Reads | 50,000 |
| GC% | 51.2% |
| Avg Quality | 32.4 |
| Avg Length | 150 bp |

### After Filtering
| Metric | Value |
|--------|-------|
| Reads Passing | 45,123 (90.2%) |
| GC% | 51.3% |
| Avg Quality | 35.1 |
| Avg Length | 149 bp |

## Performance Comparison

| Method | Time | Memory |
|--------|------|--------|
| Custom Python | 12.3s | 245 MB |
| seqkit | 1.8s | 32 MB |
| fastp | 2.1s | 48 MB |

## K-mer Analysis
Top 10 most common 6-mers: ...

## Conclusions
- 9.8% of reads filtered due to low quality
- seqkit is 6.8x faster than custom Python
- All methods produce identical results
```

## Notes for Claude Code

When implementing this plan:

1. **Create notebook incrementally**: Start with Part 0, test, then add Part 1, etc.
2. **Generate test data first**: Ensure `create_test_fastq()` works before other code
3. **Test tool integration early**: Verify seqkit and fastp are available
4. **Use markdown cells extensively**: Explain concepts before code
5. **Include visualizations**: Use matplotlib for performance comparisons
6. **Make exercises progressive**: Each builds on previous concepts
7. **Provide both template and solution**: Help students learn by trying first
8. **Add plenty of comments**: Explain non-obvious code
9. **Use type hints**: Make function signatures clear
10. **Include error handling**: Show students proper exception handling

## Success Criteria

A successful implementation will:
- Run completely on a standard laptop (8GB RAM, 4 cores)
- Process test files in seconds, not minutes
- Teach concepts through hands-on practice
- Show clear performance differences between approaches
- Integrate seamlessly with seqkit and fastp
- Prepare students for real-world bioinformatics workflows
- Be self-contained (no external data downloads required)
- Include comprehensive explanations and examples
- Provide immediate feedback on correctness
- Scale concepts from small examples to big data thinking
