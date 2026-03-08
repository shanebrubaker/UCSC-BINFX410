# Teaching Notes: Spark Genomics Exercise Module

## Suggested Lecture Outline (50 minutes)

### Introduction to Distributed Computing (10 min)
- Why single-machine processing has limits (memory, CPU)
- The MapReduce paradigm and its evolution
- Where Spark fits: in-memory distributed computing
- Real-world scale: gnomAD has 76K genomes, UK Biobank has 500K

### Spark Architecture (15 min)
- Driver vs. executors
- SparkSession as the entry point
- DataFrames as the primary API (not RDDs)
- Lazy evaluation and the Catalyst optimizer
- Draw the architecture on the board: Driver -> Cluster Manager -> Executors

### Live Demo: pandas vs Spark (10 min)
- Show the same groupBy operation in both
- Emphasize: for small data, pandas is faster (overhead!)
- Show the Spark UI during execution
- Key message: "Use the right tool for the right scale"

### Exercise Overview (5 min)
- Walk through the directory structure
- Show how to start the Docker environment
- Demonstrate opening Jupyter and running a cell
- Point out where to find help (common_issues.md)

### Q&A (10 min)
- Common questions: "Why not just use a bigger machine?"
- "How does this relate to cloud computing?"
- "When would I actually use this in bioinformatics?"

---

## Key Points by Notebook Section

### Notebook 00: Setup
- Ensure every student has a working environment before moving on
- The Spark UI is a critical learning tool - make sure everyone can access it
- Common issue: port 4040 conflicts if students have other Spark sessions

### Notebook 01: Basics
- **Lazy evaluation**: Draw the transformation chain on the board. Show how
  `.explain()` reveals the plan without executing.
- **Partitions**: Use the analogy of splitting a deck of cards among players.
  Each player processes their cards independently.
- **Shuffles**: This is THE key concept. Draw data moving between partitions.
  Emphasize the cost: disk I/O + network transfer.
- **Caching**: "Would you re-read a book every time someone asks you a question
  about it?"

### Notebook 02: Exercise
- Let students struggle with the join (Task 5) - it's the hardest part
- The optimization challenge (Task 7) often sparks good discussion
- Performance comparison (Task 6) usually surprises students - Spark is
  slower on small data!

### Notebook 03: Advanced
- Window functions are powerful but confusing - use the "sliding window over
  a sorted table" mental model
- UDF performance comparison is eye-opening - always prefer built-in functions
- ML section is a preview, not a deep dive

---

## Common Student Misconceptions

1. **"More nodes = always faster"**
   - Reality: Communication overhead can dominate for small datasets
   - Demo: Show how Spark on 100 rows is slower than pandas

2. **"Spark replaces pandas"**
   - Reality: pandas is better for datasets that fit in memory
   - Rule of thumb: Use Spark when data > 50% of available RAM

3. **"Lazy evaluation is wasteful"**
   - Reality: It enables the Catalyst optimizer to rewrite queries
   - Demo: Show how `.explain()` reveals filter pushdown

4. **"Shuffles are bad and should always be avoided"**
   - Reality: Some operations (groupBy, join) require shuffles
   - The goal is to minimize unnecessary shuffles and reduce data shuffled

5. **"Caching everything helps performance"**
   - Reality: Caching uses memory that could be used for computation
   - Only cache DataFrames that are reused multiple times

---

## Live Demo Suggestions

1. **Spark UI walkthrough**: Run a groupBy, then walk through Jobs -> Stages -> Tasks
   in the UI. Show the DAG visualization.

2. **Shuffle visualization**: Run a groupBy on an unpartitioned vs. pre-partitioned
   DataFrame. Show the difference in shuffle bytes read/written.

3. **Catalyst optimizer**: Use `.explain(True)` to show Parsed -> Analyzed -> Optimized
   -> Physical plans. Point out filter pushdown.

4. **Partition skew**: Repartition by a skewed column and show uneven partition sizes.

---

## Discussion Prompts

1. "You have 1 TB of variant data and a laptop with 16 GB RAM. What are your options?"
2. "A colleague says Spark is overkill for bioinformatics. How would you respond?"
3. "Why does the bioinformatics community use tools like Hail instead of raw Spark?"
4. "If you could only optimize one thing in a Spark job, what would it be and why?"
5. "How would you decide between Spark, Dask, and just buying a bigger machine?"
6. "What would change if we were processing real-time sequencing data instead of static files?"

---

## Connections to Real-World Bioinformatics

- **GATK**: Uses Spark internally for some tools (e.g., MarkDuplicatesSpark)
- **Hail**: Built on Spark, designed specifically for genetic data analysis
- **ADAM**: Spark-based genomic data processing library
- **gnomAD**: Processed 76K genomes using Hail/Spark on Google Cloud
- **UK Biobank**: 500K samples, analysis often uses distributed computing
- **Nextflow/WDL**: Pipeline orchestrators that can launch Spark jobs as steps

---

## Cloud Computing Connections

- **AWS EMR**: Managed Spark clusters, pay-per-hour
- **GCP Dataproc**: Google's managed Spark, integrates with BigQuery
- **Azure HDInsight**: Microsoft's managed Spark offering
- Key message: The concepts learned locally transfer directly to cloud

---

## Time Allocation

**Session 1 (2 hours)**: Setup + Basics
- 50 min: Lecture (outline above)
- 15 min: Notebook 00 (setup verification)
- 45 min: Notebook 01 (guided, with pauses for discussion)
- 10 min: Wrap-up and preview of exercise

**Session 2 (2.5 hours)**: Main Exercise
- 10 min: Quick review of key concepts
- 2 hours: Notebook 02 (students work independently, instructor circulates)
- 20 min: Group discussion of results and reflection questions

**Homework**: Written exercises (exercises/ folder) + Notebook 03 (optional)
