# Exercise 3: Spark Query Optimization

## Introduction

In big data, the difference between an optimized and unoptimized query can be hours
vs. minutes. Spark's Catalyst optimizer handles many optimizations automatically, but
understanding optimization principles helps you write better queries and troubleshoot
performance issues.

---

## Part A: Filter Pushdown (3 questions)

**Q1.** Compare these two approaches for finding rare variants in genes:

```python
# Approach A: Join first, then filter
result_a = variants.join(genes, ...).filter(F.col("AF") < 0.01)

# Approach B: Filter first, then join
result_b = variants.filter(F.col("AF") < 0.01).join(genes, ...)
```

Measure the execution time of both. Which is faster and by how much?

**Answer:**


**Q2.** Use `.explain()` on both approaches. Does Spark's Catalyst optimizer
automatically push the filter before the join in Approach A? Look for "PushedFilters"
in the physical plan.

**Answer:**


**Q3.** When might filter pushdown NOT help performance? Give an example scenario.

**Answer:**


---

## Part B: Broadcasting (3 questions)

**Q4.** Compare a regular join vs. a broadcast join when joining variants with genes:

```python
# Regular join
result_regular = variants.join(genes, ...)

# Broadcast join
result_broadcast = variants.join(F.broadcast(genes), ...)
```

Measure both. What is the speedup from broadcasting?

**Answer:**


**Q5.** What is the default size limit for automatic broadcasting in Spark?
How would you change it? When should you NOT use broadcast joins?

```python
# Hint: spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
```

**Answer:**


**Q6.** If both tables in a join are very large (e.g., 10 GB each), what join
strategy should Spark use? Explain why broadcasting wouldn't work.

**Answer:**


---

## Part C: Caching Strategy (3 questions)

**Q7.** You need to run 5 different aggregations on the filtered variants DataFrame.
Should you cache it? Measure the total time with and without caching.

**Answer:**


**Q8.** What is the difference between `MEMORY_ONLY` and `MEMORY_AND_DISK` storage
levels? When would you use each?

```python
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)
```

**Answer:**


**Q9.** You cached a large DataFrame but your subsequent operations are slower than
expected. What might be going wrong? List at least 3 possible causes.

**Answer:**


---

## Part D: File Format Optimization (3 questions)

**Q10.** Load the same data from CSV and from Parquet (after saving it). Compare the
time to execute `select("CHROM", "POS").filter(F.col("CHROM") == "chr1").count()`
on both. What is the speedup from Parquet?

**Answer:**


**Q11.** Explain **column pruning** and **predicate pushdown** in the context of Parquet.
How do they reduce I/O?

**Answer:**


**Q12.** You need to repeatedly query variants by chromosome. Should you save your
Parquet file partitioned by CHROM? What are the trade-offs?

```python
df.write.partitionBy("CHROM").parquet("variants_by_chrom.parquet")
```

**Answer:**


---

## Part E: Reflection (3 questions)

**Q13.** Design a Spark pipeline to analyze variant data from 100,000 whole genomes
(~500 TB of data). Describe your partitioning strategy, file format, join approach,
and what you would cache. What hardware would you need?

**Answer:**


**Q14.** When is optimization NOT worth the effort? Give an example of a scenario
where spending time optimizing a Spark query would be a waste.

**Answer:**


**Q15.** What are the trade-offs between memory usage and compute time in Spark?
How does caching exemplify this trade-off?

**Answer:**

