# Exercise 2: Genomic Data Aggregation

## Introduction

Aggregation is essential for summarizing large genomic datasets. Per-chromosome statistics,
population-level summaries, and cross-tabulations help researchers identify patterns and
anomalies. In this exercise, you'll practice aggregation operations and analyze their
impact on Spark's execution.

---

## Part A: GroupBy Exercises (5 questions)

**Q1.** Calculate the total variant count per chromosome. Which chromosome has the most
variants? Does this match your expectations based on chromosome size?

**Answer:**


**Q2.** Calculate the mean quality score (`QUAL`) per chromosome. Are quality scores
uniform across chromosomes, or do you see variation?

**Answer:**


**Q3.** Count variants by consequence type (`CONSEQUENCE`). What is the most common
consequence? Does the distribution match expectations for whole-genome data?

**Answer:**


**Q4.** Create a cross-tabulation of `FILTER` status by `CONSEQUENCE` type. Are
low-quality variants enriched in any particular consequence category?

**Answer:**


**Q5.** For each chromosome, calculate the ratio of missense to synonymous variants.
Which chromosome has the highest ratio? What might this indicate?

**Answer:**


---

## Part B: Window Functions (3 questions)

**Q6.** Use a window function to rank variants by quality score within each chromosome.
Show the top 5 highest-quality variants per chromosome for chr1, chr2, and chr3.

```python
# Hint:
window = Window.partitionBy("CHROM").orderBy(F.col("QUAL").desc())
# Use F.row_number().over(window)
```

**Answer:**


**Q7.** Calculate a running count of variants by position within each chromosome.
For chr1, at what position have you seen 50% of all chr1 variants?

**Answer:**


**Q8.** Find the top 3 genes with the most variants per chromosome using a window
function (not just a simple groupBy). Show the result for chr1-chr5.

```python
# Hint: First group by CHROM and gene_name to get counts,
# then use a window function to rank within each CHROM.
```

**Answer:**


---

## Part C: Shuffle Analysis (4 questions)

**Q9.** Which of the aggregation operations above caused shuffles? List them and
explain why each required (or didn't require) a shuffle.

**Answer:**


**Q10.** Look at the Spark UI for a `groupBy("CHROM").count()` operation. How many
stages does it have? What does each stage do?

**Answer:**


**Q11.** If you repartitioned the DataFrame by `CHROM` before running
`groupBy("CHROM").count()`, would it still require a shuffle? Test this and explain.

**Answer:**


**Q12.** What is **data skew** and how would it affect chromosome-level aggregations?
Which chromosomes might cause skew in our dataset and why?

**Answer:**

