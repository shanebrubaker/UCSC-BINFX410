# Exercise 1: Variant Filtering with Spark

## Introduction

Quality filtering is a critical step in variant analysis. Poor-quality variant calls
can lead to false discoveries. The GATK Best Practices recommend "hard filtering" when
machine learning-based filtering (VQSR) isn't feasible. In this exercise, you'll apply
filters and analyze their impact using Spark.

---

## Part A: Basic Filtering (5 questions)

**Q1.** Apply the filter `QUAL >= 30` to the variants DataFrame. How many variants pass?
What percentage are removed?

**Answer:**


**Q2.** Apply the filter `DP >= 10`. How many variants pass? Compare with Q1.

**Answer:**


**Q3.** Apply the filter `GQ >= 20`. How many variants pass?

**Answer:**


**Q4.** Which single filter removes the most variants? Why do you think this is the case
biologically?

**Answer:**


**Q5.** Apply all three filters simultaneously. Is the number of variants that pass all
three equal to the minimum of the individual filter counts? Why or why not?

**Answer:**


---

## Part B: Compound Filtering (3 questions)

**Q6.** Write a Spark query that applies filters in this order: QUAL first, then DP,
then GQ. Count the remaining variants after each step. Does the order of filtering
affect the final result?

**Answer:**


**Q7.** Does the order of `filter()` calls affect Spark performance? Use `.explain()`
on two different orderings and compare the physical plans.

**Answer:**


**Q8.** Create a new column `filter_status` that categorizes each variant as:
- "HIGH_QUALITY": passes all three filters
- "MEDIUM_QUALITY": passes QUAL >= 30 but fails DP or GQ
- "LOW_QUALITY": fails QUAL >= 30

How many variants fall into each category?

**Answer:**


---

## Part C: Spark Concepts (4 questions)

**Q9.** Did any of the filtering operations above cause a shuffle? Why or why not?
(Hint: check the Spark UI or use `.explain()`)

**Answer:**


**Q10.** What is the difference between `filter()` and `where()` in Spark?

**Answer:**


**Q11.** Explain the difference between **narrow** and **wide** transformations.
Which category does `filter()` belong to?

**Answer:**


**Q12.** If you had 1 billion variants partitioned across 100 partitions, and you
applied a filter that removes 90% of variants, what would happen to the partition
sizes? Would you want to repartition afterward? Why?

**Answer:**

