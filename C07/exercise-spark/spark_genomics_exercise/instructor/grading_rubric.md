# Grading Rubric: Spark Variant QC Exercise

**Total: 100 points**

---

## Code Functionality (40 points)

| Task | Points | Criteria |
|------|--------|----------|
| Task 1: Load and Explore | 5 | Correct CSV loading with header/inferSchema, schema displayed, count correct |
| Task 2: Quality Filtering | 5 | All three filters applied correctly, percentage calculation accurate |
| Task 3: Per-Chromosome Stats | 5 | Correct groupBy with all 5 metrics, results displayed |
| Task 4: AF Stratification | 5 | Correct frequency bins, frequency_class column created properly |
| Task 5: Gene Join | 5 | Correct positional join logic (CHROM + position range), top genes identified |
| Task 6: Performance Comparison | 5 | Both Spark and pandas approaches implemented, valid timing comparison |
| Task 7: Optimization | 5 | At least 2 optimization strategies attempted, measurable improvement |
| Task 8: Write Results | 5 | Parquet written successfully, file size comparison calculated |

**Deductions:**
- -1 per task for code that runs but produces incorrect results
- -2 per task for code that errors (partial credit for approach)
- -0 for minor style issues (not the focus of this exercise)

---

## Understanding (30 points)

### Spark UI Questions (10 points)
| Question | Points | Full Credit |
|----------|--------|-------------|
| Task 2: Did filtering cause a shuffle? | 2 | Correctly identifies filter as narrow transformation (no shuffle) |
| Task 3: How many stages in groupBy? | 3 | Identifies 2 stages (partial agg + final agg), explains why |
| Task 5: Did the join shuffle? | 2 | Identifies shuffle in join, mentions SortMergeJoin |
| Task 7: BroadcastHashJoin explanation | 3 | Explains that broadcast sends small table to all workers |

### Reflection Questions (10 points)
| Question | Points | Full Credit |
|----------|--------|-------------|
| When to use Spark vs pandas | 3 | Mentions data size exceeding memory, distributed processing needs |
| Costs of distributed computing | 2 | Lists at least 3: serialization, shuffles, overhead, complexity |
| Design for 100K genomes | 3 | Mentions partitioning, file format, cluster sizing, join strategy |
| Spark UI learnings | 2 | Specific observations from their own query execution |

### Optimization Explanations (10 points)
| Question | Points | Full Credit |
|----------|--------|-------------|
| Which optimization helped most | 4 | Identifies filter-before-join or broadcasting, explains mechanism |
| Filter order effect | 3 | Explains Catalyst optimizer may reorder, or data reduction matters |
| Parquet advantages | 3 | Lists at least 3: columnar, compression, predicate pushdown, schema |

---

## Optimization Results (20 points)

| Criterion | Points | Full Credit |
|-----------|--------|-------------|
| Measurable improvement achieved | 10 | Optimized query is at least 1.5x faster than unoptimized |
| Multiple strategies attempted | 5 | At least 2 of: filter pushdown, broadcast join, caching, column pruning |
| explain() analysis included | 5 | Shows and discusses physical plan before and after optimization |

**Partial credit:**
- 5 points if only one optimization attempted but well-explained
- 3 points if optimization attempted but no measurable improvement

---

## Code Quality (10 points)

| Criterion | Points | Full Credit |
|-----------|--------|-------------|
| Comments and readability | 5 | Key operations commented, code is easy to follow |
| Variable naming | 3 | Descriptive names (not single letters), consistent style |
| Organization | 2 | Logical flow, no unnecessary code duplication |

---

## Grade Scale

| Grade | Points | Description |
|-------|--------|-------------|
| A | 90-100 | Excellent understanding, all tasks complete, thoughtful analysis |
| B | 80-89 | Good understanding, most tasks complete, minor gaps |
| C | 70-79 | Adequate understanding, core tasks complete, some conceptual gaps |
| D | 60-69 | Basic completion, significant conceptual gaps |
| F | < 60 | Incomplete or major errors |

---

## Quick Assessment Checklist

For rapid grading, check these key indicators:

- [ ] Notebook runs end-to-end without errors
- [ ] Filtering results are reasonable (~20-30% filtered out)
- [ ] Per-chromosome stats show chr1 with most variants
- [ ] Gene join identifies intergenic vs genic variants
- [ ] Performance comparison shows pandas faster on small data
- [ ] At least one optimization shows measurable speedup
- [ ] Reflection answers demonstrate conceptual understanding
- [ ] Parquet file sizes are smaller than CSV
