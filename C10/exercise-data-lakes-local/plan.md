# Study Plan — BINFX410 Chapter 10

Work through the notebooks in order. Each session builds on the previous one.

---

## Session 1 — Notebook 01: Introduction and Setup (~45 min)

**Goal:** Understand the three architectures at a conceptual level and generate the genomics dataset.

1. Read all markdown cells in `01_introduction_and_setup.ipynb` before running any code.
2. Study the comparison table (cell 4) — know which row maps to which architecture.
3. Run cells 7–12 to generate `./raw_data/` (samples, genes, variant_calls, variants).
4. Run the dataset summary cells and note the row counts, file sizes, and distributions.

**Exercises to complete:**
- [ ] **1.1** — Answer all four DataFrame questions (avg depth, unique PASS samples, non-PASS fraction, top biotype by quality)
- [ ] **1.2** — Sketch a star schema for this genomics dataset (fact table grain = variant)
- [ ] **1.3** — Justify architecture choices for the hospital reporting, sequencing lab, and biotech scenarios

**Check:** `./raw_data/` contains four CSV files before moving on.

---

## Session 2 — Notebook 02: Data Lake (~60 min)

**Goal:** Build a medallion lake, understand Parquet, and see schema drift in action.

1. Read section 1 carefully — schema-on-read vs schema-on-write is a key exam concept.
2. Run Bronze ingestion cells. Compare CSV vs Parquet sizes in the output.
3. Step through Silver cleaning cells — note every transformation applied and why.
4. Build Gold aggregates. Understand what each Gold table pre-computes.
5. Run DuckDB Parquet queries — observe that no import step is needed.
6. Run schema drift cells (section 7) and read the "What went wrong?" explanation.
7. Run data swamp cells (section 8) and study the minimal catalog pattern.

**Exercises to complete:**
- [ ] **2.1** — Top 3 tissue types by PASS variant count (3-table JOIN in DuckDB)
- [ ] **2.2** — Build `consequence_monthly_counts.parquet` Gold table
- [ ] **2.3a** — Implement `check_schema_compatibility(path1, path2)`
- [ ] **2.3b** — Write a short design note: how would a pipeline detect schema changes automatically?

**Check:** `./data_lake/bronze/`, `./data_lake/silver/`, `./data_lake/gold/` all contain `.parquet` files.

---

## Session 3 — Notebook 03: Data Warehouse (~75 min)

**Goal:** Build a star schema, load it via ETL, and run OLAP queries with window functions.

1. Read section 1 (OLTP vs OLAP, star schema diagram) — know the surrogate key pattern.
2. Run the `dim_date` creation cell — study how it populates date rows with pre-computed attributes.
3. Step through each dimension table creation, noting the `ROW_NUMBER() OVER` surrogate key pattern.
4. Step through fact table creation — understand why `fact_variants` has pre-computed `signal_strength`.
5. Run all 5 analytical queries and trace which tables each joins.
6. Run `EXPLAIN` and `EXPLAIN ANALYZE` — note the operator names (Hash Join, Projection, etc.).

**Exercises to complete:**
- [ ] **3.1** — Cohort analysis (first-call date per sample, months since first call)
- [ ] **3.2** — Running total + 3-month moving average using `SUM() OVER` with ROWS frame
- [ ] **3.3** — Gene co-occurrence: self-join `fact_variants` on `call_id`, deduplicate pairs with `<`

**Check:** `./warehouse/genomics.duckdb` exists and contains 5 tables.

---

## Session 4 — Notebook 04: Lakehouse (~75 min)

**Goal:** Understand the Delta Lake transaction log, ACID semantics, time travel, and MERGE.

1. Read section 1 carefully — the transaction log diagram is the core mental model.
2. Run the write cells and inspect the `_delta_log/` JSON files manually (section 4).
3. Run the schema enforcement cells — observe which writes are blocked and why.
4. Run schema evolution with `schema_mode='merge'` — compare to schema drift in Notebook 02.
5. Run time travel cells — compare row counts at version 0 vs current.
6. Study the MERGE cell structure: `when_matched_update_all()` + `when_not_matched_insert_all()`.
7. Run OPTIMIZE and VACUUM — note the trade-off between time travel history and storage cleanup.

**Exercises to complete:**
- [ ] **4.1a** — Row counts at each version (0 through current)
- [ ] **4.1b** — Find samples whose `tissue_type` changed between v0 and current
- [ ] **4.2** — MERGE that marks old LowQuality calls (before 2022) as Reanalysis_Needed
- [ ] **4.3** — Cohort retention table via DuckDB `delta_scan()`

**Check:** `./lakehouse/` contains four subdirectories, each with a `_delta_log/` folder.

---

## Session 5 — Notebook 05: Comparison and Capstone (~90 min)

**Goal:** Consolidate understanding through benchmarking, a decision framework, and real genomics questions.

1. Read the feature comparison table (section 1) — reconcile each row against what you built.
2. Run the benchmark (section 2) — understand why the same query has different timing profiles.
3. Study the decision framework flowchart (section 4) — practice applying it to new scenarios.
4. For each of the 5 Dr. Kim questions, write the query yourself before looking at any provided solution.
5. Generate all charts. The charts are part of the "deliverable" for the capstone.

**Exercises to complete:**
- [ ] **5.1** — Highest-confidence somatic calls: scatter plot af_tumor vs quality_score + interpretation
- [ ] **5.2** — Scale test: re-generate dataset at 10x size, re-run benchmarks, explain differences
- [ ] **5.3** — Clinical genomics architecture design: warehouse vs lakehouse for dual clinical+research use

---

## Reference: Directory Structure After All Notebooks

```
exercise-data-lakes-local/
+-- raw_data/                        # generated by notebook 01
|   +-- samples.csv
|   +-- genes.csv
|   +-- variant_calls.csv
|   +-- variants.csv
+-- data_lake/                       # generated by notebook 02
|   +-- bronze/   *.parquet          # raw CSV -> Parquet (no transformation)
|   +-- silver/   *.parquet          # cleaned, typed, derived columns
|   +-- gold/     *.parquet          # pre-aggregated analytics tables
|   +-- demo_drift/                  # schema drift demonstration files
|   +-- catalog.json                 # minimal data catalog example
+-- warehouse/                       # generated by notebook 03
|   +-- genomics.duckdb              # DuckDB file with star schema
+-- lakehouse/                       # generated by notebook 04
|   +-- samples/      + _delta_log/
|   +-- genes/        + _delta_log/
|   +-- variant_calls/ + _delta_log/
|   +-- variants/     + _delta_log/
+-- figures/                         # generated by notebook 05
|   +-- benchmark_bar.png
|   +-- benchmark_boxplot.png
|   +-- capstone_q*.png
+-- requirements.txt
+-- README.md
+-- plan.md
```

---

## Key Concepts to Know

| Concept | Where covered | One-line summary |
|---------|--------------|-----------------|
| Schema-on-write vs schema-on-read | NB 01, 02 | Write = validated at ingest; Read = validated at query time |
| Medallion architecture | NB 02 | Bronze (raw) -> Silver (clean) -> Gold (aggregated) |
| Parquet columnar format | NB 02 | Stores columns together; enables predicate pushdown |
| Star schema | NB 03 | Fact tables (events/measures) + dimension tables (context) |
| Surrogate key | NB 03 | Warehouse-internal PK that decouples from source system IDs |
| OLAP vs OLTP | NB 03 | Analytics (few large scans) vs transactions (many small reads/writes) |
| signal_strength | NB 02-05 | depth * af_tumor — proxy for evidence strength of a variant call |
| Delta transaction log | NB 04 | JSON log of add/remove file actions = ACID on object storage |
| Time travel | NB 04 | Read table at any prior version via `DeltaTable(path, version=N)` |
| MERGE (upsert) | NB 04 | Update existing rows + insert new rows in a single atomic operation |
| OPTIMIZE / VACUUM | NB 04 | Compact small files / delete old unreferenced files |
| Data swamp | NB 02 | Lake without governance: undiscoverable, undocumented files |
| Schema drift | NB 02, 04 | Upstream schema changes that break downstream consumers |
| Mutation burden | NB 03-05 | Total PASS variants per sample — key clinical genomics metric |
