# Exercise: SQLite for Bioinformatics

A hands-on introduction to SQLite using a gene expression dataset. Work through the notebooks in order.

---

## Setup

SQLite ships with Python — no extra install needed. You will need `pandas` for display:

```bash
pip install pandas jupyter
```

Then launch Jupyter:

```bash
jupyter notebook
```

---

## Notebooks

| Notebook | Topics |
|---|---|
| `01_setup.ipynb` | Connect, create schema, insert data, inspect tables |
| `02_basic_queries.ipynb` | SELECT, WHERE, ORDER BY, LIMIT, LIKE, IN, BETWEEN, GROUP BY |
| `03_joins_aggregations.ipynb` | INNER JOIN, LEFT JOIN, AVG/MAX/MIN/SUM, HAVING, CASE |
| `04_indexes.ipynb` | CREATE INDEX, EXPLAIN QUERY PLAN, performance timing |
| `05_transactions.ipynb` | INSERT, UPDATE, DELETE, BEGIN/COMMIT/ROLLBACK |
| `06_export.ipynb` | Export to CSV/TSV, pivot tables, round-trip verification |

---

## Dataset

The notebooks use a synthetic cancer genomics dataset:

- **15 genes** — well-known oncogenes and tumor suppressors (BRCA1, TP53, KRAS, etc.)
- **10 samples** — breast, lung, and skin tissue; tumor and normal conditions
- **150 expression rows** — TPM and raw count values for every gene/sample pair

---

## Key SQL Concepts Covered

- Schema design with primary keys, foreign keys, and constraints
- All major DML statements: SELECT, INSERT, UPDATE, DELETE
- Filtering: WHERE, LIKE, IN, BETWEEN
- Aggregation: GROUP BY, HAVING, COUNT, AVG, MAX, MIN, SUM
- Joins: INNER JOIN, LEFT JOIN across multiple tables
- Performance: indexes, EXPLAIN QUERY PLAN
- Transactions: BEGIN, COMMIT, ROLLBACK
- Data export: CSV, TSV, pivot tables
