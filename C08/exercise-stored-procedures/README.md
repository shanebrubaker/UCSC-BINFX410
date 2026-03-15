# PostgreSQL Stored Procedures Exercise

This exercise builds on the `sequencing_qc` database created in the **exercise-sql-warmups** exercise.
Make sure you have completed that exercise and the database is available before starting this one.

## Prerequisites

- PostgreSQL installed and running (see `exercise-sql-warmups/README.md`)
- The `sequencing_qc` database populated with data from `sql_warmups.ipynb`
- Python packages installed:
  ```bash
  pip install psycopg2-binary sqlalchemy jupyter pandas
  ```

## What You'll Learn

This exercise covers the core PostgreSQL programmability features:

| # | Topic |
|---|-------|
| 1 | Scalar functions with `CREATE FUNCTION` |
| 2 | Table-valued functions returning `SETOF` / `TABLE` |
| 3 | Functions with `OUT` parameters |
| 4 | `CREATE PROCEDURE` and transaction control |
| 5 | `INOUT` parameters and overloading |
| 6 | Trigger functions and `CREATE TRIGGER` |
| 7 | Exception handling with `RAISE` and `BEGIN…EXCEPTION` |
| 8 | Calling stored routines from Python |

## Running the Exercise

```bash
jupyter notebook stored_procedures.ipynb
```

## Database Connection

Same as the warmups exercise:

- **Host**: localhost
- **Port**: 5432
- **Database**: sequencing_qc
- **User**: Your Mac username
- **Password**: (none for local connections)
