# PostgreSQL Installation and Setup for SQL Warmups

## Installing PostgreSQL on Mac

### Option 1: Using Homebrew (Recommended)

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install PostgreSQL**:
   ```bash
   brew install postgresql@15
   ```

3. **Start PostgreSQL service**:
   ```bash
   brew services start postgresql@15
   ```

4. **Verify installation**:
   ```bash
   psql --version
   ```

### Option 2: Using Postgres.app

1. Download Postgres.app from https://postgresapp.com/
2. Move the app to your Applications folder
3. Double-click to start PostgreSQL
4. Click "Initialize" to create a new server

## Creating the Database

1. **Access PostgreSQL**:
   ```bash
   psql postgres
   ```

2. **Create a database for the warmups**:
   ```sql
   CREATE DATABASE sequencing_qc;
   ```

3. **Exit psql**:
   ```sql
   \q
   ```

## Python Setup

1. **Install required Python packages**:
   ```bash
   pip install psycopg2-binary sqlalchemy jupyter pandas matplotlib seaborn numpy
   ```

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open** `sql_warmups.ipynb` and run the cells

## Database Connection Details

- **Host**: localhost
- **Port**: 5432 (default)
- **Database**: sequencing_qc
- **User**: Your Mac username (default)
- **Password**: (none by default for local connections)

## Connecting to the Database

```bash
psql -U shanebrubaker -h localhost -p 5432 sequencing_qc
```

## Example psql Commands

Once connected, try these commands to explore the database:

### Meta-commands (no semicolon needed)
```
\dt                  -- list all tables
\d patients          -- describe the patients table structure
\d+ qc_metrics       -- describe qc_metrics with extra detail
\l                   -- list all databases
\q                   -- quit psql
```

### Basic queries
```sql
-- Count rows in each table
SELECT COUNT(*) FROM patients;
SELECT COUNT(*) FROM samples;
SELECT COUNT(*) FROM sequencing_runs;
SELECT COUNT(*) FROM qc_metrics;

-- Preview a table
SELECT * FROM patients LIMIT 5;
SELECT * FROM qc_metrics LIMIT 5;
```

### Join across tables
```sql
-- Patient → sample → run → QC metrics
SELECT p.patient_id, p.diagnosis, s.sample_type, q.mean_coverage, q.pass_filter
FROM patients p
JOIN samples s ON p.patient_id = s.patient_id
JOIN sequencing_runs r ON s.sample_id = r.sample_id
JOIN qc_metrics q ON r.run_id = q.run_id
LIMIT 10;
```

### Aggregation
```sql
-- Average coverage by diagnosis
SELECT p.diagnosis, AVG(q.mean_coverage) AS avg_coverage
FROM patients p
JOIN samples s ON p.patient_id = s.patient_id
JOIN sequencing_runs r ON s.sample_id = r.sample_id
JOIN qc_metrics q ON r.run_id = q.run_id
GROUP BY p.diagnosis
ORDER BY avg_coverage DESC;

-- Pass/fail rate by sequencer platform
SELECT r.platform,
       COUNT(*) AS total_runs,
       SUM(CASE WHEN q.pass_filter THEN 1 ELSE 0 END) AS passed,
       ROUND(100.0 * AVG(q.pass_filter::int), 1) AS pass_pct
FROM sequencing_runs r
JOIN qc_metrics q ON r.run_id = q.run_id
GROUP BY r.platform;
```

### Filtering
```sql
-- Runs that failed QC with low coverage
SELECT r.run_id, q.mean_coverage, q.mean_quality_score, q.error_rate
FROM sequencing_runs r
JOIN qc_metrics q ON r.run_id = q.run_id
WHERE q.pass_filter = FALSE
ORDER BY q.mean_coverage ASC
LIMIT 10;
```

## Troubleshooting

- If you get "connection refused", ensure PostgreSQL is running: `brew services list`
- If port 5432 is in use, you can specify a different port when starting PostgreSQL
- For permission issues, check your PostgreSQL user roles: `psql -l`
