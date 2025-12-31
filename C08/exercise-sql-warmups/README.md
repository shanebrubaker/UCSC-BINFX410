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

## Troubleshooting

- If you get "connection refused", ensure PostgreSQL is running: `brew services list`
- If port 5432 is in use, you can specify a different port when starting PostgreSQL
- For permission issues, check your PostgreSQL user roles: `psql -l`
