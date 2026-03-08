# Common Issues and Troubleshooting Guide

## Quick Fixes (Top 3 Issues)

1. **Jupyter won't connect**: Wait 30 seconds after `docker-compose up`, then try
   http://localhost:8888/lab. Check `docker-compose logs` for errors.

2. **Out of memory**: Use the `small` dataset first. Increase Docker memory to 4+ GB
   in Docker Desktop > Settings > Resources.

3. **"Only one SparkContext may be running"**: Restart the Jupyter kernel
   (Kernel > Restart) before creating a new SparkSession.

---

## Environment Issues

### Docker not starting

**Symptoms**: `docker-compose up` fails or hangs.

**Causes & Solutions**:
- Docker Desktop not running -> Start Docker Desktop, wait for it to be ready
- Insufficient disk space -> Free up at least 5 GB
- Port conflict -> Check if another service uses 8888 or 4040:
  ```bash
  lsof -i :8888
  lsof -i :4040
  ```
- Docker version too old -> Update to Docker 20.10+
- On Mac: Allocate at least 4 GB memory in Docker Desktop > Settings > Resources

### Spark UI not accessible

**Symptoms**: http://localhost:4040 shows "page not found" or doesn't load.

**Causes & Solutions**:
- No active SparkSession -> Run the SparkSession creation cell first
- Port not exposed -> Verify docker-compose.yml has `"4040:4040"` in ports
- Multiple SparkSessions -> Each gets a different port (4040, 4041, ...).
  Restart kernel to release the port.
- Container networking issue -> Try http://127.0.0.1:4040 instead

### Jupyter not connecting

**Symptoms**: Browser shows connection refused at localhost:8888.

**Causes & Solutions**:
- Container still starting -> Wait 30-60 seconds, check `docker-compose logs`
- Port conflict -> Another Jupyter instance on 8888. Change port in docker-compose.yml:
  ```yaml
  ports:
    - "8889:8888"
  ```
- Firewall blocking -> Check local firewall settings

### Import errors

**Symptoms**: `ModuleNotFoundError: No module named 'pyspark'`

**Causes & Solutions**:
- Running outside Docker -> Ensure you're using the Docker Jupyter, not a local one
- Container needs rebuild -> `docker-compose build --no-cache`
- Wrong Python kernel -> Select "Python 3" kernel in Jupyter

---

## Performance Issues

### Out of memory errors

**Symptoms**: `java.lang.OutOfMemoryError`, kernel dies, or container crashes.

**Causes & Solutions**:
- Dataset too large -> Start with `small`, then `medium`. Only use `large` for
  Task 6 performance comparison.
- Too many cached DataFrames -> Unpersist DataFrames you no longer need:
  ```python
  df.unpersist()
  ```
- Increase driver memory in SparkSession config:
  ```python
  .config("spark.driver.memory", "3g")
  ```
- Increase Docker container memory limit in docker-compose.yml:
  ```yaml
  mem_limit: 6g
  ```
- Reduce shuffle partitions:
  ```python
  .config("spark.sql.shuffle.partitions", "4")
  ```

### Operations very slow

**Symptoms**: Queries take minutes when they should take seconds.

**Causes & Solutions**:
- Too many partitions -> Check with `df.rdd.getNumPartitions()`. For local mode,
  use 2-8 partitions:
  ```python
  .config("spark.sql.shuffle.partitions", "8")
  ```
- Data skew -> One partition has much more data than others. Repartition:
  ```python
  df = df.repartition(8)
  ```
- Using UDFs instead of built-in functions -> Replace with Spark SQL functions
- Unnecessary shuffles -> Filter before groupBy/join
- Not caching reused DataFrames -> Add `.cache()` for DataFrames used multiple times
- Collect on large DataFrames -> Use `.show()` instead of `.collect()` for display

### Spark session hangs

**Symptoms**: Cell execution never completes, spinning indicator.

**Causes & Solutions**:
- Garbage collection overhead -> Restart kernel, reduce data size
- Too many cached DataFrames -> Run `spark.catalog.clearCache()`
- Deadlock (rare) -> Restart the Docker container:
  ```bash
  docker-compose restart
  ```

---

## Data Issues

### Data not loading

**Symptoms**: `FileNotFoundException` or `AnalysisException: Path does not exist`.

**Causes & Solutions**:
- Wrong path -> Inside Docker, data is at `/workspace/data/`. Check:
  ```python
  import os
  print(os.listdir("/workspace/data"))
  ```
- Volume mount issue -> Verify docker-compose.yml has `../data:/workspace/data`
- Data not generated -> Run `generate_data.py` first:
  ```python
  !python /workspace/data/generate_data.py --size medium --output-dir /workspace/data
  ```
- Running outside Docker -> Adjust paths to your local directory

### Schema inference wrong

**Symptoms**: All columns are strings, or numeric columns have wrong types.

**Causes & Solutions**:
- Missing `inferSchema=True` -> Add it to `spark.read.csv()`:
  ```python
  spark.read.csv(path, header=True, inferSchema=True)
  ```
- For BED files, define schema explicitly:
  ```python
  schema = "chrom STRING, start INT, end INT, gene_name STRING, strand STRING"
  spark.read.csv(path, sep="\t", header=False, schema=schema)
  ```

### Join produces too many rows

**Symptoms**: Join result has millions more rows than expected.

**Causes & Solutions**:
- Cartesian product -> Missing or incomplete join condition. Ensure all conditions
  are specified:
  ```python
  # Wrong: only chromosome match (many-to-many)
  variants.join(genes, variants.CHROM == genes.chrom)

  # Right: chromosome + position range
  variants.join(genes,
      (variants.CHROM == genes.chrom) &
      (variants.POS >= genes.start) &
      (variants.POS <= genes.end))
  ```
- Duplicate keys in one table -> Check for duplicates:
  ```python
  genes.groupBy("chrom", "start", "end").count().filter(F.col("count") > 1).show()
  ```

---

## Notebook Issues

### Kernel dies during large operations

**Symptoms**: "Kernel Restarting" message appears unexpectedly.

**Causes & Solutions**:
- Container out of memory -> Increase `mem_limit` in docker-compose.yml
- Use smaller dataset -> Switch to `variants_small.csv` for testing
- Close other notebooks -> Each open notebook may have a SparkSession consuming memory

### Cannot create SparkSession

**Symptoms**: Error about existing SparkContext.

**Causes & Solutions**:
- Only one SparkSession per notebook -> Restart kernel (Kernel > Restart)
- Previous notebook didn't clean up -> Run `spark.stop()` in the other notebook
- Use `getOrCreate()` instead of `create()`:
  ```python
  spark = SparkSession.builder.appName("...").getOrCreate()
  ```

### Plots not displaying

**Symptoms**: Matplotlib plots don't appear in notebook output.

**Causes & Solutions**:
- Add magic command at top of notebook:
  ```python
  %matplotlib inline
  ```
- Backend issue -> Set backend before importing pyplot:
  ```python
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  ```
- Use `plt.show()` after creating plots
