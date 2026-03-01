# NextFlow Pipeline Monitoring Guide

This guide explains how to monitor the execution of the Sequence QC Pipeline.

## Table of Contents
- [Real-time Console Monitoring](#real-time-console-monitoring)
- [Built-in HTML Reports](#built-in-html-reports)
- [NextFlow Tower (Seqera Platform)](#nextflow-tower-seqera-platform)
- [Command-line Monitoring](#command-line-monitoring)
- [Log Files](#log-files)

---

## Real-time Console Monitoring

When you run the pipeline, NextFlow displays real-time progress in the terminal:

```bash
nextflow run main.nf
```

### Console Output Explanation

```
executor >  local (33)
[d2/92cb64] SPLIT_REFERENCE (Splitting reference) | 1 of 1 ✔
[90/a019c4] ALIGN_TO_CHROMOSOME (chr1)             | 10 of 10 ✔
[5c/bde9df] QC_CHROMOSOME (chr1)                   | 10 of 10 ✔
[6a/ed95ae] TRIM_AND_CLEAN (chr1)                  | 10 of 10 ✔
[b4/fe666b] COMBINE_CLEANED (Combining...)         | 1 of 1 ✔
[c9/b66241] GENERATE_QC_REPORT (Final QC Report)   | 1 of 1 ✔
[82/bdfa39] CHROMOSOME_QC_SUMMARY (Chromosome...)  | 1 of 1 ✔
```

**Key Elements:**
- `executor >  local (33)`: Shows 33 total tasks executed locally
- `[d2/92cb64]`: Work directory hash (first 2 chars of subdirectory)
- `| 10 of 10`: Progress indicator (completed/total)
- `✔`: Green checkmark indicates successful completion

---

## Built-in HTML Reports

NextFlow automatically generates three HTML reports in `results/trace/`:

### 1. Execution Timeline (`execution_timeline.html`)

**View it:**
```bash
# macOS
open results/trace/execution_timeline.html

# Linux
xdg-open results/trace/execution_timeline.html

# Windows
start results/trace/execution_timeline.html
```

**What it shows:**
- Visual timeline of when each task started and completed
- Parallel execution visualization
- Task duration bars
- Process dependencies
- Helps identify bottlenecks

**Example Timeline:**
```
Split Reference    [▓]
Align chr1         [     ▓▓▓]
Align chr2          [    ▓▓▓]
Align chr3           [   ▓▓▓]
...
```

### 2. Execution Report (`execution_report.html`)

**View it:**
```bash
open results/trace/execution_report.html
```

**What it shows:**
- Resource usage statistics (CPU, memory)
- Task duration statistics
- Success/failure rates
- Per-process resource consumption
- Execution time breakdown

**Includes:**
- Summary metrics
- Top 20 longest-running processes
- CPU and memory efficiency
- Task status distribution

### 3. Pipeline DAG (`pipeline_dag.html`)

**View it:**
```bash
open results/trace/pipeline_dag.html
```

**What it shows:**
- Directed Acyclic Graph (DAG) of workflow
- Visual representation of process dependencies
- Data flow between processes
- Helps understand workflow structure

---

## NextFlow Tower (Seqera Platform)

**NextFlow Tower** (now Seqera Platform) is the official cloud-based monitoring solution.

### Features

- **Real-time monitoring dashboard**
- **Pipeline execution tracking**
- **Resource monitoring**
- **Collaborative workflow management**
- **Historical run comparisons**
- **Cloud and HPC integration**

### Quick Start with Tower

#### Option 1: Free Cloud Service

1. **Sign up** at https://cloud.seqera.io

2. **Get your access token:**
   - Log in to Seqera Platform
   - Go to "Your Profile" > "Your Tokens"
   - Create new token

3. **Configure NextFlow:**
   ```bash
   export TOWER_ACCESS_TOKEN=<your-token>
   export NXF_VER=23.10.0  # Use compatible version
   ```

4. **Run with Tower:**
   ```bash
   nextflow run main.nf -with-tower
   ```

5. **View dashboard:**
   - Go to https://cloud.seqera.io
   - View your running/completed workflows
   - Real-time updates!

#### Option 2: Self-hosted Tower

For institutional deployments:
```bash
# Requires Docker
docker run -d \
  -p 8080:8080 \
  -e TOWER_ROOT_USERS=<your-email> \
  cr.seqera.io/private/nf-tower:latest
```

Visit: http://localhost:8080

### Tower Dashboard Features

**When monitoring with Tower, you'll see:**

1. **Workflow List**: All your pipeline runs
2. **Run Details**:
   - Real-time progress
   - Resource usage graphs
   - Task-level details
   - Error messages
3. **Metrics**:
   - CPU utilization
   - Memory usage
   - Task duration
   - Success/failure rates
4. **Logs**: Access to all task logs
5. **Reports**: Downloadable execution reports

### Tower Configuration in nextflow.config

You can configure Tower in your config file:

```groovy
tower {
    accessToken = '<your-token>'
    enabled = true
    endpoint = 'https://cloud.seqera.io'  // or your custom endpoint
    workspaceId = '<your-workspace-id>'
}
```

---

## Command-line Monitoring

### View Execution History

```bash
# List all pipeline runs
nextflow log

# Output:
# TIMESTAMP          DURATION  RUN NAME        STATUS  ...
# 2025-12-28 14:44  1m 32s    distraught_raman OK     ...
# 2025-12-28 14:33  15s       silly_marconi    OK     ...
```

### View Specific Run Details

```bash
# View all tasks from a specific run
nextflow log distraught_raman

# View specific fields
nextflow log distraught_raman -f 'name,status,duration,realtime'

# View only failed tasks
nextflow log distraught_raman -f 'name,status,exit' -F 'status == "FAILED"'

# View task resource usage
nextflow log distraught_raman -f 'name,peak_rss,peak_vmem,%cpu'
```

### Monitor Active Run

While pipeline is running, monitor in another terminal:

```bash
# Watch log file in real-time
tail -f .nextflow.log

# Count completed tasks
watch -n 1 'nextflow log | tail -1'
```

---

## Log Files

### NextFlow Logs

**Location**: `.nextflow.log` (current directory)

**View it:**
```bash
# View entire log
cat .nextflow.log

# View last 100 lines
tail -100 .nextflow.log

# Search for errors
grep ERROR .nextflow.log

# Follow in real-time
tail -f .nextflow.log
```

### Execution Trace

**Location**: `results/trace/execution_trace.txt`

**Format**: Tab-separated values with detailed task information

**View it:**
```bash
# View as table (requires column command)
column -t -s $'\t' results/trace/execution_trace.txt | less -S

# View specific columns
cut -f1,2,3,9 results/trace/execution_trace.txt

# Fields include:
# - task_id, hash, native_id, process, tag, name
# - status, exit, duration, realtime
# - cpus, memory, disk
# - %cpu, %mem, rss, vmem
```

### Task Work Directories

Each task has a work directory: `work/<hash>/<hash>/`

**Inspect a task:**
```bash
# Find work directory for a specific task
nextflow log distraught_raman -f 'hash,name' | grep "ALIGN_TO_CHROMOSOME (chr1)"

# Example output: d2/92cb64  ALIGN_TO_CHROMOSOME (chr1)

# Navigate to work directory
cd work/d2/92cb64*

# View files
ls -la

# Key files:
# .command.sh   - The actual script that was run
# .command.out  - Standard output
# .command.err  - Standard error
# .command.log  - Combined output
# .exitcode     - Exit code (0 = success)
```

**Inspect task output:**
```bash
# View what the task printed
cat .command.out

# View errors
cat .command.err

# Check exit code
cat .exitcode
```

---

## Monitoring Best Practices

### 1. Use Descriptive Run Names

```bash
# Custom run name for easy identification
nextflow run main.nf -name my-qc-run-batch1
```

### 2. Enable All Reporting

Already configured in `nextflow.config`:
```groovy
trace.enabled = true
timeline.enabled = true
report.enabled = true
dag.enabled = true
```

### 3. Set Up Tower for Production

For production workflows, use Tower to:
- Monitor long-running jobs
- Get email notifications
- Track resource usage
- Debug failures quickly

### 4. Preserve Logs

```bash
# Archive logs after each run
mkdir -p logs/$(date +%Y%m%d_%H%M%S)
cp -r results/trace .nextflow.log logs/$(date +%Y%m%d_%H%M%S)/
```

### 5. Monitor Resource Usage

Check if tasks need more resources:
```bash
# View peak memory usage per process
nextflow log -f 'process,peak_rss' | sort | uniq
```

---

## Quick Reference Commands

```bash
# Run with monitoring
nextflow run main.nf -with-tower

# View real-time log
tail -f .nextflow.log

# List all runs
nextflow log

# View last run details
nextflow log $(nextflow log | tail -1 | awk '{print $3}')

# Open timeline report
open results/trace/execution_timeline.html

# Check for failed tasks
nextflow log -f status -F 'status == "FAILED"'

# View resource usage
nextflow log -f 'name,peak_rss,%cpu'

# Clean up old runs (be careful!)
nextflow clean -f

# View help
nextflow log -h
```

---

## Troubleshooting Monitoring

### Problem: Reports not generated

**Solution:**
Check `nextflow.config` has:
```groovy
report.enabled = true
timeline.enabled = true
```

### Problem: Tower connection fails

**Solution:**
```bash
# Check token
echo $TOWER_ACCESS_TOKEN

# Test connection
curl -H "Authorization: Bearer $TOWER_ACCESS_TOKEN" \
  https://cloud.seqera.io/api/user-info
```

### Problem: Can't find work directory

**Solution:**
```bash
# Use nextflow log to find exact path
nextflow log -f 'hash,name' | grep <task-name>

# Then navigate
cd work/<first-2-chars>/<full-hash>*/
```

---

## Dashboard URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Seqera Cloud | https://cloud.seqera.io | Official Tower cloud service |
| Seqera Documentation | https://docs.seqera.io | Tower documentation |
| NextFlow Documentation | https://nextflow.io/docs/latest | NextFlow docs |
| Local Timeline | `file://$(pwd)/results/trace/execution_timeline.html` | Local timeline |
| Local Report | `file://$(pwd)/results/trace/execution_report.html` | Local report |
| Local DAG | `file://$(pwd)/results/trace/pipeline_dag.html` | Local DAG |

---

## Example: Full Monitoring Session

```bash
# 1. Run pipeline with Tower
export TOWER_ACCESS_TOKEN=<your-token>
nextflow run main.nf -with-tower -name seq-qc-500k-reads

# 2. In another terminal, monitor log
tail -f .nextflow.log

# 3. After completion, view reports
open results/trace/execution_timeline.html
open results/trace/execution_report.html

# 4. Check summary
nextflow log seq-qc-500k-reads -f 'name,status,duration'

# 5. View in Tower dashboard
# Go to https://cloud.seqera.io and click on "seq-qc-500k-reads"
```

---

## Additional Resources

- **NextFlow Documentation**: https://www.nextflow.io/docs/latest/
- **Seqera Platform**: https://seqera.io/platform/
- **Community Forum**: https://community.seqera.io/
- **GitHub**: https://github.com/nextflow-io/nextflow

---

For questions specific to this pipeline, see [README.md](README.md) or [QUICK_START.md](QUICK_START.md).
