# Quick Access: Monitoring Dashboards

## 🎯 View Your Results Now

### Local HTML Dashboards (No Setup Required)

```bash
# Timeline - When each task ran
open results/trace/execution_timeline.html

# Resource Usage - CPU, memory, duration statistics  
open results/trace/execution_report.html

# Pipeline Diagram - Visual workflow structure
open results/trace/pipeline_dag.html

# QC Report - Quality visualizations
open results/qc_report/qc_report.html
```

### Quick View All Reports

```bash
# Open all dashboards at once (macOS)
open results/trace/execution_timeline.html \
     results/trace/execution_report.html \
     results/trace/pipeline_dag.html \
     results/qc_report/qc_report.html

# Linux
xdg-open results/trace/execution_timeline.html &
xdg-open results/trace/execution_report.html &
xdg-open results/trace/pipeline_dag.html &
xdg-open results/qc_report/qc_report.html &
```

---

## 🌐 NextFlow Tower (Real-time Cloud Dashboard)

### Why Use Tower?
- ✨ Real-time monitoring of running workflows
- 📊 Beautiful interactive dashboards
- 📈 Resource usage graphs
- 🔔 Email notifications
- 📜 Historical run tracking
- 👥 Team collaboration

### Quick Setup (2 minutes)

**1. Create free account:**
https://cloud.seqera.io

**2. Get access token:**
- Login → Click your profile (top right) → "Your tokens"
- Click "Add token" → Copy the token

**3. Configure and run:**
```bash
# Set your token
export TOWER_ACCESS_TOKEN=eyJ...your-token-here

# Run pipeline with Tower
nextflow run main.nf -with-tower
```

**4. View dashboard:**
https://cloud.seqera.io/orgs/community/workspaces/default/watch

---

## 📱 Access Links

| Dashboard | Local File | Cloud URL (after setup) |
|-----------|-----------|-------------------------|
| **Tower Main** | N/A | https://cloud.seqera.io |
| **Execution Timeline** | `file://$(pwd)/results/trace/execution_timeline.html` | Available in Tower |
| **Resource Report** | `file://$(pwd)/results/trace/execution_report.html` | Available in Tower |
| **Pipeline DAG** | `file://$(pwd)/results/trace/pipeline_dag.html` | Available in Tower |
| **QC Report** | `file://$(pwd)/results/qc_report/qc_report.html` | N/A (local only) |

---

## 🔍 Monitoring Commands

```bash
# List all pipeline runs
nextflow log

# View details of last run
nextflow log $(nextflow log | tail -1 | awk '{print $3}')

# Check execution trace
column -t -s $'\t' results/trace/execution_trace.txt | less -S

# View chromosome summary
cat results/qc_report/chromosome_qc_summary.txt

# Count tasks per process
nextflow log -f 'process' | sort | uniq -c
```

---

## 📖 Full Documentation

- **[MONITORING.md](MONITORING.md)** - Complete monitoring guide
- **[SUMMARY.md](SUMMARY.md)** - Pipeline execution summary
- **[README.md](README.md)** - Full documentation
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide

---

## 🎬 Example: Run with Full Monitoring

```bash
# Step 1: Set up Tower (optional but recommended)
export TOWER_ACCESS_TOKEN=<your-token>

# Step 2: Run pipeline with monitoring
nextflow run main.nf -with-tower -name my-qc-run

# Step 3: While running, open Tower dashboard
# Visit: https://cloud.seqera.io
# Click on "my-qc-run" to see real-time progress

# Step 4: After completion, view local reports
open results/trace/execution_timeline.html
open results/qc_report/qc_report.html
```

---

## ⚡ Current Run Summary

**Latest Execution: distraught_raman**
- ✅ Status: Completed successfully
- 📊 Tasks: 34 total (10 chr × 3 processes + 4 combined)
- 🧬 Chromosomes: 10/10 processed in parallel
- ✓ QC Results: 10/10 passed
- 📁 Results: `results/` directory

**View it now:**
```bash
open results/trace/execution_timeline.html
```

---

Need help? See [MONITORING.md](MONITORING.md) for detailed instructions.
