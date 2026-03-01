#!/bin/bash

#
# Quick Test Script for Sequence QC Pipeline
# This script automates the testing process
#

set -e  # Exit on error

echo "========================================="
echo "Sequence QC Pipeline - Test Script"
echo "========================================="
echo ""

# Check if NextFlow is installed
echo "1. Checking NextFlow installation..."
if ! command -v nextflow &> /dev/null; then
    echo "   ERROR: NextFlow is not installed or not in PATH"
    echo "   Please install NextFlow following the instructions in README.md"
    exit 1
fi

echo "   NextFlow version: $(nextflow -version | head -n 1)"
echo "   ✓ NextFlow found"
echo ""

# Check Python installation
echo "2. Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "   ERROR: Python 3 is not installed"
    exit 1
fi

echo "   Python version: $(python3 --version)"
echo "   ✓ Python found"
echo ""

# Check Python dependencies
echo "3. Checking Python dependencies..."
python3 -c "import matplotlib, numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ matplotlib and numpy found"
else
    echo "   WARNING: matplotlib or numpy not found"
    echo "   Installing dependencies..."
    pip install matplotlib numpy
fi
echo ""

# Make scripts executable
echo "4. Making scripts executable..."
chmod +x generate_reference.py generate_fastq.py qc_analysis.py
echo "   ✓ Scripts are now executable"
echo ""

# Generate test data
echo "5. Generating test data..."
echo "   Generating reference genome..."
python3 generate_reference.py -o reference.fasta -s 42

echo "   Generating FASTQ data (10,000 reads)..."
python3 generate_fastq.py -o reads.fastq -n 10000 -r reference.fasta -s 42

echo "   ✓ Test data generated"
echo ""

# Display file sizes
echo "6. Test data summary:"
if [ -f reference.fasta ]; then
    ref_size=$(du -h reference.fasta | cut -f1)
    echo "   reference.fasta: $ref_size"
fi
if [ -f reads.fastq ]; then
    reads_size=$(du -h reads.fastq | cut -f1)
    echo "   reads.fastq: $reads_size"
fi
echo ""

# Run NextFlow pipeline
echo "7. Running NextFlow pipeline..."
echo "   This may take a few minutes..."
echo ""

nextflow run main.nf

echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo ""
echo "Results are available in the 'results' directory:"
echo ""
echo "  Main outputs:"
echo "  - results/combined_cleaned.fastq - Combined cleaned reads"
echo "  - results/qc_report/qc_report.html - Interactive QC report"
echo "  - results/qc_report/chromosome_qc_summary.txt - Per-chromosome summary"
echo ""
echo "  Visualizations:"
echo "  - results/qc_report/qc_plots/*.png - Quality plots"
echo ""
echo "To view the QC report, open in your browser:"
echo "  open results/qc_report/qc_report.html    # macOS"
echo "  xdg-open results/qc_report/qc_report.html # Linux"
echo ""
echo "To view the chromosome summary:"
echo "  cat results/qc_report/chromosome_qc_summary.txt"
echo ""
echo "To view execution reports:"
echo "  open results/trace/execution_timeline.html"
echo "  open results/trace/execution_report.html"
echo ""
