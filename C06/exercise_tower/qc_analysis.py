#!/usr/bin/env python3
"""
QC Analysis and Visualization Script
Generates QC report with visualizations from cleaned FASTQ data.
"""

import argparse
import json
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def parse_fastq(fastq_file):
    """Parse FASTQ file and extract quality metrics."""
    reads = []
    quality_scores = []

    with open(fastq_file, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline().strip()

            if qual:
                # Convert quality string to scores
                scores = [ord(c) - 33 for c in qual]
                quality_scores.append(scores)

                reads.append({
                    'length': len(seq),
                    'avg_quality': sum(scores) / len(scores) if scores else 0,
                    'gc_content': (seq.count('G') + seq.count('C')) / len(seq) * 100 if len(seq) > 0 else 0
                })

    return reads, quality_scores

def plot_quality_distribution(reads, output_file):
    """Plot 1: Distribution of average read quality scores."""
    qualities = [r['avg_quality'] for r in reads]

    plt.figure(figsize=(10, 6))
    plt.hist(qualities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Average Quality Score (Phred)', fontsize=12)
    plt.ylabel('Number of Reads', fontsize=12)
    plt.title('Distribution of Average Read Quality Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=30, color='green', linestyle='--', label='Q30 threshold', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_quality_per_position(quality_scores, output_file):
    """Plot 2: Per-base quality scores across all reads."""
    if not quality_scores:
        return

    # Calculate mean quality at each position
    max_length = max(len(q) for q in quality_scores)
    position_qualities = defaultdict(list)

    for scores in quality_scores:
        for i, score in enumerate(scores):
            position_qualities[i].append(score)

    positions = sorted(position_qualities.keys())
    mean_qualities = [np.mean(position_qualities[p]) for p in positions]
    q25 = [np.percentile(position_qualities[p], 25) for p in positions]
    q75 = [np.percentile(position_qualities[p], 75) for p in positions]

    plt.figure(figsize=(12, 6))
    plt.plot(positions, mean_qualities, color='steelblue', linewidth=2, label='Mean')
    plt.fill_between(positions, q25, q75, color='steelblue', alpha=0.3, label='25th-75th percentile')
    plt.xlabel('Position in Read (bp)', fontsize=12)
    plt.ylabel('Quality Score (Phred)', fontsize=12)
    plt.title('Quality Score Distribution Across Read Positions', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=30, color='green', linestyle='--', label='Q30', linewidth=2)
    plt.axhline(y=20, color='orange', linestyle='--', label='Q20', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_read_length_distribution(reads, output_file):
    """Plot 3: Distribution of read lengths."""
    lengths = [r['length'] for r in reads]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Read Length (bp)', fontsize=12)
    plt.ylabel('Number of Reads', fontsize=12)
    plt.title('Distribution of Read Lengths After Trimming', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=np.mean(lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(lengths):.1f} bp', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_gc_content(reads, output_file):
    """Plot 4: GC content distribution."""
    gc_contents = [r['gc_content'] for r in reads]

    plt.figure(figsize=(10, 6))
    plt.hist(gc_contents, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.xlabel('GC Content (%)', fontsize=12)
    plt.ylabel('Number of Reads', fontsize=12)
    plt.title('GC Content Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=50, color='red', linestyle='--', label='50% GC', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_chromosome_comparison(initial_qc_files, trim_stats_files, output_file):
    """Plot 5: Per-chromosome read counts (before and after cleaning)."""
    # Load data
    initial_data = {}
    # Handle both glob patterns (string) and file lists
    if isinstance(initial_qc_files, str):
        files = glob.glob(initial_qc_files)
    else:
        files = initial_qc_files

    for file in files:
        with open(file) as f:
            data = json.load(f)
            initial_data[data['chromosome']] = data['total_reads']

    trim_data = {}
    # Handle both glob patterns (string) and file lists
    if isinstance(trim_stats_files, str):
        files = glob.glob(trim_stats_files)
    else:
        files = trim_stats_files

    for file in files:
        chr_name = file.split('/')[-1].replace('_trim_stats.json', '')
        with open(file) as f:
            data = json.load(f)
            trim_data[chr_name] = data['passed']

    # Sort chromosomes
    chromosomes = sorted(initial_data.keys(),
                         key=lambda x: int(x.replace('chr', '')) if x.replace('chr', '').isdigit() else 999)

    initial_counts = [initial_data[chr] for chr in chromosomes]
    cleaned_counts = [trim_data.get(chr, 0) for chr in chromosomes]

    # Plot
    x = np.arange(len(chromosomes))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, initial_counts, width, label='Before Cleaning',
            color='lightcoral', edgecolor='black')
    plt.bar(x + width/2, cleaned_counts, width, label='After Cleaning',
            color='lightgreen', edgecolor='black')

    plt.xlabel('Chromosome', fontsize=12)
    plt.ylabel('Number of Reads', fontsize=12)
    plt.title('Read Counts Per Chromosome: Before vs After Cleaning', fontsize=14, fontweight='bold')
    plt.xticks(x, chromosomes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def generate_html_report(stats, plot_dir, output_file):
    """Generate HTML QC report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sequence QC Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .stat-label {{
                font-size: 14px;
                opacity: 0.9;
                margin-bottom: 5px;
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: bold;
            }}
            .plot {{
                margin: 30px 0;
                text-align: center;
            }}
            .plot img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .plot-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            .timestamp {{
                text-align: right;
                color: #7f8c8d;
                font-size: 12px;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sequence QC Report</h1>

            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Total Reads</div>
                    <div class="stat-value">{stats['total_reads']:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Bases</div>
                    <div class="stat-value">{stats['total_bases']:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Average Read Length</div>
                    <div class="stat-value">{stats['avg_length']:.0f} bp</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Average Quality</div>
                    <div class="stat-value">Q{stats['avg_quality']:.1f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Average GC Content</div>
                    <div class="stat-value">{stats['avg_gc']:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Q30 Rate</div>
                    <div class="stat-value">{stats['q30_rate']:.1f}%</div>
                </div>
            </div>

            <h2>Quality Visualizations</h2>

            <div class="plot">
                <div class="plot-title">1. Read Quality Score Distribution</div>
                <img src="{plot_dir}/quality_distribution.png" alt="Quality Distribution">
            </div>

            <div class="plot">
                <div class="plot-title">2. Per-Position Quality Scores</div>
                <img src="{plot_dir}/quality_per_position.png" alt="Quality Per Position">
            </div>

            <div class="plot">
                <div class="plot-title">3. Read Length Distribution</div>
                <img src="{plot_dir}/read_length_distribution.png" alt="Read Length Distribution">
            </div>

            <div class="plot">
                <div class="plot-title">4. GC Content Distribution</div>
                <img src="{plot_dir}/gc_content.png" alt="GC Content">
            </div>

            <div class="plot">
                <div class="plot-title">5. Per-Chromosome Read Counts</div>
                <img src="{plot_dir}/chromosome_comparison.png" alt="Chromosome Comparison">
            </div>

            <div class="timestamp">
                Generated by Sequence QC Pipeline
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_file, 'w') as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description='Generate QC analysis and visualizations')
    parser.add_argument('--fastq', required=True, help='Combined cleaned FASTQ file')
    parser.add_argument('--initial-qc', required=True, nargs='+', help='Initial QC JSON files')
    parser.add_argument('--trim-stats', required=True, nargs='+', help='Trim stats JSON files')
    parser.add_argument('--output-html', required=True, help='Output HTML report')
    parser.add_argument('--output-json', required=True, help='Output JSON summary')
    parser.add_argument('--plot-dir', required=True, help='Directory for plots')
    args = parser.parse_args()

    print("Analyzing FASTQ data...")
    reads, quality_scores = parse_fastq(args.fastq)

    print(f"Generating visualizations in {args.plot_dir}...")

    # Generate plots
    plot_quality_distribution(reads, f"{args.plot_dir}/quality_distribution.png")
    plot_quality_per_position(quality_scores, f"{args.plot_dir}/quality_per_position.png")
    plot_read_length_distribution(reads, f"{args.plot_dir}/read_length_distribution.png")
    plot_gc_content(reads, f"{args.plot_dir}/gc_content.png")
    plot_chromosome_comparison(args.initial_qc, args.trim_stats,
                              f"{args.plot_dir}/chromosome_comparison.png")

    # Calculate summary statistics
    stats = {
        'total_reads': len(reads),
        'total_bases': sum(r['length'] for r in reads),
        'avg_length': np.mean([r['length'] for r in reads]) if reads else 0,
        'avg_quality': np.mean([r['avg_quality'] for r in reads]) if reads else 0,
        'avg_gc': np.mean([r['gc_content'] for r in reads]) if reads else 0,
        'q30_rate': sum(1 for r in reads if r['avg_quality'] >= 30) / len(reads) * 100 if reads else 0
    }

    # Generate HTML report
    print(f"Generating HTML report: {args.output_html}")
    generate_html_report(stats, args.plot_dir, args.output_html)

    # Write JSON summary
    with open(args.output_json, 'w') as f:
        json.dump(stats, f, indent=2)

    print("QC analysis complete!")
    print(f"  Total reads: {stats['total_reads']:,}")
    print(f"  Average quality: Q{stats['avg_quality']:.1f}")
    print(f"  Q30 rate: {stats['q30_rate']:.1f}%")

if __name__ == '__main__':
    main()
