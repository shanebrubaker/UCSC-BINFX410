#!/usr/bin/env nextflow

/*
 * Sequence QC Pipeline with Chromosome-level Processing
 *
 * This workflow:
 * 1. Splits reference FASTA into individual chromosomes
 * 2. Aligns reads to each chromosome separately
 * 3. Performs QC on each chromosome's reads
 * 4. Trims adapters and cleans data
 * 5. Combines cleaned data
 * 6. Generates final QC report with visualizations
 * 7. Produces per-chromosome QC summary
 */

nextflow.enable.dsl=2

// Parameters
params.reference = "reference.fasta"
params.reads = "reads.fastq"
params.outdir = "results"
params.adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
params.min_quality = 25
params.min_length = 50

// Print pipeline information
log.info """
========================================
Sequence QC Pipeline
========================================
Reference  : ${params.reference}
Reads      : ${params.reads}
Output dir : ${params.outdir}
Adapter    : ${params.adapter}
Min quality: ${params.min_quality}
Min length : ${params.min_length}
========================================
"""

/*
 * Process 1: Split reference FASTA into individual chromosomes
 */
process SPLIT_REFERENCE {
    tag "Splitting reference"
    publishDir "${params.outdir}/chromosomes", mode: 'copy'

    input:
    path reference

    output:
    path "chr*.fasta"

    script:
    """
    #!/usr/bin/env python3

    current_chr = None
    current_seq = []
    files = {}

    with open('${reference}', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous chromosome
                if current_chr:
                    filename = f"{current_chr}.fasta"
                    with open(filename, 'w') as out:
                        out.write(f">{current_chr}\\n")
                        seq = ''.join(current_seq)
                        for i in range(0, len(seq), 80):
                            out.write(seq[i:i+80] + '\\n')

                # Start new chromosome
                current_chr = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Save last chromosome
        if current_chr:
            filename = f"{current_chr}.fasta"
            with open(filename, 'w') as out:
                out.write(f">{current_chr}\\n")
                seq = ''.join(current_seq)
                for i in range(0, len(seq), 80):
                    out.write(seq[i:i+80] + '\\n')

    print(f"Split reference into chromosomes")
    """
}

/*
 * Process 2: Align reads to each chromosome
 * (Simplified alignment - extracts reads that match chromosome sequence)
 */
process ALIGN_TO_CHROMOSOME {
    tag "${chromosome.baseName}"
    publishDir "${params.outdir}/aligned", mode: 'copy'

    input:
    tuple path(reads), path(chromosome)

    output:
    tuple val("${chromosome.baseName}"), path("${chromosome.baseName}_subset.fastq")

    script:
    """
    #!/usr/bin/env python3

    import re

    # Read chromosome sequence
    chr_seq = []
    with open('${chromosome}', 'r') as f:
        for line in f:
            if not line.startswith('>'):
                chr_seq.append(line.strip())
    chr_sequence = ''.join(chr_seq)

    # Create kmers from chromosome for faster matching
    # Using smaller kmer size (20bp) for better sensitivity with sequencing errors
    kmer_size = 20
    chr_kmers = set()
    for i in range(len(chr_sequence) - kmer_size + 1):
        chr_kmers.add(chr_sequence[i:i+kmer_size])

    # Process FASTQ and extract matching reads
    matched_reads = []
    with open('${reads}', 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline().strip()
            qual = f.readline().strip()

            # Check if read matches chromosome (simple kmer matching)
            if len(seq) >= kmer_size:
                for i in range(len(seq) - kmer_size + 1):
                    kmer = seq[i:i+kmer_size]
                    if kmer in chr_kmers:
                        matched_reads.append((header, seq, plus, qual))
                        break

    # Write matched reads
    with open('${chromosome.baseName}_subset.fastq', 'w') as out:
        for header, seq, plus, qual in matched_reads:
            out.write(f"{header}\\n{seq}\\n{plus}\\n{qual}\\n")

    print(f"Chromosome ${chromosome.baseName}: {len(matched_reads)} reads aligned")
    """
}

/*
 * Process 3: Run initial QC on chromosome subset
 */
process QC_CHROMOSOME {
    tag "${chr_name}"
    publishDir "${params.outdir}/qc_initial", mode: 'copy'

    input:
    tuple val(chr_name), path(fastq)

    output:
    tuple val(chr_name), path(fastq), path("${chr_name}_initial_qc.json")

    script:
    """
    #!/usr/bin/env python3

    import json

    # Analyze FASTQ file
    total_reads = 0
    total_bases = 0
    quality_sum = 0
    n_count = 0
    low_quality_reads = 0
    adapter_contaminated = 0

    with open('${fastq}', 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline().strip()

            total_reads += 1
            total_bases += len(seq)

            # Count N bases
            n_count += seq.count('N')

            # Calculate average quality
            if qual:
                avg_qual = sum(ord(c) - 33 for c in qual) / len(qual)
                quality_sum += avg_qual

                if avg_qual < ${params.min_quality}:
                    low_quality_reads += 1

            # Check for adapter contamination
            if '${params.adapter}' in seq:
                adapter_contaminated += 1

    # Calculate statistics
    avg_quality = quality_sum / total_reads if total_reads > 0 else 0
    avg_length = total_bases / total_reads if total_reads > 0 else 0

    qc_stats = {
        'chromosome': '${chr_name}',
        'total_reads': total_reads,
        'total_bases': total_bases,
        'avg_length': avg_length,
        'avg_quality': avg_quality,
        'n_bases': n_count,
        'low_quality_reads': low_quality_reads,
        'adapter_contaminated': adapter_contaminated,
        'low_quality_pct': (low_quality_reads / total_reads * 100) if total_reads > 0 else 0,
        'adapter_contaminated_pct': (adapter_contaminated / total_reads * 100) if total_reads > 0 else 0
    }

    with open('${chr_name}_initial_qc.json', 'w') as out:
        json.dump(qc_stats, out, indent=2)

    print(f"QC for ${chr_name}: {total_reads} reads, avg quality: {avg_quality:.2f}")
    """
}

/*
 * Process 4: Trim adapters and clean reads
 */
process TRIM_AND_CLEAN {
    tag "${chr_name}"
    publishDir "${params.outdir}/cleaned", mode: 'copy'

    input:
    tuple val(chr_name), path(fastq), path(qc_json)

    output:
    tuple val(chr_name), path("${chr_name}_cleaned.fastq"), path("${chr_name}_trim_stats.json")

    script:
    """
    #!/usr/bin/env python3

    import json

    adapter = '${params.adapter}'
    min_quality = ${params.min_quality}
    min_length = ${params.min_length}

    cleaned_reads = []
    stats = {
        'total_input': 0,
        'adapter_trimmed': 0,
        'quality_filtered': 0,
        'length_filtered': 0,
        'n_filtered': 0,
        'passed': 0
    }

    with open('${fastq}', 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline().strip()

            stats['total_input'] += 1
            original_length = len(seq)

            # Trim adapter
            adapter_pos = seq.find(adapter)
            if adapter_pos != -1:
                seq = seq[:adapter_pos]
                qual = qual[:adapter_pos]
                stats['adapter_trimmed'] += 1

            # Filter reads with N bases
            if 'N' in seq:
                stats['n_filtered'] += 1
                continue

            # Filter by length
            if len(seq) < min_length:
                stats['length_filtered'] += 1
                continue

            # Filter by quality
            if qual:
                avg_qual = sum(ord(c) - 33 for c in qual) / len(qual)
                if avg_qual < min_quality:
                    stats['quality_filtered'] += 1
                    continue

            # Quality trim from 3' end
            trimmed = False
            for i in range(len(qual) - 1, -1, -1):
                if ord(qual[i]) - 33 < 20:  # Quality threshold for trimming
                    seq = seq[:i]
                    qual = qual[:i]
                    trimmed = True
                else:
                    break

            # Check length again after quality trimming
            if len(seq) < min_length:
                stats['length_filtered'] += 1
                continue

            stats['passed'] += 1
            cleaned_reads.append((header, seq, plus, qual))

    # Write cleaned reads
    with open('${chr_name}_cleaned.fastq', 'w') as out:
        for header, seq, plus, qual in cleaned_reads:
            out.write(f"{header}{seq}\\n{plus}{qual}\\n")

    # Write stats
    with open('${chr_name}_trim_stats.json', 'w') as out:
        json.dump(stats, out, indent=2)

    print(f"Cleaned ${chr_name}: {stats['passed']} / {stats['total_input']} reads passed")
    """
}

/*
 * Process 5: Combine all cleaned chromosome FASTQs
 */
process COMBINE_CLEANED {
    tag "Combining all chromosomes"
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path cleaned_files

    output:
    path "combined_cleaned.fastq"

    script:
    """
    cat ${cleaned_files} > combined_cleaned.fastq

    # Count total reads
    total_reads=\$(grep -c "^@" combined_cleaned.fastq || true)
    echo "Combined FASTQ contains \${total_reads} reads"
    """
}

/*
 * Process 6: Generate final QC report with visualizations
 */
process GENERATE_QC_REPORT {
    tag "Final QC Report"
    publishDir "${params.outdir}/qc_report", mode: 'copy'

    input:
    path combined_fastq
    path initial_qc_files
    path trim_stats_files
    path qc_script

    output:
    path "qc_report.html"
    path "qc_plots/*.png"
    path "qc_summary.json"

    script:
    """
    mkdir -p qc_plots

    python3 ${qc_script} \
        --fastq ${combined_fastq} \
        --initial-qc ${initial_qc_files} \
        --trim-stats ${trim_stats_files} \
        --output-html qc_report.html \
        --output-json qc_summary.json \
        --plot-dir qc_plots
    """
}

/*
 * Process 7: Generate per-chromosome QC summary
 */
process CHROMOSOME_QC_SUMMARY {
    tag "Chromosome QC Summary"
    publishDir "${params.outdir}/qc_report", mode: 'copy'

    input:
    path initial_qc_files
    path trim_stats_files

    output:
    path "chromosome_qc_summary.txt"
    path "chromosome_qc_summary.json"

    script:
    """
    #!/usr/bin/env python3

    import json
    import glob

    # Load all QC files
    initial_qc = {}
    for file in glob.glob('*_initial_qc.json'):
        with open(file) as f:
            data = json.load(f)
            initial_qc[data['chromosome']] = data

    # Load all trim stats
    trim_stats = {}
    for file in glob.glob('*_trim_stats.json'):
        chr_name = file.replace('_trim_stats.json', '')
        with open(file) as f:
            trim_stats[chr_name] = json.load(f)

    # Define QC pass criteria
    def passes_qc(chr_name):
        if chr_name not in initial_qc or chr_name not in trim_stats:
            return False, "Missing data"

        initial = initial_qc[chr_name]
        trimmed = trim_stats[chr_name]

        # Criteria for passing QC:
        # 1. At least 50% of reads pass filtering
        # 2. Average quality > 25
        # 3. Adapter contamination < 20%

        if trimmed['total_input'] == 0:
            return False, "No reads"

        pass_rate = (trimmed['passed'] / trimmed['total_input']) * 100

        reasons = []
        passed = True

        if pass_rate < 50:
            reasons.append(f"Low pass rate ({pass_rate:.1f}%)")
            passed = False

        if initial['avg_quality'] < 25:
            reasons.append(f"Low avg quality ({initial['avg_quality']:.1f})")
            passed = False

        if initial['adapter_contaminated_pct'] > 20:
            reasons.append(f"High adapter contamination ({initial['adapter_contaminated_pct']:.1f}%)")
            passed = False

        if passed:
            return True, "PASS"
        else:
            return False, "; ".join(reasons)

    # Generate summary
    summary = {}
    with open('chromosome_qc_summary.txt', 'w') as out:
        out.write("=" * 80 + "\\n")
        out.write("CHROMOSOME QC SUMMARY\\n")
        out.write("=" * 80 + "\\n\\n")

        for chr_name in sorted(initial_qc.keys()):
            passed, reason = passes_qc(chr_name)
            status = "PASS" if passed else "FAIL"

            initial = initial_qc[chr_name]
            trimmed = trim_stats.get(chr_name, {})

            out.write(f"Chromosome: {chr_name}\\n")
            out.write(f"  Status: {status}\\n")
            if not passed:
                out.write(f"  Reason: {reason}\\n")
            out.write(f"  Input reads: {initial['total_reads']:,}\\n")
            out.write(f"  Cleaned reads: {trimmed.get('passed', 0):,}\\n")
            out.write(f"  Pass rate: {(trimmed.get('passed', 0) / initial['total_reads'] * 100):.1f}%\\n")
            out.write(f"  Avg quality: {initial['avg_quality']:.2f}\\n")
            out.write(f"  Adapter contamination: {initial['adapter_contaminated_pct']:.1f}%\\n")
            out.write("\\n")

            summary[chr_name] = {
                'status': status,
                'passed': passed,
                'reason': reason,
                'input_reads': initial['total_reads'],
                'cleaned_reads': trimmed.get('passed', 0),
                'pass_rate': (trimmed.get('passed', 0) / initial['total_reads'] * 100) if initial['total_reads'] > 0 else 0,
                'avg_quality': initial['avg_quality'],
                'adapter_contamination_pct': initial['adapter_contaminated_pct']
            }

        # Overall summary
        total_passed = sum(1 for s in summary.values() if s['passed'])
        total_chromosomes = len(summary)

        out.write("=" * 80 + "\\n")
        out.write(f"Overall: {total_passed}/{total_chromosomes} chromosomes passed QC\\n")
        out.write("=" * 80 + "\\n")

    # Write JSON summary
    with open('chromosome_qc_summary.json', 'w') as out:
        json.dump(summary, out, indent=2)

    print(f"QC Summary: {total_passed}/{total_chromosomes} chromosomes passed")
    """
}

/*
 * Workflow
 */
workflow {
    // Input channels
    reference_ch = Channel.fromPath(params.reference)
    reads_ch = Channel.fromPath(params.reads)
    qc_script_ch = Channel.fromPath("${projectDir}/qc_analysis.py")

    // Split reference into chromosomes
    SPLIT_REFERENCE(reference_ch)

    // Flatten the output to emit each chromosome file separately
    chromosomes_ch = SPLIT_REFERENCE.out.flatten()

    // Combine reads with each chromosome (Cartesian product)
    ALIGN_TO_CHROMOSOME(reads_ch.combine(chromosomes_ch))

    // QC each chromosome subset
    QC_CHROMOSOME(ALIGN_TO_CHROMOSOME.out)

    // Trim and clean each chromosome
    TRIM_AND_CLEAN(QC_CHROMOSOME.out)

    // Combine all cleaned FASTQs
    cleaned_fastqs = TRIM_AND_CLEAN.out.map { chr, fastq, stats -> fastq }.collect()
    COMBINE_CLEANED(cleaned_fastqs)

    // Collect all QC and stats files
    initial_qc_files = QC_CHROMOSOME.out.map { chr, fastq, qc -> qc }.collect()
    trim_stats_files = TRIM_AND_CLEAN.out.map { chr, fastq, stats -> stats }.collect()

    // Generate final QC report
    GENERATE_QC_REPORT(
        COMBINE_CLEANED.out,
        initial_qc_files,
        trim_stats_files,
        qc_script_ch
    )

    // Generate chromosome QC summary
    CHROMOSOME_QC_SUMMARY(
        initial_qc_files,
        trim_stats_files
    )
}

workflow.onComplete {
    log.info ""
    log.info "========================================"
    log.info "Pipeline completed!"
    log.info "Results: ${params.outdir}"
    log.info "========================================"
}
