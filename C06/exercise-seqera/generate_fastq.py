#!/usr/bin/env python3
"""
Generate faux FASTQ data with various quality issues for testing QC pipeline.
Issues include:
- Low quality scores in some reads
- Adapter contamination
- Overrepresented sequences
- N bases (ambiguous bases)
- Quality score degradation toward read ends
"""

import random
import argparse

# Common Illumina adapter sequence
ADAPTER_SEQ = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"

def quality_to_char(quality):
    """Convert quality score (0-40) to Phred+33 character."""
    return chr(min(quality + 33, 126))

def generate_quality_scores(length, mean_quality=30, end_degradation=False, low_quality=False):
    """Generate quality scores for a read."""
    scores = []

    for i in range(length):
        if low_quality:
            # Intentionally low quality
            base_quality = random.randint(10, 25)
        else:
            base_quality = random.randint(mean_quality - 5, min(mean_quality + 5, 40))

        # Add quality degradation toward the end of the read
        if end_degradation and i > length * 0.7:
            degradation = int((i - length * 0.7) / (length * 0.3) * 15)
            base_quality = max(5, base_quality - degradation)

        scores.append(quality_to_char(base_quality))

    return ''.join(scores)

def generate_read(length=150, reference=None):
    """Generate a random DNA sequence read."""
    bases = ['A', 'T', 'G', 'C']

    if reference and random.random() < 0.7:
        # 70% of reads come from reference (simulating real alignment)
        chr_name = random.choice(list(reference.keys()))
        chr_seq = reference[chr_name]

        if len(chr_seq) > length:
            start_pos = random.randint(0, len(chr_seq) - length)
            read = chr_seq[start_pos:start_pos + length]

            # Add some sequencing errors (1% error rate)
            read_list = list(read)
            for i in range(len(read_list)):
                if random.random() < 0.01:
                    read_list[i] = random.choice(bases)

            return ''.join(read_list)

    # Generate random sequence
    return ''.join(random.choice(bases) for _ in range(length))

def add_adapter_contamination(read, quality):
    """Add adapter contamination to a read."""
    contamination_length = random.randint(10, 25)
    adapter_fragment = ADAPTER_SEQ[:contamination_length]

    # Add adapter to the end
    contaminated_read = read[:-contamination_length] + adapter_fragment

    # Adjust quality scores
    contaminated_quality = quality

    return contaminated_read, contaminated_quality

def add_n_bases(read, quality, n_count=None):
    """Add N bases (ambiguous bases) to a read."""
    if n_count is None:
        n_count = random.randint(1, 5)

    read_list = list(read)
    for _ in range(n_count):
        pos = random.randint(0, len(read_list) - 1)
        read_list[pos] = 'N'

    return ''.join(read_list), quality

def load_reference(reference_file):
    """Load reference genome for generating realistic reads."""
    sequences = {}
    current_chr = None
    current_seq = []

    try:
        with open(reference_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_chr:
                        sequences[current_chr] = ''.join(current_seq)
                    current_chr = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_chr:
                sequences[current_chr] = ''.join(current_seq)

        return sequences
    except FileNotFoundError:
        print(f"Warning: Reference file {reference_file} not found. Generating random reads.")
        return None

def write_fastq(reads, output_file):
    """Write reads to FASTQ file."""
    with open(output_file, 'w') as f:
        for i, (seq, qual) in enumerate(reads, 1):
            f.write(f"@read_{i}\n")
            f.write(f"{seq}\n")
            f.write("+\n")
            f.write(f"{qual}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate faux FASTQ data with quality issues')
    parser.add_argument('-o', '--output', default='reads.fastq',
                        help='Output FASTQ file (default: reads.fastq)')
    parser.add_argument('-n', '--num-reads', type=int, default=100000,
                        help='Number of reads to generate (default: 100000)')
    parser.add_argument('-l', '--read-length', type=int, default=150,
                        help='Read length (default: 150)')
    parser.add_argument('-r', '--reference', default='reference.fasta',
                        help='Reference genome file (default: reference.fasta)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load reference genome
    reference = load_reference(args.reference)

    print(f"Generating {args.num_reads:,} reads with quality issues...")

    reads = []
    issue_counts = {
        'low_quality': 0,
        'adapter_contamination': 0,
        'n_bases': 0,
        'end_degradation': 0,
        'overrepresented': 0,
    }

    # Generate an overrepresented sequence
    overrepresented_seq = ''.join(random.choice(['A', 'T', 'G', 'C'])
                                   for _ in range(args.read_length))

    for i in range(args.num_reads):
        # Determine what issues this read will have
        issue_type = random.random()

        if issue_type < 0.05:
            # 5% overrepresented sequences (PCR duplicates)
            seq = overrepresented_seq
            qual = generate_quality_scores(args.read_length)
            issue_counts['overrepresented'] += 1

        elif issue_type < 0.15:
            # 10% low quality reads
            seq = generate_read(args.read_length, reference)
            qual = generate_quality_scores(args.read_length, low_quality=True)
            issue_counts['low_quality'] += 1

        elif issue_type < 0.25:
            # 10% adapter contamination
            seq = generate_read(args.read_length, reference)
            qual = generate_quality_scores(args.read_length)
            seq, qual = add_adapter_contamination(seq, qual)
            issue_counts['adapter_contamination'] += 1

        elif issue_type < 0.30:
            # 5% with N bases
            seq = generate_read(args.read_length, reference)
            qual = generate_quality_scores(args.read_length)
            seq, qual = add_n_bases(seq, qual)
            issue_counts['n_bases'] += 1

        elif issue_type < 0.50:
            # 20% with end degradation
            seq = generate_read(args.read_length, reference)
            qual = generate_quality_scores(args.read_length, end_degradation=True)
            issue_counts['end_degradation'] += 1

        else:
            # 50% normal reads
            seq = generate_read(args.read_length, reference)
            qual = generate_quality_scores(args.read_length)

        reads.append((seq, qual))

    write_fastq(reads, args.output)

    print(f"\nFASTQ file generated successfully!")
    print(f"  Output file: {args.output}")
    print(f"  Total reads: {args.num_reads:,}")
    print(f"  Read length: {args.read_length}")
    print(f"\nQuality issues introduced:")
    print(f"  Low quality reads: {issue_counts['low_quality']:,} ({issue_counts['low_quality']/args.num_reads*100:.1f}%)")
    print(f"  Adapter contamination: {issue_counts['adapter_contamination']:,} ({issue_counts['adapter_contamination']/args.num_reads*100:.1f}%)")
    print(f"  Reads with N bases: {issue_counts['n_bases']:,} ({issue_counts['n_bases']/args.num_reads*100:.1f}%)")
    print(f"  End degradation: {issue_counts['end_degradation']:,} ({issue_counts['end_degradation']/args.num_reads*100:.1f}%)")
    print(f"  Overrepresented sequences: {issue_counts['overrepresented']:,} ({issue_counts['overrepresented']/args.num_reads*100:.1f}%)")

if __name__ == '__main__':
    main()
