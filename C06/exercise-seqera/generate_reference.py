#!/usr/bin/env python3
"""
Generate a faux reference genome with 10 chromosomes.
Each chromosome has different length and base composition.
"""

import random
import argparse

def generate_chromosome(chr_num, length):
    """Generate a random DNA sequence for a chromosome."""
    bases = ['A', 'T', 'G', 'C']

    # Add some variation in GC content per chromosome
    gc_bias = 0.5 + (chr_num - 5) * 0.02  # Range from ~40% to ~60%

    sequence = []
    for _ in range(length):
        if random.random() < gc_bias:
            sequence.append(random.choice(['G', 'C']))
        else:
            sequence.append(random.choice(['A', 'T']))

    return ''.join(sequence)

def write_fasta(sequences, output_file):
    """Write sequences to a FASTA file."""
    with open(output_file, 'w') as f:
        for chr_name, seq in sequences.items():
            f.write(f">{chr_name}\n")
            # Write sequence in 80-character lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate faux reference genome')
    parser.add_argument('-o', '--output', default='reference.fasta',
                        help='Output FASTA file (default: reference.fasta)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate 10 chromosomes with varying lengths
    # Lengths range from 50kb to 200kb
    chromosome_lengths = {
        'chr1': 200000,
        'chr2': 180000,
        'chr3': 160000,
        'chr4': 140000,
        'chr5': 120000,
        'chr6': 100000,
        'chr7': 80000,
        'chr8': 70000,
        'chr9': 60000,
        'chr10': 50000,
    }

    print(f"Generating reference genome with {len(chromosome_lengths)} chromosomes...")
    sequences = {}

    for chr_name, length in chromosome_lengths.items():
        print(f"  Generating {chr_name}: {length:,} bp")
        sequences[chr_name] = generate_chromosome(int(chr_name[3:]), length)

    write_fasta(sequences, args.output)

    total_length = sum(chromosome_lengths.values())
    print(f"\nGenome generated successfully!")
    print(f"  Output file: {args.output}")
    print(f"  Total length: {total_length:,} bp")
    print(f"  Number of chromosomes: {len(chromosome_lengths)}")

if __name__ == '__main__':
    main()
