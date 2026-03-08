#!/usr/bin/env python3
"""Generate realistic synthetic variant data for a genomics Spark exercise.

This script produces VCF-like CSV files with realistic distributions of
genomic variants, along with companion files for gene annotations, sample
metadata, and known pathogenic variants. The data is designed for teaching
PySpark operations such as filtering, grouping, joining, and aggregation.

Usage:
    python generate_data.py --size small      # 1,000 variants
    python generate_data.py --size medium     # 100,000 variants
    python generate_data.py --size large      # 1,000,000 variants

Output files (written to --output-dir, default is the script's directory):
    variants_{size}.csv   - Main variant call data
    genes.bed             - Gene annotations (BED format, tab-separated)
    samples.csv           - Sample metadata
    known_variants.csv    - Known pathogenic variants for join exercises

All random generation is seeded (numpy seed=42) for full reproducibility.
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional tqdm import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:

    class _FallbackTqdm:
        """Minimal stand-in when tqdm is not installed."""

        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self._iterable = iterable
            self._total = total or (len(iterable) if iterable is not None else 0)
            self._desc = desc or ""
            self._n = 0

        def __iter__(self):
            for item in self._iterable:
                yield item
                self._n += 1
                if self._total and self._n % max(1, self._total // 10) == 0:
                    pct = self._n / self._total * 100
                    print(f"  {self._desc}: {pct:5.1f}% ({self._n}/{self._total})")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n: int = 1) -> None:
            self._n += n

    tqdm = _FallbackTqdm  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_SIZES: Dict[str, int] = {
    "small": 1_000,
    "medium": 100_000,
    "large": 1_000_000,
}

CHROMOSOMES: List[str] = [
    "chr1", "chr2", "chr3", "chr4", "chr5",
    "chr6", "chr7", "chr8", "chr9", "chr10",
    "chr11", "chr12", "chr13", "chr14", "chr15",
    "chr16", "chr17", "chr18", "chr19", "chr20",
    "chr21", "chr22", "chrX", "chrY",
]

# Approximate proportion of the genome each chromosome represents (GRCh38).
CHROM_PROPORTIONS: List[float] = [
    0.085, 0.083, 0.068, 0.065, 0.062,
    0.058, 0.054, 0.050, 0.048, 0.046,
    0.046, 0.046, 0.039, 0.037, 0.035,
    0.031, 0.028, 0.027, 0.020, 0.022,
    0.016, 0.017, 0.053, 0.020,
]

# Approximate chromosome lengths (GRCh38, rounded).
CHROM_LENGTHS: Dict[str, int] = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559,
    "chr4": 190_214_555, "chr5": 181_538_259, "chr6": 170_805_979,
    "chr7": 159_345_973, "chr8": 145_138_636, "chr9": 138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16": 90_338_345, "chr17": 83_257_441, "chr18": 80_373_285,
    "chr19": 58_617_616, "chr20": 64_444_167, "chr21": 46_709_983,
    "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}

BASES: List[str] = ["A", "C", "G", "T"]

FILTER_VALUES: List[str] = ["PASS", "LowQual", "LowDP", "LowGQ"]
FILTER_WEIGHTS: List[float] = [0.70, 0.12, 0.10, 0.08]

CONSEQUENCES: List[str] = [
    "intron_variant",
    "intergenic_variant",
    "synonymous_variant",
    "missense_variant",
    "stop_gained",
    "splice_donor_variant",
    "frameshift_variant",
]
CONSEQUENCE_WEIGHTS: List[float] = [0.35, 0.30, 0.15, 0.12, 0.02, 0.02, 0.04]

GENE_NAMES: List[str] = [
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "PTEN",
    "APC", "RB1", "MYC", "CDKN2A", "VHL", "NF1", "NF2", "ATM", "CHEK2",
    "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "CDH1", "STK11", "SMAD4",
    "BMPR1A", "MUTYH", "AKT1", "ALK", "ERBB2", "FGFR1", "FGFR2",
    "FGFR3", "FLT3", "IDH1", "IDH2", "JAK2", "KIT", "MET", "NRAS",
    "PDGFRA", "RET", "ROS1", "SMO", "ABL1", "BCR", "NOTCH1", "NOTCH2",
    "CTNNB1", "TERT", "GATA3", "ESR1", "AR", "FOXL2", "WT1", "PAX8",
    "SOX9", "RUNX1", "ETV6", "TET2", "DNMT3A", "NPM1", "CEBPA", "FLI1",
    "EWSR1", "SS18", "CCND1", "CDK4", "CDK6", "MDM2", "ARID1A", "ARID1B",
    "SMARCA4", "SMARCB1", "KDM6A", "KMT2A", "KMT2D", "CREBBP", "EP300",
    "SETD2", "BAP1", "PHF6", "ASXL1", "EZH2", "SUZ12", "SF3B1", "SRSF2",
    "U2AF1", "ZRSR2", "STAG2", "RAD21", "BCOR", "CIC", "FUBP1", "MAP2K1",
    "MAP2K2", "PTCH1", "DICER1", "POLE", "POLD1", "SDHA",
]

POPULATIONS: List[str] = ["EUR", "AFR", "EAS", "SAS", "AMR"]
POPULATION_WEIGHTS: List[float] = [0.35, 0.25, 0.20, 0.12, 0.08]

PHENOTYPES: List[str] = ["case", "control"]
PHENOTYPE_WEIGHTS: List[float] = [0.45, 0.55]

SEXES: List[str] = ["male", "female"]

CLINICAL_SIGNIFICANCE: List[str] = [
    "pathogenic", "likely_pathogenic", "benign",
    "likely_benign", "uncertain_significance",
]
CLINICAL_WEIGHTS: List[float] = [0.20, 0.15, 0.25, 0.20, 0.20]

DISEASES: List[str] = [
    "breast_cancer", "lung_cancer", "colorectal_cancer", "prostate_cancer",
    "ovarian_cancer", "diabetes_type2", "diabetes_type1", "heart_disease",
    "cardiomyopathy", "alzheimers", "parkinsons", "cystic_fibrosis",
    "sickle_cell_disease", "huntingtons", "marfan_syndrome",
    "lynch_syndrome", "li_fraumeni_syndrome", "neurofibromatosis",
    "retinoblastoma", "wilms_tumor",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_weights(weights: List[float]) -> np.ndarray:
    """Return a numpy array of weights that sums exactly to 1.0."""
    arr = np.array(weights, dtype=np.float64)
    return arr / arr.sum()


def _generate_bimodal_quality(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Generate bimodal quality scores (80% high ~85, 20% low ~15)."""
    high_mask = rng.random(n) < 0.80
    scores = np.empty(n, dtype=np.float64)
    n_high = int(high_mask.sum())
    scores[high_mask] = rng.normal(loc=85.0, scale=10.0, size=n_high)
    scores[~high_mask] = rng.normal(loc=15.0, scale=8.0, size=n - n_high)
    return np.clip(np.round(scores, 2), 0.0, 100.0)


def _generate_correlated_dp_gq(
    rng: np.random.RandomState, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated read depth (DP) and genotype quality (GQ)."""
    dp = rng.lognormal(mean=3.2, sigma=0.6, size=n)
    dp = np.clip(np.round(dp), 1, 100).astype(int)
    noise = rng.normal(0, 8, size=n)
    gq = dp * 0.8 + noise + 10
    gq = np.clip(np.round(gq), 0, 99).astype(int)
    return dp, gq


def _alt_base(ref: str, rng: np.random.RandomState) -> str:
    """Return a random alternate base different from ref."""
    alternatives = [b for b in BASES if b != ref]
    return alternatives[rng.randint(len(alternatives))]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------


def generate_genes(
    rng: np.random.RandomState, output_path: str, n_genes: int = 500
) -> List[Tuple[str, int, int, str]]:
    """Generate a BED file of gene annotations and return the records."""
    print(f"Generating {n_genes} gene annotations ...")
    chrom_probs = _normalise_weights(CHROM_PROPORTIONS)
    strands = ["+", "-"]

    used_names: List[str] = list(GENE_NAMES)
    while len(used_names) < n_genes:
        used_names.append(f"GENE{len(used_names) - len(GENE_NAMES) + 1}")
    rng.shuffle(used_names)  # type: ignore[arg-type]
    gene_names = used_names[:n_genes]

    records: List[Tuple[str, int, int, str]] = []
    with open(output_path, "w") as fh:
        for i in range(n_genes):
            chrom = CHROMOSOMES[rng.choice(len(CHROMOSOMES), p=chrom_probs)]
            max_pos = CHROM_LENGTHS[chrom]
            gene_len = int(np.clip(rng.lognormal(10.3, 1.0), 1_000, 2_000_000))
            start = rng.randint(1, max(2, max_pos - gene_len))
            end = start + gene_len
            strand = strands[rng.randint(2)]
            fh.write(f"{chrom}\t{start}\t{end}\t{gene_names[i]}\t{strand}\n")
            records.append((chrom, start, end, gene_names[i]))

    print(f"  Wrote {n_genes} gene records to {output_path}")
    return records


def generate_samples(
    rng: np.random.RandomState, output_path: str, n_samples: int = 200
) -> None:
    """Generate sample metadata CSV."""
    print(f"Generating {n_samples} sample records ...")
    pop_probs = _normalise_weights(POPULATION_WEIGHTS)
    pheno_probs = _normalise_weights(PHENOTYPE_WEIGHTS)

    with open(output_path, "w") as fh:
        fh.write("sample_id,population,sex,phenotype\n")
        for i in range(n_samples):
            sample_id = f"SAMPLE_{i + 1:04d}"
            population = POPULATIONS[rng.choice(len(POPULATIONS), p=pop_probs)]
            sex = SEXES[rng.randint(2)]
            phenotype = PHENOTYPES[rng.choice(len(PHENOTYPES), p=pheno_probs)]
            fh.write(f"{sample_id},{population},{sex},{phenotype}\n")
    print(f"  Wrote {n_samples} sample records to {output_path}")


def generate_known_variants(
    rng: np.random.RandomState, output_path: str, n_known: int = 1_000
) -> None:
    """Generate known pathogenic variants CSV for join exercises."""
    print(f"Generating {n_known} known pathogenic variants ...")
    chrom_probs = _normalise_weights(CHROM_PROPORTIONS)
    clin_probs = _normalise_weights(CLINICAL_WEIGHTS)

    with open(output_path, "w") as fh:
        fh.write("CHROM,POS,ID,clinical_significance,disease\n")
        rs_counter = 50_000
        for _ in range(n_known):
            chrom = CHROMOSOMES[rng.choice(len(CHROMOSOMES), p=chrom_probs)]
            pos = rng.randint(1, CHROM_LENGTHS[chrom])
            rs_id = f"rs{rs_counter}"
            rs_counter += rng.randint(1, 30)
            clin_sig = CLINICAL_SIGNIFICANCE[
                rng.choice(len(CLINICAL_SIGNIFICANCE), p=clin_probs)
            ]
            disease = DISEASES[rng.randint(len(DISEASES))]
            fh.write(f"{chrom},{pos},{rs_id},{clin_sig},{disease}\n")
    print(f"  Wrote {n_known} known variant records to {output_path}")


def generate_variants(
    rng: np.random.RandomState,
    n_variants: int,
    gene_records: List[Tuple[str, int, int, str]],
    output_path: str,
) -> None:
    """Generate the main variants CSV file."""
    print(f"Generating {n_variants:,} variants ...")

    chrom_probs = _normalise_weights(CHROM_PROPORTIONS)
    chrom_indices = rng.choice(len(CHROMOSOMES), size=n_variants, p=chrom_probs)
    chroms = [CHROMOSOMES[i] for i in chrom_indices]

    positions = np.array(
        [rng.randint(1, CHROM_LENGTHS[c]) for c in chroms], dtype=np.int64
    )

    ref_indices = rng.randint(0, 4, size=n_variants)
    refs = [BASES[i] for i in ref_indices]
    alts = [_alt_base(r, rng) for r in refs]

    # Variant IDs: ~60% have an rs-number, rest are novel.
    has_rsid = rng.random(n_variants) < 0.60
    rs_counter = 100_000
    ids: List[str] = []
    for flag in has_rsid:
        if flag:
            ids.append(f"rs{rs_counter}")
            rs_counter += rng.randint(1, 50)
        else:
            ids.append(".")

    quals = _generate_bimodal_quality(rng, n_variants)

    filter_probs = _normalise_weights(FILTER_WEIGHTS)
    filters = rng.choice(FILTER_VALUES, size=n_variants, p=filter_probs)

    dp, gq = _generate_correlated_dp_gq(rng, n_variants)

    # Allele frequency: beta distribution skewed toward rare variants.
    af = np.round(rng.beta(0.5, 5.0, size=n_variants), 6)

    # Build gene lookup for assignment.
    gene_by_chrom: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for chrom, start, end, name in gene_records:
        gene_by_chrom[chrom].append((start, end, name))
    for key in gene_by_chrom:
        gene_by_chrom[key].sort()

    def _find_gene(chrom: str, pos: int) -> str:
        for gstart, gend, gname in gene_by_chrom.get(chrom, []):
            if gstart <= pos <= gend:
                return gname
        return "."

    consequence_probs = _normalise_weights(CONSEQUENCE_WEIGHTS)
    consequences_raw = rng.choice(CONSEQUENCES, size=n_variants, p=consequence_probs)

    genes: List[str] = []
    consequences: List[str] = []
    for idx in tqdm(range(n_variants), desc="Assigning genes", total=n_variants):
        g = _find_gene(chroms[idx], positions[idx])
        if g == ".":
            consequences.append("intergenic_variant")
        else:
            consequences.append(consequences_raw[idx])
        genes.append(g)

    print(f"Writing {output_path} ...")
    with open(output_path, "w") as fh:
        fh.write("CHROM,POS,ID,REF,ALT,QUAL,FILTER,DP,AF,GQ,GENE,CONSEQUENCE\n")
        for i in tqdm(range(n_variants), desc="Writing rows", total=n_variants):
            fh.write(
                f"{chroms[i]},{positions[i]},{ids[i]},{refs[i]},{alts[i]},"
                f"{quals[i]},{filters[i]},{dp[i]},{af[i]},{gq[i]},"
                f"{genes[i]},{consequences[i]}\n"
            )
    print(f"  Wrote {n_variants:,} variant records.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic genomic variant data for Spark exercises.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_data.py --size small\n"
            "  python generate_data.py --size medium\n"
            "  python generate_data.py --size large\n"
        ),
    )
    parser.add_argument(
        "--size",
        choices=list(DATASET_SIZES.keys()),
        default="small",
        help="Dataset size to generate (default: small).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same directory as this script).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for data generation."""
    args = parse_args(argv)
    size: str = args.size
    n_variants: int = DATASET_SIZES[size]

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Genomics Synthetic Data Generator")
    print(f"  Size      : {size} ({n_variants:,} variants)")
    print(f"  Output dir: {output_dir}")
    print("  Seed      : 42")
    print("=" * 60)

    rng = np.random.RandomState(42)

    genes_path = os.path.join(output_dir, "genes.bed")
    gene_records = generate_genes(rng, genes_path, n_genes=500)

    samples_path = os.path.join(output_dir, "samples.csv")
    generate_samples(rng, samples_path, n_samples=200)

    known_path = os.path.join(output_dir, "known_variants.csv")
    generate_known_variants(rng, known_path, n_known=1_000)

    variants_path = os.path.join(output_dir, f"variants_{size}.csv")
    generate_variants(rng, n_variants, gene_records, variants_path)

    print()
    print("=" * 60)
    print("Data generation complete. Files created:")
    for fpath in [genes_path, samples_path, known_path, variants_path]:
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {os.path.basename(fpath):30s} {size_mb:8.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
