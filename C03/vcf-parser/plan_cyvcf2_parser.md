# Plan: cyvcf2-based VCF Parser Notebook

## Overview
Create a second Jupyter notebook (`vcf_parser_cyvcf2.ipynb`) that uses the cyvcf2 library instead of manual parsing. cyvcf2 is a fast Cython wrapper around htslib, providing significantly better performance for large VCF files.

## Key Differences from Manual Parser

| Feature | Manual Parser | cyvcf2 Parser |
|---------|---------------|---------------|
| Speed | Slower (pure Python) | Fast (C/Cython) |
| Dependencies | Standard library + pandas | cyvcf2 + numpy |
| Compressed files | gzip module | Native support (including bgzip) |
| Index support | None | Supports .tbi/.csi indexes |
| Memory | Loads all to DataFrame | Can iterate lazily |

## Implementation Plan

### 1. CyVCF2Parser Class
Create a wrapper class similar to the manual parser but using cyvcf2:

```python
from cyvcf2 import VCF

class CyVCF2Parser:
    def __init__(self, file_path=None)
    def parse(self, file_path) -> self
    def get_variant_count() -> int
    def get_chromosomes() -> List[str]
    def get_variants_by_chromosome(chrom) -> pd.DataFrame
    def get_variant_types() -> pd.Series
    def summary() -> Dict
```

### 2. Custom Exceptions
Reuse the same exception hierarchy:
- `VCFError` (base)
- `VCFFileNotFoundError`
- `VCFFormatError`

### 3. Unit Tests
Adapt existing tests to work with cyvcf2 parser:
- Test parsing valid VCF
- Test metadata extraction
- Test sample detection
- Test chromosome filtering
- Test variant type classification
- Test error handling

### 4. Visualizations
Reuse the same 5 visualizations:
1. Variant distribution by chromosome (bar chart)
2. Variant types distribution (pie chart)
3. Quality score distribution (histogram + boxplot)
4. Allele frequency vs read depth (scatter plot)
5. Genomic positions across chromosomes

### 5. Performance Comparison
Add a cell comparing parsing speed between manual and cyvcf2 parsers.

## Files to Create
- `/Users/shanebrubaker/work/BINFX410/C03/vcf-parser/vcf_parser_cyvcf2.ipynb`

## Dependencies
```
cyvcf2
numpy
pandas
matplotlib
seaborn
```

Note: cyvcf2 requires htslib. Install via:
```bash
pip install cyvcf2
# or
conda install -c bioconda cyvcf2
```

## Verification
1. Run all cells in the notebook
2. Confirm unit tests pass
3. Verify visualizations render correctly
4. Compare output with original manual parser to ensure consistency
