# DecodeME AlphaGenome Analysis Pipeline

A robust, modular pipeline for post-GWAS functional characterization using **AlphaGenome**. This tool takes summary statistics (REGENIE format), identifies significant loci, refines them using LD information and Bayesian fine-mapping, and predicts functional effects (RNA expression, chromatin accessibility, splicing) in relevant tissues.

## Features

*   **Robust Clumping:** Identifies independent genomic loci using distance-based clumping with deterministic sorting.
*   **LD-Informed Fine-Mapping:** Integrates with **LDlink API** to filter variants based on genetic linkage ($R^2 \ge 0.1$) rather than just physical proximity.
*   **Two-Tier Fallback:** If LD data is unavailable (e.g., rare variants missing from 1000G), the pipeline automatically:
    1.  Attempts to use nearby "Neighbor Proxies".
    2.  Falls back to a reduced physical window (±50kb) to maintain signal focus.
*   **Multimodal Scoring:** Queries AlphaGenome for RNA-Seq, ATAC-Seq, and Splicing predictions.
*   **Tissue Specificity:** Automatically filters results for **Brain**, **Blood**, and **T-Cells** (UBERON/CL ontologies).
*   **Performance:** Uses `awk` for high-speed file scanning and implements aggressive caching for API calls and regional data.

## Setup

### Prerequisites
*   Python 3.8+
*   Dependencies: `pandas`, `numpy`, `requests`, `tqdm`, `scipy`
*   **AlphaGenome** Python library installed and authenticated.

### Environment Variables
You must set the following API keys in your environment:

```bash
# Required for functional annotation
export ALPHAGENOME_API_KEY="your_alphagenome_key"

# Optional (highly recommended) for LD-informed fine-mapping
export LDLINK_API_TOKEN="your_ldlink_token"
```

## Usage

### Basic Run
Run the pipeline with default settings (interactive mode). It will scan the GWAS file, show identified loci, and ask you which to analyze.

```bash
python3 analyze_decodeme.py
```

### Targeted Analysis
Run analysis on a specific chromosome or region to save time or focus on a hit.

```bash
# Analyze only Chromosome 6
python3 analyze_decodeme.py --chrom 6

# Analyze a specific region
python3 analyze_decodeme.py --chrom 6 --start 25000000 --end 35000000
```

### Custom Thresholds & Configuration
Adjust the statistical stringency or window sizes.

```bash
python3 analyze_decodeme.py 
  --sig-threshold 8.0 
  --window 1000000 
  --credible-set 0.99 
  --output my_custom_analysis
```

### Automated / Batch Mode
Run non-interactively (processes all loci automatically) and disable LDlink if needed.

```bash
python3 analyze_decodeme.py --non-interactive --no-ldlink
```

## Pipeline Steps

1.  **Lead Identification:** Scans the input summary statistics (`--gwas`) for variants exceeding the significance threshold (`--sig-threshold`). Uses `awk` for memory efficiency.
2.  **Clumping:** Groups significant variants into independent loci (peaks) based on physical distance (`--window`).
3.  **Regional Extraction:** For each locus, extracts all variants within the window.
4.  **LD Filtering (Tiered):**
    *   *Primary:* Queries LDlink for variants in LD ($R^2 \ge 0.1$) with the lead SNP.
    *   *Fallback 1:* If lead is missing, tries nearby neighbors as proxies.
    *   *Fallback 2:* If all else fails, reduces the window to ±50kb (`--fallback-window`) to focus on the immediate vicinity.
5.  **Fine-Mapping:** Calculates Approximate Bayes Factors (ABF) and defines a **95% Credible Set** (`--credible-set`) of likely causal variants.
6.  **Annotation:** Queries AlphaGenome to predict functional impacts for every variant in the credible set.
7.  **Filtering:** Filters predictions for relevant tissues (Brain, Blood, Immune) and saves to CSV.

## Caching
The pipeline creates two cache directories to speed up subsequent runs:
*   `region_cache/`: Stores extracted GWAS summary statistics for specific windows.
*   `ld_cache/`: Stores LDlink API responses.
*   `*_leads.csv`: Caches the initial scan of significant hits.

To reset the analysis completely, you can delete these files/directories.
