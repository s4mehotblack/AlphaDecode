import pandas as pd
import numpy as np
import os
import subprocess
import io
import requests
import time
import sys
import argparse
from tqdm import tqdm
from scipy.stats import norm

# --- Configuration ---
GWAS_FILE = 'DecodeME-SummaryStatistics/gwas_1.regenie'
QC_LIST_FILE = 'DecodeME-SummaryStatistics/gwas_qced.var'
OUTPUT_FILE = 'decodeme_alphagenome_results.csv'
LEADS_CACHE_FILE = 'decodeme_leads_cache.csv'
CACHE_DIR = 'region_cache'
LD_CACHE_DIR = 'ld_cache'
API_KEY = os.environ.get('ALPHAGENOME_API_KEY')
LDLINK_TOKEN = os.environ.get('LDLINK_API_TOKEN')

# Global Constants
SIGNIFICANCE_THRESHOLD = 7.3  # LOG10P
WINDOW_BP = 500_000 # +/- 500kb
CREDIBLE_SET_THRESHOLD = 0.95

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LD_CACHE_DIR, exist_ok=True)

# --- LDLink Client ---
class LDLinkClient:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://ldlink.nih.gov/LDlinkRest/ldproxy"

    def get_proxies(self, chrom, pos, pop="EUR", r2_threshold=0.1):
        """
        Fetches proxies for a variant using GRCh38 coordinates.
        Addresses: Coordinate normalization and API rate limiting.
        Checks local cache before querying API.
        """
        # 1. Check Cache
        # Normalize chrom for filename (no 'chr')
        chrom_clean = str(chrom).replace('chr', '')
        cache_filename = os.path.join(LD_CACHE_DIR, f"proxies_chr{chrom_clean}_{pos}_{pop}.csv")
        
        if os.path.exists(cache_filename):
            print(f"Loading LD proxies from cache: {cache_filename}")
            return pd.read_csv(cache_filename)

        # 2. Prepare API Request
        # Fix: Prepend 'chr' for LDlink GRCh38 requirements
        # LDlink expects variants like "chr1:12345" for GRCh38
        variant = f"chr{chrom_clean}:{pos}"
        
        params = {
            "var": variant,
            "pop": pop,
            "r2_d": "r2",
            "genome_build": "grch38", # Non-negotiable for DecodeME
            "token": self.token
        }
        
        # Rate Limiting: LDlink prefers sequential requests (~1s delay)
        time.sleep(1.1) 
        
        try:
            print(f"Querying LDlink for {variant}...")
            response = requests.get(self.base_url, params=params)
            if response.status_code != 200:
                print(f"LDlink Error {response.status_code}: {response.text}")
                return pd.DataFrame()

            # LDlink can return error messages in text even with 200 OK sometimes
            if "error" in response.text.lower() and "\t" not in response.text:
                 print(f"LDlink returned error message: {response.text}")
                 return pd.DataFrame()

            df = pd.read_csv(io.StringIO(response.text), sep='\t')
            
            # Check for expected columns
            if 'Coord' not in df.columns:
                print("LDlink response missing 'Coord' column.")
                return pd.DataFrame()

            # Fix: Normalize coordinates back to GWAS format (strip 'chr' and split)
            # Coord format: chr1:12345
            df[['chr_str', 'GENPOS']] = df['Coord'].str.split(':', expand=True)
            df['CHROM'] = df['chr_str'].str.replace('chr', '', case=False)
            df['GENPOS'] = df['GENPOS'].astype(int)
            
            # Fix: Parse alleles "(A/G)" into a sorted tuple for robust matching
            # Handle cases where Alleles might be missing or malformed
            if 'Alleles' in df.columns:
                 df['LD_ALLELES'] = df['Alleles'].str.strip('()').str.upper().str.split('/')\
                                    .apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else None)
            
            # Filter by R2
            df_filtered = df[df['R2'] >= r2_threshold]
            
            # 3. Save to Cache
            if not df_filtered.empty:
                df_filtered.to_csv(cache_filename, index=False)
            
            return df_filtered
        except Exception as e:
            print(f"LDlink Request failed: {e}")
            return pd.DataFrame()

# --- AlphaGenome Setup ---
try:
    import alphagenome as ag
    from alphagenome.models import dna_client, variant_scorers
    from alphagenome.data import genome
    
    # Try to initialize client
    if API_KEY:
        AG_CLIENT = dna_client.create(api_key=API_KEY)
        ORGANISM = dna_client.Organism.HOMO_SAPIENS
        print("AlphaGenome Client initialized.")
    else:
        print("Warning: ALPHAGENOME_API_KEY not set.")
        AG_CLIENT = None
        ORGANISM = None

except ImportError:
    print("AlphaGenome library not found. Functional Annotation will be skipped.")
    ag = None
    AG_CLIENT = None

# --- Module 1: GWAS Processor ---
class GWASProcessor:
    def __init__(self, gwas_path, qc_path):
        self.gwas_path = gwas_path
        self.qc_path = qc_path
        self.valid_ids = self._load_qc_list()

    def _load_qc_list(self):
        print(f"Loading QC list from {self.qc_path}...")
        try:
            qc_df = pd.read_csv(self.qc_path, header=0, usecols=['ID'])
            return set(qc_df['ID'])
        except Exception as e:
            print(f"Error loading QC list: {e}")
            return set()

    def get_significant_hits(self, threshold=SIGNIFICANCE_THRESHOLD, 
                             chrom_filter=None, start_filter=None, end_filter=None):
        """Pass 1: Identify genome-wide significant hits using fast AWK scan."""
        df = pd.DataFrame()
        
        if os.path.exists(LEADS_CACHE_FILE):
            print(f"Loading significant hits from cache: {LEADS_CACHE_FILE}")
            # Try reading as CSV, then as space-separated if that fails to find LOG10P
            df = pd.read_csv(LEADS_CACHE_FILE)
            if 'LOG10P' not in df.columns:
                df = pd.read_csv(LEADS_CACHE_FILE, sep=r'\s+')
        else:
            print(f"Scanning for significant hits (LOG10P >= {threshold}) using awk...")
            # LOG10P is column 16. Using awk for speed.
            cmd = f"awk 'NR==1 || $16 >= {threshold}' {self.gwas_path}"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    df = pd.read_csv(io.StringIO(result.stdout), sep=r'\s+')
                    # Apply QC filter if needed
                    if not df.empty and self.valid_ids:
                        df = df[df['ID'].isin(self.valid_ids)]
                    
                    if not df.empty:
                        df.to_csv(LEADS_CACHE_FILE, index=False)
                        print(f"Found {len(df)} significant QC-passed hits.")
                else:
                    print(f"Awk failed: {result.stderr}")
            except Exception as e:
                print(f"Error in awk scan: {e}")
        
        if df.empty:
            return df

        # Apply Subsection Filters
        if chrom_filter:
            print(f"Filtering hits for Chromosome {chrom_filter}...")
            # Ensure filtering works for string '1' and int 1
            df = df[df['CHROM'].astype(str) == str(chrom_filter)]
        
        if start_filter:
            print(f"Filtering hits >= {start_filter}...")
            df = df[df['GENPOS'] >= start_filter]
            
        if end_filter:
            print(f"Filtering hits <= {end_filter}...")
            df = df[df['GENPOS'] <= end_filter]

        return df

    def get_region_variants(self, chrom, start, end):
        """Pass 2: Extract all variants in a window using fast AWK scan."""
        region_cache_file = os.path.join(CACHE_DIR, f"region_{chrom}_{start}_{end}.csv")
        if os.path.exists(region_cache_file):
            print(f"Loading region from cache: {region_cache_file}")
            return pd.read_csv(region_cache_file)

        target_chrom = str(chrom).replace('chr', '')
        print(f"Extracting region {target_chrom}:{start}-{end} using awk...")
        
        # AWK to get header + matching rows
        cmd = f"awk 'NR==1 || ($1 == \"{target_chrom}\" && $2 >= {start} && $2 <= {end})' {self.gwas_path}"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                df = pd.read_csv(io.StringIO(result.stdout), sep=r'\s+')
                if not df.empty:
                    # QC Filter
                    if self.valid_ids:
                        df = df[df['ID'].isin(self.valid_ids)]
                    
                    if not df.empty:
                        df.to_csv(region_cache_file, index=False)
                        return df
            else:
                print(f"Awk failed: {result.stderr}")
        except Exception as e:
            print(f"Error in awk region extraction: {e}")
            
        return pd.DataFrame()

# --- Module 2: LD Engine ---
class LDEngine:
    @staticmethod
    def clump_hits(df, window=WINDOW_BP):
        print("Performing distance-based clumping...")
        if df.empty: return df
        df = df.sort_values(by='LOG10P', ascending=False)
        clumps = []
        candidates = df.copy()
        while not candidates.empty:
            lead = candidates.iloc[0]
            clumps.append(lead)
            pos = lead['GENPOS']
            chrom = lead['CHROM']
            # Remove neighbors on same chrom within window
            candidates = candidates[
                (candidates['CHROM'] != chrom) | 
                (candidates['GENPOS'] < pos - window) | 
                (candidates['GENPOS'] > pos + window)
            ]
        return pd.DataFrame(clumps).reset_index(drop=True)

    @staticmethod
    def merge_gwas_with_ld(region_df, proxies_df):
        """
        Robustly merges local summary stats with LD data.
        Ensures alleles match regardless of REF/ALT orientation.
        """
        if region_df.empty or proxies_df.empty:
            return pd.DataFrame()

        # 1. Prepare GWAS alleles for matching
        # Convert to sorted tuple of strings
        region_df = region_df.copy()
        region_df['GWAS_ALLELES'] = region_df[['ALLELE0', 'ALLELE1']].apply(
            lambda x: tuple(sorted([str(val).upper() for val in x])), axis=1
        )
        
        # 2. Ensure types match for the join
        region_df['CHROM'] = region_df['CHROM'].astype(str)
        proxies_df['CHROM'] = proxies_df['CHROM'].astype(str)
        
        # 3. Merge on position
        # Inner join to keep only variants present in both
        merged = pd.merge(region_df, proxies_df, on=['CHROM', 'GENPOS'])
        
        if merged.empty:
            return merged

        # 4. Filter for allele consistency
        # This keeps only rows where the two nucleotides match the LD pair
        # Handle cases where LD_ALLELES might be None
        matched = merged[merged['GWAS_ALLELES'] == merged['LD_ALLELES']].copy()
        
        return matched

# --- Module 3: Fine Mapper ---
class FineMapper:
    @staticmethod
    def calculate_abf(beta, se, prior_variance=0.04):
        """Calculates Approximate Bayes Factor (Wakefield)."""
        # Z score
        z = beta / se
        # Variance of the estimate
        v = se**2
        # Shrinkage factor
        r = prior_variance / (prior_variance + v)
        # ABF
        abf = np.sqrt(1 - r) * np.exp((z**2 / 2) * r)
        return abf

    @staticmethod
    def define_credible_set(df_region):
        """Defines 95% Credible Set for a locus."""
        if df_region.empty: return pd.DataFrame()
        df = df_region.copy()
        df['ABF'] = df.apply(lambda row: FineMapper.calculate_abf(row['BETA'], row['SE']), axis=1)
        sum_abf = df['ABF'].sum()
        df['PP'] = df['ABF'] / sum_abf
        df = df.sort_values('PP', ascending=False)
        df['CUM_PP'] = df['PP'].cumsum()
        n_variants = len(df[df['CUM_PP'] < CREDIBLE_SET_THRESHOLD]) + 1
        return df.head(n_variants)

# --- Module 4: Functional Annotator ---
class FunctionalAnnotator:
    def __init__(self):
        # Configure Scorer Suite
        self.scorers = []
        if ag and AG_CLIENT:
            all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
            if 'RNA_SEQ' in all_scorers: self.scorers.append(all_scorers['RNA_SEQ'])
            if 'ATAC_ACTIVE' in all_scorers: self.scorers.append(all_scorers['ATAC_ACTIVE'])
            if 'SPLICE_JUNCTIONS' in all_scorers: self.scorers.append(all_scorers['SPLICE_JUNCTIONS'])

    def annotate_variants(self, df):
        if not AG_CLIENT or df.empty: return pd.DataFrame()
        results = []
        print(f"Annotating {len(df)} variants in Credible Set with AlphaGenome...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                chrom = f"chr{row['CHROM']}" if not str(row['CHROM']).startswith('chr') else str(row['CHROM'])
                variant = genome.Variant(chromosome=chrom, position=int(row['GENPOS']),
                                         reference_bases=row['ALLELE0'], alternate_bases=row['ALLELE1'],
                                         name=row['ID'])
                interval = variant.reference_interval.resize(1048576)
                res = AG_CLIENT.score_variant(interval=interval, variant=variant,
                                              variant_scorers=self.scorers, organism=ORGANISM)
                results.append(res)
            except Exception as e:
                print(f"Error annotating {row['ID']}: {e}")
        
        if results:
            # Tidy scores
            try:
                return variant_scorers.tidy_scores(results)
            except Exception as e:
                print(f"Error tidying scores: {e}")
        return pd.DataFrame()

    def filter_by_ontology(self, df_scores):
        """Filters results for relevant tissues (Brain, Blood, T-Cells)."""
        if df_scores.empty: return df_scores
        target_ontologies = ['UBERON:0000955', 'UBERON:0000178', 'CL:0000084']
        if 'ontology_curie' in df_scores.columns:
            return df_scores[df_scores['ontology_curie'].isin(target_ontologies)]
        return df_scores

def get_final_filename(base_name):
    """Handles filename collisions by prompting for overwrite or timestamp."""
    filename = f"{base_name}.csv"
    if os.path.exists(filename):
        print(f"\nWarning: File '{filename}' already exists.")
        choice = input("Enter '1' to add timestamp, '2' to overwrite, or 'q' to quit: ").strip().lower()
        if choice == '1':
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{base_name}_{timestamp}.csv"
        elif choice == '2':
            print(f"Overwriting {filename}...")
        else:
            print("Exiting.")
            sys.exit(0)
    return filename

# --- Main Pipeline ---
def main():
    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="DecodeME AlphaGenome Analysis Pipeline")
    parser.add_argument("--chrom", help="Filter by Chromosome (e.g., 1)")
    parser.add_argument("--start", type=int, help="Filter start position (bp)")
    parser.add_argument("--end", type=int, help="Filter end position (bp)")
    args = parser.parse_args()

    print("Starting pipeline...")
    processor = GWASProcessor(GWAS_FILE, QC_LIST_FILE)
    
    # Initialize LD Client if token available
    ld_client = None
    if LDLINK_TOKEN:
        ld_client = LDLinkClient(LDLINK_TOKEN)
        print("LDlink Client initialized (LD-informed analysis enabled).")
    else:
        print("LDLINK_API_TOKEN not set. Using physical window analysis.")
    
    # 1. Identify Lead SNPs (With Filtering)
    leads_df = processor.get_significant_hits(chrom_filter=args.chrom, 
                                              start_filter=args.start, 
                                              end_filter=args.end)
    print(f"Leads loaded: {len(leads_df)} rows.")
    if leads_df.empty:
        print("No significant hits found matching QC criteria. Exiting.")
        return
        
    # 2. Clump into independent loci
    clumps_df = LDEngine.clump_hits(leads_df)
    print(f"\nIdentified {len(clumps_df)} independent genomic loci.")
    
    if clumps_df.empty:
        print("Clumping resulted in 0 loci. Exiting.")
        return

    # 3. Pre-calculate Credible Set sizes for informed selection
    print("Calculating Credible Set sizes for each locus...")
    cs_sizes = []
    credible_sets = {} # Cache full sets to avoid re-calculating after selection
    
    for idx, row in tqdm(clumps_df.iterrows(), total=len(clumps_df), desc="Analyzing Loci"):
        start, end = int(row['GENPOS'] - WINDOW_BP), int(row['GENPOS'] + WINDOW_BP)
        region_df = processor.get_region_variants(row['CHROM'], start, end)
        
        if region_df.empty:
            cs_sizes.append(0)
            continue
        
        # --- LD Filtering Logic ---
        fine_mapping_input = region_df
        
        if ld_client:
            proxies = ld_client.get_proxies(row['CHROM'], row['GENPOS'])
            if not proxies.empty:
                ld_filtered = LDEngine.merge_gwas_with_ld(region_df, proxies)
                if not ld_filtered.empty:
                    fine_mapping_input = ld_filtered
                else:
                    # Fallback if merger fails but proxies exist
                    pass 
            else:
                 pass
        # --------------------------

        credible_set = FineMapper.define_credible_set(fine_mapping_input)
        cs_sizes.append(len(credible_set))
        credible_sets[idx] = credible_set
    
    print(f"Credible sets calculated. Non-zero counts: {sum(1 for s in cs_sizes if s > 0)}")
    
    clumps_df['CS_Size'] = cs_sizes
    
    # Store sets in a way that aligns with the final DataFrame indices
    valid_mask = clumps_df['CS_Size'] > 0
    
    if not valid_mask.any():
        print("No loci with >0 credible variants found. Exiting.")
        return

    final_credible_sets = {i: credible_sets[old_idx] 
                          for i, old_idx in enumerate(clumps_df.index[valid_mask])}
    
    clumps_df = clumps_df[valid_mask].reset_index(drop=True)
    print(f"Final loci count after filtering empty sets: {len(clumps_df)}")

    print("\nSummary of Identified Loci:")
    print(clumps_df[['CHROM', 'GENPOS', 'ID', 'LOG10P', 'CS_Size']].head(20))
    
    # 4. User Selection
    print(f"\nAlphaGenome annotation can be time-consuming (~1-2s per variant).")
    selection = input("Enter locus index to analyze, 'all' for all, or 'q' to quit: ").strip().lower()
    
    if selection == 'q':
        return
    elif selection != 'all':
        try:
            idx = int(selection)
            if 0 <= idx < len(clumps_df):
                selected_indices = [idx]
                output_base = f"decodeme_locus_{idx}"
            else:
                 print(f"Invalid index. Please select between 0 and {len(clumps_df)-1}.")
                 return
        except ValueError:
            print("Invalid input. Please enter a number, 'all', or 'q'.")
            return
    else:
        selected_indices = clumps_df.index.tolist()
        output_base = "decodeme_all_loci"

    # Resolve filename with collision check
    output_filename = get_final_filename(output_base)

    annotator = FunctionalAnnotator()
    final_results = []
    
    # 5. Execute AlphaGenome Annotation
    for idx in selected_indices:
        row = clumps_df.iloc[idx]
        credible_set = final_credible_sets.get(idx)
        
        if credible_set is None or credible_set.empty:
            continue
            
        print(f"\n--- Running AlphaGenome for Locus {row['ID']} ({len(credible_set)} variants) ---")
        scores = annotator.annotate_variants(credible_set)
        
        # Filter by prioritized ontologies (Brain, Blood, T-Cells)
        filtered_scores = annotator.filter_by_ontology(scores)
        
        if not filtered_scores.empty:
            # Create a copy to avoid SettingWithCopyWarning and add Locus_ID
            filtered_scores = filtered_scores.copy()
            filtered_scores.loc[:, 'Locus_ID'] = row['ID']
            final_results.append(filtered_scores)
    
    # 6. Final Export
    if final_results:
        final_df = pd.concat(final_results, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print(f"\nPipeline Complete. Multimodal results saved to {output_filename}")
    else:
        print("\nAnalysis produced no results after ontology filtering.")

if __name__ == "__main__":
    main()
