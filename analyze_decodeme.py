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
LEADS_CACHE_FILE = 'decodeme_leads_cache.csv'
CACHE_DIR = 'region_cache'
LD_CACHE_DIR = 'ld_cache'
API_KEY = os.environ.get('ALPHAGENOME_API_KEY')
LDLINK_TOKEN = os.environ.get('LDLINK_API_TOKEN')

# Global Constants
SIGNIFICANCE_THRESHOLD = 7.3  # LOG10P
WINDOW_BP = 500_000 # +/- 500kb
CREDIBLE_SET_THRESHOLD = 0.95

# Ensure cache directories exist
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
        Checks local cache before querying API.
        """
        # 1. Check Cache
        chrom_clean = str(chrom).replace('chr', '')
        cache_filename = os.path.join(LD_CACHE_DIR, f"proxies_chr{chrom_clean}_{pos}_{pop}.csv")
        
        if os.path.exists(cache_filename):
            print(f"Loading LD proxies from cache: {cache_filename}")
            df = pd.read_csv(cache_filename)
            # Fix: Re-parse alleles to ensure they are tuples (read_csv makes them strings)
            if 'Alleles' in df.columns:
                 df['LD_ALLELES'] = df['Alleles'].str.strip('()').str.upper().str.split('/')\
                                    .apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else None)
            return df

        # 2. Prepare API Request
        variant = f"chr{chrom_clean}:{pos}"
        params = {
            "var": variant,
            "pop": pop,
            "r2_d": "r2",
            "genome_build": "grch38",
            "token": self.token
        }
        
        # Rate Limiting
        time.sleep(1.1) 
        
        try:
            print(f"Querying LDlink for {variant}...")
            response = requests.get(self.base_url, params=params)
            if response.status_code != 200:
                print(f"LDlink Error {response.status_code}: {response.text}")
                return pd.DataFrame()

            if "error" in response.text.lower() and "\t" not in response.text:
                 print(f"LDlink returned error message: {response.text}")
                 return pd.DataFrame()

            df = pd.read_csv(io.StringIO(response.text), sep='\t')
            
            if 'Coord' not in df.columns:
                print("LDlink response missing 'Coord' column.")
                return pd.DataFrame()

            # Normalize coordinates
            df[['chr_str', 'GENPOS']] = df['Coord'].str.split(':', expand=True)
            df['CHROM'] = df['chr_str'].str.replace('chr', '', case=False)
            df['GENPOS'] = df['GENPOS'].astype(int)
            
            # Fix: Parse alleles "(A/G)" into a sorted tuple for robust matching
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
            df = pd.read_csv(LEADS_CACHE_FILE)
            if 'LOG10P' not in df.columns:
                df = pd.read_csv(LEADS_CACHE_FILE, sep=r'\s+')
        else:
            print(f"Scanning for significant hits (LOG10P >= {threshold}) using awk...")
            cmd = f"awk 'NR==1 || $16 >= {threshold}' {self.gwas_path}"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    df = pd.read_csv(io.StringIO(result.stdout), sep=r'\s+')
                    if not df.empty and self.valid_ids:
                        df = df[df['ID'].isin(self.valid_ids)]
                    if not df.empty:
                        df.to_csv(LEADS_CACHE_FILE, index=False)
                        print(f"Found {len(df)} significant QC-passed hits.")
                else:
                    print(f"Awk failed: {result.stderr}")
            except Exception as e:
                print(f"Error in awk scan: {e}")
        
        if df.empty: return df

        # Apply Subsection Filters
        if chrom_filter:
            print(f"Filtering hits for Chromosome {chrom_filter}...")
            df = df[df['CHROM'].astype(str) == str(chrom_filter)]
        if start_filter:
            df = df[df['GENPOS'] >= start_filter]
        if end_filter:
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
        cmd = f"awk 'NR==1 || ($1 == \"{target_chrom}\" && $2 >= {start} && $2 <= {end})' {self.gwas_path}"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                df = pd.read_csv(io.StringIO(result.stdout), sep=r'\s+')
                if not df.empty:
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
        # Fix: Deterministic sorting by adding ID as secondary key
        df = df.sort_values(by=['LOG10P', 'ID'], ascending=[False, True])
        clumps = []
        candidates = df.copy()
        while not candidates.empty:
            lead = candidates.iloc[0]
            clumps.append(lead)
            pos, chrom = lead['GENPOS'], lead['CHROM']
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

        # Prepare GWAS alleles
        region_df = region_df.copy()
        region_df['GWAS_ALLELES'] = region_df[['ALLELE0', 'ALLELE1']].apply(
            lambda x: tuple(sorted([str(val).upper() for val in x])), axis=1
        )
        
        region_df['CHROM'] = region_df['CHROM'].astype(str)
        proxies_df['CHROM'] = proxies_df['CHROM'].astype(str)
        
        # Merge on position
        merged = pd.merge(region_df, proxies_df, on=['CHROM', 'GENPOS'])
        if merged.empty: return merged

        # Filter for allele consistency
        matched = merged[merged['GWAS_ALLELES'] == merged['LD_ALLELES']].copy()
        return matched

    @staticmethod
    def get_ld_data_with_fallback(ld_client, chrom, lead_pos, region_df, neighbor_radius_bp=10000, max_neighbors=5):
        """
        Tries to get LD data for lead SNP. If fails (e.g. missing in 1000G), 
        tries nearby significant neighbors as proxies.
        Returns: (proxies_df, strategy_description)
        """
        # Attempt 1: Lead SNP
        proxies = ld_client.get_proxies(chrom, lead_pos)
        if not proxies.empty:
            return proxies, "Lead SNP"

        # Attempt 2: Neighbor Proxy
        if not region_df.empty:
            mask = (region_df['GENPOS'] >= lead_pos - neighbor_radius_bp) & \
                   (region_df['GENPOS'] <= lead_pos + neighbor_radius_bp) & \
                   (region_df['GENPOS'] != lead_pos)
            
            neighbors = region_df[mask].sort_values(by='LOG10P', ascending=False).head(max_neighbors)
            
            for _, row in neighbors.iterrows():
                print(f"  > Lead missing/failed. Trying neighbor {row['ID']} (P={row['LOG10P']:.2f})...")
                proxies = ld_client.get_proxies(row['CHROM'], row['GENPOS'])
                if not proxies.empty:
                    return proxies, f"Neighbor Proxy ({row['ID']})"

        # Attempt 3: Fallback
        return pd.DataFrame(), "Physical Window (Distance Fallback)"

# --- Module 3: Fine Mapper ---
class FineMapper:
# ... (rest of file) ...

# --- Main Pipeline ---
def main():
    # ... (arg parsing and setup) ...
    parser = argparse.ArgumentParser(description="DecodeME AlphaGenome Analysis Pipeline")
    parser.add_argument("--chrom", help="Filter by Chromosome (e.g., 1)")
    parser.add_argument("--start", type=int, help="Filter start position (bp)")
    parser.add_argument("--end", type=int, help="Filter end position (bp)")
    args = parser.parse_args()

    print("Starting pipeline...")
    processor = GWASProcessor(GWAS_FILE, QC_LIST_FILE)
    ld_client = LDLinkClient(LDLINK_TOKEN) if LDLINK_TOKEN else None
    if not ld_client: print("LDLINK_API_TOKEN not set. Using physical window analysis.")

    leads_df = processor.get_significant_hits(chrom_filter=args.chrom, start_filter=args.start, end_filter=args.end)
    print(f"Leads loaded: {len(leads_df)} rows.")
    if leads_df.empty: return
        
    clumps_df = LDEngine.clump_hits(leads_df)
    print(f"\nIdentified {len(clumps_df)} independent genomic loci.")
    
    print("Calculating Credible Set sizes for each locus...")
    cs_sizes, credible_sets = [], {}
    
    for idx, row in tqdm(clumps_df.iterrows(), total=len(clumps_df), desc="Analyzing Loci"):
        start, end = int(row['GENPOS'] - WINDOW_BP), int(row['GENPOS'] + WINDOW_BP)
        region_df = processor.get_region_variants(row['CHROM'], start, end)
        if region_df.empty:
            cs_sizes.append(0); continue
        
        fine_mapping_input = region_df
        if ld_client:
            proxies, strategy = LDEngine.get_ld_data_with_fallback(ld_client, row['CHROM'], row['GENPOS'], region_df)
            
            if not proxies.empty:
                ld_filtered = LDEngine.merge_gwas_with_ld(region_df, proxies)
                if not ld_filtered.empty:
                    fine_mapping_input = ld_filtered
                else:
                    print(f"  Warning: LD filtering for {row['ID']} resulted in 0 variants (Allele mismatch?). Strategy: {strategy}. Falling back to physical window.")
            else:
                print(f"  Warning: No LD data for {row['ID']} after fallback attempts. Strategy: {strategy}.")

        credible_set = FineMapper.define_credible_set(fine_mapping_input)
        cs_sizes.append(len(credible_set))
        credible_sets[idx] = credible_set
    
    clumps_df['CS_Size'] = cs_sizes
# ... (rest of main) ...
    valid_mask = clumps_df['CS_Size'] > 0
    if not valid_mask.any():
        print("No loci with >0 credible variants found. Exiting."); return

    final_credible_sets = {i: credible_sets[old_idx] for i, old_idx in enumerate(clumps_df.index[valid_mask])}
    clumps_df = clumps_df[valid_mask].reset_index(drop=True)

    print("\nSummary of Identified Loci:")
    print(clumps_df[['CHROM', 'GENPOS', 'ID', 'LOG10P', 'CS_Size']].head(20))
    
    selection = input("\nEnter locus index to analyze, 'all' for all, or 'q' to quit: ").strip().lower()
    if selection == 'q': return
    elif selection != 'all':
        try:
            idx = int(selection)
            if 0 <= idx < len(clumps_df):
                selected_indices = [idx]
                output_base = f"decodeme_locus_{idx}"
            else:
                 print(f"Invalid index. Please select between 0 and {len(clumps_df)-1}."); return
        except ValueError:
            print("Invalid input. Please enter a number, 'all', or 'q'."); return
    else:
        selected_indices = clumps_df.index.tolist()
        output_base = "decodeme_all_loci"

    output_filename = get_final_filename(output_base)
    annotator, final_results = FunctionalAnnotator(), []
    
    for idx in selected_indices:
        row = clumps_df.iloc[idx]
        credible_set = final_credible_sets.get(idx)
        if credible_set is None or credible_set.empty: continue
            
        print(f"\n--- Running AlphaGenome for Locus {row['ID']} ({len(credible_set)} variants) ---")
        scores = annotator.annotate_variants(credible_set)
        filtered_scores = annotator.filter_by_ontology(scores)
        if not filtered_scores.empty:
            filtered_scores = filtered_scores.copy()
            filtered_scores.loc[:, 'Locus_ID'] = row['ID']
            final_results.append(filtered_scores)
    
    if final_results:
        final_df = pd.concat(final_results, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print(f"\nPipeline Complete. Multimodal results saved to {output_filename}")
    else:
        print("\nAnalysis produced no results after ontology filtering.")

if __name__ == "__main__":
    main()