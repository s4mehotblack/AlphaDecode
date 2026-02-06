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
from pathlib import Path

# --- Configuration Defaults ---
DEFAULT_GWAS_FILE = 'DecodeME-SummaryStatistics/gwas_1.regenie'
DEFAULT_QC_FILE = 'DecodeME-SummaryStatistics/gwas_qced.var'
DEFAULT_OUTPUT_BASE = 'decodeme_results'
DEFAULT_SIG_THRESHOLD = 7.3  # LOG10P (approx 5e-8)
DEFAULT_WINDOW_BP = 500_000  # +/- 500kb
DEFAULT_FALLBACK_WINDOW_BP = 50_000 # +/- 50kb
DEFAULT_CREDIBLE_THRESHOLD = 0.95
DEFAULT_CACHE_DIR = 'region_cache'
DEFAULT_LD_CACHE_DIR = 'ld_cache'

# --- LDLink Client ---
class LDLinkClient:
    def __init__(self, token, cache_dir):
        self.token = token
        self.base_url = "https://ldlink.nih.gov/LDlinkRest/ldproxy"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_proxies(self, chrom, pos, pop="EUR", r2_threshold=0.1):
        """
        Fetches proxies for a variant using GRCh38 coordinates.
        Checks local cache before querying API.
        """
        # 1. Check Cache
        chrom_clean = str(chrom).replace('chr', '')
        cache_filename = os.path.join(self.cache_dir, f"proxies_chr{chrom_clean}_{pos}_{pop}.csv")
        
        if os.path.exists(cache_filename):
            print(f"Loading LD proxies from cache: {cache_filename}")
            df = pd.read_csv(cache_filename)
            # Ensure alleles are re-parsed as tuples (CSV loads them as strings)
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

            # Normalize coordinates back to GWAS format
            df[['chr_str', 'GENPOS']] = df['Coord'].str.split(':', expand=True)
            df['CHROM'] = df['chr_str'].str.replace('chr', '', case=False)
            df['GENPOS'] = df['GENPOS'].astype(int)
            
            # Parse alleles "(A/G)" into a sorted tuple for robust matching
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
except ImportError:
    ag = None

# --- Module 1: GWAS Processor ---
class GWASProcessor:
    def __init__(self, gwas_path, qc_path, cache_dir):
        self.gwas_path = gwas_path
        self.qc_path = qc_path
        self.cache_dir = cache_dir
        self.valid_ids = None # Loaded on demand
        
        # Derive a dataset-specific cache prefix for leads
        gwas_stem = Path(gwas_path).stem
        self.leads_cache_file = os.path.join(self.cache_dir, f"{gwas_stem}_leads.csv")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_qc_list(self):
        if self.valid_ids is not None:
            return self.valid_ids
            
        if not self.qc_path or not os.path.exists(self.qc_path):
            print("QC file not found. Skipping QC filtering.")
            self.valid_ids = set()
            return self.valid_ids
            
        print(f"Loading QC list from {self.qc_path}...")
        try:
            qc_df = pd.read_csv(self.qc_path, header=0, usecols=['ID'])
            self.valid_ids = set(qc_df['ID'])
            return self.valid_ids
        except Exception as e:
            print(f"Error loading QC list: {e}")
            self.valid_ids = set()
            return self.valid_ids

    def get_significant_hits(self, threshold, skip_qc=False, chrom_filter=None, start_filter=None, end_filter=None):
        """Pass 1: Identify significant hits. Supports filtering by subsection."""
        df = pd.DataFrame()
        
        # 1. Load or Generate Genome-Wide Cache
        if os.path.exists(self.leads_cache_file):
            print(f"Loading significant hits from cache: {self.leads_cache_file}")
            df = pd.read_csv(self.leads_cache_file)
            if 'LOG10P' not in df.columns:
                df = pd.read_csv(self.leads_cache_file, sep=r'\s+')
        else:
            print(f"Scanning for significant hits (LOG10P >= {threshold}) using awk...")
            cmd = f"awk 'NR==1 || $16 >= {threshold}' {self.gwas_path}"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    # Note: We save the FULL significant set to cache, WITHOUT QC filtering yet.
                    # This allows toggling QC filtering without re-scanning 1.1GB.
                    df = pd.read_csv(io.StringIO(result.stdout), sep=r'\s+')
                    if not df.empty:
                        df.to_csv(self.leads_cache_file, index=False)
                        print(f"Found {len(df)} significant hits (Genome-wide). Cache saved.")
                else:
                    print(f"Awk failed: {result.stderr}")
            except Exception as e:
                print(f"Error in awk scan: {e}")
        
        if df.empty: return df

        # 2. Apply QC Filter (Unless skipped)
        if not skip_qc:
            valid_ids = self.load_qc_list()
            if valid_ids:
                count_before = len(df)
                df = df[df['ID'].isin(valid_ids)]
                print(f"QC Filter applied: {count_before} -> {len(df)} variants retained.")
            else:
                print("Warning: QC list empty or missing. Skipping QC filter.")

        # 3. Apply Subsection Filters (Last)
        if chrom_filter:
            print(f"Filtering hits for Chromosome {chrom_filter}...")
            # Robust filtering for string/int and extra spaces
            df = df[df['CHROM'].astype(str).str.strip() == str(chrom_filter).strip()]
        
        if start_filter:
            df = df[df['GENPOS'] >= start_filter]
        if end_filter:
            df = df[df['GENPOS'] <= end_filter]

        return df

    def get_region_variants(self, chrom, start, end):
        """Pass 2: Extract all variants in a window using fast AWK scan."""
        region_cache_file = os.path.join(self.cache_dir, f"region_{chrom}_{start}_{end}.csv")
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
                    # Note: We keep original regional behavior (caching results)
                    # We don't apply QC filter *inside* the cache file to keep it reusable
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
    def clump_hits(df, window):
        print(f"Performing distance-based clumping (window: {window}bp)...")
        if df.empty: return df
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
        if region_df.empty or proxies_df.empty: return pd.DataFrame()

        region_df = region_df.copy()
        region_df['GWAS_ALLELES'] = region_df[['ALLELE0', 'ALLELE1']].apply(
            lambda x: tuple(sorted([str(val).upper() for val in x])), axis=1
        )
        
        region_df['CHROM'] = region_df['CHROM'].astype(str)
        proxies_df['CHROM'] = proxies_df['CHROM'].astype(str)
        
        merged = pd.merge(region_df, proxies_df, on=['CHROM', 'GENPOS'])
        if merged.empty: return merged

        matched = merged[merged['GWAS_ALLELES'] == merged['LD_ALLELES']].copy()
        return matched

    @staticmethod
    def get_ld_data_with_fallback(ld_client, chrom, lead_pos, region_df, neighbor_radius_bp=10000, max_neighbors=5):
        proxies = ld_client.get_proxies(chrom, lead_pos)
        if not proxies.empty:
            return proxies, "Lead SNP"

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

        return pd.DataFrame(), "Physical Window (Distance Fallback)"

# --- Module 3: Fine Mapper ---
class FineMapper:
    @staticmethod
    def calculate_abf(beta, se, prior_variance=0.04):
        z = beta / se
        v = se**2
        r = prior_variance / (prior_variance + v)
        abf = np.sqrt(1 - r) * np.exp((z**2 / 2) * r)
        return abf

    @staticmethod
    def define_credible_set(df_region, threshold):
        if df_region.empty: return pd.DataFrame()
        df = df_region.copy()
        df['ABF'] = df.apply(lambda row: FineMapper.calculate_abf(row['BETA'], row['SE']), axis=1)
        sum_abf = df['ABF'].sum()
        df['PP'] = df['ABF'] / sum_abf
        df = df.sort_values('PP', ascending=False)
        df['CUM_PP'] = df['PP'].cumsum()
        n_variants = len(df[df['CUM_PP'] < threshold]) + 1
        return df.head(n_variants)

# --- Module 4: Functional Annotator ---
class FunctionalAnnotator:
    def __init__(self, api_key):
        self.client = None
        self.organism = None
        self.scorers = []
        if not api_key: return
        if ag:
            try:
                self.client = dna_client.create(api_key=api_key)
                self.organism = dna_client.Organism.HOMO_SAPIENS
                print("AlphaGenome Client initialized.")
                all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
                if 'RNA_SEQ' in all_scorers: self.scorers.append(all_scorers['RNA_SEQ'])
                if 'ATAC_ACTIVE' in all_scorers: self.scorers.append(all_scorers['ATAC_ACTIVE'])
                if 'SPLICE_JUNCTIONS' in all_scorers: self.scorers.append(all_scorers['SPLICE_JUNCTIONS'])
            except Exception as e:
                print(f"Failed to initialize AlphaGenome client: {e}")

    def annotate_variants(self, df):
        if not self.client or df.empty: return pd.DataFrame()
        results = []
        print(f"Annotating {len(df)} variants in Credible Set with AlphaGenome...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                chrom = f"chr{row['CHROM']}" if not str(row['CHROM']).startswith('chr') else str(row['CHROM'])
                variant = genome.Variant(chromosome=chrom, position=int(row['GENPOS']),
                                         reference_bases=row['ALLELE0'], alternate_bases=row['ALLELE1'],
                                         name=row['ID'])
                interval = variant.reference_interval.resize(1048576)
                res = self.client.score_variant(interval=interval, variant=variant,
                                              variant_scorers=self.scorers, organism=self.organism)
                results.append(res)
            except Exception as e:
                print(f"Error annotating {row['ID']}: {e}")
        if results:
            try: return variant_scorers.tidy_scores(results)
            except Exception as e: print(f"Error tidying scores: {e}")
        return pd.DataFrame()

    def filter_by_ontology(self, df_scores):
        if df_scores.empty: return df_scores
        target_ontologies = ['UBERON:0000955', 'UBERON:0000178', 'CL:0000084']
        if 'ontology_curie' in df_scores.columns:
            return df_scores[df_scores['ontology_curie'].isin(target_ontologies)]
        return df_scores

def get_final_filename(base_name):
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
            print("Exiting."); sys.exit(0)
    return filename

# --- Main Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="DecodeME AlphaGenome Analysis Pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gwas", default=DEFAULT_GWAS_FILE, help="Path to GWAS summary stats")
    parser.add_argument("--qc", default=DEFAULT_QC_FILE, help="Path to QC list")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_BASE, help="Output base name")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Region cache dir")
    parser.add_argument("--ld-cache-dir", default=DEFAULT_LD_CACHE_DIR, help="LD cache dir")
    parser.add_argument("--sig-threshold", type=float, default=DEFAULT_SIG_THRESHOLD, help="LOG10P threshold")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_BP, help="Window size (bp)")
    parser.add_argument("--fallback-window", type=int, default=DEFAULT_FALLBACK_WINDOW_BP, help="Reduced window size (bp)")
    parser.add_argument("--credible-set", type=float, default=DEFAULT_CREDIBLE_THRESHOLD, help="Credible set threshold")
    parser.add_argument("--chrom", help="Filter by Chromosome")
    parser.add_argument("--start", type=int, help="Filter start position")
    parser.add_argument("--end", type=int, help="Filter end position")
    parser.add_argument("--no-ldlink", action="store_true", help="Disable LDlink")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC filtering (Use with caution!)")
    parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode")
    args = parser.parse_args()

    ag_key = os.environ.get('ALPHAGENOME_API_KEY')
    ld_token = os.environ.get('LDLINK_API_TOKEN')

    print("--- Starting DecodeME Analysis Pipeline ---")
    processor = GWASProcessor(args.gwas, args.qc, args.cache_dir)
    ld_client = LDLinkClient(ld_token, args.ld_cache_dir) if ld_token and not args.no_ldlink else None
    
    leads_df = processor.get_significant_hits(threshold=args.sig_threshold, skip_qc=args.skip_qc,
                                              chrom_filter=args.chrom, start_filter=args.start, end_filter=args.end)
    print(f"Leads loaded: {len(leads_df)} rows.")
    if leads_df.empty: return
        
    clumps_df = LDEngine.clump_hits(leads_df, window=args.window)
    print(f"\nIdentified {len(clumps_df)} independent genomic loci.")
    if clumps_df.empty: return

    print("Calculating Credible Set sizes...")
    cs_sizes, credible_sets = [], {}
    
    for idx, row in tqdm(clumps_df.iterrows(), total=len(clumps_df), desc="Analyzing Loci"):
        start, end = int(row['GENPOS'] - args.window), int(row['GENPOS'] + args.window)
        region_df = processor.get_region_variants(row['CHROM'], start, end)
        if region_df.empty:
            cs_sizes.append(0); continue
        
        # Apply QC to regional variants if not skipped
        if not args.skip_qc:
            valid_ids = processor.load_qc_list()
            if valid_ids:
                region_df = region_df[region_df['ID'].isin(valid_ids)]
        
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
                    fine_mapping_input = region_df[(region_df['GENPOS'] >= row['GENPOS'] - args.fallback_window) & 
                                                   (region_df['GENPOS'] <= row['GENPOS'] + args.fallback_window)]
            else:
                fine_mapping_input = region_df[(region_df['GENPOS'] >= row['GENPOS'] - args.fallback_window) & 
                                               (region_df['GENPOS'] <= row['GENPOS'] + args.fallback_window)]

        credible_set = FineMapper.define_credible_set(fine_mapping_input, args.credible_set)
        cs_sizes.append(len(credible_set))
        credible_sets[idx] = credible_set
    
    clumps_df['CS_Size'] = cs_sizes
    valid_mask = clumps_df['CS_Size'] > 0
    if not valid_mask.any():
        print("No loci with >0 credible variants found. Exiting."); return

    final_credible_sets = {i: credible_sets[old_idx] for i, old_idx in enumerate(clumps_df.index[valid_mask])}
    clumps_df = clumps_df[valid_mask].reset_index(drop=True)

    print("\nSummary of Identified Loci:")
    print(clumps_df[['CHROM', 'GENPOS', 'ID', 'LOG10P', 'CS_Size']].head(20))
    
    selected_indices = []
    if args.non_interactive:
        selected_indices = clumps_df.index.tolist()
        output_base = f"{args.output}_all"
    else:
        selection = input("\nEnter locus index to analyze, 'all' for all, or 'q' to quit: ").strip().lower()
        if selection == 'q': return
        elif selection != 'all':
            try:
                idx = int(selection)
                if 0 <= idx < len(clumps_df):
                    selected_indices = [idx]
                    output_base = f"{args.output}_locus_{idx}"
                else:
                     print("Invalid index."); return
            except ValueError: return
        else:
            selected_indices = clumps_df.index.tolist()
            output_base = f"{args.output}_all"

    output_filename = get_final_filename(output_base)
    annotator, final_results = FunctionalAnnotator(ag_key), []
    
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
        print(f"\nPipeline Complete. Saved to {output_filename}")
    else:
        print("\nAnalysis produced no results.")

if __name__ == "__main__":
    main()