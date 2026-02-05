import pandas as pd
import numpy as np
import os
import subprocess
import io
from tqdm import tqdm
from scipy.stats import norm

# --- Configuration ---
GWAS_FILE = 'DecodeME-SummaryStatistics/gwas_1.regenie'
QC_LIST_FILE = 'DecodeME-SummaryStatistics/gwas_qced.var'
OUTPUT_FILE = 'decodeme_alphagenome_results.csv'
LEADS_CACHE_FILE = 'decodeme_leads_cache.csv'
CACHE_DIR = 'region_cache'
API_KEY = os.environ.get('ALPHAGENOME_API_KEY')

# Global Constants
SIGNIFICANCE_THRESHOLD = 7.3  # LOG10P
WINDOW_BP = 500_000 # +/- 500kb
CREDIBLE_SET_THRESHOLD = 0.95

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

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

    def get_significant_hits(self, threshold=SIGNIFICANCE_THRESHOLD):
        """Pass 1: Identify genome-wide significant hits using fast AWK scan."""
        if os.path.exists(LEADS_CACHE_FILE):
            print(f"Loading significant hits from cache: {LEADS_CACHE_FILE}")
            # Try reading as CSV, then as space-separated if that fails to find LOG10P
            df = pd.read_csv(LEADS_CACHE_FILE)
            if 'LOG10P' not in df.columns:
                df = pd.read_csv(LEADS_CACHE_FILE, sep=r'\s+')
            return df

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
                    return df
            else:
                print(f"Awk failed: {result.stderr}")
        except Exception as e:
            print(f"Error in awk scan: {e}")
            
        return pd.DataFrame()

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
    def define_credible_set(df_region):
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
            try:
                return variant_scorers.tidy_scores(results)
            except Exception as e:
                print(f"Error tidying scores: {e}")
        return pd.DataFrame()

    def filter_by_ontology(self, df_scores):
        if df_scores.empty: return df_scores
        target_ontologies = ['UBERON:0000955', 'UBERON:0000178', 'CL:0000084']
        if 'ontology_curie' in df_scores.columns:
            return df_scores[df_scores['ontology_curie'].isin(target_ontologies)]
        return df_scores

# --- Main Pipeline ---
def main():
    processor = GWASProcessor(GWAS_FILE, QC_LIST_FILE)
    
    # 1. Identify Lead SNPs
    leads_df = processor.get_significant_hits()
    if leads_df.empty:
        print("No significant hits found matching QC criteria.")
        return
        
    # 2. Clump into independent loci
    clumps_df = LDEngine.clump_hits(leads_df)
    print(f"\nIdentified {len(clumps_df)} independent genomic loci.")
    
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
            
        credible_set = FineMapper.define_credible_set(region_df)
        cs_sizes.append(len(credible_set))
        credible_sets[idx] = credible_set
    
    clumps_df['CS_Size'] = cs_sizes
    
    # Store sets in a way that aligns with the final DataFrame indices
    valid_mask = clumps_df['CS_Size'] > 0
    final_credible_sets = {i: credible_sets[old_idx] 
                          for i, old_idx in enumerate(clumps_df.index[valid_mask])}
    
    clumps_df = clumps_df[valid_mask].reset_index(drop=True)

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
            else:
                 print(f"Invalid index. Please select between 0 and {len(clumps_df)-1}.")
                 return
        except ValueError:
            print("Invalid input. Please enter a number, 'all', or 'q'.")
            return
    else:
        selected_indices = clumps_df.index.tolist()

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
            filtered_scores['Locus_ID'] = row['ID']
            final_results.append(filtered_scores)
    
    # 6. Final Export
    if final_results:
        final_df = pd.concat(final_results, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nPipeline Complete. Multimodal results saved to {OUTPUT_FILE}")
    else:
        print("\nAnalysis produced no results after ontology filtering.")
