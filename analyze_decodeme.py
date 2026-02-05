import pandas as pd
import numpy as np
import os

# --- Configuration ---
GWAS_FILE = 'DecodeME-SummaryStatistics/gwas_1.regenie'
QC_LIST_FILE = 'DecodeME-SummaryStatistics/gwas_qced.var'
OUTPUT_FILE = 'decodeme_alphagenome_results.csv'

# API Key - loaded from environment variable
API_KEY = os.environ.get('ALPHAGENOME_API_KEY')

# Thresholds
SIGNIFICANCE_THRESHOLD = 7.3  # approx 5e-8
CLUMP_WINDOW_BP = 500_000     # +/- 500kb for distance-based clumping

# Mocking the AlphaGenome library import for demonstration purposes
class MockAlphaGenome:
    class genome:
        class Interval:
            def __init__(self, chrom, start, end, strand='+'):
                pass
            def resize(self, length):
                return self
        class Variant:
            def __init__(self, chromosome, position, reference_bases, alternate_bases, name=None):
                self.reference_interval = MockAlphaGenome.genome.Interval('1', 1, 1)
    
    class variant_scorers:
        RECOMMENDED_VARIANT_SCORERS = {'RNA_SEQ': 'Mock_RNA_Scorer'}
        @staticmethod
        def tidy_scores(results):
            return pd.DataFrame(results)

    class models:
        class dna_client:
            class Organism:
                HOMO_SAPIENS = 'human'
            @staticmethod
            def create(api_key):
                return MockAlphaGenome.DnaClient(api_key)

    class DnaClient:
        def __init__(self, api_key):
            self.api_key = api_key
        def score_variant(self, interval, variant, variant_scorers, organism):
            return {
                "variant_id": "mock_id",
                "RNA_Seq_Score": np.random.rand()
            }

# Switch to real library if available
client = None
try:
    import alphagenome as ag
    from alphagenome.models import dna_client, variant_scorers
    from alphagenome.data import genome
    from tqdm import tqdm

    if not API_KEY:
        print("Warning: ALPHAGENOME_API_KEY environment variable not set.")
    
    client = dna_client.create(api_key=API_KEY)
    print("AlphaGenome client initialized successfully.")
    ORGANISM = dna_client.Organism.HOMO_SAPIENS
except (ImportError, AttributeError) as e:
    print(f"AlphaGenome library initialization failed: {e}. Using Mock client.")
    ag = MockAlphaGenome()
    variant_scorers = ag.variant_scorers
    genome = ag.genome
    client = ag.models.dna_client.create(API_KEY)
    ORGANISM = ag.models.dna_client.Organism.HOMO_SAPIENS
    from tqdm import tqdm

def load_and_filter_data():
    """
    Loads GWAS summary stats, filters by QC list and Significance.
    Uses chunking to handle large files.
    """
    print(f"Loading QC list from {QC_LIST_FILE}...")
    # Read QC variants into a Set for O(1) lookup
    try:
        # Optimizing QC load: only read ID column if possible, or just ID
        qc_df = pd.read_csv(QC_LIST_FILE, header=0, usecols=['ID'])
        valid_ids = set(qc_df['ID'])
        del qc_df # Free memory
    except Exception as e:
        print(f"Error reading QC file: {e}")
        return pd.DataFrame()

    print(f"Loading and filtering GWAS stats from {GWAS_FILE}...")
    
    filtered_chunks = []
    chunk_size = 100000
    total_processed = 0
    
    # Read GWAS file in chunks
    try:
        # Explicitly specifying types for efficiency
        dtype_spec = {
            'CHROM': str, 
            'GENPOS': int, 
            'ID': str, 
            'LOG10P': float
        }
        
        reader = pd.read_csv(GWAS_FILE, sep=r'\s+', chunksize=chunk_size)
        
        for chunk in reader:
            total_processed += len(chunk)
            
            # 1. Filter by Significance first (faster, numerical)
            chunk = chunk[chunk['LOG10P'] >= SIGNIFICANCE_THRESHOLD]
            
            if not chunk.empty:
                # 2. Filter by QC List (string lookup)
                chunk = chunk[chunk['ID'].isin(valid_ids)]
                
                if not chunk.empty:
                    filtered_chunks.append(chunk)
            
            if total_processed % 1_000_000 == 0:
                print(f"Processed {total_processed} variants...")

    except Exception as e:
        print(f"Error reading GWAS file: {e}")
        return pd.DataFrame()
    
    print(f"Total variants processed: {total_processed}")
    
    if filtered_chunks:
        df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Variants after all filters: {len(df)}")
        return df
    else:
        return pd.DataFrame()

def perform_clumping(df, window_bp=CLUMP_WINDOW_BP):
    """
    Performs greedy distance-based clumping.
    """
    if df.empty:
        return df

    print("Performing distance-based clumping...")
    # Sort by Significance (descending)
    df = df.sort_values(by='LOG10P', ascending=False)
    
    clumped_hits = []
    
    # Group by Chromosome first to avoid cross-chromosome calc
    for chrom, group in df.groupby('CHROM'):
        candidates = group.copy()
        
        while not candidates.empty:
            # Take top hit
            lead_snp = candidates.iloc[0]
            clumped_hits.append(lead_snp)
            
            # Define window
            pos = lead_snp['GENPOS']
            start = pos - window_bp
            end = pos + window_bp
            
            # Remove variants within window (including self)
            candidates = candidates[
                (candidates['GENPOS'] < start) | (candidates['GENPOS'] > end)
            ]
            
    clumped_df = pd.DataFrame(clumped_hits).reset_index(drop=True)
    print(f"Found {len(clumped_df)} independent loci.")
    return clumped_df

def user_select_loci(df):
    """
    Displays loci and asks user for selection.
    """
    if df.empty:
        return df

    print("\nIdentified Loci (Top Hits):")
    display_cols = ['CHROM', 'GENPOS', 'ID', 'LOG10P']
    print(df[display_cols].to_string())
    
    while True:
        choice = input("\nEnter locus index to test, 'all' for all, or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            return pd.DataFrame()
        elif choice == 'all':
            return df
        else:
            try:
                idx = int(choice)
                if 0 <= idx < len(df):
                    return df.iloc[[idx]]
                else:
                    print(f"Invalid index. Please enter a value between 0 and {len(df)-1}")
            except ValueError:
                print("Invalid input. Please enter an index, 'all', or 'q'.")

def run_alphagenome_analysis(hits_df):
    """
    Queries AlphaGenome for each variant.
    """
    all_variant_results = []
    # Supported length from error message: 1048576
    SUPPORTED_SEQ_LENGTH = 1048576
    
    print(f"\nRunning AlphaGenome analysis on {len(hits_df)} selected loci...")
    # Use tqdm for progress bar
    for idx, row in tqdm(hits_df.iterrows(), total=len(hits_df)):
        chrom = str(row['CHROM'])
        # Prepend 'chr' if not already present
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"
            
        pos = int(row['GENPOS'])
        ref = row['ALLELE0']
        alt = row['ALLELE1']
        variant_id = row['ID']
        
        # 1. Define Variant using name and keyword arguments
        variant = genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=ref,
            alternate_bases=alt,
            name=variant_id
        )
        
        # 2. Define Interval using resize on the variant's reference interval
        interval = variant.reference_interval.resize(SUPPORTED_SEQ_LENGTH)
        
        # 3. Run score_variant
        try:
            # Select scorers. Using RNA_SEQ as a default/example.
            scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]
            
            variant_scores = client.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=scorers,
                organism=ORGANISM
            )
            
            all_variant_results.append(variant_scores)
            
        except Exception as e:
            print(f"Error processing {variant_id}: {e}")

    if not all_variant_results:
        return pd.DataFrame()

    print("Tidying scores...")
    try:
        # Use tidy_scores as per forestglip.py
        df_scores = variant_scorers.tidy_scores(all_variant_results)
        
        # Merge back original GWAS info (LOG10P etc.) for context
        # Extract ID from tidy results (assuming 'name' or 'variant_id' column exists)
        # Check column names in tidy_scores output
        # Based on typical tidy_scores, it should have variant-level columns
        return df_scores
    except Exception as e:
        print(f"Error tidying scores: {e}")
        return pd.DataFrame()


def main():
    # A. Data Prep & QC
    df_sig = load_and_filter_data()
    
    if df_sig.empty:
        print("No significant variants found matching QC criteria.")
        return

    # B. Clumping (Optimization)
    df_clumped = perform_clumping(df_sig)
    
    # D. User Selection (New)
    df_selected = user_select_loci(df_clumped)
    
    if df_selected.empty:
        print("No loci selected. Exiting.")
        return

    # C. AlphaGenome Implementation
    results_df = run_alphagenome_analysis(df_selected)
    
    # Save Results
    if not results_df.empty:
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")
        print(results_df.head())
    else:
        print("\nAnalysis produced no results.")

if __name__ == "__main__":
    main()
