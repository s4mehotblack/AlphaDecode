import pandas as pd
import glob
import sys
import os

def get_unique_genes(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    if not files:
        print("No files found matching the provided arguments.")
        return

    unique_genes = set()
    for f in files:
        if not os.path.isfile(f):
            continue
        try:
            # Use low_memory=False for large files
            df = pd.read_csv(f, usecols=lambda x: x == 'gene_name', low_memory=False)
            if 'gene_name' in df.columns:
                genes = df['gene_name'].dropna().unique()
                for gene in genes:
                    g_str = str(gene).strip()
                    if g_str and g_str.lower() != 'nan':
                        unique_genes.add(g_str)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if unique_genes:
        print(f"Found {len(unique_genes)} unique genes:")
        for gene in sorted(list(unique_genes)):
            print(gene)
    else:
        print("No gene names found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 show_unique_genes.py <file_or_pattern1> [file_or_pattern2 ...]")
    else:
        get_unique_genes(sys.argv[1:])