import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import numpy as np
import hashlib
from io import StringIO
from pathlib import Path

# Try importing Polars for performance
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Try importing AlphaGenome for Micro-Viz
try:
    import alphagenome.visualization.plot_components as plot_components
    from alphagenome.models import dna_client, variant_scorers
    from alphagenome.data import genome
    HAS_AG = True
except ImportError:
    HAS_AG = False

st.set_page_config(page_title="DecodeME Functional Workbench", layout="wide")

# --- Helper Functions ---

@st.cache_data
def get_available_files():
    """Scans for result CSVs, excluding cache directories."""
    files = []
    exclude_dirs = {'region_cache', 'ld_cache', 'results_cache', '.git', '.venv', '__pycache__'}
    for root, dirs, filenames in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in filenames:
            if f.endswith('.csv') and 'cache' not in f and 'leads' not in f:
                files.append(os.path.join(root, f))
    return sorted(files, key=os.path.getmtime, reverse=True)

@st.cache_data
def get_unique_values_lazy(file_path, column_name):
    """Extracts unique values for a column efficiently using Lazy Scan."""
    if HAS_POLARS:
        try:
            lf = pl.scan_csv(file_path)
            if column_name in lf.collect_schema().names():
                q = lf.select(column_name).unique()
                vals = q.collect(engine="streaming").to_series().to_list()
                return sorted([str(x) for x in vals if x is not None])
            return []
        except Exception:
            pass
    # Pandas fallback
    try:
        header = pd.read_csv(file_path, nrows=0)
        if column_name not in header.columns: return []
        vals = set()
        for chunk in pd.read_csv(file_path, usecols=[column_name], chunksize=100000):
            vals.update(chunk[column_name].dropna().unique())
        return sorted([str(x) for x in vals])
    except Exception: return []

@st.cache_data
def load_filtered_subset(file_path, locus_id, score_thresh, genes=None, biosamples=None, assays=None, ignore_thresh_for_assays=False):
    """Loads ONLY the rows matching all filter criteria."""
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path)
            schema = q.collect_schema().names()
            
            if locus_id != 'All' and "Locus_ID" in schema:
                q = q.filter(pl.col('Locus_ID') == locus_id)
            
            if not ignore_thresh_for_assays and score_thresh > 0 and "quantile_score" in schema:
                q = q.filter(pl.col('quantile_score').abs() >= score_thresh)
            
            if genes and "gene_name" in schema:
                q = q.filter(pl.col('gene_name').is_in(genes))
            if biosamples and "biosample_name" in schema:
                q = q.filter(pl.col('biosample_name').is_in(biosamples))
            if assays and "output_type" in schema:
                q = q.filter(pl.col('output_type').is_in(assays))
            
            return q.collect(engine="streaming").to_pandas()
        except Exception as e:
            st.error(f"Polars failed: {e}")
            return pd.DataFrame()
            
    # Pandas Fallback
    try:
        chunks = []
        header = pd.read_csv(file_path, nrows=0)
        has_locus = 'Locus_ID' in header.columns
        has_score = 'quantile_score' in header.columns
        has_gene = 'gene_name' in header.columns
        has_bio = 'biosample_name' in header.columns
        has_assay = 'output_type' in header.columns
        
        for chunk in pd.read_csv(file_path, chunksize=100000):
            if locus_id != 'All' and has_locus: chunk = chunk[chunk['Locus_ID'].astype(str) == str(locus_id)]
            if not ignore_thresh_for_assays and score_thresh > 0 and has_score: 
                chunk = chunk[chunk['quantile_score'].abs() >= score_thresh]
            if genes and has_gene: chunk = chunk[chunk['gene_name'].isin(genes)]
            if biosamples and has_bio: chunk = chunk[chunk['biosample_name'].isin(biosamples)]
            if assays and has_assay: chunk = chunk[chunk['output_type'].isin(assays)]
            
            if not chunk.empty: chunks.append(chunk)
        if chunks: return pd.concat(chunks, ignore_index=True)
    except Exception as e: st.error(f"Pandas failed: {e}")
    return pd.DataFrame()

def parse_variant_id(vid):
    """
    Parses variant string into (chrom, pos, ref, alt).
    Handles:
    1. CHR:POS:REF:ALT (standard)
    2. CHR:POS:REF>ALT (found in DecodeME results)
    """
    if not vid or not isinstance(vid, str): return None, None, None, None
    
    parts = vid.split(':')
    
    # Format 1: CHR:POS:REF:ALT (4+ parts)
    if len(parts) >= 4:
        try:
            return parts[0], int(parts[1]), parts[2], parts[3]
        except ValueError:
            pass
            
    # Format 2: CHR:POS:REF>ALT (3 parts, alleles joined by '>')
    if len(parts) == 3:
        chrom = parts[0]
        try:
            pos = int(parts[1])
            alleles = parts[2].split('>')
            if len(alleles) == 2:
                return chrom, pos, alleles[0], alleles[1]
        except ValueError:
            pass
            
    return None, None, None, None

# --- Main App ---

st.title("🧬 DecodeME Functional Workbench")

# 1. Sidebar: Configuration
with st.sidebar:
    st.header("1. Data Source")
    files = get_available_files()
    if not files: st.error("No CSV files found."); st.stop()
    selected_file = st.selectbox("Select Results File", files)
    if st.button("Refresh Files"): get_available_files.clear(); st.rerun()
    
    st.divider()
    st.header("2. Primary Filters")
    with st.spinner("Indexing file..."):
        locus_ids = get_unique_values_lazy(selected_file, "Locus_ID")
        gene_names = get_unique_values_lazy(selected_file, "gene_name")
        output_types = get_unique_values_lazy(selected_file, "output_type")
        biosamples = get_unique_values_lazy(selected_file, "biosample_name")

    selected_locus = st.selectbox("Locus ID", ['All'] + locus_ids, index=min(1, len(locus_ids)))
    score_threshold = st.slider("Min. Quantile Score (Abs)", 0.0, 1.0, 0.5, 0.05)
    
    st.header("3. Refine View")
    sel_genes = st.multiselect("Filter by Genes", gene_names)
    sel_biosamples = st.multiselect("Filter by Tissues", biosamples)
    sel_assays = st.multiselect("Filter by Assay Types", output_types)

# --- Data Loading ---
with st.spinner(f"Filtering records..."):
    filtered_df = load_filtered_subset(selected_file, selected_locus, score_threshold, 
                                       genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays)

# --- Diagnostics ---
with st.expander("📊 Data Diagnostics"):
    if not filtered_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variants", len(filtered_df['variant_id'].unique()))
        c2.metric("Tissues", len(filtered_df['biosample_name'].unique()) if 'biosample_name' in filtered_df.columns else 0)
        c3.metric("Assays", len(filtered_df['output_type'].unique()) if 'output_type' in filtered_df.columns else 0)
        c4.metric("Rows", f"{len(filtered_df):,}")
    else:
        st.warning("No data found with current filters.")

if filtered_df.empty:
    st.stop()

st.markdown(f"**Loaded:** `{selected_file}` | **Locus:** `{selected_locus}`")

# --- Module 1: Macro-Visualization ---
st.subheader("1. Macro-View: Functional Landscape")
tab1, tab2 = st.tabs(["🔥 Functional Fingerprint (Heatmap)", "🔗 Mechanism (Scatter)"])

with tab1:
    st.info("Heatmap shows average impact score per variant across biosamples.")
    h_thresh = st.slider("Heatmap-specific Score Threshold", 0.0, 1.0, score_threshold, 0.05, key="h_thresh")
    h_df = filtered_df[filtered_df['quantile_score'].abs() >= h_thresh]
    
    if 'biosample_name' in h_df.columns and 'variant_id' in h_df.columns:
        heatmap_data = h_df.groupby(['variant_id', 'biosample_name'])['quantile_score'].mean().reset_index()
        
        if len(heatmap_data) > 35000:
            st.error(f"Data density too high ({len(heatmap_data)} points). Please filter for specific Genes or Tissues in the sidebar.")
        elif heatmap_data.empty:
            st.warning("No data points for heatmap at this threshold.")
        else:
            fig = px.density_heatmap(heatmap_data, x="biosample_name", y="variant_id", z="quantile_score",
                                     color_continuous_scale="RdBu_r", title=f"Impact Heatmap: {selected_locus}")
            fig.update_layout(height=max(400, min(1500, len(h_df['variant_id'].unique()) * 25)))
            st.plotly_chart(fig, width="stretch")
    else: st.error("Incompatible data format for heatmap.")

with tab2:
    st.markdown("Compare scores across different assay types for the same variants.")
    incl_sub = st.checkbox("Include sub-threshold scores for correlation", value=True)
    if incl_sub:
        m_df = load_filtered_subset(selected_file, selected_locus, 0.0, 
                                    genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays, ignore_thresh_for_assays=True)
    else:
        m_df = filtered_df

    if 'output_type' in m_df.columns:
        pivot_df = m_df.pivot_table(index=['variant_id', 'biosample_name'], 
                                    columns='output_type', values='quantile_score', aggfunc='mean').reset_index()
        assays_found = [c for c in pivot_df.columns if c not in ['variant_id', 'biosample_name']]
        
        if len(assays_found) >= 2:
            c1, c2 = st.columns(2)
            x_ax = c1.selectbox("X-Axis Assay", assays_found, index=0)
            y_ax = c2.selectbox("Y-Axis Assay", assays_found, index=min(1, len(assays_found)-1))
            fig = px.scatter(pivot_df, x=x_ax, y=y_ax, color="biosample_name", 
                             hover_data=['variant_id'], title=f"Assay Correlation: {x_ax} vs {y_ax}")
            fig.add_hline(y=0, line_dash="dash", line_color="grey"); fig.add_vline(x=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning(f"Correlation requires at least two assay types. Currently only found: {assays_found}")

# --- Module 2: Micro-Visualization ---
st.divider()
st.subheader("2. Micro-View: Molecular Deep Dive")

if not HAS_AG: st.warning("AlphaGenome library not installed.")
else:
    target_vars = sorted(filtered_df['variant_id'].unique())
    sel_var = st.selectbox("Select Variant for Track Visualization", target_vars)
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("### API Control")
        api_key = st.text_input("AlphaGenome API Key", value=os.environ.get("ALPHAGENOME_API_KEY", ""), type="password")
        if st.button("🧬 Generate Tracks"):
            if not api_key: st.error("API Key required.")
            else:
                with st.spinner("Querying AlphaGenome API..."):
                    try:
                        client = dna_client.create(api_key=api_key)
                        chrom, pos, ref, alt = parse_variant_id(sel_var)
                        if chrom:
                            # Prepend chr if missing
                            chrom = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
                            var = genome.Variant(chromosome=chrom, position=pos, reference_bases=ref, alternate_bases=alt, name=sel_var)
                            interval = var.reference_interval.resize(131072)
                            res = client.score_variant(interval=interval, variant=var, organism=dna_client.Organism.HOMO_SAPIENS)
                            if res:
                                st.success(f"Track data received for {sel_var}")
                                st.dataframe(res[0].to_df())
                        else:
                            st.error(f"Could not parse variant ID: {sel_var}")
                    except Exception as e: st.error(f"API Error: {e}")
    with c1:
        st.info("Interactive tracks will appear here once generated.")