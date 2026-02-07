import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import numpy as np
import hashlib
import matplotlib.pyplot as plt
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
    from alphagenome.models import dna_client, variant_scorers, dna_output
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
def get_global_curie_map(file_path):
    """Builds a global map of Biosample Name -> Ontology CURIE from the entire file."""
    if HAS_POLARS:
        try:
            lf = pl.scan_csv(file_path)
            schema = lf.collect_schema().names()
            if 'biosample_name' in schema and 'ontology_curie' in schema:
                q = lf.select(['biosample_name', 'ontology_curie']).unique()
                df = q.collect(engine="streaming").to_pandas()
                return dict(zip(df['biosample_name'], df['ontology_curie']))
        except Exception: pass
    
    # Pandas fallback
    try:
        mapping = {}
        for chunk in pd.read_csv(file_path, usecols=['biosample_name', 'ontology_curie'], chunksize=100000):
            mapping.update(dict(zip(chunk['biosample_name'], chunk['ontology_curie'])))
        return mapping
    except Exception: return {}

@st.cache_data
def get_variant_context(file_path, variant_id):
    """Loads all rows for a specific variant, ignoring score thresholds."""
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path).filter(pl.col('variant_id') == variant_id)
            return q.collect(engine="streaming").to_pandas()
        except Exception: return pd.DataFrame()
    return pd.DataFrame() # Simplification for brevity, fallback logic is similar to load_filtered

@st.cache_data
def get_snps_by_chrom_lazy(file_path, chrom):
    """Fetches unique variant IDs for a specific chromosome."""
    if HAS_POLARS:
        try:
            lf = pl.scan_csv(file_path)
            schema = lf.collect_schema().names()
            if "variant_id" in schema:
                if "CHROM" in schema:
                    q = lf.filter(pl.col("CHROM").astype(pl.Utf8) == str(chrom)).select("variant_id").unique()
                else:
                    q = lf.filter(pl.col("variant_id").str.starts_with(f"{chrom}:") | 
                                  pl.col("variant_id").str.starts_with(f"chr{chrom}:")).select("variant_id").unique()
                
                vals = q.collect(engine="streaming").to_series().to_list()
                return sorted([str(x) for x in vals if x is not None])
        except Exception: pass
    return []

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
    if not vid or not isinstance(vid, str): return None, None, None, None
    parts = vid.split(':')
    if len(parts) >= 4:
        try: return parts[0], int(parts[1]), parts[2], parts[3]
        except ValueError: pass
    if len(parts) == 3:
        chrom = parts[0]
        try:
            pos = int(parts[1])
            alleles = parts[2].split('>')
            if len(alleles) == 2: return chrom, pos, alleles[0], alleles[1]
        except ValueError: pass
    return None, None, None, None

def safe_defaults(defaults, options):
    """Ensures defaults are present in options to prevent Streamlit crashes."""
    if not defaults: return []
    options_set = set(options)
    return [d for d in defaults if d in options_set]

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
        all_chroms = get_unique_values_lazy(selected_file, "CHROM")
        if not all_chroms: all_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
        
        # Load global CURIE map once
        global_curie_map = get_global_curie_map(selected_file)

    selected_locus = st.selectbox("Locus ID", ['All'] + locus_ids, index=min(1, len(locus_ids)))
    score_threshold = st.slider("Min. Quantile Score (Abs)", 0.0, 1.0, 0.5, 0.05)
    
    st.header("3. Refine View")
    sel_genes = st.multiselect("Filter by Genes", gene_names)
    sel_biosamples = st.multiselect("Filter by Tissues", biosamples)
    sel_assays = st.multiselect("Filter by Assay Types", output_types)

# --- Data Loading ---
with st.spinner(f"Filtering records..."):
    # We load with threshold initially
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

# Don't stop here, Micro-View needs to run even if Macro-View is empty due to strict filters

st.markdown(f"**Loaded:** `{selected_file}` | **Locus:** `{selected_locus}`")

# --- Module 1: Macro-Visualization ---
st.subheader("1. Macro-View: Functional Landscape")
tab1, tab2, tab3 = st.tabs(["🔥 Functional Fingerprint", "🔗 Mechanism", "🔎 Quick Lookup"])

with tab1:
    if filtered_df.empty:
        st.info("No data passing filters for Heatmap.")
    else:
        st.info("Heatmap shows average impact score per variant across biosamples.")
        h_thresh = st.slider("Heatmap-specific Score Threshold", 0.0, 1.0, score_threshold, 0.05, key="h_thresh")
        h_df = filtered_df[filtered_df['quantile_score'].abs() >= h_thresh]
        
        if 'biosample_name' in h_df.columns and 'variant_id' in h_df.columns:
            heatmap_data = h_df.groupby(['variant_id', 'biosample_name'])['quantile_score'].mean().reset_index()
            if len(heatmap_data) > 30000:
                st.error(f"Data density too high ({len(heatmap_data)} points). Filter by Genes/Tissues.")
            elif heatmap_data.empty:
                st.warning("No data points for heatmap at this threshold.")
            else:
                fig = px.density_heatmap(heatmap_data, x="biosample_name", y="variant_id", z="quantile_score",
                                         color_continuous_scale="RdBu_r", title=f"Impact Heatmap: {selected_locus}")
                fig.update_layout(height=max(400, min(1500, len(h_df['variant_id'].unique()) * 25)))
                st.plotly_chart(fig, width="stretch")
        else: st.error("Incompatible data format for heatmap.")

with tab2:
    st.markdown("Compare scores across different assay types.")
    incl_sub = st.checkbox("Include sub-threshold scores for correlation", value=True)
    
    if incl_sub:
        m_df = load_filtered_subset(selected_file, selected_locus, 0.0, 
                                    genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays, ignore_thresh_for_assays=True)
    else:
        m_df = filtered_df

    if not m_df.empty and 'output_type' in m_df.columns:
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
            st.warning(f"Correlation requires at least two assay types. Found: {assays_found}")
    else:
        st.info("No data for correlation.")

with tab3:
    st.markdown("### 🔎 SNP Discovery Portal")
    st.caption("Drill down to a specific variant and tissue using hierarchical filters.")
    
    col_c, col_s, col_m = st.columns([1, 2, 1])
    with col_c:
        selected_chrom = st.selectbox("Chromosome", all_chroms)
    
    with col_s:
        is_manual = st.checkbox("Manual Variant Entry")
        if is_manual:
            lookup_vid = st.text_input("Enter Variant ID (e.g. chr1:123:A:C)", "")
        else:
            with st.spinner(f"Loading SNPs for Chr {selected_chrom}..."):
                snps_in_chrom = get_snps_by_chrom_lazy(selected_file, selected_chrom)
            lookup_vid = st.selectbox("Search SNPs in Results", snps_in_chrom)
    
    if lookup_vid:
        st.divider()
        # Fetch dedicated context for this SNP (ignoring global score thresholds)
        snp_df = get_variant_context(selected_file, lookup_vid)
        
        with col_m:
            if not snp_df.empty:
                max_row = snp_df.loc[snp_df['quantile_score'].abs().idxmax()]
                st.metric("Max Impact", f"{max_row['quantile_score']:.2f}")
                st.caption(f"In {max_row['biosample_name']}")
            else:
                st.info("Variant details not in file.")

        st.markdown("#### 🎯 Smart Tissue Selection")
        suggested = []
        if not snp_df.empty and 'biosample_name' in snp_df.columns:
            suggested = snp_df.sort_values('quantile_score', ascending=False)['biosample_name'].unique().tolist()[:5]
        
        # Use safe defaults logic here too, though portal choices are usually self-contained
        sel_portal_tissues = st.multiselect("Select Tissues for Deep Dive", biosamples, default=safe_defaults(suggested, biosamples))
        
        if st.button("🧬 Send to Molecular Deep Dive"):
            st.session_state['sel_var_portal'] = lookup_vid
            st.session_state['sel_tissues_portal'] = sel_portal_tissues
            st.success(f"Configured for {lookup_vid}. Scroll down.")

# --- Module 2: Micro-Visualization ---
st.divider()
st.subheader("2. Micro-View: Molecular Deep Dive")

if not HAS_AG: st.warning("AlphaGenome library not installed.")
else:
    # 1. Determine Source of Variants
    # Mix global filter vars with portal var
    portal_vid = st.session_state.get('sel_var_portal')
    portal_tissues = st.session_state.get('sel_tissues_portal', [])
    
    if not filtered_df.empty:
        target_vars = sorted(filtered_df['variant_id'].unique().tolist())
    else:
        target_vars = []
    
    if portal_vid and portal_vid not in target_vars:
        target_vars.insert(0, portal_vid) # Prepend
    
    if not target_vars:
        st.warning("No variants selected or filtered.")
    else:
        selected_idx = target_vars.index(portal_vid) if portal_vid in target_vars else 0
        sel_var = st.selectbox("Select Variant for Track Visualization", target_vars, index=selected_idx)
        
        # 2. Determine Context (Tissues) for this SPECIFIC variant
        # We re-fetch context to ignore global thresholds
        var_context_df = get_variant_context(selected_file, sel_var)
        
        # Build available tissues from this variant's context + global dictionary + portal selection
        # (Portal selection might be for a variant not in file, so we allow global biosamples list)
        
        if not var_context_df.empty and 'biosample_name' in var_context_df.columns:
            # Tissues where this variant has data
            var_tissues = sorted(var_context_df['biosample_name'].unique())
        else:
            # Fallback for manual entry variants
            var_tissues = biosamples
            
        # Defaults
        # If this var came from Portal, use portal tissues
        if sel_var == portal_vid and portal_tissues:
            defaults = portal_tissues
        else:
            # Otherwise top 5 impactful
            defaults = []
            if not var_context_df.empty:
                defaults = var_context_df.sort_values('quantile_score', ascending=False)['biosample_name'].head(5).tolist()
        
        st.markdown("##### 🎯 Track Selection")
        # Allow selection from ANY biosample known in the file (global list)
        # But maybe highlight those relevant?
        # Streamlit simple approach: Show all biosamples.
        
        # Ensure defaults exist in options
        final_defaults = safe_defaults(defaults, biosamples)
        
        sel_names = st.multiselect("Select Tissues to Plot", biosamples, default=final_defaults)
        
        c1, col_ctrl = st.columns([3, 1])
        with col_ctrl:
            st.markdown("### API Control")
            api_key = st.text_input("AlphaGenome API Key", value=os.environ.get("ALPHAGENOME_API_KEY", ""), type="password")
            
            can_generate = api_key and len(sel_names) > 0 and len(sel_names) <= 50
            if st.button("🧬 Generate Tracks", disabled=not can_generate):
                with st.spinner("Querying AlphaGenome API..."):
                    try:
                        # Use Global Map to get CURIEs
                        # If a name is missing from map (manual entry?), we skip it or warn
                        sel_curies = []
                        for name in sel_names:
                            if name in global_curie_map:
                                sel_curies.append(global_curie_map[name])
                        
                        if len(sel_curies) != len(sel_names):
                            st.warning(f"Some selected tissues missing ontology mapping. Using {len(sel_curies)} valid tissues.")
                        
                        client = dna_client.create(api_key=api_key)
                        chrom, pos, ref, alt = parse_variant_id(sel_var)
                        if chrom:
                            chrom_norm = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
                            var = genome.Variant(chromosome=chrom_norm, position=pos, reference_bases=ref, alternate_bases=alt, name=sel_var)
                            interval = var.reference_interval.resize(131072)
                            res = client.predict_variant(interval=interval, variant=var, requested_outputs=[dna_output.OutputType.RNA_SEQ, dna_output.OutputType.ATAC], ontology_terms=sel_curies, organism=dna_client.Organism.HOMO_SAPIENS)
                            
                            components = []
                            if res.reference.rna_seq and res.alternate.rna_seq: components.append(plot_components.OverlaidTracks({'REF': res.reference.rna_seq, 'ALT': res.alternate.rna_seq}, ylabel_template='RNA-Seq', alpha=0.6))
                            if res.reference.atac and res.alternate.atac: components.append(plot_components.OverlaidTracks({'REF': res.reference.atac, 'ALT': res.alternate.atac}, ylabel_template='ATAC-Seq', alpha=0.6))
                            annotations = [plot_components.VariantAnnotation([var])]
                            
                            if components:
                                st.session_state['fig_out'] = plot_components.plot(components, interval, annotations=annotations, title=f"Track Overlay: {sel_var}")
                                st.success("Visualization generated.")
                            else: st.warning("No tracks returned.")
                        else: st.error("Invalid variant ID.")
                    except Exception as e: st.error(f"API Error: {e}")

        with c1:
            if 'fig_out' in st.session_state: st.pyplot(st.session_state['fig_out'])
            else: st.info("Configuration complete. Click 'Generate Tracks' to view.")
