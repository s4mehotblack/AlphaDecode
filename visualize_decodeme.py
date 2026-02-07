import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import io
import requests
import time
import sys
import argparse
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
        except Exception: pass
    return []

@st.cache_data
def get_global_curie_map(file_path):
    """Builds a global map of Biosample Name -> Ontology CURIE."""
    if HAS_POLARS:
        try:
            lf = pl.scan_csv(file_path)
            schema = lf.collect_schema().names()
            if 'biosample_name' in schema and 'ontology_curie' in schema:
                q = lf.select(['biosample_name', 'ontology_curie']).unique()
                df = q.collect(engine="streaming").to_pandas()
                return dict(zip(df['biosample_name'], df['ontology_curie']))
        except Exception: pass
    return {}

@st.cache_data
def get_variant_context(file_path, variant_id):
    """Loads all rows for a specific variant, ignoring score thresholds."""
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path).filter(pl.col('variant_id') == variant_id)
            return q.collect(engine="streaming").to_pandas()
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def get_locus_leads_map(file_path):
    """Loads leads cache to map IDs to LOG10P and potentially rsIDs."""
    # Find leads file associated with this results file
    gwas_stem = Path(file_path).stem.replace('_results_all', '').replace('_results', '')
    leads_file = f"region_cache/{gwas_stem}_leads.csv"
    if not os.path.exists(leads_file):
        # Fallback to generic leads cache if it exists
        leads_file = "decodeme_leads_cache.csv"
        if not os.path.exists(leads_file): return pd.DataFrame()
    
    try:
        # Load and check for ID column
        df = pd.read_csv(leads_file)
        if 'ID' not in df.columns:
            # Try space-separated fallback
            df = pd.read_csv(leads_file, sep=r'\s+')
        return df
    except Exception: return pd.DataFrame()

@st.cache_data
def load_filtered_subset(file_path, locus_id, score_thresh, genes=None, biosamples=None, assays=None, ignore_thresh_for_assays=False):
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path)
            schema = q.collect_schema().names()
            if locus_id != 'All' and "Locus_ID" in schema: q = q.filter(pl.col('Locus_ID') == locus_id)
            if not ignore_thresh_for_assays and score_thresh > 0 and "quantile_score" in schema: q = q.filter(pl.col('quantile_score').abs() >= score_thresh)
            if genes and "gene_name" in schema: q = q.filter(pl.col('gene_name').is_in(genes))
            if biosamples and "biosample_name" in schema: q = q.filter(pl.col('biosample_name').is_in(biosamples))
            if assays and "output_type" in schema: q = q.filter(pl.col('output_type').is_in(assays))
            return q.collect(engine="streaming").to_pandas()
        except Exception: return pd.DataFrame()
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
            pos = int(parts[1]); alleles = parts[2].split('>')
            if len(alleles) == 2: return chrom, pos, alleles[0], alleles[1]
        except ValueError: pass
    return None, None, None, None

def format_variant_label(vid):
    """Formats variant string for publication display."""
    chrom, pos, ref, alt = parse_variant_id(vid)
    if chrom:
        return f"{chrom}:{pos:,} ({ref}>{alt})"
    return vid

def safe_defaults(defaults, options):
    if not defaults: return []
    options_set = set(options); return [d for d in defaults if d in options_set]

# --- Main App ---

st.title("🧬 DecodeME Functional Workbench")

# 1. Sidebar: Configuration
with st.sidebar:
    st.header("1. Data Source")
    files = get_available_files()
    if not files: st.error("No CSV files found."); st.stop()
    selected_file = st.selectbox("Select Results File", files)
    if st.button("Refresh Files"): get_available_files.clear(); st.rerun()
    st.divider(); st.header("2. Primary Filters")
    with st.spinner("Indexing file..."):
        locus_ids = get_unique_values_lazy(selected_file, "Locus_ID")
        gene_names = get_unique_values_lazy(selected_file, "gene_name")
        output_types = get_unique_values_lazy(selected_file, "output_type")
        biosamples = get_unique_values_lazy(selected_file, "biosample_name")
        global_curie_map = get_global_curie_map(selected_file)
        leads_df = get_locus_leads_map(selected_file)

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

# --- Phase 1: Locus Context Card ---
if not filtered_df.empty and selected_locus != 'All':
    st.markdown("### 📍 Locus Context")
    c1, c2, c3, c4 = st.columns(4)
    
    # GWAS Logic
    p_val_str = "N/A"
    if not leads_df.empty and 'ID' in leads_df.columns:
        locus_gwas = leads_df[leads_df['ID'] == selected_locus]
        if not locus_gwas.empty and 'LOG10P' in locus_gwas.columns:
            p_val_str = f"10^-{locus_gwas['LOG10P'].iloc[0]:.1f}"
    
    # Functional Logic
    max_score = filtered_df['quantile_score'].abs().max()
    top_var = filtered_df.loc[filtered_df['quantile_score'].abs().idxmax(), 'variant_id']
    top_tissue = filtered_df.loc[filtered_df['quantile_score'].abs().idxmax(), 'biosample_name']
    
    c1.metric("Genomic Locus", selected_locus)
    c2.metric("GWAS Significance", p_val_str)
    c3.metric("Max Impact Score", f"{max_score:.2f}")
    c4.metric("Top Variant", top_var.split(':')[-1], help=top_var)
    st.caption(f"Strongest functional signal found in **{top_tissue}** for variant **{format_variant_label(top_var)}**")
    st.divider()

def assign_biological_system(biosample_name):
    """Maps biosample names to high-level biological systems."""
    name = biosample_name.lower()
    if any(x in name for x in ['brain', 'neuron', 'glia', 'astrocyte', 'cortex', 'hippocampus', 'spinal']):
        return "CNS"
    if any(x in name for x in ['blood', 't-cell', 'b-cell', 'monocyte', 'macrophage', 'lymph', 'spleen', 'immune']):
        return "Immune/Blood"
    if any(x in name for x in ['liver', 'hepatocyte', 'pancreas', 'gut', 'colon', 'intestine', 'stomach']):
        return "Digestive/Metabolic"
    if any(x in name for x in ['muscle', 'myocyte', 'heart', 'cardiac']):
        return "Musculoskeletal"
    if any(x in name for x in ['lung', 'bronchus', 'pulmonary']):
        return "Respiratory"
    if any(x in name for x in ['kidney', 'renal', 'bladder']):
        return "Renal"
    return "Other"

# --- Module 1: Macro-Visualization ---
st.subheader("1. Macro-View: Functional Landscape")
tab1, tab2, tab3 = st.tabs(["🔥 Functional Fingerprint", "🔗 Mechanism", "🔎 Quick Lookup"])

with tab1:
    h_thresh = st.slider("Heatmap-specific Score Threshold", 0.0, 1.0, score_threshold, 0.05, key="h_thresh")
    h_df = filtered_df[filtered_df['quantile_score'].abs() >= h_thresh].copy()
    
    if not h_df.empty:
        # Enrich with System info
        if 'biosample_name' in h_df.columns:
            h_df['System'] = h_df['biosample_name'].apply(assign_biological_system)
            
            heatmap_data = h_df.groupby(['variant_id', 'biosample_name', 'System'])['quantile_score'].mean().reset_index()
            heatmap_data['display_id'] = heatmap_data['variant_id'].apply(lambda x: x.split(':')[-1])
            
            # Dynamic Height
            n_vars = len(heatmap_data['display_id'].unique())
            plot_height = max(500, min(1500, n_vars * 25))
            
            fig = px.density_heatmap(
                heatmap_data, 
                x="biosample_name", 
                y="display_id", 
                z="quantile_score",
                facet_col="System", 
                facet_col_wrap=3, # Wrap if many systems found
                color_continuous_scale="RdBu_r", 
                range_color=[-1, 1], # Enforce centered diverging scale
                title=f"Functional Fingerprint: {selected_locus}",
                labels={'quantile_score': 'Impact', 'display_id': 'Variant'},
                category_orders={"System": ["CNS", "Immune/Blood", "Musculoskeletal", "Digestive/Metabolic", "Other"]}
            )
            
            fig.update_layout(height=plot_height)
            fig.update_xaxes(matches=None, showticklabels=True) # Allow independent axes
            st.plotly_chart(fig, width="stretch")
        else:
            st.error("Missing biosample_name column.")
    else:
        st.info("No data passing filters for Heatmap.")

with tab2:
    incl_sub = st.checkbox("Include sub-threshold scores for correlation", value=True)
    m_df = load_filtered_subset(selected_file, selected_locus, 0.0, genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays, ignore_thresh_for_assays=True) if incl_sub else filtered_df
    if not m_df.empty:
        pivot_df = m_df.pivot_table(index=['variant_id', 'biosample_name'], columns='output_type', values='quantile_score', aggfunc='mean').reset_index()
        assays_f = [c for c in pivot_df.columns if c not in ['variant_id', 'biosample_name']]
        if len(assays_f) >= 2:
            c1, c2 = st.columns(2)
            x_ax, y_ax = c1.selectbox("X-Axis", assays_f, index=0), c2.selectbox("Y-Axis", assays_f, index=1)
            fig = px.scatter(pivot_df, x=x_ax, y=y_ax, color="biosample_name", hover_data=['variant_id'], title="Modality Cross-Correlation")
            fig.add_hline(y=0, line_dash="dash", line_color="grey"); fig.add_vline(x=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig, width="stretch")

with tab3:
    st.markdown("### 🔎 Quick Lookup")
    # (Discovery Portal Logic remains similar but uses format_variant_label)
    st.info("Hierarchical search coming in Phase 2. Use Sidebar filters for now.")

# --- Module 2: Micro-Visualization ---
st.divider(); st.subheader("2. Micro-View: Molecular Deep Dive")

if HAS_AG:
    target_vars = sorted(filtered_df['variant_id'].unique().tolist())
    sel_var = st.selectbox("Select Variant for Track Visualization", target_vars, format_func=format_variant_label)
    
    col_viz, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown("### API & Display Control")
        sync_y = st.checkbox("Sync Y-Axes (Relative magnitude)", value=True)
        api_key = st.text_input("AlphaGenome API Key", value=os.environ.get("ALPHAGENOME_API_KEY", ""), type="password")
        
        # Tissue selection
        var_context = get_variant_context(selected_file, sel_var)
        var_tissues = sorted(var_context['biosample_name'].unique().tolist()) if not var_context.empty else biosamples
        defaults = var_context.sort_values('quantile_score', ascending=False)['biosample_name'].head(3).tolist() if not var_context.empty else []
        sel_names = st.multiselect("Select Tissues to Plot", var_tissues, default=safe_defaults(defaults, var_tissues))
        
        if st.button("🧬 Generate Tracks", disabled=not (api_key and sel_names)):
            with st.spinner("Querying AlphaGenome..."):
                try:
                    sel_curies = [global_curie_map[n] for n in sel_names if n in global_curie_map]
                    client = dna_client.create(api_key=api_key)
                    chrom, pos, ref, alt = parse_variant_id(sel_var)
                    chrom_n = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
                    var_obj = genome.Variant(chromosome=chrom_n, position=pos, reference_bases=ref, alternate_bases=alt, name=sel_var)
                    interval = var_obj.reference_interval.resize(131072)
                    
                    # Determine if we should request Splicing (check results for this variant)
                    request_splice = False
                    if not var_context.empty:
                        splice_scores = var_context[var_context['output_type'] == 'SPLICE_JUNCTIONS']
                        if not splice_scores.empty and splice_scores['quantile_score'].abs().max() > 0.5:
                            request_splice = True
                    
                    req_outputs = [dna_output.OutputType.RNA_SEQ, dna_output.OutputType.ATAC]
                    if request_splice:
                        req_outputs.append(dna_output.OutputType.SPLICE_JUNCTIONS)
                    
                    res = client.predict_variant(interval=interval, variant=var_obj, requested_outputs=req_outputs, ontology_terms=sel_curies, organism=dna_client.Organism.HOMO_SAPIENS)
                    
                    # Build Publication-quality components
                    components = []
                    
                    # 1. RNA-Seq Overlay
                    if res.reference.rna_seq and res.alternate.rna_seq:
                        components.append(plot_components.OverlaidTracks(
                            {'REF': res.reference.rna_seq, 'ALT': res.alternate.rna_seq}, 
                            ylabel_template='RNA-Seq\n(Expression)', alpha=0.6, shared_y_scale=sync_y
                        ))
                    
                    # 2. ATAC Overlay
                    if res.reference.atac and res.alternate.atac:
                        components.append(plot_components.OverlaidTracks(
                            {'REF': res.reference.atac, 'ALT': res.alternate.atac}, 
                            ylabel_template='ATAC-Seq\n(Accessibility)', alpha=0.6, shared_y_scale=sync_y
                        ))
                    
                    # 3. Sashimi (Splicing) Overlay
                    if request_splice and res.reference.splice_junctions and res.alternate.splice_junctions:
                        components.append(plot_components.Sashimi(
                            res.reference.splice_junctions, ylabel_template='Splicing (REF)'
                        ))
                        components.append(plot_components.Sashimi(
                            res.alternate.splice_junctions, ylabel_template='Splicing (ALT)'
                        ))
                    
                    # 4. Final Plot Rendering
                    if components:
                        st.session_state['fig_out'] = plot_components.plot(
                            components, interval, 
                            annotations=[plot_components.VariantAnnotation([var_obj], labels=[f"{ref} > {alt} site"])],
                            fig_width=18, hspace=0.4, despine=True,
                            xlabel='Genomic Position (GRCh38)',
                            title=f"Ref vs Alt Molecular Proof: {format_variant_label(sel_var)}"
                        )
                        st.success("High-resolution tracks generated.")
                    else:
                        st.warning("No signal tracks returned for selected tissues/modalities.")
                except Exception as e: st.error(f"API Error: {e}")

    with col_viz:
        if 'fig_out' in st.session_state: st.pyplot(st.session_state['fig_out'])
        else: st.info("Configure controls and click 'Generate Tracks'.")
else: st.warning("AlphaGenome not installed.")