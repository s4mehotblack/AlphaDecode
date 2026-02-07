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
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path).filter(pl.col('variant_id') == variant_id)
            return q.collect(engine="streaming").to_pandas()
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def get_locus_leads_map(file_path):
    gwas_stem = Path(file_path).stem.replace('_results_all', '').replace('_results', '')
    leads_file = f"region_cache/{gwas_stem}_leads.csv"
    if not os.path.exists(leads_file):
        leads_file = "decodeme_leads_cache.csv"
        if not os.path.exists(leads_file): return pd.DataFrame()
    try:
        df = pd.read_csv(leads_file)
        if 'ID' not in df.columns: df = pd.read_csv(leads_file, sep=r'\s+')
        return df
    except Exception: return pd.DataFrame()

@st.cache_data
def get_snps_by_chrom_lazy(file_path, chrom):
    if HAS_POLARS:
        try:
            lf = pl.scan_csv(file_path)
            schema = lf.collect_schema().names()
            if "variant_id" in schema:
                if "CHROM" in schema:
                    q = lf.filter(pl.col("CHROM").astype(pl.Utf8) == str(chrom)).select("variant_id").unique()
                else:
                    q = lf.filter(pl.col("variant_id").str.starts_with(f"{chrom}:") | pl.col("variant_id").str.starts_with(f"chr{chrom}:")).select("variant_id").unique()
                vals = q.collect(engine="streaming").to_series().to_list()
                return sorted([str(x) for x in vals if x is not None])
        except Exception: pass
    return []

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
    chrom, pos, ref, alt = parse_variant_id(vid)
    if chrom: return f"{chrom}:{pos:,} ({ref}>{alt})"
    return vid

def safe_defaults(defaults, options):
    if not defaults: return []
    options_set = set(options); return [d for d in defaults if d in options_set]

def assign_biological_system(biosample_name):
    name = str(biosample_name).lower()
    if any(x in name for x in ['brain', 'neuron', 'glia', 'astrocyte', 'cortex', 'hippocampus', 'spinal']): return "CNS"
    if any(x in name for x in ['blood', 't-cell', 'b-cell', 'monocyte', 'macrophage', 'lymph', 'spleen', 'immune']): return "Immune/Blood"
    if any(x in name for x in ['liver', 'hepatocyte', 'pancreas', 'gut', 'colon', 'intestine', 'stomach']): return "Digestive/Metabolic"
    if any(x in name for x in ['muscle', 'myocyte', 'heart', 'cardiac']): return "Musculoskeletal"
    if any(x in name for x in ['lung', 'bronchus', 'pulmonary']): return "Respiratory"
    if any(x in name for x in ['kidney', 'renal', 'bladder']): return "Renal"
    return "Other"

# --- Main App ---

st.title("🧬 DecodeME Functional Workbench")

with st.sidebar:
    st.header("Data Source")
    files = get_available_files()
    if not files: st.error("No CSV files found."); st.stop()
    selected_file = st.selectbox("Select Results File", files)
    if st.button("Refresh Files"): get_available_files.clear(); st.rerun()
    st.divider(); st.header("Primary Filters")
    with st.spinner("Indexing file..."):
        locus_ids = get_unique_values_lazy(selected_file, "Locus_ID")
        gene_names = get_unique_values_lazy(selected_file, "gene_name")
        output_types = get_unique_values_lazy(selected_file, "output_type")
        biosamples = get_unique_values_lazy(selected_file, "biosample_name")
        all_chroms = get_unique_values_lazy(selected_file, "CHROM")
        if not all_chroms: all_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
        global_curie_map = get_global_curie_map(selected_file)
        leads_df = get_locus_leads_map(selected_file)
    selected_locus = st.selectbox("Locus ID", ['All'] + locus_ids, index=min(1, len(locus_ids)))
    score_threshold = st.slider("Min. Quantile Score (Abs)", 0.0, 1.0, 0.5, 0.05)
    st.header("Refine View")
    sel_genes = st.multiselect("Filter by Genes", gene_names)
    sel_biosamples = st.multiselect("Filter by Tissues", biosamples)
    sel_assays = st.multiselect("Filter by Assay Types", output_types)

with st.spinner(f"Filtering records..."):
    filtered_df = load_filtered_subset(selected_file, selected_locus, score_threshold, genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays)

if not filtered_df.empty and selected_locus != 'All':
    st.markdown("### 📍 Locus Overview")
    if not leads_df.empty and 'ID' in leads_df.columns:
        locus_gwas = leads_df[leads_df['ID'] == selected_locus]
        if not locus_gwas.empty and 'LOG10P' in locus_gwas.columns:
            st.info(f"**GWAS Lead Signal:** P = 10^-{locus_gwas['LOG10P'].iloc[0]:.1f}")
    st.markdown("**Top Functional Candidates (Ranked by Impact):**")
    top_hits = filtered_df.sort_values('quantile_score', key=abs, ascending=False).head(10)
    display_cols = ['variant_id', 'biosample_name', 'output_type', 'quantile_score']
    cols_to_show = [c for c in display_cols if c in top_hits.columns]
    top_hits_display = top_hits[cols_to_show].copy()
    if 'variant_id' in top_hits_display.columns:
        top_hits_display['Variant'] = top_hits_display['variant_id'].apply(format_variant_label)
        top_hits_display = top_hits_display[['Variant'] + [c for c in cols_to_show if c != 'variant_id']]
    # Fix: width='stretch' instead of use_container_width
    st.dataframe(top_hits_display, width='stretch', hide_index=True)
    st.divider()

st.subheader("1. Macro-View: Functional Landscape")
tab1, tab2, tab3 = st.tabs(["🔥 Functional Fingerprint", "🔗 Mechanism", "🔎 Quick Lookup"])

with tab1:
    h_thresh = st.slider("Heatmap-specific Score Threshold", 0.0, 1.0, score_threshold, 0.05, key="h_thresh")
    h_df = filtered_df[filtered_df['quantile_score'].abs() >= h_thresh].copy()
    if not h_df.empty and 'biosample_name' in h_df.columns:
        h_df['System'] = h_df['biosample_name'].apply(assign_biological_system)
        available_systems = sorted(h_df['System'].unique())
        c1, c2 = st.columns([2, 1])
        selected_system = c1.selectbox("Select Biological System", available_systems)
        # Fix: Axis orientation toggle to prevent squashing
        orient = c2.radio("Axis Layout", ["Variant focus", "Tissue focus"], horizontal=True)
        sys_df = h_df[h_df['System'] == selected_system]
        heatmap_data = sys_df.groupby(['variant_id', 'biosample_name'])['quantile_score'].mean().reset_index()
        heatmap_data['display_id'] = heatmap_data['variant_id'].apply(lambda x: x.split(':')[-1])
        
        # Swapping axes based on toggle
        x_col, y_col = ("biosample_name", "display_id") if orient == "Variant focus" else ("display_id", "biosample_name")
        plot_height = max(400, min(1500, len(heatmap_data[y_col].unique()) * 25))
        
        fig = px.density_heatmap(heatmap_data, x=x_col, y=y_col, z="quantile_score",
                                 color_continuous_scale="RdBu_r", range_color=[-1, 1],
                                 title=f"Impact Heatmap: {selected_system}",
                                 labels={'quantile_score': 'Impact'})
        fig.update_layout(height=plot_height)
        st.plotly_chart(fig, width="stretch")
    else: st.info("No data for Heatmap.")

with tab2:
    incl_sub = st.checkbox("Include sub-threshold scores", value=True)
    m_df = load_filtered_subset(selected_file, selected_locus, 0.0, genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays, ignore_thresh_for_assays=True) if incl_sub else filtered_df
    if not m_df.empty and 'output_type' in m_df.columns:
        pivot_df = m_df.pivot_table(index=['variant_id', 'biosample_name'], columns='output_type', values='quantile_score', aggfunc='mean').reset_index()
        assays_f = [c for c in pivot_df.columns if c not in ['variant_id', 'biosample_name']]
        if len(assays_f) >= 2:
            c1, c2 = st.columns(2)
            x_ax, y_ax = c1.selectbox("X-Axis", assays_f, index=0), c2.selectbox("Y-Axis", assays_f, index=1)
            fig = px.scatter(pivot_df, x=x_ax, y=y_ax, color="biosample_name", hover_data=['variant_id'], title="Modality Cross-Correlation")
            fig.add_hline(y=0, line_dash="dash", line_color="grey"); fig.add_vline(x=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig, width="stretch")
    else: st.info("No data for correlation.")

with tab3:
    st.markdown("### 🔎 Quick Lookup Portal")
    col_c, col_s = st.columns([1, 3])
    with col_c: selected_chrom = st.selectbox("Chromosome", all_chroms)
    with col_s:
        is_manual = st.checkbox("Manual Variant Entry")
        if is_manual: lookup_vid = st.text_input("Enter Variant ID (e.g. chr1:123:A:C)", "")
        else:
            with st.spinner(f"Loading SNPs for Chr {selected_chrom}..."): snps_in_chrom = get_snps_by_chrom_lazy(selected_file, selected_chrom)
            lookup_vid = st.selectbox("Search SNPs in Results", snps_in_chrom)
    if lookup_vid:
        st.divider()
        snp_df = get_variant_context(selected_file, lookup_vid)
        if not snp_df.empty:
            max_row = snp_df.loc[snp_df['quantile_score'].abs().idxmax()]
            st.metric("Max Impact", f"{max_row['quantile_score']:.2f}")
            st.caption(f"In {max_row['biosample_name']}")
            suggested = snp_df.sort_values('quantile_score', ascending=False)['biosample_name'].unique().tolist()[:5]
        else: suggested = []
        st.markdown("#### 🎯 Smart Tissue Selection")
        sel_portal_tissues = st.multiselect("Select Tissues for Deep Dive", biosamples, default=safe_defaults(suggested, biosamples))
        if st.button("🧬 Send to Molecular Deep Dive"):
            st.session_state['sel_var_portal'] = lookup_vid; st.session_state['sel_tissues_portal'] = sel_portal_tissues
            st.success(f"Configured for {lookup_vid}. Scroll down.")

st.divider(); st.subheader("2. Micro-View: Molecular Deep Dive")
if HAS_AG:
    portal_vid = st.session_state.get('sel_var_portal')
    portal_tissues = st.session_state.get('sel_tissues_portal', [])
    target_vars = sorted(filtered_df['variant_id'].unique().tolist())
    if portal_vid and portal_vid not in target_vars: target_vars.insert(0, portal_vid)
    if not target_vars: st.warning("No variants selected.")
    else:
        with st.expander("⚙️ Configure & Generate Tracks", expanded=True):
            # Vertical Stack for better usability
            selected_idx = target_vars.index(portal_vid) if portal_vid in target_vars else 0
            sel_var = st.selectbox("1. Select Variant for Analysis", target_vars, index=selected_idx, format_func=format_variant_label)
            
            var_context = get_variant_context(selected_file, sel_var)
            defaults = portal_tissues if sel_var == portal_vid and portal_tissues else (var_context.sort_values('quantile_score', ascending=False)['biosample_name'].head(3).tolist() if not var_context.empty else [])
            sel_names = st.multiselect("2. Select Tissues to Plot (Limit 50)", biosamples, default=safe_defaults(defaults, biosamples))
            
            c_opt, c_key, c_btn = st.columns([1, 2, 1])
            with c_opt:
                sync_y = st.checkbox("Sync Y-Axes", value=True, help="Use same scale for REF and ALT to show relative effect magnitude.")
            with c_key:
                api_key = st.text_input("AlphaGenome API Key", value=os.environ.get("ALPHAGENOME_API_KEY", ""), type="password")
            with c_btn:
                st.write("") # spacer
                btn_label = "🧬 Generate Tracks"
                if st.button(btn_label, type="primary", width='stretch', disabled=not (api_key and sel_names)):
                    with st.spinner("Querying AlphaGenome..."):
                        try:
                            sel_curies = [global_curie_map[n] for n in sel_names if n in global_curie_map]
                            client = dna_client.create(api_key=api_key)
                            chrom, pos, ref, alt = parse_variant_id(sel_var)
                            chrom_n = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
                            var_obj = genome.Variant(chromosome=chrom_n, position=pos, reference_bases=ref, alternate_bases=alt, name=sel_var)
                            interval = var_obj.reference_interval.resize(131072)
                            request_splice = False
                            if not var_context.empty:
                                splice_scores = var_context[var_context['output_type'] == 'SPLICE_JUNCTIONS']
                                if not splice_scores.empty and splice_scores['quantile_score'].abs().max() > 0.5: request_splice = True
                            req_outputs = [dna_output.OutputType.RNA_SEQ, dna_output.OutputType.ATAC]
                            if request_splice: req_outputs.append(dna_output.OutputType.SPLICE_JUNCTIONS)
                            res = client.predict_variant(interval=interval, variant=var_obj, requested_outputs=req_outputs, ontology_terms=sel_curies, organism=dna_client.Organism.HOMO_SAPIENS)
                            components = []
                            if res.reference.rna_seq and res.alternate.rna_seq: components.append(plot_components.OverlaidTracks({'REF': res.reference.rna_seq, 'ALT': res.alternate.rna_seq}, ylabel_template='RNA-Seq\n(Expr)', alpha=0.6, shared_y_scale=sync_y))
                            if res.reference.atac and res.alternate.atac: components.append(plot_components.OverlaidTracks({'REF': res.reference.atac, 'ALT': res.alternate.atac}, ylabel_template='ATAC-Seq\n(Access)', alpha=0.6, shared_y_scale=sync_y))
                            if request_splice and res.reference.splice_junctions:
                                components.append(plot_components.Sashimi(res.reference.splice_junctions, ylabel_template='Splice(REF)'))
                                components.append(plot_components.Sashimi(res.alternate.splice_junctions, ylabel_template='Splice(ALT)'))
                            if components:
                                st.session_state['fig_out'] = plot_components.plot(components, interval, annotations=[plot_components.VariantAnnotation([var_obj], labels=[f"{ref}>{alt}"])], fig_width=18, hspace=0.4, despine=True, xlabel='Genomic Position (GRCh38)', title=f"Ref vs Alt Signal: {format_variant_label(sel_var)}")
                                st.success("Tracks generated!")
                            else: st.warning("No signal tracks returned.")
                        except Exception as e: st.error(f"API Error: {e}")
        if 'fig_out' in st.session_state: st.pyplot(st.session_state['fig_out'])
        else: st.info("Configure controls above and click 'Generate Tracks'.")
else: st.warning("AlphaGenome not installed.")