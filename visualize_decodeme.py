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
            cols = ['variant_id', 'biosample_name', 'output_type', 'quantile_score', 'ontology_curie']
            q = pl.scan_csv(file_path).filter(pl.col('variant_id') == variant_id)
            schema = q.collect_schema().names()
            existing_cols = [c for c in cols if c in schema]
            return q.select(existing_cols).collect(engine="streaming").to_pandas()
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
def get_snps_with_meta_lazy(file_path, chrom):
    if HAS_POLARS:
        try:
            lf = pl.scan_csv(file_path)
            schema = lf.collect_schema().names()
            if "variant_id" in schema and "quantile_score" in schema:
                if "CHROM" in schema: q = lf.filter(pl.col("CHROM").astype(pl.Utf8) == str(chrom))
                else: q = lf.filter(pl.col("variant_id").str.starts_with(f"{chrom}:") | 
                                  pl.col("variant_id").str.starts_with(f"chr{chrom}:"))
                q = q.group_by("variant_id").agg(pl.col("quantile_score").abs().max().alias("max_impact"))
                return q.sort("max_impact", descending=True).limit(500).collect(engine="streaming").to_pandas()
        except Exception: pass
    return pd.DataFrame(columns=['variant_id', 'max_impact'])

@st.cache_data
def load_heatmap_data(file_path, locus_id, score_thresh, genes=None, biosamples=None, assays=None):
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path)
            schema = q.collect_schema().names()
            if locus_id != 'All' and "Locus_ID" in schema: q = q.filter(pl.col('Locus_ID') == locus_id)
            if score_thresh > 0 and "quantile_score" in schema: q = q.filter(pl.col('quantile_score').abs() >= score_thresh)
            if genes and "gene_name" in schema: q = q.filter(pl.col('gene_name').is_in(genes))
            if biosamples and "biosample_name" in schema: q = q.filter(pl.col('biosample_name').is_in(biosamples))
            if assays and "output_type" in schema: q = q.filter(pl.col('output_type').is_in(assays))
            q = q.group_by(['variant_id', 'biosample_name']).agg(pl.col('quantile_score').mean())
            return q.collect(engine="streaming").to_pandas()
        except Exception as e: st.error(f"Heatmap agg failed: {e}")
    return pd.DataFrame()

@st.cache_data
def load_top_hits(file_path, locus_id, score_thresh):
    if HAS_POLARS:
        try:
            q = pl.scan_csv(file_path)
            schema = q.collect_schema().names()
            cols = ['variant_id', 'biosample_name', 'output_type', 'quantile_score']
            if "Locus_ID" in schema and locus_id != 'All': q = q.filter(pl.col('Locus_ID') == locus_id)
            existing = [c for c in cols if c in schema]
            q = q.select(existing)
            if "quantile_score" in schema:
                q = q.filter(pl.col('quantile_score').abs() >= score_thresh)
            return q.collect(engine="streaming").to_pandas()
        except Exception: pass
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

def format_pval(log10p, threshold=7.3, has_leads=True):
    """Converts LOG10P to scientific notation or sub-significant label."""
    try:
        if pd.isna(log10p):
            if has_leads:
                # Variant exists in results but not in leads cache -> it was below threshold
                return f"< {10**(-threshold):.0e}"
            else:
                return "N/A"
        lp = float(log10p)
        if lp < 0: return "N/A"
        return f"{10**(-lp):.2e}"
    except (ValueError, TypeError): return "N/A"

def normalize_id_for_join(id_str):
    if not isinstance(id_str, str): return id_str
    return id_str.lower().replace('chr', '').replace('>', ':').strip()

# --- Main App ---

st.title("🧬 DecodeME Functional Workbench")

with st.sidebar:
    st.header("Data Source")
    files = get_available_files()
    if not files: st.error("No CSV found."); st.stop()
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
    score_threshold = st.slider("Min. Functional Score (Abs)", 0.0, 1.0, 0.5, 0.05)
    leads_threshold = st.slider("Leads Sig. Threshold (LOG10P)", 5.0, 10.0, 7.3, 0.1, help="The threshold used during 'analyze_decodeme.py' to define Lead SNPs.")
    
    st.header("Refine View")
    sel_genes = st.multiselect("Filter by Genes", gene_names)
    sel_biosamples = st.multiselect("Filter by Tissues", biosamples)
    sel_assays = st.multiselect("Filter by Assay Types", output_types)
    st.divider()
    st.markdown("**Candidate Table Options**")
    unique_pos = st.checkbox("Unique Genomic Positions Only", value=True)
    sort_by = st.selectbox("Sort Candidates By", ["Functional Impact", "GWAS P-value"])

with st.spinner("Analyzing Top Candidates..."):
    top_hits = load_top_hits(selected_file, selected_locus, score_threshold)

if not top_hits.empty and selected_locus != 'All':
    st.markdown("### 📍 Locus Context & Candidate Selection")
    lead_p, lead_id = "N/A", "N/A"
    has_leads_data = not leads_df.empty and 'ID' in leads_df.columns
    
    if has_leads_data:
        locus_gwas = leads_df[leads_df['ID'] == selected_locus]
        if not locus_gwas.empty:
            lead_p, lead_id = format_pval(locus_gwas['LOG10P'].iloc[0], leads_threshold, True), locus_gwas['ID'].iloc[0]

    top_row = top_hits.loc[top_hits['quantile_score'].abs().idxmax()]
    c_stat, c_func = st.columns(2)
    with c_stat: st.info(f"**Statistical Context (GWAS)**\n\n**Lead SNP:** {lead_id}\n\n**P-value:** {lead_p}")
    with c_func: st.success(f"**Functional Evidence (AlphaGenome)**\n\n**Top Variant:** {top_row['variant_id'].split(':')[-1]}\n\n**Max Impact:** {top_row['quantile_score']:.2f} ({top_row['biosample_name']})")

    st.markdown("**Interactive Candidate Table:** Select a row to sync with the Deep Dive.")
    working_hits = top_hits.copy()
    if has_leads_data:
        working_hits.loc[:, 'join_key'] = working_hits['variant_id'].apply(normalize_id_for_join)
        leads_copy = leads_df[['ID', 'LOG10P']].copy()
        leads_copy.loc[:, 'join_key'] = leads_copy['ID'].apply(normalize_id_for_join)
        leads_copy = leads_copy.drop_duplicates('join_key')
        enriched = pd.merge(working_hits, leads_copy[['join_key', 'LOG10P']], on='join_key', how='left')
    else:
        enriched = working_hits
        enriched.loc[:, 'LOG10P'] = np.nan

    if unique_pos:
        enriched.loc[:, 'genpos'] = enriched['variant_id'].apply(lambda x: parse_variant_id(x)[1])
        enriched.loc[:, 'abs_quantile'] = enriched['quantile_score'].abs()
        idx = enriched.groupby('genpos')['abs_quantile'].idxmax()
        enriched = enriched.loc[idx].drop(columns=['abs_quantile', 'genpos'])
    
    if sort_by == "GWAS P-value": enriched = enriched.sort_values('LOG10P', ascending=False)
    else: enriched = enriched.sort_values('quantile_score', key=abs, ascending=False)

    display_df = enriched.head(20).copy()
    display_df.loc[:, 'GWAS P'] = display_df['LOG10P'].apply(lambda x: format_pval(x, leads_threshold, has_leads_data))
    display_df.loc[:, 'Variant'] = display_df['variant_id'].apply(format_variant_label)
    cols_f = ['Variant', 'GWAS P', 'biosample_name', 'output_type', 'quantile_score']
    selection = st.dataframe(display_df[[c for c in cols_f if c in display_df.columns]], width='stretch', hide_index=True, on_select="rerun", selection_mode="single-row")
    if selection.selection.rows:
        st.session_state['sel_var_portal'] = enriched.iloc[selection.selection.rows[0]]['variant_id']
        st.toast(f"Synchronized with {st.session_state['sel_var_portal']}")
    st.divider()

st.subheader("1. Macro-View: Functional Landscape")
tab1, tab2, tab3 = st.tabs(["🔥 Functional Fingerprint", "🔗 Mechanism", "🔎 Quick Lookup"])

with tab1:
    h_thresh = st.slider("Heatmap Score Threshold", 0.0, 1.0, score_threshold, 0.05, key="h_t")
    with st.spinner("Aggregating heatmap..."):
        heatmap_data = load_heatmap_data(selected_file, selected_locus, h_thresh, genes=sel_genes, biosamples=sel_biosamples, assays=sel_assays)
    if not heatmap_data.empty:
        heatmap_data = heatmap_data.copy()
        heatmap_data.loc[:, 'System'] = heatmap_data['biosample_name'].apply(assign_biological_system)
        available_systems = sorted(heatmap_data['System'].unique())
        c1, c2 = st.columns([2, 1])
        selected_system = c1.selectbox("System", available_systems)
        orient = c2.radio("Axis", ["Variant focus", "Tissue focus"], horizontal=True)
        sys_df = heatmap_data[heatmap_data['System'] == selected_system].copy()
        plot_data = sys_df.groupby(['variant_id', 'biosample_name'])['quantile_score'].mean().reset_index()
        plot_data.loc[:, 'display_id'] = plot_data['variant_id'].apply(lambda x: x.split(':')[-1])
        x_c, y_c = ("biosample_name", "display_id") if orient == "Variant focus" else ("display_id", "biosample_name")
        fig = px.density_heatmap(plot_data, x=x_c, y=y_c, z="quantile_score", color_continuous_scale="RdBu_r", range_color=[-1, 1], title=f"Impact: {selected_system}")
        fig.update_layout(height=max(400, min(1200, len(plot_data[y_c].unique()) * 25)))
        st.plotly_chart(fig, width="stretch")
    else: st.info("No data passing filters.")

with tab2:
    st.info("Correlations require sub-threshold data. Tab pending optimization.")

with tab3:
    st.markdown("### 🔎 Quick Lookup Portal")
    col_c, col_s = st.columns([1, 3])
    with col_c: selected_chrom = st.selectbox("Chromosome", all_chroms)
    with col_s:
        is_manual = st.checkbox("Manual Entry")
        if is_manual: lookup_vid = st.text_input("Variant ID", "")
        else:
            with st.spinner("Indexing SNPs..."): snp_meta = get_snps_with_meta_lazy(selected_file, selected_chrom)
            if not snp_meta.empty:
                snp_labels = [f"{row['variant_id'].split(':')[-1]} (Impact: {row['max_impact']:.2f})" for _, row in snp_meta.iterrows()]
                label_to_id = dict(zip(snp_labels, snp_meta['variant_id']))
                sel_label = st.selectbox("Search SNPs", snp_labels)
                lookup_vid = label_to_id.get(sel_label)
            else: lookup_vid = None
    if lookup_vid:
        st.divider()
        snp_df = get_variant_context(selected_file, lookup_vid)
        suggested = snp_df.sort_values('quantile_score', ascending=False)['biosample_name'].unique().tolist()[:5] if not snp_df.empty else []
        sel_portal_tissues = st.multiselect("Select Tissues", biosamples, default=safe_defaults(suggested, biosamples))
        if st.button("🧬 Send to Molecular Deep Dive", key="portal_btn"):
            st.session_state['sel_var_portal'] = lookup_vid; st.session_state['sel_tissues_portal'] = sel_portal_tissues
            st.success("Synchronized!")

st.divider(); st.subheader("2. Micro-View: Molecular Deep Dive")
if HAS_AG:
    portal_vid = st.session_state.get('sel_var_portal')
    portal_tissues = st.session_state.get('sel_tissues_portal', [])
    target_vars = []
    if portal_vid: target_vars.append(portal_vid)
    if target_vars:
        with st.expander("⚙️ Configure & Generate Tracks", expanded=True):
            sel_var = st.selectbox("Variant", target_vars, index=0, format_func=format_variant_label)
            var_context = get_variant_context(selected_file, sel_var)
            defaults = portal_tissues if sel_var == portal_vid and portal_tissues else (var_context.sort_values('quantile_score', ascending=False)['biosample_name'].head(3).tolist() if not var_context.empty else [])
            sel_names = st.multiselect("Tissues (Max 50)", biosamples, default=safe_defaults(defaults, biosamples))
            c_opt, c_key, c_btn = st.columns([1, 2, 1])
            with c_opt: sync_y = st.checkbox("Sync Y-Axes", value=True)
            with c_key: api_key = st.text_input("AlphaGenome API Key", value=os.environ.get("ALPHAGENOME_API_KEY", ""), type="password")
            with c_btn:
                st.write(""); 
                if st.button("🧬 Generate Tracks", type="primary", width='stretch', disabled=not (api_key and sel_names)):
                    with st.spinner("Querying API..."):
                        try:
                            sel_curies = [global_curie_map[n] for n in sel_names if n in global_curie_map]
                            client = dna_client.create(api_key=api_key)
                            chrom, pos, ref, alt = parse_variant_id(sel_var)
                            chrom_n = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
                            var_obj = genome.Variant(chromosome=chrom_n, position=pos, reference_bases=ref, alternate_bases=alt, name=sel_var)
                            interval = var_obj.reference_interval.resize(131072)
                            request_splice = not var_context.empty and not var_context[var_context['output_type'] == 'SPLICE_JUNCTIONS'].empty
                            req = [dna_output.OutputType.RNA_SEQ, dna_output.OutputType.ATAC]
                            if request_splice: req.append(dna_output.OutputType.SPLICE_JUNCTIONS)
                            res = client.predict_variant(interval=interval, variant=var_obj, requested_outputs=req, ontology_terms=sel_curies, organism=dna_client.Organism.HOMO_SAPIENS)
                            comp = []
                            if res.reference.rna_seq: comp.append(plot_components.OverlaidTracks({'REF': res.reference.rna_seq, 'ALT': res.alternate.rna_seq}, ylabel_template='RNA-Seq', alpha=0.6, shared_y_scale=sync_y))
                            if res.reference.atac: comp.append(plot_components.OverlaidTracks({'REF': res.reference.atac, 'ALT': res.alternate.atac}, ylabel_template='ATAC-Seq', alpha=0.6, shared_y_scale=sync_y))
                            if request_splice and res.reference.splice_junctions:
                                comp.append(plot_components.Sashimi(res.reference.splice_junctions, ylabel_template='Splice(REF)'))
                                comp.append(plot_components.Sashimi(res.alternate.splice_junctions, ylabel_template='Splice(ALT)'))
                            if comp: st.session_state['fig_out'] = plot_components.plot(comp, interval, annotations=[plot_components.VariantAnnotation([var_obj], labels=[f"{ref}>{alt}"])], fig_width=18, hspace=0.4, despine=True, xlabel='Genomic Position (GRCh38)', title=f"Signal: {format_variant_label(sel_var)}")
                        except Exception as e: st.error(f"API Error: {e}")
        if 'fig_out' in st.session_state: st.pyplot(st.session_state['fig_out'])
    else: st.info("Use the Candidate Table or Quick Lookup to select a variant for Deep Dive.")
else: st.warning("AlphaGenome not installed.")