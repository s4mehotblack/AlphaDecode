import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# --- Configuration & Setup ---
st.set_page_config(
    page_title="AlphaGenome Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Library Imports ---
HAS_AG = False
try:
    import alphagenome.visualization.plot_components as plot_components
    from alphagenome.models import dna_client, variant_scorers, dna_output
    from alphagenome.data import genome
    HAS_AG = True
except ImportError:
    st.error("⚠️ AlphaGenome library not found. Please ensure it is installed in your environment.")

# --- Constants & Presets ---
COMMON_TISSUES = {
    "Brain (UBERON:0000955)": "UBERON:0000955",
    "Whole Blood (UBERON:0000178)": "UBERON:0000178",
    "T-cell (CL:0000084)": "CL:0000084",
    "Liver (UBERON:0002107)": "UBERON:0002107",
    "Heart (UBERON:0000948)": "UBERON:0000948",
    "Skeletal Muscle (UBERON:0001134)": "UBERON:0001134",
    "Lung (UBERON:0002048)": "UBERON:0002048",
    "Kidney (UBERON:0002113)": "UBERON:0002113",
    "Skin (UBERON:0002097)": "UBERON:0002097",
    "Bone Marrow (UBERON:0002371)": "UBERON:0002371"
}

# --- Helper Functions ---

def parse_variant_string(var_str):
    """Parses 'chr:pos:ref:alt' string."""
    try:
        parts = var_str.strip().split(':')
        if len(parts) == 4:
            chrom = parts[0].replace('chr', '')
            pos = int(parts[1])
            ref = parts[2].upper()
            alt = parts[3].upper()
            return chrom, pos, ref, alt
    except Exception:
        pass
    return None, None, None, None

def get_scorers_list():
    """Returns list of available scorer keys."""
    if HAS_AG:
        try:
            return list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.keys())
        except AttributeError:
            return ['RNA_SEQ', 'ATAC_ACTIVE', 'SPLICE_JUNCTIONS']
    return []

# --- UI: Sidebar ---

with st.sidebar:
    st.title("🧬 AG Explorer")
    
    # 1. Authentication
    st.subheader("1. Authentication")
    api_key_env = os.environ.get("ALPHAGENOME_API_KEY", "")
    api_key = st.text_input("API Key", value=api_key_env, type="password")
    
    # 2. Variant Input
    st.subheader("2. Variant Definition")
    input_mode = st.radio("Input Mode", ["Manual Fields", "String Parse"], horizontal=True)
    
    sel_chrom, sel_pos, sel_ref, sel_alt = None, None, None, None
    
    if input_mode == "Manual Fields":
        col_c, col_p = st.columns([1, 2])
        with col_c:
            chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
            sel_chrom = st.selectbox("Chr", chroms, index=0)
        with col_p:
            sel_pos = st.number_input("Position (1-based)", min_value=1, value=100000)
        
        col_r, col_a = st.columns(2)
        with col_r: sel_ref = st.text_input("Ref Allele", "A")
        with col_a: sel_alt = st.text_input("Alt Allele", "G")
    else:
        var_str = st.text_input("Variant String", placeholder="chr1:12345:A:G")
        if var_str:
            c, p, r, a = parse_variant_string(var_str)
            if c:
                st.success(f"Parsed: Chr{c} : {p} ({r} > {a})")
                sel_chrom, sel_pos, sel_ref, sel_alt = c, p, r, a
            else:
                st.error("Invalid format. Use chr:pos:ref:alt")

    # 3. Model Configuration
    st.subheader("3. Model Config")
    # Organism (Placeholder for now, assuming Homo Sapiens is default/only option easily exposed)
    organism_name = st.selectbox("Organism", ["HOMO_SAPIENS"])
    
    # Scorers
    available_scorers = get_scorers_list()
    default_scorers = [s for s in available_scorers if s in ['RNA_SEQ', 'ATAC_ACTIVE']]
    selected_scorers = st.multiselect("Variant Scorers", available_scorers, default=default_scorers)

    # 4. Tissues / Biosamples
    st.subheader("4. Tissues / Biosamples")
    tissue_mode = st.radio("Selection Mode", ["Curated List", "Manual IDs"], horizontal=True)
    
    selected_ontology_ids = []
    if tissue_mode == "Curated List":
        sel_names = st.multiselect("Select Tissues", list(COMMON_TISSUES.keys()), default=["Brain (UBERON:0000955)"])
        selected_ontology_ids = [COMMON_TISSUES[n] for n in sel_names]
    else:
        custom_ids = st.text_area("Enter Ontology IDs (one per line)", "UBERON:0000955\\nCL:0000084")
        selected_ontology_ids = [line.strip() for line in custom_ids.splitlines() if line.strip()]

    # 5. Output Types (Tracks)
    st.subheader("5. Visualization Tracks")
    track_types = st.multiselect("Generate Tracks For:", ["RNA_SEQ", "ATAC_SEQ", "SPLICE_JUNCTIONS"], default=["RNA_SEQ", "ATAC_SEQ"])

# --- Main Page ---

st.header("AlphaGenome Functional Workbench")

if not HAS_AG:
    st.warning("AlphaGenome library is not installed. This tool requires the `alphagenome` python package.")
    st.stop()

if not api_key:
    st.info("Please enter your AlphaGenome API Key in the sidebar to proceed.")
    st.stop()

# Execution Block
if st.button("🚀 Fetch Predictions", type="primary"):
    if not (sel_chrom and sel_pos and sel_ref and sel_alt):
        st.error("Please define a valid variant.")
        st.stop()
    
    if not selected_ontology_ids:
        st.error("Please select at least one tissue/biosample.")
        st.stop()
        
    status = st.status("Running Analysis...", expanded=True)
    
    try:
        # 1. Initialize Client
        status.write("Initializing Client...")
        client = dna_client.create(api_key=api_key)
        
        # 2. Prepare Variant
        status.write("Preparing Variant Object...")
        chrom_str = f"chr{sel_chrom}" if not str(sel_chrom).startswith('chr') else str(sel_chrom)
        variant = genome.Variant(
            chromosome=chrom_str,
            position=int(sel_pos),
            reference_bases=sel_ref,
            alternate_bases=sel_alt
        )
        
        # Resize interval (defaulting to 1Mb for context, similar to scan script)
        interval = variant.reference_interval.resize(131072) # 128kb window for viz
        
        # 3. Determine Requested Outputs for Visualization
        req_outputs = []
        if "RNA_SEQ" in track_types: req_outputs.append(dna_output.OutputType.RNA_SEQ)
        if "ATAC_SEQ" in track_types: req_outputs.append(dna_output.OutputType.ATAC)
        if "SPLICE_JUNCTIONS" in track_types: req_outputs.append(dna_output.OutputType.SPLICE_JUNCTIONS)
        
        # 4. Fetch Prediction (Tracks)
        status.write(f"Fetching predictions for {len(selected_ontology_ids)} tissues...")
        prediction_res = client.predict_variant(
            interval=interval,
            variant=variant,
            requested_outputs=req_outputs,
            ontology_terms=selected_ontology_ids,
            organism=dna_client.Organism[organism_name]
        )
        
        # 5. Calculate Scores (Optional - if scorers selected)
        scores_df = pd.DataFrame()
        if selected_scorers:
            status.write("Calculating variant scores...")
            # We use a larger interval for accurate scoring if needed, but here we reuse or create new
            # Ideally scoring uses specific interval logic internal to score_variant usually
            scorers_to_use = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in selected_scorers]
            score_res = client.score_variant(
                interval=variant.reference_interval.resize(1048576), # 1Mb for scoring
                variant=variant,
                variant_scorers=scorers_to_use,
                organism=dna_client.Organism[organism_name]
            )
            if score_res:
                scores_df = variant_scorers.tidy_scores(score_res)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        # --- Visualization & Results ---
        
        tab_viz, tab_scores, tab_meta, tab_json = st.tabs(["🧬 Molecular Tracks", "📊 Variant Scores", "🔍 Track Metadata", "📄 Raw Response"])
        
        with tab_viz:
            st.subheader(f"Molecular Impact: {chrom_str}:{sel_pos} ({sel_ref}>{sel_alt})")
            
            # Visualization Controls
            c1, c2 = st.columns(2)
            sync_y = c1.checkbox("Sync Y-Axes (Ref/Alt)", value=True)
            show_diff = c2.checkbox("Show Difference Tracks", value=True)
            
            # Build Plot Components
            comp = []
            
            def add_tracks(ref_track, alt_track, label_base):
                if ref_track is not None and alt_track is not None:
                    # Check if empty
                    if not getattr(ref_track, 'values', np.array([])).size: return
                    
                    # Create Label
                    meta = ref_track.metadata
                    name = meta['biosample_name'].iloc[0] if 'biosample_name' in meta.columns else label_base
                    
                    comp.append(plot_components.OverlaidTracks(
                        {'REF': ref_track, 'ALT': alt_track},
                        ylabel_template=f"{name}\\n{label_base}",
                        shared_y_scale=sync_y,
                        alpha=0.6
                    ))
                    
                    if show_diff:
                        diff = alt_track - ref_track
                        comp.append(plot_components.Tracks(
                            diff,
                            ylabel_template="Difference",
                            track_colors='red',
                            filled=True
                        ))

            # RNA
            if "RNA_SEQ" in track_types:
                add_tracks(prediction_res.reference.rna_seq, prediction_res.alternate.rna_seq, "RNA-Seq")
            
            # ATAC
            if "ATAC_SEQ" in track_types:
                add_tracks(prediction_res.reference.atac, prediction_res.alternate.atac, "ATAC-Seq")

            # Splice
            if "SPLICE_JUNCTIONS" in track_types:
                if prediction_res.reference.splice_junctions:
                     comp.append(plot_components.Sashimi(prediction_res.reference.splice_junctions, ylabel_template='Ref Splice'))
                if prediction_res.alternate.splice_junctions:
                     comp.append(plot_components.Sashimi(prediction_res.alternate.splice_junctions, ylabel_template='Alt Splice'))

            # Gene Annotations
            try:
                comp.append(plot_components.GeneAnnotation(ylabel_template='Genes'))
            except Exception:
                pass # Often requires local DB

            if comp:
                fig = plot_components.plot(
                    comp,
                    interval,
                    annotations=[plot_components.VariantAnnotation([variant], labels=["Target"])],
                    fig_width=18,
                    xlabel=f"Position ({chrom_str})",
                    title=f"Variant Effect: {sel_ref} > {sel_alt}"
                )
                st.pyplot(fig)
            else:
                st.warning("No track data returned for the selected tissues.")

        with tab_scores:
            if not scores_df.empty:
                st.dataframe(scores_df.style.background_gradient(cmap="RdBu_r", subset=['score']), use_container_width=True)
                csv = scores_df.to_csv(index=False)
                st.download_button("Download Scores CSV", csv, "alphagenome_scores.csv", "text/csv")
            else:
                st.info("No scores calculated (did you select scorers?).")

        with tab_meta:
            meta_dfs = []
            if prediction_res.reference.rna_seq: meta_dfs.append(prediction_res.reference.rna_seq.metadata.assign(Type="RNA_SEQ"))
            if prediction_res.reference.atac: meta_dfs.append(prediction_res.reference.atac.metadata.assign(Type="ATAC_SEQ"))
            
            if meta_dfs:
                full_meta = pd.concat(meta_dfs, ignore_index=True)
                st.dataframe(full_meta, use_container_width=True)
            else:
                st.info("No metadata available.")

        with tab_json:
            st.json(str(prediction_res))

    except Exception as e:
        status.update(label="Error Occurred", state="error")
        st.error(f"Analysis Failed: {str(e)}")
        st.info("Tip: Check your API Key and ensure the Tissue IDs are valid.")
