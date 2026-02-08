# DecodeME Functional Workbench: User Guide

This guide documents the **DecodeME Functional Workbench** (`visualize_decodeme.py`), an interactive dashboard designed to transform raw AlphaGenome predictions into biological insights. It serves as the visualization companion to the statistical pipeline (`analyze_decodeme.py`).

---

## 🚀 Quick Start

Ensure your virtual environment is active and your API keys are set.

```bash
# 1. Set Keys (if not already set)
export ALPHAGENOME_API_KEY="your_key"

# 2. Launch the Workbench
./.venv/bin/streamlit run visualize_decodeme.py
```

The tool will open in your browser at `http://localhost:8501`.

---

## 🖥️ Workbench Features

### 1. Data Source & Primary Filters (Sidebar)
*   **Results File:** The app automatically scans your directory for `.csv` output files (excluding caches). Select the dataset you wish to explore.
*   **Locus ID:** The primary filter. Selecting a locus (e.g., `20:48914387...`) instantly narrows the analysis to that genomic region.
*   **Score Threshold:** Filters variants by their **absolute functional impact** (Quantile Score).
    *   *Default (0.5):* Shows moderate-to-high impact variants.
    *   *High (0.9):* Shows only the strongest "driver" candidates.
*   **Leads Sig. Threshold:** Adjusts the threshold (e.g. 7.3) used to label sub-significant variants in the candidate table.

### 2. Actionable Intelligence Hub (Top Panel)
This section answers the question: *"Is the statistical lead also the functional lead?"*

*   **Split-Screen Metrics:**
    *   **Left (GWAS):** Displays the P-value of the statistically strongest SNP in the locus.
    *   **Right (Functional):** Displays the AlphaGenome score of the biologically strongest SNP.
*   **Interactive Candidate Table:** Lists the top functional candidates.
    *   **Filtering Logic:** To ensure diversity, the table identifies the top 30 **distinct variants** first, then displays the single most impactful tissue for each. This prevents one "super-variant" from dominating the entire list.
    *   **GWAS P:** Shows the P-value for each candidate. Sub-significant credible variants are labeled as `< 5e-08` (or your chosen threshold).
    *   **Sync Action:** **Clicking any row** instantly synchronizes the entire dashboard. The "Micro-View" will pre-load that exact **Variant** and **Tissue**.

### 3. Macro-View: Functional Landscape
Visualize patterns across the entire locus.

*   **🔥 Functional Fingerprint (Heatmap):**
    *   **Labels:** Axes display the **Genomic Position** and **Allele Swap** (e.g., `48,916,462 (A>G)`) for clear identification.
    *   **System Grouping:** Tissues are grouped by biological system (CNS, Immune, Metabolic).
    *   **Diverging Scale:** Red (Gain) vs. Blue (Loss).
*   **🔗 Mechanism (Scatter Plot):**
    *   Correlates scores between two assays (e.g., RNA-Seq vs. ATAC-Seq) to propose mechanisms (e.g. enhancer-driven expression).

### 4. Micro-View: Molecular Deep Dive
The "Microscope" for proving causality.

*   **Configuration:**
    *   **Tissue Selection Mode:**
        *   **"Synced from Table" (Default):** Focuses on the specific tissue you clicked in the candidate table.
        *   **"Top 5 Global Hits":** Automatically loads the 5 highest-scoring tissues for that variant, regardless of the macro-view.
    *   **Show Difference Tracks:** When checked, plots a **Delta Track** (`ALT - REF`) below the overlaid signals. Red filled areas indicate increased signal; Blue indicate decreased signal.
*   **Generated Tracks:**
    *   **Rich Labels:** Y-axes display unambiguous context: `{Tissue Name} ({Type}) | {Assay}` (e.g., *Tibial Nerve (tissue) | RNA-Seq*).
    *   **Signal Tracks:** Overlays Reference (Blue) vs. Alternate (Orange) signals.
    *   **Variant Annotation:** A vertical line explicitly marks the mutation site (e.g., `A > G site`).
    *   **Splicing Arcs:** Automatically appear if the variant has a high `SPLICE_JUNCTIONS` impact score.
*   **Metadata Inspection:** An expander below the plot reveals the full metadata table (Donor ID, Age, Sex, etc.) for the visualized tracks.

---

## 🧠 Design Decisions & Assumptions

### 1. Memory vs. Speed (The Polars Decision)
*   **Design:** The tool uses **Polars LazyFrames** and **Streaming** execution.
*   **Reason:** The raw results file can exceed 1.2GB. Loading this into Pandas causes Out-of-Memory (OOM) crashes on standard machines.
*   **Impact:** The app scans the file on disk and only loads the tiny fraction of rows needed for the current view.

### 2. The "Credible Set" Disconnect
*   **Observation:** You may see `N/A` or `< 5e-08` in the "GWAS P" column for strong functional candidates.
*   **Reason:** The "Leads" cache only stores genome-wide significant hits ($P < 5 \times 10^{-8}$). However, fine-mapping Credible Sets include sub-significant variants if they are in high LD with a lead.
*   **Decision:** We label these clearly rather than hiding them, as they are often biologically critical "functional proxies."

### 3. Difference Track Calculation
*   **Method:** The "Difference" track is calculated mathematically as `Alternate_Value - Reference_Value` at every base pair.
*   **Visualization:** It uses a filled plot with a diverging colormap centered at zero. This provides an instant visual readout of the *magnitude* and *direction* of the variant's effect, which can be subtle in overlaid plots.

---

## ❓ FAQ & Troubleshooting

**Q: Why is only one tissue shown in the Deep Dive when I click the table?**
**A:** Because you are in **"Synced from Table"** mode. The tool assumes you want to investigate the specific finding you clicked on. To see more tissues, switch the radio button to **"Top 5 Global Hits"** or manually add them in the dropdown.

**Q: Why did the API return an error about 'assay_title'?**
**A:** We use robust fallback logic now. If a specific metadata field is missing from the AlphaGenome response, the label will degrade gracefully to a simpler format (e.g., just the output type) rather than crashing.

**Q: Why don't I see splicing arcs?**
**A:** Splicing tracks are visually complex. The app checks the `SPLICE_JUNCTIONS` score for the selected variant. Only if it exceeds `0.5` (predicting a splice defect) does it request and plot the splicing data.

**Q: Can I analyze a variant that isn't in my file?**
**A:** Yes! Go to the **"🔎 Quick Lookup"** tab, check **"Manual Entry"**, and type any variant ID. You can then send this "theoretical" variant to the Deep Dive to get real-time AlphaGenome predictions.