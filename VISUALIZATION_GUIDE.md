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

### 2. Actionable Intelligence Hub (Top Panel)
This section answers the question: *"Is the statistical lead also the functional lead?"*

*   **Split-Screen Metrics:**
    *   **Left (GWAS):** Displays the P-value of the statistically strongest SNP in the locus.
    *   **Right (Functional):** Displays the AlphaGenome score of the biologically strongest SNP.
    *   *Insight:* If these IDs differ, the causal variant may be a functional proxy, not the GWAS lead.
*   **Interactive Candidate Table:** Lists the top 15 functional candidates.
    *   **GWAS P:** Shows the P-value for each candidate (joined from your leads cache).
    *   **Sync Action:** **Clicking any row** instantly synchronizes the entire dashboard (Deep Dive & Quick Lookup) to that specific variant and tissue.

### 3. Macro-View: Functional Landscape
Visualize patterns across the entire locus.

*   **🔥 Functional Fingerprint (Heatmap):**
    *   **System Grouping:** Tissues are automatically grouped by biological system (CNS, Immune, Metabolic) to reduce clutter.
    *   **Diverging Scale:**
        *   🔴 **Red:** Gain of Function (Increased expression/accessibility).
        *   🔵 **Blue:** Loss of Function (Decreased expression/accessibility).
    *   **Axis Toggle:** Switch between "Variant Focus" (comparing one SNP across tissues) and "Tissue Focus" (comparing many SNPs in one tissue).
*   **🔗 Mechanism (Scatter Plot):**
    *   Correlates scores between two assays (e.g., RNA-Seq vs. ATAC-Seq) to propose mechanisms.
    *   *Example:* A variant in the top-right quadrant increases chromatin accessibility (ATAC) AND gene expression (RNA), suggesting an enhancer gain-of-function.

### 4. Micro-View: Molecular Deep Dive
The "Microscope" for proving causality.

*   **Configuration:**
    *   **Variant:** Pre-filled from your selection in the table/heatmap.
    *   **Tissue Selection:** Choose specific tissues to visualize. Defaults to the tissue you clicked on ("Synced") but can be toggled to show the "Top 5 Global Hits".
    *   **Sync Y-Axes:** When checked (default), forces Reference and Alternate tracks to share the same Y-axis scale, visually emphasizing the *magnitude* of the change.
*   **Generated Tracks:**
    *   **Signal Tracks:** Overlays the predicted signal for Reference (Blue) vs. Alternate (Orange) alleles.
    *   **Splicing Arcs (Sashimi):** Automatically appears if the variant is predicted to disrupt splicing (`Score > 0.5`).
    *   **Variant Annotation:** A vertical line explicitly marks the mutation site (e.g., `A > G site`).

---

## 🧠 Design Decisions & Assumptions

### 1. Memory vs. Speed (The Polars Decision)
*   **Design:** The tool uses **Polars LazyFrames** and **Streaming** execution.
*   **Reason:** The raw results file can exceed 1.2GB. Loading this into Pandas on a standard laptop causes Out-of-Memory (OOM) crashes.
*   **Impact:** The app scans the file on disk and only loads the tiny fraction of rows needed for the current view.

### 2. The "Credible Set" Disconnect
*   **Observation:** You may see `N/A` in the "GWAS P" column for some strong functional candidates.
*   **Reason:** The "Leads" cache only stores genome-wide significant hits ($P < 5 	imes 10^{-8}$). However, the "Credible Set" includes sub-significant variants ($P \approx 10^{-5}$) if they are in high LD with a lead.
*   **Decision:** We label these as `< 5e-08` or `N/A` rather than hiding them, as they are often biologically critical despite lower statistical significance.

### 3. Automated Splicing Detection
*   **Design:** Sashimi plots are not shown by default.
*   **Reason:** Splicing tracks are visually complex.
*   **Logic:** The app checks the `SPLICE_JUNCTIONS` score for the selected variant. Only if it exceeds `0.5` (predicting a splice defect) does it request the splicing data from the API.

---

## ❓ FAQ & Troubleshooting

**Q: Why do I see "No data passing filters" in the Heatmap?**
**A:** Your `Score Threshold` might be too high. Try lowering it to `0.1`. Alternatively, you may have selected a filter (e.g., "Gene: IL2RA") that doesn't exist in the selected Locus.

**Q: Why are the GWAS P-values "N/A" in the top table?**
**A:** This usually means the variant is a "functional proxy" that did not reach genome-wide significance in the GWAS but was included in the analysis because it is linked to a Lead SNP.

**Q: Why is my specific tissue missing from the "Deep Dive" list?**
**A:** The list defaults to the top 3-5 tissues where the variant has the *highest impact*. To see a specific tissue, select it manually from the "Select Tissues" dropdown or click its cell in the Heatmap to "Sync" it.

**Q: Why did the app crash with `StreamlitAPIException`?**
**A:** This was a known issue with list synchronization (fixed in v2.0). It occurred when a "Default" tissue selected in one view didn't exist in the filtered context of another view. The current version uses a `safe_defaults` validator to prevent this.

**Q: Can I analyze a variant that isn't in my file?**
**A:** Yes! Go to the **"🔎 Quick Lookup"** tab, check **"Manual Entry"**, and type any variant ID (e.g., `chr6:12345:A:T`). You can then send this "theoretical" variant to the Deep Dive to get real-time AlphaGenome predictions.
