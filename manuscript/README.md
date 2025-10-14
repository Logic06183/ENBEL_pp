# ENBEL Climate-Health Manuscript

LaTeX manuscript for publication: "Climate-Driven Variation in Health Biomarkers: A Machine Learning Analysis of HIV Clinical Cohorts in Johannesburg, South Africa"

## Quick Start

### Prerequisites

You'll need a LaTeX distribution installed:

**macOS:**
```bash
# Using MacTeX (recommended)
brew install --cask mactex

# Or BasicTeX for a smaller install
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-latexextra
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**Windows:**
- Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Compilation

#### Option 1: Using latexmk (Recommended)

```bash
cd manuscript
latexmk -pdf main.tex
```

This automatically handles multiple compilation passes needed for references, citations, and cross-references.

#### Option 2: Manual compilation

```bash
cd manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

#### Option 3: Using VS Code

1. Install the [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension
2. Open `main.tex`
3. Click "Build LaTeX project" or press `Ctrl+Alt+B` (Windows/Linux) or `Cmd+Alt+B` (macOS)

### Viewing the PDF

The compiled PDF will be located at:
```
manuscript/main.pdf
```

## File Structure

```
manuscript/
├── main.tex              # Main manuscript file
├── references.bib        # BibTeX bibliography
├── figures/             # Directory for figure files
│   └── .gitkeep
├── tables/              # Directory for table files (if separated)
│   └── .gitkeep
└── sections/            # Optional: separate section files
    ├── introduction.tex
    ├── methods.tex
    ├── results.tex
    └── discussion.tex
```

## Writing Workflow

### 1. Development Mode

Enable line numbers for manuscript review:
```latex
\linenumbers  % Already enabled in main.tex
```

### 2. Adding Figures

Place figures in the `figures/` directory and reference them:

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/performance_tiers.pdf}
    \caption{Your caption here.}
    \label{fig:performance_tiers}
\end{figure}
```

Supported formats:
- Vector graphics: `.pdf`, `.eps` (preferred for publication)
- Raster graphics: `.png`, `.jpg` (use high resolution: 300+ DPI)

### 3. Adding References

Add citations to `references.bib`:

```bibtex
@article{author2024,
  title={Article Title},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  volume={10},
  pages={123--145},
  year={2024}
}
```

Cite in text:
```latex
\citep{author2024}           % (Author, 2024)
\citet{author2024}           % Author (2024)
\citep{author2024,other2023} % Multiple citations
```

### 4. Adding Tables

```latex
\begin{table}[ht]
    \centering
    \caption{Table caption}
    \label{tab:results}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\
        \midrule
        Data 1 & 0.928 & $p < 0.001$ \\
        Data 2 & 0.390 & $p < 0.01$ \\
        \bottomrule
    \end{tabular}
\end{table}
```

## Manuscript Sections

Current outline:

1. **Abstract** - Completed draft
2. **Introduction** - TODO
   - Climate health in the Global South
   - Promise of ML for climate-health research
   - Study objectives
3. **Methods** - TODO
   - Study setting and population
   - Climate data integration
   - Machine learning pipeline
   - Statistical analysis
4. **Results** - Partial draft
   - Biomarker performance tiers
   - Hematocrit findings
   - Lipid metabolism
   - Immune markers
5. **Discussion** - TODO
   - Principal findings
   - Clinical implications
   - Public health recommendations
   - Methodological considerations
   - Limitations
   - Future directions
6. **Conclusions** - Completed draft

## Key Findings to Highlight

Based on `results/refined_analysis/COMPLETE_BIOMARKER_ANALYSIS.md`:

### Tier 1: Excellent (R² > 0.30)
- **Hematocrit**: R² = 0.928 (flagship finding)
- **Lipids**: Total cholesterol (0.390), LDL (0.370), HDL (0.330)
- **Metabolic**: Creatinine (0.306)

### Tier 2: Moderate (R² = 0.05-0.30)
- Glucose, blood pressure markers

### Tier 3: Poor (R² < 0.05)
- CD4, ALT, AST (require DLNM approaches)

## Figures to Create

Priority figures needed:

1. **Figure 1**: Study design flowchart
2. **Figure 2**: Geographic distribution of samples
3. **Figure 3**: Biomarker performance tiers (bar plot with R² values)
4. **Figure 4**: Hematocrit SHAP analysis (waterfall + beeswarm)
5. **Figure 5**: Temporal patterns in top biomarkers
6. **Figure 6**: Climate variable importance across biomarkers
7. **Supplementary**: Individual SHAP plots for all biomarkers

## Data Sources for Figures

Figures can be generated from:
- Analysis results: `../results/refined_analysis/`
- SHAP plots: `../results/refined_analysis/shap_plots/`
- Scripts: `../scripts/visualization/`

## Journal Target

Consider targeting:
- **Environmental Health Perspectives** (impact factor: 11.0)
- **The Lancet Planetary Health** (impact factor: 24.1)
- **Nature Climate Change** (impact factor: 30.7)
- **GeoHealth** (open access, climate-health focus)

Each journal has specific formatting requirements - check author guidelines before submission.

## Citation Style

Currently using `natbib` with `plainnat` style (author-year format). Common alternatives:

- `plainnat` - Author-year format
- `abbrvnat` - Abbreviated author-year
- `unsrtnat` - Unsorted, numbered
- `apa` - APA style
- `vancouver` - Vancouver numbered style (common in medical journals)

Change in `main.tex`:
```latex
\bibliographystyle{plainnat}  % Change to desired style
```

## Submission Checklist

Before submission:

- [ ] Remove line numbers (`\linenumbers`)
- [ ] Check all TODOs are addressed
- [ ] Verify all figures have high-resolution versions
- [ ] Ensure all citations are in references.bib
- [ ] Run spell check
- [ ] Verify author names, affiliations, and order
- [ ] Add funding information
- [ ] Add data availability statement
- [ ] Add ethics statement (if required)
- [ ] Add competing interests statement
- [ ] Check journal-specific formatting requirements
- [ ] Generate final PDF
- [ ] Create supplementary materials document

## Collaboration

This manuscript is written in the `publication-manuscript` branch using a Git worktree.

### Switching to manuscript worktree:
```bash
cd ../ENBEL_pp_manuscript
```

### Committing changes:
```bash
git add manuscript/
git commit -m "docs: update manuscript introduction section"
git push origin publication-manuscript
```

### Merging back to main:
```bash
git checkout main
git merge publication-manuscript
```

## Tips

1. **Compile frequently** - Catch LaTeX errors early
2. **Use comments** - Mark sections needing work with `% TODO:`
3. **Version control** - Commit after completing each section
4. **Backup** - Keep local backups of the manuscript directory
5. **Collaborative editing** - Use `\textcolor{red}{new text}` for tracked changes
6. **Reference management** - Consider using Zotero/Mendeley to manage `references.bib`

## Troubleshooting

### Common LaTeX errors:

**"Undefined control sequence"**
- Check for typos in LaTeX commands
- Verify all required packages are loaded

**"Citation undefined"**
- Run `bibtex main` then compile twice more
- Check citation key matches entry in references.bib

**"Missing $ inserted"**
- Mathematical expressions must be in math mode: `$x = 5$` or `\[x = 5\]`

**"File not found"**
- Check figure paths are correct
- Ensure figures are in the `figures/` directory

### Getting help:

- [LaTeX Stack Exchange](https://tex.stackexchange.com/)
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)

## Contact

For questions about manuscript content, contact [lead author].

For technical LaTeX questions, see troubleshooting resources above.
