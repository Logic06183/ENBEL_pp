# Methodology Slides - Complete Set (8 Slides)

Comprehensive slide deck explaining the ENBEL climate-health analytical methodology.

## Quick Navigation

| Slide | Topic | Type | File Size |
|-------|-------|------|-----------|
| 01 | Imputation Flow | Technical | 12 KB |
| 02 | Spatial-Demographic Matching | Technical | 9.3 KB |
| 03 | Validation Results | Technical | 10 KB |
| 04 | DLNM Concept | Conceptual | 11 KB |
| 05 | ML vs Causal Inference | Conceptual | 9.3 KB |
| 06 | Hematocrit Paradox | Conceptual | 8.8 KB |
| 07 | Data Integration | Conceptual | 12 KB |
| 08 | Two-Stage Workflow | Conceptual | 10 KB |
| **Total** | **8 slides** | **3 Technical + 5 Conceptual** | **82 KB** |

---

## Part 1: Technical Methodology (Slides 1-3)

### Slide 01: Imputation Flow Diagram
**Purpose**: Complete overview of socioeconomic imputation pipeline

**Key Content**:
- 4-step process: Data Sources → KNN Matching → Ecological Backup → Validation
- Clinical cohort (n=11,398) + GCRO surveys (n=58,616)
- KNN parameters: k=10, 40% spatial, 60% demographic, 15km max distance
- Validation: 20% holdout, 5-fold CV

**Visual Design**: Numbered circles, gradient badges, color progression (Navy → Teal → Blue → Red)

---

### Slide 02: Spatial-Demographic KNN Matching
**Purpose**: Technical details of the matching algorithm

**Key Content**:
- **Left Panel**: Geographic visualization with distance circles
  - Clinical participant (center)
  - GCRO neighbors (colored by distance)
  - 10km and 15km radius circles
  - Connection lines to nearest neighbors

- **Right Panel**: Mathematical foundation
  - Combined distance metric: d(i,j) = α·d_spatial + (1-α)·d_demographic
  - α = 0.40 (spatial weight), 1-α = 0.60 (demographic weight)
  - Spatial component: Euclidean distance (UTM Zone 35S)
  - Demographic component: Gower distance (Sex, Race, Age)
  - Selection: k=10 nearest, d_max ≤ 15 km

---

### Slide 03: Validation Results
**Purpose**: Performance metrics demonstrating imputation quality

**Key Content**:
- **Performance Table**:
  | Variable | RMSE | MAE | Correlation |
  |----------|------|-----|-------------|
  | HEAT_VULNERABILITY_SCORE | 0.142 | 0.108 | **0.842** |
  | vuln_Housing | 0.185 | 0.136 | **0.756** |
  | vuln_employment | 0.203 | 0.158 | **0.698** |
  | economic_vulnerability | 0.227 | 0.172 | **0.631** |
  | Education | 0.298 | 0.215 | **0.542** |

- **Summary Statistics**:
  - Coverage: 98.7% successful matches
  - Mean correlation: 0.694 across 5 variables
  - Cross-validation: 5-fold CV, stable (SD < 0.05)

**Visual Design**: Professional table with gradient header, alternating rows, bold correlations

---

## Part 2: Conceptual Framework (Slides 4-8)

### Slide 04: DLNM Conceptual Framework
**Purpose**: Explain why distributed lag non-linear models are necessary

**Key Content**:
- **Why DLNM?**
  1. Non-Linear Effects: J-shaped, U-shaped, thresholds
  2. Lagged Effects: 0-21 day delays in biomarker response
  3. Complex Interactions: Temperature × Time interactions

- **DLNM Solution**:
  - Crossbasis function: Temperature × Lag interaction surface
  - 3D response surface visualization
  - Natural splines (4 df for temperature, 4 df for lag)

- **Study Parameters**: 0-21 days, 4 df each, conditional logistic, case-crossover design

**Visual Design**: Split panel (problems vs solutions), 3D grid representation

---

### Slide 05: ML vs Causal Inference Comparison
**Purpose**: Explain complementary nature of ML and DLNM approaches

**Key Content**:
- **Machine Learning (Left Panel)**:
  - Purpose: Association & Prediction
  - ✅ Strengths: High-dimensional, complex interactions, fast screening, explainable AI
  - ⚠️ Limitations: Association ≠ causation, confounding, between-person effects

- **Causal Inference - DLNM (Right Panel)**:
  - Purpose: Causation & Validation
  - ✅ Strengths: Controls confounding, within-person, lagged effects, gold standard
  - ⚠️ Limitations: Computationally intensive, requires repeated measures, limited features

- **Bottom Message**: "Complementary Approaches: Two Truths"
  - ML identifies associations (socioeconomic vulnerability)
  - DLNM validates acute effects (temperature)

**Visual Design**: Side-by-side panels with gradient headers, distinct colors

---

### Slide 06: The Hematocrit Paradox
**Purpose**: Explain why excellent ML performance doesn't always mean causal temperature effects

**Key Content**:
- **The Paradox**: ML R² = 0.937 (Excellent!) but DLNM OR = 220 (CI: 0.38-128,020) (Not Significant)

- **ML Findings**:
  - R² = 0.937 (n=2,120)
  - Top feature: HEAT_VULNERABILITY_SCORE = 18.4 (96% of importance)
  - Climate features: < 4% combined

- **DLNM Findings**:
  - OR = 220 (n=2,099)
  - 95% CI: 0.38 - 128,020 (includes 1.0)
  - Not significant (wide uncertainty)

- **Explanation: Two Different Truths**:
  - **ML (Between-Person)**: Captures stable socioeconomic vulnerability
    - Vulnerable populations → lower hematocrit (chronic stress, nutrition)
    - High R² from structural factors, NOT acute temperature

  - **DLNM (Within-Person)**: Tests acute day-to-day temperature
    - Controls ALL time-invariant factors (SES fixed)
    - No significant acute temperature effect

**Visual Design**: Paradox statement box (red), two panels (ML vs DLNM), detailed explanation

---

### Slide 07: Data Integration Framework
**Purpose**: Show how three data sources are harmonized

**Key Content**:
- **Three Data Sources**:
  1. **Clinical Trials** (🏥): n=11,398, 15 trials, 2002-2021, 19 biomarkers
  2. **GCRO Surveys** (🏘️): n=58,616, QoL surveys, 2011-2021, socioeconomic data
  3. **ERA5 Climate** (🌡️): 99.5% coverage, 0.25° resolution, 16 features

- **Integration**: Arrows converging to integrated dataset

- **Final Dataset**: Clinical + Socioeconomic + Climate
  - 19 biomarkers + heat vulnerability + daily temperature
  - Demographics + housing quality + lag windows
  - Study IDs + economic status + heat indices

**Visual Design**: Three boxes (top), arrows converging (middle), integrated panels (bottom)

---

### Slide 08: Two-Stage Analytical Workflow
**Purpose**: Show how ML and DLNM work together in practice

**Key Content**:
- **Stage 1: ML Screening**
  - Purpose: Rapid biomarker screening (hypothesis-generating)
  - Methods: Random Forest, XGBoost, LightGBM, 5-fold CV, R²/RMSE/MAE, SHAP
  - Output: 6 candidate biomarkers (R² > 0.13)
    - Hematocrit, Total Chol, LDL, HDL, LDL Chol, Creatinine

- **Stage 2: Causal Validation**
  - Purpose: Confirm causality (hypothesis-testing)
  - Methods: Case-crossover DLNM, time-stratified, 0-21 day lag, OR/95% CI
  - Output: 1 validated biomarker (p < 0.05)
    - ✅ FASTING HDL (OR = 69.48, CI: 1.05-4583)

- **Key Insight**: Only 1/6 validated
  - ML identifies associations (fast, broad)
  - DLNM confirms causation (slow, rigorous)
  - Both perspectives needed

**Visual Design**: Two large panels (stage 1 vs stage 2), arrow between, insight box at bottom

---

## Suggested Presentation Order

### For Technical Audience (Methodologists, Statisticians):
```
1. Slide 01 (Imputation Flow)
2. Slide 02 (Spatial Matching)
3. Slide 03 (Validation Results)
4. Slide 04 (DLNM Concept)
5. Slide 05 (ML vs Causal)
6. Slide 08 (Two-Stage Workflow)
7. Slide 06 (Hematocrit Paradox)
8. Slide 07 (Data Integration)
```

### For General Audience (Public Health, Policymakers):
```
1. Slide 07 (Data Integration) - Start with big picture
2. Slide 08 (Two-Stage Workflow) - Show overall approach
3. Slide 01 (Imputation Flow) - Explain data linkage
4. Slide 05 (ML vs Causal) - Two complementary methods
5. Slide 06 (Hematocrit Paradox) - Key insight
6. Slide 03 (Validation Results) - Quality assurance
```

### For Conference Presentation (15 min):
```
1. Slide 07 (Data Integration)
2. Slide 01 (Imputation Flow)
3. Slide 05 (ML vs Causal)
4. Slide 08 (Two-Stage Workflow)
5. Slide 06 (Hematocrit Paradox)
```

### For Seminar (45 min):
Use all 8 slides in numerical order

---

## Scientific Accuracy

All slides based on actual ENBEL methodology from:

**Source Code**:
- `src/enbel_pp/imputation.py`
- `scripts/imputation/practical_enbel_imputation.py`
- `R/case_crossover_dlnm_validation.R`

**Documentation**:
- `CASE_CROSSOVER_DLNM_VALIDATION_RESULTS.md`
- `COMPREHENSIVE_FEATURE_SPACE_ANALYSIS.md`
- Manuscript Methods section

**Data**:
- Actual validation metrics from imputation runs
- Real DLNM results (1/6 validated)
- Authentic SHAP importance values

---

## Technical Specifications

### Branding
- **Colors**: Wits PHR palette
  - Navy: `#2c5cda`
  - Teal: `#00bec5`
  - Blue: `#20a3fc`
  - Red: `#cc1a1b`
- **Typography**: Inter, Arial, sans-serif
- **Layout**: 1920×1080px (16:9)

### Figma Compatibility
- ✅ All text as `<text>` elements (editable)
- ✅ Inline style attributes (no CSS classes)
- ✅ Font sizes without 'px' units
- ✅ Letter spacing in pixels
- ✅ Named layer groups

### File Information
- **Total slides**: 8
- **Total size**: 82 KB (highly shareable)
- **Average size**: 10.25 KB per slide
- **Format**: SVG 1.1
- **Compatibility**: All modern browsers, Figma, Illustrator, Inkscape

---

## Usage Instructions

### Import to Figma
```bash
1. Open Figma (https://figma.com)
2. File → Import
3. Select slide(s)
4. Edit as needed
```

### View in Browser
```bash
cd presentation_slides_final/methodology
open slide_01_imputation_flow.svg
```

### Batch Processing
```bash
# Convert all to PNG (requires Inkscape)
for file in slide_*.svg; do
    inkscape "$file" --export-type=png --export-dpi=150
done

# Create PDF deck
svgslides slide_*.svg --output methodology.pdf
```

---

## Customization

### Change Colors
Find and replace hex codes:
```bash
# Example: Change navy to another blue
sed -i 's/#2c5cda/#1976D2/g' slide_*.svg
```

### Update Data
Edit slide 03 validation table:
1. Open in text editor
2. Find `<g id="metrics-table">`
3. Update values in `<text>` elements
4. Save

### Add Your Logo
In Figma:
1. Import slide
2. File → Place Image
3. Position in bottom right
4. Export

---

## What Makes These Slides Special

1. **Based on Real Research**: Not generic templates, actual ENBEL methodology
2. **Conceptually Rich**: Explains *why*, not just *how*
3. **Publication-Ready**: Suitable for high-impact journals
4. **Pedagogically Sound**: Clear explanations for teaching
5. **Figma-Editable**: Easy to customize for your needs
6. **Scientifically Accurate**: Reviewed against source code and data
7. **Professionally Designed**: Wits PHR branding throughout
8. **Comprehensive**: Technical + conceptual coverage

---

## Common Questions

**Q: Can I use these for my own research?**
A: Yes, with attribution. Modify as needed for your context.

**Q: Do I need all 8 slides?**
A: No. Choose based on your audience and time constraints (see suggested orders above).

**Q: How do I explain the hematocrit paradox to non-statisticians?**
A: Slide 06 is designed for general audiences. Key message: "ML found socioeconomic factors matter most, not daily temperature changes."

**Q: Can I create additional slides in this style?**
A: Yes! Use existing slides as templates. Follow Wits PHR branding guidelines.

**Q: Where can I learn more about DLNM?**
A: See Gasparrini et al. (2010) "Distributed lag non-linear models" in Statistics in Medicine.

---

## Version History

**v2.0** (2025-10-30)
- Added 5 conceptual slides (04-08)
- Complete 8-slide methodology deck
- Comprehensive documentation

**v1.0** (2025-10-30)
- Initial release (slides 01-03)
- Technical imputation methodology
- Figma-ready SVG format

---

## Citation

If using these slides in publications or presentations:

```
Saunders, C. et al. (2025). ENBEL Climate-Health Research: Methodology Slide Deck.
Wits Planetary Health Research. https://github.com/...
```

---

## Contact

- **Technical Questions**: Refer to source code and documentation
- **Scientific Questions**: See manuscript Methods section
- **Design Questions**: Wits PHR branding guidelines

---

**Built with**: SVG Slide Master skills + ENBEL methodology expertise
**Optimized for**: Figma editing, professional presentations, publication figures
**Ready for**: Conferences, seminars, thesis defenses, grant proposals

🎯 **Complete methodology story told in 8 professional slides**
