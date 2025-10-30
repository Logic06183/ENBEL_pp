# ENBEL Presentation Slide Improvement Plan

**Source PDF**: `/Users/craig/Downloads/CP_ENBEL_141025.pdf`
**Branch**: `feat/perfect-presentation-slides`
**Goal**: Create publication-ready SVG versions with authentic underlying data

---

## Slide-by-Slide Analysis & Improvement Plan

### ✅ Slide 1: Title Slide
**Status**: Good as-is
**Improvements**: None needed - clean design with proper branding

---

### 🔧 Slide 2: Why Heat Matters in Johannesburg
**Current**: Three satellite maps (Population, Surface Temp, Vegetation)
**Issues**:
- Maps appear to be placeholder/simulated data
- Need real Landsat 8/9 and Sentinel-2 data
- Color schemes could be more accessible

**Improvement Plan**:
1. **Data Sources**:
   - Population: WorldPop actual data for Johannesburg
   - Surface Temperature: Real Landsat 8/9 thermal data
   - Vegetation: Sentinel-2 NDVI change 2015-2023
2. **Visual Enhancements**:
   - Use authentic satellite imagery
   - Improve color accessibility (colorblind-friendly)
   - Add scale bars and north arrows
   - Ensure consistent projection across all three maps
3. **Script**: Create R/Python script using Google Earth Engine or raster processing

**Priority**: HIGH - This is a key visual showing the problem context

---

### 🔧 Slide 3: Research Dataset Overview
**Current**: Three boxes (Clinical Studies, Climate Data, Socioeconomic)
**Issues**:
- Good design but could use better visual hierarchy
- Icons could be more professional
- Color scheme could match overall presentation better

**Improvement Plan**:
1. Enhance visual consistency with slide theme
2. Use professional iconography
3. Add subtle data visualization elements (mini charts)
4. Ensure perfect 16:9 aspect ratio
5. Make text more scannable

**Priority**: MEDIUM - Design polish

---

### 🔧 Slide 4: Temporal Coverage Overview
**Current**: Timeline showing 17 studies + 4 GCRO waves
**Issues**:
- Good concept but execution could be cleaner
- Some overlaps are hard to read
- Could show patient counts more clearly

**Improvement Plan**:
1. **Use Real Data**: Extract exact dates from `CLINICAL_DATASET_COMPLETE_CLIMATE.csv`
2. **Visual Improvements**:
   - Cleaner timeline visualization
   - Better spacing for overlapping studies
   - Show patient counts on hover/tooltip simulation
   - Add vertical markers for GCRO survey years
3. **Script**: Python matplotlib or R ggplot2 with actual study dates

**Priority**: HIGH - Critical for showing data comprehensiveness

**Script to Create**:
```python
# create_temporal_coverage_final.py
# Use real study dates from clinical dataset
# Generate clean SVG timeline with proper scaling
```

---

### 🔧 Slide 5: Johannesburg Study Distribution
**Current**: Map with clinical sites and GCRO points
**Issues**:
- GCRO points look too uniform (all same blue)
- Need better distinction between survey waves
- Clinical site labels could be clearer

**Improvement Plan**:
1. **Use Existing R Script**: `create_johannesburg_map_with_ward_distribution.R`
2. **Enhancements Needed**:
   - Make GCRO survey wave colors MORE distinctive
   - Increase point opacity slightly for better visibility
   - Ensure clinical site labels don't overlap
   - Add subtle background shading for geographic context
3. **Already Have**:
   - Authentic JHB shapefile ✓
   - Ward-level GCRO distribution ✓
   - Real clinical site coordinates ✓

**Priority**: HIGH - Already started, just needs final polish

**Action**: Run and refine existing R script

---

### 🔧 Slide 6: Three-Stage Validation Framework
**Current**: Clean three-box diagram
**Issues**:
- Very clean, mostly design polish needed
- Could add visual flow indicators
- Icons could be more distinctive

**Improvement Plan**:
1. Enhance visual flow with connecting arrows
2. Add subtle background gradients
3. Use more professional icons
4. Ensure perfect alignment and spacing
5. Make it more visually engaging while maintaining clarity

**Priority**: LOW - Design polish only

---

### 🔧 Slide 7: Explainable AI Feature Discovery
**Current**: SHAP explanation with key discoveries
**Issues**:
- Layout is good but could be more visually striking
- Discovery boxes could have subtle visual enhancements
- Could add mini visualizations

**Improvement Plan**:
1. Add subtle data visualization elements to each discovery box
2. Enhance color coding for different physiological systems
3. Make numerical findings more prominent
4. Add visual interest without cluttering

**Priority**: MEDIUM - Content is strong, needs visual polish

---

### 🔧 Slide 8: SHAP Analysis CD4 Model
**Current**: Three SHAP plots (Beeswarm, Feature Importance, Dependence)
**Issues**:
- **CRITICAL**: Need to verify these are REAL SHAP values, not simulated
- Plots look good but need to be regenerated from actual model
- Could improve plot styling for consistency

**Improvement Plan**:
1. **Regenerate from Real Data**:
   - Train actual Random Forest model on CD4 data
   - Generate real SHAP values using TreeExplainer
   - Use actual clinical dataset: `CLINICAL_DATASET_COMPLETE_CLIMATE.csv`
2. **Visual Enhancements**:
   - Consistent color scheme across all three plots
   - Better axis labels and legends
   - Improve plot arrangement for balance
   - Ensure high-resolution output

**Priority**: CRITICAL - Must use real SHAP values

**Script to Create**:
```python
# generate_real_cd4_shap_analysis.py
# Steps:
# 1. Load clinical data
# 2. Train Random Forest on CD4 ~ climate features
# 3. Generate SHAP values
# 4. Create three publication-quality plots
# 5. Export as SVG
```

---

### ✅ Slide 9: From Discovery to Impact
**Status**: Good as-is
**Improvements**: Minor visual polish only

---

### ✅ Slide 10: International Collaboration
**Status**: Good layout
**Improvements**: Ensure all logos are high-resolution

---

### ✅ Slide 11: References
**Status**: Acceptable
**Improvements**: Format consistency check

---

### 🔧 Slide 12: Imputation Methodology (Appendix)
**Current**: Comprehensive 6-panel methodology diagram
**Issues**:
- **CRITICAL**: Need real validation metrics, not simulated
- Plots need to be generated from actual imputation results
- Geographic distribution needs real coordinates

**Improvement Plan**:
1. **Run Real Imputation**:
   - Use actual clinical + GCRO data
   - Apply KNN + Ecological stratification
   - Generate real validation metrics
2. **Plots to Generate**:
   - B: Real KNN distance weighting curve
   - C: Actual missing data patterns from GCRO surveys
   - D: Real cross-validation performance (R² = 0.74 mentioned)
   - E: Actual geographic distribution of matched records
   - F: Real temporal stability across months
3. **Visual Consistency**: Match color scheme to main presentation

**Priority**: CRITICAL - This is a key methodology validation slide

**Script to Create**:
```python
# create_real_imputation_methodology_slide.py
# Steps:
# 1. Run actual imputation pipeline
# 2. Generate all 6 validation plots
# 3. Combine into single 16:9 SVG
# 4. Use real metrics throughout
```

---

## Implementation Strategy

### Phase 1: Critical Data-Driven Slides (Week 1)
**These MUST use real data:**

1. **Slide 8: SHAP Analysis** 🔴 CRITICAL
   - Generate real SHAP values from trained model
   - Script: `generate_real_cd4_shap_analysis.py`

2. **Slide 12: Imputation Methodology** 🔴 CRITICAL
   - Run real imputation validation
   - Script: `create_real_imputation_methodology_slide.py`

3. **Slide 4: Temporal Coverage** 🟡 HIGH
   - Use exact study dates from dataset
   - Script: `create_temporal_coverage_final.py`

### Phase 2: Geographic & Visual Enhancements (Week 2)

4. **Slide 2: Satellite Maps** 🟡 HIGH
   - Real satellite data processing
   - Script: `create_satellite_climate_health_patterns.py` or `.R`

5. **Slide 5: Study Distribution Map** 🟡 HIGH
   - Refine existing R script
   - Action: Polish `create_johannesburg_map_with_ward_distribution.R`

### Phase 3: Design Polish (Week 3)

6. **Slide 3: Dataset Overview** 🟢 MEDIUM
   - Visual hierarchy improvements

7. **Slide 7: Feature Discovery** 🟢 MEDIUM
   - Visual enhancements

8. **Slide 6: Validation Framework** 🔵 LOW
   - Design polish only

### Phase 4: Quality Assurance (Week 4)

9. Test all slides in Figma
10. Verify 16:9 aspect ratio consistency
11. Check color accessibility
12. Generate PDF versions
13. Peer review for scientific accuracy

---

## Data Sources & Paths

### Clinical Data
```bash
CLINICAL_DATA="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
```
- 10,202 patients (PDF says 11,398 - verify!)
- 30+ biomarkers including CD4
- 2002-2021 date range

### GCRO Data
```bash
GCRO_DATA="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"
```
- 58,616 households
- 4 survey waves: 2011, 2014, 2018, 2021
- Ward-level geographic data

### Geographic Data
```bash
JHB_SHAPEFILE="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/map_vector_folder_JHB/JHB/metropolitan municipality jhb.shp"
```
- Authentic Johannesburg metropolitan boundary
- Use for all maps

### Output Directory
```bash
OUTPUT_DIR="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/"
```

---

## Technical Standards

### File Format
- **Primary**: SVG (scalable vector graphics)
- **Aspect Ratio**: 16:9 (1920×1080 or 1600×900)
- **DPI**: 150 minimum for raster elements
- **Color Space**: sRGB
- **Fonts**: Embed or convert to paths

### Naming Convention
```
enbel_[slide_name]_final_v[version].svg
```

Examples:
- `enbel_cd4_shap_analysis_final_v2.svg`
- `enbel_study_distribution_map_final_v3.svg`
- `enbel_imputation_methodology_final_v1.svg`

### Color Scheme (from PDF)
- **Primary Blue**: #2E7AB5 (used in headers)
- **Clinical Studies**: Blue tones
- **Climate Data**: Green tones
- **Socioeconomic**: Orange tones
- **Ensure colorblind accessibility**: Use ColorBrewer palettes

### Typography
- **Headers**: Sans-serif, bold
- **Body**: Sans-serif, regular
- **Data Labels**: Clear, readable at small sizes
- **Avoid**: Overly decorative fonts

---

## Quality Checklist

Before marking any slide as "complete":

- [ ] Uses authentic underlying data (not simulated/placeholder)
- [ ] 16:9 aspect ratio verified
- [ ] Text is legible at presentation size
- [ ] Colors are accessible (colorblind-friendly)
- [ ] SVG exports cleanly to Figma
- [ ] All data sources are documented
- [ ] Scientific accuracy verified
- [ ] Consistent with overall presentation style
- [ ] High-resolution graphics (150+ DPI for raster)
- [ ] Proper attribution/citations included

---

## Git Workflow

```bash
# Work in presentation slides branch
cd "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection"

# After each slide improvement
git add presentation_slides_final/enbel_[slide_name]_final.svg
git add presentation_slides_final/create_[slide_name].py  # or .R
git commit -m "feat: regenerate [slide name] with real data

- Use authentic [data source]
- Improve [specific enhancement]
- Generate publication-quality SVG
- Verify scientific accuracy

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push regularly
git push origin feat/perfect-presentation-slides
```

---

## Success Metrics

- [ ] All data-driven slides use 100% authentic data (no simulation)
- [ ] All slides export perfectly to Figma
- [ ] Presentation flows cohesively with consistent visual language
- [ ] Scientific reviewers approve accuracy
- [ ] Slides are publication-ready for peer review
- [ ] Complete documentation of data sources
- [ ] Reproducible generation pipeline

---

**Next Immediate Actions**:

1. Start with **Slide 8 (SHAP Analysis)** - Most critical
2. Verify clinical data patient count discrepancy (10,202 vs 11,398)
3. Generate real SHAP values from actual CD4 model
4. Create `generate_real_cd4_shap_analysis.py` script

---

**Updated**: 2025-10-14
**Estimated Completion**: 3-4 weeks for comprehensive refinement
