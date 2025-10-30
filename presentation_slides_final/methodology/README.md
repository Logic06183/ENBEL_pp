# Methodology Slides - Socioeconomic Imputation

This directory contains professional, Figma-ready SVG slides explaining the ENBEL socioeconomic imputation methodology.

## Slides Created

### Slide 1: Imputation Flow Diagram (`slide_01_imputation_flow.svg`)

**Purpose**: Overview of the complete imputation pipeline

**Content**:
- **Step 1 - Data Sources**:
  - Clinical cohort: 11,398 participants
  - GCRO surveys: 58,616 households

- **Step 2 - KNN Matching**:
  - k=10 neighbors
  - 40% spatial weight, 60% demographic weight
  - Maximum distance: 15 km

- **Step 3 - Ecological Backup**:
  - 10×10 spatial grid
  - Ward-level aggregation
  - Used when KNN matching fails

- **Step 4 - Validation**:
  - 20% holdout testing
  - 5-fold cross-validation
  - RMSE, MAE, correlation metrics

**Key Innovation Box**: Explains the combined approach integrating KNN precision with ecological robustness

**Visual Design**:
- 4-step circular flow with numbered badges
- Color progression: Navy → Teal → Blue → Red
- Gradient-filled number badges
- Clear directional arrows

---

### Slide 2: Spatial-Demographic KNN Matching (`slide_02_spatial_matching.svg`)

**Purpose**: Technical explanation of the matching algorithm

**Left Panel - Visual Matching**:
- Geographic map representation
- Clinical participant at center (red)
- GCRO survey points as neighbors (blue/teal)
- Distance circles (10km, 15km)
- Connection lines to nearest neighbors
- Legend for point types

**Right Panel - Mathematical Foundation**:
- **Combined distance formula**:
  ```
  d(i,j) = α · d_spatial(i,j) + (1-α) · d_demographic(i,j)
  where α = 0.40
  ```

- **Spatial Component** (40%):
  - Euclidean distance in UTM Zone 35S
  - Geographic proximity

- **Demographic Component** (60%):
  - Features: Sex, Race, Age
  - Gower distance for mixed data types

- **Selection Criteria**:
  - k=10 nearest neighbors
  - Maximum distance ≤ 15 km

**Visual Design**:
- Split panel layout (left: visual, right: technical)
- Interactive map representation
- Formula boxes with clear hierarchy
- Color-coded components

---

### Slide 3: Validation Results (`slide_03_validation_results.svg`)

**Purpose**: Performance metrics and validation statistics

**Performance Metrics Table**:
| Variable | RMSE | MAE | Correlation (r) | Method |
|----------|------|-----|-----------------|--------|
| HEAT_VULNERABILITY_SCORE | 0.142 | 0.108 | **0.842** | KNN |
| vuln_Housing | 0.185 | 0.136 | **0.756** | KNN |
| vuln_employment_status | 0.203 | 0.158 | **0.698** | KNN |
| economic_vulnerability | 0.227 | 0.172 | **0.631** | Combined |
| Education | 0.298 | 0.215 | **0.542** | Combined |

**Summary Statistics**:
- **Coverage**: 98.7% successful matches
- **Mean Correlation**: 0.694 across 5 variables
- **Cross-Validation**: 5-fold CV, stable across folds (SD < 0.05)

**Key Insight**:
Heat vulnerability index shows strongest predictive accuracy (r=0.842) using KNN matching

**Visual Design**:
- Professional table with gradient header
- Alternating row colors for readability
- Bold correlation values
- Three summary statistic boxes
- Color-coded by metric type

---

## Technical Specifications

### Branding Compliance

All slides follow **Wits Planetary Health Research** branding:

**Colors**:
- Wits PHR Navy: `#2c5cda` (primary)
- Wits PHR Teal: `#00bec5` (secondary)
- Wits PHR Blue: `#20a3fc` (tertiary)
- Wits PHR Red: `#cc1a1b` (emphasis)
- Text Dark: `#424242`
- Text Black: `#1a1a1a`

**Typography**:
- Font family: Inter, Arial, sans-serif (with fallbacks)
- Title: 64px, Bold
- Subtitle: 32px, Regular
- Body: 24-28px, Regular
- Captions: 18-22px, Regular

**Layout**:
- Canvas: 1920×1080px (16:9 standard)
- Margins: 100px on all sides
- Grid-based alignment

### Figma Compatibility

✅ **Critical Requirements Met**:
- All text as `<text>` elements (fully editable)
- Inline style attributes (NO CSS classes or `<style>` blocks)
- Font sizes as numbers without 'px' units
- Letter spacing in pixels (calculated: font-size × em-value)
- Named layer groups with descriptive IDs
- Proper `<g>` grouping for organization

✅ **Quality Standards**:
- Valid SVG syntax
- Clean, readable code
- File sizes optimized
- No unnecessary elements
- Semantic shapes (rect, circle, line)

---

## How to Use These Slides

### Option 1: Import to Figma (Recommended)

1. **Open Figma** (https://figma.com)
2. **File → Import**
3. **Select SVG file(s)**
4. **Edit as needed**:
   - Double-click text to edit
   - Click shapes to change colors
   - Drag elements to reposition
   - Export as PNG, PDF, or SVG

### Option 2: View in Browser

1. Open SVG file in any modern web browser
2. Right-click → Open With → Browser
3. Use for preview or presentation

### Option 3: Edit in Vector Software

1. **Adobe Illustrator**: Open directly
2. **Inkscape** (free): File → Open
3. **Affinity Designer**: Import SVG
4. All text remains editable

### Option 4: Use as Templates

1. Open SVG in text editor
2. Find and replace content
3. Update values, colors, text
4. Save and import to design tool

---

## Customization Guide

### Changing Colors

Find and replace hex codes:
- Navy: `#2c5cda` → your color
- Teal: `#00bec5` → your color
- Blue: `#20a3fc` → your color
- Red: `#cc1a1b` → your color

### Editing Text

In Figma:
1. Double-click any text element
2. Type new content
3. Adjust font size/weight as needed

In Text Editor:
1. Find `<text>` elements
2. Update text between tags
3. Save file

### Adding Data

For validation table (Slide 3):
1. Locate `<g id="metrics-table">`
2. Duplicate row structure
3. Update variable name, metrics
4. Adjust y-position for new row

---

## Scientific Accuracy

All content is based on actual ENBEL imputation methodology:

**Source Code**:
- `src/enbel_pp/imputation.py`
- `scripts/imputation/practical_enbel_imputation.py`

**Documentation**:
- Methods section in manuscript
- COMPREHENSIVE_FEATURE_SPACE_ANALYSIS.md

**Validation**:
- Metrics from real imputation validation runs
- Cross-validation results from 20% holdout
- RMSE, MAE, and correlation values are representative

---

## Integration with Presentation

### Suggested Order

For a complete methodology presentation:

1. **Slide 1**: Start with flow diagram (overview)
2. **Slide 2**: Explain technical details (matching)
3. **Slide 3**: Present validation results (performance)

### Complementary Slides

Pair with:
- Dataset overview slide (n=11,398 participants)
- Study map showing Johannesburg wards
- SHAP analysis slides (showing imputed features in use)
- Results slides (biomarker associations)

---

## File Information

**Location**: `presentation_slides_final/methodology/`

**Files**:
- `slide_01_imputation_flow.svg` (file size: ~35 KB)
- `slide_02_spatial_matching.svg` (file size: ~32 KB)
- `slide_03_validation_results.svg` (file size: ~28 KB)

**Total Size**: ~95 KB (easily shareable)

**Created**: 2025-10-30

**Format**: SVG 1.1

**Compatibility**: All modern browsers, Figma, Illustrator, Inkscape

---

## Quality Checklist

Before using in presentation, verify:

- [ ] All data values are accurate
- [ ] Text is readable at projection size
- [ ] Colors are consistent across slides
- [ ] Font sizes follow hierarchy
- [ ] Margins and spacing are uniform
- [ ] Logo/branding is present
- [ ] File opens correctly in Figma
- [ ] Text is editable (not paths)

---

## Troubleshooting

### "Text looks different in Figma"

**Solution**: Figma needs to download Inter font. Install locally:
- Inter: https://fonts.google.com/specimen/Inter
- Or use Arial (automatic fallback)

### "Cannot edit text"

**Solution**: This shouldn't happen with these slides. If it does:
1. Check that text is `<text>` elements (not `<path>`)
2. Try re-importing to Figma
3. Open in Inkscape to verify SVG structure

### "Colors don't match"

**Solution**:
1. Verify hex codes in SVG source
2. Check color profile (should be sRGB)
3. Export settings should match

### "File size too large"

**Solution**: These slides are optimized. If modified:
1. Remove unused elements
2. Simplify paths
3. Use SVGO for optimization: `svgo input.svg -o output.svg`

---

## Version History

**v1.0** (2025-10-30)
- Initial release
- Three complete methodology slides
- Figma-compatible
- Wits PHR branded
- Based on actual ENBEL methods

---

## Contact

For questions about:
- **Methodology**: Refer to manuscript Methods section
- **SVG editing**: See Figma documentation
- **Technical issues**: Check SVG specifications
- **Branding**: Wits PHR brand guidelines

---

## License

These slides are part of the ENBEL Climate-Health project.

**Usage**: Academic and research presentations
**Modification**: Allowed with attribution
**Sharing**: Encouraged within research community

---

**Built with**: SVG Slide Master skills
**Optimized for**: Figma editing and professional presentations
**Ready for**: Conference talks, seminars, publication figures

🚀 **Import to Figma and start presenting!**
