# ENBEL Presentation Slides Perfection Roadmap

**Branch**: `feat/perfect-study-distribution-map`
**Focus**: Create publication-ready SVG visualizations using authentic underlying data
**Worktree**: `ENBEL_pp_map_perfection/`

## Current Status

- **Total SVG slides**: 32
- **Generation scripts**: 16 (R and Python)
- **Primary data sources**:
  - Clinical: `CLINICAL_DATASET_COMPLETE_CLIMATE.csv` (11,398 records)
  - GCRO: `GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv` (58,616 records)
  - Shapefile: `map_vector_folder_JHB/JHB/metropolitan municipality jhb.shp`

## Completed Improvements

### 1. Study Distribution Map ✓
- **File**: `johannesburg_study_distribution_16x9_shapefile.svg`
- **Script**: `create_johannesburg_map_with_ward_distribution.R`
- **Improvements**:
  - Authentic JHB shapefile boundary
  - Ward-level GCRO distribution (1,032 points across 258 wards)
  - Highly distinctive survey wave colors (Blue, Teal, Purple, Orange)
  - Clinical sites sized by patient enrollment
  - 16:9 aspect ratio for presentations

## Slide Categories to Perfect

### A. Maps & Geographic Visualizations
- [x] `johannesburg_study_distribution_16x9_shapefile.svg` - Ward distribution
- [ ] `enbel_clean_data_focused_map.svg` - Needs data-driven approach
- [ ] `enbel_corrected_locations_map.svg` - Verify clinical site locations
- [ ] `enbel_distinct_data_types_map.svg` - Enhance data type clarity
- [ ] `enbel_final_scientific_map.svg` - Apply scientific standards
- [ ] `enbel_jhb_shapefile_map.svg` - Integrate with shapefile
- [ ] `enbel_johannesburg_detailed_map.svg` - Add detail layers

### B. DLNM (Distributed Lag Non-Linear Model) Plots
- [ ] `enbel_cd4_dlnm_real_success.svg` - CD4-specific analysis
- [ ] `enbel_dlnm_analysis_final.svg` - Main DLNM visualization
- [ ] `enbel_dlnm_authentic_lag_plots.svg` - Lag-specific plots
- [ ] `enbel_dlnm_classic_final.svg` - Classic representation
- [ ] `enbel_dlnm_high_performance_simulation.svg` - Simulation results
- [ ] `enbel_dlnm_native_final.svg` - Native R output
- [ ] `enbel_dlnm_native_R_final.svg` - R-based visualization
- [ ] `enbel_dlnm_real_final.svg` - Real data DLNM
- [ ] `enbel_dlnm_working_final.svg` - Working implementation

### C. SHAP (Explainable AI) Plots
- [ ] `enbel_cd4_heat_analysis_final.svg` - CD4-heat SHAP analysis
- [ ] `enbel_shap_comprehensive_final.svg` - Comprehensive SHAP
- [ ] `enbel_shap_real_waterfall.svg` - Waterfall plots
- [ ] `enbel_shap_three_pathways.svg` - Three-pathway analysis

### D. Methodology & Data Quality
- [ ] `enbel_imputation_methodology_final.svg` - Imputation methods
- [ ] `enbel_imputation_real_final.svg` - Real imputation results
- [ ] `enbel_temporal_coverage_16x9.svg` - Temporal coverage visualization
- [ ] `enbel_temporal_patterns_comprehensive.svg` - Pattern analysis

### E. Overview & Summary Slides
- [ ] `enbel_data_overview_comprehensive.svg` - Dataset overview
- [ ] `enbel_editable_text_overview.svg` - Text-based overview
- [ ] `enbel_figma_compatible.svg` - Figma integration test

## Systematic Improvement Strategy

### Phase 1: Data Verification (Week 1)
1. Audit all visualization scripts against raw data
2. Identify synthetic/simulated data vs. real data
3. Document data sources for each visualization
4. Create data validation pipeline

### Phase 2: Core Visualizations (Week 2-3)
1. **Maps**: Use authentic shapefiles + real coordinates
2. **DLNM**: Run actual DLNM models on clinical data
3. **SHAP**: Generate real SHAP values from trained models
4. **Temporal**: Extract real temporal patterns from data

### Phase 3: Design Consistency (Week 4)
1. Standardize color schemes across all slides
2. Apply consistent typography (fonts, sizes)
3. Ensure 16:9 aspect ratio for all slides
4. Add unified header/footer templates
5. Optimize for Figma import

### Phase 4: Quality Assurance (Week 5)
1. Peer review all visualizations
2. Verify scientific accuracy
3. Test Figma compatibility
4. Generate PDF versions
5. Create presentation assembly guide

## Key Principles

### 1. Data Authenticity
- **Always use real data** from the actual datasets
- **No synthetic data** unless explicitly labeled
- **Document data sources** in each script
- **Validate coordinates** against known locations

### 2. Scientific Rigor
- **Proper statistics**: Real p-values, confidence intervals
- **Reproducible**: Set seeds, version control
- **Transparent**: Show sample sizes, methods
- **Peer-reviewable**: Publication-quality standards

### 3. Design Excellence
- **Minimalist aesthetic**: Remove chart junk
- **Color accessibility**: Colorblind-friendly palettes
- **Typography**: Clear, readable fonts
- **Layout**: Logical visual hierarchy
- **Format**: SVG for scalability, 16:9 for presentations

### 4. Technical Standards
- **Resolution**: 150 DPI minimum
- **Vector graphics**: SVG preferred
- **File size**: Optimize without quality loss
- **Compatibility**: Test in Figma, PowerPoint, LaTeX beamer
- **Version control**: Git commit each improvement

## Available Scripts

### R Scripts (8)
- `create_johannesburg_map_with_ward_distribution.R` ✓
- `create_johannesburg_map_r.R`
- DLNM analysis scripts (various)
- Temporal analysis scripts

### Python Scripts (8)
- `create_distinct_data_types_map.py`
- SHAP visualization scripts
- Imputation methodology scripts
- Data overview scripts

## Data Access Paths

```bash
# Clinical data
CLINICAL_DATA="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"

# GCRO socioeconomic data
GCRO_DATA="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"

# Johannesburg shapefile
JHB_SHAPEFILE="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/map_vector_folder_JHB/JHB/metropolitan municipality jhb.shp"

# Output directory
OUTPUT_DIR="/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/"
```

## Workflow

### For Each Slide Improvement:

1. **Audit**: Review current SVG and generation script
2. **Verify**: Check data sources (real vs. synthetic)
3. **Regenerate**: Run script with real data
4. **Review**: Visual QA and scientific accuracy check
5. **Commit**: Git commit with descriptive message
6. **Document**: Update this roadmap with status

### Git Workflow:

```bash
# Work in the dedicated worktree
cd "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection"

# After each improvement
git add presentation_slides_final/
git commit -m "feat: improve [slide name] with real data"

# Push regularly
git push origin feat/perfect-study-distribution-map
```

## Success Metrics

- [ ] All visualizations use authentic underlying data
- [ ] Zero synthetic/placeholder data in final slides
- [ ] Consistent design language across all slides
- [ ] 16:9 aspect ratio for all presentation slides
- [ ] Figma-compatible SVG format
- [ ] Publication-ready quality (peer-reviewable)
- [ ] Complete documentation of data sources
- [ ] Reproducible generation pipeline

## Next Steps

1. Review remaining 31 SVG slides
2. Prioritize by presentation order
3. Start with high-impact visualizations (maps, key DLNM plots)
4. Work systematically through categories A→E
5. Test Figma import after each batch

---

**Updated**: 2025-10-14
**Branch Status**: Active development
**Completion Target**: ~5 weeks for comprehensive refinement
