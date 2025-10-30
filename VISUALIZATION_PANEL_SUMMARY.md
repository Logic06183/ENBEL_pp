# Visualization Panel Summary

**Date**: 2025-10-30
**Purpose**: Streamlined SHAP attribution visualizations for Phase 2 feature testing
**Location**: `results/visualization_panel/`

---

## Files Generated (14 total)

### 1. Cross-Model Summaries (3 files)

**01_feature_importance_heatmap.png**
- **Type**: Heatmap
- **Shows**: Feature importance across all 10 biomarkers
- **Key Finding**: HEAT_VULNERABILITY_SCORE dominates for most biomarkers
- **Use**: Identify which features matter across biomarkers

**02_model_comparison.png**
- **Type**: Horizontal bar chart
- **Shows**: R² for each biomarker (climate-only models)
- **Key Finding**: Hematocrit (0.935), CD4 (0.710), FASTING LDL (0.374) are top performers
- **Use**: Rank biomarkers by predictability

**03_demographics_impact.png**
- **Type**: Horizontal bar chart (ΔR²)
- **Shows**: Impact of adding demographics + study features
- **Key Finding**: Most biomarkers WORSE with demographics (overfitting/sample loss)
- **Use**: Justify climate-only models

### 2. Detailed SHAP for Top 3 Biomarkers (9 files)

For each biomarker (CD4, Hematocrit, FASTING LDL):

**04_[biomarker]_beeswarm.png**
- **Type**: SHAP beeswarm plot
- **Shows**: Distribution of SHAP values for each feature
- **Interpretation**:
  - Red = High feature value
  - Blue = Low feature value
  - X-axis = SHAP value (impact on prediction)
- **Use**: See how feature values affect predictions

**04_[biomarker]_importance.png**
- **Type**: SHAP bar plot
- **Shows**: Mean absolute SHAP value (feature importance ranking)
- **Use**: Quick feature importance ranking

**04_[biomarker]_dependence.png**
- **Type**: 3-panel dependence plots
- **Shows**: Relationship between feature value and SHAP value for top 3 features
- **Interpretation**: Non-linear relationships, interactions
- **Use**: Understand feature effects in detail

### 3. Summary Table (2 files)

**05_summary_table.csv**
- **Type**: CSV data table
- **Shows**: Complete results for all biomarkers
- **Use**: Import into papers, reports

**05_summary_table.png**
- **Type**: Styled table image
- **Shows**: Same as CSV, formatted for presentation
- **Use**: Slide presentations, quick reference

---

## Key Findings from Visualizations

### 🌟 CD4 is the Climate Biomarker

**From beeswarm plot:**
- HEAT_VULNERABILITY_SCORE dominates (as expected)
- climate_7d_mean_temp shows clear positive effect
- climate_daily_max_temp shows secondary effect

**From dependence plots:**
- Non-linear relationships visible
- Temperature effects modulated by vulnerability
- Climate features interact

**Biological plausibility:**
- Heat stress → immune suppression
- HIV population vulnerable to temperature extremes
- Chronic heat exposure + acute spikes both matter

### ⚠️ Hematocrit Caveat

**From feature importance heatmap:**
- HEAT_VULNERABILITY_SCORE: 19.03 (96% of importance)
- All climate features: <0.5 combined (4% of importance)

**Interpretation:**
- R² = 0.935 is **socioeconomic**, not climate-driven
- Reflects structural inequality (housing, income, chronic stress)
- NOT an acute temperature biomarker
- Report as "vulnerability biomarker" not "climate biomarker"

**Data quality issue:**
- Hemoglobin/hematocrit correlation = 0.008 (should be 0.85-0.95)
- Suggests unit conversion error or data entry issue
- Recommend investigation before publication

### 📊 Demographics Don't Help

**From demographics impact chart:**
- ✅ **Help** (ΔR² > 0.03): Weight (+0.048), CD4 (+0.037)
- ❌ **Hurt** (ΔR² < -0.03): FASTING LDL (-0.041), FASTING HDL (-0.033)
- ℹ️ **Neutral**: 6 biomarkers (minimal change)

**Why demographics hurt:**
1. Sample size loss: 8,577 → 5,788 (32% drop)
2. More features + less data = overfitting
3. Age/Sex may not matter for these biomarkers

**Recommendation:** Use climate-only models (better sample size, comparable performance)

---

## Features Identified for Phase 2 Testing

Based on SHAP attribution analysis, these features show consistent importance:

### Tier 1: Strong Signal (test first)
1. **HEAT_VULNERABILITY_SCORE** - Dominates across biomarkers
2. **climate_7d_mean_temp** - Short-term climate adaptation window
3. **climate_daily_max_temp** - Acute heat stress indicator

### Tier 2: Moderate Signal (test if Tier 1 works)
4. **climate_heat_stress_index** - Composite heat metric
5. **climate_daily_min_temp** - Nighttime temperature (recovery)
6. **month** - Seasonal patterns

### Tier 3: Weak Signal (exploratory)
7. **climate_season** (categorical) - Coarse temporal grouping
8. **season** (categorical) - Redundant with climate_season

### NOT recommended:
- **climate_daily_mean_temp** - Redundant with max/min temps
- **Demographic features** - Hurt more than help (sample loss)

---

## Recommendations for Phase 2

### 1. Focus on CD4 as Primary Outcome
- Strong climate sensitivity (R² = 0.710)
- Biologically plausible
- Sufficient sample size (2,333 observations)
- HIV population clinically relevant

### 2. Investigate Hematocrit Data Quality
- Before using in publications, resolve:
  - Hemoglobin/hematocrit correlation issue (0.008 vs expected 0.85-0.95)
  - Unit conversion verification
  - Data entry validation
- If valid, report as "socioeconomic vulnerability biomarker" not "climate biomarker"

### 3. Stick with Climate-Only Models
- Demographics add minimal value (ΔR² < 0.04 for most)
- Sample size loss hurts performance
- Simpler models easier to interpret and validate

### 4. Next Phase Testing Priorities

**DLNM Analysis (Distributed Lag Non-linear Models):**
- CD4 with 0-30 day temperature lags
- Capture delayed immune responses
- Test for non-linear dose-response curves
- Expected improvement: +0.05 to +0.15 R²

**Feature Engineering:**
- Temperature variability (daily range, weekly SD)
- Heat wave indicators (consecutive days >threshold)
- Cooling degree days
- Interaction terms (vulnerability × temperature)

**Temporal Analysis:**
- Within-person repeated measures (if available)
- Time-series models for seasonal patterns
- Heat wave event studies

**Climate Projections:**
- Use trained models to project CD4 changes under future scenarios
- Estimate healthcare burden from climate change
- Identify vulnerable populations for targeted interventions

---

## Visualization Best Practices (SVG Guidelines)

### What Worked Well:
✅ **Focused on top 3 biomarkers** - Avoided overwhelming with 10 biomarkers × 6 plots = 60 images
✅ **Cross-model summaries first** - Heatmap and bar charts provide context before details
✅ **Native SHAP plots** - Beeswarm and dependence plots are publication-standard
✅ **Color coding** - Red/green for positive/negative effects intuitive
✅ **Annotations** - Model name, R², sample size on each plot

### For SVG Conversion (Future):
- **Beeswarm plots** → Good for SVG (vector scatter points)
- **Heatmaps** → Excellent for SVG (clean cells, text labels)
- **Dependence plots** → Good for SVG (line + scatter)
- **Tables** → Better as PNG (text rendering issues in SVG)

### Publication-Ready Checklist:
- [ ] All fonts readable at 300 DPI
- [ ] Color-blind friendly palettes (use when possible)
- [ ] Axis labels clear and descriptive
- [ ] Legends positioned consistently
- [ ] Figure captions added externally (not in image)
- [ ] File naming convention consistent
- [ ] SVG versions for vector graphics (if needed)

---

## File Organization

```
results/visualization_panel/
├── 01_feature_importance_heatmap.png      # Cross-biomarker summary
├── 02_model_comparison.png                # R² ranking
├── 03_demographics_impact.png             # ΔR² chart
├── 04_CD4_cell_count_cells_µL_beeswarm.png
├── 04_CD4_cell_count_cells_µL_importance.png
├── 04_CD4_cell_count_cells_µL_dependence.png
├── 04_Hematocrit_%_beeswarm.png
├── 04_Hematocrit_%_importance.png
├── 04_Hematocrit_%_dependence.png
├── 04_FASTING_LDL_beeswarm.png
├── 04_FASTING_LDL_importance.png
├── 04_FASTING_LDL_dependence.png
├── 05_summary_table.csv                   # Data table
└── 05_summary_table.png                   # Styled table
```

---

## Technical Notes

### Models Used:
- **CD4**: LightGBM (R² = 0.710)
- **Hematocrit**: RandomForest (R² = 0.935)
- **FASTING LDL**: RandomForest (R² = 0.374)

### Hyperparameters (Tuned):
- RandomForest: n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5
- XGBoost: n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8
- LightGBM: n_estimators=100, max_depth=5, learning_rate=0.1, num_leaves=31

### SHAP Configuration:
- TreeExplainer (exact for tree models)
- Background: 1,000 samples (if n_train > 1,000)
- Test set: 20% of data
- Random seed: 42 (reproducible)

---

## Next Steps

1. **Review visualizations** - Examine all 14 files for insights
2. **Identify priority features** - Use Tier 1-3 rankings above
3. **Plan Phase 2 DLNM analysis** - Focus on CD4 with temperature lags
4. **Investigate hematocrit** - Resolve data quality issues before publication
5. **Write up findings** - Use visualizations in manuscript/presentations

---

**Generated**: 2025-10-30
**Script**: `scripts/create_visualization_panel.py`
**Analysis**: Multi-model comparison with tuned hyperparameters
**Status**: ✅ Complete and ready for Phase 2
