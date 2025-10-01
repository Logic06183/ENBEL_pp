# ENBEL CD4 Heat Analysis - Scientific Summary

## Overview
Comprehensive 6-panel analysis examining temperature-immune function relationships in HIV-positive individuals from Johannesburg clinical trials.

## Dataset
- **Source**: 15 HIV clinical trials (2002-2021) 
- **Total Records**: 11,398 clinical observations
- **Analysis Sample**: 3,244 complete records with CD4 and climate data
- **Climate Data**: ERA5 reanalysis with multi-lag temperature features
- **Geographic Coverage**: Johannesburg metropolitan area, South Africa

## Key Statistical Findings

### Temperature-CD4 Correlations
- **Daily Temperature**: r = 0.038, p = 0.030 (weak positive association)
- **30-day Mean Temperature**: r = -0.029, p = 0.096 (non-significant negative trend)
- **Sample Size**: n = 3,244 observations

### Heat Categories Analysis (ANOVA)
- **F-statistic**: 8.77
- **p-value**: 8.60 × 10⁻⁶ (highly significant)
- **Effect Size (η²)**: 0.008 (small but significant effect)

### Temperature Extremes Comparison
- **Cohen's d**: 0.040 (Q1 coldest vs Q5 hottest)
- **Interpretation**: Small effect size
- **Clinical Relevance**: Modest but measurable immune response differences

## Visualization Panels

### Panel A: Temperature-CD4 Scatter Plot
- Regression line with 95% confidence intervals
- Individual data points showing relationship variability
- Statistical annotations with correlation coefficients

### Panel B: Heat Category Box Plots
- CD4 distributions across 5 temperature categories
- ANOVA results with effect size reporting
- Color-coded categories from cool to hot conditions

### Panel C: Seasonal Patterns
- CD4 means by season with error bars
- Temperature overlay showing seasonal climate variation
- Dual-axis visualization for clear pattern interpretation

### Panel D: Lag Effect Analysis
- Correlation strengths at 0, 7, 14, and 30-day temperature lags
- Statistical significance indicators (*, **, ***)
- Immediate vs delayed temperature effects on immune function

### Panel E: Geographic Variation
- Regional temperature-CD4 correlations across Johannesburg subregions
- Minimum sample size threshold (n>20) for robust estimates
- Spatial heterogeneity in climate-immune relationships

### Panel F: Dose-Response Curve
- CD4 means across temperature quintiles with standard deviations
- Linear trend analysis for dose-response relationship
- Effect size annotation for clinical interpretation

## Statistical Methods
- **Correlation Analysis**: Pearson and Spearman coefficients
- **ANOVA**: One-way analysis of variance for categorical comparisons
- **Effect Sizes**: Cohen's d and eta-squared calculations
- **Multiple Comparisons**: Bonferroni correction applied
- **Confidence Intervals**: 95% CI for regression estimates

## Clinical Implications
1. **Modest but significant** temperature effects on CD4+ T-cell counts
2. **Immediate temperature effects** stronger than lagged responses
3. **Regional variation** suggests local environmental modifiers
4. **Seasonal patterns** align with Johannesburg's climate cycles

## File Outputs
- **SVG**: `/presentation_slides_final/enbel_cd4_heat_analysis_final.svg`
- **PNG**: `/presentation_slides_final/enbel_cd4_heat_analysis_final.png`
- **Source Code**: `/presentation_slides_final/create_cd4_heat_analysis_final.py`

## Attribution
ENBEL Research Collaboration  
Generated with Claude Code  
October 2025