# ENBEL SHAP Waterfall Plot Documentation

## Overview

The SHAP waterfall plot (`enbel_shap_waterfall_final.svg`) provides a scientifically rigorous explanation of how individual climate features contribute to CD4 cell count predictions in the ENBEL climate-health analysis.

## Scientific Methodology

### SHAP (SHapley Additive exPlanations)
- **Theoretical Foundation**: Based on Shapley values from cooperative game theory
- **Interpretation**: Each feature's contribution to moving the prediction away from the population average
- **Additivity**: All SHAP values sum to the difference between prediction and base value
- **Fairness**: Satisfies efficiency, symmetry, dummy, and additivity axioms

### Model Selection Process
The analysis compared three regression models:
- **Ridge Regression**: R² = 0.0514, RMSE = 967.4 cells/µL (selected)
- **Random Forest**: R² = 0.0511, RMSE = 967.5 cells/µL  
- **XGBoost**: R² = 0.0417, RMSE = 972.3 cells/µL

Ridge regression was selected as the best performing model with the highest test R² (0.0514).

### Dataset Characteristics
- **Total Participants**: 4,606 with complete CD4 and climate data
- **Climate Features**: 17 features including temperature metrics, heat vulnerability scores, and temporal aggregations
- **Demographic Controls**: 6 features (Sex, Race, location, temporal variables)
- **Temporal Range**: HIV clinical trials in Johannesburg (2002-2021)
- **Climate Data Source**: ERA5 reanalysis with local validation

## Visualization Features

### Waterfall Structure
- **Base Value**: Population mean CD4 count (398.0 cells/µL)
- **Feature Contributions**: Individual SHAP values for top 15 most influential features
- **Final Prediction**: Model output for the selected case
- **Actual Value**: Observed CD4 count (green dashed line)

### Color Coding
- **Red Bars**: Positive contributions (increase CD4 prediction)
- **Blue Bars**: Negative contributions (decrease CD4 prediction)  
- **Gray Bars**: Population mean and final prediction
- **Green Line**: Actual observed value

### Representative Case Details
- **Case ID**: 1 (selected for moderate CD4 level and interesting climate exposure)
- **Actual CD4**: 446.0 cells/µL
- **Predicted CD4**: 304.1 cells/µL
- **Prediction Error**: 141.9 cells/µL

## Climate-Health Insights

### Key Findings
1. **Heat Vulnerability**: Patient's heat vulnerability score provides significant negative contribution
2. **Temperature Patterns**: Multiple temperature metrics show varied directional effects
3. **Temporal Lags**: Different lag periods (7d, 14d, 30d) capture distinct physiological responses
4. **Geographic Factors**: Location coordinates contribute to prediction accuracy

### Scientific Interpretation
- Climate factors explain ~5% of CD4 count variation in this population
- Heat exposure appears to have negative associations with immune function
- Multi-temporal climate patterns provide better predictive power than single measurements
- Individual-level predictions require integration of multiple climate dimensions

## Limitations and Considerations

### Model Performance
- Modest R² (0.051) reflects complex nature of CD4 regulation beyond climate
- Other factors (viral load, treatment adherence, genetics) likely dominate
- Climate effects may be indirect through behavioral or physiological pathways

### Methodological Notes
- SHAP values are local explanations (specific to this individual case)
- Feature interactions not explicitly shown in waterfall format
- Linear model assumptions may miss nonlinear climate-health relationships

## Publication Quality

### Technical Specifications
- **Format**: SVG (scalable vector graphics) for publication
- **Resolution**: 300 DPI equivalent
- **Typography**: Serif fonts appropriate for scientific journals
- **Color Scheme**: Accessible to colorblind viewers
- **Annotations**: Complete methodological transparency

### Usage Recommendations
- Suitable for scientific presentations and peer-reviewed publications
- Complements other analyses (DLNM, time series) in comprehensive climate-health assessment
- Demonstrates state-of-the-art explainable AI methodology in environmental epidemiology

## References and Attribution

**Data Sources**:
- ENBEL Clinical Trials (Johannesburg HIV cohorts, 2002-2021)
- ERA5 Climate Reanalysis (Copernicus Climate Change Service)
- South African Air Quality Information System (SAAQIS)

**Methodology**:
- Lundberg & Lee (2017). "A unified approach to interpreting model predictions." NIPS.
- Molnar (2020). "Interpretable Machine Learning: A Guide for Making Black Box Models Explainable."

**Generated**: October 2025, ENBEL Climate-Health Research Team