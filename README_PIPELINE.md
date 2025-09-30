# ENBEL Climate-Health Analysis Pipeline

## Overview

This repository contains a comprehensive, methodologically sound pipeline for analyzing climate-health relationships using clinical trial data from Johannesburg, South Africa. The pipeline implements:

1. **Methodological Imputation**: Ecological and KNN-based imputation for missing socioeconomic variables
2. **Machine Learning Analysis**: Multi-system biomarker analysis with SHAP explainability
3. **DLNM Validation**: Distributed Lag Non-linear Models for temporal validation

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install R packages (in R console)
install.packages(c('dlnm', 'mgcv', 'ggplot2', 'dplyr', 'jsonlite'))

# 3. Run the complete pipeline
python run_analysis_pipeline.py
```

## Repository Structure

```
ENBEL_pp/
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/             # Intermediate processing
│   └── imputed/              # Imputed datasets
├── src/enbel_pp/
│   ├── imputation_pipeline.py    # Ecological + KNN imputation
│   ├── biomarker_ml_pipeline.py  # ML analysis with SHAP
│   └── pipeline.py               # Core pipeline infrastructure
├── R/dlnm_validation/
│   └── dlnm_validation_pipeline.R # DLNM validation scripts
├── results/
│   ├── ml_analysis/          # ML and SHAP results
│   └── dlnm_validation/      # DLNM validation results
└── notebooks/                # Analysis notebooks
    ├── 01_imputation/
    ├── 02_ml_analysis/
    └── 03_validation/
```

## Methodology

### 1. Imputation Strategy

The pipeline implements a two-stage imputation approach:

#### Ecological Imputation
- Uses ward-level aggregates from GCRO socioeconomic data
- Matches by geographic location (ward) and temporal period (year)
- Imputes categorical variables using mode, continuous using mean
- Preserves population-level patterns

#### KNN Imputation
- Distance-weighted neighbors in standardized feature space
- Uses demographic and climate features to find similar observations
- Handles remaining missing values after ecological imputation
- Validates imputed distributions against original data

**Key Variables Imputed:**
- Dwelling type (formal/informal housing)
- Education level
- Income category
- Employment status
- Household characteristics

### 2. Machine Learning Analysis

The ML pipeline analyzes biomarkers across multiple physiological systems:

#### Physiological Systems
- **Immune**: CD4 count, lymphocytes, white blood cells
- **Metabolic**: Glucose, HbA1c, insulin
- **Cardiovascular**: Blood pressure (systolic/diastolic), heart rate
- **Renal**: Creatinine, urea, eGFR, albumin
- **Lipid**: Cholesterol (total/HDL/LDL), triglycerides
- **Hematological**: Hemoglobin, hematocrit, RBC, platelets

#### Model Selection
- **Random Forest**: Baseline model with feature importance
- **XGBoost**: Optimized gradient boosting
- **LightGBM**: Fast gradient boosting for large datasets

#### Quality Metrics
- R² score (minimum 0.3 for SHAP analysis)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Explained variance
- Cross-validation scores

#### SHAP Analysis
- Tree-based explainers for all models
- Feature importance ranking
- Climate-specific feature identification
- Interaction effects for top predictors

### 3. DLNM Validation

The R-based DLNM validation:
- Creates crossbasis functions for lag-response relationships
- Models non-linear and delayed effects
- Generates 3D response surfaces
- Compares R² values with ML models
- Validates temporal patterns identified by SHAP

## Step-by-Step Execution

### Option 1: Automated Pipeline

```bash
python run_analysis_pipeline.py
```

This runs all steps automatically:
1. Data organization
2. Imputation
3. ML analysis
4. DLNM validation

### Option 2: Manual Step Execution

```python
# Step 1: Imputation
from src.enbel_pp.imputation_pipeline import run_full_imputation_pipeline

imputed_df = run_full_imputation_pipeline(
    clinical_path="data/raw/clinical_dataset.csv",
    gcro_path="data/raw/gcro_socioeconomic.csv",
    output_dir="data/imputed"
)

# Step 2: ML Analysis
from src.enbel_pp.biomarker_ml_pipeline import run_biomarker_ml_analysis

results = run_biomarker_ml_analysis(
    data_path="data/imputed/clinical_imputed_latest.csv",
    output_dir="results/ml_analysis",
    min_model_quality=0.3
)

# Step 3: DLNM Validation (in R)
# Rscript R/dlnm_validation/dlnm_validation_pipeline.R \
#   results/ml_analysis/ml_results.json \
#   data/imputed/clinical_imputed.csv
```

## Configuration

### Imputation Parameters
```python
# In imputation_pipeline.py
n_neighbors = 5           # KNN neighbors
match_variables = ['ward', 'year']  # Ecological matching
```

### ML Parameters
```python
# In biomarker_ml_pipeline.py
min_model_quality = 0.3   # Minimum R² for SHAP
cv_folds = 5             # Cross-validation folds
test_size = 0.2          # Train-test split
```

### DLNM Parameters
```r
# In dlnm_validation_pipeline.R
lag = 21                 # Maximum lag days
df_temp = 4             # Temperature spline df
df_lag = 4              # Lag spline df
```

## Output Files

### Imputation Results
- `data/imputed/clinical_imputed_YYYYMMDD_HHMMSS.csv` - Imputed dataset
- `data/imputed/imputation_report_*.txt` - Methodology report
- `data/imputed/imputation_log_*.json` - Detailed imputation log

### ML Analysis Results
- `results/ml_analysis/ml_analysis_results_*.json` - Full results
- `results/ml_analysis/ml_analysis_report_*.txt` - Summary report
- SHAP values and feature importance for each biomarker

### DLNM Validation
- `results/dlnm_validation/dlnm_validation_results.json` - Validation metrics
- `results/dlnm_validation/dlnm_validation_report.txt` - Summary report
- Lag-response plots for validated biomarkers

## Reproducibility

The pipeline ensures reproducibility through:
- Fixed random seeds (default: 42)
- Timestamped outputs
- Comprehensive logging
- JSON serialization of all parameters
- Version control of dependencies

## Performance Expectations

Based on previous analyses:
- **CD4 count**: R² ~0.65-0.70 (high climate sensitivity)
- **Glucose**: R² ~0.55-0.60 (moderate sensitivity)  
- **Blood pressure**: R² ~0.40-0.50
- **Other biomarkers**: R² 0.20-0.40 range

Models with R² < 0.3 are flagged as insufficient for reliable SHAP interpretation.

## Troubleshooting

### Missing Dependencies
```bash
# Python packages
pip install pandas numpy scikit-learn xgboost lightgbm shap

# R packages
R -e "install.packages(c('dlnm', 'mgcv', 'ggplot2', 'dplyr', 'jsonlite'))"
```

### Data Issues
- Ensure clinical data has standard column names
- Check date formats (YYYY-MM-DD expected)
- Verify geographic codes match between datasets

### Memory Issues
- For large datasets, consider chunking in imputation
- Reduce cv_folds for faster processing
- Use subset of parameter grid for optimization

## Citation

If using this pipeline, please cite:
```
ENBEL Climate-Health Analysis Pipeline
GitHub: https://github.com/[your-repo]
Version: 1.0.0
```

## Contact

For questions or issues, please open a GitHub issue or contact the ENBEL project team.