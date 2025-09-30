# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a climate-health research repository focused on analyzing relationships between climate variables (temperature, humidity, pressure) and health biomarkers using machine learning, XAI (eXplainable AI), and advanced statistical methods. The dataset contains clinical trial data from Johannesburg with 18,205 rows × 343 columns.

## Key Commands

### Running Analysis
```bash
# Main ML pipeline (5-10 minutes on M4 Mac)
python 01_initial_analysis/supervised_ml/optimized_interpretable_ml_pipeline.py

# Data imputation
python implement_multidimensional_imputation.py

# SHAP visualization generation
python create_shap_visualizations.py

# Data overview visualization
python create_comprehensive_data_overview_final.py
```

### Dependencies
```bash
pip install -r analysis_scripts/requirements.txt
```

Key packages: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, scipy

## Repository Structure

The codebase is organized into progressive analysis stages:

- **01_initial_analysis/**: Initial ML exploration and optimization
  - `supervised_ml/`: Core ML pipelines for biomarker prediction
  - `optimization/`: Advanced climate-health optimization methods

- **02_cohort_separation/**: Validation frameworks for rigorous analysis
  - `validation_frameworks/`: Cross-validation, interpretability, XAI frameworks

- **03_flexible_discovery/**: Alternative analysis approaches
  - `ecological/`: Time series and multilevel frameworks
  - `supervised/`: Enhanced discovery analysis
  - `unsupervised/`: Signal detection and alternative methods

- **04_rigorous_methodology/**: Final validation and XAI breakthrough analyses
  - `final_analysis/`: Comprehensive final analysis and validation
  - `xai_validation/`: Novel XAI validation frameworks

- **analysis_scripts/utilities/**: Core utility functions for analysis
- **documentation/**: Scientific reports, methodologies, and insights

## Data Files

Main datasets (CSV format):
- `CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv` (36MB): Clinical data with imputed socioeconomic variables
- `CLINICAL_WITH_IMPUTED_SOCIOECONOMIC.csv` (35MB): Alternative imputation version
- `FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv` (66MB): Complete dataset with climate lag features (0-21 days)

## Key Analysis Components

### Biomarkers Analyzed
- CD4 cell count (immune system)
- Creatinine (kidney function) 
- Hemoglobin (oxygen transport)
- Blood pressure (cardiovascular)
- Glucose levels (metabolic)
- Cholesterol markers (HDL, LDL, Total)

### Climate Features
- Temperature, humidity, pressure measurements
- Multi-day lag features (0-21 days)
- 266 total climate features after engineering

### Model Types Used
- Random Forest (250 trees, max_depth=15)
- XGBoost (learning_rate=0.05, max_depth=8)
- LightGBM
- Neural networks (MLPRegressor)
- ElasticNet

## Output Directories

- `optimized_results/`: Model results and performance metrics
- `trained_models/`: Saved model files
- `*.svg`, `*.png`: Generated visualizations (SHAP plots, DLNM plots, slides)

## Performance Expectations

- R² improvements: +0.02 to +0.10 over baseline
- R² > 0.05 indicates clinically meaningful relationships
- Analysis runtime: 5-10 minutes on M4 Mac
- Memory usage: 2-4GB peak

## Visualization Generation

The repository includes multiple visualization scripts for creating publication-quality figures:
- SHAP beeswarm, waterfall, and summary plots
- DLNM (Distributed Lag Non-linear Models) plots
- Data overview slides
- SVG to PNG conversion utilities

## Important Notes

- All scripts use `n_jobs=-1` for parallel processing on Apple Silicon
- Scripts include automatic progress logging with timestamps
- Results are saved with timestamps for versioning
- The codebase emphasizes interpretability and scientific rigor