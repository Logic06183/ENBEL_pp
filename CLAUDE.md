# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ENBEL Climate-Health Analysis Pipeline - a comprehensive research framework analyzing climate impacts on health biomarkers in Johannesburg, South Africa. Integrates 15 HIV clinical trials (11,398 records) with GCRO socioeconomic data (58,616 household records) and ERA5 climate reanalysis for advanced ML/XAI analysis.

## Key Commands

### Setup and Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"        # Development tools
pip install -e ".[geospatial]" # Geospatial analysis
pip install -e ".[all]"        # Everything
```

### Running Tests
```bash
# Run all tests with coverage (configured for 80% minimum)
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m slow         # Slow-running tests
pytest -n auto         # Parallel execution

# Run a single test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v --tb=short
```

### Code Quality
```bash
# Format code (line length 88)
black src/ tests/

# Sort imports
isort src/ tests/

# Lint check
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running Pipelines
```bash
# Main pipeline with different modes
python -m enbel_pp.pipeline --mode simple
python -m enbel_pp.pipeline --mode improved
python -m enbel_pp.pipeline --mode state_of_the_art

# Standalone analysis scripts
python simple_ml_pipeline.py                    # Quick baseline
python improved_climate_health_pipeline.py      # Enhanced features
python state_of_the_art_climate_health_pipeline.py  # Full optimization
python practical_enbel_imputation.py           # Advanced imputation

# R-based DLNM analysis
Rscript R/dlnm_validation/dlnm_validation_pipeline.R
```

## Architecture

### Pipeline Core (`src/enbel_pp/`)

**`pipeline.py`**: Main `ClimateHealthPipeline` class orchestrating analysis
- Three modes: simple (baseline), improved (optimized), state_of_the_art (XAI+full optimization)
- Manages data loading, preprocessing, model training, evaluation
- Integrates with SHAP for explainability

**`config.py`**: Configuration management
- `ENBELConfig` class for centralized settings
- YAML-based configuration (configs/default.yaml)
- Path management, ML hyperparameters, biomarker lists

**`ml_utils.py`**: ML utilities
- `prepare_features_safely()`: Feature engineering with lag windows
- `train_model_with_cv()`: Cross-validation framework
- `evaluate_model_performance()`: Comprehensive metrics
- `apply_multiple_testing_correction()`: Bonferroni correction

**`data_validation.py`**: Quality assurance
- `validate_biomarker_data()`: Unit conversion, range checks
- `validate_file_exists()`: Path validation
- South African medical standards enforcement

**`imputation.py`**: Missing data handling
- Multiple imputation strategies (mean, median, KNN, iterative)
- Preserves temporal relationships
- Handles both clinical and socioeconomic data

### Data Processing Flow

1. **Data Ingestion**
   - Load CLINICAL_DATASET_COMPLETE_CLIMATE.csv (99.5% climate coverage)
   - Load GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv
   - Validate coordinates, dates, biomarker ranges

2. **Feature Engineering**
   - Create climate lags (7, 14, 30 days)
   - Calculate heat stress indices
   - Generate temporal features (season, month)
   - Compute temperature anomalies

3. **Model Pipeline**
   - Feature selection (SelectKBest, mutual information)
   - Train models: RandomForest, XGBoost, LightGBM
   - Hyperparameter optimization via Optuna/GridSearchCV
   - 5-fold cross-validation with stratification

4. **Evaluation & XAI**
   - SHAP analysis for feature importance
   - Generate waterfall, beeswarm, dependency plots
   - Calculate R², MAE, RMSE metrics
   - Apply multiple testing corrections

### R Integration (DLNM Analysis)

**`R/dlnm_validation/dlnm_validation_pipeline.R`**: Main DLNM script
- Distributed lag non-linear models for temporal effects
- Cross-basis functions for temperature-lag interactions
- Generates 3D response surfaces and lag-specific plots

**Key R packages**:
- `dlnm`: Core DLNM functionality
- `mgcv`: GAM modeling
- `splines`: Basis functions
- `ggplot2`: Visualization

## Critical Patterns

### Reproducibility
```python
# Always set seed at start
from enbel_pp.config import set_reproducible_environment
set_reproducible_environment(42)
```

### Error Handling
```python
# Wrapped data loading
try:
    df = pd.read_csv(path)
    validation_result = validate_biomarker_data(df, biomarker)
    if validation_result['status'] == 'error':
        logger.error(f"Validation failed: {validation_result['message']}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
```

### Performance Optimization
- Use `n_jobs=-1` for parallel processing
- Chunk large datasets (>50K rows)
- Prefer LightGBM for speed on high-dimensional data
- Cache intermediate results in `cache/` directory

## Common Development Tasks

### Adding New Biomarker
1. Edit `configs/default.yaml`:
   ```yaml
   biomarkers:
     - "New_Biomarker_Name"
   ```
2. Add unit conversion in `src/enbel_pp/data_validation.py`
3. Run pipeline: `python -m enbel_pp.pipeline --target "New_Biomarker_Name"`

### Custom Configuration
```python
from enbel_pp.pipeline import ClimateHealthPipeline

# Use custom config
pipeline = ClimateHealthPipeline(
    analysis_mode='improved',
    config_file='configs/custom.yaml'
)
pipeline.run(target_biomarker='CD4 cell count (cells/µL)')
```

### Debugging ML Models
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get intermediate results
pipeline = ClimateHealthPipeline(analysis_mode='simple')
pipeline.run(target_biomarker='glucose', save_intermediate=True)
# Check cache/ directory for saved features and models
```

### Generating Publication Figures
```bash
# SHAP visualizations
python generate_real_shap_plots.py

# DLNM plots
Rscript archive/analysis_scripts/create_dlnm_plots.R

# Custom SVG figures
python create_shap_waterfall_plots.py
python create_temporal_patterns_slide.py
```

## Dataset Details

### Clinical Dataset (11,398 records)
- **Key columns**: Patient biomarkers, dates, coordinates
- **Biomarkers**: CD4, glucose, creatinine, hemoglobin, cholesterol (SA units)
- **Climate**: 16 ERA5-derived features with multi-lag analysis
- **Coverage**: 2002-2021, 99.5% climate matched

### GCRO Dataset (58,616 records)  
- **Key columns**: Dwelling type, income, education, heat vulnerability
- **Geographic**: Ward-level aggregation across Johannesburg
- **Temporal**: 6 survey waves (2011-2021)
- **Vulnerability**: Composite indices for heat exposure risk

### Model Performance Benchmarks
- CD4 count: R² = 0.699 (highly climate-sensitive)
- Glucose: R² = 0.600 (moderate sensitivity)  
- Blood pressure: R² = 0.450
- Other biomarkers: R² = 0.3-0.5 range