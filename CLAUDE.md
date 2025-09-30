# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the ENBEL Climate-Health Analysis Pipeline - a scientific research project analyzing climate impacts on health biomarkers in Johannesburg, South Africa. The repository contains:

- Clinical datasets from 15 HIV clinical trials (11,398 records)
- Socioeconomic data from GCRO surveys (58,616 household records)
- ERA5 climate data integration
- Machine learning analysis pipelines with XAI/SHAP explainability
- R and Python analysis scripts for DLNM and statistical modeling

## Key Commands

### Setup and Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install with development dependencies
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow         # Slow tests

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running Analysis Pipelines
```bash
# Run the main pipeline
python -m enbel_pp.pipeline

# Run simple ML pipeline
python simple_ml_pipeline.py

# Run improved pipeline
python improved_climate_health_pipeline.py

# Run state-of-the-art pipeline
python state_of_the_art_climate_health_pipeline.py

# Run practical imputation
python practical_enbel_imputation.py
```

## Architecture

### Core Module Structure (`src/enbel_pp/`)
- **`pipeline.py`**: Main `ClimateHealthPipeline` class with three analysis modes (simple, improved, state_of_the_art)
- **`config.py`**: Configuration management using YAML files, handles paths and ML parameters
- **`data_validation.py`**: Data quality checks and biomarker validation functions
- **`ml_utils.py`**: Machine learning utilities, feature selection, cross-validation, and model evaluation
- **`imputation.py`**: Advanced imputation strategies for handling missing data

### Configuration System (`configs/`)
- **`default.yaml`**: Base configuration
- **`development.yaml`**: Development environment settings
- **`production.yaml`**: Production deployment settings

### Analysis Modes
1. **Simple**: Quick exploration with basic Random Forest, no hyperparameter tuning
2. **Improved**: Feature selection, hyperparameter optimization, multiple models
3. **State-of-the-art**: Full optimization, SHAP analysis, extensive cross-validation

### Data Flow
1. Load clinical/GCRO datasets → Validate biomarkers → Feature identification
2. Climate feature engineering → Multiple lag analysis → Temporal aggregations
3. Model training (RF, XGBoost, LightGBM) → Cross-validation → Hyperparameter optimization
4. SHAP explainability analysis → Multiple testing correction → Results export

## Important Context

### Biomarker Standards
- Uses South African medical standards for biomarkers
- Key biomarkers: CD4 count, glucose, hemoglobin, creatinine, cholesterol
- All units standardized (e.g., glucose in mmol/L, creatinine in umol/L)

### Climate Variables
- ERA5 reanalysis data with ~31km resolution
- Multi-lag features (7-day, 14-day, 30-day rolling averages)
- Heat stress indices and temperature anomalies
- 99.5% climate coverage achieved through careful data integration

### Model Performance Targets
- CD4 count: R² ~0.699 (high climate sensitivity)
- Glucose: R² ~0.600 (moderate sensitivity)
- Other biomarkers: R² 0.3-0.5 range

### Key Files to Edit
- `src/enbel_pp/pipeline.py`: Main analysis pipeline logic
- `src/enbel_pp/ml_utils.py`: Add new ML algorithms or metrics
- `configs/default.yaml`: Adjust analysis parameters
- `practical_enbel_imputation.py`: Modify imputation strategies

### Testing Approach
- Unit tests in `tests/test_*.py`
- Fixtures in `tests/conftest.py`
- Minimum 80% code coverage required
- Pre-commit hooks enforce code quality

## Critical Patterns

### Error Handling
- All data loading wrapped in validation checks
- Biomarker analysis returns error dict on failure
- Comprehensive logging at INFO level

### Reproducibility
- Random seeds set via `set_reproducible_environment()`
- All analysis timestamped
- Results saved as JSON with full configuration

### Performance Considerations
- Large datasets (70K+ records) - use chunking where needed
- XGBoost/LightGBM for faster training on large features
- Parallel cross-validation when possible

## Common Tasks

### Adding a New Biomarker
1. Add to biomarker list in `configs/default.yaml`
2. Ensure proper unit conversion in `data_validation.py`
3. Run pipeline with new biomarker in target list

### Changing Analysis Parameters
1. Edit `configs/default.yaml` or create custom config
2. Pass config file to `ClimateHealthPipeline(config_file='path/to/config.yaml')`

### Generating Visualizations
- SHAP plots: Use functions in analysis scripts
- DLNM plots: Run R scripts in `archive/analysis_scripts/`
- SVG generation: Various `create_*.py` scripts for publication figures