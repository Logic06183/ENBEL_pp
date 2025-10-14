# Repository Structure

This document describes the organized structure of the ENBEL Climate-Health Analysis Pipeline repository.

## Directory Organization

### `/scripts/` - Analysis and Processing Scripts

#### `/scripts/pipelines/` - Main Analysis Pipelines
Core pipeline scripts for running the complete analysis workflow:
- `simple_ml_pipeline.py` - Baseline ML pipeline with basic features
- `improved_climate_health_pipeline.py` - Enhanced pipeline with optimized features
- `state_of_the_art_climate_health_pipeline.py` - Full-featured pipeline with XAI
- `fast_improved_pipeline.py` - Performance-optimized version
- `demo_state_of_the_art_pipeline.py` - Demonstration script
- `run_analysis_pipeline.py` - Main pipeline runner
- `run_new_dataset_pipeline.py` - Pipeline for new dataset integration

**Usage:**
```bash
python scripts/pipelines/simple_ml_pipeline.py
python scripts/pipelines/improved_climate_health_pipeline.py
python scripts/pipelines/state_of_the_art_climate_health_pipeline.py
```

#### `/scripts/analysis/` - Specialized Analysis Scripts
Domain-specific analysis tools:
- `*cd4*` - CD4 count analysis scripts
- `analyze_biomarker_performance.py` - Biomarker model evaluation
- `deep_dive_cd4_heat_analysis.py` - Detailed CD4-heat relationship analysis
- `investigate_heat_vulnerability_score.py` - Heat vulnerability scoring
- `heat_vulnerability_summary.py` - Heat vulnerability summarization

#### `/scripts/imputation/` - Data Imputation Methods
Missing data handling strategies:
- `practical_enbel_imputation.py` - Production-ready imputation
- `corrected_imputation_methodology.py` - Methodology validation
- `fixed_multidimensional_imputation.py` - Multi-dimensional imputation
- `simple_imputation_test.py` - Basic imputation testing

#### `/scripts/testing/` - Test Scripts
Validation and testing utilities:
- `test_pipeline_*.py` - Pipeline validation scripts
- `test_creatinine_*.py` - Creatinine model tests
- `test_imputation.py` - Imputation testing
- `test_new_dataset.py` - New dataset validation
- `validate_simple_pipeline.py` - Simple pipeline validator

#### `/scripts/utilities/` - Helper Scripts
Supporting utilities:
- `setup_and_validate.py` - Environment setup and validation
- `generate_presentation_outputs.py` - Generate presentation materials
- `create_deidentified_dataset.py` - Data de-identification

#### `/scripts/visualization/` - Visualization Scripts

##### `/scripts/visualization/shap/` - SHAP Explainability Plots
SHAP (SHapley Additive exPlanations) visualization scripts:
- `create_*shap*.py` - SHAP analysis and plotting
- `generate_real_shap_*.py` - Real data SHAP visualizations
- Feature importance waterfalls, beeswarms, dependency plots

##### `/scripts/visualization/dlnm/` - DLNM Visualizations
Distributed Lag Non-linear Model plots (Python):
- `create_*dlnm*.py` - DLNM visualization scripts
- Lag-response surfaces, temporal patterns

##### `/scripts/visualization/temporal/` - Temporal Analysis Plots
Time-series and coverage visualizations:
- `create_temporal_*.py` - Temporal pattern plots
- `create_*coverage*.py` - Data coverage visualizations

### `/R/` - R Scripts and Analysis

#### `/R/dlnm_analysis/` - DLNM Core Analysis
R-based distributed lag non-linear models:
- `create_*dlnm*.R` - DLNM model fitting scripts
- `dlnm_validation_*.R` - Model validation
- Uses `dlnm`, `mgcv`, `splines` packages

**Usage:**
```bash
Rscript R/dlnm_analysis/create_cd4_dlnm_final.R
Rscript R/dlnm_validation/dlnm_validation_pipeline.R
```

#### `/R/dlnm_validation/` - DLNM Validation Framework
Cross-validation and sensitivity analysis for DLNM models

#### `/R/utilities/` - R Helper Functions
Supporting R utilities and helper functions

### `/src/enbel_pp/` - Core Package
Production Python package (installable via `pip install -e .`):
- `pipeline.py` - Main `ClimateHealthPipeline` class
- `config.py` - Configuration management
- `ml_utils.py` - ML utilities and feature engineering
- `data_validation.py` - Data quality checks
- `imputation.py` - Imputation strategies

### `/tests/` - Unit and Integration Tests
Formal test suite (run with `pytest`):
```bash
pytest                    # All tests
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -n auto           # Parallel execution
```

### `/configs/` - Configuration Files
YAML configuration files:
- `default.yaml` - Default pipeline configuration
- Custom configs for different analysis modes

### `/data/` - Data Directory
Dataset storage:
- `/data/raw/` - Original datasets
- Clinical and GCRO socioeconomic data
- **Note:** Data files not tracked in git (see `.gitignore`)

### `/docs/` - Documentation

#### `/docs/summaries/` - Analysis Summaries
Research summaries and metadata:
- `CLINICAL_METADATA.md` - Clinical dataset documentation
- `GCRO_DATA_DICTIONARY.md` - GCRO data dictionary
- `GCRO_METADATA.json` - GCRO metadata
- `PACKAGE_SUMMARY.json` - Package overview
- `enhanced_dataset_analysis_summary.md` - Dataset analysis
- `IMPLEMENTATION_SUMMARY.md` - Implementation notes

#### `/docs/methodology/` - Methodological Documentation
Research methods and validation documentation

### `/archive/` - Historical Code
Archived code from previous versions:
- `/archive/old_analysis_20250930/` - Analysis scripts before refactor
- `/archive/old_svg_slides_20250930/` - Previous slide versions
- `/archive/visualizations/` - Legacy visualization code

### `/presentation_slides_final/` - Presentation Materials
Final presentation slides and figures:
- SVG visualizations for publications
- Map vector data for Johannesburg study area

### `/presentation_slides_archive/` - Archived Presentations
Previous versions of presentation materials

### `/reanalysis_outputs/` - Analysis Results
Output from pipeline runs:
- `/reanalysis_outputs/figures_svg/` - SVG figures
- `/reanalysis_outputs/presentation_statistics/` - Statistical outputs
- `/reanalysis_outputs/validation/` - Validation results

### `/utils/` - Utility Modules
Supporting utility code (legacy, prefer `/src/enbel_pp/`)

### Root Configuration Files
- `CLAUDE.md` - AI assistant instructions
- `README.md` - Main project documentation
- `pyproject.toml` - Python package configuration
- `requirements.txt` - Python dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `Dockerfile` - Docker containerization

## Quick Start

### 1. Installation
```bash
# Install package in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[all]"
```

### 2. Run Analysis
```bash
# Simple baseline pipeline
python scripts/pipelines/simple_ml_pipeline.py

# State-of-the-art with XAI
python scripts/pipelines/state_of_the_art_climate_health_pipeline.py

# DLNM validation in R
Rscript R/dlnm_validation/dlnm_validation_pipeline.R
```

### 3. Run Tests
```bash
pytest                    # All tests
pytest -v --tb=short     # Verbose with short tracebacks
```

### 4. Generate Visualizations
```bash
# SHAP plots
python scripts/visualization/shap/generate_real_shap_plots.py

# DLNM plots
Rscript R/dlnm_analysis/create_cd4_dlnm_final.R
```

## File Naming Conventions

### Python Scripts
- `*_pipeline.py` - End-to-end pipeline scripts
- `test_*.py` - Testing scripts
- `create_*.py` - Visualization/output generation
- `generate_*.py` - Batch generation utilities
- `run_*.py` - Main execution scripts
- `validate_*.py` - Validation utilities

### R Scripts
- `*_dlnm_*.R` - DLNM analysis scripts
- `*_validation_*.R` - Validation frameworks
- `create_*.R` - Plotting scripts

## Migration Notes

This structure represents a reorganization performed on 2025-10-14 to improve:
1. **Discoverability** - Related scripts grouped logically
2. **Maintainability** - Clear separation of concerns
3. **Usability** - Intuitive navigation for new contributors

### Changes from Previous Structure
- Moved 63 Python scripts from root to `/scripts/`
- Organized 29 R scripts into `/R/` subdirectories
- Consolidated 10 markdown docs into `/docs/`
- Preserved core package in `/src/enbel_pp/`

## Contributing

When adding new scripts:
1. Place in appropriate `/scripts/` subdirectory
2. Follow naming conventions above
3. Update this `STRUCTURE.md` if adding new categories
4. Add corresponding tests to `/tests/`

## Questions?

See:
- `CLAUDE.md` - Development instructions
- `README.md` - Project overview
- `/docs/summaries/` - Dataset documentation
