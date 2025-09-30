# ENBEL Climate-Health Analysis Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/enbel/climate-health-analysis/workflows/CI/badge.svg)](https://github.com/enbel/climate-health-analysis/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://enbel.github.io/climate-health-analysis)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/r/enbel/climate-health-analysis)

> **A comprehensive, reproducible machine learning pipeline for analyzing climate impacts on health biomarkers using explainable AI techniques.**

## 🌟 Overview

The ENBEL (Environmental Biomarker Epidemiological Learning) Climate-Health Analysis Pipeline is a state-of-the-art scientific computing framework designed to investigate the complex relationships between climate variables and human health biomarkers. Built with reproducibility, scalability, and scientific rigor as core principles, this pipeline enables researchers to conduct robust climate-health analyses using machine learning and explainable AI techniques.

### 🎯 Key Features

- **🔬 Scientific Rigor**: Built-in statistical validation, multiple testing correction, and reproducibility guarantees
- **🤖 Advanced ML**: Support for Random Forest, XGBoost, LightGBM with hyperparameter optimization
- **🔍 Explainable AI**: Integrated SHAP analysis for interpretable climate-health relationships
- **⚙️ Configurable**: Flexible YAML-based configuration system for different analysis scenarios
- **🐳 Containerized**: Docker support for consistent environments and reproducible deployments
- **📊 Comprehensive**: End-to-end pipeline from data validation to scientific reporting
- **🧪 Quality Assured**: Extensive testing framework with automated CI/CD

### 🌡️ Research Applications

- **Climate-Health Impact Modeling**: Quantify relationships between temperature, humidity, and health outcomes
- **Biomarker Response Analysis**: Analyze how cardiovascular, metabolic, and immune markers respond to climate
- **Vulnerable Population Studies**: Identify climate-sensitive health indicators in different populations
- **Urban Heat Island Research**: Study differential health impacts across urban environments
- **Temporal Pattern Discovery**: Uncover lag effects and seasonal patterns in climate-health relationships

## 📊 Scientific Context

This pipeline analyzes climate-health relationships using data from Johannesburg, South Africa, including:

- **11,398 clinical participants** from 15 harmonized HIV clinical trials (2002-2021)
- **58,616 household survey participants** from GCRO socioeconomic surveys (2011-2021)
- **16 ERA5-derived climate variables** with multi-lag temporal analysis
- **13+ health biomarkers** including cardiovascular, metabolic, and immune indicators

### 🔬 Key Findings Enabled

- Temperature variability more predictive than mean temperature for health outcomes
- Immune function (CD4 count) highly climate-sensitive (R² = 0.699)
- Dwelling type critical for heat vulnerability assessment
- Multi-lag climate effects identified in biomarker responses

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 8GB+ RAM recommended
- 2GB+ disk space for models and results

### Installation

#### Option 1: pip install (recommended)
```bash
pip install enbel-pp
```

#### Option 2: Docker (for reproducibility)
```bash
docker pull enbel/climate-health-analysis:latest
docker run -v $(pwd)/data:/app/data enbel/climate-health-analysis
```

#### Option 3: From source
```bash
git clone https://github.com/enbel/climate-health-analysis.git
cd climate-health-analysis
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from enbel_pp import ClimateHealthPipeline

# Initialize pipeline
pipeline = ClimateHealthPipeline(analysis_mode='improved')

# Load your climate-health dataset
pipeline.load_data('your_climate_health_data.csv')

# Run comprehensive analysis
results = pipeline.run_comprehensive_analysis()

# Generate summary report
print(pipeline.generate_summary_report())
```

### Command Line Interface

```bash
# Run analysis pipeline
enbel-pipeline --config configs/production.yaml --mode state-of-the-art

# Validate configuration
enbel-config --validate configs/your-config.yaml

# Run with Docker
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           enbel/climate-health-analysis improved
```

## 📁 Project Structure

```
ENBEL_pp/
├── src/enbel_pp/           # Main package source
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Main analysis pipeline
│   ├── data_validation.py  # Data quality assurance
│   ├── ml_utils.py         # Machine learning utilities
│   └── visualization.py    # Scientific plotting
├── configs/                # Configuration files
│   ├── default.yaml        # Default settings
│   ├── development.yaml    # Development configuration
│   └── production.yaml     # Production settings
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── docker/                 # Container configurations
├── results/                # Analysis outputs (gitignored)
├── models/                 # Trained models (gitignored)
└── figures/                # Generated plots (gitignored)
```

## 🔧 Configuration

The pipeline uses YAML configuration files for flexible analysis setup:

```yaml
# Basic configuration example
analysis_mode: improved
random_state: 42

biomarkers:
  - "systolic blood pressure"
  - "FASTING GLUCOSE"
  - "CD4 cell count (cells/µL)"

ml_settings:
  cv_folds: 5
  test_size: 0.2
  feature_selection: true
  hyperparameter_tuning: true

random_forest:
  n_estimators: 250
  max_depth: 15
  min_samples_split: 10
```

## 🧪 Analysis Modes

### Simple Mode
- Quick exploratory analysis
- Basic Random Forest models
- 3-fold cross-validation
- ~10 minutes runtime

### Improved Mode (Recommended)
- Feature selection and engineering
- Hyperparameter optimization
- 5-fold cross-validation
- Statistical validation
- ~30 minutes runtime

### State-of-the-Art Mode
- Full hyperparameter optimization
- SHAP explainability analysis
- 10-fold cross-validation
- Comprehensive validation
- ~2 hours runtime

## 📊 Example Results

```python
# Load and examine results
from enbel_pp.results import ResultsManager

manager = ResultsManager()
results = manager.load_latest_results('improved_analysis')

# View performance summary
print(f"Average R²: {results['summary_statistics']['mean_r2']:.3f}")
print(f"Best biomarker: {results['top_biomarker']} (R² = {results['best_r2']:.3f})")

# Access model-specific results
bp_results = results['biomarker_results']['systolic blood pressure']
print(f"Systolic BP prediction R²: {bp_results['best_score']:.3f}")
print(f"Top climate features: {bp_results['top_features'][:5]}")
```

## 🔬 Scientific Validation

### Reproducibility
- Fixed random seeds across all operations
- Environment containerization with Docker
- Comprehensive version tracking
- Statistical validation of results

### Statistical Rigor
- Multiple testing correction (Bonferroni)
- Cross-validation with confidence intervals
- Permutation tests for feature importance
- Bootstrap analysis for uncertainty quantification

### Quality Assurance
- Automated data validation
- Model performance benchmarking
- Comprehensive testing suite (80%+ coverage)
- Continuous integration with GitHub Actions

## 📚 Documentation

- **[User Guide](docs/user_guide.md)**: Complete usage instructions
- **[API Reference](docs/api/)**: Detailed API documentation
- **[Scientific Methods](docs/methodology.md)**: Statistical and ML methodology
- **[Configuration Guide](docs/configuration.md)**: Configuration options
- **[Examples](notebooks/examples/)**: Jupyter notebook tutorials
- **[Contributing](CONTRIBUTING.md)**: How to contribute

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/enbel/climate-health-analysis.git
cd climate-health-analysis

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=src/enbel_pp --cov-report=html
```

### Docker Development

```bash
# Build development image
docker-compose -f docker/docker-compose.yml build enbel-dev

# Start development container
docker-compose -f docker/docker-compose.yml run --rm enbel-dev shell

# Run tests in container
docker-compose -f docker/docker-compose.yml run --rm enbel-test
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Types of Contributions
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 🔬 Scientific methodology enhancements

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{enbel_climate_health_2024,
  title={ENBEL Climate-Health Analysis Pipeline},
  author={ENBEL Project Team},
  year={2024},
  url={https://github.com/enbel/climate-health-analysis},
  version={1.0.0}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: HEAT Center Research Projects, Gauteng City-Region Observatory (GCRO), ERA5 Reanalysis
- **Funding**: [Your funding sources]
- **Contributors**: [List key contributors]
- **Inspiration**: Built on best practices from the scientific Python ecosystem

## 📞 Support

- **Documentation**: [https://enbel.github.io/climate-health-analysis](https://enbel.github.io/climate-health-analysis)
- **Issues**: [GitHub Issues](https://github.com/enbel/climate-health-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/enbel/climate-health-analysis/discussions)
- **Email**: enbel@example.com

## 📈 Project Status

- **Current Version**: 1.0.0
- **Development Status**: Active
- **Maintenance**: Actively maintained
- **Python Support**: 3.9, 3.10, 3.11
- **Platform Support**: Linux, macOS, Windows

---

**Built with ❤️ for climate-health research by the ENBEL team**