"""
ENBEL Climate-Health Analysis Utilities
=======================================

Comprehensive utilities for data validation, configuration management,
and machine learning pipeline components.

Modules:
--------
- config: Configuration management and reproducibility settings
- data_validation: Data integrity validation and quality checking
- ml_utils: Machine learning utilities and statistical corrections

Author: ENBEL Project Team
"""

from .config import ENBELConfig, get_config, set_reproducible_environment
from .data_validation import (
    validate_file_exists, 
    validate_data_files, 
    validate_biomarker_data,
    validate_climate_data,
    run_full_validation,
    DataValidationError
)
from .ml_utils import (
    prepare_features_safely,
    train_model_with_cv,
    calculate_shap_values,
    apply_multiple_testing_correction,
    evaluate_model_performance,
    save_model_results,
    MLPipelineError
)

__all__ = [
    # Config
    'ENBELConfig',
    'get_config', 
    'set_reproducible_environment',
    
    # Data validation
    'validate_file_exists',
    'validate_data_files',
    'validate_biomarker_data', 
    'validate_climate_data',
    'run_full_validation',
    'DataValidationError',
    
    # ML utilities
    'prepare_features_safely',
    'train_model_with_cv',
    'calculate_shap_values',
    'apply_multiple_testing_correction',
    'evaluate_model_performance',
    'save_model_results',
    'MLPipelineError'
]

__version__ = "1.0.0"