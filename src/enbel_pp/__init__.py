"""
ENBEL Climate-Health Analysis Core Package
==========================================

Core functionality for climate-health analysis including:
- Configuration management
- Data validation utilities
- Machine learning utilities
- Pipeline orchestration

This is the main entry point for the ENBEL analysis package.
"""

from .config import ENBELConfig, get_config, set_reproducible_environment
from .data_validation import (
    validate_file_exists,
    validate_dataframe_schema,
    validate_biomarker_data,
    DataValidationError
)
from .ml_utils import (
    prepare_features_safely,
    train_model_with_cv,
    apply_multiple_testing_correction,
    evaluate_model_performance,
    MLPipelineError
)
from .pipeline import ClimateHealthPipeline

__all__ = [
    # Configuration
    'ENBELConfig',
    'get_config',
    'set_reproducible_environment',
    
    # Data validation
    'validate_file_exists',
    'validate_dataframe_schema', 
    'validate_biomarker_data',
    'DataValidationError',
    
    # ML utilities
    'prepare_features_safely',
    'train_model_with_cv',
    'apply_multiple_testing_correction',
    'evaluate_model_performance',
    'MLPipelineError',
    
    # Pipeline
    'ClimateHealthPipeline',
]