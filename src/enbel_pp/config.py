#!/usr/bin/env python3
"""
Configuration and Setup Utilities for ENBEL Climate-Health Analysis
===================================================================

Centralized configuration management, path handling, and reproducibility
settings for the entire analysis pipeline.

Author: ENBEL Project Team
"""

import os
import random
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ENBELConfig:
    """
    Centralized configuration management for ENBEL analysis pipeline.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to YAML configuration file
        """
        # Set base directory relative to config.py location
        self.BASE_DIR = Path(__file__).parent.parent
        
        # Default configuration
        self.config = {
            'random_state': 42,
            'paths': {
                'data_dir': self.BASE_DIR,
                'results_dir': self.BASE_DIR / 'results',
                'models_dir': self.BASE_DIR / 'trained_models',
                'figures_dir': self.BASE_DIR / 'figures',
                'logs_dir': self.BASE_DIR / 'logs'
            },
            'data_files': {
                'full_dataset': 'FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv',
                'clinical_imputed': 'CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv',
                'clinical_original': 'CLINICAL_WITH_IMPUTED_SOCIOECONOMIC.csv'
            },
            'biomarkers': [
                'CD4 cell count (cells/µL)',
                'FASTING GLUCOSE', 
                'FASTING LDL',
                'FASTING TOTAL CHOLESTEROL',
                'FASTING HDL',
                'FASTING TRIGLYCERIDES',
                'Creatinine (mg/dL)',
                'ALT (U/L)',
                'AST (U/L)',
                'Hemoglobin (g/dL)',
                'Hematocrit (%)',
                'systolic blood pressure',
                'diastolic blood pressure'
            ],
            'ml_settings': {
                'cv_folds': 5,
                'test_size': 0.2,
                'n_repeats': 10,
                'max_features': 100,
                'alpha_bonferroni': None  # Will be calculated as 0.05/n_biomarkers
            },
            'random_forest': {
                'n_estimators': 250,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'n_jobs': -1
            },
            'xgboost': {
                'learning_rate': 0.05,
                'max_depth': 8,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1
            },
            'imputation': {
                'k_neighbors': 10,
                'max_distance_km': 15,
                'demographic_weight': 0.6,
                'spatial_weight': 0.4,
                'min_matches': 3,
                'min_similarity_score': 0.3
            }
        }
        
        # Load custom config if provided
        if config_file:
            self.load_config(config_file)
            
        # Calculate derived settings
        self._calculate_derived_settings()
        
        # Set up directories
        self.setup_directories()
        
        # Set random seeds
        self.set_random_seeds()
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_file : str
            Path to YAML configuration file
        """
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                # Merge with defaults
                self._deep_update(self.config, custom_config)
                logger.info(f"Loaded configuration from: {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update nested dictionary.
        
        Parameters:
        -----------
        base_dict : Dict
            Base dictionary to update
        update_dict : Dict
            Update dictionary with new values
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _calculate_derived_settings(self) -> None:
        """Calculate settings derived from base configuration."""
        # Calculate Bonferroni correction
        n_biomarkers = len(self.config['biomarkers'])
        self.config['ml_settings']['alpha_bonferroni'] = 0.05 / n_biomarkers
        
        # Convert string paths to Path objects
        for key, path in self.config['paths'].items():
            self.config['paths'][key] = Path(path)
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_name, dir_path in self.config['paths'].items():
            if dir_name != 'data_dir':  # Don't create data_dir
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
    
    def set_random_seeds(self, seed: Optional[int] = None) -> None:
        """
        Set random seeds for reproducibility.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed to use. If None, uses configured random_state.
        """
        if seed is None:
            seed = self.config['random_state']
            
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set seeds for ML libraries
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
            
        logger.info(f"Set random seeds to: {seed}")
    
    def get_data_path(self, data_type: str) -> Path:
        """
        Get full path to a data file.
        
        Parameters:
        -----------
        data_type : str
            Type of data file ('full_dataset', 'clinical_imputed', etc.)
            
        Returns:
        --------
        Path
            Full path to the data file
        """
        if data_type not in self.config['data_files']:
            raise ValueError(f"Unknown data type: {data_type}")
        
        filename = self.config['data_files'][data_type]
        return self.config['paths']['data_dir'] / filename
    
    def get_output_path(self, output_type: str, filename: str) -> Path:
        """
        Get full path for an output file.
        
        Parameters:
        -----------
        output_type : str
            Type of output ('results', 'models', 'figures', 'logs')
        filename : str
            Name of the output file
            
        Returns:
        --------
        Path
            Full path for the output file
        """
        if f'{output_type}_dir' not in self.config['paths']:
            raise ValueError(f"Unknown output type: {output_type}")
        
        return self.config['paths'][f'{output_type}_dir'] / filename
    
    def get_biomarkers(self) -> list:
        """Get list of biomarkers for analysis."""
        return self.config['biomarkers'].copy()
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration."""
        return self.config['ml_settings'].copy()
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest', 'xgboost', etc.)
            
        Returns:
        --------
        Dict[str, Any]
            Model configuration parameters
        """
        if model_type not in self.config:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.config[model_type].copy()
        # Add random state
        config['random_state'] = self.config['random_state']
        return config
    
    def save_config(self, output_file: str) -> None:
        """
        Save current configuration to YAML file.
        
        Parameters:
        -----------
        output_file : str
            Path for output configuration file
        """
        # Convert Path objects to strings for YAML serialization
        config_to_save = {}
        for key, value in self.config.items():
            if key == 'paths':
                config_to_save[key] = {k: str(v) for k, v in value.items()}
            else:
                config_to_save[key] = value
        
        with open(output_file, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_file}")

# Global configuration instance
config = ENBELConfig()

def get_config() -> ENBELConfig:
    """Get global configuration instance."""
    return config

def set_reproducible_environment(seed: int = 42) -> None:
    """
    Set up reproducible environment for the entire pipeline.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    """
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set pandas options for consistency
    import pandas as pd
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    
    # Disable warnings that clutter output
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    logger.info(f"Reproducible environment set with seed: {seed}")

if __name__ == "__main__":
    # Test configuration
    config = ENBELConfig()
    print("Configuration loaded successfully!")
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Number of biomarkers: {len(config.get_biomarkers())}")
    print(f"Bonferroni alpha: {config.config['ml_settings']['alpha_bonferroni']:.6f}")
    
    # Save example configuration
    config.save_config("config_example.yaml")