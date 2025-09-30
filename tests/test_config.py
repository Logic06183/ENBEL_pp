#!/usr/bin/env python3
"""
Test Configuration Management
=============================

Tests for ENBEL configuration system including:
- Configuration loading and validation
- Environment setup
- Path management
- Parameter validation

Author: ENBEL Project Team
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import numpy as np

from enbel_pp.config import ENBELConfig, get_config, set_reproducible_environment


class TestENBELConfig:
    """Test ENBEL configuration management."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = ENBELConfig()
        
        # Check basic properties
        assert isinstance(config.config, dict)
        assert 'random_state' in config.config
        assert 'biomarkers' in config.config
        assert 'ml_settings' in config.config
        assert 'paths' in config.config
        
        # Check default values
        assert config.config['random_state'] == 42
        assert isinstance(config.config['biomarkers'], list)
        assert len(config.config['biomarkers']) > 0
    
    def test_biomarkers_configuration(self):
        """Test biomarker configuration."""
        config = ENBELConfig()
        biomarkers = config.get_biomarkers()
        
        assert isinstance(biomarkers, list)
        assert len(biomarkers) > 0
        
        # Check for expected biomarkers
        expected_biomarkers = [
            'CD4 cell count (cells/µL)',
            'systolic blood pressure',
            'FASTING GLUCOSE'
        ]
        
        for biomarker in expected_biomarkers:
            assert biomarker in biomarkers
    
    def test_ml_configuration(self):
        """Test ML configuration."""
        config = ENBELConfig()
        ml_config = config.get_ml_config()
        
        assert isinstance(ml_config, dict)
        assert 'cv_folds' in ml_config
        assert 'test_size' in ml_config
        assert 'alpha_bonferroni' in ml_config
        
        # Check reasonable values
        assert ml_config['cv_folds'] >= 3
        assert 0 < ml_config['test_size'] < 1
        assert ml_config['alpha_bonferroni'] > 0
    
    def test_model_configuration(self):
        """Test model-specific configuration."""
        config = ENBELConfig()
        
        # Test Random Forest config
        rf_config = config.get_model_config('random_forest')
        assert isinstance(rf_config, dict)
        assert 'n_estimators' in rf_config
        assert 'max_depth' in rf_config
        assert 'random_state' in rf_config
        
        # Test XGBoost config
        xgb_config = config.get_model_config('xgboost')
        assert isinstance(xgb_config, dict)
        assert 'learning_rate' in xgb_config
        assert 'max_depth' in xgb_config
        assert 'random_state' in xgb_config
    
    def test_custom_configuration_loading(self, temp_dir):
        """Test loading custom configuration from file."""
        # Create custom config file
        custom_config = {
            'random_state': 123,
            'ml_settings': {
                'cv_folds': 8,
                'test_size': 0.25
            },
            'biomarkers': ['test_biomarker']
        }
        
        config_file = temp_dir / "custom_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(custom_config, f)
        
        # Load custom configuration
        config = ENBELConfig(str(config_file))
        
        # Check custom values were loaded
        assert config.config['random_state'] == 123
        assert config.config['ml_settings']['cv_folds'] == 8
        assert config.config['ml_settings']['test_size'] == 0.25
        assert 'test_biomarker' in config.config['biomarkers']
    
    def test_bonferroni_correction_calculation(self):
        """Test automatic Bonferroni correction calculation."""
        config = ENBELConfig()
        
        n_biomarkers = len(config.get_biomarkers())
        expected_alpha = 0.05 / n_biomarkers
        
        ml_config = config.get_ml_config()
        assert abs(ml_config['alpha_bonferroni'] - expected_alpha) < 1e-10
    
    def test_path_management(self):
        """Test path management functionality."""
        config = ENBELConfig()
        
        # Test path methods exist and return Path objects
        for output_type in ['results', 'models', 'figures', 'logs']:
            output_path = config.get_output_path(output_type, 'test_file.txt')
            assert isinstance(output_path, Path)
            assert output_path.name == 'test_file.txt'
    
    def test_configuration_saving(self, temp_dir):
        """Test configuration saving to file."""
        config = ENBELConfig()
        
        output_file = temp_dir / "saved_config.yaml"
        config.save_config(str(output_file))
        
        # Check file was created
        assert output_file.exists()
        
        # Check file content
        with open(output_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert isinstance(saved_config, dict)
        assert 'random_state' in saved_config
        assert 'biomarkers' in saved_config
    
    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        config = ENBELConfig()
        
        with pytest.raises(ValueError, match="Unknown model type"):
            config.get_model_config('invalid_model')
    
    def test_invalid_output_type(self):
        """Test handling of invalid output types."""
        config = ENBELConfig()
        
        with pytest.raises(ValueError, match="Unknown output type"):
            config.get_output_path('invalid_type', 'test.txt')


class TestReproducibleEnvironment:
    """Test reproducible environment setup."""
    
    def test_set_reproducible_environment(self):
        """Test reproducible environment setup."""
        # Set environment with specific seed
        set_reproducible_environment(123)
        
        # Generate random numbers
        random1 = np.random.random(5)
        
        # Reset with same seed
        set_reproducible_environment(123)
        random2 = np.random.random(5)
        
        # Should be identical
        np.testing.assert_array_equal(random1, random2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        # Set environment with seed 1
        set_reproducible_environment(1)
        random1 = np.random.random(5)
        
        # Set environment with seed 2
        set_reproducible_environment(2)
        random2 = np.random.random(5)
        
        # Should be different
        assert not np.array_equal(random1, random2)


class TestGlobalConfiguration:
    """Test global configuration management."""
    
    def test_get_config(self):
        """Test global configuration access."""
        config = get_config()
        
        assert isinstance(config, ENBELConfig)
        assert hasattr(config, 'config')
        assert hasattr(config, 'get_biomarkers')
    
    def test_config_persistence(self):
        """Test that global config persists across calls."""
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same instance
        assert config1 is config2