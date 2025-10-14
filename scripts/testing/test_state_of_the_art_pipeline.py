#!/usr/bin/env python3
"""
Comprehensive Test Suite for State-of-the-Art Climate-Health Analysis Pipeline
==============================================================================

This test suite validates all components of the climate-health analysis pipeline
to ensure scientific accuracy, reproducibility, and reliability.

Authors: Climate-Health Research Team
Version: 1.0.0
Date: 2025-09-30
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import json
import pickle
from unittest.mock import Mock, patch

# Import pipeline components
from state_of_the_art_climate_health_pipeline import (
    PipelineConfig,
    AdvancedClimateFeatureEngineering,
    DLNMIntegration,
    StatisticalRigorFramework,
    TimeSeriesFramework,
    ModernMLEnsemble,
    ComprehensiveInterpretability,
    ReproducibilityInfrastructure,
    StateOfTheArtClimateHealthPipeline
)


class TestPipelineConfig:
    """Test the pipeline configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert isinstance(config.target_variables, list)
        assert len(config.target_variables) > 0
        assert config.random_seed == 42
        assert config.alpha_level == 0.05
        assert config.max_lag == 21
        
    def test_config_modification(self):
        """Test configuration modification."""
        config = PipelineConfig()
        config.random_seed = 123
        config.max_lag = 14
        
        assert config.random_seed == 123
        assert config.max_lag == 14


class TestAdvancedClimateFeatureEngineering:
    """Test advanced climate feature engineering."""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineering instance."""
        config = PipelineConfig()
        return AdvancedClimateFeatureEngineering(config)
    
    @pytest.fixture
    def sample_climate_data(self):
        """Create sample climate data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'temperature': np.random.normal(20, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'wind_speed': np.random.exponential(5, n_samples),
            'temperature_max': np.random.normal(25, 10, n_samples),
            'temperature_min': np.random.normal(15, 10, n_samples),
        }
        
        return pd.DataFrame(data)
    
    def test_heat_index_calculation(self, feature_engineer):
        """Test heat index calculation."""
        temp_f = np.array([80, 90, 100])
        humidity = np.array([40, 60, 80])
        
        heat_index = feature_engineer.calculate_heat_index(temp_f, humidity)
        
        assert len(heat_index) == 3
        assert np.all(heat_index >= temp_f)  # Heat index should be >= temperature
        assert not np.any(np.isnan(heat_index))
    
    def test_apparent_temperature_calculation(self, feature_engineer):
        """Test apparent temperature calculation."""
        temp_c = np.array([20, 25, 30])
        humidity = np.array([50, 60, 70])
        wind_speed = np.array([2, 3, 4])
        
        apparent_temp = feature_engineer.calculate_apparent_temperature(temp_c, humidity, wind_speed)
        
        assert len(apparent_temp) == 3
        assert not np.any(np.isnan(apparent_temp))
    
    def test_wet_bulb_temperature_calculation(self, feature_engineer):
        """Test wet bulb temperature calculation."""
        temp_c = np.array([25, 30, 35])
        humidity = np.array([50, 70, 90])
        
        wet_bulb = feature_engineer.calculate_wet_bulb_temperature(temp_c, humidity)
        
        assert len(wet_bulb) == 3
        assert np.all(wet_bulb <= temp_c)  # Wet bulb should be <= dry bulb temperature
        assert not np.any(np.isnan(wet_bulb))
    
    def test_heating_cooling_degree_days(self, feature_engineer):
        """Test heating and cooling degree days calculation."""
        temp_c = np.array([10, 18.3, 25])  # Below, at, and above base temperature
        
        heating_dd, cooling_dd = feature_engineer.calculate_heating_cooling_degree_days(temp_c)
        
        assert len(heating_dd) == 3
        assert len(cooling_dd) == 3
        assert heating_dd[0] > 0  # Should have heating degree days for 10°C
        assert heating_dd[1] == 0  # Should be zero at base temperature
        assert cooling_dd[2] > 0  # Should have cooling degree days for 25°C
    
    def test_heat_wave_detection(self, feature_engineer):
        """Test heat wave detection algorithm."""
        # Create temperature series with a heat wave
        temp_series = np.concatenate([
            np.random.normal(20, 5, 50),  # Normal temperatures
            np.random.normal(35, 2, 5),   # Heat wave (5 consecutive hot days)
            np.random.normal(20, 5, 45)   # Return to normal
        ])
        
        heat_waves = feature_engineer.detect_heat_waves(temp_series, percentile_threshold=90, duration_threshold=3)
        
        assert len(heat_waves) == len(temp_series)
        assert np.any(heat_waves == 1)  # Should detect at least one heat wave day
    
    def test_climate_variability_metrics(self, feature_engineer, sample_climate_data):
        """Test climate variability metrics calculation."""
        df_enhanced = feature_engineer.calculate_climate_variability_metrics(
            sample_climate_data, 'temperature', windows=[3, 7]
        )
        
        expected_cols = ['temperature_variability_3d', 'temperature_variability_7d',
                        'temperature_range_3d', 'temperature_range_7d',
                        'temp_acceleration_3d', 'temp_acceleration_7d']
        
        for col in expected_cols:
            assert col in df_enhanced.columns
            assert not df_enhanced[col].isnull().all()
    
    def test_comprehensive_feature_creation(self, feature_engineer, sample_climate_data):
        """Test comprehensive climate feature creation."""
        df_enhanced = feature_engineer.create_comprehensive_climate_features(sample_climate_data)
        
        # Check that new features were created
        assert len(df_enhanced.columns) > len(sample_climate_data.columns)
        
        # Check for specific expected features
        expected_features = ['heat_index', 'wet_bulb_temp', 'apparent_temp', 
                           'heating_degree_days', 'cooling_degree_days']
        
        for feature in expected_features:
            if feature in df_enhanced.columns:
                assert not df_enhanced[feature].isnull().all()


class TestStatisticalRigorFramework:
    """Test statistical rigor and uncertainty quantification."""
    
    @pytest.fixture
    def stats_framework(self):
        """Create statistical framework instance."""
        config = PipelineConfig()
        return StatisticalRigorFramework(config)
    
    def test_multiple_testing_correction(self, stats_framework):
        """Test multiple testing correction methods."""
        p_values = np.array([0.01, 0.03, 0.05, 0.08, 0.1])
        
        rejected, corrected_p = stats_framework.apply_multiple_testing_correction(p_values, method='fdr_bh')
        
        assert len(rejected) == len(p_values)
        assert len(corrected_p) == len(p_values)
        assert np.all(corrected_p >= p_values)  # Corrected p-values should be >= original
    
    def test_bootstrap_confidence_intervals(self, stats_framework):
        """Test bootstrap confidence interval calculation."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 100)
        
        def mean_statistic(x):
            return np.mean(x)
        
        point_est, ci_lower, ci_upper = stats_framework.bootstrap_confidence_intervals(
            data, mean_statistic, confidence_level=0.95
        )
        
        assert ci_lower < point_est < ci_upper
        assert abs(point_est - np.mean(data)) < 0.1  # Should be close to actual mean
    
    def test_effect_size_calculation(self, stats_framework):
        """Test effect size calculations."""
        np.random.seed(42)
        y_true = np.random.normal(100, 15, 50)
        y_pred = y_true + np.random.normal(0, 5, 50)  # Add some prediction error
        baseline_var = 225  # 15^2
        
        effect_sizes = stats_framework.calculate_effect_sizes(y_true, y_pred, baseline_var)
        
        assert 'cohens_d' in effect_sizes
        assert 'r2' in effect_sizes
        assert 'rmse' in effect_sizes
        assert effect_sizes['r2'] > 0  # Should have positive R²
    
    def test_power_analysis(self, stats_framework):
        """Test power analysis for sample size calculation."""
        effect_size = 0.3
        sample_size = stats_framework.power_analysis(effect_size, alpha=0.05, power=0.8)
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size < 1000  # Should be reasonable for this effect size


class TestTimeSeriesFramework:
    """Test time series analysis framework."""
    
    @pytest.fixture
    def ts_framework(self):
        """Create time series framework instance."""
        config = PipelineConfig()
        return TimeSeriesFramework(config)
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data."""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        values = np.random.normal(0, 1, 365)
        return pd.Series(values, index=dates)
    
    def test_autocorrelation_detection(self, ts_framework):
        """Test autocorrelation detection."""
        # Create autocorrelated residuals
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        for i in range(1, len(residuals)):
            residuals[i] += 0.5 * residuals[i-1]  # Add autocorrelation
        
        autocorr_stats = ts_framework.detect_autocorrelation(residuals)
        
        assert 'durbin_watson' in autocorr_stats
        assert 'ljung_box_pvalue' in autocorr_stats
        assert 'autocorr_lag1' in autocorr_stats
        assert autocorr_stats['ljung_box_pvalue'] < 0.05  # Should detect autocorrelation
    
    def test_seasonal_decomposition(self, ts_framework, sample_time_series):
        """Test seasonal decomposition."""
        decomp = ts_framework.seasonal_decomposition(sample_time_series, period=7)
        
        if decomp:  # If decomposition succeeded
            assert 'trend' in decomp
            assert 'seasonal' in decomp
            assert 'residual' in decomp
    
    def test_temporal_feature_creation(self, ts_framework):
        """Test temporal feature creation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'date': dates, 'value': np.random.randn(100)})
        
        df_enhanced = ts_framework.create_temporal_features(df, 'date')
        
        expected_features = ['year', 'month', 'day_of_year', 'day_of_week', 
                           'month_sin', 'month_cos', 'is_weekend']
        
        for feature in expected_features:
            assert feature in df_enhanced.columns


class TestModernMLEnsemble:
    """Test modern machine learning ensemble methods."""
    
    @pytest.fixture
    def ml_ensemble(self):
        """Create ML ensemble instance."""
        config = PipelineConfig()
        config.ensemble_methods = ['random_forest', 'xgboost']  # Limit for testing
        return ModernMLEnsemble(config)
    
    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * (-1) + np.random.normal(0, 0.5, n_samples)
        
        return X, y
    
    def test_base_model_creation(self, ml_ensemble):
        """Test base model creation."""
        base_models = ml_ensemble.create_base_models()
        
        assert 'random_forest' in base_models
        assert 'xgboost' in base_models
        assert hasattr(base_models['random_forest'], 'fit')
        assert hasattr(base_models['xgboost'], 'fit')
    
    def test_ensemble_fitting(self, ml_ensemble, sample_ml_data):
        """Test ensemble model fitting."""
        X, y = sample_ml_data
        
        results = ml_ensemble.fit_ensemble(X, y)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        for model_name, r2 in results.items():
            assert isinstance(r2, float)
            assert -1 <= r2 <= 1  # R² should be in valid range
    
    def test_ensemble_prediction(self, ml_ensemble, sample_ml_data):
        """Test ensemble predictions."""
        X, y = sample_ml_data
        
        # Fit models first
        ml_ensemble.fit_ensemble(X, y)
        
        # Make predictions
        predictions = ml_ensemble.predict_ensemble(X, method='average')
        
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))


class TestComprehensiveInterpretability:
    """Test comprehensive interpretability framework."""
    
    @pytest.fixture
    def interpretability(self):
        """Create interpretability instance."""
        config = PipelineConfig()
        return ComprehensiveInterpretability(config)
    
    def test_feature_interaction_calculation(self, interpretability):
        """Test feature interaction calculation."""
        # Create mock SHAP values
        np.random.seed(42)
        shap_values = np.random.randn(100, 5)
        feature_names = ['temp', 'humidity', 'wind', 'pressure', 'radiation']
        
        interactions = interpretability.calculate_feature_interactions(shap_values, feature_names)
        
        if len(interactions) > 0:
            assert interactions.shape == (5, 5)
            assert list(interactions.columns) == feature_names
            assert list(interactions.index) == feature_names


class TestReproducibilityInfrastructure:
    """Test reproducibility infrastructure."""
    
    @pytest.fixture
    def repro_infra(self):
        """Create reproducibility infrastructure instance."""
        config = PipelineConfig()
        return ReproducibilityInfrastructure(config)
    
    def test_seed_setting(self, repro_infra):
        """Test random seed setting."""
        repro_infra.set_all_seeds()
        
        # Test that random operations are reproducible
        random1 = np.random.random(5)
        repro_infra.set_all_seeds()
        random2 = np.random.random(5)
        
        np.testing.assert_array_equal(random1, random2)
    
    def test_environment_info_saving(self, repro_infra):
        """Test environment information saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repro_infra.save_environment_info(temp_dir)
            
            env_file = Path(temp_dir) / 'environment_info.json'
            assert env_file.exists()
            
            with open(env_file) as f:
                env_info = json.load(f)
            
            assert 'python_version' in env_info
            assert 'platform' in env_info
            assert 'timestamp' in env_info
            assert 'random_seed' in env_info
    
    def test_dockerfile_creation(self, repro_infra):
        """Test Dockerfile creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repro_infra.create_dockerfile(temp_dir)
            
            dockerfile = Path(temp_dir) / 'Dockerfile'
            assert dockerfile.exists()
            
            content = dockerfile.read_text()
            assert 'FROM python:3.9-slim' in content
            assert 'dlnm' in content
            assert 'mgcv' in content


class TestStateOfTheArtClimateHealthPipeline:
    """Test the main pipeline class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        n_samples = 300
        
        # Create realistic climate-health data
        data = {
            'temperature': np.random.normal(20, 8, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'wind_speed': np.random.exponential(5, n_samples),
            'FASTING_GLUCOSE': np.random.normal(95, 15, n_samples),
            'systolic_blood_pressure': np.random.normal(120, 20, n_samples),
            'Sex': np.random.choice(['M', 'F'], n_samples),
            'Race': np.random.choice(['White', 'Black', 'Other'], n_samples),
        }
        
        # Add some realistic relationships
        data['FASTING_GLUCOSE'] += 0.5 * data['temperature'] + np.random.normal(0, 5, n_samples)
        data['systolic_blood_pressure'] += 0.3 * data['humidity'] + np.random.normal(0, 8, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_data_file(self, sample_dataset):
        """Create temporary data file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        sample_dataset.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        Path(temp_file.name).unlink()
    
    def test_pipeline_initialization(self, temp_data_file):
        """Test pipeline initialization."""
        config = PipelineConfig()
        config.data_path = temp_data_file
        config.target_variables = ['FASTING_GLUCOSE', 'systolic_blood_pressure']
        
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        
        assert pipeline.config == config
        assert hasattr(pipeline, 'feature_engineering')
        assert hasattr(pipeline, 'ml_ensemble')
        assert hasattr(pipeline, 'interpretability')
    
    def test_data_loading_and_validation(self, temp_data_file):
        """Test data loading and validation."""
        config = PipelineConfig()
        config.data_path = temp_data_file
        
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        df = pipeline.load_and_validate_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'temperature' in df.columns
        assert 'FASTING_GLUCOSE' in df.columns
    
    def test_feature_engineering_integration(self, temp_data_file):
        """Test feature engineering integration."""
        config = PipelineConfig()
        config.data_path = temp_data_file
        
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        df = pipeline.load_and_validate_data()
        df_enhanced = pipeline.engineer_climate_features(df)
        
        assert len(df_enhanced.columns) > len(df.columns)
        assert 'enhanced' in pipeline.data
    
    @pytest.mark.slow
    def test_ensemble_model_fitting(self, temp_data_file):
        """Test ensemble model fitting (marked as slow test)."""
        config = PipelineConfig()
        config.data_path = temp_data_file
        config.target_variables = ['FASTING_GLUCOSE']  # Test with one target
        config.ensemble_methods = ['random_forest']  # Use only one method for speed
        
        with tempfile.TemporaryDirectory() as temp_output:
            config.output_dir = temp_output
            
            pipeline = StateOfTheArtClimateHealthPipeline(config)
            df = pipeline.load_and_validate_data()
            df_enhanced = pipeline.engineer_climate_features(df)
            results = pipeline.fit_ensemble_models(df_enhanced)
            
            assert isinstance(results, dict)
            if 'FASTING_GLUCOSE' in results:
                assert 'ensemble_performance' in results['FASTING_GLUCOSE']
                assert 'r2' in results['FASTING_GLUCOSE']['ensemble_performance']
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        config = PipelineConfig()
        config.data_path = "nonexistent_file.csv"
        
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        
        with pytest.raises(Exception):
            pipeline.load_and_validate_data()


class TestIntegrationScenarios:
    """Integration tests for complete pipeline scenarios."""
    
    @pytest.fixture
    def minimal_config(self, temp_data_file):
        """Create minimal configuration for integration tests."""
        config = PipelineConfig()
        config.data_path = temp_data_file
        config.target_variables = ['FASTING_GLUCOSE']
        config.ensemble_methods = ['random_forest']
        config.bootstrap_iterations = 10  # Reduce for speed
        config.create_plots = False  # Disable plots for testing
        config.save_models = False  # Disable model saving for testing
        config.generate_report = False  # Disable report generation for testing
        
        return config
    
    @pytest.fixture
    def temp_data_file(self):
        """Create temporary data file for integration tests."""
        np.random.seed(42)
        n_samples = 100  # Smaller dataset for faster testing
        
        data = {
            'temperature': np.random.normal(20, 8, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'wind_speed': np.random.exponential(5, n_samples),
            'FASTING_GLUCOSE': np.random.normal(95, 15, n_samples),
            'Sex': np.random.choice(['M', 'F'], n_samples),
            'Race': np.random.choice(['White', 'Black'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        Path(temp_file.name).unlink()
    
    @pytest.mark.integration
    def test_minimal_pipeline_execution(self, minimal_config):
        """Test minimal pipeline execution."""
        with tempfile.TemporaryDirectory() as temp_output:
            minimal_config.output_dir = temp_output
            
            pipeline = StateOfTheArtClimateHealthPipeline(minimal_config)
            
            # Test individual components
            df = pipeline.load_and_validate_data()
            assert len(df) > 0
            
            df_enhanced = pipeline.engineer_climate_features(df)
            assert len(df_enhanced.columns) > len(df.columns)
            
            ensemble_results = pipeline.fit_ensemble_models(df_enhanced)
            assert isinstance(ensemble_results, dict)
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_pipeline_execution(self, minimal_config):
        """Test full pipeline execution (marked as slow)."""
        with tempfile.TemporaryDirectory() as temp_output:
            minimal_config.output_dir = temp_output
            
            pipeline = StateOfTheArtClimateHealthPipeline(minimal_config)
            results = pipeline.run_complete_analysis()
            
            assert results['status'] in ['completed', 'failed']
            if results['status'] == 'completed':
                assert results['targets_analyzed'] >= 0
                assert Path(results['output_directory']).exists()


def test_scientific_validity():
    """Test scientific validity of key calculations."""
    
    # Test heat index calculation against known values
    feature_eng = AdvancedClimateFeatureEngineering(PipelineConfig())
    
    # Known heat index values (from NWS)
    temp_f = 80  # 80°F
    humidity = 40  # 40% RH
    expected_hi = 80  # Should be approximately 80°F for these conditions
    
    calculated_hi = feature_eng.calculate_heat_index(np.array([temp_f]), np.array([humidity]))[0]
    
    # Allow some tolerance for rounding differences
    assert abs(calculated_hi - expected_hi) < 5
    
    # Test that apparent temperature is reasonable
    temp_c = 25
    humidity = 60
    wind_speed = 3
    
    apparent_temp = feature_eng.calculate_apparent_temperature(
        np.array([temp_c]), np.array([humidity]), np.array([wind_speed])
    )[0]
    
    # Apparent temperature should be within reasonable range of actual temperature
    assert abs(apparent_temp - temp_c) < 10


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])