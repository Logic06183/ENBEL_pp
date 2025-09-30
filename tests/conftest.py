"""
Pytest Configuration and Fixtures
=================================

Shared test configuration and fixtures for the ENBEL test suite.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enbel_pp.config import set_reproducible_environment


@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """Set up reproducible test environment for all tests."""
    set_reproducible_environment(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def synthetic_climate_data():
    """Generate synthetic climate-health data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Climate features
    data = {
        'temperature_daily': np.random.normal(25, 5, n_samples),
        'temperature_7d': np.random.normal(25, 3, n_samples),
        'humidity_daily': np.random.normal(60, 15, n_samples),
        'humidity_7d': np.random.normal(60, 10, n_samples),
        'heat_index': np.random.normal(30, 8, n_samples),
    }
    
    # Add lag features
    for lag in [1, 3, 7]:
        data[f'temperature_lag{lag}'] = np.random.normal(25, 5, n_samples)
        data[f'humidity_lag{lag}'] = np.random.normal(60, 15, n_samples)
    
    # Demographic features
    data['Sex'] = np.random.choice(['Male', 'Female'], n_samples)
    data['Race'] = np.random.choice(['Black', 'White', 'Coloured', 'Indian'], n_samples)
    data['age'] = np.random.normal(45, 15, n_samples)
    
    # Health biomarkers with realistic climate relationships
    temp_effect = data['temperature_daily'] * 0.5
    humidity_effect = data['humidity_daily'] * -0.2
    noise = np.random.normal(0, 5, n_samples)
    
    data['systolic blood pressure'] = 120 + temp_effect + humidity_effect + noise
    data['FASTING GLUCOSE'] = 95 + temp_effect * 0.8 + noise * 0.3
    data['CD4 cell count (cells/µL)'] = 500 + temp_effect * -2 + noise * 10
    data['Hemoglobin (g/dL)'] = 13 + temp_effect * 0.1 + noise * 0.2
    
    # Add some missing values
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    df.loc[missing_indices, 'systolic blood pressure'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=30, replace=False)
    df.loc[missing_indices, 'FASTING GLUCOSE'] = np.nan
    
    return df


@pytest.fixture
def synthetic_climate_data_small():
    """Generate small synthetic dataset for quick tests."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'temperature_daily': np.random.normal(25, 5, n_samples),
        'humidity_daily': np.random.normal(60, 15, n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.normal(45, 15, n_samples),
    }
    
    # Simple biomarker
    temp_effect = data['temperature_daily'] * 0.5
    noise = np.random.normal(0, 2, n_samples)
    data['systolic blood pressure'] = 120 + temp_effect + noise
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = MagicMock()
    config.get_biomarkers.return_value = [
        'systolic blood pressure',
        'FASTING GLUCOSE',
        'CD4 cell count (cells/µL)'
    ]
    config.get_ml_config.return_value = {
        'cv_folds': 3,
        'test_size': 0.2,
        'alpha_bonferroni': 0.01
    }
    config.get_model_config.return_value = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    return config


@pytest.fixture
def sample_biomarker_data():
    """Create sample biomarker validation data."""
    return pd.DataFrame({
        'CD4 cell count (cells/µL)': [500, 600, np.nan, 700, 800],
        'FASTING GLUCOSE': [90, 95, 100, np.nan, 110],
        'systolic blood pressure': [120, 130, 140, 150, np.nan],
        'temperature': [20, 25, 30, 35, 40],
        'other_column': [1, 2, 3, 4, 5]
    })


@pytest.fixture
def test_data_file(temp_dir, synthetic_climate_data):
    """Create a test data file."""
    data_file = temp_dir / "test_data.csv"
    synthetic_climate_data.to_csv(data_file, index=False)
    return data_file


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow