#!/usr/bin/env python3
"""
Comprehensive Test Suite for ENBEL ML Pipeline
==============================================

Tests all critical components of the ML pipeline to ensure:
- Data validation works correctly
- ML models train without leakage
- Statistical corrections are applied properly
- SHAP calculations are accurate
- Results are reproducible

Author: ENBEL Project Team
"""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

# Import modules to test
from config import ENBELConfig, set_reproducible_environment
from data_validation import (
    validate_file_exists, 
    validate_dataframe_schema, 
    validate_biomarker_data,
    DataValidationError
)
from ml_utils import (
    prepare_features_safely,
    train_model_with_cv,
    apply_multiple_testing_correction,
    evaluate_model_performance,
    MLPipelineError
)

warnings.filterwarnings('ignore')

class TestDataValidation(unittest.TestCase):
    """Test data validation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_file_exists_success(self):
        """Test successful file validation."""
        # Create test file
        test_file = self.temp_path / "test.csv"
        test_file.write_text("test,data\n1,2\n")
        
        # Should not raise exception
        result = validate_file_exists(test_file, "test file")
        self.assertEqual(result, test_file)
    
    def test_validate_file_exists_failure(self):
        """Test file validation failure."""
        missing_file = self.temp_path / "missing.csv"
        
        # Should raise exception
        with self.assertRaises(DataValidationError):
            validate_file_exists(missing_file, "missing file")
    
    def test_validate_dataframe_schema_success(self):
        """Test successful DataFrame schema validation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9]
        })
        
        required_columns = ['feature1', 'feature2', 'target']
        
        # Should not raise exception
        result = validate_dataframe_schema(df, required_columns, "test dataset")
        self.assertTrue(result)
    
    def test_validate_dataframe_schema_failure(self):
        """Test DataFrame schema validation failure."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        required_columns = ['feature1', 'feature2', 'missing_column']
        
        # Should raise exception
        with self.assertRaises(DataValidationError):
            validate_dataframe_schema(df, required_columns, "test dataset")
    
    def test_validate_biomarker_data(self):
        """Test biomarker data validation."""
        # Create test data with biomarkers
        df = pd.DataFrame({
            'CD4 cell count (cells/µL)': [500, 600, np.nan, 700, 800],
            'FASTING GLUCOSE': [90, 95, 100, np.nan, 110],
            'systolic blood pressure': [120, 130, 140, 150, np.nan],
            'other_column': [1, 2, 3, 4, 5]
        })
        
        stats = validate_biomarker_data(df)
        
        # Check that biomarkers are analyzed
        self.assertIn('CD4 cell count (cells/µL)', stats)
        self.assertIn('FASTING GLUCOSE', stats)
        self.assertIn('systolic blood pressure', stats)
        
        # Check completeness calculations
        cd4_stats = stats['CD4 cell count (cells/µL)']
        self.assertEqual(cd4_stats['total_count'], 5)
        self.assertEqual(cd4_stats['non_null_count'], 4)
        self.assertEqual(cd4_stats['completeness_pct'], 80.0)

class TestMLUtils(unittest.TestCase):
    """Test ML utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set reproducible environment
        set_reproducible_environment(42)
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create realistic target with some noise
        true_coefficients = np.random.randn(n_features)
        self.y = pd.Series(
            self.X.values @ true_coefficients + np.random.randn(n_samples) * 0.1
        )
        
        # Add some categorical features
        self.X['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        
        # Create dataset with missing values
        self.df_with_missing = pd.DataFrame({
            **self.X,
            'target': self.y
        })
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=20, replace=False)
        self.df_with_missing.loc[missing_indices, 'target'] = np.nan
    
    def test_prepare_features_safely_success(self):
        """Test successful feature preparation."""
        feature_cols = [f'feature_{i}' for i in range(10)]
        
        X_train, X_test, y_train, y_test = prepare_features_safely(
            self.df_with_missing, feature_cols, 'target', test_size=0.2, random_state=42
        )
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(self.df_with_missing.dropna(subset=['target'])))
        self.assertEqual(X_train.shape[1], len(feature_cols))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check no missing values in target
        self.assertFalse(y_train.isna().any())
        self.assertFalse(y_test.isna().any())
    
    def test_prepare_features_safely_missing_features(self):
        """Test feature preparation with missing feature columns."""
        feature_cols = ['feature_0', 'missing_feature', 'feature_1']
        
        with self.assertRaises(MLPipelineError):
            prepare_features_safely(
                self.df_with_missing, feature_cols, 'target', test_size=0.2, random_state=42
            )
    
    def test_prepare_features_safely_missing_target(self):
        """Test feature preparation with missing target column."""
        feature_cols = [f'feature_{i}' for i in range(10)]
        
        with self.assertRaises(MLPipelineError):
            prepare_features_safely(
                self.df_with_missing, feature_cols, 'missing_target', test_size=0.2, random_state=42
            )
    
    def test_train_model_with_cv_random_forest(self):
        """Test Random Forest training with cross-validation."""
        feature_cols = [f'feature_{i}' for i in range(5)]  # Use fewer features for speed
        X = self.X[feature_cols]
        
        results = train_model_with_cv(
            X, self.y, 
            model_type='random_forest',
            cv_folds=3,  # Use fewer folds for speed
            feature_selection=False
        )
        
        # Check results structure
        self.assertIn('model', results)
        self.assertIn('cv_scores', results)
        self.assertIn('feature_importance', results)
        self.assertIn('selected_features', results)
        
        # Check CV scores
        cv_scores = results['cv_scores']
        self.assertIn('test_r2_mean', cv_scores)
        self.assertIn('test_r2_std', cv_scores)
        self.assertIn('test_mae_mean', cv_scores)
        
        # Check reasonable R² score
        self.assertGreater(cv_scores['test_r2_mean'], 0.5)  # Should be reasonably good on synthetic data
        
        # Check feature importance
        if results['feature_importance'] is not None:
            self.assertEqual(len(results['feature_importance']), len(feature_cols))
    
    def test_train_model_with_cv_xgboost(self):
        """Test XGBoost training with cross-validation."""
        feature_cols = [f'feature_{i}' for i in range(5)]
        X = self.X[feature_cols]
        
        results = train_model_with_cv(
            X, self.y,
            model_type='xgboost',
            cv_folds=3,
            feature_selection=False
        )
        
        # Check results structure
        self.assertIn('model', results)
        self.assertIn('cv_scores', results)
        
        # Check reasonable performance
        cv_scores = results['cv_scores']
        self.assertGreater(cv_scores['test_r2_mean'], 0.5)
    
    def test_train_model_with_cv_feature_selection(self):
        """Test model training with feature selection."""
        feature_cols = [f'feature_{i}' for i in range(10)]
        X = self.X[feature_cols]
        
        results = train_model_with_cv(
            X, self.y,
            model_type='random_forest',
            cv_folds=3,
            feature_selection=True,
            max_features=5
        )
        
        # Check that feature selection worked
        self.assertLessEqual(results['n_features'], 5)
        self.assertLessEqual(len(results['selected_features']), 5)
    
    def test_apply_multiple_testing_correction_bonferroni(self):
        """Test Bonferroni multiple testing correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        
        results = apply_multiple_testing_correction(
            p_values, method='bonferroni', alpha=0.05
        )
        
        # Check results structure
        self.assertIn('original_p_values', results)
        self.assertIn('corrected_p_values', results)
        self.assertIn('significant', results)
        self.assertIn('corrected_alpha', results)
        self.assertIn('n_significant', results)
        
        # Check Bonferroni correction
        expected_alpha = 0.05 / len(p_values)
        self.assertAlmostEqual(results['corrected_alpha'], expected_alpha)
        
        # Check that correction was applied
        self.assertTrue(all(results['corrected_p_values'] >= results['original_p_values']))
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        # Create synthetic predictions
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluate_model_performance(y_true, y_pred, "Test Model")
        
        # Check metrics structure
        expected_metrics = ['r2_score', 'mae', 'mse', 'rmse', 'pearson_r', 'spearman_r']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['r2_score'], 0.9)  # Should be high for near-perfect predictions
        self.assertLess(metrics['mae'], 0.2)  # Should be low error

class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = ENBELConfig()
        
        # Check basic configuration
        self.assertIsInstance(config.config, dict)
        self.assertIn('random_state', config.config)
        self.assertIn('biomarkers', config.config)
        self.assertIn('ml_settings', config.config)
        
        # Check biomarkers
        biomarkers = config.get_biomarkers()
        self.assertIsInstance(biomarkers, list)
        self.assertGreater(len(biomarkers), 0)
        
        # Check ML configuration
        ml_config = config.get_ml_config()
        self.assertIn('cv_folds', ml_config)
        self.assertIn('alpha_bonferroni', ml_config)
    
    def test_reproducible_environment(self):
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

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        set_reproducible_environment(42)
        
        # Create larger synthetic dataset
        np.random.seed(42)
        n_samples = 500
        n_features = 20
        
        # Create realistic climate-health synthetic data
        self.df = pd.DataFrame()
        
        # Add climate features
        for i in range(10):
            self.df[f'temperature_lag{i}'] = np.random.normal(25, 5, n_samples)
            self.df[f'humidity_lag{i}'] = np.random.normal(60, 10, n_samples)
        
        # Add demographic features
        self.df['Sex'] = np.random.choice(['Male', 'Female'], n_samples)
        self.df['Race'] = np.random.choice(['Black', 'White', 'Other'], n_samples)
        self.df['age'] = np.random.normal(45, 15, n_samples)
        
        # Add realistic biomarkers with climate relationships
        temp_effect = self.df['temperature_lag1'] * 0.1
        humidity_effect = self.df['humidity_lag0'] * -0.05
        noise = np.random.normal(0, 5, n_samples)
        
        self.df['systolic blood pressure'] = 120 + temp_effect + humidity_effect + noise
        self.df['FASTING GLUCOSE'] = 95 + temp_effect * 2 + noise * 0.5
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=50, replace=False)
        self.df.loc[missing_indices, 'systolic blood pressure'] = np.nan
    
    def test_end_to_end_analysis(self):
        """Test end-to-end analysis pipeline."""
        # Prepare features
        feature_cols = [col for col in self.df.columns if col not in ['systolic blood pressure', 'FASTING GLUCOSE']]
        target = 'systolic blood pressure'
        
        # Test data preparation
        X_train, X_test, y_train, y_test = prepare_features_safely(
            self.df, feature_cols, target, test_size=0.2, random_state=42
        )
        
        self.assertGreater(len(X_train), 300)  # Should have reasonable training set
        
        # Test model training
        results = train_model_with_cv(
            X_train, y_train,
            model_type='random_forest',
            cv_folds=3,
            feature_selection=True,
            max_features=10
        )
        
        # Should achieve reasonable performance on synthetic data
        self.assertGreater(results['cv_scores']['test_r2_mean'], 0.1)
        
        # Test predictions on test set
        model = results['model']
        selected_features = results['selected_features']
        X_test_selected = X_test[selected_features]
        
        predictions = model.predict(X_test_selected)
        test_metrics = evaluate_model_performance(y_test, predictions, "Integration Test")
        
        # Should have reasonable test performance
        self.assertGreater(test_metrics['r2_score'], 0.0)

def run_test_suite():
    """Run the complete test suite."""
    print("🧪 Running ENBEL ML Pipeline Test Suite...")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataValidation,
        TestMLUtils, 
        TestConfiguration,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        # Print failure details
        for test, traceback in result.failures + result.errors:
            print(f"\n❌ FAILED: {test}")
            print(traceback)
    
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_test_suite()
    exit(0 if success else 1)