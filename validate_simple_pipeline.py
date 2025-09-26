#!/usr/bin/env python3
"""
Simple Pipeline Validation Script
=================================

This script validates that the simple pipeline works correctly.
It's designed to be easy to understand and verify.

Author: ENBEL Project Team
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def validate_environment():
    """Check that the Python environment is ready."""
    print("Validating Environment")
    print("-" * 30)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("ERROR: Need Python 3.8 or higher")
        return False
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"PASS: {package}")
        except ImportError:
            print(f"MISSING: {package}")
            missing.append(package)
    
    if missing:
        print(f"\nERROR: Install missing packages: pip install {' '.join(missing)}")
        return False
    
    print("✅ Environment ready!")
    return True

def validate_data_files():
    """Check that required data files exist."""
    print("\nValidating Data Files")
    print("-" * 30)
    
    required_files = [
        'DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv',
        'DEIDENTIFIED_CLINICAL_IMPUTED.csv',
        'DEIDENTIFIED_CLINICAL_ORIGINAL.csv'
    ]
    
    all_good = True
    for filename in required_files:
        file_path = Path(filename)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"FOUND: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"MISSING: {filename}")
            all_good = False
    
    if all_good:
        print("✅ All data files present!")
    else:
        print("ERROR: Some data files are missing!")
    
    return all_good

def test_data_loading():
    """Test that we can load and process the data."""
    print("\n📊 Testing Data Loading")
    print("-" * 30)
    
    try:
        # Load de-identified dataset
        df = pd.read_csv('DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv', low_memory=False)
        print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
        # Check for key biomarkers
        biomarkers = ['systolic blood pressure', 'FASTING GLUCOSE', 'CD4 cell count (cells/µL)']
        found_biomarkers = [b for b in biomarkers if b in df.columns]
        print(f"Found {len(found_biomarkers)}/{len(biomarkers)} key biomarkers")
        
        # Check for climate features
        climate_features = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'humid'])]
        print(f"Found {len(climate_features)} climate features")
        
        # Test basic data quality
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells) * 100
        print(f"Missing data: {missing_pct:.1f}% (expected for this type of dataset)")
        
        print("✅ Data loading successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: Data loading failed: {e}")
        return False

def test_simple_ml():
    """Test a basic machine learning workflow."""
    print("\n🤖 Testing ML Workflow")
    print("-" * 30)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        import xgboost as xgb
        
        # Load data
        df = pd.read_csv('DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv', low_memory=False)
        
        # Test with systolic blood pressure
        target = 'systolic blood pressure'
        if target not in df.columns:
            print(f"ERROR: Target '{target}' not found")
            return False
        
        # Get some climate features
        temp_features = [col for col in df.columns if 'temp' in col.lower() and df[col].dtype in ['float64', 'int64']][:5]
        
        if len(temp_features) < 3:
            print("ERROR: Not enough temperature features found")
            return False
        
        # Clean data
        data_clean = df[[target] + temp_features].dropna()
        
        if len(data_clean) < 100:
            print(f"ERROR: Not enough clean data: {len(data_clean)}")
            return False
        
        X = data_clean[temp_features]
        y = data_clean[target]
        
        print(f"Test dataset: {len(X):,} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test Random Forest
        rf = RandomForestRegressor(n_estimators=10, random_state=42)  # Small for speed
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        print(f"Random Forest R²: {rf_r2:.4f}")
        
        # Test XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        print(f"XGBoost R²: {xgb_r2:.4f}")
        
        print("✅ ML workflow working!")
        return True
        
    except Exception as e:
        print(f"ERROR: ML workflow failed: {e}")
        return False

def test_simple_pipeline():
    """Test the actual simple pipeline script."""
    print("\n🚀 Testing Simple Pipeline")
    print("-" * 30)
    
    try:
        # Try to import the simple pipeline
        sys.path.append('.')
        from simple_ml_pipeline import SimpleClimateHealthPipeline
        
        # Create pipeline instance
        pipeline = SimpleClimateHealthPipeline()
        print("Pipeline initialized")
        
        # Test data loading
        data = pipeline.load_data()
        print("Data loading method works")
        
        # Test feature selection
        features = pipeline.select_climate_features()
        print(f"Feature selection works ({len(features)} features)")
        
        # Test data preparation for one biomarker
        X, y = pipeline.prepare_data_for_biomarker('systolic blood pressure')
        if X is not None:
            print(f"Data preparation works ({len(X)} samples)")
        else:
            print("⚠️ Data preparation returned None (may be expected)")
        
        print("✅ Simple pipeline components working!")
        return True
        
    except Exception as e:
        print(f"ERROR: Simple pipeline test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Simple Pipeline Validation")
    print("=" * 50)
    print("This script checks if everything is ready for the simple ML pipeline.\n")
    
    tests = [
        ("Environment", validate_environment),
        ("Data Files", validate_data_files),
        ("Data Loading", test_data_loading),
        ("ML Workflow", test_simple_ml),
        ("Simple Pipeline", test_simple_pipeline)
    ]
    
    results = {}
    
    for test_name, test_function in tests:
        print(f"\n{'='*50}")
        print(f"TEST: {test_name}")
        print('='*50)
        
        try:
            success = test_function()
            results[test_name] = success
        except Exception as e:
            print(f"ERROR: Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 VALIDATION SUMMARY")
    print('='*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your environment is ready for the simple pipeline")
        print("✅ You can now run: python simple_ml_pipeline.py")
    else:
        print(f"\nERROR: {total - passed} test(s) failed")
        print("Please fix the issues above before running the pipeline")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)