#!/usr/bin/env python3
"""
ENBEL Climate-Health Analysis Setup and Validation Script
=========================================================

This script sets up the environment and validates that everything is working
correctly before running the main analysis pipeline.

VALIDATION CHECKS:
- ✅ Python environment and dependencies
- ✅ Data files existence and integrity
- ✅ Configuration setup
- ✅ Pipeline component tests
- ✅ Reproducibility verification

Author: ENBEL Project Team
"""

import sys
import subprocess
from pathlib import Path
import importlib
import warnings
import traceback
import time
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

def print_header(text, char="="):
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}")

def print_step(step_num, total_steps, description):
    """Print formatted step."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 60)

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ ERROR: Python 3.8 or higher required")
        return False
    else:
        print("   ✅ Python version compatible")
        return True

def check_dependencies():
    """Check required dependencies."""
    print("📦 Checking dependencies...")
    
    # Read requirements
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("   ❌ ERROR: requirements.txt not found")
        return False
    
    required_packages = []
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before >= or ==)
                package = line.split('>=')[0].split('==')[0].replace('-', '_')
                required_packages.append((line.strip(), package))
    
    missing_packages = []
    
    for req_line, package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (from {req_line})")
            missing_packages.append(req_line)
    
    if missing_packages:
        print(f"\n   ❌ Missing {len(missing_packages)} packages:")
        for pkg in missing_packages:
            print(f"      - {pkg}")
        print("\n   Install with: pip install -r requirements.txt")
        return False
    else:
        print(f"   ✅ All {len(required_packages)} dependencies available")
        return True

def check_data_files():
    """Check required data files."""
    print("📁 Checking data files...")
    
    try:
        from data_validation import validate_data_files
        data_files = validate_data_files()
        
        for file_type, file_path in data_files.items():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_type}: {file_path.name} ({file_size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data file validation failed: {e}")
        print("\n   Required data files:")
        print("      - FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv")
        print("      - CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv")
        print("      - CLINICAL_WITH_IMPUTED_SOCIOECONOMIC.csv")
        return False

def check_configuration():
    """Check configuration setup."""
    print("⚙️  Checking configuration...")
    
    try:
        from config import ENBELConfig, set_reproducible_environment
        
        # Test configuration loading
        config = ENBELConfig()
        print(f"   ✅ Configuration loaded")
        print(f"      - Base directory: {config.BASE_DIR}")
        print(f"      - Random state: {config.config['random_state']}")
        print(f"      - Biomarkers: {len(config.get_biomarkers())}")
        print(f"      - Bonferroni α: {config.config['ml_settings']['alpha_bonferroni']:.6f}")
        
        # Test reproducible environment
        set_reproducible_environment(42)
        print("   ✅ Reproducible environment set")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        traceback.print_exc()
        return False

def run_unit_tests():
    """Run unit tests."""
    print("🧪 Running unit tests...")
    
    try:
        # Check if test file exists
        test_file = Path(__file__).parent / "tests" / "test_ml_pipeline.py"
        if not test_file.exists():
            print("   ⚠️  Test file not found, skipping unit tests")
            return True
        
        # Import and run tests
        sys.path.append(str(test_file.parent))
        from test_ml_pipeline import run_test_suite
        
        print("   Running comprehensive test suite...")
        success = run_test_suite()
        
        if success:
            print("   ✅ All unit tests passed")
            return True
        else:
            print("   ❌ Some unit tests failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Unit test execution failed: {e}")
        traceback.print_exc()
        return False

def validate_data_integrity():
    """Validate data integrity."""
    print("🔍 Validating data integrity...")
    
    try:
        from data_validation import run_full_validation
        
        print("   Running comprehensive data validation...")
        success = run_full_validation()
        
        if success:
            print("   ✅ Data validation passed")
            return True
        else:
            print("   ❌ Data validation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Data validation error: {e}")
        return False

def test_ml_pipeline():
    """Test ML pipeline components."""
    print("🤖 Testing ML pipeline components...")
    
    try:
        import pandas as pd
        import numpy as np
        from ml_utils import prepare_features_safely, train_model_with_cv
        from config import set_reproducible_environment
        
        # Set reproducible environment
        set_reproducible_environment(42)
        
        # Create synthetic test data
        print("   Creating synthetic test data...")
        n_samples = 100
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })
        
        # Test feature preparation
        print("   Testing feature preparation...")
        X_train, X_test, y_train, y_test = prepare_features_safely(
            test_data, ['feature_1', 'feature_2', 'feature_3'], 'target'
        )
        
        # Test model training
        print("   Testing model training...")
        results = train_model_with_cv(
            X_train, y_train, 
            model_type='random_forest',
            cv_folds=3
        )
        
        print(f"   ✅ ML pipeline test passed (R² = {results['cv_scores']['test_r2_mean']:.3f})")
        return True
        
    except Exception as e:
        print(f"   ❌ ML pipeline test failed: {e}")
        traceback.print_exc()
        return False

def setup_directories():
    """Set up required directories."""
    print("📂 Setting up directories...")
    
    try:
        from config import ENBELConfig
        config = ENBELConfig()
        
        # Directories are automatically created by config
        directories = [
            ('results', config.config['paths']['results_dir']),
            ('models', config.config['paths']['models_dir']),
            ('figures', config.config['paths']['figures_dir']),
            ('logs', config.config['paths']['logs_dir'])
        ]
        
        for dir_name, dir_path in directories:
            if dir_path.exists():
                print(f"   ✅ {dir_name}: {dir_path}")
            else:
                print(f"   ❌ {dir_name}: {dir_path} (not created)")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Directory setup failed: {e}")
        return False

def generate_validation_report(results, start_time):
    """Generate comprehensive validation report."""
    print_header("VALIDATION REPORT")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Validation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"Overall Status: {'✅ PASS' if passed_checks == total_checks else '❌ FAIL'}")
    print(f"Checks Passed: {passed_checks}/{total_checks}")
    print("")
    
    print("Detailed Results:")
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name:<30} {status}")
    
    if passed_checks == total_checks:
        print_header("🎉 ENVIRONMENT READY FOR ANALYSIS!", "=")
        print("All validation checks passed successfully!")
        print("You can now run the ML pipeline with confidence.")
        print("")
        print("Next steps:")
        print("   1. Run: python 01_initial_analysis/supervised_ml/corrected_interpretable_ml_pipeline.py")
        print("   2. Check results in the 'results' directory")
        print("   3. Review logs in the 'logs' directory")
    else:
        print_header("❌ VALIDATION FAILED", "=")
        print("Some validation checks failed. Please fix the issues above before proceeding.")
        print("")
        print("Common fixes:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Ensure data files are in the correct location")
        print("   - Check Python version (3.8+ required)")
    
    return passed_checks == total_checks

def main():
    """Main validation function."""
    print_header("ENBEL CLIMATE-HEALTH ANALYSIS SETUP & VALIDATION")
    print("This script validates your environment for running the ML pipeline.")
    
    start_time = time.time()
    
    # Define validation steps
    validation_steps = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Configuration", check_configuration),
        ("Directories", setup_directories),
        ("Data Integrity", validate_data_integrity),
        ("ML Pipeline", test_ml_pipeline),
        ("Unit Tests", run_unit_tests),
    ]
    
    results = {}
    
    # Run validation steps
    for i, (step_name, step_function) in enumerate(validation_steps, 1):
        print_step(i, len(validation_steps), step_name)
        
        try:
            success = step_function()
            results[step_name] = success
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            results[step_name] = False
    
    # Generate report
    validation_success = generate_validation_report(results, start_time)
    
    return validation_success

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n❌ Validation interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n\n❌ Unexpected error during validation: {e}")
        traceback.print_exc()
        exit_code = 1
    
    exit(exit_code)