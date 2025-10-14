#!/usr/bin/env python3
"""
Test Script for ENBEL Imputation Methodology
============================================

This script tests the imputation methodology with synthetic data to ensure
it works correctly before applying to real ENBEL data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from enbel_pp.imputation import SocioeconomicImputer
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_test_data():
    """Create synthetic data that mimics ENBEL structure."""
    print("Creating synthetic test data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Johannesburg area coordinates (rough bounds)
    lat_min, lat_max = -26.4, -26.0
    lon_min, lon_max = 27.8, 28.4
    
    # Create GCRO-like donor data (has socioeconomic variables)
    n_donors = 1000
    
    donor_data = pd.DataFrame({
        'latitude': np.random.uniform(lat_min, lat_max, n_donors),
        'longitude': np.random.uniform(lon_min, lon_max, n_donors),
        'Sex': np.random.choice(['Male', 'Female'], n_donors),
        'Race': np.random.choice(['Black', 'White', 'Coloured', 'Indian'], n_donors, 
                                p=[0.7, 0.15, 0.1, 0.05]),  # South African demographics
        'Education': np.random.choice([1, 2, 3, 4, 5], n_donors, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'employment_status': np.random.choice([1, 2, 3], n_donors, p=[0.6, 0.3, 0.1]),
        'vuln_Housing': np.random.choice([1, 2, 3], n_donors, p=[0.5, 0.3, 0.2]),
        'heat_vulnerability_index': np.random.uniform(1, 5, n_donors)
    })
    
    # Create Clinical-like recipient data (missing socioeconomic variables)
    n_recipients = 500
    
    recipient_data = pd.DataFrame({
        'latitude': np.random.uniform(lat_min, lat_max, n_recipients),
        'longitude': np.random.uniform(lon_min, lon_max, n_recipients),
        'Sex': np.random.choice(['Male', 'Female'], n_recipients),
        'Race': np.random.choice(['Black', 'White', 'Coloured', 'Indian'], n_recipients,
                                p=[0.7, 0.15, 0.1, 0.05]),
        # Add some health variables (these won't be used for imputation)
        'systolic_bp': np.random.normal(130, 20, n_recipients),
        'glucose': np.random.normal(95, 15, n_recipients)
    })
    
    print(f"Created donor data: {len(donor_data):,} records")
    print(f"Created recipient data: {len(recipient_data):,} records")
    
    return donor_data, recipient_data

def test_knn_imputation():
    """Test KNN-based imputation."""
    print("\n" + "="*60)
    print("TESTING KNN IMPUTATION")
    print("="*60)
    
    donor_data, recipient_data = create_synthetic_test_data()
    
    # Variables to impute
    target_vars = ['Education', 'employment_status', 'vuln_Housing']
    
    # Initialize imputer
    imputer = SocioeconomicImputer(
        method='knn',
        k_neighbors=5,
        spatial_weight=0.4,
        max_distance_km=10,
        random_state=42
    )
    
    # Perform imputation
    try:
        result = imputer.fit_and_impute(donor_data, recipient_data, target_vars)
        
        print("✅ KNN imputation successful!")
        
        # Check results
        for var in target_vars:
            imputed_col = f'{var}_imputed'
            confidence_col = f'{var}_confidence'
            
            if imputed_col in result.columns:
                n_imputed = result[imputed_col].notna().sum()
                mean_confidence = result[confidence_col].mean()
                print(f"  {var}: {n_imputed:,} imputed (confidence: {mean_confidence:.3f})")
        
        return True
    
    except Exception as e:
        print(f"❌ KNN imputation failed: {e}")
        return False

def test_ecological_imputation():
    """Test ecological imputation."""
    print("\n" + "="*60)
    print("TESTING ECOLOGICAL IMPUTATION")
    print("="*60)
    
    donor_data, recipient_data = create_synthetic_test_data()
    
    target_vars = ['Education', 'employment_status', 'vuln_Housing']
    
    imputer = SocioeconomicImputer(
        method='ecological',
        spatial_weight=0.5,
        random_state=42
    )
    
    try:
        result = imputer.fit_and_impute(donor_data, recipient_data, target_vars)
        
        print("✅ Ecological imputation successful!")
        
        for var in target_vars:
            imputed_col = f'{var}_imputed'
            confidence_col = f'{var}_confidence'
            
            if imputed_col in result.columns:
                n_imputed = result[imputed_col].notna().sum()
                mean_confidence = result[confidence_col].mean()
                print(f"  {var}: {n_imputed:,} imputed (confidence: {mean_confidence:.3f})")
        
        return True
    
    except Exception as e:
        print(f"❌ Ecological imputation failed: {e}")
        return False

def test_combined_imputation():
    """Test combined imputation approach."""
    print("\n" + "="*60)
    print("TESTING COMBINED IMPUTATION")
    print("="*60)
    
    donor_data, recipient_data = create_synthetic_test_data()
    
    target_vars = ['Education', 'employment_status', 'vuln_Housing']
    
    imputer = SocioeconomicImputer(
        method='combined',
        k_neighbors=8,
        spatial_weight=0.4,
        max_distance_km=15,
        random_state=42
    )
    
    try:
        result = imputer.fit_and_impute(donor_data, recipient_data, target_vars)
        
        print("✅ Combined imputation successful!")
        
        for var in target_vars:
            imputed_col = f'{var}_imputed'
            confidence_col = f'{var}_confidence'
            
            if imputed_col in result.columns:
                n_imputed = result[imputed_col].notna().sum()
                mean_confidence = result[confidence_col].mean()
                min_conf = result[confidence_col].min()
                max_conf = result[confidence_col].max()
                print(f"  {var}: {n_imputed:,} imputed")
                print(f"    Confidence: mean={mean_confidence:.3f}, range=[{min_conf:.3f}, {max_conf:.3f}]")
        
        # Display validation results if available
        if hasattr(imputer, 'validation_scores') and imputer.validation_scores:
            print("\nValidation Results:")
            for var, scores in imputer.validation_scores.items():
                print(f"  {var}: RMSE={scores['rmse']:.3f}, MAE={scores['mae']:.3f}, r={scores['correlation']:.3f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Combined imputation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation functionality."""
    print("\n" + "="*60)
    print("TESTING DATA VALIDATION")
    print("="*60)
    
    # Create data with some issues
    donor_data = pd.DataFrame({
        'latitude': [-26.2, -26.3, 100.0, -26.1],  # One invalid coordinate
        'longitude': [28.0, 28.1, 28.2, 200.0],    # One invalid coordinate
        'Sex': ['Male', 'Female', 'Male', 'Female'],
        'Race': ['Black', 'White', 'Black', 'White'],
        'Education': [1, 2, 3, 4]
    })
    
    recipient_data = pd.DataFrame({
        'latitude': [-26.2, -26.3],
        'longitude': [28.0, 28.1],
        'Sex': ['Male', 'Female'],
        'Race': ['Black', 'White']
    })
    
    imputer = SocioeconomicImputer(random_state=42)
    
    try:
        donor_clean, recipient_clean = imputer.validate_data(donor_data, recipient_data)
        
        print("✅ Data validation successful!")
        print(f"  Donor data: {len(donor_clean)}/{len(donor_data)} valid records")
        print(f"  Recipient data: {len(recipient_clean)}/{len(recipient_data)} valid records")
        
        return True
    
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ENBEL IMPUTATION METHODOLOGY TEST SUITE")
    print("="*80)
    
    tests = [
        ("Data Validation", test_data_validation),
        ("KNN Imputation", test_knn_imputation),
        ("Ecological Imputation", test_ecological_imputation),
        ("Combined Imputation", test_combined_imputation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The imputation methodology is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)