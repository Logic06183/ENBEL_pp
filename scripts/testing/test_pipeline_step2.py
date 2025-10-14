#!/usr/bin/env python3
"""
Test Script - Step 2: Test ML Pipeline
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

print("=" * 60)
print("TESTING STEP 2: ML PIPELINE")
print("=" * 60)

# Test 1: Load and prepare data
print("\n1. Loading and preparing data...")
df = pd.read_csv('data/raw/clinical_dataset.csv', low_memory=False)
print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Simple imputation for testing
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Selected {len(numeric_cols)} numeric columns")

df_numeric = df[numeric_cols].copy()

# Handle infinite values
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

# Impute missing values for testing
imputer = SimpleImputer(strategy='mean')
df_imputed_array = imputer.fit_transform(df_numeric)
df_imputed = pd.DataFrame(df_imputed_array, columns=numeric_cols)

print(f"   After imputation: {df_imputed.isnull().sum().sum()} missing values")

# Test 2: Identify features and targets
print("\n2. Identifying features and targets...")

# Climate features
climate_patterns = ['temp', 'humid', 'pressure', 'wind', 'lag']
climate_features = [col for col in df_imputed.columns 
                   if any(pattern in col.lower() for pattern in climate_patterns)]

# Biomarker targets  
biomarker_patterns = ['glucose', 'cd4', 'cholesterol', 'hemoglobin', 'creatinine']
biomarker_targets = [col for col in df_imputed.columns
                    if any(pattern in col.lower() for pattern in biomarker_patterns)]

print(f"   Climate features: {len(climate_features)}")
print(f"   Biomarker targets: {len(biomarker_targets)}")

if len(climate_features) > 0:
    print(f"   Climate examples: {climate_features[:3]}")
if len(biomarker_targets) > 0:
    print(f"   Biomarker examples: {biomarker_targets[:3]}")

# Test 3: Run simple ML model
if len(climate_features) > 0 and len(biomarker_targets) > 0:
    print("\n3. Testing ML model...")
    
    # Use first biomarker with enough data
    target_col = None
    for biomarker in biomarker_targets:
        non_zero_count = (df_imputed[biomarker] > 0).sum()
        if non_zero_count > 100:  # Need enough non-zero values
            target_col = biomarker
            break
    
    if target_col:
        print(f"   Using target: {target_col}")
        
        # Prepare features and target
        X = df_imputed[climate_features[:20]]  # Use first 20 climate features
        y = df_imputed[target_col]
        
        # Remove zero/invalid values
        valid_mask = (y > 0) & (y.notna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) > 50:
            print(f"   Valid samples: {len(X)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            print(f"   R² score: {r2:.3f}")
            
            # Feature importance
            feature_importance = model.feature_importances_
            important_features = list(zip(climate_features[:20], feature_importance))
            important_features.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   Top 3 important features:")
            for i, (feature, importance) in enumerate(important_features[:3]):
                print(f"      {i+1}. {feature}: {importance:.3f}")
                
        else:
            print(f"   Too few valid samples: {len(X)}")
    else:
        print("   No suitable biomarker target found")
else:
    print("\n3. Cannot test ML - missing features or targets")

# Test 4: Check SHAP requirements
print("\n4. Checking SHAP requirements...")
try:
    import shap
    print("   ✓ SHAP installed and importable")
    
    if 'model' in locals() and 'X_test' in locals():
        # Test SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:5])  # Test on 5 samples
        print(f"   ✓ SHAP values calculated: shape {shap_values.shape}")
    else:
        print("   ⚠ Model not available for SHAP testing")
        
except ImportError:
    print("   ✗ SHAP not installed")
except Exception as e:
    print(f"   ⚠ SHAP error: {e}")

print("\n" + "=" * 60)
print("ML PIPELINE TEST SUMMARY")
print("=" * 60)
print(f"✓ Data preprocessed: {df_imputed.shape}")
print(f"✓ Features identified: {len(climate_features)} climate")
print(f"✓ Targets identified: {len(biomarker_targets)} biomarkers")

if 'r2' in locals():
    print(f"✓ ML model trained: R² = {r2:.3f}")
    quality = "Good" if r2 > 0.3 else "Moderate" if r2 > 0.1 else "Poor"
    print(f"✓ Model quality: {quality}")
else:
    print("✗ ML model not trained")

try:
    import shap
    print("✓ SHAP available for explainability")
except:
    print("✗ SHAP needs installation")

print("\nNext steps:")
print("1. Improve feature selection and engineering")
print("2. Add hyperparameter optimization")
print("3. Implement systematic biomarker analysis")
print("4. Add SHAP analysis for good models")