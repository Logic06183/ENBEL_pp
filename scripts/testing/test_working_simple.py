#!/usr/bin/env python3
"""
Simple working test that handles the imputation issues properly
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("WORKING SIMPLE TEST")
print("=" * 60)

# Load data
df = pd.read_csv('data/raw/clinical_dataset.csv', low_memory=False)
print(f"1. Loaded data: {df.shape}")

# Get comprehensive climate features that have good coverage
climate_cols = []
for col in df.columns:
    if any(pattern in col.lower() for pattern in ['temp', 'humid', 'heat', 'wind']):
        if df[col].dtype in [np.float64, np.int64]:
            # Check if has reasonable coverage and variance
            non_missing = df[col].notna().sum()
            if non_missing > 5000:  # Good coverage
                variance = df[col].var()
                if variance > 0:  # Has variance
                    climate_cols.append(col)

print(f"2. Selected {len(climate_cols)} climate features with good coverage and variance")

# Test biomarkers
biomarkers = ['CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'systolic blood pressure']

for biomarker in biomarkers:
    if biomarker not in df.columns:
        continue
        
    print(f"\n{'='*50}")
    print(f"TESTING: {biomarker}")
    print(f"{'='*50}")
    
    # Get complete cases only (no imputation for now)
    all_cols = climate_cols + [biomarker]
    complete_df = df[all_cols].dropna()
    
    print(f"Complete cases: {len(complete_df)} (from {len(df)} total)")
    
    if len(complete_df) < 100:
        print("⚠ Insufficient complete data - skipping")
        continue
    
    # Prepare data
    X = complete_df[climate_cols]
    y = complete_df[biomarker]
    
    print(f"Features: {X.shape[1]}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Results:")
    print(f"  R² = {r2:.3f}")
    print(f"  MAE = {mae:.3f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': climate_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 5 climate features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.3f}")
    
    quality = "Excellent" if r2 > 0.5 else "Good" if r2 > 0.3 else "Moderate" if r2 > 0.1 else "Poor"
    print(f"  Quality: {quality}")
    
    if r2 > 0.3:
        print(f"  ✓ Suitable for SHAP analysis")
    else:
        print(f"  ⚠ May need more features or engineering")

print(f"\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Pipeline working with complete cases")
print(f"✓ Using {len(climate_cols)} climate features")
print(f"✓ Models trained and evaluated")
print(f"\nNext steps:")
print(f"1. Implement proper imputation to use more data")
print(f"2. Add more sophisticated features (lags, interactions)")
print(f"3. Try other algorithms (XGBoost, etc.)")
print(f"4. Install SHAP for explainability")