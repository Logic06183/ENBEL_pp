#!/usr/bin/env python3
"""
Test with the new HEAT export dataset v1.0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TESTING NEW HEAT EXPORT DATASET v1.0")
print("=" * 60)

# Load new clinical dataset
df = pd.read_csv('data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', low_memory=False)
print(f"1. Loaded new clinical dataset: {df.shape}")

# Check data completeness
print(f"\n2. Data quality overview:")
print(f"   Records: {len(df):,}")
print(f"   Features: {len(df.columns)}")

# Check for missing data patterns
missing_summary = df.isnull().sum()
high_missing = missing_summary[missing_summary > 0].sort_values(ascending=False)
print(f"   Columns with missing data: {len(high_missing)}")

# Show biomarkers
biomarker_patterns = ['cd4', 'glucose', 'cholesterol', 'hemoglobin', 'creatinine', 'blood_pressure']
biomarkers = []
for pattern in biomarker_patterns:
    matching = [col for col in df.columns if pattern in col.lower()]
    biomarkers.extend(matching)

print(f"\n3. Available biomarkers ({len(biomarkers)}):")
for biomarker in biomarkers:
    non_missing = df[biomarker].notna().sum()
    coverage = (non_missing / len(df)) * 100
    print(f"   {biomarker:<35} {non_missing:>6} ({coverage:5.1f}%)")

# Show climate features
climate_patterns = ['climate', 'temp', 'heat', 'humid']
climate_features = []
for pattern in climate_patterns:
    matching = [col for col in df.columns if pattern in col.lower() and df[col].dtype in [np.float64, np.int64]]
    climate_features.extend(matching)

climate_features = list(set(climate_features))  # Remove duplicates
print(f"\n4. Climate features ({len(climate_features)}):")
for climate in climate_features[:10]:
    non_missing = df[climate].notna().sum()
    coverage = (non_missing / len(df)) * 100
    print(f"   {climate:<35} {non_missing:>6} ({coverage:5.1f}%)")

if len(climate_features) > 10:
    print(f"   ... and {len(climate_features) - 10} more climate features")

# Test creatinine model with new data
print(f"\n" + "="*50)
print("CREATININE MODEL WITH NEW DATASET")
print("="*50)

# Use the new creatinine column (umol/L)
creatinine_col = 'creatinine_umol_L'
if creatinine_col in df.columns:
    print(f"5. Analyzing {creatinine_col}:")
    
    creat_data = df[creatinine_col].dropna()
    print(f"   Non-missing: {len(creat_data)}")
    print(f"   Mean: {creat_data.mean():.1f} µmol/L")
    print(f"   Range: {creat_data.min():.1f} - {creat_data.max():.1f} µmol/L")
    
    # Normal ranges in µmol/L
    # Men: 62-106 µmol/L, Women: 44-80 µmol/L
    # Using general range 44-106 µmol/L
    normal = (creat_data >= 44) & (creat_data <= 106)
    elevated = creat_data > 106
    low = creat_data < 44
    
    print(f"   Normal (44-106): {normal.sum()} ({normal.mean()*100:.1f}%)")
    print(f"   Elevated (>106): {elevated.sum()} ({elevated.mean()*100:.1f}%)")
    print(f"   Low (<44): {low.sum()} ({low.mean()*100:.1f}%)")
    
    # Prepare model data
    print(f"\n6. Building creatinine prediction model:")
    
    # Get climate features for modeling
    model_climate = [col for col in climate_features if df[col].notna().sum() > len(df) * 0.5]  # >50% coverage
    print(f"   Using {len(model_climate)} climate features with good coverage")
    
    # Get complete cases
    model_cols = model_climate + [creatinine_col]
    complete_df = df[model_cols].dropna()
    print(f"   Complete cases: {len(complete_df)} ({len(complete_df)/len(df)*100:.1f}%)")
    
    if len(complete_df) > 50:
        X = complete_df[model_climate]
        y = complete_df[creatinine_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
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
        
        print(f"\n7. Model performance:")
        print(f"   R² = {r2:.3f}")
        print(f"   MAE = {mae:.1f} µmol/L")
        
        # Clinical significance
        mae_percent = (mae / y.mean()) * 100
        print(f"   MAE as % of mean: {mae_percent:.1f}%")
        
        # Feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': model_climate,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n8. Top climate predictors:")
        for i, row in importance_df.head(8).iterrows():
            print(f"   {row['feature']:<35} {row['importance']:.4f}")
        
        # SHAP analysis if model is reasonable
        if r2 > 0.1:
            print(f"\n9. SHAP analysis:")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test.iloc[:min(20, len(X_test))])
                
                # SHAP feature importance
                shap_importance = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({
                    'feature': model_climate,
                    'shap_importance': shap_importance
                }).sort_values('shap_importance', ascending=False)
                
                print(f"   SHAP values calculated for {len(X_test.iloc[:20])} samples")
                print(f"   Top SHAP predictors:")
                for i, row in shap_df.head(5).iterrows():
                    print(f"     {row['feature']:<35} {row['shap_importance']:.4f}")
                
                # Expected value
                expected = explainer.expected_value
                if isinstance(expected, np.ndarray):
                    expected = expected[0]
                print(f"   Baseline creatinine: {expected:.1f} µmol/L")
                
            except Exception as e:
                print(f"   SHAP analysis failed: {e}")
        else:
            print(f"\n9. Model R² too low for reliable SHAP analysis")
    
    else:
        print(f"   ✗ Insufficient complete data: {len(complete_df)}")

# Test glucose model (should be better)
print(f"\n" + "="*50)
print("GLUCOSE MODEL WITH NEW DATASET")
print("="*50)

glucose_col = 'fasting_glucose_mmol_L'
if glucose_col in df.columns:
    print(f"10. Analyzing {glucose_col}:")
    
    glucose_data = df[glucose_col].dropna()
    print(f"    Non-missing: {len(glucose_data)}")
    print(f"    Mean: {glucose_data.mean():.1f} mmol/L")
    print(f"    Range: {glucose_data.min():.1f} - {glucose_data.max():.1f} mmol/L")
    
    # Normal glucose: 3.9-5.5 mmol/L (fasting)
    # Prediabetes: 5.6-6.9 mmol/L
    # Diabetes: ≥7.0 mmol/L
    normal_glucose = (glucose_data >= 3.9) & (glucose_data <= 5.5)
    prediabetes = (glucose_data >= 5.6) & (glucose_data <= 6.9)
    diabetes = glucose_data >= 7.0
    
    print(f"    Normal (3.9-5.5): {normal_glucose.sum()} ({normal_glucose.mean()*100:.1f}%)")
    print(f"    Prediabetes (5.6-6.9): {prediabetes.sum()} ({prediabetes.mean()*100:.1f}%)")
    print(f"    Diabetes (≥7.0): {diabetes.sum()} ({diabetes.mean()*100:.1f}%)")
    
    # Quick glucose model
    model_cols_glucose = model_climate + [glucose_col]
    complete_glucose = df[model_cols_glucose].dropna()
    
    print(f"    Complete glucose cases: {len(complete_glucose)}")
    
    if len(complete_glucose) > 100:
        X_g = complete_glucose[model_climate]
        y_g = complete_glucose[glucose_col]
        
        X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(
            X_g, y_g, test_size=0.2, random_state=42
        )
        
        glucose_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        glucose_model.fit(X_g_train, y_g_train)
        
        y_g_pred = glucose_model.predict(X_g_test)
        glucose_r2 = r2_score(y_g_test, y_g_pred)
        glucose_mae = mean_absolute_error(y_g_test, y_g_pred)
        
        print(f"    Glucose model: R² = {glucose_r2:.3f}, MAE = {glucose_mae:.2f} mmol/L")

print(f"\n" + "=" * 60)
print("NEW DATASET SUMMARY")
print("=" * 60)
print(f"✓ Successfully loaded new HEAT export dataset")
print(f"✓ {len(df):,} clinical records with {len(df.columns)} features") 
print(f"✓ {len(biomarkers)} biomarkers identified")
print(f"✓ {len(climate_features)} climate features available")

if 'r2' in locals():
    print(f"✓ Creatinine model: R² = {r2:.3f}")
if 'glucose_r2' in locals():
    print(f"✓ Glucose model: R² = {glucose_r2:.3f}")

print(f"\nNext steps:")
print(f"1. Update pipeline to use new dataset format")
print(f"2. Test all biomarkers systematically")
print(f"3. Implement imputation with GCRO data")
print(f"4. Run full SHAP analysis on good models")