#!/usr/bin/env python3
"""
Improved test with more climate features and better feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("IMPROVED PIPELINE TEST WITH FULL CLIMATE FEATURES")
print("=" * 60)

# Load data
df = pd.read_csv('data/raw/clinical_dataset.csv', low_memory=False)
print(f"1. Loaded data: {df.shape}")

# Get comprehensive climate features
climate_patterns = ['temp', 'humid', 'pressure', 'wind', 'lag', 'heat', 'climate']
climate_cols = []

for col in df.columns:
    if any(pattern in col.lower() for pattern in climate_patterns):
        if df[col].dtype in [np.float64, np.int64]:
            non_missing = df[col].notna().sum()
            if non_missing > 1000:  # Good data coverage
                climate_cols.append(col)

print(f"2. Found {len(climate_cols)} climate features with good coverage")

# Select key biomarkers
biomarkers = ['CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 
              'Creatinine (mg/dL)', 'Hemoglobin (g/dL)', 'systolic blood pressure', 
              'diastolic blood pressure']

available_biomarkers = [col for col in biomarkers if col in df.columns]
print(f"3. Available biomarkers: {available_biomarkers}")

# Test each biomarker
results = {}

for biomarker in available_biomarkers:
    print(f"\n{'='*40}")
    print(f"TESTING: {biomarker}")
    print(f"{'='*40}")
    
    # Check data availability
    target_data = df[biomarker].dropna()
    print(f"Non-missing values: {len(target_data)}")
    
    if len(target_data) < 200:
        print(f"⚠ Skipping - insufficient data")
        continue
    
    print(f"Target stats: mean={target_data.mean():.2f}, std={target_data.std():.2f}")
    
    # Prepare data
    features_and_target = climate_cols + [biomarker]
    model_df = df[features_and_target].copy()
    
    # Remove rows where target is missing
    model_df = model_df.dropna(subset=[biomarker])
    print(f"Samples with target: {len(model_df)}")
    
    if len(model_df) < 200:
        print(f"⚠ Skipping - insufficient complete data")
        continue
    
    # Separate features and target
    X = model_df[climate_cols]
    y = model_df[biomarker]
    
    # Handle missing values in features
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        print(f"Imputing {missing_before} missing feature values...")
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), 
            columns=climate_cols, 
            index=X.index
        )
    else:
        X_imputed = X
    
    # Feature selection - remove features with zero variance
    feature_variance = X_imputed.var()
    good_features = feature_variance[feature_variance > 0].index.tolist()
    X_final = X_imputed[good_features]
    
    print(f"Using {len(good_features)} features (removed {len(climate_cols) - len(good_features)} zero-variance)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Standardize features for some algorithms
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=good_features,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=good_features,
        index=X_test.index
    )
    
    # Train Random Forest (doesn't need scaling)
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        random_state=42, 
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    print(f"Random Forest:")
    print(f"  R² = {r2_rf:.3f}")
    print(f"  MAE = {mae_rf:.3f}")
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': good_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 5 features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.3f}")
    
    # Store results
    results[biomarker] = {
        'n_samples': len(model_df),
        'n_features': len(good_features),
        'r2': r2_rf,
        'mae': mae_rf,
        'top_features': importance_df.head(5)['feature'].tolist()
    }
    
    # Check if good enough for SHAP
    if r2_rf > 0.3:
        print(f"  ✓ Model quality sufficient for SHAP analysis")
        
        # Try SHAP if available
        try:
            import shap
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test.iloc[:10])
            print(f"  ✓ SHAP analysis successful: {shap_values.shape}")
            results[biomarker]['shap_available'] = True
        except ImportError:
            print(f"  ⚠ SHAP not installed")
            results[biomarker]['shap_available'] = False
        except Exception as e:
            print(f"  ⚠ SHAP error: {e}")
            results[biomarker]['shap_available'] = False
    else:
        print(f"  ⚠ R² < 0.3 - insufficient for reliable interpretation")
        results[biomarker]['shap_available'] = False

print(f"\n" + "=" * 60)
print("COMPREHENSIVE TEST RESULTS")
print("=" * 60)

# Summary
successful_models = {k: v for k, v in results.items() if v['r2'] > 0.1}
good_models = {k: v for k, v in results.items() if v['r2'] > 0.3}

print(f"Total biomarkers tested: {len(results)}")
print(f"Models with R² > 0.1: {len(successful_models)}")
print(f"Models with R² > 0.3 (good): {len(good_models)}")

if good_models:
    print(f"\nGood models for SHAP analysis:")
    for biomarker, result in good_models.items():
        print(f"  {biomarker}: R² = {result['r2']:.3f}")

if results:
    print(f"\nAll results:")
    for biomarker, result in results.items():
        quality = "Excellent" if result['r2'] > 0.5 else "Good" if result['r2'] > 0.3 else "Moderate" if result['r2'] > 0.1 else "Poor"
        print(f"  {biomarker[:40]:40} R² = {result['r2']:6.3f} ({quality})")

print(f"\nNext steps:")
print(f"1. Focus on biomarkers with R² > 0.3 for detailed SHAP analysis")
print(f"2. Try XGBoost and other algorithms")
print(f"3. Add hyperparameter optimization")
print(f"4. Implement temporal validation with DLNM")