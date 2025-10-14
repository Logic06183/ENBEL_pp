#!/usr/bin/env python3
"""
Complete Pipeline for New HEAT Export Dataset v1.0
==================================================

This script runs the full analysis pipeline with the new cleaned dataset:
1. Load and validate new clinical + GCRO data
2. Run imputation with proper methodology
3. Test all biomarkers systematically
4. Perform SHAP analysis on good models
5. Generate comprehensive results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ENBEL CLIMATE-HEALTH ANALYSIS - NEW DATASET PIPELINE")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# STEP 1: Load and validate data
print(f"\n{'='*50}")
print("STEP 1: DATA LOADING AND VALIDATION")
print("="*50)

# Load clinical data
clinical_df = pd.read_csv('data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', low_memory=False)
print(f"✓ Clinical data loaded: {clinical_df.shape}")

# Load GCRO socioeconomic data
gcro_df = pd.read_csv('data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv', low_memory=False)
print(f"✓ GCRO data loaded: {gcro_df.shape}")

# Check data quality
print(f"\nData quality check:")
print(f"  Clinical coverage: {clinical_df.notna().sum().sum() / (clinical_df.shape[0] * clinical_df.shape[1]) * 100:.1f}%")
print(f"  GCRO coverage: {gcro_df.notna().sum().sum() / (gcro_df.shape[0] * gcro_df.shape[1]) * 100:.1f}%")

# STEP 2: Identify available features
print(f"\n{'='*50}")
print("STEP 2: FEATURE IDENTIFICATION")
print("="*50)

# Biomarkers in new format
biomarker_mapping = {
    'immune': ['CD4 cell count (cells/µL)'],
    'metabolic': ['fasting_glucose_mmol_L'],
    'cardiovascular': ['systolic_blood_pressure', 'diastolic_blood_pressure'],
    'renal': ['creatinine_umol_L', 'creatinine clearance'],
    'lipid': ['total_cholesterol_mg_dL', 'hdl_cholesterol_mg_dL', 'ldl_cholesterol_mg_dL'],
    'hematological': ['hemoglobin_g_dL']
}

# Find available biomarkers
available_biomarkers = {}
for system, markers in biomarker_mapping.items():
    available = [m for m in markers if m in clinical_df.columns]
    if available:
        available_biomarkers[system] = available

print(f"Available biomarkers by system:")
total_biomarkers = 0
for system, markers in available_biomarkers.items():
    print(f"  {system.capitalize()}: {len(markers)} biomarkers")
    for marker in markers:
        non_missing = clinical_df[marker].notna().sum()
        coverage = (non_missing / len(clinical_df)) * 100
        print(f"    - {marker}: {non_missing} ({coverage:.1f}%)")
        total_biomarkers += 1

# Climate features
climate_patterns = ['climate_', 'HEAT_']
climate_features = []
for col in clinical_df.columns:
    if any(pattern in col for pattern in climate_patterns):
        if clinical_df[col].dtype in [np.float64, np.int64]:
            non_missing = clinical_df[col].notna().sum()
            if non_missing > len(clinical_df) * 0.5:  # >50% coverage
                climate_features.append(col)

print(f"\nClimate features: {len(climate_features)} with >50% coverage")
for feat in climate_features:
    coverage = clinical_df[feat].notna().sum() / len(clinical_df) * 100
    print(f"  {feat}: {coverage:.1f}%")

# STEP 3: Simple imputation (KNN for now)
print(f"\n{'='*50}")
print("STEP 3: DATA IMPUTATION")
print("="*50)

print(f"Implementing KNN imputation for missing values...")

# For each biomarker system, create imputed dataset
imputed_results = {}

for system, markers in available_biomarkers.items():
    print(f"\nProcessing {system} system...")
    
    for marker in markers:
        print(f"  Imputing {marker}...")
        
        # Select features for this biomarker
        feature_cols = climate_features + [marker]
        biomarker_df = clinical_df[feature_cols].copy()
        
        # Check data availability
        complete_cases = biomarker_df.dropna().shape[0]
        total_cases = biomarker_df.shape[0]
        
        print(f"    Complete cases: {complete_cases}/{total_cases} ({complete_cases/total_cases*100:.1f}%)")
        
        if complete_cases < 50:
            print(f"    ⚠ Insufficient data - skipping")
            continue
        
        # Separate target from features
        target_data = biomarker_df[marker].copy()
        feature_data = biomarker_df[climate_features].copy()
        
        # Impute climate features first
        climate_missing = feature_data.isnull().sum().sum()
        if climate_missing > 0:
            print(f"    Imputing {climate_missing} missing climate values...")
            climate_imputer = KNNImputer(n_neighbors=5)
            feature_imputed = pd.DataFrame(
                climate_imputer.fit_transform(feature_data),
                columns=climate_features,
                index=feature_data.index
            )
        else:
            feature_imputed = feature_data
        
        # Store results
        imputed_results[marker] = {
            'features': feature_imputed,
            'target': target_data,
            'complete_cases': complete_cases,
            'system': system
        }
        
        print(f"    ✓ Imputation complete")

print(f"\n✓ Imputed data ready for {len(imputed_results)} biomarkers")

# STEP 4: Model training and evaluation
print(f"\n{'='*50}")
print("STEP 4: MACHINE LEARNING ANALYSIS")
print("="*50)

model_results = {}
successful_models = 0

for marker, data in imputed_results.items():
    print(f"\nTraining model for {marker}...")
    
    # Get complete cases only
    features = data['features']
    target = data['target']
    
    # Remove rows where target is missing
    complete_mask = target.notna()
    X = features[complete_mask]
    y = target[complete_mask]
    
    if len(X) < 50:
        print(f"  ⚠ Insufficient data: {len(X)} samples")
        continue
    
    print(f"  Training samples: {len(X)}")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  R² = {r2:.3f}, MAE = {mae:.3f}")
    
    # Quality assessment
    if r2 > 0.3:
        quality = "Excellent"
    elif r2 > 0.1:
        quality = "Good"
    elif r2 > 0.05:
        quality = "Moderate"
    else:
        quality = "Poor"
    
    print(f"  Quality: {quality}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': climate_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Store results
    model_results[marker] = {
        'model': model,
        'r2': r2,
        'mae': mae,
        'quality': quality,
        'n_samples': len(X),
        'system': data['system'],
        'feature_importance': importance_df,
        'X_test': X_test,
        'y_test': y_test
    }
    
    if r2 > 0.05:
        successful_models += 1
    
    # Show top features
    print(f"  Top 3 climate predictors:")
    for i, row in importance_df.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

print(f"\n✓ Trained {len(model_results)} models ({successful_models} with R² > 0.05)")

# STEP 5: SHAP Analysis for good models
print(f"\n{'='*50}")
print("STEP 5: SHAP EXPLAINABILITY ANALYSIS")
print("="*50)

shap_results = {}
models_with_shap = 0

for marker, results in model_results.items():
    if results['r2'] > 0.1:  # Only analyze decent models
        print(f"\nSHAP analysis for {marker} (R² = {results['r2']:.3f})...")
        
        try:
            model = results['model']
            X_test = results['X_test']
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (subset for speed)
            test_subset = X_test.iloc[:min(30, len(X_test))]
            shap_values = explainer.shap_values(test_subset)
            
            # Feature importance from SHAP
            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': climate_features,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            # Store SHAP results
            shap_results[marker] = {
                'shap_values': shap_values,
                'feature_importance': shap_df,
                'baseline': explainer.expected_value,
                'n_samples': len(test_subset)
            }
            
            models_with_shap += 1
            
            print(f"  ✓ SHAP values calculated for {len(test_subset)} samples")
            print(f"  Top 3 SHAP predictors:")
            for i, row in shap_df.head(3).iterrows():
                print(f"    {row['feature']}: {row['shap_importance']:.4f}")
            
            # Expected value
            expected = explainer.expected_value
            if isinstance(expected, np.ndarray):
                expected = expected[0]
            print(f"  Baseline prediction: {expected:.2f}")
            
        except Exception as e:
            print(f"  ⚠ SHAP analysis failed: {e}")
    
    else:
        print(f"Skipping SHAP for {marker} (R² = {results['r2']:.3f} too low)")

print(f"\n✓ SHAP analysis completed for {models_with_shap} models")

# STEP 6: Generate comprehensive report
print(f"\n{'='*50}")
print("STEP 6: RESULTS SUMMARY")
print("="*50)

print(f"\nCOMPREHENSIVE ANALYSIS RESULTS")
print(f"=" * 40)
print(f"Dataset: HEAT Export v1.0")
print(f"Clinical records: {len(clinical_df):,}")
print(f"GCRO records: {len(gcro_df):,}")
print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}")

print(f"\nBIOMARKER MODEL PERFORMANCE:")
print(f"{'Biomarker':<35} {'System':<15} {'R²':<8} {'Quality':<12} {'SHAP'}")
print(f"{'-'*80}")

# Sort by R² score
sorted_results = sorted(model_results.items(), key=lambda x: x[1]['r2'], reverse=True)

for marker, results in sorted_results:
    shap_available = "✓" if marker in shap_results else "✗"
    print(f"{marker:<35} {results['system']:<15} {results['r2']:<8.3f} {results['quality']:<12} {shap_available}")

print(f"\nTOP CLIMATE PREDICTORS (across all models):")
all_features = {}
for marker, results in model_results.items():
    for _, row in results['feature_importance'].head(5).iterrows():
        feature = row['feature']
        importance = row['importance']
        if feature in all_features:
            all_features[feature] += importance
        else:
            all_features[feature] = importance

# Sort by total importance
sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
print(f"{'Feature':<40} {'Total Importance'}")
print(f"{'-'*60}")
for feature, importance in sorted_features[:10]:
    print(f"{feature:<40} {importance:.4f}")

print(f"\nKEY FINDINGS:")
excellent_models = [k for k, v in model_results.items() if v['r2'] > 0.3]
good_models = [k for k, v in model_results.items() if v['r2'] > 0.1]

if excellent_models:
    print(f"✓ {len(excellent_models)} biomarker(s) show excellent climate relationships (R² > 0.3)")
    for marker in excellent_models:
        print(f"  - {marker}: R² = {model_results[marker]['r2']:.3f}")

if good_models:
    print(f"✓ {len(good_models)} biomarker(s) show good climate relationships (R² > 0.1)")

if models_with_shap > 0:
    print(f"✓ {models_with_shap} model(s) suitable for detailed SHAP interpretation")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_summary = {
    'timestamp': timestamp,
    'dataset_info': {
        'clinical_records': len(clinical_df),
        'gcro_records': len(gcro_df),
        'climate_features': len(climate_features),
        'biomarkers_analyzed': len(model_results)
    },
    'model_performance': {marker: {'r2': v['r2'], 'mae': v['mae'], 'quality': v['quality']} 
                         for marker, v in model_results.items()},
    'top_climate_features': dict(sorted_features[:10])
}

import json
with open(f'results/new_dataset_analysis_{timestamp}.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Results saved to: results/new_dataset_analysis_{timestamp}.json")

print(f"\n" + "=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"✓ New dataset integrated and analyzed")
print(f"✓ {len(model_results)} biomarker models trained")
print(f"✓ {models_with_shap} models with SHAP interpretation")
print(f"✓ Results saved with timestamp: {timestamp}")
print(f"\nNext steps:")
print(f"1. Review high-performing models in detail")
print(f"2. Run DLNM validation for top models")
print(f"3. Generate publication-ready visualizations")
print(f"4. Conduct sensitivity analyses")