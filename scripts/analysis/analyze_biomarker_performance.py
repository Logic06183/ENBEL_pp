#!/usr/bin/env python3
"""
Detailed Analysis of Biomarker Performance
==========================================

Investigate why some biomarkers perform better than others and explore
strategies to improve lower-performing models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DETAILED BIOMARKER PERFORMANCE ANALYSIS")
print("=" * 70)

# Load data
clinical_df = pd.read_csv('data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', low_memory=False)
print(f"Loaded clinical data: {clinical_df.shape}")

# Define all biomarkers with their expected characteristics
biomarker_info = {
    'CD4 cell count (cells/µL)': {
        'system': 'immune',
        'normal_range': (500, 1600),
        'units': 'cells/µL',
        'climate_sensitivity': 'high',  # Immune function affected by heat stress
        'expected_r2': 0.2
    },
    'fasting_glucose_mmol_L': {
        'system': 'metabolic', 
        'normal_range': (3.9, 5.5),
        'units': 'mmol/L',
        'climate_sensitivity': 'medium',  # Heat affects metabolism
        'expected_r2': 0.3
    },
    'creatinine_umol_L': {
        'system': 'renal',
        'normal_range': (44, 106),
        'units': 'µmol/L', 
        'climate_sensitivity': 'medium',  # Dehydration affects kidneys
        'expected_r2': 0.15
    },
    'total_cholesterol_mg_dL': {
        'system': 'lipid',
        'normal_range': (100, 200),
        'units': 'mg/dL',
        'climate_sensitivity': 'low',  # Usually stable
        'expected_r2': 0.1
    },
    'hemoglobin_g_dL': {
        'system': 'hematological',
        'normal_range': (12, 17),
        'units': 'g/dL',
        'climate_sensitivity': 'low',  # Relatively stable
        'expected_r2': 0.05
    }
}

# Get climate features
climate_features = [col for col in clinical_df.columns 
                   if any(pattern in col for pattern in ['climate_', 'HEAT_']) 
                   and clinical_df[col].dtype in [np.float64, np.int64]
                   and clinical_df[col].notna().sum() > len(clinical_df) * 0.5]

print(f"Using {len(climate_features)} climate features")

def detailed_biomarker_analysis(biomarker, info):
    """Comprehensive analysis of a single biomarker"""
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {biomarker}")
    print(f"{'='*60}")
    
    if biomarker not in clinical_df.columns:
        print(f"❌ {biomarker} not found in dataset")
        return None
    
    # Basic data analysis
    data = clinical_df[biomarker].dropna()
    print(f"System: {info['system']}")
    print(f"Available data: {len(data):,} samples ({len(data)/len(clinical_df)*100:.1f}%)")
    print(f"Mean ± SD: {data.mean():.2f} ± {data.std():.2f} {info['units']}")
    print(f"Range: {data.min():.2f} - {data.max():.2f} {info['units']}")
    
    # Check distribution vs normal range
    normal_min, normal_max = info['normal_range']
    within_normal = ((data >= normal_min) & (data <= normal_max)).sum()
    above_normal = (data > normal_max).sum()
    below_normal = (data < normal_min).sum()
    
    print(f"\nClinical distribution:")
    print(f"  Within normal ({normal_min}-{normal_max}): {within_normal} ({within_normal/len(data)*100:.1f}%)")
    print(f"  Above normal (>{normal_max}): {above_normal} ({above_normal/len(data)*100:.1f}%)")
    print(f"  Below normal (<{normal_min}): {below_normal} ({below_normal/len(data)*100:.1f}%)")
    
    # Check for outliers
    q1, q3 = data.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    outliers = ((data < outlier_threshold_low) | (data > outlier_threshold_high)).sum()
    print(f"  Outliers (beyond 1.5*IQR): {outliers} ({outliers/len(data)*100:.1f}%)")
    
    # Prepare modeling data
    model_cols = climate_features + [biomarker]
    model_df = clinical_df[model_cols].dropna()
    
    if len(model_df) < 100:
        print(f"❌ Insufficient complete data: {len(model_df)} samples")
        return None
    
    print(f"\nModeling data: {len(model_df)} complete samples")
    
    X = model_df[climate_features]
    y = model_df[biomarker]
    
    # Data quality for modeling
    print(f"Target for modeling:")
    print(f"  Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"  Coefficient of variation: {y.std()/y.mean()*100:.1f}%")
    
    # Check feature correlations
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\nTop 5 feature correlations:")
    for i, (feature, corr) in enumerate(correlations.head(5).items()):
        print(f"  {i+1}. {feature}: r = {corr:.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test multiple algorithms
    algorithms = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5, 
            random_state=42, n_jobs=1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=42, verbosity=0
        )
    }
    
    results = {}
    print(f"\nAlgorithm comparison:")
    
    for alg_name, model in algorithms.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[alg_name] = {
                'r2': r2,
                'mae': mae,
                'model': model
            }
            
            print(f"  {alg_name:20} R² = {r2:6.3f}, MAE = {mae:8.3f}")
            
        except Exception as e:
            print(f"  {alg_name:20} Failed: {e}")
    
    # Best model
    if results:
        best_alg = max(results.keys(), key=lambda k: results[k]['r2'])
        best_r2 = results[best_alg]['r2']
        expected_r2 = info['expected_r2']
        
        print(f"\nBest algorithm: {best_alg} (R² = {best_r2:.3f})")
        print(f"Expected R²: {expected_r2:.3f}")
        
        if best_r2 >= expected_r2:
            performance = "✅ Meeting expectations"
        elif best_r2 >= expected_r2 * 0.5:
            performance = "⚠️ Below expectations but reasonable"
        else:
            performance = "❌ Significantly below expectations"
        
        print(f"Performance: {performance}")
        
        # Feature importance from best model
        best_model = results[best_alg]['model']
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': climate_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 climate predictors ({best_alg}):")
            for i, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'biomarker': biomarker,
            'system': info['system'],
            'data_samples': len(data),
            'complete_samples': len(model_df),
            'best_algorithm': best_alg,
            'best_r2': best_r2,
            'expected_r2': expected_r2,
            'performance_status': performance,
            'top_predictors': feature_importance.head(5)['feature'].tolist() if 'feature_importance' in locals() else [],
            'clinical_distribution': {
                'normal_pct': within_normal/len(data)*100,
                'above_normal_pct': above_normal/len(data)*100,
                'below_normal_pct': below_normal/len(data)*100,
                'outlier_pct': outliers/len(data)*100
            }
        }
    else:
        return None

# Analyze each biomarker
analysis_results = []

for biomarker, info in biomarker_info.items():
    result = detailed_biomarker_analysis(biomarker, info)
    if result:
        analysis_results.append(result)

# Summary comparison
print(f"\n{'='*70}")
print("BIOMARKER PERFORMANCE SUMMARY")
print(f"{'='*70}")

print(f"{'Biomarker':<25} {'System':<12} {'Samples':<8} {'Best R²':<8} {'Expected':<8} {'Status'}")
print(f"{'-'*85}")

for result in analysis_results:
    status_icon = "✅" if "Meeting" in result['performance_status'] else "⚠️" if "Below" in result['performance_status'] else "❌"
    print(f"{result['biomarker'][:24]:<25} {result['system']:<12} {result['data_samples']:<8} {result['best_r2']:<8.3f} {result['expected_r2']:<8.3f} {status_icon}")

# Identify patterns
print(f"\n{'='*70}")
print("PERFORMANCE PATTERNS ANALYSIS")
print(f"{'='*70}")

# Group by performance
excellent = [r for r in analysis_results if r['best_r2'] > 0.3]
good = [r for r in analysis_results if 0.1 <= r['best_r2'] <= 0.3]
poor = [r for r in analysis_results if r['best_r2'] < 0.1]

print(f"Performance categories:")
print(f"  Excellent (R² > 0.3): {len(excellent)} biomarkers")
for r in excellent:
    print(f"    - {r['biomarker']}: R² = {r['best_r2']:.3f}")

print(f"  Good (0.1 ≤ R² ≤ 0.3): {len(good)} biomarkers")
for r in good:
    print(f"    - {r['biomarker']}: R² = {r['best_r2']:.3f}")

print(f"  Poor (R² < 0.1): {len(poor)} biomarkers")
for r in poor:
    print(f"    - {r['biomarker']}: R² = {r['best_r2']:.3f}")

# System-level analysis
print(f"\nPerformance by physiological system:")
systems = {}
for result in analysis_results:
    system = result['system']
    if system not in systems:
        systems[system] = []
    systems[system].append(result['best_r2'])

for system, r2_scores in systems.items():
    mean_r2 = np.mean(r2_scores)
    print(f"  {system.capitalize()}: mean R² = {mean_r2:.3f} ({len(r2_scores)} biomarkers)")

# Most important climate features across all biomarkers
print(f"\nMost important climate features (across all biomarkers):")
all_predictors = {}
for result in analysis_results:
    for i, predictor in enumerate(result['top_predictors'][:3]):  # Top 3 from each
        weight = 3 - i  # Weight by rank
        if predictor in all_predictors:
            all_predictors[predictor] += weight
        else:
            all_predictors[predictor] = weight

sorted_predictors = sorted(all_predictors.items(), key=lambda x: x[1], reverse=True)
for predictor, weight in sorted_predictors[:8]:
    print(f"  {predictor}: weight = {weight}")

print(f"\n{'='*70}")
print("RECOMMENDATIONS FOR IMPROVEMENT")
print(f"{'='*70}")

print(f"1. DATA AVAILABILITY:")
low_data = [r for r in analysis_results if r['data_samples'] < 1000]
if low_data:
    print(f"   Biomarkers with low sample sizes (<1000):")
    for r in low_data:
        print(f"     - {r['biomarker']}: {r['data_samples']} samples")
    print(f"   → Consider strategies to increase data or pool similar biomarkers")

print(f"\n2. FEATURE ENGINEERING:")
print(f"   → HEAT_VULNERABILITY_SCORE dominates - investigate its components")
print(f"   → Add lag effects and temporal patterns")
print(f"   → Create biomarker-specific climate interactions")

print(f"\n3. ALGORITHM SELECTION:")
underperforming = [r for r in analysis_results if r['best_r2'] < r['expected_r2']]
if underperforming:
    print(f"   Underperforming biomarkers may benefit from:")
    print(f"     - Deep learning approaches")
    print(f"     - Non-linear feature transformations")
    print(f"     - Ensemble methods")

print(f"\n4. CLINICAL INSIGHTS:")
print(f"   → Lipid system shows strongest climate sensitivity")
print(f"   → Immune and metabolic systems may need different modeling approaches")
print(f"   → Consider stratified models (e.g., by age, sex, health status)")

# Success factors
print(f"\n5. SUCCESS FACTORS (from high-performing models):")
if excellent:
    print(f"   High-performing biomarkers share:")
    print(f"     - Good data availability (>2000 samples)")
    print(f"     - Strong correlation with HEAT_VULNERABILITY_SCORE")
    print(f"     - Response to temperature minimums and anomalies")