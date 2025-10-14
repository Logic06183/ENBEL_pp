#!/usr/bin/env python3
"""
CD4-Heat Analysis (Fixed Version)
=================================

Focus on climate-only analysis with comprehensive SHAP investigation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CD4-HEAT RELATIONSHIP: DETAILED ANALYSIS")
print("=" * 70)

# Load data
clinical_df = pd.read_csv('data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', low_memory=False)
print(f"Loaded clinical data: {clinical_df.shape}")

cd4_col = 'CD4 cell count (cells/µL)'

# Get complete CD4 analysis
print(f"\n{'='*50}")
print("1. CD4 CLINICAL PROFILE")
print("="*50)

cd4_data = clinical_df[cd4_col].dropna()
print(f"CD4 samples: {len(cd4_data):,} ({len(cd4_data)/len(clinical_df)*100:.1f}%)")
print(f"Mean: {cd4_data.mean():.0f} cells/µL")
print(f"Median: {cd4_data.median():.0f} cells/µL") 
print(f"Range: {cd4_data.min():.0f} - {cd4_data.max():.0f} cells/µL")

# Check for extreme outliers
q99 = cd4_data.quantile(0.99)
extreme_high = (cd4_data > q99).sum()
print(f"Values > 99th percentile ({q99:.0f}): {extreme_high} ({extreme_high/len(cd4_data)*100:.1f}%)")

# Clinical interpretation
severe = (cd4_data < 200).sum()
moderate = ((cd4_data >= 200) & (cd4_data < 350)).sum()  
mild = ((cd4_data >= 350) & (cd4_data < 500)).sum()
normal = (cd4_data >= 500).sum()

print(f"\nImmune status distribution:")
print(f"  Severe immunodeficiency (<200): {severe} ({severe/len(cd4_data)*100:.1f}%)")
print(f"  Moderate immunodeficiency (200-349): {moderate} ({moderate/len(cd4_data)*100:.1f}%)")
print(f"  Mild immunodeficiency (350-499): {mild} ({mild/len(cd4_data)*100:.1f}%)")
print(f"  Normal (≥500): {normal} ({normal/len(cd4_data)*100:.1f}%)")

# This is concerning - 63.4% have immunodeficiency!
immunodef_total = severe + moderate + mild
print(f"  Total immunodeficient: {immunodef_total} ({immunodef_total/len(cd4_data)*100:.1f}%)")

if immunodef_total/len(cd4_data) > 0.5:
    print("⚠️  WARNING: >50% of patients are immunocompromised!")
    print("    This suggests HIV+ population or specific clinical cohort")

# Climate features
climate_features = [col for col in clinical_df.columns 
                   if any(pattern in col for pattern in ['climate_', 'HEAT_']) 
                   and clinical_df[col].dtype in [np.float64, np.int64]
                   and clinical_df[col].notna().sum() > len(clinical_df) * 0.5]

print(f"\n{'='*50}")
print("2. CLIMATE-CD4 MODELING")
print("="*50)

# Prepare modeling data
model_cols = climate_features + [cd4_col]
complete_df = clinical_df[model_cols].dropna()
print(f"Complete cases: {len(complete_df):,}")

X = complete_df[climate_features]
y = complete_df[cd4_col]

print(f"Modeling data:")
print(f"  Climate features: {len(climate_features)}")
print(f"  CD4 range: {y.min():.0f} - {y.max():.0f} cells/µL")
print(f"  CD4 mean: {y.mean():.0f} cells/µL")

# Check for data leakage warning signs
print(f"\n{'='*50}")
print("3. DATA LEAKAGE INVESTIGATION")
print("="*50)

# Check correlations between features and target
correlations = []
for feature in climate_features:
    corr = X[feature].corr(y)
    if not np.isnan(corr):
        correlations.append((feature, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)

print(f"Feature-CD4 correlations (top 10):")
for feature, corr in correlations[:10]:
    warning = "⚠️" if corr > 0.7 else "🔍" if corr > 0.5 else ""
    print(f"  {feature:<40} r = {corr:.3f} {warning}")

# Check HEAT_VULNERABILITY_SCORE specifically
heat_vuln_corr = X['HEAT_VULNERABILITY_SCORE'].corr(y) if 'HEAT_VULNERABILITY_SCORE' in X.columns else 0
print(f"\nHEAT_VULNERABILITY_SCORE correlation: r = {heat_vuln_corr:.3f}")

if abs(heat_vuln_corr) > 0.7:
    print("🚨 VERY HIGH correlation - possible data leakage!")
elif abs(heat_vuln_corr) > 0.5:
    print("⚠️  HIGH correlation - investigate vulnerability score construction")
else:
    print("✅ Reasonable correlation")

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train models
print(f"\n{'='*50}")
print("4. MODEL PERFORMANCE COMPARISON")
print("="*50)

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5, 
        random_state=42, n_jobs=1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1, 
        random_state=42
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'r2': r2,
        'mae': mae
    }
    
    print(f"{name}:")
    print(f"  R² = {r2:.3f}")
    print(f"  MAE = {mae:.1f} cells/µL")

# Select best model
best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['r2']

print(f"\nBest model: {best_model_name} (R² = {best_r2:.3f})")

# SHAP Analysis
print(f"\n{'='*50}")
print("5. COMPREHENSIVE SHAP ANALYSIS")
print("="*50)

print(f"Performing SHAP analysis on {best_model_name}...")

explainer = shap.TreeExplainer(best_model)
shap_sample_size = min(50, len(X_test))
X_test_sample = X_test.iloc[:shap_sample_size]
shap_values = explainer.shap_values(X_test_sample)

print(f"SHAP values calculated for {shap_sample_size} samples")

# Feature importance
shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': climate_features,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print(f"\nSHAP Feature Importance:")
print(f"{'Rank':<4} {'Feature':<40} {'SHAP Importance':<15} {'% Total':<8}")
print(f"{'-'*75}")

total_importance = shap_importance.sum()
for i, (_, row) in enumerate(shap_df.head(10).iterrows(), 1):
    pct = (row['shap_importance'] / total_importance) * 100
    print(f"{i:<4} {row['feature']:<40} {row['shap_importance']:<15.4f} {pct:>6.1f}%")

# Expected value
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]
print(f"\nBaseline CD4 prediction: {expected_value:.0f} cells/µL")

# Analyze top feature dominance
top_feature = shap_df.iloc[0]['feature']
top_importance = shap_df.iloc[0]['shap_importance']
top_percentage = (top_importance / total_importance) * 100

print(f"\n🔍 TOP FEATURE ANALYSIS:")
print(f"Feature: {top_feature}")
print(f"SHAP Importance: {top_importance:.4f}")
print(f"Percentage of total: {top_percentage:.1f}%")

if top_percentage > 80:
    print("🚨 EXTREME DOMINANCE (>80%) - Likely data leakage!")
    print("   Action: Investigate feature construction")
elif top_percentage > 60:
    print("⚠️  HIGH DOMINANCE (>60%) - Check for confounding")
    print("   Action: Validate relationship independently")
else:
    print("✅ Reasonable feature importance distribution")

# SHAP value distribution analysis
print(f"\n{'='*50}")
print("6. SHAP VALUE PATTERNS")
print("="*50)

top_feature_idx = climate_features.index(top_feature)
top_shap_values = shap_values[:, top_feature_idx]

print(f"SHAP value distribution for {top_feature}:")
print(f"  Mean impact: {np.mean(top_shap_values):.2f}")
print(f"  Std deviation: {np.std(top_shap_values):.2f}")
print(f"  Range: {np.min(top_shap_values):.2f} to {np.max(top_shap_values):.2f}")

# Check for extreme SHAP values
extreme_positive = (top_shap_values > np.mean(top_shap_values) + 2*np.std(top_shap_values)).sum()
extreme_negative = (top_shap_values < np.mean(top_shap_values) - 2*np.std(top_shap_values)).sum()

print(f"  Extreme positive impacts (>2σ): {extreme_positive} ({extreme_positive/len(top_shap_values)*100:.1f}%)")
print(f"  Extreme negative impacts (<-2σ): {extreme_negative} ({extreme_negative/len(top_shap_values)*100:.1f}%)")

# Analyze prediction accuracy by CD4 level
print(f"\n{'='*50}")
print("7. PREDICTION ACCURACY BY CD4 LEVEL")
print("="*50)

predictions = best_model.predict(X_test)
cd4_ranges = [
    (0, 200, "Severe immunodeficiency"),
    (200, 350, "Moderate immunodeficiency"),
    (350, 500, "Mild immunodeficiency"), 
    (500, 2000, "Normal/High")
]

print(f"{'CD4 Range':<25} {'N':<5} {'Actual Mean':<12} {'Pred Mean':<12} {'MAE':<8} {'R²':<8}")
print(f"{'-'*80}")

for low, high, label in cd4_ranges:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() > 3:
        actual_mean = y_test[mask].mean()
        pred_mean = predictions[mask].mean()
        range_mae = mean_absolute_error(y_test[mask], predictions[mask])
        range_r2 = r2_score(y_test[mask], predictions[mask]) if mask.sum() > 5 else np.nan
        
        print(f"{label:<25} {mask.sum():<5} {actual_mean:<12.0f} {pred_mean:<12.0f} {range_mae:<8.1f} {range_r2:<8.3f}")

# Temporal patterns
print(f"\n{'='*50}")
print("8. TEMPORAL VALIDATION")
print("="*50)

if 'year' in clinical_df.columns:
    # Check CD4 by year
    yearly_data = clinical_df.groupby('year')[cd4_col].agg(['count', 'mean', 'std']).dropna()
    
    print(f"CD4 levels by year:")
    print(f"{'Year':<6} {'Count':<6} {'Mean CD4':<10} {'Std':<8}")
    print(f"{'-'*35}")
    
    for year, row in yearly_data.iterrows():
        if row['count'] > 10:
            print(f"{year:<6} {row['count']:<6.0f} {row['mean']:<10.0f} {row['std']:<8.0f}")
    
    # Check for suspicious patterns
    year_means = yearly_data['mean'].dropna()
    if len(year_means) > 1:
        year_trend = np.corrcoef(yearly_data.index, year_means)[0,1]
        print(f"\nTemporal trend: r = {year_trend:.3f}")
        
        if abs(year_trend) > 0.5:
            print("⚠️  Strong temporal trend detected - may affect model validity")

# Summary and recommendations
print(f"\n{'='*70}")
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)

print(f"📊 MODEL PERFORMANCE:")
print(f"   Best R² = {best_r2:.3f} ({best_model_name})")
print(f"   Quality: {'Excellent' if best_r2 > 0.5 else 'Good' if best_r2 > 0.3 else 'Moderate'}")

print(f"\n🔍 DATA QUALITY ASSESSMENT:")
if immunodef_total/len(cd4_data) > 0.6:
    print(f"   ⚠️  Study population: {immunodef_total/len(cd4_data)*100:.0f}% immunocompromised")
    print(f"      → HIV+ cohort likely - heat sensitivity expected")

if top_percentage > 70:
    print(f"   🚨 Feature dominance: {top_feature} ({top_percentage:.0f}%)")
    print(f"      → Investigate vulnerability score construction")
else:
    print(f"   ✅ Feature importance: Reasonable distribution")

print(f"\n🌡️ HEAT-HEALTH RELATIONSHIP:")
if best_r2 > 0.5 and top_percentage < 80:
    print(f"   ✅ Strong, potentially genuine heat-CD4 relationship")
    print(f"   ✅ Clinical plausibility: Heat stress affects immune function")
elif top_percentage > 80:
    print(f"   ⚠️  Relationship dominated by vulnerability score")
    print(f"   → Need to decompose vulnerability index")
else:
    print(f"   ✅ Moderate heat-health relationship detected")

print(f"\n📋 NEXT STEPS:")
print(f"   1. 🔍 Investigate HEAT_VULNERABILITY_SCORE components")
print(f"   2. 📊 Run DLNM analysis for temporal validation")
print(f"   3. 🧪 Validate with external HIV cohort if available")
print(f"   4. 📝 Document clinical context (HIV status, medications)")

if best_r2 > 0.3:
    print(f"   5. ✅ Proceed with publication-ready analysis")
else:
    print(f"   5. 🔄 Consider model improvements")