#!/usr/bin/env python3
"""
Deep Dive: CD4-Heat Relationship Analysis
=========================================

Comprehensive analysis of the CD4-heat relationship including:
1. Detailed SHAP analysis
2. Confounding factor investigation
3. Clinical validation
4. Temporal pattern analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEEP DIVE: CD4-HEAT RELATIONSHIP ANALYSIS")
print("=" * 70)

# Load data
clinical_df = pd.read_csv('data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', low_memory=False)
print(f"Loaded clinical data: {clinical_df.shape}")

# Focus on CD4 analysis
cd4_col = 'CD4 cell count (cells/µL)'
print(f"\n{'='*50}")
print("1. CD4 DATA QUALITY AND CLINICAL CONTEXT")
print("="*50)

# Basic CD4 analysis
cd4_data = clinical_df[cd4_col].dropna()
print(f"CD4 samples: {len(cd4_data):,} ({len(cd4_data)/len(clinical_df)*100:.1f}%)")
print(f"Mean CD4: {cd4_data.mean():.0f} cells/µL")
print(f"Median CD4: {cd4_data.median():.0f} cells/µL")
print(f"Range: {cd4_data.min():.0f} - {cd4_data.max():.0f} cells/µL")

# Clinical categories for CD4
print(f"\nClinical CD4 categories:")
severe_immunodef = (cd4_data < 200).sum()
moderate_immunodef = ((cd4_data >= 200) & (cd4_data < 350)).sum()
mild_immunodef = ((cd4_data >= 350) & (cd4_data < 500)).sum()
normal_low = ((cd4_data >= 500) & (cd4_data < 800)).sum()
normal_high = (cd4_data >= 800).sum()

print(f"  Severe immunodeficiency (<200): {severe_immunodef} ({severe_immunodef/len(cd4_data)*100:.1f}%)")
print(f"  Moderate immunodef (200-349): {moderate_immunodef} ({moderate_immunodef/len(cd4_data)*100:.1f}%)")
print(f"  Mild immunodef (350-499): {mild_immunodef} ({mild_immunodef/len(cd4_data)*100:.1f}%)")
print(f"  Normal-low (500-799): {normal_low} ({normal_low/len(cd4_data)*100:.1f}%)")
print(f"  Normal-high (≥800): {normal_high} ({normal_high/len(cd4_data)*100:.1f}%)")

# Check for potential confounding variables
print(f"\n{'='*50}")
print("2. CONFOUNDING VARIABLE INVESTIGATION")
print("="*50)

# Check what other variables are available that might confound
demographic_cols = ['Sex', 'Race', 'Age', 'age_group']
clinical_cols = ['study_code', 'study_type', 'from_hiv_study', 'target_population']
temporal_cols = ['primary_date', 'year', 'month', 'season']

available_confounders = []
for col_group, cols in [('demographic', demographic_cols), 
                       ('clinical', clinical_cols), 
                       ('temporal', temporal_cols)]:
    found_cols = [col for col in cols if col in clinical_df.columns]
    if found_cols:
        available_confounders.extend(found_cols)
        print(f"{col_group.capitalize()} variables: {found_cols}")

print(f"\nChecking confounding relationships...")

# Create dataset with CD4 and all available variables
analysis_cols = [cd4_col] + available_confounders
cd4_analysis_df = clinical_df[analysis_cols].copy()

# Check relationships between confounders and CD4
confounder_correlations = []
for col in available_confounders:
    if col in cd4_analysis_df.columns:
        if cd4_analysis_df[col].dtype in ['object', 'category']:
            # For categorical variables, check group differences
            try:
                groups = cd4_analysis_df.groupby(col)[cd4_col].agg(['count', 'mean', 'std']).dropna()
                if len(groups) > 1:
                    print(f"\n{col} groups (CD4 means):")
                    for idx, row in groups.iterrows():
                        if row['count'] > 10:  # Only show groups with substantial data
                            print(f"  {idx}: {row['mean']:.0f} ± {row['std']:.0f} cells/µL (n={row['count']})")
            except:
                pass
        else:
            # For numeric variables, calculate correlation
            corr = cd4_analysis_df[col].corr(cd4_analysis_df[cd4_col])
            if not np.isnan(corr):
                confounder_correlations.append((col, abs(corr)))

if confounder_correlations:
    print(f"\nNumeric variable correlations with CD4:")
    confounder_correlations.sort(key=lambda x: x[1], reverse=True)
    for var, corr in confounder_correlations:
        print(f"  {var}: r = {corr:.3f}")

# Get climate features
climate_features = [col for col in clinical_df.columns 
                   if any(pattern in col for pattern in ['climate_', 'HEAT_']) 
                   and clinical_df[col].dtype in [np.float64, np.int64]
                   and clinical_df[col].notna().sum() > len(clinical_df) * 0.5]

print(f"\n{'='*50}")
print("3. COMPREHENSIVE CD4-CLIMATE MODELING")
print("="*50)

# Prepare modeling data
all_features = climate_features + available_confounders
model_cols = [cd4_col] + all_features

# Get complete cases
complete_df = clinical_df[model_cols].dropna()
print(f"Complete cases: {len(complete_df):,}")

if len(complete_df) < 100:
    print("❌ Insufficient complete data")
    exit()

# Separate climate and confounding features
X_climate = complete_df[climate_features]
X_confounders = complete_df[available_confounders] if available_confounders else pd.DataFrame()
y = complete_df[cd4_col]

print(f"Features: {len(climate_features)} climate + {len(available_confounders)} confounders")
print(f"Target range: {y.min():.0f} - {y.max():.0f} cells/µL")

# Split data
X_all = complete_df[all_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")

# Model 1: Climate features only
print(f"\n{'='*50}")
print("4. MODEL COMPARISON: CLIMATE vs CONFOUNDERS")
print("="*50)

# Climate-only model
X_train_climate = X_train[climate_features]
X_test_climate = X_test[climate_features]

climate_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42
)
climate_model.fit(X_train_climate, y_train)
climate_pred = climate_model.predict(X_test_climate)
climate_r2 = r2_score(y_test, climate_pred)
climate_mae = mean_absolute_error(y_test, climate_pred)

print(f"Climate-only model:")
print(f"  R² = {climate_r2:.3f}")
print(f"  MAE = {climate_mae:.1f} cells/µL")

# Full model (climate + confounders)
full_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42
)
full_model.fit(X_train, y_train)
full_pred = full_model.predict(X_test)
full_r2 = r2_score(y_test, full_pred)
full_mae = mean_absolute_error(y_test, full_pred)

print(f"\nFull model (climate + confounders):")
print(f"  R² = {full_r2:.3f}")
print(f"  MAE = {full_mae:.1f} cells/µL")

# Calculate the contribution of confounders
confounder_contribution = full_r2 - climate_r2
print(f"\nConfounder contribution: ΔR² = {confounder_contribution:.3f}")

if confounder_contribution > 0.1:
    print("⚠️  Significant confounding detected!")
elif confounder_contribution > 0.05:
    print("⚠️  Moderate confounding detected")
else:
    print("✅ Minimal confounding - climate relationship appears genuine")

# Model 3: Confounders only (if available)
if len(available_confounders) > 0:
    try:
        # Handle categorical variables for confounders-only model
        X_train_conf = X_train[available_confounders]
        X_test_conf = X_test[available_confounders]
        
        # Simple preprocessing for categorical variables
        for col in available_confounders:
            if X_train_conf[col].dtype == 'object':
                # Label encode categorical variables
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_train_conf[col] = le.fit_transform(X_train_conf[col].astype(str))
                X_test_conf[col] = le.transform(X_test_conf[col].astype(str))
        
        confounder_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42
        )
        confounder_model.fit(X_train_conf, y_train)
        confounder_pred = confounder_model.predict(X_test_conf)
        confounder_r2 = r2_score(y_test, confounder_pred)
        
        print(f"\nConfounders-only model:")
        print(f"  R² = {confounder_r2:.3f}")
        
    except Exception as e:
        print(f"\nConfounders-only model failed: {e}")

# SHAP Analysis
print(f"\n{'='*50}")
print("5. DETAILED SHAP ANALYSIS")
print("="*50)

print(f"Performing SHAP analysis on best model...")

# Use the climate-only model for cleaner interpretation
explainer = shap.TreeExplainer(climate_model)
shap_values = explainer.shap_values(X_test_climate.iloc[:100])  # Use subset for speed

print(f"SHAP values calculated for {len(X_test_climate.iloc[:100])} samples")

# Feature importance from SHAP
shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': climate_features,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print(f"\nSHAP Feature Importance (Climate-only model):")
print(f"{'Feature':<40} {'SHAP Importance':<15} {'% of Total'}")
print(f"{'-'*70}")

total_shap = shap_importance.sum()
for i, row in shap_df.head(10).iterrows():
    pct = (row['shap_importance'] / total_shap) * 100
    print(f"{row['feature']:<40} {row['shap_importance']:<15.4f} {pct:6.1f}%")

# Expected value (baseline)
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]
print(f"\nBaseline CD4 count: {expected_value:.0f} cells/µL")

# Analyze HEAT_VULNERABILITY_SCORE dominance
heat_vuln_importance = shap_df[shap_df['feature'] == 'HEAT_VULNERABILITY_SCORE']['shap_importance'].iloc[0]
heat_vuln_pct = (heat_vuln_importance / total_shap) * 100

print(f"\n🔍 HEAT_VULNERABILITY_SCORE Analysis:")
print(f"  SHAP Importance: {heat_vuln_importance:.4f}")
print(f"  % of total importance: {heat_vuln_pct:.1f}%")

if heat_vuln_pct > 80:
    print("⚠️  HEAT_VULNERABILITY_SCORE dominates (>80% of importance)")
    print("    This suggests either:")
    print("    1. A genuine strong heat-health relationship, OR")
    print("    2. Potential data leakage/confounding in the vulnerability score")
elif heat_vuln_pct > 60:
    print("⚠️  HEAT_VULNERABILITY_SCORE is dominant (>60% of importance)")
else:
    print("✅ HEAT_VULNERABILITY_SCORE shows reasonable importance")

# Check individual SHAP value distributions
print(f"\n{'='*50}")
print("6. SHAP VALUE DISTRIBUTION ANALYSIS")
print("="*50)

# Top feature SHAP value statistics
top_feature = shap_df.iloc[0]['feature']
top_feature_idx = climate_features.index(top_feature)
top_shap_values = shap_values[:, top_feature_idx]

print(f"SHAP value distribution for {top_feature}:")
print(f"  Mean: {np.mean(top_shap_values):.2f}")
print(f"  Std: {np.std(top_shap_values):.2f}")
print(f"  Range: {np.min(top_shap_values):.2f} to {np.max(top_shap_values):.2f}")

# Check for patients with extreme SHAP values
high_impact_idx = np.where(np.abs(top_shap_values) > np.std(top_shap_values) * 2)[0]
print(f"  Patients with high impact (>2σ): {len(high_impact_idx)} ({len(high_impact_idx)/len(top_shap_values)*100:.1f}%)")

# Clinical validation
print(f"\n{'='*50}")
print("7. CLINICAL VALIDATION")
print("="*50)

# Check if predictions make clinical sense
y_test_subset = y_test.iloc[:100]
predictions = climate_model.predict(X_test_climate.iloc[:100])

# Calculate prediction accuracy by CD4 range
cd4_ranges = [
    (0, 200, "Severe immunodeficiency"),
    (200, 350, "Moderate immunodeficiency"), 
    (350, 500, "Mild immunodeficiency"),
    (500, 1000, "Normal range"),
    (1000, 10000, "High normal")
]

print(f"Prediction accuracy by CD4 range:")
print(f"{'Range':<25} {'N':<5} {'MAE':<10} {'R²':<8}")
print(f"{'-'*50}")

for low, high, label in cd4_ranges:
    mask = (y_test_subset >= low) & (y_test_subset < high)
    if mask.sum() > 5:  # Need enough samples
        range_mae = mean_absolute_error(y_test_subset[mask], predictions[mask])
        range_r2 = r2_score(y_test_subset[mask], predictions[mask])
        print(f"{label:<25} {mask.sum():<5} {range_mae:<10.1f} {range_r2:<8.3f}")

# Temporal analysis if date available
print(f"\n{'='*50}")
print("8. TEMPORAL PATTERN ANALYSIS")
print("="*50)

if 'primary_date' in clinical_df.columns:
    # Convert date and check temporal patterns
    clinical_df['date'] = pd.to_datetime(clinical_df['primary_date'], errors='coerce')
    clinical_df['year'] = clinical_df['date'].dt.year
    clinical_df['month'] = clinical_df['date'].dt.month
    
    temporal_df = clinical_df[[cd4_col, 'year', 'month']].dropna()
    
    if len(temporal_df) > 100:
        print(f"Temporal analysis with {len(temporal_df)} dated samples:")
        
        # CD4 by year
        yearly_cd4 = temporal_df.groupby('year')[cd4_col].agg(['mean', 'count']).dropna()
        if len(yearly_cd4) > 1:
            print(f"\nCD4 trends by year:")
            for year, row in yearly_cd4.iterrows():
                if row['count'] > 10:
                    print(f"  {year}: {row['mean']:.0f} cells/µL (n={row['count']})")
        
        # CD4 by month (seasonal patterns)
        monthly_cd4 = temporal_df.groupby('month')[cd4_col].agg(['mean', 'count']).dropna()
        if len(monthly_cd4) > 1:
            print(f"\nCD4 seasonal patterns (by month):")
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, row in monthly_cd4.iterrows():
                if row['count'] > 5:
                    month_name = months[int(month)-1] if 1 <= month <= 12 else str(month)
                    print(f"  {month_name}: {row['mean']:.0f} cells/µL (n={row['count']})")

print(f"\n{'='*50}")
print("9. SUMMARY AND RECOMMENDATIONS")
print("="*50)

print(f"CD4-Heat Relationship Summary:")
print(f"✓ Climate-only model R² = {climate_r2:.3f}")
print(f"✓ Model performance: {'Excellent' if climate_r2 > 0.5 else 'Good' if climate_r2 > 0.3 else 'Moderate'}")

if confounder_contribution < 0.05:
    print(f"✅ Minimal confounding detected - relationship appears genuine")
else:
    print(f"⚠️  Confounding contribution: ΔR² = {confounder_contribution:.3f}")

print(f"✓ HEAT_VULNERABILITY_SCORE importance: {heat_vuln_pct:.1f}%")

print(f"\nRecommendations:")
if heat_vuln_pct > 80:
    print(f"1. 🔍 Investigate HEAT_VULNERABILITY_SCORE components")
    print(f"2. 🔍 Validate relationship with independent data")
    print(f"3. 📊 Consider DLNM analysis for temporal validation")
else:
    print(f"1. ✅ Relationship appears robust")
    print(f"2. 📊 Proceed with DLNM validation")
    print(f"3. 📝 Strong candidate for publication")

if severe_immunodef > len(cd4_data) * 0.5:
    print(f"4. ⚠️  High proportion of immunocompromised patients - check study population")

print(f"\nNext steps:")
print(f"1. Run DLNM analysis to validate temporal patterns")
print(f"2. Investigate heat vulnerability score construction")
print(f"3. Validate findings in external dataset if available")