#!/usr/bin/env python3
"""
Focused test on creatinine biomarker model
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
print("CREATININE BIOMARKER MODEL ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_csv('data/raw/clinical_dataset.csv', low_memory=False)
print(f"1. Loaded data: {df.shape}")

# Check available creatinine columns
creatinine_cols = [col for col in df.columns if 'creat' in col.lower()]
print(f"\n2. Available creatinine columns:")
for col in creatinine_cols:
    non_missing = df[col].notna().sum()
    print(f"   {col}: {non_missing} non-missing values")

# Use the main creatinine column
target = 'Creatinine (mg/dL)'
if target not in df.columns:
    print(f"   ✗ {target} not found, trying alternatives...")
    # Try other possible names
    alternatives = [col for col in creatinine_cols if 'mg' in col or 'umol' in col]
    if alternatives:
        target = alternatives[0]
        print(f"   Using: {target}")
    else:
        print("   ✗ No suitable creatinine column found")
        exit()

# Check target data
target_data = df[target].dropna()
print(f"\n3. Creatinine data analysis:")
print(f"   Non-missing values: {len(target_data)}")
print(f"   Mean: {target_data.mean():.3f} mg/dL")
print(f"   Std: {target_data.std():.3f} mg/dL")
print(f"   Range: {target_data.min():.3f} - {target_data.max():.3f} mg/dL")
print(f"   Normal range typically: 0.6-1.2 mg/dL")

# Clinical interpretation
normal_range = (target_data >= 0.6) & (target_data <= 1.2)
elevated = target_data > 1.2
low = target_data < 0.6

print(f"   Normal (0.6-1.2): {normal_range.sum()} ({normal_range.mean()*100:.1f}%)")
print(f"   Elevated (>1.2): {elevated.sum()} ({elevated.mean()*100:.1f}%)")
print(f"   Low (<0.6): {low.sum()} ({low.mean()*100:.1f}%)")

# Get climate features - focus on renal-relevant ones
print(f"\n4. Selecting climate features...")

# All climate features
climate_cols = []
for col in df.columns:
    if any(pattern in col.lower() for pattern in ['temp', 'humid', 'heat', 'wind']):
        if df[col].dtype in [np.float64, np.int64]:
            non_missing = df[col].notna().sum()
            variance = df[col].var()
            if non_missing > 5000 and variance > 0:
                climate_cols.append(col)

# Filter out confidence scores and focus on actual climate variables
actual_climate = [col for col in climate_cols if 'confidence' not in col.lower()]
print(f"   Total climate features: {len(climate_cols)}")
print(f"   Actual climate (no confidence): {len(actual_climate)}")

# Prioritize features relevant to kidney function
kidney_relevant = []
for col in actual_climate:
    if any(pattern in col.lower() for pattern in [
        'temp', 'heat', 'dehydration', 'stress',  # Heat stress affects kidneys
        'lag', 'rolling',  # Delayed effects
        'humidity',  # Dehydration risk
        'apparent_temp', 'wet_bulb'  # Heat stress indices
    ]):
        kidney_relevant.append(col)

print(f"   Kidney-relevant features: {len(kidney_relevant)}")

# Use kidney-relevant features or fall back to all climate
features_to_use = kidney_relevant if len(kidney_relevant) > 20 else actual_climate
print(f"   Using {len(features_to_use)} features for modeling")

if len(features_to_use) > 0:
    print(f"   Example features:")
    for feat in features_to_use[:5]:
        print(f"     - {feat}")

# Get complete cases
print(f"\n5. Preparing model data...")
all_cols = features_to_use + [target]
complete_df = df[all_cols].dropna()
print(f"   Complete cases: {len(complete_df)} (from {len(df)} total)")
print(f"   Data retention: {len(complete_df)/len(df)*100:.1f}%")

if len(complete_df) < 50:
    print("   ✗ Insufficient complete data")
    exit()

# Prepare data
X = complete_df[features_to_use]
y = complete_df[target]

print(f"   Features shape: {X.shape}")
print(f"   Target range in model data: {y.min():.3f} - {y.max():.3f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Train model with hyperparameters tuned for small dataset
model = RandomForestRegressor(
    n_estimators=200,  # More trees for stability
    max_depth=8,       # Prevent overfitting on small data
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=1
)

print(f"\n6. Training creatinine model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"   Model performance:")
print(f"   R² = {r2:.3f}")
print(f"   MAE = {mae:.3f} mg/dL")

# Clinical significance
mae_percent = (mae / y.mean()) * 100
print(f"   MAE as % of mean: {mae_percent:.1f}%")

# Quality assessment
if r2 > 0.3:
    quality = "Good - suitable for SHAP"
elif r2 > 0.1:
    quality = "Moderate - some signal"
else:
    quality = "Poor - insufficient signal"

print(f"   Model quality: {quality}")

# Feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': features_to_use,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n7. Top climate predictors for creatinine:")
print(f"   {'Feature':<40} {'Importance':<12} {'Type'}")
print(f"   {'-'*60}")

for i, row in importance_df.head(10).iterrows():
    feature = row['feature']
    importance = row['importance']
    
    # Categorize feature
    if 'temp' in feature.lower():
        feat_type = "Temperature"
    elif 'humid' in feature.lower():
        feat_type = "Humidity"
    elif 'heat' in feature.lower():
        feat_type = "Heat stress"
    elif 'wind' in feature.lower():
        feat_type = "Wind"
    elif 'lag' in feature.lower():
        feat_type = "Lag effect"
    else:
        feat_type = "Other"
    
    print(f"   {feature:<40} {importance:<12.4f} {feat_type}")

# SHAP analysis if model is good enough
if r2 > 0.1:  # Lower threshold for creatinine
    print(f"\n8. SHAP analysis for creatinine model...")
    
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for test set
        test_subset = X_test.iloc[:min(50, len(X_test))]  # Use subset for speed
        shap_values = explainer.shap_values(test_subset)
        
        print(f"   SHAP values calculated for {len(test_subset)} samples")
        
        # Feature importance from SHAP
        feature_importance_shap = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'feature': features_to_use,
            'shap_importance': feature_importance_shap
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\n   Top SHAP predictors:")
        for i, row in shap_importance_df.head(5).iterrows():
            print(f"     {row['feature']}: {row['shap_importance']:.4f}")
        
        # Expected value
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
        print(f"   Baseline creatinine: {expected_value:.3f} mg/dL")
        
        # Clinical insights
        print(f"\n9. Clinical insights:")
        top_feature = shap_importance_df.iloc[0]['feature']
        print(f"   Most predictive climate factor: {top_feature}")
        
        # Check for heat-related features in top 5
        heat_features = [f for f in shap_importance_df.head(5)['feature'] 
                        if any(term in f.lower() for term in ['temp', 'heat', 'apparent'])]
        
        if heat_features:
            print(f"   Heat-related predictors: {len(heat_features)}")
            print(f"     → Suggests heat stress affects kidney function")
        
        # Check for lag effects
        lag_features = [f for f in shap_importance_df.head(5)['feature'] 
                       if 'lag' in f.lower()]
        
        if lag_features:
            print(f"   Lag effects detected: {len(lag_features)}")
            print(f"     → Suggests delayed climate impact on kidneys")
        
    except Exception as e:
        print(f"   SHAP analysis failed: {e}")

else:
    print(f"\n8. Model R² too low for reliable SHAP analysis")

print(f"\n" + "=" * 60)
print("CREATININE MODEL SUMMARY")
print("=" * 60)
print(f"✓ Dataset: {len(complete_df)} complete samples")
print(f"✓ Model: R² = {r2:.3f}, MAE = {mae:.3f} mg/dL")
print(f"✓ Quality: {quality}")

if r2 > 0.1:
    print(f"✓ Climate-kidney relationships detected")
    print(f"✓ Top predictor: {importance_df.iloc[0]['feature']}")
else:
    print(f"⚠ Weak climate signal - may need:")
    print(f"  - More sophisticated features")
    print(f"  - Interaction terms")
    print(f"  - Different modeling approach")

print(f"\nClinical relevance:")
print(f"- Normal creatinine: 0.6-1.2 mg/dL")
print(f"- Model error: ±{mae:.3f} mg/dL")
print(f"- Clinically significant if MAE < 0.2 mg/dL")

if mae < 0.2:
    print(f"✓ Model error within clinically acceptable range")
else:
    print(f"⚠ Model error may be too large for clinical use")