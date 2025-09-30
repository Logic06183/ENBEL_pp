#!/usr/bin/env python3
"""
Improved creatinine analysis with feature engineering and clinical focus
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("IMPROVED CREATININE MODEL WITH FEATURE ENGINEERING")
print("=" * 60)

# Load data - use the full dataset with climate lags
df = pd.read_csv('data/raw/full_dataset.csv', low_memory=False)
print(f"1. Loaded data: {df.shape}")

target = 'Creatinine (mg/dL)'
print(f"\n2. Analyzing creatinine clinical patterns...")

# Clinical analysis
creat_data = df[target].dropna()
print(f"   Total samples: {len(creat_data)}")

# Define clinical categories
normal = (creat_data >= 0.6) & (creat_data <= 1.2)
mild_elevation = (creat_data > 1.2) & (creat_data <= 1.5)
moderate_elevation = (creat_data > 1.5) & (creat_data <= 2.0)
severe_elevation = creat_data > 2.0
low_values = creat_data < 0.6

print(f"   Normal (0.6-1.2): {normal.sum()} ({normal.mean()*100:.1f}%)")
print(f"   Mild elevation (1.2-1.5): {mild_elevation.sum()} ({mild_elevation.mean()*100:.1f}%)")
print(f"   Moderate elevation (1.5-2.0): {moderate_elevation.sum()} ({moderate_elevation.mean()*100:.1f}%)")
print(f"   Severe elevation (>2.0): {severe_elevation.sum()} ({severe_elevation.mean()*100:.1f}%)")
print(f"   Low (<0.6): {low_values.sum()} ({low_values.mean()*100:.1f}%)")

# Focus on cases with abnormal values (more likely to show climate effects)
print(f"\n3. Feature engineering for kidney function...")

# Get base climate features
base_climate = []
for col in df.columns:
    if any(pattern in col.lower() for pattern in ['temp', 'humid', 'heat', 'wind']):
        if df[col].dtype in [np.float64, np.int64]:
            if 'confidence' not in col.lower():
                non_missing = df[col].notna().sum()
                if non_missing > 1000:
                    base_climate.append(col)

print(f"   Base climate features: {len(base_climate)}")

# Create kidney-specific engineered features
df_eng = df.copy()

# Heat stress indicators (important for kidney function)
if 'heat_index' in df.columns and 'temperature' in df.columns:
    df_eng['severe_heat_stress'] = (df['heat_index'] > 40).astype(int)  # >40°C heat index
    df_eng['prolonged_heat'] = (df['temperature'] > 30).astype(int)     # >30°C temperature
    print(f"   ✓ Created heat stress indicators")

# Dehydration risk factors
temp_cols = [col for col in base_climate if 'temp' in col.lower() and 'max' in col.lower()]
humid_cols = [col for col in base_climate if 'humid' in col.lower() and 'min' in col.lower()]

if temp_cols and humid_cols:
    # High temp + low humidity = dehydration risk
    df_eng['dehydration_risk'] = (df[temp_cols[0]] > df[temp_cols[0]].quantile(0.8)) & \
                                (df[humid_cols[0]] < df[humid_cols[0]].quantile(0.2))
    df_eng['dehydration_risk'] = df_eng['dehydration_risk'].astype(int)
    print(f"   ✓ Created dehydration risk factor")

# Heat wave indicators (consecutive hot days)
if 'temperature' in df.columns:
    # Sort by date if available, otherwise by index
    if 'primary_date' in df.columns:
        df_sorted = df.sort_values('primary_date')
        temp_series = df_sorted['temperature']
    else:
        temp_series = df['temperature']
    
    # Simple heat wave: temp > 75th percentile
    hot_threshold = temp_series.quantile(0.75)
    df_eng['heat_wave_day'] = (df['temperature'] > hot_threshold).astype(int)
    print(f"   ✓ Created heat wave indicator (>{hot_threshold:.1f}°C)")

# Get engineered features
engineered_features = ['severe_heat_stress', 'prolonged_heat', 'dehydration_risk', 'heat_wave_day']
available_engineered = [f for f in engineered_features if f in df_eng.columns]

# Combine with key climate features
key_climate = [col for col in base_climate if any(term in col.lower() for term in [
    'heat_index', 'temperature', 'humidity', 'apparent_temp', 'wet_bulb'
])]

# Add lag features that might be important for kidney effects
lag_features = [col for col in base_climate if 'lag' in col.lower()][:10]  # Top 10 lag features

all_features = key_climate + lag_features + available_engineered

print(f"   Key climate: {len(key_climate)}")
print(f"   Lag features: {len(lag_features)}")  
print(f"   Engineered: {len(available_engineered)}")
print(f"   Total features: {len(all_features)}")

# Test different modeling approaches
print(f"\n4. Testing multiple modeling approaches...")

results = {}

# Get complete cases
model_cols = all_features + [target]
complete_df = df_eng[model_cols].dropna()
print(f"   Complete cases: {len(complete_df)}")

if len(complete_df) < 50:
    print("   ✗ Insufficient data")
    exit()

X = complete_df[all_features]
y = complete_df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Approach 1: Standard Random Forest
print(f"\n   Approach 1: Standard Random Forest")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"     R² = {rf_r2:.3f}, MAE = {rf_mae:.3f}")
results['Random Forest'] = {'r2': rf_r2, 'mae': rf_mae, 'model': rf_model}

# Approach 2: Focus on abnormal values only
print(f"\n   Approach 2: Focus on abnormal creatinine (>1.2 or <0.6)")
abnormal_mask = (y < 0.6) | (y > 1.2)
if abnormal_mask.sum() > 20:  # Need enough abnormal cases
    X_abn = X[abnormal_mask]
    y_abn = y[abnormal_mask]
    
    if len(X_abn) > 20:
        X_abn_train, X_abn_test, y_abn_train, y_abn_test = train_test_split(
            X_abn, y_abn, test_size=0.2, random_state=42
        )
        
        rf_abn = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=1
        )
        rf_abn.fit(X_abn_train, y_abn_train)
        abn_pred = rf_abn.predict(X_abn_test)
        abn_r2 = r2_score(y_abn_test, abn_pred)
        abn_mae = mean_absolute_error(y_abn_test, abn_pred)
        
        print(f"     R² = {abn_r2:.3f}, MAE = {abn_mae:.3f} (n={len(X_abn)})")
        results['Abnormal Values'] = {'r2': abn_r2, 'mae': abn_mae, 'model': rf_abn}
    else:
        print(f"     Insufficient abnormal cases: {len(X_abn)}")
else:
    print(f"     No abnormal cases found")

# Approach 3: Classification approach (normal vs elevated)
print(f"\n   Approach 3: Classification (normal vs elevated)")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create binary target: elevated (>1.2) vs normal/low (<=1.2)
y_binary = (y > 1.2).astype(int)
elevated_count = y_binary.sum()

print(f"     Elevated cases: {elevated_count} ({elevated_count/len(y)*100:.1f}%)")

if elevated_count > 5:  # Need some elevated cases
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=1,
        class_weight='balanced'  # Handle imbalanced classes
    )
    rf_classifier.fit(X_train_bin, y_train_bin)
    bin_pred = rf_classifier.predict(X_test_bin)
    bin_accuracy = accuracy_score(y_test_bin, bin_pred)
    
    print(f"     Accuracy = {bin_accuracy:.3f}")
    
    # Feature importance for classification
    bin_importance = rf_classifier.feature_importances_
    bin_features_df = pd.DataFrame({
        'feature': all_features,
        'importance': bin_importance
    }).sort_values('importance', ascending=False)
    
    results['Classification'] = {
        'accuracy': bin_accuracy, 
        'model': rf_classifier,
        'importance': bin_features_df
    }
else:
    print(f"     Too few elevated cases for classification")

# Summary and best approach
print(f"\n" + "=" * 60)
print("CREATININE MODEL COMPARISON")
print("=" * 60)

best_r2 = -1
best_approach = None

for approach, result in results.items():
    if 'r2' in result:
        print(f"{approach:20} R² = {result['r2']:6.3f}, MAE = {result['mae']:.3f}")
        if result['r2'] > best_r2:
            best_r2 = result['r2']
            best_approach = approach
    elif 'accuracy' in result:
        print(f"{approach:20} Accuracy = {result['accuracy']:.3f}")

if best_approach and best_r2 > 0.1:
    print(f"\n✓ Best approach: {best_approach} (R² = {best_r2:.3f})")
    
    # Show feature importance for best model
    best_model = results[best_approach]['model']
    feature_imp = best_model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': all_features,
        'importance': feature_imp
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop predictors for creatinine ({best_approach}):")
    for i, row in imp_df.head(8).iterrows():
        feat_type = "Engineered" if row['feature'] in available_engineered else "Climate"
        print(f"  {row['feature']:<35} {row['importance']:.4f} ({feat_type})")

elif 'Classification' in results:
    print(f"\n✓ Classification approach shows some promise")
    if 'importance' in results['Classification']:
        class_imp = results['Classification']['importance']
        print(f"\nTop predictors for elevated creatinine:")
        for i, row in class_imp.head(5).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")

else:
    print(f"\n⚠ All approaches show weak climate-creatinine relationships")
    print(f"Possible reasons:")
    print(f"  - Creatinine is less sensitive to acute climate changes")
    print(f"  - Need longer-term exposure measures")
    print(f"  - Individual factors (age, medication) dominate")
    print(f"  - Need different feature engineering approach")

print(f"\nClinical insights:")
print(f"- Creatinine reflects kidney function")
print(f"- May be more sensitive to chronic vs acute climate exposure")
print(f"- Heat stress effects might be delayed or cumulative")
print(f"- Consider interaction with age, medication, comorbidities")