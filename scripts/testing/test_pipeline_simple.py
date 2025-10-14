#!/usr/bin/env python3
"""
Simple working test of the pipeline with real data
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SIMPLE PIPELINE TEST")
print("=" * 60)

# Load data
df = pd.read_csv('data/raw/clinical_dataset.csv', low_memory=False)
print(f"1. Loaded data: {df.shape}")

# Get key biomarker and climate columns that exist and have data
biomarkers = ['CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 
              'Creatinine (mg/dL)', 'Hemoglobin (g/dL)']

climate_features = ['temperature', 'humidity', 'wind_speed', 'pressure', 'temperature_max', 
                   'temperature_min', 'humidity_avg', 'wind_speed_avg']

# Check which columns actually exist
available_biomarkers = [col for col in biomarkers if col in df.columns]
available_climate = [col for col in climate_features if col in df.columns and 
                    df[col].dtype in [np.float64, np.int64]]

print(f"2. Available biomarkers: {len(available_biomarkers)}")
print(f"   {available_biomarkers}")
print(f"3. Available climate features: {len(available_climate)}")
print(f"   {available_climate[:5]}...")

if len(available_biomarkers) > 0 and len(available_climate) > 0:
    # Take first biomarker with enough data
    target = available_biomarkers[0]
    
    # Check data availability
    target_data = df[target].dropna()
    print(f"\n4. Analyzing {target}:")
    print(f"   Non-missing values: {len(target_data)}")
    print(f"   Mean: {target_data.mean():.2f}")
    print(f"   Range: {target_data.min():.2f} - {target_data.max():.2f}")
    
    if len(target_data) > 100:
        # Prepare data for ML
        features_and_target = available_climate + [target]
        model_df = df[features_and_target].copy()
        
        # Remove rows where target is missing
        model_df = model_df.dropna(subset=[target])
        print(f"\n5. Model data preparation:")
        print(f"   Samples with target data: {len(model_df)}")
        
        # Simple imputation for remaining missing climate data
        X = model_df[available_climate]
        y = model_df[target]
        
        # Check for missing values in features
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X), 
                columns=available_climate, 
                index=X.index
            )
            print(f"   Imputed {missing_before} missing feature values")
        else:
            X_imputed = X
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n6. Model Results:")
        print(f"   R² score: {r2:.3f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': available_climate,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 3 important climate features:")
        for i, row in importance_df.head(3).iterrows():
            print(f"     {row['feature']}: {row['importance']:.3f}")
        
        # Try SHAP if available
        try:
            import shap
            print(f"\n7. Testing SHAP explainability:")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:10])
            print(f"   ✓ SHAP values calculated for 10 samples")
            print(f"   Shape: {shap_values.shape}")
        except ImportError:
            print(f"\n7. SHAP not available - install with: pip install shap")
        except Exception as e:
            print(f"\n7. SHAP error: {e}")
        
        print(f"\n" + "=" * 60)
        print("SIMPLE TEST RESULTS")
        print("=" * 60)
        print(f"✓ Data loaded and processed")
        print(f"✓ Target: {target} ({len(target_data)} samples)")
        print(f"✓ Features: {len(available_climate)} climate variables")
        print(f"✓ Model trained: R² = {r2:.3f}")
        
        quality = "Excellent" if r2 > 0.5 else "Good" if r2 > 0.3 else "Moderate" if r2 > 0.1 else "Poor"
        print(f"✓ Model quality: {quality}")
        
        if r2 > 0.3:
            print(f"✓ Model is good enough for SHAP analysis")
        else:
            print(f"⚠ Model R² < 0.3 - may need feature engineering")
            
    else:
        print(f"✗ Not enough data for {target}: {len(target_data)} samples")
else:
    print("✗ Missing required biomarkers or climate features")

print(f"\nNext steps:")
print(f"1. Test with more biomarkers")
print(f"2. Add feature engineering (lags, rolling averages)")
print(f"3. Implement proper imputation strategy")
print(f"4. Add hyperparameter optimization")