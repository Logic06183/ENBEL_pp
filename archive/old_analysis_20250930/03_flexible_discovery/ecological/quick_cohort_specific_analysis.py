#!/usr/bin/env python3
"""
Quick Cohort-Specific Analysis
==============================
Immediate implementation to demonstrate proper handling of separated cohorts
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("COHORT-SPECIFIC CLIMATE-HEALTH ANALYSIS")
    print("="*70)
    print("\nDemonstrating proper approach for your data structure\n")
    
    # Load data
    df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    # Separate cohorts properly
    print("1. SEPARATING COHORTS")
    print("-"*40)
    
    # Method 1: Using dataset_group if available
    if 'dataset_group' in df.columns:
        clinical = df[df['dataset_group'] == 'clinical'].copy()
        survey = df[df['dataset_group'] == 'socioeconomic'].copy()
    else:
        # Method 2: Using biomarker presence
        has_biomarkers = df[['FASTING GLUCOSE', 'systolic blood pressure']].notna().any(axis=1)
        clinical = df[has_biomarkers].copy()
        survey = df[~has_biomarkers].copy()
    
    print(f"Clinical cohort: {len(clinical):,} participants")
    print(f"Survey cohort: {len(survey):,} participants")
    
    # Analyze clinical cohort
    print("\n2. CLINICAL COHORT ANALYSIS")
    print("-"*40)
    
    biomarkers = ['systolic blood pressure', 'diastolic blood pressure', 'FASTING GLUCOSE']
    climate_vars = ['temperature', 'humidity', 'heat_index', 'apparent_temp']
    
    results = {}
    
    for biomarker in biomarkers:
        if biomarker not in clinical.columns:
            continue
            
        # Get non-missing data
        valid_data = clinical[clinical[biomarker].notna()].copy()
        
        if len(valid_data) < 100:
            continue
            
        # Select features
        features = [v for v in climate_vars if v in valid_data.columns]
        
        # Add some lag features
        lag_features = [col for col in valid_data.columns if 'lag' in col.lower()][:10]
        features.extend(lag_features)
        
        # Add demographics if available
        if 'Sex' in valid_data.columns:
            valid_data['Sex_encoded'] = (valid_data['Sex'] == 'Male').astype(int)
            features.append('Sex_encoded')
        
        # Prepare data
        X = valid_data[features].fillna(valid_data[features].median())
        y = valid_data[biomarker]
        
        # Remove any remaining issues
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            continue
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Simple model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        # Cross-validation
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        # Fit for feature importance
        model.fit(X_scaled, y)
        
        # Store results
        results[biomarker] = {
            'n': len(X),
            'r2_mean': scores.mean(),
            'r2_std': scores.std(),
            'top_features': sorted(
                zip(features, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
        print(f"\n{biomarker}:")
        print(f"  Samples: {len(X):,}")
        print(f"  R² Score: {scores.mean():.3f} (±{scores.std():.3f})")
        print(f"  Top predictors:")
        for feat, imp in results[biomarker]['top_features']:
            print(f"    - {feat}: {imp:.3f}")
    
    # Analyze survey cohort
    print("\n3. SURVEY COHORT ANALYSIS")  
    print("-"*40)
    
    # Check what outcomes are available in survey data
    survey_outcomes = ['heat_vulnerability_index', 'housing_vulnerability', 'economic_vulnerability']
    available_outcomes = [o for o in survey_outcomes if o in survey.columns]
    
    if available_outcomes:
        outcome = available_outcomes[0]
        print(f"\nAnalyzing: {outcome}")
        
        valid_survey = survey[survey[outcome].notna()].copy()
        
        # Use socioeconomic predictors
        predictors = ['Education', 'employment_status', 'temperature', 'humidity']
        available_predictors = [p for p in predictors if p in valid_survey.columns]
        
        if len(available_predictors) > 1:
            # Prepare survey data
            for col in available_predictors:
                if valid_survey[col].dtype == 'object':
                    valid_survey[col] = pd.Categorical(valid_survey[col]).codes
            
            X_survey = valid_survey[available_predictors].fillna(0)
            y_survey = valid_survey[outcome]
            
            mask = ~(X_survey.isnull().any(axis=1) | y_survey.isnull())
            X_survey = X_survey[mask]
            y_survey = y_survey[mask]
            
            if len(X_survey) > 100:
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                scores = cross_val_score(model, X_survey, y_survey, cv=5, scoring='r2')
                
                print(f"  Samples: {len(X_survey):,}")
                print(f"  R² Score: {scores.mean():.3f} (±{scores.std():.3f})")
    else:
        print("  No suitable outcomes in survey data")
    
    # Ecological aggregation attempt
    print("\n4. ECOLOGICAL AGGREGATION (Neighborhood-Level)")
    print("-"*40)
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create simple geographic bins
        n_bins = 20
        
        df['lat_bin'] = pd.qcut(df['latitude'].dropna(), n_bins, duplicates='drop', labels=False)
        df['lon_bin'] = pd.qcut(df['longitude'].dropna(), n_bins, duplicates='drop', labels=False)
        df['geo_unit'] = df['lat_bin'].astype(str) + '_' + df['lon_bin'].astype(str)
        
        # Aggregate clinical data
        clinical_agg = df[df['dataset_group'] == 'clinical'].groupby('geo_unit').agg({
            'systolic blood pressure': 'mean',
            'FASTING GLUCOSE': 'mean',
            'temperature': 'mean',
            'humidity': 'mean'
        }).dropna()
        
        # Aggregate survey data  
        survey_agg = df[df['dataset_group'] == 'socioeconomic'].groupby('geo_unit').agg({
            'housing_vulnerability': 'mean',
            'economic_vulnerability': 'mean'
        }).dropna()
        
        # Merge ecological data
        ecological = clinical_agg.merge(survey_agg, left_index=True, right_index=True, how='inner')
        
        if len(ecological) > 10:
            print(f"  Created {len(ecological)} geographic units with both data types")
            print(f"  Can now model relationships at ecological level")
            
            # Example ecological model
            if 'systolic blood pressure' in ecological.columns and 'housing_vulnerability' in ecological.columns:
                X_eco = ecological[['temperature', 'humidity', 'housing_vulnerability']].fillna(ecological.median())
                y_eco = ecological['systolic blood pressure']
                
                if len(X_eco) > 10:
                    from sklearn.linear_model import LinearRegression
                    eco_model = LinearRegression()
                    eco_model.fit(X_eco, y_eco)
                    r2 = eco_model.score(X_eco, y_eco)
                    print(f"  Ecological model R²: {r2:.3f}")
                    print(f"  (Note: Ecological inference, not individual-level)")
        else:
            print("  Insufficient geographic overlap for ecological analysis")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✓ Separated cohorts properly")
    print("✓ Analyzed each cohort with available variables")
    print("✓ Demonstrated ecological aggregation approach")
    print("\n⚠ KEY INSIGHT: These are DIFFERENT populations")
    print("  - Cannot model them together as single cohort")
    print("  - Must use appropriate methods for each data type")
    print("  - Ecological analysis most promising for integration")
    
    print("\n📊 NEXT STEPS:")
    print("  1. Focus on clinical cohort for biomarker-climate relationships")
    print("  2. Use ecological aggregation to link with socioeconomic factors")
    print("  3. Be transparent about data limitations in publication")
    print("  4. Consider data collection improvements for future studies")
    
    return results

if __name__ == "__main__":
    results = main()