#!/usr/bin/env python3
"""
Rapid ML Multi-Biomarker Analysis
=================================
Efficient version of rigorous ML multi-biomarker analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def rapid_ml_multivariate_analysis():
    """Rapid ML multi-biomarker analysis"""
    print("🔬 RAPID ML MULTI-BIOMARKER CLIMATE-HEALTH ANALYSIS")
    print("=" * 55)
    
    # Load data
    df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    # Biomarker systems
    systems = {
        'cardiovascular': ['systolic blood pressure', 'diastolic blood pressure'],
        'metabolic': ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL'],
        'immune_blood': ['CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)']
    }
    
    # Climate variables from successful analysis
    climate_vars = ['temperature_tas_lag0', 'temperature_tas_lag1', 'temperature_tas_lag2', 'temperature_tas_lag3']
    available_climate = [c for c in climate_vars if c in df.columns]
    
    print(f"Climate predictors: {len(available_climate)}")
    
    all_results = {}
    
    for system_name, biomarkers in systems.items():
        available_biomarkers = [b for b in biomarkers if b in df.columns]
        
        if len(available_biomarkers) < 2:
            continue
            
        # Get complete data
        system_data = df.dropna(subset=available_biomarkers + available_climate)
        
        if len(system_data) < 500:
            print(f"\n{system_name}: Insufficient data ({len(system_data)} samples)")
            continue
            
        print(f"\n🏥 {system_name.upper()} System")
        print(f"Biomarkers: {available_biomarkers}")
        print(f"Sample size: {len(system_data):,}")
        
        # Create composite target (standardized average)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(system_data[available_biomarkers])
        composite = np.mean(standardized, axis=1)
        
        # Climate data
        X = system_data[available_climate]
        
        # Test ML models
        models = {
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        best_r2 = -999
        best_model_name = None
        
        for model_name, model in models.items():
            scores = cross_val_score(model, X, composite, cv=5, scoring='r2')
            mean_r2 = np.mean(scores)
            
            print(f"  {model_name}: R² = {mean_r2:.4f}")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_model_name = model_name
        
        # Statistical significance
        if best_r2 > 0.02:
            # Quick permutation test
            best_model = models[best_model_name]
            null_scores = []
            for _ in range(50):
                y_perm = np.random.permutation(composite)
                perm_score = cross_val_score(best_model, X, y_perm, cv=3, scoring='r2')
                null_scores.append(np.mean(perm_score))
            
            p_value = np.mean(np.array(null_scores) >= best_r2)
            
            print(f"  Best model: {best_model_name} (R² = {best_r2:.4f}, p = {p_value:.4f})")
            
            if p_value < 0.05:
                # Multi-task learning
                Y_multi = system_data[available_biomarkers].values
                multi_model = MultiTaskElasticNet(alpha=1.0, random_state=42)
                multi_scores = cross_val_score(multi_model, X, Y_multi, cv=5, scoring='r2')
                multi_r2 = np.mean(multi_scores)
                
                print(f"  Multi-task learning: R² = {multi_r2:.4f}")
                
                # Simple validation
                best_climate = available_climate[0]  # Use first climate var
                simple_corr, simple_p = pearsonr(system_data[best_climate], composite)
                
                print(f"  Validation correlation: r = {simple_corr:.3f}, p = {simple_p:.4f}")
                
                all_results[system_name] = {
                    'composite_r2': best_r2,
                    'composite_p': p_value,
                    'multi_task_r2': multi_r2,
                    'validation_corr': simple_corr,
                    'validation_p': simple_p,
                    'n_samples': len(system_data),
                    'biomarkers': available_biomarkers
                }
                
                print(f"  ✅ SIGNIFICANT MULTI-BIOMARKER RELATIONSHIP")
            else:
                print(f"  Not significant (p = {p_value:.4f})")
        else:
            print(f"  Below threshold (R² = {best_r2:.4f})")
    
    # Summary
    print("\n" + "=" * 55)
    print("🎯 RAPID ML MULTI-BIOMARKER SUMMARY")
    print("=" * 55)
    
    if all_results:
        print(f"Significant systems: {len(all_results)}")
        
        for system_name, results in all_results.items():
            print(f"\n🏆 {system_name.upper()}:")
            print(f"  Composite R² = {results['composite_r2']:.4f}")
            print(f"  Multi-task R² = {results['multi_task_r2']:.4f}")
            print(f"  Validation r = {results['validation_corr']:.3f}")
            print(f"  Sample size = {results['n_samples']:,}")
            print(f"  Biomarkers = {len(results['biomarkers'])}")
    else:
        print("❌ No significant multi-biomarker relationships detected")
        print("Single biomarker approaches remain most effective.")
    
    return all_results

if __name__ == "__main__":
    results = rapid_ml_multivariate_analysis()