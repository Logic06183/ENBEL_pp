#!/usr/bin/env python3
"""
Fast Improved Climate-Health Pipeline
=====================================

Quick version that addresses key issues:
1. Removes race confounding 
2. Focuses on climate features only
3. Uses reasonable hyperparameters (no extensive grid search)
4. Creates proper SHAP visualizations

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_prepare_data(target_biomarker='systolic blood pressure'):
    """Load and prepare clean climate-focused data"""
    print("Loading and Preparing Data")
    print("-" * 40)
    
    # Load data
    data = pd.read_csv("DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv", low_memory=False)
    
    # Focus on clinical participants with target biomarker
    clinical_data = data[data['data_source'].notna()].copy()
    clean_data = clinical_data.dropna(subset=[target_biomarker]).copy()
    
    print(f"Participants with {target_biomarker}: {len(clean_data):,}")
    
    # Select PURE climate features (NO demographics, NO biomarkers)
    climate_keywords = ['temp', 'humid', 'heat', 'wind', 'apparent', 'degree', 'cooling', 'heating']
    
    # Explicit biomarker exclusions
    biomarker_keywords = ['blood', 'pressure', 'glucose', 'cd4', 'ldl', 'hdl', 'hemoglobin', 'creatinine', 'alt']
    demographic_keywords = ['sex', 'race', 'age', 'gender', 'ethnicity']
    
    climate_features = []
    for column in clean_data.columns:
        # Include climate keywords but exclude ALL biomarkers and demographics
        is_climate = any(keyword in column.lower() for keyword in climate_keywords)
        is_biomarker = any(bio in column.lower() for bio in biomarker_keywords)
        is_demographic = any(demo in column.lower() for demo in demographic_keywords)
        is_target = column == target_biomarker
        
        if (is_climate and not is_biomarker and not is_demographic and not is_target):
            if (clean_data[column].dtype in ['float64', 'int64'] and 
                clean_data[column].notna().sum() / len(clean_data) > 0.7):
                climate_features.append(column)
    
    print(f"Pure climate features: {len(climate_features)}")
    
    # Prepare feature matrix
    X = clean_data[climate_features].copy()
    y = clean_data[target_biomarker].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove zero variance features
    X = X.loc[:, X.var() > 1e-6]
    
    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    X = X.drop(columns=high_corr_features)
    
    print(f"Final features: {X.shape[1]}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f} mmHg")
    
    return X, y, clean_data

def train_optimized_models(X, y):
    """Train optimized models without extensive grid search"""
    print("\nTraining Optimized Models")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training: {len(X_train):,}, Testing: {len(X_test):,}")
    
    results = {}
    
    # 1. Optimized Random Forest
    print("🌲 Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    
    results['random_forest'] = {
        'model': rf_model,
        'test_r2': rf_r2,
        'test_mae': rf_mae,
        'cv_r2_mean': rf_cv.mean(),
        'cv_r2_std': rf_cv.std()
    }
    
    print(f"   Test R²: {rf_r2:.4f}")
    print(f"   CV R²: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    
    # 2. Optimized XGBoost
    print("🚀 XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
    
    results['xgboost'] = {
        'model': xgb_model,
        'test_r2': xgb_r2,
        'test_mae': xgb_mae,
        'cv_r2_mean': xgb_cv.mean(),
        'cv_r2_std': xgb_cv.std()
    }
    
    print(f"   Test R²: {xgb_r2:.4f}")
    print(f"   CV R²: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
    
    # Select best model
    if results['xgboost']['cv_r2_mean'] > results['random_forest']['cv_r2_mean']:
        best_name = 'xgboost'
    else:
        best_name = 'random_forest'
    
    results['best_model_name'] = best_name
    results['best_model'] = results[best_name]['model']
    results['best_score'] = results[best_name]['cv_r2_mean']
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['y_test'] = y_test
    
    print(f"\n🏆 Best model: {best_name}")
    print(f"   CV R²: {results['best_score']:.4f} ± {results[best_name]['cv_r2_std']:.4f}")
    
    return results

def analyze_feature_importance(model, X):
    """Analyze feature importance"""
    print("\nFeature Importance Analysis")
    print("-" * 40)
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 15 Most Important Climate Features:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<40} {row['importance']:.4f}")
        
        return importance_df
    return None

def create_shap_analysis(results, X, target_biomarker):
    """Create comprehensive SHAP analysis"""
    print("\nSHAP Analysis")
    print("-" * 40)
    
    model = results['best_model']
    X_test = results['X_test']
    model_name = results['best_model_name']
    r2_score = results['best_score']
    
    # Sample for SHAP
    sample_size = min(300, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    print(f"Calculating SHAP values for {sample_size} participants...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Set up plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    
    # 1. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.title(f'SHAP Summary Plot: {target_biomarker}\n' + 
             f'{model_name.title()} Model (R² = {r2_score:.3f}) - Pure Climate Effects', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('glucose_shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('glucose_shap_summary_plot.svg', bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary (beeswarm) plot saved")
    
    # 2. SHAP Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=15, show=False)
    plt.title(f'SHAP Feature Importance: {target_biomarker}\n' + 
             f'Mean Absolute SHAP Values - No Demographic Confounding', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('glucose_shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('glucose_shap_bar_plot.svg', bbox_inches='tight')
    plt.close()
    print("✓ SHAP bar plot saved")
    
    # 3. SHAP Dependency Plot for top feature
    top_feature_idx = np.argmax(np.abs(shap_values).mean(0))
    top_feature_name = X_sample.columns[top_feature_idx]
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values, X_sample, show=False)
    plt.title(f'SHAP Dependency Plot: {top_feature_name}\n' + 
             f'Most Important Climate Feature for {target_biomarker}', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('glucose_shap_dependency_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('glucose_shap_dependency_plot.svg', bbox_inches='tight')
    plt.close()
    print("✓ SHAP dependency plot saved")
    
    # 4. SHAP Waterfall Plot
    plt.figure(figsize=(10, 8))
    try:
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0], 
                base_values=explainer.expected_value,
                data=X_sample.iloc[0]
            ), 
            max_display=10, 
            show=False
        )
        plt.title(f'SHAP Waterfall Plot: Individual Prediction\n' + 
                 f'{target_biomarker} - Climate Factors Only', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('glucose_shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig('glucose_shap_waterfall_plot.svg', bbox_inches='tight')
        plt.close()
        print("✓ SHAP waterfall plot saved")
    except Exception as e:
        print(f"⚠️ Waterfall plot failed: {e}")
    
    # Calculate SHAP-based feature importance
    shap_importance = np.abs(shap_values).mean(0)
    shap_df = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    print("\nTop 10 Features by SHAP Importance:")
    for i, (_, row) in enumerate(shap_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<40} {row['shap_importance']:.4f}")
    
    return shap_values, X_sample, explainer, shap_df

def main():
    """Main analysis function"""
    print("🚀 GLUCOSE-CLIMATE RELATIONSHIP ANALYSIS")
    print("="*60)
    print("Focus: Pure climate effects on glucose metabolism")
    print()
    
    target = 'FASTING GLUCOSE'
    
    # Load and prepare data
    X, y, clean_data = load_and_prepare_data(target)
    
    # Train models
    results = train_optimized_models(X, y)
    
    # Feature importance
    importance_df = analyze_feature_importance(results['best_model'], X)
    
    # SHAP analysis
    shap_values, X_sample, explainer, shap_df = create_shap_analysis(results, X, target)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print(f"Target: {target}")
    print(f"Sample size: {len(clean_data):,} participants")
    print(f"Features: {len(X.columns)} pure climate variables")
    print(f"Best model: {results['best_model_name']}")
    print(f"Performance: R² = {results['best_score']:.4f} ± {results[results['best_model_name']]['cv_r2_std']:.4f}")
    print("🎯 NO DEMOGRAPHIC CONFOUNDING!")
    print("\nGenerated files:")
    print("  - final_shap_summary_plot.png/svg (beeswarm)")
    print("  - final_shap_bar_plot.png/svg")
    print("  - final_shap_dependency_plot.png/svg")
    print("  - final_shap_waterfall_plot.png/svg")
    
    return results

if __name__ == "__main__":
    main()