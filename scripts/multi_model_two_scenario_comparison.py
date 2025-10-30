"""
Multi-Model Two-Scenario Comparison
====================================

Compares climate-only vs full model (climate + demographics) across:
- Random Forest, XGBoost, LightGBM
- Hyperparameter tuning via RandomizedSearchCV
- Top 10 biomarkers

Author: ENBEL Team
Date: 2025-10-30
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import shap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
np.random.seed(42)

print("="*80)
print("MULTI-MODEL TWO-SCENARIO COMPARISON")
print("="*80)
print("Scenario A: Climate-only (9 features)")
print("Scenario B: Full model (13 features: Climate + Demographics)")
print("Models: Random Forest, XGBoost, LightGBM")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top 10 biomarkers
BIOMARKERS = [
    'Hematocrit (%)',
    'CD4 cell count (cells/µL)',
    'FASTING LDL',
    'FASTING HDL',
    'Albumin (g/dL)',
    'creatinine_umol_L',
    'White blood cell count (×10³/µL)',
    'Lymphocyte count (×10³/µL)',
    'Neutrophil count (×10³/µL)',
    'weight_kg'
]

# Scenario A features (climate-only)
SCENARIO_A_FEATURES = [
    'climate_daily_mean_temp',
    'climate_daily_max_temp',
    'climate_daily_min_temp',
    'climate_7d_mean_temp',
    'climate_heat_stress_index',
    'climate_season',
    'month',
    'season',
    'HEAT_VULNERABILITY_SCORE'
]

# Scenario B features (full model)
SCENARIO_B_FEATURES = SCENARIO_A_FEATURES + [
    'Age (at enrolment)',
    'Sex',
    'study_source',
    'year'
]

CATEGORICAL_FEATURES_A = ['climate_season', 'season']
CATEGORICAL_FEATURES_B = ['climate_season', 'season', 'Sex', 'study_source']

# Hyperparameter grids for RandomizedSearchCV (REDUCED for speed)
PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators': [50, 100],
        'max_depth': [10, 15, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [50, 100],
        'max_depth': [5, 7, -1],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_datasets():
    """Load both scenario datasets."""
    base_dir = Path(__file__).resolve().parents[1]

    # Scenario A (climate-only)
    scenario_a_path = base_dir / "results" / "modeling" / "MODELING_DATASET_SCENARIO_B.csv"
    df_a = pd.read_csv(scenario_a_path, low_memory=False)

    # Scenario B (full model)
    scenario_b_path = base_dir / "results" / "modeling" / "MODELING_DATASET_SCENARIO_B_FULL.csv"
    df_b = pd.read_csv(scenario_b_path, low_memory=False)

    print(f"\n📊 Datasets loaded:")
    print(f"   Scenario A: {len(df_a):,} records, {len(SCENARIO_A_FEATURES)} features")
    print(f"   Scenario B: {len(df_b):,} records, {len(SCENARIO_B_FEATURES)} features")

    return df_a, df_b

# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def prepare_features(df, features, categorical_features, target_biomarker):
    """
    Prepare features for modeling.

    Returns
    -------
    X_encoded : pd.DataFrame
        Feature matrix (one-hot encoded, float64)
    y : pd.Series
        Target variable
    feature_names : list
        Feature names after encoding
    n_samples : int
        Number of samples
    """
    # Filter to records with target biomarker
    df_subset = df[df[target_biomarker].notna()].copy()

    if len(df_subset) < 200:
        raise ValueError(f"Insufficient data: {len(df_subset)} records")

    X = df_subset[features].copy()
    y = df_subset[target_biomarker].copy()

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

    # Ensure all columns are numeric
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            X_encoded[col] = pd.Categorical(X_encoded[col]).codes
        elif X_encoded[col].dtype == 'bool':
            X_encoded[col] = X_encoded[col].astype(int)

    # Convert to float64 for model compatibility
    X_encoded = X_encoded.astype(np.float64)

    feature_names = X_encoded.columns.tolist()

    return X_encoded, y, feature_names, len(df_subset)

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model_with_tuning(X, y, model_name, random_state=42):
    """
    Train model with HYPERPARAMETER TUNING (medium intensity).

    Returns
    -------
    best_model : trained model
    best_params : dict
        Best hyperparameters found
    metrics : dict
        R², MAE, RMSE on test set
    X_train, X_test, y_train, y_test : data splits
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Initialize base model
    if model_name == 'RandomForest':
        base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_name == 'XGBoost':
        base_model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0)
    elif model_name == 'LightGBM':
        base_model = lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Hyperparameter tuning with RandomizedSearchCV
    print(f"      Tuning hyperparameters (10 iterations, 3-fold CV)...")
    search = RandomizedSearchCV(
        base_model,
        PARAM_GRIDS[model_name],
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_
    params = search.best_params_

    print(f"      Best params: {params}")

    # Evaluate on test set
    y_pred_test = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, y_pred_test),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

    return model, params, metrics, X_train, X_test, y_train, y_test

# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def compute_shap_values(model, X_train, X_test, max_background=1000):
    """Compute SHAP values."""
    # Sample background
    if len(X_train) > max_background:
        background = shap.sample(X_train, max_background, random_state=42)
    else:
        background = X_train

    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

    return explainer, shap_values

# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

def analyze_biomarker(biomarker, df_a, df_b):
    """
    Analyze a single biomarker across both scenarios and all models.

    Returns
    -------
    results : dict
        Comprehensive results for this biomarker
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {biomarker}")
    print(f"{'='*80}")

    results = {
        'biomarker': biomarker,
        'timestamp': datetime.now().isoformat(),
        'scenarios': {}
    }

    # Scenario A: Climate-only
    print(f"\n📍 Scenario A: Climate-only ({len(SCENARIO_A_FEATURES)} features)")
    try:
        X_a, y_a, features_a, n_a = prepare_features(
            df_a, SCENARIO_A_FEATURES, CATEGORICAL_FEATURES_A, biomarker
        )
        print(f"   Samples: {n_a} | Features after encoding: {len(features_a)}")

        scenario_a_results = {'n_samples': n_a, 'n_features': len(features_a), 'models': {}}

        # Train all 3 models
        for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            print(f"\n   🔧 {model_name}:")
            model, params, metrics, X_train, X_test, y_train, y_test = train_model_with_tuning(
                X_a, y_a, model_name
            )

            print(f"      R² = {metrics['r2']:.3f} | MAE = {metrics['mae']:.2f} | RMSE = {metrics['rmse']:.2f}")

            scenario_a_results['models'][model_name] = {
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'best_params': params
            }

        # SHAP for best model (highest R²)
        best_model_name = max(scenario_a_results['models'],
                              key=lambda m: scenario_a_results['models'][m]['r2'])
        print(f"\n   🎯 Best model: {best_model_name} (R² = {scenario_a_results['models'][best_model_name]['r2']:.3f})")
        print(f"      Computing SHAP values...")

        # Re-train best model for SHAP
        best_model, _, _, X_train, X_test, _, _ = train_model_with_tuning(X_a, y_a, best_model_name)
        explainer, shap_values = compute_shap_values(best_model, X_train, X_test)

        # Save SHAP importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': features_a,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        scenario_a_results['best_model'] = best_model_name
        scenario_a_results['shap_importance'] = importance_df.to_dict('records')
        scenario_a_results['shap_values'] = shap_values
        scenario_a_results['X_test'] = X_test
        scenario_a_results['feature_names'] = features_a

        results['scenarios']['A'] = scenario_a_results

    except Exception as e:
        print(f"   ❌ Error in Scenario A: {e}")
        results['scenarios']['A'] = {'error': str(e)}

    # Scenario B: Full model
    print(f"\n📍 Scenario B: Full model ({len(SCENARIO_B_FEATURES)} features)")
    try:
        X_b, y_b, features_b, n_b = prepare_features(
            df_b, SCENARIO_B_FEATURES, CATEGORICAL_FEATURES_B, biomarker
        )
        print(f"   Samples: {n_b} | Features after encoding: {len(features_b)}")

        scenario_b_results = {'n_samples': n_b, 'n_features': len(features_b), 'models': {}}

        # Train all 3 models
        for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            print(f"\n   🔧 {model_name}:")
            model, params, metrics, X_train, X_test, y_train, y_test = train_model_with_tuning(
                X_b, y_b, model_name
            )

            print(f"      R² = {metrics['r2']:.3f} | MAE = {metrics['mae']:.2f} | RMSE = {metrics['rmse']:.2f}")

            scenario_b_results['models'][model_name] = {
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'best_params': params
            }

        # SHAP for best model
        best_model_name = max(scenario_b_results['models'],
                              key=lambda m: scenario_b_results['models'][m]['r2'])
        print(f"\n   🎯 Best model: {best_model_name} (R² = {scenario_b_results['models'][best_model_name]['r2']:.3f})")
        print(f"      Computing SHAP values...")

        best_model, _, _, X_train, X_test, _, _ = train_model_with_tuning(X_b, y_b, best_model_name)
        explainer, shap_values = compute_shap_values(best_model, X_train, X_test)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': features_b,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        scenario_b_results['best_model'] = best_model_name
        scenario_b_results['shap_importance'] = importance_df.to_dict('records')
        scenario_b_results['shap_values'] = shap_values
        scenario_b_results['X_test'] = X_test
        scenario_b_results['feature_names'] = features_b

        results['scenarios']['B'] = scenario_b_results

    except Exception as e:
        print(f"   ❌ Error in Scenario B: {e}")
        results['scenarios']['B'] = {'error': str(e)}

    # Calculate ΔR²
    if 'A' in results['scenarios'] and 'B' in results['scenarios']:
        if 'error' not in results['scenarios']['A'] and 'error' not in results['scenarios']['B']:
            r2_a_best = results['scenarios']['A']['models'][results['scenarios']['A']['best_model']]['r2']
            r2_b_best = results['scenarios']['B']['models'][results['scenarios']['B']['best_model']]['r2']
            delta_r2 = r2_b_best - r2_a_best

            results['delta_r2'] = delta_r2
            results['climate_contribution'] = 'Demographics improve model' if delta_r2 > 0 else 'No improvement from demographics'

            print(f"\n📊 ΔR² (Full - Climate): {delta_r2:+.3f}")
            if delta_r2 > 0.05:
                print(f"   ✅ Demographics significantly improve prediction")
            elif delta_r2 < -0.05:
                print(f"   ⚠️  Demographics worsen prediction (possible overfitting)")
            else:
                print(f"   ℹ️  Demographics have minimal impact")

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    # Load datasets
    df_a, df_b = load_datasets()

    # Output directory
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "results" / "multi_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all biomarkers
    all_results = []

    for i, biomarker in enumerate(BIOMARKERS, 1):
        print(f"\n\n{'#'*80}")
        print(f"BIOMARKER {i}/{len(BIOMARKERS)}")
        print(f"{'#'*80}")

        try:
            results = analyze_biomarker(biomarker, df_a, df_b)
            all_results.append(results)

            # Save individual results
            biomarker_safe = biomarker.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            result_path = output_dir / f"{biomarker_safe}_results.json"

            # Remove numpy arrays before saving (can't serialize)
            results_to_save = results.copy()
            for scenario in ['A', 'B']:
                if scenario in results_to_save['scenarios']:
                    if 'shap_values' in results_to_save['scenarios'][scenario]:
                        del results_to_save['scenarios'][scenario]['shap_values']
                    if 'X_test' in results_to_save['scenarios'][scenario]:
                        del results_to_save['scenarios'][scenario]['X_test']

            with open(result_path, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)

            print(f"\n✅ Results saved: {result_path}")

        except Exception as e:
            print(f"\n❌ Failed to analyze {biomarker}: {e}")
            all_results.append({
                'biomarker': biomarker,
                'error': str(e)
            })

    # Save master results
    master_path = output_dir / "master_results.json"

    # Remove numpy arrays
    all_results_to_save = []
    for result in all_results:
        result_copy = result.copy()
        if 'scenarios' in result_copy:
            for scenario in ['A', 'B']:
                if scenario in result_copy['scenarios']:
                    if 'shap_values' in result_copy['scenarios'][scenario]:
                        del result_copy['scenarios'][scenario]['shap_values']
                    if 'X_test' in result_copy['scenarios'][scenario]:
                        del result_copy['scenarios'][scenario]['X_test']
        all_results_to_save.append(result_copy)

    with open(master_path, 'w') as f:
        json.dump(all_results_to_save, f, indent=2, default=str)

    print(f"\n\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Master results: {master_path}")
    print(f"✅ Individual results: {output_dir}")
    print(f"\nNext: Generate summary visualizations and comparison tables")

if __name__ == '__main__':
    main()
