"""
Exploratory SHAP Attribution Analysis
======================================

Discover which climate and socioeconomic features drive biomarker responses.

Focus: EXPLAINABILITY and ATTRIBUTION, not just prediction.

For each biomarker:
1. Train Random Forest model (interpretable, robust)
2. Compute SHAP values (feature attribution)
3. Generate comprehensive visualizations:
   - Feature importance rankings
   - SHAP summary plots (beeswarm)
   - SHAP dependence plots with interactions
   - Stratified analysis by vulnerability level
4. Identify key drivers and patterns

Goal: Understand WHAT features matter and HOW they influence biomarkers.

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
np.random.seed(42)


# 9 features for attribution analysis
FEATURES = [
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

# Categorical features (need encoding)
CATEGORICAL_FEATURES = ['climate_season', 'season']

# All biomarkers to analyze
BIOMARKERS = [
    'CD4 cell count (cells/µL)',
    'HIV viral load (copies/mL)',
    'Hematocrit (%)',
    'hemoglobin_g_dL',
    'Platelets (×10³/µL)',
    'White blood cell count (×10³/µL)',
    'Lymphocyte count (×10³/µL)',
    'Neutrophil count (×10³/µL)',
    'Total Cholesterol (mg/dL)',
    'LDL Cholesterol (mg/dL)',
    'HDL Cholesterol (mg/dL)',
    'FASTING LDL',
    'FASTING HDL',
    'FASTING TRIGLYCERIDES',
    'Triglycerides (mg/dL)',
    'fasting_glucose_mmol_L',
    'creatinine_umol_L',
    'ALT (U/L)',
    'AST (U/L)',
    'Systolic blood pressure (mmHg)',
    'Diastolic blood pressure (mmHg)',
    'BMI (kg/m²)',
    'weight_kg',
    'Alkaline phosphatase (U/L)',
    'Albumin (g/dL)',
    'Bilirubin (mg/dL)',
    'Calcium (mg/dL)'
]


def load_modeling_dataset(data_path):
    """Load the clean modeling dataset."""
    print("="*80)
    print("EXPLORATORY SHAP ATTRIBUTION ANALYSIS")
    print("="*80)

    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features\n")

    return df


def prepare_features(df, target_biomarker):
    """
    Prepare features and target for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling dataset
    target_biomarker : str
        Biomarker to predict

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (encoded)
    y : pd.Series
        Target variable
    feature_names : list
        Feature names after encoding
    """
    # Check if biomarker exists and has sufficient data
    if target_biomarker not in df.columns:
        raise ValueError(f"Biomarker '{target_biomarker}' not found in dataset")

    # Filter to records with biomarker
    df_subset = df[df[target_biomarker].notna()].copy()

    if len(df_subset) < 200:
        raise ValueError(f"Insufficient data for {target_biomarker}: {len(df_subset)} records")

    # Prepare features
    X = df_subset[FEATURES].copy()
    y = df_subset[target_biomarker].copy()

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)

    feature_names = X_encoded.columns.tolist()

    return X_encoded, y, feature_names, len(df_subset)


def train_random_forest(X, y, random_state=42):
    """
    Train Random Forest model.

    Uses default hyperparameters for consistency and interpretability.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

    return model, X_train, X_test, y_train, y_test, metrics


def compute_shap_values(model, X_train, X_test):
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model
    X_train : pd.DataFrame
        Training features (for background)
    X_test : pd.DataFrame
        Test features (for explanation)

    Returns
    -------
    explainer : shap.TreeExplainer
        SHAP explainer
    shap_values : np.ndarray
        SHAP values for test set
    """
    print(f"   Computing SHAP values...")

    # Create explainer with training data as background
    # Use a sample for speed if dataset is large
    if len(X_train) > 1000:
        background = shap.sample(X_train, 1000, random_state=42)
    else:
        background = X_train

    explainer = shap.TreeExplainer(model, background)

    # Compute SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    return explainer, shap_values


def create_shap_visualizations(biomarker, shap_values, X_test, feature_names,
                                output_dir, metrics):
    """
    Create comprehensive SHAP visualizations.

    Parameters
    ----------
    biomarker : str
        Biomarker name
    shap_values : np.ndarray
        SHAP values
    X_test : pd.DataFrame
        Test features
    feature_names : list
        Feature names
    output_dir : Path
        Output directory
    metrics : dict
        Model performance metrics
    """
    print(f"   Generating SHAP visualizations...")

    # Create biomarker-specific directory
    biomarker_dir = output_dir / biomarker.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
    biomarker_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Importance (mean absolute SHAP)
    fig, ax = plt.subplots(figsize=(10, 8))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)

    # Plot
    ax.barh(range(len(importance_df)), importance_df['importance'], color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance: {biomarker}\nR² = {metrics["r2_test"]:.3f}',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(biomarker_dir / '01_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SHAP Summary Plot (Beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                     show=False, max_display=15)
    plt.title(f'SHAP Summary: {biomarker}\nR² = {metrics["r2_test"]:.3f}',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(biomarker_dir / '02_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. SHAP Summary Plot (Bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                     plot_type='bar', show=False, max_display=15)
    plt.title(f'SHAP Feature Importance: {biomarker}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(biomarker_dir / '03_shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Top feature dependence plots
    # Find top 3 most important features
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    for i, feat_idx in enumerate(top_features_idx, 1):
        feat_name = feature_names[feat_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx, shap_values, X_test,
            feature_names=feature_names,
            show=False,
            ax=ax
        )
        plt.title(f'SHAP Dependence: {feat_name}\n{biomarker}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(biomarker_dir / f'04_dependence_top{i}_{feat_name.replace("/", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    print(f"   ✅ Visualizations saved to: {biomarker_dir}")

    return importance_df


def analyze_vulnerability_stratification(biomarker, shap_values, X_test, y_test,
                                         feature_names, output_dir):
    """
    Analyze SHAP patterns stratified by vulnerability level.

    Parameters
    ----------
    biomarker : str
        Biomarker name
    shap_values : np.ndarray
        SHAP values
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test targets
    feature_names : list
        Feature names
    output_dir : Path
        Output directory
    """
    print(f"   Analyzing vulnerability stratification...")

    # Get vulnerability score
    vuln_idx = feature_names.index('HEAT_VULNERABILITY_SCORE')
    vuln_scores = X_test.iloc[:, vuln_idx].values

    # Create quartiles
    vuln_quartiles = pd.qcut(vuln_scores, q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

    # Create biomarker-specific directory
    biomarker_dir = output_dir / biomarker.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')

    # Compare feature importance across vulnerability quartiles
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    quartile_importance = {}

    for i, quartile in enumerate(['Low', 'Medium-Low', 'Medium-High', 'High']):
        ax = axes[i]

        # Filter to quartile
        mask = vuln_quartiles == quartile
        shap_subset = shap_values[mask]

        if len(shap_subset) < 10:
            continue

        # Mean absolute SHAP
        mean_abs = np.abs(shap_subset).mean(axis=0)
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs
        }).sort_values('importance', ascending=False).head(10)

        quartile_importance[quartile] = importance

        # Plot
        ax.barh(range(len(importance)), importance['importance'], color='coral', edgecolor='black')
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(importance['feature'], fontsize=9)
        ax.set_xlabel('Mean |SHAP|', fontsize=10)
        ax.set_title(f'Vulnerability: {quartile}\n(n={mask.sum()})', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Feature Importance by Vulnerability Level\n{biomarker}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(biomarker_dir / '05_vulnerability_stratification.png', dpi=300, bbox_inches='tight')
    plt.close()

    return quartile_importance


def generate_attribution_report(biomarker, metrics, importance_df,
                                quartile_importance, n_samples):
    """
    Generate text report summarizing attribution findings.

    Parameters
    ----------
    biomarker : str
        Biomarker name
    metrics : dict
        Model performance
    importance_df : pd.DataFrame
        Feature importance
    quartile_importance : dict
        Stratified importance
    n_samples : int
        Number of samples

    Returns
    -------
    report : dict
        Attribution report
    """
    # Top 5 features
    top5 = importance_df.tail(5)

    # Climate vs socioeconomic contribution
    climate_features = [f for f in importance_df['feature']
                       if any(x in f for x in ['climate', 'month', 'season'])
                       and 'HEAT' not in f]
    climate_importance = importance_df[importance_df['feature'].isin(climate_features)]['importance'].sum()

    socio_importance = importance_df[importance_df['feature'] == 'HEAT_VULNERABILITY_SCORE']['importance'].values
    socio_importance = socio_importance[0] if len(socio_importance) > 0 else 0

    total_importance = importance_df['importance'].sum()

    report = {
        'biomarker': biomarker,
        'n_samples': n_samples,
        'performance': {
            'r2_test': float(metrics['r2_test']),
            'mae_test': float(metrics['mae_test']),
            'rmse_test': float(metrics['rmse_test'])
        },
        'top_5_features': [
            {
                'feature': row['feature'],
                'importance': float(row['importance']),
                'pct_contribution': float(row['importance'] / total_importance * 100)
            }
            for _, row in top5.iterrows()
        ],
        'climate_contribution_pct': float(climate_importance / total_importance * 100),
        'socioeconomic_contribution_pct': float(socio_importance / total_importance * 100),
        'interpretation': 'Climate-driven' if climate_importance > socio_importance else 'Socioeconomic-driven'
    }

    return report


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "results" / "modeling" / "MODELING_DATASET_SCENARIO_B.csv"
    output_dir = base_dir / "results" / "shap_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_modeling_dataset(data_path)

    # Analyze each biomarker
    all_reports = []

    print(f"\n" + "="*80)
    print(f"ANALYZING {len(BIOMARKERS)} BIOMARKERS")
    print("="*80)

    for i, biomarker in enumerate(BIOMARKERS, 1):
        print(f"\n[{i}/{len(BIOMARKERS)}] {biomarker}")
        print("-" * 80)

        try:
            # Prepare data
            X, y, feature_names, n_samples = prepare_features(df, biomarker)
            print(f"   Samples: {n_samples:,} | Features: {len(feature_names)}")

            # Train model
            print(f"   Training Random Forest...")
            model, X_train, X_test, y_train, y_test, metrics = train_random_forest(X, y)
            print(f"   R² = {metrics['r2_test']:.3f} | MAE = {metrics['mae_test']:.2f}")

            # Compute SHAP
            explainer, shap_values = compute_shap_values(model, X_train, X_test)

            # Visualizations
            importance_df = create_shap_visualizations(
                biomarker, shap_values, X_test, feature_names,
                output_dir, metrics
            )

            # Vulnerability stratification
            quartile_importance = analyze_vulnerability_stratification(
                biomarker, shap_values, X_test, y_test,
                feature_names, output_dir
            )

            # Generate report
            report = generate_attribution_report(
                biomarker, metrics, importance_df,
                quartile_importance, n_samples
            )
            all_reports.append(report)

            print(f"   ✅ Complete!")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue

    # Save master report
    master_report = {
        'timestamp': datetime.now().isoformat(),
        'n_biomarkers_analyzed': len(all_reports),
        'n_features': len(FEATURES),
        'features': FEATURES,
        'biomarker_reports': all_reports
    }

    report_path = output_dir / 'attribution_master_report.json'
    with open(report_path, 'w') as f:
        json.dump(master_report, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("EXPLORATORY SHAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ Analyzed: {len(all_reports)} biomarkers")
    print(f"✅ Master report: {report_path}")
    print(f"✅ Visualizations: {output_dir}/")

    # Print top findings
    print("\n📊 SUMMARY OF KEY FINDINGS:")
    print("-" * 80)

    for report in sorted(all_reports, key=lambda x: x['performance']['r2_test'], reverse=True)[:10]:
        print(f"\n{report['biomarker']}:")
        print(f"   R² = {report['performance']['r2_test']:.3f}")
        print(f"   Interpretation: {report['interpretation']}")
        print(f"   Top feature: {report['top_5_features'][-1]['feature']} "
              f"({report['top_5_features'][-1]['pct_contribution']:.1f}%)")

    print("\n🎯 Next: Review visualizations in results/shap_attribution/")


if __name__ == '__main__':
    main()
