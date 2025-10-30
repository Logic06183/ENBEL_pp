"""
Create Streamlined Visualization Panel
======================================

Focus on key findings:
1. CD4 as climate-sensitive biomarker
2. Hematocrit caveat (socioeconomic dominance)
3. Model comparison across biomarkers

Author: ENBEL Team
Date: 2025-10-30
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
np.random.seed(42)

print("="*80)
print("CREATING STREAMLINED VISUALIZATION PANEL")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Focus on top 3 biomarkers
TOP_BIOMARKERS = [
    'CD4 cell count (cells/µL)',
    'Hematocrit (%)',
    'FASTING LDL'
]

# Climate features
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

CATEGORICAL_FEATURES = ['climate_season', 'season']

# =============================================================================
# LOAD DATA AND RESULTS
# =============================================================================

base_dir = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(base_dir / "results" / "modeling" / "MODELING_DATASET_SCENARIO_B.csv",
                 low_memory=False)

# Load results
with open(base_dir / "results" / "multi_model_comparison" / "master_results.json", 'r') as f:
    results = json.load(f)

# Output directory
output_dir = base_dir / "results" / "visualization_panel"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n✅ Data loaded: {len(df):,} records")
print(f"✅ Results loaded: {len(results)} biomarkers")

# =============================================================================
# 1. CROSS-BIOMARKER FEATURE IMPORTANCE HEATMAP
# =============================================================================

print("\n" + "="*80)
print("1. CREATING FEATURE IMPORTANCE HEATMAP")
print("="*80)

# Extract feature importance from SHAP for all biomarkers
importance_matrix = []
biomarker_names = []

for result in results:
    if 'error' in result or 'A' not in result['scenarios']:
        continue

    scenario_a = result['scenarios']['A']
    if 'error' in scenario_a or 'shap_importance' not in scenario_a:
        continue

    biomarker = result['biomarker']
    biomarker_names.append(biomarker)

    # Get feature importance
    shap_importance = {item['feature']: item['importance']
                      for item in scenario_a['shap_importance']}

    importance_matrix.append(shap_importance)

# Convert to DataFrame
df_importance = pd.DataFrame(importance_matrix, index=biomarker_names)

# Get top 10 most important features overall
top_features = df_importance.sum(axis=0).nlargest(10).index.tolist()
df_importance_top = df_importance[top_features]

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_importance_top, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Mean |SHAP| Value'}, ax=ax)
plt.title('Feature Importance Across Biomarkers (Climate-Only Models)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Biomarkers', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '01_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: 01_feature_importance_heatmap.png")

# =============================================================================
# 2. MODEL COMPARISON BAR CHART
# =============================================================================

print("\n" + "="*80)
print("2. CREATING MODEL COMPARISON CHART")
print("="*80)

# Extract R² for climate-only models
model_comparison = []
for result in results:
    if 'error' in result or 'A' not in result['scenarios']:
        continue

    scenario_a = result['scenarios']['A']
    if 'error' in scenario_a:
        continue

    biomarker = result['biomarker']
    best_model = scenario_a.get('best_model', 'Unknown')
    r2 = scenario_a['models'].get(best_model, {}).get('r2', 0)

    model_comparison.append({
        'Biomarker': biomarker,
        'R²': r2,
        'Model': best_model
    })

df_comparison = pd.DataFrame(model_comparison).sort_values('R²', ascending=True)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'RandomForest': '#2ecc71', 'XGBoost': '#3498db', 'LightGBM': '#e74c3c'}
bar_colors = [colors.get(m, '#95a5a6') for m in df_comparison['Model']]

bars = ax.barh(df_comparison['Biomarker'], df_comparison['R²'], color=bar_colors)
ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Biomarker', fontsize=12, fontweight='bold')
ax.set_title('Model Performance (Climate-Only, Tuned Hyperparameters)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1.0)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, r2) in enumerate(zip(bars, df_comparison['R²'])):
    ax.text(r2 + 0.01, bar.get_y() + bar.get_height()/2,
            f'{r2:.3f}', va='center', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[m], label=m)
                  for m in ['RandomForest', 'XGBoost', 'LightGBM']]
ax.legend(handles=legend_elements, loc='lower right', title='Best Model')

plt.tight_layout()
plt.savefig(output_dir / '02_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: 02_model_comparison.png")

# =============================================================================
# 3. DEMOGRAPHICS IMPACT (ΔR²)
# =============================================================================

print("\n" + "="*80)
print("3. CREATING DEMOGRAPHICS IMPACT CHART")
print("="*80)

# Extract ΔR²
delta_r2_data = []
for result in results:
    if 'error' in result or 'delta_r2' not in result:
        continue

    delta_r2_data.append({
        'Biomarker': result['biomarker'],
        'ΔR²': result['delta_r2']
    })

df_delta = pd.DataFrame(delta_r2_data).sort_values('ΔR²')

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 8))
colors_delta = ['#e74c3c' if x < 0 else '#2ecc71' for x in df_delta['ΔR²']]

bars = ax.barh(df_delta['Biomarker'], df_delta['ΔR²'], color=colors_delta)
ax.set_xlabel('ΔR² (Full Model - Climate Only)', fontsize=12, fontweight='bold')
ax.set_ylabel('Biomarker', fontsize=12, fontweight='bold')
ax.set_title('Impact of Adding Demographics + Study Features',
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, delta in zip(bars, df_delta['ΔR²']):
    x_pos = delta + (0.005 if delta > 0 else -0.005)
    ha = 'left' if delta > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{delta:+.3f}', va='center', ha=ha, fontsize=9)

# Add annotations
ax.text(0.5, 0.98, 'Positive: Demographics help',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=10, style='italic', color='#2ecc71')
ax.text(0.5, 0.02, 'Negative: Demographics hurt (overfitting/sample loss)',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=10, style='italic', color='#e74c3c')

plt.tight_layout()
plt.savefig(output_dir / '03_demographics_impact.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: 03_demographics_impact.png")

# =============================================================================
# 4. DETAILED SHAP PLOTS FOR TOP 3 BIOMARKERS
# =============================================================================

print("\n" + "="*80)
print("4. CREATING DETAILED SHAP PLOTS FOR TOP BIOMARKERS")
print("="*80)

def prepare_features(df, features, categorical_features, target_biomarker):
    """Prepare features for modeling."""
    df_subset = df[df[target_biomarker].notna()].copy()

    X = df_subset[features].copy()
    y = df_subset[target_biomarker].copy()

    # One-hot encode
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

    # Ensure numeric
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            X_encoded[col] = pd.Categorical(X_encoded[col]).codes
        elif X_encoded[col].dtype == 'bool':
            X_encoded[col] = X_encoded[col].astype(int)

    X_encoded = X_encoded.astype(np.float64)
    feature_names = X_encoded.columns.tolist()

    return X_encoded, y, feature_names

for biomarker in TOP_BIOMARKERS:
    print(f"\n📊 Processing: {biomarker}")

    # Find result
    result = next((r for r in results if r['biomarker'] == biomarker), None)
    if not result or 'A' not in result['scenarios']:
        print(f"   ⚠️  Skipping (no data)")
        continue

    scenario_a = result['scenarios']['A']
    if 'error' in scenario_a:
        print(f"   ⚠️  Skipping (error in scenario)")
        continue

    # Prepare data
    X, y, feature_names = prepare_features(df, SCENARIO_A_FEATURES,
                                           CATEGORICAL_FEATURES, biomarker)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train best model
    best_model_name = scenario_a['best_model']
    r2 = scenario_a['models'][best_model_name]['r2']

    if best_model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     min_samples_split=10, min_samples_leaf=5,
                                     random_state=42, n_jobs=-1)
    elif best_model_name == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                random_state=42, n_jobs=-1, verbosity=0)
    elif best_model_name == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                 random_state=42, n_jobs=-1, verbosity=-1)

    model.fit(X_train, y_train)

    # Compute SHAP
    if len(X_train) > 1000:
        background = shap.sample(X_train, 1000, random_state=42)
    else:
        background = X_train

    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

    # Create safe filename
    biomarker_safe = biomarker.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

    # Plot 1: Beeswarm (shows distribution of SHAP values)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                     max_display=10, show=False)
    plt.title(f'{biomarker}\nSHAP Summary (Beeswarm) | {best_model_name} | R² = {r2:.3f}',
             fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / f'04_{biomarker_safe}_beeswarm.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Bar (feature importance)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type='bar',
                     feature_names=feature_names, max_display=10, show=False)
    plt.title(f'{biomarker}\nFeature Importance | {best_model_name} | R² = {r2:.3f}',
             fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / f'04_{biomarker_safe}_importance.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Top 3 dependence plots
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_3_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, idx in enumerate(top_3_idx):
        shap.dependence_plot(idx, shap_values, X_test,
                            feature_names=feature_names,
                            ax=axes[i], show=False)

    plt.suptitle(f'{biomarker}\nTop 3 Feature Dependence Plots | {best_model_name} | R² = {r2:.3f}',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'04_{biomarker_safe}_dependence.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Created 3 SHAP plots")

# =============================================================================
# 5. SUMMARY STATISTICS TABLE
# =============================================================================

print("\n" + "="*80)
print("5. CREATING SUMMARY TABLE")
print("="*80)

# Create summary table
summary_data = []
for result in results:
    if 'error' in result:
        continue

    biomarker = result['biomarker']

    # Scenario A
    if 'A' in result['scenarios'] and 'error' not in result['scenarios']['A']:
        scenario_a = result['scenarios']['A']
        best_model_a = scenario_a.get('best_model', 'N/A')
        r2_a = scenario_a['models'].get(best_model_a, {}).get('r2', 0)
        mae_a = scenario_a['models'].get(best_model_a, {}).get('mae', 0)
    else:
        best_model_a = 'N/A'
        r2_a = 0
        mae_a = 0

    # Scenario B
    if 'B' in result['scenarios'] and 'error' not in result['scenarios']['B']:
        scenario_b = result['scenarios']['B']
        best_model_b = scenario_b.get('best_model', 'N/A')
        r2_b = scenario_b['models'].get(best_model_b, {}).get('r2', 0)
        mae_b = scenario_b['models'].get(best_model_b, {}).get('mae', 0)
    else:
        best_model_b = 'N/A'
        r2_b = 0
        mae_b = 0

    delta_r2 = result.get('delta_r2', r2_b - r2_a)

    summary_data.append({
        'Biomarker': biomarker,
        'Climate R²': f'{r2_a:.3f}',
        'Climate MAE': f'{mae_a:.2f}',
        'Climate Model': best_model_a,
        'Full R²': f'{r2_b:.3f}',
        'Full MAE': f'{mae_b:.2f}',
        'Full Model': best_model_b,
        'ΔR²': f'{delta_r2:+.3f}'
    })

df_summary = pd.DataFrame(summary_data)

# Save as CSV
df_summary.to_csv(output_dir / '05_summary_table.csv', index=False)
print(f"✅ Saved: 05_summary_table.csv")

# Create styled table image
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                cellLoc='center', loc='center',
                colWidths=[0.20, 0.10, 0.10, 0.12, 0.10, 0.10, 0.12, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(df_summary.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(df_summary) + 1):
    for j in range(len(df_summary.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

plt.title('Model Performance Summary (Tuned Hyperparameters)',
         fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / '05_summary_table.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: 05_summary_table.png")

# =============================================================================
# COMPLETE
# =============================================================================

print("\n" + "="*80)
print("VISUALIZATION PANEL COMPLETE")
print("="*80)
print(f"\n📁 Output directory: {output_dir}")
print(f"\n📊 Files created:")
print(f"   1. 01_feature_importance_heatmap.png")
print(f"   2. 02_model_comparison.png")
print(f"   3. 03_demographics_impact.png")
print(f"   4. 04_[biomarker]_beeswarm.png (×3 biomarkers)")
print(f"   5. 04_[biomarker]_importance.png (×3 biomarkers)")
print(f"   6. 04_[biomarker]_dependence.png (×3 biomarkers)")
print(f"   7. 05_summary_table.csv")
print(f"   8. 05_summary_table.png")
print(f"\n✅ Total: ~14 visualizations")
print(f"\n🎯 Next: Review visualizations and identify features for Phase 2 testing")
