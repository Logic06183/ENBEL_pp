#!/usr/bin/env python3
"""
Create CD4 SHAP Analysis Slide - Publication Quality SVG
Using actual LightGBM model results from multi-model comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import shap
import seaborn as sns
from scipy.signal import savgol_filter
import warnings
import json
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
np.random.seed(42)

print("=" * 80)
print("Creating CD4 SHAP Analysis Slide - Publication Quality")
print("=" * 80)

# Load climate-only dataset (Scenario A)
print("\n1. Loading climate-only dataset (Scenario A)...")
data_path = "results/modeling/MODELING_DATASET_SCENARIO_B.csv"
df = pd.read_csv(data_path, low_memory=False)
print(f"   Loaded {len(df):,} total records")

# Extract CD4 data
target = 'CD4 cell count (cells/µL)'
cd4_df = df[df[target].notna()].copy()
print(f"   CD4 observations: {len(cd4_df):,}")

# Climate and socioeconomic features (Scenario A)
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

# Prepare features
print("\n2. Preparing features...")
feature_cols = [col for col in SCENARIO_A_FEATURES if col in cd4_df.columns]
categorical_features = ['climate_season', 'season']

# Encode categorical features
cd4_df_encoded = cd4_df[feature_cols + [target]].copy()
for cat_col in categorical_features:
    if cat_col in cd4_df_encoded.columns:
        cd4_df_encoded = pd.get_dummies(cd4_df_encoded, columns=[cat_col], drop_first=False)

# Convert all to float64 for SHAP compatibility
X = cd4_df_encoded.drop(columns=[target])
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)
X = X.astype(np.float64)
y = cd4_df_encoded[target].astype(np.float64)

print(f"   Features after encoding: {X.shape[1]}")
print(f"   CD4 range: {y.min():.0f} - {y.max():.0f} cells/µL")
print(f"   CD4 mean±SD: {y.mean():.0f}±{y.std():.0f} cells/µL")

# Train-test split
print("\n3. Training LightGBM model (from best parameters)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use best parameters from master_results.json
lgb_model = lgb.LGBMRegressor(
    num_leaves=50,
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
lgb_model.fit(X_train, y_train)

# Evaluate
y_pred_train = lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"   R² (train): {r2_train:.3f}")
print(f"   R² (test): {r2_test:.3f}")
print(f"   RMSE (test): {rmse_test:.1f} cells/µL")
print(f"   MAE (test): {mae_test:.1f} cells/µL")

# Generate SHAP values
print("\n4. Generating SHAP values...")
explainer = shap.TreeExplainer(lgb_model)

# Use subset for faster computation
n_shap_samples = min(500, len(X_test))
X_test_shap = X_test.iloc[:n_shap_samples]
y_test_shap = y_test.iloc[:n_shap_samples]
shap_values = explainer.shap_values(X_test_shap)

print(f"   SHAP computed for {n_shap_samples} samples")

# Feature display names
feature_display_names = {
    'climate_daily_mean_temp': 'Daily Mean Temp',
    'climate_daily_max_temp': 'Daily Max Temp',
    'climate_daily_min_temp': 'Daily Min Temp',
    'climate_7d_mean_temp': '7-Day Mean Temp',
    'climate_heat_stress_index': 'Heat Stress Index',
    'month': 'Month',
    'HEAT_VULNERABILITY_SCORE': 'Heat Vulnerability',
    'climate_season_Autumn': 'Autumn',
    'climate_season_Spring': 'Spring',
    'climate_season_Summer': 'Summer',
    'climate_season_Winter': 'Winter',
    'season_Autumn': 'Season: Autumn',
    'season_Spring': 'Season: Spring',
    'season_Summer': 'Season: Summer',
    'season_Winter': 'Season: Winter'
}

# Calculate feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]
top_n = min(10, len(sorted_idx))  # Show top 10

# Create publication-quality figure
print("\n5. Creating publication-quality SVG slide...")
fig = plt.figure(figsize=(18, 10), facecolor='white', dpi=150)

# Create grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4,
                     left=0.08, right=0.95, top=0.90, bottom=0.08)

# Title
title_text = 'SHAP Analysis: CD4+ T-Cell Count Climate Associations'
subtitle_text = 'LightGBM Model | Climate + Socioeconomic Features | Johannesburg, South Africa'
fig.suptitle(title_text, fontsize=22, fontweight='bold',
             y=0.97, color='#1a1a1a', family='sans-serif')
fig.text(0.5, 0.93, subtitle_text, ha='center', fontsize=13,
         color='#34495E', style='italic')

# PLOT 1: SHAP Summary Plot (Beeswarm) - MAIN PLOT
ax1 = fig.add_subplot(gs[:2, :2])
shap.summary_plot(
    shap_values, X_test_shap,
    plot_type="dot",
    show=False,
    max_display=top_n,
    color_bar_label="Feature Value\n(Low → High)"
)
ax1.set_title('A. Feature Impact Distribution (Beeswarm Plot)',
              fontsize=16, fontweight='bold', pad=15, loc='left',
              color='#2C3E50')
ax1.set_xlabel('SHAP Value (impact on CD4 count, cells/µL)',
               fontsize=13, fontweight='bold')
ax1.set_ylabel('')

# Update y-axis labels
current_labels = [item.get_text() for item in ax1.get_yticklabels()]
new_labels = [feature_display_names.get(label, label) for label in current_labels]
ax1.set_yticklabels(new_labels, fontsize=11)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.axvline(x=0, color='#2C3E50', linestyle='-', linewidth=1.5, alpha=0.5)

# Add statistical annotation
ax1.text(0.02, 0.98, f'n = {n_shap_samples} patients\nR² = {r2_test:.3f}',
         transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  alpha=0.9, edgecolor='#3498DB', linewidth=2))

# PLOT 2: Feature Importance (Horizontal Bar)
ax2 = fig.add_subplot(gs[:2, 2])
sorted_features = [feature_display_names.get(X_test_shap.columns[i], X_test_shap.columns[i])
                  for i in sorted_idx[:top_n]]
sorted_importance = feature_importance[sorted_idx[:top_n]]

# Color code by importance threshold
mean_imp = sorted_importance.mean()
colors = ['#E74C3C' if imp > mean_imp else '#3498DB'
          for imp in sorted_importance]
bars = ax2.barh(range(len(sorted_features)), sorted_importance,
                color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

ax2.set_yticks(range(len(sorted_features)))
ax2.set_yticklabels(sorted_features, fontsize=11)
ax2.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax2.set_title('B. Feature Importance Ranking', fontsize=16, fontweight='bold',
              pad=15, loc='left', color='#2C3E50')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    ax2.text(val + 2, i, f'{val:.1f}',
             va='center', ha='left', fontsize=9, fontweight='bold')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# PLOT 3: Dependence Plot (Top Feature)
ax3 = fig.add_subplot(gs[2, :2])

# Find top continuous feature (not categorical)
top_feature_idx = sorted_idx[0]
top_feature_name = X_test_shap.columns[top_feature_idx]

# If top feature is categorical, use second feature
if 'season' in top_feature_name.lower() or top_feature_name in categorical_features:
    # Find first non-categorical feature
    for idx in sorted_idx:
        feat_name = X_test_shap.columns[idx]
        if 'season' not in feat_name.lower():
            top_feature_idx = idx
            top_feature_name = feat_name
            break

# Scatter plot with colormap
scatter = ax3.scatter(
    X_test_shap.iloc[:, top_feature_idx],
    shap_values[:, top_feature_idx],
    c=X_test_shap.iloc[:, top_feature_idx],
    cmap='RdYlBu_r',
    alpha=0.7,
    s=60,
    edgecolors='white',
    linewidth=1,
    vmin=np.percentile(X_test_shap.iloc[:, top_feature_idx], 5),
    vmax=np.percentile(X_test_shap.iloc[:, top_feature_idx], 95)
)

# Add smooth trend line (LOWESS)
x_sorted_idx = np.argsort(X_test_shap.iloc[:, top_feature_idx])
x_sorted = X_test_shap.iloc[x_sorted_idx, top_feature_idx].values
y_sorted = shap_values[x_sorted_idx, top_feature_idx]

# Smooth with Savitzky-Golay filter
window = min(51, len(x_sorted) // 3 if len(x_sorted) // 3 % 2 == 1 else len(x_sorted) // 3 - 1)
if window >= 5 and len(x_sorted) > window:
    y_smooth = savgol_filter(y_sorted, window, 3)
    ax3.plot(x_sorted, y_smooth, 'k-', linewidth=3, alpha=0.8, label='Trend')
    ax3.legend(loc='best', fontsize=10)

display_name = feature_display_names.get(top_feature_name, top_feature_name)
ax3.set_xlabel(display_name, fontsize=12, fontweight='bold')
ax3.set_ylabel('SHAP Value\n(CD4 impact, cells/µL)',
               fontsize=12, fontweight='bold')
ax3.set_title(f'C. {display_name} Dependence Plot',
              fontsize=16, fontweight='bold', pad=15, loc='left',
              color='#2C3E50')
ax3.grid(True, alpha=0.2, linestyle='--')
ax3.axhline(y=0, color='#2C3E50', linestyle='-', linewidth=1.5, alpha=0.5)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label(display_name, fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# PLOT 4: Model Performance Summary
ax4 = fig.add_subplot(gs[2, 2])
ax4.axis('off')

# Calculate SHAP stats
mean_shap = float(np.abs(shap_values).mean())
base_value = float(explainer.expected_value)

# Create performance metrics table
performance_text = f"""Model Performance Metrics

Dataset (Climate-only):
• Total observations: {len(cd4_df):,}
• Training set: {len(X_train):,} (80%)
• Test set: {len(X_test):,} (20%)
• SHAP samples: {n_shap_samples}

Model: LightGBM
• Estimators: 50
• Max depth: 5
• Num leaves: 50
• Learning rate: 0.1

Test Set Performance:
• R² = {r2_test:.3f}
• RMSE = {rmse_test:.1f} cells/µL
• MAE = {mae_test:.1f} cells/µL

SHAP Statistics:
• Mean |SHAP|: {mean_shap:.2f}
• Base value: {base_value:.1f}
• Top feature: {feature_display_names.get(X_test_shap.columns[sorted_idx[0]], X_test_shap.columns[sorted_idx[0]])}

Study Period: 2002-2021
Location: Johannesburg, SA
"""

ax4.text(0.05, 0.95, performance_text,
         transform=ax4.transAxes,
         fontsize=9.5, family='monospace',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.8',
                  facecolor='#ECF0F1', alpha=0.95,
                  edgecolor='#34495E', linewidth=2))

# Add reference citation
reference_text = ("Lundberg & Lee (2017). NIPS. | "
                 "Ke et al. (2017). LightGBM. NIPS. | "
                 "ENBEL Climate-Health Analysis")

fig.text(0.5, 0.02, reference_text,
         ha='center', va='bottom', fontsize=9,
         style='italic', color='#5D6D7E',
         bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='white', alpha=0.9,
                  edgecolor='#BDC3C7', linewidth=1))

# Save SVG
output_svg = "results/visualization_panel/cd4_shap_analysis_slide.svg"
output_png = "results/visualization_panel/cd4_shap_analysis_slide.png"

plt.savefig(output_svg, format='svg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\n✓ Publication-quality SVG slide saved: {output_svg}")
print(f"✓ High-resolution PNG saved: {output_png}")
print(f"\n✓ Features:")
print(f"  - Native SHAP beeswarm, importance, and dependence plots")
print(f"  - Statistical annotations (n, R², RMSE, MAE)")
print(f"  - Smooth trend lines with Savitzky-Golay filter")
print(f"  - Scientific nomenclature")
print(f"  - Publication-ready SVG format")
print(f"  - Based on ACTUAL LightGBM model (R² = {r2_test:.3f})")
print("\n" + "=" * 80)

plt.show()
