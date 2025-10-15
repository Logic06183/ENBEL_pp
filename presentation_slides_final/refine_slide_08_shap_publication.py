#!/usr/bin/env python3
"""
Refine Slide 8: SHAP Analysis CD4 Model - Publication Quality
Enhanced scientific visualization with proper statistical annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
np.random.seed(42)

print("=" * 80)
print("REFINING SLIDE 8: SHAP Analysis CD4 Model (Publication Quality)")
print("=" * 80)

# Load and prepare data
print("\n1. Loading clinical data...")
data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
df = pd.read_csv(data_path, low_memory=False)
print(f"   Loaded {len(df):,} total records")

# Extract CD4 data
print("\n2. Preparing CD4 dataset...")
cd4_col = 'CD4 cell count (cells/µL)'
cd4_df = df[[cd4_col] + [c for c in df.columns if c != cd4_col]].copy()
cd4_df = cd4_df[cd4_df[cd4_col].notna()].copy()

# Climate features
climate_features = [
    'climate_heat_stress_index',
    'climate_daily_mean_temp',
    'climate_7d_mean_temp',
    'climate_14d_mean_temp',
    'climate_30d_mean_temp',
    'climate_temp_anomaly',
    'climate_heat_day_p90'
]

feature_cols = [col for col in climate_features if col in cd4_df.columns]
cd4_df = cd4_df[[cd4_col] + feature_cols].dropna()
cd4_df = cd4_df.rename(columns={cd4_col: 'cd4_value'})

X = cd4_df[feature_cols]
y = cd4_df['cd4_value']

print(f"   Complete cases: {len(cd4_df):,} observations")
print(f"   Features: {len(feature_cols)}")
print(f"   CD4 range: {y.min():.0f} - {y.max():.0f} cells/µL")
print(f"   CD4 mean±SD: {y.mean():.0f}±{y.std():.0f} cells/µL")

# Train model
print("\n3. Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

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
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test.iloc[:200])  # Larger sample
shap_data = X_test.iloc[:200]

# Create publication-quality figure
print("\n5. Creating publication-quality visualization...")
fig = plt.figure(figsize=(18, 10), facecolor='white', dpi=150)

# Create grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4,
                     left=0.08, right=0.95, top=0.92, bottom=0.08)

# Title with metadata
title_text = 'SHAP Analysis: CD4+ T-Cell Count Climate Associations'
fig.suptitle(title_text, fontsize=22, fontweight='bold',
             y=0.98, color='#1a1a1a', family='sans-serif')

# Feature display names (scientific)
feature_display_names = {
    'climate_heat_stress_index': 'Heat Stress Index',
    'climate_daily_mean_temp': 'Daily Mean Temperature',
    'climate_7d_mean_temp': 'Temperature (7-day mean)',
    'climate_14d_mean_temp': 'Temperature (14-day mean)',
    'climate_30d_mean_temp': 'Temperature (30-day mean)',
    'climate_temp_anomaly': 'Temperature Anomaly',
    'climate_heat_day_p90': 'Extreme Heat Days (P90)'
}

# Calculate feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]

# PLOT 1: SHAP Summary Plot (Beeswarm) - MAIN PLOT
ax1 = fig.add_subplot(gs[:2, :2])
shap.summary_plot(
    shap_values, shap_data,
    plot_type="dot",
    show=False,
    color_bar_label="Feature Value\n(Low → High)"
)
ax1.set_title('A. SHAP Feature Impact Distribution',
              fontsize=16, fontweight='bold', pad=15, loc='left')
ax1.set_xlabel('SHAP Value (impact on CD4 count, cells/µL)',
               fontsize=13, fontweight='bold')
ax1.set_ylabel('Climate Features', fontsize=13, fontweight='bold')

# Update y-axis labels with scientific names
y_labels = [feature_display_names.get(f, f) for f in shap_data.columns[sorted_idx]]
ax1.set_yticklabels(y_labels, fontsize=11)
ax1.grid(True, alpha=0.2, linestyle='--', axis='x')
ax1.axvline(x=0, color='#2C3E50', linestyle='-', linewidth=1.5, alpha=0.5)

# Add statistical annotation
ax1.text(0.02, 0.98, f'n = {len(shap_data)} observations',
         transform=ax1.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white',
                  alpha=0.8, edgecolor='#3498DB', linewidth=1.5))

# PLOT 2: Feature Importance (Horizontal Bar)
ax2 = fig.add_subplot(gs[:2, 2])
sorted_features = [feature_display_names.get(f, f)
                  for f in shap_data.columns[sorted_idx]]
sorted_importance = feature_importance[sorted_idx]

# Color code by importance threshold
colors = ['#E74C3C' if imp > sorted_importance.mean() else '#3498DB'
          for imp in sorted_importance]
bars = ax2.barh(range(len(sorted_features)), sorted_importance,
                color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

ax2.set_yticks(range(len(sorted_features)))
ax2.set_yticklabels(sorted_features, fontsize=11)
ax2.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax2.set_title('B. Feature Importance', fontsize=16, fontweight='bold',
              pad=15, loc='left')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.2, linestyle='--', axis='x')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    ax2.text(val + 1, i, f'{val:.1f}',
             va='center', ha='left', fontsize=9, fontweight='bold')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# PLOT 3: Dependence Plot (Heat Stress Index)
ax3 = fig.add_subplot(gs[2, :2])
if 'climate_heat_stress_index' in shap_data.columns:
    feature_idx = list(shap_data.columns).index('climate_heat_stress_index')

    # Scatter plot with colormap
    scatter = ax3.scatter(
        shap_data['climate_heat_stress_index'],
        shap_values[:, feature_idx],
        c=shap_data['climate_heat_stress_index'],
        cmap='RdYlBu_r',
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=1,
        vmin=shap_data['climate_heat_stress_index'].quantile(0.05),
        vmax=shap_data['climate_heat_stress_index'].quantile(0.95)
    )

    # Add smooth trend line (LOWESS)
    from scipy.signal import savgol_filter
    x_sorted = np.sort(shap_data['climate_heat_stress_index'])
    y_sorted = shap_values[:, feature_idx][np.argsort(shap_data['climate_heat_stress_index'])]

    # Smooth with Savitzky-Golay filter
    window = min(51, len(x_sorted) // 3 if len(x_sorted) // 3 % 2 == 1 else len(x_sorted) // 3 - 1)
    if window >= 5:
        y_smooth = savgol_filter(y_sorted, window, 3)
        ax3.plot(x_sorted, y_smooth, 'k-', linewidth=3, alpha=0.8, label='Trend')

    ax3.set_xlabel('Heat Stress Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('SHAP Value\n(CD4 impact, cells/µL)',
                   fontsize=12, fontweight='bold')
    ax3.set_title('C. Heat Stress Index Dependence',
                  fontsize=16, fontweight='bold', pad=15, loc='left')
    ax3.grid(True, alpha=0.2, linestyle='--')
    ax3.axhline(y=0, color='#2C3E50', linestyle='-', linewidth=1.5, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Heat Stress Index', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# PLOT 4: Model Performance Summary
ax4 = fig.add_subplot(gs[2, 2])
ax4.axis('off')

# Calculate mean SHAP
mean_shap = float(np.abs(shap_values).mean())
base_value = float(explainer.expected_value)

# Create performance metrics table
performance_text = f"""Model Performance Metrics

Dataset:
• Observations: {len(cd4_df):,}
• Training set: {len(X_train):,} (80%)
• Test set: {len(X_test):,} (20%)

Model: Random Forest
• Trees: 100
• Max depth: 10
• Features: {len(feature_cols)}

Test Set Performance:
• R² = {r2_test:.3f}
• RMSE = {rmse_test:.1f} cells/µL
• MAE = {mae_test:.1f} cells/µL

SHAP Analysis:
• Samples: {len(shap_data)}
• Mean |SHAP|: {mean_shap:.2f}
• Base value: {base_value:.1f}

Study Period: 2012-2018
Location: Johannesburg, SA
"""

ax4.text(0.05, 0.95, performance_text,
         transform=ax4.transAxes,
         fontsize=10, family='monospace',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.8',
                  facecolor='#ECF0F1', alpha=0.95,
                  edgecolor='#34495E', linewidth=2))

# Add reference citation
reference_text = ("Reference: Lundberg & Lee (2017). "
                 "A Unified Approach to Interpreting Model Predictions. "
                 "NIPS 2017.")

fig.text(0.5, 0.02, reference_text,
         ha='center', va='bottom', fontsize=9,
         style='italic', color='#5D6D7E',
         bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='white', alpha=0.9,
                  edgecolor='#BDC3C7', linewidth=1))

# Save
output_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection/presentation_slides_final/main_presentation/slide_08_shap_cd4_analysis.svg"
plt.savefig(output_path, format='svg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\n✓ Publication-quality slide saved: {output_path}")
print(f"✓ Enhanced with:")
print(f"  - Statistical annotations")
print(f"  - Smooth trend lines")
print(f"  - Performance metrics table")
print(f"  - Scientific nomenclature")
print(f"  - Publication-ready formatting")
print("\n" + "=" * 80)
