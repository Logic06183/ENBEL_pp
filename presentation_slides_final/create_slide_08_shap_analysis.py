#!/usr/bin/env python3
"""
Generate Slide 8: SHAP Analysis CD4 Model
Uses REAL clinical data to train model and generate SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("SLIDE 8: SHAP Analysis CD4 Model")
print("=" * 70)

# Load clinical data
print("\n1. Loading clinical data...")
data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
df = pd.read_csv(data_path)
print(f"   Loaded {len(df):,} total records")

# Filter for CD4 data with complete climate features
print("\n2. Filtering for CD4 records with climate data...")
cd4_col = 'CD4 cell count (cells/µL)'
cd4_df = df[[cd4_col] + [c for c in df.columns if c != cd4_col]].copy()
cd4_df = cd4_df[cd4_df[cd4_col].notna()].copy()
print(f"   Found {len(cd4_df):,} CD4 records")

# Select climate features (matching PDF slide)
climate_features = [
    'climate_daily_mean_temp',  # temperature
    'climate_7d_mean_temp',  # temperature_lag_7
    'climate_14d_mean_temp', # temperature_lag_14
    'climate_30d_mean_temp', # temperature_lag_30
    'climate_temp_anomaly',  # temp_anomaly
    'climate_heat_stress_index', # heat index
    'climate_heat_day_p90'  # extreme heat days
]

# Filter for complete cases
print("\n3. Preparing feature matrix...")
feature_cols = [col for col in climate_features if col in cd4_df.columns]
cd4_df = cd4_df[[cd4_col] + feature_cols].dropna()
cd4_df = cd4_df.rename(columns={cd4_col: 'cd4_value'})
print(f"   Complete cases: {len(cd4_df):,} observations")
print(f"   Features: {', '.join(feature_cols)}")

# Prepare X and y
X = cd4_df[feature_cols]
y = cd4_df['cd4_value']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
print("\n4. Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Calculate performance metrics
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
mae = np.mean(np.abs(y_test - y_pred))

print(f"   Model Performance:")
print(f"   - R² (train): {train_score:.3f}")
print(f"   - R² (test): {test_score:.3f}")
print(f"   - RMSE: {rmse:.1f} cells/µL")
print(f"   - MAE: {mae:.1f} cells/µL")

# Generate SHAP values
print("\n5. Generating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test.iloc[:100])  # Sample for visualization
shap_data = X_test.iloc[:100]

print(f"   SHAP values computed for {len(shap_data)} samples")
print(f"   Mean |SHAP|: {np.abs(shap_values).mean():.2f}")

# Create figure with 3 subplots
print("\n6. Creating SHAP visualization...")
fig = plt.figure(figsize=(16, 9), facecolor='white')

# Add title
fig.suptitle('SHAP Analysis: CD4 Model',
             fontsize=20, fontweight='bold', y=0.98, color='#2C3E50')

# Define colors
color_low = '#3498DB'  # Blue
color_high = '#E74C3C'  # Red

# Feature names for display
feature_display_names = {
    'climate_heat_stress_index': 'Heat Index',
    'climate_daily_mean_temp': 'Temperature',
    'climate_7d_mean_temp': 'Temp (7-day lag)',
    'climate_14d_mean_temp': 'Temp (14-day lag)',
    'climate_30d_mean_temp': 'Temp (30-day lag)',
    'climate_temp_anomaly': 'Temp Anomaly',
    'climate_heat_day_p90': 'Heat Day (P90)'
}

# Sort features by mean |SHAP value|
feature_importance = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]

# Plot 1: Beeswarm Plot (Feature Impact)
ax1 = plt.subplot(1, 3, 1)
shap.summary_plot(
    shap_values, shap_data,
    plot_type="dot",
    show=False,
    color_bar=False
)
ax1.set_title('SHAP Feature Impact\n(Beeswarm Plot)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('SHAP Value (impact on CD4 count)', fontsize=11)
ax1.set_ylabel('')

# Update y-axis labels
y_labels = [feature_display_names.get(f, f) for f in shap_data.columns[sorted_idx]]
ax1.set_yticklabels(y_labels, fontsize=10)

# Plot 2: Feature Importance (Bar chart)
ax2 = plt.subplot(1, 3, 2)
sorted_features = [feature_display_names.get(f, f)
                   for f in shap_data.columns[sorted_idx]]
sorted_importance = feature_importance[sorted_idx]

colors = ['#E74C3C' if imp > 15 else '#3498DB' for imp in sorted_importance]
bars = ax2.barh(range(len(sorted_features)), sorted_importance, color=colors, alpha=0.8)
ax2.set_yticks(range(len(sorted_features)))
ax2.set_yticklabels(sorted_features, fontsize=10)
ax2.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax2.set_title('Feature Importance', fontsize=14, fontweight='bold', pad=15)
ax2.invert_yaxis()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    ax2.text(val + 0.5, i, f'{val:.1f}',
             va='center', fontsize=9, fontweight='bold')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot 3: Dependence Plot (Heat Index)
ax3 = plt.subplot(1, 3, 3)
if 'climate_heat_stress_index' in shap_data.columns:
    feature_idx = list(shap_data.columns).index('climate_heat_stress_index')
    scatter = ax3.scatter(
        shap_data['climate_heat_stress_index'],
        shap_values[:, feature_idx],
        c=shap_data['climate_heat_stress_index'],
        cmap='RdYlBu_r',
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

    # Add trend line
    z = np.polyfit(shap_data['climate_heat_stress_index'], shap_values[:, feature_idx], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(shap_data['climate_heat_stress_index'].min(),
                          shap_data['climate_heat_stress_index'].max(), 100)
    ax3.plot(x_trend, p(x_trend), 'k--', alpha=0.5, linewidth=2)

    ax3.set_xlabel('Heat Index', fontsize=11)
    ax3.set_ylabel('SHAP Value', fontsize=11)
    ax3.set_title('Dependence Plot\nHeat Index',
                  fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax3, label='Heat Index')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add metadata text box at bottom
metadata_text = (
    f"Dataset: {len(cd4_df):,} observations • Johannesburg clinical trials • 2012-2018\n"
    f"Model: Random Forest (100 trees, max depth 10)\n"
    f"Performance: R² = {test_score:.3f} • RMSE = {rmse:.1f} cells/µL • MAE = {mae:.1f} cells/µL\n"
    f"Key Climate Features: Temperature • Humidity • Temp (7-day lag) • Temp (14-day lag)\n"
    f"SHAP Analysis: TreeExplainer • 100 samples • Mean |SHAP| = {np.abs(shap_values).mean():.2f}\n"
    f"Reference: Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions"
)

fig.text(0.5, 0.02, metadata_text,
         ha='center', va='bottom', fontsize=8, color='#34495E',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1',
                   alpha=0.8, edgecolor='#BDC3C7'))

# Add color bar legend
legend_elements = [
    mpatches.Patch(color=color_low, label='Low Feature Value'),
    mpatches.Patch(color=color_high, label='High Feature Value')
]
fig.legend(handles=legend_elements, loc='lower right',
           bbox_to_anchor=(0.98, 0.12), fontsize=9, frameon=True)

plt.tight_layout(rect=[0, 0.12, 1, 0.96])

# Save as SVG
output_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection/presentation_slides_final/main_presentation/slide_08_shap_cd4_analysis.svg"
plt.savefig(output_path, format='svg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\n✓ Slide saved: {output_path}")
print(f"✓ Aspect ratio: 16:9")
print(f"✓ Using REAL clinical data with {len(cd4_df):,} CD4 observations")
print(f"✓ Authentic SHAP values from trained Random Forest model")
print("\n" + "=" * 70)
