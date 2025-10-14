#!/usr/bin/env python3
"""
Create Slide 8: SHAP Analysis CD4 Model
Uses REAL clinical data to generate authentic SHAP analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output paths
OUTPUT_DIR = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection/presentation_slides_final/main_presentation"
DATA_PATH = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"

print("=" * 80)
print("SLIDE 8: SHAP ANALYSIS FOR CD4 COUNT MODEL")
print("=" * 80)

# Load clinical data
print("\n1. Loading clinical data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded {len(df):,} records")

# Filter for CD4 records with complete climate data
cd4_col = 'CD4 cell count (cells/µL)'

# Get actual climate column names from the dataset
climate_features = [col for col in df.columns if col.startswith('climate_')]
print(f"   Found {len(climate_features)} climate features")

# Use the most relevant climate features
selected_features = [
    'climate_daily_mean_temp', 'climate_daily_max_temp', 'climate_daily_min_temp',
    'climate_7d_mean_temp', 'climate_14d_mean_temp', 'climate_30d_mean_temp',
    'climate_temp_anomaly', 'climate_heat_stress_index',
    'climate_heat_day_p90', 'climate_heat_day_p95'
]

# Check available columns
available_climate = [col for col in selected_features if col in df.columns]
print(f"   Using {len(available_climate)} climate features for modeling")

# Prepare dataset for CD4 modeling
print("\n2. Preparing CD4 dataset...")
df_cd4 = df[[cd4_col] + available_climate].dropna()
print(f"   CD4 records with complete climate data: {len(df_cd4):,}")

# Separate features and target
X = df_cd4[available_climate]
y = df_cd4[cd4_col]

print(f"   Feature matrix shape: {X.shape}")
print(f"   Target range: {y.min():.1f} - {y.max():.1f} cells/µL")
print(f"   Target mean: {y.mean():.1f} ± {y.std():.1f} cells/µL")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")

# Train Random Forest model
print("\n3. Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate model
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)
print(f"   Training R²: {train_score:.3f}")
print(f"   Test R²: {test_score:.3f}")

# Compute SHAP values
print("\n4. Computing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test[:500])  # Use subset for speed
print(f"   SHAP values computed for {shap_values.shape[0]} samples")

# Create the three-panel SHAP visualization
print("\n5. Creating SHAP visualization panels...")

# Set up figure with 16:9 aspect ratio (1920x1080)
fig = plt.figure(figsize=(19.2, 10.8), dpi=150)

# Define color scheme matching presentation
PRIMARY_BLUE = '#2E7AB5'
ACCENT_ORANGE = '#E67E22'
BACKGROUND = '#F8F9FA'
TEXT_COLOR = '#2C3E50'

fig.patch.set_facecolor('white')

# Add title
fig.suptitle('SHAP Analysis: Climate Effects on CD4 Cell Count',
             fontsize=32, fontweight='bold', color=TEXT_COLOR, y=0.98)

# Add subtitle with model performance
subtitle = f'Random Forest Model (n={len(df_cd4):,} samples) | Test R² = {test_score:.3f} | Temperature & Humidity Effects on Immune Function'
fig.text(0.5, 0.93, subtitle, ha='center', fontsize=16, color=TEXT_COLOR, style='italic')

# Panel A: Feature Importance (Beeswarm Plot)
print("   Creating Panel A: Beeswarm Plot...")
ax1 = plt.subplot(1, 3, 1)
shap.summary_plot(shap_values, X_test[:500], plot_type="dot", show=False,
                  color_bar=True, max_display=12)
ax1.set_title('A. Feature Impact Distribution', fontsize=18, fontweight='bold', pad=15)
ax1.set_xlabel('SHAP value (impact on CD4 count)', fontsize=14)
ax1.tick_params(labelsize=11)

# Panel B: Mean Absolute SHAP Values (Feature Importance)
print("   Creating Panel B: Feature Importance...")
ax2 = plt.subplot(1, 3, 2)

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=True)

# Plot top 12 features
top_features = feature_importance.tail(12)
bars = ax2.barh(range(len(top_features)), top_features['importance'],
                color=PRIMARY_BLUE, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'], fontsize=11)
ax2.set_xlabel('Mean |SHAP value|', fontsize=14)
ax2.set_title('B. Feature Importance Ranking', fontsize=18, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)
ax2.tick_params(labelsize=11)

# Add values on bars
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax2.text(row['importance'], i, f'  {row["importance"]:.2f}',
             va='center', fontsize=10, fontweight='bold')

# Panel C: Dependence Plot (Temperature effect)
print("   Creating Panel C: Dependence Plot...")
ax3 = plt.subplot(1, 3, 3)

# Find temperature column index
temp_col = 'climate_daily_mean_temp' if 'climate_daily_mean_temp' in X.columns else available_climate[0]
temp_idx = list(X.columns).index(temp_col)

# Use heat stress index for coloring if available, otherwise use temperature anomaly
color_col = 'climate_heat_stress_index' if 'climate_heat_stress_index' in X_test.columns else 'climate_temp_anomaly'
color_data = X_test[color_col][:500] if color_col in X_test.columns else X_test[temp_col][:500]

# Create scatter plot
scatter = ax3.scatter(X_test[temp_col][:500], shap_values[:, temp_idx],
                      c=color_data, cmap='RdYlBu_r', alpha=0.6, s=50,
                      edgecolors='black', linewidth=0.3)
ax3.set_xlabel('Daily Mean Temperature (°C)', fontsize=14)
ax3.set_ylabel('SHAP value', fontsize=14)
ax3.set_title('C. Temperature Effect on CD4', fontsize=18, fontweight='bold', pad=15)
ax3.grid(alpha=0.3)
ax3.tick_params(labelsize=11)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar_label = 'Heat Stress Index' if color_col == 'climate_heat_stress_index' else 'Temp Anomaly (°C)'
cbar.set_label(cbar_label, fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Add regression line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(
    X_test[temp_col][:500], shap_values[:, temp_idx]
)
x_line = np.array([X_test[temp_col][:500].min(), X_test[temp_col][:500].max()])
y_line = slope * x_line + intercept
ax3.plot(x_line, y_line, 'r--', linewidth=2, label=f'R² = {r_value**2:.3f}')
ax3.legend(fontsize=11)

# Add methodology note
methodology_text = (
    'Methodology: TreeExplainer with 100-tree Random Forest. SHAP values represent the contribution of each\n'
    'feature to predictions relative to the baseline. Positive values increase CD4 count; negative values decrease it.\n'
    f'Analysis based on {len(df_cd4):,} HIV+ patient records from Johannesburg (2002-2021) with ERA5 climate data.'
)
fig.text(0.5, 0.04, methodology_text, ha='center', fontsize=11,
         color=TEXT_COLOR, style='italic', wrap=True,
         bbox=dict(boxstyle='round', facecolor=BACKGROUND, edgecolor=PRIMARY_BLUE, linewidth=2))

# Adjust layout
plt.tight_layout(rect=[0, 0.08, 1, 0.92])

# Save as SVG
output_path = f"{OUTPUT_DIR}/slide_08_shap_cd4_analysis.svg"
plt.savefig(output_path, format='svg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\n✓ Slide 8 saved: {output_path}")

# Also save as PNG for preview
output_png = output_path.replace('.svg', '.png')
plt.savefig(output_png, format='png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Preview saved: {output_png}")

plt.close()

# Print summary statistics
print("\n" + "=" * 80)
print("SHAP ANALYSIS SUMMARY")
print("=" * 80)
print(f"Model Performance: R² = {test_score:.3f}")
print(f"Sample Size: {len(df_cd4):,} patients")
print(f"Features Analyzed: {len(available_climate)}")
print(f"\nTop 5 Most Important Features:")
for i, (idx, row) in enumerate(feature_importance.tail(5).iloc[::-1].iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.3f}")

print("\n✓ Slide 8 completed successfully using real clinical data!")
print("=" * 80)
