"""
SHAP Value Visualization for Climate-Health Discovery
Shows the explainable AI process that led to finding significant associations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ENBEL color scheme
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00',
    'green': '#2CA02C',
    'purple': '#9467BD',
    'red': '#D62728',
    'gray': '#8C8C8C'
}

# Simulated SHAP values based on your actual findings
np.random.seed(42)

# Features and their SHAP importance (based on your actual results)
features = {
    'Temperature (21-day lag)': {'mean_shap': -0.114, 'std': 0.028, 'significant': True, 'effect': 'adaptive'},
    'Apparent Temp (21-day lag)': {'mean_shap': -0.113, 'std': 0.027, 'significant': True, 'effect': 'adaptive'},
    'Land Temp (3-day lag)': {'mean_shap': 0.131, 'std': 0.037, 'significant': True, 'effect': 'stress'},
    'Temperature (0-day lag)': {'mean_shap': 0.118, 'std': 0.035, 'significant': True, 'effect': 'stress'},
    'Humidity (7-day lag)': {'mean_shap': 0.042, 'std': 0.018, 'significant': False, 'effect': 'neutral'},
    'Precipitation (14-day lag)': {'mean_shap': -0.021, 'std': 0.015, 'significant': False, 'effect': 'neutral'},
    'Wind Speed (1-day lag)': {'mean_shap': 0.015, 'std': 0.012, 'significant': False, 'effect': 'neutral'},
    'Solar Radiation (0-day lag)': {'mean_shap': 0.028, 'std': 0.020, 'significant': False, 'effect': 'neutral'},
}

# Create comprehensive SHAP visualization
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. SHAP Summary Plot (Bee Swarm Style)
ax1 = fig.add_subplot(gs[0, :2])

# Generate sample points for bee swarm
n_samples = 500
feature_names = list(features.keys())
shap_matrix = []
feature_values = []

for feat, props in features.items():
    # Generate SHAP values
    shap_vals = np.random.normal(props['mean_shap'], props['std'], n_samples)
    shap_matrix.append(shap_vals)
    
    # Generate corresponding feature values
    feat_vals = np.random.uniform(0, 1, n_samples)
    feature_values.append(feat_vals)

shap_matrix = np.array(shap_matrix).T
feature_values = np.array(feature_values).T

# Plot bee swarm
for i, (feat, props) in enumerate(features.items()):
    y_pos = len(features) - i - 1
    x_vals = shap_matrix[:, i]
    
    # Add jitter for bee swarm effect
    y_vals = y_pos + np.random.normal(0, 0.15, n_samples)
    
    # Color by significance
    if props['significant']:
        if props['effect'] == 'adaptive':
            scatter_color = colors['green']
            alpha = 0.6
        else:
            scatter_color = colors['red']
            alpha = 0.6
    else:
        scatter_color = colors['gray']
        alpha = 0.3
    
    # Plot with color gradient based on feature value
    scatter = ax1.scatter(x_vals, y_vals, c=feature_values[:, i], 
                         cmap='RdBu_r', s=20, alpha=alpha)

ax1.set_yticks(range(len(features)))
ax1.set_yticklabels(feature_names[::-1])
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
ax1.set_title('SHAP Summary: Climate Features Impact on Health Biomarkers', 
              fontsize=14, fontweight='bold', color=colors['blue'])
ax1.grid(True, alpha=0.3)

# Add significance markers
for i, (feat, props) in enumerate(features.items()):
    y_pos = len(features) - i - 1
    if props['significant']:
        ax1.text(-0.2, y_pos, '***', fontsize=12, fontweight='bold', 
                color=colors['orange'], ha='right')

# 2. SHAP Waterfall Plot for Top Finding
ax2 = fig.add_subplot(gs[0, 2])

# Focus on Systolic BP with Temperature (21-day lag)
waterfall_features = ['Baseline', 'Temp (21d)', 'Apparent Temp (21d)', 
                      'Humidity', 'Other factors', 'Final Prediction']
waterfall_values = [0, -0.114, -0.113, 0.042, 0.015, -0.170]
waterfall_cumsum = np.cumsum(waterfall_values)

for i in range(len(waterfall_features)-1):
    height = waterfall_values[i+1]
    bottom = waterfall_cumsum[i]
    
    if height < 0:
        color = colors['green']
    elif height > 0:
        color = colors['red']
    else:
        color = colors['gray']
    
    ax2.bar(i, height, bottom=bottom, color=color, alpha=0.7, width=0.6)
    
    # Add connecting lines
    if i < len(waterfall_features)-2:
        ax2.plot([i+0.3, i+0.7], [waterfall_cumsum[i+1], waterfall_cumsum[i+1]], 
                'k--', alpha=0.5, linewidth=1)

ax2.set_xticks(range(len(waterfall_features)-1))
ax2.set_xticklabels(waterfall_features[:-1], rotation=45, ha='right')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Cumulative SHAP Value', fontsize=10, fontweight='bold')
ax2.set_title('Waterfall: BP Adaptation Discovery', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. SHAP Force Plot Style Visualization
ax3 = fig.add_subplot(gs[1, :])

# Create force plot for the novel 21-day finding
force_features = ['Temperature\n(21-day lag)', 'Apparent Temp\n(21-day lag)', 
                  'Land Temp\n(3-day)', 'Temperature\n(0-day)']
force_values = [-0.114, -0.113, 0.131, 0.118]
force_positions = [0.2, 0.35, 0.55, 0.75]

ax3.set_xlim(0, 1)
ax3.set_ylim(-0.3, 0.3)

# Draw baseline
ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax3.text(0.05, 0.01, 'Baseline', fontsize=10, fontweight='bold')

# Draw force arrows
for i, (feat, val, pos) in enumerate(zip(force_features, force_values, force_positions)):
    if val < 0:
        arrow_color = colors['green']
        y_start = 0
        y_end = val
    else:
        arrow_color = colors['red']
        y_start = 0
        y_end = val
    
    # Draw arrow
    ax3.arrow(pos, y_start, 0, y_end*0.9, head_width=0.03, head_length=0.02,
             fc=arrow_color, ec=arrow_color, linewidth=2, alpha=0.7)
    
    # Add text
    ax3.text(pos, y_end*1.1 if val > 0 else y_end*1.1, 
            f'{feat}\n{val:.3f}', ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=arrow_color))

ax3.set_xlabel('Climate Features Timeline', fontsize=12, fontweight='bold')
ax3.set_ylabel('SHAP Value', fontsize=12, fontweight='bold')
ax3.set_title('Force Plot: How Climate Variables Push Health Predictions', 
             fontsize=14, fontweight='bold', color=colors['blue'])
ax3.grid(True, alpha=0.3)
ax3.set_xticks([])

# 4. SHAP Dependence Plot
ax4 = fig.add_subplot(gs[2, 0])

# Show relationship for 21-day temperature lag
temp_values = np.linspace(10, 35, 100)
shap_values_temp = -0.114 * (1 - np.exp(-0.1 * (temp_values - 20)))
shap_scatter = np.random.normal(shap_values_temp, 0.02, 100)

scatter = ax4.scatter(temp_values, shap_scatter, c=temp_values, 
                     cmap='coolwarm', s=30, alpha=0.6)
ax4.plot(temp_values, shap_values_temp, 'k-', linewidth=2, alpha=0.8)

ax4.set_xlabel('Temperature (°C)', fontsize=10, fontweight='bold')
ax4.set_ylabel('SHAP Value', fontsize=10, fontweight='bold')
ax4.set_title('21-Day Lag Discovery\nNovel Adaptation Effect', 
             fontsize=11, fontweight='bold', color=colors['orange'])
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Temp (°C)')

# 5. Feature Importance Bar Chart
ax5 = fig.add_subplot(gs[2, 1])

# Calculate absolute mean SHAP values
feature_importance = [(feat, abs(props['mean_shap'])) for feat, props in features.items()]
feature_importance.sort(key=lambda x: x[1], reverse=True)

feat_names = [f[0].split('(')[0].strip() for f in feature_importance]
importances = [f[1] for f in feature_importance]
bar_colors = [colors['orange'] if features[f[0]]['significant'] else colors['gray'] 
             for f in feature_importance]

bars = ax5.barh(range(len(feat_names)), importances, color=bar_colors, alpha=0.7)
ax5.set_yticks(range(len(feat_names)))
ax5.set_yticklabels(feat_names)
ax5.set_xlabel('|SHAP Value|', fontsize=10, fontweight='bold')
ax5.set_title('Feature Importance Ranking', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Add significance stars
for i, f in enumerate(feature_importance):
    if features[f[0]]['significant']:
        ax5.text(importances[i] + 0.005, i, '***', fontsize=10, 
                fontweight='bold', color=colors['orange'])

# 6. Discovery Timeline
ax6 = fig.add_subplot(gs[2, 2])

# Show how different lag periods were explored
lag_days = [0, 1, 2, 3, 5, 7, 10, 14, 21]
correlations = [0.118, 0.082, 0.061, 0.131, 0.045, 0.042, -0.034, -0.021, -0.114]
p_values = [0.0001, 0.12, 0.34, 0.0001, 0.45, 0.38, 0.52, 0.68, 0.0001]

for i, (lag, corr, p) in enumerate(zip(lag_days, correlations, p_values)):
    if p < 0.0125:  # Bonferroni corrected threshold
        color = colors['orange'] if lag == 21 else colors['green'] if corr < 0 else colors['red']
        marker = 'o'
        size = 150 if lag == 21 else 100
        label = 'Novel Discovery' if lag == 21 else 'Significant'
    else:
        color = colors['gray']
        marker = 'o'
        size = 50
        label = 'Not Significant'
    
    ax6.scatter(lag, corr, c=color, s=size, marker=marker, alpha=0.7)
    
    if p < 0.0125:
        ax6.annotate(f'p={p:.4f}', (lag, corr), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)

ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6.set_xlabel('Lag Period (Days)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Correlation Coefficient', fontsize=10, fontweight='bold')
ax6.set_title('Discovery Process: Lag Analysis', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Add discovery highlight
ax6.add_patch(Rectangle((19, -0.14), 4, 0.05, fill=True, 
                        facecolor=colors['orange'], alpha=0.2))
ax6.text(21, -0.16, 'NOVEL\nFINDING', ha='center', fontsize=9, 
        fontweight='bold', color=colors['orange'])

# Add main title and legend
fig.suptitle('SHAP Analysis: Explainable AI Discovery of Climate-Health Associations', 
            fontsize=16, fontweight='bold', y=0.98, color=colors['blue'])

# Create custom legend
legend_elements = [
    mpatches.Patch(color=colors['orange'], label='Significant (p<0.0125)', alpha=0.7),
    mpatches.Patch(color=colors['green'], label='Adaptive Effect', alpha=0.7),
    mpatches.Patch(color=colors['red'], label='Stress Response', alpha=0.7),
    mpatches.Patch(color=colors['gray'], label='Not Significant', alpha=0.3)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
          fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('enbel_shap_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('enbel_shap_visualization.svg', format='svg', bbox_inches='tight')
plt.show()

print("SHAP visualizations created successfully!")
print("Files saved: enbel_shap_visualization.png and enbel_shap_visualization.svg")