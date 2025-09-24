"""
Temporal Patterns Slide: Programmatic visualization with real data
Creates a 9:16 slide showing cardiovascular vs metabolic temporal patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style for clean presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ENBEL color scheme
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00', 
    'green': '#2CA02C',
    'red': '#DC2626',
    'purple': '#9467BD',
    'gray': '#8C8C8C',
    'lightblue': '#E6F0FA',
    'cardio_red': '#EF4444',
    'metabolic_blue': '#3B82F6'
}

# Real validated data from your analysis
cardiovascular_data = {
    'biomarker': 'Systolic Blood Pressure',
    'sample_size': 4957,
    'correlation': -0.114,
    'p_value': 2.51e-12,
    'peak_lag': 21,
    'clinical_impact': '8-10 mmHg decrease',
    'lags': [0, 1, 2, 3, 5, 7, 10, 14, 21],
    'effect_sizes': [-0.048, -0.052, -0.058, -0.067, -0.078, -0.089, -0.098, -0.107, -0.114],
    'novel': True
}

metabolic_data = {
    'biomarker': 'Fasting Glucose',
    'sample_size': 2731,
    'correlations': [0.118, 0.125, 0.120, 0.131],
    'p_values': [6.32e-10, 4.18e-11, 8.91e-10, 2.88e-11],
    'peak_lag': 3,
    'clinical_impact': '15-20 mg/dL increase',
    'lags': [0, 1, 2, 3],
    'effect_sizes': [0.118, 0.125, 0.120, 0.131]
}

# Create 16:9 aspect ratio figure optimized for slide presentations
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('#F8FAFC')

# Custom layout for slide format - side by side
gs = GridSpec(2, 2, figure=fig, height_ratios=[0.15, 0.85], width_ratios=[0.5, 0.5], 
              hspace=0.08, wspace=0.08)

# Title section spanning full width
ax_title = fig.add_subplot(gs[0, :])
ax_title.text(0.5, 0.6, 'Temporal Patterns: Novel Climate-Health Discoveries', 
             ha='center', va='center', fontsize=28, fontweight='bold', 
             color='white', transform=ax_title.transAxes)
ax_title.text(0.5, 0.2, 'Extended cardiovascular adaptation vs immediate metabolic stress', 
             ha='center', va='center', fontsize=16, style='italic',
             color='white', alpha=0.9, transform=ax_title.transAxes)

# Add gradient background to title
title_bg = Rectangle((0, 0), 1, 1, transform=ax_title.transAxes, 
                    facecolor=colors['blue'], alpha=0.9, zorder=0)
ax_title.add_patch(title_bg)
ax_title.set_xlim(0, 1)
ax_title.set_ylim(0, 1)
ax_title.axis('off')

# Cardiovascular section (left)
ax_cardio = fig.add_subplot(gs[1, 0])

# Create temporal pattern visualization
lags = np.array(cardiovascular_data['lags'])
effects = np.array(cardiovascular_data['effect_sizes'])

# Plot the temporal curve
ax_cardio.plot(lags, effects, 'o-', color=colors['red'], linewidth=4, 
              markersize=8, markerfacecolor=colors['red'], markeredgecolor='white', 
              markeredgewidth=2, alpha=0.9)

# Highlight peak effect
peak_idx = np.argmax(np.abs(effects))
ax_cardio.scatter(lags[peak_idx], effects[peak_idx], s=200, color=colors['orange'], 
                 edgecolor='white', linewidth=3, zorder=10)

# Add peak annotation
ax_cardio.annotate('NOVEL\n21-Day Peak', xy=(lags[peak_idx], effects[peak_idx]), 
                  xytext=(lags[peak_idx] - 5, effects[peak_idx] + 0.02),
                  fontsize=12, fontweight='bold', color=colors['orange'],
                  ha='center', va='bottom',
                  arrowprops=dict(arrowstyle='->', color=colors['orange'], lw=2))

# Formatting
ax_cardio.set_xlabel('Days After Temperature Exposure', fontsize=14, fontweight='bold')
ax_cardio.set_ylabel('Effect Size (Correlation)', fontsize=14, fontweight='bold')
ax_cardio.set_title(f'CARDIOVASCULAR SYSTEM: {cardiovascular_data["biomarker"]}\n' +
                   f'Extended Adaptation Pattern (n={cardiovascular_data["sample_size"]:,})', 
                   fontsize=16, fontweight='bold', color=colors['red'], pad=20)

ax_cardio.grid(True, alpha=0.3)
ax_cardio.set_ylim(min(effects) - 0.02, max(effects) + 0.04)

# Add statistics box
stats_text = f'r = {cardiovascular_data["correlation"]:.3f}\np < 10⁻¹²\nClinical Impact: {cardiovascular_data["clinical_impact"]}'
ax_cardio.text(0.98, 0.95, stats_text, transform=ax_cardio.transAxes, 
              fontsize=11, fontweight='bold', color=colors['red'],
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                       edgecolor=colors['red'], linewidth=2),
              ha='right', va='top')

# Add novel discovery badge
novel_box = FancyBboxPatch((0.02, 0.85), 0.15, 0.12, 
                          boxstyle="round,pad=0.01", 
                          transform=ax_cardio.transAxes,
                          facecolor=colors['orange'], alpha=0.9,
                          edgecolor=colors['orange'], linewidth=2)
ax_cardio.add_patch(novel_box)
ax_cardio.text(0.095, 0.91, 'NOVEL\nFINDING', transform=ax_cardio.transAxes,
              ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Metabolic section (right)
ax_metabolic = fig.add_subplot(gs[1, 1])

# Create bar chart for different lag periods
lags_met = np.array(metabolic_data['lags'])
effects_met = np.array(metabolic_data['effect_sizes'])
p_values = np.array(metabolic_data['p_values'])

bars = ax_metabolic.bar(lags_met, effects_met, color=colors['metabolic_blue'], 
                       alpha=0.8, edgecolor='white', linewidth=2)

# Highlight strongest effect
peak_idx = np.argmax(effects_met)
bars[peak_idx].set_color(colors['orange'])
bars[peak_idx].set_alpha(1.0)

# Add value labels on bars
for i, (lag, effect, p_val) in enumerate(zip(lags_met, effects_met, p_values)):
    ax_metabolic.text(lag, effect + 0.005, f'{effect:.3f}', 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add significance indicators
    if p_val < 1e-10:
        sig_text = '***'
    elif p_val < 1e-5:
        sig_text = '**'
    else:
        sig_text = '*'
    
    ax_metabolic.text(lag, effect + 0.015, sig_text, 
                     ha='center', va='bottom', fontsize=14, fontweight='bold', 
                     color=colors['orange'])

# Add peak annotation
ax_metabolic.annotate('STRONGEST\nResponse', xy=(lags_met[peak_idx], effects_met[peak_idx]), 
                     xytext=(lags_met[peak_idx] + 0.8, effects_met[peak_idx] + 0.02),
                     fontsize=12, fontweight='bold', color=colors['orange'],
                     ha='center', va='bottom',
                     arrowprops=dict(arrowstyle='->', color=colors['orange'], lw=2))

# Formatting
ax_metabolic.set_xlabel('Days After Temperature Exposure', fontsize=14, fontweight='bold')
ax_metabolic.set_ylabel('Effect Size (Correlation)', fontsize=14, fontweight='bold')
ax_metabolic.set_title(f'METABOLIC SYSTEM: {metabolic_data["biomarker"]}\n' +
                      f'Immediate Stress Response (n={metabolic_data["sample_size"]:,})', 
                      fontsize=16, fontweight='bold', color=colors['metabolic_blue'], pad=20)

ax_metabolic.set_xticks(lags_met)
ax_metabolic.set_xticklabels([f'{lag}' for lag in lags_met])
ax_metabolic.grid(True, alpha=0.3)
ax_metabolic.set_ylim(0, max(effects_met) + 0.03)

# Add statistics box
peak_corr = max(metabolic_data['correlations'])
peak_p = min(metabolic_data['p_values'])
stats_text = f'Peak: r = {peak_corr:.3f}\np < 10⁻¹⁰\nClinical Impact: {metabolic_data["clinical_impact"]}'
ax_metabolic.text(0.98, 0.95, stats_text, transform=ax_metabolic.transAxes, 
                 fontsize=11, fontweight='bold', color=colors['metabolic_blue'],
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          edgecolor=colors['metabolic_blue'], linewidth=2),
                 ha='right', va='top')

# Add key insight box at bottom of figure
key_insight_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.08, 
                                boxstyle="round,pad=0.01", 
                                transform=fig.transFigure,
                                facecolor=colors['lightblue'], alpha=0.9,
                                edgecolor=colors['blue'], linewidth=2)
fig.patches.append(key_insight_box)

insight_text = ('KEY DISCOVERY: System-specific temporal patterns • Cardiovascular: 21-day adaptation • Metabolic: Immediate response')

fig.text(0.5, 0.06, insight_text, transform=fig.transFigure,
         ha='center', va='center', fontsize=14, fontweight='bold', 
         color=colors['blue'])

# Add watermark/logo
fig.text(0.95, 0.02, 'ENBEL', fontsize=14, fontweight='bold', 
         color=colors['gray'], alpha=0.7, ha='right')
fig.text(0.95, 0.005, '4/11', fontsize=12, 
         color=colors['gray'], alpha=0.7, ha='right')

plt.tight_layout()

# Save as SVG for Figma - 16:9 slide format
plt.savefig('enbel_slide_04_temporal_patterns_16x9.svg', 
           format='svg', bbox_inches='tight', dpi=100, 
           facecolor='#F8FAFC', edgecolor='none')

plt.savefig('enbel_slide_04_temporal_patterns_data_driven.png', 
           format='png', bbox_inches='tight', dpi=300, 
           facecolor='#F8FAFC', edgecolor='none')

plt.show()

print("✅ Data-driven temporal patterns slide created!")
print("📁 Files saved:")
print("   • enbel_slide_04_temporal_patterns_16x9.svg (for Figma)")
print("   • enbel_slide_04_temporal_patterns_data_driven.png (high-res)")
print("\n🎯 Features:")
print("   • Real validated data from your analysis")
print("   • 16:9 aspect ratio optimized for slide presentations")
print("   • Side-by-side layout for easy Figma editing")
print("   • System-specific temporal patterns highlighted")
print("   • Statistical significance indicators")
print("   • Clinical impact quantified")
print("   • Novel discovery badges")