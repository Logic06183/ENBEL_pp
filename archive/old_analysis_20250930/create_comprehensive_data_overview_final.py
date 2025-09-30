#!/usr/bin/env python3
"""
Create a comprehensive, detailed data overview slide for ENBEL analysis
Professional Beamer-style presentation with extensive methodological detail
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set up the figure with 16:9 aspect ratio and better DPI
fig = plt.figure(figsize=(16, 9), dpi=100)
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Professional color palette - enhanced
colors = {
    'header_blue': '#00539B',
    'primary_blue': '#1f77b4',
    'secondary_blue': '#7fb3d3',
    'accent_orange': '#FF7F00',
    'accent_green': '#2CA02C',
    'light_gray': '#f8f9fa',
    'medium_gray': '#e9ecef',
    'dark_gray': '#495057',
    'success_green': '#28a745',
    'warning_orange': '#fd7e14',
    'soft_blue': '#e3f2fd',
    'soft_green': '#e8f5e8',
    'soft_orange': '#fff3e0'
}

# Header section with blue background
header_rect = FancyBboxPatch((0, 8.2), 16, 0.8, 
                            boxstyle="round,pad=0", 
                            facecolor=colors['header_blue'], 
                            edgecolor='none')
ax.add_patch(header_rect)

# Main title
ax.text(8, 8.6, 'Integrated Climate-Health Dataset: Comprehensive Multi-Study Analysis', 
        fontsize=18, fontweight='bold', ha='center', va='center', color='white')
ax.text(8, 8.35, 'Multiple Study Types, Extensive Biomarkers & Socioeconomic Integration', 
        fontsize=12, ha='center', va='center', color='white', style='italic')

# Left section: Clinical Studies & Biomarkers - better spacing
left_section = FancyBboxPatch((0.2, 4.4), 7.4, 3.6, 
                             boxstyle="round,pad=0.06", 
                             facecolor=colors['light_gray'], 
                             edgecolor=colors['medium_gray'], 
                             linewidth=1.5)
ax.add_patch(left_section)

# Clinical Studies header
ax.text(0.4, 7.8, 'CLINICAL STUDIES DATA', fontsize=14, fontweight='bold', 
        color=colors['header_blue'])

# Participants summary
ax.text(0.4, 7.5, '9,103 participants from 17 studies (2002-2021)', 
        fontsize=12, fontweight='bold', color=colors['dark_gray'])

# Study types breakdown with visual bars
study_types = [
    ('HIV Treatment Trials', 3119, 34.3, colors['primary_blue']),
    ('COVID Studies', 2858, 31.4, colors['accent_orange']),
    ('Metabolic Health Studies', 1782, 19.6, colors['accent_green']),
    ('TB Prevention Trials', 1182, 13.0, colors['secondary_blue']),
    ('General Clinical Studies', 162, 1.8, colors['medium_gray'])
]

y_pos = 7.2
ax.text(0.4, y_pos, 'Study Types Breakdown:', fontsize=10, fontweight='bold')
y_pos -= 0.15

for study, count, pct, color in study_types:
    # Progress bar
    bar_width = (pct / 35.0) * 3.5  # Scale to fit
    bar_rect = Rectangle((0.5, y_pos - 0.05), bar_width, 0.08, 
                        facecolor=color, alpha=0.8)
    ax.add_patch(bar_rect)
    
    # Text
    ax.text(0.5, y_pos, f'• {study}: {count:,} ({pct}%)', 
            fontsize=9, va='center')
    y_pos -= 0.2

# Target populations
ax.text(0.4, 6.0, 'Target Populations:', fontsize=10, fontweight='bold')
ax.text(0.5, 5.85, '• General population: 91.5%', fontsize=9)
ax.text(0.5, 5.7, '• HIV+ adults: 8.5%', fontsize=9)
ax.text(0.4, 5.5, 'Location: Johannesburg region, South Africa', 
        fontsize=9, style='italic')

# Biomarkers section - better spacing
biomarker_section = FancyBboxPatch((0.2, 0.5), 7.4, 3.8, 
                                  boxstyle="round,pad=0.06", 
                                  facecolor=colors['light_gray'], 
                                  edgecolor=colors['medium_gray'], 
                                  linewidth=1.5)
ax.add_patch(biomarker_section)

ax.text(0.4, 4.1, 'COMPREHENSIVE BIOMARKERS', fontsize=14, fontweight='bold', 
        color=colors['header_blue'])
ax.text(0.4, 3.85, '30+ biomarker variables tracked', fontsize=12, fontweight='bold')

# Biomarker categories with coverage indicators
biomarker_cats = [
    ('Cardiovascular', ['Blood pressure (54.5% coverage)', 'Heart rate'], colors['accent_orange']),
    ('Metabolic', ['Fasting glucose', 'Cholesterol panel', 'Triglycerides'], colors['accent_green']),
    ('Immunologic', ['CD4 counts', 'Viral loads'], colors['primary_blue']),
    ('Hematologic', ['Hemoglobin'], colors['secondary_blue']),
    ('Renal', ['Creatinine'], colors['warning_orange']),
    ('Hepatic', ['ALT', 'AST'], colors['success_green']),
    ('Anthropometric', ['Height (73%)', 'Weight (73%)'], colors['dark_gray'])
]

y_pos = 3.6
for category, markers, color in biomarker_cats:
    # Category header with enhanced color indicator
    circle = plt.Circle((0.45, y_pos), 0.04, facecolor=color, alpha=0.9, 
                       edgecolor='white', linewidth=1)
    ax.add_patch(circle)
    ax.text(0.55, y_pos, f'{category}:', fontsize=10, fontweight='bold', va='center')
    
    # Markers with better formatting
    marker_text = ', '.join(markers)
    ax.text(0.6, y_pos - 0.12, f'  {marker_text}', fontsize=8, va='top', color=colors['dark_gray'])
    y_pos -= 0.35

# Right section: Climate Variables & Sources - better spacing
right_section = FancyBboxPatch((8.4, 4.4), 7.4, 3.6, 
                              boxstyle="round,pad=0.06", 
                              facecolor=colors['light_gray'], 
                              edgecolor=colors['medium_gray'], 
                              linewidth=1.5)
ax.add_patch(right_section)

ax.text(8.6, 7.8, 'CLIMATE VARIABLES & SOURCES', fontsize=14, fontweight='bold', 
        color=colors['header_blue'])
ax.text(8.6, 7.5, '98+ climate variables (2017-2022)', 
        fontsize=12, fontweight='bold')

# Data sources - accurate information
ax.text(8.6, 7.25, 'Data Sources: ERA5 Reanalysis, SAWS Ground Stations', 
        fontsize=10, style='italic', color=colors['dark_gray'])

# Climate variable categories - accurate from your data
climate_vars = [
    ('Temperature (TAS)', '30+ vars', 'Air temp, land surface temp, apparent temp'),
    ('Humidity', '9 vars', 'Relative humidity, specific humidity'),
    ('Wind Speed (WS)', '9 vars', 'Surface wind components'),
    ('Precipitation', '9 vars', 'Daily totals, intensity patterns'),
    ('Lag Variables', '27+ vars', '0-30 day temporal windows')
]

y_pos = 6.9
ax.text(8.6, y_pos, 'Variable Categories:', fontsize=10, fontweight='bold')
y_pos -= 0.15

for var_type, count, description in climate_vars:
    ax.text(8.7, y_pos, f'• {var_type} ({count}): {description}', fontsize=9)
    y_pos -= 0.2

# Quality control info - more detailed
y_pos -= 0.1
ax.text(8.6, y_pos, 'Temporal Resolution: 6-hourly → daily aggregations', fontsize=9)
ax.text(8.6, y_pos - 0.15, 'Lag Analysis: 0-30 day windows (9 lag periods)', fontsize=9)
ax.text(8.6, y_pos - 0.3, 'Coverage: 95.8% completeness, 2,191 observation days', fontsize=9)
ax.text(8.6, y_pos - 0.45, 'Extreme Events: 182 days (8.3% heat, 4.1% cold)', fontsize=9)

# Data Integration Framework (Bottom Center) - better spacing
integration_section = FancyBboxPatch((8.4, 0.5), 7.4, 3.8, 
                                    boxstyle="round,pad=0.06", 
                                    facecolor=colors['light_gray'], 
                                    edgecolor=colors['medium_gray'], 
                                    linewidth=1.5)
ax.add_patch(integration_section)

ax.text(8.6, 4.1, 'ENHANCED DATA INTEGRATION', fontsize=14, fontweight='bold', 
        color=colors['header_blue'])

# Layer 1: Spatiotemporal Matching - better spacing
layer1_rect = FancyBboxPatch((8.6, 3.4), 7.0, 0.6, 
                            boxstyle="round,pad=0.03", 
                            facecolor='white', 
                            edgecolor=colors['primary_blue'], 
                            linewidth=1.5)
ax.add_patch(layer1_rect)
ax.text(8.7, 3.85, 'Layer 1: Spatiotemporal Matching', fontsize=10, fontweight='bold')
ax.text(8.75, 3.7, '• GPS coordinate alignment (clinic vs. home addresses)', fontsize=8)
ax.text(8.75, 3.58, '• Temporal alignment (visit dates → exposure windows)', fontsize=8)
ax.text(8.75, 3.46, '• Multiple lag periods for delayed physiological effects', fontsize=8)

# Layer 2: Socioeconomic Imputation (Highlighted) - better spacing
layer2_rect = FancyBboxPatch((8.6, 2.4), 7.0, 0.95, 
                            boxstyle="round,pad=0.03", 
                            facecolor=colors['success_green'], 
                            alpha=0.1,
                            edgecolor=colors['success_green'], 
                            linewidth=2)
ax.add_patch(layer2_rect)

ax.text(8.7, 3.25, 'Layer 2: COMPREHENSIVE Socioeconomic Imputation', 
        fontsize=10, fontweight='bold', color=colors['success_green'])

# Success highlight
success_rect = FancyBboxPatch((8.75, 3.05), 3.4, 0.12, 
                             boxstyle="round,pad=0.01", 
                             facecolor=colors['success_green'], 
                             alpha=0.9)
ax.add_patch(success_rect)
ax.text(10.45, 3.11, 'SUCCESS: 57% → 84.3% coverage', 
        fontsize=9, fontweight='bold', ha='center', color='white')

ax.text(8.75, 2.92, '• Method: 80% demographic + 20% spatial weighting', fontsize=8)
ax.text(8.75, 2.8, '• Education (100%), Housing vulnerability (94.4%)', fontsize=8)
ax.text(8.75, 2.68, '• Economic vulnerability (100%), Heat vulnerability (100%)', fontsize=8)
ax.text(8.75, 2.56, '• Quality: 87-99% high confidence matches', fontsize=8)
ax.text(8.75, 2.44, '• Innovation: Fixed demographic encoding, clinic coordinates', fontsize=8)

# Layer 3: Analytical Framework - better spacing
layer3_rect = FancyBboxPatch((8.6, 1.7), 7.0, 0.6, 
                            boxstyle="round,pad=0.03", 
                            facecolor='white', 
                            edgecolor=colors['accent_orange'], 
                            linewidth=1.5)
ax.add_patch(layer3_rect)
ax.text(8.7, 2.15, 'Layer 3: Analytical Framework', fontsize=10, fontweight='bold')
ax.text(8.75, 2.0, '• Individual-level analysis with SES stratification', fontsize=8)
ax.text(8.75, 1.88, '• Climate-health associations across vulnerability gradients', fontsize=8)
ax.text(8.75, 1.76, '• Multi-study meta-analysis capabilities', fontsize=8)

# Key Statistics Dashboard (Bottom) - enhanced styling
stats_section = FancyBboxPatch((0.2, 0.1), 15.6, 0.35, 
                              boxstyle="round,pad=0.03", 
                              facecolor=colors['header_blue'], 
                              alpha=0.9,
                              edgecolor=colors['primary_blue'],
                              linewidth=2)
ax.add_patch(stats_section)

# Key statistics with enhanced visual design
stats = [
    ('9,103 participants', '●'),
    ('5 study types', '●'),
    ('30+ biomarkers', '●'),
    ('98+ climate variables', '●'),
    ('84.3% SES coverage', '●'),
    ('2017-2022 period', '●')
]

x_positions = np.linspace(1.5, 14.5, len(stats))
for i, (stat, symbol) in enumerate(stats):
    # Add subtle background circles for each stat
    circle = plt.Circle((x_positions[i], 0.25), 0.12, 
                       facecolor='white', alpha=0.1, edgecolor='white', linewidth=1)
    ax.add_patch(circle)
    
    ax.text(x_positions[i], 0.3, symbol, fontsize=14, ha='center', va='center', 
            color='white', alpha=0.8)
    ax.text(x_positions[i], 0.18, stat, fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')

plt.tight_layout()
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_comprehensive_data_overview_final.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_comprehensive_data_overview_final.svg', 
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

print("Comprehensive data overview slide created successfully!")
print("Files saved:")
print("- enbel_comprehensive_data_overview_final.png")
print("- enbel_comprehensive_data_overview_final.svg")