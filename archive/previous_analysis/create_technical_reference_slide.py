#!/usr/bin/env python3
"""
Create comprehensive technical reference slide for ENBEL dataset
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with academic dimensions
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Define colors
header_blue = '#00539B'
light_blue = '#E3F2FD'
table_bg = '#F8F9FA'
border_color = '#CCCCCC'
text_dark = '#2E3440'

# Header
header_rect = FancyBboxPatch(
    (2, 92), 96, 6,
    boxstyle="round,pad=0.1",
    facecolor=header_blue,
    edgecolor='none'
)
ax.add_patch(header_rect)

ax.text(50, 95, 'ENBEL Dataset: Technical Reference & Imputation Details',
        ha='center', va='center', fontsize=20, fontweight='bold', color='white')

# Section 1: Dataset Composition (Top Left)
section1_rect = FancyBboxPatch(
    (2, 72), 46, 18,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section1_rect)

ax.text(25, 88, '1. Dataset Composition', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=text_dark)

# Dataset composition table
composition_data = [
    ['Component', 'N', '%', 'Purpose'],
    ['Total Dataset', '18,205', '100%', ''],
    ['Clinical Cohort', '9,103', '50.0%', 'HIV treatment trials (Johannesburg)'],
    ['GCRO Cohort', '9,102', '50.0%', 'Socioeconomic survey (imputation source)']
]

y_start = 84.5
for i, row in enumerate(composition_data):
    y_pos = y_start - i * 2.5
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 5 + j * 10
            if j == 2:  # Purpose column wider
                ax.text(x_pos + 15, y_pos, cell, ha='left', va='center', 
                       fontsize=9, fontweight='bold', color=text_dark)
            else:
                ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                       fontsize=9, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 5 + j * 10
            if j == 2:  # Purpose column
                ax.text(x_pos + 15, y_pos, cell, ha='left', va='center', 
                       fontsize=8, color=text_dark)
            else:
                ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                       fontsize=8, color=text_dark)

# Section 2: GCRO Imputation Process (Top Right)
section2_rect = FancyBboxPatch(
    (52, 72), 46, 18,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section2_rect)

ax.text(75, 88, '2. GCRO K-NN Imputation Process', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=text_dark)

# Imputation process details
imputation_text = [
    "Multi-dimensional Matching Criteria:",
    "• Step 1: Demographics (60% weight)",
    "  - Sex matching (50% penalty if mismatch)",
    "  - Race matching (40% penalty if mismatch)",
    "• Step 2: Spatial proximity (40% weight)",
    "  - GPS coordinates (Haversine distance)",
    "  - Exponential decay (5km scale)",
    "  - Hard cutoff: 15km maximum",
    "• Step 3: Quality thresholds",
    "  - Minimum 3 matches required",
    "  - Combined similarity score >0.3",
    "  - Valid GPS for both cohorts"
]

y_pos = 85
for line in imputation_text:
    ax.text(54, y_pos, line, ha='left', va='center', 
           fontsize=8, color=text_dark)
    y_pos -= 1.2

# Section 3: Imputation Results (Middle Left)
section3_rect = FancyBboxPatch(
    (2, 40), 46, 30,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section3_rect)

ax.text(25, 68, '3. Imputation Results & Coverage', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=text_dark)

# Imputation results table
imputation_vars = [
    ['Variable', 'Coverage', 'Method'],
    ['Education', '68-75%', 'Mode selection'],
    ['Employment Status', '72-80%', 'Mode selection'],
    ['Housing Vulnerability', '65-78%', 'Weighted average'],
    ['Economic Vulnerability', '70-82%', 'Weighted average'],
    ['Heat Vulnerability Index', '64-79%', 'Weighted average']
]

y_start = 64
for i, row in enumerate(imputation_vars):
    y_pos = y_start - i * 2.2
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 5 + j * 13
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=9, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 5 + j * 13
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=8, color=text_dark)

# Additional imputation notes
notes_text = [
    "Quality Assurance:",
    "• Confidence scoring based on match quality",
    "• Cross-validation with known overlaps",
    "• Sensitivity analysis for missing data",
    "",
    "Why not 100% coverage:",
    "• Geographic constraints (15km cutoff)",
    "• Quality thresholds (min 3 matches)",
    "• Missing GPS coordinates",
    "• Demographic mismatch penalties"
]

y_pos = 52
for line in notes_text:
    if line == "":
        y_pos -= 1
    else:
        ax.text(4, y_pos, line, ha='left', va='center', 
               fontsize=8, color=text_dark, fontweight='bold' if line.endswith(':') else 'normal')
        y_pos -= 1.2

# Section 4: Analysis Features (Middle Right)
section4_rect = FancyBboxPatch(
    (52, 40), 46, 30,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section4_rect)

ax.text(75, 68, '4. Actual Analysis Features Used', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=text_dark)

analysis_text = [
    "CRITICAL: Not all 343 features used in DLNM/SHAP",
    "",
    "Core Climate Features in Models (~50-80 total):",
    "• Base weather: temperature, humidity, wind_speed,",
    "  heat_index (lag 0)",
    "• Key lags: temperature_lag1, lag2, lag3, lag5, lag7",
    "• Heat indices: heat_index_lag0-21, UTCI, WBGT",
    "• SAAQIS: ERA5 temp, land temp, wind",
    "  (lags 0,1,2,3,5,7)",
    "",
    "Features NOT used in core analysis:",
    "• Multi-day aggregates (temp_mean_3d, etc.)",
    "• Derived variability measures",
    "• Temperature change/acceleration terms",
    "• Complex interaction terms",
    "",
    "Model Selection Criteria:",
    "• Epidemiological relevance",
    "• Statistical significance in univariate",
    "• Collinearity assessment (VIF <5)",
    "• Biological plausibility"
]

y_pos = 65
for line in analysis_text:
    if line == "":
        y_pos -= 1
    else:
        font_weight = 'bold' if (line.startswith('CRITICAL') or line.endswith(':')) else 'normal'
        font_size = 9 if line.startswith('CRITICAL') else 8
        ax.text(54, y_pos, line, ha='left', va='center', 
               fontsize=font_size, fontweight=font_weight, color=text_dark)
        y_pos -= 1.2

# Section 5: Biomarkers & Data Sources (Bottom)
section5_rect = FancyBboxPatch(
    (2, 2), 96, 36,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section5_rect)

ax.text(50, 36, '5. Target Variables (Biomarkers) & Data Sources', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=text_dark)

# Biomarkers table
biomarkers_data = [
    ['Biomarker', 'Clinical Relevance', 'Sample Size*', 'Units'],
    ['CD4 Cell Count', 'Immune function (HIV)', '~8,500', 'cells/µL'],
    ['Systolic Blood Pressure', 'Cardiovascular risk', '~17,800', 'mmHg'],
    ['Diastolic Blood Pressure', 'Cardiovascular risk', '~17,800', 'mmHg'],
    ['Fasting Glucose', 'Metabolic health', '~12,000', 'mg/dL'],
    ['Fasting Total Cholesterol', 'Cardiovascular risk', '~11,800', 'mg/dL'],
    ['Fasting LDL', 'Cardiovascular risk', '~11,600', 'mg/dL'],
    ['Fasting HDL', 'Cardiovascular risk', '~11,600', 'mg/dL'],
    ['Hemoglobin', 'Anemia/oxygen transport', '~16,500', 'g/dL'],
    ['Creatinine', 'Kidney function', '~15,200', 'mg/dL']
]

# Split into two columns for better layout
col1_data = biomarkers_data[:6]  # Header + 5 biomarkers
col2_data = [biomarkers_data[0]] + biomarkers_data[6:]  # Header + remaining

# Column 1
y_start = 32
for i, row in enumerate(col1_data):
    y_pos = y_start - i * 2.5
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 5 + j * 11
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=9, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 5 + j * 11
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=8, color=text_dark)

# Column 2
y_start = 32
for i, row in enumerate(col2_data):
    y_pos = y_start - i * 2.5
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 52 + j * 11
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=9, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 52 + j * 11
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=8, color=text_dark)

# Data sources section
ax.text(50, 18, 'Data Sources & Coverage', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_dark)

sources_data = [
    ['Source', 'Description', 'Coverage', 'Variables'],
    ['Clinical Trials', 'HIV treatment studies, observational cohorts', '2016-2019', 'All biomarkers, demographics'],
    ['ERA5 Reanalysis', 'Hourly meteorological data', '2016-2019', 'Temperature, humidity, pressure, wind'],
    ['SAAQIS', 'South African Air Quality Network', '2016-2019', 'Local weather, air quality'],
    ['GCRO Survey', 'Gauteng City-Region Observatory', '2016-2017', 'Socioeconomic, vulnerability indices'],
    ['GPS Coordinates', 'Participant locations', '2016-2019', 'Spatial matching, exposure assignment']
]

y_start = 15
for i, row in enumerate(sources_data):
    y_pos = y_start - i * 1.8
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 4 + j * 23
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=9, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 4 + j * 23
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=8, color=text_dark)

# Footer note
ax.text(50, 4, '*Sample sizes vary due to missing values and visit patterns. Geographic coverage: Johannesburg metropolitan area.',
        ha='center', va='center', fontsize=8, style='italic', color=text_dark)

plt.tight_layout()
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.svg', 
            format='svg', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.png', 
            format='png', dpi=300, bbox_inches='tight', facecolor='white')

print("Technical reference slide created successfully!")
print("Files saved:")
print("- /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.svg")
print("- /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.png")