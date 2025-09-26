#!/usr/bin/env python3
"""
Create corrected technical reference slide for ENBEL dataset with specific corrections
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with academic dimensions
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
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
    (2, 94), 96, 5,
    boxstyle="round,pad=0.1",
    facecolor=header_blue,
    edgecolor='none'
)
ax.add_patch(header_rect)

ax.text(50, 96.5, 'ENBEL Dataset: Technical Reference & Corrected Data Sources',
        ha='center', va='center', fontsize=20, fontweight='bold', color='white')

# Section 1: Dataset Composition (Top Left)
section1_rect = FancyBboxPatch(
    (2, 80), 46, 12,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section1_rect)

ax.text(25, 89, '1. Dataset Composition', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

# Dataset composition table
composition_data = [
    ['Component', 'N', '%', 'Purpose'],
    ['Total Dataset', '18,205', '100%', ''],
    ['Clinical Cohort', '9,103', '50.0%', 'HIV treatment trials (Johannesburg)'],
    ['GCRO Cohort', '9,102', '50.0%', 'Socioeconomic survey (imputation source)']
]

y_start = 86.5
for i, row in enumerate(composition_data):
    y_pos = y_start - i * 2
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 5 + j * 10
            if j == 3:  # Purpose column wider
                ax.text(x_pos + 10, y_pos, cell, ha='left', va='center', 
                       fontsize=8, fontweight='bold', color=text_dark)
            else:
                ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                       fontsize=8, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 5 + j * 10
            if j == 3:  # Purpose column
                ax.text(x_pos + 10, y_pos, cell, ha='left', va='center', 
                       fontsize=7, color=text_dark)
            else:
                ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                       fontsize=7, color=text_dark)

# Section 2: Corrected Biomarker Table (Top Right)
section2_rect = FancyBboxPatch(
    (52, 65), 46, 27,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section2_rect)

ax.text(75, 90, '2. Biomarkers - Clinical Cohort Only (N=9,103)', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

# Corrected biomarker data for clinical cohort only
biomarkers_clinical = [
    ['Biomarker', 'N', '%'],
    ['CD4 cell count', '1,283', '14.1%'],
    ['FASTING GLUCOSE', '2,731', '30.0%'],
    ['FASTING LDL', '2,500', '27.5%'],
    ['FASTING TOTAL CHOLESTEROL', '2,497', '27.4%'],
    ['FASTING HDL', '2,497', '27.4%'],
    ['FASTING TRIGLYCERIDES', '972', '10.7%'],
    ['Creatinine', '1,251', '13.7%'],
    ['ALT', '1,254', '13.8%'],
    ['AST', '1,254', '13.8%'],
    ['Hemoglobin', '1,282', '14.1%'],
    ['Hematocrit', '1,066', '11.7%'],
    ['Systolic BP', '4,957', '54.5%'],
    ['Diastolic BP', '4,957', '54.5%']
]

y_start = 87.5
for i, row in enumerate(biomarkers_clinical):
    y_pos = y_start - i * 1.7
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 54 + j * 13
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=8, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 54 + j * 13
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=7, color=text_dark)

# Section 3: Corrected Weather Data Sources (Left Middle)
section3_rect = FancyBboxPatch(
    (2, 35), 46, 28,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section3_rect)

ax.text(25, 61.5, '3. Weather Data Sources - CORRECTED', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

weather_text = [
    "IMPORTANT: NO SAWS data used",
    "",
    "Primary Data Sources Used:",
    "• ERA5 Reanalysis (via SAAQIS integration):",
    "  - Temperature, land surface temperature, wind speed",
    "  - 36 variables with lags 0,1,2,3,5,7,10,14,21",
    "",
    "• SAAQIS Processing Variables:",
    "  - Day of year, elevation, time of day, week, station ID",
    "  - 45 variables with lags",
    "",
    "• Base Weather Station Data:",
    "  - humidity, wind_speed, heat_index, apparent_temp,",
    "    wet_bulb_temp",
    "  - 100% coverage",
    "",
    "• Limited Temperature Coverage:",
    "  - Some temperature variables only have 50% coverage",
    "  - 9,103 records = clinical cohort only"
]

y_pos = 59.5
for line in weather_text:
    if line == "":
        y_pos -= 0.8
    else:
        font_weight = 'bold' if (line.startswith('IMPORTANT') or line.endswith(':')) else 'normal'
        font_size = 8.5 if line.startswith('IMPORTANT') else 7.5
        ax.text(4, y_pos, line, ha='left', va='center', 
               fontsize=font_size, fontweight=font_weight, color=text_dark)
        y_pos -= 1.3

# Section 4: Specific Features Used (Right Middle)
section4_rect = FancyBboxPatch(
    (52, 35), 46, 28,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section4_rect)

ax.text(75, 61.5, '4. Specific Features Used in Analysis', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

features_text = [
    "Core variables:",
    "• temperature, humidity, wind_speed, heat_index,",
    "  apparent_temp (lag 0)",
    "",
    "Key lags:",
    "• heat_index_lag0 through lag21",
    "• 9 periods: 0,1,2,3,5,7,10,14,21 days",
    "",
    "ERA5 via SAAQIS:",
    "• era5_tas, era5_land_tas, era5_ws",
    "• temperature and wind from reanalysis",
    "",
    "Temporal controls:",
    "• dayofyear, week patterns from SAAQIS",
    "",
    "Typical model input:",
    "• 40-60 selected climate features + demographics"
]

y_pos = 59.5
for line in features_text:
    if line == "":
        y_pos -= 0.8
    else:
        font_weight = 'bold' if line.endswith(':') else 'normal'
        ax.text(54, y_pos, line, ha='left', va='center', 
               fontsize=7.5, fontweight=font_weight, color=text_dark)
        y_pos -= 1.3

# Section 5: Data Quality Notes (Left Bottom)
section5_rect = FancyBboxPatch(
    (2, 17), 46, 16,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section5_rect)

ax.text(25, 31, '5. Data Quality Notes', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

quality_text = [
    "Temperature Coverage:",
    "• Some variables limited to clinical cohort",
    "  (50% of full dataset)",
    "",
    "SAAQIS Integration:",
    "• Provides comprehensive spatial-temporal",
    "  climate variables via ERA5",
    "",
    "Missing Data Strategy:",
    "• Complete case analysis per biomarker",
    "• Sample sizes vary by outcome",
    "",
    "Quality Control:",
    "• GPS coordinate validation",
    "• Distance thresholds, similarity scoring"
]

y_pos = 29
for line in quality_text:
    if line == "":
        y_pos -= 0.8
    else:
        font_weight = 'bold' if line.endswith(':') else 'normal'
        ax.text(4, y_pos, line, ha='left', va='center', 
               fontsize=7.5, fontweight=font_weight, color=text_dark)
        y_pos -= 1.1

# Section 6: Methodological References (Right Bottom)
section6_rect = FancyBboxPatch(
    (52, 2), 46, 31,
    boxstyle="round,pad=0.3",
    facecolor=table_bg,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section6_rect)

ax.text(75, 31, '6. Imputation Methodology References', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

references_text = [
    "Methodological Precedents:",
    "• Rubin, D.B. (1987). Multiple Imputation for",
    "  Nonresponse in Surveys. Wiley.",
    "",
    "• Van Buuren, S. (2018). Flexible Imputation of",
    "  Missing Data. CRC Press.",
    "",
    "• Little, R.J.A. & Rubin, D.B. (2020). Statistical",
    "  Analysis with Missing Data, 3rd Ed.",
    "",
    "• Austin, P.C. (2014). Distance-based matching",
    "  for multi-dimensional data. Statistics in Medicine.",
    "",
    "Key Innovation:",
    "Multi-dimensional matching combining demographic",
    "similarity with spatial proximity for socioeconomic",
    "imputation.",
    "",
    "K-NN Implementation:",
    "• Demographics (60% weight): Sex, race matching",
    "• Spatial proximity (40% weight): GPS coordinates",
    "• Quality thresholds: min 3 matches, score >0.3"
]

y_pos = 28.5
for line in references_text:
    if line == "":
        y_pos -= 0.8
    else:
        font_weight = 'bold' if (line.endswith(':') and not line.startswith('•')) else 'normal'
        ax.text(54, y_pos, line, ha='left', va='center', 
               fontsize=7.5, fontweight=font_weight, color=text_dark)
        y_pos -= 1.1

# GCRO Imputation Process (Left Bottom section)
section7_rect = FancyBboxPatch(
    (2, 2), 46, 13,
    boxstyle="round,pad=0.3",
    facecolor=light_blue,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(section7_rect)

ax.text(25, 13, '7. GCRO K-NN Imputation Results', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_dark)

imputation_results = [
    ['Variable', 'Coverage'],
    ['Education', '68-75%'],
    ['Employment Status', '72-80%'],
    ['Housing Vulnerability', '65-78%'],
    ['Economic Vulnerability', '70-82%'],
    ['Heat Vulnerability Index', '64-79%']
]

y_start = 11
for i, row in enumerate(imputation_results):
    y_pos = y_start - i * 1.3
    if i == 0:  # Header
        for j, cell in enumerate(row):
            x_pos = 5 + j * 18
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=8, fontweight='bold', color=text_dark)
    else:
        for j, cell in enumerate(row):
            x_pos = 5 + j * 18
            ax.text(x_pos, y_pos, cell, ha='left', va='center', 
                   fontsize=7.5, color=text_dark)

plt.tight_layout()
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_corrected.svg', 
            format='svg', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_corrected.png', 
            format='png', dpi=300, bbox_inches='tight', facecolor='white')

# Also update the original files
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.svg', 
            format='svg', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.png', 
            format='png', dpi=300, bbox_inches='tight', facecolor='white')

print("Corrected technical reference slide created successfully!")
print("Files saved:")
print("- /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_corrected.svg")
print("- /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_corrected.png")
print("- /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.svg (updated)")
print("- /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_technical_reference_slide.png (updated)")