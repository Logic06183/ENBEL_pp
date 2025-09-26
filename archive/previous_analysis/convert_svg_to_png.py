#!/usr/bin/env python3
"""
Convert the fixed spacing CMIP6 attribution framework SVG to PNG
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with 16:9 aspect ratio and high DPI
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('#FAFAFA')

# Remove axes
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)
ax.set_aspect('equal')
ax.axis('off')

# Title section
ax.text(960, 1000, 'Health Impact Attribution Framework & Future Climate Scenarios', 
        fontsize=28, fontweight='bold', ha='center', va='center', color='#1A365D')
ax.text(960, 960, 'From Validated Discoveries to Policy-Relevant Climate Projections', 
        fontsize=16, ha='center', va='center', color='#4A5568')

# Attribution pathway boxes
# Box 1: Validated Findings
box1 = FancyBboxPatch((80, 640), 320, 220, boxstyle="round,pad=10", 
                      facecolor='#E8F5E8', edgecolor='#2E8B57', linewidth=2)
ax.add_patch(box1)
ax.text(240, 820, 'Validated Findings', fontsize=14, fontweight='bold', 
        ha='center', va='center', color='#2E8B57')
ax.text(100, 780, '• Temperature-health dose-response', fontsize=10, ha='left', va='top', color='#2C5530')
ax.text(100, 760, '• Clinical effect sizes validated', fontsize=10, ha='left', va='top', color='#2C5530')
ax.text(100, 740, '• 2.9 mmHg/°C blood pressure¹', fontsize=10, ha='left', va='top', color='#2C5530')
ax.text(100, 720, '• 8.2 mg/dL/°C glucose¹', fontsize=10, ha='left', va='top', color='#2C5530')

# Arrow 1
ax.arrow(420, 750, 80, 0, head_width=15, head_length=20, fc='#2C5282', ec='#2C5282', linewidth=3)

# Box 2: Attribution Pathway
box2 = FancyBboxPatch((520, 640), 320, 220, boxstyle="round,pad=10", 
                      facecolor='#E3F2FD', edgecolor='#2563EB', linewidth=2)
ax.add_patch(box2)
ax.text(680, 820, 'Attribution Pathway', fontsize=14, fontweight='bold', 
        ha='center', va='center', color='#2563EB')
ax.text(540, 780, '• Temperature warming scenarios', fontsize=10, ha='left', va='top', color='#1E3A8A')
ax.text(540, 760, '• Health impact calculations', fontsize=10, ha='left', va='top', color='#1E3A8A')
ax.text(540, 740, '• Population exposure models', fontsize=10, ha='left', va='top', color='#1E3A8A')
ax.text(540, 720, '• Risk quantification', fontsize=10, ha='left', va='top', color='#1E3A8A')

# Arrow 2
ax.arrow(860, 750, 80, 0, head_width=15, head_length=20, fc='#2C5282', ec='#2C5282', linewidth=3)

# Box 3: CMIP6 Scenarios
box3 = FancyBboxPatch((960, 640), 320, 220, boxstyle="round,pad=10", 
                      facecolor='#FFF3E0', edgecolor='#EA580C', linewidth=2)
ax.add_patch(box3)
ax.text(1120, 820, 'CMIP6 Scenarios²', fontsize=14, fontweight='bold', 
        ha='center', va='center', color='#EA580C')
ax.text(980, 780, '• SSP pathways 2050-2100', fontsize=10, ha='left', va='top', color='#9A3412')
ax.text(980, 760, '• Regional temperature projections', fontsize=10, ha='left', va='top', color='#9A3412')
ax.text(980, 740, '• Johannesburg Metro focus', fontsize=10, ha='left', va='top', color='#9A3412')
ax.text(980, 720, '• Policy-relevant timeframes', fontsize=10, ha='left', va='top', color='#9A3412')

# Scenarios table
table_bg = FancyBboxPatch((80, 240), 1200, 380, boxstyle="round,pad=10", 
                          facecolor='white', edgecolor='#E2E8F0', linewidth=2)
ax.add_patch(table_bg)

# Table header
header_bg = FancyBboxPatch((80, 560), 1200, 60, boxstyle="round,pad=10", 
                           facecolor='#F7FAFC', edgecolor='#E2E8F0', linewidth=1)
ax.add_patch(header_bg)

# Headers
headers = ['Scenario', 'Description', 'Temp. Rise', 'BP Impact', 'Glucose Impact', 'Population at Risk³']
header_x = [140, 300, 480, 640, 820, 1080]
for i, header in enumerate(headers):
    ax.text(header_x[i], 590, header, fontsize=12, fontweight='bold', 
            ha='center', va='center', color='#2D3748')

# Table data
scenarios = [
    ['SSP1-2.6', 'Paris Agreement', '+1.5°C', '+4.4 mmHg', '+12.3 mg/dL', '0.84M people', '#2E8B57'],
    ['SSP2-4.5', 'Current policies', '+2.5°C', '+7.3 mmHg', '+20.5 mg/dL', '1.12M people', '#D69E2E'],
    ['SSP3-7.0', 'Regional rivalry', '+4.0°C', '+11.6 mmHg', '+32.8 mg/dL', '1.44M people', '#E53E3E'],
    ['SSP5-8.5', 'Fossil fuel intensive', '+5.0°C', '+14.5 mmHg', '+41.0 mg/dL', '1.80M people', '#C53030']
]

row_y = [525, 450, 375, 300]
for i, row in enumerate(scenarios):
    color = row[6]
    # Scenario name and population at risk in color, others in dark gray
    ax.text(140, row_y[i], row[0], fontsize=11, fontweight='bold', ha='center', va='center', color=color)
    ax.text(300, row_y[i], row[1], fontsize=11, ha='center', va='center', color='#4A5568')
    ax.text(480, row_y[i], row[2], fontsize=11, fontweight='bold', ha='center', va='center', color='#E53E3E')
    ax.text(640, row_y[i], row[3], fontsize=11, ha='center', va='center', color='#4A5568')
    ax.text(820, row_y[i], row[4], fontsize=11, ha='center', va='center', color='#4A5568')
    ax.text(1080, row_y[i], row[5], fontsize=11, fontweight='bold', ha='center', va='center', color=color)

# Key findings box
findings_bg = FancyBboxPatch((1340, 360), 500, 250, boxstyle="round,pad=10", 
                             facecolor='#F7FAFC', edgecolor='#4299E1', linewidth=2)
ax.add_patch(findings_bg)

ax.text(1590, 580, 'Key Attribution Findings:', fontsize=14, fontweight='bold', 
        ha='center', va='center', color='#2B6CB0')

findings_text = [
    '• Each 1°C warming = 2.9 mmHg BP increase',
    '   + 8.2 mg/dL glucose increase¹',
    '• Johannesburg Metro population:',
    '   5.6M (baseline exposure)³',
    '• Health impacts scale proportionally',
    '   with temperature rise',
    '• Population at risk calculated for',
    '   hypertension/diabetes thresholds'
]

y_pos = 545
for text in findings_text:
    ax.text(1360, y_pos, text, fontsize=10, ha='left', va='center', color='#2D3748')
    y_pos -= 20

# References box
refs_bg = FancyBboxPatch((1340, 240), 500, 90, boxstyle="round,pad=10", 
                         facecolor='#F7FAFC', edgecolor='#A0AEC0', linewidth=2)
ax.add_patch(refs_bg)

ax.text(1360, 315, 'References:', fontsize=12, fontweight='bold', color='#2D3748')
ax.text(1360, 295, '¹ ENBEL Study (2024)', fontsize=10, color='#4A5568')
ax.text(1360, 275, '² IPCC AR6 CMIP6 (2021)', fontsize=10, color='#4A5568')
ax.text(1360, 255, '³ Stats SA Census (2022)', fontsize=10, color='#4A5568')

# Footer
ax.text(960, 30, 'Climate Health Attribution Analysis | Johannesburg Metropolitan Area', 
        fontsize=10, ha='center', va='center', color='#718096')

# Add grid lines for table
# Horizontal lines
for y in [560, 485, 410, 335]:
    ax.plot([80, 1280], [y, y], color='#E2E8F0', linewidth=1)

# Vertical lines
for x in [200, 400, 560, 720, 920]:
    ax.plot([x, x], [240, 620], color='#E2E8F0', linewidth=1)

plt.tight_layout()
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_cmip6_attribution_framework_fixed_spacing.png', 
            dpi=300, bbox_inches='tight', facecolor='#FAFAFA', edgecolor='none')
plt.show()

print("✅ Fixed spacing CMIP6 attribution framework slide created successfully!")
print("Files generated:")
print("- SVG: /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_cmip6_attribution_framework_fixed_spacing.svg")
print("- PNG: /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_cmip6_attribution_framework_fixed_spacing.png")