import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Create figure with 16:9 aspect ratio
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# ENBEL colors
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00', 
    'green': '#2CA02C',
    'red': '#DC2626',
    'light_gray': '#F5F5F5',
    'dark_gray': '#666666',
    'white': '#FFFFFF'
}

# Background
fig.patch.set_facecolor(colors['white'])

# Title
ax.text(50, 95, 'Health Impact Attribution Framework & Future Climate Scenarios', 
        fontsize=24, fontweight='bold', ha='center', va='top', 
        color=colors['blue'])

# Subtitle
ax.text(50, 90, 'From Validated Discoveries to Policy-Relevant Climate Projections', 
        fontsize=16, ha='center', va='top', 
        color=colors['dark_gray'], style='italic')

# Attribution pathway boxes - simplified and cleaner
pathway_boxes = [
    {'x': 5, 'y': 70, 'w': 25, 'h': 15, 'color': colors['green'], 
     'title': 'Validated Findings', 
     'content': ['• Temperature-health dose-response', '• Clinical effect sizes validated', '• 2.9 mmHg/°C blood pressure¹', '• 8.2 mg/dL/°C glucose¹']},
    
    {'x': 37.5, 'y': 70, 'w': 25, 'h': 15, 'color': colors['orange'], 
     'title': 'Attribution Pathway', 
     'content': ['• Temperature warming scenarios', '• Health impact calculations', '• Population exposure models', '• Risk quantification']},
    
    {'x': 70, 'y': 70, 'w': 25, 'h': 15, 'color': colors['blue'], 
     'title': 'CMIP6 Scenarios²', 
     'content': ['• SSP pathways 2050-2100', '• Regional temperature projections', '• Johannesburg Metro focus', '• Policy-relevant timeframes']}
]

# Draw pathway boxes
for i, box in enumerate(pathway_boxes):
    # Main box
    rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'], 
                         boxstyle="round,pad=0.5", 
                         facecolor=box['color'], alpha=0.1,
                         edgecolor=box['color'], linewidth=2)
    ax.add_patch(rect)
    
    # Title
    ax.text(box['x'] + box['w']/2, box['y'] + box['h'] - 2, box['title'], 
            fontsize=14, fontweight='bold', ha='center', va='top', 
            color=box['color'])
    
    # Content
    for j, line in enumerate(box['content']):
        ax.text(box['x'] + 1, box['y'] + box['h'] - 5 - j*2.5, line, 
                fontsize=10, ha='left', va='top', 
                color=colors['dark_gray'])
    
    # Arrow to next box
    if i < len(pathway_boxes) - 1:
        ax.annotate('', xy=(pathway_boxes[i+1]['x'] - 1, box['y'] + box['h']/2), 
                   xytext=(box['x'] + box['w'] + 1, box['y'] + box['h']/2),
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors['blue']))

# SSP Scenarios Table - cleaner design
table_y = 55
table_height = 35

# Table header background
header_rect = Rectangle((5, table_y + table_height - 8), 90, 8, 
                       facecolor=colors['blue'], alpha=0.8)
ax.add_patch(header_rect)

# Table headers
headers = ['Scenario', 'Description', 'Temp. Rise', 'BP Impact', 'Glucose Impact', 'Population at Risk³']
header_positions = [12, 25, 40, 52, 67, 82]

for i, (header, pos) in enumerate(zip(headers, header_positions)):
    ax.text(pos, table_y + table_height - 4, header, 
            fontsize=11, fontweight='bold', ha='center', va='center', 
            color=colors['white'])

# Table data
scenarios = [
    ['SSP1-2.6', 'Paris Agreement', '+1.5°C', '+4.4 mmHg', '+12.3 mg/dL', '0.84M people'],
    ['SSP2-4.5', 'Current policies', '+2.5°C', '+7.3 mmHg', '+20.5 mg/dL', '1.12M people'],
    ['SSP3-7.0', 'Regional rivalry', '+4.0°C', '+11.6 mmHg', '+32.8 mg/dL', '1.44M people'],
    ['SSP5-8.5', 'Fossil fuel intensive', '+5.0°C', '+14.5 mmHg', '+41.0 mg/dL', '1.80M people']
]

row_colors = [colors['light_gray'], colors['white']]

for i, row in enumerate(scenarios):
    # Alternate row background
    if i % 2 == 0:
        row_rect = Rectangle((5, table_y + table_height - 16 - i*7), 90, 7, 
                           facecolor=row_colors[0], alpha=0.5)
        ax.add_patch(row_rect)
    
    # Scenario name with color coding
    scenario_color = [colors['green'], colors['orange'], colors['red'], colors['red']][i]
    ax.text(header_positions[0], table_y + table_height - 12.5 - i*7, row[0], 
            fontsize=11, fontweight='bold', ha='center', va='center', 
            color=scenario_color)
    
    # Other columns
    for j, (value, pos) in enumerate(zip(row[1:], header_positions[1:]), 1):
        weight = 'bold' if j >= 3 else 'normal'  # Bold for health impacts and population
        color = colors['red'] if j >= 3 else colors['dark_gray']  # Red for impacts
        ax.text(pos, table_y + table_height - 12.5 - i*7, value, 
                fontsize=10, fontweight=weight, ha='center', va='center', 
                color=color)

# Key findings box
findings_rect = FancyBboxPatch((5, 5), 60, 10, 
                              boxstyle="round,pad=1", 
                              facecolor=colors['light_gray'], alpha=0.7,
                              edgecolor=colors['blue'], linewidth=2)
ax.add_patch(findings_rect)

ax.text(7, 13, 'Key Attribution Findings:', 
        fontsize=12, fontweight='bold', ha='left', va='top', 
        color=colors['blue'])

findings_text = [
    '• Each 1°C warming = 2.9 mmHg BP increase + 8.2 mg/dL glucose increase¹',
    '• Johannesburg Metro population: 5.6M (baseline exposure)³',
    '• Health impacts scale proportionally with temperature rise',
    '• Population at risk calculated for hypertension/diabetes thresholds'
]

for i, finding in enumerate(findings_text):
    ax.text(7, 11 - i*1.8, finding, 
            fontsize=10, ha='left', va='top', 
            color=colors['dark_gray'])

# References box
ref_rect = FancyBboxPatch((68, 5), 27, 10, 
                         boxstyle="round,pad=1", 
                         facecolor=colors['white'], alpha=0.9,
                         edgecolor=colors['dark_gray'], linewidth=1)
ax.add_patch(ref_rect)

ax.text(70, 13, 'References:', 
        fontsize=10, fontweight='bold', ha='left', va='top', 
        color=colors['dark_gray'])

references = [
    '¹ ENBEL Study (2024)',
    '² IPCC AR6 CMIP6 (2021)', 
    '³ Stats SA Census (2022)'
]

for i, ref in enumerate(references):
    ax.text(70, 11.5 - i*1.5, ref, 
            fontsize=9, ha='left', va='top', 
            color=colors['dark_gray'])

# Save as SVG
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_perfected_cmip6_attribution.svg', 
            dpi=300, bbox_inches='tight', format='svg', 
            facecolor='white', edgecolor='none')

# Save as PNG
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_perfected_cmip6_attribution.png', 
            dpi=300, bbox_inches='tight', format='png', 
            facecolor='white', edgecolor='none')

plt.show()

print("Perfected CMIP6 Attribution Framework slide created!")
print("Files saved:")
print("- SVG: /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_perfected_cmip6_attribution.svg")
print("- PNG: /Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_perfected_cmip6_attribution.png")