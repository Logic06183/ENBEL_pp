import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with high DPI for crisp output
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Color palette - sophisticated and accessible
primary_blue = '#2E86AB'
accent_teal = '#A23B72'
warm_orange = '#F18F01'
soft_gray = '#C73E1D'
light_blue = '#E8F4F8'
cream = '#FDF7E7'
medium_gray = '#666666'
light_gray = '#E5E5E5'

# Background
background = FancyBboxPatch((0.2, 0.2), 15.6, 8.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor='white', 
                           edgecolor=light_gray, 
                           linewidth=1.5)
ax.add_patch(background)

# Title section
title_bg = FancyBboxPatch((0.5, 7.8), 15, 0.8, 
                         boxstyle="round,pad=0.02", 
                         facecolor=primary_blue, 
                         alpha=0.95)
ax.add_patch(title_bg)

ax.text(8, 8.2, 'Integrated Climate-Health Dataset: Comprehensive Multi-Study Analysis', 
        ha='center', va='center', fontsize=20, fontweight='bold', color='white')

# LEFT PANEL - Clinical Studies (simplified)
clinical_bg = FancyBboxPatch((0.8, 3.5), 4.8, 3.8, 
                            boxstyle="round,pad=0.08", 
                            facecolor=light_blue, 
                            edgecolor=primary_blue, 
                            linewidth=1.5)
ax.add_patch(clinical_bg)

ax.text(3.2, 6.9, 'CLINICAL STUDIES', ha='center', va='center', 
        fontsize=16, fontweight='bold', color=primary_blue)

# Clinical content - ultra simplified
clinical_text = [
    "9,103 participants from 17 studies",
    "Study period: 2002-2021",
    "Multiple study types: HIV treatment,\nCOVID, metabolic health, TB prevention",
    "Comprehensive health monitoring:\n30+ biomarkers including cardiovascular,\nmetabolic, immunologic markers",
    "Location: Johannesburg, South Africa"
]

y_positions = [6.4, 6.0, 5.5, 4.7, 3.9]
for i, text in enumerate(clinical_text):
    ax.text(1.2, y_positions[i], f"• {text}", ha='left', va='top', 
            fontsize=11, color=medium_gray, linespacing=1.2)

# RIGHT PANEL - Climate Data (much simpler)
climate_bg = FancyBboxPatch((10.4, 3.5), 4.8, 3.8, 
                           boxstyle="round,pad=0.08", 
                           facecolor=cream, 
                           edgecolor=warm_orange, 
                           linewidth=1.5)
ax.add_patch(climate_bg)

ax.text(12.8, 6.9, 'CLIMATE DATA', ha='center', va='center', 
        fontsize=16, fontweight='bold', color=warm_orange)

# Climate content - simplified
climate_text = [
    "High-resolution meteorological data",
    "Daily weather measurements\nmatched to participant visits",
    "Multiple climate factors: Temperature,\nhumidity, heat stress, wind, precipitation",
    "Temporal analysis: Immediate and\ndelayed climate effects examined",
    "Data quality: Validated\nmeteorological records"
]

for i, text in enumerate(climate_text):
    ax.text(10.8, y_positions[i], f"• {text}", ha='left', va='top', 
            fontsize=11, color=medium_gray, linespacing=1.2)

# CENTER PANEL - Integration (extremely simplified)
integration_bg = FancyBboxPatch((6.2, 4.2), 3.6, 3.1, 
                               boxstyle="round,pad=0.08", 
                               facecolor='white', 
                               edgecolor=accent_teal, 
                               linewidth=2)
ax.add_patch(integration_bg)

ax.text(8, 7.0, 'INTEGRATED APPROACH', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=accent_teal)

# Simple 3-step process
steps = [
    "Step 1: Match participant visits\nto local weather conditions",
    "Step 2: Add socioeconomic context\nfrom community surveys", 
    "Outcome: Comprehensive individual-\nlevel dataset linking climate,\nhealth, and social factors"
]

step_y = [6.4, 5.7, 4.8]
for i, step in enumerate(steps):
    if i < 2:
        ax.text(8, step_y[i], f"{step}", ha='center', va='center', 
                fontsize=10, color=medium_gray, linespacing=1.1)
    else:
        ax.text(8, step_y[i], f"{step}", ha='center', va='center', 
                fontsize=10, color=accent_teal, fontweight='bold', linespacing=1.1)

# Simple arrows between steps
arrow1 = patches.FancyArrowPatch((8, 6.1), (8, 5.9), 
                                arrowstyle='->', mutation_scale=15, 
                                color=accent_teal, linewidth=1.5)
ax.add_patch(arrow1)

arrow2 = patches.FancyArrowPatch((8, 5.4), (8, 5.2), 
                                arrowstyle='->', mutation_scale=15, 
                                color=accent_teal, linewidth=1.5)
ax.add_patch(arrow2)

# BOTTOM STATISTICS - simplified
stats_bg = FancyBboxPatch((1.5, 1.0), 13, 1.8, 
                         boxstyle="round,pad=0.06", 
                         facecolor=light_gray, 
                         alpha=0.3)
ax.add_patch(stats_bg)

ax.text(8, 2.4, 'DATASET OVERVIEW', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=primary_blue)

# Key statistics in a clean row
stats = [
    ("9,103", "Participants\nAnalyzed"),
    ("17", "Clinical Studies\nIntegrated"), 
    ("2002-2021", "Study\nPeriod"),
    ("Individual-Level", "Climate-Health\nAssociations")
]

x_positions = [3, 6, 10, 13]
for i, (number, label) in enumerate(stats):
    ax.text(x_positions[i], 1.8, number, ha='center', va='center', 
            fontsize=16, fontweight='bold', color=primary_blue)
    ax.text(x_positions[i], 1.4, label, ha='center', va='center', 
            fontsize=9, color=medium_gray, linespacing=1.1)

# Subtle connecting lines between panels
line1 = plt.Line2D([5.6, 6.2], [5.5, 5.5], color=light_gray, linewidth=1, alpha=0.7)
ax.add_line(line1)
line2 = plt.Line2D([9.8, 10.4], [5.5, 5.5], color=light_gray, linewidth=1, alpha=0.7)
ax.add_line(line2)

# Subtle footer
ax.text(8, 0.3, 'Comprehensive dataset enabling population-level climate-health research in urban African context', 
        ha='center', va='center', fontsize=10, style='italic', color=medium_gray)

plt.tight_layout()

# Save as both PNG and SVG
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_ultra_simple_data_overview.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_ultra_simple_data_overview.svg', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()
print("Ultra-simplified data overview slide created successfully!")
print("Files saved: enbel_ultra_simple_data_overview.png and enbel_ultra_simple_data_overview.svg")