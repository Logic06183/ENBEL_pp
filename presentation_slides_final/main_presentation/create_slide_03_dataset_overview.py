#!/usr/bin/env python3
"""
Create Slide 3: Research Dataset Overview
Three-panel layout showcasing clinical, climate, and socioeconomic data
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Paths
OUTPUT_DIR = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection/presentation_slides_final/main_presentation"

# Colors
PRIMARY_BLUE = '#2E7AB5'
CLINICAL_BLUE = '#3498DB'
CLIMATE_GREEN = '#27AE60'
SOCIO_ORANGE = '#E67E22'
TEXT_COLOR = '#2C3E50'
BACKGROUND = '#F8F9FA'
ACCENT_GOLD = '#F39C12'

print("=" * 80)
print("SLIDE 3: RESEARCH DATASET OVERVIEW")
print("=" * 80)

# Create figure
fig = plt.figure(figsize=(19.2, 10.8), dpi=150)
fig.patch.set_facecolor('white')

# Title
fig.suptitle('Research Datasets: Integrating Clinical, Climate, and Socioeconomic Data',
             fontsize=34, fontweight='bold', color=TEXT_COLOR, y=0.96)

subtitle = 'Multi-source data integration for comprehensive climate-health analysis in Johannesburg, South Africa'
fig.text(0.5, 0.91, subtitle, ha='center', fontsize=16, color=TEXT_COLOR, style='italic')

# Create main axis
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# ============================================================================
# PANEL 1: Clinical Studies (Left)
# ============================================================================
print("\n1. Creating Clinical Studies panel...")

x1, y1, w1, h1 = 8, 30, 24, 50
# Main box
clinical_box = FancyBboxPatch((x1, y1), w1, h1,
                              boxstyle="round,pad=1",
                              edgecolor=CLINICAL_BLUE, facecolor=BACKGROUND,
                              linewidth=4, zorder=2)
ax.add_patch(clinical_box)

# Icon circle
icon_circle = Circle((x1 + w1/2, y1 + h1 - 8), 4,
                     facecolor=CLINICAL_BLUE, edgecolor='white', linewidth=3, zorder=3)
ax.add_patch(icon_circle)
ax.text(x1 + w1/2, y1 + h1 - 8, '⚕', ha='center', va='center',
        fontsize=32, color='white', fontweight='bold')

# Title
ax.text(x1 + w1/2, y1 + h1 - 15, 'Clinical Studies',
        ha='center', va='top', fontsize=18, fontweight='bold', color=CLINICAL_BLUE)

# Main count
ax.text(x1 + w1/2, y1 + h1 - 22, '11,398',
        ha='center', va='top', fontsize=32, fontweight='bold', color=TEXT_COLOR)
ax.text(x1 + w1/2, y1 + h1 - 28, 'Patient Records',
        ha='center', va='top', fontsize=13, color=TEXT_COLOR, style='italic')

# Details
details = [
    '• 17 Clinical Trials',
    '• HIV, TB/HIV, COVID-19',
    '• 2002-2021 (20 years)',
    '',
    'Key Biomarkers:',
    '  CD4 cell count',
    '  Hemoglobin',
    '  Creatinine',
    '  Glucose',
    '  Blood pressure',
    '',
    '✓ Geocoded locations',
    '✓ Daily timestamps',
    '✓ SA medical units'
]

y_text = y1 + h1 - 35
for detail in details:
    ax.text(x1 + w1/2, y_text, detail,
            ha='center', va='top', fontsize=10, color=TEXT_COLOR)
    y_text -= 3

# ============================================================================
# PANEL 2: Climate Data (Center)
# ============================================================================
print("   Creating Climate Data panel...")

x2, y2, w2, h2 = 38, 30, 24, 50
# Main box
climate_box = FancyBboxPatch((x2, y2), w2, h2,
                             boxstyle="round,pad=1",
                             edgecolor=CLIMATE_GREEN, facecolor=BACKGROUND,
                             linewidth=4, zorder=2)
ax.add_patch(climate_box)

# Icon circle
icon_circle2 = Circle((x2 + w2/2, y2 + h2 - 8), 4,
                      facecolor=CLIMATE_GREEN, edgecolor='white', linewidth=3, zorder=3)
ax.add_patch(icon_circle2)
ax.text(x2 + w2/2, y2 + h2 - 8, '🌡', ha='center', va='center',
        fontsize=28, color='white', fontweight='bold')

# Title
ax.text(x2 + w2/2, y2 + h2 - 15, 'Climate Data',
        ha='center', va='top', fontsize=18, fontweight='bold', color=CLIMATE_GREEN)

# Main count
ax.text(x2 + w2/2, y2 + h2 - 22, 'ERA5',
        ha='center', va='top', fontsize=28, fontweight='bold', color=TEXT_COLOR)
ax.text(x2 + w2/2, y2 + h2 - 28, 'Reanalysis Dataset',
        ha='center', va='top', fontsize=13, color=TEXT_COLOR, style='italic')

# Details
details2 = [
    '• 99.5% coverage',
    '• Hourly → Daily',
    '• 0.25° resolution',
    '',
    'Climate Variables:',
    '  Temperature (°C)',
    '  Humidity (%)',
    '  Precipitation (mm)',
    '  Wind speed (m/s)',
    '  Heat stress index',
    '',
    '✓ Multi-lag (7,14,30d)',
    '✓ Anomaly detection',
    '✓ Heat wave ID'
]

y_text = y2 + h2 - 35
for detail in details2:
    ax.text(x2 + w2/2, y_text, detail,
            ha='center', va='top', fontsize=10, color=TEXT_COLOR)
    y_text -= 3

# ============================================================================
# PANEL 3: Socioeconomic Data (Right)
# ============================================================================
print("   Creating Socioeconomic Data panel...")

x3, y3, w3, h3 = 68, 30, 24, 50
# Main box
socio_box = FancyBboxPatch((x3, y3), w3, h3,
                           boxstyle="round,pad=1",
                           edgecolor=SOCIO_ORANGE, facecolor=BACKGROUND,
                           linewidth=4, zorder=2)
ax.add_patch(socio_box)

# Icon circle
icon_circle3 = Circle((x3 + w3/2, y3 + h3 - 8), 4,
                      facecolor=SOCIO_ORANGE, edgecolor='white', linewidth=3, zorder=3)
ax.add_patch(icon_circle3)
ax.text(x3 + w3/2, y3 + h3 - 8, '🏘', ha='center', va='center',
        fontsize=28, color='white', fontweight='bold')

# Title
ax.text(x3 + w3/2, y3 + h3 - 15, 'Socioeconomic',
        ha='center', va='top', fontsize=18, fontweight='bold', color=SOCIO_ORANGE)

# Main count
ax.text(x3 + w3/2, y3 + h3 - 22, '58,616',
        ha='center', va='top', fontsize=32, fontweight='bold', color=TEXT_COLOR)
ax.text(x3 + w3/2, y3 + h3 - 28, 'GCRO Households',
        ha='center', va='top', fontsize=13, color=TEXT_COLOR, style='italic')

# Details
details3 = [
    '• Quality of Life surveys',
    '• 6 waves (2011-2021)',
    '• Ward-level data',
    '',
    'Key Variables:',
    '  Dwelling type',
    '  Income quintiles',
    '  Education level',
    '  Employment status',
    '  Heat vulnerability',
    '',
    '✓ Geocoded wards',
    '✓ Climate matched',
    '✓ Vulnerability index'
]

y_text = y3 + h3 - 35
for detail in details3:
    ax.text(x3 + w3/2, y_text, detail,
            ha='center', va='top', fontsize=10, color=TEXT_COLOR)
    y_text -= 3

# ============================================================================
# Add Integration Arrows
# ============================================================================
print("   Adding integration arrows...")

# Arrow 1: Clinical to Climate
arrow1 = FancyArrowPatch((x1 + w1, y1 + h1/2), (x2, y2 + h2/2),
                        arrowstyle='->', mutation_scale=40,
                        linewidth=3, color=PRIMARY_BLUE, zorder=1)
ax.add_patch(arrow1)
ax.text((x1 + w1 + x2)/2, y1 + h1/2 + 3, 'Match',
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        color=PRIMARY_BLUE, style='italic')

# Arrow 2: Climate to Socio
arrow2 = FancyArrowPatch((x2 + w2, y2 + h2/2), (x3, y3 + h3/2),
                        arrowstyle='->', mutation_scale=40,
                        linewidth=3, color=PRIMARY_BLUE, zorder=1)
ax.add_patch(arrow2)
ax.text((x2 + w2 + x3)/2, y2 + h2/2 + 3, 'Integrate',
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        color=PRIMARY_BLUE, style='italic')

# ============================================================================
# Add Summary Statistics Box
# ============================================================================
print("   Adding summary statistics...")

summary_box = FancyBboxPatch((15, 10), 70, 15,
                             boxstyle="round,pad=1",
                             edgecolor=ACCENT_GOLD, facecolor='white',
                             linewidth=3, zorder=2, alpha=0.95)
ax.add_patch(summary_box)

ax.text(50, 20, 'Integrated Dataset Summary', ha='center', va='center',
        fontsize=16, fontweight='bold', color=TEXT_COLOR)

summary_stats = [
    '70,014 Total Records',
    '20 Years Coverage (2002-2021)',
    '99.5% Climate Match',
    '4.1M Climate Observations',
    '250+ Wards',
    '17 Biomarkers'
]

x_stat = 20
for stat in summary_stats:
    ax.text(x_stat, 14, stat, ha='center', va='center',
            fontsize=11, fontweight='bold', color=TEXT_COLOR,
            bbox=dict(boxstyle='round', facecolor=BACKGROUND,
                     edgecolor=PRIMARY_BLUE, linewidth=1.5, pad=0.5))
    x_stat += 13

# Add methodology note
methodology_text = (
    'Data Integration: Clinical records geocoded and matched to ERA5 climate reanalysis using spatiotemporal keys (±24h, ±10km).\n'
    'GCRO household surveys linked to climate via ward centroids. All datasets harmonized to South African medical and geographic standards.\n'
    'Quality assurance: Biomarker range validation, coordinate precision assessment, temporal consistency checks.'
)
fig.text(0.5, 0.03, methodology_text, ha='center', fontsize=10,
         color=TEXT_COLOR, style='italic', wrap=True,
         bbox=dict(boxstyle='round', facecolor=BACKGROUND,
                   edgecolor=PRIMARY_BLUE, linewidth=2))

# Adjust layout
plt.tight_layout(rect=[0, 0.08, 1, 0.88])

# Save outputs
output_svg = f"{OUTPUT_DIR}/slide_03_dataset_overview.svg"
output_png = output_svg.replace('.svg', '.png')

plt.savefig(output_svg, format='svg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\n✓ Slide 3 saved: {output_svg}")

plt.savefig(output_png, format='png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Preview saved: {output_png}")

plt.close()

print("\n✓ Slide 3 completed successfully!")
print("=" * 80)
