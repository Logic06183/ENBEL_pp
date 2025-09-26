#!/usr/bin/env python3
"""
Create Data Overview Slide for ENBEL Presentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np

def create_data_overview_slide():
    # ENBEL color scheme
    colors = {
        'blue': '#00539B',
        'orange': '#FF7F00', 
        'green': '#2CA02C',
        'red': '#DC2626',
        'lightblue': '#E6F0FA',
        'gray': '#8C8C8C'
    }
    
    # Create figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#F8FAFC')
    
    # Grid layout with title bar
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.12, 0.88], hspace=0.02)
    
    # Title section with blue header
    ax_title = fig.add_subplot(gs[0])
    ax_title.text(0.5, 0.55, 'Integrated Climate-Health Dataset Overview', 
                 ha='center', va='center', fontsize=28, fontweight='bold', 
                 color='white', transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15, 'Clinical Trial Cohorts with Spatiotemporal Climate Variables', 
                 ha='center', va='center', fontsize=16, style='italic',
                 color='white', alpha=0.9, transform=ax_title.transAxes)
    
    # Add blue background to title
    title_bg = Rectangle((0, 0), 1, 1, transform=ax_title.transAxes, 
                        facecolor=colors['blue'], alpha=1.0, zorder=0)
    ax_title.add_patch(title_bg)
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis('off')
    
    # Main content area
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Main dataset box (clinical trials)
    clinical_box = FancyBboxPatch((5, 48), 40, 35,
                                  boxstyle="round,pad=0.02",
                                  facecolor=colors['lightblue'], edgecolor=colors['blue'],
                                  linewidth=2.5)
    ax.add_patch(clinical_box)
    
    # Clinical data content
    ax.text(25, 78, "Clinical Trial Data", fontsize=18, weight='bold', 
            ha='center', color=colors['blue'])
    
    # Key statistics with corrected dates
    stats_y = 73
    ax.text(10, stats_y, "• 9,103 participants", fontsize=12, color='#2c3e50')
    ax.text(10, stats_y-3.5, "• Multiple HIV treatment trials", fontsize=12, color='#2c3e50')
    ax.text(10, stats_y-7, "• Johannesburg region", fontsize=12, color='#2c3e50')
    ax.text(10, stats_y-10.5, "• 2002-2021 study period", fontsize=12, color='#2c3e50')
    
    # Biomarkers section
    ax.text(10, 59, "Key Biomarkers:", fontsize=12, weight='bold', color=colors['blue'])
    biomarkers = [
        "• Blood pressure (sys/dia)",
        "• Fasting glucose",
        "• Cholesterol (HDL/LDL)",
        "• CD4 cell count",
        "• BMI & weight"
    ]
    for i, marker in enumerate(biomarkers):
        ax.text(10, 55.5-i*3, marker, fontsize=11, color='#34495e')
    
    # Climate data box
    climate_box = FancyBboxPatch((55, 48), 40, 35,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#FFF5E6', edgecolor=colors['orange'],
                                 linewidth=2.5)
    ax.add_patch(climate_box)
    
    ax.text(75, 78, "Climate Variables", fontsize=18, weight='bold',
            ha='center', color=colors['orange'])
    
    # Climate variables
    climate_y = 73
    ax.text(60, climate_y, "• Temperature (daily mean/max)", fontsize=12, color='#2c3e50')
    ax.text(60, climate_y-3.5, "• Humidity & heat index", fontsize=12, color='#2c3e50')
    ax.text(60, climate_y-7, "• Precipitation patterns", fontsize=12, color='#2c3e50')
    ax.text(60, climate_y-10.5, "• Wind speed & pressure", fontsize=12, color='#2c3e50')
    
    # Temporal features
    ax.text(60, 59, "Temporal Features:", fontsize=12, weight='bold', color=colors['orange'])
    temporal = [
        "• Multiple lag periods (0-30 days)",
        "• Seasonal indicators",
        "• Moving averages",
        "• Extreme event flags",
        "• Cumulative exposures"
    ]
    for i, temp in enumerate(temporal):
        ax.text(60, 55.5-i*3, temp, fontsize=11, color='#34495e')
    
    # Integration methodology box
    integration_box = FancyBboxPatch((10, 18), 80, 22,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#F0F8FF', edgecolor=colors['green'],
                                    linewidth=2.5)
    ax.add_patch(integration_box)
    
    ax.text(50, 35, "Data Integration Approach", fontsize=16, weight='bold',
            ha='center', color=colors['green'])
    
    # Integration details
    ax.text(50, 31, "Spatiotemporal Matching Framework", fontsize=13,
            ha='center', style='italic', color=colors['green'])
    
    # Three columns for integration
    col1_x, col2_x, col3_x = 25, 50, 75
    
    # Location matching
    location_circle = Circle((col1_x, 25), 2, facecolor=colors['lightblue'], 
                            edgecolor=colors['blue'], linewidth=2)
    ax.add_patch(location_circle)
    ax.text(col1_x, 25, "GPS", fontsize=12, ha='center', va='center', 
            weight='bold', color=colors['blue'])
    ax.text(col1_x, 20, "Spatial Matching", fontsize=11, ha='center', color='#2c3e50')
    ax.text(col1_x, 17.5, "Clinic coordinates", fontsize=10, ha='center', 
            style='italic', color='#7f8c8d')
    
    # Time alignment
    time_circle = Circle((col2_x, 25), 2, facecolor='#FFF5E6',
                        edgecolor=colors['orange'], linewidth=2)
    ax.add_patch(time_circle)
    ax.text(col2_x, 25, "TIME", fontsize=12, ha='center', va='center',
            weight='bold', color=colors['orange'])
    ax.text(col2_x, 20, "Temporal Alignment", fontsize=11, ha='center', color='#2c3e50')
    ax.text(col2_x, 17.5, "Visit dates → Climate", fontsize=10, ha='center',
            style='italic', color='#7f8c8d')
    
    # Lag analysis
    lag_circle = Circle((col3_x, 25), 2, facecolor='#E8F5E9',
                       edgecolor=colors['green'], linewidth=2)
    ax.add_patch(lag_circle)
    ax.text(col3_x, 25, "LAG", fontsize=12, ha='center', va='center',
            weight='bold', color=colors['green'])
    ax.text(col3_x, 20, "Lag Analysis", fontsize=11, ha='center', color='#2c3e50')
    ax.text(col3_x, 17.5, "0-30 day windows", fontsize=10, ha='center',
            style='italic', color='#7f8c8d')
    
    # Connection arrows
    arrow1 = FancyArrow(45, 65, 10, 0, width=1, head_width=2.5,
                       head_length=2, fc=colors['green'], ec=colors['green'], alpha=0.5)
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrow(25, 48, 0, -8, width=1, head_width=2.5,
                       head_length=2, fc=colors['green'], ec=colors['green'], alpha=0.5)
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrow(75, 48, 0, -8, width=1, head_width=2.5,
                       head_length=2, fc=colors['green'], ec=colors['green'], alpha=0.5)
    ax.add_patch(arrow3)
    
    # Key insights box (bottom)
    insights_box = FancyBboxPatch((5, 3), 90, 11,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#F8F9FA', edgecolor=colors['gray'],
                                 linewidth=2, alpha=0.95)
    ax.add_patch(insights_box)
    
    ax.text(50, 10.5, "Key Dataset Characteristics", fontsize=13, weight='bold',
            ha='center', color=colors['blue'])
    
    # Three columns of insights
    ax.text(22, 7, "• Individual-level data", fontsize=11, color=colors['green'])
    ax.text(22, 5, "• Longitudinal design", fontsize=11, color=colors['green'])
    
    ax.text(50, 7, "• High temporal resolution", fontsize=11, color=colors['green'], ha='center')
    ax.text(50, 5, "• Multi-lag modeling", fontsize=11, color=colors['green'], ha='center')
    
    ax.text(78, 7, "• Rich biomarker panel", fontsize=11, color=colors['green'], ha='right')
    ax.text(78, 5, "• Validated climate data", fontsize=11, color=colors['green'], ha='right')
    
    plt.tight_layout()
    plt.savefig('enbel_data_overview_slide.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig('enbel_data_overview_slide.png', format='png', bbox_inches='tight', dpi=300)
    print("Data overview slide saved as 'enbel_data_overview_slide.svg' and '.png'")
    
    plt.show()

if __name__ == "__main__":
    create_data_overview_slide()