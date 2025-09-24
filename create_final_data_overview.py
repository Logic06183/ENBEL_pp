#!/usr/bin/env python3
"""
Create final clean data overview slide without coverage percentages
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np

# ENBEL color scheme
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00', 
    'green': '#2CA02C',
    'red': '#DC2626',
    'purple': '#9467BD',
    'gray': '#8C8C8C',
    'lightblue': '#E6F0FA',
    'darkblue': '#003d72'
}

def create_final_overview():
    # Create figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#F8FAFC')
    
    # Grid layout with title bar
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.12, 0.65, 0.23], 
                  width_ratios=[0.33, 0.33, 0.34], hspace=0.03, wspace=0.03)
    
    # Title section with blue header
    ax_title = fig.add_subplot(gs[0, :])
    title_bg = Rectangle((0, 0), 1, 1, transform=ax_title.transAxes, 
                        facecolor=colors['blue'], alpha=1.0, zorder=0)
    ax_title.add_patch(title_bg)
    ax_title.text(0.5, 0.55, 'Integrated Climate-Health Dataset Overview', 
                 ha='center', va='center', fontsize=26, fontweight='bold', 
                 color='white', transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15, 'Comprehensive Multi-Study Analysis with Socioeconomic Integration', 
                 ha='center', va='center', fontsize=14, style='italic',
                 color='white', alpha=0.95, transform=ax_title.transAxes)
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis('off')
    
    # LEFT PANEL - Clinical Studies
    ax_clinical = fig.add_subplot(gs[1, 0])
    ax_clinical.axis('off')
    
    # Clinical box
    clinical_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                  boxstyle="round,pad=0.02",
                                  facecolor=colors['lightblue'], 
                                  edgecolor=colors['blue'],
                                  linewidth=2.5, alpha=0.95)
    ax_clinical.add_patch(clinical_box)
    
    # Clinical content
    ax_clinical.text(0.5, 0.92, 'Clinical Studies', fontsize=16, weight='bold',
                    ha='center', color=colors['blue'], transform=ax_clinical.transAxes)
    
    y_pos = 0.82
    content = [
        ('• 9,103 participants', 12),
        ('• 17 studies (2002-2021)', 12),
        ('• Study types:', 11),
        ('  - HIV treatment', 10),
        ('  - COVID studies', 10),
        ('  - Metabolic health', 10),
        ('  - TB prevention', 10),
        ('  - General clinical', 10),
        ('', 8),
        ('• 30+ biomarkers:', 11),
        ('  - Cardiovascular', 10),
        ('  - Metabolic', 10),
        ('  - Immunologic', 10),
        ('• Johannesburg, SA', 11)
    ]
    
    for text, size in content:
        ax_clinical.text(0.1, y_pos, text, fontsize=size, 
                        color='#2c3e50', transform=ax_clinical.transAxes)
        y_pos -= 0.055
    
    # CENTER PANEL - Integration
    ax_integration = fig.add_subplot(gs[1, 1])
    ax_integration.axis('off')
    
    # Integration box
    integration_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#FFF5F0', 
                                    edgecolor=colors['orange'],
                                    linewidth=2.5, alpha=0.95)
    ax_integration.add_patch(integration_box)
    
    ax_integration.text(0.5, 0.92, 'Data Integration', fontsize=16, weight='bold',
                       ha='center', color=colors['orange'], transform=ax_integration.transAxes)
    
    # Step 1
    ax_integration.text(0.1, 0.78, 'Step 1: Spatiotemporal', fontsize=12, weight='bold',
                       color=colors['blue'], transform=ax_integration.transAxes)
    ax_integration.text(0.1, 0.71, '• Match visits to weather', fontsize=10,
                       color='#2c3e50', transform=ax_integration.transAxes)
    ax_integration.text(0.1, 0.65, '• GPS coordination', fontsize=10,
                       color='#2c3e50', transform=ax_integration.transAxes)
    ax_integration.text(0.1, 0.59, '• Multiple lag periods', fontsize=10,
                       color='#2c3e50', transform=ax_integration.transAxes)
    
    # Step 2
    ax_integration.text(0.1, 0.45, 'Step 2: Socioeconomic', fontsize=12, weight='bold',
                       color=colors['green'], transform=ax_integration.transAxes)
    ax_integration.text(0.1, 0.38, '• G-SORO survey data', fontsize=10,
                       color='#2c3e50', transform=ax_integration.transAxes)
    ax_integration.text(0.1, 0.32, '• Demographic matching', fontsize=10,
                       color='#2c3e50', transform=ax_integration.transAxes)
    ax_integration.text(0.1, 0.26, '• Vulnerability indices', fontsize=10,
                       color='#2c3e50', transform=ax_integration.transAxes)
    
    # Result
    ax_integration.text(0.5, 0.12, '→ Comprehensive dataset', fontsize=11, weight='bold',
                       style='italic', ha='center', color=colors['purple'], 
                       transform=ax_integration.transAxes)
    
    # RIGHT PANEL - Climate Data
    ax_climate = fig.add_subplot(gs[1, 2])
    ax_climate.axis('off')
    
    # Climate box
    climate_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#F0FFF0', 
                                 edgecolor=colors['green'],
                                 linewidth=2.5, alpha=0.95)
    ax_climate.add_patch(climate_box)
    
    ax_climate.text(0.5, 0.92, 'Climate Data', fontsize=16, weight='bold',
                   ha='center', color=colors['green'], transform=ax_climate.transAxes)
    
    y_pos = 0.82
    climate_content = [
        ('• Daily meteorological', 11),
        ('• 2002-2021 period', 11),
        ('', 8),
        ('• Variables:', 11),
        ('  - Temperature', 10),
        ('  - Humidity', 10),
        ('  - Heat index', 10),
        ('  - Wind speed', 10),
        ('  - Precipitation', 10),
        ('', 8),
        ('• Data sources:', 11),
        ('  - ERA5 Reanalysis', 10),
        ('  - SAWS stations', 10),
        ('• Quality validated', 11)
    ]
    
    for text, size in climate_content:
        ax_climate.text(0.1, y_pos, text, fontsize=size, 
                       color='#2c3e50', transform=ax_climate.transAxes)
        y_pos -= 0.055
    
    # BOTTOM STATISTICS BAR
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Stats background
    stats_box = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                               boxstyle="round,pad=0.02",
                               facecolor='#F8F9FA', 
                               edgecolor=colors['gray'],
                               linewidth=2, alpha=0.95)
    ax_stats.add_patch(stats_box)
    
    # Statistics - NO COVERAGE PERCENTAGE
    stats = [
        ('9,103', 'Participants', 0.15),
        ('17', 'Studies', 0.3),
        ('30+', 'Biomarkers', 0.45),
        ('19 years', 'Study Period', 0.6),
        ('Individual', 'Level Analysis', 0.8)
    ]
    
    for number, label, x_pos in stats:
        ax_stats.text(x_pos, 0.55, number, fontsize=20, weight='bold',
                     ha='center', color=colors['blue'], transform=ax_stats.transAxes)
        ax_stats.text(x_pos, 0.35, label, fontsize=10, ha='center',
                     color=colors['gray'], transform=ax_stats.transAxes)
    
    plt.tight_layout()
    
    # Save as both SVG and PNG
    plt.savefig('enbel_data_overview_final.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('enbel_data_overview_final.png', format='png', bbox_inches='tight', dpi=150)
    print("Final data overview slide saved as 'enbel_data_overview_final.svg' and '.png'")

if __name__ == "__main__":
    create_final_overview()