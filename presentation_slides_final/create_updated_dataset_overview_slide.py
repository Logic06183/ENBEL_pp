#!/usr/bin/env python3
"""
Create Updated ENBEL Climate-Health Dataset Overview Slide
Recreates the original format with verified, updated numbers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_dataset_overview_slide():
    """Create the comprehensive dataset overview slide with updated verified numbers"""
    
    # Create figure with 16:9 aspect ratio
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Color scheme matching original
    blue_header = '#1e3a8a'  # Deep blue
    blue_clinical = '#3b82f6'  # Clinical blue
    green_integration = '#10b981'  # Integration green
    orange_climate = '#f59e0b'  # Climate orange
    white = '#ffffff'
    dark_text = '#1f2937'
    light_gray = '#f3f4f6'
    
    # Background
    bg_rect = patches.Rectangle((0, 0), 16, 9, linewidth=0, 
                               facecolor=light_gray, alpha=0.3)
    ax.add_patch(bg_rect)
    
    # Header section with gradient effect
    header_rect = FancyBboxPatch((0, 7.5), 16, 1.5, 
                                boxstyle="round,pad=0.02",
                                facecolor=blue_header, 
                                edgecolor='none')
    ax.add_patch(header_rect)
    
    # Header text
    ax.text(8, 8.6, 'Integrated Climate-Health Dataset: Comprehensive Multi-Study Analysis', 
            fontsize=22, fontweight='bold', color=white, ha='center', va='center')
    ax.text(8, 8.1, 'Multiple Study Types, Extensive Biomarkers & Socioeconomic Integration', 
            fontsize=14, color=white, ha='center', va='center', alpha=0.9)
    
    # WITS logo placeholder (top right)
    logo_rect = patches.Rectangle((14.5, 8.2), 1.2, 0.6, 
                                 linewidth=2, facecolor=white, edgecolor=blue_header)
    ax.add_patch(logo_rect)
    ax.text(15.1, 8.5, 'WITS', fontsize=12, fontweight='bold', 
            color=blue_header, ha='center', va='center')
    
    # Section 1: Clinical Studies (Blue)
    clinical_rect = FancyBboxPatch((0.3, 1.5), 4.8, 5.8, 
                                  boxstyle="round,pad=0.05",
                                  facecolor=blue_clinical, 
                                  edgecolor='none', alpha=0.15)
    ax.add_patch(clinical_rect)
    
    # Clinical header
    clinical_header = FancyBboxPatch((0.4, 6.8), 4.6, 0.4, 
                                    boxstyle="round,pad=0.02",
                                    facecolor=blue_clinical, 
                                    edgecolor='none')
    ax.add_patch(clinical_header)
    
    ax.text(2.7, 7.0, 'CLINICAL STUDIES', fontsize=16, fontweight='bold', 
            color=white, ha='center', va='center')
    
    # Main clinical numbers
    ax.text(2.7, 6.4, '11,398 participants from 17 studies', 
            fontsize=18, fontweight='bold', color=blue_clinical, ha='center', va='center')
    ax.text(2.7, 6.1, '(2002-2021)', 
            fontsize=12, color=blue_clinical, ha='center', va='center', style='italic')
    
    # Study breakdown based on actual data categorization  
    study_data = [
        ('General Clinical Studies', '6,083', '53.4%'),
        ('HIV Treatment & Prevention Studies', '3,974', '34.9%'),
        ('COVID-19 Studies', '998', '8.8%'),
        ('TB Prevention Trials', '343', '3.0%')
    ]
    
    y_pos = 5.6
    for study_type, count, percentage in study_data:
        ax.text(0.6, y_pos, f'• {study_type}:', fontsize=11, fontweight='bold', 
                color=dark_text, ha='left', va='center')
        ax.text(4.8, y_pos, f'{count} ({percentage})', fontsize=11, 
                color=blue_clinical, ha='right', va='center', fontweight='bold')
        y_pos -= 0.3
    
    # Biomarkers section
    ax.text(2.7, 4.2, 'BIOMARKERS (30+ variables)', fontsize=12, fontweight='bold', 
            color=dark_text, ha='center', va='center')
    
    biomarker_categories = [
        'Cardiovascular: BP monitoring',
        'Metabolic: Glucose, lipids, HbA1c',
        'Immunologic: CD4, viral loads',
        'Anthropometric: Height/weight'
    ]
    
    y_pos = 3.8
    for category in biomarker_categories:
        ax.text(0.6, y_pos, f'• {category}', fontsize=10, 
                color=dark_text, ha='left', va='center')
        y_pos -= 0.25
    
    # Section 2: Integration Framework (Green/Orange)
    integration_rect = FancyBboxPatch((5.5, 1.5), 5.0, 5.8, 
                                     boxstyle="round,pad=0.05",
                                     facecolor=green_integration, 
                                     edgecolor='none', alpha=0.15)
    ax.add_patch(integration_rect)
    
    # Integration header
    integration_header = FancyBboxPatch((5.6, 6.8), 4.8, 0.4, 
                                       boxstyle="round,pad=0.02",
                                       facecolor=green_integration, 
                                       edgecolor='none')
    ax.add_patch(integration_header)
    
    ax.text(8.0, 7.0, 'INTEGRATION FRAMEWORK', fontsize=16, fontweight='bold', 
            color=white, ha='center', va='center')
    
    # Step 1: Spatiotemporal Matching
    step1_rect = FancyBboxPatch((5.7, 5.5), 4.6, 1.1, 
                               boxstyle="round,pad=0.03",
                               facecolor=orange_climate, 
                               edgecolor='none', alpha=0.2)
    ax.add_patch(step1_rect)
    
    ax.text(8.0, 6.4, 'STEP 1: SPATIOTEMPORAL MATCHING', fontsize=12, fontweight='bold', 
            color=orange_climate, ha='center', va='center')
    
    step1_details = [
        '• GPS coordinate alignment',
        '• Temporal alignment: Visit dates',
        '• Multiple lag periods (0-30 days) for delayed effects'
    ]
    
    y_pos = 6.0
    for detail in step1_details:
        ax.text(5.9, y_pos, detail, fontsize=10, 
                color=dark_text, ha='left', va='center')
        y_pos -= 0.2
    
    # Step 2: Socioeconomic Integration
    step2_rect = FancyBboxPatch((5.7, 3.8), 4.6, 1.5, 
                               boxstyle="round,pad=0.03",
                               facecolor=green_integration, 
                               edgecolor='none', alpha=0.2)
    ax.add_patch(step2_rect)
    
    ax.text(8.0, 5.0, 'STEP 2: SOCIOECONOMIC INTEGRATION', fontsize=12, fontweight='bold', 
            color=green_integration, ha='center', va='center')
    
    ax.text(8.0, 4.6, 'GCRO Community Survey', fontsize=11, fontweight='bold', 
            color=dark_text, ha='center', va='center')
    ax.text(8.0, 4.4, '(58,616 Johannesburg residents)', fontsize=10, 
            color=dark_text, ha='center', va='center', style='italic')
    
    step2_details = [
        'Method: Multi-dimensional matching algorithm',
        'Variables Added: Education, employment status,',
        'housing/economic vulnerability'
    ]
    
    y_pos = 4.1
    for detail in step2_details:
        ax.text(5.9, y_pos, f'• {detail}', fontsize=10, 
                color=dark_text, ha='left', va='center')
        y_pos -= 0.15
    
    # Section 3: Climate Data (Orange)
    climate_rect = FancyBboxPatch((11.0, 1.5), 4.7, 5.8, 
                                 boxstyle="round,pad=0.05",
                                 facecolor=orange_climate, 
                                 edgecolor='none', alpha=0.15)
    ax.add_patch(climate_rect)
    
    # Climate header
    climate_header = FancyBboxPatch((11.1, 6.8), 4.5, 0.4, 
                                   boxstyle="round,pad=0.02",
                                   facecolor=orange_climate, 
                                   edgecolor='none')
    ax.add_patch(climate_header)
    
    ax.text(13.35, 7.0, 'CLIMATE DATA', fontsize=16, fontweight='bold', 
            color=white, ha='center', va='center')
    
    # Climate data sources
    ax.text(13.35, 6.4, 'Data Sources: ERA5 Reanalysis', 
            fontsize=14, fontweight='bold', color=orange_climate, ha='center', va='center')
    ax.text(13.35, 6.1, '(2002-2021)', 
            fontsize=12, color=orange_climate, ha='center', va='center', style='italic')
    
    # Climate categories
    climate_categories = [
        ('Temperature:', 'Mean, max, min, ranges'),
        ('Heat Index:', 'Multiple lag periods'),
        ('Humidity:', 'Daily patterns'),
        ('Wind:', 'Speed, gusts, direction')
    ]
    
    y_pos = 5.6
    for category, description in climate_categories:
        ax.text(11.3, y_pos, f'• {category}', fontsize=11, fontweight='bold', 
                color=dark_text, ha='left', va='center')
        ax.text(11.3, y_pos-0.15, f'  {description}', fontsize=10, 
                color=dark_text, ha='left', va='center')
        y_pos -= 0.4
    
    # Temporal coverage
    ax.text(13.35, 3.8, 'Temporal Coverage:', fontsize=11, fontweight='bold', 
            color=dark_text, ha='center', va='center')
    ax.text(13.35, 3.5, '0-30 day lag analysis', fontsize=11, 
            color=orange_climate, ha='center', va='center', fontweight='bold')
    
    # Bottom statistics bar
    stats_rect = FancyBboxPatch((0.5, 0.3), 15.0, 0.8, 
                               boxstyle="round,pad=0.03",
                               facecolor=blue_header, 
                               edgecolor='none')
    ax.add_patch(stats_rect)
    
    # Statistics
    stats = [
        ('11,398', 'Participants'),
        ('17', 'Studies'),
        ('30+', 'Biomarkers'),
        ('19 years', 'Study Period'),
        ('Individual', 'Level Analysis')
    ]
    
    x_positions = [1.8, 4.2, 6.6, 9.0, 11.4, 13.8]
    for i, (number, label) in enumerate(stats):
        ax.text(x_positions[i], 0.85, number, fontsize=16, fontweight='bold', 
                color=white, ha='center', va='center')
        ax.text(x_positions[i], 0.55, label, fontsize=11, 
                color=white, ha='center', va='center')
        
        # Add separator lines (except after last item)
        if i < len(stats) - 1:
            ax.plot([x_positions[i] + 0.8, x_positions[i] + 0.8], [0.4, 1.0], 
                   color=white, alpha=0.5, linewidth=1)
    
    # Data quality indicator
    ax.text(15.2, 1.8, '99.5%', fontsize=12, fontweight='bold', 
            color=green_integration, ha='center', va='center')
    ax.text(15.2, 1.6, 'Climate', fontsize=10, 
            color=green_integration, ha='center', va='center')
    ax.text(15.2, 1.45, 'Coverage', fontsize=10, 
            color=green_integration, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save as both PNG and SVG
    output_path_png = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_updated_dataset_overview_final.png'
    output_path_svg = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_updated_dataset_overview_final.svg'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Updated dataset overview slide saved to:")
    print(f"PNG: {output_path_png}")
    print(f"SVG: {output_path_svg}")
    
    plt.show()
    
    return output_path_svg

if __name__ == "__main__":
    create_dataset_overview_slide()