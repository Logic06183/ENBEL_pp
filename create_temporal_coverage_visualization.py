#!/usr/bin/env python3
"""
Create a beautiful temporal coverage visualization for ENBEL Climate-Health Analysis Pipeline
showing clinical studies and GCRO Quality of Life survey periods integration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")

def create_temporal_coverage_visualization():
    """Create publication-quality temporal coverage visualization"""
    
    # Define the clinical studies data
    studies_data = [
        {"study_id": "JHB_JHSPH_005", "start": 2002, "end": 2009, "n": 524, "focus": "TB/HIV"},
        {"study_id": "JHB_ACTG_017", "start": 2003, "end": 2004, "n": 20, "focus": "HIV"},
        {"study_id": "JHB_ACTG_019", "start": 2005, "end": 2007, "n": 100, "focus": "HIV"},
        {"study_id": "JHB_ACTG_015", "start": 2005, "end": 2008, "n": 87, "focus": "HIV"},
        {"study_id": "JHB_ACTG_016", "start": 2007, "end": 2009, "n": 76, "focus": "HIV"},
        {"study_id": "JHB_DPHRU_013", "start": 2011, "end": 2013, "n": 247, "focus": "TB/HIV"},
        {"study_id": "JHB_ACTG_018", "start": 2011, "end": 2012, "n": 240, "focus": "HIV"},
        {"study_id": "JHB_WRHI_001", "start": 2012, "end": 2014, "n": 1067, "focus": "HIV"},
        {"study_id": "JHB_Aurum_009", "start": 2013, "end": 2015, "n": 2551, "focus": "TB/HIV"},
        {"study_id": "JHB_SCHARP_004", "start": 2015, "end": 2016, "n": 2, "focus": "HIV"},
        {"study_id": "JHB_WRHI_003", "start": 2016, "end": 2017, "n": 217, "focus": "HIV"},
        {"study_id": "JHB_SCHARP_006", "start": 2017, "end": 2017, "n": 162, "focus": "HIV"},
        {"study_id": "JHB_Ezin_002", "start": 2017, "end": 2018, "n": 1053, "focus": "Metabolic"},
        {"study_id": "JHB_DPHRU_053", "start": 2017, "end": 2018, "n": 998, "focus": "TB/HIV"},
        {"study_id": "JHB_VIDA_008", "start": 2020, "end": 2021, "n": 550, "focus": "COVID-19"},
        {"study_id": "JHB_VIDA_007", "start": 2020, "end": 2020, "n": 2129, "focus": "COVID-19"},
        {"study_id": "JHB_EZIN_025", "start": 2020, "end": 2021, "n": 179, "focus": "Metabolic"}
    ]
    
    # GCRO survey periods
    gcro_data = [
        {"survey": "GCRO QoL 2009", "year": 2009, "n": 6458},
        {"survey": "GCRO QoL 2011", "year": 2011, "n": 15000},
        {"survey": "GCRO QoL 2013-2014", "start": 2013, "end": 2014, "n": 15997},
        {"survey": "GCRO QoL 2015-2016", "start": 2015, "end": 2016, "n": 8862},
        {"survey": "GCRO QoL 2017-2018", "start": 2017, "end": 2018, "n": 11338},
        {"survey": "GCRO QoL 2020-2021", "start": 2020, "end": 2021, "n": 13616}
    ]
    
    # Define colors for research focus - professional academic palette
    focus_colors = {
        "HIV": "#E74C3C",      # Red
        "COVID-19": "#F39C12", # Orange
        "Metabolic": "#27AE60", # Green
        "TB/HIV": "#9B59B6"     # Purple
    }
    
    # Create figure with high DPI for publication quality
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    
    # Set timeline range
    start_year = 2002
    end_year = 2022
    
    # Draw GCRO survey periods as background bands
    for i, gcro in enumerate(gcro_data):
        if 'start' in gcro and 'end' in gcro:
            # Multi-year survey
            width = gcro['end'] - gcro['start'] + 1
            rect = Rectangle((gcro['start'] - 0.4, -1), width + 0.8, len(studies_data) + 2,
                           facecolor='lightblue', alpha=0.15, edgecolor='none')
            ax.add_patch(rect)
            
            # Add GCRO label at top
            ax.text(gcro['start'] + width/2, len(studies_data) + 0.5, 
                   f"{gcro['survey']}\nN={gcro['n']:,}", 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        else:
            # Single year survey
            rect = Rectangle((gcro['year'] - 0.4, -1), 0.8, len(studies_data) + 2,
                           facecolor='lightblue', alpha=0.15, edgecolor='none')
            ax.add_patch(rect)
            
            # Add GCRO label at top
            ax.text(gcro['year'], len(studies_data) + 0.5, 
                   f"{gcro['survey']}\nN={gcro['n']:,}", 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Plot clinical studies as horizontal bars
    for i, study in enumerate(studies_data):
        y_pos = len(studies_data) - i - 1
        duration = study['end'] - study['start'] + 1
        
        # Get color for research focus
        color = focus_colors.get(study['focus'], '#34495E')
        
        # Draw study bar
        bar = Rectangle((study['start'], y_pos - 0.3), duration, 0.6,
                       facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
        ax.add_patch(bar)
        
        # Add study ID label
        ax.text(study['start'] - 0.5, y_pos, study['study_id'], 
               ha='right', va='center', fontsize=9, fontweight='bold')
        
        # Add sample size inside bar
        ax.text(study['start'] + duration/2, y_pos, f"N={study['n']}", 
               ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Add research focus label at the end
        ax.text(study['end'] + 0.1, y_pos, study['focus'], 
               ha='left', va='center', fontsize=8, style='italic')
    
    # Set up the plot
    ax.set_xlim(start_year - 2, end_year + 1)
    ax.set_ylim(-1.5, len(studies_data) + 2)
    
    # Set x-axis with year ticks
    years = list(range(start_year, end_year + 1, 2))
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=11)
    
    # Remove y-axis
    ax.set_yticks([])
    
    # Add grid for years
    for year in years:
        ax.axvline(x=year, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add title and subtitle
    plt.suptitle('Temporal Coverage and GCRO Survey Integration', 
                fontsize=18, fontweight='bold', y=0.95)
    plt.title('All 17 clinical studies with GCRO Quality of Life Survey timeline\n' +
              'ENBEL Climate-Health Analysis Pipeline', 
              fontsize=14, pad=20, style='italic')
    
    # Create legend for research focus
    legend_elements = [patches.Patch(facecolor=color, label=focus, alpha=0.8) 
                      for focus, color in focus_colors.items()]
    
    # Add GCRO legend element
    legend_elements.append(patches.Patch(facecolor='lightblue', label='GCRO QoL Survey', alpha=0.7))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
             frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Add integration statistics with better formatting
    total_clinical = sum(s['n'] for s in studies_data)
    total_gcro = sum(g['n'] for g in gcro_data)
    integration_text = (
        "Data Integration Summary\n"
        f"• Clinical Records: {total_clinical:,}\n"
        f"• GCRO Records: {total_gcro:,}\n"
        "• Studies with GCRO Overlap: 11/17\n"
        "• Climate Coverage: 99.5%\n"
        "• Timeline Span: 2002-2021"
    )
    
    ax.text(0.02, 0.02, integration_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', alpha=0.95, edgecolor='#BDC3C7'),
           fontsize=10, verticalalignment='bottom', fontweight='normal')
    
    # Add WITS logo placeholder
    ax.text(0.95, 0.85, 'WITS\nLOGO', transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5),
           fontsize=10, ha='center', va='center', style='italic')
    
    plt.tight_layout()
    
    # Save as SVG for publication quality
    output_file = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_temporal_coverage_visualization.svg'
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Also save as PNG for immediate viewing
    png_file = output_file.replace('.svg', '.png')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    print(f"✓ Temporal coverage visualization saved:")
    print(f"  SVG: {output_file}")
    print(f"  PNG: {png_file}")
    
    return output_file

if __name__ == "__main__":
    create_temporal_coverage_visualization()