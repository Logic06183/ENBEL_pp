#!/usr/bin/env python3
"""
Create Scientific Publication-Quality Temporal Coverage Visualization
==================================================================

Creates a high-quality SVG visualization showing the temporal coverage of 
clinical studies and GCRO surveys for publication in peer-reviewed journals.

Features:
- Professional grayscale base with selective color accents
- Scientific typography and layout
- Precise sample size annotations
- Publication-ready formatting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from datetime import datetime, date
import numpy as np
import pandas as pd

# Set publication-quality parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#2C3E50',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#ECF0F1',
    'grid.linewidth': 0.3,
    'text.color': '#2C3E50',
    'axes.labelcolor': '#2C3E50',
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50'
})

def create_scientific_temporal_coverage():
    """Create publication-quality temporal coverage visualization."""
    
    # Clinical studies data (corrected and validated)
    clinical_studies = [
        {'name': 'JHB_JHSPH_005', 'start': '2002-01-01', 'end': '2009-12-31', 'n': 524, 'type': 'TB/HIV'},
        {'name': 'JHB_ACTG_017', 'start': '2003-01-01', 'end': '2004-12-31', 'n': 20, 'type': 'HIV'},
        {'name': 'JHB_ACTG_019', 'start': '2005-01-01', 'end': '2007-12-31', 'n': 100, 'type': 'HIV'},
        {'name': 'JHB_ACTG_015', 'start': '2005-01-01', 'end': '2008-12-31', 'n': 87, 'type': 'HIV'},
        {'name': 'JHB_ACTG_016', 'start': '2007-01-01', 'end': '2009-12-31', 'n': 76, 'type': 'HIV'},
        {'name': 'JHB_DPHRU_013', 'start': '2011-01-01', 'end': '2013-12-31', 'n': 247, 'type': 'HIV/TB'},
        {'name': 'JHB_ACTG_018', 'start': '2011-01-01', 'end': '2012-12-31', 'n': 240, 'type': 'HIV'},
        {'name': 'JHB_WRHI_001', 'start': '2012-01-01', 'end': '2014-12-31', 'n': 1067, 'type': 'HIV'},
        {'name': 'JHB_Aurum_009', 'start': '2013-01-01', 'end': '2015-12-31', 'n': 2551, 'type': 'TB/HIV'},
        {'name': 'JHB_SCHARP_004', 'start': '2015-01-01', 'end': '2016-12-31', 'n': 2, 'type': 'HIV'},
        {'name': 'JHB_WRHI_003', 'start': '2016-01-01', 'end': '2017-12-31', 'n': 217, 'type': 'HIV'},
        {'name': 'JHB_SCHARP_006', 'start': '2017-01-01', 'end': '2017-12-31', 'n': 162, 'type': 'HIV'},
        {'name': 'JHB_Ezin_002', 'start': '2017-01-01', 'end': '2018-12-31', 'n': 1053, 'type': 'Metabolic'},
        {'name': 'JHB_DPHRU_053', 'start': '2017-01-01', 'end': '2018-12-31', 'n': 998, 'type': 'HIV/TB'},
        {'name': 'JHB_VIDA_008', 'start': '2020-01-01', 'end': '2021-12-31', 'n': 550, 'type': 'COVID-19'},
        {'name': 'JHB_VIDA_007', 'start': '2020-01-01', 'end': '2020-12-31', 'n': 2129, 'type': 'COVID-19'},
        {'name': 'JHB_EZIN_025', 'start': '2020-01-01', 'end': '2021-12-31', 'n': 179, 'type': 'Metabolic'}
    ]
    
    # GCRO survey data
    gcro_surveys = [
        {'name': 'GCRO QoL 2011', 'start': '2011-01-01', 'end': '2011-12-31', 'n': 15000},
        {'name': 'GCRO QoL 2014', 'start': '2014-01-01', 'end': '2014-12-31', 'n': 15000},
        {'name': 'GCRO QoL 2018', 'start': '2018-01-01', 'end': '2018-12-31', 'n': 15000},
        {'name': 'GCRO QoL 2021', 'start': '2021-01-01', 'end': '2021-12-31', 'n': 13616}
    ]
    
    # Color scheme for study types
    study_colors = {
        'HIV': '#3498DB',       # Deep blue
        'COVID-19': '#E67E22',  # Warm orange
        'Metabolic': '#27AE60', # Forest green
        'TB/HIV': '#8E44AD',    # Purple
        'HIV/TB': '#8E44AD'     # Purple (same as TB/HIV)
    }
    
    # Create figure with publication dimensions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Temporal Coverage and Socioeconomic Survey Integration', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Add subtitle
    fig.text(0.5, 0.92, 'Clinical Studies (N=10,202) and GCRO Quality of Life Surveys (N=58,616)',
             ha='center', fontsize=11, style='italic')
    
    # Convert dates to datetime objects
    start_date = datetime(2002, 1, 1)
    end_date = datetime(2022, 1, 1)
    
    # Plot clinical studies
    ax1.set_title('A. Clinical Studies Timeline', fontsize=12, fontweight='bold', pad=20)
    
    for i, study in enumerate(clinical_studies):
        start = datetime.strptime(study['start'], '%Y-%m-%d')
        end = datetime.strptime(study['end'], '%Y-%m-%d')
        duration = (end - start).days
        
        # Get color for study type
        color = study_colors[study['type']]
        
        # Create timeline bar
        bar = ax1.barh(i, duration, left=start, height=0.6, 
                      color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add study name and sample size
        ax1.text(start + pd.Timedelta(days=duration/2), i, 
                f"{study['name']}\n(N={study['n']:,})", 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Customize clinical studies axis
    ax1.set_ylim(-0.5, len(clinical_studies) - 0.5)
    ax1.set_xlim(start_date, end_date)
    ax1.set_yticks(range(len(clinical_studies)))
    ax1.set_yticklabels([s['type'] for s in clinical_studies], fontsize=9)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.set_xlabel('')
    
    # Format x-axis for clinical studies
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Plot GCRO surveys
    ax2.set_title('B. GCRO Quality of Life Surveys', fontsize=12, fontweight='bold', pad=20)
    
    for i, survey in enumerate(gcro_surveys):
        start = datetime.strptime(survey['start'], '%Y-%m-%d')
        end = datetime.strptime(survey['end'], '%Y-%m-%d')
        
        # Create survey blocks
        rect = patches.Rectangle((start, i-0.3), end-start, 0.6,
                               facecolor='#EBF5FB', edgecolor='#3498DB', 
                               linewidth=2, alpha=0.9)
        ax2.add_patch(rect)
        
        # Add survey name and sample size
        ax2.text(start + (end-start)/2, i, 
                f"{survey['name']}\n(N={survey['n']:,})", 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Customize GCRO axis
    ax2.set_ylim(-0.5, len(gcro_surveys) - 0.5)
    ax2.set_xlim(start_date, end_date)
    ax2.set_yticks(range(len(gcro_surveys)))
    ax2.set_yticklabels(['Household\nSurveys'] * len(gcro_surveys), fontsize=9)
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    
    # Format x-axis for GCRO surveys
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Create legend for study types
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='white')
                      for color in study_colors.values()]
    legend_labels = list(study_colors.keys())
    
    ax1.legend(legend_elements, legend_labels, loc='upper right', 
              title='Study Types', frameon=True, fancybox=False, 
              edgecolor='#2C3E50', fontsize=9)
    
    # Add statistical summary box
    summary_text = (
        "Dataset Summary:\n"
        "• Clinical Studies: 17 studies\n"
        "• Unique Patients: 10,202\n"
        "• Total Records: 11,398\n"
        "• GCRO Surveys: 4 waves\n"
        "• Household Records: 58,616\n"
        "• Temporal Span: 2002-2021"
    )
    
    ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9FA', 
                     edgecolor='#2C3E50', linewidth=0.5),
            verticalalignment='top', fontsize=8, fontfamily='monospace')
    
    # Add methodology note
    fig.text(0.5, 0.02, 
             'Data Sources: ENBEL Clinical Dataset (validated) and GCRO Quality of Life Survey Series (2011-2021)',
             ha='center', fontsize=9, style='italic', color='#34495E')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08)
    
    return fig

def main():
    """Generate and save the scientific temporal coverage visualization."""
    print("Creating scientific publication-quality temporal coverage visualization...")
    
    # Create the visualization
    fig = create_scientific_temporal_coverage()
    
    # Save as high-quality SVG
    output_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_scientific_temporal_coverage.svg'
    fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # Also save as high-quality PNG for presentations
    png_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_scientific_temporal_coverage.png'
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"✅ Scientific temporal coverage visualization saved:")
    print(f"   SVG: {output_path}")
    print(f"   PNG: {png_path}")
    
    # Display summary statistics
    print("\n📊 Validated Dataset Summary:")
    print("   Clinical Studies: 17 studies")
    print("   Unique Patients: 10,202") 
    print("   Total Clinical Records: 11,398")
    print("   GCRO Survey Waves: 4")
    print("   Total Household Records: 58,616")
    print("   Temporal Coverage: 2002-2021 (20 years)")
    
    plt.show()

if __name__ == "__main__":
    main()