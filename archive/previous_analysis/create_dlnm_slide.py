#!/usr/bin/env python3
"""
Create a single presentation slide combining all DLNM plots
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle
import os

# ENBEL color scheme
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00', 
    'green': '#2CA02C',
    'red': '#DC2626',
    'purple': '#9467BD',
    'gray': '#8C8C8C',
    'lightblue': '#E6F0FA',
    'white': '#FFFFFF'
}

def create_dlnm_slide():
    """Create a comprehensive DLNM slide with all 6 plots"""
    
    # Create figure with presentation dimensions (16:9 aspect ratio)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(colors['white'])
    
    # Main title
    fig.suptitle('Distributed Lag Non-Linear Models (DLNM): Climate-Health Response Analysis', 
                fontsize=24, weight='bold', y=0.96, color=colors['blue'])
    
    # Subtitle
    fig.text(0.5, 0.92, 
            'Non-linear and delayed temperature effects on cardiovascular and metabolic biomarkers',
            fontsize=16, ha='center', style='italic', color=colors['gray'])
    
    # Define plot positions for 2x3 grid
    positions = [
        [0.08, 0.55, 0.28, 0.32],   # BP 3D (top left)
        [0.38, 0.55, 0.28, 0.32],   # BP overall (top center)
        [0.68, 0.55, 0.28, 0.32],   # BP slices (top right)
        [0.08, 0.15, 0.28, 0.32],   # Glucose 3D (bottom left)
        [0.38, 0.15, 0.28, 0.32],   # Glucose overall (bottom center)
        [0.68, 0.15, 0.28, 0.32],   # Glucose slices (bottom right)
    ]
    
    # Plot files and titles
    plot_info = [
        ('enbel_dlnm_bp_3d.png', 'A. Blood Pressure: 3D Response Surface'),
        ('enbel_dlnm_bp_overall.png', 'B. BP: Cumulative Effect (0-30 days)'),
        ('enbel_dlnm_bp_slices.png', 'C. BP: Response at Specific Lags'),
        ('enbel_dlnm_glucose_3d.png', 'D. Glucose: 3D Response Surface'),
        ('enbel_dlnm_glucose_overall.png', 'E. Glucose: Cumulative Effect (0-10 days)'),
        ('enbel_dlnm_glucose_slices.png', 'F. Glucose: Response at Specific Lags'),
    ]
    
    # Add each plot
    for i, ((filename, title), pos) in enumerate(zip(plot_info, positions)):
        if os.path.exists(filename):
            # Create subplot
            ax = fig.add_axes(pos)
            
            try:
                # Load and display image
                img = mpimg.imread(filename)
                ax.imshow(img, aspect='equal')
                ax.axis('off')
                
                # Add title
                ax.text(0.5, 1.08, title, transform=ax.transAxes, 
                       fontsize=13, weight='bold', ha='center', color=colors['blue'])
                
                # Add border
                rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, edgecolor=colors['gray'], 
                               linewidth=1, alpha=0.5)
                ax.add_patch(rect)
                
            except Exception as e:
                # Fallback if image can't be loaded
                ax.text(0.5, 0.5, f'Plot {i+1}\n{title}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color=colors['gray'])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        else:
            print(f"Warning: {filename} not found")
    
    # Add section labels
    fig.text(0.01, 0.71, 'BLOOD PRESSURE', fontsize=16, weight='bold', 
             color=colors['red'], rotation=90, va='center')
    fig.text(0.01, 0.31, 'GLUCOSE', fontsize=16, weight='bold', 
             color=colors['orange'], rotation=90, va='center')
    
    # Add key findings box
    findings_text = '''Key Validated Findings:
• Blood Pressure: Delayed cardiovascular response peaking at lag 21 days (p<0.001)
• Glucose: Immediate metabolic response at lag 0-3 days (p<0.001)  
• Non-linear temperature thresholds identified for both outcomes
• Socioeconomic vulnerability modifies climate-health associations'''
    
    fig.text(0.5, 0.08, findings_text,
            fontsize=12, ha='center', va='top', color=colors['blue'],
            bbox=dict(boxstyle="round,pad=0.8", facecolor=colors['lightblue'], 
                     edgecolor=colors['blue'], alpha=0.8))
    
    # Add methodology note
    fig.text(0.99, 0.02, 
            'DLNM Analysis: Natural splines, 30-day cardiovascular lag, 10-day metabolic lag. N=2000 BP, N=1500 glucose.',
            fontsize=10, ha='right', style='italic', color=colors['gray'])
    
    # Save as SVG
    plt.savefig('enbel_dlnm_comprehensive_slide.svg', format='svg', 
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    plt.savefig('enbel_dlnm_comprehensive_slide.png', format='png', 
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    
    print("DLNM comprehensive slide saved as 'enbel_dlnm_comprehensive_slide.svg' and '.png'")
    plt.show()

if __name__ == "__main__":
    create_dlnm_slide()