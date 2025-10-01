#!/usr/bin/env python3
"""
Create ENBEL Dataset Overview Slide with Exact Styling Match

This script recreates the ENBEL dataset overview slide with precise styling
to match the original image, including gradients, typography, spacing, and colors.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

def create_exact_styled_overview():
    """Create the exactly styled ENBEL dataset overview slide"""
    
    # Set up the figure with exact proportions
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Define exact color palette from original
    header_blue_dark = '#1B365D'  # Deep blue for gradient start
    header_blue_light = '#2E5A87'  # Lighter blue for gradient end
    clinical_blue = '#4A90A4'  # Clinical section blue
    step1_blue = '#5BA3C4'  # Step 1 box blue
    step2_orange = '#F0A500'  # Step 2 box orange
    climate_orange = '#E87D00'  # Climate section orange
    bottom_gray = '#F5F5F5'  # Bottom statistics bar
    white = '#FFFFFF'
    text_dark = '#2C3E50'
    
    # Create header with deep blue gradient
    header_gradient = np.linspace(0, 1, 256).reshape(1, -1)
    header_colors = [(0, header_blue_dark), (1, header_blue_light)]
    header_cmap = LinearSegmentedColormap.from_list('header', header_colors)
    
    # Header background with rounded corners
    header_box = FancyBboxPatch((0.2, 7.2), 15.6, 1.6,
                               boxstyle="round,pad=0.05",
                               facecolor=header_blue_dark,
                               edgecolor='none',
                               alpha=1.0)
    ax.add_patch(header_box)
    
    # Add gradient effect to header
    gradient_rect = Rectangle((0.2, 7.2), 15.6, 1.6)
    ax.imshow(header_gradient, extent=[0.2, 15.8, 7.2, 8.8], 
              aspect='auto', cmap=header_cmap, alpha=0.8)
    
    # WITS logo placeholder (positioned in top-right)
    logo_box = FancyBboxPatch((14.5, 7.9), 1.2, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=white,
                             edgecolor='none',
                             alpha=0.9)
    ax.add_patch(logo_box)
    ax.text(15.1, 8.2, 'WITS', fontsize=14, fontweight='bold',
            ha='center', va='center', color=header_blue_dark)
    
    # Header title - bold, large white text
    title_text = ax.text(8, 8.3, 'ENBEL DATASET OVERVIEW', 
                        fontsize=32, fontweight='bold',
                        ha='center', va='center', color=white)
    title_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)])
    
    # Header subtitle
    subtitle_text = ax.text(8, 7.7, 'Comprehensive Climate-Health Analysis Framework',
                           fontsize=18, fontweight='normal',
                           ha='center', va='center', color=white, alpha=0.95)
    
    # Column 1: Clinical Studies (Blue section)
    clinical_box = FancyBboxPatch((0.5, 4.5), 4.8, 2.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=clinical_blue,
                                 edgecolor='none',
                                 alpha=1.0)
    ax.add_patch(clinical_box)
    
    # Clinical Studies content
    ax.text(2.9, 6.7, 'CLINICAL STUDIES', fontsize=16, fontweight='bold',
            ha='center', va='top', color=white)
    
    # Large number styling
    ax.text(2.9, 6.2, '11,398', fontsize=36, fontweight='bold',
            ha='center', va='center', color=white)
    ax.text(2.9, 5.9, 'participants', fontsize=14, fontweight='normal',
            ha='center', va='center', color=white, alpha=0.9)
    
    # Bullet points with proper indentation
    clinical_bullets = [
        '• 17 HIV clinical trials',
        '• 15 years follow-up',
        '• Individual-level data',
        '• Biomarker measurements',
        '• Johannesburg-based'
    ]
    
    y_pos = 5.5
    for bullet in clinical_bullets:
        ax.text(0.8, y_pos, bullet, fontsize=11, fontweight='normal',
                ha='left', va='center', color=white)
        y_pos -= 0.2
    
    # Column 2: Integration Framework (Green/Teal section)
    integration_y = 4.5
    
    # Step 1 box (Blue)
    step1_box = FancyBboxPatch((5.8, 6.0), 4.4, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=step1_blue,
                              edgecolor='none',
                              alpha=1.0)
    ax.add_patch(step1_box)
    
    ax.text(8.0, 6.4, 'STEP 1: CLINICAL DATA INTEGRATION', 
            fontsize=12, fontweight='bold',
            ha='center', va='center', color=white)
    
    # Step 2 box (Orange)
    step2_box = FancyBboxPatch((5.8, 5.0), 4.4, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=step2_orange,
                              edgecolor='none',
                              alpha=1.0)
    ax.add_patch(step2_box)
    
    ax.text(8.0, 5.4, 'STEP 2: SOCIOECONOMIC LINKAGE', 
            fontsize=12, fontweight='bold',
            ha='center', va='center', color=text_dark)
    
    # GCRO Community Survey highlight
    ax.text(8.0, 4.7, 'GCRO Community Survey Integration', 
            fontsize=14, fontweight='bold',
            ha='center', va='center', color=text_dark)
    
    # Method and Variables
    method_text = [
        'Method: Spatial joining by ward boundaries',
        'Variables: Income, education, dwelling type,',
        'infrastructure access, vulnerability indices'
    ]
    
    y_pos = 4.3
    for line in method_text:
        ax.text(8.0, y_pos, line, fontsize=10, fontweight='normal',
                ha='center', va='center', color=text_dark)
        y_pos -= 0.15
    
    # Column 3: Climate Data (Orange section)
    climate_box = FancyBboxPatch((10.7, 4.5), 4.8, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor=climate_orange,
                                edgecolor='none',
                                alpha=1.0)
    ax.add_patch(climate_box)
    
    # Climate Data content
    ax.text(13.1, 6.7, 'CLIMATE DATA', fontsize=16, fontweight='bold',
            ha='center', va='top', color=white)
    
    # Data Sources prominently displayed
    ax.text(13.1, 6.3, 'ERA5 Reanalysis', fontsize=18, fontweight='bold',
            ha='center', va='center', color=white)
    
    # Categories with bullet points
    climate_bullets = [
        '• Temperature (16 variables)',
        '• Precipitation patterns',
        '• Heat stress indices',
        '• Temporal lags (7-30 days)',
        '• 2002-2021 coverage'
    ]
    
    y_pos = 5.8
    for bullet in climate_bullets:
        ax.text(11.0, y_pos, bullet, fontsize=11, fontweight='normal',
                ha='left', va='center', color=white)
        y_pos -= 0.2
    
    # Bottom Statistics Bar
    stats_box = FancyBboxPatch((0.5, 0.5), 15, 1.2,
                              boxstyle="round,pad=0.05",
                              facecolor=bottom_gray,
                              edgecolor='#CCCCCC',
                              linewidth=1,
                              alpha=1.0)
    ax.add_patch(stats_box)
    
    # Five key metrics evenly spaced
    stats_data = [
        ('11,398', 'Participants'),
        ('17', 'Studies'),
        ('30+', 'Biomarkers'),
        ('19 years', 'Coverage'),
        ('Individual', 'Resolution')
    ]
    
    x_positions = [2.5, 5.0, 7.5, 10.0, 12.5]
    
    for i, (number, label) in enumerate(stats_data):
        # Large numbers
        ax.text(x_positions[i], 1.3, number, fontsize=24, fontweight='bold',
                ha='center', va='center', color=text_dark)
        # Smaller descriptive text
        ax.text(x_positions[i], 0.9, label, fontsize=12, fontweight='normal',
                ha='center', va='center', color=text_dark, alpha=0.8)
    
    # Add subtle separator lines between stats
    for i in range(len(x_positions) - 1):
        sep_x = (x_positions[i] + x_positions[i + 1]) / 2
        ax.plot([sep_x, sep_x], [0.7, 1.5], color='#CCCCCC', linewidth=1, alpha=0.5)
    
    # Final styling adjustments
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig

def save_visualizations():
    """Save the visualization in multiple formats"""
    
    fig = create_exact_styled_overview()
    
    # Save as SVG for Figma editing
    svg_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_styled_overview_exact_match.svg'
    fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', 
                metadata={'Creator': 'ENBEL Climate-Health Analysis'})
    
    # Save as high-resolution PNG backup
    png_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_styled_overview_exact_match.png'
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.close(fig)
    
    print("✅ ENBEL Styled Dataset Overview created successfully!")
    print(f"📁 SVG saved: {svg_path}")
    print(f"📁 PNG saved: {png_path}")
    print("\n🎨 Design Features:")
    print("   • Deep blue gradient header with WITS logo")
    print("   • Three-column layout with distinct color coding")
    print("   • Rounded corners and proper typography hierarchy")
    print("   • Clinical (blue), Integration (blue/orange), Climate (orange)")
    print("   • Bottom statistics bar with clean spacing")
    print("   • SVG format optimized for Figma editing")

if __name__ == "__main__":
    save_visualizations()