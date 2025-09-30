#!/usr/bin/env python3
"""
Create properly spaced CMIP6 attribution slide with correct BP relationship
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
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
    'white': '#FFFFFF',
    'lightgray': '#F5F5F5'
}

def create_fixed_spacing_cmip6():
    """Create CMIP6 slide with perfect spacing and correct BP relationship"""
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(colors['white'])
    
    # Title with proper spacing
    fig.text(0.5, 0.95, 'Complex Climate-Health Attribution Framework', 
             fontsize=22, weight='bold', ha='center', color=colors['blue'])
    
    fig.text(0.5, 0.91, 
            'Mixed health effects: Cardiovascular benefits (BP↓) vs Metabolic risks (Glucose↑)',
            fontsize=14, ha='center', style='italic', color=colors['gray'])
    
    # Attribution pathway boxes with proper spacing
    box_width = 0.22
    box_height = 0.18
    box_y = 0.7
    box_spacing = 0.09
    
    boxes = [
        {'title': 'VALIDATED\nFINDINGS', 'x': 0.08, 'color': colors['green'],
         'content': '• BP: -2.9 mmHg/°C ✓\n• Glucose: +8.2 mg/dL/°C ✓\n• Lag-21 cardiovascular\n• Immediate metabolic'},
        {'title': 'ATTRIBUTION\nPATHWAY', 'x': 0.39, 'color': colors['orange'], 
         'content': '• DLNM dose-response\n• Population scaling\n• Mixed health effects\n• Risk-benefit analysis'},
        {'title': 'CMIP6\nSCENARIOS', 'x': 0.7, 'color': colors['blue'],
         'content': '• SSP1-2.6 to SSP5-8.5\n• Temperature projections\n• Health impact scaling\n• Policy implications'}
    ]
    
    for i, box in enumerate(boxes):
        ax = fig.add_axes([box['x'], box_y, box_width, box_height])
        
        # Box background
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                                   facecolor=box['color'], alpha=0.15, 
                                   edgecolor=box['color'], linewidth=2))
        
        # Title background
        ax.add_patch(FancyBboxPatch((0, 0.75), 1, 0.25, boxstyle="round,pad=0.02",
                                   facecolor=box['color'], alpha=0.8))
        
        # Title text
        ax.text(0.5, 0.875, box['title'], ha='center', va='center',
               fontsize=11, weight='bold', color='white')
        
        # Content text with proper spacing
        ax.text(0.05, 0.65, box['content'], ha='left', va='top',
               fontsize=10, color='black', linespacing=1.2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add arrows between boxes with proper spacing
        if i < len(boxes) - 1:
            arrow_start_x = box['x'] + box_width + 0.01
            arrow_end_x = boxes[i+1]['x'] - 0.01
            arrow_y = box_y + box_height/2
            
            arrow = patches.FancyArrowPatch((arrow_start_x, arrow_y), 
                                          (arrow_end_x, arrow_y),
                                          arrowstyle='->', mutation_scale=20,
                                          color=colors['gray'], linewidth=3)
            fig.add_artist(arrow)
    
    # Scenarios table with proper spacing
    table_y_start = 0.25
    table_height = 0.35
    
    # Table title
    fig.text(0.5, 0.58, 'CMIP6 Climate Scenarios & Health Impacts', 
             ha='center', fontsize=16, weight='bold', color=colors['blue'])
    
    # Create table data with CORRECTED BP signs
    scenarios_data = [
        ['Scenario', 'Temp Change', 'BP Impact', 'Glucose Impact', 'Net Health Effect'],
        ['SSP1-2.6', '+1.5°C', '-4.4 mmHg', '+12.3 mg/dL', 'Mixed (CV↓/Met↑)'],
        ['SSP2-4.5', '+2.5°C', '-7.3 mmHg', '+20.5 mg/dL', 'Mixed (CV↓/Met↑)'],
        ['SSP3-7.0', '+4.0°C', '-11.6 mmHg', '+32.8 mg/dL', 'Mixed (CV↓/Met↑)'],
        ['SSP5-8.5', '+5.0°C', '-14.5 mmHg', '+41.0 mg/dL', 'Mixed (CV↓/Met↑)']
    ]
    
    # Table with proper spacing
    table_ax = fig.add_axes([0.1, table_y_start, 0.8, table_height])
    
    row_height = 0.16
    col_widths = [0.18, 0.16, 0.18, 0.18, 0.3]
    
    # Header row
    y_pos = 0.85
    x_pos = 0
    for j, (header, width) in enumerate(zip(scenarios_data[0], col_widths)):
        rect = patches.Rectangle((x_pos, y_pos), width, row_height, 
                               facecolor=colors['blue'], alpha=0.8, 
                               edgecolor='white', linewidth=1)
        table_ax.add_patch(rect)
        table_ax.text(x_pos + width/2, y_pos + row_height/2, header, 
                     ha='center', va='center', fontsize=11, 
                     weight='bold', color='white')
        x_pos += width
    
    # Data rows with proper spacing
    row_colors = [colors['lightgray'], 'white', colors['lightgray'], 'white']
    
    for i, (row, row_color) in enumerate(zip(scenarios_data[1:], row_colors)):
        y_pos = 0.85 - (i+1) * row_height
        x_pos = 0
        
        for j, (cell, width) in enumerate(zip(row, col_widths)):
            rect = patches.Rectangle((x_pos, y_pos), width, row_height,
                                   facecolor=row_color, edgecolor=colors['gray'], 
                                   linewidth=0.5, alpha=0.7)
            table_ax.add_patch(rect)
            
            # Color code the health impacts
            text_color = 'black'
            if 'mmHg' in cell and '-' in cell:  # Negative BP (beneficial)
                text_color = colors['blue']
            elif 'mg/dL' in cell and '+' in cell:  # Positive glucose (harmful)
                text_color = colors['red']
            
            table_ax.text(x_pos + width/2, y_pos + row_height/2, cell,
                         ha='center', va='center', fontsize=10, 
                         color=text_color, weight='bold' if text_color != 'black' else 'normal')
            x_pos += width
    
    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(0, 1)
    table_ax.axis('off')
    
    # Key findings box with proper spacing
    key_findings = """KEY RELATIONSHIPS (Johannesburg Metro 5.6M):
• Blood Pressure: DECREASES 2.9 mmHg per °C (cardiovascular benefit)
• Glucose: INCREASES 8.2 mg/dL per °C (metabolic risk)
• Mixed health effects require nuanced policy responses"""
    
    fig.text(0.03, 0.18, key_findings, ha='left', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.6", facecolor=colors['lightblue'], 
                      edgecolor=colors['blue'], alpha=0.9))
    
    # References with proper positioning
    references = """References:
1. ENBEL Study (2024): Climate-health relationships, N=6,180
2. IPCC AR6 (2021): CMIP6 temperature projections  
3. Stats SA (2022): Johannesburg Metro population"""
    
    fig.text(0.97, 0.18, references, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['lightgray'], 
                      edgecolor=colors['gray'], alpha=0.8), style='italic')
    
    # Bottom note with proper spacing
    fig.text(0.5, 0.02, 
            'CV = Cardiovascular, Met = Metabolic | Mixed effects require integrated health system responses',
            fontsize=11, ha='center', weight='bold', color=colors['gray'])
    
    plt.tight_layout()
    
    # Save with proper spacing
    plt.savefig('enbel_cmip6_perfect_spacing.svg', format='svg', 
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    plt.savefig('enbel_cmip6_perfect_spacing.png', format='png',
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    
    print("CMIP6 slide with perfect spacing and correct BP relationship created")
    plt.show()

if __name__ == "__main__":
    create_fixed_spacing_cmip6()