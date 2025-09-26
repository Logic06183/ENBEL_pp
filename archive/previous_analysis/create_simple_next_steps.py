#!/usr/bin/env python3
"""
Create a simplified but effective next steps framework slide
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
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

def create_simple_next_steps():
    """Create simplified next steps framework slide"""
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(colors['white'])
    
    # Title
    fig.suptitle('From Discovery to Impact: ENBEL Climate-Health Translation Framework', 
                fontsize=20, weight='bold', y=0.95, color=colors['blue'])
    
    fig.text(0.5, 0.91, 
            'Validated findings → CMIP6 scenarios → Policy-ready interventions',
            fontsize=14, ha='center', style='italic', color=colors['gray'])
    
    # Create three main pillars
    pillars = [
        {
            'title': 'SCIENTIFIC FOUNDATION',
            'subtitle': 'Validated Discoveries',
            'content': ['• Lag-21 cardiovascular effects (p<0.001)', 
                       '• Immediate glucose response (p<0.001)',
                       '• Socioeconomic vulnerability interactions',
                       '• DLNM dose-response relationships',
                       '• Non-linear temperature thresholds'],
            'color': colors['blue'],
            'position': [0.05, 0.45, 0.28, 0.4]
        },
        {
            'title': 'CLIMATE INTEGRATION', 
            'subtitle': 'CMIP6 Scenario Planning',
            'content': ['• SSP1-2.6: +1.5°C → 540K at risk',
                       '• SSP2-4.5: +2.5°C → 900K at risk', 
                       '• SSP3-7.0: +4.0°C → 1.44M at risk',
                       '• SSP5-8.5: +5.0°C → 1.8M at risk',
                       '• Regional downscaling for Johannesburg'],
            'color': colors['green'],
            'position': [0.36, 0.45, 0.28, 0.4]
        },
        {
            'title': 'POLICY TRANSLATION',
            'subtitle': 'Implementation Ready',
            'content': ['• Early warning system protocols',
                       '• 21-day cardiovascular monitoring',
                       '• Vulnerable population targeting',
                       '• Healthcare system preparedness',
                       '• Economic impact assessments'],
            'color': colors['red'],
            'position': [0.67, 0.45, 0.28, 0.4]
        }
    ]
    
    # Create pillar boxes
    for pillar in pillars:
        ax = fig.add_axes(pillar['position'])
        
        # Main box
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03",
                                   facecolor=pillar['color'], alpha=0.1, 
                                   edgecolor=pillar['color'], linewidth=2))
        
        # Title box
        ax.add_patch(FancyBboxPatch((0, 0.85), 1, 0.15, boxstyle="round,pad=0.02",
                                   facecolor=pillar['color'], edgecolor='none'))
        
        # Title text
        ax.text(0.5, 0.925, pillar['title'], ha='center', va='center',
               fontsize=12, weight='bold', color='white')
        
        # Subtitle
        ax.text(0.5, 0.8, pillar['subtitle'], ha='center', va='center',
               fontsize=10, weight='bold', color=pillar['color'])
        
        # Content
        content_text = '\n'.join(pillar['content'])
        ax.text(0.05, 0.75, content_text, ha='left', va='top',
               fontsize=10, color='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Implementation timeline
    fig.text(0.5, 0.42, 'IMPLEMENTATION TIMELINE', ha='center', va='center',
             fontsize=16, weight='bold', color=colors['blue'])
    
    # Timeline boxes
    timeline_items = [
        ('IMMEDIATE\n(0-6 months)', 'Tool Development', colors['blue']),
        ('SHORT TERM\n(6-12 months)', 'Pilot Testing', colors['green']),
        ('MEDIUM TERM\n(1-2 years)', 'System Integration', colors['orange']), 
        ('LONG TERM\n(2-5 years)', 'Full Implementation', colors['red'])
    ]
    
    timeline_y = 0.25
    box_width = 0.2
    
    for i, (phase, desc, color) in enumerate(timeline_items):
        x_pos = 0.1 + i * 0.2
        
        # Phase box
        ax = fig.add_axes([x_pos, timeline_y, box_width-0.02, 0.12])
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=0.2, 
                                   edgecolor=color, linewidth=2))
        
        ax.text(0.5, 0.7, phase, ha='center', va='center',
               fontsize=10, weight='bold', color=color)
        ax.text(0.5, 0.3, desc, ha='center', va='center',
               fontsize=9, color='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Arrow to next phase
        if i < len(timeline_items) - 1:
            arrow_start_x = x_pos + box_width - 0.01
            arrow_end_x = x_pos + 0.2 - 0.01
            arrow_y = timeline_y + 0.06
            
            arrow = patches.FancyArrowPatch((arrow_start_x, arrow_y), 
                                          (arrow_end_x, arrow_y),
                                          arrowstyle='->', mutation_scale=15,
                                          color=colors['gray'], linewidth=2)
            fig.add_artist(arrow)
    
    # Key deliverables
    deliverables_text = '''IMMEDIATE DELIVERABLES:
• CMIP6-health calculator for Johannesburg Metro
• Clinical decision support for 21-day monitoring
• Early warning system integration protocols  
• Economic assessment for adaptation funding'''
    
    fig.text(0.02, 0.2, deliverables_text, ha='left', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['lightblue'], 
                      edgecolor=colors['blue'], alpha=0.8))
    
    # Success metrics
    metrics_text = '''SUCCESS METRICS:
• >80% early warning accuracy
• >90% vulnerable population coverage
• 3:1 cost-benefit ratio achieved
• Policy adoption in 3+ metro areas'''
    
    fig.text(0.98, 0.2, metrics_text, ha='right', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', 
                      edgecolor=colors['red'], alpha=0.8))
    
    # Partnership footer
    fig.text(0.5, 0.02, 
            'PARTNERSHIP NETWORK: City of Johannesburg • NICD • WITS • CSIR • WHO Africa Regional Office',
            fontsize=11, ha='center', weight='bold', color=colors['gray'])
    
    plt.savefig('enbel_implementation_framework.svg', format='svg', 
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    plt.savefig('enbel_implementation_framework.png', format='png',
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    
    print("Implementation framework slide saved")
    plt.show()

if __name__ == "__main__":
    create_simple_next_steps()