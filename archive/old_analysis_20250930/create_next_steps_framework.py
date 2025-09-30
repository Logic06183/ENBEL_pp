#!/usr/bin/env python3
"""
Create a comprehensive next steps framework slide
Linking current discoveries to future research and policy implementation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
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

def create_next_steps_framework():
    """Create comprehensive next steps framework slide"""
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(colors['white'])
    
    # Title
    fig.suptitle('Research Translation Framework: From Discovery to Implementation', 
                fontsize=22, weight='bold', y=0.96, color=colors['blue'])
    
    fig.text(0.5, 0.92, 
            'Validated climate-health relationships → CMIP6 scenarios → Policy-ready interventions',
            fontsize=14, ha='center', style='italic', color=colors['gray'])
    
    # Create circular framework with 6 interconnected components
    center_x, center_y = 0.5, 0.55
    radius = 0.25
    
    # Central hub
    central_circle = Circle((center_x, center_y), 0.08, facecolor=colors['blue'], 
                           edgecolor='white', linewidth=3)
    fig.add_artist(central_circle)
    fig.text(center_x, center_y, 'ENBEL\nFRAMEWORK', ha='center', va='center', 
             fontsize=11, weight='bold', color='white')
    
    # Six framework components
    components = [
        {
            'title': '1. VALIDATED\nDISCOVERIES',
            'content': '• Lag-21 cardiovascular\n• Immediate glucose\n• SES vulnerability\n• Non-linear thresholds',
            'color': colors['blue'],
            'angle': 0
        },
        {
            'title': '2. MECHANISTIC\nUNDERSTANDING', 
            'content': '• DLNM methodology\n• XAI interpretation\n• Dose-response curves\n• Confidence intervals',
            'color': colors['purple'],
            'angle': 60
        },
        {
            'title': '3. CLIMATE\nPROJECTIONS',
            'content': '• CMIP6 integration\n• SSP scenarios\n• Regional downscaling\n• Uncertainty bands',
            'color': colors['green'],
            'angle': 120
        },
        {
            'title': '4. POPULATION\nSCALING',
            'content': '• Johannesburg 5.6M\n• Vulnerable groups\n• Healthcare capacity\n• Economic burden',
            'color': colors['orange'],
            'angle': 180
        },
        {
            'title': '5. INTERVENTION\nDESIGN',
            'content': '• Early warning systems\n• Clinical protocols\n• Community outreach\n• Infrastructure adaptation',
            'color': colors['red'],
            'angle': 240
        },
        {
            'title': '6. POLICY\nIMPLEMENTATION',
            'content': '• Evidence packages\n• Stakeholder engagement\n• Monitoring systems\n• Evaluation frameworks',
            'color': colors['gray'],
            'angle': 300
        }
    ]
    
    # Position and create component boxes
    component_positions = []
    for i, comp in enumerate(components):
        angle_rad = np.radians(comp['angle'])
        x = center_x + radius * np.cos(angle_rad)
        y = center_y + radius * np.sin(angle_rad)
        component_positions.append((x, y))
        
        # Create component box
        ax = fig.add_axes([x-0.08, y-0.06, 0.16, 0.12])
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                                   facecolor=comp['color'], alpha=0.2, 
                                   edgecolor=comp['color'], linewidth=2))
        
        # Title
        ax.text(0.5, 0.85, comp['title'], ha='center', va='center', 
               fontsize=9, weight='bold', color=comp['color'])
        
        # Content
        ax.text(0.5, 0.35, comp['content'], ha='center', va='center',
               fontsize=8, color='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Connection to center
        conn = ConnectionPatch((x, y), (center_x, center_y), "data", "data",
                              arrowstyle='-', shrinkA=40, shrinkB=30,
                              color=comp['color'], alpha=0.6, linewidth=2)
        fig.add_artist(conn)
    
    # Add interconnections between adjacent components
    for i in range(len(components)):
        next_i = (i + 1) % len(components)
        conn = ConnectionPatch(component_positions[i], component_positions[next_i], 
                              "data", "data", arrowstyle='->', 
                              shrinkA=35, shrinkB=35, color=colors['gray'], 
                              alpha=0.4, linewidth=1, linestyle='dashed')
        fig.add_artist(conn)
    
    # Timeline at bottom
    timeline_y = 0.15
    timeline_items = [
        ('Phase 1\nValidation\n(Complete)', 0.1, colors['blue']),
        ('Phase 2\nScenario Modeling\n(6 months)', 0.3, colors['green']),
        ('Phase 3\nIntervention Design\n(12 months)', 0.5, colors['orange']),
        ('Phase 4\nPolicy Pilot\n(18 months)', 0.7, colors['red']),
        ('Phase 5\nFull Implementation\n(24 months)', 0.9, colors['purple'])
    ]
    
    # Timeline line
    timeline_line = patches.Rectangle((0.05, timeline_y-0.003), 0.9, 0.006, 
                                    facecolor=colors['gray'], edgecolor='none')
    fig.add_artist(timeline_line)
    
    for item, x_pos, color in timeline_items:
        # Timeline marker
        circle = Circle((x_pos, timeline_y), 0.015, facecolor=color, edgecolor='white', linewidth=2)
        fig.add_artist(circle)
        
        # Timeline text
        fig.text(x_pos, timeline_y - 0.06, item, ha='center', va='top',
                fontsize=10, weight='bold', color=color)
    
    # Key deliverables box
    deliverables_text = """Immediate Deliverables:
• CMIP6-health impact calculator for Johannesburg health department
• Clinical decision support tool for 21-day cardiovascular monitoring  
• Early warning system integration with existing heat-health protocols
• Economic impact assessment for climate adaptation funding proposals"""
    
    fig.text(0.02, 0.35, deliverables_text, ha='left', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.6", facecolor=colors['lightblue'], 
                      edgecolor=colors['blue'], alpha=0.8))
    
    # Success metrics box  
    metrics_text = """Success Metrics:
• Health system preparedness index
• Early warning system accuracy (>80%)
• Vulnerable population coverage (>90%)
• Policy adoption across 3 metro areas
• Cost-benefit ratio >3:1"""
    
    fig.text(0.98, 0.35, metrics_text, ha='right', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.6", facecolor='lightcoral', 
                      edgecolor=colors['red'], alpha=0.8))
    
    # Partnership network
    fig.text(0.5, 0.02, 
            'Partnership Network: City of Johannesburg • NICD • WITS School of Public Health • CSIR Climate Studies • WHO Africa',
            fontsize=10, ha='center', style='italic', color=colors['gray'])
    
    plt.savefig('enbel_next_steps_framework.svg', format='svg', 
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    plt.savefig('enbel_next_steps_framework.png', format='png',
                bbox_inches='tight', dpi=150, facecolor=colors['white'])
    
    print("Next steps framework slide saved")
    plt.show()

if __name__ == "__main__":
    create_next_steps_framework()