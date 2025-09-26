"""
Final Slide 13: Climate-Health Applications Summary
Clean, visual slide with key headers and minimal text
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style for clean presentation
plt.style.use('seaborn-v0_8-whitegrid')

# ENBEL color scheme
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00', 
    'green': '#2CA02C',
    'red': '#DC2626',
    'purple': '#9467BD',
    'gray': '#64748B',
    'lightblue': '#E6F0FA',
    'clinical_red': '#EF4444',
    'research_blue': '#3B82F6',
    'policy_green': '#10B981'
}

# Create 16:9 slide format
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('#F8FAFC')

# Custom layout - header + 3 columns + action bar
gs = GridSpec(4, 3, figure=fig, height_ratios=[0.15, 0.6, 0.15, 0.1], 
              width_ratios=[0.33, 0.34, 0.33], hspace=0.08, wspace=0.06)

# Title section
ax_title = fig.add_subplot(gs[0, :])
ax_title.text(0.5, 0.6, 'Climate-Health Discoveries Enable Precision Medicine', 
             ha='center', va='center', fontsize=28, fontweight='bold', 
             color='white', transform=ax_title.transAxes)
ax_title.text(0.5, 0.2, 'Translating validated findings into clinical excellence and policy implementation', 
             ha='center', va='center', fontsize=16, style='italic',
             color='white', alpha=0.9, transform=ax_title.transAxes)

# Gradient background
title_bg = Rectangle((0, 0), 1, 1, transform=ax_title.transAxes, 
                    facecolor=colors['blue'], alpha=0.9, zorder=0)
ax_title.add_patch(title_bg)
ax_title.set_xlim(0, 1)
ax_title.set_ylim(0, 1)
ax_title.axis('off')

# Clinical Applications (Left)
ax_clinical = fig.add_subplot(gs[1, 0])

# Clinical header box
clinical_header = FancyBboxPatch((0.05, 0.85), 0.9, 0.12,
                                boxstyle="round,pad=0.02",
                                facecolor=colors['clinical_red'], alpha=0.9,
                                edgecolor=colors['clinical_red'], linewidth=2)
ax_clinical.add_patch(clinical_header)
ax_clinical.text(0.5, 0.91, 'CLINICAL APPLICATIONS', 
                ha='center', va='center', fontsize=16, fontweight='bold', 
                color='white', transform=ax_clinical.transAxes)

# Clinical content boxes
clinical_items = [
    {'title': 'Extended Monitoring', 'icon': '📊', 'subtitle': '21-day protocols'},
    {'title': 'Evidence-Based Decisions', 'icon': '🎯', 'subtitle': 'Climate-informed care'},
    {'title': 'Vulnerable Populations', 'icon': '🛡️', 'subtitle': 'Targeted protection'}
]

for i, item in enumerate(clinical_items):
    y_pos = 0.65 - (i * 0.22)
    
    # Item box
    item_box = FancyBboxPatch((0.1, y_pos - 0.08), 0.8, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='white', alpha=0.9,
                             edgecolor=colors['clinical_red'], linewidth=1)
    ax_clinical.add_patch(item_box)
    
    # Icon and text
    ax_clinical.text(0.2, y_pos, item['icon'], ha='center', va='center', 
                    fontsize=24, transform=ax_clinical.transAxes)
    ax_clinical.text(0.35, y_pos + 0.03, item['title'], ha='left', va='center', 
                    fontsize=12, fontweight='bold', color=colors['clinical_red'],
                    transform=ax_clinical.transAxes)
    ax_clinical.text(0.35, y_pos - 0.03, item['subtitle'], ha='left', va='center', 
                    fontsize=10, color=colors['gray'], style='italic',
                    transform=ax_clinical.transAxes)

ax_clinical.set_xlim(0, 1)
ax_clinical.set_ylim(0, 1)
ax_clinical.axis('off')

# Research Paradigm Shift (Center)
ax_research = fig.add_subplot(gs[1, 1])

# Research header box
research_header = FancyBboxPatch((0.05, 0.85), 0.9, 0.12,
                                boxstyle="round,pad=0.02",
                                facecolor=colors['research_blue'], alpha=0.9,
                                edgecolor=colors['research_blue'], linewidth=2)
ax_research.add_patch(research_header)
ax_research.text(0.5, 0.91, 'RESEARCH PARADIGM SHIFT', 
                ha='center', va='center', fontsize=16, fontweight='bold', 
                color='white', transform=ax_research.transAxes)

# Research content boxes
research_items = [
    {'title': 'XAI-Guided Discovery', 'icon': '🧠', 'subtitle': 'Hypothesis generation'},
    {'title': 'Multi-System Analysis', 'icon': '🔬', 'subtitle': 'Integrated physiology'},
    {'title': 'African Climate-Health', 'icon': '🌍', 'subtitle': 'Global South research'}
]

for i, item in enumerate(research_items):
    y_pos = 0.65 - (i * 0.22)
    
    # Item box
    item_box = FancyBboxPatch((0.1, y_pos - 0.08), 0.8, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='white', alpha=0.9,
                             edgecolor=colors['research_blue'], linewidth=1)
    ax_research.add_patch(item_box)
    
    # Icon and text
    ax_research.text(0.2, y_pos, item['icon'], ha='center', va='center', 
                    fontsize=24, transform=ax_research.transAxes)
    ax_research.text(0.35, y_pos + 0.03, item['title'], ha='left', va='center', 
                    fontsize=12, fontweight='bold', color=colors['research_blue'],
                    transform=ax_research.transAxes)
    ax_research.text(0.35, y_pos - 0.03, item['subtitle'], ha='left', va='center', 
                    fontsize=10, color=colors['gray'], style='italic',
                    transform=ax_research.transAxes)

ax_research.set_xlim(0, 1)
ax_research.set_ylim(0, 1)
ax_research.axis('off')

# Policy Implementation (Right)
ax_policy = fig.add_subplot(gs[1, 2])

# Policy header box
policy_header = FancyBboxPatch((0.05, 0.85), 0.9, 0.12,
                              boxstyle="round,pad=0.02",
                              facecolor=colors['policy_green'], alpha=0.9,
                              edgecolor=colors['policy_green'], linewidth=2)
ax_policy.add_patch(policy_header)
ax_policy.text(0.5, 0.91, 'POLICY IMPLEMENTATION', 
              ha='center', va='center', fontsize=16, fontweight='bold', 
              color='white', transform=ax_policy.transAxes)

# Policy content boxes
policy_items = [
    {'title': 'Health Surveillance', 'icon': '📈', 'subtitle': 'Population monitoring'},
    {'title': 'Healthcare Policies', 'icon': '🏥', 'subtitle': 'Climate-informed care'},
    {'title': 'Heat Mitigation', 'icon': '🌡️', 'subtitle': 'Urban adaptation'}
]

for i, item in enumerate(policy_items):
    y_pos = 0.65 - (i * 0.22)
    
    # Item box
    item_box = FancyBboxPatch((0.1, y_pos - 0.08), 0.8, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='white', alpha=0.9,
                             edgecolor=colors['policy_green'], linewidth=1)
    ax_policy.add_patch(item_box)
    
    # Icon and text
    ax_policy.text(0.2, y_pos, item['icon'], ha='center', va='center', 
                  fontsize=24, transform=ax_policy.transAxes)
    ax_policy.text(0.35, y_pos + 0.03, item['title'], ha='left', va='center', 
                  fontsize=12, fontweight='bold', color=colors['policy_green'],
                  transform=ax_policy.transAxes)
    ax_policy.text(0.35, y_pos - 0.03, item['subtitle'], ha='left', va='center', 
                  fontsize=10, color=colors['gray'], style='italic',
                  transform=ax_policy.transAxes)

ax_policy.set_xlim(0, 1)
ax_policy.set_ylim(0, 1)
ax_policy.axis('off')

# Action Required Section
ax_action = fig.add_subplot(gs[2, :])

# Action background
action_bg = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                          boxstyle="round,pad=0.02",
                          facecolor=colors['orange'], alpha=0.9,
                          edgecolor=colors['orange'], linewidth=2)
ax_action.add_patch(action_bg)

ax_action.text(0.1, 0.5, '⚡ IMMEDIATE ACTION REQUIRED', 
              ha='left', va='center', fontsize=18, fontweight='bold', 
              color='white', transform=ax_action.transAxes)

action_items = [
    'Clinical: 21-day cardiovascular monitoring',
    'Research: XAI-guided discovery framework',
    'Policy: Evidence-based heat health systems'
]

action_text = ' • '.join(action_items)
ax_action.text(0.5, 0.5, action_text, 
              ha='center', va='center', fontsize=14, fontweight='bold', 
              color='white', transform=ax_action.transAxes)

ax_action.set_xlim(0, 1)
ax_action.set_ylim(0, 1)
ax_action.axis('off')

# Translational Impact Footer
ax_footer = fig.add_subplot(gs[3, :])

footer_bg = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                          boxstyle="round,pad=0.02",
                          facecolor=colors['blue'], alpha=0.9,
                          edgecolor=colors['blue'], linewidth=2)
ax_footer.add_patch(footer_bg)

ax_footer.text(0.5, 0.5, 'Translational Impact: Clinical Excellence • Research Innovation • Policy Implementation', 
              ha='center', va='center', fontsize=16, fontweight='bold', 
              color='white', transform=ax_footer.transAxes)

ax_footer.set_xlim(0, 1)
ax_footer.set_ylim(0, 1)
ax_footer.axis('off')

# Add slide number and ENBEL logo
fig.text(0.95, 0.02, 'ENBEL', fontsize=14, fontweight='bold', 
         color=colors['gray'], alpha=0.7, ha='right')
fig.text(0.95, 0.005, '13/13', fontsize=12, 
         color=colors['gray'], alpha=0.7, ha='right')

plt.tight_layout()

# Save as SVG for Figma
plt.savefig('enbel_slide_13_final_applications.svg', 
           format='svg', bbox_inches='tight', dpi=100, 
           facecolor='#F8FAFC', edgecolor='none')

plt.savefig('enbel_slide_13_final_applications.png', 
           format='png', bbox_inches='tight', dpi=300, 
           facecolor='#F8FAFC', edgecolor='none')

plt.show()

print("✅ Final applications slide created!")
print("📁 Files saved:")
print("   • enbel_slide_13_final_applications.svg (for Figma)")
print("   • enbel_slide_13_final_applications.png (high-res)")
print("\n🎯 Features:")
print("   • Clean 3-column layout with minimal text")
print("   • Visual icons for each application area")
print("   • Consolidated key messages")
print("   • Professional color-coding by domain")
print("   • Action-oriented immediate steps")
print("   • Translational impact emphasis")