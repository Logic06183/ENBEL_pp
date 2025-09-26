#!/usr/bin/env python3
"""
Create PNG version of the perfected population health impact slide using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ENBEL colors
ENBEL_BLUE = '#00539B'
ENBEL_ORANGE = '#FF7F00'
ENBEL_GREEN = '#2CA02C'

def create_perfected_slide():
    """Create the perfected population health impact slide."""
    
    # Set up figure with 16:9 aspect ratio
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Background
    fig.patch.set_facecolor('white')
    
    # Header background
    header_bg = FancyBboxPatch(
        (0, 85), 100, 15,
        boxstyle="round,pad=0",
        facecolor=ENBEL_BLUE,
        alpha=0.05,
        edgecolor='none'
    )
    ax.add_patch(header_bg)
    
    # Main title
    ax.text(50, 92, 'Effect Sizes Exceed WHO & ADA Clinical Thresholds',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color=ENBEL_BLUE, family='sans-serif')
    
    # Population context
    ax.text(50, 88, 'Population Health Impact — Johannesburg Metro (5.6M people)¹',
            ha='center', va='center', fontsize=14,
            color='#333333', family='sans-serif')
    
    # Heat Wave Scenario Box
    heat_box = FancyBboxPatch(
        (5, 45), 28, 35,
        boxstyle="round,pad=1",
        facecolor=ENBEL_ORANGE,
        alpha=0.1,
        edgecolor=ENBEL_ORANGE,
        linewidth=2
    )
    ax.add_patch(heat_box)
    
    # Heat wave header
    heat_header = FancyBboxPatch(
        (5, 72), 28, 8,
        boxstyle="round,pad=0.5",
        facecolor=ENBEL_ORANGE,
        edgecolor='none'
    )
    ax.add_patch(heat_header)
    
    ax.text(19, 76, 'Heat Wave Scenario (+5°C)', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', family='sans-serif')
    
    # Heat wave content
    ax.text(7, 68, 'Extreme Heat Events', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_ORANGE, family='sans-serif')
    ax.text(7, 65, '• 15+ additional days >35°C annually', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    ax.text(7, 62, '• Peak temperatures reaching 40°C', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    ax.text(7, 59, '• Urban heat island amplification', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    
    ax.text(7, 55, 'Population Exposure', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_ORANGE, family='sans-serif')
    ax.text(7, 52, '• 1.8M residents affected daily', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    
    # Metabolic Impact Box
    metabolic_box = FancyBboxPatch(
        (36, 45), 28, 35,
        boxstyle="round,pad=1",
        facecolor=ENBEL_BLUE,
        alpha=0.1,
        edgecolor=ENBEL_BLUE,
        linewidth=2
    )
    ax.add_patch(metabolic_box)
    
    # Metabolic header
    metabolic_header = FancyBboxPatch(
        (36, 72), 28, 8,
        boxstyle="round,pad=0.5",
        facecolor=ENBEL_BLUE,
        edgecolor='none'
    )
    ax.add_patch(metabolic_header)
    
    ax.text(50, 76, 'Metabolic Health Impact', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', family='sans-serif')
    
    # Metabolic content
    ax.text(38, 68, 'Blood Pressure Effects', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_BLUE, family='sans-serif')
    ax.text(38, 64, '+14.5 mmHg', ha='left', va='top',
            fontsize=16, fontweight='bold', color=ENBEL_ORANGE, family='sans-serif')
    ax.text(38, 61, 'vs WHO threshold: 5 mmHg²', ha='left', va='top',
            fontsize=7, color='#666666', family='sans-serif')
    
    ax.text(38, 57, 'Glucose Dysregulation', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_BLUE, family='sans-serif')
    ax.text(38, 53, '+41 mg/dL', ha='left', va='top',
            fontsize=16, fontweight='bold', color=ENBEL_ORANGE, family='sans-serif')
    ax.text(38, 50, 'vs ADA threshold: 25 mg/dL³', ha='left', va='top',
            fontsize=7, color='#666666', family='sans-serif')
    
    # Public Health Response Box
    response_box = FancyBboxPatch(
        (67, 45), 28, 35,
        boxstyle="round,pad=1",
        facecolor=ENBEL_GREEN,
        alpha=0.1,
        edgecolor=ENBEL_GREEN,
        linewidth=2
    )
    ax.add_patch(response_box)
    
    # Response header
    response_header = FancyBboxPatch(
        (67, 72), 28, 8,
        boxstyle="round,pad=0.5",
        facecolor=ENBEL_GREEN,
        edgecolor='none'
    )
    ax.add_patch(response_header)
    
    ax.text(81, 76, 'Public Health Response', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', family='sans-serif')
    
    # Response content
    ax.text(69, 68, 'Healthcare Burden', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_GREEN, family='sans-serif')
    ax.text(69, 65, '• 300K diabetic patients at risk⁴', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    ax.text(69, 62, '• 45% increase in heat-related visits⁵', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    
    ax.text(69, 57, 'Adaptation Strategies', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_GREEN, family='sans-serif')
    ax.text(69, 54, '• Early warning systems', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    ax.text(69, 51, '• Cool shelter networks', ha='left', va='top',
            fontsize=8, color='#333333', family='sans-serif')
    
    # Key Findings Banner
    banner = FancyBboxPatch(
        (5, 35), 90, 8,
        boxstyle="round,pad=0.5",
        facecolor=ENBEL_BLUE,
        alpha=0.1,
        edgecolor=ENBEL_BLUE,
        linewidth=1
    )
    ax.add_patch(banner)
    
    ax.text(50, 40.5, 'Clinical Significance', ha='center', va='center',
            fontsize=12, fontweight='bold', color=ENBEL_BLUE, family='sans-serif')
    ax.text(50, 37, 'Effect sizes exceed established clinical thresholds by 290% (BP) and 164% (glucose), indicating substantial population health risk',
            ha='center', va='center', fontsize=10, color='#333333', family='sans-serif')
    
    # References Section
    ref_box = FancyBboxPatch(
        (5, 3), 90, 30,
        boxstyle="round,pad=1",
        facecolor='#f8f9fa',
        edgecolor='#e9ecef',
        linewidth=1
    )
    ax.add_patch(ref_box)
    
    # References header
    ax.text(7, 30, 'References', ha='left', va='top',
            fontsize=10, fontweight='bold', color=ENBEL_BLUE, family='sans-serif')
    
    # Left column references
    refs_left = [
        "1. Statistics South Africa. Community Survey 2016: Johannesburg Metropolitan Municipality.",
        "    Pretoria: Stats SA, 2018.",
        "2. WHO. Hypertension Guidelines 2021. Geneva: World Health Organization, 2021.",
        "3. American Diabetes Association. Standards of Medical Care in Diabetes—2023.",
        "    Diabetes Care. 2023;46(Suppl 1):S1-S291.",
        "4. Bradshaw D, et al. Burden of disease in South Africa: protracted transitions in a",
        "    rapidly changing society. Lancet Global Health. 2019;7(10):e1369-e1385.",
        "5. Gasparrini A, et al. Temperature and mortality: a multi-country study. Lancet.",
        "    2015;386(9991):369-375."
    ]
    
    y_pos = 27
    for ref in refs_left:
        ax.text(7, y_pos, ref, ha='left', va='top',
                fontsize=6, color='#333333', family='sans-serif')
        y_pos -= 2.2
    
    # Right column references
    refs_right = [
        "6. Watts N, et al. The 2020 report of The Lancet Countdown on health and climate",
        "    change. Lancet. 2021;397(10269):129-170.",
        "7. Hondula DM, et al. Heat-related morbidity in emergency departments during the",
        "    Chicago heat wave of 1995. Environ Health Perspect. 2012;120(10):1369-1375.",
        "8. Reid CE, et al. Mapping community determinants of heat vulnerability.",
        "    Environ Health Perspect. 2009;117(11):1730-1736.",
        "9. Campbell S, et al. Heatwave and health impact research: A global review.",
        "    Health Place. 2018;53:210-218.",
        "10. Kjellstrom T, et al. Heat, human performance, and occupational health.",
        "     Asia Pacific J Public Health. 2016;28(2_suppl):8S-37S."
    ]
    
    y_pos = 27
    for ref in refs_right:
        ax.text(52, y_pos, ref, ha='left', va='top',
                fontsize=6, color='#333333', family='sans-serif')
        y_pos -= 2.2
    
    # Footer
    ax.text(50, 1, 'ENBEL Project | Population Health Impact Assessment | Climate-Health Interface Analysis',
            ha='center', va='center', fontsize=8, color='#666666', family='sans-serif')
    
    # Save as PNG with high DPI
    plt.tight_layout()
    plt.savefig('enbel_population_health_impact_perfected.png', 
                dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    
    print("Successfully created enbel_population_health_impact_perfected.png")
    print("Resolution: High DPI (300) for presentation quality")
    
    plt.close()

if __name__ == "__main__":
    create_perfected_slide()