#!/usr/bin/env python3
"""
Create Color Interpretation Guide for SHAP Beeswarm Plots
Explains what the red/blue colors actually mean and how to interpret patterns
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# ENBEL color scheme
colors = {
    'blue': '#00539B',
    'orange': '#FF7F00', 
    'green': '#2CA02C',
    'red': '#DC2626',
    'purple': '#9467BD',
    'gray': '#8C8C8C',
    'lightblue': '#E6F0FA',
}

def create_color_guide():
    """
    Create guide explaining the color patterns in beeswarm plots
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#F8FAFC')
    
    # Main title
    fig.suptitle('SHAP Beeswarm Plot Colors: What Red and Blue Actually Mean', 
                fontsize=22, weight='bold', y=0.96, color=colors['blue'])
    fig.text(0.5, 0.92, 'Understanding the relationship between feature values and their impact on predictions',
            fontsize=13, ha='center', style='italic', color=colors['gray'])
    
    # Create layout - 2 rows, 2 columns
    ax_concept = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax_example1 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax_example2 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    
    # === CONCEPT EXPLANATION ===
    ax_concept.axis('off')
    
    concept_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                boxstyle="round,pad=0.02",
                                facecolor=colors['lightblue'], edgecolor=colors['blue'],
                                linewidth=2, alpha=0.95)
    ax_concept.add_patch(concept_box)
    
    ax_concept.text(0.5, 0.8, 'Color Coding in SHAP Beeswarm Plots', fontsize=16, weight='bold',
                   ha='center', color=colors['blue'], transform=ax_concept.transAxes)
    
    # Left side - Red explanation
    red_box = FancyBboxPatch((0.1, 0.3), 0.35, 0.4,
                            boxstyle="round,pad=0.02",
                            facecolor='#FFE5E5', edgecolor=colors['red'],
                            linewidth=2, alpha=0.8)
    ax_concept.add_patch(red_box)
    
    ax_concept.text(0.275, 0.65, 'RED DOTS', fontsize=14, weight='bold',
                   ha='center', color=colors['red'], transform=ax_concept.transAxes)
    ax_concept.text(0.275, 0.6, 'HIGH feature values', fontsize=12, weight='bold',
                   ha='center', color=colors['red'], transform=ax_concept.transAxes)
    
    red_examples = [
        "• High temperature (35°C)",
        "• High heat index (45°C)", 
        "• Older age (70+ years)",
        "• High economic vulnerability"
    ]
    
    y_red = 0.52
    for example in red_examples:
        ax_concept.text(0.12, y_red, example, fontsize=10,
                       color='#2c3e50', transform=ax_concept.transAxes)
        y_red -= 0.04
    
    # Right side - Blue explanation  
    blue_box = FancyBboxPatch((0.55, 0.3), 0.35, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor='#E5F0FF', edgecolor='#0066CC',
                             linewidth=2, alpha=0.8)
    ax_concept.add_patch(blue_box)
    
    ax_concept.text(0.725, 0.65, 'BLUE DOTS', fontsize=14, weight='bold',
                   ha='center', color='#0066CC', transform=ax_concept.transAxes)
    ax_concept.text(0.725, 0.6, 'LOW feature values', fontsize=12, weight='bold',
                   ha='center', color='#0066CC', transform=ax_concept.transAxes)
    
    blue_examples = [
        "• Low temperature (15°C)",
        "• Low heat index (20°C)",
        "• Younger age (25 years)", 
        "• Low economic vulnerability"
    ]
    
    y_blue = 0.52
    for example in blue_examples:
        ax_concept.text(0.57, y_blue, example, fontsize=10,
                       color='#2c3e50', transform=ax_concept.transAxes)
        y_blue -= 0.04
    
    # Central key message
    ax_concept.text(0.5, 0.2, 'KEY: Colors show feature VALUE, position shows IMPACT', 
                   fontsize=14, weight='bold', ha='center',
                   color=colors['purple'], transform=ax_concept.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # === EXAMPLE 1: POSITIVE RELATIONSHIP ===
    ax_example1.set_title('Example 1: Temperature → Blood Pressure\n(Positive Relationship)', 
                         fontsize=14, weight='bold', color=colors['blue'], pad=10)
    
    # Simulate positive relationship data
    np.random.seed(42)
    n_points = 100
    
    # Generate temperature values (feature values)
    temp_values = np.random.uniform(10, 40, n_points)  # 10°C to 40°C
    
    # Generate SHAP values - higher temp = higher SHAP (positive relationship)
    shap_values = (temp_values - 25) * 0.3 + np.random.normal(0, 0.5, n_points)
    
    # Color by temperature value
    colors_norm = (temp_values - temp_values.min()) / (temp_values.max() - temp_values.min())
    scatter_colors = plt.cm.RdYlBu_r(colors_norm)
    
    # Add jitter for visualization
    y_positions = np.random.normal(0, 0.1, n_points)
    
    ax_example1.scatter(shap_values, y_positions, c=scatter_colors, 
                       alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
    
    ax_example1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_example1.set_xlabel('SHAP Value (Impact on BP)', fontsize=11)
    ax_example1.set_ylabel('', fontsize=11)
    ax_example1.set_ylim(-0.5, 0.5)
    ax_example1.set_yticks([])
    ax_example1.spines['top'].set_visible(False)
    ax_example1.spines['right'].set_visible(False)
    ax_example1.spines['left'].set_visible(False)
    ax_example1.grid(axis='x', alpha=0.2)
    
    # Add pattern explanation
    ax_example1.text(0.02, 0.95, 'PATTERN: Red dots (hot days) on RIGHT → heat increases BP',
                    transform=ax_example1.transAxes, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE5E5', alpha=0.8))
    ax_example1.text(0.02, 0.85, 'Blue dots (cool days) on LEFT → cold decreases BP',
                    transform=ax_example1.transAxes, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#E5F0FF', alpha=0.8))
    
    # Add arrows
    ax_example1.annotate('Hot weather\nincreases BP', xy=(2, 0.3), xytext=(1, 0.4),
                        arrowprops=dict(arrowstyle='->', color=colors['red'], lw=2),
                        fontsize=9, ha='center', color=colors['red'], weight='bold')
    ax_example1.annotate('Cold weather\ndecreases BP', xy=(-2, 0.3), xytext=(-1, 0.4),
                        arrowprops=dict(arrowstyle='->', color='#0066CC', lw=2),
                        fontsize=9, ha='center', color='#0066CC', weight='bold')
    
    # === EXAMPLE 2: COMPLEX RELATIONSHIP ===
    ax_example2.set_title('Example 2: Age → Health Outcome\n(Complex Relationship)', 
                         fontsize=14, weight='bold', color=colors['blue'], pad=10)
    
    # Simulate complex age relationship
    age_values = np.random.uniform(20, 80, n_points)
    
    # Complex relationship: middle-aged people most vulnerable
    age_centered = age_values - 50  # Center at 50
    shap_values_age = -0.1 * age_centered**2 / 100 + 2 + np.random.normal(0, 0.8, n_points)
    
    # Color by age
    colors_norm_age = (age_values - age_values.min()) / (age_values.max() - age_values.min())
    scatter_colors_age = plt.cm.RdYlBu_r(colors_norm_age)
    
    y_positions_age = np.random.normal(0, 0.1, n_points)
    
    ax_example2.scatter(shap_values_age, y_positions_age, c=scatter_colors_age,
                       alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
    
    ax_example2.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_example2.set_xlabel('SHAP Value (Impact on Health)', fontsize=11)
    ax_example2.set_ylabel('', fontsize=11)
    ax_example2.set_ylim(-0.5, 0.5)
    ax_example2.set_yticks([])
    ax_example2.spines['top'].set_visible(False)
    ax_example2.spines['right'].set_visible(False)
    ax_example2.spines['left'].set_visible(False)
    ax_example2.grid(axis='x', alpha=0.2)
    
    # Add pattern explanation
    ax_example2.text(0.02, 0.95, 'PATTERN: Red (elderly) mixed with blue (young)',
                    transform=ax_example2.transAxes, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFCC', alpha=0.8))
    ax_example2.text(0.02, 0.85, '→ Complex, non-linear age effects',
                    transform=ax_example2.transAxes, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFCC', alpha=0.8))
    
    # Bottom interpretation guide
    fig.text(0.5, 0.02, 'How to Read: (1) Color = feature value intensity, (2) X-position = impact on prediction, (3) Spread = importance, (4) Pattern = relationship type',
            fontsize=12, ha='center', weight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['lightblue'], alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.91])
    
    # Save
    plt.savefig('enbel_color_interpretation_guide.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('enbel_color_interpretation_guide.png', format='png', bbox_inches='tight', dpi=150)
    print("Color interpretation guide saved as 'enbel_color_interpretation_guide.svg' and '.png'")
    
    plt.show()

if __name__ == "__main__":
    create_color_guide()