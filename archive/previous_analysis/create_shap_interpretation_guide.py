#!/usr/bin/env python3
"""
Create SHAP Interpretation Guide Slide
Educational slide explaining how to read beeswarm plots and understand SHAP values
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

def create_shap_guide():
    """
    Create comprehensive SHAP interpretation guide
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#F8FAFC')
    
    # Main title
    fig.suptitle('How to Interpret SHAP Values and Beeswarm Plots', 
                fontsize=24, weight='bold', y=0.98, color=colors['blue'])
    fig.text(0.5, 0.94, 'A guide to understanding explainable AI outputs in climate-health research',
            fontsize=14, ha='center', style='italic', color=colors['gray'])
    
    # Create layout
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=1)
    ax_beeswarm = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2)
    ax_concepts = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=1)
    ax_reading = plt.subplot2grid((3, 4), (2, 2), colspan=2, rowspan=1)
    
    # === SHAP CONCEPTS SECTION ===
    ax_main.axis('off')
    
    # Background box for concepts
    concept_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                boxstyle="round,pad=0.02",
                                facecolor=colors['lightblue'], 
                                edgecolor=colors['blue'],
                                linewidth=2, alpha=0.95)
    ax_main.add_patch(concept_box)
    
    # Title
    ax_main.text(0.5, 0.85, 'What are SHAP Values?', fontsize=16, weight='bold',
                ha='center', color=colors['blue'], transform=ax_main.transAxes)
    
    # Concept explanations
    concepts = [
        ('SHAP = SHapley Additive exPlanations', 0.7, 12, 'bold'),
        ('• How much each feature contributes to a prediction', 0.6, 11, 'normal'),
        ('• Positive values push prediction higher (red)', 0.52, 11, 'normal'),
        ('• Negative values push prediction lower (blue)', 0.44, 11, 'normal'),
        ('• Values add up: Base + SHAP₁ + SHAP₂ + ... = Final Prediction', 0.36, 11, 'normal'),
        ('• Fair attribution: Every feature gets its due credit', 0.28, 11, 'normal')
    ]
    
    for text, y, size, weight in concepts:
        ax_main.text(0.1, y, text, fontsize=size, weight=weight,
                    color='#2c3e50', transform=ax_main.transAxes)
    
    # === BEESWARM PLOT EXAMPLE ===
    ax_beeswarm.set_title('Example: SHAP Beeswarm Plot', fontsize=14, weight='bold', 
                         color=colors['blue'], pad=10)
    
    # Simulate beeswarm plot data
    features = ['Heat Index (lag 0)', 'Age', 'Economic Vulnerability', 'Education Level', 
                'Housing Quality', 'Heat Index (lag 7)', 'Sex', 'Previous Health']
    n_points = 50
    
    np.random.seed(42)  # Reproducible
    
    for i, feature in enumerate(features):
        # Generate SHAP values for this feature
        if 'Heat' in feature:
            # Heat features: mostly positive impact
            shap_values = np.random.exponential(1.5, n_points) * np.random.choice([1, -1], n_points, p=[0.8, 0.2])
            feature_values = np.random.uniform(25, 45, n_points)  # Temperature range
        elif 'Age' in feature:
            # Age: mixed impact
            shap_values = np.random.normal(0, 2, n_points)
            feature_values = np.random.uniform(20, 80, n_points)
        elif 'Vulnerability' in feature:
            # Economic vulnerability: positive impact (bad)
            shap_values = np.random.exponential(1.2, n_points) * np.random.choice([1, -1], n_points, p=[0.9, 0.1])
            feature_values = np.random.uniform(0, 1, n_points)
        else:
            # Other features: mixed
            shap_values = np.random.normal(0, 1.5, n_points)
            feature_values = np.random.uniform(0, 1, n_points)
        
        # Add jitter for beeswarm effect
        y_positions = np.full(n_points, i) + np.random.normal(0, 0.1, n_points)
        
        # Color based on feature values (low=blue, high=red)
        colors_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
        scatter_colors = plt.cm.RdYlBu_r(colors_norm)
        
        ax_beeswarm.scatter(shap_values, y_positions, c=scatter_colors, 
                          alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    
    # Formatting
    ax_beeswarm.set_yticks(range(len(features)))
    ax_beeswarm.set_yticklabels(features, fontsize=10)
    ax_beeswarm.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
    ax_beeswarm.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_beeswarm.set_xlim(-4, 6)
    ax_beeswarm.spines['top'].set_visible(False)
    ax_beeswarm.spines['right'].set_visible(False)
    ax_beeswarm.grid(axis='x', alpha=0.2)
    
    # Add annotations
    ax_beeswarm.annotate('Increases\nPrediction', xy=(4, 7.5), fontsize=10, 
                        ha='center', color=colors['red'], weight='bold')
    ax_beeswarm.annotate('Decreases\nPrediction', xy=(-3, 7.5), fontsize=10,
                        ha='center', color=colors['blue'], weight='bold')
    
    # Add color bar explanation
    ax_beeswarm.text(0.02, 0.98, 'Color: Blue = Low feature value, Red = High feature value',
                    transform=ax_beeswarm.transAxes, fontsize=9, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # === KEY CONCEPTS BOX ===
    ax_concepts.axis('off')
    
    concepts_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#FFF5F0', 
                                 edgecolor=colors['orange'],
                                 linewidth=2, alpha=0.95)
    ax_concepts.add_patch(concepts_box)
    
    ax_concepts.text(0.5, 0.9, 'Key Insights from Beeswarm Plots', fontsize=12, weight='bold',
                    ha='center', color=colors['orange'], transform=ax_concepts.transAxes)
    
    insights = [
        '1. Feature Importance',
        '   Wider spread = more important',
        '',
        '2. Direction of Impact',
        '   Right side increases prediction',
        '   Left side decreases prediction',
        '',
        '3. Feature Value Effects',
        '   Red dots = high feature values',
        '   Blue dots = low feature values',
        '',
        '4. Individual Variations',
        '   Each dot = one person',
        '   Spread shows variability'
    ]
    
    y_pos = 0.8
    for insight in insights:
        if insight.startswith((' ', '\t')):
            # Indented text
            ax_concepts.text(0.15, y_pos, insight.strip(), fontsize=10,
                           color='#2c3e50', transform=ax_concepts.transAxes)
        elif insight == '':
            pass  # Skip empty lines
        else:
            # Main points
            ax_concepts.text(0.1, y_pos, insight, fontsize=10, weight='bold',
                           color=colors['orange'], transform=ax_concepts.transAxes)
        y_pos -= 0.08
    
    # === HOW TO READ BOX ===
    ax_reading.axis('off')
    
    reading_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor='#F0FFF0', 
                                edgecolor=colors['green'],
                                linewidth=2, alpha=0.95)
    ax_reading.add_patch(reading_box)
    
    ax_reading.text(0.5, 0.9, 'How to Read the Plot', fontsize=12, weight='bold',
                   ha='center', color=colors['green'], transform=ax_reading.transAxes)
    
    reading_steps = [
        '1. Look at feature ranking',
        '   Top features = most important',
        '',
        '2. Check the spread',
        '   Wide = high impact variability',
        '   Narrow = consistent impact',
        '',
        '3. Notice the colors',
        '   Pattern shows how feature',
        '   values affect the outcome',
        '',
        '4. Find interactions',
        '   Same feature, different colors',
        '   → complex relationships'
    ]
    
    y_pos = 0.8
    for step in reading_steps:
        if step.startswith((' ', '\t')):
            # Indented text
            ax_reading.text(0.15, y_pos, step.strip(), fontsize=10,
                          color='#2c3e50', transform=ax_reading.transAxes)
        elif step == '':
            pass  # Skip empty lines
        else:
            # Main points
            ax_reading.text(0.1, y_pos, step, fontsize=10, weight='bold',
                          color=colors['green'], transform=ax_reading.transAxes)
        y_pos -= 0.08
    
    # Add bottom note
    fig.text(0.5, 0.02, 'SHAP values help us understand "why" the model made each prediction, not just "what" it predicted',
            fontsize=11, ha='center', weight='bold', style='italic', color=colors['purple'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Save
    plt.savefig('enbel_shap_interpretation_guide.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('enbel_shap_interpretation_guide.png', format='png', bbox_inches='tight', dpi=150)
    print("SHAP interpretation guide saved as 'enbel_shap_interpretation_guide.svg' and '.png'")
    
    plt.show()

if __name__ == "__main__":
    create_shap_guide()