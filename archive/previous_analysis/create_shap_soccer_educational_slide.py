#!/usr/bin/env python3
"""
SHAP Educational Slide: Soccer/Football Analogy for Non-Technical Audiences
===========================================================================
Creates a professional educational slide explaining SHAP beeswarm and waterfall plots
using soccer analogies. Designed for 16:10 aspect ratio presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Configure matplotlib for text rendering (not paths)
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def create_shap_soccer_educational_slide():
    """Create comprehensive SHAP education slide using soccer analogies"""
    
    # Create figure with 16:10 aspect ratio
    fig = plt.figure(figsize=(16, 10), facecolor='#ffffff')
    
    # Define color palette matching existing slides
    primary_blue = '#2E86AB'
    accent_orange = '#F18F01'
    positive_green = '#27AE60'
    negative_red = '#E74C3C'
    light_gray = '#F5F5F5'
    medium_gray = '#666666'
    dark_gray = '#2C3E50'
    
    # Create main axis
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 16)
    ax_main.set_ylim(0, 10)
    ax_main.axis('off')
    
    # Add blue header bar matching existing slides
    header_rect = Rectangle((0, 8.8), 16, 1.2, facecolor=primary_blue, edgecolor='none')
    ax_main.add_patch(header_rect)
    
    # Main title in header
    ax_main.text(8, 9.4, 'Understanding SHAP Plots: Match Predictions Through Soccer Analytics',
                fontsize=20, fontweight='bold', ha='center', va='center', color='white')
    
    # Subtitle
    ax_main.text(8, 8.4, 'How AI explains its predictions using familiar football concepts',
                fontsize=12, ha='center', va='center', color=medium_gray, style='italic')
    
    # ==================== LEFT PANEL: BEESWARM PLOT ====================
    ax_bee = fig.add_axes([0.05, 0.15, 0.42, 0.55])
    
    # Title for beeswarm section
    ax_main.text(3.8, 7.8, 'BEESWARM PLOT: Season Overview',
                fontsize=14, fontweight='bold', ha='center', color=dark_gray)
    ax_main.text(3.8, 7.4, '"How Each Factor Impacts All Matches"',
                fontsize=11, ha='center', color=medium_gray, style='italic')
    
    # Generate synthetic beeswarm data
    np.random.seed(42)
    features = ['Player Fitness', 'Home Advantage', 'Recent Form', 
                'Opposition Strength', 'Weather Conditions', 'Rest Days', 
                'Key Injuries', 'Head-to-Head Record']
    n_samples = 50
    
    # Create realistic SHAP values for each feature
    shap_data = []
    for i, feature in enumerate(features):
        if feature in ['Player Fitness', 'Home Advantage', 'Recent Form']:
            # Mostly positive impact
            values = np.random.normal(0.08, 0.04, n_samples)
            values = np.clip(values, -0.02, 0.2)
        elif feature in ['Opposition Strength', 'Key Injuries']:
            # Mostly negative impact
            values = np.random.normal(-0.06, 0.03, n_samples)
            values = np.clip(values, -0.15, 0.02)
        else:
            # Mixed impact
            values = np.random.normal(0, 0.05, n_samples)
            values = np.clip(values, -0.1, 0.1)
        
        # Add jitter for y-axis to create beeswarm effect
        y_positions = i + np.random.normal(0, 0.15, n_samples)
        y_positions = np.clip(y_positions, i - 0.3, i + 0.3)
        
        # Color based on feature value (for this example, use performance metric)
        colors = plt.cm.RdYlGn(0.5 + values * 2.5)
        
        # Plot the beeswarm
        ax_bee.scatter(values, y_positions, c=colors, s=20, alpha=0.6, edgecolor='none')
    
    # Customize beeswarm plot
    ax_bee.set_yticks(range(len(features)))
    ax_bee.set_yticklabels(features, fontsize=10)
    ax_bee.set_xlabel('Impact on Win Probability', fontsize=11, color=medium_gray)
    ax_bee.set_xlim(-0.15, 0.2)
    ax_bee.axvline(x=0, color=dark_gray, linestyle='-', linewidth=1, alpha=0.5)
    
    # Add impact labels
    ax_bee.text(-0.075, 7.5, '← Hurts Arsenal', fontsize=9, ha='center', color=negative_red)
    ax_bee.text(0.075, 7.5, 'Helps Arsenal →', fontsize=9, ha='center', color=positive_green)
    
    # Grid for readability
    ax_bee.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax_bee.set_facecolor(light_gray)
    
    # Add colorbar legend for beeswarm
    ax_cb = fig.add_axes([0.48, 0.35, 0.015, 0.15])
    cb_data = np.linspace(0, 1, 100).reshape(-1, 1)
    ax_cb.imshow(cb_data, cmap='RdYlGn', aspect='auto')
    ax_cb.set_xticks([])
    ax_cb.set_yticks([0, 50, 100])
    ax_cb.set_yticklabels(['Poor', 'Avg', 'Good'], fontsize=8)
    ax_cb.set_ylabel('Performance', fontsize=9, color=medium_gray)
    
    # Beeswarm explanation box
    bee_explain_bg = FancyBboxPatch((0.5, 1.5), 7, 1.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#E8F4FF', edgecolor=primary_blue,
                                    linewidth=1.5)
    ax_main.add_patch(bee_explain_bg)
    
    ax_main.text(3.75, 3.0, 'Reading the Beeswarm:', fontsize=11, fontweight='bold', color=primary_blue)
    explanations_bee = [
        '• Each dot = one match in the season',
        '• Position left/right = hurt or helped win probability',
        '• Color = player/factor performance level that match',
        '• Spread shows consistency across matches'
    ]
    
    y_pos = 2.6
    for exp in explanations_bee:
        ax_main.text(0.8, y_pos, exp, fontsize=9, color=dark_gray)
        y_pos -= 0.3
    
    # ==================== RIGHT PANEL: WATERFALL PLOT ====================
    ax_water = fig.add_axes([0.56, 0.35, 0.38, 0.35])
    
    # Title for waterfall section
    ax_main.text(12, 7.8, 'WATERFALL PLOT: Single Match Breakdown',
                fontsize=14, fontweight='bold', ha='center', color=dark_gray)
    ax_main.text(12, 7.4, '"Arsenal vs Chelsea - Step-by-Step Prediction"',
                fontsize=11, ha='center', color=medium_gray, style='italic')
    
    # Waterfall data
    categories = ['Base Rate', 'Saka Form\n(+15%)', 'Home Adv.\n(+8%)', 
                  'Recent Form\n(+5%)', 'Weather\n(-2%)', 'Chelsea Str.\n(-11%)', 'Final']
    values = [45, 15, 8, 5, -2, -11, 0]  # Last value is calculated
    cumulative = []
    current = 45
    for val in values[1:-1]:
        cumulative.append(current)
        current += val
    cumulative.append(current)
    
    # Create waterfall bars
    x_pos = np.arange(len(categories))
    
    # Base rate bar
    ax_water.bar(0, values[0], color=primary_blue, alpha=0.7, width=0.6)
    
    # Incremental bars
    for i in range(1, len(values) - 1):
        color = positive_green if values[i] > 0 else negative_red
        bottom = cumulative[i-1]
        ax_water.bar(i, values[i], bottom=bottom, color=color, alpha=0.7, width=0.6)
        
        # Add connecting lines
        if i < len(values) - 2:
            ax_water.plot([i + 0.3, i + 0.7], [cumulative[i], cumulative[i]], 
                         'k--', alpha=0.3, linewidth=1)
        
        # Add value labels
        y_pos_label = bottom + values[i]/2
        ax_water.text(i, y_pos_label, f'{values[i]:+.0f}%', 
                     ha='center', va='center', fontsize=9, 
                     fontweight='bold', color='white')
    
    # Final prediction bar
    final_value = cumulative[-1]
    ax_water.bar(len(categories) - 1, final_value, color=accent_orange, alpha=0.8, width=0.6)
    ax_water.text(len(categories) - 1, final_value/2, f'{final_value:.0f}%', 
                 ha='center', va='center', fontsize=10, 
                 fontweight='bold', color='white')
    
    # Customize waterfall plot
    ax_water.set_xticks(x_pos)
    ax_water.set_xticklabels(categories, fontsize=9, rotation=0)
    ax_water.set_ylabel('Win Probability (%)', fontsize=10, color=medium_gray)
    ax_water.set_ylim(0, 80)
    ax_water.grid(True, axis='y', alpha=0.2, linestyle='--')
    ax_water.set_facecolor(light_gray)
    
    # Add threshold line
    ax_water.axhline(y=50, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_water.text(6.2, 51, 'Even odds', fontsize=8, color=medium_gray, style='italic')
    
    # Waterfall explanation box
    water_explain_bg = FancyBboxPatch((8.5, 1.5), 7, 1.8,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#FFF5E6', edgecolor=accent_orange,
                                      linewidth=1.5)
    ax_main.add_patch(water_explain_bg)
    
    ax_main.text(12, 3.0, 'Reading the Waterfall:', fontsize=11, fontweight='bold', color=accent_orange)
    explanations_water = [
        '• Start with baseline prediction (league average)',
        '• Add/subtract each factor\'s contribution',
        '• Green = positive factors, Red = negative factors',
        '• Orange bar = final match prediction'
    ]
    
    y_pos = 2.6
    for exp in explanations_water:
        ax_main.text(9, y_pos, exp, fontsize=9, color=dark_gray)
        y_pos -= 0.3
    
    # ==================== BOTTOM INSIGHT SECTION ====================
    insight_bg = FancyBboxPatch((2, 0.3), 12, 0.8,
                                boxstyle="round,pad=0.03",
                                facecolor=primary_blue, alpha=0.1,
                                edgecolor=primary_blue, linewidth=1.5)
    ax_main.add_patch(insight_bg)
    
    ax_main.text(8, 0.85, 'Key Insight:', fontsize=11, fontweight='bold', color=primary_blue)
    ax_main.text(8, 0.5, 'SHAP helps us understand WHY the AI made its prediction - just like a coach explains match tactics!',
                fontsize=10, ha='center', color=dark_gray, style='italic')
    
    # Save as SVG with text elements preserved
    plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_shap_soccer_educational.svg', 
                format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/enbel_shap_soccer_educational.png', 
                format='png', dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Created SHAP educational slide with soccer analogies:")
    print("   - enbel_shap_soccer_educational.svg (with editable text)")
    print("   - enbel_shap_soccer_educational.png")
    
    return fig

if __name__ == "__main__":
    # Create the educational slide
    fig = create_shap_soccer_educational_slide()
    plt.show()