#!/usr/bin/env python3
"""
Create educational SHAP visualization slide with soccer/football analogies
Matches the style of the ENBEL presentation slides exactly
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set font to ensure text remains as text in SVG
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Color palette matching reference
HEADER_BLUE = '#2E5894'
POSITIVE_GREEN = '#52A373'
NEGATIVE_RED = '#E74C3C'
NEUTRAL_GRAY = '#95A5A6'
TEXT_DARK = '#2C3E50'
LIGHT_GRAY = '#ECF0F1'
WHITE = '#FFFFFF'

def create_waterfall_plot(ax, player_title, features, values, base_value, final_value):
    """Create a horizontal waterfall plot for individual player performance analysis"""
    
    # Clear axes
    ax.clear()
    ax.set_facecolor('white')
    
    # Calculate cumulative values
    cumulative = [base_value]
    for v in values:
        cumulative.append(cumulative[-1] + v)
    
    # Plot horizontal bars
    y_positions = list(range(len(features) + 2))
    y_positions.reverse()
    
    # Base bar
    ax.barh(y_positions[0], base_value, height=0.6, 
            color=NEUTRAL_GRAY, alpha=0.7, edgecolor='none')
    ax.text(base_value/2, y_positions[0], f'{base_value:.1f}/10', 
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Feature bars
    for i, (feat, val) in enumerate(zip(features, values)):
        color = POSITIVE_GREEN if val > 0 else NEGATIVE_RED
        start = cumulative[i]
        ax.barh(y_positions[i+1], val, left=start, height=0.6,
                color=color, alpha=0.8, edgecolor='none')
        
        # Add value label
        label_x = start + val/2
        ax.text(label_x, y_positions[i+1], f'{val:+.1f}', 
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Connect with line
        if i < len(features) - 1:
            ax.plot([cumulative[i+1], cumulative[i+1]], 
                   [y_positions[i+1]-0.35, y_positions[i+2]+0.35],
                   color=NEUTRAL_GRAY, linestyle='--', alpha=0.5, linewidth=1)
    
    # Final prediction bar
    ax.barh(y_positions[-1], final_value, height=0.6,
            color='#34495E', alpha=0.9, edgecolor='none')
    ax.text(final_value/2, y_positions[-1], f'{final_value:.1f}/10',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Labels
    y_labels = ['Average Player\nRating'] + features + ['Final\nRating']
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # Formatting
    ax.set_xlim(0, 10)
    ax.set_xlabel('Player Rating (0-10)', fontsize=9)
    ax.set_title(player_title, fontsize=10, fontweight='bold', pad=10)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add vertical line at 6.5 (average rating)
    ax.axvline(x=6.5, color=NEUTRAL_GRAY, linestyle=':', alpha=0.5, linewidth=1)

def create_beeswarm_plot(ax, plot_title, features, show_player_focus=True):
    """Create a horizontal beeswarm plot showing feature importance across matches"""
    
    # Clear axes
    ax.clear()
    ax.set_facecolor('white')
    
    np.random.seed(42)  # For reproducibility
    n_matches = 30  # Number of matches in season
    
    # Generate synthetic SHAP values for each feature
    shap_data = []
    feature_values = []
    
    for i, feat in enumerate(features):
        # Generate SHAP values with different distributions for each feature
        if i == 0:  # Most important feature
            values = np.random.normal(0.5, 0.3, n_matches)
        elif i == 1:
            values = np.random.normal(0.2, 0.25, n_matches)
        elif i == 2:
            values = np.random.normal(-0.1, 0.2, n_matches)
        else:
            values = np.random.normal(0, 0.15, n_matches)
        
        # Clip values
        values = np.clip(values, -1, 1)
        shap_data.append(values)
        
        # Generate feature values (for coloring)
        if show_player_focus:
            # Player performance metrics (0-100 scale)
            feat_vals = np.random.uniform(20, 100, n_matches)
        else:
            # Context factors (varied scales)
            feat_vals = np.random.uniform(0, 1, n_matches)
        feature_values.append(feat_vals)
    
    # Plot beeswarm
    y_positions = list(range(len(features)))
    y_positions.reverse()
    
    for i, (feat, shap_vals, feat_vals) in enumerate(zip(features, shap_data, feature_values)):
        # Add jitter to y position for beeswarm effect
        y_jitter = np.random.uniform(-0.2, 0.2, len(shap_vals))
        y_pos = [y_positions[i] + jitter for jitter in y_jitter]
        
        # Normalize feature values for coloring
        norm_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
        
        # Create color array
        colors = []
        for nv in norm_vals:
            if nv > 0.6:
                colors.append(NEGATIVE_RED)
            elif nv > 0.4:
                colors.append('#F39C12')  # Orange
            else:
                colors.append('#3498DB')  # Blue
        
        # Plot dots
        scatter = ax.scatter(shap_vals, y_pos, c=colors, s=30, alpha=0.7, edgecolors='none')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color=TEXT_DARK, linestyle='-', alpha=0.3, linewidth=1.5)
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Impact on Prediction (SHAP value)', fontsize=9)
    ax.set_title(plot_title, fontsize=10, fontweight='bold', pad=10)
    
    # Formatting
    ax.set_xlim(-1, 1)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add legend for color meaning
    if show_player_focus:
        legend_title = "Performance"
        labels = ['High (80-100)', 'Med (50-80)', 'Low (0-50)']
    else:
        legend_title = "Factor Level"
        labels = ['High', 'Medium', 'Low']
    
    colors = [NEGATIVE_RED, '#F39C12', '#3498DB']
    legend_elements = [plt.scatter([], [], c=c, s=30, alpha=0.7, label=l) 
                      for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, title=legend_title, loc='lower right', 
             fontsize=8, title_fontsize=8, framealpha=0.9, edgecolor='none')

def create_slide():
    """Create the complete educational SHAP slide"""
    
    # Create figure with specific size for presentation
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(WHITE)
    
    # Create main axes for the slide
    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.set_xlim(0, 1)
    main_ax.set_ylim(0, 1)
    main_ax.axis('off')
    
    # Add header bar
    header_rect = Rectangle((0, 0.92), 1, 0.08, facecolor=HEADER_BLUE, 
                           edgecolor='none', transform=main_ax.transData)
    main_ax.add_patch(header_rect)
    
    # Add header text
    main_ax.text(0.5, 0.96, "Understanding SHAP: Individual Player Performance Analysis",
                ha='center', va='center', fontsize=18, color=WHITE, 
                fontweight='bold', transform=main_ax.transData)
    
    # Add subtitle
    main_ax.text(0.5, 0.88, "How each factor contributes to predicting individual player ratings",
                ha='center', va='center', fontsize=11, color=TEXT_DARK,
                style='italic', transform=main_ax.transData)
    
    # Create 2x2 grid of plots
    # Top left: Waterfall plot for Saka's performance analysis
    ax1 = fig.add_axes([0.05, 0.48, 0.42, 0.35])
    features1 = ['Personal Fitness\nLevel', 'Recent Form\n(goals/assists)', 'Opposition\nDefensive Strength', 'Playing Position\n(wing vs central)', 'Team Support\n(Ødegaard partnership)']
    values1 = [0.8, 0.6, -0.4, 0.3, 0.5]
    create_waterfall_plot(ax1, "Individual Analysis: Saka's Match Rating Prediction",
                         features1, values1, base_value=6.5, final_value=7.8)
    
    # Top right: Waterfall plot for Ødegaard's performance analysis
    ax2 = fig.add_axes([0.53, 0.48, 0.42, 0.35])
    features2 = ['Creative Freedom\n(tactical setup)', 'Physical\nCondition', 'Opposition\nMidfield Pressure', 'Team Possession\nStyle', 'Match Importance\n(pressure)']
    values2 = [0.5, 0.4, -0.6, 0.3, 0.1]
    create_waterfall_plot(ax2, "Individual Analysis: Ødegaard's Match Rating Prediction",
                         features2, values2, base_value=6.5, final_value=7.2)
    
    # Bottom left: Beeswarm plot for players
    ax3 = fig.add_axes([0.05, 0.08, 0.42, 0.35])
    player_features = ['Saka\nPerformance', 'Odegaard\nCreativity', 'Rice\nDefensive', 'Gabriel\nHeaders', 'Ramsdale\nSaves']
    create_beeswarm_plot(ax3, "Season Overview: Star Players' Impact", 
                        player_features, show_player_focus=True)
    
    # Bottom right: Beeswarm plot for context
    ax4 = fig.add_axes([0.53, 0.08, 0.42, 0.35])
    context_features = ['Home/Away', 'Opposition\nRank', 'Fixture\nCongestion', 'Weather\nConditions', 'Injury\nCount']
    create_beeswarm_plot(ax4, "Season Overview: Context Factors",
                        context_features, show_player_focus=False)
    
    # Add explanatory text at bottom
    explanation = ("SHAP (SHapley Additive exPlanations) shows how each factor contributes to individual player rating predictions. "
                  "Waterfall plots show single match analysis, while beeswarm plots show patterns across the season.")
    main_ax.text(0.5, 0.03, explanation,
                ha='center', va='center', fontsize=10, color=TEXT_DARK,
                wrap=True, transform=main_ax.transData)
    
    # Add small flow diagram at bottom
    flow_y = 0.01
    flow_elements = ["Personal Factors", "→", "Contribute to Rating", "→", "Individual Performance"]
    flow_x_positions = [0.15, 0.28, 0.42, 0.58, 0.75]
    
    for i, (text, x) in enumerate(zip(flow_elements, flow_x_positions)):
        if text == "→":
            main_ax.text(x, flow_y, text, ha='center', va='center', 
                        fontsize=12, color=HEADER_BLUE, fontweight='bold',
                        transform=main_ax.transData)
        else:
            # Add small box around text
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=LIGHT_GRAY, 
                            edgecolor=HEADER_BLUE, linewidth=1, alpha=0.8)
            main_ax.text(x, flow_y, text, ha='center', va='center',
                        fontsize=9, color=TEXT_DARK, bbox=bbox_props,
                        transform=main_ax.transData)
    
    return fig

# Create and save the slide
fig = create_slide()

# Save as SVG
svg_filename = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/shap_soccer_education_slide.svg'
fig.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"SVG saved as: {svg_filename}")

# Save as PNG
png_filename = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/shap_soccer_education_slide.png'
fig.savefig(png_filename, format='png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print(f"PNG saved as: {png_filename}")

plt.close(fig)
print("Slide generation completed successfully!")