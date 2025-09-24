#!/usr/bin/env python3
"""
Create SHAP Waterfall Plots for Climate-Health Analysis
Shows individual prediction explanations for presentation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
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

def create_waterfall_plot(ax, features, values, base_value, prediction, title, case_description):
    """
    Create a SHAP waterfall plot for a single prediction
    """
    # Sort features by absolute impact
    sorted_idx = np.argsort(np.abs(values))[::-1]
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    # Limit to top features for clarity
    max_features = 8
    if len(features) > max_features:
        other_sum = sum(values[max_features:])
        features = features[:max_features] + ['Other features']
        values = values[:max_features] + [other_sum]
    
    # Calculate cumulative values for waterfall
    cumulative = [base_value]
    for val in values:
        cumulative.append(cumulative[-1] + val)
    
    # Plot setup
    ax.set_title(title, fontsize=14, weight='bold', pad=10, color=colors['blue'])
    ax.text(0.5, 1.02, case_description, fontsize=10, style='italic', 
            ha='center', transform=ax.transAxes, color=colors['gray'])
    
    # Draw waterfall bars
    bar_height = 0.6
    y_positions = range(len(features))
    
    for i, (feature, value, cum_start) in enumerate(zip(features, values, cumulative[:-1])):
        # Determine color based on positive/negative impact
        color = colors['red'] if value > 0 else colors['green']
        alpha = 0.8 if feature != 'Other features' else 0.5
        
        # Draw the bar
        bar = Rectangle((min(cum_start, cum_start + value), i - bar_height/2),
                       abs(value), bar_height, 
                       facecolor=color, alpha=alpha, edgecolor='white', linewidth=1)
        ax.add_patch(bar)
        
        # Add value label
        label_x = cum_start + value/2
        ax.text(label_x, i, f'{value:+.3f}', ha='center', va='center',
               fontsize=9, weight='bold', color='white')
        
        # Add connecting line to next bar
        if i < len(features) - 1:
            ax.plot([cum_start + value, cum_start + value], 
                   [i + bar_height/2, i + 0.9],
                   'k--', alpha=0.3, linewidth=0.5)
    
    # Add feature labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features, fontsize=10)
    
    # Add base and prediction values
    ax.axvline(x=base_value, color=colors['gray'], linestyle='--', alpha=0.5, linewidth=1)
    ax.text(base_value, -0.8, f'Base: {base_value:.3f}', ha='center', 
            fontsize=9, color=colors['gray'])
    
    ax.axvline(x=prediction, color=colors['blue'], linestyle='-', alpha=0.8, linewidth=2)
    ax.text(prediction, len(features) + 0.2, f'Prediction: {prediction:.3f}', 
            ha='center', fontsize=10, weight='bold', color=colors['blue'])
    
    # Formatting
    ax.set_xlabel('Impact on Blood Pressure (mmHg)', fontsize=10)
    ax.set_xlim(min(base_value, prediction) - 2, max(base_value, prediction) + 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)

def create_waterfall_examples():
    """
    Create multiple waterfall plot examples
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#F8FAFC')
    
    # Main title
    fig.suptitle('SHAP Waterfall Plots: Individual Prediction Explanations', 
                fontsize=24, weight='bold', y=0.98, color=colors['blue'])
    fig.text(0.5, 0.94, 'How each feature contributes to individual climate-health predictions',
            fontsize=14, ha='center', style='italic', color=colors['gray'])
    
    # Example 1: High-risk individual during heat wave
    ax1 = plt.subplot(2, 2, 1)
    features1 = ['Heat index (lag 0)', 'Age > 65', 'Economic vulnerability', 
                'Heat index (lag 7)', 'Previous BP', 'Male', 'Humidity', 'Education']
    values1 = np.array([8.5, 4.2, 3.8, 2.1, 1.5, -0.8, 1.2, -0.5])
    base1 = 120.0
    pred1 = base1 + sum(values1)
    create_waterfall_plot(ax1, features1, values1, base1, pred1,
                         'Case 1: Vulnerable Individual During Heat Wave',
                         'Elderly male with economic vulnerability, Day 0 of heat wave')
    
    # Example 2: Resilient individual with protective factors
    ax2 = plt.subplot(2, 2, 2)
    features2 = ['Temperature (lag 21)', 'Higher education', 'Good housing', 
                'Young age', 'Heat index (lag 0)', 'Female', 'Employment', 'Previous BP']
    values2 = np.array([3.2, -4.5, -3.2, -2.8, 2.1, -1.5, -1.8, 0.8])
    base2 = 120.0
    pred2 = base2 + sum(values2)
    create_waterfall_plot(ax2, features2, values2, base2, pred2,
                         'Case 2: Protected Individual with Adaptive Capacity',
                         'Young female with good SES, 3 weeks post-heat exposure')
    
    # Example 3: Moderate risk with mixed factors
    ax3 = plt.subplot(2, 2, 3)
    features3 = ['Heat index (lag 3)', 'Moderate vulnerability', 'Temperature range',
                'Age 45-65', 'Wind speed', 'Precipitation', 'Urban residence', 'BMI']
    values3 = np.array([4.5, 2.3, 1.8, 1.2, -0.9, -1.5, 0.8, 0.6])
    base3 = 120.0
    pred3 = base3 + sum(values3)
    create_waterfall_plot(ax3, features3, values3, base3, pred3,
                         'Case 3: Moderate Risk with Mixed Factors',
                         'Middle-aged urban resident, 3 days post-heat exposure')
    
    # Example 4: Metabolic impact (glucose)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Case 4: Metabolic Response to Heat Stress', 
                 fontsize=14, weight='bold', pad=10, color=colors['blue'])
    ax4.text(0.5, 0.94, 'Diabetic patient during sustained heat period',
            fontsize=10, style='italic', ha='center', 
            transform=ax4.transAxes, color=colors['gray'])
    
    # Different scale for glucose
    features4 = ['Heat duration', 'Diabetic status', 'Heat index (lag 1)',
                'Poor housing', 'Dehydration risk', 'Age > 65', 'Medication', 'Activity level']
    values4 = np.array([12.5, 8.3, 6.2, 4.8, 3.5, 2.1, -5.2, -2.8])
    base4 = 95.0  # Baseline glucose
    pred4 = base4 + sum(values4)
    
    # Modified plot for glucose
    sorted_idx = np.argsort(np.abs(values4))[::-1]
    features4_sorted = [features4[i] for i in sorted_idx]
    values4_sorted = [values4[i] for i in sorted_idx]
    
    cumulative = [base4]
    for val in values4_sorted:
        cumulative.append(cumulative[-1] + val)
    
    bar_height = 0.6
    y_positions = range(len(features4_sorted))
    
    for i, (feature, value, cum_start) in enumerate(zip(features4_sorted, values4_sorted, cumulative[:-1])):
        color = colors['orange'] if value > 0 else colors['green']
        bar = Rectangle((min(cum_start, cum_start + value), i - bar_height/2),
                       abs(value), bar_height, 
                       facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
        ax4.add_patch(bar)
        
        label_x = cum_start + value/2
        ax4.text(label_x, i, f'{value:+.1f}', ha='center', va='center',
                fontsize=9, weight='bold', color='white')
        
        if i < len(features4_sorted) - 1:
            ax4.plot([cum_start + value, cum_start + value], 
                    [i + bar_height/2, i + 0.9],
                    'k--', alpha=0.3, linewidth=0.5)
    
    ax4.set_yticks(y_positions)
    ax4.set_yticklabels(features4_sorted, fontsize=10)
    ax4.axvline(x=base4, color=colors['gray'], linestyle='--', alpha=0.5, linewidth=1)
    ax4.text(base4, -0.8, f'Base: {base4:.0f}', ha='center', fontsize=9, color=colors['gray'])
    ax4.axvline(x=pred4, color=colors['orange'], linestyle='-', alpha=0.8, linewidth=2)
    ax4.text(pred4, len(features4_sorted) + 0.2, f'Prediction: {pred4:.0f}', 
            ha='center', fontsize=10, weight='bold', color=colors['orange'])
    ax4.set_xlabel('Impact on Glucose (mg/dL)', fontsize=10)
    ax4.set_xlim(base4 - 10, pred4 + 10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='x', alpha=0.2)
    
    # Add interpretation note at bottom
    fig.text(0.5, 0.02, 'Red/Orange bars increase the prediction | Green bars decrease the prediction | Width shows impact magnitude',
            fontsize=11, ha='center', style='italic', color=colors['gray'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Save
    plt.savefig('enbel_shap_waterfall_plots.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('enbel_shap_waterfall_plots.png', format='png', bbox_inches='tight', dpi=150)
    print("SHAP waterfall plots saved as 'enbel_shap_waterfall_plots.svg' and '.png'")
    
    plt.show()

if __name__ == "__main__":
    create_waterfall_examples()