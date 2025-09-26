#!/usr/bin/env python3
"""
Create CORRECTED SHAP Interpretation Guide
Based on the actual beeswarm plots showing statistical significance patterns
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

def create_corrected_interpretation():
    """
    Create corrected interpretation based on actual analysis results
    """
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('#F8FAFC')
    
    # Main title
    fig.suptitle('Understanding Your SHAP Results: Why "Non-Significant" Features Matter', 
                fontsize=22, weight='bold', y=0.96, color=colors['blue'])
    fig.text(0.5, 0.92, 'How to interpret beeswarm plots when exploring the feature space systematically',
            fontsize=13, ha='center', style='italic', color=colors['gray'])
    
    # Create three-panel layout
    ax_problem = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    ax_interpretation = plt.subplot2grid((2, 3), (0, 1), colspan=1) 
    ax_insight = plt.subplot2grid((2, 3), (0, 2), colspan=1)
    ax_workflow = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    
    # === PROBLEM EXPLANATION ===
    ax_problem.axis('off')
    problem_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor='#FFF5F5', edgecolor=colors['red'],
                                linewidth=2, alpha=0.95)
    ax_problem.add_patch(problem_box)
    
    ax_problem.text(0.5, 0.9, 'The Confusion', fontsize=14, weight='bold',
                   ha='center', color=colors['red'], transform=ax_problem.transAxes)
    
    problem_text = [
        "You're seeing beeswarm plots",
        "where many features cluster",
        "near zero SHAP values.",
        "",
        "This doesn't mean no relationship!",
        "",
        "It means SYSTEMATIC",
        "EXPLORATION found most",
        "features have weak or",
        "inconsistent effects.",
        "",
        "This IS the discovery process!"
    ]
    
    y_pos = 0.8
    for line in problem_text:
        if line == "":
            y_pos -= 0.04
        else:
            weight = 'bold' if 'SYSTEMATIC' in line or 'discovery' in line else 'normal'
            size = 10 if weight == 'normal' else 11
            ax_problem.text(0.1, y_pos, line, fontsize=size, weight=weight,
                           color='#2c3e50', transform=ax_problem.transAxes)
            y_pos -= 0.06
    
    # === CORRECT INTERPRETATION ===
    ax_interpretation.axis('off')
    interp_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                               boxstyle="round,pad=0.02",
                               facecolor=colors['lightblue'], edgecolor=colors['blue'],
                               linewidth=2, alpha=0.95)
    ax_interpretation.add_patch(interp_box)
    
    ax_interpretation.text(0.5, 0.9, 'What This Actually Shows', fontsize=14, weight='bold',
                          ha='center', color=colors['blue'], transform=ax_interpretation.transAxes)
    
    interp_text = [
        "1. Feature Space Explored:",
        "   1,092 climate features tested",
        "",
        "2. Systematic Filtering:",
        "   XAI → Statistical → DLNM",
        "",
        "3. Most features near zero:",
        "   Expected! Most climate",
        "   variables don't strongly",
        "   predict health outcomes",
        "",
        "4. The few that spread wide:",
        "   These ARE your discoveries!",
        "   (Temperature lags, etc.)"
    ]
    
    y_pos = 0.8
    for line in interp_text:
        if line == "":
            y_pos -= 0.03
        elif line.startswith(('1.', '2.', '3.', '4.')):
            ax_interpretation.text(0.1, y_pos, line, fontsize=10, weight='bold',
                                  color=colors['blue'], transform=ax_interpretation.transAxes)
            y_pos -= 0.05
        else:
            ax_interpretation.text(0.15, y_pos, line.strip(), fontsize=9,
                                  color='#2c3e50', transform=ax_interpretation.transAxes)
            y_pos -= 0.05
    
    # === KEY INSIGHT ===
    ax_insight.axis('off')
    insight_box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor='#F0FFF0', edgecolor=colors['green'],
                                linewidth=2, alpha=0.95)
    ax_insight.add_patch(insight_box)
    
    ax_insight.text(0.5, 0.9, 'The Discovery Value', fontsize=14, weight='bold',
                   ha='center', color=colors['green'], transform=ax_insight.transAxes)
    
    insight_text = [
        "XAI as Exploration Tool:",
        "",
        "✓ Tested 1,092 features",
        "✓ Most showed weak effects", 
        "✓ Identified key patterns:",
        "   • Lag-21 cardiovascular",
        "   • Immediate glucose",
        "   • SES interactions",
        "",
        "Without XAI systematic",
        "exploration, you'd miss",
        "these specific patterns!",
        "",
        "The 'null' results validate",
        "the importance of the",
        "significant findings."
    ]
    
    y_pos = 0.8
    for line in insight_text:
        if line == "":
            y_pos -= 0.03
        elif line.startswith('✓'):
            ax_insight.text(0.1, y_pos, line, fontsize=10,
                           color=colors['green'], transform=ax_insight.transAxes)
            y_pos -= 0.05
        elif 'XAI' in line or 'null' in line:
            ax_insight.text(0.1, y_pos, line, fontsize=10, weight='bold',
                           color=colors['green'], transform=ax_insight.transAxes)
            y_pos -= 0.05
        else:
            ax_insight.text(0.1, y_pos, line, fontsize=9,
                           color='#2c3e50', transform=ax_insight.transAxes)
            y_pos -= 0.05
    
    # === WORKFLOW DIAGRAM ===
    ax_workflow.axis('off')
    workflow_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#FAFAFA', edgecolor=colors['purple'],
                                 linewidth=2, alpha=0.95)
    ax_workflow.add_patch(workflow_box)
    
    ax_workflow.text(0.5, 0.85, 'Your XAI Discovery Workflow in Practice', fontsize=16, weight='bold',
                    ha='center', color=colors['purple'], transform=ax_workflow.transAxes)
    
    # Workflow steps
    steps = [
        ("1,092\nFeatures", "Generate", 0.15, colors['blue']),
        ("SHAP\nAnalysis", "Explore", 0.35, colors['orange']), 
        ("~200\nCandidates", "Filter", 0.55, colors['green']),
        ("Statistical\nValidation", "Validate", 0.75, colors['red']),
        ("Key\nFindings", "Discover", 0.9, colors['purple'])
    ]
    
    # Draw workflow
    y_center = 0.45
    for i, (label, action, x, color) in enumerate(steps):
        # Draw box
        box = FancyBboxPatch((x-0.06, y_center-0.1), 0.12, 0.2,
                            boxstyle="round,pad=0.01",
                            facecolor=color, alpha=0.2,
                            edgecolor=color, linewidth=2)
        ax_workflow.add_patch(box)
        
        # Add text
        ax_workflow.text(x, y_center+0.05, label, ha='center', va='center',
                        fontsize=10, weight='bold', color=color,
                        transform=ax_workflow.transAxes)
        ax_workflow.text(x, y_center-0.05, action, ha='center', va='center',
                        fontsize=9, style='italic', color=color,
                        transform=ax_workflow.transAxes)
        
        # Add arrow
        if i < len(steps) - 1:
            next_x = steps[i+1][2]
            ax_workflow.annotate('', xy=(next_x-0.06, y_center), 
                               xytext=(x+0.06, y_center),
                               arrowprops=dict(arrowstyle='->', lw=2, color=colors['gray']))
    
    # Add explanatory text below workflow
    explanations = [
        "Most features cluster near zero → Expected for comprehensive exploration",
        "Few features show wide spread → These are your discoveries!",
        "Beeswarm width indicates importance, not statistical significance"
    ]
    
    y_exp = 0.25
    for exp in explanations:
        ax_workflow.text(0.5, y_exp, f"• {exp}", ha='center', fontsize=11,
                        color='#2c3e50', transform=ax_workflow.transAxes)
        y_exp -= 0.05
    
    # Add bottom key message
    fig.text(0.5, 0.02, 'Key Message: Your beeswarm plots show successful systematic exploration - most features have weak effects, validating the importance of the few strong patterns discovered',
            fontsize=12, ha='center', weight='bold', style='italic', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['lightblue'], alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.91])
    
    # Save
    plt.savefig('enbel_corrected_shap_interpretation.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('enbel_corrected_shap_interpretation.png', format='png', bbox_inches='tight', dpi=150)
    print("Corrected SHAP interpretation saved as 'enbel_corrected_shap_interpretation.svg' and '.png'")
    
    plt.show()

if __name__ == "__main__":
    create_corrected_interpretation()