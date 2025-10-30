#!/usr/bin/env python3
"""
Create SVG visualization slide for patient-level Temperature×Vulnerability interaction
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def create_interaction_slide():
    """Create professional slide showing patient-level interaction validation"""

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')

    # Main title
    fig.text(0.5, 0.95, 'Patient-Level Validation: Temperature × Vulnerability Interaction',
             fontsize=26, fontweight='bold', ha='center', color='#1a4e8a')
    fig.text(0.5, 0.91, 'SHAP Findings Confirmed with Statistical Rigor | n=2,917 patients',
             fontsize=16, ha='center', color='#555', style='italic')

    # Left panel: Main finding
    ax1 = fig.add_axes([0.05, 0.45, 0.43, 0.40])
    ax1.set_facecolor('#fff9f0')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    rect1 = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=3,
                             edgecolor='#ff6b00', facecolor='none')
    ax1.add_patch(rect1)

    ax1.text(0.5, 0.92, 'KEY FINDING: CHOLESTEROL INTERACTION', fontsize=18,
            fontweight='bold', ha='center', color='#ff6b00')

    ax1.text(0.5, 0.80, 'Temperature × Vulnerability: p < 0.001***',
            fontsize=16, fontweight='bold', ha='center', color='#c62828')

    # Key statistics
    ax1.text(0.08, 0.68, '• Sample:', fontsize=13, color='#333')
    ax1.text(0.30, 0.68, '2,917 patients', fontsize=13, fontweight='bold', color='#000')

    ax1.text(0.08, 0.60, '• Interaction coef:', fontsize=13, color='#333')
    ax1.text(0.30, 0.60, '5.001 (SE=0.888)', fontsize=13, fontweight='bold', color='#000')

    ax1.text(0.08, 0.52, '• t-statistic:', fontsize=13, color='#333')
    ax1.text(0.30, 0.52, 't = 5.63', fontsize=13, fontweight='bold', color='#000')

    ax1.text(0.08, 0.44, '• P-value:', fontsize=13, color='#333')
    ax1.text(0.30, 0.44, 'p < 0.001***', fontsize=13, fontweight='bold', color='#c62828')

    ax1.text(0.08, 0.36, '• Likelihood ratio:', fontsize=13, color='#333')
    ax1.text(0.30, 0.36, 'χ² = 31.65, p < 0.001', fontsize=13, fontweight='bold', color='#c62828')

    ax1.text(0.08, 0.28, '• AIC improvement:', fontsize=13, color='#333')
    ax1.text(0.30, 0.28, 'ΔAIC = -31.1', fontsize=13, fontweight='bold', color='#1565c0')

    # Effect size box
    effect_box = patches.FancyBboxPatch((0.06, 0.08), 0.88, 0.14,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#ffe0b2', edgecolor='#ff6b00', linewidth=2)
    ax1.add_patch(effect_box)
    ax1.text(0.5, 0.16, 'Effect Size: High vulnerability patients show',
            fontsize=12, ha='center', color='#333')
    ax1.text(0.5, 0.11, '10× STRONGER cholesterol response to temperature',
            fontsize=12, fontweight='bold', ha='center', color='#ff6b00')

    # Right panel: Effect comparison
    ax2 = fig.add_axes([0.52, 0.45, 0.43, 0.40])
    ax2.set_facecolor('#f0f4ff')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    rect2 = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=3,
                             edgecolor='#1565c0', facecolor='none')
    ax2.add_patch(rect2)

    ax2.text(0.5, 0.92, 'TEMPERATURE EFFECT BY VULNERABILITY',
            fontsize=18, fontweight='bold', ha='center', color='#1565c0')

    # Bar chart visualization
    bar_ax = fig.add_axes([0.57, 0.58, 0.33, 0.20])
    categories = ['Low\nVulnerability', 'High\nVulnerability']
    effects = [-0.88, 9.12]
    colors = ['#2166ac', '#b2182b']

    bars = bar_ax.bar(categories, effects, color=colors, alpha=0.7, width=0.5)
    bar_ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    bar_ax.set_ylabel('Cholesterol Change (mg/dL per SD temp)', fontsize=11, fontweight='bold')
    bar_ax.set_ylim(-3, 12)
    bar_ax.grid(axis='y', alpha=0.3)
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, value in zip(bars, effects):
        height = bar.get_height()
        bar_ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom' if value > 0 else 'top',
                   fontsize=13, fontweight='bold')

    # Interpretation box
    interp_box = patches.FancyBboxPatch((0.06, 0.08), 0.88, 0.28,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#bbdefb', edgecolor='#1565c0', linewidth=2)
    ax2.add_patch(interp_box)
    ax2.text(0.5, 0.29, 'BIOLOGICAL INTERPRETATION:',
            fontsize=12, fontweight='bold', ha='center', color='#1565c0')
    ax2.text(0.5, 0.23, '• Low vulnerability: Cholesterol stable/decreases with heat',
            fontsize=11, ha='center', color='#333')
    ax2.text(0.5, 0.18, '• High vulnerability: Sharp cholesterol increase with heat',
            fontsize=11, ha='center', color='#333')
    ax2.text(0.5, 0.13, '→ Vulnerable populations show metabolic hypersensitivity',
            fontsize=11, ha='center', color='#333', style='italic', fontweight='bold')

    # Bottom panel: Methodological contribution
    ax3 = fig.add_axes([0.05, 0.05, 0.90, 0.32])
    ax3.set_facecolor('#f0f9f0')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    rect3 = patches.Rectangle((0.01, 0.01), 0.98, 0.98, linewidth=3,
                             edgecolor='#2e7d32', facecolor='none')
    ax3.add_patch(rect3)

    ax3.text(0.5, 0.88, 'RESOLVING THE VULNERABILITY PARADOX: Simpson\'s Paradox Detected',
            fontsize=18, fontweight='bold', ha='center', color='#2e7d32')

    # Three columns: Meta-regression vs Patient-level vs Implication

    # Column 1: Meta-regression problem
    col1_x = 0.18
    ax3.text(col1_x, 0.75, 'Study-Level Analysis',
            fontsize=14, fontweight='bold', ha='center', color='#c62828')
    ax3.text(col1_x, 0.68, '(Meta-Regression)',
            fontsize=11, ha='center', color='#666', style='italic')

    ax3.text(col1_x, 0.58, 'k = 3-7 studies', fontsize=11, ha='center', color='#333')
    ax3.text(col1_x, 0.52, 'Correlation: r = -0.891', fontsize=11, ha='center',
            color='#c62828', fontweight='bold')
    ax3.text(col1_x, 0.46, 'NOT significant (p=0.35)', fontsize=11, ha='center', color='#c62828')

    ax3.text(col1_x, 0.36, '❌ PARADOX:', fontsize=10, ha='center', color='#c62828',
            fontweight='bold')
    ax3.text(col1_x, 0.30, 'High vuln studies', fontsize=9, ha='center', color='#333')
    ax3.text(col1_x, 0.25, '→ WEAKER effects', fontsize=9, ha='center', color='#333',
            style='italic')

    ax3.text(col1_x, 0.15, 'Problem:', fontsize=9, ha='center', color='#c62828',
            fontweight='bold')
    ax3.text(col1_x, 0.10, 'Underpowered', fontsize=9, ha='center', color='#333')
    ax3.text(col1_x, 0.05, 'Ecological fallacy', fontsize=9, ha='center', color='#333')

    # Arrow
    arrow1 = patches.FancyArrowPatch((0.30, 0.50), (0.44, 0.50),
                                    arrowstyle='->', lw=3, color='#2e7d32',
                                    mutation_scale=30)
    ax3.add_patch(arrow1)
    ax3.text(0.37, 0.54, 'RESOLUTION', fontsize=10, ha='center',
            color='#2e7d32', fontweight='bold')

    # Column 2: Patient-level solution
    col2_x = 0.5
    ax3.text(col2_x, 0.75, 'Patient-Level Analysis',
            fontsize=14, fontweight='bold', ha='center', color='#1565c0')
    ax3.text(col2_x, 0.68, '(Mixed Effects)',
            fontsize=11, ha='center', color='#666', style='italic')

    ax3.text(col2_x, 0.58, 'n = 2,917 patients', fontsize=11, ha='center', color='#333')
    ax3.text(col2_x, 0.52, 'Interaction: p < 0.001***', fontsize=11, ha='center',
            color='#1565c0', fontweight='bold')
    ax3.text(col2_x, 0.46, 'HIGHLY significant', fontsize=11, ha='center',
            color='#1565c0', fontweight='bold')

    ax3.text(col2_x, 0.36, '✓ EXPECTED:', fontsize=10, ha='center', color='#2e7d32',
            fontweight='bold')
    ax3.text(col2_x, 0.30, 'High vuln patients', fontsize=9, ha='center', color='#333')
    ax3.text(col2_x, 0.25, '→ STRONGER effects', fontsize=9, ha='center', color='#333',
            style='italic')

    ax3.text(col2_x, 0.15, 'Solution:', fontsize=9, ha='center', color='#2e7d32',
            fontweight='bold')
    ax3.text(col2_x, 0.10, 'Adequate power', fontsize=9, ha='center', color='#333')
    ax3.text(col2_x, 0.05, 'Within-study analysis', fontsize=9, ha='center', color='#333')

    # Arrow
    arrow2 = patches.FancyArrowPatch((0.62, 0.50), (0.76, 0.50),
                                    arrowstyle='->', lw=3, color='#2e7d32',
                                    mutation_scale=30)
    ax3.add_patch(arrow2)
    ax3.text(0.69, 0.54, 'VALIDATION', fontsize=10, ha='center',
            color='#2e7d32', fontweight='bold')

    # Column 3: SHAP validation
    col3_x = 0.82
    ax3.text(col3_x, 0.75, 'SHAP Validation',
            fontsize=14, fontweight='bold', ha='center', color='#ff6b00')
    ax3.text(col3_x, 0.68, '(ML/XAI)',
            fontsize=11, ha='center', color='#666', style='italic')

    ax3.text(col3_x, 0.58, 'Vulnerability feature', fontsize=11, ha='center', color='#333')
    ax3.text(col3_x, 0.52, 'identified as important', fontsize=11, ha='center', color='#333')
    ax3.text(col3_x, 0.46, 'by SHAP analysis', fontsize=11, ha='center', color='#333')

    ax3.text(col3_x, 0.36, '✓ CONFIRMED:', fontsize=10, ha='center', color='#2e7d32',
            fontweight='bold')
    ax3.text(col3_x, 0.30, 'Vulnerability MODIFIES', fontsize=9, ha='center', color='#333')
    ax3.text(col3_x, 0.25, 'climate effects', fontsize=9, ha='center', color='#333',
            style='italic')

    ax3.text(col3_x, 0.15, 'Impact:', fontsize=9, ha='center', color='#2e7d32',
            fontweight='bold')
    ax3.text(col3_x, 0.10, 'ML findings validated', fontsize=9, ha='center', color='#333')
    ax3.text(col3_x, 0.05, 'Mechanistic proof', fontsize=9, ha='center', color='#333')

    # Save as SVG
    output_file = '../interaction_validation_slide.svg'
    plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=150)
    plt.savefig(output_file.replace('.svg', '.png'), format='png',
               bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Saved: {output_file}")
    print(f"✓ Saved: {output_file.replace('.svg', '.png')}")

if __name__ == '__main__':
    create_interaction_slide()
