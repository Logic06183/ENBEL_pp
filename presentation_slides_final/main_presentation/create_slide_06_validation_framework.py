#!/usr/bin/env python3
"""
Create Slide 6: Three-Stage Validation Framework
Visual flow diagram of the validation methodology
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
import numpy as np

OUTPUT_DIR = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection/presentation_slides_final/main_presentation"

# Colors
PRIMARY_BLUE = '#2E7AB5'
STAGE1_BLUE = '#3498DB'
STAGE2_PURPLE = '#9B59B6'
STAGE3_GREEN = '#27AE60'
TEXT_COLOR = '#2C3E50'
BACKGROUND = '#F8F9FA'
ACCENT_GOLD = '#F39C12'

print("Creating Slide 6: Three-Stage Validation Framework...")

# Create figure
fig = plt.figure(figsize=(19.2, 10.8), dpi=150)
fig.patch.set_facecolor('white')

# Title
fig.suptitle('Three-Stage Validation Framework: From Discovery to Confirmation',
             fontsize=34, fontweight='bold', color=TEXT_COLOR, y=0.96)

subtitle = 'Rigorous statistical validation with multiple testing correction and cross-validation'
fig.text(0.5, 0.91, subtitle, ha='center', fontsize=16, color=TEXT_COLOR, style='italic')

# Create axis
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Stage 1: Discovery
x1, y1 = 15, 55
box1 = FancyBboxPatch((x1-8, y1-12), 16, 24, boxstyle="round,pad=1",
                       edgecolor=STAGE1_BLUE, facecolor=BACKGROUND,
                       linewidth=4, zorder=2)
ax.add_patch(box1)

# Stage 1 icon
circle1 = Circle((x1, y1+8), 3, facecolor=STAGE1_BLUE, edgecolor='white', linewidth=2, zorder=3)
ax.add_patch(circle1)
ax.text(x1, y1+8, '1', ha='center', va='center', fontsize=22, color='white', fontweight='bold')

ax.text(x1, y1+3, 'DISCOVERY', ha='center', va='top', fontsize=16, fontweight='bold', color=STAGE1_BLUE)
ax.text(x1, y1-1, 'Exploratory', ha='center', va='top', fontsize=11, style='italic', color=TEXT_COLOR)

details1 = [
    'Random Forest',
    'Feature Selection',
    'SHAP Analysis',
    'α = 0.05'
]
y_txt = y1-5
for txt in details1:
    ax.text(x1, y_txt, txt, ha='center', va='top', fontsize=9, color=TEXT_COLOR)
    y_txt -= 2.5

# Stage 2: Screening
x2, y2 = 50, 55
box2 = FancyBboxPatch((x2-8, y2-12), 16, 24, boxstyle="round,pad=1",
                       edgecolor=STAGE2_PURPLE, facecolor=BACKGROUND,
                       linewidth=4, zorder=2)
ax.add_patch(box2)

circle2 = Circle((x2, y2+8), 3, facecolor=STAGE2_PURPLE, edgecolor='white', linewidth=2, zorder=3)
ax.add_patch(circle2)
ax.text(x2, y2+8, '2', ha='center', va='center', fontsize=22, color='white', fontweight='bold')

ax.text(x2, y2+3, 'SCREENING', ha='center', va='top', fontsize=16, fontweight='bold', color=STAGE2_PURPLE)
ax.text(x2, y2-1, 'Confirmatory', ha='center', va='top', fontsize=11, style='italic', color=TEXT_COLOR)

details2 = [
    'XGBoost + LGBM',
    'Cross-Validation',
    'Multiple Testing',
    'Bonferroni'
]
y_txt = y2-5
for txt in details2:
    ax.text(x2, y_txt, txt, ha='center', va='top', fontsize=9, color=TEXT_COLOR)
    y_txt -= 2.5

# Stage 3: Validation
x3, y3 = 85, 55
box3 = FancyBboxPatch((x3-8, y3-12), 16, 24, boxstyle="round,pad=1",
                       edgecolor=STAGE3_GREEN, facecolor=BACKGROUND,
                       linewidth=4, zorder=2)
ax.add_patch(box3)

circle3 = Circle((x3, y3+8), 3, facecolor=STAGE3_GREEN, edgecolor='white', linewidth=2, zorder=3)
ax.add_patch(circle3)
ax.text(x3, y3+8, '3', ha='center', va='center', fontsize=22, color='white', fontweight='bold')

ax.text(x3, y3+3, 'VALIDATION', ha='center', va='top', fontsize=16, fontweight='bold', color=STAGE3_GREEN)
ax.text(x3, y3-1, 'Independent', ha='center', va='top', fontsize=11, style='italic', color=TEXT_COLOR)

details3 = [
    'DLNM Models',
    'Temporal Lags',
    'Effect Sizes',
    'CI 95%'
]
y_txt = y3-5
for txt in details3:
    ax.text(x3, y_txt, txt, ha='center', va='top', fontsize=9, color=TEXT_COLOR)
    y_txt -= 2.5

# Arrows
arrow1 = FancyArrowPatch((x1+8, y1), (x2-8, y2), arrowstyle='->', mutation_scale=30,
                         linewidth=3, color=PRIMARY_BLUE, zorder=1)
ax.add_patch(arrow1)
ax.text((x1+x2)/2, y1+3, 'Filter', ha='center', va='bottom', fontsize=11,
        fontweight='bold', color=PRIMARY_BLUE, style='italic')

arrow2 = FancyArrowPatch((x2+8, y2), (x3-8, y3), arrowstyle='->', mutation_scale=30,
                         linewidth=3, color=PRIMARY_BLUE, zorder=1)
ax.add_patch(arrow2)
ax.text((x2+x3)/2, y2+3, 'Confirm', ha='center', va='bottom', fontsize=11,
        fontweight='bold', color=PRIMARY_BLUE, style='italic')

# Add performance metrics boxes
metrics_y = 30
metrics = [
    ('Discovery', 'n=100 features', '→ 20 candidates', STAGE1_BLUE),
    ('Screening', '20 candidates', '→ 8 significant', STAGE2_PURPLE),
    ('Validation', '8 significant', '→ 5 confirmed', STAGE3_GREEN)
]

for i, (stage, input_txt, output_txt, color) in enumerate(metrics):
    x = 15 + i * 35
    box = FancyBboxPatch((x-7, metrics_y-4), 14, 8, boxstyle="round,pad=0.5",
                         edgecolor=color, facecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, metrics_y+2, stage, ha='center', va='top', fontsize=11, fontweight='bold', color=color)
    ax.text(x, metrics_y, input_txt, ha='center', va='center', fontsize=9, color=TEXT_COLOR)
    ax.text(x, metrics_y-2, output_txt, ha='center', va='bottom', fontsize=9, color=TEXT_COLOR, fontweight='bold')

# Add methodology box
method_box = FancyBboxPatch((10, 8), 80, 12, boxstyle="round,pad=1",
                            edgecolor=ACCENT_GOLD, facecolor=BACKGROUND,
                            linewidth=3, alpha=0.95)
ax.add_patch(method_box)

ax.text(50, 16, 'Statistical Safeguards', ha='center', va='center',
        fontsize=14, fontweight='bold', color=TEXT_COLOR)

safeguards = [
    '✓ 5-Fold Cross-Validation',
    '✓ Bonferroni Correction',
    '✓ Independent Test Sets',
    '✓ Effect Size Thresholds',
    '✓ Temporal Consistency',
    '✓ Biological Plausibility'
]

x_safe = 15
for i, safe in enumerate(safeguards):
    ax.text(x_safe + (i%3)*26, 12 - (i//3)*3, safe, ha='left', va='center',
            fontsize=10, color=TEXT_COLOR, fontweight='bold')

# Methodology note
methodology_text = (
    'Three-stage framework prevents false discovery while maintaining statistical power. Stage 1: Exploratory analysis with Random Forest.\n'
    'Stage 2: Confirmatory analysis with multiple algorithms and Bonferroni correction for multiple testing (α/k where k=number of tests).\n'
    'Stage 3: Independent validation using Distributed Lag Non-linear Models (DLNM) to assess temporal effects and dose-response relationships.'
)
fig.text(0.5, 0.03, methodology_text, ha='center', fontsize=10,
         color=TEXT_COLOR, style='italic', wrap=True,
         bbox=dict(boxstyle='round', facecolor='white',
                   edgecolor=PRIMARY_BLUE, linewidth=2))

plt.tight_layout(rect=[0, 0.08, 1, 0.88])

output_svg = f"{OUTPUT_DIR}/slide_06_validation_framework.svg"
output_png = output_svg.replace('.svg', '.png')

plt.savefig(output_svg, format='svg', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(output_png, format='png', dpi=150, bbox_inches='tight', facecolor='white')

print(f"✓ Slide 6 saved: {output_svg}")
print(f"✓ Preview saved: {output_png}")
plt.close()
