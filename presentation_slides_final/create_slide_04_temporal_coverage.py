#!/usr/bin/env python3
"""
Generate Slide 4: Temporal Coverage Timeline
Uses REAL clinical data to show actual study dates and patient counts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SLIDE 4: Temporal Coverage Timeline")
print("=" * 70)

# Load clinical data
print("\n1. Loading clinical data...")
data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
df = pd.read_csv(data_path, low_memory=False)
print(f"   Loaded {len(df):,} total records")

# Load GCRO data for survey waves
print("\n2. Loading GCRO survey data...")
gcro_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"
gcro_df = pd.read_csv(gcro_path, low_memory=False)
print(f"   Loaded {len(gcro_df):,} GCRO records")

# Extract study information
print("\n3. Extracting study timeline data...")
df['primary_date'] = pd.to_datetime(df['primary_date'], errors='coerce')
df['year'] = df['primary_date'].dt.year

# Group by study source
study_summary = df.groupby('study_source').agg({
    'anonymous_patient_id': 'count',
    'year': ['min', 'max']
}).reset_index()
study_summary.columns = ['study', 'patient_count', 'year_start', 'year_end']
study_summary = study_summary.sort_values('year_start')

print(f"   Found {len(study_summary)} unique studies")
print(f"   Date range: {study_summary['year_start'].min():.0f} - {study_summary['year_end'].max():.0f}")
print(f"   Total patients: {study_summary['patient_count'].sum():,}")

# GCRO survey waves
gcro_surveys = gcro_df['survey_year'].value_counts().sort_index()
print(f"\n   GCRO Survey waves: {list(gcro_surveys.index)}")
print(f"   Total households: {gcro_surveys.sum():,}")

# Create figure
print("\n4. Creating timeline visualization...")
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')

# Define research focus categories
research_focus = {
    'HIV': '#3498DB',      # Blue
    'TB/HIV': '#2ECC71',   # Green
    'Metabolic': '#9B59B6', # Purple
    'COVID': '#E74C3C'     # Red
}

# Categorize studies by research type
def categorize_study(study_name):
    study_lower = study_name.lower()
    if 'covid' in study_lower or 'vida' in study_lower:
        return 'COVID', research_focus['COVID']
    elif 'aurum' in study_lower or ('tb' in study_lower and 'hiv' in study_lower):
        return 'TB/HIV', research_focus['TB/HIV']
    elif 'metabolic' in study_lower:
        return 'Metabolic', research_focus['Metabolic']
    else:
        return 'HIV', research_focus['HIV']

# Plot clinical studies
y_position = 0.7
study_positions = []

for idx, row in study_summary.iterrows():
    study_name = row['study']
    year_start = row['year_start']
    year_end = row['year_end']
    patient_count = row['patient_count']

    # Get research type
    research_type, color = categorize_study(study_name)

    # Calculate bar width (duration)
    duration = year_end - year_start + 0.5

    # Plot study bar
    rect = Rectangle((year_start, y_position), duration, 0.08,
                     facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)

    # Add label with patient count
    mid_year = year_start + duration / 2
    label = f"N={patient_count:,}" if patient_count > 100 else f"N={patient_count}"

    ax.text(mid_year, y_position + 0.04, label,
           ha='center', va='center', fontsize=8, fontweight='bold',
           color='white', bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor=color, alpha=0.9, edgecolor='none'))

    study_positions.append((study_name, y_position, research_type))
    y_position -= 0.09

# Add "Clinical Studies" label
ax.text(2001.5, 0.78, 'Clinical Studies',
       fontsize=13, fontweight='bold', color='#2C3E50',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1',
                alpha=0.9, edgecolor='#3498DB', linewidth=2))

# Plot GCRO survey waves
gcro_y = 0.15
gcro_color = '#27AE60'  # Green for GCRO

for year in [2011, 2014, 2018, 2021]:
    if year in gcro_surveys.index:
        count = gcro_surveys[year]
        # Plot as vertical marker
        rect = Rectangle((year - 0.3, gcro_y), 0.6, 0.08,
                        facecolor=gcro_color, edgecolor='white',
                        linewidth=2, alpha=0.85)
        ax.add_patch(rect)

        # Add year label
        ax.text(year, gcro_y + 0.04, str(year),
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white')

# Add "GCRO Surveys" label
ax.text(2001.5, 0.23, 'GCRO Surveys',
       fontsize=13, fontweight='bold', color='#2C3E50',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1',
                alpha=0.9, edgecolor='#27AE60', linewidth=2))

# Create legend
legend_elements = []

# Add research focus legend
legend_elements.append(mpatches.Patch(color='white', label='Research Focus'))
for focus_type, color in research_focus.items():
    legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='white',
                                         linewidth=2, label=f'  {focus_type}', alpha=0.85))

# Add spacer
legend_elements.append(mpatches.Patch(color='white', label=''))

# Add GCRO legend
legend_elements.append(mpatches.Patch(facecolor=gcro_color, edgecolor='white',
                                     linewidth=2, label='GCRO Surveys', alpha=0.85))

# Position legend
legend = ax.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), fontsize=11, frameon=True,
                  title='Research Focus', title_fontsize=12)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_edgecolor('#BDC3C7')
legend.get_title().set_fontweight('bold')

# Set title
title_text = 'Temporal Coverage Overview\n17 Clinical Studies (10,202 patients) + 4 GCRO Survey Waves (58,616 records) • 2002-2021'
ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

# Format axes
ax.set_xlim(2001, 2022)
ax.set_ylim(0, 0.9)
ax.set_xlabel('Year', fontsize=13, fontweight='bold', color='#34495E')
ax.set_yticks([])  # No y-axis labels needed

# X-axis formatting
ax.set_xticks(range(2002, 2022, 2))
ax.tick_params(axis='x', labelsize=11, colors='#34495E')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#BDC3C7')

# Add grid for years
ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Add summary text box
summary_text = (
    f"19-Year Study Period | 1,092 Climate Features | Individual-Level Analysis\n"
    f"ERA5 climate data matched to clinical records | GCRO socioeconomic integration"
)

ax.text(0.5, 0.02, summary_text,
       transform=ax.transAxes, ha='center', va='bottom',
       fontsize=11, color='#2C3E50',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498DB',
                alpha=0.2, edgecolor='#3498DB', linewidth=2))

plt.tight_layout()

# Save as SVG
output_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_map_perfection/presentation_slides_final/main_presentation/slide_04_temporal_coverage.svg"
plt.savefig(output_path, format='svg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\n✓ Slide saved: {output_path}")
print(f"✓ Aspect ratio: 16:9")
print(f"✓ Studies plotted: {len(study_summary)}")
print(f"✓ Date range: {study_summary['year_start'].min():.0f}-{study_summary['year_end'].max():.0f}")
print(f"✓ Total clinical patients: {study_summary['patient_count'].sum():,}")
print(f"✓ GCRO survey waves: {len(gcro_surveys)}")
print(f"✓ Total GCRO households: {gcro_surveys.sum():,}")
print("\n" + "=" * 70)
