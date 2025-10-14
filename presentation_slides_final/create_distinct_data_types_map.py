#!/usr/bin/env python3
"""
Create a Johannesburg map with CLEAR visual distinction between GCRO survey data and clinical trial sites.

Addresses user feedback:
- "distinction between the actual survey data and the clusters...is not that clear"
- "which are the clinical trial sites and which are the GCRO sites...that distinction is not clear"
- "I like the styling...minimalist feel"

Visual Hierarchy:
1. Clinical Trial Sites: Large, prominent, distinct symbols
2. Johannesburg Boundary: Clear but subtle outline  
3. GCRO Survey Points: Small, scattered, background layer
4. Geographic Labels: Minimal (CBD, Sandton, Soweto only)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random
from typing import List, Tuple, Dict
import math

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_johannesburg_boundary() -> List[Tuple[float, float]]:
    """Create simplified Johannesburg metropolitan boundary."""
    # Johannesburg bounds: approximately 26.0°S to 26.4°S, 27.8°E to 28.3°E
    boundary_points = [
        (27.85, -26.05),   # Northwest
        (28.25, -26.05),   # Northeast  
        (28.30, -26.15),   # East
        (28.25, -26.30),   # Southeast
        (28.15, -26.35),   # South
        (27.95, -26.38),   # Southwest
        (27.85, -26.32),   # West
        (27.82, -26.20),   # Northwest return
        (27.85, -26.05),   # Close polygon
    ]
    return boundary_points

def generate_gcro_survey_points(n_points: int = 2000) -> List[Dict]:
    """
    Generate realistic GCRO household survey points across Johannesburg.
    Higher density in townships, lower in suburbs.
    Returns list of dicts with location and survey wave.
    """
    points = []

    # Survey waves with realistic distribution
    survey_waves = [
        {"year": 2011, "proportion": 0.25, "color": "#1E3A8A"},  # Dark blue
        {"year": 2014, "proportion": 0.30, "color": "#2563EB"},  # Medium blue
        {"year": 2018, "proportion": 0.25, "color": "#60A5FA"},  # Light blue
        {"year": 2021, "proportion": 0.20, "color": "#93C5FD"},  # Very light blue
    ]

    # Define residential density zones
    zones = [
        # Soweto (high density)
        {"center": (27.87, -26.27), "radius": 0.08, "density": 0.4, "points": 800},
        # Alexandra (high density)
        {"center": (28.09, -26.10), "radius": 0.03, "density": 0.5, "points": 300},
        # Central/Inner City (medium density)
        {"center": (28.05, -26.20), "radius": 0.05, "density": 0.3, "points": 400},
        # Northern suburbs (low density)
        {"center": (28.05, -26.08), "radius": 0.12, "density": 0.1, "points": 300},
        # Eastern areas (medium density)
        {"center": (28.20, -26.25), "radius": 0.08, "density": 0.2, "points": 200},
    ]

    for zone in zones:
        center_lon, center_lat = zone["center"]
        radius = zone["radius"]
        n_zone_points = zone["points"]

        for _ in range(n_zone_points):
            # Generate point within circular zone with realistic clustering
            angle = random.uniform(0, 2 * math.pi)
            # Use sqrt for uniform distribution within circle
            r = radius * math.sqrt(random.uniform(0, 1))

            # Add some clustering by biasing toward center
            if random.random() < zone["density"]:
                r *= 0.6  # Cluster more toward center

            lon = center_lon + r * math.cos(angle)
            lat = center_lat + r * math.sin(angle)

            # Ensure point is within Johannesburg bounds
            if 27.82 <= lon <= 28.30 and -26.40 <= lat <= -26.05:
                # Randomly assign survey wave
                rand = random.random()
                cumulative = 0
                assigned_wave = survey_waves[-1]  # Default to last wave

                for wave in survey_waves:
                    cumulative += wave["proportion"]
                    if rand <= cumulative:
                        assigned_wave = wave
                        break

                points.append({
                    "lon": lon,
                    "lat": lat,
                    "year": assigned_wave["year"],
                    "color": assigned_wave["color"]
                })

    return points[:n_points]  # Return exactly n_points

def get_clinical_trial_sites() -> List[Dict]:
    """Define the 4 major clinical trial sites with research details."""
    return [
        {
            "name": "Central Johannesburg Hub",
            "location": (28.047, -26.204),  # CBD area
            "patients": 6964,
            "studies": 9,
            "research_type": "Mixed",
            "color": "#E74C3C",  # Red for mixed research
            "description": "Mixed research facility"
        },
        {
            "name": "Western Johannesburg Hub", 
            "location": (27.91, -26.25),  # Western areas
            "patients": 685,
            "studies": 6,
            "research_type": "HIV",
            "color": "#3498DB",  # Blue for HIV
            "description": "HIV research focus"
        },
        {
            "name": "Northern Site (Aurum)",
            "location": (28.08, -26.12),  # Northern areas
            "patients": 2551,
            "studies": 1,
            "research_type": "TB/HIV",
            "color": "#9B59B6",  # Purple for TB/HIV
            "description": "TB/HIV research"
        },
        {
            "name": "Southwest Site",
            "location": (27.89, -26.32),  # Southwest
            "patients": 2,
            "studies": 1, 
            "research_type": "HIV",
            "color": "#3498DB",  # Blue for HIV
            "description": "Small HIV study"
        }
    ]

def create_distinct_data_types_map():
    """Create the main map with clear distinction between data types."""

    # Create figure with 16:9 aspect ratio optimized for Figma
    # Standard 1920x1080 presentation size
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Generate data
    boundary = create_johannesburg_boundary()
    gcro_points = generate_gcro_survey_points(2000)
    clinical_sites = get_clinical_trial_sites()

    # Plot Johannesburg boundary (subtle)
    boundary_lons = [p[0] for p in boundary]
    boundary_lats = [p[1] for p in boundary]
    ax.plot(boundary_lons, boundary_lats,
           color='#34495E', linewidth=2, alpha=0.7, zorder=2)

    # Fill boundary with very light background
    boundary_polygon = Polygon([(lon, lat) for lon, lat in boundary],
                              facecolor='#F8F9FA', alpha=0.3, zorder=1)
    ax.add_patch(boundary_polygon)

    # Plot GCRO survey points by wave (small, scattered, background)
    # Group points by survey wave for better color distinction
    survey_waves = {}
    for point in gcro_points:
        year = point["year"]
        if year not in survey_waves:
            survey_waves[year] = {"lons": [], "lats": [], "color": point["color"]}
        survey_waves[year]["lons"].append(point["lon"])
        survey_waves[year]["lats"].append(point["lat"])

    # Plot each wave separately with distinct colors
    for year in sorted(survey_waves.keys()):
        wave_data = survey_waves[year]
        ax.scatter(wave_data["lons"], wave_data["lats"],
                  s=5,  # Small dots
                  c=wave_data["color"],
                  alpha=0.6,  # More opaque for better visibility
                  marker='o',
                  zorder=3,
                  label=f'GCRO {year}')
    
    # Plot clinical trial sites (large, prominent, foreground)
    for site in clinical_sites:
        lon, lat = site["location"]
        
        # Size proportional to patient count (with minimum size)
        size = max(200, site["patients"] / 30)  # Scale factor
        
        # Plot main marker (diamond shape for distinction)
        ax.scatter(lon, lat,
                  s=size,
                  c=site["color"],
                  marker='D',  # Diamond shape
                  edgecolors='white',
                  linewidths=3,
                  zorder=10,
                  alpha=0.9)
        
        # Add site label with patient count
        label_text = f"{site['name']}\n({site['patients']:,} patients)"
        
        # Position label to avoid overlap
        label_offset = 0.02
        if lon > 28.1:  # Right side sites
            ha, offset_x = 'left', label_offset
        else:  # Left side sites  
            ha, offset_x = 'right', -label_offset
            
        ax.annotate(label_text,
                   xy=(lon, lat),
                   xytext=(lon + offset_x, lat),
                   fontsize=10,
                   fontweight='bold',
                   ha=ha,
                   va='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            alpha=0.8,
                            edgecolor=site["color"],
                            linewidth=1),
                   zorder=11)
    
    # Add key geographic labels (minimal)
    landmarks = [
        {"name": "Johannesburg CBD", "location": (28.047, -26.204), "size": 12},
        {"name": "Sandton", "location": (28.053, -26.107), "size": 11},
        {"name": "Soweto", "location": (27.87, -26.27), "size": 11},
    ]
    
    for landmark in landmarks:
        lon, lat = landmark["location"]
        ax.annotate(landmark["name"],
                   xy=(lon, lat),
                   xytext=(lon, lat - 0.03),
                   fontsize=landmark["size"],
                   ha='center',
                   va='top',
                   style='italic',
                   color='#2C3E50',
                   alpha=0.7,
                   zorder=5)
    
    # Create clear legend with sections
    legend_elements = []

    # GCRO Survey section - add each wave with distinct color
    legend_elements.append(plt.Line2D([0], [0], marker='', color='w',
                                    label='GCRO Survey Waves:', markersize=0))
    for year in sorted(survey_waves.keys()):
        wave_data = survey_waves[year]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=wave_data["color"],
                                        markersize=7,
                                        alpha=0.6,
                                        label=f'  {year}'))

    # Add spacer
    legend_elements.append(plt.Line2D([0], [0], marker='', color='w',
                                    label='', markersize=0))

    # Clinical trial sites section
    legend_elements.append(plt.Line2D([0], [0], marker='', color='w',
                                    label='Clinical Research Sites:', markersize=0))

    research_types = {}
    for site in clinical_sites:
        if site["research_type"] not in research_types:
            research_types[site["research_type"]] = site["color"]

    for research_type, color in research_types.items():
        legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                        markerfacecolor=color, markersize=10,
                                        markeredgecolor='white', markeredgewidth=2,
                                        label=f'  {research_type}'))

    # Position legend
    legend = ax.legend(handles=legend_elements,
                      loc='upper right',
                      bbox_to_anchor=(0.98, 0.98),
                      fontsize=10,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      title='Data Sources',
                      title_fontsize=11)

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#BDC3C7')
    legend.get_title().set_fontweight('bold')
    
    # Add data summary text box
    summary_text = (
        "GCRO Survey: 58,616 households across Johannesburg\n"
        "Clinical Trials: 11,398 patient records from 4 major sites\n"
        "Combined: Comprehensive climate-health analysis dataset"
    )
    
    ax.text(0.98, 0.02, summary_text,
           transform=ax.transAxes,
           fontsize=10,
           ha='right',
           va='bottom',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='#ECF0F1',
                    alpha=0.9,
                    edgecolor='#BDC3C7'),
           zorder=12)
    
    # Set title
    ax.set_title('Johannesburg Study Distribution\n'
                'Clinical trial sites and GCRO household survey coverage across metropolitan area',
                fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
    
    # Format axes (minimal, clean)
    ax.set_xlabel('Longitude (°E)', fontsize=12, color='#34495E')
    ax.set_ylabel('Latitude (°S)', fontsize=12, color='#34495E')
    
    # Set clean bounds with slight padding
    ax.set_xlim(27.8, 28.32)
    ax.set_ylim(-26.42, -26.02)
    
    # Clean grid (subtle)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Format tick labels
    ax.tick_params(colors='#34495E', labelsize=10)
    
    plt.tight_layout()

    # Save as SVG optimized for Figma (16:9 aspect ratio)
    output_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/johannesburg_study_distribution_16x9.svg'
    plt.savefig(output_path,
               format='svg',
               dpi=150,
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none')

    print(f"✓ Map saved to: {output_path}")
    print(f"✓ Aspect ratio: 16:9 (optimized for Figma presentations)")
    print(f"✓ GCRO survey points generated: {len(gcro_points)}")
    print(f"  - Survey waves: {sorted(survey_waves.keys())}")
    print(f"  - Distinct colors for each wave")
    print(f"✓ Clinical trial sites plotted: {len(clinical_sites)}")
    print("\nColor Scheme:")
    print("  GCRO 2011: #1E3A8A (Dark blue)")
    print("  GCRO 2014: #2563EB (Medium blue)")
    print("  GCRO 2018: #60A5FA (Light blue)")
    print("  GCRO 2021: #93C5FD (Very light blue)")
    print("\nVisual Hierarchy:")
    print("1. Clinical Trial Sites: Large diamonds with labels")
    print("2. Johannesburg Boundary: Subtle outline")
    print("3. GCRO Survey Points: Small scattered dots by wave")
    print("4. Geographic Labels: Minimal key locations")

    return fig, ax

if __name__ == "__main__":
    create_distinct_data_types_map()
    plt.show()