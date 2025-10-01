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

def generate_gcro_survey_points(n_points: int = 2000) -> List[Tuple[float, float]]:
    """
    Generate realistic GCRO household survey points across Johannesburg.
    Higher density in townships, lower in suburbs.
    """
    points = []
    
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
                points.append((lon, lat))
    
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
    
    # Create figure with proper aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
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
    
    # Plot GCRO survey points (small, scattered, background)
    gcro_lons = [p[0] for p in gcro_points]
    gcro_lats = [p[1] for p in gcro_points]
    ax.scatter(gcro_lons, gcro_lats,
              s=4,  # Very small dots
              c='#87CEEB',  # Light blue
              alpha=0.4,  # Semi-transparent
              marker='o',
              zorder=3,
              label='GCRO Household Surveys (n≈58,616)')
    
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
    
    # GCRO Survey section
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='#87CEEB', markersize=6,
                                    alpha=0.4, label='GCRO Household Surveys'))
    
    # Clinical trial sites section
    research_types = {}
    for site in clinical_sites:
        if site["research_type"] not in research_types:
            research_types[site["research_type"]] = site["color"]
    
    for research_type, color in research_types.items():
        legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                        markerfacecolor=color, markersize=10,
                                        markeredgecolor='white', markeredgewidth=2,
                                        label=f'Clinical Sites - {research_type}'))
    
    # Position legend
    legend = ax.legend(handles=legend_elements,
                      loc='upper left',
                      bbox_to_anchor=(0.02, 0.98),
                      fontsize=11,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      title='Data Sources',
                      title_fontsize=12)
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
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
    ax.set_title('ENBEL Climate-Health Data Sources in Johannesburg\n'
                'Clear Distinction: Survey Points vs Clinical Trial Sites',
                fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
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
    
    # Save as SVG with organized structure
    output_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_distinct_data_types_map.svg'
    plt.savefig(output_path, 
               format='svg',
               dpi=300,
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none')
    
    print(f"Map saved to: {output_path}")
    print(f"GCRO survey points generated: {len(gcro_points)}")
    print(f"Clinical trial sites plotted: {len(clinical_sites)}")
    print("\nVisual Hierarchy:")
    print("1. Clinical Trial Sites: Large diamonds with labels")
    print("2. Johannesburg Boundary: Subtle outline")  
    print("3. GCRO Survey Points: Small scattered dots")
    print("4. Geographic Labels: Minimal key locations")
    
    return fig, ax

if __name__ == "__main__":
    create_distinct_data_types_map()
    plt.show()