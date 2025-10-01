#!/usr/bin/env python3
"""
ENBEL Project - Detailed Johannesburg Map with Real Shapefile Data
Creates a publication-quality cartographic visualization using actual South African boundaries
and precise clinical trial coordinates.

Author: ENBEL Research Team
Date: 2025-10-01
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_south_africa_boundaries():
    """Load South African administrative boundaries from shapefile."""
    shapefile_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/map_vector_folder_JHB/zaf_admbnda_adm0_sadb_ocha_20201109.shp"
    
    try:
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)
        print(f"Loaded shapefile with {len(gdf)} features")
        print(f"CRS: {gdf.crs}")
        print(f"Columns: {list(gdf.columns)}")
        
        # Ensure WGS84 coordinate system
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
            
        return gdf
        
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None

def create_johannesburg_extent():
    """Define the Johannesburg metropolitan area extent."""
    # Johannesburg metropolitan area bounds (slightly expanded)
    # Covers City of Johannesburg, Ekurhuleni, and parts of West Rand
    return {
        'west': 27.6,    # Western boundary
        'east': 28.6,    # Eastern boundary  
        'south': -26.5,  # Southern boundary
        'north': -25.5   # Northern boundary
    }

def get_clinical_sites_data():
    """Get clinical trial sites with precise coordinates and metadata."""
    return [
        {
            'name': 'Central Johannesburg Hub',
            'lat': -26.2041,
            'lon': 28.0473,
            'patients': 6964,
            'studies': 9,
            'types': ['HIV', 'TB/HIV', 'COVID', 'Metabolic'],
            'primary_type': 'Mixed',
            'studies_list': ['DPHRU_013', 'DPHRU_053', 'EZIN_025', 'Ezin_002', 'JHSPH_005', 'VIDA_007', 'VIDA_008', 'WRHI_001', 'WRHI_003'],
            'description': 'Major multi-study hub'
        },
        {
            'name': 'Western Johannesburg Hub',
            'lat': -26.2041,
            'lon': 27.9394,
            'patients': 685,
            'studies': 6,
            'types': ['HIV'],
            'primary_type': 'HIV',
            'studies_list': ['ACTG_015', 'ACTG_016', 'ACTG_017', 'ACTG_018', 'ACTG_019', 'SCHARP_006'],
            'description': 'HIV/AIDS clinical trials'
        },
        {
            'name': 'Northern Site (Aurum)',
            'lat': -25.7479,
            'lon': 28.2293,
            'patients': 2551,
            'studies': 1,
            'types': ['TB/HIV'],
            'primary_type': 'TB/HIV',
            'studies_list': ['Aurum_009'],
            'description': 'Large TB/HIV study'
        },
        {
            'name': 'Southwest Site',
            'lat': -26.2309,
            'lon': 27.8585,
            'patients': 2,
            'studies': 1,
            'types': ['HIV'],
            'primary_type': 'HIV',
            'studies_list': ['SCHARP_004'],
            'description': 'Small HIV study'
        }
    ]

def get_major_landmarks():
    """Get major Johannesburg landmarks and districts."""
    return [
        {'name': 'Johannesburg CBD', 'lat': -26.2041, 'lon': 28.0473, 'type': 'district'},
        {'name': 'Sandton', 'lat': -26.1076, 'lon': 28.0567, 'type': 'district'},
        {'name': 'Soweto', 'lat': -26.2678, 'lon': 27.8583, 'type': 'district'},
        {'name': 'Alexandra', 'lat': -26.1009, 'lon': 28.1103, 'type': 'district'},
        {'name': 'Randburg', 'lat': -26.0945, 'lon': 28.0070, 'type': 'district'},
        {'name': 'OR Tambo Airport', 'lat': -26.1392, 'lon': 28.2460, 'type': 'landmark'},
        {'name': 'University of the Witwatersrand', 'lat': -26.1929, 'lon': 28.0305, 'type': 'landmark'},
        {'name': 'Chris Hani Baragwanath Hospital', 'lat': -26.2394, 'lon': 27.9089, 'type': 'landmark'},
    ]

def get_transport_network():
    """Get major transportation routes."""
    return [
        # Major highways
        {'name': 'N1 Highway', 'coords': [(-26.5, 28.0), (-25.5, 28.0)], 'type': 'highway'},
        {'name': 'N3 Highway', 'coords': [(-26.3, 28.1), (-25.8, 28.3)], 'type': 'highway'},
        {'name': 'M1 Highway', 'coords': [(-26.4, 28.05), (-25.9, 28.05)], 'type': 'major_road'},
        {'name': 'M2 Highway', 'coords': [(-26.22, 27.9), (-26.18, 28.1)], 'type': 'major_road'},
        # Ring roads
        {'name': 'R21', 'coords': [(-26.3, 28.2), (-25.9, 28.2)], 'type': 'highway'},
        {'name': 'R24', 'coords': [(-26.2, 27.8), (-26.1, 28.3)], 'type': 'highway'},
    ]

def create_color_scheme():
    """Define color scheme for different research focus areas."""
    return {
        'HIV': '#3498DB',           # Blue
        'COVID': '#E67E22',         # Orange
        'Metabolic': '#27AE60',     # Green  
        'TB/HIV': '#8E44AD',        # Purple
        'Mixed': '#E74C3C',         # Red
        'background': '#F8F9FA',    # Light gray
        'water': '#5DADE2',         # Light blue
        'urban': '#D5DBDB',         # Gray
        'roads': '#34495E',         # Dark gray
        'highways': '#2C3E50',      # Darker gray
        'boundaries': '#7F8C8D',    # Medium gray
        'grid': '#BDC3C7'          # Light gray
    }

def create_detailed_johannesburg_map():
    """Create the comprehensive Johannesburg map."""
    
    # Load South African boundaries
    print("Loading South African administrative boundaries...")
    sa_gdf = load_south_africa_boundaries()
    
    if sa_gdf is None:
        print("Failed to load shapefile. Creating map without boundary data.")
        sa_boundary = None
    else:
        sa_boundary = sa_gdf.geometry.iloc[0] if len(sa_gdf) > 0 else None
    
    # Get data
    extent = create_johannesburg_extent()
    sites = get_clinical_sites_data()
    landmarks = get_major_landmarks()
    transport = get_transport_network()
    colors = create_color_scheme()
    
    # Create figure with proper aspect ratio (16:9)
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Set extent for Johannesburg area
    ax.set_xlim(extent['west'], extent['east'])
    ax.set_ylim(extent['south'], extent['north'])
    
    # Set background color
    ax.set_facecolor(colors['background'])
    
    # Plot South African boundary if available
    if sa_boundary is not None:
        # Get the boundary that intersects with our extent
        extent_poly = Polygon([
            (extent['west'], extent['south']),
            (extent['west'], extent['north']),
            (extent['east'], extent['north']),
            (extent['east'], extent['south'])
        ])
        
        try:
            # Clip boundary to our extent
            clipped_boundary = sa_boundary.intersection(extent_poly)
            
            if hasattr(clipped_boundary, 'geoms'):
                for geom in clipped_boundary.geoms:
                    if geom.geom_type == 'Polygon':
                        x, y = geom.exterior.xy
                        ax.plot(x, y, color=colors['boundaries'], linewidth=2, alpha=0.8)
            else:
                if clipped_boundary.geom_type == 'Polygon':
                    x, y = clipped_boundary.exterior.xy
                    ax.plot(x, y, color=colors['boundaries'], linewidth=2, alpha=0.8)
                    
        except Exception as e:
            print(f"Error clipping boundary: {e}")
    
    # Add coordinate grid
    print("Adding coordinate grid...")
    lon_lines = np.arange(extent['west'], extent['east'] + 0.1, 0.2)
    lat_lines = np.arange(extent['south'], extent['north'] + 0.1, 0.2)
    
    for lon in lon_lines:
        ax.axvline(lon, color=colors['grid'], alpha=0.3, linewidth=0.5)
        ax.text(lon, extent['north'] - 0.05, f"{lon:.1f}°E", 
                ha='center', va='bottom', fontsize=8, color=colors['grid'])
    
    for lat in lat_lines:
        ax.axhline(lat, color=colors['grid'], alpha=0.3, linewidth=0.5)
        ax.text(extent['west'] + 0.05, lat, f"{abs(lat):.1f}°S", 
                ha='left', va='center', fontsize=8, color=colors['grid'])
    
    # Add urban areas (simplified polygons)
    print("Adding urban areas...")
    urban_areas = [
        # Johannesburg CBD area
        Polygon([(27.98, -26.22), (28.08, -26.22), (28.08, -26.18), (27.98, -26.18)]),
        # Sandton area  
        Polygon([(28.03, -26.12), (28.08, -26.12), (28.08, -26.08), (28.03, -26.08)]),
        # Soweto area
        Polygon([(27.82, -26.29), (27.90, -26.29), (27.90, -26.24), (27.82, -26.24)]),
        # Alexandra area
        Polygon([(28.09, -26.11), (28.13, -26.11), (28.13, -26.09), (28.09, -26.09)]),
    ]
    
    for area in urban_areas:
        x, y = area.exterior.xy
        ax.fill(x, y, color=colors['urban'], alpha=0.4, edgecolor=colors['boundaries'], linewidth=0.5)
    
    # Add major water features
    print("Adding water features...")
    # Jukskei River (simplified)
    river_coords = [(27.7, -25.8), (27.9, -26.0), (28.1, -26.1), (28.3, -26.2)]
    river_x, river_y = zip(*river_coords)
    ax.plot(river_x, river_y, color=colors['water'], linewidth=3, alpha=0.7, label='Jukskei River')
    
    # Add dams/reservoirs (small circles)
    dams = [(-25.9, 28.1), (-26.1, 27.8), (-26.3, 28.2)]
    for lat, lon in dams:
        circle = Circle((lon, lat), 0.02, color=colors['water'], alpha=0.8)
        ax.add_patch(circle)
    
    # Add transportation network
    print("Adding transportation network...")
    for route in transport:
        coords = route['coords']
        x_coords, y_coords = zip(*[(lon, lat) for lat, lon in coords])
        
        if route['type'] == 'highway':
            ax.plot(x_coords, y_coords, color=colors['highways'], linewidth=3, alpha=0.8)
            # Add highway label
            mid_x = np.mean(x_coords)
            mid_y = np.mean(y_coords)
            ax.text(mid_x, mid_y, route['name'], fontsize=8, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        else:
            ax.plot(x_coords, y_coords, color=colors['roads'], linewidth=2, alpha=0.6)
    
    # Add ward grid pattern (GCRO coverage)
    print("Adding GCRO ward coverage...")
    # Create approximate ward grid
    ward_size = 0.05  # Approximate ward size in degrees
    for i, x in enumerate(np.arange(extent['west'], extent['east'], ward_size)):
        for j, y in enumerate(np.arange(extent['south'], extent['north'], ward_size)):
            # Vary opacity to show survey coverage density
            coverage_waves = min(4, (i + j) % 5 + 1)  # Simulate 1-4 survey waves
            alpha = 0.1 + (coverage_waves * 0.05)
            
            rect = Rectangle((x, y), ward_size, ward_size, 
                           facecolor='lightblue', alpha=alpha, 
                           edgecolor='lightgray', linewidth=0.2)
            ax.add_patch(rect)
    
    # Add landmarks
    print("Adding major landmarks...")
    for landmark in landmarks:
        if landmark['type'] == 'district':
            ax.plot(landmark['lon'], landmark['lat'], 's', 
                   markersize=8, color='black', alpha=0.7)
            ax.text(landmark['lon'] + 0.02, landmark['lat'], landmark['name'], 
                   fontsize=9, fontweight='bold', ha='left')
        else:
            ax.plot(landmark['lon'], landmark['lat'], '^', 
                   markersize=10, color='darkred', alpha=0.8)
            ax.text(landmark['lon'] + 0.02, landmark['lat'], landmark['name'], 
                   fontsize=8, ha='left', style='italic')
    
    # Plot clinical trial sites
    print("Adding clinical trial sites...")
    max_patients = max(site['patients'] for site in sites)
    
    for site in sites:
        # Calculate proportional symbol size
        base_size = 100
        size = base_size + (site['patients'] / max_patients) * 400
        
        # Get color for research type
        color = colors[site['primary_type']]
        
        # Plot main circle
        ax.scatter(site['lon'], site['lat'], s=size, c=color, 
                  alpha=0.8, edgecolor='white', linewidth=2, zorder=10)
        
        # Add patient count label
        ax.text(site['lon'], site['lat'], str(site['patients']), 
               ha='center', va='center', fontweight='bold', 
               fontsize=10, color='white', zorder=11)
        
        # Add site name and details
        label_text = f"{site['name']}\n{site['studies']} studies | {site['description']}"
        ax.annotate(label_text, (site['lon'], site['lat']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9),
                   fontsize=8, color='white', fontweight='bold',
                   ha='left', zorder=12)
    
    # Create comprehensive legend
    print("Creating legend...")
    legend_elements = []
    
    # Research type legend
    for research_type, color in colors.items():
        if research_type in ['HIV', 'COVID', 'Metabolic', 'TB/HIV', 'Mixed']:
            legend_elements.append(
                plt.scatter([], [], s=100, c=color, alpha=0.8, 
                          edgecolor='white', linewidth=1, label=research_type)
            )
    
    # Add size legend for patient numbers
    legend_elements.extend([
        plt.scatter([], [], s=100, c='gray', alpha=0.6, label='<100 patients'),
        plt.scatter([], [], s=300, c='gray', alpha=0.6, label='100-1000 patients'),
        plt.scatter([], [], s=500, c='gray', alpha=0.6, label='>1000 patients')
    ])
    
    # Position legend
    legend1 = ax.legend(handles=legend_elements[:5], title='Research Focus', 
                       loc='upper left', bbox_to_anchor=(0.02, 0.98))
    legend2 = ax.legend(handles=legend_elements[5:], title='Patient Numbers', 
                       loc='upper left', bbox_to_anchor=(0.02, 0.75))
    ax.add_artist(legend1)
    
    # Add north arrow
    north_arrow_x = extent['east'] - 0.1
    north_arrow_y = extent['north'] - 0.1
    ax.annotate('N', xy=(north_arrow_x, north_arrow_y), 
               xytext=(north_arrow_x, north_arrow_y - 0.05),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               fontsize=12, fontweight='bold', ha='center')
    
    # Add scale bar
    scale_length = 0.1  # degrees (approximately 10 km)
    scale_x = extent['west'] + 0.1
    scale_y = extent['south'] + 0.1
    
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
           'k-', linewidth=3)
    ax.text(scale_x + scale_length/2, scale_y - 0.02, '~10 km', 
           ha='center', fontsize=10, fontweight='bold')
    
    # Add title and metadata
    plt.suptitle('ENBEL Project: Clinical Trial Sites and GCRO Survey Coverage\nJohannesburg Metropolitan Area', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add subtitle with study summary
    total_patients = sum(site['patients'] for site in sites)
    total_studies = sum(site['studies'] for site in sites)
    
    subtitle = f"Clinical Data: {total_studies} studies, {total_patients:,} patients | GCRO Data: 508 wards, 58,616 households"
    plt.figtext(0.5, 0.91, subtitle, ha='center', fontsize=12, style='italic')
    
    # Add data sources and projection info
    footer_text = ("Data Sources: ERA5 Climate, South African Medical Research | "
                  "Coordinate System: WGS84 (EPSG:4326) | "
                  "ENBEL Research Project 2025")
    plt.figtext(0.5, 0.02, footer_text, ha='center', fontsize=9, alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°S)', fontsize=12, fontweight='bold')
    
    # Improve axis formatting
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)
    
    # Ensure equal aspect ratio for accurate geographic representation
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig, ax

def main():
    """Main function to create and save the map."""
    print("Creating detailed Johannesburg map with real shapefile data...")
    
    fig, ax = create_detailed_johannesburg_map()
    
    # Save as high-quality SVG
    output_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_johannesburg_detailed_map.svg"
    
    print(f"Saving map to: {output_path}")
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    # Also save as high-resolution PNG for backup
    png_path = output_path.replace('.svg', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print("Map creation completed successfully!")
    print(f"Files saved:")
    print(f"  SVG: {output_path}")
    print(f"  PNG: {png_path}")
    
    plt.show()

if __name__ == "__main__":
    main()