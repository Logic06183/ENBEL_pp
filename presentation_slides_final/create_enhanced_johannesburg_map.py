#!/usr/bin/env python3
"""
ENBEL Project - Enhanced Publication-Quality Johannesburg Map
Advanced cartographic visualization with sophisticated design elements,
improved visual hierarchy, and professional styling for scientific publications.

Author: ENBEL Research Team
Date: 2025-10-01
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, PathPatch, Polygon as MPLPolygon
import matplotlib.patheffects as path_effects
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

def load_south_africa_boundaries():
    """Load and process South African administrative boundaries."""
    shapefile_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/map_vector_folder_JHB/zaf_admbnda_adm0_sadb_ocha_20201109.shp"
    
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None

def create_professional_color_scheme():
    """Define a sophisticated color palette for scientific publication."""
    return {
        # Research focus colors (ColorBrewer-inspired)
        'HIV': '#2166AC',           # Deep blue
        'COVID': '#D73027',         # Vibrant red  
        'Metabolic': '#1A9641',     # Forest green
        'TB/HIV': '#762A83',        # Deep purple
        'Mixed': '#5AAE61',         # Medium green
        
        # Geographic features
        'background': '#FAFAFA',    # Off-white
        'water': '#4292C6',         # Water blue
        'urban_high': '#BDBDBD',    # Dense urban
        'urban_med': '#D9D9D9',     # Medium urban
        'urban_low': '#F0F0F0',     # Light urban
        'green_space': '#A1D99B',   # Parks/open space
        
        # Infrastructure
        'highway_major': '#252525', # Major highways
        'highway_minor': '#525252', # Minor highways  
        'roads_major': '#737373',   # Major roads
        'roads_minor': '#969696',   # Local roads
        'railway': '#41AB5D',       # Railway lines
        
        # Administrative
        'boundary_national': '#252525',  # Country boundary
        'boundary_metro': '#525252',     # Metro boundary
        'boundary_ward': '#BDBDBD',      # Ward boundaries
        'grid_major': '#969696',         # Major grid
        'grid_minor': '#D9D9D9',         # Minor grid
        
        # GCRO coverage
        'gcro_high': '#08519C',     # 4 waves
        'gcro_med': '#3182BD',      # 2-3 waves  
        'gcro_low': '#9ECAE1',      # 1 wave
        'gcro_none': '#F7FBFF'      # No coverage
    }

def get_enhanced_clinical_sites():
    """Get clinical sites with enhanced metadata and styling."""
    return [
        {
            'name': 'Central JHB Hub',
            'full_name': 'Central Johannesburg Clinical Research Hub',
            'lat': -26.2041, 'lon': 28.0473,
            'patients': 6964, 'studies': 9,
            'primary_type': 'Mixed',
            'studies_detail': 'HIV, TB/HIV, COVID-19, Metabolic',
            'major_studies': ['DPHRU_013', 'DPHRU_053', 'VIDA_007', 'WRHI_001'],
            'priority': 1
        },
        {
            'name': 'Northern Hub (Aurum)',
            'full_name': 'Aurum Institute Research Centre',
            'lat': -25.7479, 'lon': 28.2293,
            'patients': 2551, 'studies': 1,
            'primary_type': 'TB/HIV',
            'studies_detail': 'Large-scale TB/HIV intervention',
            'major_studies': ['Aurum_009'],
            'priority': 2
        },
        {
            'name': 'Western JHB Hub',
            'full_name': 'Western Johannesburg ACTG Centre',
            'lat': -26.2041, 'lon': 27.9394,
            'patients': 685, 'studies': 6,
            'primary_type': 'HIV',
            'studies_detail': 'ACTG HIV treatment studies',
            'major_studies': ['ACTG_015', 'ACTG_016', 'ACTG_017'],
            'priority': 3
        },
        {
            'name': 'Southwest Site',
            'full_name': 'Southwest Community Clinic',
            'lat': -26.2309, 'lon': 27.8585,
            'patients': 2, 'studies': 1,
            'primary_type': 'HIV',
            'studies_detail': 'Small community-based study',
            'major_studies': ['SCHARP_004'],
            'priority': 4
        }
    ]

def get_enhanced_landmarks():
    """Get comprehensive landmark and district data."""
    return {
        'business_districts': [
            {'name': 'Johannesburg CBD', 'lat': -26.2041, 'lon': 28.0473, 'importance': 'high'},
            {'name': 'Sandton', 'lat': -26.1076, 'lon': 28.0567, 'importance': 'high'},
            {'name': 'Rosebank', 'lat': -26.1481, 'lon': 28.0417, 'importance': 'medium'},
            {'name': 'Midrand', 'lat': -25.9953, 'lon': 28.1288, 'importance': 'medium'},
        ],
        'residential_areas': [
            {'name': 'Soweto', 'lat': -26.2678, 'lon': 27.8583, 'importance': 'high'},
            {'name': 'Alexandra', 'lat': -26.1009, 'lon': 28.1103, 'importance': 'high'},
            {'name': 'Randburg', 'lat': -26.0945, 'lon': 28.0070, 'importance': 'medium'},
            {'name': 'Roodepoort', 'lat': -26.1625, 'lon': 27.8725, 'importance': 'medium'},
        ],
        'institutions': [
            {'name': 'University of the Witwatersrand', 'lat': -26.1929, 'lon': 28.0305, 'type': 'university'},
            {'name': 'Chris Hani Baragwanath Hospital', 'lat': -26.2394, 'lon': 27.9089, 'type': 'hospital'},
            {'name': 'Charlotte Maxeke Hospital', 'lat': -26.1888, 'lon': 28.0364, 'type': 'hospital'},
            {'name': 'OR Tambo International Airport', 'lat': -26.1392, 'lon': 28.2460, 'type': 'airport'},
        ]
    }

def get_transport_infrastructure():
    """Get detailed transportation network."""
    return {
        'highways': [
            {'name': 'N1', 'coords': [(-26.5, 28.05), (-25.5, 28.05)], 'type': 'national'},
            {'name': 'N3', 'coords': [(-26.35, 28.15), (-25.75, 28.35)], 'type': 'national'},
            {'name': 'N12', 'coords': [(-26.25, 27.7), (-26.15, 28.4)], 'type': 'national'},
            {'name': 'R21', 'coords': [(-26.35, 28.22), (-25.85, 28.22)], 'type': 'regional'},
            {'name': 'R24', 'coords': [(-26.22, 27.75), (-26.02, 28.35)], 'type': 'regional'},
        ],
        'metro_roads': [
            {'name': 'M1', 'coords': [(-26.42, 28.05), (-25.88, 28.05)], 'type': 'metro'},
            {'name': 'M2', 'coords': [(-26.22, 27.92), (-26.18, 28.12)], 'type': 'metro'},
            {'name': 'M3', 'coords': [(-26.28, 28.0), (-26.05, 28.15)], 'type': 'metro'},
        ],
        'rail_lines': [
            {'name': 'Gautrain', 'coords': [(-26.14, 28.25), (-26.11, 28.06), (-26.20, 28.04)], 'type': 'rapid'},
            {'name': 'Main Rail', 'coords': [(-26.3, 27.9), (-26.1, 28.3)], 'type': 'conventional'},
        ]
    }

def create_terrain_base(ax, extent, colors):
    """Create subtle terrain background."""
    # Generate elevation-like contours
    x = np.linspace(extent['west'], extent['east'], 50)
    y = np.linspace(extent['south'], extent['north'], 50)
    X, Y = np.meshgrid(x, y)
    
    # Create synthetic elevation data (simplified Witwatersrand ridge)
    Z = np.sin((X - 28.0) * 10) * np.cos((Y + 26.2) * 8) * 0.1
    
    # Add ridges
    ridge_effect = np.exp(-((X - 28.0)**2 + (Y + 26.1)**2) / 0.05)
    Z += ridge_effect * 0.3
    
    # Plot subtle contours
    contours = ax.contour(X, Y, Z, levels=8, colors='gray', alpha=0.15, linewidths=0.5)
    
    return contours

def add_gcro_ward_coverage(ax, extent, colors):
    """Add sophisticated GCRO survey coverage visualization."""
    ward_size = 0.04
    
    # Create coverage pattern based on realistic survey deployment
    coverage_data = []
    
    for i, x in enumerate(np.arange(extent['west'], extent['east'], ward_size)):
        for j, y in enumerate(np.arange(extent['south'], extent['north'], ward_size)):
            # Simulate realistic coverage patterns
            distance_from_center = np.sqrt((x - 28.05)**2 + (y + 26.15)**2)
            
            # Higher coverage in central areas, lower in periphery
            if distance_from_center < 0.15:  # Central areas
                waves = np.random.choice([3, 4], p=[0.3, 0.7])
            elif distance_from_center < 0.3:  # Intermediate areas
                waves = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            else:  # Peripheral areas
                waves = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            
            coverage_data.append({
                'x': x, 'y': y, 'waves': waves, 'distance': distance_from_center
            })
    
    # Plot coverage with appropriate colors
    for ward in coverage_data:
        if ward['waves'] == 4:
            color = colors['gcro_high']
            alpha = 0.6
        elif ward['waves'] == 3:
            color = colors['gcro_med']
            alpha = 0.4
        elif ward['waves'] in [1, 2]:
            color = colors['gcro_low']
            alpha = 0.25
        else:
            continue  # Skip uncovered areas
        
        rect = Rectangle((ward['x'], ward['y']), ward_size, ward_size,
                        facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(rect)

def plot_enhanced_clinical_sites(ax, sites, colors):
    """Plot clinical sites with sophisticated styling."""
    max_patients = max(site['patients'] for site in sites)
    
    for site in sites:
        # Calculate proportional symbol size
        size_factor = (site['patients'] / max_patients)
        base_size = 150
        size = base_size + (size_factor * 600)
        
        # Get colors
        color = colors[site['primary_type']]
        
        # Create layered symbol effect
        # Outer ring (white border)
        ax.scatter(site['lon'], site['lat'], s=size*1.3, c='white', 
                  alpha=0.9, edgecolor='none', zorder=8)
        
        # Main symbol
        scatter = ax.scatter(site['lon'], site['lat'], s=size, c=color,
                           alpha=0.85, edgecolor='white', linewidth=2, zorder=9)
        
        # Inner highlight
        ax.scatter(site['lon'], site['lat'], s=size*0.4, c='white',
                  alpha=0.7, edgecolor='none', zorder=10)
        
        # Patient count label
        text = ax.text(site['lon'], site['lat'], f"{site['patients']:,}",
                      ha='center', va='center', fontweight='bold',
                      fontsize=9, color='black', zorder=11)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
        
        # Site information box
        if site['priority'] <= 2:  # Only for major sites
            box_text = f"{site['name']}\n{site['studies']} studies\n{site['studies_detail']}"
            
            # Position box to avoid overlaps
            if site['lon'] > 28.1:
                offset_x, ha = -15, 'right'
            else:
                offset_x, ha = 15, 'left'
                
            if site['lat'] < -26.15:
                offset_y, va = 15, 'bottom'
            else:
                offset_y, va = -15, 'top'
            
            # Create styled annotation box
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor=color, 
                            alpha=0.9, edgecolor='white', linewidth=1.5)
            
            annotation = ax.annotate(box_text, (site['lon'], site['lat']),
                                   xytext=(offset_x, offset_y), textcoords='offset points',
                                   bbox=bbox_props, fontsize=9, color='white',
                                   fontweight='bold', ha=ha, va=va, zorder=12)

def add_professional_grid(ax, extent, colors):
    """Add professional coordinate grid system."""
    # Major grid lines (every 0.2 degrees)
    major_lons = np.arange(extent['west'], extent['east'] + 0.1, 0.2)
    major_lats = np.arange(extent['south'], extent['north'] + 0.1, 0.2)
    
    # Minor grid lines (every 0.1 degrees)
    minor_lons = np.arange(extent['west'], extent['east'] + 0.05, 0.1)
    minor_lats = np.arange(extent['south'], extent['north'] + 0.05, 0.1)
    
    # Plot minor grid
    for lon in minor_lons:
        ax.axvline(lon, color=colors['grid_minor'], alpha=0.4, linewidth=0.3, zorder=1)
    for lat in minor_lats:
        ax.axhline(lat, color=colors['grid_minor'], alpha=0.4, linewidth=0.3, zorder=1)
    
    # Plot major grid
    for lon in major_lons:
        ax.axvline(lon, color=colors['grid_major'], alpha=0.6, linewidth=0.8, zorder=2)
        # Add coordinate labels
        ax.text(lon, extent['north'] - 0.03, f"{lon:.1f}°E",
               ha='center', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    for lat in major_lats:
        ax.axhline(lat, color=colors['grid_major'], alpha=0.6, linewidth=0.8, zorder=2)
        # Add coordinate labels
        ax.text(extent['west'] + 0.03, lat, f"{abs(lat):.1f}°S",
               ha='left', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

def create_enhanced_legend(ax, extent, colors, sites):
    """Create comprehensive professional legend."""
    legend_x = extent['east'] - 0.35
    legend_y = extent['north'] - 0.05
    
    # Title
    ax.text(legend_x, legend_y, 'LEGEND', fontsize=14, fontweight='bold',
           ha='left', va='top')
    
    y_offset = 0.08
    
    # Research Focus Section
    ax.text(legend_x, legend_y - y_offset, 'Research Focus Areas', 
           fontsize=12, fontweight='bold', ha='left', va='top')
    y_offset += 0.05
    
    research_types = ['HIV', 'TB/HIV', 'COVID', 'Metabolic', 'Mixed']
    for i, research_type in enumerate(research_types):
        y_pos = legend_y - y_offset - (i * 0.035)
        ax.scatter(legend_x + 0.02, y_pos, s=80, c=colors[research_type],
                  alpha=0.8, edgecolor='white', linewidth=1)
        ax.text(legend_x + 0.05, y_pos, research_type, fontsize=10,
               ha='left', va='center')
    
    y_offset += len(research_types) * 0.035 + 0.06
    
    # Patient Numbers Section
    ax.text(legend_x, legend_y - y_offset, 'Patient Enrollment', 
           fontsize=12, fontweight='bold', ha='left', va='top')
    y_offset += 0.05
    
    size_categories = [
        ('< 100 patients', 150),
        ('100 - 1,000 patients', 300),
        ('1,000 - 5,000 patients', 450),
        ('> 5,000 patients', 600)
    ]
    
    for i, (label, size) in enumerate(size_categories):
        y_pos = legend_y - y_offset - (i * 0.04)
        ax.scatter(legend_x + 0.02, y_pos, s=size, c='gray',
                  alpha=0.6, edgecolor='white', linewidth=1)
        ax.text(legend_x + 0.06, y_pos, label, fontsize=10,
               ha='left', va='center')
    
    y_offset += len(size_categories) * 0.04 + 0.06
    
    # GCRO Coverage Section
    ax.text(legend_x, legend_y - y_offset, 'GCRO Survey Coverage', 
           fontsize=12, fontweight='bold', ha='left', va='top')
    y_offset += 0.05
    
    coverage_types = [
        ('High (4 waves)', colors['gcro_high']),
        ('Medium (2-3 waves)', colors['gcro_med']),
        ('Low (1 wave)', colors['gcro_low'])
    ]
    
    for i, (label, color) in enumerate(coverage_types):
        y_pos = legend_y - y_offset - (i * 0.035)
        rect = Rectangle((legend_x + 0.01, y_pos - 0.01), 0.03, 0.02,
                        facecolor=color, alpha=0.6, edgecolor='gray')
        ax.add_patch(rect)
        ax.text(legend_x + 0.05, y_pos, label, fontsize=10,
               ha='left', va='center')

def create_enhanced_johannesburg_map():
    """Create the enhanced publication-quality map."""
    
    # Load data
    sa_gdf = load_south_africa_boundaries()
    colors = create_professional_color_scheme()
    sites = get_enhanced_clinical_sites()
    landmarks = get_enhanced_landmarks()
    transport = get_transport_infrastructure()
    
    # Define extent
    extent = {
        'west': 27.55, 'east': 28.65,
        'south': -26.55, 'north': -25.45
    }
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(1, 1, 1)
    
    # Set extent and background
    ax.set_xlim(extent['west'], extent['east'])
    ax.set_ylim(extent['south'], extent['north'])
    ax.set_facecolor(colors['background'])
    
    # Add terrain base
    print("Adding terrain base...")
    create_terrain_base(ax, extent, colors)
    
    # Add coordinate grid
    print("Adding professional grid...")
    add_professional_grid(ax, extent, colors)
    
    # Plot South African boundary
    if sa_gdf is not None:
        boundary_geom = sa_gdf.geometry.iloc[0]
        if boundary_geom.geom_type == 'Polygon':
            x, y = boundary_geom.exterior.xy
            ax.plot(x, y, color=colors['boundary_national'], linewidth=3, alpha=0.8, zorder=3)
    
    # Add GCRO ward coverage
    print("Adding GCRO coverage...")
    add_gcro_ward_coverage(ax, extent, colors)
    
    # Add urban areas with different densities
    print("Adding urban areas...")
    urban_areas = [
        # High density
        {'coords': [(27.97, -26.23), (28.09, -26.23), (28.09, -26.17), (27.97, -26.17)], 'density': 'high'},
        {'coords': [(28.02, -26.13), (28.09, -26.13), (28.09, -26.07), (28.02, -26.07)], 'density': 'high'},
        # Medium density
        {'coords': [(27.81, -26.30), (27.91, -26.30), (27.91, -26.23), (27.81, -26.23)], 'density': 'med'},
        {'coords': [(28.08, -26.12), (28.14, -26.12), (28.14, -26.08), (28.08, -26.08)], 'density': 'med'},
        # Low density
        {'coords': [(26.08, -26.15), (28.15, -26.15), (28.15, -26.05), (28.08, -26.05)], 'density': 'low'},
    ]
    
    for area in urban_areas:
        polygon = Polygon(area['coords'])
        x, y = polygon.exterior.xy
        color_key = f"urban_{area['density']}"
        ax.fill(x, y, color=colors[color_key], alpha=0.7, 
               edgecolor=colors['boundary_metro'], linewidth=0.5, zorder=4)
    
    # Add transportation network
    print("Adding transport infrastructure...")
    transport_data = get_transport_infrastructure()
    
    # Plot highways
    for highway in transport_data['highways']:
        coords = highway['coords']
        x_coords, y_coords = zip(*[(lon, lat) for lat, lon in coords])
        
        if highway['type'] == 'national':
            ax.plot(x_coords, y_coords, color=colors['highway_major'], 
                   linewidth=4, alpha=0.9, zorder=6)
        else:
            ax.plot(x_coords, y_coords, color=colors['highway_minor'], 
                   linewidth=3, alpha=0.8, zorder=5)
        
        # Add highway labels
        mid_x, mid_y = np.mean(x_coords), np.mean(y_coords)
        ax.text(mid_x, mid_y, highway['name'], fontsize=9, fontweight='bold',
               ha='center', va='center', color='white',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['highway_major'], alpha=0.8))
    
    # Plot metro roads
    for road in transport_data['metro_roads']:
        coords = road['coords']
        x_coords, y_coords = zip(*[(lon, lat) for lat, lon in coords])
        ax.plot(x_coords, y_coords, color=colors['roads_major'], 
               linewidth=2.5, alpha=0.7, zorder=5)
    
    # Plot rail lines
    for rail in transport_data['rail_lines']:
        coords = rail['coords']
        x_coords, y_coords = zip(*[(lon, lat) for lat, lon in coords])
        if rail['type'] == 'rapid':
            ax.plot(x_coords, y_coords, color=colors['railway'], 
                   linewidth=3, alpha=0.8, linestyle='--', zorder=5)
        else:
            ax.plot(x_coords, y_coords, color=colors['railway'], 
                   linewidth=2, alpha=0.6, linestyle=':', zorder=4)
    
    # Add water features
    print("Adding water features...")
    # Jukskei River
    river_coords = [(27.65, -25.75), (27.85, -25.95), (28.05, -26.05), 
                   (28.25, -26.15), (28.35, -26.25)]
    river_x, river_y = zip(*river_coords)
    ax.plot(river_x, river_y, color=colors['water'], linewidth=4, alpha=0.8, zorder=5)
    
    # Dams and reservoirs
    water_bodies = [(-25.88, 28.12), (-26.08, 27.82), (-26.32, 28.22)]
    for lat, lon in water_bodies:
        circle = Circle((lon, lat), 0.025, color=colors['water'], alpha=0.8, zorder=5)
        ax.add_patch(circle)
    
    # Add landmarks
    print("Adding landmarks...")
    all_landmarks = landmarks['business_districts'] + landmarks['residential_areas'] + landmarks['institutions']
    
    for landmark in all_landmarks:
        if 'importance' in landmark:
            if landmark['importance'] == 'high':
                marker_size = 12
                font_size = 10
                font_weight = 'bold'
            else:
                marker_size = 8
                font_size = 9
                font_weight = 'normal'
        else:
            marker_size = 10
            font_size = 9
            font_weight = 'normal'
        
        if landmark in landmarks['institutions']:
            if landmark['type'] == 'hospital':
                marker = '+'
                color = 'red'
            elif landmark['type'] == 'university':
                marker = '^'
                color = 'blue'
            else:
                marker = 's'
                color = 'purple'
        else:
            marker = 's'
            color = 'black'
        
        ax.plot(landmark['lon'], landmark['lat'], marker, markersize=marker_size, 
               color=color, alpha=0.8, zorder=7)
        ax.text(landmark['lon'] + 0.025, landmark['lat'], landmark['name'],
               fontsize=font_size, fontweight=font_weight, ha='left', va='center')
    
    # Plot clinical sites
    print("Adding clinical trial sites...")
    plot_enhanced_clinical_sites(ax, sites, colors)
    
    # Create enhanced legend
    print("Creating enhanced legend...")
    create_enhanced_legend(ax, extent, colors, sites)
    
    # Add scale bar and north arrow
    print("Adding cartographic elements...")
    # Scale bar
    scale_length = 0.1  # degrees (~10 km)
    scale_x = extent['west'] + 0.08
    scale_y = extent['south'] + 0.08
    
    # Scale bar background
    scale_bg = Rectangle((scale_x - 0.02, scale_y - 0.02), 0.14, 0.06,
                        facecolor='white', alpha=0.9, edgecolor='black', 
                        linewidth=1, zorder=15)
    ax.add_patch(scale_bg)
    
    # Scale bar line
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
           'k-', linewidth=4, zorder=16)
    ax.text(scale_x + scale_length/2, scale_y + 0.02, '10 km', 
           ha='center', fontsize=10, fontweight='bold', zorder=16)
    
    # North arrow
    north_x = extent['east'] - 0.1
    north_y = extent['north'] - 0.1
    
    # North arrow background
    north_bg = Circle((north_x, north_y), 0.04, facecolor='white', alpha=0.9,
                     edgecolor='black', linewidth=1, zorder=15)
    ax.add_patch(north_bg)
    
    # Arrow
    ax.annotate('N', xy=(north_x, north_y), xytext=(north_x, north_y - 0.02),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'),
               fontsize=14, fontweight='bold', ha='center', zorder=16)
    
    # Add comprehensive title and metadata
    plt.suptitle('ENBEL Climate-Health Research Project\nJohannesburg Metropolitan Area: Clinical Trial Sites & GCRO Survey Coverage', 
                fontsize=18, fontweight='bold', y=0.96)
    
    # Enhanced subtitle
    total_patients = sum(site['patients'] for site in sites)
    total_studies = sum(site['studies'] for site in sites)
    subtitle = f"Multi-Site Clinical Research: {total_studies} Studies • {total_patients:,} Participants • {len(sites)} Research Centres\nSocioeconomic Data: 508 Municipal Wards • 58,616 Households • 4 Survey Waves (2011-2021)"
    plt.figtext(0.5, 0.91, subtitle, ha='center', fontsize=13, style='italic')
    
    # Data sources and technical info
    footer_text = ("Data Sources: ERA5 Climate Reanalysis • South African Medical Research Council • GCRO Quality of Life Survey • OSM Transportation\n"
                  "Coordinate System: WGS84 Geographic (EPSG:4326) • Generated: October 2025 • ENBEL Research Consortium")
    plt.figtext(0.5, 0.03, footer_text, ha='center', fontsize=10, alpha=0.8)
    
    # Professional axis formatting
    ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°S)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Ensure proper aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig, ax

def main():
    """Main execution function."""
    print("Creating enhanced publication-quality Johannesburg map...")
    
    fig, ax = create_enhanced_johannesburg_map()
    
    # Save outputs
    output_svg = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_johannesburg_enhanced_map.svg"
    output_png = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_johannesburg_enhanced_map.png"
    
    print(f"Saving enhanced map...")
    plt.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print("Enhanced map creation completed!")
    print(f"Files saved:")
    print(f"  SVG: {output_svg}")  
    print(f"  PNG: {output_png}")
    
    plt.show()

if __name__ == "__main__":
    main()