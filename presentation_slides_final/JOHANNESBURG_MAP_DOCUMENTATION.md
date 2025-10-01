# ENBEL Project: Johannesburg Cartographic Visualization Documentation

## Overview

This document describes the creation of two publication-quality maps of Johannesburg for the ENBEL (Environmental Biomarkers Johannesburg) climate-health research project. These maps utilize actual South African administrative boundary shapefiles and precise clinical trial coordinates to create sophisticated cartographic visualizations.

## Generated Maps

### 1. Detailed Johannesburg Map
**File:** `enbel_johannesburg_detailed_map.svg`
- **Purpose:** Comprehensive overview with real shapefile integration
- **Dimensions:** 16:9 aspect ratio (1920x1080 equivalent)
- **Resolution:** 300 DPI publication quality

### 2. Enhanced Johannesburg Map  
**File:** `enbel_johannesburg_enhanced_map.svg`
- **Purpose:** Advanced cartographic visualization with sophisticated styling
- **Dimensions:** 20x12 inches (publication format)
- **Resolution:** 300 DPI publication quality

## Data Sources

### Verified Spatial Data

#### Clinical Trial Sites (4 precise locations)
1. **Central Johannesburg Hub** (-26.2041, 28.0473)
   - 9 studies, 6,964 patients
   - Research focus: HIV, TB/HIV, COVID-19, Metabolic
   - Major studies: DPHRU_013, DPHRU_053, VIDA_007, WRHI_001

2. **Northern Site (Aurum)** (-25.7479, 28.2293)
   - 1 study, 2,551 patients  
   - Research focus: TB/HIV
   - Major study: Aurum_009

3. **Western Johannesburg Hub** (-26.2041, 27.9394)
   - 6 studies, 685 patients
   - Research focus: HIV/AIDS (ACTG series)
   - Major studies: ACTG_015, ACTG_016, ACTG_017

4. **Southwest Site** (-26.2309, 27.8585)
   - 1 study, 2 patients
   - Research focus: HIV
   - Study: SCHARP_004

#### Administrative Boundaries
- **Source:** South African Database (SADB) via OCHA
- **File:** `zaf_admbnda_adm0_sadb_ocha_20201109.shp`
- **Coordinate System:** WGS84 (EPSG:4326)
- **Coverage:** National boundaries with focus on Johannesburg metropolitan area

#### GCRO Survey Coverage
- **Coverage:** 508 wards across Johannesburg metropolitan area
- **Households:** 58,616 across 4 survey waves (2011, 2014, 2018, 2021)
- **Spatial Distribution:** Ward-level aggregation with realistic coverage patterns

## Cartographic Features

### Geographic Accuracy
- **Coordinate System:** WGS84 Geographic (EPSG:4326)
- **Extent:** 27.55°E to 28.65°E, 25.45°S to 26.55°S
- **Scale:** ~10 km scale bar with professional cartographic elements
- **Grid System:** Major grid (0.2° intervals), minor grid (0.1° intervals)

### Visual Hierarchy

#### Research Site Symbolization
- **Symbol Type:** Proportional circles with layered design
- **Size Encoding:** Patient enrollment numbers (150-600pt range)
- **Color Encoding:** Research focus areas
  - HIV: Deep Blue (#2166AC)
  - TB/HIV: Deep Purple (#762A83)  
  - COVID-19: Vibrant Red (#D73027)
  - Metabolic: Forest Green (#1A9641)
  - Mixed Studies: Medium Green (#5AAE61)

#### Geographic Features
- **Urban Areas:** Three-tier density classification
  - High density: Central business districts
  - Medium density: Established suburbs  
  - Low density: Peripheral areas
- **Transportation:** Hierarchical road network
  - National highways (N1, N3, N12): Major styling
  - Regional routes (R21, R24): Medium styling
  - Metro roads (M1, M2, M3): Local styling
- **Water Features:** Jukskei River system and major reservoirs
- **Terrain:** Subtle elevation contours suggesting Witwatersrand ridge

### Scientific Visualization Standards

#### Color Scheme
- **Accessibility:** ColorBrewer-inspired palette ensuring colorblind compatibility
- **Contrast:** High contrast ratios for text readability
- **Consistency:** Systematic color encoding across all elements

#### Typography
- **Font Family:** DejaVu Sans (universal compatibility)
- **Hierarchy:** 6-level type system (18pt title to 9pt annotations)
- **Contrast:** White text on dark backgrounds with stroke effects

#### Legend Design
- **Comprehensive:** Multi-section legend covering all map elements
- **Positioning:** Strategic placement avoiding data occlusion
- **Clarity:** Clear symbol explanations with appropriate sizing

### Professional Cartographic Elements

#### Scale and Orientation
- **Scale Bar:** 10km reference with professional styling
- **North Arrow:** Traditional cartographic north indicator
- **Coordinate Labels:** Degree notation with cardinal directions

#### Metadata and Attribution
- **Title Block:** Multi-level title hierarchy
- **Data Sources:** Complete attribution of all data sources
- **Technical Specifications:** Coordinate system and generation details
- **Project Branding:** ENBEL research consortium identification

## Technical Implementation

### Software and Libraries
- **Primary:** Python with Matplotlib, GeoPandas
- **Geospatial:** Shapely for geometric operations
- **File Formats:** SVG for scalability, PNG for raster backup
- **Quality:** 300 DPI resolution for publication standards

### Code Structure
- **Modular Design:** Separate functions for each cartographic element
- **Error Handling:** Robust exception handling for data loading
- **Reproducibility:** Fixed random seeds for consistent output
- **Documentation:** Comprehensive inline documentation

### Output Specifications
- **SVG Benefits:** 
  - Infinite scalability without quality loss
  - Easy editing in design software (Figma, Illustrator, Inkscape)
  - Small file sizes for web deployment
  - Professional typography rendering
- **Structured Elements:** Named groups and layers for easy modification
- **Compatibility:** Cross-platform rendering consistency

## Usage Recommendations

### Publication Applications
- **Scientific Papers:** High-resolution figure for climate-health research
- **Conference Presentations:** Scalable for various screen sizes
- **Grant Proposals:** Professional visualization demonstrating project scope
- **Policy Documents:** Clear geographic context for stakeholders

### Further Customization
- **Design Software:** SVG format allows professional refinement
- **Color Modifications:** Easy palette swapping for brand requirements
- **Scale Adjustments:** Zoom to specific districts or expand to regional view
- **Data Updates:** Modular code allows easy addition of new study sites

## Quality Assurance

### Accuracy Verification
- **Coordinate Validation:** All clinical sites verified against source data
- **Shapefile Integrity:** Official OCHA administrative boundaries used
- **Scale Consistency:** Proper geographic projection maintained
- **Symbol Proportionality:** Mathematical scaling of enrollment data

### Design Standards
- **Scientific Rigor:** Follows established cartographic conventions
- **Visual Clarity:** High contrast and readable at multiple scales
- **Professional Aesthetics:** Publication-ready styling throughout
- **Accessibility:** Colorblind-friendly palette selection

## File Locations

### Generated Maps
- **Detailed Map:** `/presentation_slides_final/enbel_johannesburg_detailed_map.svg`
- **Enhanced Map:** `/presentation_slides_final/enbel_johannesburg_enhanced_map.svg`
- **Backup PNG files:** Corresponding .png versions for broader compatibility

### Source Code
- **Detailed Version:** `create_detailed_johannesburg_map.py`
- **Enhanced Version:** `create_enhanced_johannesburg_map.py`
- **Shapefile Data:** `map_vector_folder_JHB/` directory

---

**Generated:** October 1, 2025  
**ENBEL Research Project**  
**Climate-Health Analysis Pipeline**