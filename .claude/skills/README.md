# R Cartography Skills - Expert Guide

## Overview

This skills package provides comprehensive, expert-level guidance for creating publication-quality maps in R with SVG export. Suitable for:

- **Peer-reviewed journals** (Nature, Science, The Lancet, etc.)
- **Conference presentations** (talks and posters)
- **Media publications** (newspapers, magazines)
- **Social media** (Twitter, Instagram, LinkedIn)
- **Reports and dissertations**

---

## Skills Structure

### 1. [R_CARTOGRAPHY_FUNDAMENTALS.md](R_CARTOGRAPHY_FUNDAMENTALS.md)
**Core mapping workflows and techniques**

**What you'll learn:**
- Essential R packages (sf, ggplot2, tmap)
- Three main workflows (ggplot2+sf, tmap, mapsf)
- Multi-panel and faceted maps
- Inset maps for context
- Color palette selection (viridis, scico, ColorBrewer)
- Projection selection guide
- Complete examples with code

**Best for:**
- Getting started with R cartography
- Choosing the right workflow for your project
- Understanding package ecosystems
- Quick reference templates

**Example use cases:**
- Creating a choropleth map of survey data
- Adding survey points to district boundaries
- Making temporal comparison maps (facets)
- Building multi-layer study region maps

---

### 2. [SVG_EXPORT_WORKFLOWS.md](SVG_EXPORT_WORKFLOWS.md)
**High-quality vector graphics export**

**What you'll learn:**
- SVG vs PNG vs PDF (when to use each)
- Export methods (ggsave, svglite, Cairo)
- Size guidelines for different contexts
- Font handling and embedding
- File size optimization
- Multi-format export pipelines
- Post-processing with Inkscape
- Troubleshooting common issues

**Best for:**
- Preparing figures for journal submission
- Creating scalable graphics for presentations
- Understanding technical export options
- Optimizing file sizes for web use

**Example use cases:**
- Exporting a map for Nature journal submission (89mm width)
- Creating presentation slides (16:9 ratio)
- Generating social media graphics (Instagram square)
- Batch-exporting in multiple formats (SVG + PNG + PDF)

---

### 3. [PUBLICATION_MAP_STYLING.md](PUBLICATION_MAP_STYLING.md)
**Context-specific design for maximum impact**

**What you'll learn:**
- Five complete styling approaches:
  1. **Peer-reviewed journals** (conservative, rigorous)
  2. **Conference presentations** (bold, high-contrast)
  3. **Newspapers/magazines** (eye-catching, accessible)
  4. **Scientific posters** (large format, self-contained)
  5. **Social media** (instant impact, mobile-optimized)
- Color theory for maps
- Typography guidelines
- Text size hierarchies
- Branded templates
- Context-specific checklists

**Best for:**
- Adapting your map to the right audience
- Understanding design principles
- Creating visually compelling graphics
- Building organization-specific styles

**Example use cases:**
- Converting a journal map to a presentation slide
- Creating a Twitter-friendly version of your research
- Designing a conference poster with proper hierarchy
- Building a branded map template for your institution

---

### 4. [SCIENTIFIC_CARTOGRAPHY_STANDARDS.md](SCIENTIFIC_CARTOGRAPHY_STANDARDS.md)
**Rigorous geospatial analysis and reproducibility**

**What you'll learn:**
- Coordinate reference systems (CRS) and projections
- Scale bars and distance calculations
- North arrows and orientation
- Data quality and uncertainty visualization
- Spatial resolution documentation
- Topology validation
- Spatial autocorrelation testing
- Reproducibility standards
- Complete metadata documentation
- Quality control checklists

**Best for:**
- Ensuring scientific rigor
- Meeting journal requirements
- Validating spatial data
- Creating reproducible research
- Advanced spatial analysis

**Example use cases:**
- Selecting the right projection for your region
- Testing for spatial autocorrelation (Moran's I)
- Documenting data provenance
- Validating geometry integrity
- Creating LISA (Local Indicators of Spatial Association) maps

---

## Quick Start Guide

### For First-Time Users

**Start here:** [R_CARTOGRAPHY_FUNDAMENTALS.md](R_CARTOGRAPHY_FUNDAMENTALS.md)

1. Install required packages:
```r
install.packages(c("sf", "ggplot2", "ggspatial", "viridis", "svglite"))
```

2. Follow the "Basic Template" section (Workflow 1: ggplot2 + sf)

3. Save your map:
```r
ggsave("my_first_map.svg", width = 7, height = 5, dpi = 300)
```

### For Journal Submissions

**Read these in order:**

1. [R_CARTOGRAPHY_FUNDAMENTALS.md](R_CARTOGRAPHY_FUNDAMENTALS.md) - Build your map
2. [SCIENTIFIC_CARTOGRAPHY_STANDARDS.md](SCIENTIFIC_CARTOGRAPHY_STANDARDS.md) - Ensure rigor
3. [PUBLICATION_MAP_STYLING.md](PUBLICATION_MAP_STYLING.md) - Apply journal style
4. [SVG_EXPORT_WORKFLOWS.md](SVG_EXPORT_WORKFLOWS.md) - Export to journal specs

**Key checklist:**
- [ ] Use appropriate projection (UTM or equal-area)
- [ ] Add scale bar with units
- [ ] Include north arrow
- [ ] Use colorblind-safe palette
- [ ] Document all data sources in caption
- [ ] Export at 300 DPI
- [ ] Test in grayscale
- [ ] Match journal size requirements

### For Presentations

**Focus on:** [PUBLICATION_MAP_STYLING.md](PUBLICATION_MAP_STYLING.md) - Style 2

**Quick tips:**
- Base size = 18pt (readable from distance)
- High contrast colors
- Bold, simple message
- 16:9 aspect ratio (10" × 5.625")
- Minimal text

### For Media/Public Engagement

**Focus on:** [PUBLICATION_MAP_STYLING.md](PUBLICATION_MAP_STYLING.md) - Styles 3 & 5

**Quick tips:**
- No technical jargon
- Eye-catching colors
- Human-interest framing
- Self-explanatory
- Square format for social media

---

## Common Workflows

### Workflow 1: Basic Choropleth Map

**Goal:** Show district-level data with colors

**Skills needed:** R_CARTOGRAPHY_FUNDAMENTALS.md (Workflow 1)

**Time:** 15 minutes

```r
library(sf)
library(ggplot2)

# Load data
districts <- st_read("districts.shp")

# Create map
ggplot() +
  geom_sf(data = districts, aes(fill = variable_name)) +
  scale_fill_viridis_c() +
  theme_minimal()
```

---

### Workflow 2: Study Region Map with Points

**Goal:** Show survey locations on district boundaries

**Skills needed:** R_CARTOGRAPHY_FUNDAMENTALS.md (Basic Template)

**Time:** 30 minutes

```r
library(sf)
library(ggplot2)
library(ggspatial)

# Load data
districts <- st_read("districts.shp")
points <- st_read("survey_points.gpkg")

# Transform to common CRS
districts <- st_transform(districts, crs = 32735)
points <- st_transform(points, crs = 32735)

# Create map
ggplot() +
  geom_sf(data = districts, fill = "gray95", color = "gray40") +
  geom_sf(data = points, aes(color = category), size = 2) +
  annotation_scale(location = "br") +
  annotation_north_arrow(location = "tl") +
  theme_minimal()
```

---

### Workflow 3: Temporal Comparison (Facets)

**Goal:** Show changes across time periods

**Skills needed:** R_CARTOGRAPHY_FUNDAMENTALS.md (Multi-Panel Maps)

**Time:** 45 minutes

```r
library(sf)
library(ggplot2)
library(dplyr)

# Prepare temporal data
districts_temporal <- districts %>%
  st_join(survey_data) %>%
  group_by(district_id, year) %>%
  summarize(mean_value = mean(variable), .groups = "drop")

# Create faceted map
ggplot() +
  geom_sf(data = districts_temporal, aes(fill = mean_value)) +
  scale_fill_viridis_c() +
  facet_wrap(~ year, ncol = 3) +
  theme_minimal()
```

---

### Workflow 4: Publication-Ready Scientific Map

**Goal:** Complete map with all cartographic elements

**Skills needed:** All four skill files

**Time:** 2 hours

**Steps:**
1. Load and validate data (SCIENTIFIC_CARTOGRAPHY_STANDARDS.md)
2. Select appropriate projection (SCIENTIFIC_CARTOGRAPHY_STANDARDS.md)
3. Create base map (R_CARTOGRAPHY_FUNDAMENTALS.md)
4. Apply journal styling (PUBLICATION_MAP_STYLING.md)
5. Add cartographic elements (scale, north arrow, legend)
6. Export to SVG (SVG_EXPORT_WORKFLOWS.md)
7. Validate and document (SCIENTIFIC_CARTOGRAPHY_STANDARDS.md)

---

## Package Installation

### Essential Packages

```r
# Core spatial
install.packages("sf")           # Simple Features
install.packages("terra")        # Raster data

# Visualization
install.packages("ggplot2")      # Grammar of graphics
install.packages("tmap")         # Thematic maps
install.packages("ggspatial")    # Scale bars, north arrows

# Color palettes
install.packages("viridis")      # Colorblind-safe
install.packages("scico")        # Scientific palettes
install.packages("RColorBrewer") # Classic palettes

# Export
install.packages("svglite")      # High-quality SVG
install.packages("Cairo")        # Alternative graphics device

# Data manipulation
install.packages("dplyr")
install.packages("tidyr")
```

### Advanced Packages

```r
# Spatial analysis
install.packages("spdep")        # Spatial autocorrelation
install.packages("spatstat")     # Spatial statistics

# Geometry operations
install.packages("rmapshaper")   # Simplify geometries
install.packages("lwgeom")       # Advanced geometry operations

# Utilities
install.packages("cowplot")      # Combine plots
install.packages("patchwork")    # Layout multiple plots
install.packages("ggrepel")      # Non-overlapping labels
install.packages("ggrastr")      # Rasterize complex layers
```

---

## Data Sources

### Recommended Sources for Spatial Data

#### Administrative Boundaries
- **GADM**: https://gadm.org/ (global boundaries)
- **Statistics South Africa**: https://www.statssa.gov.za/ (SA boundaries)
- **Natural Earth**: https://www.naturalearthdata.com/ (physical features)

#### Climate Data
- **ERA5**: https://cds.climate.copernicus.eu/ (reanalysis)
- **WorldClim**: https://www.worldclim.org/ (historical climate)
- **CHIRPS**: https://www.chc.ucsb.edu/data/chirps (precipitation)

#### Population & Demographics
- **WorldPop**: https://www.worldpop.org/ (population density)
- **GCRO**: https://gcro.ac.za/ (Johannesburg regional data)
- **Census data**: Country-specific statistical agencies

---

## Troubleshooting

### Common Issues

#### "Error: CRS not found"
→ See SCIENTIFIC_CARTOGRAPHY_STANDARDS.md - CRS section

#### "Fonts not rendering correctly"
→ See SVG_EXPORT_WORKFLOWS.md - Font Handling section

#### "SVG file is too large"
→ See SVG_EXPORT_WORKFLOWS.md - Optimizing SVG File Size section

#### "Labels overlapping"
→ See R_CARTOGRAPHY_FUNDAMENTALS.md - Common Issues section

#### "Colors look different in print"
→ See PUBLICATION_MAP_STYLING.md - Color Theory section

---

## Contributing

These skills are designed to be living documents. Suggestions for improvements:

1. Open an issue with specific examples
2. Provide code snippets that didn't work as expected
3. Share better approaches or newer packages
4. Request additional topics or workflows

---

## Version History

- **v1.0** (2025-01-30): Initial release
  - R_CARTOGRAPHY_FUNDAMENTALS.md
  - SVG_EXPORT_WORKFLOWS.md
  - PUBLICATION_MAP_STYLING.md
  - SCIENTIFIC_CARTOGRAPHY_STANDARDS.md

---

## License

These skills are provided under MIT License. Use freely for academic, commercial, or personal projects.

---

## Citation

If these skills helped your research, please cite as:

```
Claude Code Expert System (2025). R Cartography Skills: Expert Guide for
Publication-Quality Maps. https://github.com/your-repo/cartography-skills
```

---

**Last Updated**: 2025-01-30
**Maintained by**: Claude Code Expert System
**Version**: 1.0
