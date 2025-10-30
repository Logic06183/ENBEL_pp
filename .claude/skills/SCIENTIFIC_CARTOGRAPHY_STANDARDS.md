# Scientific Cartography Standards

## Rigorous Geospatial Analysis and Visualization

This skill provides comprehensive guidance for ensuring scientific rigor, accuracy, and reproducibility in cartographic work. Essential for peer-reviewed publications and professional geospatial analysis.

---

## The Pillars of Scientific Cartography

### 1. Spatial Accuracy
- Proper coordinate reference systems (CRS)
- Appropriate projections for the region
- Documented transformations
- Validated spatial relationships

### 2. Metadata Completeness
- Data sources with citations
- Processing methods documented
- Spatial resolution specified
- Temporal coverage stated
- Uncertainty quantified

### 3. Cartographic Elements
- Scale indicators
- Orientation (north arrow)
- Legend with units
- Coordinate grid (if appropriate)
- Inset maps for context

### 4. Reproducibility
- Code documented and versioned
- Random seeds set
- Package versions recorded
- Data provenance tracked
- Processing pipeline transparent

---

## Coordinate Reference Systems (CRS)

### Understanding CRS Components

```r
library(sf)

# Check CRS of spatial data
st_crs(districts)

# Example output interpretation:
# EPSG:32735 = UTM Zone 35S (Universal Transverse Mercator, Zone 35, Southern Hemisphere)
# Components:
#   - Datum: WGS 84 (World Geodetic System 1984)
#   - Projection: Transverse Mercator
#   - Zone: 35 (longitude range: 24°E to 30°E)
#   - Hemisphere: South
#   - Units: meters
```

### Selecting Appropriate Projections

#### For Local/Regional Analysis (e.g., Johannesburg)

```r
# UTM Zone 35S (preserves distances and areas locally)
districts <- st_transform(districts, crs = 32735)

# Rationale:
# - Johannesburg center: 28.05°E, 26.20°S
# - Falls in UTM Zone 35 (24°E - 30°E)
# - Minimal distortion within zone
# - Units in meters (convenient for measurements)
# - Standard for South African spatial data
```

#### For Country-Wide Analysis (South Africa)

```r
# Albers Equal Area Conic (preserves area)
districts <- st_transform(
  districts,
  crs = "+proj=aea +lat_1=-24 +lat_2=-33 +lat_0=-28.5 +lon_0=25 +datum=WGS84"
)

# Rationale:
# - Standard parallels at -24° and -33° (SA extent)
# - Central meridian at 25°E (center of country)
# - Equal-area property (accurate area calculations)
# - Suitable for choropleth maps
```

#### For Global Analysis

```r
# Robinson (compromise, visually pleasing)
world <- st_transform(world, crs = "+proj=robin")

# Mollweide (equal-area)
world <- st_transform(world, crs = "+proj=moll")

# Natural Earth (aesthetic, low distortion)
world <- st_transform(world, crs = "+proj=natearth")

# Winkel Tripel (used by National Geographic)
world <- st_transform(world, crs = "+proj=wintri")
```

#### For Web Mapping

```r
# Web Mercator (EPSG:3857) - for interactive maps only
districts_web <- st_transform(districts, crs = 3857)

# WARNING: Severely distorts areas at high latitudes
# DO NOT use for scientific analysis
# Acceptable for: Google Maps, Leaflet, Mapbox
```

### CRS Transformation Best Practices

```r
# Always check CRS before analysis
st_crs(layer1)
st_crs(layer2)

# Transform all layers to common CRS
target_crs <- 32735  # UTM 35S

layer1_proj <- st_transform(layer1, crs = target_crs)
layer2_proj <- st_transform(layer2, crs = target_crs)
layer3_proj <- st_transform(layer3, crs = target_crs)

# Verify transformation
st_crs(layer1_proj) == st_crs(layer2_proj)  # Should be TRUE
```

### Documenting Projections in Figures

```r
# Include in caption
labs(
  caption = paste0(
    "Projection: UTM Zone 35S (EPSG:32735, WGS 84 datum) | ",
    "Distortion: <0.1% within study area"
  )
)

# Or in methods text:
# "All spatial data were transformed to UTM Zone 35S (EPSG:32735)
# using the WGS 84 datum. This projection preserves distances and
# areas within the Johannesburg metropolitan region with less than
# 0.1% distortion."
```

---

## Scale and Distance

### Scale Bars

```r
library(ggspatial)

# Basic scale bar
annotation_scale(
  location = "br",        # bottom-right
  width_hint = 0.25,      # 25% of plot width
  style = "ticks",        # or "bar"
  line_width = 0.5,
  text_cex = 0.7
)

# Customized scale bar
annotation_scale(
  location = "br",
  width_hint = 0.2,
  unit_category = "metric",  # or "imperial"
  text_face = "bold",
  text_family = "Arial",
  text_col = "gray20",
  line_col = "gray20",
  pad_x = unit(0.5, "cm"),
  pad_y = unit(0.5, "cm")
)
```

### Calculating Distances

```r
library(sf)

# Distance between two points (in CRS units)
point1 <- st_point(c(28.05, -26.20)) %>%
  st_sfc(crs = 4326) %>%
  st_transform(32735)

point2 <- st_point(c(28.15, -26.10)) %>%
  st_sfc(crs = 4326) %>%
  st_transform(32735)

distance_m <- st_distance(point1, point2)
print(paste("Distance:", round(distance_m / 1000, 2), "km"))

# Area calculation
district_area_m2 <- st_area(districts)
district_area_km2 <- district_area_m2 / 1e6

districts$area_km2 <- as.numeric(district_area_km2)
```

### Scale Validation

```r
# Verify scale accuracy
bbox <- st_bbox(districts)
map_width_m <- bbox$xmax - bbox$xmin
map_height_m <- bbox$ymax - bbox$ymin

cat("Map dimensions:\n")
cat("  Width:", round(map_width_m / 1000, 2), "km\n")
cat("  Height:", round(map_height_m / 1000, 2), "km\n")

# Check against known distances
# Example: Johannesburg CBD to OR Tambo Airport ≈ 22 km
```

---

## North Arrow and Orientation

### Types of North

```r
# True North (geographic pole)
annotation_north_arrow(
  which_north = "true",
  location = "tl"
)

# Grid North (parallel to map projection grid)
annotation_north_arrow(
  which_north = "grid",
  location = "tl"
)

# Magnetic North (compass direction)
annotation_north_arrow(
  which_north = "magnetic",
  location = "tl"
)
```

### North Arrow Styles

```r
# Fancy orienteering style (traditional)
annotation_north_arrow(
  location = "tl",
  which_north = "true",
  pad_x = unit(0.3, "in"),
  pad_y = unit(0.3, "in"),
  style = north_arrow_fancy_orienteering(
    fill = c("gray20", "white"),
    line_col = "gray20",
    text_size = 10
  )
)

# Minimal style (modern)
annotation_north_arrow(
  location = "tr",
  which_north = "true",
  style = north_arrow_minimal(
    line_width = 1,
    line_col = "gray20",
    fill = "gray20",
    text_col = "gray20"
  )
)

# Nautical style
annotation_north_arrow(
  location = "bl",
  style = north_arrow_nautical()
)
```

### When to Omit North Arrow

- Global maps (no single "up" direction)
- Maps with standard orientation (north up, implicit)
- Small-scale inset maps
- Schematic/conceptual diagrams

---

## Data Quality and Uncertainty

### Documenting Data Sources

```r
# Complete citation in caption
labs(
  caption = paste0(
    "Data Sources:\n",
    "Administrative boundaries: Statistics South Africa (2021), ",
    "Municipal Demarcation Board v2021.1\n",
    "Survey data: Greater Capital Region Observatory (GCRO), ",
    "Quality of Life Survey 2021, doi:10.1234/gcro.2021\n",
    "Climate data: ERA5 reanalysis (Hersbach et al., 2020), ",
    "Copernicus Climate Data Store\n",
    "Projection: UTM Zone 35S (EPSG:32735)"
  )
)
```

### Visualizing Uncertainty

#### Confidence Intervals as Error Bars

```r
# For point data with uncertainty
ggplot(survey_points) +
  geom_sf(aes(color = mean_value)) +
  geom_errorbar(
    aes(x = lon, y = lat,
        ymin = mean_value - se,
        ymax = mean_value + se),
    width = 0.01
  )
```

#### Transparency for Data Quality

```r
# Lower alpha for uncertain data
ggplot() +
  geom_sf(
    data = districts,
    aes(fill = estimate, alpha = data_quality),
    color = "gray50"
  ) +
  scale_alpha_continuous(
    range = c(0.3, 1.0),
    name = "Data Quality\n(sample size)",
    guide = guide_legend(override.aes = list(fill = "gray50"))
  )
```

#### Bivariate Maps (Value + Uncertainty)

```r
library(biscale)

# Create bivariate classes
districts_bivariate <- bi_class(
  districts,
  x = estimate,
  y = uncertainty,
  style = "quantile",
  dim = 3
)

# Plot
ggplot() +
  geom_sf(
    data = districts_bivariate,
    aes(fill = bi_class),
    color = "white",
    size = 0.1,
    show.legend = FALSE
  ) +
  bi_scale_fill(pal = "DkViolet", dim = 3) +
  bi_theme() +
  labs(title = "Estimate (red) and Uncertainty (blue)")

# Add bivariate legend
bi_legend(
  pal = "DkViolet",
  dim = 3,
  xlab = "Higher estimate",
  ylab = "Higher uncertainty"
)
```

---

## Spatial Resolution and Aggregation

### Documenting Spatial Resolution

```r
# For raster data
library(terra)

climate_raster <- rast("climate_data.tif")
res_km <- res(climate_raster) / 1000  # Convert m to km

cat("Spatial resolution:", res_km[1], "km ×", res_km[2], "km\n")

# Include in figure caption
labs(
  caption = paste0(
    "Climate data resolution: ",
    round(res_km[1], 1), " km × ",
    round(res_km[2], 1), " km ",
    "(ERA5 reanalysis, 0.25° grid)"
  )
)
```

### Appropriate Aggregation

```r
# Point data → Polygon aggregation
library(dplyr)

# Calculate summary statistics by district
district_summary <- survey_points %>%
  st_drop_geometry() %>%
  group_by(district_id) %>%
  summarize(
    n_samples = n(),
    mean_value = mean(value, na.rm = TRUE),
    sd_value = sd(value, na.rm = TRUE),
    se_value = sd_value / sqrt(n_samples),
    .groups = "drop"
  )

# Join to spatial data
districts_with_data <- districts %>%
  left_join(district_summary, by = "district_id")

# Visualize with sample size annotation
ggplot() +
  geom_sf(data = districts_with_data, aes(fill = mean_value)) +
  geom_sf_text(
    data = districts_with_data,
    aes(label = paste0("n=", n_samples)),
    size = 2.5,
    color = "gray20"
  )
```

### Modifiable Areal Unit Problem (MAUP)

```r
# Sensitivity analysis: Multiple aggregation levels
results_ward <- aggregate_to_ward(data)
results_district <- aggregate_to_district(data)
results_metro <- aggregate_to_metro(data)

# Compare results
cat("Mean value by aggregation level:\n")
cat("  Ward:", mean(results_ward$value), "\n")
cat("  District:", mean(results_district$value), "\n")
cat("  Metro:", mean(results_metro$value), "\n")

# Document in methods:
# "We conducted sensitivity analyses at multiple spatial aggregation
# levels (ward, district, metropolitan) to assess potential bias from
# the Modifiable Areal Unit Problem (MAUP). Results were consistent
# across scales (correlation r = 0.94, p < 0.001)."
```

---

## Topology and Spatial Relationships

### Ensuring Valid Geometries

```r
library(sf)

# Check for invalid geometries
invalid <- !st_is_valid(districts)
if (any(invalid)) {
  warning(sum(invalid), " invalid geometries found")

  # Fix invalid geometries
  districts <- st_make_valid(districts)

  # Verify fix
  if (all(st_is_valid(districts))) {
    cat("✓ All geometries now valid\n")
  }
}
```

### Testing Spatial Relationships

```r
# Check for gaps between polygons
gaps <- st_difference(
  st_union(districts),
  st_convex_hull(st_union(districts))
)

if (st_is_empty(gaps)) {
  cat("✓ No gaps in polygon coverage\n")
} else {
  warning("Gaps detected in polygon coverage")
}

# Check for overlaps
overlaps <- st_intersection(districts, districts)
overlaps <- overlaps %>%
  filter(id.1 != id.2)

if (nrow(overlaps) == 0) {
  cat("✓ No overlapping polygons\n")
} else {
  warning(nrow(overlaps), " overlapping polygons detected")
}
```

### Spatial Joins with Validation

```r
# Join points to polygons
points_with_district <- st_join(
  survey_points,
  districts,
  join = st_within
)

# Validate join
n_unmatched <- sum(is.na(points_with_district$district_id))

cat("Spatial join results:\n")
cat("  Total points:", nrow(survey_points), "\n")
cat("  Matched to districts:", nrow(survey_points) - n_unmatched, "\n")
cat("  Unmatched:", n_unmatched, "\n")

if (n_unmatched > 0) {
  warning(n_unmatched, " points outside district boundaries")

  # Identify problematic points
  unmatched_points <- points_with_district %>%
    filter(is.na(district_id))

  # Plot for inspection
  ggplot() +
    geom_sf(data = districts, fill = "gray90") +
    geom_sf(data = unmatched_points, color = "red", size = 2)
}
```

---

## Reproducibility Standards

### Session Information

```r
# Always include session info in supplementary materials
writeLines(
  capture.output(sessionInfo()),
  "output/session_info.txt"
)

# Key information to report:
# - R version
# - Package versions (sf, ggplot2, terra, etc.)
# - Operating system
# - GDAL/PROJ versions
```

### Reproducible Random Seeds

```r
# Set seed for any stochastic operations
set.seed(42)

# Spatially-stratified sampling (reproducible)
sample_points <- st_sample(
  districts,
  size = 1000,
  type = "random"
)
# Note: st_sample() uses R's RNG, so set.seed() ensures reproducibility
```

### Code Documentation

```r
#!/usr/bin/env Rscript
#
# create_vulnerability_map.R
#
# Purpose: Generate publication-quality map of heat vulnerability
# Author: Your Name <email@example.com>
# Date: 2025-01-30
# R Version: 4.3.1
#
# Dependencies:
#   - sf (≥1.0-14)
#   - ggplot2 (≥3.4.0)
#   - viridis (≥0.6.4)
#   - ggspatial (≥1.1.9)
#
# Inputs:
#   - data/johannesburg_districts.shp (SA 2021 boundaries)
#   - data/vulnerability_index.csv (GCRO 2021 survey)
#
# Outputs:
#   - figures/figure1_vulnerability_map.svg (7" × 5.5")
#   - figures/figure1_vulnerability_map.png (300 DPI)
#
# Citation:
#   Please cite as: Surname, A. (2025). Heat vulnerability in
#   Johannesburg. Journal Name, 10(1), 1-20. doi:10.1234/journal.2025
#
# License: MIT

# Load required packages
library(sf)
library(ggplot2)
library(viridis)
library(ggspatial)

# Set reproducible environment
set.seed(42)
options(scipen = 999)

# Define paths (relative to project root)
DATA_DIR <- "data"
OUTPUT_DIR <- "figures"

# ... rest of script ...
```

### Data Provenance

```r
# Create metadata file
metadata <- list(
  title = "Heat Vulnerability Map - Johannesburg",
  description = "Ward-level heat vulnerability index",
  created_date = Sys.Date(),
  author = "Your Name",
  data_sources = list(
    boundaries = list(
      name = "Municipal boundaries",
      provider = "Statistics South Africa",
      year = 2021,
      url = "https://example.com/data",
      accessed = "2025-01-15",
      license = "CC BY 4.0"
    ),
    vulnerability = list(
      name = "Quality of Life Survey",
      provider = "GCRO",
      year = 2021,
      doi = "10.1234/gcro.2021",
      sample_size = 58616,
      spatial_resolution = "Ward-level aggregation",
      license = "Open Database License"
    )
  ),
  projection = list(
    epsg = 32735,
    name = "UTM Zone 35S",
    datum = "WGS 84",
    units = "meters"
  ),
  processing = list(
    steps = c(
      "1. Load shapefiles",
      "2. Transform to UTM 35S",
      "3. Join vulnerability data",
      "4. Validate geometries",
      "5. Create map visualization"
    ),
    software = list(
      R = R.version.string,
      sf = as.character(packageVersion("sf")),
      ggplot2 = as.character(packageVersion("ggplot2"))
    )
  )
)

# Save metadata
jsonlite::write_json(
  metadata,
  "output/map_metadata.json",
  pretty = TRUE
)
```

---

## Quality Control Checklist

### Pre-Analysis Checklist

- [ ] **All spatial data loaded successfully**
- [ ] **CRS checked for all layers**
- [ ] **Geometries validated (no invalid features)**
- [ ] **Spatial extent confirmed (bounding boxes)**
- [ ] **Attribute data joined correctly**
- [ ] **Missing data quantified**
- [ ] **Outliers identified**
- [ ] **Temporal coverage verified**

### Analysis Checklist

- [ ] **Appropriate CRS selected for region**
- [ ] **All layers transformed to common CRS**
- [ ] **Spatial relationships tested (joins, intersections)**
- [ ] **Topology validated (no gaps, overlaps)**
- [ ] **Distance/area calculations verified**
- [ ] **Sample sizes adequate (n ≥ 30 per unit)**
- [ ] **Statistical tests appropriate for spatial data**
- [ ] **Spatial autocorrelation assessed (if relevant)**

### Visualization Checklist

- [ ] **Scale bar present with correct units**
- [ ] **North arrow included (and type specified)**
- [ ] **Legend has clear title and units**
- [ ] **Color palette is colorblind-safe**
- [ ] **Projection documented in caption**
- [ ] **Data sources cited**
- [ ] **Spatial resolution stated**
- [ ] **Sample sizes reported**
- [ ] **Uncertainty visualized or noted**
- [ ] **Temporal coverage specified**

### Publication Checklist

- [ ] **Figure meets journal requirements (size, format, DPI)**
- [ ] **All text is legible at published size**
- [ ] **Caption provides complete information**
- [ ] **Methods section describes spatial analysis**
- [ ] **Code and data available (or archived)**
- [ ] **Session information saved**
- [ ] **Reproducibility tested (fresh R session)**
- [ ] **Ethical approval obtained (if using human data)**
- [ ] **Privacy protected (no personally identifiable locations)**

---

## Advanced: Spatial Autocorrelation

### Testing for Spatial Autocorrelation

```r
library(spdep)

# Create spatial weights matrix
neighbors <- poly2nb(districts, queen = TRUE)
weights <- nb2listw(neighbors, style = "W")

# Moran's I test
moran_test <- moran.test(
  districts$vulnerability_index,
  weights,
  alternative = "two.sided"
)

cat("Moran's I:", round(moran_test$estimate[1], 3), "\n")
cat("p-value:", format.pval(moran_test$p.value), "\n")

# Interpretation:
# Moran's I = 0.67, p < 0.001
# → Significant positive spatial autocorrelation
# → Neighboring areas have similar vulnerability levels

# Document in methods:
# "We assessed spatial autocorrelation using Moran's I statistic.
# Results indicated significant positive spatial autocorrelation
# (I = 0.67, p < 0.001), suggesting clustering of similar
# vulnerability levels. This justifies the use of spatial regression
# models."
```

### Visualizing Spatial Autocorrelation

```r
# Local Moran's I (LISA)
local_moran <- localmoran(
  districts$vulnerability_index,
  weights
)

districts$lisa_category <- case_when(
  local_moran[, 5] > 0.05 ~ "Not significant",
  local_moran[, 1] > 0 & districts$vulnerability_index > mean(districts$vulnerability_index) ~ "High-High",
  local_moran[, 1] > 0 & districts$vulnerability_index < mean(districts$vulnerability_index) ~ "Low-Low",
  local_moran[, 1] < 0 & districts$vulnerability_index > mean(districts$vulnerability_index) ~ "High-Low",
  local_moran[, 1] < 0 & districts$vulnerability_index < mean(districts$vulnerability_index) ~ "Low-High"
)

# LISA map
ggplot() +
  geom_sf(data = districts, aes(fill = lisa_category)) +
  scale_fill_manual(
    values = c(
      "High-High" = "#d7191c",
      "Low-Low" = "#2c7bb6",
      "High-Low" = "#fdae61",
      "Low-High" = "#abd9e9",
      "Not significant" = "gray90"
    ),
    name = "LISA Category"
  ) +
  theme_minimal() +
  labs(
    title = "Local Indicators of Spatial Association (LISA)",
    caption = "High-High: Hotspots | Low-Low: Coldspots | High-Low/Low-High: Spatial outliers"
  )
```

---

## Case Study: Complete Scientific Map

### Full Implementation with All Standards

```r
#!/usr/bin/env Rscript
# scientific_map_example.R
# Complete implementation of scientific cartography standards

library(sf)
library(ggplot2)
library(ggspatial)
library(viridis)
library(dplyr)
library(spdep)

# ============================================================================
# 1. DATA LOADING AND VALIDATION
# ============================================================================

# Load data
districts <- st_read("data/johannesburg_districts.shp")
survey_data <- read.csv("data/vulnerability_survey.csv")

# Validate geometries
if (!all(st_is_valid(districts))) {
  districts <- st_make_valid(districts)
  cat("✓ Fixed invalid geometries\n")
}

# Check CRS
cat("Original CRS:", st_crs(districts)$input, "\n")

# Transform to appropriate projection (UTM 35S)
districts <- st_transform(districts, crs = 32735)
cat("✓ Transformed to UTM Zone 35S\n")

# Join survey data
districts <- districts %>%
  left_join(survey_data, by = "district_id")

# Validate join
n_missing <- sum(is.na(districts$vulnerability_index))
cat("Districts with data:", nrow(districts) - n_missing, "/", nrow(districts), "\n")

# ============================================================================
# 2. SPATIAL ANALYSIS
# ============================================================================

# Calculate areas
districts$area_km2 <- as.numeric(st_area(districts)) / 1e6

# Test spatial autocorrelation
neighbors <- poly2nb(districts, queen = TRUE)
weights <- nb2listw(neighbors, style = "W")
moran <- moran.test(districts$vulnerability_index, weights)

cat("\nSpatial autocorrelation:\n")
cat("  Moran's I:", round(moran$estimate[1], 3), "\n")
cat("  p-value:", format.pval(moran$p.value), "\n")

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

# Create scientific map
map <- ggplot() +
  # Main data layer
  geom_sf(
    data = districts,
    aes(fill = vulnerability_index),
    color = "gray40",
    linewidth = 0.3
  ) +

  # Color scale (colorblind-safe)
  scale_fill_viridis_c(
    option = "rocket",
    name = "Heat Vulnerability Index\n(dimensionless, 0-100 scale)",
    na.value = "gray90",
    guide = guide_colorbar(
      barwidth = 10,
      barheight = 0.5,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +

  # Scale bar
  annotation_scale(
    location = "br",
    width_hint = 0.25,
    text_cex = 0.7,
    style = "ticks"
  ) +

  # North arrow
  annotation_north_arrow(
    location = "tl",
    which_north = "true",
    pad_x = unit(0.2, "in"),
    pad_y = unit(0.2, "in"),
    style = north_arrow_fancy_orienteering(
      fill = c("gray20", "white"),
      line_col = "gray20"
    )
  ) +

  # Theme
  theme_minimal(base_size = 10, base_family = "Arial") +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray92", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(15, 15, 15, 15),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  # Complete metadata
  labs(
    title = "Heat Vulnerability in Johannesburg Metropolitan Area",
    subtitle = "Ward-level composite vulnerability index (2021)",
    caption = paste0(
      "Data: Greater Capital Region Observatory (GCRO) Quality of Life Survey 2021 | ",
      "n = 58,616 household surveys across 258 wards | ",
      "Vulnerability index integrates: dwelling type, income, health access, age demographics | ",
      "Projection: UTM Zone 35S (EPSG:32735, WGS 84 datum) | ",
      "Spatial autocorrelation: Moran's I = ", round(moran$estimate[1], 3),
      " (p < 0.001) | ",
      "Analysis: ", R.version.string, " with sf ", packageVersion("sf")
    )
  )

# ============================================================================
# 4. EXPORT
# ============================================================================

# Create output directory
dir.create("figures", showWarnings = FALSE, recursive = TRUE)

# Save publication version (SVG)
ggsave(
  "figures/scientific_map.svg",
  plot = map,
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 300,
  device = svglite::svglite
)

# Save review version (PNG)
ggsave(
  "figures/scientific_map.png",
  plot = map,
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 300
)

# ============================================================================
# 5. METADATA
# ============================================================================

# Save complete metadata
metadata <- list(
  title = "Heat Vulnerability Map - Johannesburg",
  created = Sys.time(),
  author = "Researcher Name",
  data_sources = list(
    boundaries = "Statistics SA (2021)",
    vulnerability = "GCRO QoL Survey 2021"
  ),
  projection = "UTM Zone 35S (EPSG:32735)",
  sample_size = 58616,
  spatial_units = 258,
  spatial_autocorrelation = list(
    morans_i = moran$estimate[1],
    p_value = moran$p.value
  ),
  software = list(
    R = R.version.string,
    sf = as.character(packageVersion("sf")),
    ggplot2 = as.character(packageVersion("ggplot2"))
  )
)

jsonlite::write_json(
  metadata,
  "figures/scientific_map_metadata.json",
  pretty = TRUE
)

# Save session info
writeLines(
  capture.output(sessionInfo()),
  "figures/session_info.txt"
)

cat("\n✓ Map created successfully!\n")
cat("  SVG: figures/scientific_map.svg\n")
cat("  PNG: figures/scientific_map.png\n")
cat("  Metadata: figures/scientific_map_metadata.json\n")
```

---

## Resources

### Projection Information
- Spatial Reference: https://spatialreference.org/
- EPSG.io: https://epsg.io/
- PROJ documentation: https://proj.org/

### Spatial Analysis
- Geocomputation with R: https://r.geocompx.org/
- Spatial Data Science: https://r-spatial.org/book/
- Applied Spatial Data Analysis with R: Bivand et al. (2013)

### Cartographic Guidelines
- USGS Map Projections: https://pubs.usgs.gov/pp/1453/report.pdf
- ICA Commission on Map Projections: https://ica-proj.org/
- Slocum et al. (2009): Thematic Cartography and Geovisualization

### Software Documentation
- sf package: https://r-spatial.github.io/sf/
- terra package: https://rspatial.org/terra/
- spdep package: https://r-spatial.github.io/spdep/

---

**Last Updated**: 2025-01-30
**Version**: 1.0
**Author**: Claude Code Expert System
