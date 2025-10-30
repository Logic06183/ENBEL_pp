# R Cartography Fundamentals

## Expert-Level R Mapping for Scientific Publications

This skill provides comprehensive guidance for creating publication-quality maps in R using modern geospatial workflows. Suitable for peer-reviewed journals, conference presentations, and media outlets.

---

## Core Principles

### 1. Scientific Rigor
- **Always specify coordinate reference systems (CRS)**
- **Include scale bars, north arrows, and legends**
- **Document data sources in captions**
- **Use appropriate projections for the region**
- **Maintain spatial accuracy and precision**

### 2. Visual Excellence
- **Design for the medium**: Slides need larger text, journals need detail
- **Use colorblind-safe palettes**
- **Minimize chart junk, maximize data-ink ratio**
- **Ensure accessibility (508 compliance)**

### 3. Reproducibility
- **Version control all code**
- **Document package versions**
- **Use relative paths**
- **Set random seeds where applicable**

---

## Essential R Packages

```r
# Geospatial Core
library(sf)           # Simple Features for vector data
library(terra)        # Raster data handling (replaces raster)
library(stars)        # Spatiotemporal arrays

# Mapping Frameworks
library(ggplot2)      # Grammar of graphics
library(tmap)         # Thematic maps (static & interactive)
library(mapsf)        # Cartographic styling

# Data Wrangling
library(dplyr)        # Data manipulation
library(tidyr)        # Data tidying

# Styling & Export
library(cowplot)      # Publication-ready ggplot themes
library(svglite)      # High-quality SVG export
library(ggspatial)    # Scale bars and north arrows for ggplot

# Color Palettes
library(viridis)      # Colorblind-safe sequential palettes
library(RColorBrewer) # Classic palettes
library(scico)        # Scientific color maps
library(cols4all)     # All color palettes in one place
```

---

## Workflow 1: ggplot2 + sf (Most Flexible)

### Advantages
- Full control over every element
- Integrates with ggplot2 ecosystem
- Best for complex multi-panel figures
- Excellent SVG output quality

### Basic Template

```r
library(sf)
library(ggplot2)
library(ggspatial)
library(viridis)

# Load spatial data
districts <- st_read("data/districts.shp")
points <- st_read("data/survey_locations.gpkg")

# Ensure correct projection (example: South African Albers Equal Area)
districts <- st_transform(districts, crs = 32735)  # UTM Zone 35S
points <- st_transform(points, crs = 32735)

# Create map
map <- ggplot() +
  # Base layer (polygons)
  geom_sf(data = districts,
          aes(fill = variable_name),
          color = "gray30",
          linewidth = 0.3) +

  # Point layer
  geom_sf(data = points,
          aes(size = sample_size, color = category),
          alpha = 0.7) +

  # Color scales
  scale_fill_viridis_c(
    option = "rocket",
    name = "Variable Name\n(Units)",
    guide = guide_colorbar(
      barwidth = 12,
      barheight = 0.6,
      title.position = "top"
    )
  ) +

  scale_color_brewer(
    palette = "Set2",
    name = "Category"
  ) +

  scale_size_continuous(
    range = c(1, 6),
    name = "Sample Size"
  ) +

  # Cartographic elements
  annotation_scale(
    location = "br",           # bottom-right
    width_hint = 0.2,
    style = "ticks",
    line_width = 0.5,
    text_cex = 0.7
  ) +

  annotation_north_arrow(
    location = "tl",           # top-left
    which_north = "true",
    pad_x = unit(0.2, "in"),
    pad_y = unit(0.2, "in"),
    style = north_arrow_fancy_orienteering(
      fill = c("gray30", "white"),
      line_col = "gray30"
    )
  ) +

  # Publication theme
  theme_minimal(base_size = 11, base_family = "Arial") +
  theme(
    # Panel
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.2),
    panel.grid.minor = element_blank(),

    # Legend
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    legend.background = element_rect(fill = "white", color = "gray60", linewidth = 0.3),
    legend.margin = margin(t = 5, r = 5, b = 5, l = 5),

    # Plot margins
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10),

    # Axis (usually removed for maps)
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank()
  ) +

  # Labels
  labs(
    title = "Study Region: Johannesburg Metropolitan Area",
    subtitle = "Survey locations and administrative districts (n=1,032)",
    caption = "Data: GCRO 2021 | Projection: UTM Zone 35S (EPSG:32735)"
  )

# Display
print(map)

# Save as SVG (publication quality)
ggsave(
  filename = "output/map_district_survey.svg",
  plot = map,
  width = 10,
  height = 8,
  units = "in",
  dpi = 300,
  device = "svg"
)

# Save as PNG (for quick review)
ggsave(
  filename = "output/map_district_survey.png",
  plot = map,
  width = 10,
  height = 8,
  units = "in",
  dpi = 300
)
```

---

## Workflow 2: tmap (Rapid Thematic Mapping)

### Advantages
- Fast development for choropleth maps
- Built-in cartographic elements
- Easy switch between static and interactive
- Excellent for exploratory analysis

### Basic Template

```r
library(tmap)
library(sf)

# Load data
districts <- st_read("data/districts.shp")
districts <- st_transform(districts, crs = 32735)

# Set tmap mode
tmap_mode("plot")  # Static maps ("view" for interactive)

# Create map
map <- tm_shape(districts) +
  tm_polygons(
    col = "variable_name",
    title = "Variable Name\n(Units)",
    palette = "YlOrRd",
    border.col = "gray30",
    border.alpha = 0.5,
    style = "quantile",        # or "jenks", "equal", "pretty"
    n = 5,
    legend.hist = TRUE
  ) +

  tm_scale_bar(
    position = c("right", "bottom"),
    text.size = 0.6
  ) +

  tm_compass(
    type = "arrow",
    position = c("left", "top"),
    size = 2
  ) +

  tm_credits(
    text = "Data: GCRO 2021 | Projection: UTM 35S",
    position = c("left", "bottom"),
    size = 0.5
  ) +

  tm_layout(
    title = "District-Level Analysis",
    title.size = 1.2,
    title.position = c("center", "top"),
    legend.outside = TRUE,
    legend.outside.position = "right",
    legend.frame = TRUE,
    bg.color = "white",
    frame = TRUE,
    inner.margins = c(0.05, 0.05, 0.05, 0.05)
  )

# Display
print(map)

# Save
tmap_save(
  tm = map,
  filename = "output/map_choropleth.svg",
  width = 10,
  height = 8,
  units = "in",
  dpi = 300
)
```

---

## Workflow 3: Multi-Panel Maps (Faceting)

### For Temporal or Multi-Variable Comparisons

```r
library(ggplot2)
library(sf)
library(dplyr)

# Example: Survey waves across time
districts_temporal <- districts %>%
  st_join(survey_data) %>%
  group_by(district_id, survey_wave) %>%
  summarize(
    mean_value = mean(variable_name, na.rm = TRUE),
    .groups = "drop"
  )

# Faceted map
map_facets <- ggplot() +
  geom_sf(data = districts_temporal,
          aes(fill = mean_value),
          color = "gray50",
          linewidth = 0.2) +

  scale_fill_viridis_c(
    option = "mako",
    name = "Mean Value",
    na.value = "gray90"
  ) +

  facet_wrap(~ survey_wave, ncol = 3) +

  theme_minimal(base_size = 10) +
  theme(
    strip.background = element_rect(fill = "gray95", color = "gray60"),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  labs(
    title = "Temporal Evolution of Variable Across Survey Waves",
    caption = "Data: GCRO 2011-2021 | Each panel represents one survey wave"
  )

ggsave(
  "output/map_temporal_facets.svg",
  plot = map_facets,
  width = 12,
  height = 8,
  units = "in",
  dpi = 300
)
```

---

## Advanced: Inset Maps

### For Context and Study Area Location

```r
library(ggplot2)
library(sf)
library(cowplot)

# Main map (detailed study area)
main_map <- ggplot() +
  geom_sf(data = districts, aes(fill = variable_name)) +
  scale_fill_viridis_c(option = "viridis") +
  theme_minimal() +
  labs(title = "Study Area Detail")

# Inset map (regional context)
country <- st_read("data/south_africa_provinces.shp")
study_bbox <- st_bbox(districts) %>% st_as_sfc()

inset_map <- ggplot() +
  geom_sf(data = country, fill = "gray90", color = "gray50") +
  geom_sf(data = study_bbox, fill = NA, color = "red", linewidth = 1.5) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "lightblue", color = "black"),
    plot.background = element_rect(fill = "white", color = "black", linewidth = 0.5)
  )

# Combine with cowplot
combined_map <- ggdraw() +
  draw_plot(main_map) +
  draw_plot(inset_map,
            x = 0.05, y = 0.65,   # Position
            width = 0.25, height = 0.25)  # Size

# Save
save_plot(
  filename = "output/map_with_inset.svg",
  plot = combined_map,
  base_width = 10,
  base_height = 8
)
```

---

## Color Palette Selection

### Scientific Color Maps (Perceptually Uniform)

```r
# For continuous variables
library(scico)

# Temperature (blue-red)
scale_fill_scico(palette = "vik", direction = -1)

# Elevation (green-brown)
scale_fill_scico(palette = "bamako")

# Density (white-purple)
scale_fill_scico(palette = "lajolla")

# Diverging (blue-white-red)
scale_fill_scico(palette = "roma", midpoint = 0)
```

### Viridis (Colorblind-Safe)

```r
library(viridis)

# Sequential
scale_fill_viridis_c(option = "viridis")  # Blue-green-yellow
scale_fill_viridis_c(option = "magma")    # Black-purple-yellow
scale_fill_viridis_c(option = "inferno")  # Black-red-yellow
scale_fill_viridis_c(option = "plasma")   # Purple-pink-yellow
scale_fill_viridis_c(option = "rocket")   # Black-red-white
scale_fill_viridis_c(option = "mako")     # Dark blue-light blue
scale_fill_viridis_c(option = "turbo")    # Rainbow (use sparingly)
```

### ColorBrewer (Categorical & Sequential)

```r
library(RColorBrewer)

# Display all palettes
display.brewer.all(colorblindFriendly = TRUE)

# Sequential (single hue)
scale_fill_brewer(palette = "Blues")
scale_fill_brewer(palette = "Greens")

# Diverging
scale_fill_brewer(palette = "RdYlBu")
scale_fill_brewer(palette = "PuOr")

# Qualitative (categorical)
scale_fill_brewer(palette = "Set2")     # 8 colors, colorblind-safe
scale_fill_brewer(palette = "Dark2")    # 8 colors, darker
```

---

## Projection Selection Guide

### For South Africa / Johannesburg

```r
# UTM Zone 35S (preserves area and distance in region)
st_transform(data, crs = 32735)

# South African Albers Equal Area (for country-wide maps)
st_transform(data, crs = "+proj=aea +lat_1=-24 +lat_2=-33 +lat_0=-28.5 +lon_0=25 +datum=WGS84")

# WGS84 Geographic (lat/lon, for web maps only)
st_transform(data, crs = 4326)
```

### For Global Maps

```r
# Robinson (good compromise for world maps)
st_transform(data, crs = "+proj=robin")

# Mollweide (equal-area)
st_transform(data, crs = "+proj=moll")

# Natural Earth (aesthetic)
st_transform(data, crs = "+proj=natearth")
```

### Check Current CRS

```r
st_crs(data)
```

---

## Quality Control Checklist

Before exporting final maps:

- [ ] **CRS specified and appropriate for region**
- [ ] **Scale bar present with correct units**
- [ ] **North arrow included (unless global map)**
- [ ] **Legend has clear title and units**
- [ ] **Data source cited in caption**
- [ ] **Projection documented**
- [ ] **Colors are colorblind-safe**
- [ ] **Text is readable at intended size**
- [ ] **No overlapping labels**
- [ ] **Grid lines appropriate (or removed)**
- [ ] **File saved as SVG for publications**
- [ ] **Code is reproducible and commented**

---

## SVG Export Best Practices

### For Publications (Journals)

```r
ggsave(
  filename = "figure1_map.svg",
  plot = map,
  width = 7,        # Single column: 3.5-4in, Double column: 7in
  height = 6,       # Maintain aspect ratio
  units = "in",
  dpi = 300,        # High resolution
  device = svglite::svglite,
  scaling = 1.0
)
```

### For Presentations (Slides)

```r
ggsave(
  filename = "slide_map.svg",
  plot = map,
  width = 10,       # Widescreen 16:9 ratio
  height = 5.625,
  units = "in",
  dpi = 150,        # Lower DPI for slides
  device = "svg"
)
```

### For Posters

```r
ggsave(
  filename = "poster_map.svg",
  plot = map,
  width = 12,
  height = 9,
  units = "in",
  dpi = 300,
  device = "svg"
)
```

---

## Common Issues and Solutions

### Issue: Overlapping Labels

```r
# Solution 1: Use ggrepel
library(ggrepel)

geom_text_repel(
  aes(label = district_name),
  size = 3,
  box.padding = 0.5,
  point.padding = 0.3,
  segment.color = "gray50",
  max.overlaps = 20
)

# Solution 2: Reduce label count
districts_labeled <- districts %>%
  filter(population > 50000)  # Only label large areas
```

### Issue: Slow Rendering with Large Shapefiles

```r
# Solution: Simplify geometry
library(rmapshaper)

districts_simplified <- ms_simplify(
  districts,
  keep = 0.05,        # Keep 5% of vertices
  keep_shapes = TRUE  # Preserve small polygons
)
```

### Issue: Misaligned Layers

```r
# Solution: Ensure all layers have same CRS
target_crs <- 32735

layer1 <- st_transform(layer1, crs = target_crs)
layer2 <- st_transform(layer2, crs = target_crs)
layer3 <- st_transform(layer3, crs = target_crs)
```

---

## Example: Complete Publication-Ready Map

```r
#!/usr/bin/env Rscript
# create_publication_map.R
# Author: Your Name
# Date: 2025-01-30
# Purpose: Generate Figure 1 for manuscript

library(sf)
library(ggplot2)
library(ggspatial)
library(viridis)
library(dplyr)

# Set working directory
setwd("/path/to/project")

# Load data
districts <- st_read("data/johannesburg_districts.shp") %>%
  st_transform(crs = 32735)

survey_points <- st_read("data/survey_locations.gpkg") %>%
  st_transform(crs = 32735) %>%
  filter(year == 2021)

# Calculate summary statistics
districts <- districts %>%
  left_join(
    survey_points %>%
      st_drop_geometry() %>%
      group_by(district_id) %>%
      summarize(
        n_surveys = n(),
        mean_vulnerability = mean(heat_vulnerability, na.rm = TRUE)
      ),
    by = "district_id"
  )

# Create map
map <- ggplot() +
  # District boundaries with vulnerability
  geom_sf(
    data = districts,
    aes(fill = mean_vulnerability),
    color = "white",
    linewidth = 0.4
  ) +

  # Survey points
  geom_sf(
    data = survey_points,
    aes(size = sample_size),
    color = "#E63946",
    alpha = 0.6,
    shape = 16
  ) +

  # Color scale
  scale_fill_viridis_c(
    option = "rocket",
    name = "Heat Vulnerability\nIndex (0-100)",
    na.value = "gray90",
    guide = guide_colorbar(
      barwidth = 10,
      barheight = 0.5,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +

  scale_size_continuous(
    range = c(1, 5),
    name = "Sample Size",
    guide = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      override.aes = list(color = "#E63946", alpha = 0.8)
    )
  ) +

  # Cartographic elements
  annotation_scale(
    location = "br",
    width_hint = 0.25,
    style = "ticks",
    line_width = 0.8,
    text_cex = 0.8
  ) +

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
  ) +

  # Theme
  theme_minimal(base_size = 12, base_family = "Arial") +
  theme(
    panel.background = element_rect(fill = "aliceblue", color = NA),
    panel.grid.major = element_line(color = "white", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    legend.background = element_rect(fill = "white", color = "gray70", linewidth = 0.3),
    legend.margin = margin(t = 10, r = 10, b = 10, l = 10),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
    plot.caption = element_text(size = 8, hjust = 0, color = "gray50"),
    plot.margin = margin(t = 15, r = 15, b = 15, l = 15),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank()
  ) +

  # Labels
  labs(
    title = "Study Region: Johannesburg Metropolitan Area",
    subtitle = "Heat vulnerability and survey locations (2021 GCRO Quality of Life Survey)",
    caption = paste0(
      "Data: Greater Capital Region Observatory (GCRO) | ",
      "Projection: UTM Zone 35S (EPSG:32735) | ",
      "n = ", nrow(survey_points), " survey locations across ",
      nrow(districts), " districts"
    )
  )

# Display
print(map)

# Save publication version (SVG)
ggsave(
  filename = "figures/figure1_study_region_map.svg",
  plot = map,
  width = 7,
  height = 6,
  units = "in",
  dpi = 300,
  device = svglite::svglite
)

# Save presentation version (PNG)
ggsave(
  filename = "figures/figure1_study_region_map.png",
  plot = map,
  width = 10,
  height = 8,
  units = "in",
  dpi = 300
)

cat("Map created successfully!\n")
cat("  SVG: figures/figure1_study_region_map.svg\n")
cat("  PNG: figures/figure1_study_region_map.png\n")
```

---

## Additional Resources

- **sf documentation**: https://r-spatial.github.io/sf/
- **ggplot2 spatial**: https://ggplot2.tidyverse.org/reference/ggsf.html
- **tmap book**: https://r-tmap.github.io/tmap-book/
- **Geocomputation with R**: https://r.geocompx.org/
- **Cartography best practices**: Slocum et al. (2009) "Thematic Cartography and Geovisualization"

---

**Last Updated**: 2025-01-30
**Version**: 1.0
**Author**: Claude Code Expert System
