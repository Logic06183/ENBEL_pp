# R Cartography Quick Reference Card

## Essential Commands Cheat Sheet

---

## 1. Load and Transform Data

```r
library(sf)
library(ggplot2)
library(ggspatial)

# Load spatial data
districts <- st_read("districts.shp")
points <- st_read("points.gpkg")

# Check CRS
st_crs(districts)

# Transform to common projection
districts <- st_transform(districts, crs = 32735)  # UTM 35S
points <- st_transform(points, crs = 32735)

# Validate geometries
districts <- st_make_valid(districts)
```

---

## 2. Basic Map Template

```r
# Minimal working example
map <- ggplot() +
  geom_sf(data = districts, aes(fill = variable)) +
  scale_fill_viridis_c() +
  theme_minimal()

# Save
ggsave("map.svg", width = 7, height = 5, dpi = 300)
```

---

## 3. Complete Scientific Map

```r
map <- ggplot() +
  # Data layers
  geom_sf(data = districts, aes(fill = vulnerability),
          color = "gray40", linewidth = 0.3) +
  geom_sf(data = points, aes(size = sample_size),
          color = "red", alpha = 0.7) +

  # Color scales
  scale_fill_viridis_c(
    option = "rocket",
    name = "Vulnerability Index\n(0-100)",
    guide = guide_colorbar(barwidth = 10, barheight = 0.5)
  ) +

  scale_size_continuous(range = c(1, 5), name = "Sample Size") +

  # Cartographic elements
  annotation_scale(location = "br", width_hint = 0.25) +
  annotation_north_arrow(location = "tl", which_north = "true") +

  # Theme
  theme_minimal(base_size = 11, base_family = "Arial") +
  theme(
    legend.position = "bottom",
    panel.grid = element_line(color = "gray90"),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  # Labels
  labs(
    title = "Study Region Map",
    subtitle = "District-level vulnerability analysis",
    caption = "Data: Source (Year) | Projection: UTM 35S | n = X"
  )

# Export
ggsave("scientific_map.svg", plot = map,
       width = 7, height = 5.5, dpi = 300, device = svglite::svglite)
```

---

## 4. Common Projections

```r
# Local (Johannesburg)
st_transform(data, crs = 32735)  # UTM Zone 35S

# Regional (South Africa)
st_transform(data, crs = "+proj=aea +lat_1=-24 +lat_2=-33")  # Albers

# Global
st_transform(data, crs = "+proj=robin")  # Robinson
st_transform(data, crs = "+proj=moll")   # Mollweide

# Web mapping only (NOT for analysis)
st_transform(data, crs = 3857)  # Web Mercator
```

---

## 5. Color Palettes

```r
# Sequential (low → high)
scale_fill_viridis_c(option = "viridis")  # Blue-green-yellow
scale_fill_viridis_c(option = "magma")    # Purple-yellow
scale_fill_viridis_c(option = "rocket")   # Black-red-white
scale_fill_viridis_c(option = "mako")     # Blue shades

# Diverging (negative ← 0 → positive)
scale_fill_distiller(palette = "RdBu", direction = -1)
scale_fill_distiller(palette = "PuOr", direction = -1)

# Categorical
scale_fill_brewer(palette = "Set2")   # 8 colors, colorblind-safe
scale_fill_brewer(palette = "Dark2")  # 8 colors, darker
```

---

## 6. Export Formats

```r
# Journal (SVG, 300 DPI)
ggsave("figure.svg", width = 7, height = 5, dpi = 300,
       device = svglite::svglite)

# Presentation (PNG, 150 DPI)
ggsave("slide.png", width = 10, height = 5.625, dpi = 150)

# Social media (square)
ggsave("instagram.png", width = 1080, height = 1080, units = "px", dpi = 96)

# Multi-format export
formats <- c("svg", "png", "pdf")
for (fmt in formats) {
  ggsave(paste0("figure.", fmt), plot = map,
         width = 7, height = 5, dpi = 300)
}
```

---

## 7. Faceted Maps (Temporal)

```r
# Prepare data
districts_temporal <- districts %>%
  left_join(temporal_data) %>%
  group_by(district_id, year)

# Create facets
ggplot() +
  geom_sf(data = districts_temporal, aes(fill = value)) +
  scale_fill_viridis_c() +
  facet_wrap(~ year, ncol = 3) +
  theme_minimal() +
  theme(panel.grid = element_blank())
```

---

## 8. Inset Maps

```r
library(cowplot)

# Main map
main_map <- ggplot() + geom_sf(data = districts)

# Inset (regional context)
country <- st_read("country.shp")
inset_map <- ggplot() +
  geom_sf(data = country, fill = "gray90") +
  geom_sf(data = st_bbox(districts) %>% st_as_sfc(),
          fill = NA, color = "red", linewidth = 2) +
  theme_void()

# Combine
ggdraw() +
  draw_plot(main_map) +
  draw_plot(inset_map, x = 0.05, y = 0.65, width = 0.25, height = 0.25)
```

---

## 9. Quality Checks

```r
# Validate geometries
if (!all(st_is_valid(districts))) {
  districts <- st_make_valid(districts)
}

# Check CRS match
st_crs(layer1) == st_crs(layer2)

# Calculate areas
districts$area_km2 <- as.numeric(st_area(districts)) / 1e6

# Test spatial join
n_matched <- nrow(st_join(points, districts))
cat("Matched points:", n_matched, "/", nrow(points), "\n")
```

---

## 10. Spatial Autocorrelation

```r
library(spdep)

# Create spatial weights
neighbors <- poly2nb(districts, queen = TRUE)
weights <- nb2listw(neighbors, style = "W")

# Moran's I test
moran_test <- moran.test(districts$value, weights)

# Report
cat("Moran's I:", round(moran_test$estimate[1], 3), "\n")
cat("p-value:", format.pval(moran_test$p.value), "\n")
```

---

## 11. Context-Specific Styling

### Journal Style

```r
theme_journal <- function(base_size = 10) {
  theme_minimal(base_size = base_size, base_family = "Arial") +
    theme(
      legend.position = "bottom",
      plot.caption = element_text(hjust = 0, size = 8, color = "gray40")
    )
}
```

### Presentation Style

```r
theme_presentation <- function(base_size = 18) {
  theme_void(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = 26, hjust = 0.5),
      legend.position = "right"
    )
}
```

### Media Style

```r
theme_media <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      panel.grid = element_blank()
    )
}
```

---

## 12. Common Errors and Fixes

### CRS Mismatch

```r
# ERROR: layers don't align
# FIX: Transform all to common CRS
target_crs <- 32735
layer1 <- st_transform(layer1, target_crs)
layer2 <- st_transform(layer2, target_crs)
```

### Invalid Geometries

```r
# ERROR: "Self-intersection"
# FIX: Make valid
districts <- st_make_valid(districts)
```

### Overlapping Labels

```r
# ERROR: Text overlaps
# FIX: Use ggrepel
library(ggrepel)
geom_text_repel(aes(label = name), size = 3, max.overlaps = 20)
```

### Large SVG Files

```r
# ERROR: SVG file is 50+ MB
# FIX 1: Simplify geometries
library(rmapshaper)
districts <- ms_simplify(districts, keep = 0.05)

# FIX 2: Rasterize complex layers
library(ggrastr)
rasterize(geom_point(...), dpi = 300)
```

---

## 13. Package Installation

```r
# Essential packages
install.packages(c(
  "sf",           # Spatial data
  "ggplot2",      # Plotting
  "ggspatial",    # Scale bars, north arrows
  "viridis",      # Colors
  "svglite",      # SVG export
  "dplyr",        # Data manipulation
  "cowplot"       # Combine plots
))

# Advanced packages
install.packages(c(
  "tmap",         # Thematic maps
  "spdep",        # Spatial autocorrelation
  "rmapshaper",   # Simplify geometries
  "ggrepel",      # Non-overlapping labels
  "scico"         # Scientific colors
))
```

---

## 14. Journal Size Guidelines

```r
# Nature
width = 89,  height = 80,  units = "mm"  # Single column
width = 183, height = 120, units = "mm"  # Double column

# Science
width = 9,  height = 7,  units = "cm"   # Single
width = 18, height = 12, units = "cm"   # Double

# PLOS / BMC
width = 3.5, height = 3,   units = "in"  # Single
width = 7,   height = 5.5, units = "in"  # Double

# The Lancet
width = 87,  height = 120, units = "mm"  # Single
width = 180, height = 240, units = "mm"  # Double
```

---

## 15. Complete Workflow

```r
#!/usr/bin/env Rscript
# complete_map_workflow.R

library(sf)
library(ggplot2)
library(ggspatial)
library(viridis)
library(svglite)

# 1. LOAD DATA
districts <- st_read("data/districts.shp")
points <- st_read("data/points.gpkg")
data <- read.csv("data/values.csv")

# 2. VALIDATE
districts <- st_make_valid(districts)
cat("✓ Geometries valid\n")

# 3. TRANSFORM
districts <- st_transform(districts, crs = 32735)
points <- st_transform(points, crs = 32735)
cat("✓ CRS transformed to UTM 35S\n")

# 4. JOIN DATA
districts <- districts %>% left_join(data, by = "district_id")
cat("✓ Data joined\n")

# 5. CREATE MAP
map <- ggplot() +
  geom_sf(data = districts, aes(fill = vulnerability)) +
  geom_sf(data = points, color = "red", size = 2) +
  scale_fill_viridis_c(option = "rocket") +
  annotation_scale(location = "br") +
  annotation_north_arrow(location = "tl") +
  theme_minimal() +
  labs(title = "Study Region", caption = "Data: Source 2021")

# 6. EXPORT
ggsave("output/map.svg", plot = map,
       width = 7, height = 5.5, dpi = 300, device = svglite)
ggsave("output/map.png", plot = map,
       width = 7, height = 5.5, dpi = 300)

cat("✓ Maps exported to output/\n")
```

---

## 16. Metadata Template

```r
# Save complete metadata
metadata <- list(
  title = "Map Title",
  created = Sys.time(),
  author = "Your Name",
  data_sources = list(
    boundaries = "Source (Year)",
    variables = "Dataset name and DOI"
  ),
  projection = "UTM Zone 35S (EPSG:32735)",
  sample_size = 1000,
  software = list(
    R = R.version.string,
    sf = as.character(packageVersion("sf")),
    ggplot2 = as.character(packageVersion("ggplot2"))
  )
)

jsonlite::write_json(metadata, "output/metadata.json", pretty = TRUE)
```

---

## 17. Quality Control Checklist

**Before exporting:**

- [ ] CRS appropriate for region
- [ ] All layers use same CRS
- [ ] Geometries are valid
- [ ] Scale bar present
- [ ] North arrow included
- [ ] Legend has units
- [ ] Colors are colorblind-safe
- [ ] Data sources cited
- [ ] Text readable at final size
- [ ] File size reasonable (<5 MB)

---

## 18. Further Help

### Detailed Guides
- **R_CARTOGRAPHY_FUNDAMENTALS.md** - Core workflows
- **SVG_EXPORT_WORKFLOWS.md** - Export options
- **PUBLICATION_MAP_STYLING.md** - Context-specific styles
- **SCIENTIFIC_CARTOGRAPHY_STANDARDS.md** - Rigor and validation

### Online Resources
- sf package: https://r-spatial.github.io/sf/
- ggplot2: https://ggplot2.tidyverse.org/
- Geocomputation with R: https://r.geocompx.org/

---

**Quick Reference Card v1.0**
**Last Updated**: 2025-01-30
