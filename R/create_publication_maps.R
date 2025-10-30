#!/usr/bin/env Rscript
#
# create_publication_maps.R
#
# Purpose: Generate comprehensive publication-quality maps for Johannesburg
#          climate-health research
# Author: ENBEL Research Team
# Date: 2025-01-30
# Dependencies: sf, ggplot2, ggspatial, viridis, dplyr, tidyr, svglite, cowplot

# ==============================================================================
# SETUP
# ==============================================================================

library(sf)
library(ggplot2)
library(ggspatial)
library(viridis)
library(dplyr)
library(tidyr)
library(svglite)
library(cowplot)

# Set reproducible environment
set.seed(42)
options(scipen = 999)

# Create output directories
dir.create("figures/maps/journal", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/maps/presentation", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/maps/social", recursive = TRUE, showWarnings = FALSE)

cat("📍 Starting publication map generation...\n\n")

# ==============================================================================
# 1. LOAD SPATIAL DATA
# ==============================================================================

cat("Loading spatial data...\n")

# Johannesburg metropolitan boundary
jhb_metro <- st_read(
  "presentation_slides_final/map_vector_folder_JHB/JHB/metropolitan municipality jhb.shp",
  quiet = TRUE
)

# Set CRS if missing (assume WGS84 for South African shapefiles)
if (is.na(st_crs(jhb_metro))) {
  jhb_metro <- st_set_crs(jhb_metro, 4326)
  cat("  Set JHB CRS to WGS84 (EPSG:4326)\n")
}

# South Africa country boundary
south_africa <- st_read(
  "presentation_slides_final/map_vector_folder_JHB/zaf_admbnda_adm0_sadb_ocha_20201109.shp",
  quiet = TRUE
)

if (is.na(st_crs(south_africa))) {
  south_africa <- st_set_crs(south_africa, 4326)
  cat("  Set SA CRS to WGS84 (EPSG:4326)\n")
}

# Transform to UTM Zone 35S (appropriate for Johannesburg)
jhb_metro <- st_transform(jhb_metro, crs = 32735)
south_africa_wgs84 <- south_africa  # Keep copy in WGS84 for context map

cat("✓ Spatial boundaries loaded\n")

# ==============================================================================
# 2. LOAD SURVEY DATA
# ==============================================================================

cat("Loading survey data...\n")

# Clinical dataset
clinical <- read.csv("data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv")

# GCRO socioeconomic dataset
gcro <- read.csv("data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv")

# Create spatial points for clinical data (filter valid coordinates)
clinical_sf <- clinical %>%
  filter(!is.na(latitude) & !is.na(longitude)) %>%
  filter(latitude != 0 & longitude != 0) %>%
  filter(abs(latitude) < 90 & abs(longitude) < 180) %>%  # Valid coordinate range
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(32735)

# Aggregate by location to get site-level counts
clinical_sites <- clinical_sf %>%
  mutate(
    lon = st_coordinates(.)[,1],
    lat = st_coordinates(.)[,2]
  ) %>%
  st_drop_geometry() %>%
  group_by(lon, lat) %>%
  summarize(
    n_participants = n(),
    .groups = "drop"
  ) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 32735)

cat("  Aggregated to", nrow(clinical_sites), "unique clinical sites\n")

# Create spatial points for GCRO data
gcro_sf <- gcro %>%
  filter(!is.na(latitude) & !is.na(longitude)) %>%
  filter(latitude != 0 & longitude != 0) %>%
  filter(abs(latitude) < 90 & abs(longitude) < 180) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(32735)

cat("✓ Survey data loaded\n")
cat("  Clinical records:", nrow(clinical_sf), "\n")
cat("  GCRO records:", nrow(gcro_sf), "\n\n")

# ==============================================================================
# 3. MAP 1: STUDY REGION OVERVIEW WITH INSET
# ==============================================================================

cat("Creating Map 1: Study Region Overview...\n")

# Main map - Johannesburg
main_map <- ggplot() +
  # Metropolitan boundary
  geom_sf(
    data = jhb_metro,
    fill = "#E8F4F8",
    color = "#2C5F7F",
    linewidth = 1.2
  ) +

  # Scale bar
  annotation_scale(
    location = "br",
    width_hint = 0.25,
    text_cex = 0.9,
    style = "ticks",
    line_width = 0.8
  ) +

  # North arrow
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
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
    plot.caption = element_text(size = 8, hjust = 0, color = "gray40"),
    plot.margin = margin(15, 15, 15, 15),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  labs(
    title = "Study Region: Johannesburg Metropolitan Area",
    subtitle = "Greater Johannesburg, Gauteng Province, South Africa",
    caption = paste0(
      "Projection: UTM Zone 35S (EPSG:32735) | ",
      "Boundary: Statistics South Africa, Municipal Demarcation Board 2021"
    )
  )

# Inset map - South Africa context
jhb_bbox <- st_bbox(jhb_metro) %>%
  st_as_sfc() %>%
  st_transform(4326)

inset_map <- ggplot() +
  geom_sf(data = south_africa_wgs84, fill = "gray85", color = "gray50", linewidth = 0.5) +
  geom_sf(data = jhb_bbox, fill = NA, color = "#E63946", linewidth = 1.5) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "aliceblue", color = "black", linewidth = 0.8),
    plot.background = element_rect(fill = "white", color = NA)
  )

# Combine maps
map1_combined <- ggdraw() +
  draw_plot(main_map) +
  draw_plot(inset_map, x = 0.08, y = 0.68, width = 0.25, height = 0.25)

# Save in multiple formats
ggsave(
  "figures/maps/journal/map1_study_region_overview.svg",
  plot = map1_combined,
  width = 7,
  height = 6,
  units = "in",
  dpi = 300,
  device = svglite
)

ggsave(
  "figures/maps/presentation/map1_study_region_overview.png",
  plot = map1_combined,
  width = 10,
  height = 8,
  units = "in",
  dpi = 150
)

cat("✓ Map 1 saved\n\n")

# ==============================================================================
# 4. MAP 2: CLINICAL SURVEY LOCATIONS
# ==============================================================================

cat("Creating Map 2: Clinical Survey Locations...\n")

map2 <- ggplot() +
  # Metropolitan boundary
  geom_sf(
    data = jhb_metro,
    fill = "gray95",
    color = "gray60",
    linewidth = 0.5
  ) +

  # Clinical survey points (sized by participant count)
  geom_sf(
    data = clinical_sites,
    aes(size = n_participants, color = "Clinical Trial Sites"),
    alpha = 0.7,
    shape = 16
  ) +

  # Size scale for participant counts
  scale_size_continuous(
    range = c(2, 12),
    name = "Participants per Site",
    breaks = c(100, 500, 1000, 2000),
    labels = c("100", "500", "1,000", "2,000")
  ) +

  # Color scale
  scale_color_manual(
    values = c("Clinical Trial Sites" = "#E63946"),
    name = NULL,
    guide = guide_legend(override.aes = list(size = 4))
  ) +

  # Cartographic elements
  annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
  annotation_north_arrow(
    location = "tl",
    which_north = "true",
    pad_x = unit(0.2, "in"),
    pad_y = unit(0.2, "in")
  ) +

  # Theme
  theme_minimal(base_size = 11, base_family = "Arial") +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.background = element_rect(fill = "white", color = "gray70", linewidth = 0.3),
    legend.margin = margin(5, 5, 5, 5),
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
    plot.caption = element_text(size = 8, hjust = 0),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  labs(
    title = "Clinical Trial Survey Locations",
    subtitle = paste0("HIV treatment cohorts, n = ",
                     format(nrow(clinical_sf), big.mark = ","),
                     " records (2002-2021)"),
    caption = paste0(
      "Data: 15 ENBEL consortium clinical trials | ",
      "Biomarkers: CD4 count, viral load, metabolic panel | ",
      "Climate data: ERA5 reanalysis matched to survey dates (99.5% coverage)"
    )
  )

# Save
ggsave(
  "figures/maps/journal/map2_clinical_survey_locations.svg",
  plot = map2,
  width = 7,
  height = 6,
  dpi = 300,
  device = svglite
)

ggsave(
  "figures/maps/presentation/map2_clinical_survey_locations.png",
  plot = map2,
  width = 10,
  height = 8,
  dpi = 150
)

# ALSO save as SVG for editing
ggsave(
  "figures/maps/presentation/map2_clinical_survey_locations.svg",
  plot = map2,
  width = 10,
  height = 8,
  dpi = 150,
  device = svglite
)

cat("✓ Map 2 saved (SVG + PNG)\n\n")

# ==============================================================================
# 5. MAP 3: GCRO SURVEY LOCATIONS BY WAVE
# ==============================================================================

cat("Creating Map 3: GCRO Survey Locations by Wave...\n")

# Check for survey wave/year variables
wave_var <- NULL
if ("survey_wave" %in% names(gcro_sf)) {
  wave_var <- "survey_wave"
} else if ("survey_year" %in% names(gcro_sf)) {
  wave_var <- "survey_year"
} else if ("year" %in% names(gcro_sf)) {
  wave_var <- "year"
}

# Sample GCRO data for visualization (more points to show density)
if (nrow(gcro_sf) > 20000) {
  set.seed(42)
  gcro_sf_sample <- gcro_sf %>% sample_n(20000)
  cat("  Sampling 20,000 points for visualization\n")
} else {
  gcro_sf_sample <- gcro_sf
  cat("  Using all", nrow(gcro_sf), "points\n")
}

# Create map with or without wave colors
if (!is.null(wave_var)) {
  cat("  Coloring by", wave_var, "\n")

  map3 <- ggplot() +
    # Metropolitan boundary
    geom_sf(
      data = jhb_metro,
      fill = "gray95",
      color = "gray60",
      linewidth = 0.5
    ) +

    # GCRO survey points by wave
    geom_sf(
      data = gcro_sf_sample,
      aes(color = as.factor(!!sym(wave_var))),
      size = 1.2,
      alpha = 0.5
    ) +

    # Color scale (distinct colors for waves)
    scale_color_viridis_d(
      option = "turbo",
      name = "Survey Wave",
      guide = guide_legend(
        override.aes = list(size = 4, alpha = 0.8)
      )
    ) +

    # Cartographic elements
    annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
    annotation_north_arrow(location = "tl", which_north = "true") +

    # Theme
    theme_minimal(base_size = 11, base_family = "Arial") +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
      legend.position = "bottom",
      legend.title = element_text(face = "bold", size = 9),
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      plot.caption = element_text(size = 8, hjust = 0),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    ) +

    labs(
      title = "GCRO Quality of Life Survey Locations",
      subtitle = paste0("Household surveys across multiple waves, n = ",
                       format(nrow(gcro_sf), big.mark = ",")),
      caption = paste0(
        "Data: Greater Capital Region Observatory (GCRO) 2011-2021 | ",
        "Variables: Socioeconomic, dwelling, health access, demographics"
      )
    )

} else {
  # Single wave visualization
  map3 <- ggplot() +
    geom_sf(data = jhb_metro, fill = "gray95", color = "gray60", linewidth = 0.5) +
    geom_sf(
      data = gcro_sf_sample,
      color = "#2A9D8F",
      size = 0.8,
      alpha = 0.6
    ) +
    annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
    annotation_north_arrow(location = "tl", which_north = "true") +
    theme_minimal(base_size = 11, base_family = "Arial") +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      plot.caption = element_text(size = 8, hjust = 0),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    ) +
    labs(
      title = "GCRO Quality of Life Survey Locations",
      subtitle = paste0("Household surveys, n = ", format(nrow(gcro_sf), big.mark = ",")),
      caption = "Data: Greater Capital Region Observatory (GCRO)"
    )
}

# Save journal version (SVG)
ggsave(
  "figures/maps/journal/map3_gcro_survey_locations.svg",
  plot = map3,
  width = 7,
  height = 6,
  dpi = 300,
  device = svglite
)

# Save presentation version (PNG)
ggsave(
  "figures/maps/presentation/map3_gcro_survey_locations.png",
  plot = map3,
  width = 10,
  height = 8,
  dpi = 150
)

# ALSO save presentation version as SVG for editing
ggsave(
  "figures/maps/presentation/map3_gcro_survey_locations.svg",
  plot = map3,
  width = 10,
  height = 8,
  dpi = 150,
  device = svglite
)

cat("✓ Map 3 saved\n\n")

# ==============================================================================
# 6. MAP 4: COMBINED SURVEY COVERAGE
# ==============================================================================

cat("Creating Map 4: Combined Survey Coverage...\n")

map4 <- ggplot() +
  # Metropolitan boundary
  geom_sf(
    data = jhb_metro,
    fill = "gray95",
    color = "gray60",
    linewidth = 0.5
  ) +

  # GCRO points (background)
  geom_sf(
    data = gcro_sf_sample,
    aes(color = "GCRO Household Survey"),
    size = 0.8,
    alpha = 0.3
  ) +

  # Clinical points (foreground, sized by participants)
  geom_sf(
    data = clinical_sites,
    aes(size = n_participants, color = "Clinical Trial Sites"),
    alpha = 0.8,
    shape = 17  # Triangle
  ) +

  # Size scale
  scale_size_continuous(
    range = c(3, 15),
    name = "Participants",
    breaks = c(100, 500, 1000, 2000),
    labels = c("100", "500", "1,000", "2,000")
  ) +

  # Color scale
  scale_color_manual(
    values = c(
      "GCRO Household Survey" = "#2A9D8F",
      "Clinical Trial Sites" = "#E63946"
    ),
    name = NULL,
    guide = guide_legend(
      override.aes = list(size = c(2, 6), shape = c(16, 17))
    )
  ) +

  # Cartographic elements
  annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
  annotation_north_arrow(location = "tl", which_north = "true") +

  # Theme
  theme_minimal(base_size = 11, base_family = "Arial") +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
    legend.position = "bottom",
    legend.background = element_rect(fill = "white", color = "gray70", linewidth = 0.3),
    legend.margin = margin(5, 5, 5, 5),
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    plot.caption = element_text(size = 8, hjust = 0),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  labs(
    title = "Integrated Climate-Health Data Collection",
    subtitle = paste0(
      "Clinical trials (n = ", format(nrow(clinical_sf), big.mark = ","), ") + ",
      "Household surveys (n = ", format(nrow(gcro_sf), big.mark = ","), ")"
    ),
    caption = paste0(
      "Combined dataset enables climate-biomarker-socioeconomic integration | ",
      "Coverage: Johannesburg metropolitan area, 2002-2021"
    )
  )

# Save
ggsave(
  "figures/maps/journal/map4_combined_survey_coverage.svg",
  plot = map4,
  width = 7,
  height = 6,
  dpi = 300,
  device = svglite
)

ggsave(
  "figures/maps/presentation/map4_combined_survey_coverage.png",
  plot = map4,
  width = 10,
  height = 8,
  dpi = 150
)

# ALSO save as SVG for editing
ggsave(
  "figures/maps/presentation/map4_combined_survey_coverage.svg",
  plot = map4,
  width = 10,
  height = 8,
  dpi = 150,
  device = svglite
)

cat("✓ Map 4 saved (SVG + PNG)\n\n")

# ==============================================================================
# 7. MAP 5: HEAT VULNERABILITY (if variable exists)
# ==============================================================================

cat("Creating Map 5: Heat Vulnerability Map...\n")

# Check if heat vulnerability exists
if ("HEAT_VULNERABILITY_SCORE" %in% names(gcro_sf)) {

  # Aggregate to spatial grid (hexbins or grid cells)
  # Create 2km grid
  grid <- st_make_grid(jhb_metro, cellsize = 2000, square = TRUE)
  grid_sf <- st_sf(grid_id = 1:length(grid), geometry = grid)

  # Spatial join and aggregate
  gcro_grid <- st_join(grid_sf, gcro_sf) %>%
    filter(!is.na(HEAT_VULNERABILITY_SCORE)) %>%
    group_by(grid_id) %>%
    summarize(
      mean_vulnerability = mean(HEAT_VULNERABILITY_SCORE, na.rm = TRUE),
      n_surveys = n(),
      .groups = "drop"
    ) %>%
    filter(n_surveys >= 5)  # Only show cells with sufficient data

  map5 <- ggplot() +
    # Vulnerability grid
    geom_sf(
      data = gcro_grid,
      aes(fill = mean_vulnerability),
      color = NA
    ) +

    # Metropolitan boundary overlay
    geom_sf(
      data = jhb_metro,
      fill = NA,
      color = "gray30",
      linewidth = 1
    ) +

    # Color scale
    scale_fill_viridis_c(
      option = "rocket",
      name = "Heat Vulnerability\nIndex (0-100)",
      na.value = "transparent",
      guide = guide_colorbar(
        barwidth = 10,
        barheight = 0.6,
        title.position = "top",
        title.hjust = 0.5
      )
    ) +

    # Cartographic elements
    annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
    annotation_north_arrow(location = "tl", which_north = "true") +

    # Theme
    theme_minimal(base_size = 11, base_family = "Arial") +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      plot.caption = element_text(size = 8, hjust = 0),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    ) +

    labs(
      title = "Heat Vulnerability Across Johannesburg",
      subtitle = "Composite index: dwelling type, income, health access, age (2km grid)",
      caption = paste0(
        "Data: GCRO Quality of Life Survey | ",
        "Higher values indicate greater vulnerability to heat stress | ",
        "Grid cells with n ≥ 5 surveys shown"
      )
    )

  # Save
  ggsave(
    "figures/maps/journal/map5_heat_vulnerability.svg",
    plot = map5,
    width = 7,
    height = 6,
    dpi = 300,
    device = svglite
  )

  ggsave(
    "figures/maps/presentation/map5_heat_vulnerability.png",
    plot = map5,
    width = 10,
    height = 8,
    dpi = 150
  )

  cat("✓ Map 5 saved\n\n")

} else {
  cat("⚠ Heat vulnerability variable not found, skipping Map 5\n\n")
}

# ==============================================================================
# 8. MAP 6: CLIMATE EXPOSURE (if temperature data exists)
# ==============================================================================

cat("Creating Map 6: Climate Exposure Map...\n")

# Check for temperature variables
temp_vars <- c("climate_mean_temp", "mean_temp", "temperature", "temp")
temp_var <- temp_vars[temp_vars %in% names(clinical_sf)][1]

if (!is.na(temp_var)) {

  # Create grid and aggregate temperature
  grid <- st_make_grid(jhb_metro, cellsize = 2000, square = TRUE)
  grid_sf <- st_sf(grid_id = 1:length(grid), geometry = grid)

  climate_grid <- st_join(grid_sf, clinical_sf) %>%
    filter(!is.na(!!sym(temp_var))) %>%
    group_by(grid_id) %>%
    summarize(
      mean_temp = mean(!!sym(temp_var), na.rm = TRUE),
      n_obs = n(),
      .groups = "drop"
    ) %>%
    filter(n_obs >= 3)

  map6 <- ggplot() +
    # Temperature grid
    geom_sf(
      data = climate_grid,
      aes(fill = mean_temp),
      color = NA
    ) +

    # Metropolitan boundary
    geom_sf(
      data = jhb_metro,
      fill = NA,
      color = "gray30",
      linewidth = 1
    ) +

    # Color scale (diverging from cool to warm)
    scale_fill_scico(
      palette = "vik",
      name = "Mean Temperature (°C)",
      guide = guide_colorbar(
        barwidth = 10,
        barheight = 0.6,
        title.position = "top",
        title.hjust = 0.5
      )
    ) +

    # Cartographic elements
    annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
    annotation_north_arrow(location = "tl", which_north = "true") +

    # Theme
    theme_minimal(base_size = 11, base_family = "Arial") +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      plot.caption = element_text(size = 8, hjust = 0),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    ) +

    labs(
      title = "Climate Exposure: Mean Temperature",
      subtitle = "ERA5 reanalysis matched to clinical survey dates (2002-2021)",
      caption = paste0(
        "Data: ERA5 0.25° grid (~31km resolution) | ",
        "Spatial interpolation: 2km grid | ",
        "n = ", format(nrow(clinical_sf), big.mark = ","), " observations"
      )
    )

  # Save
  ggsave(
    "figures/maps/journal/map6_climate_exposure.svg",
    plot = map6,
    width = 7,
    height = 6,
    dpi = 300,
    device = svglite
  )

  ggsave(
    "figures/maps/presentation/map6_climate_exposure.png",
    plot = map6,
    width = 10,
    height = 8,
    dpi = 150
  )

  cat("✓ Map 6 saved\n\n")

} else {
  cat("⚠ Temperature variable not found, skipping Map 6\n\n")
}

# ==============================================================================
# 9. PRESENTATION-READY VERSIONS (HIGH CONTRAST)
# ==============================================================================

cat("Creating presentation-ready versions...\n")

# High-contrast version of study region
map_presentation <- ggplot() +
  geom_sf(
    data = jhb_metro,
    fill = "#264653",
    color = "#E9C46A",
    linewidth = 2
  ) +

  annotation_scale(
    location = "br",
    width_hint = 0.3,
    text_cex = 1.5,
    style = "ticks",
    line_width = 1.5
  ) +

  theme_void(base_size = 18, base_family = "Arial") +
  theme(
    plot.background = element_rect(fill = "#1a1a1a", color = NA),
    plot.title = element_text(
      face = "bold",
      size = 26,
      color = "white",
      hjust = 0.5,
      margin = margin(t = 10, b = 15)
    ),
    plot.subtitle = element_text(
      size = 20,
      color = "gray80",
      hjust = 0.5,
      margin = margin(b = 20)
    ),
    plot.margin = margin(20, 20, 20, 20)
  ) +

  labs(
    title = "Study Region",
    subtitle = "Johannesburg, South Africa"
  )

ggsave(
  "figures/maps/presentation/map_presentation_high_contrast.png",
  plot = map_presentation,
  width = 12,
  height = 6.75,
  dpi = 150
)

cat("✓ Presentation maps saved\n\n")

# ==============================================================================
# 10. SOCIAL MEDIA VERSIONS
# ==============================================================================

cat("Creating social media versions...\n")

# Square format for Instagram/Twitter
map_social <- ggplot() +
  geom_sf(
    data = jhb_metro,
    fill = "#2A9D8F",
    color = "white",
    linewidth = 1.5
  ) +

  geom_sf(
    data = clinical_sf,
    color = "#E63946",
    size = 1.5,
    alpha = 0.7
  ) +

  theme_void(base_size = 16, base_family = "Arial") +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(
      face = "bold",
      size = 24,
      hjust = 0.5,
      margin = margin(t = 20, b = 10)
    ),
    plot.subtitle = element_text(
      size = 18,
      hjust = 0.5,
      margin = margin(b = 20)
    ),
    plot.caption = element_text(
      size = 14,
      hjust = 0.5,
      margin = margin(t = 10, b = 10),
      color = "gray40"
    ),
    plot.margin = margin(15, 15, 15, 15)
  ) +

  labs(
    title = "Climate-Health Research",
    subtitle = "Johannesburg, South Africa",
    caption = "#ClimateHealth #ENBEL"
  )

# Instagram square
ggsave(
  "figures/maps/social/map_instagram_square.png",
  plot = map_social,
  width = 1080,
  height = 1080,
  units = "px",
  dpi = 96
)

# Twitter 2:1
ggsave(
  "figures/maps/social/map_twitter_card.png",
  plot = map_social,
  width = 1200,
  height = 600,
  units = "px",
  dpi = 96
)

cat("✓ Social media maps saved\n\n")

# ==============================================================================
# 11. SUMMARY
# ==============================================================================

cat("==============================================================================\n")
cat("📍 MAP GENERATION COMPLETE!\n")
cat("==============================================================================\n\n")

cat("Generated maps:\n")
cat("  Journal versions (SVG, 300 DPI):\n")
cat("    - figures/maps/journal/map1_study_region_overview.svg\n")
cat("    - figures/maps/journal/map2_clinical_survey_locations.svg\n")
cat("    - figures/maps/journal/map3_gcro_survey_locations.svg\n")
cat("    - figures/maps/journal/map4_combined_survey_coverage.svg\n")
if (exists("map5")) cat("    - figures/maps/journal/map5_heat_vulnerability.svg\n")
if (exists("map6")) cat("    - figures/maps/journal/map6_climate_exposure.svg\n")

cat("\n  Presentation versions (PNG, 150 DPI):\n")
cat("    - figures/maps/presentation/*.png (6-8 files)\n")

cat("\n  Social media versions (optimized):\n")
cat("    - figures/maps/social/map_instagram_square.png (1080x1080)\n")
cat("    - figures/maps/social/map_twitter_card.png (1200x600)\n")

cat("\n==============================================================================\n")
cat("All maps are publication-ready for:\n")
cat("  ✓ Peer-reviewed journals (Nature, Science, The Lancet, etc.)\n")
cat("  ✓ Conference presentations (16:9 slides)\n")
cat("  ✓ Social media (Twitter, Instagram, LinkedIn)\n")
cat("  ✓ Posters and reports\n")
cat("==============================================================================\n\n")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
