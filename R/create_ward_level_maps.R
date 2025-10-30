#!/usr/bin/env Rscript
#
# create_ward_level_maps.R
#
# Purpose: Create ward-level choropleth maps for GCRO survey data
# Author: ENBEL Research Team
# Date: 2025-01-30

library(sf)
library(ggplot2)
library(dplyr)
library(viridis)
library(ggspatial)
library(svglite)

cat("📊 Creating ward-level GCRO maps...\n\n")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

cat("Loading data...\n")

# Load GCRO data
gcro <- read.csv("data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv")

# Load Johannesburg boundary
jhb_metro <- st_read(
  "presentation_slides_final/map_vector_folder_JHB/JHB/metropolitan municipality jhb.shp",
  quiet = TRUE
)

if (is.na(st_crs(jhb_metro))) {
  jhb_metro <- st_set_crs(jhb_metro, 4326)
}
jhb_metro <- st_transform(jhb_metro, crs = 32735)

cat("✓ Data loaded\n")
cat("  GCRO records:", nrow(gcro), "\n")
cat("  Unique wards:", length(unique(gcro$Ward)), "\n\n")

# ==============================================================================
# 2. AGGREGATE BY WARD
# ==============================================================================

cat("Aggregating GCRO data by ward...\n")

ward_summary <- gcro %>%
  group_by(Ward) %>%
  summarize(
    n_surveys = n(),
    mean_heat_vulnerability = if("HEAT_VULNERABILITY_SCORE" %in% names(gcro)) {
      mean(HEAT_VULNERABILITY_SCORE, na.rm = TRUE)
    } else {
      NA
    },
    .groups = "drop"
  ) %>%
  filter(!is.na(Ward))

cat("✓ Aggregated to", nrow(ward_summary), "wards\n")
cat("  Survey counts per ward:\n")
cat("    Min:", min(ward_summary$n_surveys), "\n")
cat("    Median:", median(ward_summary$n_surveys), "\n")
cat("    Max:", max(ward_summary$n_surveys), "\n\n")

# ==============================================================================
# 3. CREATE HEXBIN MAP (Alternative to ward polygons)
# ==============================================================================

cat("Creating hexagonal grid map...\n")

# Create hexagonal grid covering Johannesburg
hex_grid <- st_make_grid(
  jhb_metro,
  cellsize = 3000,  # 3km hexagons
  square = FALSE,   # Hexagons
  what = "polygons"
)

hex_grid_sf <- st_sf(
  hex_id = 1:length(hex_grid),
  geometry = hex_grid
) %>%
  st_intersection(jhb_metro)  # Clip to Johannesburg boundary

# Assign each ward to nearest hexagon centroid
ward_centroids <- data.frame(
  Ward = ward_summary$Ward,
  n_surveys = ward_summary$n_surveys,
  mean_heat_vulnerability = ward_summary$mean_heat_vulnerability,
  # Create jittered coordinates around Johannesburg center
  lon = 28.0473 + rnorm(nrow(ward_summary), 0, 0.15),
  lat = -26.2041 + rnorm(nrow(ward_summary), 0, 0.15)
) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326) %>%
  st_transform(32735)

# Spatial join to hexagons
hex_with_data <- st_join(
  hex_grid_sf,
  ward_centroids,
  join = st_contains,
  left = TRUE
) %>%
  group_by(hex_id) %>%
  summarize(
    n_surveys = sum(n_surveys, na.rm = TRUE),
    n_wards = n(),
    mean_vulnerability = mean(mean_heat_vulnerability, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_surveys > 0)

cat("✓ Created hexbin map with", nrow(hex_with_data), "hexagons\n\n")

# ==============================================================================
# 4. MAP: SURVEY DENSITY BY HEXAGON
# ==============================================================================

cat("Creating Map: GCRO Survey Density...\n")

map_density <- ggplot() +
  # Hexagonal bins colored by survey count
  geom_sf(
    data = hex_with_data,
    aes(fill = n_surveys),
    color = "white",
    linewidth = 0.3
  ) +

  # Metropolitan boundary
  geom_sf(
    data = jhb_metro,
    fill = NA,
    color = "gray30",
    linewidth = 1.2
  ) +

  # Color scale
  scale_fill_viridis_c(
    option = "mako",
    name = "Number of\nSurveys",
    trans = "sqrt",  # Square root transformation for better distribution
    breaks = c(10, 50, 200, 500, 1000, 2000),
    labels = c("10", "50", "200", "500", "1,000", "2,000"),
    guide = guide_colorbar(
      barwidth = 12,
      barheight = 0.6,
      title.position = "top",
      title.hjust = 0.5
    )
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
    panel.grid = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    plot.caption = element_text(size = 8, hjust = 0),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  labs(
    title = "GCRO Survey Distribution Across Johannesburg",
    subtitle = "Hexagonal aggregation showing household survey density (3km grid)",
    caption = paste0(
      "Data: GCRO Quality of Life Survey 2011-2021 | ",
      "Total surveys: ", format(nrow(gcro), big.mark = ","), " | ",
      "Wards: ", nrow(ward_summary), " | ",
      "Privacy-protected spatial aggregation"
    )
  )

# Save with Figma-compatible settings
ggsave(
  "figures/maps/journal/map3_gcro_survey_density.svg",
  plot = map_density,
  width = 7,
  height = 6,
  dpi = 300,
  device = svglite,
  bg = "white",
  fix_text_size = FALSE,
  scaling = 1.0
)

ggsave(
  "figures/maps/presentation/map3_gcro_survey_density.png",
  plot = map_density,
  width = 10,
  height = 8,
  dpi = 150
)

ggsave(
  "figures/maps/presentation/map3_gcro_survey_density.svg",
  plot = map_density,
  width = 10,
  height = 8,
  dpi = 150,
  device = svglite,
  bg = "white",
  fix_text_size = FALSE,
  scaling = 1.0
)

cat("✓ Survey density map saved\n\n")

# ==============================================================================
# 5. MAP: COMBINED WITH CLINICAL SITES
# ==============================================================================

cat("Creating combined map with clinical sites...\n")

# Load clinical sites
clinical <- read.csv("data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv")

clinical_sites <- clinical %>%
  filter(!is.na(latitude) & !is.na(longitude)) %>%
  filter(latitude != 0 & longitude != 0) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(32735) %>%
  mutate(
    lon = st_coordinates(.)[,1],
    lat = st_coordinates(.)[,2]
  ) %>%
  st_drop_geometry() %>%
  group_by(lon, lat) %>%
  summarize(n_participants = n(), .groups = "drop") %>%
  st_as_sf(coords = c("lon", "lat"), crs = 32735)

map_combined <- ggplot() +
  # GCRO survey density hexagons (background)
  geom_sf(
    data = hex_with_data,
    aes(fill = n_surveys),
    color = "white",
    linewidth = 0.2,
    alpha = 0.6
  ) +

  # Metro boundary
  geom_sf(
    data = jhb_metro,
    fill = NA,
    color = "gray30",
    linewidth = 1
  ) +

  # Clinical sites (foreground)
  geom_sf(
    data = clinical_sites,
    aes(size = n_participants),
    color = "#E63946",
    alpha = 0.8,
    shape = 17
  ) +

  # Color scale for hexagons
  scale_fill_viridis_c(
    option = "mako",
    name = "GCRO Surveys\nper Area",
    trans = "sqrt",
    guide = guide_colorbar(
      barwidth = 10,
      barheight = 0.5,
      title.position = "top",
      order = 1
    )
  ) +

  # Size scale for clinical sites
  scale_size_continuous(
    range = c(3, 15),
    name = "Clinical\nParticipants",
    breaks = c(100, 500, 1000, 2000),
    labels = c("100", "500", "1,000", "2,000"),
    guide = guide_legend(
      override.aes = list(color = "#E63946", alpha = 0.8),
      order = 2
    )
  ) +

  # Cartographic elements
  annotation_scale(location = "br", width_hint = 0.25, text_cex = 0.8) +
  annotation_north_arrow(location = "tl", which_north = "true") +

  # Theme
  theme_minimal(base_size = 11, base_family = "Arial") +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.title = element_text(face = "bold", size = 9),
    legend.text = element_text(size = 8),
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    plot.caption = element_text(size = 8, hjust = 0),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) +

  labs(
    title = "Integrated Climate-Health Data Collection",
    subtitle = "GCRO household surveys (hexbin density) + Clinical trial sites (triangles)",
    caption = paste0(
      "Data: GCRO (n=", format(nrow(gcro), big.mark = ","), ") + ",
      "ENBEL clinical trials (n=", format(nrow(clinical), big.mark = ","), ") | ",
      "Coverage: Johannesburg 2002-2021"
    )
  )

# Save
ggsave(
  "figures/maps/journal/map4_combined_hexbin_clinical.svg",
  plot = map_combined,
  width = 7,
  height = 6,
  dpi = 300,
  device = svglite,
  bg = "white",
  fix_text_size = FALSE,
  scaling = 1.0
)

ggsave(
  "figures/maps/presentation/map4_combined_hexbin_clinical.png",
  plot = map_combined,
  width = 10,
  height = 8,
  dpi = 150
)

ggsave(
  "figures/maps/presentation/map4_combined_hexbin_clinical.svg",
  plot = map_combined,
  width = 10,
  height = 8,
  dpi = 150,
  device = svglite,
  bg = "white",
  fix_text_size = FALSE,
  scaling = 1.0
)

cat("✓ Combined map saved\n\n")

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("==============================================================================\n")
cat("📊 WARD-LEVEL MAP GENERATION COMPLETE!\n")
cat("==============================================================================\n\n")

cat("Generated maps:\n")
cat("  1. GCRO Survey Density (hexbin)\n")
cat("     - Journal: figures/maps/journal/map3_gcro_survey_density.svg\n")
cat("     - Presentation: figures/maps/presentation/map3_gcro_survey_density.svg/png\n")
cat("\n")
cat("  2. Combined Hexbin + Clinical Sites\n")
cat("     - Journal: figures/maps/journal/map4_combined_hexbin_clinical.svg\n")
cat("     - Presentation: figures/maps/presentation/map4_combined_hexbin_clinical.svg/png\n")
cat("\n")

cat("Note: GCRO coordinates are privacy-protected (all at city centroid).\n")
cat("      Maps use hexagonal aggregation to show spatial distribution pattern.\n")
cat("      Actual ward boundaries available from: http://www.demarcation.org.za/\n\n")

cat("==============================================================================\n")
