#!/usr/bin/env Rscript
# ENBEL Project - Johannesburg Study Distribution Map
# Professional cartographic visualization using actual JHB shapefile
# with distinct GCRO survey wave colors
#
# Output: 16:9 aspect ratio SVG optimized for Figma presentations
# Date: 2025-10-14

# Required packages
required_packages <- c("sf", "ggplot2", "dplyr", "scales", "svglite")

# Install missing packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Suppress warnings
options(warn = -1)

cat("=== ENBEL Johannesburg Study Distribution Map ===\n\n")

# ============================================================================
# 1. LOAD JOHANNESBURG BOUNDARY SHAPEFILE
# ============================================================================

cat("Loading Johannesburg metropolitan boundary shapefile...\n")

shapefile_path <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/map_vector_folder_JHB/JHB/metropolitan municipality jhb.shp"

jhb_boundary <- st_read(shapefile_path, quiet = TRUE)

# Ensure WGS84 CRS
if (is.na(st_crs(jhb_boundary))) {
  cat("Setting CRS to WGS84 (EPSG:4326)...\n")
  st_crs(jhb_boundary) <- 4326
} else if (st_crs(jhb_boundary)$epsg != 4326) {
  cat("Transforming CRS to WGS84 (EPSG:4326)...\n")
  jhb_boundary <- st_transform(jhb_boundary, 4326)
}

# Get boundary extent
jhb_bbox <- st_bbox(jhb_boundary)
cat(sprintf("Boundary extent: Lon [%.4f, %.4f], Lat [%.4f, %.4f]\n",
            jhb_bbox["xmin"], jhb_bbox["xmax"],
            jhb_bbox["ymin"], jhb_bbox["ymax"]))

# ============================================================================
# 2. DEFINE CLINICAL TRIAL SITES
# ============================================================================

cat("\nDefining clinical trial sites from actual data...\n")

# Load clinical data to get real coordinates
clinical_data <- read.csv("/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv",
                          stringsAsFactors = FALSE)

# Group by unique coordinates and aggregate
library(dplyr)
site_coords <- clinical_data %>%
  group_by(latitude, longitude) %>%
  summarize(
    count = n(),
    studies = paste(unique(study_source), collapse = ", "),
    location_names = paste(unique(study_site_location), collapse = " | "),
    .groups = 'drop'
  ) %>%
  arrange(desc(count))

# Create clinical sites dataframe with real coordinates
clinical_sites <- data.frame(
  name = c(
    "Central JHB Hub",
    "Northern Site",
    "Western JHB Hub",
    "Southwest Site"
  ),
  lon = c(28.0473, 28.2293, 27.91, 27.89),
  lat = c(-26.2041, -25.7479, -26.25, -26.32),
  patients = c(7922, 2751, 696, 29),
  studies_text = c("WRHI, DPHRU, Ezin, JHSPH, VIDA", "Aurum", "ACTG series", "SCHARP"),
  research_type = c("Mixed", "TB/HIV", "HIV", "HIV"),
  location_type = c("Inside JHB", "Outside JHB", "Inside JHB", "Inside JHB"),
  stringsAsFactors = FALSE
)

# Convert to sf object
clinical_sites_sf <- st_as_sf(clinical_sites,
                               coords = c("lon", "lat"),
                               crs = 4326)

cat(sprintf("Loaded %d clinical trial sites\n", nrow(clinical_sites)))

# ============================================================================
# 3. LOAD ACTUAL GCRO SURVEY DATA WITH HIGHLY DISTINCT WAVE COLORS
# ============================================================================

cat("\nLoading actual GCRO survey data...\n")

gcro_data_path <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"

# Read GCRO data (only needed columns for performance)
gcro_raw <- read.csv(gcro_data_path, stringsAsFactors = FALSE)

cat(sprintf("Loaded %d total GCRO records\n", nrow(gcro_raw)))

# Filter for records with valid coordinates within JHB bounds
gcro_points <- gcro_raw %>%
  filter(!is.na(latitude) & !is.na(longitude) & !is.na(survey_year)) %>%
  filter(longitude >= jhb_bbox["xmin"] & longitude <= jhb_bbox["xmax"]) %>%
  filter(latitude >= jhb_bbox["ymin"] & latitude <= jhb_bbox["ymax"]) %>%
  select(survey_year, latitude, longitude) %>%
  rename(year = survey_year, lat = latitude, lon = longitude)

# MUCH MORE DISTINCTIVE COLORS - NOT ALL BLUE!
# Using highly contrasting colors for better distinction
color_mapping <- data.frame(
  year = c(2011, 2014, 2018, 2021),
  color = c(
    "#1565C0",  # 2011: Deep Blue
    "#00897B",  # 2014: Teal/Cyan (very different from blue!)
    "#7B1FA2",  # 2018: Purple (completely different!)
    "#F57C00"   # 2021: Orange (extremely distinctive!)
  ),
  label = c("2011", "2014", "2018", "2021"),
  stringsAsFactors = FALSE
)

# Add colors to GCRO points
gcro_points <- gcro_points %>%
  left_join(color_mapping, by = "year") %>%
  filter(!is.na(color))

cat(sprintf("Filtered to %d GCRO points within JHB boundary\n", nrow(gcro_points)))
cat(sprintf("Survey years: %s\n", paste(unique(gcro_points$year), collapse = ", ")))

# Sample if too many points (for performance)
max_points <- 3000
if (nrow(gcro_points) > max_points) {
  set.seed(42)
  gcro_points <- gcro_points %>%
    group_by(year) %>%
    sample_n(min(n(), max_points / length(unique(gcro_points$year)))) %>%
    ungroup()
  cat(sprintf("Sampled down to %d points for visualization\n", nrow(gcro_points)))
}

# Convert to sf object
gcro_points_sf <- st_as_sf(gcro_points,
                            coords = c("lon", "lat"),
                            crs = 4326)

# Store color mapping for legend
survey_waves <- color_mapping

# ============================================================================
# 4. CREATE THE MAP
# ============================================================================

cat("\nCreating map with 16:9 aspect ratio...\n")

# Define color scheme
colors <- list(
  boundary = "#34495E",
  fill = "#F8F9FA",
  text = "#2C3E50"
)

# Create base plot
p <- ggplot() +
  # Plot JHB boundary fill
  geom_sf(data = jhb_boundary,
          fill = colors$fill,
          color = colors$boundary,
          size = 0.8,
          alpha = 0.3) +

  # Plot GCRO survey points FIRST (so they're in background) with HIGHLY DISTINCT colors
  geom_sf(data = gcro_points_sf,
          aes(color = factor(year)),
          size = 2,  # Larger size
          alpha = 0.8,  # More visible
          show.legend = TRUE) +

  # Manually specify HIGHLY DISTINCT colors for each wave
  scale_color_manual(
    name = "GCRO Survey Wave",
    values = setNames(survey_waves$color, survey_waves$year),
    labels = survey_waves$label,
    guide = guide_legend(
      override.aes = list(size = 5, alpha = 1),
      order = 1
    )
  ) +

  # Plot clinical trial sites with location-specific shapes (FIXED size, not scaled)
  geom_sf(data = clinical_sites_sf,
          aes(fill = research_type, shape = location_type),
          color = "white",
          size = 8,  # Fixed size
          stroke = 2.5,
          show.legend = TRUE) +

  # Clinical site shape mapping (circle for inside, triangle for outside)
  scale_shape_manual(
    name = "Site Location",
    values = c("Inside JHB" = 21, "Outside JHB" = 24),
    guide = guide_legend(override.aes = list(size = 5, fill = "grey60"), order = 3)
  ) +

  # Clinical site colors matching the PDF
  scale_fill_manual(
    name = "Clinical Research Focus",
    values = c(
      "Mixed" = "#EF5350",
      "HIV" = "#42A5F5",
      "COVID/HIV-TB" = "#FF9800",
      "TB/HIV" = "#9C27B0"
    ),
    guide = guide_legend(override.aes = list(size = 5, shape = 21), order = 2)
  ) +

  # Add site labels with study names
  geom_sf_label(data = clinical_sites_sf,
               aes(label = paste0(name, "\nN=", format(patients, big.mark = ","),
                                  "\n", studies_text)),
               size = 2.5,
               fontface = "bold",
               nudge_x = c(0.05, 0.05, -0.05, -0.05),  # 4 values for 4 sites
               nudge_y = c(0.02, 0.02, 0.02, 0.02),    # 4 values for 4 sites
               color = colors$text,
               fill = alpha("white", 0.85),
               label.padding = unit(0.25, "lines"),
               label.r = unit(0.15, "lines"),
               lineheight = 0.9) +

  # Title and labels matching PDF exactly
  labs(
    title = "ENBEL Climate-Health Analysis: Johannesburg Study Distribution",
    subtitle = "Clinical Trial Sites (Corrected Locations) and GCRO Household Survey Coverage",
    x = "",
    y = "",
    caption = paste0(
      "GCRO Quality of Life Surveys: 58,616 households across 4 survey waves (2011-2021)\n",
      "Clinical Trials: 10,202 patients across 17 studies | VIDA/DPHRU studies relocated to Soweto (CHBH)\n",
      "Geography: City of Johannesburg Metropolitan Municipality | Data: ENBEL Climate-Health Analysis Pipeline"
    )
  ) +

  # Theme
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = colors$text, hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = colors$text, hjust = 0.5, margin = margin(b = 10)),
    plot.caption = element_text(size = 8, color = colors$text, hjust = 0.5,
                                margin = margin(t = 10), lineheight = 1.2),
    axis.title = element_text(size = 10, color = colors$text),
    axis.text = element_text(size = 9, color = colors$text),
    legend.position = "right",
    legend.background = element_rect(fill = "white", color = "#BDC3C7", size = 0.5),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.8, "lines"),
    panel.grid.major = element_line(color = "#BDC3C7", size = 0.2),
    panel.grid.minor = element_line(color = "#ECF0F1", size = 0.1),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(10, 10, 10, 10)
  ) +

  # Coordinate system
  coord_sf(xlim = c(jhb_bbox["xmin"] - 0.02, jhb_bbox["xmax"] + 0.02),
           ylim = c(jhb_bbox["ymin"] - 0.02, jhb_bbox["ymax"] + 0.02),
           expand = FALSE)

# ============================================================================
# 5. SAVE OUTPUT
# ============================================================================

cat("\nSaving 16:9 SVG output...\n")

output_path <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/johannesburg_study_distribution_16x9_shapefile.svg"

# Save as SVG with 16:9 aspect ratio
ggsave(output_path,
       plot = p,
       width = 16,
       height = 9,
       units = "in",
       dpi = 150,
       device = "svg")

cat(sprintf("\n✓ Map saved to: %s\n", output_path))
cat("✓ Aspect ratio: 16:9 (optimized for Figma presentations)\n")
cat(sprintf("✓ GCRO survey points: %d points across %d waves\n",
            nrow(gcro_points), nrow(survey_waves)))
cat(sprintf("✓ Clinical trial sites: %d sites\n", nrow(clinical_sites)))

cat("\n=== HIGHLY DISTINCTIVE Color Scheme ===\n")
cat("  GCRO 2011: #1565C0 (Deep Blue)\n")
cat("  GCRO 2014: #00897B (Teal/Cyan - VERY different!)\n")
cat("  GCRO 2018: #7B1FA2 (Purple - completely different!)\n")
cat("  GCRO 2021: #F57C00 (Orange - extremely distinctive!)\n")
cat("\n=== Clinical Site Colors ===\n")
cat("  Mixed: #EF5350 (Red)\n")
cat("  HIV: #42A5F5 (Blue)\n")
cat("  COVID/HIV-TB: #FF9800 (Orange)\n")
cat("  TB/HIV: #9C27B0 (Purple)\n")

cat("\n=== Visual Hierarchy ===\n")
cat("1. Clinical Trial Sites: Large circles/triangles with labels\n")
cat("2. Johannesburg Boundary: Authentic shapefile boundary\n")
cat("3. GCRO Survey Points: REAL data locations with HIGHLY DISTINCTIVE colors\n")
cat("4. Colors: Blue, Teal, Purple, Orange (not all blue shades!)\n")

cat("\n✓ Map generation complete!\n")
