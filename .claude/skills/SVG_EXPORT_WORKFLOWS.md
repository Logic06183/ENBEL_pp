# SVG Export Workflows for R Graphics

## Professional Vector Graphics for Publications and Media

This skill provides expert guidance for exporting R graphics to publication-quality SVG format with optimization for different use cases.

---

## Why SVG for Scientific Graphics?

### Advantages
- **Scalable**: No quality loss at any size
- **Editable**: Can be modified in Illustrator, Inkscape, Figma
- **Small file size**: Especially for non-photographic content
- **Text remains text**: Fonts can be changed, text is searchable
- **Web-friendly**: Works in browsers and HTML documents
- **Print-ready**: Preferred by many journals

### When to Use PNG/PDF Instead
- **PNG**: Presentations with embedded images, social media
- **PDF**: LaTeX documents, multi-page reports
- **TIFF**: Some journal requirements (especially medical)

---

## Export Method 1: ggsave() with svglite

### Best for ggplot2 Graphics

```r
library(ggplot2)
library(svglite)

# Create plot
p <- ggplot(data, aes(x = temp, y = biomarker)) +
  geom_point() +
  theme_minimal()

# Export to SVG
ggsave(
  filename = "output/figure1.svg",
  plot = p,
  device = svglite::svglite,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300,
  scaling = 1.0,
  bg = "white"
)
```

### Key Parameters Explained

```r
ggsave(
  filename = "figure.svg",
  plot = p,                    # ggplot object
  device = svglite::svglite,   # Use svglite backend (better than built-in)
  width = 7,                   # Physical width
  height = 5,                  # Physical height
  units = "in",                # or "cm", "mm", "px"
  dpi = 300,                   # Resolution (affects text rendering)
  scaling = 1.0,               # Scale all elements (1.0 = 100%)
  bg = "white",                # Background color
  limitsize = FALSE            # Allow large files if needed
)
```

---

## Export Method 2: svglite() Device

### For Direct Control

```r
library(svglite)

# Open SVG device
svglite(
  filename = "output/figure2.svg",
  width = 7,
  height = 5,
  pointsize = 12,
  bg = "white",
  standalone = TRUE,         # Include XML declaration
  fix_text_size = TRUE,      # Adjust text rendering
  system_fonts = list(
    sans = "Arial",
    serif = "Times New Roman",
    mono = "Courier"
  )
)

# Create plot
plot(x, y, type = "l", col = "blue")
title("Time Series Analysis")

# Close device
dev.off()
```

---

## Export Method 3: tmap_save()

### For tmap Objects

```r
library(tmap)
library(sf)

# Create map
map <- tm_shape(districts) +
  tm_polygons(col = "population") +
  tm_scale_bar() +
  tm_compass()

# Save to SVG
tmap_save(
  tm = map,
  filename = "output/map_districts.svg",
  width = 10,
  height = 8,
  units = "in",
  dpi = 300
)
```

---

## Size Guidelines for Different Contexts

### Academic Journals

```r
# Single-column figure (typical 3.5-4 inches)
ggsave(
  "figure_single_column.svg",
  width = 3.5,
  height = 3,
  units = "in",
  dpi = 300
)

# Double-column figure (typical 7 inches)
ggsave(
  "figure_double_column.svg",
  width = 7,
  height = 5,
  units = "in",
  dpi = 300
)

# Full-page figure
ggsave(
  "figure_full_page.svg",
  width = 7,
  height = 9,
  units = "in",
  dpi = 300
)
```

### Nature/Science Journals (Specific Requirements)

```r
# Nature: 89 mm (single column) or 183 mm (double column)
ggsave(
  "nature_figure.svg",
  width = 89,
  height = 80,
  units = "mm",
  dpi = 300
)

# Science: 9 cm (single) or 18 cm (double)
ggsave(
  "science_figure.svg",
  width = 9,
  height = 7,
  units = "cm",
  dpi = 300
)
```

### Presentations

```r
# PowerPoint slide (16:9 widescreen)
ggsave(
  "slide_16_9.svg",
  width = 10,
  height = 5.625,
  units = "in",
  dpi = 150
)

# PowerPoint slide (4:3 traditional)
ggsave(
  "slide_4_3.svg",
  width = 10,
  height = 7.5,
  units = "in",
  dpi = 150
)
```

### Posters

```r
# Conference poster (A0 size, 841 x 1189 mm)
ggsave(
  "poster_figure.svg",
  width = 800,
  height = 600,
  units = "mm",
  dpi = 300
)
```

### Web/Social Media

```r
# Twitter/X card (2:1 ratio)
ggsave(
  "social_twitter.svg",
  width = 1200,
  height = 600,
  units = "px",
  dpi = 96
)

# Instagram square
ggsave(
  "social_instagram.svg",
  width = 1080,
  height = 1080,
  units = "px",
  dpi = 96
)
```

---

## Font Handling

### System Fonts in R

```r
library(systemfonts)

# List available fonts
system_fonts() %>%
  filter(grepl("Arial|Helvetica", name, ignore.case = TRUE))

# Use in ggplot
theme_minimal(
  base_family = "Arial",
  base_size = 11
)
```

### Embedding Fonts vs. Converting to Paths

```r
# Option 1: Embed fonts (editable text in Illustrator)
svglite(
  "figure_embedded_fonts.svg",
  width = 7,
  height = 5,
  system_fonts = list(sans = "Arial")
)

# Option 2: Convert text to paths (no font issues, not editable)
# This requires post-processing with tools like:
# - Inkscape CLI: inkscape --export-text-to-path
# - Cairo backend: see below
```

### Using Cairo Backend (Alternative)

```r
library(Cairo)

# Cairo SVG (good font handling)
cairo_svg(
  filename = "figure_cairo.svg",
  width = 7,
  height = 5,
  pointsize = 12,
  bg = "white"
)

plot(x, y)

dev.off()
```

---

## Optimizing SVG File Size

### Remove Unnecessary Metadata

```r
# After creating SVG, clean with svgo (Node.js tool)
# Install: npm install -g svgo

# Command line:
# svgo input.svg -o output.svg

# Or in R workflow:
system("svgo output/figure1.svg -o output/figure1_optimized.svg")
```

### Simplify Geometries Before Plotting

```r
library(sf)
library(rmapshaper)

# Original shapefile (large)
districts <- st_read("districts.shp")
object.size(districts)  # Check size

# Simplify (reduce vertices)
districts_simple <- ms_simplify(
  districts,
  keep = 0.05,          # Keep 5% of vertices
  keep_shapes = TRUE    # Don't remove small polygons
)

object.size(districts_simple)  # Much smaller

# Plot simplified version
ggplot(districts_simple) +
  geom_sf() +
  theme_minimal()
```

### Rasterize Complex Elements

```r
library(ggplot2)

# For maps with many tiny features, rasterize background
ggplot() +
  # Rasterized layer (thousands of points)
  geom_point(
    data = many_points,
    aes(x = lon, y = lat),
    size = 0.1,
    alpha = 0.3
  ) +

  # Vector layer (important features)
  geom_sf(data = districts, fill = NA, color = "black", linewidth = 0.5) +

  theme_minimal()

# Use ggrastr for selective rasterization
library(ggrastr)

ggplot() +
  rasterize(
    geom_point(data = many_points, aes(x, y)),
    dpi = 300
  ) +
  geom_sf(data = districts) +
  theme_minimal()
```

---

## Multi-Format Export Pipeline

### Generate All Required Formats at Once

```r
#' Export plot in multiple formats
#'
#' @param plot ggplot object
#' @param filename Base filename (without extension)
#' @param width Width in inches
#' @param height Height in inches
#' @param formats Vector of formats to export
export_multi <- function(plot, filename, width, height,
                        formats = c("svg", "png", "pdf")) {

  base_path <- dirname(filename)
  base_name <- basename(tools::file_path_sans_ext(filename))

  # Create output directory if needed
  if (!dir.exists(base_path)) {
    dir.create(base_path, recursive = TRUE)
  }

  # Export each format
  if ("svg" %in% formats) {
    ggsave(
      file.path(base_path, paste0(base_name, ".svg")),
      plot = plot,
      width = width,
      height = height,
      units = "in",
      dpi = 300,
      device = svglite::svglite
    )
    cat("✓ SVG saved\n")
  }

  if ("png" %in% formats) {
    ggsave(
      file.path(base_path, paste0(base_name, ".png")),
      plot = plot,
      width = width,
      height = height,
      units = "in",
      dpi = 300,
      bg = "white"
    )
    cat("✓ PNG saved\n")
  }

  if ("pdf" %in% formats) {
    ggsave(
      file.path(base_path, paste0(base_name, ".pdf")),
      plot = plot,
      width = width,
      height = height,
      units = "in",
      device = cairo_pdf
    )
    cat("✓ PDF saved\n")
  }

  if ("tiff" %in% formats) {
    ggsave(
      file.path(base_path, paste0(base_name, ".tiff")),
      plot = plot,
      width = width,
      height = height,
      units = "in",
      dpi = 300,
      compression = "lzw"
    )
    cat("✓ TIFF saved\n")
  }

  cat(paste0("All formats saved to: ", base_path, "\n"))
}

# Usage
map <- ggplot() + geom_sf(data = districts) + theme_minimal()

export_multi(
  plot = map,
  filename = "figures/figure1_map",
  width = 7,
  height = 6,
  formats = c("svg", "png", "pdf")
)
```

---

## Quality Control Checks

### Automated Testing Script

```r
#' Validate exported SVG file
#'
#' @param svg_path Path to SVG file
validate_svg <- function(svg_path) {

  if (!file.exists(svg_path)) {
    stop("SVG file not found: ", svg_path)
  }

  # Read SVG content
  svg_content <- readLines(svg_path)
  svg_text <- paste(svg_content, collapse = "\n")

  # Check file size
  file_size <- file.info(svg_path)$size / 1024  # KB
  cat(sprintf("File size: %.2f KB\n", file_size))

  if (file_size > 5000) {
    warning("Large SVG file (>5 MB). Consider optimization.")
  }

  # Check for XML declaration
  if (!grepl("^<\\?xml", svg_content[1])) {
    warning("Missing XML declaration.")
  }

  # Check for viewBox
  if (!grepl("viewBox", svg_text)) {
    warning("Missing viewBox attribute (may cause scaling issues).")
  }

  # Check for embedded fonts (warning if present)
  if (grepl("@font-face", svg_text)) {
    cat("✓ Fonts are embedded (good for portability)\n")
  } else {
    warning("Fonts may not be embedded (check compatibility).")
  }

  # Check dimensions
  width_match <- regmatches(svg_text, regexpr('width="[^"]*"', svg_text))
  height_match <- regmatches(svg_text, regexpr('height="[^"]*"', svg_text))

  cat("Dimensions:", width_match, "x", height_match, "\n")

  cat("\n✓ SVG validation complete\n")
}

# Usage
validate_svg("figures/figure1_map.svg")
```

---

## Post-Processing with Inkscape (Optional)

### Command-Line Optimization

```bash
# Install Inkscape (https://inkscape.org/)

# Convert text to paths (for font compatibility)
inkscape --export-text-to-path \
  --export-filename=output_text_as_paths.svg \
  input.svg

# Optimize (vacuum defs, remove metadata)
inkscape --vacuum-defs \
  --export-plain-svg \
  --export-filename=output_optimized.svg \
  input.svg

# Convert to PDF
inkscape --export-filename=output.pdf input.svg

# Convert to PNG (high-res)
inkscape --export-type=png \
  --export-dpi=300 \
  --export-filename=output.png \
  input.svg
```

### R Wrapper for Inkscape

```r
#' Convert SVG to PDF using Inkscape
#'
#' @param svg_path Path to SVG file
#' @param pdf_path Output PDF path (optional)
svg_to_pdf_inkscape <- function(svg_path, pdf_path = NULL) {

  if (is.null(pdf_path)) {
    pdf_path <- sub("\\.svg$", ".pdf", svg_path)
  }

  # Check if Inkscape is installed
  inkscape_cmd <- Sys.which("inkscape")
  if (inkscape_cmd == "") {
    stop("Inkscape not found. Install from https://inkscape.org/")
  }

  # Run Inkscape
  system2(
    command = "inkscape",
    args = c(
      "--export-filename", pdf_path,
      svg_path
    )
  )

  cat("PDF created:", pdf_path, "\n")
}

# Usage
svg_to_pdf_inkscape("figures/figure1_map.svg")
```

---

## Troubleshooting Common Issues

### Issue 1: Fonts Not Rendering Correctly

**Problem**: Fonts look different in Illustrator/browser

**Solutions**:
```r
# Option A: Use common system fonts
theme_minimal(base_family = "Arial")  # Widely available

# Option B: Embed fonts explicitly
svglite(
  "output.svg",
  system_fonts = list(
    sans = "Arial",
    serif = "Times New Roman",
    mono = "Courier New"
  )
)

# Option C: Convert text to paths (post-processing)
# inkscape --export-text-to-path input.svg -o output.svg
```

### Issue 2: Large File Sizes

**Problem**: SVG file is >10 MB

**Solutions**:
```r
# 1. Simplify geometries BEFORE plotting
districts <- ms_simplify(districts, keep = 0.05)

# 2. Reduce data points
data_sample <- data %>% sample_frac(0.1)  # Use 10% of data

# 3. Rasterize complex layers
library(ggrastr)
rasterize(geom_point(...), dpi = 300)

# 4. Remove unnecessary precision
districts <- st_set_precision(districts, 1e5)

# 5. Optimize with svgo
system("svgo input.svg -o output.svg")
```

### Issue 3: SVG Doesn't Scale Properly

**Problem**: SVG appears at wrong size when imported

**Solutions**:
```r
# Ensure viewBox is set
svglite(
  "output.svg",
  width = 7,
  height = 5,
  standalone = TRUE  # Includes viewBox
)

# Alternatively, use Cairo backend
cairo_svg(
  "output.svg",
  width = 7,
  height = 5
)
```

### Issue 4: Colors Look Different in Print

**Problem**: Colors appear washed out or wrong

**Solutions**:
```r
# 1. Use colorblind-safe palettes
scale_fill_viridis_c()

# 2. Test in grayscale
scale_fill_grey(start = 0.2, end = 0.8)

# 3. Specify CMYK-safe colors (for print)
# Use tools like https://www.pantone.com/

# 4. Preview in different contexts
# - PDF viewers
# - Web browsers
# - Printed on paper
```

---

## Complete Publication Workflow

### From R to Journal Submission

```r
#!/usr/bin/env Rscript
# workflow_publication_figures.R

library(ggplot2)
library(sf)
library(svglite)

# Set options
options(
  scipen = 999,  # Disable scientific notation
  digits = 3     # Decimal places
)

# Create output directories
dir.create("figures/journal", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/web", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/slides", recursive = TRUE, showWarnings = FALSE)

# Load data and create plot
data <- st_read("data/processed/districts.shp")
map <- ggplot(data) +
  geom_sf(aes(fill = vulnerability)) +
  scale_fill_viridis_c() +
  theme_minimal(base_size = 11, base_family = "Arial")

# Journal version (SVG, double-column)
ggsave(
  "figures/journal/figure1_map.svg",
  plot = map,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300,
  device = svglite::svglite,
  bg = "white"
)

# Web version (PNG, 1200px wide)
ggsave(
  "figures/web/figure1_map.png",
  plot = map,
  width = 1200,
  height = 900,
  units = "px",
  dpi = 96
)

# Slide version (SVG, 16:9)
ggsave(
  "figures/slides/figure1_map.svg",
  plot = map,
  width = 10,
  height = 5.625,
  units = "in",
  dpi = 150
)

# Also create PDF for LaTeX manuscripts
cairo_pdf(
  "figures/journal/figure1_map.pdf",
  width = 7,
  height = 5
)
print(map)
dev.off()

cat("✓ All figures exported successfully\n")
cat("  Journal: figures/journal/\n")
cat("  Web: figures/web/\n")
cat("  Slides: figures/slides/\n")
```

---

## Additional Resources

- **svglite package**: https://svglite.r-lib.org/
- **Cairo graphics**: https://www.rforge.net/Cairo/
- **Inkscape**: https://inkscape.org/
- **SVGO optimizer**: https://github.com/svg/svgo
- **Journal figure guidelines**:
  - Nature: https://www.nature.com/nature/for-authors/final-submission
  - Science: https://www.science.org/content/page/instructions-preparing-initial-manuscript
  - PLOS: https://journals.plos.org/plosone/s/figures

---

**Last Updated**: 2025-01-30
**Version**: 1.0
**Author**: Claude Code Expert System
