# Publication Map Styling Guide

## Context-Specific Design for Maximum Impact

This skill provides expert styling guidance for creating maps that are optimized for different publication contexts: peer-reviewed journals, conference presentations, newspapers, magazines, and social media.

---

## The Golden Rules of Map Styling

### 1. Know Your Audience
- **Scientists**: Precision, accuracy, complete metadata
- **Public**: Clarity, simplicity, engaging visuals
- **Policymakers**: Key insights, actionable information
- **Media**: Eye-catching, immediate comprehension

### 2. Design for the Medium
- **Print**: High contrast, larger text, consider grayscale
- **Screen**: Vibrant colors, interactive potential
- **Projection**: Simple, bold, visible from distance
- **Mobile**: Minimal detail, large touch targets

### 3. Data-Ink Ratio
- Maximize information per visual element
- Remove chart junk and unnecessary decorations
- Every element should serve a purpose

---

## Style 1: Peer-Reviewed Journal

### Characteristics
- Conservative, professional appearance
- Complete metadata and citations
- High information density
- Grayscale-compatible
- Standard fonts
- Detailed legends

### Implementation

```r
library(ggplot2)
library(sf)
library(ggspatial)

# Journal-style theme
theme_journal <- function(base_size = 10, base_family = "Arial") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # White background, subtle grid
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray92", linewidth = 0.25),
      panel.grid.minor = element_blank(),

      # Plot area
      plot.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(
        face = "bold",
        size = base_size + 2,
        hjust = 0,
        margin = margin(b = 8)
      ),
      plot.subtitle = element_text(
        size = base_size,
        color = "gray30",
        hjust = 0,
        margin = margin(b = 8)
      ),
      plot.caption = element_text(
        size = base_size - 2,
        color = "gray40",
        hjust = 0,
        margin = margin(t = 8)
      ),
      plot.margin = margin(15, 15, 15, 15),

      # Legend
      legend.position = "bottom",
      legend.title = element_text(
        face = "bold",
        size = base_size - 1
      ),
      legend.text = element_text(size = base_size - 2),
      legend.key.width = unit(1.2, "cm"),
      legend.key.height = unit(0.4, "cm"),
      legend.background = element_rect(
        fill = "white",
        color = "gray70",
        linewidth = 0.3
      ),
      legend.margin = margin(t = 5, r = 5, b = 5, l = 5),

      # Axes (usually removed for maps)
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),

      # Facets
      strip.background = element_rect(fill = "gray95", color = "gray70"),
      strip.text = element_text(face = "bold", size = base_size)
    )
}

# Create journal map
map_journal <- ggplot() +
  geom_sf(
    data = districts,
    aes(fill = vulnerability_index),
    color = "gray40",
    linewidth = 0.3
  ) +

  # Conservative color scheme (Nature prefers)
  scale_fill_viridis_c(
    option = "viridis",
    name = "Heat Vulnerability Index\n(dimensionless, 0-100 scale)",
    guide = guide_colorbar(
      barwidth = 10,
      barheight = 0.5,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +

  # Essential cartographic elements
  annotation_scale(
    location = "br",
    width_hint = 0.25,
    text_cex = 0.7,
    style = "ticks"
  ) +

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

  theme_journal(base_size = 10) +

  # Complete metadata
  labs(
    title = "Spatial Distribution of Heat Vulnerability in Johannesburg",
    subtitle = "Ward-level analysis based on composite vulnerability index (2021)",
    caption = paste0(
      "Data: Greater Capital Region Observatory (GCRO) Quality of Life Survey 2021 | ",
      "Projection: UTM Zone 35S (EPSG:32735) | ",
      "Vulnerability index integrates socioeconomic factors, dwelling characteristics, ",
      "and health infrastructure access | ",
      "n = 58,616 household surveys across 258 wards"
    )
  )

# Export for journal (double-column, 7 inches)
ggsave(
  "figures/journal_map.svg",
  plot = map_journal,
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 300,
  device = svglite::svglite
)
```

---

## Style 2: Conference Presentation

### Characteristics
- Bold, high-contrast design
- Large text (readable from 20+ feet)
- Simplified details
- Vibrant colors
- Minimal text
- Clear takeaway message

### Implementation

```r
# Presentation theme
theme_presentation <- function(base_size = 18, base_family = "Arial") {
  theme_void(base_size = base_size, base_family = base_family) +
    theme(
      # Dark background for projector
      plot.background = element_rect(fill = "#1a1a1a", color = NA),
      panel.background = element_rect(fill = "#1a1a1a", color = NA),

      # Large, bold text
      plot.title = element_text(
        face = "bold",
        size = base_size + 8,
        color = "white",
        hjust = 0.5,
        margin = margin(t = 10, b = 15)
      ),
      plot.subtitle = element_text(
        size = base_size + 2,
        color = "gray80",
        hjust = 0.5,
        margin = margin(b = 20)
      ),
      plot.caption = element_text(
        size = base_size - 4,
        color = "gray60",
        hjust = 0.5,
        margin = margin(t = 15)
      ),
      plot.margin = margin(20, 20, 20, 20),

      # Legend
      legend.position = "right",
      legend.title = element_text(
        face = "bold",
        size = base_size,
        color = "white"
      ),
      legend.text = element_text(
        size = base_size - 2,
        color = "white"
      ),
      legend.key.width = unit(1.5, "cm"),
      legend.key.height = unit(2, "cm"),
      legend.background = element_blank()
    )
}

# Create presentation map
map_presentation <- ggplot() +
  geom_sf(
    data = districts,
    aes(fill = vulnerability_index),
    color = "white",
    linewidth = 0.8
  ) +

  # High-contrast colors
  scale_fill_viridis_c(
    option = "magma",
    name = "Heat\nVulnerability",
    direction = -1,  # Light = high vulnerability
    guide = guide_colorbar(
      barwidth = 1.5,
      barheight = 10,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +

  theme_presentation(base_size = 18) +

  # Simple, impactful message
  labs(
    title = "Heat Vulnerability Across Johannesburg",
    subtitle = "Higher vulnerability in northern and eastern districts",
    caption = "GCRO 2021 | n = 58,616 households"
  )

# Export for presentation (16:9, large)
ggsave(
  "figures/presentation_map.svg",
  plot = map_presentation,
  width = 12,
  height = 6.75,
  units = "in",
  dpi = 150
)

# Alternative: Light background version
map_presentation_light <- map_presentation +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(color = "black"),
    plot.subtitle = element_text(color = "gray30"),
    plot.caption = element_text(color = "gray50"),
    legend.title = element_text(color = "black"),
    legend.text = element_text(color = "black")
  )
```

---

## Style 3: Newspaper/Magazine

### Characteristics
- Attention-grabbing
- Strong visual hierarchy
- Limited technical jargon
- Human interest angle
- Colorful but print-safe
- Self-explanatory

### Implementation

```r
# Media theme
theme_media <- function(base_size = 14, base_family = "Arial") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Clean white background
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid = element_blank(),

      # Eye-catching title
      plot.title = element_text(
        face = "bold",
        size = base_size + 6,
        hjust = 0,
        margin = margin(b = 8),
        lineheight = 1.2
      ),
      plot.subtitle = element_text(
        size = base_size + 1,
        hjust = 0,
        margin = margin(b = 12),
        lineheight = 1.3
      ),
      plot.caption = element_text(
        size = base_size - 2,
        color = "gray40",
        hjust = 0,
        margin = margin(t = 10),
        face = "italic"
      ),
      plot.margin = margin(20, 20, 20, 20),

      # Clear legend
      legend.position = "bottom",
      legend.title = element_text(
        face = "bold",
        size = base_size
      ),
      legend.text = element_text(size = base_size - 1),
      legend.key.width = unit(2, "cm"),
      legend.key.height = unit(0.5, "cm"),

      # Remove axes
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank()
    )
}

# Create media map
map_media <- ggplot() +
  geom_sf(
    data = districts,
    aes(fill = vulnerability_category),  # Categorical for clarity
    color = "white",
    linewidth = 0.5
  ) +

  # Highlight key area
  geom_sf(
    data = districts %>% filter(district_name == "Alexandra"),
    fill = NA,
    color = "#E63946",
    linewidth = 2
  ) +

  # Simple categorical colors
  scale_fill_manual(
    values = c(
      "Low" = "#2A9D8F",
      "Moderate" = "#E9C46A",
      "High" = "#F4A261",
      "Very High" = "#E76F51"
    ),
    name = "Vulnerability Level",
    guide = guide_legend(
      nrow = 1,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +

  # Label key area
  annotate(
    "text",
    x = 28.12,
    y = -26.10,
    label = "Alexandra Township\n(Highest vulnerability)",
    size = 5,
    color = "#E63946",
    fontface = "bold",
    hjust = 0
  ) +

  theme_media(base_size = 14) +

  # Human-interest framing
  labs(
    title = "Where Johannesburg Residents Are Most at Risk from Heat Waves",
    subtitle = "Communities in northern townships face higher vulnerability due to informal housing\nand limited access to cooling infrastructure",
    caption = "Source: Greater Capital Region Observatory, 2021 Quality of Life Survey | Analysis by [Your Organization]"
  )

# Export for media (flexible size)
ggsave(
  "figures/media_map.png",
  plot = map_media,
  width = 10,
  height = 8,
  units = "in",
  dpi = 300
)
```

---

## Style 4: Scientific Poster

### Characteristics
- Large format (readable from 3-5 feet)
- Bold colors and text
- Self-contained (can understand without presenter)
- Clear visual hierarchy
- Effective use of white space

### Implementation

```r
# Poster theme
theme_poster <- function(base_size = 22, base_family = "Arial") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Clean background
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor = element_blank(),

      # Very large text
      plot.title = element_text(
        face = "bold",
        size = base_size + 12,
        hjust = 0.5,
        margin = margin(b = 15)
      ),
      plot.subtitle = element_text(
        size = base_size + 4,
        hjust = 0.5,
        margin = margin(b = 20)
      ),
      plot.caption = element_text(
        size = base_size - 2,
        hjust = 0.5,
        margin = margin(t = 15)
      ),
      plot.margin = margin(30, 30, 30, 30),

      # Large legend
      legend.position = "right",
      legend.title = element_text(
        face = "bold",
        size = base_size + 2
      ),
      legend.text = element_text(size = base_size),
      legend.key.width = unit(2, "cm"),
      legend.key.height = unit(3, "cm"),

      # No axes
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank()
    )
}

# Create poster map
map_poster <- ggplot() +
  geom_sf(
    data = districts,
    aes(fill = vulnerability_index),
    color = "gray30",
    linewidth = 0.6
  ) +

  # Bold color scheme
  scale_fill_viridis_c(
    option = "rocket",
    name = "Heat\nVulnerability\nIndex",
    guide = guide_colorbar(
      barwidth = 2,
      barheight = 15,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +

  # Large scale bar
  annotation_scale(
    location = "br",
    width_hint = 0.3,
    text_cex = 1.5,
    style = "ticks",
    line_width = 1.5
  ) +

  # Large north arrow
  annotation_north_arrow(
    location = "tl",
    which_north = "true",
    pad_x = unit(0.5, "in"),
    pad_y = unit(0.5, "in"),
    style = north_arrow_fancy_orienteering(
      fill = c("gray20", "white"),
      line_col = "gray20",
      text_size = 16
    )
  ) +

  theme_poster(base_size = 22) +

  labs(
    title = "Spatial Distribution of Heat Vulnerability",
    subtitle = "Johannesburg Metropolitan Area (2021)",
    caption = "Greater Capital Region Observatory | n = 58,616 households | Poster ID: P-123"
  )

# Export for poster (large format)
ggsave(
  "figures/poster_map.svg",
  plot = map_poster,
  width = 24,
  height = 18,
  units = "in",
  dpi = 300
)
```

---

## Style 5: Social Media

### Characteristics
- Square or vertical format
- Instant visual impact
- Minimal text
- Bold, saturated colors
- Mobile-optimized
- Shareable

### Implementation

```r
# Social media theme
theme_social <- function(base_size = 16, base_family = "Arial") {
  theme_void(base_size = base_size, base_family = base_family) +
    theme(
      # Gradient background
      plot.background = element_rect(
        fill = "white",
        color = NA
      ),

      # Centered, punchy text
      plot.title = element_text(
        face = "bold",
        size = base_size + 8,
        hjust = 0.5,
        margin = margin(t = 20, b = 10),
        lineheight = 1.1
      ),
      plot.subtitle = element_text(
        size = base_size + 2,
        hjust = 0.5,
        margin = margin(b = 20)
      ),
      plot.caption = element_text(
        size = base_size - 2,
        hjust = 0.5,
        margin = margin(t = 10, b = 10),
        color = "gray40"
      ),
      plot.margin = margin(15, 15, 15, 15),

      # Compact legend
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size),
      legend.key.width = unit(3, "cm"),
      legend.key.height = unit(0.5, "cm")
    )
}

# Create social media graphic
map_social <- ggplot() +
  geom_sf(
    data = districts,
    aes(fill = vulnerability_index),
    color = "white",
    linewidth = 0.3
  ) +

  # Eye-catching gradient
  scale_fill_gradientn(
    colors = c("#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"),
    name = NULL,
    guide = guide_colorbar(
      barwidth = 15,
      barheight = 0.5
    )
  ) +

  theme_social(base_size = 16) +

  # Punchy message
  labs(
    title = "Heat Vulnerability\nAcross Johannesburg",
    subtitle = "58,000+ households surveyed",
    caption = "@YourHandle | #ClimateHealth"
  )

# Export for Instagram (square)
ggsave(
  "figures/social_instagram.png",
  plot = map_social,
  width = 1080,
  height = 1080,
  units = "px",
  dpi = 96
)

# Export for Twitter (2:1 ratio)
ggsave(
  "figures/social_twitter.png",
  plot = map_social,
  width = 1200,
  height = 600,
  units = "px",
  dpi = 96
)
```

---

## Color Theory for Maps

### Choosing the Right Palette

#### Sequential (Low to High)

```r
# For continuous variables with one direction
# Examples: temperature, population density, risk

# Viridis (blue → green → yellow)
scale_fill_viridis_c(option = "viridis")

# Mako (dark blue → light blue)
scale_fill_viridis_c(option = "mako")

# Rocket (black → red → white)
scale_fill_viridis_c(option = "rocket")

# Single-hue (light → dark)
scale_fill_distiller(palette = "Blues", direction = 1)
```

#### Diverging (Negative to Positive)

```r
# For variables with meaningful midpoint
# Examples: temperature anomaly, change from baseline

# Blue-white-red
scale_fill_distiller(palette = "RdBu", direction = -1)

# Purple-white-orange
scale_fill_distiller(palette = "PuOr", direction = -1)

# Scico roma (blue-white-red, perceptually uniform)
scale_fill_scico(palette = "roma", midpoint = 0)
```

#### Categorical (Distinct Groups)

```r
# For discrete categories with no inherent order
# Examples: land use types, administrative regions

# ColorBrewer Set2 (8 colors, colorblind-safe)
scale_fill_brewer(palette = "Set2")

# ColorBrewer Dark2 (8 colors, darker)
scale_fill_brewer(palette = "Dark2")

# Custom palette
scale_fill_manual(values = c(
  "Forest" = "#2D6A4F",
  "Urban" = "#E63946",
  "Agriculture" = "#E9C46A",
  "Water" = "#1D3557"
))
```

### Colorblind-Safe Palettes

```r
# Test your palettes
library(colorblindr)

# Create map
map <- ggplot() + geom_sf(data = districts, aes(fill = category))

# View under different color vision deficiencies
cvd_grid(map)

# Recommended palettes:
# - Viridis family (all options)
# - ColorBrewer: Set2, Dark2, Paired
# - Scico: batlow, roma, berlin
```

---

## Typography Guidelines

### Font Selection

```r
# Scientific publications
theme_minimal(base_family = "Arial")          # Universal
theme_minimal(base_family = "Helvetica")      # Clean
theme_minimal(base_family = "Times New Roman") # Traditional

# Presentations
theme_minimal(base_family = "Arial Black")    # Bold, readable
theme_minimal(base_family = "Gill Sans")      # Modern

# Media
theme_minimal(base_family = "Arial")          # Safe choice
theme_minimal(base_family = "Georgia")        # Readable, elegant
```

### Text Size Hierarchy

```r
# Journal (base_size = 10)
plot.title:    12 pt (base + 2)
plot.subtitle: 10 pt (base)
plot.caption:   8 pt (base - 2)
legend.title:   9 pt (base - 1)
legend.text:    8 pt (base - 2)

# Presentation (base_size = 18)
plot.title:    26 pt (base + 8)
plot.subtitle: 20 pt (base + 2)
plot.caption:  14 pt (base - 4)
legend.title:  18 pt (base)
legend.text:   16 pt (base - 2)

# Poster (base_size = 22)
plot.title:    34 pt (base + 12)
plot.subtitle: 26 pt (base + 4)
plot.caption:  20 pt (base - 2)
legend.title:  24 pt (base + 2)
legend.text:   22 pt (base)
```

---

## Context-Specific Checklists

### Before Submitting to Journal

- [ ] Color palette is colorblind-safe
- [ ] Works in grayscale (test with `scale_fill_grey()`)
- [ ] Font size ≥10pt for readability
- [ ] All text is editable (fonts embedded in SVG)
- [ ] Scale bar with correct units
- [ ] North arrow (unless global map)
- [ ] Complete metadata in caption
- [ ] CRS documented
- [ ] Data sources cited
- [ ] Figure fits column width (3.5" or 7")
- [ ] File format meets requirements (usually PDF/SVG)
- [ ] Resolution ≥300 DPI

### Before Conference Presentation

- [ ] Text readable from 20+ feet
- [ ] High contrast (test on projector)
- [ ] Simplified details (no visual clutter)
- [ ] Bold colors
- [ ] Key takeaway is obvious
- [ ] Aspect ratio matches slides (16:9 or 4:3)
- [ ] Works on light and dark backgrounds
- [ ] Animation/interaction tested (if applicable)

### Before Media Release

- [ ] No technical jargon
- [ ] Title grabs attention
- [ ] Context is self-explanatory
- [ ] Human-interest angle clear
- [ ] High-quality image (300 DPI PNG)
- [ ] Attribution clear
- [ ] Contact information included
- [ ] Mobile-friendly format
- [ ] Social media versions created (square, 2:1)

---

## Advanced: Branded Map Templates

### Create Organization-Specific Style

```r
# Define brand colors
brand_colors <- list(
  primary = "#1D3557",
  secondary = "#457B9D",
  accent = "#E63946",
  neutral = "#F1FAEE"
)

# Create branded theme
theme_organization <- function(base_size = 11, base_family = "Arial") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Brand colors
      plot.background = element_rect(
        fill = brand_colors$neutral,
        color = NA
      ),
      panel.background = element_rect(
        fill = "white",
        color = brand_colors$primary,
        linewidth = 1
      ),

      # Typography
      plot.title = element_text(
        face = "bold",
        color = brand_colors$primary,
        size = base_size + 4
      ),
      plot.subtitle = element_text(
        color = brand_colors$secondary,
        size = base_size + 1
      ),

      # Logo placement area (top-right)
      plot.margin = margin(20, 80, 20, 20)  # Extra space for logo
    )
}

# Add logo to map
library(magick)
library(cowplot)

# Create map
map <- ggplot() + geom_sf(data = districts) + theme_organization()

# Load logo
logo <- image_read("assets/logo.png") %>%
  image_scale("150")  # Resize to 150px wide

# Combine
map_with_logo <- ggdraw() +
  draw_plot(map) +
  draw_image(
    logo,
    x = 0.85, y = 0.90,      # Top-right position
    width = 0.12, height = 0.08
  )

# Save
save_plot(
  "figures/branded_map.svg",
  plot = map_with_logo,
  base_width = 8,
  base_height = 6
)
```

---

## Resources

### Color Palette Tools
- ColorBrewer: https://colorbrewer2.org/
- Viridis: https://sjmgarnier.github.io/viridis/
- Scico: https://github.com/thomasp85/scico
- Coolors: https://coolors.co/ (palette generator)
- Colorblind simulator: https://www.color-blindness.com/coblis-color-blindness-simulator/

### Design Inspiration
- New York Times Graphics: https://www.nytimes.com/spotlight/graphics
- The Pudding: https://pudding.cool/
- Flowing Data: https://flowingdata.com/
- Washington Post Graphics: https://www.washingtonpost.com/graphics/

### Style Guides
- Nature: https://www.nature.com/nature/for-authors/final-submission
- Science: https://www.science.org/content/page/instructions-preparing-initial-manuscript
- PLOS: https://journals.plos.org/plosone/s/figures
- BBC News Style Guide: https://www.bbc.co.uk/programmes/articles/

---

**Last Updated**: 2025-01-30
**Version**: 1.0
**Author**: Claude Code Expert System
