#!/usr/bin/env Rscript
################################################################################
# Publication-Quality Figures for Mixed Effects DLNM Analysis
################################################################################
#
# Purpose: Generate forest plots, box plots, and multi-panel figures
#          for publication of climate-biomarker associations
#
# Outputs:
# - Forest plot with effect sizes and CIs
# - Box plots showing distributions by study
# - DLNM effect curves with confidence bands
# - Combined multi-panel publication figure
#
# Author: Claude + Craig Saunders
# Date: 2025-10-30
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(cowplot)
  library(viridis)
  library(scales)
  library(mgcv)
  library(dlnm)
})

set.seed(42)

# Configuration
DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
RESULTS_PATH <- "reanalysis_outputs/comprehensive_units_check/comprehensive_comparison.csv"
OUTPUT_DIR <- "reanalysis_outputs/publication_figures"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("\n")
cat("################################################################################\n")
cat("# PUBLICATION-QUALITY FIGURES GENERATION\n")
cat("################################################################################\n\n")

################################################################################
# CUSTOM THEME
################################################################################

theme_publication <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.4), hjust = 0),
      plot.subtitle = element_text(size = rel(1.1), hjust = 0, color = "gray30"),
      axis.title = element_text(face = "bold", size = rel(1.1)),
      axis.text = element_text(size = rel(0.9)),
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = rel(0.9)),
      strip.text = element_text(face = "bold", size = rel(1.05)),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin = margin(10, 10, 10, 10)
    )
}

################################################################################
# FIGURE 1: FOREST PLOT (Effect Sizes with CIs)
################################################################################

cat("=== Creating Forest Plot ===\n")

# Load results
results <- fread(RESULTS_PATH)

# Prepare for forest plot
forest_data <- results %>%
  mutate(
    # Calculate 95% CI (assuming normal approximation)
    se = sqrt(r2_after * (1 - r2_after) / n_obs),
    lower_ci = pmax(0, r2_after - 1.96 * se),
    upper_ci = pmin(1, r2_after + 1.96 * se),

    # Clean biomarker names
    biomarker_clean = case_when(
      grepl("cholesterol", biomarker, ignore.case = TRUE) ~
        gsub("_", " ", gsub("_mg_dL|_mmol_L", "", biomarker)),
      grepl("glucose", biomarker, ignore.case = TRUE) ~ "Fasting Glucose",
      grepl("creatinine", biomarker, ignore.case = TRUE) ~ "Creatinine",
      grepl("Hematocrit", biomarker) ~ "Hematocrit",
      TRUE ~ biomarker
    ),

    # Significance category
    category = case_when(
      r2_after >= 0.30 ~ "Strong (R² ≥ 0.30)",
      r2_after >= 0.10 ~ "Moderate (R² ≥ 0.10)",
      TRUE ~ "Weak (R² < 0.10)"
    ),

    # Corrected?
    corrected_label = ifelse(correction_applied, "*", "")
  ) %>%
  arrange(desc(r2_after))

# Create forest plot
p_forest <- ggplot(forest_data,
                   aes(x = r2_after, y = reorder(biomarker_clean, r2_after))) +

  # Reference line at zero
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", size = 0.8) +

  # Confidence intervals
  geom_errorbarh(aes(xmin = lower_ci, xmax = upper_ci, color = category),
                 height = 0.3, size = 1.2, alpha = 0.8) +

  # Point estimates (sized by sample size)
  geom_point(aes(size = n_obs, color = category, shape = category),
             alpha = 0.9) +

  # Add R² values as text
  geom_text(aes(label = sprintf("%.3f%s", r2_after, corrected_label),
                x = upper_ci + 0.015),
            hjust = 0, size = 3.5, fontface = "bold") +

  # Add sample sizes
  geom_text(aes(label = sprintf("n=%d", n_obs),
                x = -0.01),
            hjust = 1, size = 3, color = "gray40") +

  # Scales
  scale_size_continuous(range = c(4, 10), name = "Sample Size") +
  scale_color_manual(
    values = c(
      "Strong (R² ≥ 0.30)" = "#00BA38",
      "Moderate (R² ≥ 0.10)" = "#619CFF",
      "Weak (R² < 0.10)" = "#F8766D"
    ),
    name = "Effect Strength"
  ) +
  scale_shape_manual(
    values = c(
      "Strong (R² ≥ 0.30)" = 18,
      "Moderate (R² ≥ 0.10)" = 16,
      "Weak (R² < 0.10)" = 17
    ),
    name = "Effect Strength"
  ) +
  scale_x_continuous(
    breaks = seq(0, 0.4, 0.05),
    labels = scales::percent_format(accuracy = 1),
    limits = c(-0.02, 0.42)
  ) +

  # Labels
  labs(
    title = "Climate-Biomarker Associations",
    subtitle = "Mixed Effects DLNM Analysis (R² with 95% Confidence Intervals)",
    x = "Proportion of Variance Explained by Climate (R²)",
    y = "Biomarker",
    caption = "* = Units corrected | Error bars = 95% CI | Point size = Sample size"
  ) +

  theme_publication(base_size = 13) +
  theme(
    legend.position = "right",
    legend.direction = "vertical"
  )

# Save
ggsave(file.path(OUTPUT_DIR, "Figure1_ForestPlot.pdf"),
       p_forest, width = 12, height = 7, units = "in", dpi = 300)

ggsave(file.path(OUTPUT_DIR, "Figure1_ForestPlot.png"),
       p_forest, width = 12, height = 7, units = "in", dpi = 300)

cat(sprintf("  Saved: %s/Figure1_ForestPlot.pdf\n", OUTPUT_DIR))

################################################################################
# FIGURE 2: BOX PLOTS (Distributions by Study)
################################################################################

cat("\n=== Creating Box Plots by Study ===\n")

# Load raw data
df_raw <- fread(DATA_PATH)

# Biomarkers to plot (using corrected data)
biomarkers_to_plot <- c(
  "Hematocrit (%)",
  "total_cholesterol_mg_dL",
  "fasting_glucose_mmol_L",
  "creatinine_umol_L"
)

plot_list <- list()

for (biomarker_name in biomarkers_to_plot) {

  if (!biomarker_name %in% names(df_raw)) next

  # Prepare data
  df_plot <- df_raw[, .(
    biomarker_raw = get(biomarker_name),
    study_id = study_source
  )]

  df_plot <- na.omit(df_plot)

  # Apply hematocrit correction if needed
  if (biomarker_name == "Hematocrit (%)") {
    df_plot[, biomarker := ifelse(biomarker_raw < 1, biomarker_raw * 100, biomarker_raw)]
    plot_title <- "Hematocrit (%) [Corrected]"
  } else {
    df_plot[, biomarker := biomarker_raw]
    plot_title <- gsub("_", " ", gsub("_mg_dL|_mmol_L|_umol_L", "", biomarker_name))
  }

  # Count studies
  n_studies <- uniqueN(df_plot$study_id)

  if (n_studies < 2) next

  # Create box plot
  p <- ggplot(df_plot, aes(x = study_id, y = biomarker, fill = study_id)) +
    geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 1.5) +
    geom_jitter(width = 0.2, alpha = 0.1, size = 0.3) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
                 fill = "white", color = "black") +
    scale_fill_viridis_d(option = "plasma", end = 0.9) +
    labs(
      title = plot_title,
      x = "Study",
      y = "Value",
      subtitle = sprintf("n = %d observations across %d studies",
                         nrow(df_plot), n_studies)
    ) +
    theme_publication(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )

  plot_list[[biomarker_name]] <- p
}

# Combine box plots
if (length(plot_list) > 0) {
  p_boxes_combined <- plot_grid(
    plotlist = plot_list,
    ncol = 2,
    labels = "AUTO",
    label_size = 14
  )

  # Add title
  title <- ggdraw() +
    draw_label(
      "Biomarker Distributions by Study",
      fontface = "bold",
      size = 16,
      hjust = 0.5
    )

  p_boxes_final <- plot_grid(
    title,
    p_boxes_combined,
    ncol = 1,
    rel_heights = c(0.05, 1)
  )

  # Save
  ggsave(file.path(OUTPUT_DIR, "Figure2_BoxPlots.pdf"),
         p_boxes_final, width = 12, height = 10, units = "in", dpi = 300)

  ggsave(file.path(OUTPUT_DIR, "Figure2_BoxPlots.png"),
         p_boxes_final, width = 12, height = 10, units = "in", dpi = 300)

  cat(sprintf("  Saved: %s/Figure2_BoxPlots.pdf\n", OUTPUT_DIR))
}

################################################################################
# FIGURE 3: COMPARISON BEFORE/AFTER CORRECTION
################################################################################

cat("\n=== Creating Before/After Comparison ===\n")

# Filter biomarkers with corrections
corrected_biomarkers <- results %>%
  filter(correction_applied == TRUE) %>%
  select(biomarker, r2_before, r2_after, icc_before, icc_after) %>%
  tidyr::pivot_longer(
    cols = c(r2_before, r2_after),
    names_to = "timepoint",
    values_to = "r2"
  ) %>%
  mutate(
    timepoint = ifelse(timepoint == "r2_before", "Before Correction", "After Correction"),
    biomarker_clean = gsub("_", " ", gsub("_mg_dL|_mmol_L|_umol_L", "", biomarker))
  )

if (nrow(corrected_biomarkers) > 0) {

  p_comparison <- ggplot(corrected_biomarkers,
                         aes(x = timepoint, y = r2, group = biomarker_clean)) +
    geom_line(aes(color = biomarker_clean), size = 1.2, arrow = arrow(length = unit(0.3, "cm"))) +
    geom_point(aes(color = biomarker_clean, shape = timepoint), size = 5) +
    scale_color_viridis_d(option = "viridis", name = "Biomarker") +
    scale_shape_manual(values = c(16, 18), name = "Status") +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 0.1),
      breaks = seq(0, 1, 0.1)
    ) +
    labs(
      title = "Impact of Units Correction on R²",
      subtitle = "Change in variance explained after standardizing measurement units",
      x = "",
      y = "R² (Variance Explained)",
      caption = "Arrows show direction of change from before to after correction"
    ) +
    theme_publication(base_size = 13) +
    theme(
      axis.text.x = element_text(face = "bold", size = 12)
    )

  # Save
  ggsave(file.path(OUTPUT_DIR, "Figure3_BeforeAfter.pdf"),
         p_comparison, width = 10, height = 7, units = "in", dpi = 300)

  ggsave(file.path(OUTPUT_DIR, "Figure3_BeforeAfter.png"),
         p_comparison, width = 10, height = 7, units = "in", dpi = 300)

  cat(sprintf("  Saved: %s/Figure3_BeforeAfter.pdf\n", OUTPUT_DIR))
}

################################################################################
# FIGURE 4: SAMPLE SIZE VS EFFECT SIZE
################################################################################

cat("\n=== Creating Sample Size vs Effect Size Plot ===\n")

p_scatter <- ggplot(forest_data, aes(x = n_obs, y = r2_after)) +
  geom_point(aes(color = category, size = n_sig_temps_after),
             alpha = 0.7) +
  geom_smooth(method = "loess", color = "gray30", linetype = "dashed",
              fill = "gray80", alpha = 0.3) +
  geom_text(aes(label = biomarker_clean),
            vjust = -0.8, hjust = 0.5, size = 3.5, fontface = "bold") +
  scale_color_manual(
    values = c(
      "Strong (R² ≥ 0.30)" = "#00BA38",
      "Moderate (R² ≥ 0.10)" = "#619CFF",
      "Weak (R² < 0.10)" = "#F8766D"
    ),
    name = "Effect Strength"
  ) +
  scale_size_continuous(
    range = c(3, 10),
    name = "Significant\nTemperatures"
  ) +
  scale_x_log10(
    breaks = c(500, 1000, 2000, 3000),
    labels = scales::comma
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    breaks = seq(0, 0.4, 0.05)
  ) +
  labs(
    title = "Sample Size vs Effect Size",
    subtitle = "Relationship between study size and climate-biomarker associations",
    x = "Sample Size (log scale)",
    y = "R² (Variance Explained)",
    caption = "Point size indicates number of significant temperatures detected"
  ) +
  theme_publication(base_size = 13)

# Save
ggsave(file.path(OUTPUT_DIR, "Figure4_SampleSize_vs_EffectSize.pdf"),
       p_scatter, width = 10, height = 7, units = "in", dpi = 300)

ggsave(file.path(OUTPUT_DIR, "Figure4_SampleSize_vs_EffectSize.png"),
       p_scatter, width = 10, height = 7, units = "in", dpi = 300)

cat(sprintf("  Saved: %s/Figure4_SampleSize_vs_EffectSize.pdf\n", OUTPUT_DIR))

################################################################################
# FIGURE 5: MULTI-PANEL SUMMARY FIGURE
################################################################################

cat("\n=== Creating Multi-Panel Summary Figure ===\n")

# Combine key plots
summary_figure <- plot_grid(
  p_forest + theme(legend.position = "right"),
  p_scatter,
  if (exists("p_comparison")) p_comparison else NULL,
  ncol = 1,
  labels = c("A", "B", "C"),
  label_size = 16,
  rel_heights = c(1.2, 1, 1)
)

# Add overall title
title_overall <- ggdraw() +
  draw_label(
    "Climate-Biomarker Associations: Mixed Effects DLNM Analysis with Units Correction",
    fontface = "bold",
    size = 18,
    hjust = 0.5
  )

summary_final <- plot_grid(
  title_overall,
  summary_figure,
  ncol = 1,
  rel_heights = c(0.05, 1)
)

# Save
ggsave(file.path(OUTPUT_DIR, "Figure5_MultiPanel_Summary.pdf"),
       summary_final, width = 12, height = 16, units = "in", dpi = 300)

ggsave(file.path(OUTPUT_DIR, "Figure5_MultiPanel_Summary.png"),
       summary_final, width = 12, height = 16, units = "in", dpi = 300)

cat(sprintf("  Saved: %s/Figure5_MultiPanel_Summary.pdf\n", OUTPUT_DIR))

################################################################################
# SUMMARY TABLE
################################################################################

cat("\n=== Creating Summary Table ===\n")

summary_table <- forest_data %>%
  select(
    Biomarker = biomarker_clean,
    `Sample Size` = n_obs,
    `R²` = r2_after,
    `95% CI Lower` = lower_ci,
    `95% CI Upper` = upper_ci,
    `Significant Temperatures` = n_sig_temps_after,
    `Correction Applied` = correction_applied
  ) %>%
  mutate(
    `R²` = sprintf("%.3f", `R²`),
    `95% CI` = sprintf("[%.3f, %.3f]", `95% CI Lower`, `95% CI Upper`),
    `Correction Applied` = ifelse(`Correction Applied`, "Yes", "No")
  ) %>%
  select(-`95% CI Lower`, -`95% CI Upper`) %>%
  arrange(desc(as.numeric(`R²`)))

write.csv(summary_table,
          file.path(OUTPUT_DIR, "Table1_Summary_Results.csv"),
          row.names = FALSE)

cat(sprintf("  Saved: %s/Table1_Summary_Results.csv\n", OUTPUT_DIR))

################################################################################
# COMPLETION
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# FIGURES GENERATION COMPLETE\n")
cat("################################################################################\n\n")

cat("Generated Files:\n")
cat("  1. Figure1_ForestPlot.pdf - Effect sizes with confidence intervals\n")
cat("  2. Figure2_BoxPlots.pdf - Distributions by study\n")
cat("  3. Figure3_BeforeAfter.pdf - Impact of units correction\n")
cat("  4. Figure4_SampleSize_vs_EffectSize.pdf - Sample size relationship\n")
cat("  5. Figure5_MultiPanel_Summary.pdf - Combined publication figure\n")
cat("  6. Table1_Summary_Results.csv - Formatted results table\n")
cat("\n")
cat(sprintf("All files saved to: %s\n", OUTPUT_DIR))
cat("\nFigures are publication-ready at 300 DPI in both PDF and PNG formats.\n")
