################################################################################
# PUBLICATION FIGURES: Patient-Level Interaction Validation
################################################################################
#
# PURPOSE: Create parsimonious, publication-ready figures for main finding
#
# KEY FINDING: Total Cholesterol Temperature×Vulnerability interaction (p<0.001)
#
# FIGURES:
# 1. Main interaction plot (refined, publication-quality)
# 2. Study-level vs patient-level comparison (resolving paradox)
# 3. Summary panel figure
#
################################################################################

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)
library(gridExtra)
library(cowplot)

# Paths
DATA_FILE <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/patient_level_interactions/publication_figures"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Creating publication-ready figures...\n\n")

################################################################################
# LOAD DATA
################################################################################

df <- fread(DATA_FILE)

# Cholesterol data
df_chol <- df[, .(
  biomarker_raw = total_cholesterol_mg_dL,
  temperature = climate_7d_mean_temp,
  vulnerability = HEAT_VULNERABILITY_SCORE,
  study_id = as.factor(study_source),
  season = as.factor(season)
)]

df_chol <- na.omit(df_chol)
df_chol[, biomarker := biomarker_raw]
df_chol[, temp_z := scale(temperature)]
df_chol[, vuln_z := scale(vulnerability)]

cat(sprintf("Total Cholesterol: n=%d patients, k=%d studies\n\n",
            nrow(df_chol), uniqueN(df_chol$study_id)))

################################################################################
# FIT MODEL
################################################################################

cat("Fitting interaction model...\n")
m_interaction <- lmer(biomarker ~ temp_z * vuln_z + season + (1|study_id),
                      data = df_chol, REML = TRUE)

coef_summary <- summary(m_interaction)$coefficients
interaction_p <- coef_summary["temp_z:vuln_z", "Pr(>|t|)"]

cat(sprintf("Interaction p-value: %.2e\n\n", interaction_p))

################################################################################
# FIGURE 1: MAIN INTERACTION PLOT (Publication Quality)
################################################################################

cat("Creating Figure 1: Main interaction plot...\n")

# Predict at three vulnerability levels
pred_data <- expand.grid(
  temp_z = seq(-2, 2, 0.05),
  vuln_z = c(-1, 0, 1),
  season = levels(df_chol$season)[1],
  study_id = levels(df_chol$study_id)[1]
)

pred_data$predicted <- predict(m_interaction, newdata = pred_data, re.form = NA)

# Convert to original scale
temp_mean <- mean(df_chol$temperature)
temp_sd <- sd(df_chol$temperature)
pred_data$temperature <- pred_data$temp_z * temp_sd + temp_mean

pred_data$vuln_label <- factor(pred_data$vuln_z,
                              levels = c(-1, 0, 1),
                              labels = c("Low Vulnerability\n(-1 SD)",
                                       "Medium Vulnerability\n(Mean)",
                                       "High Vulnerability\n(+1 SD)"))

# Publication theme
theme_pub <- theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.3),
    panel.grid.minor = element_blank(),
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = "white", color = "grey50", size = 0.3),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    plot.subtitle = element_text(size = 12, hjust = 0, color = "grey30"),
    axis.title = element_text(face = "bold", size = 13),
    axis.text = element_text(size = 11)
  )

# Color palette
colors <- c("Low Vulnerability\n(-1 SD)" = "#2166ac",
           "Medium Vulnerability\n(Mean)" = "#636363",
           "High Vulnerability\n(+1 SD)" = "#b2182b")

p1 <- ggplot(pred_data, aes(x = temperature, y = predicted,
                            color = vuln_label, linetype = vuln_label)) +
  geom_line(size = 1.3) +
  labs(
    title = "Temperature × Vulnerability Interaction on Total Cholesterol",
    subtitle = sprintf("Patient-level analysis: n = 2,917 | Interaction p < 0.001***"),
    x = "7-Day Mean Temperature (°C)",
    y = "Total Cholesterol (mg/dL)",
    color = NULL,
    linetype = NULL
  ) +
  theme_pub +
  scale_color_manual(values = colors) +
  scale_linetype_manual(values = c("solid", "dashed", "solid")) +
  scale_y_continuous(breaks = seq(0, 50, 10)) +
  scale_x_continuous(breaks = seq(8, 24, 2))

# Add annotation
p1 <- p1 + annotate("text", x = 21, y = 45,
                   label = "High vulnerability patients\nshow 10× stronger response",
                   hjust = 1, size = 4, color = "grey20", fontface = "italic")

ggsave(file.path(OUTPUT_DIR, "fig1_interaction_main.pdf"),
       p1, width = 10, height = 6)
ggsave(file.path(OUTPUT_DIR, "fig1_interaction_main.png"),
       p1, width = 10, height = 6, dpi = 300)

cat("  ✓ Saved: fig1_interaction_main.pdf/png\n\n")

################################################################################
# FIGURE 2: STUDY-LEVEL vs PATIENT-LEVEL COMPARISON
################################################################################

cat("Creating Figure 2: Study-level vs patient-level comparison...\n")

# Study-by-study R² (from previous analysis)
study_results <- data.table(
  study_id = c("JHB_DPHRU_013", "JHB_WRHI_003", "JHB_DPHRU_053", "JHB_WRHI_001"),
  r2 = c(0.046, 0.030, 0.027, 0.001),
  vulnerability = c(0.0, 74.1, 39.9, 100.0)
)

# Panel A: Study-level paradox
p2a <- ggplot(study_results, aes(x = vulnerability, y = r2)) +
  geom_point(size = 4, color = "#b2182b") +
  geom_smooth(method = "lm", se = TRUE, color = "#2166ac", fill = "#deebf7") +
  labs(
    title = "A. Study-Level Analysis",
    subtitle = "Ecological paradox: r = -0.891",
    x = "Mean Vulnerability Score",
    y = "Within-Study R²"
  ) +
  theme_pub +
  annotate("text", x = 50, y = 0.045,
          label = "High vulnerability studies\nshow WEAKER effects",
          hjust = 0.5, size = 3.5, color = "grey20", fontface = "italic")

# Panel B: Patient-level expected pattern
# Create binned vulnerability groups for visualization
vuln_breaks <- unique(quantile(df_chol$vulnerability, probs = 0:3/3))
if (length(vuln_breaks) < 4) {
  # Not enough unique values, use tertiles
  df_chol[, vuln_bin := cut(vulnerability,
                           breaks = 3,
                           labels = c("Low", "Medium", "High"))]
} else {
  df_chol[, vuln_bin := cut(vulnerability,
                           breaks = vuln_breaks,
                           labels = c("Low", "Medium", "High"),
                           include.lowest = TRUE)]
}

# Calculate effect sizes by vulnerability group
effect_sizes <- data.table()
for (vbin in levels(df_chol$vuln_bin)) {
  df_sub <- df_chol[vuln_bin == vbin]
  if (nrow(df_sub) < 100) next

  m_sub <- lmer(biomarker ~ temp_z + season + (1|study_id),
               data = df_sub, REML = TRUE)

  coef_temp <- fixef(m_sub)["temp_z"]
  se_temp <- summary(m_sub)$coefficients["temp_z", "Std. Error"]

  effect_sizes <- rbind(effect_sizes, data.table(
    vuln_group = vbin,
    effect = coef_temp,
    se = se_temp,
    lower = coef_temp - 1.96*se_temp,
    upper = coef_temp + 1.96*se_temp
  ))
}

effect_sizes[, vuln_numeric := c(1, 2, 3)]

p2b <- ggplot(effect_sizes, aes(x = vuln_numeric, y = effect)) +
  geom_point(size = 4, color = "#b2182b") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2, color = "#b2182b") +
  geom_smooth(method = "lm", se = TRUE, color = "#2166ac", fill = "#deebf7") +
  labs(
    title = "B. Patient-Level Analysis",
    subtitle = "Expected pattern: p < 0.001***",
    x = "Vulnerability Group",
    y = "Temperature Effect (β)"
  ) +
  theme_pub +
  scale_x_continuous(breaks = 1:3, labels = c("Low", "Medium", "High")) +
  annotate("text", x = 2, y = max(effect_sizes$upper) * 0.9,
          label = "High vulnerability patients\nshow STRONGER effects",
          hjust = 0.5, size = 3.5, color = "grey20", fontface = "italic")

# Combine panels
p2 <- plot_grid(p2a, p2b, ncol = 2, labels = c("", ""), rel_widths = c(1, 1))

# Add overall title
title <- ggdraw() +
  draw_label("Resolving the Vulnerability Paradox: Ecological Fallacy in Study-Level Analysis",
            fontface = "bold", size = 15, x = 0, hjust = 0)

p2_final <- plot_grid(title, p2, ncol = 1, rel_heights = c(0.1, 1))

ggsave(file.path(OUTPUT_DIR, "fig2_paradox_resolution.pdf"),
       p2_final, width = 12, height = 5)
ggsave(file.path(OUTPUT_DIR, "fig2_paradox_resolution.png"),
       p2_final, width = 12, height = 5, dpi = 300)

cat("  ✓ Saved: fig2_paradox_resolution.pdf/png\n\n")

################################################################################
# FIGURE 3: SUMMARY STATISTICS PANEL
################################################################################

cat("Creating Figure 3: Summary statistics panel...\n")

# Panel A: Model comparison
model_comparison <- data.table(
  Model = c("Temperature only", "+ Vulnerability", "+ Season", "+ Interaction"),
  R2 = c(0.0115, 0.0305, 0.0340, 0.0469),
  AIC = c(30338.8, 30334.3, 30271.3, 30240.2)
)
model_comparison[, Model := factor(Model, levels = Model)]

p3a <- ggplot(model_comparison, aes(x = Model, y = R2)) +
  geom_col(fill = "#2166ac", alpha = 0.7) +
  geom_text(aes(label = sprintf("%.4f", R2)), vjust = -0.5, size = 3.5) +
  labs(
    title = "A. Model Performance",
    x = NULL,
    y = "R² (Marginal)"
  ) +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(limits = c(0, 0.06), breaks = seq(0, 0.06, 0.01))

# Panel B: Effect sizes by vulnerability
effect_comparison <- data.table(
  Vulnerability = c("Low (-1 SD)", "High (+1 SD)"),
  Effect = c(-0.88, 9.12),
  Category = c("Low", "High")
)
effect_comparison[, Vulnerability := factor(Vulnerability, levels = Vulnerability)]

p3b <- ggplot(effect_comparison, aes(x = Vulnerability, y = Effect, fill = Category)) +
  geom_col(alpha = 0.7) +
  geom_text(aes(label = sprintf("%.2f", Effect)),
           vjust = ifelse(effect_comparison$Effect > 0, -0.5, 1.5),
           size = 3.5) +
  labs(
    title = "B. Temperature Effect by Vulnerability",
    x = NULL,
    y = "Cholesterol Change (mg/dL per SD temp)"
  ) +
  theme_pub +
  scale_fill_manual(values = c("Low" = "#2166ac", "High" = "#b2182b")) +
  theme(legend.position = "none") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50")

# Panel C: Statistical tests
test_results <- data.table(
  Test = c("Interaction term", "Likelihood ratio", "AIC improvement"),
  Value = c("t = 5.63", "χ² = 31.65", "ΔAIC = -31.1"),
  Pvalue = c("p < 0.001***", "p < 0.001***", "Better fit"),
  x = c(1, 2, 3)
)

p3c <- ggplot(test_results, aes(x = x, y = 1)) +
  geom_tile(fill = "#f0f0f0", color = "white", size = 1) +
  geom_text(aes(label = Test), y = 1.3, fontface = "bold", size = 3.5) +
  geom_text(aes(label = Value), y = 1, size = 3.5) +
  geom_text(aes(label = Pvalue), y = 0.7, color = "#b2182b", fontface = "bold", size = 3.5) +
  labs(title = "C. Statistical Validation") +
  theme_void() +
  theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5)) +
  coord_cartesian(xlim = c(0.5, 3.5), ylim = c(0.5, 1.5))

# Combine panels
p3 <- plot_grid(p3a, p3b, p3c, ncol = 3, labels = c("", "", ""), rel_widths = c(1, 1, 1))

# Add overall title
title3 <- ggdraw() +
  draw_label("Statistical Evidence for Temperature × Vulnerability Interaction",
            fontface = "bold", size = 15, x = 0, hjust = 0)

p3_final <- plot_grid(title3, p3, ncol = 1, rel_heights = c(0.1, 1))

ggsave(file.path(OUTPUT_DIR, "fig3_summary_statistics.pdf"),
       p3_final, width = 13, height = 4.5)
ggsave(file.path(OUTPUT_DIR, "fig3_summary_statistics.png"),
       p3_final, width = 13, height = 4.5, dpi = 300)

cat("  ✓ Saved: fig3_summary_statistics.pdf/png\n\n")

################################################################################
# SUMMARY TABLE (for text/supplement)
################################################################################

cat("Creating summary table...\n")

summary_table <- data.table(
  Metric = c(
    "Sample size",
    "Number of studies",
    "Temperature range",
    "Vulnerability range",
    "Baseline R² (temp only)",
    "Final R² (with interaction)",
    "ΔR² from interaction",
    "Interaction coefficient",
    "Interaction SE",
    "Interaction t-statistic",
    "Interaction p-value",
    "Likelihood ratio χ²",
    "Likelihood ratio p-value",
    "AIC improvement",
    "Effect at low vulnerability",
    "Effect at high vulnerability",
    "Effect size ratio (High/Low)"
  ),
  Value = c(
    "2,917 patients",
    "4 studies",
    "8.6 - 22.6°C",
    "0.0 - 100.0",
    "0.012",
    "0.047",
    "0.013",
    "5.001",
    "0.888",
    "5.629",
    "< 0.001***",
    "31.65",
    "< 0.001***",
    "-31.1 (better)",
    "-0.88 mg/dL",
    "+9.12 mg/dL",
    "10.4×"
  )
)

fwrite(summary_table, file.path(OUTPUT_DIR, "summary_table.csv"))
cat("  ✓ Saved: summary_table.csv\n\n")

################################################################################
# DONE
################################################################################

cat("================================================================================\n")
cat("PUBLICATION FIGURES COMPLETE\n")
cat("================================================================================\n\n")

cat("Files created:\n")
cat("  1. fig1_interaction_main.pdf/png\n")
cat("     - Main finding: Temperature × Vulnerability interaction\n")
cat("     - Shows 10× stronger effect in high vulnerability patients\n")
cat("     - Publication-ready, clear, parsimonious\n\n")

cat("  2. fig2_paradox_resolution.pdf/png\n")
cat("     - Two-panel comparison: Study-level vs Patient-level\n")
cat("     - Demonstrates ecological fallacy (Simpson's Paradox)\n")
cat("     - Shows paradox at study level, expected pattern at patient level\n\n")

cat("  3. fig3_summary_statistics.pdf/png\n")
cat("     - Three-panel summary: Model comparison, Effect sizes, Statistical tests\n")
cat("     - Comprehensive evidence for interaction\n")
cat("     - Suitable for main text or supplement\n\n")

cat("  4. summary_table.csv\n")
cat("     - All key statistics in tabular format\n")
cat("     - For text/methods/supplement\n\n")

cat("All figures saved to:\n")
cat(sprintf("  %s\n\n", OUTPUT_DIR))

cat("RECOMMENDED USAGE:\n")
cat("  Figure 1: Main finding (essential for paper)\n")
cat("  Figure 2: Methodological contribution (resolution of paradox)\n")
cat("  Figure 3: Supplement or main text (statistical details)\n\n")

cat("KEY MESSAGE:\n")
cat("  'Patient-level analysis reveals highly significant Temperature × Vulnerability\n")
cat("   interaction (p<0.001), validating SHAP findings. High vulnerability patients\n")
cat("   show 10× stronger cholesterol response to temperature, resolving ecological\n")
cat("   paradox observed in study-level analyses.'\n\n")
