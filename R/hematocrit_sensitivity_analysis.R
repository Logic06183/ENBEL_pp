#!/usr/bin/env Rscript
################################################################################
# Hematocrit Sensitivity Analysis - Investigating High R²
################################################################################
#
# Purpose: Investigate why Hematocrit R² = 0.961 is so high
#          Test for data leakage, overfitting, and study artifacts
#
# Author: Claude + Craig Saunders
# Date: 2025-10-30
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)
  library(dlnm)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
})

set.seed(42)

# Configuration
DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/hematocrit_sensitivity"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("\n")
cat("################################################################################\n")
cat("# HEMATOCRIT SENSITIVITY ANALYSIS\n")
cat("################################################################################\n\n")

# Load and prepare data
cat("Loading data...\n")
df <- fread(DATA_PATH)

df_clean <- df[, .(
  biomarker = `Hematocrit (%)`,
  temperature = climate_7d_mean_temp,
  vulnerability = HEAT_VULNERABILITY_SCORE,
  study_id = as.factor(study_source),
  patient_id = paste0(study_source, "_", anonymous_patient_id),
  season = as.factor(season),
  date = as.Date(primary_date)
)]

df_clean <- na.omit(df_clean)

cat(sprintf("N observations: %d\n", nrow(df_clean)))
cat(sprintf("N studies: %d\n", uniqueN(df_clean$study_id)))
cat(sprintf("N patients: %d\n", uniqueN(df_clean$patient_id)))

################################################################################
# 1. DESCRIPTIVE STATISTICS BY STUDY
################################################################################

cat("\n=== 1. Descriptive Statistics by Study ===\n")

study_stats <- df_clean[, .(
  n = .N,
  mean_hematocrit = mean(biomarker),
  sd_hematocrit = sd(biomarker),
  min_hematocrit = min(biomarker),
  max_hematocrit = max(biomarker),
  mean_temp = mean(temperature),
  sd_temp = sd(temperature),
  min_temp = min(temperature),
  max_temp = max(temperature)
), by = study_id]

print(study_stats)

# Check if studies have very different baseline hematocrit
cat("\nBetween-study hematocrit differences:\n")
cat(sprintf("  Range of study means: %.2f - %.2f\n",
            min(study_stats$mean_hematocrit),
            max(study_stats$mean_hematocrit)))
cat(sprintf("  SD of study means: %.2f\n",
            sd(study_stats$mean_hematocrit)))

################################################################################
# 2. VARIANCE DECOMPOSITION
################################################################################

cat("\n=== 2. Variance Decomposition ===\n")

# Calculate within-study and between-study variance
df_clean[, study_mean_hematocrit := mean(biomarker), by = study_id]
df_clean[, deviation_from_study_mean := biomarker - study_mean_hematocrit]

within_study_var <- var(df_clean$deviation_from_study_mean)
between_study_var <- var(df_clean$study_mean_hematocrit)
total_var <- var(df_clean$biomarker)

cat(sprintf("Total variance: %.4f\n", total_var))
cat(sprintf("Within-study variance: %.4f (%.1f%%)\n",
            within_study_var,
            100 * within_study_var / total_var))
cat(sprintf("Between-study variance: %.4f (%.1f%%)\n",
            between_study_var,
            100 * between_study_var / total_var))

# Calculate ICC
icc <- between_study_var / total_var
cat(sprintf("\nIntraclass Correlation (ICC): %.3f\n", icc))
cat("  Interpretation: %.1f%% of variance is between-study\n", 100 * icc)

################################################################################
# 3. PROGRESSIVE MODEL COMPARISON
################################################################################

cat("\n=== 3. Progressive Model Comparison ===\n")

# Create crossbasis
cb_temp <- crossbasis(
  df_clean$temperature,
  lag = 14,
  argvar = list(fun = "ns", df = 3),
  arglag = list(fun = "ns", df = 3)
)

df_cb <- cbind(df_clean, as.data.table(cb_temp))

# Model A: Temperature only (no controls, no random effects)
cat("\nModel A: Temperature only (no controls)...\n")
m_a <- gam(biomarker ~ cb_temp, data = df_cb, method = "REML")

# Model B: Temperature + season (no vulnerability, no random effects)
cat("Model B: Temperature + season...\n")
m_b <- gam(biomarker ~ cb_temp + season, data = df_cb, method = "REML")

# Model C: Temperature + vulnerability (no season, no random effects)
cat("Model C: Temperature + vulnerability...\n")
m_c <- gam(biomarker ~ cb_temp + vulnerability, data = df_cb, method = "REML")

# Model D: Temperature + season + vulnerability (full controls, no random effects)
cat("Model D: Full controls (no random effects)...\n")
m_d <- gam(biomarker ~ cb_temp + season + vulnerability, data = df_cb, method = "REML")

# Model E: Only random study intercept (no predictors!)
cat("Model E: Random study intercept only (no predictors)...\n")
m_e <- gam(biomarker ~ s(study_id, bs = "re"), data = df_cb, method = "REML")

# Model F: Random study + season (no temperature!)
cat("Model F: Random study + season (no temperature)...\n")
m_f <- gam(biomarker ~ season + s(study_id, bs = "re"), data = df_cb, method = "REML")

# Model G: Full model with random intercept
cat("Model G: Full model with random intercept...\n")
m_g <- gam(biomarker ~ cb_temp + season + vulnerability + s(study_id, bs = "re"),
           data = df_cb, method = "REML")

# Model H: Full model with random slope (original best model)
cat("Model H: Full model with random slope...\n")
m_h <- gam(biomarker ~ cb_temp + season + vulnerability +
             s(study_id, bs = "re") + s(study_id, temperature, bs = "re"),
           data = df_cb, method = "REML")

# Compare all models
comparison <- data.table(
  model = c("A: Temp only", "B: Temp+Season", "C: Temp+Vuln",
            "D: Full controls", "E: Study RE only", "F: Study+Season",
            "G: Full+Study RE", "H: Full+Slope RE"),
  r2 = c(summary(m_a)$r.sq, summary(m_b)$r.sq, summary(m_c)$r.sq,
         summary(m_d)$r.sq, summary(m_e)$r.sq, summary(m_f)$r.sq,
         summary(m_g)$r.sq, summary(m_h)$r.sq),
  aic = c(AIC(m_a), AIC(m_b), AIC(m_c), AIC(m_d),
          AIC(m_e), AIC(m_f), AIC(m_g), AIC(m_h)),
  deviance_expl = c(summary(m_a)$dev.expl, summary(m_b)$dev.expl,
                    summary(m_c)$dev.expl, summary(m_d)$dev.expl,
                    summary(m_e)$dev.expl, summary(m_f)$dev.expl,
                    summary(m_g)$dev.expl, summary(m_h)$dev.expl) * 100
)

comparison[, r2_increase := r2 - r2[1]]
comparison[, aic_delta := aic - min(aic)]

cat("\nProgressive Model Comparison:\n")
print(comparison[order(aic)])

# Save comparison
fwrite(comparison, file.path(OUTPUT_DIR, "model_comparison.csv"))

################################################################################
# 4. CRITICAL INSIGHT: STUDY RANDOM EFFECT ALONE
################################################################################

cat("\n=== 4. CRITICAL TEST: Random Study Effect Alone ===\n")

cat(sprintf("\nModel E (Study RE only, no predictors): R² = %.3f\n", summary(m_e)$r.sq))
cat(sprintf("Model H (Full model): R² = %.3f\n", summary(m_h)$r.sq))
cat(sprintf("Difference: %.3f\n", summary(m_h)$r.sq - summary(m_e)$r.sq))

cat("\n!! KEY FINDING !!\n")
if (summary(m_e)$r.sq > 0.90) {
  cat("WARNING: Study random effects ALONE explain >90%% of variance!\n")
  cat("This suggests the high R² is driven by study-level differences,\n")
  cat("NOT by temperature or other predictors.\n")
} else {
  cat("Study random effects explain %.1f%% of variance.\n", summary(m_e)$r.sq * 100)
  cat("Temperature and other predictors add meaningful information.\n")
}

################################################################################
# 5. WITHIN-STUDY MODELS
################################################################################

cat("\n=== 5. Within-Study Models (Separate Analysis per Study) ===\n")

within_study_results <- data.table()

for (study in unique(df_cb$study_id)) {
  cat(sprintf("\nStudy: %s\n", study))

  df_study <- df_cb[study_id == study]

  tryCatch({
    # Simple model: temperature + season within this study
    m_study <- gam(biomarker ~ cb_temp + season, data = df_study, method = "REML")

    result <- data.table(
      study_id = study,
      n = nrow(df_study),
      r2 = summary(m_study)$r.sq,
      aic = AIC(m_study),
      mean_hematocrit = mean(df_study$biomarker)
    )

    cat(sprintf("  N = %d, R² = %.3f\n", result$n, result$r2))

    within_study_results <- rbind(within_study_results, result)

  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", e$message))
  })
}

cat("\nWithin-Study R² Summary:\n")
print(within_study_results)
cat(sprintf("Mean within-study R²: %.3f\n", mean(within_study_results$r2)))

fwrite(within_study_results, file.path(OUTPUT_DIR, "within_study_results.csv"))

################################################################################
# 6. PERMUTATION TEST
################################################################################

cat("\n=== 6. Permutation Test (Shuffled Temperature) ===\n")

# Shuffle temperature within each study (preserve study structure)
df_cb_perm <- copy(df_cb)
df_cb_perm[, temperature_shuffled := sample(temperature), by = study_id]

# Create new crossbasis with shuffled temperature
cb_temp_perm <- crossbasis(
  df_cb_perm$temperature_shuffled,
  lag = 14,
  argvar = list(fun = "ns", df = 3),
  arglag = list(fun = "ns", df = 3)
)

df_cb_perm <- cbind(df_cb_perm[, .(biomarker, study_id, season, vulnerability)],
                    as.data.table(cb_temp_perm))

# Fit model with shuffled temperature
cat("Fitting model with shuffled temperature...\n")
m_perm <- gam(biomarker ~ cb_temp_perm + season + vulnerability +
                s(study_id, bs = "re"),
              data = df_cb_perm, method = "REML")

cat(sprintf("\nOriginal model R²: %.3f\n", summary(m_g)$r.sq))
cat(sprintf("Permuted temperature R²: %.3f\n", summary(m_perm)$r.sq))
cat(sprintf("Difference: %.3f\n", summary(m_g)$r.sq - summary(m_perm)$r.sq))

if (summary(m_perm)$r.sq > 0.90) {
  cat("\n!! WARNING !!\n")
  cat("Shuffled temperature still gives R² > 0.90!\n")
  cat("This confirms that temperature is NOT driving the high R².\n")
  cat("The random study effects are capturing almost all variance.\n")
}

################################################################################
# 7. RESIDUAL DIAGNOSTICS
################################################################################

cat("\n=== 7. Residual Diagnostics ===\n")

# Extract residuals from full model
df_cb$residuals <- residuals(m_h)
df_cb$fitted <- fitted(m_h)

# Check residuals by study
residual_stats <- df_cb[, .(
  mean_residual = mean(residuals),
  sd_residual = sd(residuals),
  mean_fitted = mean(fitted)
), by = study_id]

cat("\nResidual statistics by study:\n")
print(residual_stats)

# Save diagnostic plots
pdf(file.path(OUTPUT_DIR, "residual_diagnostics.pdf"), width = 12, height = 8)

# Plot 1: Residuals vs Fitted
p1 <- ggplot(df_cb, aes(x = fitted, y = residuals, color = study_id)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Fitted Values",
       x = "Fitted Values",
       y = "Residuals") +
  theme_minimal()

# Plot 2: Q-Q plot
p2 <- ggplot(df_cb, aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals") +
  theme_minimal()

# Plot 3: Residuals by study
p3 <- ggplot(df_cb, aes(x = study_id, y = residuals, fill = study_id)) +
  geom_boxplot() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals by Study",
       x = "Study",
       y = "Residuals") +
  theme_minimal()

# Plot 4: Fitted values by study
p4 <- ggplot(df_cb, aes(x = study_id, y = fitted, fill = study_id)) +
  geom_boxplot() +
  labs(title = "Fitted Values by Study",
       x = "Study",
       y = "Fitted Values") +
  theme_minimal()

print(p1)
print(p2)
print(p3)
print(p4)

dev.off()

cat(sprintf("\nResidual plots saved: %s/residual_diagnostics.pdf\n", OUTPUT_DIR))

################################################################################
# 8. LEAVE-ONE-STUDY-OUT ANALYSIS
################################################################################

cat("\n=== 8. Leave-One-Study-Out Analysis ===\n")

loso_results <- data.table()

for (study in unique(df_cb$study_id)) {
  cat(sprintf("\nLeaving out study: %s\n", study))

  df_loso <- df_cb[study_id != study]

  tryCatch({
    m_loso <- gam(biomarker ~ cb_temp + season + vulnerability +
                    s(study_id, bs = "re"),
                  data = df_loso, method = "REML")

    result <- data.table(
      excluded_study = as.character(study),
      n_obs = nrow(df_loso),
      n_studies = uniqueN(df_loso$study_id),
      r2 = summary(m_loso)$r.sq,
      aic = AIC(m_loso)
    )

    cat(sprintf("  R² = %.3f\n", result$r2))

    loso_results <- rbind(loso_results, result)

  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", e$message))
  })
}

cat("\nLeave-One-Study-Out Results:\n")
print(loso_results)

fwrite(loso_results, file.path(OUTPUT_DIR, "leave_one_study_out.csv"))

################################################################################
# 9. VISUALIZE STUDY DIFFERENCES
################################################################################

cat("\n=== 9. Visualizing Study-Level Differences ===\n")

pdf(file.path(OUTPUT_DIR, "study_differences.pdf"), width = 12, height = 8)

# Plot 1: Hematocrit distribution by study
p1 <- ggplot(df_clean, aes(x = biomarker, fill = study_id)) +
  geom_density(alpha = 0.5) +
  labs(title = "Hematocrit Distribution by Study",
       x = "Hematocrit (%)",
       y = "Density") +
  theme_minimal()

# Plot 2: Hematocrit vs Temperature by study
p2 <- ggplot(df_clean, aes(x = temperature, y = biomarker, color = study_id)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = FALSE) +
  labs(title = "Hematocrit vs Temperature by Study",
       x = "Temperature (°C)",
       y = "Hematocrit (%)") +
  theme_minimal()

# Plot 3: Study means
p3 <- ggplot(study_stats, aes(x = study_id, y = mean_hematocrit, fill = study_id)) +
  geom_col() +
  geom_errorbar(aes(ymin = mean_hematocrit - sd_hematocrit,
                    ymax = mean_hematocrit + sd_hematocrit),
                width = 0.2) +
  labs(title = "Mean Hematocrit by Study (±SD)",
       x = "Study",
       y = "Mean Hematocrit (%)") +
  theme_minimal()

print(p1)
print(p2)
print(p3)

dev.off()

cat(sprintf("\nStudy difference plots saved: %s/study_differences.pdf\n", OUTPUT_DIR))

################################################################################
# 10. FINAL SUMMARY
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# SENSITIVITY ANALYSIS SUMMARY\n")
cat("################################################################################\n\n")

cat("Key Findings:\n")
cat(sprintf("1. Between-study variance: %.1f%% of total variance\n", 100 * icc))
cat(sprintf("2. Study random effects alone (no predictors): R² = %.3f\n", summary(m_e)$r.sq))
cat(sprintf("3. Full model with random effects: R² = %.3f\n", summary(m_h)$r.sq))
cat(sprintf("4. Temperature contribution: R² increase = %.3f\n",
            summary(m_h)$r.sq - summary(m_e)$r.sq))
cat(sprintf("5. Mean within-study R²: %.3f\n", mean(within_study_results$r2)))

cat("\nConclusion:\n")
if (summary(m_e)$r.sq > 0.90) {
  cat("!! HIGH R² IS AN ARTIFACT !!\n")
  cat("The R² = 0.961 is driven almost entirely by study-level differences,\n")
  cat("NOT by temperature or climate variables.\n\n")
  cat("Studies simply have different baseline hematocrit levels.\n")
  cat("Random effects are capturing study identity, not climate effects.\n\n")
  cat("RECOMMENDATION: Report within-study R² (%.3f) as the true climate effect.\n",
      mean(within_study_results$r2))
} else {
  cat("The high R² includes both study-level and temperature effects.\n")
  cat("Temperature contributes meaningfully beyond study differences.\n")
}

cat(sprintf("\nAll results saved to: %s\n", OUTPUT_DIR))
