#!/usr/bin/env Rscript
################################################################################
# Hematocrit Units Correction and Re-Analysis
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)
  library(dlnm)
  library(ggplot2)
})

set.seed(42)

DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/hematocrit_corrected"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("\n")
cat("################################################################################\n")
cat("# HEMATOCRIT UNITS CORRECTION AND RE-ANALYSIS\n")
cat("################################################################################\n\n")

# Load data
df <- fread(DATA_PATH)

df_clean <- df[, .(
  biomarker_raw = `Hematocrit (%)`,
  temperature = climate_7d_mean_temp,
  vulnerability = HEAT_VULNERABILITY_SCORE,
  study_id = as.factor(study_source),
  season = as.factor(season)
)]

df_clean <- na.omit(df_clean)

# Check distributions by study BEFORE correction
cat("=== BEFORE Units Correction ===\n")
study_stats_before <- df_clean[, .(
  n = .N,
  mean = mean(biomarker_raw),
  sd = sd(biomarker_raw),
  min = min(biomarker_raw),
  max = max(biomarker_raw)
), by = study_id]

print(study_stats_before)

# UNITS CORRECTION
# If values are < 1.0, multiply by 100 (convert from decimal to percentage)
df_clean[, biomarker := ifelse(biomarker_raw < 1.0, biomarker_raw * 100, biomarker_raw)]

cat("\n=== AFTER Units Correction ===\n")
study_stats_after <- df_clean[, .(
  n = .N,
  mean = mean(biomarker),
  sd = sd(biomarker),
  min = min(biomarker),
  max = max(biomarker)
), by = study_id]

print(study_stats_after)

# Check if correction worked
cat("\n=== Variance Decomposition (Corrected Units) ===\n")
df_clean[, study_mean := mean(biomarker), by = study_id]
df_clean[, deviation := biomarker - study_mean]

within_var <- var(df_clean$deviation)
between_var <- var(df_clean$study_mean)
total_var <- var(df_clean$biomarker)

icc <- between_var / total_var

cat(sprintf("Total variance: %.4f\n", total_var))
cat(sprintf("Within-study variance: %.4f (%.1f%%)\n",
            within_var, 100 * within_var / total_var))
cat(sprintf("Between-study variance: %.4f (%.1f%%)\n",
            between_var, 100 * between_var / total_var))
cat(sprintf("ICC: %.3f\n\n", icc))

if (icc < 0.20) {
  cat("✓ Units correction successful! ICC < 0.20\n")
} else {
  cat("⚠ High ICC persists (%.3f). May still have issues.\n", icc)
}

# Re-fit models with corrected units
cat("\n=== Re-fitting Models with Corrected Units ===\n")

# Create crossbasis
cb_temp <- crossbasis(
  df_clean$temperature,
  lag = 14,
  argvar = list(fun = "ns", df = 3),
  arglag = list(fun = "ns", df = 3)
)

df_cb <- cbind(df_clean, as.data.table(cb_temp))

# Model 1: Baseline (no random effects)
cat("\nModel 1: Baseline (no random effects)...\n")
m1 <- gam(biomarker ~ cb_temp + season + vulnerability,
          data = df_cb, method = "REML")

# Model 2: Random intercept by study
cat("Model 2: Random intercept by study...\n")
m2 <- gam(biomarker ~ cb_temp + season + vulnerability + s(study_id, bs = "re"),
          data = df_cb, method = "REML")

# Model 3: Random slope
cat("Model 3: Random slope by study...\n")
m3 <- tryCatch({
  gam(biomarker ~ cb_temp + season + vulnerability +
        s(study_id, bs = "re") + s(study_id, temperature, bs = "re"),
      data = df_cb, method = "REML")
}, error = function(e) {
  cat(sprintf("  ERROR: %s\n", e$message))
  return(NULL)
})

# Model E: Study random effects only (no predictors)
cat("Model E: Study random effects only...\n")
m_e <- gam(biomarker ~ s(study_id, bs = "re"),
           data = df_cb, method = "REML")

# Compare
comparison <- data.table(
  model = c("Baseline", "Random intercept", "Random slope", "Study RE only"),
  r2 = c(summary(m1)$r.sq, summary(m2)$r.sq,
         if(!is.null(m3)) summary(m3)$r.sq else NA,
         summary(m_e)$r.sq),
  aic = c(AIC(m1), AIC(m2),
          if(!is.null(m3)) AIC(m3) else NA,
          AIC(m_e))
)

comparison <- na.omit(comparison)
comparison[, aic_delta := aic - min(aic)]
comparison[, r2_gain_from_RE := r2 - r2[model == "Baseline"]]

cat("\n=== Model Comparison (Corrected Units) ===\n")
print(comparison[order(aic)])

cat("\n=== Key Comparisons ===\n")
cat(sprintf("Study RE only (no predictors): R² = %.3f\n", comparison[model == "Study RE only"]$r2))
cat(sprintf("Baseline (with climate): R² = %.3f\n", comparison[model == "Baseline"]$r2))
cat(sprintf("Best model: R² = %.3f\n", max(comparison$r2)))

# Test DLNM significance
best_model <- m2
pred <- crosspred(cb_temp, best_model, at = seq(15, 30, 1), cen = 18)

sig_idx <- which(pred$alllow > 0 | pred$allhigh < 0)
cat(sprintf("\nSignificant DLNM effects: %d temperatures\n", length(sig_idx)))

if (length(sig_idx) > 0) {
  cat(sprintf("Temperature range: %.1f - %.1f°C\n",
              min(pred$predvar[sig_idx]), max(pred$predvar[sig_idx])))
}

# Save corrected data
fwrite(df_clean[, .(study_id, biomarker_raw, biomarker, temperature, season, vulnerability)],
       file.path(OUTPUT_DIR, "hematocrit_corrected_data.csv"))

cat(sprintf("\n✓ Corrected data saved: %s/hematocrit_corrected_data.csv\n", OUTPUT_DIR))
cat(sprintf("✓ Results saved: %s\n", OUTPUT_DIR))
