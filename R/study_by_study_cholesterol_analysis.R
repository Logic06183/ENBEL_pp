################################################################################
# STUDY-BY-STUDY CHOLESTEROL ANALYSIS
################################################################################
#
# PURPOSE: Analyze climate-cholesterol relationship WITHIN each study separately
#          to isolate within-study effects vs between-study heterogeneity
#
# COMPARISON:
#   - Mixed effects (current): Partial pooling, assumes similar effect across studies
#   - Study-by-study: No pooling, each study analyzed independently
#
# Date: 2025-10-30
################################################################################

library(data.table)
library(mgcv)
library(dlnm)
library(ggplot2)

# Paths
DATA_FILE <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/study_by_study_cholesterol"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("################################################################################\n")
cat("# STUDY-BY-STUDY ANALYSIS: TOTAL CHOLESTEROL\n")
cat("################################################################################\n\n")

cat("GOAL: Isolate WITHIN-STUDY climate effects (no pooling)\n")
cat("COMPARISON: Mixed effects (partial pooling) vs study-by-study (no pooling)\n\n")

################################################################################
# 1. DATA PREPARATION
################################################################################

cat("=== 1. Loading Data ===\n")

# Load dataset
df <- fread(DATA_FILE)

# Select cholesterol records with required columns
df_chol <- df[, .(
  cholesterol_raw = total_cholesterol_mg_dL,
  temperature = climate_7d_mean_temp,
  vulnerability = HEAT_VULNERABILITY_SCORE,
  study_id = as.factor(study_source),
  season = as.factor(season),
  date = as.Date(primary_date),
  year = year,
  month = month
)]

# Remove missing values
df_chol <- na.omit(df_chol)

# Keep only studies with cholesterol data
cat(sprintf("  Total cholesterol records: %d\n", nrow(df_chol)))
cat(sprintf("  Number of studies: %d\n", uniqueN(df_chol$study_id)))

# Check study-level means to detect units issues
study_means <- df_chol[, .(mean_chol = mean(cholesterol_raw, na.rm = TRUE),
                           n = .N), by = study_id]
setorder(study_means, -n)

cat("\n  Study-level means (raw):\n")
print(study_means)

# Units correction: convert mmol/L to mg/dL if needed
needs_conversion <- any(study_means$mean_chol < 15) & any(study_means$mean_chol > 30)

if (needs_conversion) {
  cat("\n  ⚠️  Units mixing detected. Converting mmol/L to mg/dL...\n")
  df_chol[, cholesterol := ifelse(cholesterol_raw < 15,
                                  cholesterol_raw * 38.67,  # mmol/L to mg/dL
                                  cholesterol_raw)]
  cat("  ✓ Conversion applied\n")
} else {
  df_chol[, cholesterol := cholesterol_raw]
  cat("\n  ✓ Units consistent across studies\n")
}

# Check after conversion
study_means_corrected <- df_chol[, .(mean_chol = mean(cholesterol, na.rm = TRUE),
                                      sd_chol = sd(cholesterol, na.rm = TRUE),
                                      n = .N), by = study_id]
setorder(study_means_corrected, -n)

cat("\n  Study-level means (after correction):\n")
print(study_means_corrected)

cat(sprintf("\n  Complete cases: %d\n", nrow(df_chol)))

################################################################################
# 2. STUDY-BY-STUDY MODELS (NO POOLING)
################################################################################

cat("\n=== 2. Fitting Separate Models for Each Study ===\n\n")

# Initialize results table
study_results <- data.table()

# Fit model for each study separately
for (study in unique(df_chol$study_id)) {

  cat(sprintf("STUDY: %s\n", study))

  # Subset to this study
  df_study <- df_chol[study_id == study]

  cat(sprintf("  N = %d\n", nrow(df_study)))
  cat(sprintf("  Temperature range: %.1f - %.1f°C\n",
              min(df_study$temperature), max(df_study$temperature)))
  cat(sprintf("  Cholesterol range: %.1f - %.1f mg/dL\n",
              min(df_study$cholesterol), max(df_study$cholesterol)))

  # Skip if too few observations
  if (nrow(df_study) < 100) {
    cat("  ⚠️  Too few observations (n < 100). Skipping.\n\n")
    next
  }

  # Create DLNM crossbasis for this study
  cb_temp_study <- crossbasis(
    df_study$temperature,
    lag = 14,
    argvar = list(fun = "ns", df = 3),
    arglag = list(fun = "ns", df = 3)
  )

  # Model 1: Temperature only (no confounding control)
  m_temp_only <- gam(
    cholesterol ~ cb_temp_study,
    data = df_study,
    method = "REML"
  )

  r2_temp_only <- summary(m_temp_only)$r.sq
  cat(sprintf("  Model 1 (Temp only): R² = %.3f\n", r2_temp_only))

  # Model 2: Temperature + Season
  m_temp_season <- gam(
    cholesterol ~ cb_temp_study + season,
    data = df_study,
    method = "REML"
  )

  r2_temp_season <- summary(m_temp_season)$r.sq
  cat(sprintf("  Model 2 (Temp + Season): R² = %.3f\n", r2_temp_season))

  # Model 3: Temperature + Season + Vulnerability (FULL)
  m_full <- gam(
    cholesterol ~ cb_temp_study + season + vulnerability,
    data = df_study,
    method = "REML"
  )

  r2_full <- summary(m_full)$r.sq
  aic_full <- AIC(m_full)
  cat(sprintf("  Model 3 (Full): R² = %.3f, AIC = %.1f\n", r2_full, aic_full))

  # Test DLNM significance
  pred_study <- crosspred(cb_temp_study, m_full, at = seq(10, 30, 0.5), cen = 18)
  sig_temps <- sum(pred_study$alllow > 0 | pred_study$allhigh < 0)
  cat(sprintf("  Significant temperatures: %d\n", sig_temps))

  # Find maximum effect
  max_effect_idx <- which.max(abs(pred_study$allfit))
  max_temp <- pred_study$predvar[max_effect_idx]
  max_effect <- pred_study$allfit[max_effect_idx]
  max_lower <- pred_study$alllow[max_effect_idx]
  max_upper <- pred_study$allhigh[max_effect_idx]

  cat(sprintf("  Maximum effect: %.2f mg/dL at %.1f°C (95%% CI: [%.2f, %.2f])\n",
              max_effect, max_temp, max_lower, max_upper))

  # Store results
  study_results <- rbind(study_results, data.table(
    study_id = study,
    n = nrow(df_study),
    temp_range = sprintf("%.1f-%.1f", min(df_study$temperature), max(df_study$temperature)),
    chol_mean = mean(df_study$cholesterol),
    chol_sd = sd(df_study$cholesterol),
    r2_temp_only = r2_temp_only,
    r2_temp_season = r2_temp_season,
    r2_full = r2_full,
    aic = aic_full,
    n_sig_temps = sig_temps,
    max_effect = max_effect,
    max_temp = max_temp,
    max_lower = max_lower,
    max_upper = max_upper
  ))

  cat("\n")
}

# Print summary table
cat("=== Summary: Study-by-Study Results ===\n\n")
print(study_results)

# Save results
fwrite(study_results, file.path(OUTPUT_DIR, "study_by_study_results.csv"))

################################################################################
# 3. COMPARE WITH MIXED EFFECTS (PARTIAL POOLING)
################################################################################

cat("\n=== 3. Comparison: Mixed Effects vs Study-by-Study ===\n\n")

# Fit mixed effects model (partial pooling) for comparison
cb_temp_pooled <- crossbasis(
  df_chol$temperature,
  lag = 14,
  argvar = list(fun = "ns", df = 3),
  arglag = list(fun = "ns", df = 3)
)

# Mixed effects model
m_mixed <- gam(
  cholesterol ~ cb_temp_pooled + season + vulnerability + s(study_id, bs = "re"),
  data = df_chol,
  method = "REML"
)

r2_mixed <- summary(m_mixed)$r.sq
cat(sprintf("Mixed Effects Model (Partial Pooling):\n"))
cat(sprintf("  R² = %.3f (pooled across all studies)\n", r2_mixed))
cat(sprintf("  N = %d\n", nrow(df_chol)))

# Test pooled DLNM significance
pred_mixed <- crosspred(cb_temp_pooled, m_mixed, at = seq(10, 30, 0.5), cen = 18)
sig_temps_mixed <- sum(pred_mixed$alllow > 0 | pred_mixed$allhigh < 0)
cat(sprintf("  Significant temperatures: %d\n\n", sig_temps_mixed))

# Calculate summary statistics for study-by-study
cat("Study-by-Study Models (No Pooling):\n")
cat(sprintf("  Mean R² = %.3f (SD = %.3f)\n",
            mean(study_results$r2_full), sd(study_results$r2_full)))
cat(sprintf("  Range R² = %.3f - %.3f\n",
            min(study_results$r2_full), max(study_results$r2_full)))
cat(sprintf("  Median N sig temps = %d (range: %d - %d)\n",
            median(study_results$n_sig_temps),
            min(study_results$n_sig_temps),
            max(study_results$n_sig_temps)))

################################################################################
# 4. VARIANCE DECOMPOSITION
################################################################################

cat("\n=== 4. Variance Decomposition ===\n\n")

# Calculate between-study variance
between_study_var <- var(study_means_corrected$mean_chol)
cat(sprintf("Between-study variance: %.2f (mg/dL)²\n", between_study_var))

# Calculate within-study variance (average of study-specific variances)
within_study_var <- mean(study_means_corrected$sd_chol^2)
cat(sprintf("Within-study variance: %.2f (mg/dL)²\n", within_study_var))

# Intraclass correlation (ICC)
total_var <- between_study_var + within_study_var
icc <- between_study_var / total_var
cat(sprintf("\nIntraclass Correlation (ICC): %.3f\n", icc))
cat(sprintf("Interpretation: %.1f%% of variance is BETWEEN studies\n", icc * 100))
cat(sprintf("              %.1f%% of variance is WITHIN studies\n", (1 - icc) * 100))

if (icc > 0.5) {
  cat("\n⚠️  HIGH ICC: Most variance is between-study\n")
  cat("   → Mixed effects captures between-study heterogeneity\n")
  cat("   → Study-by-study reveals within-study effects\n")
} else if (icc > 0.2) {
  cat("\n⚠️  MODERATE ICC: Substantial between-study variance\n")
  cat("   → Both approaches important\n")
} else {
  cat("\n✓ LOW ICC: Most variance is within-study\n")
  cat("  → Study-by-study and mixed effects should agree\n")
}

################################################################################
# 5. HETEROGENEITY ANALYSIS
################################################################################

cat("\n=== 5. Heterogeneity Across Studies ===\n\n")

# Test heterogeneity in R²
cat("Heterogeneity in Climate Effect (R²):\n")
cat(sprintf("  Range: %.3f - %.3f (%.1fx difference)\n",
            min(study_results$r2_full), max(study_results$r2_full),
            max(study_results$r2_full) / min(study_results$r2_full)))

# Which study has strongest effect?
strongest_study <- study_results[which.max(r2_full)]
cat(sprintf("\n  Strongest effect: %s (R² = %.3f, n = %d)\n",
            strongest_study$study_id, strongest_study$r2_full, strongest_study$n))

# Which study has weakest effect?
weakest_study <- study_results[which.min(r2_full)]
cat(sprintf("  Weakest effect: %s (R² = %.3f, n = %d)\n",
            weakest_study$study_id, weakest_study$r2_full, weakest_study$n))

# Test if all studies show positive R²
if (all(study_results$r2_full > 0)) {
  cat("\n✓ All studies show positive R² (climate effect present in all)\n")
} else {
  cat("\n⚠️  Some studies show R² ≤ 0 (no climate effect detected)\n")
}

################################################################################
# 6. VISUALIZATION
################################################################################

cat("\n=== 6. Creating Visualization ===\n")

pdf(file.path(OUTPUT_DIR, "study_by_study_comparison.pdf"), width = 14, height = 10)

par(mfrow = c(2, 3))

# Panel A: R² by study
barplot(study_results$r2_full, names.arg = study_results$study_id,
        col = "steelblue", las = 2,
        main = "A) R² by Study (No Pooling)",
        ylab = "R² (Within-Study)",
        ylim = c(0, max(study_results$r2_full) * 1.2))
abline(h = r2_mixed, col = "red", lwd = 2, lty = 2)
legend("topright", legend = sprintf("Mixed Effects: %.3f", r2_mixed),
       col = "red", lwd = 2, lty = 2, cex = 0.8)

# Panel B: Sample size vs R²
plot(study_results$n, study_results$r2_full,
     pch = 16, cex = 2, col = "steelblue",
     xlab = "Sample Size", ylab = "R² (Within-Study)",
     main = "B) Sample Size vs Effect Size")
text(study_results$n, study_results$r2_full,
     labels = study_results$study_id, pos = 4, cex = 0.8)
abline(h = r2_mixed, col = "red", lwd = 2, lty = 2)

# Panel C: Temperature range vs R²
temp_ranges_numeric <- sapply(strsplit(study_results$temp_range, "-"), function(x) {
  as.numeric(x[2]) - as.numeric(x[1])
})
plot(temp_ranges_numeric, study_results$r2_full,
     pch = 16, cex = 2, col = "steelblue",
     xlab = "Temperature Range (°C)", ylab = "R² (Within-Study)",
     main = "C) Temperature Range vs Effect Size")
text(temp_ranges_numeric, study_results$r2_full,
     labels = study_results$study_id, pos = 4, cex = 0.8)

# Panel D: Confounding effect (R² change)
study_results[, r2_change_season := r2_temp_season - r2_temp_only]
study_results[, r2_change_vuln := r2_full - r2_temp_season]

barplot(t(as.matrix(study_results[, .(r2_temp_only, r2_change_season, r2_change_vuln)])),
        names.arg = study_results$study_id, las = 2,
        col = c("lightblue", "orange", "darkgreen"),
        main = "D) Confounding Effects by Study",
        ylab = "R² Contribution",
        legend.text = c("Temperature Only", "+ Season", "+ Vulnerability"),
        args.legend = list(x = "topright", cex = 0.7))

# Panel E: Effect sizes with confidence intervals
x_pos <- 1:nrow(study_results)
plot(x_pos, study_results$max_effect,
     ylim = range(c(study_results$max_lower, study_results$max_upper)),
     xlab = "", ylab = "Maximum Effect (mg/dL)",
     xaxt = "n", pch = 16, cex = 2, col = "steelblue",
     main = "E) Maximum Effects with 95% CI")
arrows(x_pos, study_results$max_lower, x_pos, study_results$max_upper,
       angle = 90, code = 3, length = 0.1, col = "steelblue")
abline(h = 0, col = "red", lty = 2)
axis(1, at = x_pos, labels = study_results$study_id, las = 2)

# Panel F: Number of significant temperatures
barplot(study_results$n_sig_temps, names.arg = study_results$study_id,
        col = "darkgreen", las = 2,
        main = "F) Number of Significant Temperatures",
        ylab = "N Significant Temps")

dev.off()

cat(sprintf("  Saved: %s/study_by_study_comparison.pdf\n", OUTPUT_DIR))

################################################################################
# 7. FINAL SUMMARY
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# SUMMARY: WITHIN-STUDY vs BETWEEN-STUDY EFFECTS\n")
cat("################################################################################\n\n")

cat("VARIANCE DECOMPOSITION:\n")
cat(sprintf("  ICC = %.3f (%.1f%% between-study, %.1f%% within-study)\n",
            icc, icc * 100, (1 - icc) * 100))

cat("\nMIXED EFFECTS (Partial Pooling):\n")
cat(sprintf("  R² = %.3f\n", r2_mixed))
cat(sprintf("  Assumes SIMILAR climate effect across studies\n"))
cat(sprintf("  Captures both within-study and between-study variance\n"))

cat("\nSTUDY-BY-STUDY (No Pooling):\n")
cat(sprintf("  Mean R² = %.3f (SD = %.3f)\n",
            mean(study_results$r2_full), sd(study_results$r2_full)))
cat(sprintf("  Range R² = %.3f - %.3f (%.1fx heterogeneity)\n",
            min(study_results$r2_full), max(study_results$r2_full),
            max(study_results$r2_full) / min(study_results$r2_full)))
cat(sprintf("  Isolates WITHIN-STUDY climate effects\n"))
cat(sprintf("  Reveals heterogeneity across studies\n"))

cat("\nKEY INSIGHTS:\n")
if (abs(r2_mixed - mean(study_results$r2_full)) < 0.05) {
  cat("  ✓ Mixed effects ≈ Mean study-by-study R²\n")
  cat("  → Pooling is reasonable, minimal heterogeneity\n")
} else if (r2_mixed > mean(study_results$r2_full)) {
  cat("  ⚠️  Mixed effects R² > Mean study-by-study R²\n")
  cat("  → Mixed model captures between-study variance\n")
  cat("  → Inflated estimate of within-study climate effect\n")
} else {
  cat("  ⚠️  Mixed effects R² < Mean study-by-study R²\n")
  cat("  → Pooling shrinks estimates toward overall mean\n")
}

if (sd(study_results$r2_full) / mean(study_results$r2_full) > 0.5) {
  cat("\n  ⚠️  HIGH HETEROGENEITY across studies (CV > 50%)\n")
  cat("  → Climate effect varies substantially by study\n")
  cat("  → Study-by-study analysis reveals important differences\n")
} else {
  cat("\n  ✓ LOW HETEROGENEITY across studies (CV < 50%)\n")
  cat("  → Climate effect relatively consistent\n")
  cat("  → Mixed effects pooling is appropriate\n")
}

cat("\nRECOMMENDATION:\n")
cat("  Report BOTH approaches:\n")
cat("  1. Study-by-study: Shows within-study effects (no confounding by study)\n")
cat("  2. Mixed effects: Borrows strength across studies, provides pooled estimate\n")
cat("  3. Heterogeneity tests: Quantify variation across studies\n")

cat("\n=== Analysis Complete ===\n")
