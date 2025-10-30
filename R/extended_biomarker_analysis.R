################################################################################
# EXTENDED BIOMARKER ANALYSIS: HIGH-POWER CANDIDATES
################################################################################
#
# PURPOSE: Analyze biomarkers with larger sample sizes (5-7 studies)
#          to achieve adequate statistical power
#
# BIOMARKERS:
# - BMI, Weight, Height (7 studies each) - HIGH POWER
# - Heart Rate, Body Temperature (4 studies) - Thermal relevance
# - HIV Viral Load, Hemoglobin, Platelet Count (4 studies) - Clinical relevance
#
# Date: 2025-10-30
################################################################################

library(data.table)
library(mgcv)
library(dlnm)
library(metafor)
library(ggplot2)

# Paths
DATA_FILE <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/extended_biomarker_analysis"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("################################################################################\n")
cat("# EXTENDED BIOMARKER ANALYSIS: HIGH-POWER CANDIDATES\n")
cat("################################################################################\n\n")

################################################################################
# 1. LOAD DATA
################################################################################

df <- fread(DATA_FILE)

# Define extended biomarker set
biomarkers <- list(
  # HIGH POWER (7 studies)
  bmi = list(
    col = "BMI (kg/m²)",
    name = "BMI",
    units = "kg/m²",
    expected_studies = 7
  ),
  weight = list(
    col = "weight_kg",
    name = "Weight",
    units = "kg",
    expected_studies = 7
  ),
  height = list(
    col = "height_m",
    name = "Height",
    units = "m",
    expected_studies = 7
  ),
  # THERMAL RELEVANCE (4 studies)
  heart_rate = list(
    col = "heart_rate_bpm",
    name = "Heart Rate",
    units = "bpm",
    expected_studies = 4
  ),
  body_temp = list(
    col = "body_temperature_celsius",
    name = "Body Temperature",
    units = "°C",
    expected_studies = 4
  ),
  # CLINICAL RELEVANCE (4 studies)
  viral_load = list(
    col = "HIV viral load (copies/mL)",
    name = "HIV Viral Load",
    units = "copies/mL",
    expected_studies = 4
  ),
  hemoglobin = list(
    col = "hemoglobin_g_dL",
    name = "Hemoglobin",
    units = "g/dL",
    expected_studies = 4
  ),
  platelet = list(
    col = "Platelet count (×10³/µL)",
    name = "Platelet Count",
    units = "×10³/µL",
    expected_studies = 4
  )
)

cat("Biomarkers selected:\n")
for (bio_name in names(biomarkers)) {
  cat(sprintf("  - %s (%d expected studies)\n",
              biomarkers[[bio_name]]$name,
              biomarkers[[bio_name]]$expected_studies))
}
cat("\n")

################################################################################
# 2. ANALYSIS FUNCTION
################################################################################

analyze_biomarker_extended <- function(biomarker_config, df_full) {

  biomarker_name <- biomarker_config$name
  cat(sprintf("\n================================================================================\n"))
  cat(sprintf("BIOMARKER: %s\n", biomarker_name))
  cat(sprintf("================================================================================\n\n"))

  # Extract data
  df_bio <- df_full[, .(
    biomarker_raw = get(biomarker_config$col),
    temperature = climate_7d_mean_temp,
    vulnerability = HEAT_VULNERABILITY_SCORE,
    study_id = as.factor(study_source),
    season = as.factor(season),
    date = as.Date(primary_date),
    year = year,
    month = month
  )]

  df_bio <- na.omit(df_bio)
  df_bio[, biomarker := biomarker_raw]

  cat(sprintf("Total %s records: %d\n", biomarker_name, nrow(df_bio)))
  cat(sprintf("Studies with data: %d\n\n", uniqueN(df_bio$study_id)))

  if (nrow(df_bio) < 100 || uniqueN(df_bio$study_id) < 3) {
    cat("  ⚠️  Insufficient data. Skipping.\n")
    return(NULL)
  }

  # Study-level descriptives
  study_summary <- df_bio[, .(
    n = .N,
    mean_bio = mean(biomarker),
    sd_bio = sd(biomarker),
    mean_temp = mean(temperature),
    mean_vuln = mean(vulnerability)
  ), by = study_id]

  cat("Study-level summary:\n")
  print(study_summary)

  cat("\n--- Study-by-Study Analysis ---\n\n")

  study_results <- data.table()

  for (study in unique(df_bio$study_id)) {

    df_study <- df_bio[study_id == study]

    cat(sprintf("Study: %s (n=%d)\n", study, nrow(df_study)))

    if (nrow(df_study) < 100) {
      cat("  ⚠️  n < 100. Skipping.\n\n")
      next
    }

    # Create crossbasis
    cb_study <- crossbasis(
      df_study$temperature,
      lag = 14,
      argvar = list(fun = "ns", df = 3),
      arglag = list(fun = "ns", df = 3)
    )

    # Full model
    m_full <- gam(biomarker ~ cb_study + season + vulnerability,
                  data = df_study, method = "REML")

    r2_full <- summary(m_full)$r.sq
    aic_full <- AIC(m_full)

    # Test DLNM significance
    temp_range <- range(df_study$temperature)
    pred_study <- crosspred(cb_study, m_full,
                           at = seq(floor(temp_range[1]), ceiling(temp_range[2]), 0.5),
                           cen = median(df_study$temperature))

    sig_temps <- sum(pred_study$alllow > 0 | pred_study$allhigh < 0)

    # Maximum effect
    max_idx <- which.max(abs(pred_study$allfit))
    max_effect <- pred_study$allfit[max_idx]
    max_temp <- pred_study$predvar[max_idx]

    cat(sprintf("  R² = %.3f, Sig temps = %d\n\n", r2_full, sig_temps))

    # Store
    study_results <- rbind(study_results, data.table(
      biomarker = biomarker_name,
      study_id = study,
      n = nrow(df_study),
      mean_bio = mean(df_study$biomarker),
      sd_bio = sd(df_study$biomarker),
      mean_temp = mean(df_study$temperature),
      mean_vuln = mean(df_study$vulnerability),
      r2_full = r2_full,
      aic = aic_full,
      n_sig_temps = sig_temps,
      max_effect = max_effect,
      max_temp = max_temp
    ))
  }

  # Summary
  cat("--- Summary ---\n")
  cat(sprintf("  Studies analyzed: %d\n", nrow(study_results)))
  cat(sprintf("  Mean R² = %.3f (SD = %.3f)\n",
              mean(study_results$r2_full), sd(study_results$r2_full)))
  cat(sprintf("  Range: %.3f - %.3f\n",
              min(study_results$r2_full), max(study_results$r2_full)))

  # Vulnerability correlation
  if (nrow(study_results) > 2 & sd(study_results$mean_vuln) > 0) {
    cor_vuln <- cor(study_results$mean_vuln, study_results$r2_full)
    cat(sprintf("  Vulnerability-R² correlation: r = %.3f\n", cor_vuln))

    if (abs(cor_vuln) > 0.5) {
      if (cor_vuln < 0) {
        cat("    → PARADOX pattern detected\n")
      } else {
        cat("    → Expected pattern detected\n")
      }
    }
  }

  return(study_results)
}

################################################################################
# 3. RUN ANALYSES
################################################################################

cat("\n=== Running Analyses ===\n")

all_results <- data.table()

for (bio_name in names(biomarkers)) {
  results <- analyze_biomarker_extended(biomarkers[[bio_name]], df)
  if (!is.null(results)) {
    all_results <- rbind(all_results, results)
  }
}

# Save
fwrite(all_results, file.path(OUTPUT_DIR, "extended_biomarker_results.csv"))

################################################################################
# 4. META-REGRESSION
################################################################################

cat("\n=== Meta-Regression Analysis ===\n\n")

meta_results <- data.table()

for (bio in unique(all_results$biomarker)) {

  cat(sprintf("Biomarker: %s\n", bio))

  df_bio <- all_results[biomarker == bio]

  if (nrow(df_bio) < 3) {
    cat("  ⚠️  Too few studies\n\n")
    next
  }

  # Check vulnerability variation
  if (sd(df_bio$mean_vuln, na.rm = TRUE) < 1) {
    cat("  ⚠️  No vulnerability variation\n\n")
    next
  }

  # Calculate variance
  df_bio[, se_r2 := sqrt((1 - r2_full)^2 / (n - 2))]
  df_bio[, vi := se_r2^2]

  # Meta-analysis (pooled)
  meta_pooled <- rma(yi = r2_full, vi = vi, data = df_bio, method = "REML")

  cat(sprintf("  Pooled R² = %.4f [%.4f, %.4f], p = %.4f\n",
              meta_pooled$beta[1], meta_pooled$ci.lb,
              meta_pooled$ci.ub, meta_pooled$pval))
  cat(sprintf("  I² = %.1f%%, Q p-value = %.4f\n",
              meta_pooled$I2, meta_pooled$QEp))

  # Meta-regression
  meta_reg <- rma(yi = r2_full, vi = vi, mods = ~ mean_vuln,
                  data = df_bio, method = "REML")

  cat(sprintf("  Vulnerability slope = %.6f (SE = %.6f), p = %.4f\n",
              meta_reg$beta[2], meta_reg$se[2], meta_reg$pval[2]))

  # R² explained
  R2_explained <- max(0, (meta_pooled$tau2 - meta_reg$tau2) / (meta_pooled$tau2 + 1e-10))
  cat(sprintf("  R² explained by vulnerability: %.1f%%\n", R2_explained * 100))

  # Significance
  if (meta_reg$pval[2] < 0.05) {
    if (meta_reg$beta[2] < 0) {
      cat("  ✓✓ SIGNIFICANT PARADOX (p < 0.05)\n")
    } else {
      cat("  ✓ Significant positive relationship (p < 0.05)\n")
    }
  } else if (meta_reg$pval[2] < 0.10) {
    cat("  ⚠️  Marginal significance (p < 0.10)\n")
  } else {
    cat("  → Not significant (p ≥ 0.10)\n")
  }

  meta_results <- rbind(meta_results, data.table(
    biomarker = bio,
    k = nrow(df_bio),
    pooled_r2 = meta_pooled$beta[1],
    pooled_p = meta_pooled$pval,
    I2 = meta_pooled$I2,
    Q_p = meta_pooled$QEp,
    slope_vuln = meta_reg$beta[2],
    slope_p = meta_reg$pval[2],
    R2_explained = R2_explained
  ))

  cat("\n")
}

# Save
fwrite(meta_results, file.path(OUTPUT_DIR, "meta_regression_results.csv"))

################################################################################
# 5. CROSS-BIOMARKER SUMMARY
################################################################################

cat("================================================================================\n")
cat("# SUMMARY: EXTENDED BIOMARKER ANALYSIS\n")
cat("================================================================================\n\n")

cat("WITHIN-STUDY EFFECTS (Mean R²):\n")
summary_table <- all_results[, .(
  k = .N,
  mean_r2 = mean(r2_full),
  sd_r2 = sd(r2_full),
  min_r2 = min(r2_full),
  max_r2 = max(r2_full)
), by = biomarker]

print(summary_table)

cat("\nMETA-REGRESSION RESULTS:\n")
print(meta_results[, .(biomarker, k, pooled_r2, pooled_p, slope_vuln, slope_p)])

cat("\nSIGNIFICANT FINDINGS (p < 0.05):\n")
sig_findings <- meta_results[slope_p < 0.05]
if (nrow(sig_findings) > 0) {
  print(sig_findings[, .(biomarker, k, slope_vuln, slope_p, R2_explained)])
  cat("\n  ✓✓ SIGNIFICANT patterns detected!\n")
} else {
  cat("  → No significant vulnerability-R² relationships at p < 0.05\n")
}

cat("\nMARGINAL FINDINGS (p < 0.10):\n")
marginal_findings <- meta_results[slope_p >= 0.05 & slope_p < 0.10]
if (nrow(marginal_findings) > 0) {
  print(marginal_findings[, .(biomarker, k, slope_vuln, slope_p, R2_explained)])
  cat("\n  ⚠️  Marginal patterns detected\n")
} else {
  cat("  → No marginal findings\n")
}

cat("\n=== Analysis Complete ===\n")
