#!/usr/bin/env Rscript
################################################################################
# Comprehensive Units Check and Re-Analysis for All Biomarkers
################################################################################
#
# Purpose: Check ALL biomarkers for units inconsistencies like Hematocrit issue
#          Apply corrections and re-run mixed effects DLNM analysis
#
# Heuristics for detecting units issues:
# 1. Multimodal distribution (suggests different scales)
# 2. Extreme between-study variance (ICC > 0.5)
# 3. Study means differ by orders of magnitude
# 4. Values span multiple orders of magnitude within biomarker
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
OUTPUT_DIR <- "reanalysis_outputs/comprehensive_units_check"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# DLNM parameters
DLNM_LAG <- 14
TEMP_DF <- 3
LAG_DF <- 3
REF_TEMP <- 18

# Biomarkers to check
BIOMARKERS <- c(
  "Hematocrit (%)",
  "total_cholesterol_mg_dL",
  "hdl_cholesterol_mg_dL",
  "ldl_cholesterol_mg_dL",
  "fasting_glucose_mmol_L",
  "creatinine_umol_L"
)

cat("\n")
cat("################################################################################\n")
cat("# COMPREHENSIVE UNITS CHECK AND RE-ANALYSIS\n")
cat("################################################################################\n\n")

# Load data
cat("Loading data...\n")
df_raw <- fread(DATA_PATH)
cat(sprintf("Loaded %d observations\n", nrow(df_raw)))

################################################################################
# FUNCTIONS
################################################################################

#' Detect units issues for a biomarker
#'
#' @param df data.table with biomarker and study_id columns
#' @param biomarker_name name of biomarker
#' @return list with diagnostic results
detect_units_issues <- function(df, biomarker_name) {

  # Study-level statistics
  study_stats <- df[, .(
    n = .N,
    mean = mean(biomarker),
    sd = sd(biomarker),
    min = min(biomarker),
    max = max(biomarker),
    cv = sd(biomarker) / mean(biomarker)  # Coefficient of variation
  ), by = study_id]

  # Overall statistics
  total_var <- var(df$biomarker)
  df[, study_mean := mean(biomarker), by = study_id]
  df[, deviation := biomarker - study_mean]
  within_var <- var(df$deviation)
  between_var <- var(df$study_mean)
  icc <- between_var / total_var

  # Check for suspicious patterns
  issues <- list()

  # Issue 1: High ICC (>0.5 suggests extreme clustering)
  if (icc > 0.5) {
    issues$high_icc <- sprintf("ICC = %.3f (>0.5 threshold)", icc)
  }

  # Issue 2: Study means differ by >10x
  mean_ratio <- max(study_stats$mean) / min(study_stats$mean)
  if (mean_ratio > 10) {
    issues$mean_ratio <- sprintf("Study means differ by %.1fx", mean_ratio)
  }

  # Issue 3: Values span >100x range
  overall_range <- max(df$biomarker) / min(df$biomarker)
  if (overall_range > 100 && min(df$biomarker) > 0) {
    issues$wide_range <- sprintf("Values span %.0fx range", overall_range)
  }

  # Issue 4: One study has values <1 while others >10 (decimal vs integer)
  study_with_small_values <- study_stats[mean < 1]
  study_with_large_values <- study_stats[mean > 10]
  if (nrow(study_with_small_values) > 0 && nrow(study_with_large_values) > 0) {
    issues$decimal_vs_integer <- "Some studies have means <1, others >10"
  }

  # Issue 5: Multimodal distribution (Hartigan's dip test would be ideal, but checking variance ratio)
  if (between_var > 10 * within_var) {
    issues$multimodal <- sprintf("Between-study variance %.1fx within-study",
                                  between_var / within_var)
  }

  return(list(
    issues = issues,
    icc = icc,
    mean_ratio = mean_ratio,
    study_stats = study_stats,
    has_issues = length(issues) > 0
  ))
}

#' Apply units correction
#'
#' @param df data.table
#' @param diagnostic diagnostic results from detect_units_issues
#' @return corrected data.table with correction_applied flag
apply_units_correction <- function(df, diagnostic) {

  df_corrected <- copy(df)
  df_corrected[, correction_applied := FALSE]

  # Strategy 1: If some studies have values <1 and others >10, multiply <1 values by 100
  if (!is.null(diagnostic$issues$decimal_vs_integer)) {
    cat("  Applying correction: values <1 multiplied by 100\n")
    df_corrected[biomarker < 1, `:=`(
      biomarker = biomarker * 100,
      correction_applied = TRUE
    )]
  }

  # Strategy 2: If one study has extreme mean (>3 SD from overall), flag for investigation
  overall_mean <- mean(df_corrected$biomarker)
  overall_sd <- sd(df_corrected$biomarker)

  study_means <- df_corrected[, .(study_mean = mean(biomarker)), by = study_id]
  extreme_studies <- study_means[abs(study_mean - overall_mean) > 3 * overall_sd]

  if (nrow(extreme_studies) > 0) {
    cat(sprintf("  WARNING: %d studies have extreme means (>3 SD)\n", nrow(extreme_studies)))
    cat("  Manual review recommended\n")
  }

  return(df_corrected)
}

#' Run mixed effects DLNM for single biomarker
#'
#' @param df_clean prepared data
#' @param biomarker_name name for reporting
#' @return analysis results
run_dlnm_analysis <- function(df_clean, biomarker_name) {

  # Create crossbasis
  cb_temp <- crossbasis(
    df_clean$temperature,
    lag = DLNM_LAG,
    argvar = list(fun = "ns", df = TEMP_DF),
    arglag = list(fun = "ns", df = LAG_DF)
  )

  df_cb <- cbind(df_clean, as.data.table(cb_temp))

  # Model 1: Baseline (no random effects)
  m1 <- gam(biomarker ~ cb_temp + season + vulnerability,
            data = df_cb, method = "REML")

  # Model 2: Random intercept
  m2 <- tryCatch({
    gam(biomarker ~ cb_temp + season + vulnerability + s(study_id, bs = "re"),
        data = df_cb, method = "REML")
  }, error = function(e) NULL)

  # Model 3: Random slope
  m3 <- tryCatch({
    gam(biomarker ~ cb_temp + season + vulnerability +
          s(study_id, bs = "re") + s(study_id, temperature, bs = "re"),
        data = df_cb, method = "REML")
  }, error = function(e) NULL)

  # Model E: Study RE only (diagnostic)
  m_e <- tryCatch({
    gam(biomarker ~ s(study_id, bs = "re"),
        data = df_cb, method = "REML")
  }, error = function(e) NULL)

  # Select best model
  models <- list(baseline = m1, random_int = m2, random_slope = m3, study_only = m_e)
  models <- Filter(Negate(is.null), models)

  if (length(models) == 0) {
    return(NULL)
  }

  aics <- sapply(models, AIC)
  best_model <- models[[which.min(aics)]]
  best_name <- names(models)[which.min(aics)]

  # Test DLNM significance
  pred <- crosspred(cb_temp, best_model, at = seq(15, 30, 1), cen = REF_TEMP)
  sig_idx <- which(pred$alllow > 0 | pred$allhigh < 0)

  return(list(
    models = models,
    best_model = best_model,
    best_name = best_name,
    r2 = summary(best_model)$r.sq,
    aic = AIC(best_model),
    n_sig_temps = length(sig_idx),
    pred = pred,
    crossbasis = cb_temp
  ))
}

################################################################################
# MAIN ANALYSIS LOOP
################################################################################

all_results <- list()
comparison_table <- data.table()

for (biomarker_name in BIOMARKERS) {

  cat("\n")
  cat("========================================\n")
  cat(sprintf("BIOMARKER: %s\n", biomarker_name))
  cat("========================================\n")

  # Check if biomarker exists
  if (!biomarker_name %in% names(df_raw)) {
    cat("  Biomarker not found in data. SKIPPING.\n")
    next
  }

  # Prepare data
  df_clean <- df_raw[, .(
    biomarker = get(biomarker_name),
    temperature = climate_7d_mean_temp,
    vulnerability = HEAT_VULNERABILITY_SCORE,
    study_id = as.factor(study_source),
    season = as.factor(season)
  )]

  df_clean <- na.omit(df_clean)

  if (nrow(df_clean) < 100) {
    cat("  Insufficient data (n < 100). SKIPPING.\n")
    next
  }

  cat(sprintf("  N observations: %d\n", nrow(df_clean)))
  cat(sprintf("  N studies: %d\n", uniqueN(df_clean$study_id)))

  # STEP 1: Detect units issues
  cat("\n  === Units Diagnostics ===\n")
  diagnostic <- detect_units_issues(df_clean, biomarker_name)

  cat(sprintf("  ICC: %.3f\n", diagnostic$icc))
  cat(sprintf("  Mean ratio (max/min): %.2f\n", diagnostic$mean_ratio))

  if (diagnostic$has_issues) {
    cat("\n  ⚠️  POTENTIAL ISSUES DETECTED:\n")
    for (issue_name in names(diagnostic$issues)) {
      cat(sprintf("    - %s: %s\n", issue_name, diagnostic$issues[[issue_name]]))
    }
  } else {
    cat("\n  ✓ No obvious units issues detected\n")
  }

  # Show study statistics
  cat("\n  Study-level statistics (BEFORE correction):\n")
  print(diagnostic$study_stats)

  # STEP 2: Apply correction if needed
  df_before <- copy(df_clean)

  if (diagnostic$has_issues) {
    cat("\n  === Applying Corrections ===\n")
    df_after <- apply_units_correction(df_clean, diagnostic)

    # Re-check diagnostics
    diagnostic_after <- detect_units_issues(df_after, biomarker_name)

    cat("\n  Study-level statistics (AFTER correction):\n")
    print(diagnostic_after$study_stats)

    cat(sprintf("\n  ICC change: %.3f → %.3f\n", diagnostic$icc, diagnostic_after$icc))

    if (diagnostic_after$icc < 0.2) {
      cat("  ✓ Correction successful! ICC < 0.2\n")
      df_clean <- df_after
    } else {
      cat("  ⚠️  High ICC persists. Manual review needed.\n")
    }
  } else {
    df_after <- df_clean
    diagnostic_after <- diagnostic
  }

  # STEP 3: Run analysis (BEFORE and AFTER if correction applied)
  cat("\n  === Running Mixed Effects DLNM ===\n")

  if (diagnostic$has_issues) {
    cat("\n  BEFORE correction:\n")
    result_before <- run_dlnm_analysis(df_before, biomarker_name)

    cat("\n  AFTER correction:\n")
    result_after <- run_dlnm_analysis(df_after, biomarker_name)

    # Store comparison
    if (!is.null(result_before) && !is.null(result_after)) {
      comparison_row <- data.table(
        biomarker = biomarker_name,
        n_obs = nrow(df_clean),
        n_studies = uniqueN(df_clean$study_id),
        icc_before = diagnostic$icc,
        icc_after = diagnostic_after$icc,
        r2_before = result_before$r2,
        r2_after = result_after$r2,
        r2_change = result_after$r2 - result_before$r2,
        aic_before = result_before$aic,
        aic_after = result_after$aic,
        best_model_before = result_before$best_name,
        best_model_after = result_after$best_name,
        n_sig_temps_before = result_before$n_sig_temps,
        n_sig_temps_after = result_after$n_sig_temps,
        correction_applied = TRUE
      )

      comparison_table <- rbind(comparison_table, comparison_row)

      cat("\n  === COMPARISON ===\n")
      cat(sprintf("    ICC:  %.3f → %.3f (%.1fx change)\n",
                  comparison_row$icc_before,
                  comparison_row$icc_after,
                  comparison_row$icc_before / comparison_row$icc_after))
      cat(sprintf("    R²:   %.3f → %.3f (%.3f change)\n",
                  comparison_row$r2_before,
                  comparison_row$r2_after,
                  comparison_row$r2_change))
      cat(sprintf("    Sig temps: %d → %d\n",
                  comparison_row$n_sig_temps_before,
                  comparison_row$n_sig_temps_after))
    }

  } else {
    # No correction needed
    result <- run_dlnm_analysis(df_clean, biomarker_name)

    if (!is.null(result)) {
      comparison_row <- data.table(
        biomarker = biomarker_name,
        n_obs = nrow(df_clean),
        n_studies = uniqueN(df_clean$study_id),
        icc_before = diagnostic$icc,
        icc_after = diagnostic$icc,
        r2_before = result$r2,
        r2_after = result$r2,
        r2_change = 0,
        aic_before = result$aic,
        aic_after = result$aic,
        best_model_before = result$best_name,
        best_model_after = result$best_name,
        n_sig_temps_before = result$n_sig_temps,
        n_sig_temps_after = result$n_sig_temps,
        correction_applied = FALSE
      )

      comparison_table <- rbind(comparison_table, comparison_row)

      cat(sprintf("\n  R²: %.3f | Sig temps: %d | Best: %s\n",
                  result$r2, result$n_sig_temps, result$best_name))
    }
  }
}

################################################################################
# SUMMARY
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# COMPREHENSIVE ANALYSIS SUMMARY\n")
cat("################################################################################\n\n")

cat("=== Biomarkers Requiring Correction ===\n")
corrected <- comparison_table[correction_applied == TRUE]
if (nrow(corrected) > 0) {
  print(corrected[, .(biomarker, icc_before, icc_after, r2_before, r2_after)])
} else {
  cat("  None - all biomarkers had consistent units!\n")
}

cat("\n=== Final Results (After Corrections) ===\n")
final_results <- comparison_table[order(-r2_after)]
print(final_results[, .(
  biomarker,
  n_obs,
  n_studies,
  icc = icc_after,
  r2 = r2_after,
  best_model = best_model_after,
  n_sig_temps = n_sig_temps_after,
  correction = correction_applied
)])

# Save results
fwrite(comparison_table, file.path(OUTPUT_DIR, "comprehensive_comparison.csv"))

cat("\n")
cat("=== KEY FINDINGS ===\n")

# Which biomarkers had issues?
n_issues <- sum(comparison_table$correction_applied)
cat(sprintf("Biomarkers with units issues: %d/%d\n", n_issues, nrow(comparison_table)))

# Biggest ICC changes
if (n_issues > 0) {
  biggest_change <- corrected[which.max(abs(icc_after - icc_before))]
  cat(sprintf("\nBiggest ICC change: %s\n", biggest_change$biomarker))
  cat(sprintf("  ICC: %.3f → %.3f (%.1fx reduction)\n",
              biggest_change$icc_before,
              biggest_change$icc_after,
              biggest_change$icc_before / biggest_change$icc_after))
  cat(sprintf("  R²: %.3f → %.3f\n",
              biggest_change$r2_before,
              biggest_change$r2_after))
}

# Best biomarkers (corrected)
cat("\n=== Top Climate-Sensitive Biomarkers (Corrected) ===\n")
top3 <- final_results[1:min(3, nrow(final_results))]
for (i in 1:nrow(top3)) {
  row <- top3[i]
  cat(sprintf("%d. %s: R² = %.3f (%d sig temps)%s\n",
              i,
              row$biomarker,
              row$r2,
              row$n_sig_temps,
              ifelse(row$correction, " [corrected]", "")))
}

cat(sprintf("\n✓ Results saved: %s/comprehensive_comparison.csv\n", OUTPUT_DIR))
cat("\n=== Analysis Complete ===\n")
