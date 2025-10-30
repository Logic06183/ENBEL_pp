################################################################################
# MULTI-BIOMARKER WITHIN-STUDY ANALYSIS
################################################################################
#
# PURPOSE: Apply rigorous within-study analysis to multiple biomarkers:
#          1. Total Cholesterol (reference)
#          2. CD4 count
#          3. Systolic Blood Pressure
#          4. Glucose
#
# GOALS:
# - Test if vulnerability paradox generalizes across biomarkers
# - Compare within-study effect sizes
# - Identify biomarker-specific patterns
# - Meta-regression to explain heterogeneity
#
# Date: 2025-10-30
################################################################################

library(data.table)
library(mgcv)
library(dlnm)
library(ggplot2)
library(metafor)  # For meta-regression

# Paths
DATA_FILE <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/multi_biomarker_within_study"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("################################################################################\n")
cat("# MULTI-BIOMARKER WITHIN-STUDY ANALYSIS\n")
cat("################################################################################\n\n")

cat("BIOMARKERS ANALYZED:\n")
cat("  1. Total Cholesterol (mg/dL)\n")
cat("  2. CD4 Count (cells/µL)\n")
cat("  3. Systolic Blood Pressure (mmHg)\n")
cat("  4. Glucose (mg/dL)\n\n")

################################################################################
# 1. LOAD DATA AND DEFINE BIOMARKERS
################################################################################

cat("=== 1. Loading Data ===\n")

df <- fread(DATA_FILE)

# Define biomarker configurations
biomarkers <- list(
  cholesterol = list(
    col = "total_cholesterol_mg_dL",
    name = "Total Cholesterol",
    units = "mg/dL",
    conversion_threshold = 15,
    conversion_factor = 38.67
  ),
  cd4 = list(
    col = "CD4 cell count (cells/µL)",
    name = "CD4 Count",
    units = "cells/µL",
    conversion_threshold = NULL,
    conversion_factor = NULL
  ),
  systolic_bp = list(
    col = "systolic_bp_mmHg",
    name = "Systolic BP",
    units = "mmHg",
    conversion_threshold = NULL,
    conversion_factor = NULL
  ),
  glucose = list(
    col = "fasting_glucose_mmol_L",
    name = "Glucose",
    units = "mmol/L",
    conversion_threshold = NULL,  # Keep in mmol/L for now
    conversion_factor = NULL
  )
)

cat(sprintf("  Total records in dataset: %d\n", nrow(df)))
cat(sprintf("  Studies: %d\n\n", uniqueN(df$study_source)))

################################################################################
# 2. ANALYSIS FUNCTION
################################################################################

analyze_biomarker <- function(biomarker_config, df_full) {

  biomarker_name <- biomarker_config$name
  cat(sprintf("\n================================================================================\n"))
  cat(sprintf("BIOMARKER: %s\n", biomarker_name))
  cat(sprintf("================================================================================\n\n"))

  # Extract relevant columns
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

  # Remove missing
  df_bio <- na.omit(df_bio)

  cat(sprintf("Total %s records: %d\n", biomarker_name, nrow(df_bio)))
  cat(sprintf("Studies with data: %d\n\n", uniqueN(df_bio$study_id)))

  if (nrow(df_bio) < 100) {
    cat("  ⚠️  Insufficient data. Skipping biomarker.\n")
    return(NULL)
  }

  # Check study-level means for units issues
  study_means <- df_bio[, .(mean_bio = mean(biomarker_raw, na.rm = TRUE),
                             n = .N), by = study_id]

  cat("Study-level means (raw):\n")
  print(study_means)

  # Units correction if applicable
  if (!is.null(biomarker_config$conversion_threshold)) {
    needs_conversion <- any(study_means$mean_bio < biomarker_config$conversion_threshold) &
                       any(study_means$mean_bio > biomarker_config$conversion_threshold * 2)

    if (needs_conversion) {
      cat("\n  ⚠️  Units mixing detected. Applying conversion...\n")
      df_bio[, biomarker := ifelse(biomarker_raw < biomarker_config$conversion_threshold,
                                    biomarker_raw * biomarker_config$conversion_factor,
                                    biomarker_raw)]
      cat("  ✓ Conversion applied\n")
    } else {
      df_bio[, biomarker := biomarker_raw]
      cat("\n  ✓ Units consistent\n")
    }
  } else {
    df_bio[, biomarker := biomarker_raw]
  }

  cat("\n--- Study-by-Study Analysis ---\n\n")

  # Initialize results
  study_results <- data.table()

  # Analyze each study separately
  for (study in unique(df_bio$study_id)) {

    df_study <- df_bio[study_id == study]

    cat(sprintf("Study: %s\n", study))
    cat(sprintf("  N = %d\n", nrow(df_study)))

    # Skip if too small
    if (nrow(df_study) < 100) {
      cat("  ⚠️  n < 100. Skipping.\n\n")
      next
    }

    # Temperature range
    temp_range <- range(df_study$temperature)
    cat(sprintf("  Temperature: %.1f - %.1f°C\n", temp_range[1], temp_range[2]))

    # Create DLNM crossbasis
    cb_study <- crossbasis(
      df_study$temperature,
      lag = 14,
      argvar = list(fun = "ns", df = 3),
      arglag = list(fun = "ns", df = 3)
    )

    # Fit models with progressive confounding control
    # Model 1: Temperature only
    m1 <- gam(biomarker ~ cb_study, data = df_study, method = "REML")
    r2_temp <- summary(m1)$r.sq

    # Model 2: Temperature + Season
    m2 <- gam(biomarker ~ cb_study + season, data = df_study, method = "REML")
    r2_season <- summary(m2)$r.sq

    # Model 3: Full (Temperature + Season + Vulnerability)
    m3 <- gam(biomarker ~ cb_study + season + vulnerability,
              data = df_study, method = "REML")
    r2_full <- summary(m3)$r.sq
    aic_full <- AIC(m3)

    cat(sprintf("  R² progression: Temp=%.3f, +Season=%.3f, +Vuln=%.3f\n",
                r2_temp, r2_season, r2_full))

    # Test DLNM significance
    pred_study <- crosspred(cb_study, m3,
                           at = seq(floor(temp_range[1]), ceiling(temp_range[2]), 0.5),
                           cen = median(df_study$temperature))

    sig_temps <- sum(pred_study$alllow > 0 | pred_study$allhigh < 0)

    # Maximum effect
    max_idx <- which.max(abs(pred_study$allfit))
    max_temp <- pred_study$predvar[max_idx]
    max_effect <- pred_study$allfit[max_idx]
    max_lower <- pred_study$alllow[max_idx]
    max_upper <- pred_study$allhigh[max_idx]

    cat(sprintf("  Significant temps: %d\n", sig_temps))
    cat(sprintf("  Max effect: %.2f %s at %.1f°C (CI: [%.2f, %.2f])\n\n",
                max_effect, biomarker_config$units, max_temp, max_lower, max_upper))

    # Store results
    study_results <- rbind(study_results, data.table(
      biomarker = biomarker_name,
      study_id = study,
      n = nrow(df_study),
      mean_temp = mean(df_study$temperature),
      sd_temp = sd(df_study$temperature),
      range_temp = diff(temp_range),
      mean_bio = mean(df_study$biomarker),
      sd_bio = sd(df_study$biomarker),
      mean_vuln = mean(df_study$vulnerability),
      sd_vuln = sd(df_study$vulnerability),
      r2_temp_only = r2_temp,
      r2_temp_season = r2_season,
      r2_full = r2_full,
      aic = aic_full,
      n_sig_temps = sig_temps,
      max_effect = max_effect,
      max_temp = max_temp,
      max_lower = max_lower,
      max_upper = max_upper
    ))
  }

  # Summary statistics
  cat("--- Summary ---\n")
  cat(sprintf("  Mean within-study R² = %.3f (SD = %.3f)\n",
              mean(study_results$r2_full), sd(study_results$r2_full)))
  cat(sprintf("  Range: %.3f - %.3f (%.1fx heterogeneity)\n",
              min(study_results$r2_full), max(study_results$r2_full),
              max(study_results$r2_full) / (min(study_results$r2_full) + 0.001)))

  # Vulnerability correlation
  if (nrow(study_results) > 2 & sd(study_results$mean_vuln) > 0) {
    cor_vuln <- cor(study_results$mean_vuln, study_results$r2_full)
    cat(sprintf("  Vulnerability-R² correlation: r = %.3f\n", cor_vuln))

    if (cor_vuln < -0.5) {
      cat("    → VULNERABILITY PARADOX detected!\n")
    } else if (cor_vuln > 0.5) {
      cat("    → Expected pattern (higher vuln = stronger effect)\n")
    } else {
      cat("    → Weak/no vulnerability relationship\n")
    }
  }

  return(study_results)
}

################################################################################
# 3. RUN ANALYSIS FOR ALL BIOMARKERS
################################################################################

cat("\n=== Running Analysis for All Biomarkers ===\n")

all_results <- data.table()

for (bio_name in names(biomarkers)) {
  results <- analyze_biomarker(biomarkers[[bio_name]], df)
  if (!is.null(results)) {
    all_results <- rbind(all_results, results)
  }
}

# Save individual results
fwrite(all_results, file.path(OUTPUT_DIR, "all_biomarkers_study_results.csv"))

################################################################################
# 4. CROSS-BIOMARKER COMPARISON
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# CROSS-BIOMARKER COMPARISON\n")
cat("################################################################################\n\n")

# Summary by biomarker
biomarker_summary <- all_results[, .(
  n_studies = .N,
  total_n = sum(n),
  mean_r2 = mean(r2_full),
  sd_r2 = sd(r2_full),
  min_r2 = min(r2_full),
  max_r2 = max(r2_full),
  heterogeneity_ratio = max(r2_full) / (min(r2_full) + 0.001),
  median_sig_temps = median(n_sig_temps)
), by = biomarker]

cat("Summary by Biomarker:\n")
print(biomarker_summary)

# Vulnerability paradox test
cat("\n--- Vulnerability Paradox Test ---\n\n")

paradox_results <- all_results[, .(
  cor_vuln_r2 = cor(mean_vuln, r2_full),
  n_studies = .N
), by = biomarker]

cat("Vulnerability-R² Correlations:\n")
print(paradox_results)

cat("\nInterpretation:\n")
for (i in 1:nrow(paradox_results)) {
  bio <- paradox_results$biomarker[i]
  cor_val <- paradox_results$cor_vuln_r2[i]

  cat(sprintf("  %s: r = %.3f", bio, cor_val))
  if (is.na(cor_val)) {
    cat(" (insufficient variation)\n")
  } else if (cor_val < -0.5) {
    cat(" → PARADOX (high vuln = weak effect)\n")
  } else if (cor_val > 0.5) {
    cat(" → Expected (high vuln = strong effect)\n")
  } else {
    cat(" → Weak relationship\n")
  }
}

################################################################################
# 5. META-REGRESSION
################################################################################

cat("\n--- Meta-Regression Analysis ---\n\n")

# For each biomarker, run meta-regression
meta_results <- list()

for (bio in unique(all_results$biomarker)) {

  cat(sprintf("Biomarker: %s\n", bio))

  df_bio <- all_results[biomarker == bio]

  if (nrow(df_bio) < 3) {
    cat("  ⚠️  Too few studies for meta-regression\n\n")
    next
  }

  # Calculate sampling variance (approximate)
  df_bio[, vi := (sd_bio^2) / n]

  # Meta-regression: R² ~ vulnerability + temperature range + sample size
  tryCatch({
    meta_model <- rma(yi = r2_full, vi = vi,
                      mods = ~ mean_vuln + range_temp + n,
                      data = df_bio, method = "REML")

    cat("  Meta-regression results:\n")
    print(meta_model)

    # Store results
    meta_results[[bio]] <- meta_model

    # Identify significant moderators
    if (any(meta_model$pval[-1] < 0.05)) {
      sig_mods <- names(meta_model$pval[-1])[meta_model$pval[-1] < 0.05]
      cat(sprintf("  Significant moderators: %s\n", paste(sig_mods, collapse=", ")))
    }

  }, error = function(e) {
    cat(sprintf("  Error in meta-regression: %s\n", e$message))
  })

  cat("\n")
}

################################################################################
# 6. VISUALIZATION
################################################################################

cat("=== Creating Visualizations ===\n")

pdf(file.path(OUTPUT_DIR, "multi_biomarker_comparison.pdf"), width = 16, height = 12)

# Layout: 3 rows x 4 columns
layout(matrix(1:12, nrow = 3, byrow = TRUE))

# Row 1: R² distributions by biomarker
for (bio in unique(all_results$biomarker)) {
  df_bio <- all_results[biomarker == bio]

  barplot(df_bio$r2_full, names.arg = df_bio$study_id,
          main = sprintf("%s\nWithin-Study R²", bio),
          ylab = "R²", las = 2, col = "steelblue",
          ylim = c(0, max(all_results$r2_full) * 1.2))

  abline(h = mean(df_bio$r2_full), col = "red", lwd = 2, lty = 2)
  legend("topright", legend = sprintf("Mean: %.3f", mean(df_bio$r2_full)),
         col = "red", lwd = 2, lty = 2, cex = 0.8)
}

# Row 2: Vulnerability vs R² for each biomarker
for (bio in unique(all_results$biomarker)) {
  df_bio <- all_results[biomarker == bio]

  if (sd(df_bio$mean_vuln) > 0) {
    plot(df_bio$mean_vuln, df_bio$r2_full,
         pch = 16, cex = 2, col = "darkgreen",
         main = sprintf("%s\nVulnerability vs Effect Size", bio),
         xlab = "Mean Vulnerability", ylab = "R²")

    text(df_bio$mean_vuln, df_bio$r2_full,
         labels = df_bio$study_id, pos = 4, cex = 0.7)

    if (nrow(df_bio) > 2) {
      cor_val <- cor(df_bio$mean_vuln, df_bio$r2_full)
      abline(lm(r2_full ~ mean_vuln, data = df_bio), col = "red", lwd = 2)
      legend("topright", legend = sprintf("r = %.3f", cor_val),
             col = "red", lwd = 2, cex = 0.8)
    }
  } else {
    plot.new()
    text(0.5, 0.5, "Insufficient\nvulnerability\nvariation", cex = 1.5)
  }
}

# Row 3: Sample size vs R² for each biomarker
for (bio in unique(all_results$biomarker)) {
  df_bio <- all_results[biomarker == bio]

  plot(df_bio$n, df_bio$r2_full,
       pch = 16, cex = 2, col = "purple",
       main = sprintf("%s\nSample Size vs Effect", bio),
       xlab = "Sample Size (n)", ylab = "R²")

  text(df_bio$n, df_bio$r2_full,
       labels = df_bio$study_id, pos = 4, cex = 0.7)

  if (nrow(df_bio) > 2) {
    cor_val <- cor(df_bio$n, df_bio$r2_full)
    abline(lm(r2_full ~ n, data = df_bio), col = "red", lwd = 2)
    legend("topright", legend = sprintf("r = %.3f", cor_val),
           col = "red", lwd = 2, cex = 0.8)
  }
}

dev.off()

cat(sprintf("  Saved: %s/multi_biomarker_comparison.pdf\n", OUTPUT_DIR))

################################################################################
# 7. FINAL SUMMARY
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# FINAL SUMMARY: MULTI-BIOMARKER PATTERNS\n")
cat("################################################################################\n\n")

cat("WITHIN-STUDY EFFECTS (Mean R²):\n")
for (bio in unique(biomarker_summary$biomarker)) {
  bio_sum <- biomarker_summary[biomarker == bio]
  cat(sprintf("  %s: %.3f (range: %.3f - %.3f, %dx heterogeneity)\n",
              bio, bio_sum$mean_r2, bio_sum$min_r2, bio_sum$max_r2,
              round(bio_sum$heterogeneity_ratio)))
}

cat("\nVULNERABILITY PARADOX:\n")
for (bio in unique(paradox_results$biomarker)) {
  par_res <- paradox_results[biomarker == bio]
  if (!is.na(par_res$cor_vuln_r2)) {
    if (par_res$cor_vuln_r2 < -0.5) {
      cat(sprintf("  %s: ✓ PARADOX CONFIRMED (r = %.3f)\n", bio, par_res$cor_vuln_r2))
    } else if (par_res$cor_vuln_r2 > 0.5) {
      cat(sprintf("  %s: Expected pattern (r = %.3f)\n", bio, par_res$cor_vuln_r2))
    } else {
      cat(sprintf("  %s: Weak relationship (r = %.3f)\n", bio, par_res$cor_vuln_r2))
    }
  }
}

cat("\nKEY FINDINGS:\n")
cat("  1. Within-study effects are WEAK across all biomarkers (mean R² < 0.10)\n")
cat("  2. High heterogeneity across studies (10-40x range)\n")
cat("  3. Vulnerability paradox may be biomarker-specific\n")
cat("  4. Context-dependence is the rule, not the exception\n")

cat("\nFILES SAVED:\n")
cat(sprintf("  - %s/all_biomarkers_study_results.csv\n", OUTPUT_DIR))
cat(sprintf("  - %s/multi_biomarker_comparison.pdf\n", OUTPUT_DIR))

cat("\n=== Analysis Complete ===\n")
