################################################################################
# PATIENT-LEVEL INTERACTION ANALYSIS: Temperature × Vulnerability
################################################################################
#
# PURPOSE: Test whether vulnerability modifies climate effects at patient level
#          to validate SHAP findings showing vulnerability importance
#
# APPROACH: Mixed effects models with Temperature×Vulnerability interactions
#           using ALL patient-level data (n=thousands) for adequate power
#
# WHY THIS VALIDATES SHAP:
# - SHAP analyzes patient-level predictions
# - If SHAP shows vulnerability modifies climate effects, we should detect
#   significant Temperature×Vulnerability interactions in statistical models
# - Patient-level analysis (n=thousands) provides adequate power
#
# Date: 2025-10-30
################################################################################

library(data.table)
library(mgcv)
library(dlnm)
library(ggplot2)
library(lme4)
library(lmerTest)

# Paths
DATA_FILE <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/patient_level_interactions"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("################################################################################\n")
cat("# PATIENT-LEVEL INTERACTION ANALYSIS\n")
cat("################################################################################\n\n")

################################################################################
# 1. LOAD DATA
################################################################################

df <- fread(DATA_FILE)

# Define biomarkers showing interesting patterns
biomarkers <- list(
  # VULNERABILITY PARADOX (strong correlations at study level)
  cholesterol = list(
    col = "total_cholesterol_mg_dL",
    name = "Total Cholesterol",
    units = "mg/dL",
    expected_pattern = "PARADOX (high vuln → weak effect)"
  ),
  glucose = list(
    col = "fasting_glucose_mmol_L",
    name = "Glucose",
    units = "mmol/L",
    expected_pattern = "PARADOX (high vuln → weak effect)"
  ),
  body_temp = list(
    col = "body_temperature_celsius",
    name = "Body Temperature",
    units = "°C",
    expected_pattern = "PARADOX (r=-0.996 at study level)"
  ),
  # EXPECTED PATTERN (opposite of paradox)
  cd4 = list(
    col = "CD4 cell count (cells/µL)",
    name = "CD4 Count",
    units = "cells/µL",
    expected_pattern = "EXPECTED (high vuln → strong effect)"
  ),
  # HIGH POWER CANDIDATES
  bmi = list(
    col = "BMI (kg/m²)",
    name = "BMI",
    units = "kg/m²",
    expected_pattern = "UNKNOWN (7 studies available)"
  ),
  hemoglobin = list(
    col = "hemoglobin_g_dL",
    name = "Hemoglobin",
    units = "g/dL",
    expected_pattern = "UNKNOWN (4 studies available)"
  )
)

cat("Biomarkers selected for patient-level interaction analysis:\n")
for (bio_name in names(biomarkers)) {
  cat(sprintf("  - %s: %s\n",
              biomarkers[[bio_name]]$name,
              biomarkers[[bio_name]]$expected_pattern))
}
cat("\n")

################################################################################
# 2. ANALYSIS FUNCTION
################################################################################

analyze_patient_level_interactions <- function(biomarker_config, df_full) {

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
    year = year,
    month = month
  )]

  df_bio <- na.omit(df_bio)
  df_bio[, biomarker := biomarker_raw]

  # Standardize predictors for interpretability
  df_bio[, temp_z := scale(temperature)]
  df_bio[, vuln_z := scale(vulnerability)]

  n_total <- nrow(df_bio)
  n_studies <- uniqueN(df_bio$study_id)

  cat(sprintf("Total patients: %d\n", n_total))
  cat(sprintf("Studies: %d\n", n_studies))
  cat(sprintf("Mean biomarker: %.2f %s (SD=%.2f)\n",
              mean(df_bio$biomarker), biomarker_config$units, sd(df_bio$biomarker)))
  cat(sprintf("Temperature range: %.1f - %.1f°C\n",
              min(df_bio$temperature), max(df_bio$temperature)))
  cat(sprintf("Vulnerability range: %.1f - %.1f\n\n",
              min(df_bio$vulnerability), max(df_bio$vulnerability)))

  if (n_total < 500) {
    cat("  ⚠️  Insufficient data (n < 500). Skipping.\n")
    return(NULL)
  }

  cat("--- Model Progression ---\n\n")

  # Model 1: Temperature only (baseline)
  cat("Model 1: Temperature only\n")
  m1 <- lmer(biomarker ~ temp_z + (1|study_id),
             data = df_bio, REML = TRUE)
  r2_m1 <- as.numeric(MuMIn::r.squaredGLMM(m1)[1])
  aic_m1 <- AIC(m1)
  temp_p_m1 <- summary(m1)$coefficients["temp_z", "Pr(>|t|)"]
  cat(sprintf("  R² = %.4f, AIC = %.1f, Temp p = %.4f\n\n", r2_m1, aic_m1, temp_p_m1))

  # Model 2: + Vulnerability (main effect)
  cat("Model 2: + Vulnerability (main effect)\n")
  m2 <- lmer(biomarker ~ temp_z + vuln_z + (1|study_id),
             data = df_bio, REML = TRUE)
  r2_m2 <- as.numeric(MuMIn::r.squaredGLMM(m2)[1])
  aic_m2 <- AIC(m2)
  vuln_p_m2 <- summary(m2)$coefficients["vuln_z", "Pr(>|t|)"]
  cat(sprintf("  R² = %.4f, AIC = %.1f, Vuln p = %.4f\n", r2_m2, aic_m2, vuln_p_m2))
  cat(sprintf("  ΔR² = %.4f, ΔAIC = %.1f\n\n", r2_m2 - r2_m1, aic_m2 - aic_m1))

  # Model 3: + Season
  cat("Model 3: + Season\n")
  m3 <- lmer(biomarker ~ temp_z + vuln_z + season + (1|study_id),
             data = df_bio, REML = TRUE)
  r2_m3 <- as.numeric(MuMIn::r.squaredGLMM(m3)[1])
  aic_m3 <- AIC(m3)
  cat(sprintf("  R² = %.4f, AIC = %.1f\n", r2_m3, aic_m3))
  cat(sprintf("  ΔR² = %.4f, ΔAIC = %.1f\n\n", r2_m3 - r2_m2, aic_m3 - aic_m2))

  # Model 4: + Temperature × Vulnerability INTERACTION (KEY TEST)
  cat("Model 4: + Temperature × Vulnerability INTERACTION *** KEY TEST ***\n")
  m4 <- lmer(biomarker ~ temp_z * vuln_z + season + (1|study_id),
             data = df_bio, REML = TRUE)
  r2_m4 <- as.numeric(MuMIn::r.squaredGLMM(m4)[1])
  aic_m4 <- AIC(m4)

  # Extract interaction term
  coef_summary <- summary(m4)$coefficients
  interaction_coef <- coef_summary["temp_z:vuln_z", "Estimate"]
  interaction_se <- coef_summary["temp_z:vuln_z", "Std. Error"]
  interaction_t <- coef_summary["temp_z:vuln_z", "t value"]
  interaction_p <- coef_summary["temp_z:vuln_z", "Pr(>|t|)"]

  cat(sprintf("  R² = %.4f, AIC = %.1f\n", r2_m4, aic_m4))
  cat(sprintf("  ΔR² = %.4f, ΔAIC = %.1f\n\n", r2_m4 - r2_m3, aic_m4 - aic_m3))

  cat("  INTERACTION TERM (Temp × Vulnerability):\n")
  cat(sprintf("    Coefficient = %.6f (SE = %.6f)\n", interaction_coef, interaction_se))
  cat(sprintf("    t = %.3f, p = %.4f\n", interaction_t, interaction_p))

  # Interpret
  if (interaction_p < 0.001) {
    cat("    ✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)\n")
    sig_level <- "***"
  } else if (interaction_p < 0.01) {
    cat("    ✓✓ VERY SIGNIFICANT (p < 0.01)\n")
    sig_level <- "**"
  } else if (interaction_p < 0.05) {
    cat("    ✓ SIGNIFICANT (p < 0.05)\n")
    sig_level <- "*"
  } else if (interaction_p < 0.10) {
    cat("    ⚠️  MARGINALLY SIGNIFICANT (p < 0.10)\n")
    sig_level <- "†"
  } else {
    cat("    → Not significant (p ≥ 0.10)\n")
    sig_level <- "ns"
  }

  # Direction interpretation
  cat("\n  DIRECTION:\n")
  if (interaction_coef < 0) {
    cat("    Negative interaction: Higher vulnerability → WEAKER temperature effects\n")
    cat("    → PARADOX pattern confirmed at patient level\n")
  } else {
    cat("    Positive interaction: Higher vulnerability → STRONGER temperature effects\n")
    cat("    → EXPECTED pattern confirmed at patient level\n")
  }

  # Likelihood ratio test
  cat("\n  LIKELIHOOD RATIO TEST:\n")
  lr_test <- anova(m3, m4)
  lr_p <- lr_test[["Pr(>Chisq)"]][2]
  cat(sprintf("    χ² = %.3f, df = 1, p = %.4f\n", lr_test[["Chisq"]][2], lr_p))

  if (lr_p < 0.05) {
    cat("    ✓ Model with interaction significantly better (p < 0.05)\n")
  } else {
    cat("    → Model with interaction not significantly better\n")
  }

  # Compare with expected pattern
  cat(sprintf("\n  EXPECTED PATTERN: %s\n", biomarker_config$expected_pattern))
  if (grepl("PARADOX", biomarker_config$expected_pattern)) {
    if (interaction_coef < 0 & interaction_p < 0.05) {
      cat("  ✓✓ CONFIRMED: Paradox detected at patient level\n")
    } else if (interaction_coef < 0) {
      cat("  ⚠️  Direction consistent but not significant\n")
    } else {
      cat("  ✗ CONTRADICTION: Expected paradox, found opposite\n")
    }
  } else if (grepl("EXPECTED", biomarker_config$expected_pattern)) {
    if (interaction_coef > 0 & interaction_p < 0.05) {
      cat("  ✓✓ CONFIRMED: Expected pattern detected at patient level\n")
    } else if (interaction_coef > 0) {
      cat("  ⚠️  Direction consistent but not significant\n")
    } else {
      cat("  ✗ CONTRADICTION: Expected positive interaction, found negative\n")
    }
  }

  # Effect size: What does 1 SD change in vulnerability do to temperature effect?
  cat("\n  EFFECT SIZE:\n")
  temp_effect_low_vuln <- coef_summary["temp_z", "Estimate"] - interaction_coef
  temp_effect_high_vuln <- coef_summary["temp_z", "Estimate"] + interaction_coef
  cat(sprintf("    Temp effect at LOW vulnerability (-1 SD): %.4f\n", temp_effect_low_vuln))
  cat(sprintf("    Temp effect at HIGH vulnerability (+1 SD): %.4f\n", temp_effect_high_vuln))
  cat(sprintf("    Ratio (High/Low): %.2f\n", temp_effect_high_vuln / (temp_effect_low_vuln + 1e-10)))

  # Store results
  result <- data.table(
    biomarker = biomarker_name,
    n = n_total,
    k_studies = n_studies,
    r2_baseline = r2_m1,
    r2_with_vuln = r2_m2,
    r2_with_season = r2_m3,
    r2_with_interaction = r2_m4,
    delta_r2_interaction = r2_m4 - r2_m3,
    aic_baseline = aic_m1,
    aic_interaction = aic_m4,
    interaction_coef = interaction_coef,
    interaction_se = interaction_se,
    interaction_t = interaction_t,
    interaction_p = interaction_p,
    interaction_sig = sig_level,
    lr_test_p = lr_p,
    temp_effect_low_vuln = temp_effect_low_vuln,
    temp_effect_high_vuln = temp_effect_high_vuln,
    expected_pattern = biomarker_config$expected_pattern
  )

  # Create visualization
  cat("\n--- Creating Visualization ---\n")

  # Predict at low, medium, high vulnerability
  pred_data <- expand.grid(
    temp_z = seq(-2, 2, 0.1),
    vuln_z = c(-1, 0, 1),  # Low, Medium, High
    season = levels(df_bio$season)[1],
    study_id = levels(df_bio$study_id)[1]
  )
  pred_data$vuln_label <- factor(pred_data$vuln_z,
                                  levels = c(-1, 0, 1),
                                  labels = c("Low Vuln (-1 SD)", "Medium Vuln (Mean)", "High Vuln (+1 SD)"))

  # Get predictions
  pred_data$predicted <- predict(m4, newdata = pred_data, re.form = NA)

  # Convert back to original scale for x-axis
  temp_mean <- mean(df_bio$temperature)
  temp_sd <- sd(df_bio$temperature)
  pred_data$temperature <- pred_data$temp_z * temp_sd + temp_mean

  # Plot
  p <- ggplot(pred_data, aes(x = temperature, y = predicted, color = vuln_label, linetype = vuln_label)) +
    geom_line(size = 1.2) +
    labs(
      title = sprintf("%s: Temperature × Vulnerability Interaction", biomarker_name),
      subtitle = sprintf("p = %.4f %s | n = %d patients, k = %d studies",
                        interaction_p, sig_level, n_total, n_studies),
      x = "7-Day Mean Temperature (°C)",
      y = sprintf("%s (%s)", biomarker_name, biomarker_config$units),
      color = "Vulnerability Level",
      linetype = "Vulnerability Level"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    ) +
    scale_color_manual(values = c("Low Vuln (-1 SD)" = "#2166ac",
                                   "Medium Vuln (Mean)" = "#4d4d4d",
                                   "High Vuln (+1 SD)" = "#b2182b"))

  # Add interpretation text
  if (interaction_coef < 0) {
    p <- p + annotate("text", x = Inf, y = Inf,
                     label = "← Negative interaction (PARADOX)",
                     hjust = 1.1, vjust = 1.5, size = 4, fontface = "italic")
  } else {
    p <- p + annotate("text", x = Inf, y = Inf,
                     label = "← Positive interaction (EXPECTED)",
                     hjust = 1.1, vjust = 1.5, size = 4, fontface = "italic")
  }

  # Save
  bio_name_clean <- gsub(" ", "_", tolower(biomarker_name))
  ggsave(file.path(OUTPUT_DIR, sprintf("%s_interaction_plot.pdf", bio_name_clean)),
         p, width = 10, height = 7)
  ggsave(file.path(OUTPUT_DIR, sprintf("%s_interaction_plot.png", bio_name_clean)),
         p, width = 10, height = 7, dpi = 300)

  cat(sprintf("  Saved: %s_interaction_plot.pdf/png\n", bio_name_clean))

  return(result)
}

################################################################################
# 3. RUN ANALYSES
################################################################################

cat("\n=== Running Patient-Level Interaction Analyses ===\n")

all_results <- data.table()

for (bio_name in names(biomarkers)) {
  results <- analyze_patient_level_interactions(biomarkers[[bio_name]], df)
  if (!is.null(results)) {
    all_results <- rbind(all_results, results)
  }
}

# Save
fwrite(all_results, file.path(OUTPUT_DIR, "patient_level_interaction_results.csv"))

################################################################################
# 4. SUMMARY
################################################################################

cat("\n================================================================================\n")
cat("# SUMMARY: PATIENT-LEVEL INTERACTION ANALYSIS\n")
cat("================================================================================\n\n")

cat("SAMPLE SIZES:\n")
print(all_results[, .(biomarker, n, k_studies)])

cat("\n\nMODEL PERFORMANCE:\n")
print(all_results[, .(biomarker, r2_baseline, r2_with_interaction, delta_r2_interaction)])

cat("\n\nINTERACTION TERMS (Temperature × Vulnerability):\n")
print(all_results[, .(biomarker, interaction_coef, interaction_se, interaction_p, interaction_sig)])

cat("\n\nSIGNIFICANT INTERACTIONS (p < 0.05):\n")
sig_results <- all_results[interaction_p < 0.05]
if (nrow(sig_results) > 0) {
  print(sig_results[, .(biomarker, n, interaction_coef, interaction_p,
                       temp_effect_low_vuln, temp_effect_high_vuln)])
  cat("\n  ✓✓ SIGNIFICANT patient-level interactions detected!\n")
  cat("\n  INTERPRETATION:\n")
  for (i in 1:nrow(sig_results)) {
    bio <- sig_results$biomarker[i]
    coef <- sig_results$interaction_coef[i]
    p <- sig_results$interaction_p[i]
    pattern <- sig_results$expected_pattern[i]

    cat(sprintf("  - %s (p=%.4f): ", bio, p))
    if (coef < 0) {
      cat("Paradox confirmed - high vulnerability → weaker effects\n")
    } else {
      cat("Expected pattern - high vulnerability → stronger effects\n")
    }
    cat(sprintf("    Expected: %s\n", pattern))
  }
} else {
  cat("  → No significant interactions at p < 0.05\n")
}

cat("\n\nMARGINAL INTERACTIONS (p < 0.10):\n")
marginal_results <- all_results[interaction_p >= 0.05 & interaction_p < 0.10]
if (nrow(marginal_results) > 0) {
  print(marginal_results[, .(biomarker, n, interaction_coef, interaction_p)])
  cat("\n  ⚠️  Marginal evidence for interactions\n")
} else {
  cat("  → No marginal interactions\n")
}

cat("\n\nDIRECTIONAL CONSISTENCY:\n")
paradox_biomarkers <- all_results[grepl("PARADOX", expected_pattern)]
if (nrow(paradox_biomarkers) > 0) {
  n_paradox_confirmed <- sum(paradox_biomarkers$interaction_coef < 0)
  cat(sprintf("  Paradox biomarkers: %d/%d show negative interactions (%.0f%%)\n",
              n_paradox_confirmed, nrow(paradox_biomarkers),
              100 * n_paradox_confirmed / nrow(paradox_biomarkers)))
}

expected_biomarkers <- all_results[grepl("EXPECTED", expected_pattern)]
if (nrow(expected_biomarkers) > 0) {
  n_expected_confirmed <- sum(expected_biomarkers$interaction_coef > 0)
  cat(sprintf("  Expected pattern biomarkers: %d/%d show positive interactions (%.0f%%)\n",
              n_expected_confirmed, nrow(expected_biomarkers),
              100 * n_expected_confirmed / nrow(expected_biomarkers)))
}

cat("\n\nCOMPARISON WITH STUDY-LEVEL META-REGRESSION:\n")
cat("  Study-level approach:\n")
cat("    - Sample size: k=3-7 studies\n")
cat("    - Power: INSUFFICIENT (need k≥7 for r=0.80)\n")
cat("    - Results: Strong correlations but NOT significant\n")
cat("\n  Patient-level approach:\n")
cat("    - Sample size: n=thousands of patients\n")
cat("    - Power: ADEQUATE for moderate-to-large effects\n")
cat("    - Results: See above\n")

cat("\n\nVALIDATION OF SHAP FINDINGS:\n")
cat("  SHAP analysis shows vulnerability as important feature\n")
cat("  Patient-level interactions test if vulnerability MODIFIES climate effects\n")
cat("\n  If significant interactions found:\n")
cat("    ✓ SHAP findings validated - vulnerability truly modifies climate effects\n")
cat("  If no significant interactions:\n")
cat("    → SHAP may detect main effects, not interactions\n")
cat("    → Or interactions too small for statistical detection\n")
cat("    → Or SHAP detecting other vulnerability-related patterns\n")

cat("\n\n=== Analysis Complete ===\n")
cat(sprintf("Results saved to: %s\n", OUTPUT_DIR))
cat("Files generated:\n")
cat("  - patient_level_interaction_results.csv\n")
cat("  - [biomarker]_interaction_plot.pdf/png (for each biomarker)\n")
