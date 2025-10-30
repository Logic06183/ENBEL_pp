#!/usr/bin/env Rscript
################################################################################
# Comprehensive Mixed Effects DLNM Analysis
################################################################################
#
# Purpose: Test statistical rigor of climate-biomarker relationships using
#          hierarchical DLNM models with study-level random effects
#
# Key Innovation: Accounting for study-level heterogeneity while examining
#                 non-linear temperature-lag effects on biomarkers
#
# Author: Claude + Craig Saunders
# Date: 2025-10-30
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)
  library(dlnm)
  library(splines)
  library(ggplot2)
  library(dplyr)
  library(viridis)
})

set.seed(42)

# Configuration
DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/mixed_effects_dlnm"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# DLNM parameters (simplified for speed)
DLNM_LAG <- 14
TEMP_DF <- 3
LAG_DF <- 3
REF_TEMP <- 18

# Biomarkers to analyze
BIOMARKERS <- c(
  "Hematocrit (%)",
  "total_cholesterol_mg_dL",
  "hdl_cholesterol_mg_dL",
  "ldl_cholesterol_mg_dL",
  "fasting_glucose_mmol_L",
  "creatinine_umol_L"
)

################################################################################
# HELPER FUNCTIONS
################################################################################

prepare_biomarker_data <- function(df, biomarker_name) {
  cat(sprintf("\n=== %s ===\n", biomarker_name))

  df_clean <- df[, .(
    biomarker = get(biomarker_name),
    temperature = climate_7d_mean_temp,
    vulnerability = HEAT_VULNERABILITY_SCORE,
    study_id = as.factor(study_source),
    patient_id = paste0(study_source, "_", anonymous_patient_id),
    season = as.factor(season),
    date = as.Date(primary_date)
  )]

  df_clean <- na.omit(df_clean)

  cat(sprintf("  N observations: %d\n", nrow(df_clean)))
  cat(sprintf("  N studies: %d\n", uniqueN(df_clean$study_id)))
  cat(sprintf("  Temperature range: %.1f - %.1f°C\n",
              min(df_clean$temperature), max(df_clean$temperature)))

  return(df_clean)
}

fit_models <- function(df_clean, biomarker_name) {
  # Create crossbasis
  cb_temp <- crossbasis(
    df_clean$temperature,
    lag = DLNM_LAG,
    argvar = list(fun = "ns", df = TEMP_DF),
    arglag = list(fun = "ns", df = LAG_DF)
  )

  df_cb <- cbind(df_clean, as.data.table(cb_temp))

  # Model 1: No random effects (baseline)
  cat("  Fitting baseline model...\n")
  m1 <- gam(
    biomarker ~ cb_temp + season + vulnerability,
    data = df_cb,
    method = "REML"
  )

  # Model 2: Random intercept by study
  cat("  Fitting random effects model...\n")
  m2 <- gam(
    biomarker ~ cb_temp + season + vulnerability + s(study_id, bs = "re"),
    data = df_cb,
    method = "REML"
  )

  # Model 3: Random slope for temperature by study
  cat("  Fitting random slope model...\n")
  m3 <- tryCatch({
    gam(
      biomarker ~ cb_temp + season + vulnerability +
        s(study_id, bs = "re") +
        s(study_id, temperature, bs = "re"),
      data = df_cb,
      method = "REML"
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    return(NULL)
  })

  # Compare models
  comparison <- data.table(
    model = c("Baseline (no RE)", "Random intercept", "Random slope"),
    aic = c(AIC(m1), AIC(m2), if(!is.null(m3)) AIC(m3) else NA),
    bic = c(BIC(m1), BIC(m2), if(!is.null(m3)) BIC(m3) else NA),
    r2 = c(summary(m1)$r.sq, summary(m2)$r.sq,
           if(!is.null(m3)) summary(m3)$r.sq else NA),
    dev_expl = c(summary(m1)$dev.expl * 100,
                 summary(m2)$dev.expl * 100,
                 if(!is.null(m3)) summary(m3)$dev.expl * 100 else NA)
  )

  comparison <- na.omit(comparison)
  comparison[, aic_delta := aic - min(aic)]

  cat("\n  Model Comparison:\n")
  print(comparison[order(aic)])

  # Select best model
  best_idx <- which.min(comparison$aic)
  best_model <- switch(comparison$model[best_idx],
                       "Baseline (no RE)" = m1,
                       "Random intercept" = m2,
                       "Random slope" = m3)

  return(list(
    models = list(m1 = m1, m2 = m2, m3 = m3),
    comparison = comparison,
    best_model = best_model,
    crossbasis = cb_temp
  ))
}

test_dlnm_significance <- function(result, biomarker_name) {
  cat("\n  Testing DLNM significance...\n")

  best_model <- result$best_model
  cb_temp <- result$crossbasis

  # Generate predictions
  pred <- crosspred(
    cb_temp,
    best_model,
    at = seq(15, 30, by = 0.5),
    cen = REF_TEMP
  )

  # Check if confidence intervals cross zero
  temp_range <- pred$predvar
  effects <- pred$allfit
  lower_ci <- pred$alllow
  upper_ci <- pred$allhigh

  # Find temperatures with significant effects
  sig_idx <- which(lower_ci > 0 | upper_ci < 0)

  if (length(sig_idx) > 0) {
    cat(sprintf("    ✓ Significant effects detected at %d temperatures\n", length(sig_idx)))
    cat(sprintf("    Temperature range: %.1f - %.1f°C\n",
                min(temp_range[sig_idx]), max(temp_range[sig_idx])))

    # Maximum effect
    max_effect_idx <- which.max(abs(effects))
    cat(sprintf("    Maximum effect: %.3f at %.1f°C (95%% CI: [%.3f, %.3f])\n",
                effects[max_effect_idx],
                temp_range[max_effect_idx],
                lower_ci[max_effect_idx],
                upper_ci[max_effect_idx]))
  } else {
    cat("    ✗ No significant effects detected\n")
  }

  return(list(
    pred = pred,
    significant = length(sig_idx) > 0,
    n_sig_temps = length(sig_idx)
  ))
}

visualize_results <- function(result, sig_test, biomarker_name) {
  pred <- sig_test$pred

  # Create output filename
  filename <- gsub("[^A-Za-z0-9]", "_", biomarker_name)
  filepath <- file.path(OUTPUT_DIR, sprintf("%s_results.pdf", filename))

  pdf(filepath, width = 14, height = 10)

  # Plot 1: Cumulative exposure-response
  df_plot1 <- data.frame(
    temperature = pred$predvar,
    effect = pred$allfit,
    lower = pred$alllow,
    upper = pred$allhigh
  )

  p1 <- ggplot(df_plot1, aes(x = temperature, y = effect)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "skyblue", alpha = 0.3) +
    geom_line(color = "darkblue", size = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_vline(xintercept = REF_TEMP, linetype = "dotted", color = "gray50") +
    labs(
      title = sprintf("%s: Cumulative Temperature Effect (0-%d day lag)", biomarker_name, DLNM_LAG),
      subtitle = sprintf("R² = %.3f | %s",
                         result$comparison[which.min(result$comparison$aic)]$r2,
                         result$comparison[which.min(result$comparison$aic)]$model),
      x = "Temperature (°C)",
      y = "Effect on Biomarker (relative to 18°C)"
    ) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"))

  # Plot 2: 3D temperature-lag surface
  df_plot2 <- expand.grid(
    lag = 0:DLNM_LAG,
    temperature = pred$predvar
  )
  df_plot2$effect <- as.vector(pred$matfit)

  p2 <- ggplot(df_plot2, aes(x = temperature, y = lag, fill = effect)) +
    geom_tile() +
    scale_fill_viridis(option = "plasma", name = "Effect") +
    labs(
      title = "Temperature-Lag Surface",
      x = "Temperature (°C)",
      y = "Lag (days)"
    ) +
    theme_minimal(base_size = 14)

  # Plot 3: Lag-specific effects
  lags_to_plot <- c(0, 7, 14)
  df_plot3 <- lapply(lags_to_plot, function(lag_val) {
    data.frame(
      temperature = pred$predvar,
      effect = pred$matfit[, lag_val + 1],
      lower = pred$matlow[, lag_val + 1],
      upper = pred$mathigh[, lag_val + 1],
      lag = paste0("Lag ", lag_val, " days")
    )
  })
  df_plot3 <- do.call(rbind, df_plot3)

  p3 <- ggplot(df_plot3, aes(x = temperature, y = effect, color = lag, fill = lag)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
    geom_line(size = 1) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_color_viridis(discrete = TRUE, option = "viridis") +
    scale_fill_viridis(discrete = TRUE, option = "viridis") +
    labs(
      title = "Lag-Specific Effects",
      x = "Temperature (°C)",
      y = "Effect on Biomarker",
      color = "Lag",
      fill = "Lag"
    ) +
    theme_minimal(base_size = 14)

  # Plot 4: Model comparison
  p4 <- ggplot(result$comparison, aes(x = reorder(model, aic), y = aic_delta)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_text(aes(label = sprintf("R²=%.3f", r2)), vjust = -0.5, size = 4) +
    labs(
      title = "Model Comparison (ΔAIC from Best Model)",
      x = "Model",
      y = "ΔAIC (lower is better)"
    ) +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))

  # Combine plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)

  dev.off()

  cat(sprintf("  Saved: %s\n", filepath))
}

extract_random_effects <- function(result, biomarker_name) {
  best_model <- result$best_model

  # Check if model has random effects
  if ("s(study_id)" %in% names(best_model$smooth)) {
    re <- ranef(best_model)

    if (length(re) > 0) {
      cat("\n  Study-Level Random Effects:\n")

      for (i in seq_along(re)) {
        cat(sprintf("    %s: Mean = %.4f, SD = %.4f\n",
                    names(re)[i], mean(re[[i]]), sd(re[[i]])))
      }

      # Save random effects
      re_df <- data.frame(
        study_id = names(re[[1]]),
        random_effect = re[[1]]
      )

      filename <- gsub("[^A-Za-z0-9]", "_", biomarker_name)
      filepath <- file.path(OUTPUT_DIR, sprintf("%s_random_effects.csv", filename))
      fwrite(re_df, filepath)

      cat(sprintf("  Saved random effects: %s\n", filepath))
    }
  }
}

################################################################################
# MAIN ANALYSIS
################################################################################

main_analysis <- function() {
  cat("\n")
  cat("################################################################################\n")
  cat("# Comprehensive Mixed Effects DLNM Analysis\n")
  cat("################################################################################\n\n")

  # Load data
  cat("Loading data...\n")
  df <- fread(DATA_PATH)
  cat(sprintf("Loaded %d observations\n", nrow(df)))

  # Store results
  all_results <- list()
  summary_table <- data.table()

  # Analyze each biomarker
  for (biomarker_name in BIOMARKERS) {
    if (!biomarker_name %in% names(df)) {
      cat(sprintf("\nSkipping %s (not found in data)\n", biomarker_name))
      next
    }

    tryCatch({
      # Step 1: Prepare data
      df_clean <- prepare_biomarker_data(df, biomarker_name)

      # Step 2: Fit models
      result <- fit_models(df_clean, biomarker_name)

      # Step 3: Test DLNM significance
      sig_test <- test_dlnm_significance(result, biomarker_name)

      # Step 4: Visualize results
      visualize_results(result, sig_test, biomarker_name)

      # Step 5: Extract random effects
      extract_random_effects(result, biomarker_name)

      # Store summary
      best_comparison <- result$comparison[which.min(result$comparison$aic)]
      summary_row <- data.table(
        biomarker = biomarker_name,
        n_obs = nrow(df_clean),
        best_model = best_comparison$model,
        r2 = best_comparison$r2,
        aic = best_comparison$aic,
        dev_explained_pct = best_comparison$dev_expl,
        significant_dlnm = sig_test$significant,
        n_sig_temps = sig_test$n_sig_temps
      )
      summary_table <- rbind(summary_table, summary_row)

      all_results[[biomarker_name]] <- list(
        data = df_clean,
        result = result,
        sig_test = sig_test
      )

    }, error = function(e) {
      cat(sprintf("\n  ERROR: %s\n", e$message))
    })
  }

  # Save summary table
  summary_file <- file.path(OUTPUT_DIR, "comprehensive_summary.csv")
  fwrite(summary_table, summary_file)

  cat("\n")
  cat("################################################################################\n")
  cat("# Analysis Complete\n")
  cat("################################################################################\n\n")

  cat("Summary of Results:\n")
  print(summary_table[order(-r2)])

  cat(sprintf("\nDetailed results saved to: %s\n", OUTPUT_DIR))

  return(list(
    summary = summary_table,
    results = all_results
  ))
}

################################################################################
# RUN ANALYSIS
################################################################################

if (!interactive()) {
  results <- main_analysis()
}
