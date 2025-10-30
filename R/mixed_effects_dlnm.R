#!/usr/bin/env Rscript
################################################################################
# Mixed Effects DLNM Analysis for Climate-Biomarker Relationships
################################################################################
#
# Purpose: Implement hierarchical DLNM models with random effects to rigorously
#          test climate-biomarker associations while accounting for:
#          - Study-level heterogeneity
#          - Patient-level clustering
#          - Non-linear temperature-lag effects
#
# Model: biomarker ~ crossbasis(temp, lag) + season + vulnerability +
#                    s(study_id, bs="re") + s(patient_id, bs="re")
#
# Author: Claude + Craig Saunders
# Date: 2025-10-30
################################################################################

# Load required libraries
suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)          # GAM with random effects
  library(dlnm)          # Distributed lag non-linear models
  library(splines)       # Basis functions
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)     # Combine plots
  library(viridis)       # Color scales
})

# Set reproducibility
set.seed(42)

################################################################################
# CONFIGURATION
################################################################################

# Paths
DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/mixed_effects_dlnm"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# DLNM Parameters
DLNM_CONFIG <- list(
  lag = 21,                    # 0-21 day lag
  temp_df = 4,                 # Temperature non-linearity (4 df spline)
  lag_df = 4,                  # Lag structure flexibility (4 df spline)
  reference_temp = 18          # Minimum mortality temperature for JHB
)

# Biomarkers to analyze (based on previous high performers)
BIOMARKERS <- c(
  "Hematocrit (%)",
  "FASTING HDL CHOLESTEROL (mmol/L)",
  "Total cholesterol (mmol/L)",
  "FASTING LDL CHOLESTEROL (mmol/L)",
  "FASTING GLUCOSE (mmol/L)",
  "Creatinine (µmol/L)"
)

################################################################################
# HELPER FUNCTIONS
################################################################################

#' Load and prepare data for mixed effects DLNM
#'
#' @param biomarker_name Name of biomarker to analyze
#' @return data.table with complete cases for analysis
prepare_data <- function(biomarker_name) {
  cat(sprintf("\n=== Preparing data for: %s ===\n", biomarker_name))

  # Load data
  df <- fread(DATA_PATH)

  # Ensure required columns exist
  required_cols <- c(
    "primary_date", "study_source", "anonymous_patient_id",
    "climate_7d_mean_temp", "HEAT_VULNERABILITY_SCORE",
    biomarker_name
  )

  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  # Convert date
  df[, date := as.Date(primary_date, format = "%Y-%m-%d")]

  # Extract temporal features (use existing if available, otherwise create)
  if (!"month" %in% names(df)) {
    df[, month := month(date)]
  }
  if (!"year" %in% names(df)) {
    df[, year := year(date)]
  }
  if (!"season" %in% names(df)) {
    df[, season := case_when(
      month %in% c(12, 1, 2) ~ "Summer",
      month %in% c(3, 4, 5) ~ "Autumn",
      month %in% c(6, 7, 8) ~ "Winter",
      month %in% c(9, 10, 11) ~ "Spring"
    )]
  }
  df[, day_of_year := yday(date)]

  # Rename biomarker for easier handling
  setnames(df, biomarker_name, "biomarker", skip_absent = TRUE)

  # Create patient_id (unique identifier)
  df[, patient_id := paste0(study_source, "_", anonymous_patient_id)]

  # Rename study_source to study_id for consistency
  df[, study_id := study_source]

  # Select complete cases
  analysis_cols <- c(
    "biomarker", "climate_7d_mean_temp", "HEAT_VULNERABILITY_SCORE",
    "study_id", "patient_id", "season", "month", "year", "date", "day_of_year"
  )

  df_clean <- df[, ..analysis_cols]
  df_clean <- na.omit(df_clean)

  # Count repeated measures per patient
  df_clean[, n_obs_per_patient := .N, by = patient_id]
  df_clean[, n_patients := uniqueN(patient_id)]

  # Summary statistics
  cat(sprintf("  Total observations: %d\n", nrow(df_clean)))
  cat(sprintf("  Unique patients: %d\n", df_clean$n_patients[1]))
  cat(sprintf("  Unique studies: %d\n", uniqueN(df_clean$study_id)))
  cat(sprintf("  Observations with repeated measures: %d\n",
              sum(df_clean$n_obs_per_patient > 1)))
  cat(sprintf("  Temperature range: %.1f to %.1f°C\n",
              min(df_clean$climate_7d_mean_temp, na.rm = TRUE),
              max(df_clean$climate_7d_mean_temp, na.rm = TRUE)))
  cat(sprintf("  Biomarker range: %.2f to %.2f\n",
              min(df_clean$biomarker, na.rm = TRUE),
              max(df_clean$biomarker, na.rm = TRUE)))

  return(df_clean)
}

#' Fit hierarchical DLNM with multiple random effects structures
#'
#' @param data data.table with prepared data
#' @param biomarker_name Name of biomarker for reporting
#' @return list of fitted models
fit_mixed_effects_dlnm <- function(data, biomarker_name) {
  cat(sprintf("\n=== Fitting mixed effects DLNM for: %s ===\n", biomarker_name))

  # Create DLNM crossbasis
  cb_temp <- crossbasis(
    data$climate_7d_mean_temp,
    lag = DLNM_CONFIG$lag,
    argvar = list(fun = "ns", df = DLNM_CONFIG$temp_df),
    arglag = list(fun = "ns", df = DLNM_CONFIG$lag_df)
  )

  # Prepare data with crossbasis
  data_cb <- data.table(data)
  data_cb <- cbind(data_cb, as.data.table(cb_temp))

  # Convert factors
  data_cb[, `:=`(
    study_id = as.factor(study_id),
    patient_id = as.factor(patient_id),
    season = as.factor(season)
  )]

  # Model 1: No random effects (baseline)
  cat("\n  Fitting Model 1: No random effects (baseline)...\n")
  tryCatch({
    m1 <- gam(
      biomarker ~ cb_temp + season + HEAT_VULNERABILITY_SCORE,
      data = data_cb,
      method = "REML"
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    m1 <- NULL
  })

  # Model 2: Random intercept by study
  cat("\n  Fitting Model 2: Random intercept by study...\n")
  tryCatch({
    m2 <- gam(
      biomarker ~ cb_temp + season + HEAT_VULNERABILITY_SCORE +
        s(study_id, bs = "re"),
      data = data_cb,
      method = "REML"
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    m2 <- NULL
  })

  # Model 3: Random intercept by patient (nested in study)
  cat("\n  Fitting Model 3: Random intercept by patient...\n")
  tryCatch({
    m3 <- gam(
      biomarker ~ cb_temp + season + HEAT_VULNERABILITY_SCORE +
        s(patient_id, bs = "re"),
      data = data_cb,
      method = "REML"
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    m3 <- NULL
  })

  # Model 4: Random intercepts by both study and patient
  cat("\n  Fitting Model 4: Random intercepts by study + patient...\n")
  tryCatch({
    m4 <- gam(
      biomarker ~ cb_temp + season + HEAT_VULNERABILITY_SCORE +
        s(study_id, bs = "re") + s(patient_id, bs = "re"),
      data = data_cb,
      method = "REML"
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    m4 <- NULL
  })

  # Model 5: Random slope for temperature by study
  cat("\n  Fitting Model 5: Random slope for temperature by study...\n")
  tryCatch({
    m5 <- gam(
      biomarker ~ cb_temp + season + HEAT_VULNERABILITY_SCORE +
        s(study_id, bs = "re") +
        s(study_id, climate_7d_mean_temp, bs = "re"),
      data = data_cb,
      method = "REML"
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    m5 <- NULL
  })

  # Return list of models
  models <- list(
    m1_no_random = m1,
    m2_study_intercept = m2,
    m3_patient_intercept = m3,
    m4_study_patient_intercepts = m4,
    m5_study_random_slope = m5
  )

  # Filter out NULL models
  models <- Filter(Negate(is.null), models)

  return(list(
    models = models,
    crossbasis = cb_temp,
    data = data_cb
  ))
}

#' Compare model fit statistics
#'
#' @param model_list List of fitted gam models
#' @return data.table with comparison metrics
compare_models <- function(model_list) {
  cat("\n=== Model Comparison ===\n")

  results <- lapply(names(model_list), function(model_name) {
    model <- model_list[[model_name]]

    # Extract fit statistics
    data.table(
      model = model_name,
      aic = AIC(model),
      bic = BIC(model),
      deviance_explained = summary(model)$dev.expl * 100,
      r_squared = summary(model)$r.sq,
      n_obs = nrow(model$model),
      edf = sum(model$edf)  # Effective degrees of freedom
    )
  })

  results_dt <- rbindlist(results)

  # Rank by AIC (lower is better)
  results_dt[, aic_rank := rank(aic)]
  results_dt[, bic_rank := rank(bic)]

  # Print table
  print(results_dt[order(aic)])

  return(results_dt)
}

#' Extract and visualize DLNM predictions from best model
#'
#' @param fitted_result List containing models, crossbasis, data
#' @param biomarker_name Name of biomarker
#' @param model_comparison data.table with model comparison results
visualize_dlnm_predictions <- function(fitted_result, biomarker_name, model_comparison) {
  cat("\n=== Generating DLNM visualizations ===\n")

  # Select best model by AIC
  best_model_name <- model_comparison[order(aic)]$model[1]
  best_model <- fitted_result$models[[best_model_name]]
  cb_temp <- fitted_result$crossbasis

  cat(sprintf("  Best model: %s\n", best_model_name))
  cat(sprintf("  AIC: %.2f\n", AIC(best_model)))
  cat(sprintf("  R²: %.3f\n", summary(best_model)$r.sq))

  # Generate predictions from crossbasis
  pred <- crosspred(
    cb_temp,
    best_model,
    at = seq(10, 35, by = 0.5),
    cen = DLNM_CONFIG$reference_temp
  )

  # 1. Cumulative exposure-response curve
  p1 <- plot_cumulative_exposure_response(pred, biomarker_name, best_model_name)

  # 2. 3D temperature-lag surface
  p2 <- plot_3d_surface(pred, biomarker_name, best_model_name)

  # 3. Lag-specific effects
  p3 <- plot_lag_specific_effects(pred, biomarker_name, best_model_name)

  # 4. Contour plot
  p4 <- plot_contour(pred, biomarker_name, best_model_name)

  # Save plots
  output_file <- file.path(OUTPUT_DIR, sprintf("%s_dlnm_visualization.pdf",
                                                gsub("[^A-Za-z0-9_]", "_", biomarker_name)))

  pdf(output_file, width = 14, height = 10)
  print((p1 | p2) / (p3 | p4))
  dev.off()

  cat(sprintf("  Saved visualization: %s\n", output_file))

  # Return predictions for further analysis
  return(list(
    pred = pred,
    best_model = best_model,
    best_model_name = best_model_name
  ))
}

#' Plot cumulative exposure-response curve
plot_cumulative_exposure_response <- function(pred, biomarker_name, model_name) {
  df_plot <- data.frame(
    temperature = pred$predvar,
    effect = pred$allfit,
    lower = pred$alllow,
    upper = pred$allhigh
  )

  ggplot(df_plot, aes(x = temperature, y = effect)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "skyblue", alpha = 0.3) +
    geom_line(color = "darkblue", size = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = sprintf("%s: Cumulative Temperature Effect", biomarker_name),
      subtitle = sprintf("Model: %s", model_name),
      x = "Temperature (°C)",
      y = "Effect on Biomarker (cumulative over 0-21 days)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10)
    )
}

#' Plot 3D temperature-lag surface
plot_3d_surface <- function(pred, biomarker_name, model_name) {
  # Extract matrix for plotting
  z_matrix <- pred$matfit

  # Convert to long format for ggplot
  df_plot <- expand.grid(
    lag = 0:DLNM_CONFIG$lag,
    temperature = pred$predvar
  )
  df_plot$effect <- as.vector(z_matrix)

  ggplot(df_plot, aes(x = temperature, y = lag, fill = effect)) +
    geom_tile() +
    scale_fill_viridis(option = "plasma", name = "Effect") +
    labs(
      title = sprintf("%s: Temperature-Lag Surface", biomarker_name),
      subtitle = sprintf("Model: %s", model_name),
      x = "Temperature (°C)",
      y = "Lag (days)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10)
    )
}

#' Plot lag-specific effects at different lags
plot_lag_specific_effects <- function(pred, biomarker_name, model_name) {
  # Select specific lags to plot (0, 7, 14, 21 days)
  lags_to_plot <- c(0, 7, 14, 21)

  df_list <- lapply(lags_to_plot, function(lag_val) {
    data.frame(
      temperature = pred$predvar,
      effect = pred$matfit[, lag_val + 1],
      lower = pred$matlow[, lag_val + 1],
      upper = pred$mathigh[, lag_val + 1],
      lag = paste0("Lag ", lag_val, " days")
    )
  })

  df_plot <- do.call(rbind, df_list)

  ggplot(df_plot, aes(x = temperature, y = effect, color = lag, fill = lag)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
    geom_line(size = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    scale_color_viridis(discrete = TRUE, option = "viridis") +
    scale_fill_viridis(discrete = TRUE, option = "viridis") +
    labs(
      title = sprintf("%s: Lag-Specific Effects", biomarker_name),
      subtitle = sprintf("Model: %s", model_name),
      x = "Temperature (°C)",
      y = "Effect on Biomarker",
      color = "Lag",
      fill = "Lag"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      legend.position = "right"
    )
}

#' Plot contour plot
plot_contour <- function(pred, biomarker_name, model_name) {
  # Convert to long format
  df_plot <- expand.grid(
    lag = 0:DLNM_CONFIG$lag,
    temperature = pred$predvar
  )
  df_plot$effect <- as.vector(pred$matfit)

  ggplot(df_plot, aes(x = temperature, y = lag, z = effect)) +
    geom_contour_filled(bins = 15) +
    scale_fill_viridis(discrete = TRUE, option = "plasma", name = "Effect") +
    labs(
      title = sprintf("%s: Contour Plot", biomarker_name),
      subtitle = sprintf("Model: %s", model_name),
      x = "Temperature (°C)",
      y = "Lag (days)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10)
    )
}

#' Extract random effects and assess heterogeneity
#'
#' @param fitted_result List containing models
#' @param model_comparison data.table with model comparison
#' @param biomarker_name Name of biomarker
extract_random_effects <- function(fitted_result, model_comparison, biomarker_name) {
  cat("\n=== Extracting random effects ===\n")

  # Select best model
  best_model_name <- model_comparison[order(aic)]$model[1]
  best_model <- fitted_result$models[[best_model_name]]

  # Check if model has random effects
  if (!grepl("random", best_model_name, ignore.case = TRUE)) {
    cat("  Best model has no random effects.\n")
    return(NULL)
  }

  # Extract random effects
  random_effects <- ranef(best_model)

  # Visualize random effects distribution
  if (length(random_effects) > 0) {
    output_file <- file.path(OUTPUT_DIR, sprintf("%s_random_effects.pdf",
                                                  gsub("[^A-Za-z0-9_]", "_", biomarker_name)))

    pdf(output_file, width = 10, height = 6)

    for (i in seq_along(random_effects)) {
      re_data <- data.frame(
        id = names(random_effects[[i]]),
        effect = random_effects[[i]]
      )

      p <- ggplot(re_data, aes(x = effect)) +
        geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
        labs(
          title = sprintf("%s: Random Effects Distribution", names(random_effects)[i]),
          x = "Random Effect",
          y = "Count"
        ) +
        theme_minimal()

      print(p)

      # Calculate heterogeneity statistics
      cat(sprintf("\n  Random effect: %s\n", names(random_effects)[i]))
      cat(sprintf("    Mean: %.4f\n", mean(random_effects[[i]])))
      cat(sprintf("    SD: %.4f\n", sd(random_effects[[i]])))
      cat(sprintf("    Range: [%.4f, %.4f]\n",
                  min(random_effects[[i]]), max(random_effects[[i]])))
    }

    dev.off()
    cat(sprintf("  Saved random effects plot: %s\n", output_file))
  }

  return(random_effects)
}

################################################################################
# MAIN ANALYSIS PIPELINE
################################################################################

#' Run complete mixed effects DLNM analysis for all biomarkers
main_analysis <- function() {
  cat("\n")
  cat("################################################################################\n")
  cat("# Mixed Effects DLNM Analysis\n")
  cat("################################################################################\n\n")

  # Store results for all biomarkers
  all_results <- list()

  # Loop through biomarkers
  for (biomarker_name in BIOMARKERS) {
    cat(sprintf("\n\n========================================\n"))
    cat(sprintf("Analyzing: %s\n", biomarker_name))
    cat(sprintf("========================================\n"))

    tryCatch({
      # Step 1: Prepare data
      data <- prepare_data(biomarker_name)

      # Step 2: Fit models with different random effects structures
      fitted_result <- fit_mixed_effects_dlnm(data, biomarker_name)

      # Step 3: Compare models
      model_comparison <- compare_models(fitted_result$models)

      # Step 4: Visualize DLNM predictions from best model
      viz_result <- visualize_dlnm_predictions(fitted_result, biomarker_name, model_comparison)

      # Step 5: Extract and visualize random effects
      random_effects <- extract_random_effects(fitted_result, model_comparison, biomarker_name)

      # Store results
      all_results[[biomarker_name]] <- list(
        data = data,
        fitted_result = fitted_result,
        model_comparison = model_comparison,
        viz_result = viz_result,
        random_effects = random_effects
      )

    }, error = function(e) {
      cat(sprintf("\n  ERROR analyzing %s: %s\n", biomarker_name, e$message))
    })
  }

  # Save summary results
  save_summary_results(all_results)

  return(all_results)
}

#' Save summary results to CSV
save_summary_results <- function(all_results) {
  cat("\n=== Saving summary results ===\n")

  # Extract model comparison results
  summary_list <- lapply(names(all_results), function(biomarker) {
    result <- all_results[[biomarker]]

    if (!is.null(result$model_comparison)) {
      best_model <- result$model_comparison[order(aic)][1]
      best_model$biomarker <- biomarker
      return(best_model)
    }
    return(NULL)
  })

  summary_dt <- rbindlist(summary_list, use.names = TRUE, fill = TRUE)

  # Save to CSV
  output_file <- file.path(OUTPUT_DIR, "mixed_effects_dlnm_summary.csv")
  fwrite(summary_dt, output_file)

  cat(sprintf("  Saved summary: %s\n", output_file))
  cat("\n=== Analysis Complete ===\n")

  return(summary_dt)
}

################################################################################
# RUN ANALYSIS
################################################################################

if (!interactive()) {
  all_results <- main_analysis()
}
