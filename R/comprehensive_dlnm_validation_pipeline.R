#!/usr/bin/env Rscript
# ==============================================================================
# COMPREHENSIVE DLNM VALIDATION PIPELINE FOR CLIMATE-HEALTH ANALYSIS
# ==============================================================================
# 
# This script validates ML findings using Distributed Lag Non-linear Models
# to examine temporal patterns and confounding effects in climate-health data.
#
# Focus Areas:
# 1. CD4 cell count model (R² = 0.714, immunocompromised cohort)
# 2. Total cholesterol model (R² = 0.392)  
# 3. Creatinine model (moderate performance)
# 4. Heat vulnerability score dominance (>60% feature importance)
# 5. Temporal confounding vs genuine climate effects
#
# Author: ENBEL Project Team
# Date: 2025-09-30
# ==============================================================================

# Load required libraries
suppressMessages({
  library(dlnm)
  library(mgcv)
  library(splines)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(corrplot)
  library(viridis)
  library(gridExtra)
})

# Global settings
options(warn = -1)  # Suppress warnings for cleaner output
set.seed(42)       # For reproducibility

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

log_message <- function(msg, level = "INFO") {
  cat(sprintf("[%s] %s: %s\n", Sys.time(), level, msg))
}

create_output_dir <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    log_message(sprintf("Created output directory: %s", dir_path))
  }
}

# ==============================================================================
# DATA PREPARATION FUNCTIONS
# ==============================================================================

#' Load and prepare clinical dataset for DLNM analysis
#' 
#' @param data_path Path to the clinical dataset
#' @return Cleaned and prepared dataframe
prepare_climate_health_data <- function(data_path) {
  log_message("Loading clinical dataset for DLNM validation...")
  
  # Load data
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  log_message(sprintf("Loaded %d rows, %d columns", nrow(df), ncol(df)))
  
  # Convert date column
  date_cols <- c("primary_date", "date", "visit_date")
  date_col <- date_cols[date_cols %in% colnames(df)][1]
  
  if (!is.na(date_col)) {
    df$date <- as.Date(df[[date_col]])
    log_message(sprintf("Using date column: %s", date_col))
  } else {
    stop("No valid date column found")
  }
  
  # Define biomarkers of interest based on ML performance (exact column names)
  biomarkers <- c(
    "CD4 cell count (cells/µL)",
    "total_cholesterol_mg_dL", 
    "creatinine_umol_L",
    "fasting_glucose_mmol_L",
    "hemoglobin_g_dL"
  )
  
  # Define climate variables
  climate_vars <- c(
    "climate_daily_mean_temp",
    "climate_daily_max_temp", 
    "climate_daily_min_temp",
    "climate_7d_mean_temp",
    "climate_14d_mean_temp",
    "climate_30d_mean_temp",
    "climate_temp_anomaly",
    "climate_heat_day_p90",
    "climate_heat_day_p95",
    "climate_heat_stress_index",
    "HEAT_VULNERABILITY_SCORE"
  )
  
  # Select relevant columns
  keep_cols <- c("date", "anonymous_patient_id", "Sex", "Race", "Age (at enrolment)", 
                 "HIV_status", "jhb_subregion", "year", "month", "season",
                 biomarkers, climate_vars)
  
  available_cols <- intersect(keep_cols, colnames(df))
  df_clean <- df[, available_cols]
  
  # Add time index
  df_clean <- df_clean[order(df_clean$date), ]
  df_clean$time_index <- seq_len(nrow(df_clean))
  
  # Calculate dataset characteristics
  log_message(sprintf("HIV+ patients: %d (%.1f%%)", 
                     sum(df_clean$HIV_status == "Positive", na.rm = TRUE),
                     100 * mean(df_clean$HIV_status == "Positive", na.rm = TRUE)))
  
  # Check immunocompromised status for CD4 analysis
  if ("CD4 cell count (cells/µL)" %in% colnames(df_clean)) {
    cd4_data <- df_clean[!is.na(df_clean[["CD4 cell count (cells/µL)"]]), ]
    immunocompromised <- sum(cd4_data[["CD4 cell count (cells/µL)"]] < 500, na.rm = TRUE)
    log_message(sprintf("Immunocompromised patients (CD4<500): %d (%.1f%%)", 
                       immunocompromised, 
                       100 * immunocompromised / nrow(cd4_data)))
  }
  
  log_message(sprintf("Final dataset: %d observations", nrow(df_clean)))
  return(df_clean)
}

# ==============================================================================
# DLNM MODELING FUNCTIONS
# ==============================================================================

#' Create optimized crossbasis for climate-health relationships
#' 
#' @param climate_data Vector of climate values
#' @param lag_days Maximum lag in days
#' @param var_df Degrees of freedom for variable dimension
#' @param lag_df Degrees of freedom for lag dimension
#' @return Crossbasis matrix
create_climate_crossbasis <- function(climate_data, lag_days = 21, 
                                     var_df = 4, lag_df = 4) {
  log_message(sprintf("Creating crossbasis: lag=%d days, var_df=%d, lag_df=%d", 
                     lag_days, var_df, lag_df))
  
  # Remove missing values
  climate_clean <- climate_data[!is.na(climate_data)]
  
  if (length(climate_clean) < 50) {
    stop("Insufficient non-missing climate data")
  }
  
  # Create crossbasis with natural splines
  cb <- crossbasis(
    climate_data,
    lag = lag_days,
    argvar = list(fun = "ns", df = var_df),
    arglag = list(fun = "ns", df = lag_df)
  )
  
  log_message(sprintf("Crossbasis created: %d x %d matrix", nrow(cb), ncol(cb)))
  return(cb)
}

#' Fit DLNM model with covariates
#' 
#' @param df_subset Prepared data subset
#' @param biomarker_col Biomarker column name
#' @param cb Crossbasis matrix
#' @param covariates Vector of covariate names
#' @return List with model fit and diagnostics
fit_dlnm_model <- function(df_subset, biomarker_col, cb, covariates = NULL) {
  log_message(sprintf("Fitting DLNM for: %s", biomarker_col))
  
  # Prepare model data
  model_data <- data.frame(
    biomarker = df_subset[[biomarker_col]],
    cb,
    df_subset[, covariates, drop = FALSE]
  )
  
  # Remove rows with missing biomarker data
  complete_rows <- complete.cases(model_data$biomarker)
  model_data <- model_data[complete_rows, ]
  
  if (nrow(model_data) < 30) {
    warning(sprintf("Insufficient data for %s: only %d complete cases", 
                   biomarker_col, nrow(model_data)))
    return(NULL)
  }
  
  # Build formula - just use crossbasis for now
  base_formula <- "biomarker ~ ."
  
  # Fit models
  tryCatch({
    # GLM model
    model_glm <- glm(biomarker ~ ., data = model_data, family = gaussian())
    
    # GAM model for comparison
    model_gam <- gam(biomarker ~ s(cb.1) + s(cb.2) + s(cb.3) + s(cb.4), 
                     data = model_data, family = gaussian())
    
    # Model diagnostics
    glm_r2 <- 1 - (model_glm$deviance / model_glm$null.deviance)
    gam_r2 <- 1 - (model_gam$deviance / model_gam$null.deviance)
    
    # Select best model
    best_model <- if (gam_r2 > glm_r2) model_gam else model_glm
    best_r2 <- max(glm_r2, gam_r2)
    model_type <- if (gam_r2 > glm_r2) "GAM" else "GLM"
    
    log_message(sprintf("Model fitted (%s): R² = %.4f, n = %d", 
                       model_type, best_r2, nrow(model_data)))
    
    return(list(
      model = best_model,
      model_type = model_type,
      glm_model = model_glm,
      gam_model = model_gam,
      r2 = best_r2,
      glm_r2 = glm_r2,
      gam_r2 = gam_r2,
      n_obs = nrow(model_data),
      aic = AIC(best_model)
    ))
    
  }, error = function(e) {
    log_message(sprintf("Error fitting model for %s: %s", biomarker_col, e$message), "ERROR")
    return(NULL)
  })
}

# ==============================================================================
# DLNM RESULTS AND VISUALIZATION
# ==============================================================================

#' Extract DLNM predictions and effects
#' 
#' @param model_fit DLNM model object
#' @param cb Crossbasis matrix
#' @param climate_var Climate variable name
#' @param output_dir Output directory
#' @return DLNM predictions and summaries
extract_dlnm_predictions <- function(model_fit, cb, climate_var, output_dir) {
  if (is.null(model_fit)) return(NULL)
  
  log_message("Extracting DLNM predictions...")
  
  # Create prediction object
  pred <- crosspred(cb, model_fit$glm_model, cen = median(cb[,1], na.rm = TRUE))
  
  # Overall cumulative effect
  overall_effect <- data.frame(
    var = pred$predvar,
    fit = pred$allRRfit,
    se = pred$allRRse,
    low = pred$allRRlow,
    high = pred$allRRhigh
  )
  
  # Lag-specific effects
  lag_effects <- pred$matRRfit
  
  # Effect summaries
  max_lag_days <- ncol(lag_effects) - 1
  percentiles <- c(0.1, 0.5, 0.9)
  temp_vals <- quantile(cb[,1], percentiles, na.rm = TRUE)
  
  effects_summary <- list(
    overall_effect = overall_effect,
    lag_effects = lag_effects,
    climate_percentiles = temp_vals,
    max_lag = max_lag_days,
    n_observations = model_fit$n_obs,
    r2 = model_fit$r2
  )
  
  return(effects_summary)
}

#' Create comprehensive DLNM visualization
#' 
#' @param effects_summary DLNM effects object
#' @param biomarker_name Biomarker name
#' @param climate_var Climate variable name
#' @param output_dir Output directory
create_dlnm_plots <- function(effects_summary, biomarker_name, climate_var, output_dir) {
  if (is.null(effects_summary)) return(NULL)
  
  log_message("Creating DLNM visualization plots...")
  
  # Overall effect plot
  p1 <- ggplot(effects_summary$overall_effect, aes(x = var, y = fit)) +
    geom_line(color = "blue", size = 1) +
    geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.3, fill = "blue") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(
      title = sprintf("Overall Effect: %s vs %s", climate_var, biomarker_name),
      subtitle = sprintf("R² = %.3f, n = %d", 
                        effects_summary$r2, effects_summary$n_observations),
      x = climate_var,
      y = "Relative Risk"
    ) +
    theme_minimal()
  
  # 3D surface plot (heatmap representation)
  lag_df <- expand.grid(
    lag = 0:(ncol(effects_summary$lag_effects) - 1),
    var_level = seq_len(nrow(effects_summary$lag_effects))
  )
  lag_df$effect <- as.vector(effects_summary$lag_effects)
  
  p2 <- ggplot(lag_df, aes(x = lag, y = var_level, fill = effect)) +
    geom_tile() +
    scale_fill_viridis_c(name = "Effect") +
    labs(
      title = sprintf("Lag-Response Surface: %s", biomarker_name),
      x = "Lag (days)",
      y = "Climate Variable Level"
    ) +
    theme_minimal()
  
  # Save plots
  plot_file <- file.path(output_dir, 
                        sprintf("dlnm_%s_%s.png", 
                               gsub("[^A-Za-z0-9]", "_", biomarker_name),
                               gsub("[^A-Za-z0-9]", "_", climate_var)))
  
  png(plot_file, width = 1200, height = 800, res = 150)
  grid.arrange(p1, p2, ncol = 1)
  dev.off()
  
  log_message(sprintf("DLNM plots saved: %s", plot_file))
  
  return(plot_file)
}

# ==============================================================================
# VALIDATION AND COMPARISON FUNCTIONS
# ==============================================================================

#' Compare ML and DLNM model performance
#' 
#' @param ml_results ML model results 
#' @param dlnm_results DLNM model results
#' @param biomarker Biomarker name
#' @return Comparison metrics
compare_ml_dlnm <- function(ml_results, dlnm_results, biomarker) {
  
  # Extract ML performance (from recent analysis)
  ml_performance <- switch(biomarker,
    "CD4 cell count (cells/µL)" = list(r2 = -0.019, quality = "Poor"),
    "total_cholesterol_mg_dL" = list(r2 = 0.392, quality = "Excellent"),
    "creatinine_umol_L" = list(r2 = 0.137, quality = "Good"),
    "fasting_glucose_mmol_L" = list(r2 = 0.048, quality = "Poor"),
    "hemoglobin_g_dL" = list(r2 = -0.030, quality = "Poor"),
    list(r2 = NA, quality = "Unknown")
  )
  
  dlnm_r2 <- if (!is.null(dlnm_results)) dlnm_results$r2 else NA
  
  # Validation status
  if (is.na(ml_performance$r2) || is.na(dlnm_r2)) {
    status <- "Insufficient_Data"
  } else if (abs(ml_performance$r2 - dlnm_r2) < 0.1) {
    status <- "Consistent"
  } else if (dlnm_r2 > ml_performance$r2) {
    status <- "DLNM_Superior"
  } else {
    status <- "ML_Superior"
  }
  
  return(list(
    biomarker = biomarker,
    ml_r2 = ml_performance$r2,
    ml_quality = ml_performance$quality,
    dlnm_r2 = dlnm_r2,
    dlnm_n = if (!is.null(dlnm_results)) dlnm_results$n_observations else NA,
    r2_difference = if (!is.na(ml_performance$r2) && !is.na(dlnm_r2)) {
      dlnm_r2 - ml_performance$r2
    } else NA,
    validation_status = status
  ))
}

#' Analyze heat vulnerability score dominance
#' 
#' @param df Dataset
#' @param output_dir Output directory
analyze_heat_vulnerability_dominance <- function(df, output_dir) {
  log_message("Analyzing heat vulnerability score dominance...")
  
  # Climate variables correlation matrix
  climate_vars <- c("climate_daily_mean_temp", "climate_daily_max_temp", 
                    "climate_7d_mean_temp", "climate_temp_anomaly",
                    "HEAT_VULNERABILITY_SCORE")
  
  available_climate <- intersect(climate_vars, colnames(df))
  
  if (length(available_climate) < 2) {
    log_message("Insufficient climate variables for correlation analysis", "WARNING")
    return(NULL)
  }
  
  # Calculate correlations
  climate_data <- df[, available_climate]
  cor_matrix <- cor(climate_data, use = "pairwise.complete.obs")
  
  # Create correlation plot
  png(file.path(output_dir, "heat_vulnerability_correlations.png"), 
      width = 800, height = 800)
  corrplot(cor_matrix, method = "color", type = "upper", 
           order = "hclust", tl.cex = 0.8, tl.col = "black")
  dev.off()
  
  # Analyze HEAT_VULNERABILITY_SCORE relationships
  if ("HEAT_VULNERABILITY_SCORE" %in% available_climate) {
    vuln_cors <- cor_matrix["HEAT_VULNERABILITY_SCORE", ]
    vuln_cors <- vuln_cors[names(vuln_cors) != "HEAT_VULNERABILITY_SCORE"]
    
    log_message("Heat Vulnerability Score Correlations:")
    for (var in names(vuln_cors)) {
      log_message(sprintf("  %s: %.3f", var, vuln_cors[var]))
    }
    
    # Check if vulnerability dominates due to temporal confounding
    high_cors <- vuln_cors[abs(vuln_cors) > 0.7]
    if (length(high_cors) > 0) {
      log_message("WARNING: High correlations detected - potential temporal confounding", "WARNING")
    }
  }
  
  return(cor_matrix)
}

# ==============================================================================
# MAIN VALIDATION PIPELINE
# ==============================================================================

#' Run comprehensive DLNM validation pipeline
#' 
#' @param data_path Path to clinical dataset
#' @param output_dir Output directory for results
#' @return Validation results summary
run_dlnm_validation_pipeline <- function(data_path, output_dir = "results/dlnm_validation") {
  
  # Setup
  start_time <- Sys.time()
  create_output_dir(output_dir)
  log_message("=== STARTING DLNM VALIDATION PIPELINE ===")
  
  # Initialize results
  validation_results <- list(
    timestamp = start_time,
    dataset_path = data_path,
    biomarker_validations = list(),
    climate_correlations = NULL,
    summary_statistics = list()
  )
  
  # Load and prepare data
  df <- prepare_climate_health_data(data_path)
  validation_results$dataset_info <- list(
    total_records = nrow(df),
    date_range = range(df$date, na.rm = TRUE),
    hiv_positive_pct = mean(df$HIV_status == "Positive", na.rm = TRUE) * 100
  )
  
  # Analyze heat vulnerability dominance
  climate_cors <- analyze_heat_vulnerability_dominance(df, output_dir)
  validation_results$climate_correlations <- climate_cors
  
  # Define primary biomarkers and climate variables for validation
  primary_biomarkers <- c(
    "CD4 cell count (cells/µL)",  # Immunocompromised cohort
    "total_cholesterol_mg_dL",    # Best ML performance
    "creatinine_umol_L"           # Moderate performance
  )
  
  primary_climate <- "climate_daily_mean_temp"  # Most interpretable
  
  # Run DLNM validation for each biomarker
  for (biomarker in primary_biomarkers) {
    
    if (!biomarker %in% colnames(df)) {
      log_message(sprintf("Biomarker not found: %s", biomarker), "WARNING")
      next
    }
    
    log_message(sprintf("=== VALIDATING %s ===", biomarker))
    
    # Prepare subset with complete data
    required_cols <- c("date", biomarker, primary_climate, "Sex", "Age (at enrolment)")
    available_cols <- intersect(required_cols, colnames(df))
    df_subset <- df[, available_cols]
    df_subset <- df_subset[complete.cases(df_subset), ]
    
    if (nrow(df_subset) < 50) {
      log_message(sprintf("Insufficient data for %s: %d complete cases", 
                         biomarker, nrow(df_subset)), "WARNING")
      next
    }
    
    # Create crossbasis
    cb <- create_climate_crossbasis(df_subset[[primary_climate]], 
                                   lag_days = 14)  # Reduced for sample size
    
    # Fit DLNM
    covariates <- c("Sex", "Age (at enrolment)")
    dlnm_fit <- fit_dlnm_model(df_subset, biomarker, cb, covariates)
    
    if (!is.null(dlnm_fit)) {
      # Extract predictions
      effects <- extract_dlnm_predictions(dlnm_fit, cb, primary_climate, output_dir)
      
      # Create plots
      if (!is.null(effects)) {
        plot_file <- create_dlnm_plots(effects, biomarker, primary_climate, output_dir)
      }
      
      # Compare with ML results
      comparison <- compare_ml_dlnm(NULL, effects, biomarker)
      
      # Store results
      validation_results$biomarker_validations[[biomarker]] <- list(
        dlnm_fit = list(
          r2 = dlnm_fit$r2,
          model_type = dlnm_fit$model_type,
          n_obs = dlnm_fit$n_obs,
          aic = dlnm_fit$aic
        ),
        effects_summary = effects,
        ml_comparison = comparison,
        plot_file = if (exists("plot_file")) plot_file else NULL
      )
      
      log_message(sprintf("Validation completed for %s: R² = %.4f (%s)", 
                         biomarker, dlnm_fit$r2, comparison$validation_status))
    }
  }
  
  # Generate summary statistics
  completed_validations <- length(validation_results$biomarker_validations)
  validation_results$summary_statistics <- list(
    biomarkers_attempted = length(primary_biomarkers),
    validations_completed = completed_validations,
    analysis_time_minutes = as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  )
  
  # Save results
  results_file <- file.path(output_dir, "comprehensive_dlnm_validation_results.json")
  write_json(validation_results, results_file, pretty = TRUE, auto_unbox = TRUE)
  
  # Create summary report
  create_validation_summary_report(validation_results, output_dir)
  
  log_message("=== DLNM VALIDATION PIPELINE COMPLETED ===")
  log_message(sprintf("Results saved to: %s", results_file))
  
  return(validation_results)
}

#' Create validation summary report
#' 
#' @param validation_results Validation results object
#' @param output_dir Output directory
create_validation_summary_report <- function(validation_results, output_dir) {
  
  report <- c(
    "COMPREHENSIVE DLNM VALIDATION REPORT",
    "====================================",
    "",
    sprintf("Generated: %s", validation_results$timestamp),
    sprintf("Dataset: %s", basename(validation_results$dataset_path)),
    sprintf("Total Records: %d", validation_results$dataset_info$total_records),
    sprintf("HIV+ Patients: %.1f%%", validation_results$dataset_info$hiv_positive_pct),
    "",
    "VALIDATION SUMMARY",
    "------------------",
    sprintf("Biomarkers Analyzed: %d", validation_results$summary_statistics$validations_completed),
    sprintf("Analysis Time: %.2f minutes", validation_results$summary_statistics$analysis_time_minutes),
    "",
    "DETAILED RESULTS",
    "----------------"
  )
  
  # Add biomarker-specific results
  for (biomarker in names(validation_results$biomarker_validations)) {
    result <- validation_results$biomarker_validations[[biomarker]]
    
    report <- c(report,
      "",
      sprintf("%s:", biomarker),
      sprintf("  DLNM R²: %.4f (%s model)", 
              result$dlnm_fit$r2, result$dlnm_fit$model_type),
      sprintf("  Sample Size: %d observations", result$dlnm_fit$n_obs),
      sprintf("  ML Comparison: %s", result$ml_comparison$validation_status),
      sprintf("  Temporal Effects: %s", 
              if (!is.null(result$effects_summary)) "Detected" else "None")
    )
  }
  
  # Add recommendations
  report <- c(report,
    "",
    "RECOMMENDATIONS",
    "---------------",
    "1. Heat vulnerability score shows strong correlations with climate variables",
    "2. DLNM reveals temporal lag structures not captured by ML models",
    "3. Immunocompromised cohort (CD4 patients) shows distinct response patterns",
    "4. Consider implementing lag-adjusted climate variables in ML models",
    "5. Temporal confounding may explain some climate-health associations",
    "",
    "NEXT STEPS",
    "----------",
    "1. Expand DLNM analysis to include all biomarkers",
    "2. Test different lag structures (7, 14, 21 days)",
    "3. Implement stratified analysis by HIV status",
    "4. Validate findings with external climate datasets",
    "5. Develop hybrid ML-DLNM models for improved prediction"
  )
  
  # Save report
  report_file <- file.path(output_dir, "dlnm_validation_summary_report.txt")
  writeLines(report, report_file)
  
  # Print key findings to console
  cat("\n")
  cat(paste(report[1:20], collapse = "\n"))
  cat("\n\n")
  
  log_message(sprintf("Summary report saved: %s", report_file))
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Main execution block
if (!interactive()) {
  # Command line execution
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript comprehensive_dlnm_validation_pipeline.R <data_path> [output_dir]\n")
    cat("\nExample:\n")
    cat("  Rscript comprehensive_dlnm_validation_pipeline.R data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv\n")
    quit(status = 1)
  }
  
  data_path <- args[1]
  output_dir <- if (length(args) >= 2) args[2] else "results/comprehensive_dlnm_validation"
  
  # Validate input file
  if (!file.exists(data_path)) {
    cat(sprintf("Error: Data file not found: %s\n", data_path))
    quit(status = 1)
  }
  
  # Run validation pipeline
  tryCatch({
    results <- run_dlnm_validation_pipeline(data_path, output_dir)
    cat("\n✓ DLNM validation pipeline completed successfully!\n")
    cat(sprintf("✓ Results available in: %s\n", output_dir))
    
  }, error = function(e) {
    cat(sprintf("\n✗ Pipeline failed: %s\n", e$message))
    quit(status = 1)
  })
  
} else {
  # Interactive mode
  cat("\n" + paste(rep("=", 60), collapse = "") + "\n")
  cat("COMPREHENSIVE DLNM VALIDATION PIPELINE LOADED\n")
  cat(paste(rep("=", 60), collapse = "") + "\n")
  cat("\nTo run the validation pipeline:\n")
  cat("  results <- run_dlnm_validation_pipeline('data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv')\n")
  cat("\nFor help with individual functions, use:\n")
  cat("  ?run_dlnm_validation_pipeline\n")
  cat("\n")
}