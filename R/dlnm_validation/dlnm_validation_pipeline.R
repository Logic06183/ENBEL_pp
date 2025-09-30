# DLNM Validation Pipeline for Climate-Health Analysis
# =====================================================
# 
# This script validates ML findings using Distributed Lag Non-linear Models
# to examine the temporal and non-linear relationships between climate 
# variables and health biomarkers.
#
# Author: ENBEL Project Team
# Date: 2024

# Load required libraries
library(dlnm)
library(mgcv)
library(splines)
library(ggplot2)
library(dplyr)
library(tidyr)
library(jsonlite)

# Set up logging
log_message <- function(msg) {
  cat(paste0("[", Sys.time(), "] ", msg, "\n"))
}

#' Load and prepare data for DLNM analysis
#'
#' @param data_path Path to the CSV file with imputed data
#' @param biomarker Target biomarker column name
#' @param climate_var Climate variable to analyze
#' @return Prepared dataframe for DLNM analysis
prepare_dlnm_data <- function(data_path, biomarker, climate_var) {
  log_message("Loading data for DLNM analysis...")
  
  # Load data
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  
  # Convert date column
  if ("date" %in% colnames(df)) {
    df$date <- as.Date(df$date)
  } else if ("Date" %in% colnames(df)) {
    df$date <- as.Date(df$Date)
  }
  
  # Select relevant columns
  required_cols <- c("date", biomarker, climate_var)
  available_cols <- required_cols[required_cols %in% colnames(df)]
  
  if (length(available_cols) < 3) {
    stop(paste("Missing required columns. Need:", paste(required_cols, collapse=", ")))
  }
  
  # Filter complete cases
  df_clean <- df[, available_cols]
  df_clean <- df_clean[complete.cases(df_clean), ]
  
  # Sort by date
  df_clean <- df_clean[order(df_clean$date), ]
  
  # Add time index
  df_clean$time <- seq_len(nrow(df_clean))
  
  log_message(sprintf("Prepared %d observations for analysis", nrow(df_clean)))
  
  return(df_clean)
}

#' Create crossbasis for DLNM
#'
#' @param climate_data Vector of climate variable values
#' @param lag Maximum lag to consider
#' @param df_temp Degrees of freedom for temperature dimension
#' @param df_lag Degrees of freedom for lag dimension
#' @return Crossbasis matrix for DLNM
create_crossbasis <- function(climate_data, lag = 21, df_temp = 4, df_lag = 4) {
  log_message(sprintf("Creating crossbasis with lag=%d days", lag))
  
  # Create crossbasis
  cb <- crossbasis(
    climate_data,
    lag = lag,
    argvar = list(fun = "ns", df = df_temp),
    arglag = list(fun = "ns", df = df_lag)
  )
  
  return(cb)
}

#' Fit DLNM model
#'
#' @param df_clean Prepared dataframe
#' @param biomarker Target biomarker name
#' @param climate_var Climate variable name
#' @param cb Crossbasis matrix
#' @param covariates Additional covariates to include
#' @return Fitted DLNM model
fit_dlnm <- function(df_clean, biomarker, climate_var, cb, covariates = NULL) {
  log_message("Fitting DLNM model...")
  
  # Build formula
  if (!is.null(covariates)) {
    covar_terms <- paste(covariates, collapse = " + ")
    formula_str <- sprintf("%s ~ cb + %s", biomarker, covar_terms)
  } else {
    formula_str <- sprintf("%s ~ cb", biomarker)
  }
  
  # Fit model
  model <- glm(as.formula(formula_str), data = df_clean, family = gaussian())
  
  # Calculate model metrics
  r_squared <- 1 - (model$deviance / model$null.deviance)
  aic_value <- AIC(model)
  
  log_message(sprintf("Model fitted: R² = %.3f, AIC = %.1f", r_squared, aic_value))
  
  return(list(model = model, r_squared = r_squared, aic = aic_value))
}

#' Extract and plot DLNM results
#'
#' @param model_fit DLNM model object
#' @param cb Crossbasis matrix
#' @param climate_var Climate variable name
#' @param biomarker Biomarker name
#' @param output_dir Directory to save plots
#' @return List of DLNM predictions and plots
extract_dlnm_results <- function(model_fit, cb, climate_var, biomarker, output_dir = "results/dlnm") {
  log_message("Extracting DLNM results...")
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Predict from crossbasis
  pred <- crosspred(cb, model_fit$model, cen = median(cb[,1]))
  
  # Overall cumulative effect
  png(file.path(output_dir, sprintf("dlnm_overall_%s_%s.png", biomarker, climate_var)),
      width = 800, height = 600)
  plot(pred, "overall", main = sprintf("Overall effect of %s on %s", climate_var, biomarker),
       xlab = climate_var, ylab = sprintf("Effect on %s", biomarker))
  dev.off()
  
  # 3D plot
  png(file.path(output_dir, sprintf("dlnm_3d_%s_%s.png", biomarker, climate_var)),
      width = 800, height = 600)
  plot(pred, "3d", main = sprintf("3D lag-response for %s on %s", climate_var, biomarker),
       xlab = climate_var, ylab = "Lag (days)", zlab = "Effect")
  dev.off()
  
  # Lag-specific effects at different temperatures
  percentiles <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  temp_values <- quantile(cb[,1], percentiles)
  
  png(file.path(output_dir, sprintf("dlnm_lag_effects_%s_%s.png", biomarker, climate_var)),
      width = 1000, height = 600)
  par(mfrow = c(2, 3))
  for (i in 1:length(temp_values)) {
    plot(pred, "slices", var = temp_values[i], 
         main = sprintf("%s = %.1f (P%d)", climate_var, temp_values[i], percentiles[i]*100))
  }
  dev.off()
  
  # Extract key results
  results <- list(
    overall_effect = pred$allRRfit,
    lag_effects = pred$matRRfit,
    model_metrics = list(
      r_squared = model_fit$r_squared,
      aic = model_fit$aic
    )
  )
  
  log_message("DLNM results extracted and plots saved")
  
  return(results)
}

#' Validate ML findings with DLNM
#'
#' @param ml_results_path Path to ML results JSON file
#' @param data_path Path to imputed dataset
#' @param output_dir Output directory for results
#' @return Validation results comparing ML and DLNM findings
validate_ml_with_dlnm <- function(ml_results_path, data_path, output_dir = "results/dlnm_validation") {
  log_message("Starting DLNM validation of ML findings...")
  
  # Load ML results
  ml_results <- fromJSON(ml_results_path)
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Initialize validation results
  validation_results <- list(
    timestamp = Sys.time(),
    biomarkers_validated = list()
  )
  
  # Process each system
  for (system_name in names(ml_results$systems)) {
    system_results <- ml_results$systems[[system_name]]
    
    for (biomarker_name in names(system_results)) {
      biomarker_results <- system_results[[biomarker_name]]
      
      # Skip if no SHAP analysis
      if (is.null(biomarker_results$shap_analysis)) {
        next
      }
      
      # Get top climate features from SHAP
      top_climate <- biomarker_results$shap_analysis$top_climate_features
      
      if (length(top_climate) == 0) {
        next
      }
      
      log_message(sprintf("Validating %s with DLNM...", biomarker_name))
      
      # Run DLNM for top climate predictor
      climate_var <- top_climate[1]
      
      tryCatch({
        # Prepare data
        df_clean <- prepare_dlnm_data(data_path, biomarker_name, climate_var)
        
        # Create crossbasis
        cb <- create_crossbasis(df_clean[[climate_var]])
        
        # Fit DLNM
        dlnm_fit <- fit_dlnm(df_clean, biomarker_name, climate_var, cb)
        
        # Extract results
        dlnm_results <- extract_dlnm_results(
          dlnm_fit, cb, climate_var, biomarker_name,
          file.path(output_dir, system_name)
        )
        
        # Compare with ML results
        ml_r2 <- biomarker_results$best_r2
        dlnm_r2 <- dlnm_fit$r_squared
        
        validation_results$biomarkers_validated[[biomarker_name]] <- list(
          system = system_name,
          ml_r2 = ml_r2,
          dlnm_r2 = dlnm_r2,
          r2_difference = ml_r2 - dlnm_r2,
          climate_variable = climate_var,
          validation_status = ifelse(abs(ml_r2 - dlnm_r2) < 0.2, "Consistent", "Divergent")
        )
        
        log_message(sprintf("  ML R²=%.3f, DLNM R²=%.3f (%s)", 
                           ml_r2, dlnm_r2,
                           validation_results$biomarkers_validated[[biomarker_name]]$validation_status))
        
      }, error = function(e) {
        log_message(sprintf("  Error validating %s: %s", biomarker_name, e$message))
      })
    }
  }
  
  # Save validation results
  validation_json <- toJSON(validation_results, pretty = TRUE, auto_unbox = TRUE)
  writeLines(validation_json, file.path(output_dir, "dlnm_validation_results.json"))
  
  # Create summary report
  create_validation_report(validation_results, output_dir)
  
  log_message("DLNM validation completed")
  
  return(validation_results)
}

#' Create validation summary report
#'
#' @param validation_results Validation results object
#' @param output_dir Output directory
create_validation_report <- function(validation_results, output_dir) {
  report <- c(
    "DLNM VALIDATION REPORT",
    "=====================",
    paste("Generated:", validation_results$timestamp),
    "",
    "VALIDATION SUMMARY",
    "------------------"
  )
  
  n_validated <- length(validation_results$biomarkers_validated)
  n_consistent <- sum(sapply(validation_results$biomarkers_validated, 
                             function(x) x$validation_status == "Consistent"))
  
  report <- c(report,
    paste("Biomarkers validated:", n_validated),
    paste("Consistent findings:", n_consistent),
    paste("Divergent findings:", n_validated - n_consistent),
    "",
    "DETAILED RESULTS",
    "----------------"
  )
  
  for (biomarker in names(validation_results$biomarkers_validated)) {
    result <- validation_results$biomarkers_validated[[biomarker]]
    report <- c(report,
      "",
      paste0(biomarker, " (", result$system, " system):"),
      paste("  Climate variable:", result$climate_variable),
      paste("  ML R²:", sprintf("%.3f", result$ml_r2)),
      paste("  DLNM R²:", sprintf("%.3f", result$dlnm_r2)),
      paste("  Status:", result$validation_status)
    )
  }
  
  # Save report
  writeLines(report, file.path(output_dir, "dlnm_validation_report.txt"))
  
  # Print to console
  cat(paste(report, collapse = "\n"))
}

# Main execution
if (!interactive()) {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    stop("Usage: Rscript dlnm_validation_pipeline.R <ml_results.json> <imputed_data.csv>")
  }
  
  ml_results_path <- args[1]
  data_path <- args[2]
  output_dir <- ifelse(length(args) >= 3, args[3], "results/dlnm_validation")
  
  # Run validation
  validation_results <- validate_ml_with_dlnm(ml_results_path, data_path, output_dir)
  
} else {
  # Interactive mode - provide example usage
  cat("DLNM Validation Pipeline loaded.\n")
  cat("To run validation:\n")
  cat("  validation_results <- validate_ml_with_dlnm('ml_results.json', 'imputed_data.csv')\n")
}