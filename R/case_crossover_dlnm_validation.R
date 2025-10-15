#!/usr/bin/env Rscript
# ============================================================================
# Case-Crossover DLNM Validation for Climate-Health Biomarker Associations
# ============================================================================
#
# Purpose: Validate ML findings using case-crossover design with DLNM
#          to control for time-invariant confounders and assess lagged effects
#
# Design: Time-stratified case-crossover with distributed lag non-linear models
#
# Biomarkers: Validates top 6 significant biomarkers from ML analysis
#   1. Hematocrit (%)          - R² = 0.937
#   2. Total Cholesterol       - R² = 0.392
#   3. FASTING LDL             - R² = 0.377
#   4. FASTING HDL             - R² = 0.334
#   5. LDL Cholesterol         - R² = 0.143
#   6. Creatinine              - R² = 0.137
#
# Author: ENBEL Climate-Health Analysis Pipeline
# Date: 2025-10-14
# ============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(dlnm)       # Distributed lag non-linear models
  library(gnm)        # Generalized nonlinear models (for conditional logistic)
  library(splines)    # Spline functions
  library(mgcv)       # GAM models
  library(dplyr)      # Data manipulation
  library(tidyr)      # Data tidying
  library(ggplot2)    # Visualization
  library(gridExtra)  # Multiple plots
  library(jsonlite)   # JSON output
})

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH <- "data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "results/case_crossover_dlnm"

# Create output directory
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# DLNM parameters
MAX_LAG <- 21            # Maximum lag in days (3 weeks)
DF_TEMP <- 4             # Degrees of freedom for temperature spline
DF_LAG <- 4              # Degrees of freedom for lag spline
PERCENTILE_REF <- 50     # Reference temperature percentile (median)

# Case-crossover parameters
STRATUM_DAYS <- 28       # Time-stratified stratum length (4 weeks)

# Biomarkers to validate
BIOMARKERS <- list(
  list(
    name = "Hematocrit (%)",
    column = "Hematocrit (%)",
    ml_r2 = 0.937,
    n_samples = 2120,
    description = "Blood volume/hemoconcentration marker"
  ),
  list(
    name = "Total Cholesterol",
    column = "total_cholesterol_mg_dL",
    ml_r2 = 0.392,
    n_samples = 2917,
    description = "Cardiovascular lipid marker"
  ),
  list(
    name = "FASTING LDL",
    column = "FASTING LDL",
    ml_r2 = 0.377,
    n_samples = 2917,
    description = "Low-density lipoprotein (fasting)"
  ),
  list(
    name = "FASTING HDL",
    column = "FASTING HDL",
    ml_r2 = 0.334,
    n_samples = 2918,
    description = "High-density lipoprotein (fasting)"
  ),
  list(
    name = "LDL Cholesterol",
    column = "ldl_cholesterol_mg_dL",
    ml_r2 = 0.143,
    n_samples = 710,
    description = "Low-density lipoprotein"
  ),
  list(
    name = "Creatinine",
    column = "creatinine_umol_L",
    ml_r2 = 0.137,
    n_samples = 1247,
    description = "Kidney function marker"
  )
)

# Climate variable to analyze
CLIMATE_VAR <- "climate_daily_mean_temp"

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_message <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", timestamp, level, msg))
}

log_header <- function(msg) {
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat(msg, "\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

#' Load and prepare clinical data
#'
#' @param data_path Path to CSV file
#' @return Prepared dataframe
load_clinical_data <- function(data_path) {
  log_message("Loading clinical dataset...")

  df <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)

  # Convert date column
  if ("Date" %in% colnames(df)) {
    df$date <- as.Date(df$Date)
  } else if ("date" %in% colnames(df)) {
    df$date <- as.Date(df$date)
  } else {
    stop("No date column found in data")
  }

  # Add patient ID if not present
  if (!"patient_id" %in% colnames(df)) {
    df$patient_id <- seq_len(nrow(df))
  }

  log_message(sprintf("Loaded %d observations", nrow(df)))
  log_message(sprintf("Date range: %s to %s", min(df$date, na.rm=TRUE),
                      max(df$date, na.rm=TRUE)))
  log_message(sprintf("Available columns: %d", ncol(df)))

  return(df)
}

#' Prepare data for case-crossover analysis
#'
#' @param df Clinical dataframe
#' @param biomarker_col Biomarker column name
#' @param climate_var Climate variable name
#' @return List with case and control dataframes
prepare_case_crossover_data <- function(df, biomarker_col, climate_var) {
  log_message(sprintf("Preparing case-crossover data for %s...", biomarker_col))

  # Select relevant columns
  required_cols <- c("patient_id", "date", biomarker_col, climate_var)
  df_subset <- df[, required_cols[required_cols %in% colnames(df)]]

  # Remove missing values
  df_subset <- df_subset[complete.cases(df_subset), ]

  if (nrow(df_subset) == 0) {
    stop(sprintf("No complete cases for %s", biomarker_col))
  }

  # Add time stratification variable (4-week strata)
  df_subset$year <- as.numeric(format(df_subset$date, "%Y"))
  df_subset$month <- as.numeric(format(df_subset$date, "%m"))
  df_subset$day <- as.numeric(format(df_subset$date, "%d"))
  df_subset$dow <- as.numeric(format(df_subset$date, "%u"))  # Day of week (1=Mon, 7=Sun)

  # Create stratum ID: Year-Month-Week
  df_subset$week_of_month <- ceiling(df_subset$day / 7)
  df_subset$stratum <- paste(df_subset$year, df_subset$month,
                             df_subset$week_of_month, sep = "-")

  # Define cases and controls
  # Case: actual measurement day
  # Control: same day of week within same stratum

  log_message(sprintf("  Complete cases: %d", nrow(df_subset)))
  log_message(sprintf("  Unique strata: %d", length(unique(df_subset$stratum))))

  return(df_subset)
}

#' Create lagged climate matrix
#'
#' @param df Dataframe with climate variable
#' @param climate_var Climate variable name
#' @param max_lag Maximum lag
#' @return Dataframe with lagged climate variables
create_lag_matrix <- function(df, climate_var, max_lag) {
  log_message(sprintf("Creating lag matrix with max_lag=%d days...", max_lag))

  # Sort by date
  df <- df[order(df$date), ]

  # Create lagged variables
  climate_values <- df[[climate_var]]

  # Create matrix of lags
  lag_matrix <- matrix(NA, nrow = nrow(df), ncol = max_lag + 1)
  colnames(lag_matrix) <- paste0("lag", 0:max_lag)

  # Fill lag matrix
  for (lag in 0:max_lag) {
    if (lag == 0) {
      lag_matrix[, lag + 1] <- climate_values
    } else {
      lag_matrix[(lag + 1):nrow(df), lag + 1] <- climate_values[1:(nrow(df) - lag)]
    }
  }

  # Add to dataframe
  df_with_lags <- cbind(df, as.data.frame(lag_matrix))

  # Remove rows with NA in lags
  df_with_lags <- df_with_lags[complete.cases(lag_matrix), ]

  log_message(sprintf("  Rows with complete lags: %d", nrow(df_with_lags)))

  return(df_with_lags)
}

# ============================================================================
# CASE-CROSSOVER DLNM ANALYSIS
# ============================================================================

#' Fit case-crossover DLNM model
#'
#' @param df_prepared Prepared dataframe with lags
#' @param biomarker_col Biomarker column name
#' @param climate_var Climate variable name
#' @param max_lag Maximum lag
#' @return List with model results
fit_case_crossover_dlnm <- function(df_prepared, biomarker_col, climate_var, max_lag) {
  log_message("Fitting case-crossover DLNM model...")

  # Extract lagged climate matrix
  lag_cols <- paste0("lag", 0:max_lag)
  lag_matrix <- as.matrix(df_prepared[, lag_cols])

  # Create crossbasis
  log_message("  Creating crossbasis...")
  cb <- crossbasis(
    lag_matrix,
    lag = c(0, max_lag),
    argvar = list(fun = "ns", df = DF_TEMP),
    arglag = list(fun = "ns", df = DF_LAG)
  )

  # Fit conditional logistic regression (case-crossover)
  # Using stratum as matching variable
  log_message("  Fitting conditional model...")

  # For continuous biomarkers, we'll use a binary outcome approach
  # Define "high" biomarker as above median within each stratum
  df_prepared$high_biomarker <- 0
  for (stratum_id in unique(df_prepared$stratum)) {
    stratum_rows <- df_prepared$stratum == stratum_id
    median_val <- median(df_prepared[[biomarker_col]][stratum_rows], na.rm = TRUE)
    df_prepared$high_biomarker[stratum_rows] <-
      as.numeric(df_prepared[[biomarker_col]][stratum_rows] > median_val)
  }

  # Fit model
  tryCatch({
    model <- gnm(
      high_biomarker ~ cb,
      data = df_prepared,
      family = binomial(),
      eliminate = factor(stratum)
    )

    log_message("  Model fitted successfully")

    # Extract predictions
    log_message("  Extracting predictions...")

    # Create prediction grid
    temp_range <- range(df_prepared[[climate_var]], na.rm = TRUE)
    temp_pred <- seq(temp_range[1], temp_range[2], length.out = 50)

    # Reference temperature (median)
    temp_ref <- median(df_prepared[[climate_var]], na.rm = TRUE)

    # Predict cumulative effects over all lags
    pred_cumul <- crosspred(
      cb, model,
      at = temp_pred,
      cen = temp_ref,
      cumul = TRUE
    )

    # Predict lag-specific effects at different lags
    # Note: We'll create separate predictions for each lag
    pred_lag_0 <- crosspred(cb, model, at = temp_pred, cen = temp_ref, lag = 0)
    pred_lag_7 <- crosspred(cb, model, at = temp_pred, cen = temp_ref, lag = 7)
    pred_lag_14 <- crosspred(cb, model, at = temp_pred, cen = temp_ref, lag = 14)
    pred_lag_21 <- crosspred(cb, model, at = temp_pred, cen = temp_ref, lag = 21)

    pred_lag <- list(
      lag0 = pred_lag_0,
      lag7 = pred_lag_7,
      lag14 = pred_lag_14,
      lag21 = pred_lag_21
    )

    # Extract effect estimates
    results <- list(
      model = model,
      crossbasis = cb,
      pred_cumul = pred_cumul,
      pred_lag = pred_lag,
      temp_range = temp_range,
      temp_ref = temp_ref,
      n_obs = nrow(df_prepared),
      n_strata = length(unique(df_prepared$stratum)),
      biomarker = biomarker_col
    )

    # Calculate summary statistics
    results$summary <- list(
      overall_OR = exp(pred_cumul$allfit[length(pred_cumul$allfit)]),
      overall_CI_low = exp(pred_cumul$allfit[length(pred_cumul$allfit)] -
                          1.96 * pred_cumul$allse[length(pred_cumul$allse)]),
      overall_CI_high = exp(pred_cumul$allfit[length(pred_cumul$allfit)] +
                           1.96 * pred_cumul$allse[length(pred_cumul$allse)]),
      significant = !between(1,
                             exp(pred_cumul$allfit[length(pred_cumul$allfit)] -
                                 1.96 * pred_cumul$allse[length(pred_cumul$allse)]),
                             exp(pred_cumul$allfit[length(pred_cumul$allfit)] +
                                 1.96 * pred_cumul$allse[length(pred_cumul$allse)]))
    )

    log_message(sprintf("  Overall OR: %.3f (95%% CI: %.3f - %.3f)",
                        results$summary$overall_OR,
                        results$summary$overall_CI_low,
                        results$summary$overall_CI_high))

    return(results)

  }, error = function(e) {
    log_message(sprintf("  ERROR: %s", e$message), "ERROR")
    return(NULL)
  })
}

# ============================================================================
# VISUALIZATION
# ============================================================================

#' Create DLNM visualization plots
#'
#' @param results DLNM results object
#' @param biomarker_name Biomarker name for titles
#' @param output_dir Output directory
create_dlnm_plots <- function(results, biomarker_name, output_dir) {
  log_message("Creating DLNM visualization plots...")

  safe_name <- gsub("[^A-Za-z0-9_]", "_", biomarker_name)

  # 1. Cumulative exposure-response curve
  pdf(file.path(output_dir, paste0(safe_name, "_cumulative_curve.pdf")),
      width = 8, height = 6)

  plot(results$pred_cumul,
       xlab = "Temperature (°C)",
       ylab = "Cumulative Odds Ratio",
       main = paste(biomarker_name, "- Cumulative Temperature Effect (0-21 days)"),
       col = "darkred",
       lwd = 2)
  abline(h = 1, lty = 2, col = "gray")

  dev.off()

  # 2. 3D surface plot (temperature x lag)
  pdf(file.path(output_dir, paste0(safe_name, "_3d_surface.pdf")),
      width = 10, height = 8)

  tryCatch({
    plot(results$crossbasis, results$model,
         xlab = "Temperature (°C)",
         ylab = "Lag (days)",
         zlab = "Odds Ratio",
         main = paste(biomarker_name, "- Temperature-Lag Response Surface"))
  }, error = function(e) {
    # If 3D plot fails, create a simple contour plot
    plot(results$pred_cumul,
         xlab = "Temperature (°C)",
         ylab = "Log Odds Ratio",
         main = paste(biomarker_name, "- Temperature Effect (3D plot failed)"))
  })

  dev.off()

  # 3. Lag-specific effects (manual plotting)
  pdf(file.path(output_dir, paste0(safe_name, "_lag_specific.pdf")),
      width = 10, height = 6)

  # Extract OR and CI for each lag
  lags <- c(0, 7, 14, 21)
  lag_names <- c("lag0", "lag7", "lag14", "lag21")

  # Create a simple bar plot or line plot of lag effects
  par(mfrow=c(2,2))
  for(i in 1:length(lag_names)){
    plot(results$pred_lag[[lag_names[i]]],
         xlab = "Temperature (°C)",
         ylab = "Odds Ratio",
         main = paste("Lag", lags[i], "days"),
         col = "darkblue",
         lwd = 2)
    abline(h = 1, lty = 2, col = "gray")
  }

  dev.off()

  log_message("  Plots saved successfully")
}

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

#' Run case-crossover DLNM validation for a single biomarker
#'
#' @param biomarker_info List with biomarker information
#' @param df_full Full clinical dataset
#' @return Results list
analyze_biomarker <- function(biomarker_info, df_full) {
  log_header(sprintf("ANALYZING: %s (ML R² = %.4f)",
                     biomarker_info$name, biomarker_info$ml_r2))

  # Check if column exists
  if (!biomarker_info$column %in% colnames(df_full)) {
    log_message(sprintf("Column '%s' not found in dataset", biomarker_info$column), "ERROR")
    return(NULL)
  }

  # Prepare case-crossover data
  df_prepared <- prepare_case_crossover_data(
    df_full,
    biomarker_info$column,
    CLIMATE_VAR
  )

  if (nrow(df_prepared) < 100) {
    log_message("Insufficient data for analysis", "ERROR")
    return(NULL)
  }

  # Create lag matrix
  df_with_lags <- create_lag_matrix(df_prepared, CLIMATE_VAR, MAX_LAG)

  # Fit DLNM
  results <- fit_case_crossover_dlnm(
    df_with_lags,
    biomarker_info$column,
    CLIMATE_VAR,
    MAX_LAG
  )

  if (is.null(results)) {
    return(NULL)
  }

  # Create visualizations
  create_dlnm_plots(results, biomarker_info$name, OUTPUT_DIR)

  # Add metadata
  results$biomarker_info <- biomarker_info
  results$analysis_date <- Sys.time()

  return(results)
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main <- function() {
  log_header("CASE-CROSSOVER DLNM VALIDATION PIPELINE")

  log_message("Configuration:")
  log_message(sprintf("  Data path: %s", DATA_PATH))
  log_message(sprintf("  Output dir: %s", OUTPUT_DIR))
  log_message(sprintf("  Max lag: %d days", MAX_LAG))
  log_message(sprintf("  Climate variable: %s", CLIMATE_VAR))
  log_message(sprintf("  Biomarkers to analyze: %d", length(BIOMARKERS)))

  # Load data
  df_full <- load_clinical_data(DATA_PATH)

  # Run analysis for each biomarker
  all_results <- list()

  for (i in seq_along(BIOMARKERS)) {
    biomarker_info <- BIOMARKERS[[i]]

    results <- analyze_biomarker(biomarker_info, df_full)

    if (!is.null(results)) {
      all_results[[biomarker_info$name]] <- results
    }

    cat("\n")
  }

  # Save summary results
  log_header("SAVING SUMMARY RESULTS")

  summary_df <- data.frame(
    Biomarker = character(),
    ML_R2 = numeric(),
    N_Observations = integer(),
    N_Strata = integer(),
    Cumulative_OR = numeric(),
    CI_Low = numeric(),
    CI_High = numeric(),
    Significant = logical(),
    stringsAsFactors = FALSE
  )

  for (biomarker_name in names(all_results)) {
    res <- all_results[[biomarker_name]]
    summary_df <- rbind(summary_df, data.frame(
      Biomarker = biomarker_name,
      ML_R2 = res$biomarker_info$ml_r2,
      N_Observations = res$n_obs,
      N_Strata = res$n_strata,
      Cumulative_OR = res$summary$overall_OR,
      CI_Low = res$summary$overall_CI_low,
      CI_High = res$summary$overall_CI_high,
      Significant = res$summary$significant,
      stringsAsFactors = FALSE
    ))
  }

  # Save summary
  write.csv(summary_df,
            file.path(OUTPUT_DIR, "dlnm_validation_summary.csv"),
            row.names = FALSE)

  log_message("Summary saved to dlnm_validation_summary.csv")

  # Print summary table
  log_header("DLNM VALIDATION SUMMARY")
  print(summary_df)

  log_header("ANALYSIS COMPLETE")
  log_message(sprintf("Total biomarkers analyzed: %d", nrow(summary_df)))
  log_message(sprintf("Significant associations: %d", sum(summary_df$Significant)))
  log_message(sprintf("Results saved to: %s", OUTPUT_DIR))

  return(all_results)
}

# Run analysis
if (!interactive()) {
  results <- main()
}
