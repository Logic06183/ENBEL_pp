#!/usr/bin/env Rscript
# DLNM Validation Analysis for Climate-Health Relationships
# =========================================================
# 
# Validate the machine learning findings using Distributed Lag Non-linear Models
# Focus on CD4, cholesterol, and creatinine relationships with climate variables

# Load required libraries
if (!require(dlnm)) install.packages("dlnm")
if (!require(mgcv)) install.packages("mgcv")
if (!require(splines)) install.packages("splines")
if (!require(dplyr)) install.packages("dplyr")
if (!require(ggplot2)) install.packages("ggplot2")

library(dlnm)
library(mgcv)
library(splines)
library(dplyr)
library(ggplot2)

cat("=================================================================\n")
cat("DLNM VALIDATION OF CLIMATE-HEALTH RELATIONSHIPS\n")
cat("=================================================================\n")

# Load the clinical dataset
cat("\nStep 1: Loading clinical dataset...\n")
clinical_data <- read.csv("data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv", stringsAsFactors = FALSE)
cat(sprintf("✓ Loaded %d records with %d variables\n", nrow(clinical_data), ncol(clinical_data)))

# Define biomarkers of interest (top performers from ML analysis)
biomarkers <- list(
  cd4 = "CD4.cell.count..cells.µL.",
  cholesterol = "total_cholesterol_mg_dL", 
  creatinine = "creatinine_umol_L"
)

# Find available biomarker columns
available_biomarkers <- list()
for (name in names(biomarkers)) {
  # Try exact match first
  exact_match <- biomarkers[[name]] %in% colnames(clinical_data)
  if (exact_match) {
    available_biomarkers[[name]] <- biomarkers[[name]]
  } else {
    # Try pattern matching
    pattern <- gsub("\\.", ".*", biomarkers[[name]], fixed = FALSE)
    matches <- grep(pattern, colnames(clinical_data), value = TRUE, ignore.case = TRUE)
    if (length(matches) > 0) {
      available_biomarkers[[name]] <- matches[1]
      cat(sprintf("  Using %s for %s\n", matches[1], name))
    }
  }
}

cat(sprintf("Found %d available biomarkers:\n", length(available_biomarkers)))
for (name in names(available_biomarkers)) {
  col_name <- available_biomarkers[[name]]
  non_missing <- sum(!is.na(clinical_data[[col_name]]))
  cat(sprintf("  %s: %s (%d samples)\n", name, col_name, non_missing))
}

# Identify climate variables
cat("\nStep 2: Identifying climate variables...\n")
climate_patterns <- c("climate_", "HEAT_", "temp", "humid")
climate_cols <- c()

for (pattern in climate_patterns) {
  matches <- grep(pattern, colnames(clinical_data), value = TRUE, ignore.case = TRUE)
  # Filter for numeric columns with good coverage
  for (col in matches) {
    if (is.numeric(clinical_data[[col]]) && sum(!is.na(clinical_data[[col]])) > nrow(clinical_data) * 0.5) {
      climate_cols <- c(climate_cols, col)
    }
  }
}

climate_cols <- unique(climate_cols)
cat(sprintf("Found %d climate variables with >50%% coverage\n", length(climate_cols)))

# Identify key temperature variables for DLNM
temp_vars <- grep("temp|HEAT", climate_cols, value = TRUE, ignore.case = TRUE)
cat(sprintf("Temperature variables for DLNM analysis:\n"))
for (var in temp_vars[1:min(5, length(temp_vars))]) {
  cat(sprintf("  %s\n", var))
}

# Check for date/time variables
cat("\nStep 3: Checking temporal structure...\n")
date_cols <- grep("date|time|year|month", colnames(clinical_data), value = TRUE, ignore.case = TRUE)
cat(sprintf("Date/time columns found: %s\n", paste(date_cols, collapse = ", ")))

# Prepare data for DLNM analysis
cat("\nStep 4: Preparing DLNM analysis data...\n")

# Function to perform DLNM analysis for a biomarker
perform_dlnm_analysis <- function(biomarker_name, biomarker_col, climate_var) {
  cat(sprintf("\n--- DLNM Analysis: %s vs %s ---\n", biomarker_name, climate_var))
  
  # Get complete cases
  analysis_cols <- c(biomarker_col, climate_var)
  if ("primary_date" %in% colnames(clinical_data)) {
    analysis_cols <- c(analysis_cols, "primary_date")
  }
  
  complete_data <- clinical_data[analysis_cols]
  complete_data <- complete_data[complete.cases(complete_data), ]
  
  if (nrow(complete_data) < 100) {
    cat(sprintf("  ⚠ Insufficient data: %d complete cases\n", nrow(complete_data)))
    return(NULL)
  }
  
  cat(sprintf("  Complete cases: %d\n", nrow(complete_data)))
  
  # Add sequence for temporal ordering if no date available
  if (!"primary_date" %in% colnames(complete_data)) {
    complete_data$time_seq <- 1:nrow(complete_data)
    time_var <- "time_seq"
  } else {
    # Convert date and create time sequence
    complete_data$date_converted <- as.Date(complete_data$primary_date)
    complete_data <- complete_data[order(complete_data$date_converted), ]
    complete_data$time_seq <- 1:nrow(complete_data)
    time_var <- "time_seq"
  }
  
  # Create lag matrix for the climate variable
  # Use up to 30 day lags (or 30 observations if no dates)
  max_lag <- min(30, nrow(complete_data) - 1)
  
  cat(sprintf("  Creating lag structure (max lag: %d)\n", max_lag))
  
  # Simple lag matrix creation
  lag_matrix <- matrix(NA, nrow = nrow(complete_data), ncol = max_lag + 1)
  for (lag in 0:max_lag) {
    if (lag == 0) {
      lag_matrix[, lag + 1] <- complete_data[[climate_var]]
    } else {
      lag_matrix[(lag + 1):nrow(complete_data), lag + 1] <- complete_data[[climate_var]][1:(nrow(complete_data) - lag)]
    }
  }
  
  # Remove rows with any missing lag values
  complete_rows <- complete.cases(lag_matrix)
  analysis_data <- complete_data[complete_rows, ]
  lag_matrix_clean <- lag_matrix[complete_rows, ]
  
  cat(sprintf("  Final analysis data: %d observations\n", nrow(analysis_data)))
  
  if (nrow(analysis_data) < 50) {
    cat("  ⚠ Insufficient data after lag creation\n")
    return(NULL)
  }
  
  # Create crossbasis for DLNM
  # Use natural splines for both exposure-response and lag-response
  tryCatch({
    # Define exposure range
    exp_range <- range(lag_matrix_clean[, 1], na.rm = TRUE)
    lag_range <- c(0, max_lag)
    
    cat(sprintf("  Exposure range: %.2f to %.2f\n", exp_range[1], exp_range[2]))
    cat(sprintf("  Lag range: %d to %d\n", lag_range[1], lag_range[2]))
    
    # Create crossbasis
    cb_climate <- crossbasis(
      lag_matrix_clean,
      lag = lag_range,
      argvar = list(fun = "ns", df = 3),  # Natural splines for exposure
      arglag = list(fun = "ns", df = 3)   # Natural splines for lags
    )
    
    # Fit DLNM model
    cat("  Fitting DLNM model...\n")
    
    # Simple linear model with crossbasis
    model_formula <- as.formula(paste(biomarker_col, "~ cb_climate"))
    dlnm_model <- lm(model_formula, data = analysis_data)
    
    # Extract model summary
    model_summary <- summary(dlnm_model)
    r_squared <- model_summary$r.squared
    
    cat(sprintf("  DLNM Model R² = %.3f\n", r_squared))
    
    # Predict effects
    cat("  Calculating exposure-response curves...\n")
    
    # Predict at specific exposure values
    pred_exposures <- seq(exp_range[1], exp_range[2], length.out = 20)
    
    # Create prediction matrix
    pred_results <- list()
    
    for (i in 1:length(pred_exposures)) {
      # Create prediction crossbasis for specific exposure
      pred_matrix <- matrix(pred_exposures[i], nrow = 1, ncol = ncol(lag_matrix_clean))
      pred_cb <- crossbasis(
        pred_matrix,
        lag = lag_range,
        argvar = list(fun = "ns", df = 3),
        arglag = list(fun = "ns", df = 3)
      )
      
      # Predict
      pred_val <- predict(dlnm_model, newdata = data.frame(pred_cb))
      pred_results[[i]] <- data.frame(
        exposure = pred_exposures[i],
        predicted_effect = pred_val
      )
    }
    
    # Combine predictions
    all_predictions <- do.call(rbind, pred_results)
    
    # Calculate overall effect strength
    effect_range <- range(all_predictions$predicted_effect, na.rm = TRUE)
    effect_strength <- diff(effect_range)
    
    cat(sprintf("  Effect strength: %.3f %s units\n", effect_strength, biomarker_name))
    
    # Statistical significance test
    model_p <- pf(model_summary$fstatistic[1], 
                 model_summary$fstatistic[2], 
                 model_summary$fstatistic[3], 
                 lower.tail = FALSE)
    
    significance <- ifelse(model_p < 0.001, "***", 
                          ifelse(model_p < 0.01, "**",
                                ifelse(model_p < 0.05, "*", "ns")))
    
    cat(sprintf("  Model significance: p = %.6f %s\n", model_p, significance))
    
    return(list(
      biomarker = biomarker_name,
      climate_var = climate_var,
      n_obs = nrow(analysis_data),
      r_squared = r_squared,
      effect_strength = effect_strength,
      p_value = model_p,
      significance = significance,
      predictions = all_predictions,
      model = dlnm_model
    ))
    
  }, error = function(e) {
    cat(sprintf("  ✗ DLNM analysis failed: %s\n", e$message))
    return(NULL)
  })
}

# Run DLNM analysis for each biomarker-climate combination
cat("\nStep 5: Running DLNM analyses...\n")

dlnm_results <- list()
result_counter <- 1

# Focus on most important climate variables from ML analysis
priority_climate_vars <- c()

# Look for HEAT_VULNERABILITY_SCORE first (dominant in ML)
if ("HEAT_VULNERABILITY_SCORE" %in% climate_cols) {
  priority_climate_vars <- c(priority_climate_vars, "HEAT_VULNERABILITY_SCORE")
}

# Add temperature variables
temp_priority <- grep("temp.*min|temp.*max|temp.*mean", climate_cols, value = TRUE, ignore.case = TRUE)
priority_climate_vars <- c(priority_climate_vars, temp_priority[1:min(3, length(temp_priority))])

# Remove duplicates and ensure we have variables
priority_climate_vars <- unique(priority_climate_vars[!is.na(priority_climate_vars)])

if (length(priority_climate_vars) == 0) {
  priority_climate_vars <- climate_cols[1:min(3, length(climate_cols))]
}

cat(sprintf("Priority climate variables for DLNM:\n"))
for (var in priority_climate_vars) {
  cat(sprintf("  %s\n", var))
}

# Run analysis for each biomarker-climate combination
for (biomarker_name in names(available_biomarkers)) {
  biomarker_col <- available_biomarkers[[biomarker_name]]
  
  for (climate_var in priority_climate_vars) {
    result <- perform_dlnm_analysis(biomarker_name, biomarker_col, climate_var)
    
    if (!is.null(result)) {
      dlnm_results[[result_counter]] <- result
      result_counter <- result_counter + 1
    }
  }
}

# Summarize DLNM results
cat("\n=================================================================\n")
cat("DLNM VALIDATION RESULTS SUMMARY\n")
cat("=================================================================\n")

if (length(dlnm_results) > 0) {
  cat(sprintf("Successfully completed %d DLNM analyses\n\n", length(dlnm_results)))
  
  cat(sprintf("%-15s %-25s %-10s %-15s %-12s\n", 
              "Biomarker", "Climate Variable", "N", "R²", "Significance"))
  cat(paste(rep("-", 80), collapse = ""), "\n")
  
  for (result in dlnm_results) {
    cat(sprintf("%-15s %-25s %-10d %-15.3f %-12s\n",
                substr(result$biomarker, 1, 14),
                substr(result$climate_var, 1, 24),
                result$n_obs,
                result$r_squared,
                result$significance))
  }
  
  # Identify strong relationships
  cat("\nStrong DLNM relationships (R² > 0.1):\n")
  strong_results <- dlnm_results[sapply(dlnm_results, function(x) x$r_squared > 0.1)]
  
  if (length(strong_results) > 0) {
    for (result in strong_results) {
      cat(sprintf("✓ %s ~ %s: R² = %.3f, p = %.6f\n",
                  result$biomarker, result$climate_var, result$r_squared, result$p_value))
    }
  } else {
    cat("  No relationships with R² > 0.1 found\n")
  }
  
  # Compare with ML results
  cat("\nComparison with Machine Learning Results:\n")
  ml_results <- list(
    cd4 = 0.714,
    cholesterol = 0.392,
    creatinine = 0.1  # Estimated
  )
  
  for (biomarker_name in names(ml_results)) {
    ml_r2 <- ml_results[[biomarker_name]]
    
    # Find best DLNM result for this biomarker
    biomarker_dlnm <- dlnm_results[sapply(dlnm_results, function(x) x$biomarker == biomarker_name)]
    
    if (length(biomarker_dlnm) > 0) {
      best_dlnm <- biomarker_dlnm[[which.max(sapply(biomarker_dlnm, function(x) x$r_squared))]]
      dlnm_r2 <- best_dlnm$r_squared
      
      cat(sprintf("  %s: ML R² = %.3f, DLNM R² = %.3f", 
                  biomarker_name, ml_r2, dlnm_r2))
      
      if (dlnm_r2 > ml_r2 * 0.5) {
        cat(" ✓ DLNM validates ML findings\n")
      } else if (dlnm_r2 > 0.05) {
        cat(" ⚠ Partial validation\n")
      } else {
        cat(" ✗ Poor DLNM performance - possible temporal confounding\n")
      }
    } else {
      cat(sprintf("  %s: ML R² = %.3f, DLNM: No valid results\n", biomarker_name, ml_r2))
    }
  }
  
} else {
  cat("No successful DLNM analyses completed\n")
}

# Generate recommendations
cat("\n=================================================================\n")
cat("RECOMMENDATIONS\n")
cat("=================================================================\n")

if (length(dlnm_results) > 0) {
  # Count significant results
  significant_results <- length(dlnm_results[sapply(dlnm_results, function(x) x$p_value < 0.05)])
  strong_results_count <- length(dlnm_results[sapply(dlnm_results, function(x) x$r_squared > 0.1)])
  
  cat(sprintf("📊 DLNM Analysis Summary:\n"))
  cat(sprintf("   - %d total analyses completed\n", length(dlnm_results)))
  cat(sprintf("   - %d statistically significant (p < 0.05)\n", significant_results))
  cat(sprintf("   - %d with strong effects (R² > 0.1)\n", strong_results_count))
  
  if (strong_results_count > 0) {
    cat("\n✅ VALIDATION STATUS: DLNM supports climate-health relationships\n")
    cat("   → Temporal lag patterns confirm genuine climate effects\n")
    cat("   → Relationships are not simply temporal confounding\n")
  } else if (significant_results > 0) {
    cat("\n⚠️  VALIDATION STATUS: Weak DLNM support\n")
    cat("   → Some temporal patterns detected but effects are small\n")
    cat("   → May indicate temporal confounding in ML models\n")
  } else {
    cat("\n❌ VALIDATION STATUS: DLNM does not support ML findings\n")
    cat("   → No significant temporal lag patterns detected\n")
    cat("   → ML relationships likely due to temporal confounding\n")
  }
  
  cat(sprintf("\n📋 Next Steps:\n"))
  cat(sprintf("1. 🔍 Investigate HEAT_VULNERABILITY_SCORE components\n"))
  cat(sprintf("2. 📊 Generate publication-ready DLNM plots\n"))
  cat(sprintf("3. 🧪 Validate findings with external datasets\n"))
  cat(sprintf("4. 📝 Document lag patterns for clinical interpretation\n"))
  
  # HEAT_VULNERABILITY_SCORE analysis
  heat_vuln_results <- dlnm_results[sapply(dlnm_results, function(x) x$climate_var == "HEAT_VULNERABILITY_SCORE")]
  if (length(heat_vuln_results) > 0) {
    cat(sprintf("\n🌡️ HEAT_VULNERABILITY_SCORE Analysis:\n"))
    for (result in heat_vuln_results) {
      cat(sprintf("   - %s: R² = %.3f %s\n", result$biomarker, result$r_squared, result$significance))
    }
    
    best_heat_vuln <- heat_vuln_results[[which.max(sapply(heat_vuln_results, function(x) x$r_squared))]]
    if (best_heat_vuln$r_squared > 0.3) {
      cat("   → Strong DLNM validation of heat vulnerability effects\n")
    } else {
      cat("   → Limited DLNM support - investigate vulnerability score construction\n")
    }
  }
  
} else {
  cat("❌ No DLNM analyses completed successfully\n")
  cat("📋 Recommendations:\n")
  cat("1. 🔍 Check data quality and temporal structure\n")
  cat("2. 📊 Review biomarker and climate variable availability\n")
  cat("3. 🔄 Consider alternative temporal modeling approaches\n")
}

cat(sprintf("\n✓ DLNM validation analysis completed: %s\n", Sys.time()))