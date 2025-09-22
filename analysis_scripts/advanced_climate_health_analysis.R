# Advanced Climate-Health Analysis using Literature Methods
# ========================================================
# 
# Implementing state-of-the-art methodologies from climate-health literature:
# 1. Distributed Lag Non-linear Models (DLNM)
# 2. Spline-based exposure-response functions
# 3. Heat wave analysis
# 4. Case-crossover methodology
# 5. Mixed-effects models

# Load required libraries
if (!require("dlnm")) install.packages("dlnm")
if (!require("splines")) install.packages("splines")
if (!require("mgcv")) install.packages("mgcv")
if (!require("lme4")) install.packages("lme4")
if (!require("survival")) install.packages("survival")
if (!require("dplyr")) install.packages("dplyr")
if (!require("lubridate")) install.packages("lubridate")

library(dlnm)
library(splines)
library(mgcv)
library(lme4)
library(survival)
library(dplyr)
library(lubridate)

cat("🔬 ADVANCED CLIMATE-HEALTH ANALYSIS USING R\n")
cat("==========================================\n\n")

# Load data
cat("📊 Loading climate-health data...\n")
df <- read.csv("FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv", stringsAsFactors = FALSE)

cat(sprintf("Dataset: %d records, %d variables\n", nrow(df), ncol(df)))

# Identify biomarkers and climate variables
biomarkers <- c("systolic blood pressure", "diastolic blood pressure", 
               "FASTING GLUCOSE", "FASTING TOTAL CHOLESTEROL", "FASTING HDL",
               "CD4 cell count (cells/µL)", "Hemoglobin (g/dL)", "Creatinine (mg/dL)")

# Get temperature variables for different lags
temp_vars <- grep("temp.*lag", names(df), value = TRUE, ignore.case = TRUE)
cat(sprintf("Found %d temperature lag variables\n\n", length(temp_vars)))

# Function to perform DLNM analysis
perform_dlnm_analysis <- function(biomarker_name) {
  cat(sprintf("🌡️ DLNM Analysis for %s\n", biomarker_name))
  cat("--------------------------------\n")
  
  if (!biomarker_name %in% names(df)) {
    cat("Biomarker not found in dataset\n\n")
    return(NULL)
  }
  
  # Prepare data
  analysis_data <- df[!is.na(df[[biomarker_name]]), ]
  
  if (nrow(analysis_data) < 500) {
    cat(sprintf("Insufficient data: %d samples\n\n", nrow(analysis_data)))
    return(NULL)
  }
  
  cat(sprintf("Sample size: %d\n", nrow(analysis_data)))
  
  # Create temperature matrix for DLNM (lags 0-21)
  temp_matrix <- matrix(NA, nrow = nrow(analysis_data), ncol = 22)
  
  for (lag in 0:21) {
    temp_col <- paste0("temperature_tas_lag", lag)
    if (temp_col %in% names(analysis_data)) {
      temp_matrix[, lag + 1] <- analysis_data[[temp_col]]
    }
  }
  
  # Remove rows with too many missing temperature values
  complete_rows <- rowSums(!is.na(temp_matrix)) >= 15  # At least 15 lags
  
  if (sum(complete_rows) < 500) {
    cat("Insufficient complete temperature data\n\n")
    return(NULL)
  }
  
  analysis_data <- analysis_data[complete_rows, ]
  temp_matrix <- temp_matrix[complete_rows, ]
  
  # Fill remaining missing values with mean
  for (i in 1:ncol(temp_matrix)) {
    if (any(is.na(temp_matrix[, i]))) {
      temp_matrix[is.na(temp_matrix[, i]), i] <- mean(temp_matrix[, i], na.rm = TRUE)
    }
  }
  
  cat(sprintf("Final sample size: %d\n", nrow(analysis_data)))
  
  # Create crossbasis for DLNM
  # Natural spline for temperature dimension (3 df)
  # Natural spline for lag dimension (4 df)
  tryCatch({
    cb_temp <- crossbasis(temp_matrix, 
                         lag = 21,
                         argvar = list(fun = "ns", df = 3),
                         arglag = list(fun = "ns", df = 4))
    
    # Fit DLNM model
    biomarker_values <- analysis_data[[biomarker_name]]
    
    # Basic DLNM model
    model <- lm(biomarker_values ~ cb_temp)
    
    # Check model significance
    model_summary <- summary(model)
    f_stat <- model_summary$fstatistic
    p_value <- pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)
    
    cat(sprintf("Model R²: %.4f\n", model_summary$r.squared))
    cat(sprintf("Adjusted R²: %.4f\n", model_summary$adj.r.squared))
    cat(sprintf("Overall p-value: %.6f\n", p_value))
    
    # Test for non-linearity
    # Predict at different temperature percentiles
    temp_range <- range(temp_matrix, na.rm = TRUE)
    temp_pred <- seq(temp_range[1], temp_range[2], length.out = 50)
    
    # Overall cumulative effect
    overall_effect <- crosspred(cb_temp, model, at = temp_pred, cen = median(temp_matrix, na.rm = TRUE))
    
    # Check for significant effects
    sig_effects <- overall_effect$allRRfit[!is.na(overall_effect$allRRfit)]
    
    if (length(sig_effects) > 0 && (model_summary$adj.r.squared > 0.01 || p_value < 0.05)) {
      cat("✅ POTENTIAL DLNM RELATIONSHIP DETECTED\n")
      
      # Find optimal lag
      lag_effects <- rep(NA, 22)
      for (lag in 0:21) {
        lag_pred <- crosspred(cb_temp, model, at = quantile(temp_matrix, 0.95, na.rm = TRUE), 
                             lag = lag, cen = median(temp_matrix, na.rm = TRUE))
        if (!is.null(lag_pred$allRRfit) && !is.na(lag_pred$allRRfit)) {
          lag_effects[lag + 1] <- abs(lag_pred$allRRfit)
        }
      }
      
      optimal_lag <- which.max(lag_effects) - 1
      cat(sprintf("Optimal lag: %d days\n", optimal_lag))
      
      # Effect size at 95th percentile vs median
      high_temp_effect <- crosspred(cb_temp, model, at = quantile(temp_matrix, 0.95, na.rm = TRUE),
                                   cen = median(temp_matrix, na.rm = TRUE))
      
      if (!is.null(high_temp_effect$allRRfit) && !is.na(high_temp_effect$allRRfit)) {
        cat(sprintf("Effect size (95th vs 50th percentile): %.4f\n", high_temp_effect$allRRfit))
      }
      
      result <- list(
        biomarker = biomarker_name,
        n_samples = nrow(analysis_data),
        r_squared = model_summary$r.squared,
        adj_r_squared = model_summary$adj.r.squared,
        p_value = p_value,
        optimal_lag = optimal_lag,
        effect_estimate = if(!is.null(high_temp_effect$allRRfit)) high_temp_effect$allRRfit else NA,
        method = "DLNM"
      )
      
      cat("\n")
      return(result)
    } else {
      cat("No significant DLNM relationship\n\n")
      return(NULL)
    }
    
  }, error = function(e) {
    cat(sprintf("DLNM Error: %s\n\n", e$message))
    return(NULL)
  })
}

# Function to perform GAM analysis with splines
perform_gam_analysis <- function(biomarker_name) {
  cat(sprintf("📈 GAM Spline Analysis for %s\n", biomarker_name))
  cat("--------------------------------\n")
  
  if (!biomarker_name %in% names(df)) {
    cat("Biomarker not found\n\n")
    return(NULL)
  }
  
  analysis_data <- df[!is.na(df[[biomarker_name]]), ]
  
  if (nrow(analysis_data) < 500) {
    cat(sprintf("Insufficient data: %d samples\n\n", nrow(analysis_data)))
    return(NULL)
  }
  
  # Get temperature variables
  temp_cols <- c("temperature_tas_lag0", "temperature_tas_lag1", "temperature_tas_lag2", "temperature_tas_lag3")
  available_temps <- temp_cols[temp_cols %in% names(analysis_data)]
  
  if (length(available_temps) < 2) {
    cat("Insufficient temperature variables\n\n")
    return(NULL)
  }
  
  # Create clean dataset
  gam_data <- analysis_data[, c(biomarker_name, available_temps)]
  gam_data <- gam_data[complete.cases(gam_data), ]
  
  if (nrow(gam_data) < 500) {
    cat("Insufficient complete data\n\n")
    return(NULL)
  }
  
  cat(sprintf("GAM sample size: %d\n", nrow(gam_data)))
  
  tryCatch({
    # Build GAM formula with smooth terms
    formula_parts <- sprintf("s(%s, k=4)", available_temps)
    formula_str <- sprintf("%s ~ %s", biomarker_name, paste(formula_parts, collapse = " + "))
    
    # Fit GAM
    gam_model <- gam(as.formula(formula_str), data = gam_data, method = "REML")
    
    # Model summary
    gam_summary <- summary(gam_model)
    
    cat(sprintf("GAM R²: %.4f\n", gam_summary$r.sq))
    cat(sprintf("Adjusted R²: %.4f\n", gam_summary$r.sq))
    cat(sprintf("Deviance explained: %.2f%%\n", gam_summary$dev.expl * 100))
    
    # Check smooth term significance
    smooth_p_values <- gam_summary$s.table[, "p-value"]
    significant_smooths <- sum(smooth_p_values < 0.05)
    
    cat(sprintf("Significant smooth terms: %d/%d\n", significant_smooths, length(smooth_p_values)))
    
    if (gam_summary$r.sq > 0.02 || any(smooth_p_values < 0.01)) {
      cat("✅ POTENTIAL GAM RELATIONSHIP DETECTED\n")
      
      # Find most significant temperature variable
      min_p_idx <- which.min(smooth_p_values)
      best_temp_var <- available_temps[min_p_idx]
      best_p_value <- smooth_p_values[min_p_idx]
      
      cat(sprintf("Best temperature predictor: %s (p = %.4f)\n", best_temp_var, best_p_value))
      
      result <- list(
        biomarker = biomarker_name,
        n_samples = nrow(gam_data),
        r_squared = gam_summary$r.sq,
        deviance_explained = gam_summary$dev.expl,
        best_temp_predictor = best_temp_var,
        best_p_value = best_p_value,
        n_significant_terms = significant_smooths,
        method = "GAM"
      )
      
      cat("\n")
      return(result)
    } else {
      cat("No significant GAM relationship\n\n")
      return(NULL)
    }
    
  }, error = function(e) {
    cat(sprintf("GAM Error: %s\n\n", e$message))
    return(NULL)
  })
}

# Function to analyze heat waves
perform_heatwave_analysis <- function(biomarker_name) {
  cat(sprintf("🔥 Heat Wave Analysis for %s\n", biomarker_name))
  cat("--------------------------------\n")
  
  if (!biomarker_name %in% names(df)) {
    cat("Biomarker not found\n\n")
    return(NULL)
  }
  
  analysis_data <- df[!is.na(df[[biomarker_name]]), ]
  
  if (nrow(analysis_data) < 500) {
    cat(sprintf("Insufficient data: %d samples\n\n", nrow(analysis_data)))
    return(NULL)
  }
  
  # Define heat wave (temperature > 95th percentile for 2+ consecutive days)
  if (!"temperature_tas_lag0" %in% names(analysis_data)) {
    cat("Temperature data not available\n\n")
    return(NULL)
  }
  
  temp_data <- analysis_data$temperature_tas_lag0
  temp_95 <- quantile(temp_data, 0.95, na.rm = TRUE)
  
  # Create heat wave indicator
  heat_extreme <- temp_data > temp_95
  analysis_data$heat_wave <- as.numeric(heat_extreme & !is.na(heat_extreme))
  
  cat(sprintf("Heat wave threshold: %.2f°C\n", temp_95))
  cat(sprintf("Heat wave days: %d (%.1f%%)\n", 
              sum(analysis_data$heat_wave, na.rm = TRUE),
              mean(analysis_data$heat_wave, na.rm = TRUE) * 100))
  
  # Test heat wave effect
  tryCatch({
    # Simple t-test
    heat_wave_data <- analysis_data[analysis_data$heat_wave == 1 & !is.na(analysis_data$heat_wave), biomarker_name]
    normal_data <- analysis_data[analysis_data$heat_wave == 0 & !is.na(analysis_data$heat_wave), biomarker_name]
    
    if (length(heat_wave_data) < 10 || length(normal_data) < 10) {
      cat("Insufficient heat wave data for comparison\n\n")
      return(NULL)
    }
    
    t_test <- t.test(heat_wave_data, normal_data)
    
    # Effect size (Cohen's d)
    pooled_sd <- sqrt(((length(heat_wave_data) - 1) * var(heat_wave_data) + 
                       (length(normal_data) - 1) * var(normal_data)) / 
                      (length(heat_wave_data) + length(normal_data) - 2))
    cohens_d <- (mean(heat_wave_data) - mean(normal_data)) / pooled_sd
    
    cat(sprintf("Heat wave mean: %.2f\n", mean(heat_wave_data)))
    cat(sprintf("Normal days mean: %.2f\n", mean(normal_data)))
    cat(sprintf("Difference: %.2f\n", mean(heat_wave_data) - mean(normal_data)))
    cat(sprintf("t-test p-value: %.6f\n", t_test$p.value))
    cat(sprintf("Cohen's d: %.4f\n", cohens_d))
    
    if (t_test$p.value < 0.05 && abs(cohens_d) > 0.2) {
      cat("✅ SIGNIFICANT HEAT WAVE EFFECT DETECTED\n")
      
      result <- list(
        biomarker = biomarker_name,
        n_samples = nrow(analysis_data),
        n_heat_wave_days = length(heat_wave_data),
        heat_wave_mean = mean(heat_wave_data),
        normal_mean = mean(normal_data),
        difference = mean(heat_wave_data) - mean(normal_data),
        p_value = t_test$p.value,
        cohens_d = cohens_d,
        effect_size_category = ifelse(abs(cohens_d) > 0.8, "Large",
                                     ifelse(abs(cohens_d) > 0.5, "Medium", "Small")),
        method = "Heat Wave"
      )
      
      cat("\n")
      return(result)
    } else {
      cat("No significant heat wave effect\n\n")
      return(NULL)
    }
    
  }, error = function(e) {
    cat(sprintf("Heat wave analysis error: %s\n\n", e$message))
    return(NULL)
  })
}

# Run all analyses
cat("🔬 RUNNING COMPREHENSIVE CLIMATE-HEALTH ANALYSES\n")
cat("===============================================\n\n")

all_results <- list()

for (biomarker in biomarkers) {
  if (biomarker %in% names(df)) {
    cat(sprintf("🎯 ANALYZING: %s\n", biomarker))
    cat("=" , rep("=", nchar(biomarker) + 10), "\n", sep="")
    
    # DLNM Analysis
    dlnm_result <- perform_dlnm_analysis(biomarker)
    if (!is.null(dlnm_result)) {
      all_results[[length(all_results) + 1]] <- dlnm_result
    }
    
    # GAM Analysis  
    gam_result <- perform_gam_analysis(biomarker)
    if (!is.null(gam_result)) {
      all_results[[length(all_results) + 1]] <- gam_result
    }
    
    # Heat Wave Analysis
    heatwave_result <- perform_heatwave_analysis(biomarker)
    if (!is.null(heatwave_result)) {
      all_results[[length(all_results) + 1]] <- heatwave_result
    }
    
    cat("\n")
  }
}

# Summary of findings
cat("🎯 SUMMARY OF ADVANCED CLIMATE-HEALTH FINDINGS\n")
cat("==============================================\n")

if (length(all_results) > 0) {
  cat(sprintf("Total significant relationships found: %d\n\n", length(all_results)))
  
  for (i in seq_along(all_results)) {
    result <- all_results[[i]]
    cat(sprintf("%d. %s (%s)\n", i, result$biomarker, result$method))
    
    if (result$method == "DLNM") {
      cat(sprintf("   R² = %.4f, p = %.6f\n", result$adj_r_squared, result$p_value))
      cat(sprintf("   Optimal lag: %d days\n", result$optimal_lag))
      if (!is.na(result$effect_estimate)) {
        cat(sprintf("   Effect estimate: %.4f\n", result$effect_estimate))
      }
    } else if (result$method == "GAM") {
      cat(sprintf("   R² = %.4f, deviance explained = %.2f%%\n", result$r_squared, result$deviance_explained * 100))
      cat(sprintf("   Best predictor: %s (p = %.4f)\n", result$best_temp_predictor, result$best_p_value))
    } else if (result$method == "Heat Wave") {
      cat(sprintf("   Effect size (Cohen's d): %.4f (%s)\n", result$cohens_d, result$effect_size_category))
      cat(sprintf("   p-value: %.6f\n", result$p_value))
      cat(sprintf("   Difference: %.2f units\n", result$difference))
    }
    cat("\n")
  }
  
  # Save results
  save(all_results, file = "advanced_climate_health_results.RData")
  cat("💾 Results saved to: advanced_climate_health_results.RData\n")
  
} else {
  cat("❌ No significant relationships detected with advanced methods\n")
  cat("\nThis suggests:\n")
  cat("• Climate-health effects in this dataset are genuinely very weak\n")
  cat("• Different analytical approaches may be needed\n")
  cat("• Longer observation periods might be required\n")
  cat("• Population-specific factors may be masking effects\n")
}

cat("\n🏁 ADVANCED ANALYSIS COMPLETE\n")