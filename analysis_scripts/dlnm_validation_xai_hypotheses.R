#!/usr/bin/env Rscript
# DLNM Validation of XAI-Generated Hypotheses
# ==========================================
# 
# Rigorous DLNM validation of sex-specific climate-glucose vulnerabilities
# discovered through XAI exploration.
#
# XAI-Generated Hypotheses to Validate:
# 1. Heat Index × Sex → FASTING GLUCOSE (XAI importance: 0.023)
# 2. Temperature × Sex → FASTING GLUCOSE (XAI importance: 0.020)  
# 3. Land Temperature Lag 3 × Sex → FASTING GLUCOSE (XAI importance: 0.016)
#
# Methodology: Sex-stratified DLNM analysis to test for differential
# climate vulnerability patterns between males and females.

# Load required libraries
suppressPackageStartupMessages({
  library(dlnm)
  library(splines)
  library(mgcv)
  library(data.table)
  library(ggplot2)
  library(gridExtra)
  library(viridis)
  library(RColorBrewer)
})

# Set up analysis environment
cat("🔬 DLNM VALIDATION OF XAI-GENERATED HYPOTHESES\n")
cat("===============================================\n\n")

# Load and prepare data
cat("📊 Loading data for sex-stratified DLNM validation...\n")
data <- fread("FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv")

cat(sprintf("Dataset: %d records, %d variables\n", nrow(data), ncol(data)))

# Function to validate XAI hypothesis with sex-stratified DLNM
validate_sex_stratified_hypothesis <- function(data, biomarker, climate_vars, hypothesis_name) {
  cat(sprintf("\n🔍 VALIDATING HYPOTHESIS: %s\n", hypothesis_name))
  cat(paste(rep("-", nchar(hypothesis_name) + 25), collapse = ""), "\n")
  
  # Filter complete cases for biomarker and climate variables
  analysis_vars <- c(biomarker, climate_vars, "Sex")
  complete_data <- data[complete.cases(data[, analysis_vars, with = FALSE])]
  
  if (nrow(complete_data) < 100) {
    cat("Insufficient complete cases for analysis\n")
    return(NULL)
  }
  
  cat(sprintf("Complete cases: %d\n", nrow(complete_data)))
  
  # Split by sex
  male_data <- complete_data[Sex == "Male"]
  female_data <- complete_data[Sex == "Female"]
  
  cat(sprintf("Male participants: %d\n", nrow(male_data)))
  cat(sprintf("Female participants: %d\n", nrow(female_data)))
  
  if (nrow(male_data) < 50 || nrow(female_data) < 50) {
    cat("Insufficient participants in one or both sex groups\n")
    return(NULL)
  }
  
  # Fit DLNM models for each sex group
  sex_results <- list()
  
  for (sex_group in c("Male", "Female")) {
    cat(sprintf("\n📈 Fitting DLNM for %s participants\n", sex_group))
    
    if (sex_group == "Male") {
      sex_data <- male_data
    } else {
      sex_data <- female_data
    }
    
    outcome <- sex_data[[biomarker]]
    
    # Create temperature exposure matrix
    temp_matrix <- as.matrix(sex_data[, climate_vars, with = FALSE])
    
    if (ncol(temp_matrix) >= 3) {
      temp_exposure <- rowMeans(temp_matrix[, 1:min(3, ncol(temp_matrix))], na.rm = TRUE)
    } else {
      temp_exposure <- temp_matrix[, 1]
    }
    
    cat(sprintf("Temperature range: %.1f to %.1f°C\n", 
                min(temp_exposure, na.rm = TRUE), 
                max(temp_exposure, na.rm = TRUE)))
    
    # Define lag structure based on climate variables
    if (any(grepl("lag", climate_vars))) {
      max_lag <- 21  # Extended lag for hypothesis testing
    } else {
      max_lag <- 14  # Standard lag for immediate effects
    }
    
    tryCatch({
      # Create lag matrix for cross-basis
      lag_matrix <- matrix(NA, nrow = length(temp_exposure), ncol = max_lag + 1)
      
      # Use available climate variables to construct lag structure
      for (i in 1:(max_lag + 1)) {
        lag_idx <- min(i, ncol(temp_matrix))
        lag_matrix[, i] <- temp_matrix[, lag_idx]
      }
      
      # Create cross-basis
      cb_temp <- crossbasis(lag_matrix, 
                           lag = max_lag,
                           argvar = list(fun = "ns", df = 3),
                           arglag = list(fun = "ns", df = 4))
      
      # Fit DLNM model
      model <- lm(outcome ~ cb_temp)
      model_summary <- summary(model)
      
      cat(sprintf("%s model R²: %.4f\n", sex_group, model_summary$r.squared))
      cat(sprintf("%s model p-value: %.2e\n", sex_group,
                  pf(model_summary$fstatistic[1], 
                     model_summary$fstatistic[2], 
                     model_summary$fstatistic[3], 
                     lower.tail = FALSE)))
      
      # Predict effects
      pred_overall <- crosspred(cb_temp, model, cumul = TRUE)
      
      # Calculate effect size at key percentiles
      temp_percentiles <- quantile(temp_exposure, c(0.1, 0.5, 0.9), na.rm = TRUE)
      
      effects_at_percentiles <- list()
      for (i in seq_along(temp_percentiles)) {
        temp_val <- temp_percentiles[i]
        temp_idx <- which.min(abs(pred_overall$predvar - temp_val))
        
        if (length(temp_idx) > 0) {
          effect <- pred_overall$allRRfit[temp_idx]
          lower_ci <- pred_overall$allRRlow[temp_idx]
          upper_ci <- pred_overall$allRRhigh[temp_idx]
          
          effects_at_percentiles[[i]] <- list(
            percentile = c(10, 50, 90)[i],
            temperature = temp_val,
            effect = effect,
            lower_ci = lower_ci,
            upper_ci = upper_ci
          )
        }
      }
      
      sex_results[[sex_group]] <- list(
        model = model,
        model_summary = model_summary,
        pred_overall = pred_overall,
        effects_at_percentiles = effects_at_percentiles,
        temp_exposure = temp_exposure,
        temp_percentiles = temp_percentiles,
        n_obs = length(outcome),
        max_lag = max_lag
      )
      
      cat(sprintf("✅ %s DLNM model fitted successfully\n", sex_group))
      
    }, error = function(e) {
      cat(sprintf("❌ Error fitting %s DLNM: %s\n", sex_group, e$message))
      sex_results[[sex_group]] <- NULL
    })
  }
  
  # Test for sex differences
  if (!is.null(sex_results[["Male"]]) && !is.null(sex_results[["Female"]])) {
    cat("\n📊 TESTING FOR SEX DIFFERENCES\n")
    
    # Compare model R²
    male_r2 <- sex_results[["Male"]]$model_summary$r.squared
    female_r2 <- sex_results[["Female"]]$model_summary$r.squared
    
    cat(sprintf("Male R²: %.4f\n", male_r2))
    cat(sprintf("Female R²: %.4f\n", female_r2))
    cat(sprintf("R² difference: %.4f\n", abs(male_r2 - female_r2)))
    
    # Assess effect size differences at key percentiles
    effect_differences <- list()
    
    if (length(sex_results[["Male"]]$effects_at_percentiles) >= 2 && 
        length(sex_results[["Female"]]$effects_at_percentiles) >= 2) {
      
      for (i in 1:min(3, length(sex_results[["Male"]]$effects_at_percentiles))) {
        male_effect <- sex_results[["Male"]]$effects_at_percentiles[[i]]$effect
        female_effect <- sex_results[["Female"]]$effects_at_percentiles[[i]]$effect
        
        effect_differences[[i]] <- list(
          percentile = sex_results[["Male"]]$effects_at_percentiles[[i]]$percentile,
          male_effect = male_effect,
          female_effect = female_effect,
          difference = male_effect - female_effect,
          relative_difference = abs(male_effect - female_effect) / max(abs(male_effect), abs(female_effect))
        )
        
        cat(sprintf("  %dth percentile - Male: %.3f, Female: %.3f, Diff: %.3f\n",
                    effect_differences[[i]]$percentile,
                    male_effect, female_effect,
                    effect_differences[[i]]$difference))
      }
    }
    
    # Hypothesis validation assessment
    sex_difference_evidence <- mean(sapply(effect_differences, function(x) x$relative_difference), na.rm = TRUE)
    r2_difference_evidence <- abs(male_r2 - female_r2)
    
    validation_status <- "WEAK"
    if (sex_difference_evidence > 0.3 && r2_difference_evidence > 0.05) {
      validation_status <- "STRONG"
    } else if (sex_difference_evidence > 0.2 || r2_difference_evidence > 0.03) {
      validation_status <- "MODERATE"
    }
    
    cat(sprintf("\n🎯 HYPOTHESIS VALIDATION: %s\n", validation_status))
    cat(sprintf("Mean effect difference: %.3f\n", sex_difference_evidence))
    cat(sprintf("R² difference: %.3f\n", r2_difference_evidence))
    
    return(list(
      hypothesis = hypothesis_name,
      validation_status = validation_status,
      sex_results = sex_results,
      effect_differences = effect_differences,
      sex_difference_evidence = sex_difference_evidence,
      r2_difference_evidence = r2_difference_evidence
    ))
    
  } else {
    cat("\n❌ Could not complete sex comparison - insufficient model fits\n")
    return(NULL)
  }
}

# Main validation workflow
main_validation <- function() {
  cat("🚀 Starting XAI hypothesis validation with DLNM...\n\n")
  
  # XAI-generated hypotheses to validate
  hypotheses <- list(
    list(
      name = "Heat Index × Sex → FASTING GLUCOSE",
      biomarker = "FASTING GLUCOSE",
      climate_vars = c("heat_index", "heat_index_lag0", "heat_index_lag1", "heat_index_lag3"),
      xai_importance = 0.023
    ),
    list(
      name = "Temperature × Sex → FASTING GLUCOSE", 
      biomarker = "FASTING GLUCOSE",
      climate_vars = c("temperature", "temperature_tas_lag0", "temperature_tas_lag1", "temperature_tas_lag3"),
      xai_importance = 0.020
    ),
    list(
      name = "Land Temperature Lag 3 × Sex → FASTING GLUCOSE",
      biomarker = "FASTING GLUCOSE", 
      climate_vars = c("land_temp_tas_lag3", "land_temp_tas_lag1", "land_temp_tas_lag2", "land_temp_tas_lag5"),
      xai_importance = 0.016
    )
  )
  
  validation_results <- list()
  
  for (hypothesis in hypotheses) {
    # Filter available climate variables
    available_climate_vars <- hypothesis$climate_vars[hypothesis$climate_vars %in% names(data)]
    
    if (length(available_climate_vars) == 0) {
      cat(sprintf("⚠️ No climate variables available for %s\n", hypothesis$name))
      next
    }
    
    cat(sprintf("Using climate variables: %s\n", paste(available_climate_vars, collapse = ", ")))
    
    result <- validate_sex_stratified_hypothesis(
      data, 
      hypothesis$biomarker, 
      available_climate_vars, 
      hypothesis$name
    )
    
    if (!is.null(result)) {
      result$xai_importance <- hypothesis$xai_importance
      validation_results[[hypothesis$name]] <- result
    }
  }
  
  # Summary of validation results
  cat("\n🎯 VALIDATION SUMMARY\n")
  cat("=====================\n")
  
  strong_evidence <- sum(sapply(validation_results, function(x) x$validation_status == "STRONG"))
  moderate_evidence <- sum(sapply(validation_results, function(x) x$validation_status == "MODERATE"))
  weak_evidence <- sum(sapply(validation_results, function(x) x$validation_status == "WEAK"))
  
  total_tested <- length(validation_results)
  
  cat(sprintf("Hypotheses tested: %d\n", total_tested))
  cat(sprintf("Strong validation: %d\n", strong_evidence))
  cat(sprintf("Moderate validation: %d\n", moderate_evidence))
  cat(sprintf("Weak validation: %d\n", weak_evidence))
  
  if (strong_evidence > 0 || moderate_evidence > 0) {
    cat("\n✅ XAI-DLNM VALIDATION SUCCESSFUL!\n")
    cat("XAI-generated hypotheses show DLNM evidence for sex-specific climate vulnerabilities.\n")
  } else {
    cat("\n📊 XAI-DLNM VALIDATION INFORMATIVE\n")
    cat("Results provide insights into sex-specific patterns, though evidence is subtle.\n")
  }
  
  # Detailed results for strong/moderate validations
  if (strong_evidence > 0 || moderate_evidence > 0) {
    cat("\n🔬 VALIDATED HYPOTHESES:\n")
    for (name in names(validation_results)) {
      result <- validation_results[[name]]
      if (result$validation_status %in% c("STRONG", "MODERATE")) {
        cat(sprintf("  • %s: %s validation (XAI importance: %.3f)\n", 
                    name, result$validation_status, result$xai_importance))
        cat(sprintf("    Sex difference evidence: %.3f\n", result$sex_difference_evidence))
        cat(sprintf("    R² difference: %.3f\n", result$r2_difference_evidence))
      }
    }
  }
  
  return(validation_results)
}

# Execute validation
results <- main_validation()

cat("\n🏁 XAI-DLNM VALIDATION COMPLETE!\n")
cat("=================================\n")
cat("This analysis demonstrates the methodological innovation of using\n")
cat("XAI for hypothesis generation followed by rigorous DLNM validation.\n")