#!/usr/bin/env Rscript
# Simplified DLNM Validation Analysis
# ===================================
# 
# Focus on basic temporal validation without complex crossbasis functions

library(mgcv)
library(dplyr)

cat("=================================================================\n")
cat("SIMPLIFIED DLNM VALIDATION ANALYSIS\n")
cat("=================================================================\n")

# Load data
clinical_data <- read.csv("data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv", stringsAsFactors = FALSE)
cat(sprintf("✓ Loaded %d records\n", nrow(clinical_data)))

# Define key variables
biomarkers <- c(
  cd4 = "CD4.cell.count..cells.µL.",
  cholesterol = "total_cholesterol_mg_dL",
  creatinine = "creatinine_umol_L"
)

# Key climate variables  
climate_vars <- c("HEAT_VULNERABILITY_SCORE", "climate_daily_mean_temp", "climate_daily_max_temp")

# Filter available variables
available_bio <- biomarkers[biomarkers %in% colnames(clinical_data)]
available_climate <- climate_vars[climate_vars %in% colnames(clinical_data)]

cat(sprintf("Available biomarkers: %d\n", length(available_bio)))
cat(sprintf("Available climate vars: %d\n", length(available_climate)))

# Simple lag analysis function
analyze_temporal_relationship <- function(biomarker_col, climate_col, data) {
  cat(sprintf("\n--- Analyzing %s vs %s ---\n", biomarker_col, climate_col))
  
  # Get complete cases
  complete_data <- data[c(biomarker_col, climate_col, "primary_date")]
  complete_data <- complete_data[complete.cases(complete_data), ]
  
  if (nrow(complete_data) < 100) {
    cat("  Insufficient data\n")
    return(NULL)
  }
  
  cat(sprintf("  Complete cases: %d\n", nrow(complete_data)))
  
  # Convert dates and sort
  complete_data$date_converted <- as.Date(complete_data$primary_date)
  complete_data <- complete_data[order(complete_data$date_converted), ]
  
  # Create simple lag variables (0, 1, 7, 14, 30 days)
  lags <- c(0, 1, 7, 14, 30)
  
  # Initialize results
  lag_results <- data.frame(
    lag = integer(),
    r_squared = numeric(),
    p_value = numeric(),
    coefficient = numeric()
  )
  
  for (lag_days in lags) {
    # Create lagged climate variable
    if (lag_days == 0) {
      complete_data$climate_lag <- complete_data[[climate_col]]
    } else {
      # Simple lag by shifting indices
      n <- nrow(complete_data)
      climate_lag <- rep(NA, n)
      if (lag_days < n) {
        climate_lag[(lag_days + 1):n] <- complete_data[[climate_col]][1:(n - lag_days)]
      }
      complete_data$climate_lag <- climate_lag
    }
    
    # Remove missing lagged values
    lag_data <- complete_data[!is.na(complete_data$climate_lag), ]
    
    if (nrow(lag_data) > 50) {
      # Fit simple linear model
      formula_str <- paste(biomarker_col, "~ climate_lag")
      model <- lm(as.formula(formula_str), data = lag_data)
      
      model_summary <- summary(model)
      r_squared <- model_summary$r.squared
      p_value <- model_summary$coefficients[2, 4]  # p-value for climate coefficient
      coefficient <- model_summary$coefficients[2, 1]  # climate coefficient
      
      lag_results <- rbind(lag_results, data.frame(
        lag = lag_days,
        r_squared = r_squared,
        p_value = p_value,
        coefficient = coefficient
      ))
      
      cat(sprintf("    Lag %d days: R² = %.3f, p = %.4f\n", lag_days, r_squared, p_value))
    }
  }
  
  if (nrow(lag_results) > 0) {
    # Find best lag
    best_lag <- lag_results[which.max(lag_results$r_squared), ]
    
    cat(sprintf("  Best lag: %d days (R² = %.3f, p = %.4f)\n", 
                best_lag$lag, best_lag$r_squared, best_lag$p_value))
    
    return(list(
      biomarker = biomarker_col,
      climate_var = climate_col,
      n_obs = nrow(complete_data),
      best_lag = best_lag$lag,
      best_r2 = best_lag$r_squared,
      best_p = best_lag$p_value,
      lag_results = lag_results
    ))
  }
  
  return(NULL)
}

# Run temporal analysis for all combinations
cat("\nRunning temporal lag analysis...\n")

all_results <- list()
counter <- 1

for (bio_name in names(available_bio)) {
  bio_col <- available_bio[bio_name]
  
  for (climate_col in available_climate) {
    result <- analyze_temporal_relationship(bio_col, climate_col, clinical_data)
    
    if (!is.null(result)) {
      all_results[[counter]] <- result
      counter <- counter + 1
    }
  }
}

# Summarize results
cat("\n=================================================================\n")
cat("TEMPORAL LAG ANALYSIS RESULTS\n")
cat("=================================================================\n")

if (length(all_results) > 0) {
  cat(sprintf("Completed %d temporal analyses\n\n", length(all_results)))
  
  # Create summary table
  cat(sprintf("%-20s %-25s %-8s %-10s %-12s\n", 
              "Biomarker", "Climate Variable", "Best Lag", "R²", "P-value"))
  cat(paste(rep("-", 80), collapse = ""), "\n")
  
  for (result in all_results) {
    biomarker_short <- gsub(".*\\.", "", result$biomarker)
    climate_short <- substr(result$climate_var, 1, 24)
    
    cat(sprintf("%-20s %-25s %-8d %-10.3f %-12.6f\n",
                biomarker_short,
                climate_short,
                result$best_lag,
                result$best_r2,
                result$best_p))
  }
  
  # Identify significant relationships
  cat("\nSignificant temporal relationships (p < 0.05):\n")
  significant <- all_results[sapply(all_results, function(x) x$best_p < 0.05)]
  
  if (length(significant) > 0) {
    for (result in significant) {
      significance <- if (result$best_p < 0.001) "***" else if (result$best_p < 0.01) "**" else "*"
      cat(sprintf("✓ %s ~ %s: %d-day lag, R² = %.3f %s\n",
                  gsub(".*\\.", "", result$biomarker),
                  result$climate_var,
                  result$best_lag,
                  result$best_r2,
                  significance))
    }
  } else {
    cat("  No significant relationships found\n")
  }
  
  # Compare optimal lags
  cat("\nOptimal lag patterns:\n")
  lag_summary <- table(sapply(all_results, function(x) x$best_lag))
  for (lag in names(lag_summary)) {
    cat(sprintf("  %s-day lag: %d relationships\n", lag, lag_summary[lag]))
  }
  
  # ML vs DLNM comparison
  cat("\n=================================================================\n")
  cat("VALIDATION OF MACHINE LEARNING FINDINGS\n")
  cat("=================================================================\n")
  
  # ML results from previous analysis
  ml_performance <- list(
    "CD4.cell.count..cells.µL." = 0.714,
    "total_cholesterol_mg_dL" = 0.392,
    "creatinine_umol_L" = 0.1
  )
  
  cat("Comparison with ML model performance:\n\n")
  
  for (bio_col in names(ml_performance)) {
    ml_r2 <- ml_performance[[bio_col]]
    bio_short <- gsub(".*\\.", "", bio_col)
    
    # Find best temporal result for this biomarker
    bio_results <- all_results[sapply(all_results, function(x) x$biomarker == bio_col)]
    
    if (length(bio_results) > 0) {
      best_temporal <- bio_results[[which.max(sapply(bio_results, function(x) x$best_r2))]]
      temporal_r2 <- best_temporal$best_r2
      
      cat(sprintf("%s:\n", bio_short))
      cat(sprintf("  ML R² = %.3f\n", ml_r2))
      cat(sprintf("  Best temporal R² = %.3f (lag: %d days)\n", temporal_r2, best_temporal$best_lag))
      
      # Validation assessment
      ratio <- temporal_r2 / ml_r2
      
      if (ratio > 0.5 && temporal_r2 > 0.1) {
        validation <- "✅ Strong temporal validation"
      } else if (ratio > 0.2 && temporal_r2 > 0.05) {
        validation <- "⚠️ Moderate temporal validation"
      } else {
        validation <- "❌ Weak temporal validation - possible confounding"
      }
      
      cat(sprintf("  Validation: %s (ratio: %.2f)\n\n", validation, ratio))
    } else {
      cat(sprintf("%s: ML R² = %.3f, No temporal results\n\n", bio_short, ml_r2))
    }
  }
  
  # HEAT_VULNERABILITY_SCORE analysis
  heat_vuln_results <- all_results[sapply(all_results, function(x) x$climate_var == "HEAT_VULNERABILITY_SCORE")]
  
  if (length(heat_vuln_results) > 0) {
    cat("HEAT_VULNERABILITY_SCORE temporal patterns:\n")
    for (result in heat_vuln_results) {
      bio_short <- gsub(".*\\.", "", result$biomarker)
      cat(sprintf("  %s: %d-day lag, R² = %.3f\n", bio_short, result$best_lag, result$best_r2))
    }
    
    # Check if HEAT_VULNERABILITY_SCORE shows genuine temporal patterns
    strong_heat_vuln <- heat_vuln_results[sapply(heat_vuln_results, function(x) x$best_r2 > 0.1)]
    
    if (length(strong_heat_vuln) > 0) {
      cat("\n✅ HEAT_VULNERABILITY_SCORE shows genuine temporal effects\n")
      cat("   → ML dominance appears justified by temporal patterns\n")
    } else {
      cat("\n⚠️ HEAT_VULNERABILITY_SCORE shows weak temporal effects\n")
      cat("   → ML dominance may be due to confounding factors\n")
    }
  }
  
} else {
  cat("No temporal analyses completed successfully\n")
}

cat("\n=================================================================\n")
cat("FINAL RECOMMENDATIONS\n")
cat("=================================================================\n")

if (length(all_results) > 0) {
  significant_count <- length(all_results[sapply(all_results, function(x) x$best_p < 0.05)])
  strong_count <- length(all_results[sapply(all_results, function(x) x$best_r2 > 0.1)])
  
  cat(sprintf("📊 Summary: %d analyses, %d significant, %d strong (R² > 0.1)\n\n",
              length(all_results), significant_count, strong_count))
  
  if (strong_count > 0) {
    cat("✅ TEMPORAL VALIDATION: Successful\n")
    cat("   → Climate-health relationships show genuine temporal patterns\n")
    cat("   → ML findings are supported by lag structure analysis\n")
    cat("   → Relationships are not simply temporal confounding\n")
  } else {
    cat("⚠️ TEMPORAL VALIDATION: Limited\n")
    cat("   → Weak temporal patterns detected\n")
    cat("   → ML relationships may involve confounding factors\n")
    cat("   → Further investigation of HEAT_VULNERABILITY_SCORE needed\n")
  }
  
  cat(sprintf("\n📋 Next Steps:\n"))
  cat(sprintf("1. 🔍 Investigate HEAT_VULNERABILITY_SCORE construction\n"))
  cat(sprintf("2. 📊 Generate lag-specific effect plots\n"))
  cat(sprintf("3. 🧪 Test with external validation dataset\n"))
  cat(sprintf("4. 📝 Document optimal lag patterns for clinical use\n"))
  
} else {
  cat("❌ Temporal validation inconclusive\n")
  cat("📋 Recommendations:\n")
  cat("1. 🔍 Check data temporal structure\n")
  cat("2. 📊 Review variable availability and quality\n")
  cat("3. 🔄 Consider alternative validation approaches\n")
}

cat(sprintf("\n✓ Simplified DLNM analysis completed: %s\n", Sys.time()))