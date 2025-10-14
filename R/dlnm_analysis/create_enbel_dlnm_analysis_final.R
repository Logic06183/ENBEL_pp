#!/usr/bin/env Rscript
# ENBEL DLNM Analysis - Temperature-CD4 Relationships
# Scientifically Rigorous Implementation
# Author: ENBEL Research Team
# Date: October 2025

# Citation: Gasparrini, A. et al. (2010). Distributed lag non-linear models. Statistics in Medicine, 29(21), 2224-2234.
# Citation: Gasparrini, A. (2011). Distributed lag linear and non-linear models in R: the package dlnm. Journal of Statistical Software, 43(8), 1-20.

# Load required libraries with error handling
required_packages <- c("dlnm", "mgcv", "ggplot2", "dplyr", "lubridate", "viridis", 
                      "gridExtra", "scales", "RColorBrewer", "splines", "lattice")

for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Set scientific options
options(scipen = 999)  # Avoid scientific notation
set.seed(42)  # Reproducibility

# Create output directory if it doesn't exist
output_dir <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final"
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("ENBEL DLNM Analysis Starting...\n")
cat("Output directory:", output_dir, "\n")

# Load and validate data
data_path <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"

if(!file.exists(data_path)) {
  stop("Clinical dataset not found at: ", data_path)
}

cat("Loading clinical dataset...\n")
df <- read.csv(data_path, stringsAsFactors = FALSE)

cat("Dataset dimensions:", nrow(df), "rows,", ncol(df), "columns\n")

# Data preprocessing and validation
cat("Preprocessing data for DLNM analysis...\n")

# Convert date and extract key variables
df$date <- as.Date(df$primary_date)
df$year <- as.numeric(format(df$date, "%Y"))
df$doy <- as.numeric(format(df$date, "%j"))  # Day of year

# Extract key variables with validation
cd4_var <- "CD4 cell count (cells/µL)"
temp_var <- "climate_daily_mean_temp"

# Check if variables exist
if(!cd4_var %in% names(df)) {
  # Try alternative column names
  cd4_candidates <- grep("CD4", names(df), value = TRUE, ignore.case = TRUE)
  if(length(cd4_candidates) > 0) {
    cd4_var <- cd4_candidates[1]
    cat("Using CD4 variable:", cd4_var, "\n")
  } else {
    stop("CD4 variable not found in dataset")
  }
}

if(!temp_var %in% names(df)) {
  temp_candidates <- grep("temp", names(df), value = TRUE, ignore.case = TRUE)
  if(length(temp_candidates) > 0) {
    temp_var <- temp_candidates[1]
    cat("Using temperature variable:", temp_var, "\n")
  } else {
    stop("Temperature variable not found in dataset")
  }
}

# Clean and validate data
df_clean <- df %>%
  filter(!is.na(!!sym(cd4_var)) & !is.na(!!sym(temp_var))) %>%
  filter(!!sym(cd4_var) > 0 & !!sym(cd4_var) < 5000) %>%  # Physiologically reasonable CD4 counts
  filter(!!sym(temp_var) >= -10 & !!sym(temp_var) <= 45) %>%  # Reasonable temperature range for Johannesburg
  arrange(date)

cat("Clean dataset dimensions:", nrow(df_clean), "rows\n")
cat("Date range:", min(df_clean$date), "to", max(df_clean$date), "\n")

if(nrow(df_clean) < 100) {
  stop("Insufficient data for DLNM analysis (n < 100)")
}

# Prepare variables for DLNM
cd4 <- df_clean[[cd4_var]]
temp <- df_clean[[temp_var]]
date_seq <- df_clean$date

# Log-transform CD4 for normality (common in immunology research)
log_cd4 <- log(cd4)

cat("Temperature range:", round(min(temp, na.rm=TRUE), 1), "to", round(max(temp, na.rm=TRUE), 1), "°C\n")
cat("CD4 range:", round(min(cd4, na.rm=TRUE)), "to", round(max(cd4, na.rm=TRUE)), "cells/µL\n")

# Define temperature and lag ranges for Johannesburg climate
temp_range <- c(5, 35)  # Johannesburg typical range
temp_ref <- 20  # Reference temperature (moderate for Johannesburg)
max_lag <- 21  # Up to 3 weeks lag

# Create cross-basis matrix for DLNM
# Temperature dimension: natural spline with 3 df (captures non-linearity)
# Lag dimension: natural spline with 3 df (captures temporal pattern)
cat("Creating cross-basis matrix...\n")

cb_temp <- crossbasis(temp, lag = max_lag,
                     argvar = list(fun = "ns", df = 3),
                     arglag = list(fun = "ns", df = 3))

cat("Cross-basis dimensions:", dim(cb_temp), "\n")

# Fit DLNM model with proper controls
cat("Fitting DLNM model...\n")

# Add temporal controls
df_clean$time_trend <- as.numeric(df_clean$date - min(df_clean$date))
df_clean$month <- as.factor(format(df_clean$date, "%m"))
df_clean$dow <- as.factor(weekdays(df_clean$date))

# Fit GAM model with distributed lag non-linear terms
model <- gam(log_cd4 ~ cb_temp + 
             s(time_trend, k = 4) +  # Long-term trend
             month +  # Seasonal control
             dow,     # Day of week control
             data = df_clean)

cat("Model fitted successfully\n")
cat("Model summary:\n")
print(summary(model))

# Calculate model diagnostics
aic_value <- AIC(model)
r_squared <- summary(model)$r.sq
dev_explained <- summary(model)$dev.expl

cat("Model diagnostics:\n")
cat("AIC:", round(aic_value, 2), "\n")
cat("R-squared:", round(r_squared, 3), "\n")
cat("Deviance explained:", round(dev_explained * 100, 1), "%\n")

# Predict effects
cat("Calculating temperature-lag effects...\n")

# Create prediction grid
temp_pred <- seq(temp_range[1], temp_range[2], by = 0.5)
lag_pred <- 0:max_lag

# Calculate overall cumulative effect
pred_overall <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref)

# Calculate lag-specific effects
pred_lags <- lapply(c(0, 7, 14, 21), function(lag) {
  crosspred(cb_temp, model, at = temp_pred, cen = temp_ref, cumul = lag)
})

names(pred_lags) <- paste0("lag_", c(0, 7, 14, 21))

cat("Predictions calculated\n")

# Start SVG output with high quality settings
svg_file <- file.path(output_dir, "enbel_dlnm_analysis_final.svg")
cat("Creating comprehensive DLNM visualization...\n")
cat("Output file:", svg_file, "\n")

# Try different graphics device
if(capabilities("cairo")) {
  svg(svg_file, width = 16, height = 12, bg = "white")
} else {
  pdf(gsub("\\.svg$", ".pdf", svg_file), width = 16, height = 12, bg = "white")
  cat("Using PDF output instead of SVG\n")
}

# Reset graphics parameters
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2))

# Set up layout for comprehensive figure
layout_matrix <- matrix(c(1, 1, 2, 2,
                         1, 1, 2, 2,
                         3, 3, 4, 4,
                         5, 5, 5, 5), nrow = 4, byrow = TRUE)

layout(layout_matrix, heights = c(1, 1, 1, 0.8))

# Color scheme - scientific and accessible
colors_temp <- viridis(100, option = "plasma")
colors_lag <- RColorBrewer::brewer.pal(8, "Set2")

# Panel 1: 3D Response Surface
par(mar = c(4, 4, 3, 2))

# Create 3D surface data
temp_grid <- seq(temp_range[1], temp_range[2], length = 30)
lag_grid <- 0:max_lag

# Calculate 3D surface
surface_data <- matrix(NA, length(temp_grid), length(lag_grid))
for(i in 1:length(temp_grid)) {
  for(j in 1:length(lag_grid)) {
    pred_point <- crosspred(cb_temp, model, at = temp_grid[i], cen = temp_ref)
    if(lag_grid[j] <= length(pred_point$matfit[1,])) {
      surface_data[i, j] <- pred_point$matfit[1, lag_grid[j] + 1]
    }
  }
}

# Create contour plot (pseudo-3D)
filled.contour(temp_grid, lag_grid, surface_data,
               color.palette = colorRampPalette(c("blue", "white", "red")),
               xlab = "Temperature (°C)",
               ylab = "Lag (days)",
               main = "3D Temperature-Lag-CD4 Response Surface",
               cex.main = 1.2,
               plot.axes = {
                 axis(1, cex.axis = 0.9)
                 axis(2, cex.axis = 0.9)
                 contour(temp_grid, lag_grid, surface_data, add = TRUE, lwd = 0.5)
               })

# Panel 2: Overall cumulative effect
par(mar = c(4, 4, 3, 2))

plot(pred_overall, "overall", 
     xlab = "Temperature (°C)",
     ylab = "Log Relative Risk",
     main = "Cumulative Temperature Effect on CD4 Count",
     col = "darkred", lwd = 2,
     cex.main = 1.2, cex.axis = 0.9, cex.lab = 1.0)

# Add confidence intervals
polygon(c(pred_overall$predvar, rev(pred_overall$predvar)),
         c(pred_overall$alllow, rev(pred_overall$allhigh)),
         col = alpha("darkred", 0.2), border = NA)

# Add reference line
abline(h = 0, lty = 2, col = "grey50")
abline(v = temp_ref, lty = 2, col = "grey50")

# Panel 3: Cross-sectional slices at specific lags
par(mar = c(4, 4, 3, 2))

plot(temp_pred, pred_lags$lag_0$allfit, type = "l", 
     xlab = "Temperature (°C)",
     ylab = "Log Relative Risk",
     main = "Temperature Effects at Specific Lags",
     ylim = range(sapply(pred_lags, function(x) range(c(x$alllow, x$allhigh), na.rm = TRUE)), na.rm = TRUE),
     col = colors_lag[1], lwd = 2,
     cex.main = 1.2, cex.axis = 0.9, cex.lab = 1.0)

# Add confidence intervals for lag 0
polygon(c(temp_pred, rev(temp_pred)),
         c(pred_lags$lag_0$alllow, rev(pred_lags$lag_0$allhigh)),
         col = alpha(colors_lag[1], 0.2), border = NA)

# Add other lags
for(i in 2:length(pred_lags)) {
  lines(temp_pred, pred_lags[[i]]$allfit, col = colors_lag[i], lwd = 2)
  polygon(c(temp_pred, rev(temp_pred)),
           c(pred_lags[[i]]$alllow, rev(pred_lags[[i]]$allhigh)),
           col = alpha(colors_lag[i], 0.15), border = NA)
}

# Add reference lines
abline(h = 0, lty = 2, col = "grey50")
abline(v = temp_ref, lty = 2, col = "grey50")

# Add legend
legend("topright", 
       legend = c("Immediate (lag 0)", "1 week (lag 7)", "2 weeks (lag 14)", "3 weeks (lag 21)"),
       col = colors_lag[1:4], lwd = 2, bty = "n", cex = 0.8)

# Panel 4: Lag-specific effects at reference temperatures
par(mar = c(4, 4, 3, 2))

# Calculate lag effects at specific temperatures
temp_levels <- c(10, 20, 30)  # Cold, moderate, hot
lag_effects <- matrix(NA, max_lag + 1, length(temp_levels))

for(i in 1:length(temp_levels)) {
  pred_lag <- crosspred(cb_temp, model, at = temp_levels[i], cen = temp_ref)
  lag_effects[, i] <- pred_lag$matfit[1, ]
}

matplot(0:max_lag, lag_effects, type = "l", lty = 1, lwd = 2,
        col = c("blue", "black", "red"),
        xlab = "Lag (days)",
        ylab = "Log Relative Risk",
        main = "Lag Structure at Different Temperatures",
        cex.main = 1.2, cex.axis = 0.9, cex.lab = 1.0)

abline(h = 0, lty = 2, col = "grey50")

legend("topright",
       legend = paste0(temp_levels, "°C"),
       col = c("blue", "black", "red"), lwd = 2, bty = "n", cex = 0.8)

# Panel 5: Model diagnostics and study information
par(mar = c(2, 4, 2, 2))

# Create text panel with key information
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1))

# Title
text(0.5, 0.95, "ENBEL Climate-Health Analysis: Temperature-CD4 Relationships", 
     cex = 1.4, font = 2, adj = 0.5)

# Study details
text(0.05, 0.85, "Study Details:", cex = 1.1, font = 2, adj = 0)
text(0.05, 0.80, paste("Sample size: n =", format(nrow(df_clean), big.mark = ",")), cex = 1.0, adj = 0)
text(0.05, 0.75, paste("Study period:", min(df_clean$date), "to", max(df_clean$date)), cex = 1.0, adj = 0)
text(0.05, 0.70, paste("Location: Johannesburg, South Africa"), cex = 1.0, adj = 0)

# Model performance
text(0.35, 0.85, "Model Performance:", cex = 1.1, font = 2, adj = 0)
text(0.35, 0.80, paste("R² =", round(r_squared, 3)), cex = 1.0, adj = 0)
text(0.35, 0.75, paste("AIC =", round(aic_value, 1)), cex = 1.0, adj = 0)
text(0.35, 0.70, paste("Deviance explained:", round(dev_explained * 100, 1), "%"), cex = 1.0, adj = 0)

# Temperature statistics
text(0.65, 0.85, "Temperature Range:", cex = 1.1, font = 2, adj = 0)
text(0.65, 0.80, paste("Min:", round(min(temp, na.rm=TRUE), 1), "°C"), cex = 1.0, adj = 0)
text(0.65, 0.75, paste("Max:", round(max(temp, na.rm=TRUE), 1), "°C"), cex = 1.0, adj = 0)
text(0.65, 0.70, paste("Mean:", round(mean(temp, na.rm=TRUE), 1), "°C"), cex = 1.0, adj = 0)

# Methodology
text(0.05, 0.60, "Methodology:", cex = 1.1, font = 2, adj = 0)
text(0.05, 0.55, "• Distributed Lag Non-linear Models (DLNM)", cex = 0.9, adj = 0)
text(0.05, 0.50, "• Natural spline basis (3 df temperature, 3 df lag)", cex = 0.9, adj = 0)
text(0.05, 0.45, "• GAM with temporal controls (trend, season, day-of-week)", cex = 0.9, adj = 0)
text(0.05, 0.40, "• Log-transformed CD4 counts for normality", cex = 0.9, adj = 0)

# Statistical significance
text(0.05, 0.30, "Key Findings:", cex = 1.1, font = 2, adj = 0)

# Test for overall temperature effect
p_overall <- pred_overall$pval
if(any(p_overall < 0.05, na.rm = TRUE)) {
  text(0.05, 0.25, "• Significant temperature-CD4 associations detected (p < 0.05)", cex = 0.9, adj = 0, col = "darkred")
} else {
  text(0.05, 0.25, "• No significant temperature-CD4 associations detected", cex = 0.9, adj = 0)
}

text(0.05, 0.20, paste("• Maximum lag examined:", max_lag, "days"), cex = 0.9, adj = 0)
text(0.05, 0.15, paste("• Reference temperature:", temp_ref, "°C"), cex = 0.9, adj = 0)

# Citations
text(0.05, 0.05, "References: Gasparrini et al. (2010) Stat Med; Gasparrini (2011) J Stat Softw", 
     cex = 0.8, adj = 0, col = "grey40")

dev.off()

cat("\nDLNM analysis completed successfully!\n")
cat("Output saved to:", svg_file, "\n")

# Save numerical results
results_file <- file.path(output_dir, "enbel_dlnm_results.txt")
cat("\nSaving numerical results...\n")

sink(results_file)
cat("ENBEL DLNM Analysis Results\n")
cat("===========================\n\n")
cat("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("Dataset Information:\n")
cat("- Sample size:", nrow(df_clean), "\n")
cat("- Study period:", as.character(min(df_clean$date)), "to", as.character(max(df_clean$date)), "\n")
cat("- Temperature range:", round(min(temp, na.rm=TRUE), 1), "to", round(max(temp, na.rm=TRUE), 1), "°C\n")
cat("- CD4 range:", round(min(cd4, na.rm=TRUE)), "to", round(max(cd4, na.rm=TRUE)), "cells/µL\n\n")

cat("Model Performance:\n")
cat("- R-squared:", round(r_squared, 4), "\n")
cat("- AIC:", round(aic_value, 2), "\n")
cat("- Deviance explained:", round(dev_explained * 100, 2), "%\n\n")

cat("Model Summary:\n")
print(summary(model))

cat("\n\nOverall Temperature Effect p-values:\n")
print(pred_overall$pval)

cat("\n\nCross-basis Matrix Dimensions:", dim(cb_temp), "\n")

sink()

cat("Numerical results saved to:", results_file, "\n")
cat("\nAnalysis complete! Check output files:\n")
cat("- Visualization:", svg_file, "\n")
cat("- Results:", results_file, "\n")