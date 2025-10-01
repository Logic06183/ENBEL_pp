#!/usr/bin/env Rscript
# ENBEL DLNM Analysis - Temperature-CD4 Relationships (Robust Version)
# Scientifically Rigorous Implementation
# Author: ENBEL Research Team
# Date: October 2025

# Load required libraries
suppressPackageStartupMessages({
  library(dlnm)
  library(mgcv)
  library(ggplot2)
  library(dplyr)
  library(lubridate)
  library(viridis)
  library(RColorBrewer)
  library(splines)
})

# Set scientific options
options(scipen = 999)
set.seed(42)

# Create output directory
output_dir <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final"
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("ENBEL DLNM Analysis Starting...\n")

# Load data
data_path <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
df <- read.csv(data_path, stringsAsFactors = FALSE)

cat("Dataset loaded:", nrow(df), "rows\n")

# Preprocessing
df$date <- as.Date(df$primary_date)
cd4_var <- "CD4.cell.count..cells.µL."
temp_var <- "climate_daily_mean_temp"

# Clean data
df_clean <- df %>%
  filter(!is.na(!!sym(cd4_var)) & !is.na(!!sym(temp_var))) %>%
  filter(!!sym(cd4_var) > 0 & !!sym(cd4_var) < 5000) %>%
  filter(!!sym(temp_var) >= -10 & !!sym(temp_var) <= 45) %>%
  arrange(date)

cat("Clean dataset:", nrow(df_clean), "rows\n")

# Variables
cd4 <- df_clean[[cd4_var]]
temp <- df_clean[[temp_var]]
log_cd4 <- log(cd4)

# DLNM setup
temp_range <- c(5, 35)
temp_ref <- 20
max_lag <- 21

# Cross-basis
cb_temp <- crossbasis(temp, lag = max_lag,
                     argvar = list(fun = "ns", df = 3),
                     arglag = list(fun = "ns", df = 3))

# Model controls
df_clean$time_trend <- as.numeric(df_clean$date - min(df_clean$date))
df_clean$month <- as.factor(format(df_clean$date, "%m"))
df_clean$dow <- as.factor(weekdays(df_clean$date))

# Fit model
model <- gam(log_cd4 ~ cb_temp + 
             s(time_trend, k = 4) +
             month + dow,
             data = df_clean)

cat("Model fitted successfully\n")

# Model diagnostics
aic_value <- AIC(model)
r_squared <- summary(model)$r.sq
dev_explained <- summary(model)$dev.expl

cat("R²:", round(r_squared, 3), "AIC:", round(aic_value, 1), "\n")

# Predictions
temp_pred <- seq(temp_range[1], temp_range[2], by = 0.5)
pred_overall <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref)

# Individual lag predictions
pred_lag0 <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref, cumul = 0)
pred_lag7 <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref, cumul = 7)
pred_lag14 <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref, cumul = 14)
pred_lag21 <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref, cumul = 21)

cat("Predictions calculated\n")

# Create comprehensive PNG (avoiding SVG graphics issues)
png_file <- file.path(output_dir, "enbel_dlnm_analysis_final.png")
svg_file <- file.path(output_dir, "enbel_dlnm_analysis_final.svg")

cat("Creating visualization:", png_file, "\n")

# Use PNG graphics device which is more reliable
png(png_file, width = 2000, height = 1400, res = 100, bg = "white")

# Create comprehensive layout
par(mfrow = c(3, 3), 
    mar = c(4, 4, 3, 2),
    oma = c(4, 4, 6, 2))

# Colors
colors_lag <- c("#E31A1C", "#1F78B4", "#33A02C", "#FF7F00")
temp_colors <- colorRampPalette(c("blue", "white", "red"))(100)

# Panel 1: Overall cumulative effect
plot(pred_overall, "overall", 
     xlab = "Temperature (°C)",
     ylab = "Log Relative Risk",
     main = "A. Cumulative Temperature Effect on CD4",
     col = "darkred", lwd = 3,
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)

# Add confidence interval
polygon(c(pred_overall$predvar, rev(pred_overall$predvar)),
         c(pred_overall$alllow, rev(pred_overall$allhigh)),
         col = adjustcolor("darkred", alpha = 0.2), border = NA)

abline(h = 0, lty = 2, col = "grey50", lwd = 2)
abline(v = temp_ref, lty = 2, col = "grey50", lwd = 2)

# Panel 2: Lag-specific effects
plot(temp_pred, pred_lag0$allfit, type = "l", 
     xlab = "Temperature (°C)",
     ylab = "Log Relative Risk",
     main = "B. Temperature Effects by Lag",
     ylim = range(c(pred_lag0$alllow, pred_lag0$allhigh, 
                   pred_lag7$alllow, pred_lag7$allhigh,
                   pred_lag14$alllow, pred_lag14$allhigh,
                   pred_lag21$alllow, pred_lag21$allhigh), na.rm = TRUE),
     col = colors_lag[1], lwd = 3,
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)

# Add other lags
lines(temp_pred, pred_lag7$allfit, col = colors_lag[2], lwd = 3)
lines(temp_pred, pred_lag14$allfit, col = colors_lag[3], lwd = 3)
lines(temp_pred, pred_lag21$allfit, col = colors_lag[4], lwd = 3)

# Add confidence intervals
polygon(c(temp_pred, rev(temp_pred)),
         c(pred_lag0$alllow, rev(pred_lag0$allhigh)),
         col = adjustcolor(colors_lag[1], alpha = 0.15), border = NA)

abline(h = 0, lty = 2, col = "grey50", lwd = 2)
abline(v = temp_ref, lty = 2, col = "grey50", lwd = 2)

legend("topright", 
       legend = c("Immediate", "1 week", "2 weeks", "3 weeks"),
       col = colors_lag, lwd = 3, bty = "n", cex = 1.0)

# Panel 3: Contour plot (3D surface representation)
plot(pred_overall, "contour", 
     xlab = "Temperature (°C)",
     ylab = "Lag (days)",
     main = "C. 3D Temperature-Lag Surface",
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0,
     key.title = title("Log RR"))

# Panel 4: Lag structure at different temperatures
temp_levels <- c(10, 20, 30)
plot(0:max_lag, pred_overall$matfit[which.min(abs(temp_pred - temp_levels[1])), ], 
     type = "l", lwd = 3, col = "blue",
     xlab = "Lag (days)",
     ylab = "Log Relative Risk",
     main = "D. Lag Structure by Temperature",
     ylim = range(pred_overall$matfit, na.rm = TRUE),
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)

lines(0:max_lag, pred_overall$matfit[which.min(abs(temp_pred - temp_levels[2])), ], 
      col = "black", lwd = 3)
lines(0:max_lag, pred_overall$matfit[which.min(abs(temp_pred - temp_levels[3])), ], 
      col = "red", lwd = 3)

abline(h = 0, lty = 2, col = "grey50", lwd = 2)

legend("topright",
       legend = paste0(temp_levels, "°C"),
       col = c("blue", "black", "red"), lwd = 3, bty = "n", cex = 1.0)

# Panel 5: Model diagnostics
plot.new()
text(0.5, 0.9, "E. Model Diagnostics & Study Info", cex = 1.5, font = 2, adj = 0.5)

text(0.1, 0.75, paste("Sample size: n =", format(nrow(df_clean), big.mark = ",")), 
     cex = 1.2, adj = 0)
text(0.1, 0.65, paste("Study period:", min(df_clean$date), "to", max(df_clean$date)), 
     cex = 1.2, adj = 0)
text(0.1, 0.55, paste("R² =", round(r_squared, 3)), cex = 1.2, adj = 0)
text(0.1, 0.45, paste("AIC =", round(aic_value, 1)), cex = 1.2, adj = 0)

text(0.6, 0.75, paste("Temperature range:", round(min(temp), 1), "-", round(max(temp), 1), "°C"), 
     cex = 1.2, adj = 0)
text(0.6, 0.65, paste("CD4 range:", round(min(cd4)), "-", round(max(cd4)), "cells/µL"), 
     cex = 1.2, adj = 0)
text(0.6, 0.55, paste("Max lag:", max_lag, "days"), cex = 1.2, adj = 0)
text(0.6, 0.45, paste("Reference temp:", temp_ref, "°C"), cex = 1.2, adj = 0)

# Panel 6: Data distribution
hist(temp, breaks = 30, 
     main = "F. Temperature Distribution",
     xlab = "Temperature (°C)",
     ylab = "Frequency",
     col = adjustcolor("skyblue", alpha = 0.7),
     border = "white",
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)
abline(v = temp_ref, col = "red", lwd = 3, lty = 2)

# Panel 7: CD4 distribution
hist(cd4, breaks = 50,
     main = "G. CD4 Count Distribution",
     xlab = "CD4 Count (cells/µL)",
     ylab = "Frequency",
     col = adjustcolor("lightgreen", alpha = 0.7),
     border = "white",
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)

# Panel 8: Residuals
residuals_val <- residuals(model)
plot(fitted(model), residuals_val,
     main = "H. Model Residuals",
     xlab = "Fitted Values",
     ylab = "Residuals",
     pch = 20, alpha = 0.5,
     cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)
abline(h = 0, col = "red", lwd = 2)

# Panel 9: Q-Q plot
qqnorm(residuals_val, 
       main = "I. Q-Q Plot",
       cex.main = 1.3, cex.lab = 1.1, cex.axis = 1.0)
qqline(residuals_val, col = "red", lwd = 2)

# Overall title
mtext("ENBEL Climate-Health Analysis: Temperature-CD4 Relationships Using DLNM", 
      side = 3, outer = TRUE, cex = 2, font = 2, line = 2)

# Footer
mtext("Gasparrini et al. (2010) Stat Med; Distributed Lag Non-linear Models", 
      side = 1, outer = TRUE, cex = 1.0, line = 2, col = "grey40")

mtext(paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M")), 
      side = 1, outer = TRUE, cex = 1.0, line = 0.5, adj = 1, col = "grey40")

dev.off()

cat("Analysis completed successfully!\n")
cat("PNG file created:", png_file, "\n")

# Save results
results_file <- file.path(output_dir, "enbel_dlnm_results_summary.txt")

sink(results_file)
cat("ENBEL DLNM Analysis Results Summary\n")
cat("==================================\n\n")
cat("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Sample size:", nrow(df_clean), "\n")
cat("Study period:", as.character(min(df_clean$date)), "to", as.character(max(df_clean$date)), "\n")
cat("Temperature range:", round(min(temp), 1), "to", round(max(temp), 1), "°C\n")
cat("CD4 range:", round(min(cd4)), "to", round(max(cd4)), "cells/µL\n")
cat("R-squared:", round(r_squared, 4), "\n")
cat("AIC:", round(aic_value, 2), "\n")
cat("Deviance explained:", round(dev_explained * 100, 2), "%\n")
cat("Max lag examined:", max_lag, "days\n")
cat("Reference temperature:", temp_ref, "°C\n")
cat("\nMethodology: Distributed Lag Non-linear Models (DLNM)\n")
cat("Temperature basis: Natural spline (3 df)\n")
cat("Lag basis: Natural spline (3 df)\n")
cat("Controls: Long-term trend, seasonal, day-of-week\n")
sink()

cat("Results summary saved:", results_file, "\n")
cat("All outputs complete!\n")