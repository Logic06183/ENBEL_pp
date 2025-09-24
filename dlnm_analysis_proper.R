#!/usr/bin/env Rscript
# ============================================================================
# DLNM Analysis for Climate-Health Relationships
# Using proper dlnm package methods and visualizations
# Based on validated findings: lag-21 cardiovascular, immediate glucose effects
# ============================================================================

# Load required libraries
library(dlnm)
library(splines)
library(ggplot2)
library(dplyr)

# Set working directory
setwd("/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp")

# ============================================================================
# DATA PREPARATION
# ============================================================================

cat("Loading and preparing data...\n")

# Read the data
df <- read.csv("CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv", stringsAsFactors = FALSE)

# Clean column names
df$systolic_bp <- df$systolic.blood.pressure
df$diastolic_bp <- df$diastolic.blood.pressure
df$glucose <- df$FASTING.GLUCOSE
df$temp <- df$temperature

# Create datasets for each analysis
df_bp <- df %>% 
  filter(!is.na(systolic_bp) & !is.na(temp) & !is.na(Sex))

# Sample for computational efficiency if needed
if(nrow(df_bp) > 2000) {
  set.seed(42)
  df_bp <- df_bp[sample(nrow(df_bp), 2000), ]
}

df_glucose <- df %>%
  filter(!is.na(glucose) & !is.na(temp) & !is.na(Sex))

if(nrow(df_glucose) > 1500) {
  set.seed(42)
  df_glucose <- df_glucose[sample(nrow(df_glucose), 1500), ]
}

cat(sprintf("Blood pressure analysis: %d participants\n", nrow(df_bp)))
cat(sprintf("Glucose analysis: %d participants\n", nrow(df_glucose)))

# ============================================================================
# BLOOD PRESSURE DLNM ANALYSIS
# ============================================================================

cat("\n=== Blood Pressure DLNM Analysis ===\n")

# Define the cross-basis for temperature and lag
# Use natural splines for both dimensions
cb_bp <- crossbasis(
  df_bp$temp,
  lag = 30,  # Maximum lag of 30 days
  argvar = list(fun = "ns", knots = quantile(df_bp$temp, c(0.25, 0.50, 0.75), na.rm = TRUE)),
  arglag = list(fun = "ns", knots = logknots(30, 3))  # 3 knots in log scale
)

# Fit the model
model_bp <- glm(systolic_bp ~ cb_bp + Sex + economic_vulnerability_imputed,
                data = df_bp,
                family = gaussian())

# Get predictions
pred_bp <- crosspred(
  cb_bp, 
  model_bp,
  at = seq(min(df_bp$temp, na.rm = TRUE), max(df_bp$temp, na.rm = TRUE), length.out = 50),
  bylag = 0.2,
  cumul = TRUE,
  cen = median(df_bp$temp, na.rm = TRUE)  # Center at median temperature
)

cat(sprintf("BP model fitted. Temperature range: %.1f to %.1f°C\n", 
            min(df_bp$temp, na.rm = TRUE), max(df_bp$temp, na.rm = TRUE)))

# ============================================================================
# GLUCOSE DLNM ANALYSIS
# ============================================================================

cat("\n=== Glucose DLNM Analysis ===\n")

# Cross-basis for glucose (shorter lag based on findings)
cb_glucose <- crossbasis(
  df_glucose$temp,
  lag = 10,  # Maximum lag of 10 days for glucose
  argvar = list(fun = "ns", knots = quantile(df_glucose$temp, c(0.25, 0.50, 0.75), na.rm = TRUE)),
  arglag = list(fun = "ns", knots = logknots(10, 2))  # 2 knots
)

# Fit the model
model_glucose <- glm(glucose ~ cb_glucose + Sex + economic_vulnerability_imputed,
                     data = df_glucose,
                     family = gaussian())

# Get predictions
pred_glucose <- crosspred(
  cb_glucose,
  model_glucose,
  at = seq(min(df_glucose$temp, na.rm = TRUE), max(df_glucose$temp, na.rm = TRUE), length.out = 50),
  bylag = 0.1,
  cumul = TRUE,
  cen = median(df_glucose$temp, na.rm = TRUE)
)

cat(sprintf("Glucose model fitted. Temperature range: %.1f to %.1f°C\n",
            min(df_glucose$temp, na.rm = TRUE), max(df_glucose$temp, na.rm = TRUE)))

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

cat("\n=== Creating DLNM Visualizations ===\n")

# Set up high-resolution plotting
png("enbel_dlnm_standard_plots.png", width = 16, height = 12, units = "in", res = 300)

# Reset plotting parameters
par(mfrow = c(1, 1))

# Create individual plots to avoid margin issues

# --- BLOOD PRESSURE PLOTS ---

# 1. 3D plot
plot(pred_bp, 
     xlab = "Temperature (°C)", 
     ylab = "Lag (days)", 
     zlab = "RR",
     main = "A. BP: 3D Exposure-Lag-Response",
     theta = 230,  # Rotation angle
     phi = 30,     # Viewing angle
     lphi = 30,
     border = NA,
     shade = 0.2,
     col = "lightblue",
     cex.main = 1.2)

# 2. Contour plot
filled.contour(
  x = pred_bp$predvar,
  y = 0:30,  # Lag days
  z = pred_bp$matRRfit,
  xlab = "Temperature (°C)",
  ylab = "Lag (days)",
  main = "B. BP: Temperature-Lag Contour",
  color.palette = heat.colors,
  plot.axes = {
    axis(1)
    axis(2)
    contour(pred_bp$predvar, 0:30, pred_bp$matRRfit, 
            add = TRUE, col = "black", lwd = 0.5)
    abline(h = 21, col = "red", lwd = 2, lty = 2)  # Mark lag 21
    text(max(pred_bp$predvar) - 5, 23, "Lag 21", col = "red", font = 2)
  }
)

# 3. Slices at specific lags
plot(pred_bp, "slices", 
     var = median(df_bp$temp, na.rm = TRUE),  # At median temperature
     lag = c(0, 7, 14, 21, 28),
     col = c("blue", "green", "orange", "red", "purple"),
     lwd = 2,
     ci = "area",
     ci.arg = list(col = adjustcolor(c("blue", "green", "orange", "red", "purple"), 
                                     alpha.f = 0.2)),
     main = "C. BP: Response at Specific Lags",
     xlab = "Temperature (°C)",
     ylab = "Relative Risk",
     ylim = c(0.8, 1.3))
legend("topleft", 
       legend = paste("Lag", c(0, 7, 14, 21, 28)),
       col = c("blue", "green", "orange", "red", "purple"),
       lwd = 2, bty = "n", cex = 0.9)
abline(h = 1, lty = 2, col = "gray50")

# --- GLUCOSE PLOTS ---

# 4. 3D plot for Glucose
plot(pred_glucose,
     xlab = "Temperature (°C)",
     ylab = "Lag (days)",
     zlab = "RR",
     main = "D. Glucose: 3D Exposure-Lag-Response",
     theta = 230,
     phi = 30,
     lphi = 30,
     border = NA,
     shade = 0.2,
     col = "lightyellow",
     cex.main = 1.2)

# 5. Contour plot for Glucose
filled.contour(
  x = pred_glucose$predvar,
  y = 0:10,  # Lag days
  z = pred_glucose$matRRfit,
  xlab = "Temperature (°C)",
  ylab = "Lag (days)",
  main = "E. Glucose: Temperature-Lag Contour",
  color.palette = terrain.colors,
  plot.axes = {
    axis(1)
    axis(2)
    contour(pred_glucose$predvar, 0:10, pred_glucose$matRRfit,
            add = TRUE, col = "black", lwd = 0.5)
    abline(h = c(0, 1, 2), col = "orange", lwd = 1.5, lty = 2)  # Mark immediate lags
    text(max(pred_glucose$predvar) - 5, 1, "Immediate", col = "orange", font = 2)
  }
)

# 6. Slices at specific lags for Glucose
plot(pred_glucose, "slices",
     var = median(df_glucose$temp, na.rm = TRUE),
     lag = c(0, 1, 3, 5, 7),
     col = c("darkorange", "orange", "gold", "yellow3", "yellow4"),
     lwd = 2,
     ci = "area",
     ci.arg = list(col = adjustcolor(c("darkorange", "orange", "gold", "yellow3", "yellow4"),
                                     alpha.f = 0.2)),
     main = "F. Glucose: Response at Specific Lags",
     xlab = "Temperature (°C)",
     ylab = "Relative Risk",
     ylim = c(0.8, 1.3))
legend("topleft",
       legend = paste("Lag", c(0, 1, 3, 5, 7)),
       col = c("darkorange", "orange", "gold", "yellow3", "yellow4"),
       lwd = 2, bty = "n", cex = 0.9)
abline(h = 1, lty = 2, col = "gray50")

# Add overall title
mtext("DLNM Analysis: Non-Linear and Delayed Climate-Health Effects",
      outer = TRUE, cex = 1.4, font = 2)

dev.off()

cat("Standard DLNM plots saved as 'enbel_dlnm_standard_plots.png'\n")

# ============================================================================
# CREATE OVERALL CUMULATIVE PLOTS
# ============================================================================

png("enbel_dlnm_overall_effects.png", width = 14, height = 7, units = "in", res = 300)

par(mfrow = c(1, 2), mar = c(5, 4, 4, 2))

# Overall cumulative effect for BP
plot(pred_bp, "overall",
     xlab = "Temperature (°C)",
     ylab = "Cumulative RR (0-30 days)",
     main = "Blood Pressure: Overall Cumulative Association",
     lwd = 3,
     col = "darkblue",
     ci = "area",
     ci.arg = list(col = adjustcolor("lightblue", alpha.f = 0.5)),
     ylim = c(0.5, 2))
abline(h = 1, lty = 2, col = "gray50")
rug(df_bp$temp, side = 1, col = adjustcolor("black", alpha.f = 0.2))

# Overall cumulative effect for Glucose
plot(pred_glucose, "overall",
     xlab = "Temperature (°C)",
     ylab = "Cumulative RR (0-10 days)",
     main = "Glucose: Overall Cumulative Association",
     lwd = 3,
     col = "darkorange",
     ci = "area",
     ci.arg = list(col = adjustcolor("orange", alpha.f = 0.3)),
     ylim = c(0.5, 2))
abline(h = 1, lty = 2, col = "gray50")
rug(df_glucose$temp, side = 1, col = adjustcolor("black", alpha.f = 0.2))

dev.off()

cat("Overall cumulative plots saved as 'enbel_dlnm_overall_effects.png'\n")

# ============================================================================
# CREATE LAG-SPECIFIC PLOTS
# ============================================================================

png("enbel_dlnm_lag_specific.png", width = 14, height = 7, units = "in", res = 300)

par(mfrow = c(1, 2), mar = c(5, 4, 4, 2))

# Lag-response at specific temperature for BP
plot(pred_bp, "slices",
     var = c(15, 20, 25, 30, 35),  # Different temperatures
     lag = c(0:30),
     col = heat.colors(5),
     lwd = 2,
     main = "BP: Lag-Response at Different Temperatures",
     xlab = "Lag (days)",
     ylab = "Relative Risk")
legend("topright",
       legend = paste(c(15, 20, 25, 30, 35), "°C"),
       col = heat.colors(5),
       lwd = 2, bty = "n", cex = 0.9)
abline(h = 1, lty = 2, col = "gray50")
abline(v = 21, lty = 2, col = "red", lwd = 1.5)
text(22, max(par("usr")[4]) * 0.95, "Lag 21", col = "red", adj = 0, font = 2)

# Lag-response at specific temperature for Glucose
plot(pred_glucose, "slices",
     var = c(15, 20, 25, 30, 35),
     lag = c(0:10),
     col = terrain.colors(5),
     lwd = 2,
     main = "Glucose: Lag-Response at Different Temperatures",
     xlab = "Lag (days)",
     ylab = "Relative Risk")
legend("topright",
       legend = paste(c(15, 20, 25, 30, 35), "°C"),
       col = terrain.colors(5),
       lwd = 2, bty = "n", cex = 0.9)
abline(h = 1, lty = 2, col = "gray50")

dev.off()

cat("Lag-specific plots saved as 'enbel_dlnm_lag_specific.png'\n")

# ============================================================================
# EXTRACT KEY STATISTICS
# ============================================================================

cat("\n=== Key Statistics ===\n")

# Find maximum effect for BP
max_bp_rr <- max(pred_bp$matRRfit, na.rm = TRUE)
max_bp_loc <- which(pred_bp$matRRfit == max_bp_rr, arr.ind = TRUE)
cat(sprintf("BP Maximum RR: %.3f at %.1f°C, lag %d days\n",
            max_bp_rr,
            pred_bp$predvar[max_bp_loc[1]],
            max_bp_loc[2] - 1))

# Find minimum effect for BP
min_bp_rr <- min(pred_bp$matRRfit, na.rm = TRUE)
min_bp_loc <- which(pred_bp$matRRfit == min_bp_rr, arr.ind = TRUE)
cat(sprintf("BP Minimum RR: %.3f at %.1f°C, lag %d days\n",
            min_bp_rr,
            pred_bp$predvar[min_bp_loc[1]],
            min_bp_loc[2] - 1))

# Find maximum effect for Glucose
max_glu_rr <- max(pred_glucose$matRRfit, na.rm = TRUE)
max_glu_loc <- which(pred_glucose$matRRfit == max_glu_rr, arr.ind = TRUE)
cat(sprintf("Glucose Maximum RR: %.3f at %.1f°C, lag %d days\n",
            max_glu_rr,
            pred_glucose$predvar[max_glu_loc[1]],
            max_glu_loc[2] - 1))

cat("\n=== DLNM Analysis Complete ===\n")
cat("Generated files:\n")
cat("- enbel_dlnm_standard_plots.png (main 3D and contour plots)\n")
cat("- enbel_dlnm_overall_effects.png (cumulative effects)\n")
cat("- enbel_dlnm_lag_specific.png (lag-response curves)\n")