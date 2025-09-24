#!/usr/bin/env Rscript
# Simplified DLNM Analysis - Create Individual Plots
# Using proper dlnm package methods

library(dlnm)
library(splines)
library(dplyr)

setwd("/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp")

# Load and prepare data
cat("Loading data...\n")
df <- read.csv("CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv", stringsAsFactors = FALSE)

df$systolic_bp <- df$systolic.blood.pressure
df$glucose <- df$FASTING.GLUCOSE
df$temp <- df$temperature

# Prepare BP dataset
df_bp <- df %>% 
  filter(!is.na(systolic_bp) & !is.na(temp) & !is.na(Sex))

# Sample if too large
if(nrow(df_bp) > 2000) {
  set.seed(42)
  df_bp <- df_bp[sample(nrow(df_bp), 2000), ]
}

cat(sprintf("BP analysis: %d participants\n", nrow(df_bp)))

# Create crossbasis
cb_bp <- crossbasis(
  df_bp$temp,
  lag = 30,
  argvar = list(fun = "ns", knots = quantile(df_bp$temp, c(0.25, 0.5, 0.75), na.rm = TRUE)),
  arglag = list(fun = "ns", knots = logknots(30, 3))
)

# Fit model
model_bp <- glm(systolic_bp ~ cb_bp + Sex, data = df_bp, family = gaussian())

# Get predictions
pred_bp <- crosspred(
  cb_bp, 
  model_bp,
  at = seq(10, 40, length.out = 20),
  bylag = 1,
  cumul = TRUE,
  cen = 25
)

cat("Creating BP 3D plot...\n")

# 3D Plot for Blood Pressure
svg("enbel_dlnm_bp_3d.svg", width = 10, height = 8)
par(mar = c(5, 5, 4, 2))

plot(pred_bp,
     xlab = "Temperature (°C)",
     ylab = "Lag (days)",
     zlab = "Relative Risk",
     main = "Blood Pressure: Temperature-Lag Response Surface",
     theta = 230,
     phi = 30,
     col = "lightblue",
     border = NA,
     shade = 0.2,
     cex.main = 1.3,
     cex.lab = 1.2)

dev.off()

cat("Creating BP overall effect plot...\n")

# Overall Effect Plot
svg("enbel_dlnm_bp_overall.svg", width = 10, height = 8)
par(mar = c(5, 5, 4, 2))

plot(pred_bp, "overall",
     xlab = "Temperature (°C)",
     ylab = "Cumulative Relative Risk",
     main = "Blood Pressure: Overall Cumulative Effect (0-30 days)",
     lwd = 3,
     col = "darkblue",
     ci = "area",
     ci.arg = list(col = adjustcolor("lightblue", alpha.f = 0.3)),
     cex.main = 1.3,
     cex.lab = 1.2)
abline(h = 1, lty = 2, col = "gray50", lwd = 2)
rug(df_bp$temp, side = 1, col = adjustcolor("black", alpha.f = 0.2))

dev.off()

cat("Creating BP slice plot...\n")

# Slice Plot at specific lags
svg("enbel_dlnm_bp_slices.svg", width = 10, height = 8)
par(mar = c(5, 5, 4, 2))

plot(pred_bp, "slices",
     lag = c(0, 7, 14, 21, 28),  # Only specify lags
     col = c("blue", "green", "orange", "red", "purple"),
     lwd = 3,
     ci = "area",
     ci.arg = list(col = adjustcolor(c("blue", "green", "orange", "red", "purple"), alpha.f = 0.2)),
     main = "Blood Pressure: Response at Specific Lags",
     xlab = "Temperature (°C)",
     ylab = "Relative Risk",
     cex.main = 1.3,
     cex.lab = 1.2)

legend("topleft",
       legend = paste("Lag", c(0, 7, 14, 21, 28), "days"),
       col = c("blue", "green", "orange", "red", "purple"),
       lwd = 3, bty = "n", cex = 1.1)
abline(h = 1, lty = 2, col = "gray50", lwd = 2)

dev.off()

# Now do similar for Glucose
df_glucose <- df %>%
  filter(!is.na(glucose) & !is.na(temp) & !is.na(Sex))

if(nrow(df_glucose) > 1500) {
  set.seed(42)
  df_glucose <- df_glucose[sample(nrow(df_glucose), 1500), ]
}

cat(sprintf("Glucose analysis: %d participants\n", nrow(df_glucose)))

# Glucose crossbasis (shorter lag)
cb_glucose <- crossbasis(
  df_glucose$temp,
  lag = 10,
  argvar = list(fun = "ns", knots = quantile(df_glucose$temp, c(0.25, 0.5, 0.75), na.rm = TRUE)),
  arglag = list(fun = "ns", knots = logknots(10, 2))
)

model_glucose <- glm(glucose ~ cb_glucose + Sex, data = df_glucose, family = gaussian())

pred_glucose <- crosspred(
  cb_glucose,
  model_glucose,
  at = seq(10, 40, length.out = 20),
  bylag = 0.5,
  cumul = TRUE,
  cen = 25
)

cat("Creating Glucose 3D plot...\n")

# Glucose 3D Plot
svg("enbel_dlnm_glucose_3d.svg", width = 10, height = 8)
par(mar = c(5, 5, 4, 2))

plot(pred_glucose,
     xlab = "Temperature (°C)",
     ylab = "Lag (days)",
     zlab = "Relative Risk",
     main = "Glucose: Temperature-Lag Response Surface",
     theta = 230,
     phi = 30,
     col = "lightyellow",
     border = NA,
     shade = 0.2,
     cex.main = 1.3,
     cex.lab = 1.2)

dev.off()

cat("Creating Glucose overall effect plot...\n")

# Glucose Overall Effect
svg("enbel_dlnm_glucose_overall.svg", width = 10, height = 8)
par(mar = c(5, 5, 4, 2))

plot(pred_glucose, "overall",
     xlab = "Temperature (°C)",
     ylab = "Cumulative Relative Risk",
     main = "Glucose: Overall Cumulative Effect (0-10 days)",
     lwd = 3,
     col = "darkorange",
     ci = "area",
     ci.arg = list(col = adjustcolor("orange", alpha.f = 0.3)),
     cex.main = 1.3,
     cex.lab = 1.2)
abline(h = 1, lty = 2, col = "gray50", lwd = 2)
rug(df_glucose$temp, side = 1, col = adjustcolor("black", alpha.f = 0.2))

dev.off()

cat("Creating Glucose slice plot...\n")

# Glucose Slices
svg("enbel_dlnm_glucose_slices.svg", width = 10, height = 8)
par(mar = c(5, 5, 4, 2))

plot(pred_glucose, "slices",
     lag = c(0, 1, 3, 5, 7),
     col = c("darkorange", "orange", "gold", "yellow3", "yellow4"),
     lwd = 3,
     ci = "area",
     ci.arg = list(col = adjustcolor(c("darkorange", "orange", "gold", "yellow3", "yellow4"), alpha.f = 0.2)),
     main = "Glucose: Response at Specific Lags",
     xlab = "Temperature (°C)",
     ylab = "Relative Risk",
     cex.main = 1.3,
     cex.lab = 1.2)

legend("topleft",
       legend = paste("Lag", c(0, 1, 3, 5, 7), "days"),
       col = c("darkorange", "orange", "gold", "yellow3", "yellow4"),
       lwd = 3, bty = "n", cex = 1.1)
abline(h = 1, lty = 2, col = "gray50", lwd = 2)

dev.off()

# Print summary statistics
cat("\n=== DLNM Analysis Summary ===\n")

max_bp <- max(pred_bp$allRRfit, na.rm = TRUE)
cat(sprintf("BP Maximum RR: %.3f\n", max_bp))

max_glucose <- max(pred_glucose$allRRfit, na.rm = TRUE)
cat(sprintf("Glucose Maximum RR: %.3f\n", max_glucose))

cat("\nGenerated DLNM plots:\n")
cat("- enbel_dlnm_bp_3d.svg\n")
cat("- enbel_dlnm_bp_overall.svg\n")
cat("- enbel_dlnm_bp_slices.svg\n")
cat("- enbel_dlnm_glucose_3d.svg\n")
cat("- enbel_dlnm_glucose_overall.svg\n")
cat("- enbel_dlnm_glucose_slices.svg\n")

cat("\n=== Analysis Complete ===\n")