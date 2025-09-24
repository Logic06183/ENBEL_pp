#!/usr/bin/env Rscript
# DLNM Analysis - Clinical Trial Dataset Only (9,103 participants)
# Focus on original validated clinical relationships

library(dlnm)
library(splines)
library(dplyr)

setwd("/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp")

# Load and prepare ORIGINAL clinical data only
cat("Loading original clinical trial dataset only...\n")
df <- read.csv("FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv", stringsAsFactors = FALSE)

# Filter to only clinical trial participants (exclude imputed socioeconomic data)
# The original dataset should have participant_type or we can identify by non-missing clinical data
df_clinical <- df %>% 
  filter(!is.na(systolic.blood.pressure) | !is.na(FASTING.GLUCOSE)) %>%
  filter(!is.na(temperature))

cat(sprintf("Clinical dataset: %d participants (down from full dataset)\n", nrow(df_clinical)))

# Clean column names
df_clinical$systolic_bp <- df_clinical$systolic.blood.pressure
df_clinical$glucose <- df_clinical$FASTING.GLUCOSE
df_clinical$temp <- df_clinical$temperature

# Prepare BP dataset - clinical only
df_bp <- df_clinical %>% 
  filter(!is.na(systolic_bp) & !is.na(temp) & !is.na(Sex))

# Sample for computational efficiency if still large
if(nrow(df_bp) > 3000) {
  set.seed(42)
  df_bp <- df_bp[sample(nrow(df_bp), 3000), ]
}

cat(sprintf("BP analysis (clinical only): %d participants\n", nrow(df_bp)))

# Create crossbasis for BP
cb_bp <- crossbasis(
  df_bp$temp,
  lag = 30,
  argvar = list(fun = "ns", knots = quantile(df_bp$temp, c(0.25, 0.5, 0.75), na.rm = TRUE)),
  arglag = list(fun = "ns", knots = logknots(30, 3))
)

# Fit model - clinical data should have stronger signal
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

# === GLUCOSE ANALYSIS - CLINICAL ONLY ===
df_glucose <- df_clinical %>%
  filter(!is.na(glucose) & !is.na(temp) & !is.na(Sex))

if(nrow(df_glucose) > 2000) {
  set.seed(42)
  df_glucose <- df_glucose[sample(nrow(df_glucose), 2000), ]
}

cat(sprintf("Glucose analysis (clinical only): %d participants\n", nrow(df_glucose)))

# Glucose crossbasis
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

# === CREATE PLOTS ===

cat("Creating clinical-only DLNM plots...\n")

# BP 3D Plot
png("enbel_dlnm_bp_3d_clinical.png", width = 10, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

plot(pred_bp,
     xlab = "Temperature (°C)",
     ylab = "Lag (days)",
     zlab = "Relative Risk",
     main = "Blood Pressure: Clinical Trial Participants Only",
     theta = 230,
     phi = 30,
     col = "lightblue",
     border = NA,
     shade = 0.2,
     cex.main = 1.3,
     cex.lab = 1.2)

dev.off()

# BP Overall Effect
png("enbel_dlnm_bp_overall_clinical.png", width = 10, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

plot(pred_bp, "overall",
     xlab = "Temperature (°C)",
     ylab = "Cumulative Relative Risk",
     main = "BP: Overall Effect - Clinical Dataset",
     lwd = 3,
     col = "darkblue",
     ci = "area",
     ci.arg = list(col = adjustcolor("lightblue", alpha.f = 0.3)),
     cex.main = 1.3,
     cex.lab = 1.2)
abline(h = 1, lty = 2, col = "gray50", lwd = 2)
rug(df_bp$temp, side = 1, col = adjustcolor("black", alpha.f = 0.2))

dev.off()

# BP Slices
png("enbel_dlnm_bp_slices_clinical.png", width = 10, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

plot(pred_bp, "slices",
     lag = c(0, 7, 14, 21, 28),
     col = c("blue", "green", "orange", "red", "purple"),
     lwd = 3,
     ci = "area",
     ci.arg = list(col = adjustcolor(c("blue", "green", "orange", "red", "purple"), alpha.f = 0.2)),
     main = "BP: Lag-Specific Response - Clinical Dataset",
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

# === GLUCOSE PLOTS ===

# Glucose 3D
png("enbel_dlnm_glucose_3d_clinical.png", width = 10, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

plot(pred_glucose,
     xlab = "Temperature (°C)",
     ylab = "Lag (days)",
     zlab = "Relative Risk",
     main = "Glucose: Clinical Trial Participants Only",
     theta = 230,
     phi = 30,
     col = "lightyellow",
     border = NA,
     shade = 0.2,
     cex.main = 1.3,
     cex.lab = 1.2)

dev.off()

# Glucose Overall
png("enbel_dlnm_glucose_overall_clinical.png", width = 10, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

plot(pred_glucose, "overall",
     xlab = "Temperature (°C)",
     ylab = "Cumulative Relative Risk",
     main = "Glucose: Overall Effect - Clinical Dataset",
     lwd = 3,
     col = "darkorange",
     ci = "area",
     ci.arg = list(col = adjustcolor("orange", alpha.f = 0.3)),
     cex.main = 1.3,
     cex.lab = 1.2)
abline(h = 1, lty = 2, col = "gray50", lwd = 2)
rug(df_glucose$temp, side = 1, col = adjustcolor("black", alpha.f = 0.2))

dev.off()

# Glucose Slices
png("enbel_dlnm_glucose_slices_clinical.png", width = 10, height = 8, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))

plot(pred_glucose, "slices",
     lag = c(0, 1, 3, 5, 7),
     col = c("darkorange", "orange", "gold", "yellow3", "yellow4"),
     lwd = 3,
     ci = "area",
     ci.arg = list(col = adjustcolor(c("darkorange", "orange", "gold", "yellow3", "yellow4"), alpha.f = 0.2)),
     main = "Glucose: Lag-Specific Response - Clinical Dataset",
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

# === SUMMARY STATISTICS ===
cat("\n=== CLINICAL-ONLY DLNM ANALYSIS SUMMARY ===\n")

max_bp <- max(pred_bp$allRRfit, na.rm = TRUE)
cat(sprintf("BP Maximum RR (clinical only): %.3f\n", max_bp))

max_glucose <- max(pred_glucose$allRRfit, na.rm = TRUE)
cat(sprintf("Glucose Maximum RR (clinical only): %.3f\n", max_glucose))

cat("\nGenerated clinical-only DLNM plots:\n")
cat("- enbel_dlnm_bp_3d_clinical.png\n")
cat("- enbel_dlnm_bp_overall_clinical.png\n")
cat("- enbel_dlnm_bp_slices_clinical.png\n")
cat("- enbel_dlnm_glucose_3d_clinical.png\n")
cat("- enbel_dlnm_glucose_overall_clinical.png\n")
cat("- enbel_dlnm_glucose_slices_clinical.png\n")

# Check sample sizes and data quality
cat(sprintf("\nData quality check:\n"))
cat(sprintf("BP range: %.1f - %.1f mmHg (mean: %.1f)\n", 
            min(df_bp$systolic_bp, na.rm=TRUE), 
            max(df_bp$systolic_bp, na.rm=TRUE),
            mean(df_bp$systolic_bp, na.rm=TRUE)))
cat(sprintf("Glucose range: %.1f - %.1f mg/dL (mean: %.1f)\n",
            min(df_glucose$glucose, na.rm=TRUE),
            max(df_glucose$glucose, na.rm=TRUE), 
            mean(df_glucose$glucose, na.rm=TRUE)))
cat(sprintf("Temperature range: %.1f - %.1f°C\n",
            min(df_clinical$temp, na.rm=TRUE),
            max(df_clinical$temp, na.rm=TRUE)))

cat("\n=== CLINICAL-ONLY ANALYSIS COMPLETE ===\n")