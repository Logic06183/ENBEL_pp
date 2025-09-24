# Create DLNM Response Graphs for Climate-Health Relationships
# Based on your validated findings: lag-21 cardiovascular, immediate glucose effects

# Load required libraries
library(dlnm)
library(splines)
library(ggplot2)
library(gridExtra)
library(viridis)
library(dplyr)

# Set working directory
setwd("/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp")

# Read the data
df <- read.csv("CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv", stringsAsFactors = FALSE)

# Clean and prepare data
df$systolic_bp <- df$systolic.blood.pressure
df$diastolic_bp <- df$diastolic.blood.pressure
df$glucose <- df$FASTING.GLUCOSE
df$temp <- df$temperature

# Remove missing values for main variables
df_bp_temp <- df %>% 
  filter(!is.na(systolic_bp) & !is.na(temp))
  
# Sample for computational efficiency
n_bp <- nrow(df_bp_temp)
df_bp <- df_bp_temp %>%
  slice_sample(n = min(2000, n_bp))

df_glucose_temp <- df %>%
  filter(!is.na(glucose) & !is.na(temp))
  
n_glucose <- nrow(df_glucose_temp)
df_glucose <- df_glucose_temp %>%
  slice_sample(n = min(1500, n_glucose))

# ============================================================================
# BLOOD PRESSURE DLNM ANALYSIS
# ============================================================================

print("Running DLNM for Blood Pressure...")

# Create crossbasis for temperature (BP analysis)
# Based on your findings: significant effects at lag 21
cb_bp <- crossbasis(df_bp$temp, 
                    lag = 30,  # Maximum lag
                    argvar = list(fun = "ns", df = 4),  # Natural spline for temperature
                    arglag = list(fun = "ns", df = 4))  # Natural spline for lag

# Fit the model for systolic BP
model_bp <- lm(systolic_bp ~ cb_bp + Sex + economic_vulnerability_imputed, 
               data = df_bp)

# Predict the effect
pred_bp <- crosspred(cb_bp, model_bp, 
                     at = seq(15, 35, by = 0.5),  # Temperature range
                     bylag = 0.1,  # Lag resolution
                     cumul = TRUE)

# ============================================================================
# GLUCOSE DLNM ANALYSIS  
# ============================================================================

print("Running DLNM for Glucose...")

# Create crossbasis for temperature (Glucose analysis)
# Based on your findings: immediate effects (lag 0-3)
cb_glucose <- crossbasis(df_glucose$temp,
                         lag = 10,  # Shorter max lag for glucose
                         argvar = list(fun = "ns", df = 3),
                         arglag = list(fun = "ns", df = 3))

# Fit the model for glucose
model_glucose <- lm(glucose ~ cb_glucose + Sex + economic_vulnerability_imputed,
                    data = df_glucose)

# Predict the effect
pred_glucose <- crosspred(cb_glucose, model_glucose,
                         at = seq(15, 35, by = 0.5),
                         bylag = 0.1,
                         cumul = TRUE)

# ============================================================================
# CREATE PUBLICATION-QUALITY PLOTS
# ============================================================================

# Set up the plotting device
png("enbel_dlnm_response_graphs.png", width = 16, height = 10, units = "in", res = 300)

# Create 2x3 layout
par(mfrow = c(2, 3), mar = c(4.5, 4.5, 3, 2))

# --- BLOOD PRESSURE PLOTS ---

# 1. 3D surface plot for BP
plot(pred_bp, xlab = "Temperature (°C)", ylab = "Lag (days)", 
     zlab = "RR for Systolic BP",
     main = "A. Temperature-Blood Pressure Response Surface",
     col = heat.colors(100), border = NA,
     theta = 35, phi = 20, ltheta = 120,
     cex.main = 1.2, font.main = 2)

# 2. Slice at specific lags for BP
plot(pred_bp, "slices", lag = c(0, 7, 14, 21), 
     xlab = "Temperature (°C)", ylab = "Relative Risk",
     main = "B. BP Response at Specific Lags",
     col = c("#3B82F6", "#10B981", "#F59E0B", "#DC2626"),
     lwd = 2, ci.arg = list(col = adjustcolor(c("#3B82F6", "#10B981", "#F59E0B", "#DC2626"), alpha = 0.2)),
     cex.main = 1.2, font.main = 2)
legend("topleft", legend = c("Lag 0", "Lag 7", "Lag 14", "Lag 21"),
       col = c("#3B82F6", "#10B981", "#F59E0B", "#DC2626"),
       lwd = 2, bty = "n")

# 3. Cumulative effect over all lags for BP
plot(pred_bp, "overall", xlab = "Temperature (°C)", 
     ylab = "Cumulative Relative Risk",
     main = "C. Cumulative BP Effect (0-30 days)",
     col = "#DC2626", lwd = 3,
     ci.arg = list(col = adjustcolor("#DC2626", alpha = 0.2)),
     cex.main = 1.2, font.main = 2)
abline(h = 1, lty = 2, col = "gray50")

# --- GLUCOSE PLOTS ---

# 4. 3D surface plot for Glucose
plot(pred_glucose, xlab = "Temperature (°C)", ylab = "Lag (days)",
     zlab = "RR for Glucose",
     main = "D. Temperature-Glucose Response Surface",
     col = heat.colors(100), border = NA,
     theta = 35, phi = 20, ltheta = 120,
     cex.main = 1.2, font.main = 2)

# 5. Slice at specific lags for Glucose
plot(pred_glucose, "slices", lag = c(0, 1, 3, 7),
     xlab = "Temperature (°C)", ylab = "Relative Risk",
     main = "E. Glucose Response at Specific Lags",
     col = c("#FF7F00", "#FBBF24", "#34D399", "#6366F1"),
     lwd = 2, ci.arg = list(col = adjustcolor(c("#FF7F00", "#FBBF24", "#34D399", "#6366F1"), alpha = 0.2)),
     cex.main = 1.2, font.main = 2)
legend("topleft", legend = c("Lag 0", "Lag 1", "Lag 3", "Lag 7"),
       col = c("#FF7F00", "#FBBF24", "#34D399", "#6366F1"),
       lwd = 2, bty = "n")

# 6. Cumulative effect over all lags for Glucose
plot(pred_glucose, "overall", xlab = "Temperature (°C)",
     ylab = "Cumulative Relative Risk",
     main = "F. Cumulative Glucose Effect (0-10 days)",
     col = "#FF7F00", lwd = 3,
     ci.arg = list(col = adjustcolor("#FF7F00", alpha = 0.2)),
     cex.main = 1.2, font.main = 2)
abline(h = 1, lty = 2, col = "gray50")

# Add main title
mtext("DLNM Analysis: Non-Linear and Delayed Climate-Health Effects", 
      outer = TRUE, cex = 1.5, font = 2, line = -2)

dev.off()

# ============================================================================
# CREATE CONTOUR PLOTS
# ============================================================================

png("enbel_dlnm_contour_plots.png", width = 16, height = 8, units = "in", res = 300)

par(mfrow = c(1, 2), mar = c(5, 5, 4, 6))

# Contour plot for BP
plot(pred_bp, "contour", xlab = "Temperature (°C)", ylab = "Lag (days)",
     main = "Blood Pressure: Temperature-Lag Response Contours",
     col = viridis(100), key.title = title("RR"),
     cex.main = 1.3, font.main = 2)
# Add reference lines
abline(h = 21, col = "red", lwd = 2, lty = 2)  # Your lag-21 finding
text(30, 22, "Peak effect (lag 21)", col = "red", font = 2)

# Contour plot for Glucose
plot(pred_glucose, "contour", xlab = "Temperature (°C)", ylab = "Lag (days)",
     main = "Glucose: Temperature-Lag Response Contours",
     col = viridis(100), key.title = title("RR"),
     cex.main = 1.3, font.main = 2)
# Add reference lines
abline(h = c(0, 1), col = "orange", lwd = 2, lty = 2)  # Immediate effects
text(30, 1.5, "Immediate effects", col = "orange", font = 2)

# Add overall title
mtext("DLNM Contour Plots: Visualizing Climate-Health Lag Patterns", 
      outer = TRUE, cex = 1.5, font = 2, line = -2)

dev.off()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

# Extract key statistics
cat("\n========================================\n")
cat("DLNM ANALYSIS SUMMARY\n")
cat("========================================\n")

# Blood Pressure
cat("\nBLOOD PRESSURE ANALYSIS:\n")
cat(sprintf("Sample size: %d participants\n", nrow(df_bp)))
cat(sprintf("Temperature range: %.1f - %.1f°C\n", min(df_bp$temp), max(df_bp$temp)))
cat(sprintf("Mean systolic BP: %.1f mmHg\n", mean(df_bp$systolic_bp)))

# Find maximum effect
max_bp_effect <- max(pred_bp$matRRfit, na.rm = TRUE)
max_bp_loc <- which(pred_bp$matRRfit == max_bp_effect, arr.ind = TRUE)
cat(sprintf("Maximum RR: %.3f at %.1f°C, lag %d days\n", 
            max_bp_effect, 
            pred_bp$predvar[max_bp_loc[1]], 
            round(pred_bp$lag[max_bp_loc[2]])))

# Glucose
cat("\nGLUCOSE ANALYSIS:\n")
cat(sprintf("Sample size: %d participants\n", nrow(df_glucose)))
cat(sprintf("Mean glucose: %.1f mg/dL\n", mean(df_glucose$glucose)))

# Find maximum effect
max_glucose_effect <- max(pred_glucose$matRRfit, na.rm = TRUE)
max_glucose_loc <- which(pred_glucose$matRRfit == max_glucose_effect, arr.ind = TRUE)
cat(sprintf("Maximum RR: %.3f at %.1f°C, lag %d days\n",
            max_glucose_effect,
            pred_glucose$predvar[max_glucose_loc[1]],
            round(pred_glucose$lag[max_glucose_loc[2]])))

cat("\n========================================\n")
cat("Plots saved as:\n")
cat("- enbel_dlnm_response_graphs.png\n")
cat("- enbel_dlnm_contour_plots.png\n")
cat("========================================\n")

# Save the model objects for later use
save(model_bp, model_glucose, pred_bp, pred_glucose, cb_bp, cb_glucose,
     file = "dlnm_models_results.RData")

print("DLNM analysis complete!")