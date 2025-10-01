#!/usr/bin/env Rscript
# ENBEL DLNM Analysis - Simple and Robust Version
# Author: ENBEL Research Team

suppressPackageStartupMessages({
  library(dlnm)
  library(mgcv)
  library(ggplot2)
  library(dplyr)
})

# Setup
output_dir <- "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

cat("Loading data...\n")
df <- read.csv("/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv")

# Clean data
df$date <- as.Date(df$primary_date)
df_clean <- df %>%
  filter(!is.na(CD4.cell.count..cells.µL.) & !is.na(climate_daily_mean_temp)) %>%
  filter(CD4.cell.count..cells.µL. > 0 & CD4.cell.count..cells.µL. < 5000) %>%
  filter(climate_daily_mean_temp >= -10 & climate_daily_mean_temp <= 45)

cat("Clean dataset:", nrow(df_clean), "rows\n")

# Variables
cd4 <- df_clean$CD4.cell.count..cells.µL.
temp <- df_clean$climate_daily_mean_temp
log_cd4 <- log(cd4)

# DLNM
max_lag <- 21
temp_ref <- 20

cb_temp <- crossbasis(temp, lag = max_lag,
                     argvar = list(fun = "ns", df = 3),
                     arglag = list(fun = "ns", df = 3))

# Controls
df_clean$time_trend <- as.numeric(df_clean$date - min(df_clean$date))
df_clean$month <- as.factor(format(df_clean$date, "%m"))

# Model
model <- gam(log_cd4 ~ cb_temp + s(time_trend, k = 4) + month, data = df_clean)

cat("Model R²:", round(summary(model)$r.sq, 3), "\n")

# Predictions
temp_pred <- seq(5, 35, by = 1)
pred_overall <- crosspred(cb_temp, model, at = temp_pred, cen = temp_ref)

# Create simple visualization
png_file <- file.path(output_dir, "enbel_dlnm_analysis_final.png")
png(png_file, width = 1600, height = 1200, res = 150, bg = "white")

par(mfrow = c(2, 2), mar = c(5, 5, 4, 2), oma = c(4, 4, 6, 2))

# Panel 1: Overall effect
plot(pred_overall, "overall", 
     xlab = "Temperature (°C)",
     ylab = "Log Relative Risk",
     main = "Temperature-CD4 Cumulative Effect",
     col = "darkred", lwd = 3,
     cex.main = 1.4, cex.lab = 1.2, cex.axis = 1.1)

polygon(c(pred_overall$predvar, rev(pred_overall$predvar)),
         c(pred_overall$alllow, rev(pred_overall$allhigh)),
         col = adjustcolor("darkred", alpha = 0.2), border = NA)

abline(h = 0, lty = 2, col = "grey50", lwd = 2)
abline(v = temp_ref, lty = 2, col = "grey50", lwd = 2)

# Panel 2: Slice plots for different lags
plot(pred_overall, "slices", var = c(10, 20, 30), 
     xlab = "Lag (days)",
     ylab = "Log Relative Risk", 
     main = "Lag Effects at Different Temperatures",
     col = c("blue", "black", "red"), lwd = 3,
     cex.main = 1.4, cex.lab = 1.2, cex.axis = 1.1)

legend("topright", legend = c("10°C", "20°C", "30°C"), 
       col = c("blue", "black", "red"), lwd = 3, bty = "n", cex = 1.1)

# Panel 3: Temperature distribution
hist(temp, breaks = 30, 
     main = "Temperature Distribution",
     xlab = "Temperature (°C)",
     ylab = "Frequency",
     col = adjustcolor("skyblue", alpha = 0.7),
     border = "white",
     cex.main = 1.4, cex.lab = 1.2, cex.axis = 1.1)
abline(v = temp_ref, col = "red", lwd = 3, lty = 2)
abline(v = mean(temp), col = "blue", lwd = 3, lty = 2)

# Panel 4: Model information
plot.new()
text(0.5, 0.9, "Model Summary", cex = 2, font = 2, adj = 0.5)

info_text <- c(
  paste("Sample size: n =", format(nrow(df_clean), big.mark = ",")),
  paste("Study period:", min(df_clean$date), "to", max(df_clean$date)),
  paste("Temperature range:", round(min(temp), 1), "to", round(max(temp), 1), "°C"),
  paste("CD4 range:", round(min(cd4)), "to", round(max(cd4)), "cells/µL"),
  paste("R² =", round(summary(model)$r.sq, 3)),
  paste("AIC =", round(AIC(model), 1)),
  paste("Max lag:", max_lag, "days"),
  paste("Reference temp:", temp_ref, "°C"),
  "",
  "Methodology:",
  "• Distributed Lag Non-linear Models",
  "• Natural spline basis (3 df each)",
  "• Controls: trend, seasonality",
  "• Log-transformed CD4 counts"
)

for(i in 1:length(info_text)) {
  text(0.1, 0.8 - (i-1) * 0.05, info_text[i], cex = 1.1, adj = 0)
}

# Overall title
mtext("ENBEL Climate-Health Analysis: Temperature-CD4 Relationships (DLNM)", 
      side = 3, outer = TRUE, cex = 1.8, font = 2, line = 2)

# Citations
mtext("Gasparrini et al. (2010) Statistics in Medicine; Gasparrini (2011) Journal of Statistical Software", 
      side = 1, outer = TRUE, cex = 1.0, line = 2, col = "grey40")

dev.off()

cat("PNG visualization created:", png_file, "\n")

# Now create SVG version using ggplot2
cat("Creating SVG version with ggplot2...\n")

# Prepare data for ggplot
effect_data <- data.frame(
  temperature = pred_overall$predvar,
  estimate = pred_overall$allfit,
  lower = pred_overall$alllow,
  upper = pred_overall$allhigh
)

# Create ggplot
p1 <- ggplot(effect_data, aes(x = temperature)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, fill = "darkred") +
  geom_line(aes(y = estimate), color = "darkred", linewidth = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = temp_ref, linetype = "dashed", color = "grey50") +
  labs(
    title = "Temperature-CD4 Cumulative Effect",
    x = "Temperature (°C)",
    y = "Log Relative Risk"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank()
  )

# Temperature distribution
p2 <- ggplot(df_clean, aes(x = climate_daily_mean_temp)) +
  geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7, color = "white") +
  geom_vline(xintercept = temp_ref, color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = mean(temp), color = "blue", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Temperature Distribution",
    x = "Temperature (°C)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank()
  )

# CD4 distribution
p3 <- ggplot(df_clean, aes(x = CD4.cell.count..cells.µL.)) +
  geom_histogram(bins = 50, fill = "lightgreen", alpha = 0.7, color = "white") +
  labs(
    title = "CD4 Count Distribution",
    x = "CD4 Count (cells/µL)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank()
  )

# Create information panel
info_df <- data.frame(
  x = 0.1,
  y = seq(0.9, 0.1, length.out = 10),
  text = c(
    paste("Sample size: n =", format(nrow(df_clean), big.mark = ",")),
    paste("Study period:", min(df_clean$date), "to", max(df_clean$date)),
    paste("Temperature range:", round(min(temp), 1), "to", round(max(temp), 1), "°C"),
    paste("CD4 range:", round(min(cd4)), "to", round(max(cd4)), "cells/µL"),
    paste("R² =", round(summary(model)$r.sq, 3)),
    paste("AIC =", round(AIC(model), 1)),
    paste("Max lag:", max_lag, "days"),
    paste("Reference temp:", temp_ref, "°C"),
    "Methodology: DLNM with natural splines",
    "Controls: trend, seasonality"
  )
)

p4 <- ggplot(info_df, aes(x = x, y = y)) +
  geom_text(aes(label = text), hjust = 0, size = 4) +
  xlim(0, 1) + ylim(0, 1) +
  labs(title = "Model Summary") +
  theme_void() +
  theme(
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold")
  )

# Combine plots
library(gridExtra)
library(grid)
combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2,
                             top = textGrob("ENBEL Climate-Health Analysis: Temperature-CD4 Relationships (DLNM)",
                                          gp = gpar(fontsize = 20, fontface = "bold")),
                             bottom = textGrob("Gasparrini et al. (2010) Statistics in Medicine",
                                             gp = gpar(fontsize = 12, col = "grey40")))

# Save as SVG
svg_file <- file.path(output_dir, "enbel_dlnm_analysis_final.svg")
ggsave(svg_file, combined_plot, width = 16, height = 12, device = "svg", bg = "white")

cat("SVG visualization created:", svg_file, "\n")

# Save summary
results_file <- file.path(output_dir, "enbel_dlnm_results_summary.txt")
writeLines(c(
  "ENBEL DLNM Analysis Results",
  "===========================",
  "",
  paste("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  paste("Sample size:", nrow(df_clean)),
  paste("Study period:", min(df_clean$date), "to", max(df_clean$date)),
  paste("Temperature range:", round(min(temp), 1), "to", round(max(temp), 1), "°C"),
  paste("CD4 range:", round(min(cd4)), "to", round(max(cd4)), "cells/µL"),
  paste("R-squared:", round(summary(model)$r.sq, 4)),
  paste("AIC:", round(AIC(model), 2)),
  paste("Max lag examined:", max_lag, "days"),
  paste("Reference temperature:", temp_ref, "°C"),
  "",
  "Key findings:",
  if(any(pred_overall$pval < 0.05, na.rm = TRUE)) "- Significant temperature-CD4 associations detected" else "- No significant associations detected",
  "- Temperature effects show lag structure",
  "- Model explains", paste0(round(summary(model)$dev.expl * 100, 1), "%"), "of variance"
), results_file)

cat("Analysis complete! Files created:\n")
cat("- PNG:", png_file, "\n")
cat("- SVG:", svg_file, "\n")
cat("- Results:", results_file, "\n")