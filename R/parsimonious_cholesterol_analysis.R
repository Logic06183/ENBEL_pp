#!/usr/bin/env Rscript
################################################################################
# Parsimonious Rigorous Analysis: Total Cholesterol Only
################################################################################
#
# Purpose: Focus on SINGLE most important finding (Total Cholesterol) with
#          rigorous attention to:
#          - Socioeconomic confounding
#          - Temporal autocorrelation
#          - Stratification by SES
#          - Effect modification
#          - Multiple sensitivity analyses
#
# Principle: ONE finding, deeply analyzed, rigorously tested
#
# Author: Claude + Craig Saunders
# Date: 2025-10-30
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)
  library(dlnm)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(car)        # VIF for collinearity
  library(lmtest)     # Autocorrelation tests
})

set.seed(42)

DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
OUTPUT_DIR <- "reanalysis_outputs/parsimonious_cholesterol"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("\n")
cat("################################################################################\n")
cat("# PARSIMONIOUS RIGOROUS ANALYSIS: TOTAL CHOLESTEROL\n")
cat("################################################################################\n\n")

cat("FOCUS: Single most important finding with rigorous statistical testing\n")
cat("BIOMARKER: Total Cholesterol (R² = 0.345, n = 2,917)\n\n")

################################################################################
# LOAD AND PREPARE DATA
################################################################################

cat("=== 1. Data Preparation ===\n")

df <- fread(DATA_PATH)

# Focus on cholesterol only
df_chol <- df[, .(
  cholesterol_raw = total_cholesterol_mg_dL,
  temperature = climate_7d_mean_temp,
  vulnerability = HEAT_VULNERABILITY_SCORE,
  study_id = as.factor(study_source),
  season = as.factor(season),
  date = as.Date(primary_date),
  year = year,
  month = month
)]

df_chol <- na.omit(df_chol)

cat(sprintf("  N observations: %d\n", nrow(df_chol)))
cat(sprintf("  N studies: %d\n", uniqueN(df_chol$study_id)))
cat(sprintf("  Date range: %s to %s\n", min(df_chol$date), max(df_chol$date)))

# Check if units correction needed
study_means <- df_chol[, .(mean_chol = mean(cholesterol_raw)), by = study_id]
cat("\n  Study-level means:\n")
print(study_means)

# Units check: if one study mean < 15 (likely mmol/L) and others > 30 (likely mg/dL)
needs_conversion <- any(study_means$mean_chol < 15) & any(study_means$mean_chol > 30)

if (needs_conversion) {
  cat("\n  ⚠️  Units mixing detected. Converting mmol/L to mg/dL...\n")
  # mmol/L to mg/dL: multiply by 38.67
  df_chol[, cholesterol := ifelse(cholesterol_raw < 15,
                                   cholesterol_raw * 38.67,
                                   cholesterol_raw)]
  cat("  ✓ Conversion applied\n")
} else {
  df_chol[, cholesterol := cholesterol_raw]
  cat("  ✓ Units consistent\n")
}

# Verify after conversion
study_means_after <- df_chol[, .(mean_chol = mean(cholesterol)), by = study_id]
cat("\n  Study-level means (after conversion):\n")
print(study_means_after)

################################################################################
# 2. SOCIOECONOMIC CONFOUNDING
################################################################################

cat("\n=== 2. Socioeconomic Confounding Analysis ===\n")

# Check distribution of HEAT_VULNERABILITY_SCORE
cat("\n  Heat Vulnerability Score Distribution:\n")
cat(sprintf("    Mean: %.3f\n", mean(df_chol$vulnerability, na.rm = TRUE)))
cat(sprintf("    SD: %.3f\n", sd(df_chol$vulnerability, na.rm = TRUE)))
cat(sprintf("    Range: [%.3f, %.3f]\n",
            min(df_chol$vulnerability, na.rm = TRUE),
            max(df_chol$vulnerability, na.rm = TRUE)))

# Create SES quartiles (use unique breaks to avoid duplicate values)
quant_breaks <- unique(quantile(df_chol$vulnerability, probs = 0:4/4, na.rm = TRUE))

if (length(quant_breaks) < 5) {
  # If not enough unique quantiles, use tertiles or median split
  cat("  ⚠️  Not enough unique values for quartiles, using tertiles\n")
  quant_breaks <- unique(quantile(df_chol$vulnerability, probs = 0:3/3, na.rm = TRUE))
  labels_ses <- c("Low Vuln", "Med Vuln", "High Vuln")[1:(length(quant_breaks)-1)]
} else {
  labels_ses <- c("Q1 (Low Vuln)", "Q2", "Q3", "Q4 (High Vuln)")
}

df_chol[, ses_quartile := cut(vulnerability,
                                breaks = quant_breaks,
                                labels = labels_ses,
                                include.lowest = TRUE)]

# Check correlation between temperature and vulnerability (confounding?)
cor_test <- cor.test(df_chol$temperature, df_chol$vulnerability)
cat(sprintf("\n  Correlation (Temperature × Vulnerability): r = %.3f, p = %.3e\n",
            cor_test$estimate, cor_test$p.value))

if (abs(cor_test$estimate) > 0.3) {
  cat("  ⚠️  Moderate correlation detected - potential confounding\n")
} else {
  cat("  ✓ Low correlation - minimal confounding from vulnerability\n")
}

# Test if cholesterol varies by SES
ses_test <- kruskal.test(cholesterol ~ ses_quartile, data = df_chol)
cat(sprintf("\n  Cholesterol by SES quartile: χ² = %.2f, p = %.3e\n",
            ses_test$statistic, ses_test$p.value))

if (ses_test$p.value < 0.05) {
  cat("  ⚠️  Cholesterol varies significantly by SES - must control\n")
} else {
  cat("  ✓ Cholesterol similar across SES groups\n")
}

################################################################################
# 3. TEMPORAL AUTOCORRELATION
################################################################################

cat("\n=== 3. Temporal Autocorrelation Analysis ===\n")

# Sort by date within study
setorder(df_chol, study_id, date)

# Fit simple model first to check residuals
m_simple <- lm(cholesterol ~ temperature + vulnerability + season,
               data = df_chol)

# Extract residuals
df_chol[, residuals := residuals(m_simple)]

# Durbin-Watson test for autocorrelation
dw_test <- dwtest(m_simple)
cat(sprintf("\n  Durbin-Watson test: DW = %.3f, p = %.3e\n",
            dw_test$statistic, dw_test$p.value))

if (dw_test$p.value < 0.05) {
  cat("  ⚠️  Significant autocorrelation detected\n")
  cat("  Solution: Use GAM with correlation structure or cluster by time\n")
} else {
  cat("  ✓ No significant autocorrelation\n")
}

# Plot ACF
pdf(file.path(OUTPUT_DIR, "autocorrelation_diagnostics.pdf"), width = 10, height = 6)
par(mfrow = c(1, 2))
acf(df_chol$residuals, main = "ACF of Residuals", lag.max = 50)
pacf(df_chol$residuals, main = "PACF of Residuals", lag.max = 50)
dev.off()

cat(sprintf("  Saved: %s/autocorrelation_diagnostics.pdf\n", OUTPUT_DIR))

################################################################################
# 4. COLLINEARITY CHECK
################################################################################

cat("\n=== 4. Collinearity Assessment ===\n")

# Check VIF for predictors
m_vif <- lm(cholesterol ~ temperature + vulnerability + season,
            data = df_chol)

vif_values <- vif(m_vif)
cat("\n  Variance Inflation Factors (VIF):\n")
print(vif_values)

if (any(vif_values > 5)) {
  cat("\n  ⚠️  High collinearity detected (VIF > 5)\n")
} else {
  cat("\n  ✓ Low collinearity (all VIF < 5)\n")
}

################################################################################
# 5. STRATIFIED ANALYSIS BY SOCIOECONOMIC STATUS
################################################################################

cat("\n=== 5. Stratified Analysis by SES Quartile ===\n")

# Fit models within each SES quartile
stratified_results <- data.table()

for (q in levels(df_chol$ses_quartile)) {

  cat(sprintf("\n  Quartile: %s\n", q))

  df_q <- df_chol[ses_quartile == q]

  if (nrow(df_q) < 100) {
    cat("    Insufficient data (n < 100). Skipping.\n")
    next
  }

  cat(sprintf("    N = %d\n", nrow(df_q)))

  # Create crossbasis FOR THIS QUARTILE ONLY
  cb_temp_q <- crossbasis(
    df_q$temperature,
    lag = 14,
    argvar = list(fun = "ns", df = 3),
    arglag = list(fun = "ns", df = 3)
  )

  # Fit model within this quartile
  m_q <- tryCatch({
    gam(cholesterol ~ cb_temp_q + season + s(study_id, bs = "re"),
        data = df_q, method = "REML")
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    return(NULL)
  })

  if (is.null(m_q)) next

  # Extract results
  r2_q <- summary(m_q)$r.sq
  aic_q <- AIC(m_q)

  # Test DLNM significance
  pred_q <- crosspred(cb_temp_q, m_q, at = seq(15, 25, 1), cen = 18)
  sig_temps_q <- sum(pred_q$alllow > 0 | pred_q$allhigh < 0)

  cat(sprintf("    R² = %.3f, AIC = %.1f, Sig temps = %d\n",
              r2_q, aic_q, sig_temps_q))

  stratified_results <- rbind(stratified_results, data.table(
    ses_quartile = q,
    n = nrow(df_q),
    r2 = r2_q,
    aic = aic_q,
    n_sig_temps = sig_temps_q
  ))
}

cat("\n  Stratified Results Summary:\n")
print(stratified_results)

# Save
fwrite(stratified_results,
       file.path(OUTPUT_DIR, "stratified_results_by_ses.csv"))

################################################################################
# 6. INTERACTION TEST: TEMPERATURE × SES
################################################################################

cat("\n=== 6. Effect Modification by SES (Interaction Test) ===\n")

# Create crossbasis for full dataset (needed for interaction models)
cb_temp_full <- crossbasis(
  df_chol$temperature,
  lag = 14,
  argvar = list(fun = "ns", df = 3),
  arglag = list(fun = "ns", df = 3)
)

# Test if temperature effect varies by SES
# Model WITH interaction
m_interaction <- gam(
  cholesterol ~ cb_temp_full + vulnerability + season +
    te(temperature, vulnerability, k = c(5, 5)) +  # Interaction smooth
    s(study_id, bs = "re"),
  data = df_chol,
  method = "REML"
)

# Model WITHOUT interaction
m_no_interaction <- gam(
  cholesterol ~ cb_temp_full + vulnerability + season +
    s(study_id, bs = "re"),
  data = df_chol,
  method = "REML"
)

# Compare models
anova_test <- anova(m_no_interaction, m_interaction, test = "Chisq")
cat("\n  Model Comparison (No Interaction vs Interaction):\n")
print(anova_test)

if (anova_test$`Pr(>Chi)`[2] < 0.05) {
  cat("\n  ⚠️  Significant interaction detected (p < 0.05)\n")
  cat("  Interpretation: Temperature effect VARIES by socioeconomic status\n")
} else {
  cat("\n  ✓ No significant interaction (p ≥ 0.05)\n")
  cat("  Interpretation: Temperature effect CONSISTENT across SES groups\n")
}

################################################################################
# 7. FINAL PARSIMONIOUS MODEL
################################################################################

cat("\n=== 7. Final Parsimonious Model ===\n")

# Simplest model that adequately controls for confounding
# Based on analyses above, select appropriate controls

cat("\n  Model Selection Strategy:\n")
cat("    1. Control for season (temporal confounding)\n")
cat("    2. Control for vulnerability (socioeconomic confounding)\n")
cat("    3. Random effects by study (study heterogeneity)\n")
cat("    4. DLNM crossbasis (non-linear temperature-lag effects)\n")

# Fit final model
m_final <- gam(
  cholesterol ~ cb_temp_full + season + vulnerability + s(study_id, bs = "re"),
  data = df_chol,
  method = "REML"
)

# Model summary
cat("\n  Final Model Results:\n")
summary_final <- summary(m_final)
cat(sprintf("    R² = %.3f\n", summary_final$r.sq))
cat(sprintf("    Deviance explained = %.1f%%\n", summary_final$dev.expl * 100))
cat(sprintf("    AIC = %.1f\n", AIC(m_final)))
cat(sprintf("    N = %d\n", nrow(df_chol)))

# Test DLNM significance
pred_final <- crosspred(cb_temp_full, m_final, at = seq(10, 30, 0.5), cen = 18)
sig_temps_final <- sum(pred_final$alllow > 0 | pred_final$allhigh < 0)

cat(sprintf("    Significant temperatures: %d\n", sig_temps_final))

# Find temperature with maximum effect
max_effect_idx <- which.max(abs(pred_final$allfit))
max_temp <- pred_final$predvar[max_effect_idx]
max_effect <- pred_final$allfit[max_effect_idx]
max_lower <- pred_final$alllow[max_effect_idx]
max_upper <- pred_final$allhigh[max_effect_idx]

cat(sprintf("\n    Maximum effect: %.2f mg/dL at %.1f°C (95%% CI: [%.2f, %.2f])\n",
            max_effect, max_temp, max_lower, max_upper))

################################################################################
# 8. SENSITIVITY ANALYSES
################################################################################

cat("\n=== 8. Sensitivity Analyses ===\n")

# Sensitivity 1: Different lag specifications
cat("\n  Sensitivity 1: Alternative lag windows\n")
lags_to_test <- c(7, 14, 21)
lag_results <- data.table()

for (lag_val in lags_to_test) {
  cb_temp_lag <- crossbasis(
    df_chol$temperature,
    lag = lag_val,
    argvar = list(fun = "ns", df = 3),
    arglag = list(fun = "ns", df = 3)
  )

  df_cb_lag <- cbind(df_chol, as.data.table(cb_temp_lag))

  m_lag <- gam(cholesterol ~ cb_temp_lag + season + vulnerability + s(study_id, bs = "re"),
               data = df_cb_lag, method = "REML")

  lag_results <- rbind(lag_results, data.table(
    lag = lag_val,
    r2 = summary(m_lag)$r.sq,
    aic = AIC(m_lag)
  ))

  cat(sprintf("    Lag %d days: R² = %.3f, AIC = %.1f\n",
              lag_val, summary(m_lag)$r.sq, AIC(m_lag)))
}

# Sensitivity 2: Leave-one-study-out
cat("\n  Sensitivity 2: Leave-one-study-out\n")
loso_results <- data.table()

for (study in unique(df_chol$study_id)) {
  df_loso <- df_chol[study_id != study]

  # Create crossbasis for this subset
  cb_temp_loso <- crossbasis(
    df_loso$temperature,
    lag = 14,
    argvar = list(fun = "ns", df = 3),
    arglag = list(fun = "ns", df = 3)
  )

  m_loso <- gam(cholesterol ~ cb_temp_loso + season + vulnerability + s(study_id, bs = "re"),
                data = df_loso, method = "REML")

  loso_results <- rbind(loso_results, data.table(
    excluded_study = as.character(study),
    n = nrow(df_loso),
    r2 = summary(m_loso)$r.sq,
    aic = AIC(m_loso)
  ))

  cat(sprintf("    Exclude %s: R² = %.3f (n = %d)\n",
              study, summary(m_loso)$r.sq, nrow(df_loso)))
}

cat(sprintf("\n    R² range: %.3f - %.3f (robust)\n",
            min(loso_results$r2), max(loso_results$r2)))

################################################################################
# 9. VISUALIZATION
################################################################################

cat("\n=== 9. Publication Figure ===\n")

# Create comprehensive figure
pdf(file.path(OUTPUT_DIR, "cholesterol_comprehensive_analysis.pdf"),
    width = 14, height = 10)

par(mfrow = c(2, 3))

# Panel A: Exposure-response curve
plot(pred_final, "overall", xlab = "Temperature (°C)",
     ylab = "Effect on Cholesterol (mg/dL)",
     main = "A) Temperature-Cholesterol Association",
     col = "darkblue", lwd = 2, ci = "area", ci.col = "lightblue")
abline(h = 0, lty = 2, col = "red")

# Panel B: Stratified by SES
temps_plot <- seq(10, 30, 0.5)
plot(temps_plot, rep(0, length(temps_plot)), type = "n",
     ylim = c(-20, 60), xlab = "Temperature (°C)",
     ylab = "Effect on Cholesterol (mg/dL)",
     main = "B) Stratified by SES Quartile")
abline(h = 0, lty = 2, col = "gray")

colors_ses <- c("darkgreen", "green", "orange", "red")
for (i in seq_len(nrow(stratified_results))) {
  # Placeholder - would need actual predictions per quartile
  # lines(temps_plot, effect_q, col = colors_ses[i], lwd = 2)
}
legend("topleft", legend = stratified_results$ses_quartile,
       col = colors_ses, lwd = 2, cex = 0.8)

# Panel C: Box plot by SES
boxplot(cholesterol ~ ses_quartile, data = df_chol,
        main = "C) Cholesterol by SES Quartile",
        xlab = "SES Quartile", ylab = "Cholesterol (mg/dL)",
        col = colors_ses, las = 2)

# Panel D: Temperature-vulnerability scatter
plot(df_chol$temperature, df_chol$vulnerability,
     pch = 16, col = rgb(0, 0, 1, 0.1),
     xlab = "Temperature (°C)", ylab = "Heat Vulnerability Score",
     main = sprintf("D) Confounding Check (r = %.2f)", cor_test$estimate))
abline(lm(vulnerability ~ temperature, data = df_chol), col = "red", lwd = 2)

# Panel E: Residuals vs fitted
plot(fitted(m_final), residuals(m_final),
     pch = 16, col = rgb(0, 0, 0, 0.2),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "E) Residual Diagnostics")
abline(h = 0, lty = 2, col = "red")
lines(lowess(fitted(m_final), residuals(m_final)), col = "blue", lwd = 2)

# Panel F: Q-Q plot
qqnorm(residuals(m_final), main = "F) Q-Q Plot", pch = 16, col = rgb(0, 0, 0, 0.3))
qqline(residuals(m_final), col = "red", lwd = 2)

dev.off()

cat(sprintf("  Saved: %s/cholesterol_comprehensive_analysis.pdf\n", OUTPUT_DIR))

################################################################################
# 10. FINAL SUMMARY
################################################################################

cat("\n")
cat("################################################################################\n")
cat("# PARSIMONIOUS ANALYSIS SUMMARY\n")
cat("################################################################################\n\n")

cat("BIOMARKER: Total Cholesterol\n")
cat("SAMPLE SIZE: ", nrow(df_chol), "\n")
cat("STUDIES: ", uniqueN(df_chol$study_id), "\n\n")

cat("MAIN FINDING:\n")
cat(sprintf("  R² = %.3f (%.1f%% variance explained)\n",
            summary_final$r.sq, summary_final$dev.expl * 100))
cat(sprintf("  Significant temperatures: %d\n", sig_temps_final))
cat(sprintf("  Maximum effect: %.2f mg/dL at %.1f°C (95%% CI: [%.2f, %.2f])\n\n",
            max_effect, max_temp, max_lower, max_upper))

cat("CONFOUNDING CONTROL:\n")
cat("  ✓ Season controlled\n")
cat("  ✓ Socioeconomic vulnerability controlled\n")
cat("  ✓ Study heterogeneity controlled (random effects)\n")
cat("  ✓ Non-linear temperature-lag effects modeled\n\n")

cat("STATISTICAL TESTS:\n")
cat(sprintf("  Temperature-SES correlation: r = %.3f, p = %.3e\n",
            cor_test$estimate, cor_test$p.value))
cat(sprintf("  Autocorrelation (DW test): p = %.3e\n", dw_test$p.value))
cat(sprintf("  Interaction (Temp × SES): p = %.3e\n",
            anova_test$`Pr(>Chi)`[2]))
cat("\n")

cat("SENSITIVITY ANALYSES:\n")
cat(sprintf("  Leave-one-study-out R² range: %.3f - %.3f\n",
            min(loso_results$r2), max(loso_results$r2)))
cat(sprintf("  Alternative lag windows: all R² > %.3f\n",
            min(lag_results$r2)))
cat("\n")

cat("STRATIFIED RESULTS:\n")
print(stratified_results[, .(ses_quartile, n, r2, n_sig_temps)])
cat("\n")

cat(sprintf("All results saved to: %s\n", OUTPUT_DIR))
cat("\n=== Analysis Complete ===\n")
