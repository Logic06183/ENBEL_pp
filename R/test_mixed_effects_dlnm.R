#!/usr/bin/env Rscript
################################################################################
# Test Mixed Effects DLNM - Quick Single Biomarker Test
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(mgcv)
  library(dlnm)
  library(splines)
})

set.seed(42)

# Load data
DATA_PATH <- "../data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
cat("Loading data...\n")
df <- fread(DATA_PATH)

cat(sprintf("Loaded %d rows\n", nrow(df)))
cat(sprintf("Columns: %s\n", paste(head(names(df), 10), collapse = ", ")))

# Check for Hematocrit
biomarker_name <- "Hematocrit (%)"
if (!biomarker_name %in% names(df)) {
  stop(sprintf("Biomarker '%s' not found in data", biomarker_name))
}

cat(sprintf("\nPreparing data for %s...\n", biomarker_name))

# Select required columns
df_clean <- df[, .(
  biomarker = get(biomarker_name),
  temperature = climate_7d_mean_temp,
  vulnerability = HEAT_VULNERABILITY_SCORE,
  study_id = as.factor(study_source),
  patient_id = paste0(study_source, "_", anonymous_patient_id),
  season = as.factor(season),
  date = as.Date(primary_date)
)]

# Remove NAs
df_clean <- na.omit(df_clean)

cat(sprintf("Complete cases: %d\n", nrow(df_clean)))
cat(sprintf("Unique patients: %d\n", uniqueN(df_clean$patient_id)))
cat(sprintf("Unique studies: %d\n", uniqueN(df_clean$study_id)))

# Count repeated measures
df_clean[, n_obs := .N, by = patient_id]
cat(sprintf("Patients with >1 observation: %d\n", sum(df_clean$n_obs > 1)))

# Create DLNM crossbasis (simplified: 14-day lag, 3df each)
cat("\nCreating DLNM crossbasis...\n")
cb_temp <- crossbasis(
  df_clean$temperature,
  lag = 14,
  argvar = list(fun = "ns", df = 3),
  arglag = list(fun = "ns", df = 3)
)

# Combine with data
df_cb <- cbind(df_clean, as.data.table(cb_temp))
df_cb[, patient_id := as.factor(patient_id)]

# Model 1: No random effects
cat("\nFitting Model 1: No random effects...\n")
m1 <- gam(
  biomarker ~ cb_temp + season + vulnerability,
  data = df_cb,
  method = "REML"
)
cat(sprintf("  AIC: %.2f\n", AIC(m1)))
cat(sprintf("  R²: %.3f\n", summary(m1)$r.sq))

# Model 2: Random intercept by study
cat("\nFitting Model 2: Random intercept by study...\n")
m2 <- gam(
  biomarker ~ cb_temp + season + vulnerability + s(study_id, bs = "re"),
  data = df_cb,
  method = "REML"
)
cat(sprintf("  AIC: %.2f\n", AIC(m2)))
cat(sprintf("  R²: %.3f\n", summary(m2)$r.sq))

# Model 3: Random intercept by patient
cat("\nFitting Model 3: Random intercept by patient...\n")
m3 <- gam(
  biomarker ~ cb_temp + season + vulnerability + s(patient_id, bs = "re"),
  data = df_cb,
  method = "REML"
)
cat(sprintf("  AIC: %.2f\n", AIC(m3)))
cat(sprintf("  R²: %.3f\n", summary(m3)$r.sq))

# Compare models
cat("\n=== Model Comparison ===\n")
comparison <- data.table(
  model = c("No random effects", "Random study intercept", "Random patient intercept"),
  aic = c(AIC(m1), AIC(m2), AIC(m3)),
  r2 = c(summary(m1)$r.sq, summary(m2)$r.sq, summary(m3)$r.sq)
)
comparison[, aic_rank := rank(aic)]
print(comparison[order(aic)])

# Test DLNM prediction
cat("\n=== Testing DLNM Predictions ===\n")
best_model <- m2  # Use model 2 for testing
pred <- crosspred(cb_temp, best_model, at = seq(15, 25, 1), cen = 18)

cat(sprintf("Prediction completed successfully!\n"))
cat(sprintf("Temperature range: %.1f to %.1f°C\n", min(pred$predvar), max(pred$predvar)))
cat(sprintf("Effect range: %.3f to %.3f\n", min(pred$allfit), max(pred$allfit)))

cat("\n=== Test completed successfully! ===\n")
