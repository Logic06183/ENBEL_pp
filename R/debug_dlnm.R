# Debug DLNM pipeline
library(dlnm)
library(dplyr)

# Load data
df <- read.csv("data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv", stringsAsFactors = FALSE)
cat("Data loaded:", nrow(df), "rows,", ncol(df), "columns\n")

# Check specific columns
biomarker <- "total_cholesterol_mg_dL"
climate_var <- "climate_daily_mean_temp"

cat("Biomarker column exists:", biomarker %in% colnames(df), "\n")
cat("Climate variable exists:", climate_var %in% colnames(df), "\n")

# Check data availability
df$date <- as.Date(df$primary_date)
required_cols <- c("date", biomarker, climate_var, "Sex", "Age (at enrolment)")
available_cols <- intersect(required_cols, colnames(df))
cat("Available columns:", length(available_cols), "of", length(required_cols), "\n")
cat("Available:", paste(available_cols, collapse = ", "), "\n")

# Create subset
df_subset <- df[, available_cols]
df_subset <- df_subset[complete.cases(df_subset), ]
cat("Complete cases:", nrow(df_subset), "\n")

if (nrow(df_subset) > 50) {
  # Try creating crossbasis
  cb <- crossbasis(df_subset[[climate_var]], lag = 7, 
                   argvar = list(fun = "ns", df = 3),
                   arglag = list(fun = "ns", df = 3))
  
  cat("Crossbasis created:", nrow(cb), "x", ncol(cb), "\n")
  
  # Try simple model
  model_data <- data.frame(
    biomarker = df_subset[[biomarker]],
    cb
  )
  
  cat("Model data dimensions:", nrow(model_data), "x", ncol(model_data), "\n")
  cat("Column names:", paste(colnames(model_data), collapse = ", "), "\n")
  
  # Fit simple model
  model <- glm(biomarker ~ ., data = model_data, family = gaussian())
  cat("Model fitted successfully\n")
  cat("R-squared:", 1 - (model$deviance / model$null.deviance), "\n")
}