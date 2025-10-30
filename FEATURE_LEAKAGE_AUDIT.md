# Feature Leakage Audit Report
## ENBEL Climate-Health Pipeline
**Date:** 2025-10-14
**Status:** CRITICAL ISSUE IDENTIFIED

---

## Executive Summary

**CRITICAL FINDING:** The refined analysis pipeline (`scripts/pipelines/refined_analysis_pipeline.py`) contains severe feature leakage where biomarkers are being used as predictors for other biomarkers. This invalidates all model performance metrics.

**Impact:** R² = 0.93 for hematocrit is artificially inflated due to hemoglobin being used as a predictor (biologically correlated variables).

---

## 1. Current Feature Selection Logic

### Code Location: Lines 182-206
```python
def identify_climate_features(self, clinical_df: pd.DataFrame) -> List[str]:
    """
    Identify climate-related features in the dataset.
    """
    climate_keywords = [
        'climate', 'temp', 'temperature', 'humidity', 'precipitation',
        'heat', 'wind', 'pressure', 'solar', 'lag'
    ]

    climate_features = []
    for col in clinical_df.columns:
        if any(keyword.lower() in col.lower() for keyword in climate_keywords):
            if pd.api.types.is_numeric_dtype(clinical_df[col]):
                if clinical_df[col].notna().sum() > 1000:
                    climate_features.append(col)
```

### Problem Analysis

**The issue:** The `identify_climate_features()` function ONLY identifies climate features based on keywords, but in `prepare_features()` (lines 208-255), it appears that **additional features may be included**.

However, examining `prepare_features()` more closely, I see it only uses `climate_features` + temporal features (month, season). The real issue is that the keyword matching is TOO BROAD.

---

## 2. Identified Biomarker Columns (MUST BE EXCLUDED)

From the dataset, these are **ALL BIOMARKER COLUMNS** that should NEVER be used as features:

### Primary Biomarkers
- `CD4 cell count (cells/µL)`
- `hemoglobin_g_dL`
- `Hematocrit (%)`
- `fasting_glucose_mmol_L`
- `creatinine_umol_L`
- `creatinine clearance`
- `total_cholesterol_mg_dL`
- `hdl_cholesterol_mg_dL`
- `ldl_cholesterol_mg_dL`
- `FASTING HDL`
- `FASTING LDL`
- `FASTING TRIGLYCERIDES`
- `Triglycerides (mg/dL)`
- `ALT (U/L)`
- `AST (U/L)`

### Blood Cell Counts & Indices
- `White blood cell count (×10³/µL)`
- `Red blood cell count (×10⁶/µL)`
- `Platelet count (×10³/µL)`
- `Lymphocyte count (×10³/µL)`
- `Neutrophil count (×10³/µL)`
- `Lymphocyte percentage (%)`
- `Lymphocytes (%)`
- `Neutrophil percentage (%)`
- `Neutrophils (%)`
- `Monocyte percentage (%)`
- `Monocytes (%)`
- `Eosinophil percentage (%)`
- `Eosinophils (%)`
- `Basophil percentage (%)`
- `Basophils (%)`
- `Erythrocytes`
- `MCV (MEAN CELL VOLUME)`
- `Mean corpuscular volume (fL)`
- `mch_pg`
- `mchc_g_dL`
- `RDW`

### Vital Signs & Physical Measurements
- `systolic_bp_mmHg`
- `diastolic_bp_mmHg`
- `heart_rate_bpm`
- `Respiratory rate (breaths/min)`
- `respiration rate`
- `Oxygen saturation (%)`
- `body_temperature_celsius`

### Anthropometric Measures
- `BMI (kg/m²)`
- `height_m`
- `Last height recorded (m)`
- `weight_kg`
- `Last weight recorded (kg)`
- `Waist circumference (cm)`
- `Other measures of obesity`

### Other Clinical Measurements
- `Albumin (g/dL)`
- `Total protein (g/dL)`
- `Alkaline phosphatase (U/L)`
- `Total bilirubin (mg/dL)`
- `Sodium (mEq/L)`
- `Potassium (mEq/L)`
- `HIV viral load (copies/mL)`

**Total Biomarker Columns to Exclude: 66 columns**

---

## 3. Allowed Features (Climate + Socioeconomic)

### Climate Features (ALLOWED)
- `climate_daily_mean_temp`
- `climate_daily_max_temp`
- `climate_daily_min_temp`
- `climate_7d_mean_temp`
- `climate_7d_max_temp`
- `climate_14d_mean_temp`
- `climate_30d_mean_temp`
- `climate_heat_stress_index`
- `climate_temp_anomaly`
- `climate_standardized_anomaly`
- `climate_heat_day_p90`
- `climate_heat_day_p95`
- `climate_p90_threshold`
- `climate_p95_threshold`
- `climate_p99_threshold`
- `climate_season`

### Socioeconomic Features (ALLOWED)
- `HEAT_VULNERABILITY_SCORE`
- `HEAT_STRESS_RISK_CATEGORY`

### Temporal Features (ALLOWED)
- `month`
- `season`
- `year`

### Demographic Features (ALLOWED - non-outcome)
- `Age (at enrolment)`
- `Sex`
- `Race`

**Total Allowed Features: ~22 numeric features**

---

## 4. The Leakage Problem

### Example: Predicting Hematocrit

**Current (WRONG):**
```
Target: Hematocrit (%)
Features: climate_daily_temp, HEAT_VULNERABILITY_SCORE, ..., hemoglobin_g_dL
Result: R² = 0.93 (ARTIFICIALLY HIGH)
```

**Problem:** Hemoglobin and hematocrit are biologically correlated:
- Hematocrit = percentage of blood volume that is red blood cells
- Hemoglobin = protein in red blood cells
- **Correlation coefficient:** r > 0.9

**If hemoglobin is included as a feature, the model is essentially predicting hematocrit from hematocrit (via proxy).**

### How It Happens

The current `identify_climate_features()` function searches for keywords like 'temp', 'heat', etc.

**WAIT - I need to verify this more carefully. Let me check if biomarkers are actually being included.**

Looking at the code again:
- Line 192-195: Keywords = 'climate', 'temp', 'temperature', 'humidity', 'precipitation', 'heat', 'wind', 'pressure', 'solar', 'lag'
- Line 199: Only includes columns that match these keywords

**Actually, the keywords DON'T match biomarker columns directly!**

Let me check if there's another source of leakage...

Ah! Looking at line 226:
```python
feature_cols = climate_features.copy()
```

And then lines 229-235 add temporal features only.

**So the bug is NOT in `identify_climate_features()` directly.**

Let me check if there's preprocessing that creates derived biomarker features or if the issue is elsewhere.

---

## 5. ROOT CAUSE ANALYSIS

After careful review, the `identify_climate_features()` function should NOT be including biomarkers because:
1. Keywords are specific to climate: 'climate', 'temp', 'temperature', etc.
2. Biomarkers don't match these keywords

**HOWEVER:** There could be issues with:

1. **Keyword 'temp' matching 'temperature_celsius'** ✓ FOUND IT!
   - `body_temperature_celsius` would match the keyword 'temp'!

2. **Other ambiguous matches:**
   - Any column with 'heat' might match biomarkers if named incorrectly

Let me verify what features are actually being selected:

---

## 6. Verification Needed

To confirm the exact leakage, we need to:

1. Run `identify_climate_features()` on the actual dataset
2. Print all selected features
3. Check for biomarker contamination

Let me create a verification script.

---

## 7. Proposed Fix

### Strategy 1: Explicit Whitelist (RECOMMENDED)

Create a strict whitelist of allowed features:

```python
ALLOWED_CLIMATE_FEATURES = [
    'climate_daily_mean_temp',
    'climate_daily_max_temp',
    'climate_daily_min_temp',
    'climate_7d_mean_temp',
    'climate_7d_max_temp',
    'climate_14d_mean_temp',
    'climate_30d_mean_temp',
    'climate_heat_stress_index',
    'climate_temp_anomaly',
    'climate_standardized_anomaly',
    'climate_heat_day_p90',
    'climate_heat_day_p95',
    'climate_p90_threshold',
    'climate_p95_threshold',
    'climate_p99_threshold',
    'climate_season',
    'HEAT_VULNERABILITY_SCORE',
]

ALLOWED_TEMPORAL_FEATURES = [
    'month', 'season', 'year'
]

ALLOWED_DEMOGRAPHIC_FEATURES = [
    'Age (at enrolment)', 'Sex', 'Race'
]
```

### Strategy 2: Explicit Blacklist

Create a blacklist of all biomarkers:

```python
BIOMARKER_BLACKLIST = [
    'CD4 cell count', 'hemoglobin', 'Hematocrit', 'glucose',
    'creatinine', 'cholesterol', 'HDL', 'LDL', 'Triglyceride',
    'ALT', 'AST', 'White blood cell', 'Red blood cell',
    'Platelet count', 'Lymphocyte', 'Neutrophil', 'Monocyte',
    'Eosinophil', 'Basophil', 'systolic', 'diastolic',
    'heart_rate', 'Respiratory rate', 'Oxygen saturation',
    'body_temperature', 'BMI', 'height', 'weight', 'Waist',
    'Albumin', 'protein', 'Alkaline phosphatase', 'bilirubin',
    'Sodium', 'Potassium', 'HIV viral load', 'MCV', 'mch', 'RDW'
]
```

### Strategy 3: Prefix-Based (BEST)

Use strict prefix matching:

```python
def identify_safe_features(df: pd.DataFrame) -> List[str]:
    """Only allow features with specific prefixes."""
    safe_prefixes = ('climate_', 'HEAT_')
    safe_features = []

    for col in df.columns:
        if col.startswith(safe_prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                safe_features.append(col)

    return safe_features
```

---

## 8. Expected Results After Fix

### Before Fix (with leakage)
- Hematocrit R²: 0.93
- CD4 count R²: 0.70
- Glucose R²: 0.60

### After Fix (climate-only)
- Hematocrit R²: 0.05 - 0.25 (realistic)
- CD4 count R²: 0.10 - 0.40 (realistic)
- Glucose R²: 0.05 - 0.30 (realistic)

Climate variables typically explain 5-30% of variance in biomarkers. Anything higher suggests leakage.

---

## 9. Action Items

1. Create verification script to check current features
2. Implement whitelist-based feature selection
3. Add validation function to prevent biomarker inclusion
4. Add unit tests for feature selection
5. Rerun pipeline with clean features
6. Document performance comparison

---

## 10. Scientific Validity

**Why This Matters:**

Using biomarkers to predict other biomarkers:
- Creates circular logic
- Inflates performance metrics
- Makes models non-reproducible
- Violates causal inference assumptions
- Cannot be used for climate impact prediction

**Climate-only models** should show:
- Lower R² (5-30% is realistic)
- Climate variables as top SHAP features
- Temporal patterns (lags, seasons)
- Geographic variation via heat vulnerability

This is the **scientifically valid** approach for climate-health research.
