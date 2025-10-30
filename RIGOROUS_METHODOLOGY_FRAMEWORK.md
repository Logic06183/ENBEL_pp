# Rigorous Methodology Framework
## ENBEL Climate-Health Model Refinement
**Version:** 1.0
**Date:** 2025-10-30
**Status:** 🔬 Research-Grade Implementation Guide

---

## Overview

This document establishes a rigorous, reproducible framework for climate-health biomarker modeling following best practices from epidemiology, biostatistics, machine learning, and climate health research literature.

---

## 1. Research Quality Standards

### 1.1 Guiding Principles

**Based on:**
- STROBE guidelines for observational studies
- TRIPOD guidelines for prediction model development
- RECORD guidelines for routinely collected health data
- Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD)
- Guidelines for Accurate and Transparent Health Estimates Reporting (GATHER)

**Core Requirements:**
1. ✅ **Transparency:** All decisions documented with rationale
2. ✅ **Reproducibility:** Seed control, version control, containerization
3. ✅ **Validation:** Multiple validation strategies (temporal, spatial, cross-validation)
4. ✅ **Robustness:** Sensitivity analyses for all key assumptions
5. ✅ **Interpretability:** Explainable AI for clinical translation

---

## 2. Data Quality Assessment Framework

### 2.1 Clinical Dataset Quality Checks

**Sample:** 11,398 records from 15 HIV clinical trials (2002-2021)

#### Completeness Analysis
```python
# For each variable, calculate:
- Missing rate: n_missing / n_total
- Missing pattern: MCAR, MAR, MNAR (Little's test)
- Missingness by study: chi-square test for differential missingness
- Temporal patterns: missing by year/period
```

**Quality Metrics:**
| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| Overall completeness | ≥80% | Document in limitations |
| Biomarker completeness | ≥200 observations | Exclude from analysis |
| Climate coverage | ≥95% | Already achieved (99.5%) |
| Geographic validity | 100% within Johannesburg | Filter invalid coordinates |

#### Data Validation Rules
1. **Range checks:** Biomarkers within physiologically plausible ranges
   - CD4: 0-2000 cells/µL
   - Glucose: 2.0-30.0 mmol/L
   - BP: Systolic 60-220, Diastolic 40-140 mmHg
   - Temperature: 35.0-42.0°C

2. **Consistency checks:**
   - Date validity: primary_date within study period
   - Coordinate validity: within Johannesburg metropolitan boundary
   - Duplicate detection: same patient, date, biomarker

3. **Outlier detection:**
   - Method: Isolation Forest (contamination=0.05)
   - Z-score threshold: |z| > 5 (extreme outliers)
   - Clinical review: Outliers flagged for expert review

#### Missingness Analysis Strategy

**Step 1: Descriptive Analysis**
```r
# Calculate missingness patterns
library(mice)
library(VIM)

# 1. Missing data pattern visualization
md.pattern(clinical_data)

# 2. Missing data mechanism testing
# Little's MCAR test (H0: data are MCAR)
mcar_test <- mcar.test(clinical_data)

# 3. Correlations between missingness indicators
missing_indicators <- is.na(clinical_data)
cor(missing_indicators)
```

**Step 2: Mechanism Classification**

| Variable Category | Expected Mechanism | Handling Strategy |
|-------------------|-------------------|-------------------|
| **Biomarkers** | MAR (depends on study, date) | Multiple imputation |
| **Climate data** | MCAR (data extraction errors) | Already 99.5% complete |
| **Socioeconomic** | MAR (spatial, demographic) | KNN + multiple imputation |
| **Demographics** | MAR (enrollment procedures) | Mode imputation (minimal missing) |

**Step 3: Sensitivity Analysis**
- Complete case analysis (comparison baseline)
- Multiple imputation (m=5, m=10, m=20 datasets)
- Sensitivity to imputation method (MICE, missForest, KNN)

---

### 2.2 GCRO Dataset Quality Checks

**Sample:** 58,616 household records from 6 survey waves (2011-2021)

#### Survey Quality Assessment
```python
# 1. Response rate by ward
response_rate = households_surveyed / eligible_households

# 2. Survey wave completeness
wave_completeness = complete_surveys / total_surveys

# 3. Socioeconomic variable coverage
for var in socioeconomic_variables:
    coverage[var] = (1 - missing_rate[var]) * 100
```

**Quality Thresholds:**
| Variable | Minimum Coverage | Current Status |
|----------|------------------|----------------|
| Dwelling type | ≥90% | 95.2% ✅ |
| Income level | ≥70% | 78.3% ✅ |
| Employment status | ≥80% | 84.1% ✅ |
| Education level | ≥85% | 91.7% ✅ |
| Ward location | 100% | 100% ✅ |

#### Spatial Matching Quality
```python
# Assess imputation quality from GCRO to clinical records
matching_metrics = {
    'match_rate': n_matched / n_total,
    'mean_distance_km': np.mean(distances),
    'median_distance_km': np.median(distances),
    'within_5km': np.sum(distances <= 5) / n_total,
    'within_15km': np.sum(distances <= 15) / n_total,
}
```

**Expected Quality:**
- Match rate: ≥95% (currently achieved)
- Median distance: <5 km (ward-level matching)
- Within 15 km: 100% (max distance threshold)

---

## 3. Missing Data Handling Strategy

### 3.1 Theoretical Framework

**Literature Foundation:**
- Rubin (1976): Missing data mechanisms (MCAR, MAR, MNAR)
- Little & Rubin (2020): Statistical Analysis with Missing Data
- van Buuren & Groothuis-Oudshoorn (2011): mice: Multivariate Imputation by Chained Equations in R
- Stekhoven & Bühlmann (2012): MissForest for mixed-type data

### 3.2 Multiple Imputation Protocol

**Method:** Multivariate Imputation by Chained Equations (MICE)

**Configuration:**
```yaml
imputation:
  method: "MICE"
  n_imputations: 10  # Recommended: m ≥ % missing
  max_iterations: 10

  # Predictor matrix: Which variables predict which
  predictors:
    biomarkers: ["climate_vars", "demographics", "study_vars"]
    socioeconomic: ["spatial_vars", "demographics", "climate_vars"]

  # Imputation method by variable type
  methods:
    continuous: "pmm"  # Predictive mean matching
    binary: "logreg"  # Logistic regression
    categorical: "polyreg"  # Multinomial logistic regression
    ordinal: "polr"  # Proportional odds model
```

**Validation Approach:**
1. **Holdout validation:**
   - Remove 10% of observed values
   - Impute using remaining data
   - Compare imputed vs. true values
   - Metrics: RMSE, MAE, correlation

2. **Imputation diagnostics:**
   ```r
   # Convergence plots
   plot(imputation_object, type = "convergence")

   # Density plots: observed vs imputed
   densityplot(imputation_object)

   # Stripplots: check imputed value ranges
   stripplot(imputation_object)
   ```

3. **Sensitivity analysis:**
   - Vary imputation method (MICE vs missForest vs KNN)
   - Vary number of imputations (m = 5, 10, 20)
   - Compare model results across imputation strategies
   - Report range of estimates (min-max, IQR)

### 3.3 Feature-Specific Missing Data Handling

#### Climate Variables (99.5% complete)
**Current status:** Excellent
**Action:** Exclude 0.5% with missing climate (n≈57 records)
**Rationale:** Too few to impute reliably, climate is primary exposure

#### Biomarkers (varies by marker)
**Action:** Multiple imputation IF ≥200 observations AND ≥70% complete
**Otherwise:** Exclude biomarker from analysis
**Rationale:** Preserve statistical power, avoid imputation uncertainty

#### Socioeconomic Variables (being added)
**Current:** HEAT_VULNERABILITY_SCORE already imputed (95% coverage)
**New variables:** Apply spatial-demographic matching + MICE
**Validation:** Compare imputed vs survey-matched values where available

---

## 4. Feature Engineering Best Practices

### 4.1 Socioeconomic Feature Expansion

**Literature Foundation:**
- Solar & Irwin (2010): A conceptual framework for action on the social determinants of health
- WHO Commission on Social Determinants of Health (2008)
- Marmot Review (2010): Fair Society, Healthy Lives

**Feature Selection Criteria:**
1. ✅ **Theoretical relevance:** Documented link to health in literature
2. ✅ **Measurement quality:** Valid, reliable GCRO survey instruments
3. ✅ **Sufficient variability:** Not constant across population
4. ✅ **Low multicollinearity:** VIF < 5 (preferred < 3)

**New Socioeconomic Features (14 total):**

| Feature | Type | Rationale | Reference |
|---------|------|-----------|-----------|
| Income level | Ordinal | Economic access to healthcare | Marmot 2010 |
| Employment status | Categorical | Financial security, stress | Solar & Irwin 2010 |
| Education level | Ordinal | Health literacy, adaptive capacity | WHO 2008 |
| Dwelling type | Categorical | Heat exposure, living conditions | Kovats & Hajat 2008 |
| Household size | Continuous | Crowding, disease transmission | Baker et al. 2020 |
| Infrastructure quality | Ordinal | Service access, vulnerability | Watts et al. 2015 |
| Ward-level density | Continuous | Urban heat island effect | Harlan et al. 2006 |
| Age vulnerability | Composite | Thermoregulation capacity | Kenny et al. 2010 |

### 4.2 Feature Interaction Terms

**Theoretical Justification:**
Climate impacts on health are **modified by** socioeconomic vulnerability (WHO 2014, Ebi et al. 2021)

**Interaction Terms to Test:**
```python
interactions = [
    # Heat × Vulnerability
    'climate_daily_mean_temp × HEAT_VULNERABILITY_SCORE',
    'climate_heat_stress_index × dwelling_type_enhanced',

    # Age × Climate (differential susceptibility)
    'Age × climate_temp_anomaly',
    'Age × climate_heat_day_p95',

    # Socioeconomic × Climate
    'income_level × climate_heat_stress_index',
    'education_level × climate_temp_anomaly',
]
```

**Interaction Testing Protocol:**
1. Test each interaction term individually
2. Retain if:
   - p < 0.05 (after multiple testing correction)
   - Improves cross-validated R² by ≥0.01
   - Biologically plausible interpretation
3. Maximum 5 interactions per model (avoid overfitting)

### 4.3 Multicollinearity Assessment

**Method:** Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for all features
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

# Flag problematic features
high_vif = vif_data[vif_data['VIF'] > 5]
```

**Thresholds:**
- VIF < 3: Ideal (low multicollinearity)
- VIF 3-5: Moderate (acceptable with caution)
- VIF > 5: High (consider removing or combining)
- VIF > 10: Severe (must address)

**Resolution Strategy:**
1. Remove one of two highly correlated features (r > 0.95)
2. Create composite indices (e.g., socioeconomic vulnerability score)
3. Use regularization (Elastic Net L1+L2 penalty)
4. Principal Component Analysis (if interpretability not critical)

---

## 5. Model Development Protocol

### 5.1 Train-Test Split Strategy

**Primary Split: Temporal Validation**
```python
# RECOMMENDED for time-series climate-health data
train = data[data['year'] <= 2015]  # 2002-2015
test = data[data['year'] >= 2016]   # 2016-2021

# Rationale:
# - Mimics real-world prediction (past → future)
# - Tests model stability over climate regime shifts
# - Avoids data leakage from temporal autocorrelation
```

**Secondary Split: Random 80/20**
```python
# For comparison with temporal split
train, test = train_test_split(data,
                                test_size=0.2,
                                random_state=42,
                                stratify=data['study_source'])
```

**Validation:** 5-fold cross-validation within training set

### 5.2 Hyperparameter Optimization

**Method:** Optuna (Bayesian optimization with pruning)

**Best Practices:**
```python
import optuna

def objective(trial):
    # 1. Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
    }

    # 2. Train model with 5-fold CV
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # 3. Return mean CV score
    return cv_scores.mean()

# 4. Optimize
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(),  # Early stopping
    sampler=optuna.samplers.TPESampler(seed=42)  # Reproducible
)
study.optimize(objective, n_trials=100, timeout=3600)

# 5. Retrieve best hyperparameters
best_params = study.best_params
```

**Reproducibility Requirements:**
- ✅ Fix random seed in sampler
- ✅ Fix random seed in model
- ✅ Save optimization history (`study.trials_dataframe()`)
- ✅ Save best hyperparameters to YAML

### 5.3 Cross-Validation Strategy

**Method:** Stratified K-Fold (k=5)

**Stratification Variables:**
- Study source (account for study heterogeneity)
- Year period (2002-2010, 2011-2015, 2016-2021)

```python
from sklearn.model_selection import StratifiedKFold

# Create stratification variable
data['strata'] = data['study_source'] + '_' + data['year_period']

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, data['strata']):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Train and evaluate
    model.fit(X_train_fold, y_train_fold)
    scores.append(model.score(X_val_fold, y_val_fold))

# Report mean ± SD across folds
print(f"CV R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**Why Stratified?**
1. Ensures each fold has similar distribution of studies
2. Accounts for study-specific effects (batch effects)
3. More stable performance estimates

---

## 6. Model Evaluation Framework

### 6.1 Regression Metrics

**Primary Metric: R² (Coefficient of Determination)**
```python
r2 = 1 - (SS_residual / SS_total)
```
- **Interpretation:** Proportion of variance explained
- **Range:** -∞ to 1 (negative indicates worse than mean baseline)
- **Clinical relevance:** R² ≥ 0.30 = meaningful predictive value

**Secondary Metrics:**

1. **Mean Absolute Error (MAE)**
   ```python
   mae = np.mean(np.abs(y_true - y_pred))
   ```
   - **Advantage:** Same units as biomarker (clinically interpretable)
   - **Example:** MAE = 50 cells/µL for CD4

2. **Root Mean Squared Error (RMSE)**
   ```python
   rmse = np.sqrt(np.mean((y_true - y_pred)**2))
   ```
   - **Advantage:** Penalizes large errors more than MAE
   - **Use:** When large prediction errors are particularly harmful

3. **Mean Absolute Percentage Error (MAPE)**
   ```python
   mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   ```
   - **Advantage:** Scale-independent (comparable across biomarkers)
   - **Caution:** Undefined if y_true = 0

### 6.2 Classification Metrics (Clinical Thresholds)

**For each biomarker with established clinical thresholds:**

Example: CD4 < 200 cells/µL (severe immunosuppression)
```python
from sklearn.metrics import roc_auc_score, classification_report

# Convert to binary
y_true_binary = (y_true < 200).astype(int)
y_pred_binary = (y_pred < 200).astype(int)

# Calculate metrics
auc = roc_auc_score(y_true_binary, y_pred_proba)
sensitivity = recall_score(y_true_binary, y_pred_binary)
specificity = recall_score(1 - y_true_binary, 1 - y_pred_binary)
ppv = precision_score(y_true_binary, y_pred_binary)
```

**Metrics:**
- **AUC-ROC:** Discrimination ability (0.5 = random, 1.0 = perfect)
- **Sensitivity:** True positive rate (correctly identify high-risk)
- **Specificity:** True negative rate (correctly identify low-risk)
- **PPV:** Positive predictive value (precision)
- **F1 Score:** Harmonic mean of precision and recall

### 6.3 Calibration Assessment

**Calibration:** Do predicted probabilities match observed frequencies?

```python
from sklearn.calibration import calibration_curve

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true_binary, y_pred_proba, n_bins=10
)

# Hosmer-Lemeshow test
hl_stat, hl_pvalue = hosmer_lemeshow_test(y_true_binary, y_pred_proba)
```

**Visualization:** Calibration plot (predicted vs observed)

---

## 7. Feature Importance & Explainability

### 7.1 SHAP (SHapley Additive exPlanations)

**Theoretical Foundation:**
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions
- Shapley (1953): Game theory foundation

**Implementation:**
```python
import shap

# 1. Initialize explainer (model-agnostic)
explainer = shap.Explainer(model, X_train)

# 2. Calculate SHAP values (subsample if large dataset)
shap_values = explainer(X_test[:1000])

# 3. Generate plots
shap.plots.waterfall(shap_values[0])  # Single prediction
shap.plots.beeswarm(shap_values)      # All predictions
shap.plots.bar(shap_values)           # Mean |SHAP|
```

**Key Outputs:**
1. **Global importance:** Mean absolute SHAP value per feature
2. **Local explanations:** SHAP values for individual predictions
3. **Interaction effects:** SHAP interaction values

**Validation:**
- ✅ SHAP values sum to prediction: f(x) = E[f(X)] + Σ SHAP_i
- ✅ Feature effects align with domain knowledge (climate science, physiology)
- ✅ Stable across multiple random subsamples

### 7.2 Permutation Importance

**Complementary to SHAP (less computationally intensive)**

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2'
)

# Sort features by importance
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
```

**Interpretation:**
- How much does R² decrease when feature is randomly shuffled?
- Higher decrease = more important feature

---

## 8. Reproducibility Checklist

### 8.1 Computational Reproducibility

**Requirements:**
- ✅ **Fixed random seed:** `np.random.seed(42)`, `random_state=42`
- ✅ **Version control:** All code in Git with meaningful commits
- ✅ **Environment:** `requirements.txt` or `environment.yml` with pinned versions
- ✅ **Containerization:** Docker image for exact environment replication
- ✅ **Hardware:** Document CPU/GPU used, note if results vary by hardware

**Seed Setting (comprehensive):**
```python
import numpy as np
import random
import os

def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # If using TensorFlow/PyTorch
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)
```

### 8.2 Analytical Reproducibility

**Documentation Requirements:**
1. **Data provenance:**
   - Source of clinical data (15 trials listed)
   - GCRO survey waves and methodology
   - ERA5 climate data extraction parameters

2. **Preprocessing decisions:**
   - Outlier removal criteria and justification
   - Imputation method and validation
   - Feature engineering transformations

3. **Model specifications:**
   - Hyperparameter values (including defaults)
   - Training algorithm and convergence criteria
   - Validation strategy

4. **Analysis plan:**
   - Pre-specified hypotheses (avoid p-hacking)
   - Multiple testing correction method
   - Sensitivity analyses planned

### 8.3 File Organization

```
ENBEL_pp_model_refinement/
├── data/
│   ├── raw/                    # Original data (read-only)
│   ├── processed/              # Cleaned data with provenance
│   └── README.md               # Data dictionary and sources
├── scripts/
│   ├── 01_data_preparation.py  # Numbered order of execution
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   ├── 04_evaluation.py
│   └── utils/                  # Reusable functions
├── results/
│   ├── models/                 # Trained models (.joblib)
│   ├── metrics/                # Performance metrics (.csv)
│   ├── figures/                # Plots (.png, .svg)
│   └── logs/                   # Training logs
├── tests/
│   ├── test_data_processing.py # Unit tests
│   ├── test_models.py
│   └── test_evaluation.py
├── docs/
│   ├── METHODOLOGY.md          # This document
│   ├── RESULTS.md              # Findings
│   └── CHANGELOG.md            # Version history
├── configs/
│   └── config.yaml             # All parameters in one place
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container specification
└── README.md                   # Quick start guide
```

---

## 9. Sensitivity Analyses

### 9.1 Imputation Sensitivity

**Test:** Does model performance depend on imputation method?

```python
imputation_methods = ['MICE', 'missForest', 'KNN', 'mean']
results = {}

for method in imputation_methods:
    # Impute
    X_imputed = impute(X, method=method)

    # Train model
    model.fit(X_imputed, y)

    # Evaluate
    results[method] = {
        'r2': model.score(X_test, y_test),
        'mae': mean_absolute_error(y_test, model.predict(X_test))
    }

# Compare results
print(pd.DataFrame(results).T)
```

**Reporting:** Present range of estimates across imputation methods

### 9.2 Outlier Sensitivity

**Test:** Are results driven by outliers?

```python
# 1. Winsorize extreme values (cap at 1st/99th percentile)
X_winsorized = winsorize(X, limits=[0.01, 0.01])

# 2. Fit model with/without outliers
model_with_outliers.fit(X, y)
model_without_outliers.fit(X_outliers_removed, y_outliers_removed)

# 3. Compare coefficients/feature importance
compare_feature_importance(model_with, model_without)
```

### 9.3 Temporal Stability

**Test:** Is model performance consistent across time periods?

```python
time_periods = {
    'early': (2002, 2010),
    'middle': (2011, 2015),
    'late': (2016, 2021)
}

for period, (start, end) in time_periods.items():
    data_period = data[(data['year'] >= start) & (data['year'] <= end)]

    # Train on this period, test on others
    # OR train on all periods, evaluate on this period separately
```

### 9.4 Geographic Stability

**Test:** Leave-one-ward-out cross-validation

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
scores = []

for train_idx, test_idx in logo.split(X, y, groups=data['Ward']):
    model.fit(X[train_idx], y[train_idx])
    scores.append(model.score(X[test_idx], y[test_idx]))

# Report: mean ± SD, range
```

---

## 10. Statistical Testing & Multiple Testing Correction

### 10.1 Comparing Model Performance

**Before vs After Feature Expansion:**

```python
from scipy.stats import ttest_rel

# Paired t-test (same CV folds)
statistic, pvalue = ttest_rel(cv_scores_before, cv_scores_after)

# Effect size: Cohen's d
d = (np.mean(cv_scores_after) - np.mean(cv_scores_before)) / \
    np.std(cv_scores_after - cv_scores_before)
```

**Interpretation:**
- p < 0.05: Statistically significant improvement
- Cohen's d > 0.5: Moderate effect size
- Cohen's d > 0.8: Large effect size

### 10.2 Multiple Testing Correction

**Problem:** Testing 28 biomarkers increases false positive risk

**Solution: Bonferroni Correction**
```python
alpha = 0.05
n_tests = 28
alpha_corrected = alpha / n_tests  # 0.05 / 28 = 0.00179
```

**Alternative: False Discovery Rate (FDR)**
```python
from statsmodels.stats.multitest import multipletests

# Benjamini-Hochberg procedure
reject, pvals_corrected, _, _ = multipletests(
    pvals, alpha=0.05, method='fdr_bh'
)
```

**Reporting:** Present both uncorrected and corrected p-values

---

## 11. Documentation Standards

### 11.1 Code Documentation

**Function Docstrings (NumPy style):**
```python
def train_model_system_specific(
    system_name: str,
    biomarker: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> Tuple[object, dict]:
    """
    Train models for a specific physiological system and biomarker.

    Parameters
    ----------
    system_name : str
        Name of physiological system (e.g., 'hematological')
    biomarker : str
        Target biomarker name (e.g., 'Hematocrit (%)')
    X_train : pd.DataFrame, shape (n_samples, n_features)
        Training features
    y_train : pd.Series, shape (n_samples,)
        Training target values
    config : dict
        Configuration dictionary with model hyperparameters

    Returns
    -------
    model : object
        Trained scikit-learn model
    metrics : dict
        Dictionary containing performance metrics:
        - 'r2': R-squared score
        - 'mae': Mean absolute error
        - 'rmse': Root mean squared error

    Examples
    --------
    >>> model, metrics = train_model_system_specific(
    ...     'hematological', 'Hematocrit (%)', X_train, y_train, config
    ... )
    >>> print(f"R² = {metrics['r2']:.3f}")

    Notes
    -----
    Uses 5-fold cross-validation with stratification by study source.
    Hyperparameters are optimized using Optuna with 100 trials.

    References
    ----------
    .. [1] Lundberg et al. (2017). A unified approach to interpreting
           model predictions. NeurIPS 2017.
    """
    # Implementation
```

### 11.2 Analysis Log

**Create a running log of all analysis decisions:**

```markdown
# Analysis Log - ENBEL Model Refinement

## 2025-10-30: Initial Planning
- Decision: Use temporal validation (2002-2015 train, 2016-2021 test)
- Rationale: Climate regime may differ across periods
- Reference: Roberts et al. (2017) Cross-validation strategies

## 2025-10-31: Feature Selection
- Decision: Include 14 new socioeconomic features
- Rationale: Literature supports social determinants of health
- VIF check: All features VIF < 3 (no multicollinearity)
- Reference: Solar & Irwin (2010) Conceptual framework

## 2025-11-01: Imputation Method
- Decision: MICE with m=10 imputations
- Rationale: MAR mechanism likely (Little's test p=0.23)
- Sensitivity: Results consistent across m=5, 10, 20
```

---

## 12. Quality Assurance Procedures

### 12.1 Unit Testing

**Example: Test data processing functions**

```python
import pytest
import pandas as pd
import numpy as np

def test_leakage_checker():
    """Test that leakage checker correctly identifies biomarker leakage."""
    from leakage_checker import LeakageChecker

    checker = LeakageChecker()

    # Test case 1: Safe features
    safe_features = ['climate_daily_mean_temp', 'HEAT_VULNERABILITY_SCORE', 'Age']
    report = checker.check_features('Hematocrit (%)', safe_features)
    assert report.is_safe == True, "Safe features incorrectly flagged"

    # Test case 2: Unsafe features (biomarker leakage)
    unsafe_features = safe_features + ['hemoglobin_g_dL']
    report = checker.check_features('Hematocrit (%)', unsafe_features)
    assert report.is_safe == False, "Leakage not detected"
    assert 'hemoglobin_g_dL' in report.biomarker_leakage

def test_imputation_quality():
    """Test that imputation produces valid values."""
    # Create data with missing values
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, 20, 30, np.nan, 50]
    })

    # Impute
    df_imputed = impute_data(df, method='MICE')

    # Check: No remaining missing values
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain after imputation"

    # Check: Imputed values in reasonable range
    assert df_imputed['feature1'].between(1, 5).all(), "Imputed values out of range"
```

### 12.2 Integration Testing

**Test complete pipeline:**

```python
def test_full_pipeline():
    """Test complete modeling pipeline from data to results."""

    # 1. Load data
    data = load_clinical_data()
    assert len(data) > 10000, "Insufficient data loaded"

    # 2. Preprocess
    data_processed = preprocess_data(data)
    assert data_processed.isna().sum().sum() == 0, "Missing values after preprocessing"

    # 3. Feature engineering
    X, y = engineer_features(data_processed, target='Hematocrit (%)')
    assert X.shape[1] >= 30, "Insufficient features engineered"

    # 4. Leakage check
    checker = LeakageChecker()
    report = checker.check_features('Hematocrit (%)', X.columns.tolist())
    assert report.is_safe, "Leakage detected in engineered features"

    # 5. Train model
    model, metrics = train_model(X, y)
    assert metrics['r2'] > 0, "Model performs worse than baseline"

    # 6. Evaluate
    shap_values = calculate_shap(model, X)
    assert shap_values is not None, "SHAP calculation failed"
```

### 12.3 Continuous Integration (CI)

**GitHub Actions workflow:**

```yaml
name: Model Quality Checks

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=scripts/
      - name: Check code quality
        run: |
          flake8 scripts/
          black --check scripts/
          mypy scripts/
```

---

## 13. Reporting Standards

### 13.1 Results Tables

**Table 1: Baseline Characteristics**
- Sample size per biomarker
- Mean (SD) or Median (IQR) for continuous
- n (%) for categorical
- Stratify by study source or time period

**Table 2: Model Performance**
| Biomarker | n | Before R² | After R² | Δ R² | p-value* |
|-----------|---|-----------|----------|------|----------|
| Hematocrit | 2,120 | 0.937 | 0.945 | +0.008 | 0.23 |
| CD4 | 4,606 | -0.004 | 0.121 | +0.125 | <0.001 |

*Bonferroni-corrected

**Table 3: Feature Importance**
Top 10 features by mean |SHAP value| for each physiological system

### 13.2 Figures

**Figure 1: Study Flowchart**
- CONSORT-style diagram showing:
  - Records screened
  - Exclusions (with reasons)
  - Final analytical sample

**Figure 2: Model Performance Comparison**
- Bar plot: R² before vs after feature expansion
- Error bars: 95% CI from cross-validation

**Figure 3: SHAP Summary Plots**
- One per physiological system
- Beeswarm plot showing feature effects

**Figure 4: Calibration Plots**
- For binary clinical thresholds
- Observed vs predicted frequencies

---

## 14. Validation Checklist

Before finalizing any analysis, verify:

### Data Quality
- [ ] Outliers identified and justified
- [ ] Missing data mechanism assessed (MCAR/MAR/MNAR)
- [ ] Imputation validated on holdout data
- [ ] No duplicates in dataset
- [ ] All dates within valid range
- [ ] All coordinates within Johannesburg boundary

### Feature Engineering
- [ ] VIF < 5 for all features
- [ ] No features with zero variance
- [ ] Leakage check passed (automated validation)
- [ ] Feature distributions reasonable (no extreme skew)

### Modeling
- [ ] Random seed set and documented
- [ ] Train/test split appropriate (temporal or random)
- [ ] Cross-validation stratified appropriately
- [ ] Hyperparameters optimized (not defaults)
- [ ] Model converged (check training logs)

### Evaluation
- [ ] Multiple metrics reported (R², MAE, RMSE)
- [ ] Clinical thresholds evaluated (AUC, sensitivity, specificity)
- [ ] SHAP values calculated and validated
- [ ] Feature importance aligns with domain knowledge
- [ ] Results robust to sensitivity analyses

### Reproducibility
- [ ] All code in version control
- [ ] requirements.txt or environment.yml updated
- [ ] Random seeds documented
- [ ] Analysis decisions logged
- [ ] Results saved with timestamps

### Reporting
- [ ] Methods described in sufficient detail for replication
- [ ] Results tables formatted for publication
- [ ] Figures have clear legends and labels
- [ ] Limitations discussed transparently
- [ ] Code and data availability statement included

---

## 15. Key References

### Methodological Guidelines
1. **STROBE:** von Elm et al. (2007). The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. PLoS Medicine.

2. **TRIPOD:** Collins et al. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). BMJ.

3. **RECORD:** Benchimol et al. (2015). The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement. PLoS Medicine.

### Missing Data
4. **Rubin's Framework:** Rubin, D.B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.

5. **MICE:** van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. Journal of Statistical Software, 45(3).

### Machine Learning
6. **SHAP:** Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.

7. **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016.

8. **CatBoost:** Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS 2018.

### Climate-Health
9. **WHO:** Smith et al. (2014). Human health: impacts, adaptation, and co-benefits. In IPCC Climate Change 2014: Impacts, Adaptation, and Vulnerability.

10. **Ebi et al.:** Ebi, K.L. et al. (2021). Hot weather and heat extremes: health risks. The Lancet, 398(10301), 698-708.

### Social Determinants
11. **Solar & Irwin:** Solar, O., & Irwin, A. (2010). A conceptual framework for action on the social determinants of health. WHO.

12. **Marmot Review:** Marmot, M. (2010). Fair society, healthy lives: The Marmot Review. London: UCL.

---

## 16. Next Steps

Now that we have this rigorous framework, we can proceed systematically:

### Step 1: Data Quality Assessment (Week 1, Days 1-2)
- Run comprehensive missingness analysis
- Validate GCRO-clinical data merge
- Create data quality report

### Step 2: Feature Engineering (Week 1, Days 3-5)
- Merge 14 new socioeconomic features
- Calculate VIF, check correlations
- Run automated leakage checks

### Step 3: Pilot Testing (Week 2, Days 1-3)
- Test on 3 biomarkers (high/medium/low performance)
- Validate imputation strategy
- Compare imputation methods (sensitivity)

### Step 4: Full Implementation (Week 2-3)
- All 7 systems, all biomarkers
- Hyperparameter optimization
- SHAP analysis

### Step 5: Validation & Reporting (Week 4)
- Statistical testing
- Sensitivity analyses
- Final report and figures

---

**This framework ensures our analysis meets the highest standards of rigor, reproducibility, and transparency expected in climate-health research.**
