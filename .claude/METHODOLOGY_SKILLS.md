# Methodology Development & Causal Inference Skills

This worktree focuses on developing rigorous statistical methodologies and causal inference frameworks for climate-health research.

## Core Focus Areas

### 1. Causal Inference Methods
- **Distributed Lag Non-Linear Models (DLNM)**: Temperature-lag-response relationships
- **Case-Crossover Design**: Within-person control for time-invariant confounders
- **Time-Stratified Matching**: Temporal control for seasonality
- **Propensity Score Methods**: Observational study adjustment
- **Instrumental Variables**: Addressing unmeasured confounding

### 2. Statistical Frameworks
- **Mixed Effects Models**: Hierarchical data structures (patients nested in studies)
- **Generalized Additive Models (GAM)**: Non-linear relationships with smoothing
- **Structural Equation Modeling (SEM)**: Mediation and path analysis
- **Survival Analysis**: Time-to-event outcomes
- **Bayesian Methods**: Uncertainty quantification, prior incorporation

### 3. Validation & Robustness
- **Cross-Validation**: Time-series aware, stratified, nested
- **Sensitivity Analysis**: Assess robustness to assumptions
- **Multiple Testing Correction**: Bonferroni, FDR, permutation tests
- **Bootstrap Confidence Intervals**: Non-parametric uncertainty
- **Leave-One-Out Analysis**: Study-level sensitivity

## Key Commands for Methodology Work

### DLNM Analysis (R)
```bash
# Main DLNM validation pipeline
Rscript R/dlnm_validation/dlnm_validation_pipeline.R

# Case-crossover DLNM
Rscript R/case_crossover_dlnm_validation.R

# Continuous outcome DLNM (not case-crossover)
Rscript R/dlnm_continuous_outcomes.R

# Generate DLNM plots
Rscript R/create_dlnm_plots.R
```

### GAM Analysis (R)
```bash
# Temperature-biomarker GAM with smoothing
Rscript R/gam_analysis/temperature_biomarker_gam.R

# Mixed GAM with random effects
Rscript R/gam_analysis/mixed_gam_analysis.R
```

### Python Statistical Methods
```bash
# Mixed effects models
python scripts/methodology/mixed_effects_analysis.py

# Propensity score matching
python scripts/methodology/propensity_score_matching.py

# Mediation analysis
python scripts/methodology/mediation_analysis.py
```

## DLNM Framework (Primary Method)

### What is DLNM?
Distributed Lag Non-Linear Models capture:
1. **Non-linearity**: Temperature effects may be non-linear (U-shaped, threshold)
2. **Delayed effects**: Biomarker changes may occur days after exposure
3. **Lag structure**: Effects may accumulate or dissipate over time

### R Implementation
```r
library(dlnm)
library(mgcv)
library(splines)

# Create crossbasis (temperature × lag interaction)
cb_temp <- crossbasis(
  temperature,
  lag = 21,                    # 0-21 day lag
  argvar = list(fun = "ns", df = 4),   # Non-linear temperature (4 df)
  arglag = list(fun = "ns", df = 4)    # Non-linear lag structure (4 df)
)

# Fit model
model <- gam(
  biomarker ~ cb_temp + study_id + season,
  data = df,
  family = gaussian()
)

# Predict cumulative effects
pred <- crosspred(cb_temp, model, at = seq(10, 35, 1))

# Visualize 3D surface
plot(pred, "3d", theta = 40, phi = 30)
```

### Case-Crossover Design
```r
library(gnm)

# Time-stratified matching (28-day strata, same day-of-week)
df$stratum <- interaction(
  format(df$date, "%Y-%m"),  # Year-month
  weekdays(df$date)           # Day of week
)

# Create binary outcome (above/below median)
df$outcome_binary <- as.integer(df$biomarker > median(df$biomarker))

# Fit conditional logistic regression
model <- gnm(
  outcome_binary ~ cb_temp,
  data = df,
  family = binomial(link = "logit"),
  eliminate = stratum  # Condition on strata
)
```

### Key DLNM Parameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `lag` | 0-21 days | Biological response time |
| `argvar df` | 3-5 | Temperature non-linearity flexibility |
| `arglag df` | 3-5 | Lag structure flexibility |
| Reference temp | 18-22°C | Minimum mortality temperature for SA |

## Case-Crossover Design

### Advantages
- Controls for **all time-invariant confounders** (age, sex, genetics, SES)
- Each individual is their own control
- No unmeasured between-person confounding

### Limitations
- Requires within-person variability
- Binary outcomes lose information (vs continuous)
- Limited power for rare outcomes

### When to Use
✅ **Use case-crossover for:**
- Acute effects (days to weeks)
- Outcomes with temporal variability
- Strong unmeasured confounding suspected

❌ **Don't use case-crossover for:**
- Chronic effects (months to years)
- Outcomes that rarely change within person
- When continuous outcomes are important

## Mixed Effects Models

### Random Effects Structure
```python
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Random intercept by study
model = mixedlm(
    "biomarker ~ temp_mean + season + HEAT_VULNERABILITY_SCORE",
    data=df,
    groups=df["study_id"],
    re_formula="1"  # Random intercept only
)

# Random slope for temperature
model = mixedlm(
    "biomarker ~ temp_mean + season + HEAT_VULNERABILITY_SCORE",
    data=df,
    groups=df["study_id"],
    re_formula="~temp_mean"  # Random intercept + slope
)
```

### Interpreting Results
- **Fixed effects**: Population-average associations
- **Random effects**: Between-study heterogeneity
- **ICC**: Intraclass correlation (study-level clustering)

## Mediation Analysis

### Conceptual Model
```
Temperature → Socioeconomic Vulnerability → Biomarker
              (mediator)
```

### Implementation
```python
from statsmodels.stats.mediation import Mediation

# Step 1: Exposure → Outcome (total effect)
model_total = sm.OLS.from_formula(
    "biomarker ~ temp_mean",
    data=df
).fit()

# Step 2: Exposure → Mediator
model_med = sm.OLS.from_formula(
    "HEAT_VULNERABILITY_SCORE ~ temp_mean",
    data=df
).fit()

# Step 3: Exposure + Mediator → Outcome
model_outcome = sm.OLS.from_formula(
    "biomarker ~ temp_mean + HEAT_VULNERABILITY_SCORE",
    data=df
).fit()

# Mediation analysis
med = Mediation(
    model_outcome,
    model_med,
    exposure='temp_mean',
    mediator='HEAT_VULNERABILITY_SCORE'
).fit()

print(med.summary())
```

### Interpretation
- **Direct effect**: Temperature → Biomarker (not through vulnerability)
- **Indirect effect**: Temperature → Vulnerability → Biomarker
- **Total effect**: Direct + Indirect
- **Proportion mediated**: Indirect / Total

## Sensitivity Analysis

### 1. Leave-One-Out Study Analysis
```python
studies = df['study_id'].unique()

results = []
for study in studies:
    # Exclude one study
    df_loo = df[df['study_id'] != study]

    # Refit model
    model = train_model(df_loo)
    r2 = evaluate_model(model, df_loo)

    results.append({
        'excluded_study': study,
        'r2': r2,
        'n_patients': len(df_loo)
    })

# Check if any single study drives results
print(pd.DataFrame(results))
```

### 2. Alternative Feature Sets
```python
feature_sets = {
    'climate_only': ['temp_mean', 'temp_max', 'temp_min'],
    'socioeconomic_only': ['HEAT_VULNERABILITY_SCORE'],
    'temporal_only': ['month', 'season'],
    'full_model': all_features
}

for name, features in feature_sets.items():
    model = train_model(X[features], y)
    print(f"{name}: R² = {model.score(X_test[features], y_test):.3f}")
```

### 3. Alternative Lag Specifications
```r
# Test different lag structures
lag_options <- c(7, 14, 21, 30)

for (lag in lag_options) {
  cb_temp <- crossbasis(
    temperature,
    lag = lag,
    argvar = list(fun = "ns", df = 4),
    arglag = list(fun = "ns", df = 4)
  )

  model <- gam(biomarker ~ cb_temp, data = df)
  aic <- AIC(model)

  cat(sprintf("Lag %d days: AIC = %.2f\n", lag, aic))
}
```

## Multiple Testing Correction

### Bonferroni Correction
```python
n_biomarkers = 19
alpha = 0.05
alpha_corrected = alpha / n_biomarkers  # 0.0026

# Apply threshold
significant = p_values < alpha_corrected
```

### False Discovery Rate (FDR)
```python
from statsmodels.stats.multitest import multipletests

# Apply FDR correction
reject, pvals_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'  # Benjamini-Hochberg
)
```

### Permutation Tests
```python
def permutation_test(X, y, model, n_permutations=1000):
    # Actual test statistic
    model.fit(X, y)
    actual_r2 = model.score(X, y)

    # Permutation distribution
    perm_r2 = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        model.fit(X, y_perm)
        perm_r2.append(model.score(X, y_perm))

    # P-value
    p_value = np.mean(np.array(perm_r2) >= actual_r2)
    return actual_r2, p_value
```

## Time-Series Cross-Validation

### Implementation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train on past, test on future
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

### Stratified by Study
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, df['study_id']):
    # Ensure each fold has all studies represented
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

## Bootstrapping for Confidence Intervals

### Implementation
```python
from sklearn.utils import resample

def bootstrap_ci(X, y, model, n_bootstrap=1000, ci=0.95):
    scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        X_boot, y_boot = resample(X, y, random_state=None)

        # Train and evaluate
        model.fit(X_boot, y_boot)
        score = model.score(X_boot, y_boot)
        scores.append(score)

    # Calculate CI
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return lower, upper

# Usage
lower, upper = bootstrap_ci(X, y, RandomForestRegressor(), n_bootstrap=1000)
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
```

## Power Analysis

### Sample Size Requirements
```python
from statsmodels.stats.power import TTestIndPower

# Calculate required sample size
analysis = TTestIndPower()
sample_size = analysis.solve_power(
    effect_size=0.3,   # Cohen's d
    power=0.8,         # 80% power
    alpha=0.05         # 5% significance
)

print(f"Required sample size: {sample_size:.0f} per group")
```

### Post-Hoc Power
```python
def calculate_power(n, r2, n_predictors, alpha=0.05):
    """Calculate power for regression analysis"""
    from scipy import stats

    # Calculate F-statistic
    f_stat = (r2 / n_predictors) / ((1 - r2) / (n - n_predictors - 1))

    # Critical value
    f_crit = stats.f.ppf(1 - alpha, n_predictors, n - n_predictors - 1)

    # Power
    ncp = f_stat * n_predictors  # Non-centrality parameter
    power = 1 - stats.ncf.cdf(f_crit, n_predictors, n - n_predictors - 1, ncp)

    return power
```

## Best Practices

### Documenting Methodology
- State all assumptions explicitly
- Report model diagnostics (residuals, Q-Q plots)
- Include sensitivity analyses
- Document software versions
- Provide reproducible code

### Reporting Standards
Follow STROBE (Strengthening the Reporting of Observational Studies in Epidemiology):
- Study design clearly stated
- Setting and dates of data collection
- Eligibility criteria and sources
- Handling of missing data
- Statistical methods with justification
- Sensitivity analyses

### Common Pitfalls
1. **Temporal Mismatch**: Aligning climate exposure to biomarker measurement dates
2. **Autocorrelation**: Accounting for temporal correlation in time-series data
3. **Multiple Testing**: Correcting for testing 19 biomarkers
4. **Overfitting**: Using cross-validation, not just train-test split
5. **Confounding**: Considering socioeconomic, seasonal, study-level confounders

## Integration with ML Models

### Two-Stage Workflow
**Stage 1: ML Screening**
- Identify potentially sensitive biomarkers
- Fast, hypothesis-generating
- High-dimensional feature space

**Stage 2: Causal Validation**
- DLNM for identified biomarkers
- Rigorous, confirmatory
- Addresses confounding and temporality

### Example Workflow
```python
# Stage 1: ML screening
ml_results = []
for biomarker in biomarkers:
    model = train_ml_model(biomarker)
    r2 = evaluate_model(model)
    ml_results.append({'biomarker': biomarker, 'r2': r2})

# Identify high-performers
candidates = [b for b in ml_results if b['r2'] > 0.2]

# Stage 2: DLNM validation
for biomarker in candidates:
    # Run DLNM in R
    subprocess.run([
        'Rscript', 'R/dlnm_validation.R',
        '--biomarker', biomarker
    ])
```

## Code Organization

Place methodology code in:
- `R/dlnm_validation/` - DLNM analysis scripts
- `R/gam_analysis/` - GAM models
- `scripts/methodology/` - Python statistical methods
- `scripts/sensitivity/` - Sensitivity analyses
- `scripts/power/` - Power calculations

## Git Workflow

```bash
# Start new methodology work
git checkout -b exp/dlnm-continuous-outcomes

# Develop and test
Rscript R/dlnm_continuous_outcomes.R

# Commit with clear methodology description
git add .
git commit -m "feat: implement continuous outcome DLNM for hematocrit validation"

# Merge when validated
git checkout main
git merge exp/dlnm-continuous-outcomes
```

## Success Metrics

### Methodological Rigor
- All models include appropriate controls
- Sensitivity analyses performed
- Multiple testing corrections applied
- Assumptions validated

### Causal Evidence
- DLNM validates ML findings
- Exposure-response relationships plausible
- Temporal patterns consistent with biology
- Confounding adequately addressed

### Reproducibility
- Complete R/Python scripts available
- Package versions documented
- Results stable across runs
- Methods clearly described
