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

## Publication-Quality Visualizations in R

### Overview

Climate-health research requires publication-quality visualizations that clearly communicate effect sizes, uncertainty, and study heterogeneity. Essential plot types include forest plots, box plots, and effect-response curves.

### Required R Packages

```r
library(ggplot2)      # Core plotting
library(forestplot)   # Forest plots
library(gridExtra)    # Multi-panel layouts
library(scales)       # Axis formatting
library(viridis)      # Color palettes
library(ggpubr)       # Publication themes
library(cowplot)      # Publication-ready layouts
```

### Forest Plots (Effect Sizes with CIs)

Forest plots are the gold standard for displaying effect sizes across multiple outcomes/studies with confidence intervals.

#### Basic Forest Plot

```r
library(forestplot)
library(dplyr)

# Prepare data
results <- data.frame(
  biomarker = c("Total Cholesterol", "Creatinine", "Glucose", "Hematocrit"),
  effect = c(0.345, 0.130, 0.090, 0.030),
  lower_ci = c(0.310, 0.095, 0.060, 0.010),
  upper_ci = c(0.380, 0.165, 0.120, 0.050),
  n = c(2917, 1247, 2722, 2120)
)

# Sort by effect size
results <- results %>% arrange(desc(effect))

# Create forest plot
forestplot(
  labeltext = cbind(
    c("Biomarker", results$biomarker),
    c("N", results$n),
    c("R²", sprintf("%.3f", results$effect))
  ),
  mean = c(NA, results$effect),
  lower = c(NA, results$lower_ci),
  upper = c(NA, results$upper_ci),
  title = "Climate-Biomarker Associations (Mixed Effects DLNM)",
  xlab = "R² (Variance Explained)",
  txt_gp = fpTxtGp(
    label = gpar(cex = 1.1),
    ticks = gpar(cex = 0.9),
    xlab = gpar(cex = 1.1)
  ),
  col = fpColors(box = "royalblue", line = "darkblue", summary = "red"),
  zero = 0,
  cex = 0.9,
  lineheight = "auto",
  boxsize = 0.25,
  graphwidth = unit(3, "inches")
)
```

#### Advanced Forest Plot with ggplot2

```r
library(ggplot2)

# Create forest plot with ggplot2 (more customizable)
ggplot(results, aes(x = effect, y = reorder(biomarker, effect))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = lower_ci, xmax = upper_ci), 
                 height = 0.2, color = "darkblue", size = 1) +
  geom_point(aes(size = n), color = "royalblue", shape = 18) +
  geom_text(aes(label = sprintf("%.3f", effect), x = upper_ci + 0.02),
            hjust = 0, size = 3.5) +
  scale_size_continuous(range = c(3, 8), name = "Sample Size") +
  labs(
    title = "Climate-Biomarker Associations",
    subtitle = "Mixed Effects DLNM Results (Corrected)",
    x = "R² (Variance Explained by Climate)",
    y = "Biomarker"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )
```

### Box Plots (Distribution by Study)

Box plots show data distribution across studies and identify potential outliers/heterogeneity.

#### Multi-Panel Box Plot

```r
library(ggplot2)
library(dplyr)

# Prepare data with study information
df <- data.frame(
  biomarker_value = c(...),
  study_id = c(...),
  biomarker_name = c(...)
)

# Create box plot
ggplot(df, aes(x = study_id, y = biomarker_value, fill = study_id)) +
  geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 2) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 0.5) +
  facet_wrap(~ biomarker_name, scales = "free_y", ncol = 2) +
  scale_fill_viridis_d(option = "plasma") +
  labs(
    title = "Biomarker Distributions by Study",
    x = "Study",
    y = "Biomarker Value",
    fill = "Study"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold", size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )
```

#### Violin Plot Alternative

```r
# Violin plot (shows distribution shape)
ggplot(df, aes(x = study_id, y = biomarker_value, fill = study_id)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.1, fill = "white", alpha = 0.8) +
  scale_fill_viridis_d(option = "viridis") +
  labs(title = "Biomarker Distribution by Study (Violin Plot)") +
  theme_minimal()
```

### Effect-Response Curves (DLNM Results)

Show non-linear temperature-biomarker relationships with confidence bands.

#### DLNM Exposure-Response Curve

```r
library(dlnm)
library(ggplot2)

# After fitting DLNM model
pred <- crosspred(cb_temp, model, at = seq(10, 30, 0.5), cen = 18)

# Extract predictions
df_pred <- data.frame(
  temperature = pred$predvar,
  effect = pred$allfit,
  lower_ci = pred$alllow,
  upper_ci = pred$allhigh
)

# Plot
ggplot(df_pred, aes(x = temperature, y = effect)) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci),
              fill = "skyblue", alpha = 0.4) +
  geom_line(color = "darkblue", size = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 18, linetype = "dotted", color = "gray40") +
  annotate("text", x = 18, y = max(df_pred$effect) * 0.9,
           label = "Reference (18°C)", hjust = -0.1, size = 3.5) +
  labs(
    title = "Temperature-Biomarker Association",
    subtitle = "Cumulative Effect (0-14 day lag)",
    x = "Temperature (°C)",
    y = "Effect on Biomarker (relative to 18°C)",
    caption = "Shaded area: 95% confidence interval"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))
```

### Multi-Panel Publication Figure

Combine multiple plots into a single publication-ready figure.

```r
library(cowplot)
library(gridExtra)

# Create individual plots
p1 <- forest_plot  # Forest plot
p2 <- box_plot     # Box plot
p3 <- effect_curve # DLNM curve
p4 <- heatmap      # Temperature-lag surface

# Combine with labels
figure <- plot_grid(
  p1, p2, p3, p4,
  labels = c("A", "B", "C", "D"),
  ncol = 2,
  rel_widths = c(1, 1),
  rel_heights = c(1, 1)
)

# Add title
title <- ggdraw() +
  draw_label(
    "Climate-Biomarker Associations: Mixed Effects DLNM Analysis",
    fontface = "bold",
    size = 16,
    x = 0.5,
    hjust = 0.5
  )

# Final figure
final_figure <- plot_grid(
  title,
  figure,
  ncol = 1,
  rel_heights = c(0.1, 1)
)

# Save
ggsave("publication_figure.pdf", final_figure,
       width = 12, height = 10, units = "in", dpi = 300)
```

### Publication-Quality Themes

#### Custom Theme

```r
theme_publication <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      # Text
      plot.title = element_text(face = "bold", size = rel(1.4), hjust = 0),
      plot.subtitle = element_text(size = rel(1.1), hjust = 0),
      axis.title = element_text(face = "bold", size = rel(1.1)),
      axis.text = element_text(size = rel(0.9)),
      
      # Grid
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_blank(),
      
      # Legend
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = rel(0.9)),
      
      # Strip (facet labels)
      strip.text = element_text(face = "bold", size = rel(1.05)),
      strip.background = element_rect(fill = "gray95", color = NA),
      
      # Margins
      plot.margin = margin(10, 10, 10, 10)
    )
}

# Use in plots
ggplot(...) + theme_publication()
```

### Color Palettes for Publications

```r
# Viridis (colorblind-friendly)
scale_fill_viridis_d(option = "plasma")  # Discrete
scale_color_viridis_c(option = "viridis") # Continuous

# Brewer (for categories)
scale_fill_brewer(palette = "Set2")

# Manual (custom)
scale_fill_manual(values = c(
  "Significant" = "#00BA38",
  "Not Significant" = "#F8766D"
))
```

### Statistical Annotations

```r
library(ggpubr)

# Add p-values to plots
ggplot(df, aes(x = group, y = value)) +
  geom_boxplot() +
  stat_compare_means(
    comparisons = list(c("Control", "Treatment")),
    method = "t.test",
    label = "p.signif"
  ) +
  stat_compare_means(label.y = 50, label.x = 1.5)
```

### Saving High-Quality Figures

```r
# PDF (vector, best for journals)
ggsave("figure1.pdf", plot, width = 8, height = 6, units = "in", dpi = 300)

# PNG (raster, for presentations)
ggsave("figure1.png", plot, width = 8, height = 6, units = "in", dpi = 300)

# TIFF (required by some journals)
ggsave("figure1.tiff", plot, width = 8, height = 6, units = "in", dpi = 300, compression = "lzw")
```

### Complete Example Workflow

```r
# 1. Load data and results
results <- read.csv("mixed_effects_dlnm_results.csv")

# 2. Create forest plot
p_forest <- create_forest_plot(results)

# 3. Create box plots
p_boxes <- create_box_plots(data)

# 4. Create DLNM curves
p_dlnm <- create_dlnm_curves(model_results)

# 5. Combine into figure
figure <- plot_grid(
  p_forest, p_boxes, p_dlnm,
  labels = "AUTO",
  ncol = 2
)

# 6. Save
ggsave("Figure1_ClimateHealth.pdf", figure,
       width = 12, height = 8, units = "in", dpi = 300)
```

### Best Practices

1. **Resolution**: Always use dpi = 300 or higher for publications
2. **Size**: Check journal requirements (usually 8-12 inches wide)
3. **Colors**: Use colorblind-friendly palettes (viridis)
4. **Text**: Make sure all text is readable at final size
5. **Legends**: Clear and positioned appropriately
6. **Error bars**: Always include confidence intervals
7. **Reference lines**: Add for clinical thresholds or null effects
8. **Annotations**: Label important features (p-values, thresholds)

### Common Pitfalls to Avoid

❌ Too many colors (limit to 5-7)
❌ 3D plots (hard to interpret)
❌ Pie charts (use bar charts instead)
❌ Dual y-axes (confusing)
❌ Default ggplot2 theme (not publication-quality)
❌ Low resolution (use dpi ≥ 300)
❌ Missing error bars
❌ Inconsistent font sizes

✅ Simple, clear designs
✅ Consistent color schemes
✅ Clear axis labels with units
✅ Confidence intervals shown
✅ Professional themes
✅ High resolution
✅ Colorblind-friendly
✅ Accessible text sizes

