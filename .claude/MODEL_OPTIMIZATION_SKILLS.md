# Model Optimization & Refinement Skills

This worktree is dedicated to improving ML model performance and expanding explainability through systematic refinement.

## Core Focus Areas

### 1. Model Performance Enhancement
- **Hyperparameter Optimization**: Systematic tuning using Optuna, GridSearchCV, RandomizedSearchCV
- **Feature Engineering**: Creating informative features from climate, temporal, and socioeconomic data
- **Ensemble Methods**: Stacking, blending, weighted averaging of models
- **Cross-Validation Strategies**: Stratified, time-series aware, nested CV

### 2. Evaluation Metrics
- **Regression Metrics**: R², RMSE, MAE, MAPE, adjusted R²
- **Model Diagnostics**: Residual analysis, learning curves, validation curves
- **Statistical Testing**: Bonferroni correction, permutation tests, DeLong's test
- **Confidence Intervals**: Bootstrap, analytical CIs for metrics

### 3. Explainability (SHAP/XAI)
- **SHAP Analysis**: Feature importance, waterfall, beeswarm, dependence plots
- **Feature Interactions**: SHAP interaction values, 2D dependence plots
- **Global vs Local Explanations**: Understanding model behavior at different scales
- **Visualization Best Practices**: Publication-quality SHAP plots

## Key Commands for Model Work

### Running Model Experiments
```bash
# Run refined analysis pipeline with all features
python scripts/pipelines/refined_analysis_pipeline_FIXED.py

# Run specific biomarker analysis
python -m enbel_pp.pipeline --target "CD4 cell count (cells/µL)" --mode state_of_the_art

# Hyperparameter optimization for specific model
python scripts/optimization/tune_hyperparameters.py --biomarker "CD4" --model "lightgbm"
```

### Model Evaluation & Comparison
```bash
# Generate comprehensive evaluation report
python scripts/evaluation/evaluate_all_models.py

# Compare multiple models
python scripts/evaluation/model_comparison.py --biomarkers "CD4,Glucose,Hemoglobin"

# Cross-validation analysis
python scripts/evaluation/cross_validation_analysis.py --folds 10 --stratify study_id
```

### SHAP Analysis
```bash
# Generate SHAP plots for all biomarkers
python generate_real_shap_plots.py

# Create publication-quality SHAP visualizations
python create_shap_waterfall_plots.py

# SHAP feature interaction analysis
python scripts/xai/shap_interaction_analysis.py --biomarker "CD4"
```

## Model Development Workflow

### 1. Baseline Establishment
```python
from enbel_pp.pipeline import ClimateHealthPipeline

# Run simple baseline
pipeline = ClimateHealthPipeline(analysis_mode='simple')
baseline_results = pipeline.run(target_biomarker='CD4 cell count (cells/µL)')
```

### 2. Feature Engineering
```python
from enbel_pp.ml_utils import prepare_features_safely

# Create advanced features
features = prepare_features_safely(
    df,
    climate_cols=['temp_mean', 'temp_max', 'temp_min'],
    lag_windows=[7, 14, 30],  # Multiple lag periods
    rolling_stats=['mean', 'std', 'max', 'min'],
    create_interactions=True,  # Temperature × vulnerability
    create_anomalies=True      # Deviation from normal
)
```

### 3. Hyperparameter Optimization
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }

    model = LGBMRegressor(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 4. Model Validation
```python
from sklearn.model_selection import cross_validate

# Comprehensive cross-validation
cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
    return_train_score=True,
    return_estimator=True
)

# Check for overfitting
train_r2 = cv_results['train_r2'].mean()
test_r2 = cv_results['test_r2'].mean()
overfit_gap = train_r2 - test_r2
print(f"Overfitting gap: {overfit_gap:.3f}")
```

### 5. SHAP Analysis
```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Generate visualizations
shap.plots.waterfall(shap_values[0])  # Single prediction
shap.plots.beeswarm(shap_values)      # Global importance
shap.plots.dependence(
    ind='temp_mean_7d',
    shap_values=shap_values,
    features=X_test,
    interaction_index='HEAT_VULNERABILITY_SCORE'
)
```

## Performance Benchmarks

Current best results (as of latest analysis):

| Biomarker | R² | RMSE | Best Model | Top Feature |
|-----------|-----|------|------------|-------------|
| Hematocrit | 0.937 | 2.51% | RandomForest | HEAT_VULNERABILITY |
| Total Cholesterol | 0.392 | 0.89 mmol/L | RandomForest | HEAT_VULNERABILITY |
| FASTING LDL | 0.377 | 0.75 mmol/L | XGBoost | climate_temp_30d_mean |
| FASTING HDL | 0.334 | 0.29 mmol/L | LightGBM | climate_temp_7d_mean |

**Target**: Improve R² by 0.05-0.15 through better features and optimization.

## Feature Engineering Priorities

### High-Impact Features to Develop
1. **Vulnerability × Temperature Interactions**: `HEAT_VULNERABILITY_SCORE * temp_anomaly`
2. **Cumulative Heat Exposure**: `sum(max(temp - 25°C, 0))` over 30 days
3. **Temperature Variability**: Standard deviation of temperature over lag windows
4. **Extreme Event Indicators**: Binary flags for heat waves (3+ days >30°C)
5. **Seasonal Adjustments**: Normalized by seasonal baseline

### Data Leakage Prevention
- ✅ Use ONLY climate, socioeconomic, and temporal features
- ❌ NEVER use other biomarkers as features
- ✅ Verify with `scripts/verification/verify_feature_selection.py`
- ✅ Document feature whitelist in code

### Temporal Considerations
- Account for lag effects (0-30 days)
- Consider delayed biological responses
- Use time-stratified cross-validation for temporal data

## Model Optimization Strategies

### 1. Algorithm Selection
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = {
    'rf': RandomForestRegressor(random_state=42),
    'xgb': XGBRegressor(random_state=42),
    'lgb': LGBMRegressor(random_state=42),
    'cat': CatBoostRegressor(random_state=42, verbose=False),
    'gbm': GradientBoostingRegressor(random_state=42)
}
```

### 2. Regularization
- L1 (Lasso): Feature selection, sparsity
- L2 (Ridge): Prevent overfitting, handle multicollinearity
- ElasticNet: Combination of L1 and L2
- Early stopping: Prevent overfitting in boosting algorithms

### 3. Ensemble Methods
```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Voting ensemble
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor()),
    ('xgb', XGBRegressor()),
    ('lgb', LGBMRegressor())
])

# Stacking ensemble
ensemble = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor()),
        ('xgb', XGBRegressor())
    ],
    final_estimator=LGBMRegressor()
)
```

## Debugging Model Performance

### Common Issues
1. **Low R² (<0.1)**:
   - Check sample size (need >1000 for good performance)
   - Verify feature quality (no NaNs, proper scaling)
   - Consider non-linear relationships (use tree-based models)
   - Investigate if biomarker is truly climate-sensitive

2. **High Overfitting (train-test gap >0.1)**:
   - Reduce model complexity (max_depth, n_estimators)
   - Increase regularization
   - Add more training data
   - Use cross-validation

3. **SHAP Values Don't Match Expectations**:
   - Check for feature correlations
   - Verify feature engineering logic
   - Ensure proper baseline (expected value)
   - Consider feature interactions

### Diagnostic Tools
```python
# Learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Validation curve
from sklearn.model_selection import validation_curve

train_scores, test_scores = validation_curve(
    model, X, y,
    param_name='max_depth',
    param_range=[3, 5, 7, 9, 11, 13, 15],
    cv=5, scoring='r2'
)
```

## Best Practices

### Reproducibility
```python
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # For scikit-learn models, pass random_state=seed
    # For tensorflow/pytorch, set additional seeds
```

### Memory Efficiency
- Use `dtype=np.float32` for large datasets
- Process biomarkers sequentially, not all at once
- Cache intermediate results in `cache/` directory
- Use `gc.collect()` after processing each biomarker

### Performance Tracking
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    })
    mlflow.sklearn.log_model(model, 'model')
```

## Code Organization

Place new model code in:
- `scripts/pipelines/` - New analysis pipelines
- `scripts/optimization/` - Hyperparameter tuning scripts
- `scripts/evaluation/` - Model evaluation and comparison
- `scripts/xai/` - SHAP and explainability analysis
- `src/enbel_pp/` - Reusable utilities and classes

## Git Workflow

```bash
# Start work on new optimization
git checkout -b exp/optimize-cd4-model

# Make changes, test locally
python scripts/pipelines/refined_analysis_pipeline_FIXED.py

# Commit with descriptive message
git add .
git commit -m "feat: improve CD4 model R² from 0.70 to 0.85 with interaction features"

# When ready, merge to main
git checkout main
git merge exp/optimize-cd4-model
```

## Success Metrics

### Model Performance Goals
- **Excellent tier** (R² > 0.30): Maintain or improve
- **Moderate tier** (R² 0.05-0.30): Improve to excellent tier
- **Poor tier** (R² < 0.05): Achieve at least moderate performance

### Explainability Goals
- Identify top 5 features for each biomarker
- Quantify feature importance contributions
- Detect and characterize non-linear relationships
- Validate biological plausibility of findings

### Publication Readiness
- All models have comprehensive documentation
- SHAP plots are publication-quality
- Methods are fully reproducible
- Results align with domain knowledge
