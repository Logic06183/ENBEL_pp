# Model Refinement Plan - ENBEL Climate-Health Analysis
**Created:** 2025-10-30
**Branch:** feat/model-optimization
**Objective:** Improve biomarker prediction models while maintaining explanatory value of climate and socioeconomic variables

---

## Executive Summary

This plan outlines a comprehensive model refinement strategy to improve predictive performance across 19 biomarkers while:
- ✅ **Avoiding feature leakage** (no biomarker-to-biomarker prediction)
- ✅ **Expanding socioeconomic features** (from 1 to ~15 features)
- ✅ **Adding carefully selected health features** (non-leaking demographic/contextual variables)
- ✅ **Implementing CatBoost and interpretable models**
- ✅ **Using scientifically appropriate evaluation metrics** per physiological system
- ✅ **Grouping biomarkers by physiological systems**

---

## 1. Biomarker Groupings by Physiological System

### 1.1 Immune/Inflammatory System (n=2 biomarkers)
**Biomarkers:**
- CD4 cell count (cells/µL) - n=4,606
- HIV viral load (copies/mL)

**Safe Predictor Features:**
- ✅ All 16 climate features
- ✅ Expanded socioeconomic features (15 total)
- ✅ Age, Sex (demographic context)
- ✅ ART status (treatment context)
- ❌ NO other immune markers (WBC, lymphocytes, etc.)

**Evaluation Metrics:**
- R² (variance explained)
- RMSE (log-transformed for viral load)
- Spearman correlation (for ordinal relationships)
- Clinical thresholds: % correctly classified as <200, 200-500, >500 (CD4)

---

### 1.2 Hematological System (n=5 biomarkers)
**Biomarkers:**
- Hematocrit (%) - n=2,120 [**BEST PERFORMER: R²=0.937**]
- Hemoglobin (g/dL) - n=2,337
- Red blood cell count (×10⁶/µL)
- MCV (Mean Cell Volume)
- Platelet count (×10³/µL)

**Safe Predictor Features:**
- ✅ All 16 climate features
- ✅ Expanded socioeconomic features
- ✅ Age, Sex, BMI, Weight, Height
- ✅ ART status (affects hematology)
- ❌ NO cross-predictions within hematological panel (hemoglobin ↛ hematocrit)

**Leakage Mitigation Strategy:**
Train **separate models** for each biomarker using ONLY climate + socioeconomic + demographic features. Do NOT use any other hematological values.

**Evaluation Metrics:**
- R² (primary)
- MAE (clinically interpretable)
- % within clinical decision thresholds (anemia: Hb <12 g/dL women, <13 men)

---

### 1.3 Hepatic/Liver Function System (n=6 biomarkers)
**Biomarkers:**
- ALT (U/L) - n=1,250
- AST (U/L) - n=1,250
- Albumin (g/dL)
- Alkaline phosphatase (U/L)
- Total bilirubin (mg/dL)
- Total protein (g/dL)

**Safe Predictor Features:**
- ✅ All 16 climate features
- ✅ Expanded socioeconomic features
- ✅ Age, Sex, BMI
- ✅ ART status (hepatotoxicity)
- ❌ NO cross-predictions within liver panel (ALT ↛ AST)

**Evaluation Metrics:**
- R²
- MAE
- Log-error metrics (liver enzymes are right-skewed)
- Clinical threshold classification (ALT >40 U/L = abnormal)

---

### 1.4 Lipid/Cardiovascular System (n=5 biomarkers)
**Biomarkers:**
- FASTING HDL - n=2,918 [R²=0.334]
- FASTING LDL - n=2,917 [R²=0.377]
- FASTING TRIGLYCERIDES - n=972
- Total cholesterol (mg/dL) - n=2,917 [R²=0.392]
- HDL/LDL cholesterol (mg/dL) - n=710

**Safe Predictor Features:**
- ✅ All 16 climate features
- ✅ Expanded socioeconomic features
- ✅ Age, Sex, BMI, Waist circumference
- ❌ NO cross-predictions within lipid panel (LDL ↛ total cholesterol)

**Evaluation Metrics:**
- R²
- MAE
- Cardiovascular risk categories (Framingham scoring)
- AUC for binary classification (high cholesterol: >200 mg/dL)

---

### 1.5 Renal System (n=3 biomarkers)
**Biomarkers:**
- Creatinine (µmol/L) - n=1,247 [R²=0.137]
- Creatinine clearance - n=217
- Potassium (mEq/L)
- Sodium (mEq/L)

**Safe Predictor Features:**
- ✅ All 16 climate features
- ✅ Expanded socioeconomic features
- ✅ Age, Sex, BMI, Weight, Height
- ❌ NO cross-predictions within renal panel (creatinine ↛ creatinine clearance)

**Evaluation Metrics:**
- R²
- MAE
- eGFR classification accuracy (CKD stages 1-5)

---

### 1.6 Cardiovascular/Vital Signs System (n=4 biomarkers)
**Biomarkers:**
- Systolic BP (mmHg) - n=4,173
- Diastolic BP (mmHg) - n=4,173 [R²=0.096]
- Heart rate (bpm)
- Body temperature (celsius)

**Safe Predictor Features:**
- ✅ All 16 climate features (temperature expected to influence vitals!)
- ✅ Expanded socioeconomic features
- ✅ Age, Sex, BMI
- ⚠️ **CRITICAL LEAKAGE CONCERN:** Systolic ↔ Diastolic BP are highly correlated
  - **SOLUTION:** Train separate models, do NOT use systolic to predict diastolic or vice versa

**Evaluation Metrics:**
- R²
- MAE
- Hypertension classification (AUC for SBP ≥140 or DBP ≥90)
- Clinical guidelines: JNC8 criteria

---

### 1.7 Metabolic System (n=3 biomarkers)
**Biomarkers:**
- Fasting glucose (mmol/L) - n=2,722 [R²=0.050]
- BMI (kg/m²)
- Weight (kg) - n=285

**Safe Predictor Features:**
- ✅ All 16 climate features
- ✅ Expanded socioeconomic features
- ✅ Age, Sex, Height
- ❌ NO circular predictions (BMI ↛ Weight, Weight ↛ BMI)

**Evaluation Metrics:**
- R²
- MAE
- Diabetes classification (AUC for glucose ≥7.0 mmol/L fasting)
- Obesity classification (BMI categories)

---

## 2. Expanded Feature Space

### 2.1 Current Feature Set (20 features)
**Climate Features (16):**
- Temperature: daily mean/max/min, 7d/14d/30d means, anomalies
- Heat indices: stress index, percentile thresholds (p90/p95/p99)

**Socioeconomic Features (1):**
- HEAT_VULNERABILITY_SCORE (composite index)

**Temporal Features (3):**
- Month, Season indicators (Summer, Winter, Spring)

### 2.2 Proposed Expanded Feature Set (~35 features)

#### Additional Socioeconomic Features (14 new)
From GCRO dataset:
1. **Income Level** (q15_3_income_recode) - economic status
2. **Employment Status** (employment_status) - employment security
3. **Education Level** (std_education) - adaptive capacity
4. **Dwelling Type** (dwelling_type_enhanced) - housing quality
5. **Household Size** (Q1_03_households) - crowding
6. **Dwelling Satisfaction** (Q2_02_dwelling_dissatisfaction) - living conditions
7. **Drainage Quality** (Q2_14_Drainage) - infrastructure
8. **Age Group** (from Q11_03_age) - age vulnerability
9. **Economic Vulnerability Indicator** - composite
10. **Employment Vulnerability Indicator** - composite
11. **Education Adaptive Capacity** - composite
12. **Age Vulnerability Indicator** - composite
13. **Ward-level Density** (dwelling_count) - urbanization proxy
14. **Race** (std_race) - health equity marker

**Rationale:** These features capture socioeconomic determinants of health without creating biomarker leakage.

#### Additional Health Context Features (5 new)
From clinical dataset:
1. **Age (at enrolment)** - fundamental demographic
2. **Sex** - biological factor
3. **BMI** (for non-metabolic biomarkers) - body composition context
4. **ART Status** - treatment context (HIV-specific)
5. **Study Period** (year binned: 2002-2010, 2011-2015, 2016-2021) - temporal trends

**Leakage Prevention Rules:**
- ❌ BMI NOT used to predict Weight
- ❌ Weight/Height NOT used to predict BMI
- ❌ No biomarker-to-biomarker predictions
- ✅ Demographic/treatment context allowed

---

## 3. Model Algorithms

### 3.1 Current Models
- ✅ Random Forest
- ✅ XGBoost
- ✅ LightGBM

### 3.2 New Models to Add
1. **CatBoost** - handles categorical features natively, less tuning needed
2. **Elastic Net** - interpretable linear model with L1/L2 regularization
3. **Gradient Boosting (sklearn)** - baseline gradient boosting
4. **Extra Trees** - more randomness than Random Forest

### 3.3 Model Selection Strategy
For each biomarker:
- Train all 7 models with 5-fold cross-validation
- Hyperparameter optimization via Optuna (100 trials)
- Select best model based on primary metric
- Use SHAP for feature importance (all models)

---

## 4. Evaluation Metrics Framework

### 4.1 Regression Metrics (all biomarkers)
**Primary Metrics:**
- **R²** - variance explained (interpretability)
- **MAE** - clinical interpretability (same units as biomarker)
- **RMSE** - penalizes large errors

**Secondary Metrics:**
- **MAPE** - percentage error (for ratio interpretation)
- **Median Absolute Error** - robust to outliers
- **Max Error** - worst-case analysis

### 4.2 Classification Metrics (clinical thresholds)
For each biomarker with established clinical thresholds:

**Binary Classification:**
- **AUC-ROC** - discrimination ability
- **AUC-PR** - precision-recall (for imbalanced classes)
- **Sensitivity** - true positive rate
- **Specificity** - true negative rate
- **F1 Score** - harmonic mean of precision/recall

**Multi-class Classification:**
- **Accuracy** - overall correctness
- **Weighted F1** - accounts for class imbalance
- **Cohen's Kappa** - agreement beyond chance

### 4.3 System-Specific Metrics

| System | Primary Metric | Secondary Metric | Clinical Metric |
|--------|----------------|------------------|-----------------|
| Immune | R² | Spearman ρ | CD4 category accuracy |
| Hematological | R² | MAE | Anemia classification AUC |
| Hepatic | R² | Log-MAE | Elevated ALT/AST AUC |
| Lipid | R² | MAE | High cholesterol AUC |
| Renal | R² | MAE | eGFR stage accuracy |
| Cardiovascular | MAE | R² | Hypertension AUC |
| Metabolic | AUC | R² | Diabetes classification |

---

## 5. Implementation Strategy

### Phase 1: Feature Expansion (Week 1)
1. **Expand socioeconomic features**
   - Merge 14 new GCRO features
   - Handle missing values (multiple imputation)
   - Encode categorical variables (target encoding for high cardinality)

2. **Add health context features**
   - Age, Sex, BMI (where appropriate)
   - ART status
   - Study period indicators

3. **Feature validation**
   - Check correlations (avoid multicollinearity >0.95)
   - Validate no leakage (correlation matrix review)

### Phase 2: Model Implementation (Week 1-2)
1. **Add new models to pipeline**
   - CatBoost with categorical feature support
   - Elastic Net with standardized features
   - Extra Trees and Gradient Boosting

2. **Update hyperparameter search spaces**
   - CatBoost: depth, learning_rate, l2_leaf_reg, iterations
   - Elastic Net: alpha, l1_ratio
   - Extra Trees: n_estimators, max_features, min_samples_leaf

3. **Implement multi-metric evaluation**
   - Calculate all metrics per fold
   - Aggregate across folds (mean ± std)

### Phase 3: System-Specific Modeling (Week 2-3)
For each of 7 physiological systems:

1. **Select biomarkers**
2. **Define safe feature set** (with leakage checks)
3. **Train all 7 models** per biomarker
4. **Hyperparameter optimization** (Optuna, 100 trials)
5. **Calculate all evaluation metrics**
6. **Generate SHAP plots** for top model
7. **Save results** (models, metrics, plots)

### Phase 4: Analysis & Reporting (Week 3-4)
1. **Comparative analysis**
   - Best model per biomarker
   - Feature importance across systems
   - Climate vs socioeconomic contributions

2. **Performance improvement analysis**
   - Before/after comparison (old 20 features vs new ~35 features)
   - Statistical significance testing (paired t-test on CV scores)

3. **Documentation**
   - Model performance tables
   - Feature importance visualizations
   - Clinical interpretation guidelines

---

## 6. Expected Outcomes

### 6.1 Performance Targets
Based on current results, expected improvements:

| Biomarker Tier | Current R² | Target R² | Expected Δ |
|----------------|------------|-----------|------------|
| Excellent (>0.30) | 0.33-0.94 | 0.35-0.95 | +0.02-0.05 |
| Moderate (0.05-0.30) | 0.05-0.14 | 0.10-0.25 | +0.05-0.15 |
| Poor (<0.05) | -0.05-0.03 | 0.05-0.15 | +0.10-0.20 |

**Rationale:** Expanded socioeconomic features should most benefit biomarkers currently limited by insufficient context (immune, metabolic systems).

### 6.2 Feature Importance Insights
Expected findings:
- **Climate features:** Direct physiological effects (temperature → vitals, lipids)
- **Socioeconomic features:** Indirect effects via stress, access to care, nutrition
- **Demographics:** Age/sex confounding effects
- **Treatment context:** ART effects on hematology, liver function

### 6.3 Scientific Contributions
1. **Methodological:** Comprehensive leakage-free modeling framework
2. **Substantive:** Quantify socioeconomic vs climate contributions to health
3. **Clinical:** Identify high-risk populations (vulnerable to climate + SES)
4. **Policy:** Evidence for targeted interventions

---

## 7. Quality Assurance

### 7.1 Leakage Prevention Checklist
For EVERY biomarker model, verify:
- [ ] No other biomarkers in feature set
- [ ] No circular predictions (BMI ↔ Weight)
- [ ] No highly correlated features (r > 0.95)
- [ ] SHAP values validate expected relationships

### 7.2 Validation Framework
- **Cross-validation:** 5-fold stratified (by study and year)
- **Holdout set:** 20% for final evaluation
- **Temporal validation:** Train on 2002-2015, test on 2016-2021
- **Spatial validation:** Leave-one-ward-out (geography)

### 7.3 Reproducibility
- **Random seed:** 42 (all models)
- **Version control:** Git commits per system
- **Environment:** Docker container with pinned dependencies
- **Documentation:** README with exact commands

---

## 8. Implementation Checklist

### Data Preparation
- [ ] Load clinical dataset (11,398 records)
- [ ] Load GCRO dataset (58,616 records)
- [ ] Merge 14 new socioeconomic features
- [ ] Add 5 health context features
- [ ] Validate feature set (no leakage, correlation checks)
- [ ] Handle missing values (multiple imputation)

### Model Development
- [ ] Implement CatBoost pipeline
- [ ] Implement Elastic Net pipeline
- [ ] Implement Extra Trees pipeline
- [ ] Update hyperparameter search spaces
- [ ] Add multi-metric evaluation function

### System-Specific Modeling
- [ ] Immune/Inflammatory (2 biomarkers)
- [ ] Hematological (5 biomarkers)
- [ ] Hepatic (6 biomarkers)
- [ ] Lipid/Cardiovascular (5 biomarkers)
- [ ] Renal (3 biomarkers)
- [ ] Cardiovascular/Vitals (4 biomarkers)
- [ ] Metabolic (3 biomarkers)

### Analysis & Reporting
- [ ] Generate performance comparison tables
- [ ] Create feature importance visualizations
- [ ] Statistical significance testing
- [ ] Write final report
- [ ] Update manuscript with new results

---

## 9. File Structure

```
ENBEL_pp_model_refinement/
├── configs/
│   └── model_refinement_config.yaml          # New config with expanded features
├── scripts/
│   ├── pipelines/
│   │   └── system_specific_modeling.py       # New pipeline per system
│   ├── models/
│   │   ├── catboost_trainer.py              # CatBoost implementation
│   │   ├── elastic_net_trainer.py           # Elastic Net implementation
│   │   └── model_comparison.py              # Multi-model comparison
│   └── evaluation/
│       ├── multi_metric_evaluator.py        # All metrics calculation
│       └── leakage_checker.py               # Automated leakage detection
├── results/
│   ├── system_models/
│   │   ├── immune/                          # Per-system results
│   │   ├── hematological/
│   │   └── ...
│   └── comparative_analysis/
│       ├── before_after_comparison.csv
│       └── feature_importance_analysis.csv
└── docs/
    ├── MODEL_REFINEMENT_PLAN.md             # This document
    └── RESULTS_SUMMARY.md                   # Final results report
```

---

## 10. Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| Week 1 | Feature expansion, new model implementation | Updated dataset, CatBoost/ElasticNet/ExtraTrees code |
| Week 2 | System-specific modeling (Immune, Hematological, Hepatic) | 13 trained models, evaluation metrics |
| Week 3 | System-specific modeling (Lipid, Renal, Cardiovascular, Metabolic) | 15 trained models, SHAP plots |
| Week 4 | Comparative analysis, reporting, manuscript updates | Final report, updated manuscript |

---

## 11. Success Criteria

✅ **Technical:**
- All 28 biomarker models trained successfully
- No feature leakage detected (automated checks pass)
- SHAP values validate expected relationships
- Reproducible pipeline (Docker + seed control)

✅ **Performance:**
- Average R² improvement: +0.10 (moderate/poor biomarkers)
- At least 15/28 biomarkers show statistically significant improvement (p<0.05)
- Clinical metrics (AUC, sensitivity) ≥ 0.70 for key biomarkers

✅ **Scientific:**
- Clear quantification of climate vs socioeconomic contributions
- Interpretable feature importance rankings
- Actionable insights for public health interventions
- Publication-ready figures and tables

---

**Next Steps:** Begin Phase 1 (Feature Expansion) by merging GCRO socioeconomic features and validating the expanded feature set.
