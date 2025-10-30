# Model Refinement Setup - Summary Report
**Date:** 2025-10-30
**Branch:** feat/model-optimization
**Status:** ✅ Planning Phase Complete

---

## Overview

This document summarizes the comprehensive planning and setup completed for the ENBEL Climate-Health model refinement initiative. The goal is to improve biomarker prediction performance while maintaining the explanatory value of climate and socioeconomic variables.

---

## What Was Accomplished

### 1. Comprehensive Planning Document
**File:** `MODEL_REFINEMENT_PLAN.md`

Created a 15-page detailed plan covering:
- ✅ **7 Physiological System Groupings** with 28 biomarkers
- ✅ **Leakage Prevention Strategy** for each system
- ✅ **Expanded Feature Space** (20 → ~35 features)
- ✅ **7 ML Algorithms** (added CatBoost, Elastic Net, Extra Trees, Gradient Boosting)
- ✅ **System-Specific Evaluation Metrics** (R², MAE, AUC, clinical thresholds)
- ✅ **4-Phase Implementation Strategy** (weeks 1-4)
- ✅ **Expected Performance Targets** and success criteria

### 2. Automated Leakage Detection Utility
**File:** `scripts/evaluation/leakage_checker.py`

Implemented a robust feature leakage detection system that:
- ✅ **Detects biomarker-to-biomarker predictions** (e.g., hemoglobin → hematocrit)
- ✅ **Identifies circular predictions** (e.g., BMI ↔ Weight)
- ✅ **Flags high correlation features** (r > 0.95)
- ✅ **Generates safe feature sets** automatically
- ✅ **Produces detailed reports** with color-coded warnings

**Test Results:**
```
Example 1: UNSAFE feature set - ✅ Detected hemoglobin leakage
Example 2: SAFE feature set - ✅ Passed all checks
Example 3: Auto-generation - ✅ Successfully filtered unsafe features
```

### 3. Enhanced Configuration File
**File:** `configs/model_refinement_config.yaml`

Created a comprehensive YAML configuration with:
- ✅ **7 Physiological System Definitions** with biomarkers and metrics
- ✅ **35 Features** organized by category (climate, socioeconomic, temporal, demographic)
- ✅ **7 Model Configurations** with hyperparameter search spaces
- ✅ **Multiple Cross-Validation Strategies** (temporal, spatial, stratified)
- ✅ **Leakage Prevention Settings** with system-specific exclusions
- ✅ **Comprehensive Evaluation Metrics** (regression + classification)
- ✅ **Explainable AI Configuration** (SHAP, permutation importance)

---

## Key Improvements Over Current Approach

### Feature Space Expansion
| Category | Current | Proposed | Increase |
|----------|---------|----------|----------|
| **Climate Features** | 16 | 16 | +0 |
| **Socioeconomic Features** | 1 | 15 | +1,400% |
| **Temporal Features** | 3 | 3 | +0 |
| **Demographic Features** | 0 | 5 | +5 |
| **TOTAL** | **20** | **~35-39** | **+75-95%** |

### Model Algorithm Expansion
| Model | Current | Proposed |
|-------|---------|----------|
| Random Forest | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| CatBoost | ❌ | ✅ NEW |
| Elastic Net | ❌ | ✅ NEW |
| Extra Trees | ❌ | ✅ NEW |
| Gradient Boosting | ❌ | ✅ NEW |

### Evaluation Metrics Enhancement
| System | Primary Metric | Added Clinical Metrics |
|--------|----------------|------------------------|
| Immune | R² | Spearman ρ, CD4 category accuracy |
| Hematological | R² | MAE, Anemia classification AUC |
| Hepatic | R² | Log-MAE, Elevated enzyme AUC |
| Lipid | R² | MAE, High cholesterol AUC |
| Renal | R² | MAE, eGFR stage accuracy |
| Cardiovascular | MAE | R², Hypertension AUC |
| Metabolic | R² | AUC, Diabetes classification |

---

## Physiological System Groupings

### 1. Immune/Inflammatory System
**Biomarkers (n=2):**
- CD4 cell count (n=4,606) - Current R² = -0.004
- HIV viral load

**Safe Features:** Climate (16) + Socioeconomic (15) + Age, Sex, ART Status

**Expected Improvement:** +0.15-0.25 R² (better socioeconomic context)

---

### 2. Hematological System
**Biomarkers (n=5):**
- **Hematocrit (n=2,120)** - Current R² = 0.937 ✅ BEST PERFORMER
- Hemoglobin (n=2,337) - Current R² = -0.032
- RBC count, MCV, Platelet count

**Critical Leakage Prevention:**
- ❌ NO hemoglobin → hematocrit prediction
- ❌ NO cross-predictions within panel

**Expected Improvement:** +0.02-0.05 R² (already high performance)

---

### 3. Hepatic/Liver Function System
**Biomarkers (n=6):**
- ALT (n=1,250) - Current R² = -0.042
- AST (n=1,250) - Current R² = -0.017
- Albumin, Alkaline phosphatase, Total bilirubin, Total protein

**Critical Leakage Prevention:**
- ❌ NO ALT → AST prediction

**Expected Improvement:** +0.10-0.20 R² (currently poor, need better context)

---

### 4. Lipid/Cardiovascular System
**Biomarkers (n=5):**
- **Total Cholesterol (n=2,917)** - Current R² = 0.392 ✅ EXCELLENT
- **FASTING HDL (n=2,918)** - Current R² = 0.334 ✅ EXCELLENT
- **FASTING LDL (n=2,917)** - Current R² = 0.377 ✅ EXCELLENT
- HDL cholesterol (n=710) - Current R² = 0.072
- LDL cholesterol (n=710) - Current R² = 0.143

**Expected Improvement:** +0.05-0.10 R² (already good, refinement)

---

### 5. Renal System
**Biomarkers (n=3):**
- Creatinine (n=1,247) - Current R² = 0.137
- Creatinine clearance (n=217) - Current R² = -0.053
- Potassium, Sodium

**Critical Leakage Prevention:**
- ❌ NO creatinine → creatinine clearance prediction

**Expected Improvement:** +0.10-0.15 R²

---

### 6. Cardiovascular/Vital Signs System
**Biomarkers (n=4):**
- Systolic BP (n=4,173) - Current R² = -0.001
- Diastolic BP (n=4,173) - Current R² = 0.096
- Heart rate, Body temperature

**Critical Leakage Prevention:**
- ❌ NO systolic → diastolic BP prediction

**Expected Improvement:** +0.10-0.20 R² (climate should directly influence vitals)

---

### 7. Metabolic System
**Biomarkers (n=3):**
- Fasting glucose (n=2,722) - Current R² = 0.050
- BMI, Weight (n=285) - Current R² = 0.028

**Critical Leakage Prevention:**
- ❌ NO BMI ↔ Weight circular predictions

**Expected Improvement:** +0.10-0.25 R² (socioeconomic factors important)

---

## Expanded Socioeconomic Features (14 New Features)

Currently using only **1 feature** (HEAT_VULNERABILITY_SCORE). Adding:

1. **Income Level** (q15_3_income_recode) - Economic status
2. **Employment Status** (employment_status) - Job security
3. **Education Level** (std_education) - Adaptive capacity
4. **Dwelling Type** (dwelling_type_enhanced) - Housing quality
5. **Household Size** (Q1_03_households) - Crowding
6. **Dwelling Satisfaction** (Q2_02_dwelling_dissatisfaction)
7. **Drainage Quality** (Q2_14_Drainage) - Infrastructure
8. **Age Group** (Q11_03_age) - Age vulnerability
9. **Economic Vulnerability Indicator** - Composite
10. **Employment Vulnerability Indicator** - Composite
11. **Education Adaptive Capacity** - Composite
12. **Age Vulnerability Indicator** - Composite
13. **Ward-level Density** (dwelling_count) - Urbanization
14. **Race** (std_race) - Health equity marker

**Rationale:** Socioeconomic determinants of health are critical for understanding biomarker variability beyond climate effects.

---

## Implementation Roadmap

### Phase 1: Feature Expansion (Week 1)
**Tasks:**
1. Merge 14 new GCRO socioeconomic features
2. Add 5 demographic/health context features
3. Validate feature correlations (no multicollinearity >0.95)
4. Run automated leakage checks

**Deliverables:**
- Expanded dataset with ~35 features per biomarker
- Feature correlation matrix
- Leakage validation report

---

### Phase 2: Model Implementation (Week 1-2)
**Tasks:**
1. Implement CatBoost pipeline
2. Implement Elastic Net pipeline
3. Implement Extra Trees and Gradient Boosting pipelines
4. Update hyperparameter search spaces (Optuna, 100 trials)
5. Add multi-metric evaluation functions

**Deliverables:**
- 7 model training pipelines
- Hyperparameter optimization configs
- Evaluation metric calculators

---

### Phase 3: System-Specific Modeling (Week 2-3)
**Tasks:**
For each of 7 physiological systems:
1. Train all 7 models per biomarker
2. 5-fold cross-validation
3. Hyperparameter optimization
4. Calculate all evaluation metrics
5. Generate SHAP plots

**Deliverables:**
- 28 trained models (7 systems × 4 biomarkers average)
- Performance comparison tables
- Feature importance visualizations

---

### Phase 4: Analysis & Reporting (Week 3-4)
**Tasks:**
1. Comparative before/after analysis
2. Statistical significance testing (paired t-test)
3. Feature importance analysis across systems
4. Clinical interpretation guidelines

**Deliverables:**
- Final performance report
- Feature importance heatmaps
- Manuscript-ready figures and tables
- Updated manuscript text

---

## Expected Performance Improvements

### Overall Target
- **Average R² improvement:** +0.10 across all biomarkers
- **Statistical significance:** 15/28 biomarkers with p<0.05
- **Clinical metrics:** AUC ≥ 0.70 for key thresholds

### By Biomarker Tier
| Tier | Current Avg R² | Target Avg R² | Expected Δ |
|------|----------------|---------------|------------|
| **Excellent (>0.30)** | 0.51 | 0.54 | +0.03 |
| **Moderate (0.05-0.30)** | 0.11 | 0.19 | +0.08 |
| **Poor (<0.05)** | -0.02 | 0.10 | +0.12 |

### Key Hypotheses to Test
1. **Socioeconomic features will most benefit immune and metabolic systems**
   - Rationale: These systems influenced by stress, nutrition, access to care

2. **Climate features will most benefit cardiovascular and vital signs**
   - Rationale: Direct physiological response to temperature

3. **Combined climate + socioeconomic interactions will outperform additive effects**
   - Test with interaction terms in feature engineering

4. **CatBoost will handle high-cardinality categorical features best**
   - e.g., dwelling type, ward, education level

---

## Success Criteria

### Technical
- ✅ All 28 biomarker models trained successfully
- ✅ Zero feature leakage detected (automated checks pass)
- ✅ SHAP values validate expected relationships
- ✅ Reproducible pipeline (seed=42, Docker, Git)

### Performance
- ✅ Average R² improvement ≥ +0.10 (moderate/poor biomarkers)
- ✅ ≥15/28 biomarkers with statistically significant improvement
- ✅ Clinical AUC ≥ 0.70 for key diagnostic thresholds

### Scientific
- ✅ Clear quantification of climate vs socioeconomic contributions
- ✅ Interpretable feature importance rankings
- ✅ Actionable insights for public health
- ✅ Publication-ready figures (Lancet Planetary Health quality)

---

## File Structure Created

```
ENBEL_pp_model_refinement/
├── MODEL_REFINEMENT_PLAN.md                    ✅ Complete planning document
├── MODEL_REFINEMENT_SETUP_SUMMARY.md           ✅ This summary
├── configs/
│   └── model_refinement_config.yaml            ✅ Enhanced configuration
└── scripts/
    └── evaluation/
        └── leakage_checker.py                  ✅ Automated leakage detection
```

---

## Next Steps

### Immediate (Week 1)
1. **Merge GCRO socioeconomic features** into clinical dataset
   - Use spatial-demographic matching from existing imputation
   - Validate merge quality (>95% coverage expected)

2. **Implement feature engineering pipeline**
   - Age binning, year period categorization
   - Interaction terms (climate × vulnerability)
   - Categorical encoding (target encoding)

3. **Run initial leakage validation**
   - Test all 28 biomarker feature sets
   - Generate safety reports

### Short-term (Week 2)
1. **Implement new model trainers**
   - CatBoost with native categorical support
   - Elastic Net with feature scaling
   - Extra Trees and Gradient Boosting

2. **Test on pilot biomarkers**
   - Hematocrit (excellent performer)
   - CD4 (poor performer)
   - Glucose (moderate performer)

### Medium-term (Weeks 3-4)
1. **Complete system-specific modeling**
   - All 7 systems, all biomarkers
   - Hyperparameter optimization
   - SHAP analysis

2. **Generate comparative analysis**
   - Before/after performance
   - Feature importance insights
   - Statistical significance

---

## Resources Required

### Computational
- **CPU:** 16+ cores recommended (parallel model training)
- **RAM:** 32GB+ (XGBoost/LightGBM memory intensive)
- **Storage:** ~10GB for models and results
- **Runtime:** ~48 hours for complete analysis (7 models × 28 biomarkers × 5 CV folds)

### Data
- ✅ Clinical dataset: 11,398 records (already available)
- ✅ GCRO dataset: 58,616 records (already available)
- ✅ Climate features: 99.5% coverage (already merged)

### Software Dependencies
**New packages to install:**
```bash
pip install catboost==1.2
pip install optuna==3.3.0
pip install shap==0.42.1
```

---

## Risk Mitigation

### Risk 1: Feature Leakage Despite Checks
**Mitigation:**
- Automated leakage checker validates EVERY model
- Manual review of top SHAP features
- Correlation matrix analysis

### Risk 2: Overfitting with Expanded Features
**Mitigation:**
- Strict train/test split (80/20)
- 5-fold cross-validation
- Temporal validation (2002-2015 train, 2016-2021 test)
- Regularization in all models (L1/L2, tree depth limits)

### Risk 3: Computational Time Exceeds Estimate
**Mitigation:**
- Parallel processing (n_jobs=-1)
- Optuna pruning for hyperparameter search
- Pilot testing before full run
- Checkpoint saving (resume if interrupted)

### Risk 4: No Performance Improvement
**Mitigation:**
- Realistic expectations (+0.10 R² is meaningful)
- Focus on interpretability over pure performance
- Clinical metrics (AUC, sensitivity) may improve even if R² doesn't

---

## Scientific Contributions

This work will contribute:

1. **Methodological:** First comprehensive leakage-free climate-health modeling framework
2. **Substantive:** Quantify socioeconomic vs climate contributions (currently unknown)
3. **Clinical:** Identify high-risk populations (vulnerable to climate + low SES)
4. **Policy:** Evidence-based interventions (e.g., cooling centers in low-income wards)
5. **Reproducible Science:** Open-source pipeline with automated checks

---

## References to Key Documents

1. **FEATURE_LEAKAGE_AUDIT.md** - Original leakage identification
2. **FEATURE_LEAKAGE_FIX_2025-10-14.md** - How leakage was addressed
3. **COMPREHENSIVE_FEATURE_SPACE_ANALYSIS.md** - Current performance baseline
4. **CLAUDE.md** - Overall project documentation

---

## Conclusion

✅ **Planning Phase Complete:** Comprehensive model refinement strategy established with:
- Clear biomarker groupings by physiological system
- Expanded feature space (20 → ~35 features)
- 7 ML algorithms with hyperparameter optimization
- Automated leakage prevention
- System-specific evaluation metrics
- 4-week implementation roadmap

**Next:** Begin Phase 1 (Feature Expansion) by merging GCRO socioeconomic features.

---

**Contact:** ENBEL Team
**Last Updated:** 2025-10-30
