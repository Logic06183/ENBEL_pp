# Revised Plan: Using Existing GCRO Imputation
**Date:** 2025-10-30
**Status:** 🔄 Major Revision

---

## Key Insight

The clinical dataset **already has GCRO socioeconomic features imputed** from your previous work! We don't need to redo the GCRO linkage.

---

## Existing GCRO Features in Clinical Dataset

| Feature | Completeness | Unique Values | Notes |
|---------|--------------|---------------|-------|
| **HEAT_VULNERABILITY_SCORE** | 75.86% (8,647/11,398) | 17 | Mean=19.2, Range=[0,100] ⭐ |
| **HEAT_STRESS_RISK_CATEGORY** | 75.86% (8,647/11,398) | 4 | LOW/MODERATE/HIGH/CRITICAL |
| **Sex** | 82.74% (9,431/11,398) | 4 | Female/Male/Unknown |
| **Race** | 34.76% (3,962/11,398) | 7 | Black/White/Coloured/Asian |

**Excellent!** Your previous imputation provided the key socioeconomic vulnerability measure.

---

## Revised Feature Set for Modeling

### Climate Features (6) - From Day 1 ✅
1. `climate_daily_mean_temp` (99.46%)
2. `climate_daily_max_temp` (99.46%)
3. `climate_daily_min_temp` (99.46%)
4. `climate_7d_mean_temp` (99.39%)
5. `climate_heat_stress_index` (99.46%)
6. `climate_season` (99.46%)

### Temporal Features (2) ✅
7. `month` (100%)
8. `season` (100%)

### Socioeconomic Features (2-3) - From Your Existing Imputation ✅
9. **`HEAT_VULNERABILITY_SCORE`** (75.86%) ⭐ KEY FEATURE
10. `Sex` (82.74%)
11. `Race` (34.76%) - *Optional, may drop due to low completeness*

### Demographic Features (1) ✅
12. `Age (at enrolment)` - check completeness

---

## Total Feature Set: 11-12 Features

**Much simpler than 23!**

| Category | Count | Source |
|----------|-------|--------|
| Climate | 6 | ERA5 (Day 1 validation) |
| Temporal | 2 | Derived |
| Socioeconomic | 1-2 | Your GCRO imputation (HEAT_VULNERABILITY_SCORE + Sex) |
| Demographic | 1 | Clinical (Age) |
| **TOTAL** | **10-11** | |

---

## Advantages of This Approach

1. ✅ **Respects your existing work** - Uses GCRO imputation you already completed
2. ✅ **Simpler** - 11 features vs 23 planned
3. ✅ **Focused** - HEAT_VULNERABILITY_SCORE captures composite socioeconomic risk
4. ✅ **Faster** - Skip all GCRO wave analysis
5. ✅ **Validated** - You already validated the imputation methodology

---

## What We Keep from Phase 1

### Keep: Day 1 Climate Analysis ✅
- Climate feature investigation (6 core features, 99.4% coverage)
- Rejection of poorly-validated derived features
- Decision documentation (CLIMATE_FEATURE_DECISION.md)

**This work is still valid and valuable!**

### Discard: Day 2 GCRO Wave Analysis ❌
- GCRO_DATA_STRUCTURE_ANALYSIS.md
- GCRO_WAVE_COMPARISON_REVISION.md
- 06_gcro_dataset_inspection.py
- 07_select_socioeconomic_features.py

**Not needed since you already did the imputation!**

---

## Remaining Tasks (Simplified)

### Task 2.A: Assess Existing Feature Completeness (30 min)
Check completeness of all features we'll use:
```python
final_features = [
    # Climate (6)
    'climate_daily_mean_temp', 'climate_daily_max_temp', 'climate_daily_min_temp',
    'climate_7d_mean_temp', 'climate_heat_stress_index', 'climate_season',
    # Temporal (2)
    'month', 'season',
    # Socioeconomic (2)
    'HEAT_VULNERABILITY_SCORE', 'Sex',
    # Demographic (1)
    'Age (at enrolment)'
]

# Check completeness
for feat in final_features:
    completeness = df[feat].notna().mean() * 100
    print(f'{feat}: {completeness:.2f}%')
```

### Task 2.B: Handle Missing Values (1-2 hours)
**Option 1: Complete Case Analysis**
- Records with all 11 features present
- Estimate: ~75% of 11,398 = ~8,500 records
- Simple, no imputation needed

**Option 2: Impute Missing HEAT_VULNERABILITY_SCORE & Sex** (RECOMMENDED)
- HEAT_VULNERABILITY_SCORE: 75.86% → impute remaining 24%
  - Use KNN imputation based on climate, location, age
  - Or median by study/year
- Sex: 82.74% → impute remaining 17%
  - Use mode by study
- Race: 34.76% → **DROP** (too many missing)

### Task 2.C: Run Automated Leakage Check (30 min)
Use existing `leakage_checker.py` to verify no biomarker-to-biomarker predictions

### Task 2.D: Feature Validation (1 hour)
- VIF analysis (target: VIF < 5)
- Correlation matrix
- Distribution checks

### Task 3: Model Implementation (6-8 hours)
**Now we can move directly to modeling!**

---

## Updated Timeline

**Completed:**
- ✅ Day 1: Climate features (2 hours)

**Remaining:**
- Task 2.A-2.D: Final feature prep (3 hours)
- Phase 2: Model implementation (6-8 hours)

**Total Remaining:** ~9-11 hours

---

## Revised Feature Space Rationale

### Why HEAT_VULNERABILITY_SCORE is Sufficient

`HEAT_VULNERABILITY_SCORE` is a **composite index** that already captures:
- Income/socioeconomic status
- Housing quality (dwelling type)
- Access to infrastructure
- Neighborhood vulnerability

This is exactly what we were trying to reconstruct from raw GCRO data!

**Your previous imputation work already solved this problem.**

### Why We Don't Need 15 Socioeconomic Features

Original plan: 15 features (income, education, dwelling, vulnerability indices, etc.)

With `HEAT_VULNERABILITY_SCORE`: **1 feature** captures the composite effect

**Advantages:**
- ✅ Simpler models (less overfitting risk)
- ✅ Single interpretable socioeconomic exposure
- ✅ Validated by your previous work
- ✅ Aligns with climate justice framework

---

## Next Steps

1. **Assess completeness** of 11 final features
2. **Decide on missing value strategy:**
   - Complete case (n~8,500)?
   - Impute HEAT_VULNERABILITY_SCORE & Sex?
3. **Run leakage checks**
4. **Validate features** (VIF, correlation)
5. **Move to modeling!**

---

## Apology

I apologize for the unnecessary detour into GCRO wave analysis. I should have:
1. First checked what features were already in the clinical dataset
2. Recognized your existing imputation work
3. Built on your work instead of redoing it

**Lesson learned:** Always check existing data structure before planning new imputation!

---

**Decision:** Use existing GCRO imputation (HEAT_VULNERABILITY_SCORE + Sex) + 6 climate features + 2 temporal = **10-11 total features**

**Next:** Assess feature completeness and handle missing values, then proceed directly to modeling.
