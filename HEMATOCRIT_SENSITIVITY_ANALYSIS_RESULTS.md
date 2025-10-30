# Hematocrit Sensitivity Analysis: Critical Units Issue Discovered

**Date**: 2025-10-30
**Analysis**: Sensitivity Analysis of High R² = 0.961 Finding
**Result**: UNITS ARTIFACT DISCOVERED - True R² = 0.03

---

## Executive Summary

The initial mixed effects DLNM analysis reported **R² = 0.961 for Hematocrit**, suggesting an exceptionally strong climate-biomarker relationship. Rigorous sensitivity analysis revealed this to be a **complete artifact** of inconsistent measurement units across studies.

**Key Finding**: One study recorded hematocrit as a **decimal** (0.41 = 41%) while others used **percentages** (45.9%). The random effects model captured this units discrepancy, not climate effects.

**Corrected Result**: After standardizing units, true **R² = 0.03** (3% variance explained).

---

## Methods: Comprehensive Sensitivity Analysis

### Tests Performed

1. **Variance Decomposition** - Within-study vs between-study variance
2. **Progressive Model Comparison** - 8 models from simple to complex
3. **Critical Test** - Random effects alone vs with predictors
4. **Permutation Test** - Shuffled temperature to test significance
5. **Within-Study Analysis** - Separate models per study
6. **Leave-One-Study-Out** - Robustness to individual studies
7. **Residual Diagnostics** - Model assumptions validation
8. **Units Correction** - Standardize measurement scales

---

## Results: The Smoking Gun

### 1. Initial Descriptive Statistics (UNCORRECTED)

| Study | N | Mean Hematocrit | SD | Min | Max |
|-------|---|-----------------|-----|-----|-----|
| JHB_Ezin_002 | 1,052 | **0.41%** | 0.05 | 0.21 | 0.54 |
| JHB_SCHARP_004 | 2 | **45.90%** | 2.12 | 44.4 | 47.4 |
| JHB_WRHI_001 | 1,066 | **38.94%** | 5.49 | 13.4 | 60.3 |

**🚨 RED FLAG**: Study means differ by 100-fold! (0.41% vs 45.90%)

### 2. Variance Decomposition (UNCORRECTED)

```
Total variance:       386.81
Within-study:          15.17 (3.9%)
Between-study:        371.64 (96.1%)
ICC:                   0.961
```

**Interpretation**: 96% of variance is between studies, only 4% within studies.

### 3. Critical Model Comparison (UNCORRECTED)

| Model | R² | AIC | Interpretation |
|-------|-----|-----|----------------|
| **Study RE only (no predictors)** | **0.961** | 11,788 | Random effects alone explain 96%! |
| Full model with temperature | 0.961 | 11,712 | Temperature adds nothing! |
| Baseline (no random effects) | 0.873 | 14,197 | High R² from study differences |
| Temperature only | 0.015 | 18,505 | Temperature alone: only 1.5%! |

**🚨 SMOKING GUN**:
- Random study effects ALONE (no temperature!) → R² = 0.961
- Adding temperature → R² = 0.961 (no change!)
- **Conclusion**: The 0.961 R² is entirely from study differences, NOT climate

### 4. Permutation Test (UNCORRECTED)

```
Original model R²:           0.961
Shuffled temperature R²:     0.961
Difference:                  0.000
```

**🚨 DEFINITIVE PROOF**:
- **Shuffling temperature doesn't change R²!**
- Temperature is NOT driving the high R²
- Random study effects capture everything

---

## Root Cause: Units Discrepancy

### Problem Identified

**JHB_Ezin_002 study** recorded hematocrit as a **decimal fraction** (0.41 = 41%), while other studies used **percentage format** (45.9% = 45.9%).

This created a 100-fold difference in raw values that the random effects model interpreted as genuine study-level variation.

### Units Correction Applied

```r
# If values < 1.0, multiply by 100 (convert decimal to percentage)
df_clean[, biomarker := ifelse(biomarker_raw < 1.0,
                                biomarker_raw * 100,
                                biomarker_raw)]
```

### Results After Correction

| Study | N | Mean Hematocrit (Corrected) | SD | Min | Max |
|-------|---|----------------------------|-----|-----|-----|
| JHB_Ezin_002 | 1,052 | **40.67%** | 5.33 | 21.0 | 54.0 |
| JHB_SCHARP_004 | 2 | **45.90%** | 2.12 | 44.4 | 47.4 |
| JHB_WRHI_001 | 1,066 | **38.94%** | 5.49 | 13.4 | 60.3 |

**✅ Now in consistent units!** Study means differ by ~7% (38.94% to 45.90%), not 100-fold.

---

## Corrected Analysis Results

### Variance Decomposition (CORRECTED)

```
Total variance:       30.02
Within-study:         29.24 (97.4%)
Between-study:         0.79 (2.6%)
ICC:                   0.026
```

**✅ Dramatic change**:
- Before: 96% between-study → After: **3% between-study**
- ICC drops from 0.961 to **0.026**
- Units correction successful!

### Model Comparison (CORRECTED)

| Model | R² | AIC | ΔAIC | Interpretation |
|-------|-----|-----|------|----------------|
| Random slope | **0.030** | 13,096 | 0 | Best model |
| Baseline (no RE) | **0.028** | 13,099 | +2.4 | Nearly identical |
| Random intercept | **0.028** | 13,099 | +2.7 | Nearly identical |
| Study RE only | **0.025** | 13,180 | +83 | Study effects minimal |

**Key Insights**:
- **TRUE R² = 0.03** (3% variance explained, not 96%!)
- Random effects add **nothing** (ΔAIC = +2.4 vs baseline)
- Study heterogeneity is minimal (2.6%)
- Temperature + controls explain 2.8%, random effects add 0.2%

### DLNM Significance (CORRECTED)

```
Significant effects: 4 temperatures (20-23°C)
Effect size: Small but statistically significant
```

**Interpretation**:
- Temperature effects ARE real (4 significant temps)
- But effect size is **SMALL** (R² = 0.03, not 0.96)
- Relationship is statistically valid but weak

---

## Comparison: Before vs After

| Metric | UNCORRECTED | CORRECTED | Ratio |
|--------|-------------|-----------|-------|
| R² (full model) | 0.961 | **0.030** | **32x inflated!** |
| ICC | 0.961 | 0.026 | **37x inflated!** |
| Between-study variance | 96.1% | 2.6% | **37x inflated!** |
| Within-study variance | 3.9% | 97.4% | 25x increased |
| Study RE contribution | 96.1% | 0.2% | **480x inflated!** |

---

## Implications

### 1. Methodological Lessons

✅ **Always check descriptive statistics by study/site before modeling**
✅ **Inspect raw data distributions for multimodal patterns**
✅ **Verify consistent measurement units across data sources**
✅ **High ICC (>0.50) warrants investigation of data quality issues**
✅ **Permutation tests are essential for validating model inferences**

### 2. Statistical Interpretation

The original R² = 0.961 was **statistically correct** but **scientifically meaningless**:
- The model accurately captured that one study used different units
- Random effects correctly identified study-level heterogeneity
- But this heterogeneity was an artifact, not biological variation

**Lesson**: High R² ≠ Strong effect. Always investigate what drives the variance.

### 3. Climate-Hematocrit Relationship

**Corrected conclusion**:
- Climate explains **3% of hematocrit variance** (weak effect)
- Temperature effects are **statistically significant** (4 temps at 20-23°C)
- Effect size is **small** (not the exceptionally strong relationship initially suggested)
- Random effects modeling is **unnecessary** (ICC = 0.026)

### 4. Comparison with Other Biomarkers

| Biomarker | R² (Corrected) | Significant DLNM? | Interpretation |
|-----------|----------------|-------------------|----------------|
| **Hematocrit** | **0.030** | Yes (4 temps) | Weak but valid |
| **Cholesterol** | **0.345** | Yes (23 temps) | Strong and valid |
| Glucose | 0.090 | No | Weak, not significant |
| Creatinine | 0.130 | No | Weak, not significant |

**Revised ranking**:
1. **Cholesterol** (R² = 0.345) - strongest climate effect
2. **Creatinine** (R² = 0.130) - modest effect
3. **Glucose** (R² = 0.090) - weak effect
4. **Hematocrit** (R² = 0.030) - weakest effect (after correction)

---

## Recommendations

### For This Analysis

1. **✅ Use corrected R² = 0.03** in all reporting
2. **✅ Report significance** (4 temperatures with significant effects)
3. **✅ Emphasize small effect size** (climate explains 3% of variance)
4. **✅ Remove random effects** (ICC = 0.026, not needed)
5. **✅ Acknowledge units issue** in limitations section

### For Future Analyses

1. **Check raw data distributions** before modeling
2. **Verify measurement units** are consistent across studies
3. **Use permutation tests** to validate findings
4. **Report both R² and effect sizes** (don't rely on R² alone)
5. **Investigate high ICC values** (>0.50 suggests data quality issues)

### For Manuscript

**Original claim** (WRONG):
> "Hematocrit showed exceptional climate sensitivity (R² = 0.96) with random effects models."

**Corrected claim** (RIGHT):
> "After correcting for measurement unit inconsistencies across studies, hematocrit showed weak but statistically significant temperature associations (R² = 0.03, 4 significant temperatures at 20-23°C)."

---

## Scientific Integrity

### Transparency

This sensitivity analysis demonstrates the importance of:
- **Questioning unexpected findings** (R² = 0.96 for environmental exposure is extraordinary)
- **Rigorous diagnostic testing** (permutation tests, variance decomposition)
- **Data quality checks** (units, distributions, outliers)
- **Transparent reporting** (acknowledging and correcting errors)

### Lessons Learned

High R² can result from:
1. **Data artifacts** (units, outliers, entry errors)
2. **Study design features** (clustering, stratification)
3. **Confounding** (unmeasured variables)
4. **True strong effects** (rare for environmental exposures)

**Golden rule**: Extraordinary claims (R² = 0.96 for climate effect) require extraordinary evidence and scrutiny.

---

## Conclusions

1. **Units artifact discovered**: One study used decimal format, others percentage
2. **True R² = 0.03**: Climate explains 3% of hematocrit variance (not 96%)
3. **Statistically significant**: 4 temperatures show significant DLNM effects
4. **Small effect size**: Weak climate-hematocrit relationship
5. **No random effects needed**: ICC = 0.026 (minimal between-study variation)
6. **Cholesterol remains strongest**: R² = 0.345 is the true winner

### Final Verdict

❌ **Reject**: Hematocrit as exceptionally climate-sensitive biomarker (R² = 0.96)
✅ **Accept**: Hematocrit shows weak but significant climate associations (R² = 0.03)
✅ **Recommend**: Cholesterol (R² = 0.345) is the strongest validated climate-sensitive biomarker

---

## Files Generated

1. **R/hematocrit_sensitivity_analysis.R** - Complete diagnostic script
2. **R/hematocrit_units_correction.R** - Units correction and re-analysis
3. **reanalysis_outputs/hematocrit_sensitivity/** - Diagnostic outputs
4. **reanalysis_outputs/hematocrit_corrected/** - Corrected analysis results
5. **hematocrit_corrected_data.csv** - Standardized units dataset

---

**Analysis Completed**: 2025-10-30
**Lesson**: Always check your data! High R² = red flag, not celebration.
**Credit**: Sensitivity analysis prompted by user skepticism - excellent scientific practice!

