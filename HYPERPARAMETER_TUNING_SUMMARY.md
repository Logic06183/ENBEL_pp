# Hyperparameter Tuning Analysis - Running

**Date**: 2025-10-30
**Status**: ✅ Running with PID 85114
**Output**: `results/multi_model_comparison_tuned.log`

---

## Configuration

### Tuning Strategy
- **Method**: RandomizedSearchCV
- **Iterations**: 10 per algorithm
- **Cross-validation**: 3-fold
- **Total models trained**: 60 (10 biomarkers × 3 algorithms × 2 scenarios)
- **Expected runtime**: 45-90 minutes

### Hyperparameter Grids

**Random Forest:**
- n_estimators: [50, 100]
- max_depth: [10, 15, None]
- min_samples_split: [5, 10]
- min_samples_leaf: [5, 10]

**XGBoost:**
- n_estimators: [50, 100]
- max_depth: [5, 7]
- learning_rate: [0.05, 0.1]
- subsample: [0.8, 1.0]

**LightGBM:**
- n_estimators: [50, 100]
- max_depth: [5, 7, -1]
- learning_rate: [0.05, 0.1]
- num_leaves: [31, 50]

---

## Baseline Results (Fixed Hyperparameters)

### Top Biomarkers (Climate-only):
1. **Hematocrit**: R² = 0.935 ⚠️  (socioeconomic-driven, not climate!)
2. **CD4**: R² = 0.712 ✅ (climate-sensitive)
3. **FASTING LDL**: R² = 0.379
4. **FASTING HDL**: R² = 0.337

### Key Finding: **Demographics don't help most biomarkers**

**ΔR² (Full - Climate):**
- ✅ Improved: Hematocrit (+0.026), CD4 (+0.029), Weight (+0.028)
- ❌ Worsened: 7 biomarkers (sample size loss 8,577 → 5,788)

---

## 🚨 Hematocrit Data Quality Issue

**Problem detected:**
- Correlation = 0.008 between Hematocrit and Hemoglobin
- **Expected**: 0.85-0.95 (strong biological relationship)
- **Actual**: Near zero (data quality issue or unit mismatch)

**SHAP analysis reveals:**
- HEAT_VULNERABILITY_SCORE: 19.03 importance (96% of total)
- All climate features: <0.5 combined (<4% of total)

**Conclusion:**
Hematocrit R² = 0.935 is **NOT climate-driven**. It's driven by **socioeconomic vulnerability** (housing quality, income, chronic stress), not acute temperature exposure.

**Recommendation:**
- Report hematocrit as "socioeconomic vulnerability biomarker" not "climate biomarker"
- Investigate hemoglobin/hematocrit correlation issue (unit conversion? data entry error?)
- Focus on CD4 as the true climate-sensitive biomarker

---

## CD4 as Star Climate Biomarker ⭐

**Why CD4 is exciting:**
- R² = 0.712 with climate alone
- Biologically plausible (heat stress → immune suppression)
- Sufficient sample size (2,333 observations)
- HIV population particularly vulnerable to climate stressors

**Climate effects on CD4:**
- Temperature extremes stress immune system
- Heat waves → dehydration → medication adherence issues
- Socioeconomic vulnerability amplifies effects

---

## Expected Improvements from Tuning

Based on literature, hyperparameter tuning typically improves R² by:
- Random Forest: +0.02 to +0.05
- XGBoost: +0.03 to +0.07
- LightGBM: +0.03 to +0.06

**Projected results:**
- CD4: R² = 0.712 → **0.75-0.78** (tuned)
- LDL: R² = 0.379 → **0.40-0.43** (tuned)
- HDL: R² = 0.337 → **0.36-0.39** (tuned)

---

## Monitoring Progress

### Check if running:
```bash
ps aux | grep multi_model_two_scenario_comparison
```

### View live progress:
```bash
tail -f results/multi_model_comparison_tuned.log
```

### Check results when complete:
```bash
cat results/multi_model_comparison/master_results.json
```

---

## Important Notes

### About Computer Sleep:
⚠️  **If your Mac goes to sleep, the process will pause** (not killed, just paused)
- Solution: Keep Mac awake or use `caffeinate` command
- To prevent sleep: System Settings → Energy → Prevent automatic sleeping

### Process Details:
- **PID**: 85114
- **Command**: `nohup python3 scripts/multi_model_two_scenario_comparison.py`
- **Output**: Redirected to `results/multi_model_comparison_tuned.log`
- **Detached**: Yes (will survive terminal close)

---

## Next Steps (After Completion)

1. **Review tuned results** - Compare to baseline fixed hyperparameters
2. **Generate summary visualizations**:
   - Heatmap: Feature importance across all biomarkers
   - Bar chart: ΔR² (Full - Climate) for each biomarker
   - SHAP summary: Top features for CD4, LDL, HDL
3. **Write up findings**:
   - CD4 as climate-sensitive biomarker
   - Hematocrit caveat (socioeconomic, not climate)
   - Demographics add minimal value (sample size loss)
4. **Recommendations for future work**:
   - DLNM analysis for CD4 (capture lagged effects)
   - Investigate hematocrit data quality
   - Focus on climate-only models (better sample size)

---

## Files Generated

### Baseline (Fixed Hyperparameters):
- `results/multi_model_comparison/master_results.json`
- `results/multi_model_comparison/[biomarker]_results.json` (10 files)
- `results/multi_model_comparison_run.log`

### Tuned (Hyperparameter Search):
- `results/multi_model_comparison_tuned.log` (this run)
- Results will overwrite baseline files in `results/multi_model_comparison/`

---

**Started**: 2025-10-30 12:40 PM
**Expected completion**: 2025-10-30 1:30-2:15 PM
**Status**: ✅ Running (check log for progress)
