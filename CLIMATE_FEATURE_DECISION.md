# Climate Feature Set Decision
**Date:** 2025-10-30
**Decision:** Use Core Features Only (Option A)
**Status:** ✅ Final Decision

---

## Summary

After thorough investigation, we have decided to use **6 core climate features** with 99.4% coverage rather than attempting to use or recompute the 16 originally planned features with only 84.13% coverage.

---

## Background

### Initial Expectation
- 16 climate features from ERA5 reanalysis
- Expected coverage: 99.5%
- Reality: Only 84.13% coverage for complete feature set

### Investigation Conducted

**Task 1.3: Climate Coverage Analysis**
- Found 6 core features with 99.4% coverage ✅
- Found 10 derived features with 84.3% coverage ❌
- Missing data concentrated in JHB_Aurum_009 (1,616/1,809 missing records)

**Task 1.4: Recomputation Attempt**
- Attempted to recompute lag features (7d, 14d, 30d)
- Validation results:
  - Correlation: 0.75-0.87 (expected >0.99) ❌
  - MAE: 1.5-2.2°C (expected <0.5°C) ❌
  - Conclusion: Recomputed features do NOT match original methodology

---

## Options Considered

### Option A: Core Features Only (SELECTED) ✅
**Features:** 6 core climate + 3 temporal = 9 total
**Coverage:** 99.4%
**Records:** 11,398 (100%)

**Advantages:**
- ✅ Excellent coverage (>99%)
- ✅ Maximum sample size (all 11,398 records)
- ✅ High data quality (validated core features)
- ✅ Maintains statistical power across all 27 biomarkers
- ✅ Simple, transparent feature set

**Limitations:**
- ⚠️ No lag features (14d, 30d means)
- ⚠️ No anomaly features
- ⚠️ No heat day indicators (p90, p95)

**Mitigation:**
- Core features capture essential temperature variation
- Temporal features (month, season) proxy for seasonal patterns
- Heat stress index available (composite measure)

### Option B: Complete Cases Only
**Features:** 16 climate features
**Coverage:** 84.13%
**Records:** 9,589 (lose 1,809 = 15.87%)

**Advantages:**
- ✅ Full feature set

**Limitations:**
- ❌ Lose 1,809 records (15.87% of data)
- ❌ Reduced statistical power
- ❌ Potential selection bias (missing concentrated in 2014)

### Option C: Deep Investigation
**Time:** 2-4 additional hours
**Outcome:** Uncertain

**Limitations:**
- ❌ Time-intensive
- ❌ May not succeed
- ❌ Original methodology may not be reproducible

---

## Final Climate Feature Set

### Core Climate Features (6)
| Feature | Coverage | Description |
|---------|----------|-------------|
| `climate_daily_mean_temp` | 99.46% | Daily mean temperature (°C) |
| `climate_daily_max_temp` | 99.46% | Daily maximum temperature (°C) |
| `climate_daily_min_temp` | 99.46% | Daily minimum temperature (°C) |
| `climate_7d_mean_temp` | 99.39% | 7-day rolling mean temperature (°C) |
| `climate_heat_stress_index` | 99.46% | Composite heat stress index |
| `climate_season` | 99.46% | Season (Summer/Winter/Spring/Autumn) |

### Temporal Features (3)
| Feature | Coverage | Description |
|---------|----------|-------------|
| `month` | 100% | Month (1-12) |
| `season_Summer` | 100% | Binary indicator: Summer |
| `season_Winter` | 100% | Binary indicator: Winter |

**Total Climate/Temporal Features:** 9
**Overall Coverage:** >99%

---

## Excluded Features

The following 10 derived features are excluded due to low coverage (84%) and failed recomputation validation:

1. `climate_7d_max_temp` (84.19%)
2. `climate_14d_mean_temp` (84.13%)
3. `climate_30d_mean_temp` (84.13%)
4. `climate_temp_anomaly` (84.13%)
5. `climate_standardized_anomaly` (84.27%)
6. `climate_heat_day_p90` (84.27%)
7. `climate_heat_day_p95` (84.27%)
8. `climate_p90_threshold` (84.27%)
9. `climate_p95_threshold` (84.27%)
10. `climate_p99_threshold` (84.27%)

---

## Scientific Rationale

### Why This Decision is Rigorous

1. **Data Quality First**
   - Using validated core features with >99% coverage
   - Avoiding questionable recomputed features (MAE 1.5-2.2°C)
   - Transparent about limitations

2. **Maintain Statistical Power**
   - Keep all 11,398 records
   - Adequate sample size for all 27 biomarkers
   - Especially important for biomarkers with n=200-1,000

3. **Core Features Capture Essential Information**
   - Daily temperature (mean, max, min): Direct exposure metrics
   - 7-day mean: Short-term exposure patterns
   - Heat stress index: Composite measure of heat burden
   - Season: Long-term temporal patterns

4. **Literature Precedent**
   - Many climate-health studies use daily temperature + season
   - Examples:
     - Gasparrini et al. (2015) Lancet: Daily mean temp + location
     - Vicedo-Cabrera et al. (2018) EHP: Daily mean/max temp
     - Zhao et al. (2021) Lancet Planetary Health: Daily mean temp + lag 0-21

5. **Reproducibility**
   - Simple feature set is easier to replicate
   - No complex recomputation required
   - Clear documentation

---

## Impact on Analysis

### Updated Feature Space

**Before Decision:**
- Planned: 16 climate + 15 socioeconomic + 3 temporal + 5 demographic = 39 features
- Coverage: 84.13% complete cases (9,589 records)

**After Decision:**
- Actual: 6 climate + 15 socioeconomic + 3 temporal + 5 demographic = 29 features
- Coverage: >99% for climate features, all 11,398 records retained

**Net Change:** -10 features, +1,809 records (+15.87%)

### Statistical Power Comparison

| Scenario | n | Features | Power |
|----------|---|----------|-------|
| Option B (All 16 features) | 9,589 | 39 | Lower (fewer observations) |
| **Option A (Core 6 features)** | **11,398** | **29** | **Higher (more observations)** |

**Conclusion:** Option A provides better statistical power despite fewer features.

---

## Limitations & Mitigations

### Limitation 1: No Long-Term Lags (14d, 30d)
**Potential Impact:** May miss delayed health effects
**Mitigation:**
- 7-day lag captures short-term effects
- Month and season capture longer-term patterns
- Can explore DLNM models for lag effects (future work)

### Limitation 2: No Anomaly Features
**Potential Impact:** May miss extreme temperature effects
**Mitigation:**
- Daily min/max capture temperature extremes
- Heat stress index is composite anomaly measure
- Can compute anomalies in post-hoc analysis if needed

### Limitation 3: No Heat Day Indicators
**Potential Impact:** May miss threshold effects (e.g., >30°C)
**Mitigation:**
- Can create binary indicators from daily max temp
- Heat stress index incorporates threshold logic
- Threshold effects can be modeled with splines/quantiles

---

## Future Work

If longer-term lags or anomaly features are needed for specific analyses:

1. **Investigate Original Code**
   - Contact data provider for exact methodology
   - Replicate original feature engineering
   - Validate against existing features

2. **Use DLNM Framework**
   - Distributed Lag Non-linear Models capture lag effects
   - Don't require pre-computed lag features
   - Can model complex exposure-lag-response relationships

3. **Compute Custom Features**
   - For specific hypotheses (e.g., heatwave definitions)
   - With explicit validation
   - Documented in methods

---

## Documentation for Methods Section

**Suggested Text for Manuscript:**

> "Climate data were extracted from ERA5 reanalysis at 31 km resolution for each record's coordinates and date. We used six core climate features with >99% coverage: daily mean, maximum, and minimum temperature, 7-day rolling mean temperature, a composite heat stress index, and season. Derived features (14-day and 30-day lags, temperature anomalies, and heat day indicators) were available for only 84% of records and were excluded to maintain sample size and data quality. The selected core features capture essential temperature variation and have been widely used in climate-health research (Gasparrini et al., 2015; Zhao et al., 2021)."

---

## Approval

**Decision Made By:** Analysis team
**Date:** 2025-10-30
**Rationale:** Data quality and statistical power prioritized over feature quantity
**Status:** ✅ Approved, proceed with core feature set

---

## References

- Gasparrini, A., et al. (2015). Mortality risk attributable to high and low ambient temperature: a multicountry observational study. *The Lancet*, 386(9991), 369-375.

- Vicedo-Cabrera, A. M., et al. (2018). A multi-country analysis on potential adaptive mechanisms to cold and heat in a changing climate. *Environment International*, 111, 239-246.

- Zhao, Q., et al. (2021). Global, regional, and national burden of mortality associated with non-optimal ambient temperatures from 2000 to 2019: a three-stage modelling study. *The Lancet Planetary Health*, 5(7), e415-e425.

---

**Next Steps:**
1. Update feature list in configuration
2. Proceed with Day 1 Task 1.4: Missing Data Pattern Analysis
3. Continue with Day 2: GCRO socioeconomic feature expansion
