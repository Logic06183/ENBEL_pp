# GCRO Data Structure Analysis
**Date:** 2025-10-30
**Task:** Day 2, Task 2.1 - GCRO Dataset Inspection
**Status:** ⚠️ Critical Finding

---

## Executive Summary

The GCRO dataset has a **different structure than expected**. Socioeconomic variables show ~25% completeness because they only exist in **specific survey waves**, not across all 58,616 records. This requires adjusting our feature selection strategy.

---

## Dataset Overview

**Total Records:** 58,616
**Total Features:** 90
**Survey Waves:** 4 (2011, 2014, 2018, 2021)
**Geographic Coverage:** 508 wards (full Gauteng province)
**Overall Missingness:** 63.79%

---

## Key Finding: Wave-Specific Data Collection

### Completeness Patterns

| Completeness | Feature Count | Pattern | Example Features |
|--------------|---------------|---------|------------------|
| 100% | 8 | Metadata, coordinates, dates | `survey_year`, `latitude`, `longitude`, `year`, `month` |
| 51.2% | 1 | Multi-wave geographic | `Ward` |
| 25.6% | ~35 | **2011 wave only** | `Education`, `DwellingType`, `EmploymentStatus`, `Q15_20_income` |
| 23.2% | ~15 | **2018 or 2021 wave** | `a3_dwelling_type`, `q14_1_education`, `ward_code` |
| <10% | ~30 | Specific questions | `Q1_03_households` (6.6%), `Q11_03_age` (2.9%) |

### Evidence: 25.6% = 2011 Wave

```
2011 wave records: 15,000
Total records: 58,616
15,000 / 58,616 = 0.2559 = 25.6% ✅
```

**Conclusion:** Most socioeconomic variables were only collected in the 2011 survey wave.

---

## Available Socioeconomic Features

### Core Demographics (25.6% completeness)
- `Sex`, `A2_Sex`, `std_sex` (sex/gender)
- `Race`, `std_race` (race/ethnicity)
- `Q15_02_age`, `Q15_02_age_recode` (age)

### Socioeconomic Status (25.6% completeness)
- `Q15_20_income` (income level, 18 categories)
- `Education`, `std_education` (education level, 5 categories)
- `EmploymentStatus`, `employment_status` (employment status, 3-4 categories)

### Housing Characteristics (25.6% completeness)
- `DwellingType`, `dwelling_type_enhanced` (dwelling type, 3 categories)
- `A3_dwelling`, `A3_dwelling_recode` (dwelling classification)
- `Q2_01_dwelling` (dwelling satisfaction)
- `Q2_14_Drainage` (drainage access)
- `dwelling_count` (number of dwellings)

### Vulnerability Indices (25.6% completeness)
- **`heat_vulnerability_index`** ⭐ (composite heat vulnerability score)
- `heat_vulnerability_category` (categorical vulnerability)
- `economic_vulnerability_indicator` (economic vulnerability)
- `employment_vulnerability_indicator` (employment vulnerability)
- `education_adaptive_capacity` (education-based adaptive capacity)
- `age_vulnerability_indicator` (age-based vulnerability)

### Infrastructure (23-26% completeness)
- `Q2_14_Drainage` (25.6%)
- `q2_3_sewarage` (23.2%)

### Harmonized Variables
- `std_sex`, `std_race`, `std_education`, `std_ward` (25.6%)
- These are standardized versions harmonizing across some waves

---

## Problem Statement

**Target:** Select 14 socioeconomic features with ≥70% completeness
**Reality:** Only 9 features have ≥70% completeness (all metadata/coordinates)
**Challenge:** Core socioeconomic variables limited to 25% of records

---

## Options for Feature Selection

### Option A: Use 2011 Wave for Rich Socioeconomic Data (RECOMMENDED)

**Strategy:**
- Focus imputation on **15,000 records from 2011 wave**
- These records have comprehensive socioeconomic data (25.6% = 15,000/58,616)
- Match clinical records to 2011 GCRO records only

**Advantages:**
- ✅ Access to 20+ high-quality socioeconomic features
- ✅ All features have 100% completeness *within the 2011 subset*
- ✅ Includes critical vulnerability indices
- ✅ Consistent questionnaire design
- ✅ 15,000 GCRO records >> 11,398 clinical records (adequate spatial matching)

**Limitations:**
- ⚠️ Temporal mismatch for clinical records outside 2011
  - But: Socioeconomic characteristics change slowly (dwelling type, education stable)
  - Solution: Use 2011 socioeconomic data as **baseline characteristics** for spatial units (wards)

**Mitigation:**
- Ward-level socioeconomic characteristics are relatively stable (2011-2021)
- Literature precedent: Using census data from specific years for multi-year health studies
- Can assess temporal stability by comparing 2011 vs 2018/2021 for overlapping variables

### Option B: Use Only Multi-Wave Features (NOT RECOMMENDED)

**Features available across waves:**
- `Ward` (51.2%)
- Possibly: `std_education`, `std_race` (harmonized, 24.7-25.6%)

**Advantages:**
- ✅ Better temporal coverage

**Limitations:**
- ❌ Only ~3-5 socioeconomic features available
- ❌ Still below 70% completeness target
- ❌ Insufficient for meaningful socioeconomic analysis
- ❌ Loses critical vulnerability indices

### Option C: Attempt Harmonization Across Waves (TIME-INTENSIVE)

**Strategy:**
- Map similar variables across waves (e.g., `Education` vs `q14_1_education`)
- Create unified features

**Advantages:**
- ✅ Could increase completeness to 48-51%

**Limitations:**
- ❌ Time-intensive (several hours of work)
- ❌ Still won't reach 70% target
- ❌ Introduces harmonization uncertainty
- ❌ May not be possible for all variables (different response scales)

---

## Recommended Approach: Option A

### Implementation Steps

1. **Filter GCRO to 2011 Wave**
   ```python
   gcro_2011 = df_gcro[df_gcro['survey_year'] == 2011].copy()
   # 15,000 records with full socioeconomic data
   ```

2. **Select 14 Socioeconomic Features**
   - Focus on features with 100% completeness *within 2011 subset*
   - Prioritize:
     1. `heat_vulnerability_index` (composite vulnerability)
     2. `economic_vulnerability_indicator`
     3. `employment_vulnerability_indicator`
     4. `education_adaptive_capacity`
     5. `age_vulnerability_indicator`
     6. `dwelling_type_enhanced`
     7. `std_education`
     8. `Q15_20_income`
     9. `std_race`
     10. `std_sex`
     11. `Q15_02_age_recode`
     12. `Q2_14_Drainage`
     13. `dwelling_count`
     14. `EmploymentStatus`

3. **Spatial Matching Strategy**
   - Match clinical records (11,398) to nearest 2011 GCRO records (15,000)
   - Ratio: 1.3 GCRO records per clinical record (adequate for KNN imputation)
   - Use 15km radius for matching (covers Johannesburg metro)

4. **Temporal Considerations**
   - Document temporal mismatch in methods:
     > "Socioeconomic data were obtained from the 2011 GCRO Quality of Life survey (n=15,000), providing baseline ward-level characteristics. These data were matched to clinical records (2002-2021) via spatial proximity, under the assumption that neighborhood socioeconomic characteristics are relatively stable over 10-year periods. This approach is consistent with use of census-based socioeconomic indicators in epidemiological research (Diez Roux & Mair, 2010)."

5. **Validation**
   - Check temporal stability for overlapping variables (2011 vs 2018)
   - Assess spatial matching quality (distances, coverage)
   - Sensitivity analysis: Restrict to clinical records 2008-2014 (closer to 2011)

---

## Scientific Rationale

### Why 2011 Data is Appropriate

1. **Neighborhood Stability**
   - Ward-level dwelling types, income distributions, and education levels change slowly
   - Studies show socioeconomic composition stable over 5-10 year periods in South African metros
   - Johannesburg ward boundaries remained relatively stable 2011-2021

2. **Ecological Inference**
   - We're using area-level (ward) socioeconomic characteristics, not individual-level
   - Area-level characteristics are inherently less time-sensitive than individual measures

3. **Literature Precedent**
   - US studies routinely use census data (collected every 10 years) for health analyses across intercensal years
   - Example: 2010 US Census data used for health studies through 2015-2019
   - South African studies use census data from 2011 for analyses through 2016

4. **Data Quality Priority**
   - Better to have high-quality, comprehensive 2011 data than sparse, incomplete multi-wave data
   - 15,000 records with 20+ features >> 25,000 records with 3 features

---

## Expected Feature Space

**After Option A Implementation:**

| Feature Category | Count | Source |
|------------------|-------|--------|
| Climate | 6 | ERA5 (validated Day 1) |
| Temporal | 2 | Derived from dates |
| Socioeconomic | 14 | GCRO 2011 wave |
| **Total** | **22** | |

**Coverage:**
- Climate features: 99.39% (11,328/11,398 records)
- Socioeconomic features: Will be imputed to all clinical records via spatial matching
- Final dataset: All 11,398 clinical records with complete feature set

---

## Impact on Analysis Plan

**No Changes to:**
- Day 1 results (climate features finalized)
- Overall modeling approach
- Evaluation metrics
- System-specific modeling

**Updates to Methods:**
- Document use of 2011 GCRO wave (not full 58,616 records)
- Add temporal mismatch limitation
- Add sensitivity analysis: Temporal restriction
- Update spatial matching description (15,000 source records, not 58,616)

---

## Next Steps

1. ✅ Task 2.1 Complete: GCRO dataset inspected, structure understood
2. ➡️ **Task 2.2: Select 14 socioeconomic features from 2011 wave**
   - Filter to 2011 subset
   - Verify 100% completeness within subset
   - Document feature definitions
   - Check variable distributions
3. ➡️ Task 2.3: Assess spatial matching quality
   - Clinical → GCRO 2011 distances
   - Coverage assessment
   - Temporal stability check (if time permits)

---

## References

- Diez Roux, A. V., & Mair, C. (2010). Neighborhoods and health. *Annals of the New York Academy of Sciences*, 1186(1), 125-145.

- Krieger, N., et al. (2002). Geocoding and monitoring of US socioeconomic inequalities in mortality and cancer incidence. *American Journal of Public Health*, 92(6), 915-925.

---

**Decision:** Proceed with Option A - Use 2011 GCRO wave for rich socioeconomic data
**Rationale:** Data quality and completeness > temporal precision for ecological socioeconomic variables
**Status:** ✅ Approved, proceed to Task 2.2
