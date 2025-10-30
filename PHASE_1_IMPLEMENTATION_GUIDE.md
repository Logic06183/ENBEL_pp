# Phase 1 Implementation Guide
## Feature Expansion & Data Quality Assessment
**Version:** 1.0
**Duration:** Week 1 (5 days)
**Status:** 📋 Ready to Execute

---

## Overview

This guide provides step-by-step instructions for Phase 1 of the model refinement project, focusing on:
1. Comprehensive data quality assessment
2. Missingness analysis and handling
3. GCRO socioeconomic feature expansion
4. Feature validation and leakage checking

**Every step includes validation checks to ensure rigor and reproducibility.**

---

## Day 1: Data Quality Assessment (Clinical Dataset)

### Task 1.1: Load and Inspect Clinical Dataset
**Time:** 30 minutes

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
clinical_path = "data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
df_clinical = pd.read_csv(clinical_path)

print(f"Dataset shape: {df_clinical.shape}")
print(f"Date range: {df_clinical['primary_date'].min()} to {df_clinical['primary_date'].max()}")
print(f"Number of studies: {df_clinical['study_source'].nunique()}")
print(f"Number of unique patients: {df_clinical['anonymous_patient_id'].nunique()}")
```

**Expected Output:**
- Shape: (11,398, 114)
- Date range: 2002-2021
- Studies: 15
- Climate coverage: 99.5%

**Validation:**
- [ ] Record counts match expected (11,398 ± 100)
- [ ] All dates parseable and within study period
- [ ] No null values in patient ID

---

### Task 1.2: Biomarker Completeness Analysis
**Time:** 1 hour

```python
# Define all biomarkers of interest (28 total)
biomarkers = {
    'immune': ['CD4 cell count (cells/µL)', 'HIV viral load (copies/mL)'],
    'hematological': ['Hematocrit (%)', 'hemoglobin_g_dL', 'Red blood cell count (×10⁶/µL)',
                      'MCV (MEAN CELL VOLUME)', 'Platelet count (×10³/µL)'],
    'hepatic': ['ALT (U/L)', 'AST (U/L)', 'Albumin (g/dL)',
                'Alkaline phosphatase (U/L)', 'Total bilirubin (mg/dL)', 'Total protein (g/dL)'],
    'lipid': ['FASTING HDL', 'FASTING LDL', 'FASTING TRIGLYCERIDES',
              'total_cholesterol_mg_dL', 'hdl_cholesterol_mg_dL', 'ldl_cholesterol_mg_dL'],
    'renal': ['creatinine_umol_L', 'creatinine clearance', 'Potassium (mEq/L)', 'Sodium (mEq/L)'],
    'cardiovascular': ['systolic_bp_mmHg', 'diastolic_bp_mmHg', 'heart_rate_bpm', 'body_temperature_celsius'],
    'metabolic': ['fasting_glucose_mmol_L', 'BMI (kg/m²)', 'Last weight recorded (kg)']
}

# Calculate completeness for each biomarker
completeness = {}
for system, markers in biomarkers.items():
    for marker in markers:
        if marker in df_clinical.columns:
            n_total = len(df_clinical)
            n_observed = df_clinical[marker].notna().sum()
            pct_complete = (n_observed / n_total) * 100

            completeness[marker] = {
                'system': system,
                'n_observed': n_observed,
                'n_missing': n_total - n_observed,
                'pct_complete': pct_complete,
                'include': n_observed >= 200 and pct_complete >= 5
            }

# Convert to DataFrame and save
df_completeness = pd.DataFrame(completeness).T
df_completeness.to_csv('results/data_quality/biomarker_completeness.csv')
print(df_completeness.sort_values('pct_complete', ascending=False))
```

**Validation Criteria:**
- [ ] All 28 biomarkers present in dataset or documented as unavailable
- [ ] Biomarkers with n ≥ 200 flagged for inclusion
- [ ] Completeness report saved to results/

**Decision Point:**
- Exclude any biomarker with < 200 observations
- Document exclusions in analysis log

---

### Task 1.3: Climate Feature Coverage
**Time:** 30 minutes

```python
# Check climate feature completeness
climate_features = [col for col in df_clinical.columns if 'climate' in col.lower()]

print(f"Number of climate features: {len(climate_features)}")

for feat in climate_features:
    n_missing = df_clinical[feat].isna().sum()
    pct_missing = (n_missing / len(df_clinical)) * 100
    print(f"{feat}: {pct_missing:.2f}% missing")

# Overall climate coverage
has_all_climate = df_clinical[climate_features].notna().all(axis=1)
print(f"\nRecords with complete climate data: {has_all_climate.sum()} ({has_all_climate.mean()*100:.2f}%)")
```

**Expected Result:**
- 16 climate features identified
- Overall coverage: 99.5% (11,337 / 11,398)

**Validation:**
- [ ] Climate coverage ≥ 95%
- [ ] Missing climate patterns random (no systematic bias by study/year)

---

### Task 1.4: Missing Data Pattern Analysis
**Time:** 1.5 hours

```python
import missingno as msno
from scipy.stats import chi2_contingency

# 1. Visualize missing data patterns
msno.matrix(df_clinical[biomarkers_flat], figsize=(20, 10))
plt.savefig('results/data_quality/missing_data_matrix.png', dpi=300, bbox_inches='tight')

# 2. Missing data heatmap (correlation between missingness)
msno.heatmap(df_clinical[biomarkers_flat], figsize=(16, 12))
plt.savefig('results/data_quality/missing_data_heatmap.png', dpi=300, bbox_inches='tight')

# 3. Test for differential missingness by study
for biomarker in high_priority_biomarkers:
    missing_by_study = pd.crosstab(
        df_clinical['study_source'],
        df_clinical[biomarker].isna()
    )

    chi2, pval, dof, expected = chi2_contingency(missing_by_study)

    print(f"{biomarker}: Chi² = {chi2:.2f}, p = {pval:.4f}")
    if pval < 0.05:
        print(f"  ⚠️ Differential missingness detected (p < 0.05)")

# 4. Little's MCAR test
# (Requires R package 'naniar' or custom implementation)
```

**Interpretation:**
- If p > 0.05 for chi-square tests → missingness appears random (MCAR/MAR)
- If p < 0.05 → differential missingness by study (MAR, need to account for study)

**Validation:**
- [ ] Missing data visualizations generated
- [ ] Differential missingness tests completed
- [ ] Missingness mechanism classified (MCAR/MAR/MNAR)

---

## Day 2: GCRO Dataset Quality Assessment

### Task 2.1: Load and Inspect GCRO Dataset
**Time:** 30 minutes

```python
# Load GCRO data
gcro_path = "data/raw/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"
df_gcro = pd.read_csv(gcro_path)

print(f"GCRO Dataset shape: {df_gcro.shape}")
print(f"Survey waves: {df_gcro['survey_wave'].unique()}")
print(f"Number of wards: {df_gcro['Ward'].nunique()}")
print(f"Date range: {df_gcro['year'].min()}-{df_gcro['year'].max()}")
```

**Expected Output:**
- Shape: (58,616, 90)
- Survey waves: 6 (2011, 2013, 2015, 2017, 2019, 2021)
- Wards: ~258

**Validation:**
- [ ] Record counts match expected (58,616 ± 1000)
- [ ] All survey waves present
- [ ] Geographic coverage spans Johannesburg wards

---

### Task 2.2: Socioeconomic Feature Selection
**Time:** 2 hours

**Objective:** Select 14 new socioeconomic features based on:
1. Theoretical relevance (literature support)
2. Data quality (≥70% completeness)
3. Variability (not constant)
4. Low correlation with existing HEAT_VULNERABILITY_SCORE (r < 0.95)

```python
# Candidate socioeconomic features
candidate_features = {
    'income': ['Q15_20_income', 'q15_3_income_recode'],
    'employment': ['EmploymentStatus', 'employment_status', 'Q11_20_employment'],
    'education': ['Education', 'std_education', 'Q15_01_education'],
    'housing': ['DwellingType', 'dwelling_type_enhanced', 'A3_dwelling'],
    'household': ['Q1_03_households'],
    'infrastructure': ['Q2_02_dwelling_dissatisfaction', 'Q2_14_Drainage', 'q2_3_sewarage'],
    'vulnerability': ['economic_vulnerability_indicator', 'employment_vulnerability_indicator',
                     'education_adaptive_capacity', 'age_vulnerability_indicator'],
    'spatial': ['dwelling_count', 'Ward'],
    'demographics': ['std_race', 'Sex']
}

# Evaluate each candidate
selected_features = []
for category, features in candidate_features.items():
    for feat in features:
        if feat not in df_gcro.columns:
            continue

        # 1. Check completeness
        completeness = df_gcro[feat].notna().mean()

        # 2. Check variability
        if df_gcro[feat].dtype in ['object', 'category']:
            n_unique = df_gcro[feat].nunique()
            variability = n_unique > 1
        else:
            variability = df_gcro[feat].std() > 0

        # 3. Check correlation with HEAT_VULNERABILITY_SCORE (if numeric)
        if df_gcro[feat].dtype in [np.float64, np.int64]:
            corr = df_gcro[[feat, 'HEAT_VULNERABILITY_SCORE']].corr().iloc[0, 1]
        else:
            corr = np.nan

        # Selection criteria
        selected = (
            completeness >= 0.70 and
            variability and
            (np.isnan(corr) or abs(corr) < 0.95)
        )

        if selected:
            selected_features.append({
                'feature': feat,
                'category': category,
                'completeness': completeness,
                'n_unique': df_gcro[feat].nunique() if variability else 0,
                'corr_with_hvs': corr
            })

df_selected = pd.DataFrame(selected_features)
print(f"Selected {len(df_selected)} socioeconomic features")
print(df_selected)

# Save selection
df_selected.to_csv('results/data_quality/selected_socioeconomic_features.csv', index=False)
```

**Expected Output:**
- 14-16 features selected
- All completeness ≥ 70%
- All correlations with HEAT_VULNERABILITY_SCORE < 0.95

**Validation:**
- [ ] 14 features selected (minimum)
- [ ] Each feature has literature justification documented
- [ ] Completeness and correlation criteria met

---

### Task 2.3: Spatial Matching Quality Assessment
**Time:** 1.5 hours

**Objective:** Assess quality of GCRO → Clinical matching

```python
from scipy.spatial.distance import cdist

# For clinical records with HEAT_VULNERABILITY_SCORE already imputed,
# assess matching quality

# 1. Calculate distances between clinical and GCRO records
clinical_coords = df_clinical[['latitude', 'longitude']].values
gcro_coords = df_gcro[['latitude', 'longitude']].values

# Find nearest GCRO record for each clinical record
distances = cdist(clinical_coords, gcro_coords, metric='euclidean')
min_distances_idx = distances.argmin(axis=1)
min_distances_km = distances.min(axis=1) * 111  # rough km conversion

# 2. Calculate matching statistics
matching_stats = {
    'n_matched': len(min_distances_km),
    'mean_distance_km': min_distances_km.mean(),
    'median_distance_km': np.median(min_distances_km),
    'within_5km': (min_distances_km <= 5).sum(),
    'within_10km': (min_distances_km <= 10).sum(),
    'within_15km': (min_distances_km <= 15).sum(),
    'max_distance_km': min_distances_km.max()
}

print("Spatial Matching Quality:")
for key, val in matching_stats.items():
    print(f"  {key}: {val}")

# 3. Visualize matching distances
plt.figure(figsize=(10, 6))
plt.hist(min_distances_km, bins=50, edgecolor='black')
plt.xlabel('Distance to nearest GCRO record (km)')
plt.ylabel('Frequency')
plt.title('Spatial Matching Distances: Clinical → GCRO')
plt.axvline(5, color='red', linestyle='--', label='5 km threshold')
plt.axvline(15, color='orange', linestyle='--', label='15 km threshold')
plt.legend()
plt.savefig('results/data_quality/spatial_matching_distances.png', dpi=300, bbox_inches='tight')
```

**Expected Results:**
- Mean distance: <5 km
- Within 15 km: 100%

**Validation:**
- [ ] Matching quality report generated
- [ ] <5% of clinical records matched >15 km away

---

## Day 3: Feature Engineering & Merging

### Task 3.1: Merge GCRO Features into Clinical Dataset
**Time:** 2 hours

```python
from sklearn.neighbors import NearestNeighbors

def merge_gcro_to_clinical(df_clinical, df_gcro, features_to_merge, max_distance_km=15):
    """
    Merge GCRO socioeconomic features to clinical records via spatial matching.

    Parameters
    ----------
    df_clinical : pd.DataFrame
        Clinical dataset with latitude, longitude
    df_gcro : pd.DataFrame
        GCRO dataset with latitude, longitude and socioeconomic features
    features_to_merge : list
        List of GCRO column names to merge
    max_distance_km : float
        Maximum matching distance in kilometers

    Returns
    -------
    df_merged : pd.DataFrame
        Clinical dataset with GCRO features added
    """

    # 1. Prepare coordinates
    clinical_coords = df_clinical[['latitude', 'longitude']].values
    gcro_coords = df_gcro[['latitude', 'longitude']].values

    # 2. Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1, metric='haversine')
    nbrs.fit(np.radians(gcro_coords))

    distances, indices = nbrs.kneighbors(np.radians(clinical_coords))
    distances_km = distances[:, 0] * 6371  # Earth radius in km

    # 3. Create merged dataset
    df_merged = df_clinical.copy()

    for feat in features_to_merge:
        df_merged[f'gcro_{feat}'] = np.nan
        valid_matches = distances_km <= max_distance_km

        matched_values = df_gcro.iloc[indices[valid_matches, 0]][feat].values
        df_merged.loc[valid_matches, f'gcro_{feat}'] = matched_values

    # 4. Add matching metadata
    df_merged['gcro_match_distance_km'] = distances_km
    df_merged['gcro_match_valid'] = distances_km <= max_distance_km

    return df_merged

# Execute merge
features_to_merge = df_selected['feature'].tolist()
df_clinical_expanded = merge_gcro_to_clinical(df_clinical, df_gcro, features_to_merge)

print(f"Original features: {df_clinical.shape[1]}")
print(f"Expanded features: {df_clinical_expanded.shape[1]}")
print(f"Features added: {df_clinical_expanded.shape[1] - df_clinical.shape[1]}")

# Check merge coverage
for feat in features_to_merge:
    coverage = df_clinical_expanded[f'gcro_{feat}'].notna().mean() * 100
    print(f"{feat}: {coverage:.1f}% coverage")

# Save expanded dataset
df_clinical_expanded.to_csv('data/processed/clinical_dataset_expanded_features.csv', index=False)
```

**Validation:**
- [ ] All 14 features successfully merged
- [ ] Merge coverage ≥ 90% for each feature
- [ ] No data leakage (temporal check: GCRO survey date ≤ clinical record date)
- [ ] Expanded dataset saved

---

### Task 3.2: Create Clean Feature Set (Climate + Socioeconomic + Demographic)
**Time:** 1 hour

```python
from leakage_checker import LeakageChecker

# Define clean feature categories
climate_features = [col for col in df_clinical_expanded.columns if 'climate' in col.lower()]

socioeconomic_features = ['HEAT_VULNERABILITY_SCORE'] + \
                        [f'gcro_{feat}' for feat in features_to_merge]

temporal_features = ['month', 'season', 'year']

demographic_features = [
    'Age (at enrolment)',
    'Sex',
    'Antiretroviral Therapy Status'
]

# Combine into master feature list
all_safe_features = (
    climate_features +
    socioeconomic_features +
    temporal_features +
    demographic_features
)

print(f"Total safe features: {len(all_safe_features)}")
print(f"  Climate: {len(climate_features)}")
print(f"  Socioeconomic: {len(socioeconomic_features)}")
print(f"  Temporal: {len(temporal_features)}")
print(f"  Demographic: {len(demographic_features)}")

# Validate no leakage
checker = LeakageChecker()
for system, biomarkers_list in biomarkers.items():
    for biomarker in biomarkers_list:
        report = checker.check_features(biomarker, all_safe_features)
        if not report.is_safe:
            print(f"⚠️ Leakage detected for {biomarker}:")
            print(f"  Biomarker leakage: {report.biomarker_leakage}")
            print(f"  Circular predictions: {report.circular_predictions}")

# Save feature list
pd.DataFrame({
    'feature': all_safe_features,
    'category': [
        'climate' if f in climate_features else
        'socioeconomic' if f in socioeconomic_features else
        'temporal' if f in temporal_features else
        'demographic'
        for f in all_safe_features
    ]
}).to_csv('results/data_quality/clean_feature_set.csv', index=False)
```

**Validation:**
- [ ] ~35 clean features identified
- [ ] Leakage checks passed for all biomarkers
- [ ] Feature list saved

---

## Day 4: Feature Validation & Quality Control

### Task 4.1: Multicollinearity Assessment (VIF)
**Time:** 1.5 hours

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Prepare numeric features only for VIF
numeric_features = df_clinical_expanded[all_safe_features].select_dtypes(include=[np.number]).columns
X_numeric = df_clinical_expanded[numeric_features].dropna()

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features
vif_data["VIF"] = [
    variance_inflation_factor(X_numeric.values, i)
    for i in range(X_numeric.shape[1])
]

vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data)

# Flag high VIF features
high_vif = vif_data[vif_data['VIF'] > 5]
if len(high_vif) > 0:
    print(f"\n⚠️ {len(high_vif)} features with VIF > 5:")
    print(high_vif)
else:
    print("\n✅ All features have VIF ≤ 5 (low multicollinearity)")

# Save VIF results
vif_data.to_csv('results/data_quality/vif_analysis.csv', index=False)
```

**Decision Rule:**
- VIF > 10: Remove feature
- VIF 5-10: Review pairwise correlations, consider removal
- VIF < 5: Keep

**Validation:**
- [ ] VIF calculated for all numeric features
- [ ] High VIF features identified and flagged
- [ ] Decision made on feature retention

---

### Task 4.2: Feature Correlation Analysis
**Time:** 1 hour

```python
# Calculate correlation matrix
corr_matrix = df_clinical_expanded[numeric_features].corr()

# Find high correlations (excluding diagonal)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val > 0.95:
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_val
            })

if high_corr_pairs:
    df_high_corr = pd.DataFrame(high_corr_pairs)
    print(f"⚠️ Found {len(df_high_corr)} feature pairs with |r| > 0.95:")
    print(df_high_corr)
    df_high_corr.to_csv('results/data_quality/high_correlation_pairs.csv', index=False)
else:
    print("✅ No feature pairs with |r| > 0.95")

# Visualize correlation matrix
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('results/data_quality/feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
```

**Validation:**
- [ ] Correlation matrix generated
- [ ] High correlation pairs (r > 0.95) identified
- [ ] Visual inspection confirms no obvious redundancy

---

### Task 4.3: Imputation Strategy Finalization
**Time:** 2 hours

```python
# For each feature with missing values, determine imputation strategy

imputation_plan = {}

for feat in all_safe_features:
    missing_rate = df_clinical_expanded[feat].isna().mean()

    if missing_rate == 0:
        strategy = 'complete'
    elif missing_rate < 0.05:
        strategy = 'median' if df_clinical_expanded[feat].dtype in [np.float64, np.int64] else 'mode'
    elif missing_rate < 0.30:
        strategy = 'knn'
    else:
        strategy = 'mice'

    imputation_plan[feat] = {
        'missing_rate': missing_rate,
        'strategy': strategy
    }

df_imputation_plan = pd.DataFrame(imputation_plan).T
print(df_imputation_plan[df_imputation_plan['missing_rate'] > 0].sort_values('missing_rate', ascending=False))

# Save imputation plan
df_imputation_plan.to_csv('results/data_quality/imputation_plan.csv')
```

**Validation:**
- [ ] Imputation strategy assigned to each feature
- [ ] Strategies appropriate for missing rate and data type

---

## Day 5: Documentation & Testing

### Task 5.1: Create Comprehensive Data Codebook
**Time:** 2 hours

```markdown
# Data Codebook - ENBEL Clinical Dataset (Expanded Features)
**Version:** 2.0 (Expanded Socioeconomic Features)
**Date:** 2025-10-30
**Records:** 11,398 clinical observations
**Features:** ~49 (original) + 14-16 (GCRO) = 63-65 total

## Climate Features (16)
| Variable | Unit | Description | Source | Missingness |
|----------|------|-------------|--------|-------------|
| climate_daily_mean_temp | °C | Daily mean temperature | ERA5 | 0.5% |
| climate_7d_mean_temp | °C | 7-day rolling mean temperature | ERA5 | 0.5% |
| ... | ... | ... | ... | ... |

## Socioeconomic Features (15)
| Variable | Type | Description | Source | Missingness |
|----------|------|-------------|--------|-------------|
| HEAT_VULNERABILITY_SCORE | Continuous | Composite vulnerability index | GCRO (imputed) | 5% |
| gcro_q15_3_income_recode | Ordinal | Income level (1-10) | GCRO QoL Survey | 8% |
| ... | ... | ... | ... | ... |

## Biomarkers (28)
[Full list with units, clinical thresholds, reference ranges]

## Temporal Features (3)
month, season, year

## Demographic Features (5)
Age, Sex, ART Status, study_source, BMI
```

**Validation:**
- [ ] Codebook includes all features
- [ ] Units specified for all continuous variables
- [ ] Sources documented
- [ ] Missingness rates accurate

---

### Task 5.2: Build Unit Tests
**Time:** 2 hours

```python
# tests/test_data_quality.py
import pytest

def test_dataset_loaded():
    """Test that dataset loads successfully."""
    df = pd.read_csv('data/processed/clinical_dataset_expanded_features.csv')
    assert len(df) > 10000, "Dataset should have >10k records"
    assert df.shape[1] > 60, "Dataset should have >60 features"

def test_no_duplicate_records():
    """Test that there are no duplicate records."""
    df = pd.read_csv('data/processed/clinical_dataset_expanded_features.csv')
    duplicates = df.duplicated(subset=['anonymous_patient_id', 'primary_date'])
    assert duplicates.sum() == 0, f"Found {duplicates.sum()} duplicate records"

def test_climate_coverage():
    """Test that climate coverage meets minimum threshold."""
    df = pd.read_csv('data/processed/clinical_dataset_expanded_features.csv')
    climate_cols = [c for c in df.columns if 'climate' in c.lower()]
    coverage = df[climate_cols].notna().all(axis=1).mean()
    assert coverage >= 0.95, f"Climate coverage {coverage:.2%} below 95% threshold"

def test_feature_leakage():
    """Test that no biomarker leakage exists."""
    from leakage_checker import LeakageChecker
    checker = LeakageChecker()

    # Test hematocrit (known leakage risk with hemoglobin)
    safe_features = load_safe_features()
    report = checker.check_features('Hematocrit (%)', safe_features)
    assert report.is_safe, f"Leakage detected: {report.biomarker_leakage}"

def test_vif_acceptable():
    """Test that all features have VIF < 10."""
    vif_df = pd.read_csv('results/data_quality/vif_analysis.csv')
    high_vif = vif_df[vif_df['VIF'] > 10]
    assert len(high_vif) == 0, f"Found {len(high_vif)} features with VIF > 10"

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Validation:**
- [ ] All unit tests written
- [ ] All tests pass
- [ ] Test coverage ≥ 80%

---

### Task 5.3: Generate Phase 1 Summary Report
**Time:** 1 hour

```python
# Generate automated summary report

report = f"""
# Phase 1 Completion Report
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Status:** ✅ Complete

## Data Quality Summary
- **Clinical records:** {len(df_clinical_expanded):,}
- **Total features:** {df_clinical_expanded.shape[1]}
  - Climate: {len(climate_features)}
  - Socioeconomic: {len(socioeconomic_features)}
  - Demographic: {len(demographic_features)}
  - Biomarkers: 28

## Feature Expansion
- **Before:** 20 features (16 climate + 1 socioeconomic + 3 temporal)
- **After:** {len(all_safe_features)} features
- **Improvement:** +{len(all_safe_features) - 20} features (+{((len(all_safe_features)-20)/20)*100:.0f}%)

## Quality Checks Passed
- ✅ Climate coverage: 99.5%
- ✅ Leakage validation: All biomarkers safe
- ✅ VIF: All features < 10
- ✅ High correlations: {len(high_corr_pairs) if high_corr_pairs else 0} pairs flagged
- ✅ Unit tests: {test_pass_rate}% passing

## Next Steps
1. Proceed to Phase 2: Model Implementation
2. Implement CatBoost, Elastic Net, Extra Trees, Gradient Boosting
3. Run pilot testing on 3 biomarkers (high/medium/low performance)

## Files Generated
- data/processed/clinical_dataset_expanded_features.csv
- results/data_quality/biomarker_completeness.csv
- results/data_quality/selected_socioeconomic_features.csv
- results/data_quality/clean_feature_set.csv
- results/data_quality/vif_analysis.csv
- results/data_quality/imputation_plan.csv
- docs/DATA_CODEBOOK_v2.md
"""

# Save report
with open('PHASE_1_COMPLETION_REPORT.md', 'w') as f:
    f.write(report)

print(report)
```

**Deliverables:**
- [ ] Phase 1 completion report generated
- [ ] All quality checks documented
- [ ] Ready to proceed to Phase 2

---

## Quality Assurance Checklist (End of Phase 1)

Before proceeding to Phase 2, verify:

### Data Quality
- [ ] Dataset has >11,000 records
- [ ] Climate coverage ≥ 99%
- [ ] All biomarkers have ≥ 200 observations (or excluded)
- [ ] No duplicate records
- [ ] All dates valid and within study period

### Feature Engineering
- [ ] 14-16 GCRO socioeconomic features successfully merged
- [ ] Merge coverage ≥ 90% for each feature
- [ ] Total features: ~35-40 (climate + socioeconomic + temporal + demographic)

### Validation
- [ ] Leakage checks passed for all 28 biomarkers
- [ ] VIF < 10 for all features (ideally < 5)
- [ ] No feature pairs with |r| > 0.95
- [ ] Imputation plan documented

### Documentation
- [ ] Data codebook updated (v2.0)
- [ ] Analysis log maintained
- [ ] All decisions justified and documented
- [ ] Unit tests written and passing

### Reproducibility
- [ ] All scripts numbered and commented
- [ ] Random seeds set (seed=42)
- [ ] File paths relative (not absolute)
- [ ] Results saved with timestamps

---

## Troubleshooting Guide

### Issue 1: Low GCRO Merge Coverage (<90%)
**Solution:**
- Increase max_distance_km to 20 km
- Use temporal matching (match by year where possible)
- Check for coordinate errors in clinical dataset

### Issue 2: High VIF (>10)
**Solution:**
- Identify correlated feature pairs
- Remove one of each highly correlated pair
- Consider creating composite indices (PCA)

### Issue 3: Leakage Detected
**Solution:**
- Review feature list carefully
- Remove flagged biomarker features
- Re-run leakage checker

### Issue 4: Unit Tests Failing
**Solution:**
- Check file paths (use absolute paths in tests)
- Verify data files are in expected locations
- Review test assertions (may need adjustment)

---

## Expected Time Breakdown

| Day | Tasks | Hours |
|-----|-------|-------|
| 1 | Clinical data quality | 3.5 |
| 2 | GCRO data quality | 4.0 |
| 3 | Feature engineering | 3.0 |
| 4 | Validation & QC | 4.5 |
| 5 | Documentation & testing | 5.0 |
| **Total** | | **20 hours** |

---

## Success Criteria

Phase 1 is complete when:
1. ✅ Expanded dataset created with 35-40 features
2. ✅ All quality checks passed
3. ✅ Leakage validation completed (zero violations)
4. ✅ Unit tests passing (≥80% coverage)
5. ✅ Documentation complete (codebook, analysis log)
6. ✅ Phase 1 completion report generated

**Ready to proceed to Phase 2: Model Implementation**
