"""
Create Scenario B Dataset: Climate + Vulnerability + Demographics
NO IMPUTATION - only complete cases

Author: ENBEL Team
Date: 2025-10-30
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("="*80)
print("CREATING SCENARIO B: FULL MODEL DATASET")
print("="*80)

# Load original dataset
data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
print(f"\nLoading: {data_path}")
df = pd.read_csv(data_path, low_memory=False)
print(f"✅ Loaded: {len(df):,} records")

# Define feature sets
CLIMATE_FEATURES = [
    'climate_daily_mean_temp',
    'climate_daily_max_temp',
    'climate_daily_min_temp',
    'climate_7d_mean_temp',
    'climate_heat_stress_index',
    'climate_season'
]

TEMPORAL_FEATURES = ['month', 'season']
SOCIOECONOMIC_FEATURES = ['HEAT_VULNERABILITY_SCORE']
DEMOGRAPHIC_FEATURES = ['Age (at enrolment)', 'Sex']
STUDY_FEATURES = ['study_source', 'year']

SCENARIO_B_FEATURES = (CLIMATE_FEATURES + TEMPORAL_FEATURES +
                       SOCIOECONOMIC_FEATURES + DEMOGRAPHIC_FEATURES +
                       STUDY_FEATURES)

print(f"\n📋 Scenario B Features ({len(SCENARIO_B_FEATURES)} total):")
print(f"   • Climate: {len(CLIMATE_FEATURES)}")
print(f"   • Temporal: {len(TEMPORAL_FEATURES)}")
print(f"   • Socioeconomic: {len(SOCIOECONOMIC_FEATURES)}")
print(f"   • Demographic: {len(DEMOGRAPHIC_FEATURES)}")
print(f"   • Study: {len(STUDY_FEATURES)}")

# Check feature availability
print(f"\n📊 Feature Completeness:")
missing_features = []
for feat in SCENARIO_B_FEATURES:
    if feat not in df.columns:
        missing_features.append(feat)
        print(f"   ❌ {feat:<45} NOT FOUND")
    else:
        completeness = df[feat].notna().mean() * 100
        status = "✅" if completeness == 100 else "⚠️ "
        print(f"   {status} {feat:<45} {completeness:.1f}%")

if missing_features:
    print(f"\n❌ ERROR: Missing features: {missing_features}")
    import sys
    sys.exit(1)

# Filter to complete cases
print(f"\n🔍 Filtering to complete cases...")
has_all_features = df[SCENARIO_B_FEATURES].notna().all(axis=1)
df_complete = df[has_all_features].copy()

n_removed = len(df) - len(df_complete)
pct_retained = (len(df_complete) / len(df)) * 100

print(f"   Original: {len(df):,}")
print(f"   Complete: {len(df_complete):,}")
print(f"   Removed: {n_removed:,} ({100-pct_retained:.1f}%)")
print(f"   Retained: {pct_retained:.1f}%")

# Top 10 Biomarkers
BIOMARKERS = [
    'CD4 cell count (cells/µL)',
    'Hematocrit (%)',
    'FASTING LDL',
    'FASTING HDL',
    'Albumin (g/dL)',
    'creatinine_umol_L',
    'White blood cell count (×10³/µL)',
    'Lymphocyte count (×10³/µL)',
    'Neutrophil count (×10³/µL)',
    'weight_kg'
]

# Check biomarker availability
print(f"\n📊 Top 10 Biomarker Availability in Scenario B:")
biomarker_stats = []
for biomarker in BIOMARKERS:
    if biomarker in df_complete.columns:
        n_obs = df_complete[biomarker].notna().sum()
        pct = (n_obs / len(df_complete)) * 100
        biomarker_stats.append({
            'biomarker': biomarker,
            'n_observations': n_obs,
            'pct_complete': pct,
            'sufficient': n_obs >= 200
        })
        status = "✅" if n_obs >= 200 else "❌"
        print(f"   {status} {biomarker:<45} {n_obs:>5} obs ({pct:>5.1f}%)")

df_biomarker_stats = pd.DataFrame(biomarker_stats)
n_sufficient = df_biomarker_stats['sufficient'].sum()
print(f"\n   Total: {len(df_biomarker_stats)} biomarkers")
print(f"   Sufficient (≥200): {n_sufficient}")

# Save dataset
base_dir = Path(__file__).resolve().parents[1]
output_dir = base_dir / "results" / "modeling"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "MODELING_DATASET_SCENARIO_B_FULL.csv"

# Select columns
metadata_cols = ['anonymous_patient_id', 'study_source', 'primary_date',
                 'primary_date_parsed', 'year', 'latitude', 'longitude']
cols_to_save = SCENARIO_B_FEATURES + metadata_cols + BIOMARKERS
cols_to_save = [c for c in cols_to_save if c in df_complete.columns]

df_final = df_complete[cols_to_save].copy()
df_final.to_csv(output_path, index=False)

print(f"\n✅ Dataset saved: {output_path}")
print(f"   Records: {len(df_final):,}")
print(f"   Columns: {len(df_final.columns)}")
print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save report
report = {
    'timestamp': datetime.now().isoformat(),
    'scenario': 'B: Climate + Vulnerability + Demographics (Full Model)',
    'n_features': len(SCENARIO_B_FEATURES),
    'features': SCENARIO_B_FEATURES,
    'n_records': len(df_final),
    'pct_retained': pct_retained,
    'n_biomarkers': len(df_biomarker_stats),
    'n_biomarkers_sufficient': int(n_sufficient),
    'biomarker_statistics': df_biomarker_stats.to_dict('records'),
    'output_file': str(output_path)
}

report_path = output_dir / 'modeling_dataset_scenario_b_full_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"✅ Report saved: {report_path}")

print("\n" + "="*80)
print("SCENARIO B DATASET READY FOR MODELING")
print("="*80)
