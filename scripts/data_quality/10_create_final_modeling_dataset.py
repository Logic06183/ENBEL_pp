"""
Create Final Modeling Dataset - Scenario B (Climate + Heat Vulnerability)
==========================================================================

Create clean modeling dataset using existing imputation AS-IS:
- 6 climate features (99.4% coverage)
- 2 temporal features (100% coverage)
- 1 socioeconomic feature (75.9% coverage)

Total: 9 features
Expected records: ~8,586 (75.3% of 11,398)

NO NEW IMPUTATION - using existing work only.

Author: ENBEL Team
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
np.random.seed(42)


# FINAL FEATURE SET: SCENARIO B
FINAL_FEATURES = [
    # Climate (6)
    'climate_daily_mean_temp',
    'climate_daily_max_temp',
    'climate_daily_min_temp',
    'climate_7d_mean_temp',
    'climate_heat_stress_index',
    'climate_season',
    # Temporal (2)
    'month',
    'season',
    # Socioeconomic (1)
    'HEAT_VULNERABILITY_SCORE'
]

# Metadata columns to retain
METADATA_COLS = [
    'anonymous_patient_id',
    'study_source',
    'primary_date',
    'primary_date_parsed',
    'year',
    'latitude',
    'longitude'
]

# All 27 biomarkers for modeling
BIOMARKERS = [
    'CD4 cell count (cells/µL)',
    'HIV viral load (copies/mL)',
    'Hematocrit (%)',
    'hemoglobin_g_dL',
    'Platelets (×10³/µL)',
    'White blood cell count (×10³/µL)',
    'Lymphocyte count (×10³/µL)',
    'Neutrophil count (×10³/µL)',
    'Total Cholesterol (mg/dL)',
    'LDL Cholesterol (mg/dL)',
    'HDL Cholesterol (mg/dL)',
    'FASTING LDL',
    'FASTING HDL',
    'FASTING TRIGLYCERIDES',
    'Triglycerides (mg/dL)',
    'fasting_glucose_mmol_L',
    'creatinine_umol_L',
    'ALT (U/L)',
    'AST (U/L)',
    'Systolic blood pressure (mmHg)',
    'Diastolic blood pressure (mmHg)',
    'BMI (kg/m²)',
    'weight_kg',
    'Alkaline phosphatase (U/L)',
    'Albumin (g/dL)',
    'Bilirubin (mg/dL)',
    'Calcium (mg/dL)'
]


def load_and_filter_dataset(data_path):
    """
    Load dataset and filter to complete cases for Scenario B.

    Parameters
    ----------
    data_path : Path
        Path to clinical dataset

    Returns
    -------
    df_complete : pd.DataFrame
        Filtered dataset with all 9 features present
    """
    print("="*80)
    print("CREATING FINAL MODELING DATASET - SCENARIO B")
    print("="*80)

    # Load full dataset
    print(f"\nLoading from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Full dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features")

    # Parse date if needed
    if 'primary_date_parsed' not in df.columns and 'primary_date' in df.columns:
        df['primary_date_parsed'] = pd.to_datetime(df['primary_date'], errors='coerce')

    # Check feature availability
    print(f"\n📊 Checking feature availability:")
    missing_features = [f for f in FINAL_FEATURES if f not in df.columns]
    if missing_features:
        print(f"   ❌ Missing features: {missing_features}")
        raise ValueError(f"Required features not found: {missing_features}")

    print(f"   ✅ All {len(FINAL_FEATURES)} features found")

    # Filter to complete cases (all 9 features present)
    print(f"\n📍 Filtering to complete cases:")
    print(f"   Features required: {len(FINAL_FEATURES)}")

    has_all_features = df[FINAL_FEATURES].notna().all(axis=1)
    df_complete = df[has_all_features].copy()

    n_removed = len(df) - len(df_complete)
    pct_retained = (len(df_complete) / len(df)) * 100

    print(f"   Original records: {len(df):,}")
    print(f"   Complete cases: {len(df_complete):,}")
    print(f"   Removed: {n_removed:,} ({100-pct_retained:.1f}%)")
    print(f"   Retained: {pct_retained:.1f}%")

    if pct_retained < 70:
        print(f"   ⚠️  WARNING: Retained <70% of data")
    else:
        print(f"   ✅ Good retention (≥70%)")

    return df_complete


def validate_final_dataset(df_complete):
    """
    Validate the final modeling dataset.

    Parameters
    ----------
    df_complete : pd.DataFrame
        Complete case dataset

    Returns
    -------
    validation_results : dict
        Validation statistics
    """
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80)

    validation_results = {}

    # 1. Check completeness of final features
    print(f"\n✅ Feature Completeness Check:")
    for feat in FINAL_FEATURES:
        completeness = df_complete[feat].notna().mean() * 100
        print(f"   {feat:<45} {completeness:.1f}%")

        if completeness < 100:
            print(f"      ⚠️  WARNING: Not 100% complete in filtered dataset!")

    # 2. Check biomarker availability
    print(f"\n📊 Biomarker Availability:")
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

    df_biomarker_stats = pd.DataFrame(biomarker_stats)
    n_sufficient = df_biomarker_stats['sufficient'].sum()

    print(f"   Total biomarkers: {len(df_biomarker_stats)}")
    print(f"   With ≥200 observations: {n_sufficient}")
    print(f"   With <200 observations: {len(df_biomarker_stats) - n_sufficient}")

    # 3. Check temporal coverage
    print(f"\n📅 Temporal Coverage:")
    year_range = (df_complete['year'].min(), df_complete['year'].max())
    print(f"   Year range: {int(year_range[0])}-{int(year_range[1])}")

    year_counts = df_complete['year'].value_counts().sort_index()
    print(f"   Years with data: {len(year_counts)}")
    print(f"   Records per year (mean): {year_counts.mean():.0f}")

    # 4. Check geographic coverage
    if 'latitude' in df_complete.columns and 'longitude' in df_complete.columns:
        print(f"\n🗺️  Geographic Coverage:")
        lat_range = (df_complete['latitude'].min(), df_complete['latitude'].max())
        lon_range = (df_complete['longitude'].min(), df_complete['longitude'].max())
        print(f"   Latitude range: [{lat_range[0]:.4f}, {lat_range[1]:.4f}]")
        print(f"   Longitude range: [{lon_range[0]:.4f}, {lon_range[1]:.4f}]")

    # 5. Check study distribution
    if 'study_source' in df_complete.columns:
        print(f"\n📚 Study Distribution:")
        n_studies = df_complete['study_source'].nunique()
        print(f"   Unique studies: {n_studies}")

        study_counts = df_complete['study_source'].value_counts()
        print(f"   Records per study (mean): {study_counts.mean():.0f}")
        print(f"   Records per study (median): {study_counts.median():.0f}")

        # Top 5 studies
        print(f"\n   Top 5 studies by record count:")
        for study, count in study_counts.head(5).items():
            print(f"      {study}: {count:,} records")

    validation_results = {
        'n_records': len(df_complete),
        'n_features': len(FINAL_FEATURES),
        'n_biomarkers': len(df_biomarker_stats),
        'n_biomarkers_sufficient': int(n_sufficient),
        'year_range': [int(year_range[0]), int(year_range[1])],
        'n_studies': int(n_studies) if 'study_source' in df_complete.columns else None
    }

    return validation_results, df_biomarker_stats


def generate_summary_statistics(df_complete):
    """Generate summary statistics for final features."""

    print("\n" + "="*80)
    print("FEATURE SUMMARY STATISTICS")
    print("="*80)

    stats_list = []

    for feat in FINAL_FEATURES:
        stats = {'feature': feat}

        if df_complete[feat].dtype in ['object', 'category']:
            stats['type'] = 'categorical'
            stats['n_unique'] = int(df_complete[feat].nunique())
            stats['top_value'] = df_complete[feat].mode()[0]
            stats['top_count'] = int(df_complete[feat].value_counts().iloc[0])
        else:
            stats['type'] = 'continuous'
            stats['mean'] = float(df_complete[feat].mean())
            stats['std'] = float(df_complete[feat].std())
            stats['min'] = float(df_complete[feat].min())
            stats['max'] = float(df_complete[feat].max())
            stats['median'] = float(df_complete[feat].median())

        stats_list.append(stats)

    df_stats = pd.DataFrame(stats_list)

    print(f"\n{df_stats.to_string(index=False)}")

    return df_stats


def save_final_dataset(df_complete, output_dir):
    """Save the final modeling dataset."""

    print("\n" + "="*80)
    print("SAVING FINAL DATASET")
    print("="*80)

    # Select columns to save
    cols_to_save = FINAL_FEATURES + METADATA_COLS + BIOMARKERS

    # Filter to existing columns
    cols_to_save = [c for c in cols_to_save if c in df_complete.columns]

    df_final = df_complete[cols_to_save].copy()

    # Save to CSV
    output_path = output_dir / 'MODELING_DATASET_SCENARIO_B.csv'
    df_final.to_csv(output_path, index=False)

    print(f"\n✅ Final dataset saved to: {output_path}")
    print(f"   Records: {len(df_final):,}")
    print(f"   Columns: {len(df_final.columns)}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    output_dir = base_dir / "results" / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter
    df_complete = load_and_filter_dataset(data_path)

    # Validate
    validation_results, df_biomarker_stats = validate_final_dataset(df_complete)

    # Summary statistics
    df_stats = generate_summary_statistics(df_complete)

    # Save final dataset
    output_path = save_final_dataset(df_complete, output_dir)

    # Save validation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'scenario': 'B: Climate + Heat Vulnerability',
        'n_features': len(FINAL_FEATURES),
        'features': FINAL_FEATURES,
        'validation': validation_results,
        'feature_statistics': df_stats.to_dict('records'),
        'biomarker_statistics': df_biomarker_stats.to_dict('records'),
        'output_file': str(output_path)
    }

    report_path = output_dir / 'modeling_dataset_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "="*80)
    print("FINAL MODELING DATASET CREATED")
    print("="*80)
    print(f"✅ Report saved to: {report_path}")

    print("\n📌 SCENARIO B SUMMARY:")
    print(f"   Features: {len(FINAL_FEATURES)}")
    print(f"     • Climate: 6")
    print(f"     • Temporal: 2")
    print(f"     • Socioeconomic: 1 (HEAT_VULNERABILITY_SCORE)")
    print(f"   Records: {len(df_complete):,}")
    print(f"   Biomarkers: {len(df_biomarker_stats)} total, {validation_results['n_biomarkers_sufficient']} with ≥200 obs")
    print(f"   Coverage: {validation_results['year_range'][0]}-{validation_results['year_range'][1]}")

    print("\n📌 NEXT STEPS:")
    print("   1. ✅ Final dataset created")
    print("   2. ➡️  Run automated leakage checks")
    print("   3. ➡️  Feature validation (VIF, correlation)")
    print("   4. ➡️  Begin model optimization!")

    print("\n🎯 READY FOR PHASE 2: MODEL OPTIMIZATION")


if __name__ == '__main__':
    main()
