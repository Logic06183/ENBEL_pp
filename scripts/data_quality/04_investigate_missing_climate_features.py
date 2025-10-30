"""
Investigate Missing Derived Climate Features
=============================================

Investigation and recovery of missing derived climate features:
- Identify why JHB_Aurum_009 missing derived features
- Check if raw temperature data exists for recomputation
- Attempt to recompute lag features (7d, 14d, 30d)
- Attempt to recompute anomaly features
- Validate recomputed features against existing data

Author: ENBEL Team
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Set random seed
np.random.seed(42)


def investigate_missing_patterns(df):
    """
    Investigate why derived climate features are missing.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    investigation_results : dict
        Investigation findings
    """
    print("="*80)
    print("INVESTIGATING MISSING DERIVED CLIMATE FEATURES")
    print("="*80)

    # Core features (should be 99%+ complete)
    core_features = [
        'climate_daily_mean_temp',
        'climate_daily_max_temp',
        'climate_daily_min_temp',
        'climate_7d_mean_temp',
        'climate_heat_stress_index',
        'climate_season'
    ]

    # Derived features (missing for ~16% of records)
    derived_features = [
        'climate_7d_max_temp',
        'climate_14d_mean_temp',
        'climate_30d_mean_temp',
        'climate_temp_anomaly',
        'climate_standardized_anomaly',
        'climate_heat_day_p90',
        'climate_heat_day_p95',
        'climate_p90_threshold',
        'climate_p95_threshold',
        'climate_p99_threshold'
    ]

    # Check JHB_Aurum_009 specifically
    df_aurum = df[df['study_source'] == 'JHB_Aurum_009'].copy()

    print(f"\n📊 JHB_Aurum_009 Analysis:")
    print(f"   Total records: {len(df_aurum):,}")

    print(f"\n🌡️  Core Climate Features:")
    for feat in core_features:
        if feat in df_aurum.columns:
            completeness = (df_aurum[feat].notna().sum() / len(df_aurum)) * 100
            print(f"   {feat:<35} {completeness:>6.2f}%")

    print(f"\n🔢 Derived Climate Features:")
    for feat in derived_features:
        if feat in df_aurum.columns:
            completeness = (df_aurum[feat].notna().sum() / len(df_aurum)) * 100
            print(f"   {feat:<35} {completeness:>6.2f}%")

    # Check if records with missing derived features have core features
    missing_derived = df_aurum[df_aurum['climate_14d_mean_temp'].isna()]
    print(f"\n🔍 Records with missing derived features: {len(missing_derived):,}")

    if len(missing_derived) > 0:
        print(f"\n   Do they have core features?")
        for feat in core_features:
            if feat in missing_derived.columns:
                has_core = missing_derived[feat].notna().sum()
                pct = (has_core / len(missing_derived)) * 100
                print(f"   {feat:<35} {has_core:>6,} ({pct:>5.1f}%)")

    # Check date distribution
    if 'primary_date_parsed' in df_aurum.columns:
        print(f"\n📅 Date Distribution (JHB_Aurum_009):")
        df_aurum['year'] = df_aurum['primary_date_parsed'].dt.year
        print(df_aurum['year'].value_counts().sort_index().to_string())

        # Check if missing derived features cluster by date
        if len(missing_derived) > 0 and 'primary_date_parsed' in missing_derived.columns:
            missing_derived['year'] = missing_derived['primary_date_parsed'].dt.year
            print(f"\n📅 Missing Derived Features by Year:")
            print(missing_derived['year'].value_counts().sort_index().to_string())

    investigation_results = {
        'study': 'JHB_Aurum_009',
        'n_records': len(df_aurum),
        'n_missing_derived': len(missing_derived),
        'core_features_present': True if len(missing_derived) > 0 else False
    }

    return investigation_results, df_aurum, missing_derived


def check_recomputation_feasibility(df_aurum, missing_derived):
    """
    Check if we can recompute derived features from core features.

    Parameters
    ----------
    df_aurum : pd.DataFrame
        JHB_Aurum_009 records
    missing_derived : pd.DataFrame
        Records with missing derived features

    Returns
    -------
    feasibility : dict
        Feasibility assessment
    """
    print("\n" + "="*80)
    print("FEASIBILITY ASSESSMENT FOR RECOMPUTATION")
    print("="*80)

    feasibility = {}

    # Check if we have daily temps for recomputing lags
    if 'climate_daily_mean_temp' in missing_derived.columns:
        has_daily = missing_derived['climate_daily_mean_temp'].notna().sum()
        pct_has_daily = (has_daily / len(missing_derived)) * 100

        print(f"\n✅ Daily mean temperature available: {has_daily:,} / {len(missing_derived):,} ({pct_has_daily:.1f}%)")

        if pct_has_daily >= 95:
            print(f"   ✅ FEASIBLE: Can recompute lag features (7d, 14d, 30d)")
            feasibility['lag_features'] = 'feasible'
        else:
            print(f"   ❌ NOT FEASIBLE: Insufficient daily temperature data")
            feasibility['lag_features'] = 'not_feasible'

    # Check if we have daily temps for anomaly computation
    # (need full time series to compute mean/std)
    if 'climate_daily_mean_temp' in df_aurum.columns:
        has_daily_full = df_aurum['climate_daily_mean_temp'].notna().sum()
        pct_has_daily_full = (has_daily_full / len(df_aurum)) * 100

        print(f"\n✅ Daily temperature (full study): {has_daily_full:,} / {len(df_aurum):,} ({pct_has_daily_full:.1f}%)")

        if pct_has_daily_full >= 95:
            print(f"   ✅ FEASIBLE: Can recompute anomaly features")
            feasibility['anomaly_features'] = 'feasible'
        else:
            print(f"   ❌ NOT FEASIBLE: Insufficient data for anomaly computation")
            feasibility['anomaly_features'] = 'not_feasible'

    # Check if we have dates for proper sorting/windowing
    if 'primary_date_parsed' in missing_derived.columns:
        has_dates = missing_derived['primary_date_parsed'].notna().sum()
        pct_has_dates = (has_dates / len(missing_derived)) * 100

        print(f"\n✅ Valid dates available: {has_dates:,} / {len(missing_derived):,} ({pct_has_dates:.1f}%)")

        if pct_has_dates == 100:
            print(f"   ✅ FEASIBLE: Can sort by date for lag computation")
            feasibility['dates'] = 'feasible'
        else:
            print(f"   ⚠️  WARNING: Some records missing dates")
            feasibility['dates'] = 'partial'

    return feasibility


def recompute_lag_features(df):
    """
    Recompute lag features (7d, 14d, 30d) from daily temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset with daily temperature and dates

    Returns
    -------
    df_recomputed : pd.DataFrame
        Dataset with recomputed lag features
    """
    print("\n" + "="*80)
    print("RECOMPUTING LAG FEATURES")
    print("="*80)

    df_recomputed = df.copy()

    # Ensure dates are parsed
    if 'primary_date_parsed' not in df_recomputed.columns:
        df_recomputed['primary_date_parsed'] = pd.to_datetime(df_recomputed['primary_date'], errors='coerce')

    # Sort by patient and date for proper windowing
    df_recomputed = df_recomputed.sort_values(['anonymous_patient_id', 'primary_date_parsed'])

    print(f"\n📊 Recomputing lag features for {len(df_recomputed):,} records...")

    # Recompute 7-day max temperature
    if 'climate_daily_max_temp' in df_recomputed.columns:
        print(f"   Computing climate_7d_max_temp...")
        df_recomputed['climate_7d_max_temp_recomputed'] = df_recomputed.groupby('anonymous_patient_id')['climate_daily_max_temp'].transform(
            lambda x: x.rolling(window=7, min_periods=1).max()
        )
        n_recomputed = df_recomputed['climate_7d_max_temp_recomputed'].notna().sum()
        print(f"   ✅ Recomputed: {n_recomputed:,} values")

    # Recompute 14-day mean temperature
    if 'climate_daily_mean_temp' in df_recomputed.columns:
        print(f"   Computing climate_14d_mean_temp...")
        df_recomputed['climate_14d_mean_temp_recomputed'] = df_recomputed.groupby('anonymous_patient_id')['climate_daily_mean_temp'].transform(
            lambda x: x.rolling(window=14, min_periods=1).mean()
        )
        n_recomputed = df_recomputed['climate_14d_mean_temp_recomputed'].notna().sum()
        print(f"   ✅ Recomputed: {n_recomputed:,} values")

    # Recompute 30-day mean temperature
    if 'climate_daily_mean_temp' in df_recomputed.columns:
        print(f"   Computing climate_30d_mean_temp...")
        df_recomputed['climate_30d_mean_temp_recomputed'] = df_recomputed.groupby('anonymous_patient_id')['climate_daily_mean_temp'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        n_recomputed = df_recomputed['climate_30d_mean_temp_recomputed'].notna().sum()
        print(f"   ✅ Recomputed: {n_recomputed:,} values")

    return df_recomputed


def validate_recomputed_features(df_recomputed):
    """
    Validate recomputed features against original (where available).

    Parameters
    ----------
    df_recomputed : pd.DataFrame
        Dataset with both original and recomputed features

    Returns
    -------
    validation_results : dict
        Validation statistics
    """
    print("\n" + "="*80)
    print("VALIDATING RECOMPUTED FEATURES")
    print("="*80)

    validation_results = {}

    # Compare recomputed vs original for records that have both
    features_to_validate = [
        ('climate_7d_max_temp', 'climate_7d_max_temp_recomputed'),
        ('climate_14d_mean_temp', 'climate_14d_mean_temp_recomputed'),
        ('climate_30d_mean_temp', 'climate_30d_mean_temp_recomputed')
    ]

    for original, recomputed in features_to_validate:
        if original in df_recomputed.columns and recomputed in df_recomputed.columns:
            # Find records with both values
            both_present = df_recomputed[[original, recomputed]].dropna()

            if len(both_present) > 0:
                # Calculate correlation
                corr = both_present[original].corr(both_present[recomputed])

                # Calculate mean absolute error
                mae = np.abs(both_present[original] - both_present[recomputed]).mean()

                # Calculate RMSE
                rmse = np.sqrt(((both_present[original] - both_present[recomputed])**2).mean())

                print(f"\n📊 {original}:")
                print(f"   Records with both values: {len(both_present):,}")
                print(f"   Correlation: {corr:.4f}")
                print(f"   MAE: {mae:.4f} °C")
                print(f"   RMSE: {rmse:.4f} °C")

                if corr > 0.99 and mae < 0.5:
                    print(f"   ✅ VALIDATION PASSED: Excellent agreement")
                    validation_results[original] = 'pass'
                elif corr > 0.95:
                    print(f"   ⚠️  VALIDATION WARNING: Good but not perfect agreement")
                    validation_results[original] = 'warning'
                else:
                    print(f"   ❌ VALIDATION FAILED: Poor agreement")
                    validation_results[original] = 'fail'
            else:
                print(f"\n⚠️  {original}: No overlapping values for validation")
                validation_results[original] = 'no_overlap'

    return validation_results


def fill_missing_with_recomputed(df_recomputed):
    """
    Fill missing derived features with recomputed values.

    Parameters
    ----------
    df_recomputed : pd.DataFrame
        Dataset with recomputed features

    Returns
    -------
    df_filled : pd.DataFrame
        Dataset with missing values filled
    """
    print("\n" + "="*80)
    print("FILLING MISSING VALUES WITH RECOMPUTED FEATURES")
    print("="*80)

    df_filled = df_recomputed.copy()

    fill_mappings = {
        'climate_7d_max_temp': 'climate_7d_max_temp_recomputed',
        'climate_14d_mean_temp': 'climate_14d_mean_temp_recomputed',
        'climate_30d_mean_temp': 'climate_30d_mean_temp_recomputed'
    }

    for original, recomputed in fill_mappings.items():
        if original in df_filled.columns and recomputed in df_filled.columns:
            # Count missing before
            missing_before = df_filled[original].isna().sum()

            # Fill missing with recomputed
            df_filled[original] = df_filled[original].fillna(df_filled[recomputed])

            # Count missing after
            missing_after = df_filled[original].isna().sum()
            filled = missing_before - missing_after

            print(f"\n{original}:")
            print(f"   Missing before: {missing_before:,}")
            print(f"   Missing after: {missing_after:,}")
            print(f"   Filled: {filled:,} ({(filled/missing_before)*100:.1f}% of missing)")

    # Calculate new overall climate coverage
    climate_features = [col for col in df_filled.columns if col.startswith('climate_') and not col.endswith('_recomputed')]
    has_all_climate = df_filled[climate_features].notna().all(axis=1)
    new_coverage = (has_all_climate.sum() / len(df_filled)) * 100

    print(f"\n📊 OVERALL CLIMATE COVERAGE:")
    print(f"   Before recomputation: 84.13%")
    print(f"   After recomputation: {new_coverage:.2f}%")
    print(f"   Improvement: +{new_coverage - 84.13:.2f} percentage points")

    if new_coverage >= 99.0:
        print(f"   ✅ TARGET MET: ≥99.0% coverage achieved!")
    elif new_coverage >= 95.0:
        print(f"   ⚠️  CLOSE: ≥95.0% coverage, approaching target")
    else:
        print(f"   ❌ TARGET NOT MET: Still below 95% coverage")

    return df_filled, new_coverage


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features\n")

    # Parse dates
    if 'primary_date' in df.columns:
        df['primary_date_parsed'] = pd.to_datetime(df['primary_date'], errors='coerce')

    # Step 1: Investigate missing patterns
    investigation_results, df_aurum, missing_derived = investigate_missing_patterns(df)

    # Step 2: Check feasibility
    feasibility = check_recomputation_feasibility(df_aurum, missing_derived)

    # Step 3: Attempt recomputation if feasible
    if feasibility.get('lag_features') == 'feasible':
        df_recomputed = recompute_lag_features(df)

        # Step 4: Validate
        validation_results = validate_recomputed_features(df_recomputed)

        # Step 5: Fill missing values
        df_filled, new_coverage = fill_missing_with_recomputed(df_recomputed)

        # Save results
        if new_coverage >= 95.0:
            output_path = base_dir.parent / "data" / "processed" / "CLINICAL_DATASET_CLIMATE_RECOMPUTED.csv"
            df_filled.to_csv(output_path, index=False)
            print(f"\n✅ Saved enhanced dataset to: {output_path}")
        else:
            print(f"\n⚠️  Coverage still below 95%, not saving")
    else:
        print(f"\n❌ Cannot proceed with recomputation (feasibility check failed)")

    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'investigation': investigation_results,
        'feasibility': feasibility,
        'validation': validation_results if 'validation_results' in locals() else None,
        'new_coverage': float(new_coverage) if 'new_coverage' in locals() else None
    }

    summary_path = output_dir / 'climate_recomputation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to: {summary_path}")

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
