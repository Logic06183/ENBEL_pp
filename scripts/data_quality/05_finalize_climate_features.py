"""
Finalize Clean Climate Feature Set - Phase 1, Day 1
====================================================

Create final clean climate feature set based on Option A decision:
- 6 core climate features (99.4% coverage)
- 3 temporal features (100% coverage)
- Validate final feature set
- Generate summary report

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


# Final climate feature set (Option A)
FINAL_CLIMATE_FEATURES = [
    'climate_daily_mean_temp',
    'climate_daily_max_temp',
    'climate_daily_min_temp',
    'climate_7d_mean_temp',
    'climate_heat_stress_index',
    'climate_season'
]

TEMPORAL_FEATURES = [
    'month',
    'season'
]


def validate_final_features(df):
    """
    Validate final climate feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    validation_results : dict
        Validation statistics
    """
    print("="*80)
    print("FINAL CLIMATE FEATURE SET VALIDATION")
    print("="*80)

    validation_results = {}

    print(f"\n📊 Core Climate Features (6):")
    print("-" * 80)
    for feat in FINAL_CLIMATE_FEATURES:
        if feat in df.columns:
            n_total = len(df)
            n_observed = df[feat].notna().sum()
            pct_complete = (n_observed / n_total) * 100
            status = "✅" if pct_complete >= 99.0 else "❌"

            print(f"{status} {feat:<35} {n_observed:>6,} ({pct_complete:>6.2f}%)")

            validation_results[feat] = {
                'n_observed': int(n_observed),
                'pct_complete': float(pct_complete),
                'meets_target': bool(pct_complete >= 99.0)
            }

    print(f"\n📅 Temporal Features (2):")
    print("-" * 80)
    for feat in TEMPORAL_FEATURES:
        if feat in df.columns:
            n_total = len(df)
            n_observed = df[feat].notna().sum()
            pct_complete = (n_observed / n_total) * 100
            status = "✅" if pct_complete >= 99.0 else "❌"

            print(f"{status} {feat:<35} {n_observed:>6,} ({pct_complete:>6.2f}%)")

            validation_results[feat] = {
                'n_observed': int(n_observed),
                'pct_complete': float(pct_complete),
                'meets_target': bool(pct_complete >= 99.0)
            }

    # Overall assessment
    all_features = FINAL_CLIMATE_FEATURES + TEMPORAL_FEATURES
    n_meeting_target = sum(1 for feat in all_features if validation_results.get(feat, {}).get('meets_target', False))

    print(f"\n" + "="*80)
    print(f"📈 OVERALL VALIDATION:")
    print(f"   Total features: {len(all_features)}")
    print(f"   Meeting target (≥99%): {n_meeting_target} ({(n_meeting_target/len(all_features))*100:.1f}%)")

    if n_meeting_target == len(all_features):
        print(f"   ✅ ALL FEATURES VALIDATED")
        validation_results['overall_status'] = 'pass'
    else:
        print(f"   ⚠️  Some features below target")
        validation_results['overall_status'] = 'warning'

    return validation_results


def calculate_complete_coverage(df):
    """
    Calculate records with complete climate/temporal coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    coverage_stats : dict
        Coverage statistics
    """
    print("\n" + "="*80)
    print("COMPLETE FEATURE COVERAGE")
    print("="*80)

    all_features = FINAL_CLIMATE_FEATURES + TEMPORAL_FEATURES
    available_features = [f for f in all_features if f in df.columns]

    # Records with all features
    has_all = df[available_features].notna().all(axis=1)
    n_complete = has_all.sum()
    pct_complete = (n_complete / len(df)) * 100

    print(f"\n📍 Records with ALL climate/temporal features:")
    print(f"   Complete: {n_complete:,} / {len(df):,} ({pct_complete:.2f}%)")

    if pct_complete >= 99.0:
        print(f"   ✅ EXCELLENT: ≥99% complete coverage")
    elif pct_complete >= 95.0:
        print(f"   ✅ GOOD: ≥95% complete coverage")
    else:
        print(f"   ⚠️  FAIR: <95% complete coverage")

    coverage_stats = {
        'n_total': len(df),
        'n_complete': int(n_complete),
        'pct_complete': float(pct_complete),
        'n_features': len(available_features)
    }

    return coverage_stats


def generate_feature_summary(df):
    """
    Generate summary statistics for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    feature_summary : pd.DataFrame
        Summary statistics
    """
    print("\n" + "="*80)
    print("FEATURE SUMMARY STATISTICS")
    print("="*80)

    all_features = FINAL_CLIMATE_FEATURES + TEMPORAL_FEATURES
    summary_data = []

    for feat in all_features:
        if feat in df.columns:
            if df[feat].dtype in [np.float64, np.int64, np.float32, np.int32]:
                summary = {
                    'feature': feat,
                    'type': 'continuous' if df[feat].dtype in [np.float64, np.float32] else 'discrete',
                    'n_observed': int(df[feat].notna().sum()),
                    'mean': float(df[feat].mean()) if df[feat].dtype in [np.float64, np.float32] else None,
                    'std': float(df[feat].std()) if df[feat].dtype in [np.float64, np.float32] else None,
                    'min': float(df[feat].min()) if df[feat].dtype in [np.float64, np.float32, np.int64, np.int32] else None,
                    'max': float(df[feat].max()) if df[feat].dtype in [np.float64, np.float32, np.int64, np.int32] else None
                }
            else:
                summary = {
                    'feature': feat,
                    'type': 'categorical',
                    'n_observed': int(df[feat].notna().sum()),
                    'n_unique': int(df[feat].nunique()),
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None
                }

            summary_data.append(summary)

    feature_summary = pd.DataFrame(summary_data)

    print(f"\n{feature_summary.to_string(index=False)}")

    return feature_summary


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

    # Validate final features
    validation_results = validate_final_features(df)

    # Calculate complete coverage
    coverage_stats = calculate_complete_coverage(df)

    # Generate feature summary
    feature_summary = generate_feature_summary(df)

    # Save results
    feature_summary_path = output_dir / 'final_climate_features_summary.csv'
    feature_summary.to_csv(feature_summary_path, index=False)
    print(f"\n✅ Feature summary saved to: {feature_summary_path}")

    # Generate final report
    report = {
        'timestamp': datetime.now().isoformat(),
        'decision': 'Option A: Core Features Only',
        'n_climate_features': len(FINAL_CLIMATE_FEATURES),
        'n_temporal_features': len(TEMPORAL_FEATURES),
        'n_total_features': len(FINAL_CLIMATE_FEATURES) + len(TEMPORAL_FEATURES),
        'climate_features': FINAL_CLIMATE_FEATURES,
        'temporal_features': TEMPORAL_FEATURES,
        'validation_results': validation_results,
        'coverage_stats': coverage_stats,
        'excluded_features': [
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
        ],
        'exclusion_reason': 'Low coverage (84%) and failed recomputation validation'
    }

    report_path = output_dir / 'final_climate_features_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print("FINAL CLIMATE FEATURE SET COMPLETE")
    print("="*80)
    print(f"✅ Report saved to: {report_path}")

    print("\n📌 SUMMARY:")
    print(f"   • Climate features: {len(FINAL_CLIMATE_FEATURES)}")
    print(f"   • Temporal features: {len(TEMPORAL_FEATURES)}")
    print(f"   • Total features: {len(FINAL_CLIMATE_FEATURES) + len(TEMPORAL_FEATURES)}")
    print(f"   • Coverage: {coverage_stats['pct_complete']:.2f}%")
    print(f"   • Records: {coverage_stats['n_complete']:,} / {coverage_stats['n_total']:,}")

    print("\n📌 FEATURES TO USE IN MODELING:")
    print("   Core Climate (6):")
    for feat in FINAL_CLIMATE_FEATURES:
        print(f"     - {feat}")
    print("   Temporal (2):")
    for feat in TEMPORAL_FEATURES:
        print(f"     - {feat}")

    print("\n📌 NEXT STEPS:")
    print("   1. Day 1 complete - Climate features finalized")
    print("   2. Proceed to Day 2: GCRO socioeconomic feature expansion")
    print("   3. Total feature space: 6 climate + 15 socioeconomic + 2 temporal + 5 demographic ≈ 28 features")


if __name__ == '__main__':
    main()
