"""
Climate Feature Coverage Verification - Phase 1, Day 1, Task 1.3
================================================================

Comprehensive assessment of climate feature availability:
- Verify all 16 climate features are present
- Calculate completeness for each climate feature
- Confirm 99.5% coverage target met
- Analyze patterns in missing climate data
- Generate coverage summary

Target: ≥99% coverage for all climate features

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
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Set random seed
np.random.seed(42)


# Expected climate features (from ERA5 reanalysis)
EXPECTED_CLIMATE_FEATURES = [
    'climate_daily_mean_temp',
    'climate_daily_max_temp',
    'climate_daily_min_temp',
    'climate_7d_mean_temp',
    'climate_7d_max_temp',
    'climate_14d_mean_temp',
    'climate_30d_mean_temp',
    'climate_temp_anomaly',
    'climate_standardized_anomaly',
    'climate_heat_day_p90',
    'climate_heat_day_p95',
    'climate_heat_stress_index',
    'climate_p90_threshold',
    'climate_p95_threshold',
    'climate_p99_threshold',
    'climate_season'
]


def identify_climate_features(df):
    """
    Identify all climate-related features in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    climate_features : list
        List of climate feature column names
    """
    # Only use features that start with 'climate_' (ERA5-derived)
    climate_features = [col for col in df.columns if col.startswith('climate_')]

    return climate_features


def analyze_climate_coverage(df, climate_features):
    """
    Analyze coverage for each climate feature.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    climate_features : list
        List of climate feature names

    Returns
    -------
    coverage_df : pd.DataFrame
        Coverage statistics for each climate feature
    """
    print("="*80)
    print("CLIMATE FEATURE COVERAGE ANALYSIS")
    print("="*80)

    coverage_results = []

    for feature in climate_features:
        if feature in df.columns:
            n_total = len(df)
            n_observed = df[feature].notna().sum()
            n_missing = n_total - n_observed
            pct_complete = (n_observed / n_total) * 100

            coverage_results.append({
                'feature': feature,
                'n_total': n_total,
                'n_observed': n_observed,
                'n_missing': n_missing,
                'pct_complete': pct_complete,
                'meets_target': pct_complete >= 99.0
            })
        else:
            print(f"⚠️  Feature not found in dataset: {feature}")

    coverage_df = pd.DataFrame(coverage_results)
    coverage_df = coverage_df.sort_values('pct_complete', ascending=False)

    return coverage_df


def print_coverage_summary(coverage_df, target_pct=99.0):
    """Print climate feature coverage summary."""

    print(f"\n📊 CLIMATE FEATURE COVERAGE (Target: ≥{target_pct}%)")
    print("="*80)

    for _, row in coverage_df.iterrows():
        status = "✅" if row['meets_target'] else "❌"
        print(f"{status} {row['feature']:<40} "
              f"n={row['n_observed']:>6,} ({row['pct_complete']:>6.2f}%)")

    # Overall summary
    n_total_features = len(coverage_df)
    n_meeting_target = coverage_df['meets_target'].sum()

    print("\n" + "="*80)
    print(f"📈 OVERALL SUMMARY:")
    print(f"   Total climate features: {n_total_features}")
    print(f"   Meeting target (≥{target_pct}%): {n_meeting_target} ({(n_meeting_target/n_total_features)*100:.1f}%)")
    print(f"   Below target: {n_total_features - n_meeting_target}")

    # Calculate overall climate coverage
    # (records with ALL climate features present)
    return coverage_df


def analyze_complete_climate_coverage(df, climate_features):
    """
    Analyze records with complete climate data (all features present).

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    climate_features : list
        List of climate feature names

    Returns
    -------
    complete_stats : dict
        Statistics for complete climate coverage
    """
    print("\n" + "="*80)
    print("COMPLETE CLIMATE COVERAGE (All 16 Features)")
    print("="*80)

    # Check which records have ALL climate features
    climate_data = df[climate_features]
    has_all_climate = climate_data.notna().all(axis=1)

    n_total = len(df)
    n_complete = has_all_climate.sum()
    pct_complete = (n_complete / n_total) * 100

    complete_stats = {
        'n_total': n_total,
        'n_complete': n_complete,
        'n_missing_any': n_total - n_complete,
        'pct_complete': pct_complete
    }

    print(f"\n📍 Records with complete climate data: {n_complete:,} ({pct_complete:.2f}%)")
    print(f"📍 Records missing any climate feature: {n_total - n_complete:,} ({100-pct_complete:.2f}%)")

    if pct_complete >= 99.5:
        print(f"\n✅ Complete climate coverage target met (≥99.5%)")
    elif pct_complete >= 99.0:
        print(f"\n⚠️  Complete climate coverage slightly below target (99.0-99.5%)")
    else:
        print(f"\n❌ Complete climate coverage below target (<99.0%)")

    return complete_stats, has_all_climate


def analyze_missing_patterns(df, climate_features, has_all_climate):
    """
    Analyze patterns in missing climate data.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    climate_features : list
        List of climate feature names
    has_all_climate : pd.Series
        Boolean series indicating complete climate coverage

    Returns
    -------
    missing_patterns : dict
        Missing data pattern analysis
    """
    print("\n" + "="*80)
    print("MISSING CLIMATE DATA PATTERNS")
    print("="*80)

    # Identify records with missing climate data
    df_missing = df[~has_all_climate].copy()

    if len(df_missing) == 0:
        print("\n✅ No missing climate data detected!")
        return None

    print(f"\nAnalyzing {len(df_missing)} records with missing climate data...")

    # By study
    if 'study_source' in df.columns:
        missing_by_study = df_missing.groupby('study_source').size().sort_values(ascending=False)
        print(f"\nMissing climate by study (Top 10):")
        print(missing_by_study.head(10).to_string())

    # By year
    if 'primary_date_parsed' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df_missing['primary_date_parsed']):
            df_missing_copy = df_missing.copy()
            df_missing_copy['year'] = df_missing_copy['primary_date_parsed'].dt.year
            missing_by_year = df_missing_copy.groupby('year').size().sort_values(ascending=False)
            print(f"\nMissing climate by year:")
            print(missing_by_year.to_string())
        else:
            missing_by_year = None
    else:
        missing_by_year = None

    # Which specific features are missing most often
    missing_counts = df_missing[climate_features].isna().sum().sort_values(ascending=False)
    print(f"\nMost frequently missing climate features:")
    print(missing_counts.head(10).to_string())

    missing_patterns = {
        'n_records_missing': len(df_missing),
        'missing_by_study': missing_by_study.to_dict() if 'study_source' in df.columns else None,
        'missing_by_year': missing_by_year.to_dict() if missing_by_year is not None else None,
        'most_missing_features': missing_counts.head(10).to_dict()
    }

    return missing_patterns


def visualize_climate_coverage(coverage_df, output_path):
    """
    Create visualizations of climate feature coverage.

    Parameters
    ----------
    coverage_df : pd.DataFrame
        Coverage statistics
    output_path : Path
        Output directory for plots
    """
    print("\n📊 GENERATING VISUALIZATIONS...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top panel: Coverage percentage
    ax1 = axes[0]
    colors = ['green' if x else 'red' for x in coverage_df['meets_target']]

    coverage_df.plot(
        kind='barh',
        x='feature',
        y='pct_complete',
        ax=ax1,
        color=colors,
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )

    ax1.axvline(99.0, color='blue', linestyle='--', linewidth=2, label='99% target')
    ax1.axvline(99.5, color='darkblue', linestyle='--', linewidth=2, label='99.5% target')
    ax1.set_xlabel('Completeness (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Climate Feature', fontsize=12, fontweight='bold')
    ax1.set_title('Climate Feature Coverage - All 16 Features', fontsize=14, fontweight='bold')
    ax1.set_xlim([98.5, 100.5])  # Zoom into relevant range
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Bottom panel: Missing counts
    ax2 = axes[1]
    coverage_df.plot(
        kind='barh',
        x='feature',
        y='n_missing',
        ax=ax2,
        color=colors,
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )

    ax2.set_xlabel('Number of Missing Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Climate Feature', fontsize=12, fontweight='bold')
    ax2.set_title('Climate Feature Missing Values', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'climate_coverage_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: climate_coverage_overview.png")


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if data file exists
    if not data_path.exists():
        print(f"❌ Data file not found at {data_path}")
        return

    # Load dataset
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features\n")

    # Parse dates if needed
    if 'primary_date' in df.columns and 'primary_date_parsed' not in df.columns:
        df['primary_date_parsed'] = pd.to_datetime(df['primary_date'], errors='coerce')

    # Identify climate features
    climate_features = identify_climate_features(df)
    print(f"\n🌡️  Identified {len(climate_features)} climate features in dataset")
    print(f"   Expected: {len(EXPECTED_CLIMATE_FEATURES)} features")

    # Check if all expected features are present
    missing_expected = set(EXPECTED_CLIMATE_FEATURES) - set(climate_features)
    if missing_expected:
        print(f"\n⚠️  Missing expected features: {missing_expected}")

    # Analyze coverage
    coverage_df = analyze_climate_coverage(df, climate_features)
    print_coverage_summary(coverage_df)

    # Save coverage table
    coverage_path = output_dir / 'climate_feature_coverage.csv'
    coverage_df.to_csv(coverage_path, index=False)
    print(f"\n✅ Coverage table saved to: {coverage_path}")

    # Analyze complete climate coverage
    complete_stats, has_all_climate = analyze_complete_climate_coverage(df, climate_features)

    # Analyze missing patterns
    missing_patterns = analyze_missing_patterns(df, climate_features, has_all_climate)

    # Visualize
    visualize_climate_coverage(coverage_df, output_dir)

    # Generate summary report (convert numpy types to native Python)
    def convert_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/03_climate_feature_coverage.py',
        'n_climate_features': len(climate_features),
        'n_expected_features': len(EXPECTED_CLIMATE_FEATURES),
        'feature_coverage': convert_types(coverage_df.to_dict('records')),
        'complete_coverage': convert_types(complete_stats),
        'missing_patterns': convert_types(missing_patterns),
        'target_met': bool(complete_stats['pct_complete'] >= 99.0)
    }

    summary_path = output_dir / 'climate_coverage_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("CLIMATE FEATURE COVERAGE ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ Summary saved to: {summary_path}")
    print(f"✅ Figures saved to: {output_dir}/")

    print("\n📌 KEY FINDINGS:")
    print(f"   • Climate features identified: {len(climate_features)}")
    print(f"   • Complete coverage: {complete_stats['pct_complete']:.2f}%")
    print(f"   • Records with all climate: {complete_stats['n_complete']:,} / {complete_stats['n_total']:,}")

    if summary['target_met']:
        print(f"   • ✅ Target met (≥99.0% coverage)")
    else:
        print(f"   • ❌ Target not met (<99.0% coverage)")

    print("\n📌 NEXT STEPS:")
    print("   1. Review climate_feature_coverage.csv")
    print("   2. Proceed to Task 1.4: Missing Data Pattern Analysis")
    print("   3. If coverage <99%, investigate missing data patterns")


if __name__ == '__main__':
    main()
