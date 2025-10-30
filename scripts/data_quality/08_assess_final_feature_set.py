"""
Final Feature Set Assessment - Using Existing GCRO Imputation
==============================================================

Assess completeness of final feature set for modeling:
- 6 core climate features (from Day 1)
- 2 temporal features
- 2 socioeconomic features (from existing GCRO imputation)
- 1 demographic feature

Total: 11 features

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


# Final feature set for modeling
FINAL_FEATURES = {
    'climate': [
        'climate_daily_mean_temp',
        'climate_daily_max_temp',
        'climate_daily_min_temp',
        'climate_7d_mean_temp',
        'climate_heat_stress_index',
        'climate_season'
    ],
    'temporal': [
        'month',
        'season'
    ],
    'socioeconomic': [
        'HEAT_VULNERABILITY_SCORE',
        'Sex'
    ],
    'demographic': [
        'Age (at enrolment)'
    ]
}


def load_clinical_dataset(data_path):
    """Load clinical dataset."""
    print("="*80)
    print("LOADING CLINICAL DATASET")
    print("="*80)

    print(f"\nLoading from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features\n")

    return df


def assess_feature_completeness(df, feature_dict):
    """
    Assess completeness of all final features.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    feature_dict : dict
        Dictionary of features by category

    Returns
    -------
    completeness_df : pd.DataFrame
        Completeness statistics
    """
    print("="*80)
    print("FINAL FEATURE SET COMPLETENESS")
    print("="*80)

    results = []

    for category, features in feature_dict.items():
        print(f"\n📊 {category.upper()} FEATURES:")
        print("-" * 80)

        for feat in features:
            if feat not in df.columns:
                print(f"❌ {feat:<45} NOT FOUND IN DATASET")
                results.append({
                    'category': category,
                    'feature': feat,
                    'n_observed': 0,
                    'n_missing': len(df),
                    'pct_complete': 0.0,
                    'status': 'missing'
                })
                continue

            n_total = len(df)
            n_observed = df[feat].notna().sum()
            n_missing = n_total - n_observed
            pct_complete = (n_observed / n_total) * 100

            # Determine status
            if pct_complete >= 99.0:
                status = '✅'
                status_label = 'excellent'
            elif pct_complete >= 90.0:
                status = '✅'
                status_label = 'good'
            elif pct_complete >= 75.0:
                status = '⚠️'
                status_label = 'fair'
            else:
                status = '❌'
                status_label = 'poor'

            print(f"{status} {feat:<45} {n_observed:>6,} ({pct_complete:>6.2f}%)")

            results.append({
                'category': category,
                'feature': feat,
                'n_observed': int(n_observed),
                'n_missing': int(n_missing),
                'pct_complete': float(pct_complete),
                'status': status_label
            })

    completeness_df = pd.DataFrame(results)
    return completeness_df


def calculate_complete_case_coverage(df, feature_dict):
    """
    Calculate records with ALL features present.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    feature_dict : dict
        Dictionary of features by category

    Returns
    -------
    coverage_stats : dict
        Coverage statistics
    """
    print("\n" + "="*80)
    print("COMPLETE CASE ANALYSIS")
    print("="*80)

    # Flatten feature list
    all_features = []
    for features in feature_dict.values():
        all_features.extend([f for f in features if f in df.columns])

    print(f"\n📊 Total features: {len(all_features)}")

    # Records with all features
    has_all = df[all_features].notna().all(axis=1)
    n_complete = has_all.sum()
    pct_complete = (n_complete / len(df)) * 100

    print(f"\n📍 Records with ALL features present:")
    print(f"   Complete: {n_complete:,} / {len(df):,} ({pct_complete:.2f}%)")

    if pct_complete >= 90.0:
        print(f"   ✅ EXCELLENT: ≥90% complete coverage")
    elif pct_complete >= 75.0:
        print(f"   ✅ GOOD: ≥75% complete coverage")
    elif pct_complete >= 50.0:
        print(f"   ⚠️  FAIR: 50-75% complete coverage")
    else:
        print(f"   ❌ POOR: <50% complete coverage")

    # Missing patterns
    print(f"\n📊 Missing value patterns:")
    missing_counts = df[all_features].isna().sum()
    features_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(features_with_missing) > 0:
        for feat, count in features_with_missing.items():
            pct_missing = (count / len(df)) * 100
            print(f"   {feat}: {count:,} missing ({pct_missing:.1f}%)")
    else:
        print("   ✅ No missing values!")

    coverage_stats = {
        'n_total': len(df),
        'n_complete': int(n_complete),
        'n_incomplete': int(len(df) - n_complete),
        'pct_complete': float(pct_complete),
        'n_features': len(all_features)
    }

    return coverage_stats, has_all


def recommend_missing_value_strategy(completeness_df, coverage_stats):
    """
    Recommend strategy for handling missing values.

    Parameters
    ----------
    completeness_df : pd.DataFrame
        Feature completeness statistics
    coverage_stats : dict
        Complete case coverage statistics

    Returns
    -------
    recommendation : dict
        Recommended strategy
    """
    print("\n" + "="*80)
    print("MISSING VALUE STRATEGY RECOMMENDATION")
    print("="*80)

    # Check which features have missing values
    features_with_missing = completeness_df[completeness_df['pct_complete'] < 100.0]

    print(f"\n📊 Features with missing values: {len(features_with_missing)}/{len(completeness_df)}")

    if len(features_with_missing) == 0:
        print("\n✅ NO MISSING VALUES - No imputation needed!")
        recommendation = {
            'strategy': 'none',
            'reason': 'All features 100% complete'
        }

    elif coverage_stats['pct_complete'] >= 90.0:
        print(f"\n✅ RECOMMENDATION: Complete Case Analysis")
        print(f"   Rationale:")
        print(f"     • {coverage_stats['pct_complete']:.1f}% of records have all features")
        print(f"     • Minimal data loss ({coverage_stats['n_incomplete']:,} records)")
        print(f"     • Simple, no imputation needed")
        print(f"     • Reduces bias from imputation")

        recommendation = {
            'strategy': 'complete_case',
            'reason': f"{coverage_stats['pct_complete']:.1f}% complete coverage",
            'n_records': coverage_stats['n_complete']
        }

    elif coverage_stats['pct_complete'] >= 75.0:
        print(f"\n⚠️  RECOMMENDATION: Complete Case OR Impute Key Features")
        print(f"   Option A: Complete Case Analysis")
        print(f"     • {coverage_stats['n_complete']:,} records ({coverage_stats['pct_complete']:.1f}%)")
        print(f"     • Lose {coverage_stats['n_incomplete']:,} records")

        print(f"\n   Option B: Impute Missing Values")
        print(f"     • Keep all {coverage_stats['n_total']:,} records")
        print(f"     • Impute features with <100% completeness:")
        for _, row in features_with_missing.iterrows():
            print(f"         - {row['feature']}: {row['n_missing']:,} missing ({100-row['pct_complete']:.1f}%)")

        recommendation = {
            'strategy': 'complete_case_or_impute',
            'reason': f"{coverage_stats['pct_complete']:.1f}% complete coverage",
            'option_a': 'complete_case',
            'option_b': 'impute',
            'features_to_impute': features_with_missing['feature'].tolist()
        }

    else:
        print(f"\n❌ RECOMMENDATION: Imputation Required")
        print(f"   Rationale:")
        print(f"     • Only {coverage_stats['pct_complete']:.1f}% complete coverage")
        print(f"     • Would lose {coverage_stats['n_incomplete']:,} records ({100-coverage_stats['pct_complete']:.1f}%)")
        print(f"     • Imputation necessary to maintain sample size")

        print(f"\n   Features requiring imputation:")
        for _, row in features_with_missing.iterrows():
            print(f"     • {row['feature']}: {row['n_missing']:,} missing ({100-row['pct_complete']:.1f}%)")

        recommendation = {
            'strategy': 'impute_required',
            'reason': f"Only {coverage_stats['pct_complete']:.1f}% complete coverage",
            'features_to_impute': features_with_missing['feature'].tolist()
        }

    return recommendation


def generate_summary_visualization(completeness_df, output_dir):
    """Generate visualization of feature completeness."""
    print("\n📊 GENERATING VISUALIZATIONS...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Completeness by feature
    ax1 = axes[0]

    # Color by status
    colors = completeness_df['status'].map({
        'excellent': 'green',
        'good': 'lightgreen',
        'fair': 'orange',
        'poor': 'red',
        'missing': 'darkred'
    })

    y_pos = np.arange(len(completeness_df))
    ax1.barh(y_pos, completeness_df['pct_complete'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(completeness_df['feature'], fontsize=9)
    ax1.set_xlabel('Completeness (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Feature Set Completeness', fontsize=14, fontweight='bold')
    ax1.axvline(99, color='blue', linestyle='--', linewidth=2, label='99% target')
    ax1.axvline(90, color='orange', linestyle='--', linewidth=2, label='90% acceptable')
    ax1.set_xlim([0, 105])
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Bottom: Completeness by category
    ax2 = axes[1]
    category_complete = completeness_df.groupby('category')['pct_complete'].mean()
    category_complete.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black', linewidth=1)
    ax2.set_ylabel('Average Completeness (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature Category', fontsize=12, fontweight='bold')
    ax2.set_title('Average Completeness by Category', fontsize=14, fontweight='bold')
    ax2.axhline(99, color='blue', linestyle='--', linewidth=2, label='99% target')
    ax2.axhline(90, color='orange', linestyle='--', linewidth=2, label='90% acceptable')
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    output_path = output_dir / 'final_feature_set_completeness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: final_feature_set_completeness.png")
    plt.close()


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_clinical_dataset(data_path)

    # Assess feature completeness
    completeness_df = assess_feature_completeness(df, FINAL_FEATURES)

    # Complete case coverage
    coverage_stats, has_all = calculate_complete_case_coverage(df, FINAL_FEATURES)

    # Recommend strategy
    recommendation = recommend_missing_value_strategy(completeness_df, coverage_stats)

    # Generate visualization
    generate_summary_visualization(completeness_df, output_dir)

    # Save results
    completeness_path = output_dir / 'final_feature_completeness.csv'
    completeness_df.to_csv(completeness_path, index=False)
    print(f"\n✅ Completeness table saved to: {completeness_path}")

    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/08_assess_final_feature_set.py',
        'n_records': len(df),
        'feature_set': FINAL_FEATURES,
        'n_features': sum(len(v) for v in FINAL_FEATURES.values()),
        'completeness': completeness_df.to_dict('records'),
        'coverage_stats': coverage_stats,
        'recommendation': recommendation
    }

    summary_path = output_dir / 'final_feature_set_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("FINAL FEATURE SET ASSESSMENT COMPLETE")
    print("="*80)
    print(f"✅ Summary saved to: {summary_path}")

    print("\n📌 FEATURE SET SUMMARY:")
    for category, features in FINAL_FEATURES.items():
        print(f"   {category.capitalize()}: {len(features)} features")
    print(f"   ─────────────────────")
    print(f"   TOTAL: {sum(len(v) for v in FINAL_FEATURES.values())} features")

    print("\n📌 COVERAGE:")
    print(f"   Complete cases: {coverage_stats['n_complete']:,} / {coverage_stats['n_total']:,} ({coverage_stats['pct_complete']:.2f}%)")

    print("\n📌 RECOMMENDATION:")
    print(f"   Strategy: {recommendation['strategy']}")
    print(f"   Reason: {recommendation['reason']}")

    print("\n📌 NEXT STEPS:")
    if recommendation['strategy'] == 'none':
        print("   1. ✅ No imputation needed - proceed directly to modeling")
        print("   2. Run automated leakage checks")
        print("   3. Feature validation (VIF, correlation)")
        print("   4. Begin Phase 2: Model implementation")
    elif recommendation['strategy'] == 'complete_case':
        print("   1. Filter to complete cases")
        print("   2. Run automated leakage checks")
        print("   3. Feature validation (VIF, correlation)")
        print("   4. Begin Phase 2: Model implementation")
    else:
        print("   1. Implement imputation for missing values")
        print("   2. Validate imputation quality")
        print("   3. Run automated leakage checks")
        print("   4. Feature validation (VIF, correlation)")
        print("   5. Begin Phase 2: Model implementation")


if __name__ == '__main__':
    main()
