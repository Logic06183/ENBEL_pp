"""
Verify Existing GCRO Imputation - NO NEW IMPUTATION
====================================================

Verify the user's existing imputation work:
- Check dataset structure (one row per participant record)
- Identify which records have imputed socioeconomic data
- Understand imputation cutoff criteria
- Document what data is available AS-IS

DO NOT perform any new imputation - respect existing work.

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


def load_and_verify_structure(data_path):
    """Load dataset and verify basic structure."""
    print("="*80)
    print("DATASET STRUCTURE VERIFICATION")
    print("="*80)

    print(f"\nLoading from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features")

    # Verify one row per participant record
    print(f"\n📊 Record Structure:")
    print(f"   Total records: {len(df):,}")

    if 'anonymous_patient_id' in df.columns:
        n_unique_patients = df['anonymous_patient_id'].nunique()
        records_per_patient = len(df) / n_unique_patients
        print(f"   Unique patients: {n_unique_patients:,}")
        print(f"   Records per patient (avg): {records_per_patient:.2f}")

        if records_per_patient > 1.0:
            print(f"   ℹ️  Multiple records per patient (longitudinal data)")
        else:
            print(f"   ✅ One record per patient")

    print(f"\n   ✅ Structure: Each row = one participant record")

    return df


def analyze_imputed_socioeconomic_features(df):
    """
    Analyze existing imputed socioeconomic features.

    Understand:
    - Which features were imputed
    - What percentage have imputed values
    - Imputation cutoff criteria
    """
    print("\n" + "="*80)
    print("EXISTING SOCIOECONOMIC IMPUTATION ANALYSIS")
    print("="*80)

    # Key socioeconomic features
    socio_features = [
        'HEAT_VULNERABILITY_SCORE',
        'HEAT_STRESS_RISK_CATEGORY',
        'Sex',
        'Race'
    ]

    print(f"\n📊 Imputed Socioeconomic Features:")
    print("-" * 80)

    imputation_stats = {}

    for feat in socio_features:
        if feat not in df.columns:
            print(f"❌ {feat}: NOT FOUND")
            continue

        n_total = len(df)
        n_present = df[feat].notna().sum()
        n_missing = n_total - n_present
        pct_present = (n_present / n_total) * 100

        print(f"\n{feat}:")
        print(f"   Present: {n_present:,} / {n_total:,} ({pct_present:.2f}%)")
        print(f"   Missing: {n_missing:,} ({100-pct_present:.2f}%)")

        # Value distribution for present records
        if pct_present > 0:
            if df[feat].dtype in ['object', 'category', 'bool']:
                print(f"   Value distribution:")
                vc = df[feat].value_counts().head(10)
                for val, count in vc.items():
                    print(f"      {val}: {count:,} ({count/n_present*100:.1f}% of present)")
            else:
                print(f"   Statistics (present records):")
                print(f"      Mean ± SD: {df[feat].mean():.2f} ± {df[feat].std():.2f}")
                print(f"      Range: [{df[feat].min():.2f}, {df[feat].max():.2f}]")
                print(f"      Median: {df[feat].median():.2f}")

        imputation_stats[feat] = {
            'n_total': int(n_total),
            'n_present': int(n_present),
            'n_missing': int(n_missing),
            'pct_present': float(pct_present),
            'pct_missing': float(100 - pct_present)
        }

    return imputation_stats


def identify_imputation_cutoff_criteria(df):
    """
    Identify what criteria determined which records got imputation.

    Possible criteria:
    - Geographic (latitude/longitude availability)
    - Temporal (certain years)
    - Study-specific
    - Data quality threshold
    """
    print("\n" + "="*80)
    print("IMPUTATION CUTOFF CRITERIA ANALYSIS")
    print("="*80)

    # Check if imputation correlates with certain factors
    has_heat_vuln = df['HEAT_VULNERABILITY_SCORE'].notna()

    print(f"\n📊 Records with HEAT_VULNERABILITY_SCORE: {has_heat_vuln.sum():,} / {len(df):,}")

    # 1. Geographic availability
    if 'latitude' in df.columns and 'longitude' in df.columns:
        has_coords = df[['latitude', 'longitude']].notna().all(axis=1)
        print(f"\n🗺️  Geographic Criterion:")
        print(f"   Records with coordinates: {has_coords.sum():,} ({has_coords.mean()*100:.2f}%)")

        # How many with coords have heat vulnerability?
        coords_and_heat = (has_coords & has_heat_vuln).sum()
        coords_no_heat = (has_coords & ~has_heat_vuln).sum()

        print(f"   With coords AND heat vulnerability: {coords_and_heat:,}")
        print(f"   With coords but NO heat vulnerability: {coords_no_heat:,}")

        if coords_no_heat > 0:
            print(f"   ℹ️  {coords_no_heat:,} records have coordinates but no imputation")
            print(f"      → Suggests imputation cutoff is NOT solely based on coordinates")

    # 2. Temporal pattern
    if 'year' in df.columns:
        print(f"\n📅 Temporal Pattern:")
        year_heat_counts = df.groupby('year')['HEAT_VULNERABILITY_SCORE'].agg([
            ('total', 'count'),
            ('with_heat_vuln', lambda x: x.notna().sum()),
            ('pct', lambda x: x.notna().mean() * 100)
        ])

        print(f"\n   Imputation coverage by year:")
        print(year_heat_counts.to_string())

    # 3. Study-specific pattern
    if 'study_source' in df.columns:
        print(f"\n📚 Study-Specific Pattern:")
        study_heat_counts = df.groupby('study_source')['HEAT_VULNERABILITY_SCORE'].agg([
            ('total', 'count'),
            ('with_heat_vuln', lambda x: x.notna().sum()),
            ('pct', lambda x: x.notna().mean() * 100)
        ]).sort_values('pct', ascending=False)

        print(f"\n   Top 10 studies by imputation coverage:")
        print(study_heat_counts.head(10).to_string())

        print(f"\n   Bottom 10 studies by imputation coverage:")
        print(study_heat_counts.tail(10).to_string())

    # 4. Climate data availability
    if 'climate_daily_mean_temp' in df.columns:
        has_climate = df['climate_daily_mean_temp'].notna()

        print(f"\n🌡️  Climate Data Availability:")
        print(f"   Records with climate data: {has_climate.sum():,} ({has_climate.mean()*100:.2f}%)")

        # Overlap with heat vulnerability
        climate_and_heat = (has_climate & has_heat_vuln).sum()
        climate_no_heat = (has_climate & ~has_heat_vuln).sum()

        print(f"   With climate AND heat vulnerability: {climate_and_heat:,}")
        print(f"   With climate but NO heat vulnerability: {climate_no_heat:,}")


def create_imputation_coverage_map(df, output_dir):
    """Visualize imputation coverage patterns."""
    print("\n📊 GENERATING IMPUTATION COVERAGE VISUALIZATIONS...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Coverage by year
    if 'year' in df.columns:
        ax1 = axes[0, 0]
        year_coverage = df.groupby('year').agg({
            'HEAT_VULNERABILITY_SCORE': lambda x: x.notna().mean() * 100,
            'Sex': lambda x: x.notna().mean() * 100,
            'Race': lambda x: x.notna().mean() * 100
        })

        year_coverage.plot(kind='line', ax=ax1, marker='o', linewidth=2)
        ax1.set_title('Imputation Coverage by Year', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Coverage (%)', fontsize=12)
        ax1.set_ylim([0, 100])
        ax1.legend(title='Feature')
        ax1.grid(True, alpha=0.3)

    # 2. Coverage by study (top 15)
    if 'study_source' in df.columns:
        ax2 = axes[0, 1]
        study_coverage = df.groupby('study_source')['HEAT_VULNERABILITY_SCORE'].apply(
            lambda x: x.notna().mean() * 100
        ).sort_values(ascending=False).head(15)

        study_coverage.plot(kind='barh', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_title('Imputation Coverage by Study (Top 15)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Coverage (%)', fontsize=12)
        ax2.set_xlim([0, 100])
        ax2.grid(axis='x', alpha=0.3)

    # 3. Feature completeness comparison
    ax3 = axes[1, 0]
    features = ['HEAT_VULNERABILITY_SCORE', 'Sex', 'Race', 'Age (at enrolment)']
    completeness = []

    for feat in features:
        if feat in df.columns:
            completeness.append(df[feat].notna().mean() * 100)
        else:
            completeness.append(0)

    bars = ax3.bar(range(len(features)), completeness, color='lightcoral', edgecolor='black')
    ax3.set_xticks(range(len(features)))
    ax3.set_xticklabels(features, rotation=45, ha='right')
    ax3.set_ylabel('Completeness (%)', fontsize=12)
    ax3.set_title('Socioeconomic Feature Completeness', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 100])
    ax3.axhline(75, color='orange', linestyle='--', linewidth=2, label='75% threshold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, completeness)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Missing value patterns
    ax4 = axes[1, 1]
    missing_patterns = pd.DataFrame({
        'Climate': [df['climate_daily_mean_temp'].isna().mean() * 100],
        'Heat Vuln': [df['HEAT_VULNERABILITY_SCORE'].isna().mean() * 100],
        'Sex': [df['Sex'].isna().mean() * 100],
        'Age': [df['Age (at enrolment)'].isna().mean() * 100]
    }).T

    missing_patterns.columns = ['Missing (%)']
    missing_patterns.plot(kind='barh', ax=ax4, color='orange', edgecolor='black', legend=False)
    ax4.set_title('Missing Data Percentage', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Missing (%)', fontsize=12)
    ax4.set_xlim([0, 30])
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'existing_imputation_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: existing_imputation_coverage.png")
    plt.close()


def summarize_available_data(df, imputation_stats):
    """
    Summarize what data is available for modeling.

    This is AS-IS - no new imputation.
    """
    print("\n" + "="*80)
    print("DATA AVAILABLE FOR MODELING (AS-IS)")
    print("="*80)

    # Count records with different feature combinations
    has_climate = df['climate_daily_mean_temp'].notna()
    has_heat_vuln = df['HEAT_VULNERABILITY_SCORE'].notna()
    has_sex = df['Sex'].notna() if 'Sex' in df.columns else pd.Series([False] * len(df))
    has_age = df['Age (at enrolment)'].notna() if 'Age (at enrolment)' in df.columns else pd.Series([False] * len(df))

    # Different analysis scenarios
    print(f"\n📊 Available Records by Feature Combination:")
    print("-" * 80)

    scenarios = {
        'Climate only (6 features + 2 temporal)': has_climate,
        'Climate + Heat Vulnerability (9 features)': has_climate & has_heat_vuln,
        'Climate + Heat Vulnerability + Sex (10 features)': has_climate & has_heat_vuln & has_sex,
        'Climate + Heat Vulnerability + Sex + Age (11 features)': has_climate & has_heat_vuln & has_sex & has_age,
        'All features complete': has_climate & has_heat_vuln & has_sex & has_age
    }

    scenario_stats = {}
    for scenario_name, mask in scenarios.items():
        n_records = mask.sum()
        pct = (n_records / len(df)) * 100
        print(f"   {scenario_name:<55} {n_records:>6,} ({pct:>5.1f}%)")
        scenario_stats[scenario_name] = {
            'n_records': int(n_records),
            'pct': float(pct)
        }

    print(f"\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find best scenario with >50% coverage
    best_scenario = None
    best_n = 0

    for scenario_name, stats in scenario_stats.items():
        if stats['n_records'] > best_n:
            best_n = stats['n_records']
            best_scenario = scenario_name

    print(f"\n✅ RECOMMENDED FEATURE SET:")
    print(f"   {best_scenario}")
    print(f"   Available records: {best_n:,} ({scenario_stats[best_scenario]['pct']:.1f}%)")

    print(f"\n📌 RATIONALE:")
    print(f"   • Uses existing imputation (no new imputation needed)")
    print(f"   • Maintains maximum sample size with complete data")
    print(f"   • Respects original imputation cutoff criteria")

    return scenario_stats


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and verify structure
    df = load_and_verify_structure(data_path)

    # Analyze existing imputation
    imputation_stats = analyze_imputed_socioeconomic_features(df)

    # Identify imputation cutoff criteria
    identify_imputation_cutoff_criteria(df)

    # Visualize coverage patterns
    create_imputation_coverage_map(df, output_dir)

    # Summarize available data
    scenario_stats = summarize_available_data(df, imputation_stats)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/09_verify_existing_imputation.py',
        'n_total_records': len(df),
        'imputation_stats': imputation_stats,
        'scenario_stats': scenario_stats,
        'note': 'Using existing imputation AS-IS - no new imputation performed'
    }

    summary_path = output_dir / 'existing_imputation_verification.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("EXISTING IMPUTATION VERIFICATION COMPLETE")
    print("="*80)
    print(f"✅ Summary saved to: {summary_path}")

    print("\n📌 NEXT STEPS:")
    print("   1. ✅ Verified existing imputation structure")
    print("   2. ✅ Identified available data AS-IS")
    print("   3. ➡️  Run automated leakage checks")
    print("   4. ➡️  Feature validation (VIF, correlation)")
    print("   5. ➡️  Proceed to model optimization!")


if __name__ == '__main__':
    main()
