"""
GCRO Dataset Inspection - Phase 1, Day 2, Task 2.1
===================================================

Load and inspect GCRO socioeconomic dataset:
- Verify dataset structure and record counts
- Analyze survey wave coverage
- Assess geographic distribution (wards)
- Check temporal coverage
- Generate basic quality metrics

Expected: 58,616 household records across 6 survey waves (2011-2021)

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
np.random.seed(42)


def load_gcro_dataset(data_path):
    """
    Load GCRO dataset with proper handling.

    Parameters
    ----------
    data_path : Path
        Path to GCRO CSV file

    Returns
    -------
    df : pd.DataFrame
        GCRO dataset
    """
    print("="*80)
    print("LOADING GCRO DATASET")
    print("="*80)

    if not data_path.exists():
        raise FileNotFoundError(f"GCRO dataset not found at {data_path}")

    print(f"\nLoading from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features")

    return df


def inspect_basic_statistics(df):
    """
    Inspect basic dataset statistics.

    Parameters
    ----------
    df : pd.DataFrame
        GCRO dataset

    Returns
    -------
    basic_stats : dict
        Basic statistics
    """
    print("\n" + "="*80)
    print("BASIC DATASET STATISTICS")
    print("="*80)

    basic_stats = {
        'n_records': len(df),
        'n_features': df.shape[1],
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    print(f"\n📊 Dataset Shape:")
    print(f"   Records: {basic_stats['n_records']:,}")
    print(f"   Features: {basic_stats['n_features']:,}")
    print(f"   Memory: {basic_stats['memory_mb']:.2f} MB")

    # Check expected record count
    expected_records = 58616
    deviation_pct = abs(basic_stats['n_records'] - expected_records) / expected_records * 100

    if deviation_pct < 2.0:
        print(f"   ✅ Record count matches expected (~{expected_records:,})")
        basic_stats['record_count_validation'] = 'pass'
    else:
        print(f"   ⚠️  Record count differs from expected ({expected_records:,}): {deviation_pct:.1f}% deviation")
        basic_stats['record_count_validation'] = 'warning'

    # Data types
    print(f"\n📋 Feature Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} features")

    basic_stats['dtype_counts'] = {str(k): int(v) for k, v in dtype_counts.to_dict().items()}

    # Missing data overview
    missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"\n🔍 Overall Missingness: {missing_pct:.2f}%")
    basic_stats['overall_missing_pct'] = float(missing_pct)

    return basic_stats


def analyze_survey_waves(df):
    """
    Analyze survey wave distribution.

    Parameters
    ----------
    df : pd.DataFrame
        GCRO dataset

    Returns
    -------
    wave_stats : dict
        Survey wave statistics
    """
    print("\n" + "="*80)
    print("SURVEY WAVE ANALYSIS")
    print("="*80)

    # Identify wave columns
    wave_cols = [col for col in df.columns if 'wave' in col.lower() or 'survey' in col.lower()]

    if not wave_cols:
        # Try year column
        if 'year' in df.columns:
            wave_col = 'year'
        else:
            print("⚠️  No survey wave or year column found")
            return None
    else:
        wave_col = wave_cols[0]  # Use first matching column

    print(f"\n📅 Using wave identifier: '{wave_col}'")

    # Analyze waves
    wave_counts = df[wave_col].value_counts().sort_index()

    print(f"\n📊 Survey Wave Distribution:")
    print(f"   Total waves: {len(wave_counts)}")
    print(f"   Wave range: {wave_counts.index.min()} - {wave_counts.index.max()}")

    print(f"\n   Records per wave:")
    for wave, count in wave_counts.items():
        pct = (count / len(df)) * 100
        print(f"     {wave}: {count:>6,} ({pct:>5.1f}%)")

    # Expected waves: 2011, 2013, 2015, 2017, 2019, 2021 (6 waves)
    expected_waves = [2011, 2013, 2015, 2017, 2019, 2021]
    present_waves = wave_counts.index.tolist()

    missing_waves = set(expected_waves) - set(present_waves)
    extra_waves = set(present_waves) - set(expected_waves)

    if not missing_waves:
        print(f"\n   ✅ All expected waves present (2011-2021)")
        validation_status = 'pass'
    else:
        print(f"\n   ⚠️  Missing waves: {missing_waves}")
        validation_status = 'warning'

    if extra_waves:
        print(f"   ℹ️  Extra waves found: {extra_waves}")

    wave_stats = {
        'wave_column': wave_col,
        'n_waves': len(wave_counts),
        'wave_range': [int(wave_counts.index.min()), int(wave_counts.index.max())],
        'wave_counts': {str(k): int(v) for k, v in wave_counts.to_dict().items()},
        'missing_waves': sorted([int(x) for x in missing_waves]),
        'extra_waves': sorted([int(x) for x in extra_waves]),
        'validation_status': validation_status
    }

    return wave_stats


def analyze_geographic_coverage(df):
    """
    Analyze geographic distribution (wards).

    Parameters
    ----------
    df : pd.DataFrame
        GCRO dataset

    Returns
    -------
    geo_stats : dict
        Geographic statistics
    """
    print("\n" + "="*80)
    print("GEOGRAPHIC COVERAGE ANALYSIS")
    print("="*80)

    # Identify ward column
    ward_cols = [col for col in df.columns if 'ward' in col.lower()]

    if not ward_cols:
        print("⚠️  No ward column found")
        return None

    ward_col = ward_cols[0]
    print(f"\n🗺️  Using ward identifier: '{ward_col}'")

    # Analyze wards
    n_wards = df[ward_col].nunique()
    ward_counts = df[ward_col].value_counts().sort_index()

    print(f"\n📍 Ward Distribution:")
    print(f"   Unique wards: {n_wards}")
    print(f"   Expected wards: ~258 (Johannesburg metro)")

    # Check expected ward count
    if 250 <= n_wards <= 270:
        print(f"   ✅ Ward count within expected range")
        validation_status = 'pass'
    else:
        print(f"   ⚠️  Ward count outside expected range (250-270)")
        validation_status = 'warning'

    # Records per ward statistics
    print(f"\n   Records per ward statistics:")
    print(f"     Mean: {ward_counts.mean():.1f}")
    print(f"     Median: {ward_counts.median():.1f}")
    print(f"     Min: {ward_counts.min()}")
    print(f"     Max: {ward_counts.max()}")

    # Wards with very few records
    sparse_wards = ward_counts[ward_counts < 50]
    if len(sparse_wards) > 0:
        print(f"\n   ⚠️  {len(sparse_wards)} wards with <50 records:")
        for ward, count in sparse_wards.head(10).items():
            print(f"       Ward {ward}: {count} records")

    # Check for coordinate columns
    coord_cols = ['latitude', 'longitude', 'lat', 'lon', 'Latitude', 'Longitude']
    found_coords = [col for col in coord_cols if col in df.columns]

    if found_coords:
        print(f"\n   📍 Coordinate columns found: {found_coords}")

        # Check coordinate coverage
        for col in found_coords[:2]:  # Check first two (lat, lon)
            completeness = df[col].notna().mean() * 100
            print(f"       {col}: {completeness:.2f}% complete")
    else:
        print(f"\n   ⚠️  No coordinate columns found")

    geo_stats = {
        'ward_column': ward_col,
        'n_wards': int(n_wards),
        'records_per_ward_mean': float(ward_counts.mean()),
        'records_per_ward_median': float(ward_counts.median()),
        'records_per_ward_min': int(ward_counts.min()),
        'records_per_ward_max': int(ward_counts.max()),
        'n_sparse_wards': int(len(sparse_wards)),
        'coordinate_columns': found_coords,
        'validation_status': validation_status
    }

    return geo_stats


def analyze_key_socioeconomic_variables(df):
    """
    Quick check of key socioeconomic variables.

    Parameters
    ----------
    df : pd.DataFrame
        GCRO dataset

    Returns
    -------
    variable_stats : dict
        Key variable statistics
    """
    print("\n" + "="*80)
    print("KEY SOCIOECONOMIC VARIABLES")
    print("="*80)

    # Key variables to check
    key_variables = [
        'HEAT_VULNERABILITY_SCORE',
        'income', 'Income', 'Q15_20_income',
        'EmploymentStatus', 'employment_status',
        'Education', 'std_education',
        'DwellingType', 'dwelling_type_enhanced'
    ]

    found_variables = [var for var in key_variables if var in df.columns]

    print(f"\n📊 Found {len(found_variables)} key variables:")

    variable_stats = {}

    for var in found_variables:
        completeness = df[var].notna().mean() * 100
        n_unique = df[var].nunique()

        status = "✅" if completeness >= 70 else "⚠️"
        print(f"   {status} {var:<35} {completeness:>6.1f}% complete, {n_unique:>5} unique values")

        variable_stats[var] = {
            'completeness': float(completeness),
            'n_unique': int(n_unique),
            'meets_threshold': bool(completeness >= 70)
        }

    # Check for HEAT_VULNERABILITY_SCORE specifically
    if 'HEAT_VULNERABILITY_SCORE' in df.columns:
        hvs = df['HEAT_VULNERABILITY_SCORE']
        print(f"\n   ℹ️  HEAT_VULNERABILITY_SCORE details:")
        print(f"       Mean: {hvs.mean():.3f}")
        print(f"       Std: {hvs.std():.3f}")
        print(f"       Range: [{hvs.min():.3f}, {hvs.max():.3f}]")
    else:
        print(f"\n   ⚠️  HEAT_VULNERABILITY_SCORE not found")

    return variable_stats


def visualize_gcro_overview(df, wave_col, ward_col, output_dir):
    """
    Create overview visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        GCRO dataset
    wave_col : str
        Survey wave column name
    ward_col : str
        Ward column name
    output_dir : Path
        Output directory for plots
    """
    print("\n📊 GENERATING VISUALIZATIONS...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Survey wave distribution
    ax1 = axes[0, 0]
    wave_counts = df[wave_col].value_counts().sort_index()
    wave_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Survey Wave Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Survey Wave', fontsize=12)
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (wave, count) in enumerate(wave_counts.items()):
        ax1.text(i, count + 200, f'{count:,}', ha='center', fontsize=10)

    # 2. Records per ward histogram
    ax2 = axes[0, 1]
    ward_counts = df[ward_col].value_counts()
    ax2.hist(ward_counts, bins=30, color='lightcoral', edgecolor='black')
    ax2.set_title('Records per Ward Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Records per Ward', fontsize=12)
    ax2.set_ylabel('Number of Wards', fontsize=12)
    ax2.axvline(ward_counts.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {ward_counts.median():.0f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Missingness by column (top 20)
    ax3 = axes[1, 0]
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(20)
    missing_pct.plot(kind='barh', ax=ax3, color='orange', edgecolor='black')
    ax3.set_title('Top 20 Features by Missingness', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Missing (%)', fontsize=12)
    ax3.set_ylabel('Feature', fontsize=12)
    ax3.axvline(30, color='red', linestyle='--', linewidth=2, label='30% threshold')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)

    # 4. Data type distribution
    ax4 = axes[1, 1]
    dtype_counts = df.dtypes.value_counts()
    dtype_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', startangle=90,
                      colors=sns.color_palette("Set2", len(dtype_counts)))
    ax4.set_title('Feature Data Types', fontsize=14, fontweight='bold')
    ax4.set_ylabel('')

    plt.tight_layout()
    output_path = output_dir / 'gcro_dataset_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: gcro_dataset_overview.png")
    plt.close()


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_gcro_dataset(data_path)

    # Basic statistics
    basic_stats = inspect_basic_statistics(df)

    # Survey wave analysis
    wave_stats = analyze_survey_waves(df)

    # Geographic coverage
    geo_stats = analyze_geographic_coverage(df)

    # Key socioeconomic variables
    variable_stats = analyze_key_socioeconomic_variables(df)

    # Visualize
    if wave_stats and geo_stats:
        visualize_gcro_overview(df, wave_stats['wave_column'],
                               geo_stats['ward_column'], output_dir)

    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/06_gcro_dataset_inspection.py',
        'basic_statistics': basic_stats,
        'survey_waves': wave_stats,
        'geographic_coverage': geo_stats,
        'key_variables': variable_stats
    }

    # Save summary
    summary_path = output_dir / 'gcro_dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("GCRO DATASET INSPECTION COMPLETE")
    print("="*80)
    print(f"✅ Summary saved to: {summary_path}")

    # Validation summary
    print("\n📌 VALIDATION SUMMARY:")
    print(f"   • Record count: {basic_stats['record_count_validation'].upper()}")
    if wave_stats:
        print(f"   • Survey waves: {wave_stats['validation_status'].upper()}")
    if geo_stats:
        print(f"   • Geographic coverage: {geo_stats['validation_status'].upper()}")

    print("\n📌 KEY FINDINGS:")
    print(f"   • Records: {basic_stats['n_records']:,}")
    print(f"   • Features: {basic_stats['n_features']:,}")
    if wave_stats:
        print(f"   • Survey waves: {wave_stats['n_waves']} ({wave_stats['wave_range'][0]}-{wave_stats['wave_range'][1]})")
    if geo_stats:
        print(f"   • Wards: {geo_stats['n_wards']}")
        print(f"   • Records/ward (median): {geo_stats['records_per_ward_median']:.0f}")

    print("\n📌 NEXT STEPS:")
    print("   1. Task 2.1 complete ✅")
    print("   2. Proceed to Task 2.2: Socioeconomic feature selection")
    print("   3. Select 14 features with ≥70% completeness")


if __name__ == '__main__':
    main()
