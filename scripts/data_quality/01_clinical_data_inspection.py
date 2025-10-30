"""
Clinical Dataset Inspection - Phase 1, Day 1, Task 1.1
=======================================================

Comprehensive quality assessment of the clinical dataset:
- Load and inspect basic statistics
- Validate data types and ranges
- Check date coverage and study distribution
- Identify patient-level characteristics

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

def load_clinical_dataset(data_path):
    """
    Load clinical dataset with basic validation.

    Parameters
    ----------
    data_path : str or Path
        Path to clinical dataset CSV file

    Returns
    -------
    df : pd.DataFrame
        Loaded dataset
    """
    print("="*80)
    print("CLINICAL DATASET INSPECTION")
    print("="*80)
    print(f"\nLoading data from: {data_path}")

    # Load data
    df = pd.read_csv(data_path, low_memory=False)

    # Parse dates
    if 'primary_date' in df.columns:
        df['primary_date_parsed'] = pd.to_datetime(df['primary_date'], errors='coerce')

    print(f"✅ Dataset loaded successfully")

    return df


def inspect_basic_statistics(df):
    """
    Inspect basic dataset statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    stats : dict
        Dictionary of basic statistics
    """
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)

    stats = {
        'n_records': len(df),
        'n_features': df.shape[1],
        'n_unique_patients': df['anonymous_patient_id'].nunique() if 'anonymous_patient_id' in df.columns else 'N/A',
        'n_studies': df['study_source'].nunique() if 'study_source' in df.columns else 'N/A',
        'date_range_start': df['primary_date_parsed'].min() if 'primary_date_parsed' in df.columns else 'N/A',
        'date_range_end': df['primary_date_parsed'].max() if 'primary_date_parsed' in df.columns else 'N/A',
        'missing_rate_overall': (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    }

    # Print statistics
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} records × {df.shape[1]} features")
    print(f"👤 Unique Patients: {stats['n_unique_patients']:,}" if isinstance(stats['n_unique_patients'], int) else f"👤 Unique Patients: {stats['n_unique_patients']}")
    print(f"🏥 Number of Studies: {stats['n_studies']}")
    print(f"📅 Date Range: {stats['date_range_start']} to {stats['date_range_end']}")
    print(f"❓ Overall Missing Rate: {stats['missing_rate_overall']:.2f}%")

    # Validate against expected values
    validation_results = []

    # Check record count
    if isinstance(stats['n_records'], int):
        if 11000 <= stats['n_records'] <= 12000:
            print(f"✅ Record count within expected range (11,000-12,000)")
            validation_results.append(('record_count', True))
        else:
            print(f"⚠️  Record count ({stats['n_records']:,}) outside expected range (11,000-12,000)")
            validation_results.append(('record_count', False))

    # Check date range
    if stats['date_range_start'] != 'N/A':
        if stats['date_range_start'].year >= 2002 and stats['date_range_end'].year <= 2021:
            print(f"✅ Date range within expected study period (2002-2021)")
            validation_results.append(('date_range', True))
        else:
            print(f"⚠️  Date range outside expected study period")
            validation_results.append(('date_range', False))

    # Check patient ID
    if 'anonymous_patient_id' in df.columns:
        null_patient_ids = df['anonymous_patient_id'].isna().sum()
        if null_patient_ids == 0:
            print(f"✅ No null patient IDs")
            validation_results.append(('patient_id_complete', True))
        else:
            print(f"⚠️  Found {null_patient_ids} null patient IDs")
            validation_results.append(('patient_id_complete', False))

    stats['validation_results'] = validation_results

    return stats


def inspect_study_distribution(df):
    """
    Inspect distribution of records across studies.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    study_stats : pd.DataFrame
        Study-level statistics
    """
    print("\n" + "="*80)
    print("STUDY DISTRIBUTION")
    print("="*80)

    if 'study_source' not in df.columns:
        print("⚠️  'study_source' column not found")
        return None

    # Study-level statistics
    study_stats = df.groupby('study_source').agg({
        'anonymous_patient_id': 'count',
        'primary_date_parsed': ['min', 'max']
    }).round(2)

    study_stats.columns = ['n_records', 'date_start', 'date_end']
    study_stats = study_stats.sort_values('n_records', ascending=False)

    print(f"\n📊 Records by Study (Top 10):")
    print(study_stats.head(10).to_string())

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot: records by study
    study_stats.head(15).plot(
        kind='barh',
        y='n_records',
        ax=axes[0],
        color='steelblue',
        legend=False
    )
    axes[0].set_xlabel('Number of Records', fontsize=12)
    axes[0].set_ylabel('Study', fontsize=12)
    axes[0].set_title('Records by Study (Top 15)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Timeline plot
    if 'primary_date_parsed' in df.columns:
        df_timeline = df.groupby([df['primary_date_parsed'].dt.year, 'study_source']).size().reset_index(name='count')
        df_timeline_pivot = df_timeline.pivot(index='primary_date_parsed', columns='study_source', values='count').fillna(0)

        # Plot top 10 studies only for clarity
        top_studies = study_stats.head(10).index
        df_timeline_pivot[top_studies].plot(ax=axes[1], marker='o', linewidth=2, alpha=0.7)

        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Number of Records', fontsize=12)
        axes[1].set_title('Temporal Distribution by Study', fontsize=14, fontweight='bold')
        axes[1].legend(title='Study', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/data_quality/study_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Study distribution plot saved to results/data_quality/study_distribution.png")

    return study_stats


def inspect_temporal_coverage(df):
    """
    Inspect temporal coverage of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    temporal_stats : dict
        Temporal coverage statistics
    """
    print("\n" + "="*80)
    print("TEMPORAL COVERAGE")
    print("="*80)

    if 'primary_date_parsed' not in df.columns:
        print("⚠️  Date column not found or not parsed")
        return None

    # Remove records with missing dates
    df_dated = df[df['primary_date_parsed'].notna()].copy()

    # Calculate statistics
    temporal_stats = {
        'n_records_with_dates': len(df_dated),
        'pct_records_with_dates': (len(df_dated) / len(df)) * 100,
        'date_min': df_dated['primary_date_parsed'].min(),
        'date_max': df_dated['primary_date_parsed'].max(),
        'date_range_years': (df_dated['primary_date_parsed'].max() - df_dated['primary_date_parsed'].min()).days / 365.25
    }

    print(f"\n📅 Records with valid dates: {temporal_stats['n_records_with_dates']:,} ({temporal_stats['pct_records_with_dates']:.2f}%)")
    print(f"📅 Date range: {temporal_stats['date_min'].date()} to {temporal_stats['date_max'].date()}")
    print(f"📅 Span: {temporal_stats['date_range_years']:.1f} years")

    # Records by year
    df_dated['year'] = df_dated['primary_date_parsed'].dt.year
    records_by_year = df_dated.groupby('year').size()

    print(f"\n📊 Records by Year:")
    print(records_by_year.to_string())

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    records_by_year.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Records', fontsize=12)
    ax.set_title('Temporal Distribution of Clinical Records', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/data_quality/temporal_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Temporal distribution plot saved to results/data_quality/temporal_distribution.png")

    # Identify periods for train/test split
    print(f"\n📌 Recommended temporal split:")
    print(f"   Train: 2002-2015 (n = {len(df_dated[df_dated['year'] <= 2015]):,})")
    print(f"   Test:  2016-2021 (n = {len(df_dated[df_dated['year'] >= 2016]):,})")

    return temporal_stats


def inspect_geographic_coverage(df):
    """
    Inspect geographic coverage (Johannesburg validity).

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset

    Returns
    -------
    geo_stats : dict
        Geographic statistics
    """
    print("\n" + "="*80)
    print("GEOGRAPHIC COVERAGE")
    print("="*80)

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("⚠️  Latitude/longitude columns not found")
        return None

    # Remove records with missing coordinates
    df_geo = df[(df['latitude'].notna()) & (df['longitude'].notna())].copy()

    geo_stats = {
        'n_records_with_coords': len(df_geo),
        'pct_records_with_coords': (len(df_geo) / len(df)) * 100,
        'lat_min': df_geo['latitude'].min(),
        'lat_max': df_geo['latitude'].max(),
        'lon_min': df_geo['longitude'].min(),
        'lon_max': df_geo['longitude'].max()
    }

    print(f"\n🌍 Records with valid coordinates: {geo_stats['n_records_with_coords']:,} ({geo_stats['pct_records_with_coords']:.2f}%)")
    print(f"🌍 Latitude range: {geo_stats['lat_min']:.4f} to {geo_stats['lat_max']:.4f}")
    print(f"🌍 Longitude range: {geo_stats['lon_min']:.4f} to {geo_stats['lon_max']:.4f}")

    # Check Johannesburg validity (approximate bounds: -26.5 to -25.8 lat, 27.8 to 28.3 lon)
    johannesburg_bounds = {
        'lat_min': -26.5,
        'lat_max': -25.8,
        'lon_min': 27.8,
        'lon_max': 28.3
    }

    in_johannesburg = (
        (df_geo['latitude'] >= johannesburg_bounds['lat_min']) &
        (df_geo['latitude'] <= johannesburg_bounds['lat_max']) &
        (df_geo['longitude'] >= johannesburg_bounds['lon_min']) &
        (df_geo['longitude'] <= johannesburg_bounds['lon_max'])
    )

    n_in_johannesburg = in_johannesburg.sum()
    pct_in_johannesburg = (n_in_johannesburg / len(df_geo)) * 100

    print(f"\n📍 Records within Johannesburg bounds: {n_in_johannesburg:,} ({pct_in_johannesburg:.2f}%)")

    if pct_in_johannesburg >= 95:
        print(f"✅ Geographic validity check passed (≥95% in Johannesburg)")
    else:
        print(f"⚠️  Geographic validity concern: only {pct_in_johannesburg:.2f}% in Johannesburg")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all points
    ax.scatter(df_geo['longitude'], df_geo['latitude'], alpha=0.3, s=5, color='steelblue', label='Clinical records')

    # Highlight Johannesburg bounds
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (johannesburg_bounds['lon_min'], johannesburg_bounds['lat_min']),
        johannesburg_bounds['lon_max'] - johannesburg_bounds['lon_min'],
        johannesburg_bounds['lat_max'] - johannesburg_bounds['lat_min'],
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
        label='Johannesburg bounds'
    )
    ax.add_patch(rect)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Geographic Distribution of Clinical Records', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/data_quality/geographic_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Geographic distribution plot saved to results/data_quality/geographic_distribution.png")

    return geo_stats


def main():
    """Main execution function."""

    # Define paths (data is in parent ENBEL_pp directory)
    base_dir = Path(__file__).resolve().parents[2]  # Go up to ENBEL_pp_model_refinement root
    data_path = base_dir.parent / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if data file exists
    if not data_path.exists():
        print(f"❌ Data file not found at {data_path}")
        print("Please ensure the clinical dataset is in the correct location.")
        print(f"Expected location: {data_path}")
        return

    # Load dataset
    df = load_clinical_dataset(data_path)

    # Run inspections
    basic_stats = inspect_basic_statistics(df)
    study_stats = inspect_study_distribution(df)
    temporal_stats = inspect_temporal_coverage(df)
    geo_stats = inspect_geographic_coverage(df)

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/01_clinical_data_inspection.py',
        'basic_stats': {k: str(v) if not isinstance(v, (int, float, list)) else v for k, v in basic_stats.items()},
        'temporal_stats': {k: str(v) if not isinstance(v, (int, float)) else v for k, v in temporal_stats.items()} if temporal_stats else None,
        'geo_stats': geo_stats
    }

    # Save results
    results_path = output_dir / 'clinical_inspection_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)
    print(f"\n✅ Results saved to {results_path}")
    print(f"✅ Figures saved to {output_dir}/")

    # Print validation summary
    print("\n📋 VALIDATION SUMMARY:")
    validation_results = basic_stats.get('validation_results', [])
    for check_name, passed in validation_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check_name}")

    print("\n📌 Next Steps:")
    print("   1. Review generated plots in results/data_quality/")
    print("   2. Proceed to Task 1.2: Biomarker Completeness Analysis")
    print("   3. Check clinical_inspection_results.json for detailed statistics")


if __name__ == '__main__':
    main()
