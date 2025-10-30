"""
Biomarker Completeness Analysis - Phase 1, Day 1, Task 1.2
===========================================================

Comprehensive assessment of biomarker availability:
- Calculate completeness for all 28 biomarkers across 7 physiological systems
- Determine which biomarkers meet inclusion criteria
- Analyze missingness patterns by study and time period
- Generate completeness visualizations

Inclusion Criteria:
- n_observed ≥ 200 observations
- completeness ≥ 5%

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


# Define biomarkers by physiological system
BIOMARKERS = {
    'immune': [
        'CD4 cell count (cells/µL)',
        'HIV viral load (copies/mL)'
    ],
    'hematological': [
        'Hematocrit (%)',
        'hemoglobin_g_dL',
        'Red blood cell count (×10⁶/µL)',
        'MCV (MEAN CELL VOLUME)',
        'Platelet count (×10³/µL)'
    ],
    'hepatic': [
        'ALT (U/L)',
        'AST (U/L)',
        'Albumin (g/dL)',
        'Alkaline phosphatase (U/L)',
        'Total bilirubin (mg/dL)',
        'Total protein (g/dL)'
    ],
    'lipid_cardiovascular': [
        'FASTING HDL',
        'FASTING LDL',
        'FASTING TRIGLYCERIDES',
        'total_cholesterol_mg_dL',
        'hdl_cholesterol_mg_dL',
        'ldl_cholesterol_mg_dL'
    ],
    'renal': [
        'creatinine_umol_L',
        'creatinine clearance',
        'Potassium (mEq/L)',
        'Sodium (mEq/L)'
    ],
    'cardiovascular_vitals': [
        'systolic_bp_mmHg',
        'diastolic_bp_mmHg',
        'heart_rate_bpm',
        'body_temperature_celsius'
    ],
    'metabolic': [
        'fasting_glucose_mmol_L',
        'BMI (kg/m²)',
        'Last weight recorded (kg)'
    ]
}


def analyze_biomarker_completeness(df, biomarkers_dict, min_observations=200, min_completeness_pct=5.0):
    """
    Analyze completeness for all biomarkers.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    biomarkers_dict : dict
        Dictionary of biomarkers by physiological system
    min_observations : int
        Minimum number of observations for inclusion
    min_completeness_pct : float
        Minimum completeness percentage for inclusion

    Returns
    -------
    completeness_df : pd.DataFrame
        Completeness statistics for all biomarkers
    """
    print("="*80)
    print("BIOMARKER COMPLETENESS ANALYSIS")
    print("="*80)

    completeness_results = []

    for system, biomarker_list in biomarkers_dict.items():
        for biomarker in biomarker_list:
            if biomarker in df.columns:
                n_total = len(df)
                n_observed = df[biomarker].notna().sum()
                n_missing = n_total - n_observed
                pct_complete = (n_observed / n_total) * 100

                # Inclusion criteria
                meets_criteria = (n_observed >= min_observations) and (pct_complete >= min_completeness_pct)

                completeness_results.append({
                    'system': system,
                    'biomarker': biomarker,
                    'n_total': n_total,
                    'n_observed': n_observed,
                    'n_missing': n_missing,
                    'pct_complete': pct_complete,
                    'meets_criteria': meets_criteria
                })
            else:
                # Biomarker not in dataset
                completeness_results.append({
                    'system': system,
                    'biomarker': biomarker,
                    'n_total': len(df),
                    'n_observed': 0,
                    'n_missing': len(df),
                    'pct_complete': 0.0,
                    'meets_criteria': False
                })

    completeness_df = pd.DataFrame(completeness_results)

    return completeness_df


def print_completeness_summary(completeness_df):
    """Print completeness summary by system."""

    print("\n📊 COMPLETENESS SUMMARY BY SYSTEM")
    print("="*80)

    for system in completeness_df['system'].unique():
        system_df = completeness_df[completeness_df['system'] == system].sort_values('pct_complete', ascending=False)

        print(f"\n{system.upper().replace('_', ' ')}:")
        print("-" * 80)

        for _, row in system_df.iterrows():
            status = "✅" if row['meets_criteria'] else "❌"
            print(f"{status} {row['biomarker']:<45} "
                  f"n={row['n_observed']:>6,} ({row['pct_complete']:>5.1f}%)")

    # Overall summary
    n_total_biomarkers = len(completeness_df)
    n_meeting_criteria = completeness_df['meets_criteria'].sum()

    print("\n" + "="*80)
    print(f"📈 OVERALL SUMMARY:")
    print(f"   Total biomarkers assessed: {n_total_biomarkers}")
    print(f"   Meeting inclusion criteria: {n_meeting_criteria} ({(n_meeting_criteria/n_total_biomarkers)*100:.1f}%)")
    print(f"   Excluded (insufficient data): {n_total_biomarkers - n_meeting_criteria}")


def analyze_missingness_by_study(df, biomarker, top_n=10):
    """
    Analyze missingness patterns by study for a specific biomarker.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset
    biomarker : str
        Biomarker name
    top_n : int
        Number of top studies to analyze

    Returns
    -------
    study_completeness : pd.DataFrame
        Completeness by study
    """
    if biomarker not in df.columns:
        return None

    if 'study_source' not in df.columns:
        return None

    study_completeness = df.groupby('study_source').agg({
        biomarker: [
            ('n_total', 'count'),
            ('n_observed', lambda x: x.notna().sum()),
            ('pct_complete', lambda x: (x.notna().sum() / len(x)) * 100)
        ]
    }).round(2)

    study_completeness.columns = ['n_total', 'n_observed', 'pct_complete']
    study_completeness = study_completeness.sort_values('n_total', ascending=False).head(top_n)

    return study_completeness


def visualize_completeness(completeness_df, output_path):
    """
    Create comprehensive visualization of biomarker completeness.

    Parameters
    ----------
    completeness_df : pd.DataFrame
        Completeness statistics
    output_path : Path
        Output directory for plots
    """
    print("\n📊 GENERATING VISUALIZATIONS...")

    # Figure 1: Completeness heatmap by system
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Top panel: Bar plot of completeness
    ax1 = axes[0]
    completeness_sorted = completeness_df.sort_values('pct_complete', ascending=True)

    colors = ['red' if not x else 'green' for x in completeness_sorted['meets_criteria']]

    completeness_sorted.plot(
        kind='barh',
        x='biomarker',
        y='pct_complete',
        ax=ax1,
        color=colors,
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )

    ax1.axvline(5, color='blue', linestyle='--', linewidth=2, label='5% threshold')
    ax1.set_xlabel('Completeness (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Biomarker', fontsize=12, fontweight='bold')
    ax1.set_title('Biomarker Completeness - All 28 Biomarkers', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Bottom panel: Sample size
    ax2 = axes[1]
    completeness_sorted.plot(
        kind='barh',
        x='biomarker',
        y='n_observed',
        ax=ax2,
        color=colors,
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )

    ax2.axvline(200, color='blue', linestyle='--', linewidth=2, label='200 obs threshold')
    ax2.set_xlabel('Number of Observations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Biomarker', fontsize=12, fontweight='bold')
    ax2.set_title('Biomarker Sample Sizes', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'biomarker_completeness_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: biomarker_completeness_overview.png")

    # Figure 2: Completeness by system (grouped bar plot)
    fig, ax = plt.subplots(figsize=(14, 8))

    systems = completeness_df['system'].unique()
    x = np.arange(len(systems))
    width = 0.35

    n_meeting = [completeness_df[(completeness_df['system'] == s) & (completeness_df['meets_criteria'])].shape[0]
                 for s in systems]
    n_total = [completeness_df[completeness_df['system'] == s].shape[0] for s in systems]
    n_excluded = [total - meet for total, meet in zip(n_total, n_meeting)]

    ax.bar(x - width/2, n_meeting, width, label='Meeting Criteria', color='green', edgecolor='black')
    ax.bar(x + width/2, n_excluded, width, label='Excluded', color='red', edgecolor='black')

    ax.set_xlabel('Physiological System', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Biomarkers', fontsize=12, fontweight='bold')
    ax.set_title('Biomarker Availability by Physiological System', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in systems], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'completeness_by_system.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: completeness_by_system.png")


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

    # Analyze biomarker completeness
    completeness_df = analyze_biomarker_completeness(df, BIOMARKERS)

    # Print summary
    print_completeness_summary(completeness_df)

    # Save completeness table
    completeness_path = output_dir / 'biomarker_completeness.csv'
    completeness_df.to_csv(completeness_path, index=False)
    print(f"\n✅ Completeness table saved to: {completeness_path}")

    # Analyze missingness by study for key biomarkers
    print("\n" + "="*80)
    print("MISSINGNESS BY STUDY (Top 5 Biomarkers)")
    print("="*80)

    key_biomarkers = completeness_df[completeness_df['meets_criteria']].sort_values('pct_complete', ascending=False).head(5)['biomarker'].tolist()

    study_analyses = {}
    for biomarker in key_biomarkers:
        print(f"\n{biomarker}:")
        study_comp = analyze_missingness_by_study(df, biomarker, top_n=5)
        if study_comp is not None:
            print(study_comp.to_string())
            study_analyses[biomarker] = study_comp.to_dict()

    # Visualize
    visualize_completeness(completeness_df, output_dir)

    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/02_biomarker_completeness_analysis.py',
        'n_biomarkers_total': len(completeness_df),
        'n_biomarkers_meeting_criteria': int(completeness_df['meets_criteria'].sum()),
        'inclusion_criteria': {
            'min_observations': 200,
            'min_completeness_pct': 5.0
        },
        'biomarkers_by_system': {
            system: {
                'n_total': int((completeness_df['system'] == system).sum()),
                'n_meeting_criteria': int(completeness_df[(completeness_df['system'] == system) & (completeness_df['meets_criteria'])].shape[0])
            }
            for system in completeness_df['system'].unique()
        },
        'top_biomarkers': completeness_df[completeness_df['meets_criteria']].sort_values('pct_complete', ascending=False).head(10)[['biomarker', 'n_observed', 'pct_complete']].to_dict('records')
    }

    summary_path = output_dir / 'biomarker_completeness_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("BIOMARKER COMPLETENESS ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ Summary saved to: {summary_path}")
    print(f"✅ Figures saved to: {output_dir}/")

    print("\n📌 KEY FINDINGS:")
    print(f"   • {summary['n_biomarkers_meeting_criteria']}/{summary['n_biomarkers_total']} biomarkers meet inclusion criteria")
    print(f"   • Top biomarker: {summary['top_biomarkers'][0]['biomarker']} (n={summary['top_biomarkers'][0]['n_observed']}, {summary['top_biomarkers'][0]['pct_complete']:.1f}%)")

    print("\n📌 NEXT STEPS:")
    print("   1. Review biomarker_completeness.csv for detailed statistics")
    print("   2. Review generated visualizations")
    print("   3. Proceed to Task 1.3: Climate Feature Coverage")


if __name__ == '__main__':
    main()
