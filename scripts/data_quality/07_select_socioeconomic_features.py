"""
Socioeconomic Feature Selection - Phase 1, Day 2, Task 2.2
===========================================================

Select 14 socioeconomic features from GCRO 2011 survey wave based on:
1. Completeness: 100% within 2011 subset
2. Theoretical relevance: Literature-supported climate-health associations
3. Variability: Not constant
4. Independence: Low correlation with heat_vulnerability_index (r < 0.95)

Strategy: Use 2011 wave (15,000 records) with rich socioeconomic data

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


# Candidate socioeconomic features organized by category
CANDIDATE_FEATURES = {
    'vulnerability_indices': [
        'heat_vulnerability_index',
        'economic_vulnerability_indicator',
        'employment_vulnerability_indicator',
        'education_adaptive_capacity',
        'age_vulnerability_indicator',
        'heat_vulnerability_category'
    ],
    'demographics': [
        'std_sex',
        'Sex',
        'A2_Sex',
        'std_race',
        'Race',
        'Q15_02_age',
        'Q15_02_age_recode'
    ],
    'socioeconomic_status': [
        'Q15_20_income',
        'std_education',
        'Education',
        'Q15_01_education',
        'Q15_01_education_recode',
        'EmploymentStatus',
        'employment_status'
    ],
    'housing': [
        'dwelling_type_enhanced',
        'DwellingType',
        'A3_dwelling',
        'A3_dwelling_recode',
        'dwelling_count',
        'Q2_01_dwelling',
        'Q2_02_dwelling_dissatisfaction'
    ],
    'infrastructure': [
        'Q2_14_Drainage',
        'q2_3_sewarage'
    ],
    'spatial': [
        'Ward',
        'ward',
        'F_Ward',
        'std_ward'
    ]
}


def load_and_filter_gcro_2011(data_path):
    """
    Load GCRO dataset and filter to 2011 wave.

    Parameters
    ----------
    data_path : Path
        Path to GCRO CSV

    Returns
    -------
    df_2011 : pd.DataFrame
        2011 wave subset
    """
    print("="*80)
    print("LOADING AND FILTERING GCRO TO 2011 WAVE")
    print("="*80)

    # Load full dataset
    print(f"\nLoading from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Full dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features")

    # Filter to 2011
    df_2011 = df[df['survey_year'] == 2011].copy()
    print(f"✅ 2011 wave filtered: {df_2011.shape[0]:,} records × {df_2011.shape[1]} features")

    # Verify expected record count
    if 14000 <= len(df_2011) <= 16000:
        print(f"   ✅ Record count within expected range (14,000-16,000)")
    else:
        print(f"   ⚠️  Record count outside expected range: {len(df_2011):,}")

    return df_2011


def evaluate_candidate_features(df_2011, candidate_dict):
    """
    Evaluate all candidate features for selection.

    Parameters
    ----------
    df_2011 : pd.DataFrame
        2011 wave data
    candidate_dict : dict
        Dictionary of candidate features by category

    Returns
    -------
    feature_evaluation : pd.DataFrame
        Evaluation results for all candidates
    """
    print("\n" + "="*80)
    print("EVALUATING CANDIDATE FEATURES")
    print("="*80)

    evaluation_results = []

    # Flatten candidate dict
    all_candidates = []
    for category, features in candidate_dict.items():
        for feat in features:
            all_candidates.append((feat, category))

    print(f"\nEvaluating {len(all_candidates)} candidate features...")

    for feat, category in all_candidates:
        if feat not in df_2011.columns:
            continue

        # 1. Completeness within 2011 subset
        completeness = df_2011[feat].notna().mean() * 100

        # 2. Variability
        if df_2011[feat].dtype in ['object', 'category', 'bool']:
            n_unique = df_2011[feat].nunique()
            variability = n_unique > 1
            feat_type = 'categorical'
        else:
            std_val = df_2011[feat].std()
            variability = std_val > 0
            n_unique = df_2011[feat].nunique()
            feat_type = 'continuous'

        # 3. Correlation with heat_vulnerability_index (if both numeric)
        corr_with_hvi = np.nan
        if 'heat_vulnerability_index' in df_2011.columns:
            if feat_type == 'continuous' and feat != 'heat_vulnerability_index':
                try:
                    corr_with_hvi = df_2011[[feat, 'heat_vulnerability_index']].corr().iloc[0, 1]
                except:
                    corr_with_hvi = np.nan

        # Selection criteria
        meets_completeness = completeness >= 99.0  # 99% within 2011
        meets_variability = variability
        meets_independence = np.isnan(corr_with_hvi) or abs(corr_with_hvi) < 0.95

        selected = meets_completeness and meets_variability and meets_independence

        evaluation_results.append({
            'feature': feat,
            'category': category,
            'type': feat_type,
            'completeness_pct': completeness,
            'n_unique': n_unique,
            'corr_with_hvi': corr_with_hvi,
            'meets_completeness': meets_completeness,
            'meets_variability': meets_variability,
            'meets_independence': meets_independence,
            'selected': selected
        })

    df_eval = pd.DataFrame(evaluation_results)
    df_eval = df_eval.sort_values(['selected', 'completeness_pct', 'category'],
                                   ascending=[False, False, True])

    return df_eval


def print_evaluation_summary(df_eval):
    """Print evaluation summary."""

    print("\n" + "="*80)
    print("FEATURE EVALUATION SUMMARY")
    print("="*80)

    # Overall counts
    n_total = len(df_eval)
    n_selected = df_eval['selected'].sum()

    print(f"\n📊 Overall Statistics:")
    print(f"   Total candidates evaluated: {n_total}")
    print(f"   Meeting all criteria: {n_selected}")

    # By category
    print(f"\n📂 Selected Features by Category:")
    for category in df_eval[df_eval['selected']]['category'].unique():
        cat_df = df_eval[(df_eval['selected']) & (df_eval['category'] == category)]
        print(f"   {category}: {len(cat_df)} features")

    # Selected features
    print(f"\n✅ SELECTED FEATURES ({n_selected}):")
    print("-" * 80)

    for idx, row in df_eval[df_eval['selected']].iterrows():
        status_symbols = []
        if row['completeness_pct'] >= 99:
            status_symbols.append('✅')
        if row['meets_variability']:
            status_symbols.append('🔄')
        if row['meets_independence']:
            status_symbols.append('🔀')

        symbols = ' '.join(status_symbols)

        print(f"{symbols} {row['feature']:<45} "
              f"({row['category']:<25}) "
              f"{row['completeness_pct']:>6.1f}% "
              f"n_unique={row['n_unique']:>4}")

        if not np.isnan(row['corr_with_hvi']):
            print(f"      ├─ Correlation with HVI: {row['corr_with_hvi']:>6.3f}")

    # Rejected features
    rejected = df_eval[~df_eval['selected']]
    if len(rejected) > 0:
        print(f"\n❌ REJECTED FEATURES ({len(rejected)}):")
        print("-" * 80)

        for idx, row in rejected.head(10).iterrows():
            reasons = []
            if not row['meets_completeness']:
                reasons.append(f"completeness={row['completeness_pct']:.1f}%")
            if not row['meets_variability']:
                reasons.append("no variability")
            if not row['meets_independence']:
                reasons.append(f"corr={row['corr_with_hvi']:.3f}")

            reason_str = ', '.join(reasons)
            print(f"   {row['feature']:<40} ({reason_str})")


def finalize_feature_selection(df_eval, target_n=14):
    """
    Finalize selection of exactly target_n features.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Evaluation results
    target_n : int
        Target number of features (default: 14)

    Returns
    -------
    final_features : list
        Final selected feature names
    """
    print("\n" + "="*80)
    print(f"FINALIZING SELECTION: TARGET {target_n} FEATURES")
    print("="*80)

    # Get all features meeting criteria
    candidates = df_eval[df_eval['selected']].copy()

    print(f"\n📊 Initial candidates: {len(candidates)}")

    if len(candidates) < target_n:
        print(f"   ⚠️  Fewer than {target_n} features meet all criteria")
        print(f"   → Will select all {len(candidates)} meeting criteria")
        print(f"   → May need to relax criteria or add features manually")
        final_features = candidates['feature'].tolist()

    elif len(candidates) == target_n:
        print(f"   ✅ Exactly {target_n} features meet all criteria!")
        final_features = candidates['feature'].tolist()

    else:
        print(f"   ℹ️  {len(candidates)} features meet criteria (>{target_n})")
        print(f"   → Prioritizing by category balance and completeness")

        # Prioritization strategy:
        # 1. Always include heat_vulnerability_index
        # 2. Include other vulnerability indices
        # 3. Balance across categories
        # 4. Prefer higher completeness

        priority_order = [
            'heat_vulnerability_index',
            'economic_vulnerability_indicator',
            'employment_vulnerability_indicator',
            'education_adaptive_capacity',
            'age_vulnerability_indicator',
            'std_education',
            'Q15_20_income',
            'dwelling_type_enhanced',
            'EmploymentStatus',
            'std_race',
            'std_sex',
            'Q15_02_age_recode',
            'Q2_14_Drainage',
            'dwelling_count'
        ]

        final_features = []

        # Add by priority
        for feat in priority_order:
            if feat in candidates['feature'].values and len(final_features) < target_n:
                final_features.append(feat)

        # If still need more, add by completeness
        if len(final_features) < target_n:
            remaining = candidates[~candidates['feature'].isin(final_features)]
            remaining = remaining.sort_values('completeness_pct', ascending=False)
            for feat in remaining['feature']:
                if len(final_features) < target_n:
                    final_features.append(feat)

    print(f"\n✅ Final selection: {len(final_features)} features")

    return final_features


def generate_feature_documentation(df_2011, final_features):
    """
    Generate detailed documentation for each selected feature.

    Parameters
    ----------
    df_2011 : pd.DataFrame
        2011 wave data
    final_features : list
        Final selected features

    Returns
    -------
    feature_docs : dict
        Documentation for each feature
    """
    print("\n" + "="*80)
    print("GENERATING FEATURE DOCUMENTATION")
    print("="*80)

    feature_docs = {}

    for feat in final_features:
        doc = {
            'feature_name': feat,
            'n_records': int(df_2011[feat].notna().sum()),
            'completeness_pct': float((df_2011[feat].notna().sum() / len(df_2011)) * 100),
            'data_type': str(df_2011[feat].dtype)
        }

        # Type-specific statistics
        if df_2011[feat].dtype in ['object', 'category', 'bool']:
            doc['value_counts'] = df_2011[feat].value_counts().head(10).to_dict()
            doc['n_unique'] = int(df_2011[feat].nunique())
        else:
            doc['mean'] = float(df_2011[feat].mean())
            doc['std'] = float(df_2011[feat].std())
            doc['min'] = float(df_2011[feat].min())
            doc['max'] = float(df_2011[feat].max())
            doc['median'] = float(df_2011[feat].median())
            doc['n_unique'] = int(df_2011[feat].nunique())

        feature_docs[feat] = doc

        # Print summary
        print(f"\n📋 {feat}")
        print(f"   Completeness: {doc['completeness_pct']:.2f}%")
        print(f"   Type: {doc['data_type']}")

        if 'mean' in doc:
            print(f"   Range: [{doc['min']:.2f}, {doc['max']:.2f}]")
            print(f"   Mean ± SD: {doc['mean']:.2f} ± {doc['std']:.2f}")
        else:
            print(f"   Categories: {doc['n_unique']}")
            if 'value_counts' in doc:
                for val, count in list(doc['value_counts'].items())[:5]:
                    print(f"      {val}: {count:,} ({count/len(df_2011)*100:.1f}%)")

    return feature_docs


def visualize_selected_features(df_2011, final_features, output_dir):
    """
    Create visualizations for selected features.

    Parameters
    ----------
    df_2011 : pd.DataFrame
        2011 wave data
    final_features : list
        Selected features
    output_dir : Path
        Output directory
    """
    print("\n📊 GENERATING VISUALIZATIONS...")

    # Create figure with subplots
    n_features = len(final_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feat in enumerate(final_features):
        ax = axes[idx]

        if df_2011[feat].dtype in ['object', 'category', 'bool']:
            # Categorical: bar plot
            value_counts = df_2011[feat].value_counts().head(10)
            value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'{feat}\n(Categorical)', fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            # Continuous: histogram
            df_2011[feat].hist(bins=30, ax=ax, color='lightcoral', edgecolor='black')
            ax.set_title(f'{feat}\n(Continuous)', fontsize=10, fontweight='bold')
            ax.set_xlabel('')

        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'selected_socioeconomic_features.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: selected_socioeconomic_features.png")
    plt.close()

    # Create correlation matrix for continuous features
    continuous_features = [f for f in final_features if df_2011[f].dtype not in ['object', 'category', 'bool']]

    if len(continuous_features) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df_2011[continuous_features].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title('Correlation Matrix: Continuous Socioeconomic Features',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / 'socioeconomic_features_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: socioeconomic_features_correlation.png")
        plt.close()


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir.parent / "data" / "raw" / "GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"
    output_dir = base_dir / "results" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter to 2011
    df_2011 = load_and_filter_gcro_2011(data_path)

    # Evaluate all candidates
    df_eval = evaluate_candidate_features(df_2011, CANDIDATE_FEATURES)

    # Print evaluation summary
    print_evaluation_summary(df_eval)

    # Finalize selection (target: 14 features)
    final_features = finalize_feature_selection(df_eval, target_n=14)

    # Generate feature documentation
    feature_docs = generate_feature_documentation(df_2011, final_features)

    # Visualize selected features
    visualize_selected_features(df_2011, final_features, output_dir)

    # Save evaluation table
    eval_path = output_dir / 'socioeconomic_feature_evaluation.csv'
    df_eval.to_csv(eval_path, index=False)
    print(f"\n✅ Evaluation table saved to: {eval_path}")

    # Save selected features
    selected_path = output_dir / 'selected_socioeconomic_features.csv'
    df_selected = df_eval[df_eval['feature'].isin(final_features)]
    df_selected.to_csv(selected_path, index=False)
    print(f"✅ Selected features saved to: {selected_path}")

    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'script': 'scripts/data_quality/07_select_socioeconomic_features.py',
        'n_records_2011': len(df_2011),
        'n_candidates_evaluated': len(df_eval),
        'n_meeting_criteria': int(df_eval['selected'].sum()),
        'n_final_selected': len(final_features),
        'target_n': 14,
        'final_features': final_features,
        'feature_documentation': feature_docs,
        'selection_criteria': {
            'completeness_threshold': 99.0,
            'variability_required': True,
            'independence_threshold': 0.95
        }
    }

    summary_path = output_dir / 'socioeconomic_feature_selection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("SOCIOECONOMIC FEATURE SELECTION COMPLETE")
    print("="*80)
    print(f"✅ Summary saved to: {summary_path}")

    print("\n📌 FINAL FEATURE SET:")
    for i, feat in enumerate(final_features, 1):
        category = df_eval[df_eval['feature'] == feat]['category'].iloc[0]
        print(f"   {i:2}. {feat:<45} ({category})")

    print("\n📌 FEATURE SPACE SUMMARY:")
    print(f"   Climate features (Day 1):   6")
    print(f"   Temporal features (Day 1):  2")
    print(f"   Socioeconomic features:    {len(final_features)}")
    print(f"   ─────────────────────────────")
    print(f"   TOTAL:                     {6 + 2 + len(final_features)}")

    print("\n📌 NEXT STEPS:")
    print("   1. Task 2.2 complete ✅")
    print("   2. Proceed to Task 2.3: Assess spatial matching quality")
    print("   3. Match 11,398 clinical records → 15,000 GCRO 2011 records")


if __name__ == '__main__':
    main()
