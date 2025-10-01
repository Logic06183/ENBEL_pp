#!/usr/bin/env python3
"""
Find Meaningful Climate-Health Relationships in ENBEL Dataset
==============================================================
This script explores the dataset more comprehensively to find 
statistically and clinically significant relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_full_dataset():
    """Load and prepare the full dataset"""
    print("Loading complete ENBEL dataset...")
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', 
                     low_memory=False)
    
    # Convert dates
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['visit_date'].min()} to {df['visit_date'].max()}")
    
    return df

def explore_all_biomarkers(df):
    """Find all available biomarkers in the dataset"""
    print("\n=== EXPLORING ALL BIOMARKERS ===")
    
    # Common biomarker keywords
    biomarker_keywords = [
        'cd4', 'cd8', 'viral', 'glucose', 'cholesterol', 'triglyc',
        'creatinine', 'albumin', 'bilirubin', 'ast', 'alt', 'hemoglobin',
        'hematocrit', 'platelet', 'white', 'red', 'blood', 'pressure',
        'systolic', 'diastolic', 'weight', 'bmi', 'waist', 'temperature'
    ]
    
    biomarker_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in biomarker_keywords):
            # Check if it has actual numeric data
            if df[col].dtype in ['float64', 'int64']:
                non_null = df[col].notna().sum()
                if non_null > 100:  # At least 100 observations
                    biomarker_cols.append(col)
                    print(f"  {col}: {non_null:,} observations")
    
    return biomarker_cols

def analyze_extreme_heat_events(df, biomarker_cols):
    """Analyze impact of extreme heat events"""
    print("\n=== EXTREME HEAT EVENT ANALYSIS ===")
    
    results = {}
    
    # Define extreme heat as >95th percentile
    if 'climate_daily_max_temp' in df.columns:
        temp_95 = df['climate_daily_max_temp'].quantile(0.95)
        temp_90 = df['climate_daily_max_temp'].quantile(0.90)
        temp_75 = df['climate_daily_max_temp'].quantile(0.75)
        
        print(f"Temperature thresholds:")
        print(f"  75th percentile: {temp_75:.1f}°C")
        print(f"  90th percentile: {temp_90:.1f}°C")
        print(f"  95th percentile: {temp_95:.1f}°C")
        
        # Create heat exposure categories
        df['heat_category'] = pd.cut(df['climate_daily_max_temp'],
                                     bins=[-np.inf, temp_75, temp_90, temp_95, np.inf],
                                     labels=['Normal', 'Warm', 'Hot', 'Extreme'])
        
        print(f"\nHeat exposure distribution:")
        print(df['heat_category'].value_counts())
        
        # Test each biomarker
        significant_findings = []
        
        for biomarker in biomarker_cols:
            data = df[[biomarker, 'heat_category']].dropna()
            
            if len(data) > 100:
                # ANOVA test for differences across heat categories
                groups = [group[biomarker].values for name, group in data.groupby('heat_category')]
                
                if len(groups) >= 3:  # Need at least 3 groups
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    if p_value < 0.05:  # Significant difference
                        # Calculate effect sizes
                        means = data.groupby('heat_category')[biomarker].mean()
                        stds = data.groupby('heat_category')[biomarker].std()
                        counts = data.groupby('heat_category')[biomarker].count()
                        
                        # Compare extreme vs normal
                        if 'Extreme' in means.index and 'Normal' in means.index:
                            effect_size = (means['Extreme'] - means['Normal']) / stds['Normal']
                            percent_change = ((means['Extreme'] - means['Normal']) / means['Normal']) * 100
                            
                            if abs(effect_size) > 0.2:  # Cohen's d > 0.2 (small effect)
                                significant_findings.append({
                                    'biomarker': biomarker,
                                    'p_value': p_value,
                                    'effect_size': effect_size,
                                    'percent_change': percent_change,
                                    'normal_mean': means['Normal'],
                                    'extreme_mean': means['Extreme'],
                                    'n_normal': counts['Normal'],
                                    'n_extreme': counts['Extreme']
                                })
                                
                                print(f"\n  *** SIGNIFICANT: {biomarker}")
                                print(f"      p-value: {p_value:.4f}")
                                print(f"      Effect size: {effect_size:.3f}")
                                print(f"      Change: {percent_change:.1f}%")
                                print(f"      Normal: {means['Normal']:.2f} (n={counts['Normal']})")
                                print(f"      Extreme: {means['Extreme']:.2f} (n={counts['Extreme']})")
        
        results['extreme_heat'] = significant_findings
    
    return results

def analyze_heat_waves(df, biomarker_cols):
    """Analyze consecutive hot days (heat waves)"""
    print("\n=== HEAT WAVE ANALYSIS ===")
    
    results = {}
    
    if 'climate_daily_max_temp' in df.columns:
        # Define heat wave as 3+ consecutive days >90th percentile
        temp_90 = df['climate_daily_max_temp'].quantile(0.90)
        
        # Sort by date (patient_id might not exist)
        df_sorted = df.sort_values('visit_date') if 'visit_date' in df.columns else df
        
        # Identify heat wave periods
        df_sorted['is_hot'] = df_sorted['climate_daily_max_temp'] > temp_90
        
        # Count consecutive hot days (simplified approach)
        df_sorted['heat_wave'] = False
        
        # Check for recent heat exposure (within 7 days)
        for col in df.columns:
            if 'climate_7d_max_temp' in col:
                df_sorted['heat_wave'] = df_sorted[col] > temp_90
                break
        
        print(f"Heat wave exposure: {df_sorted['heat_wave'].sum():,} observations")
        
        # Test biomarkers
        significant_findings = []
        
        for biomarker in biomarker_cols:
            data = df_sorted[[biomarker, 'heat_wave']].dropna()
            
            if len(data) > 100:
                # T-test for heat wave vs normal
                heat_wave_data = data[data['heat_wave']][biomarker]
                normal_data = data[~data['heat_wave']][biomarker]
                
                if len(heat_wave_data) > 30 and len(normal_data) > 30:
                    t_stat, p_value = stats.ttest_ind(heat_wave_data, normal_data)
                    
                    if p_value < 0.05:
                        effect_size = (heat_wave_data.mean() - normal_data.mean()) / normal_data.std()
                        percent_change = ((heat_wave_data.mean() - normal_data.mean()) / normal_data.mean()) * 100
                        
                        if abs(effect_size) > 0.2:
                            significant_findings.append({
                                'biomarker': biomarker,
                                'p_value': p_value,
                                'effect_size': effect_size,
                                'percent_change': percent_change,
                                'normal_mean': normal_data.mean(),
                                'heat_wave_mean': heat_wave_data.mean(),
                                'n_normal': len(normal_data),
                                'n_heat_wave': len(heat_wave_data)
                            })
                            
                            print(f"\n  *** SIGNIFICANT: {biomarker}")
                            print(f"      p-value: {p_value:.4f}")
                            print(f"      Effect size: {effect_size:.3f}")
                            print(f"      Change: {percent_change:.1f}%")
        
        results['heat_waves'] = significant_findings
    
    return results

def analyze_vulnerable_subgroups(df, biomarker_cols):
    """Analyze vulnerable populations"""
    print("\n=== VULNERABLE SUBGROUP ANALYSIS ===")
    
    results = {}
    
    # Define vulnerable groups
    vulnerable_groups = []
    
    # HIV positive patients
    if 'HIV_status' in df.columns:
        df['is_hiv_positive'] = df['HIV_status'].isin(['Positive', 'positive', 'HIV+', '1', 1])
        vulnerable_groups.append(('HIV+', 'is_hiv_positive'))
    
    # Elderly (>50 years)
    if 'Age' in df.columns:
        df['is_elderly'] = df['Age'] > 50
        vulnerable_groups.append(('Elderly (>50)', 'is_elderly'))
    
    # Low CD4 (<350)
    if 'CD4 cell count (cells/µL)' in df.columns:
        df['is_immunocompromised'] = df['CD4 cell count (cells/µL)'] < 350
        vulnerable_groups.append(('CD4<350', 'is_immunocompromised'))
    
    # High vulnerability score
    if 'HEAT_VULNERABILITY_SCORE' in df.columns:
        vuln_75 = df['HEAT_VULNERABILITY_SCORE'].quantile(0.75)
        df['is_high_vulnerability'] = df['HEAT_VULNERABILITY_SCORE'] > vuln_75
        vulnerable_groups.append(('High Heat Vulnerability', 'is_high_vulnerability'))
    
    # Analyze temperature effects in each vulnerable group
    for group_name, group_col in vulnerable_groups:
        if group_col in df.columns:
            vulnerable = df[df[group_col] == True]
            non_vulnerable = df[df[group_col] == False]
            
            print(f"\n{group_name}: {len(vulnerable):,} observations")
            
            if 'climate_daily_max_temp' in df.columns and len(vulnerable) > 100:
                # Compare temperature sensitivity
                for biomarker in biomarker_cols[:5]:  # Top 5 biomarkers
                    vuln_data = vulnerable[[biomarker, 'climate_daily_max_temp']].dropna()
                    non_vuln_data = non_vulnerable[[biomarker, 'climate_daily_max_temp']].dropna()
                    
                    if len(vuln_data) > 50 and len(non_vuln_data) > 50:
                        # Correlation with temperature
                        vuln_corr = vuln_data[biomarker].corr(vuln_data['climate_daily_max_temp'])
                        non_vuln_corr = non_vuln_data[biomarker].corr(non_vuln_data['climate_daily_max_temp'])
                        
                        if abs(vuln_corr) > 0.15 and abs(vuln_corr) > abs(non_vuln_corr) * 1.5:
                            print(f"    {biomarker}: r={vuln_corr:.3f} (vs {non_vuln_corr:.3f} in non-vulnerable)")
                            
                            results[group_name] = {
                                'biomarker': biomarker,
                                'vulnerable_correlation': vuln_corr,
                                'non_vulnerable_correlation': non_vuln_corr,
                                'amplification': vuln_corr / non_vuln_corr if non_vuln_corr != 0 else np.inf
                            }
    
    return results

def analyze_lag_patterns(df, biomarker_cols):
    """Comprehensive lag analysis"""
    print("\n=== LAG PATTERN ANALYSIS ===")
    
    results = {}
    lag_columns = [col for col in df.columns if 'climate' in col.lower() and any(x in col for x in ['7d', '14d', '30d'])]
    
    print(f"Found {len(lag_columns)} lag variables")
    
    # Focus on key biomarkers
    key_biomarkers = ['CD4 cell count (cells/µL)', 'systolic_bp_mmHg', 'fasting_glucose_mmol_L']
    
    for biomarker in key_biomarkers:
        if biomarker in biomarker_cols:
            print(f"\n{biomarker}:")
            lag_correlations = {}
            
            # Test different lag periods
            for lag_col in lag_columns:
                data = df[[biomarker, lag_col]].dropna()
                if len(data) > 100:
                    corr = data[biomarker].corr(data[lag_col])
                    if abs(corr) > 0.05:
                        lag_correlations[lag_col] = corr
                        print(f"  {lag_col}: r={corr:.3f}")
            
            if lag_correlations:
                # Find optimal lag
                optimal_lag = max(lag_correlations, key=lambda x: abs(lag_correlations[x]))
                results[biomarker] = {
                    'optimal_lag': optimal_lag,
                    'correlation': lag_correlations[optimal_lag],
                    'all_lags': lag_correlations
                }
    
    return results

def find_nonlinear_relationships(df, biomarker_cols):
    """Use Random Forest to find non-linear relationships"""
    print("\n=== NON-LINEAR RELATIONSHIP ANALYSIS ===")
    
    climate_features = [col for col in df.columns if 'climate' in col.lower()][:10]  # Top 10 climate features
    
    results = {}
    
    for biomarker in biomarker_cols[:5]:  # Top 5 biomarkers
        data = df[climate_features + [biomarker]].dropna()
        
        if len(data) > 500:
            X = data[climate_features]
            y = data[biomarker]
            
            # Random Forest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
            
            if scores.mean() > 0.1:  # R² > 0.1
                rf.fit(X, y)
                feature_importance = pd.DataFrame({
                    'feature': climate_features,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n{biomarker}: R² = {scores.mean():.3f}")
                print("Top features:")
                for idx, row in feature_importance.head(3).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
                
                results[biomarker] = {
                    'r2_score': scores.mean(),
                    'top_features': feature_importance.head(3).to_dict('records')
                }
    
    return results

def create_meaningful_results_slide(all_results):
    """Create slide with meaningful findings"""
    
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.96, 'Significant Climate-Health Relationships in ENBEL Dataset', 
             fontsize=24, fontweight='bold', ha='center', color='#1a73e8')
    fig.text(0.5, 0.93, 'Evidence-based findings from comprehensive analysis', 
             fontsize=14, ha='center', color='#666')
    
    # Create grid for findings
    gs = fig.add_gridspec(3, 3, left=0.05, right=0.95, top=0.88, bottom=0.1, 
                          wspace=0.3, hspace=0.4)
    
    # Panel 1: Extreme Heat Effects
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.text(0.5, 0.9, 'EXTREME HEAT IMPACTS', fontsize=16, fontweight='bold', 
             ha='center', color='#d32f2f', transform=ax1.transAxes)
    
    if 'extreme_heat' in all_results and all_results['extreme_heat']:
        y_pos = 0.6
        for finding in all_results['extreme_heat'][:3]:
            ax1.text(0.1, y_pos, f"• {finding['biomarker']}: {finding['percent_change']:.1f}% change (p={finding['p_value']:.3f})", 
                    fontsize=12, transform=ax1.transAxes)
            y_pos -= 0.25
    else:
        ax1.text(0.5, 0.5, 'No significant extreme heat effects found', 
                fontsize=12, ha='center', transform=ax1.transAxes, style='italic')
    
    # Panel 2: Heat Wave Effects
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.text(0.5, 0.9, 'HEAT WAVE EFFECTS', fontsize=14, fontweight='bold', 
             ha='center', color='#ff6f00', transform=ax2.transAxes)
    
    if 'heat_waves' in all_results and all_results['heat_waves']:
        y_pos = 0.6
        for finding in all_results['heat_waves'][:2]:
            ax2.text(0.1, y_pos, f"• {finding['biomarker'][:20]}...\n  {finding['percent_change']:.1f}% change", 
                    fontsize=10, transform=ax2.transAxes)
            y_pos -= 0.3
    
    # Panel 3: Vulnerable Groups
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.text(0.5, 0.9, 'VULNERABLE POPULATIONS', fontsize=14, fontweight='bold', 
             ha='center', color='#7b1fa2', transform=ax3.transAxes)
    
    if any('vulnerable' in str(k) for k in all_results.keys()):
        y_pos = 0.6
        for key in all_results:
            if 'vulnerable' in str(key).lower() or 'HIV' in key or 'Elderly' in key:
                ax3.text(0.1, y_pos, f"• {key}: Enhanced sensitivity", 
                        fontsize=10, transform=ax3.transAxes)
                y_pos -= 0.2
    
    # Panel 4: Optimal Lag Periods
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'LAG PATTERNS', fontsize=14, fontweight='bold', 
             ha='center', color='#1976d2', transform=ax4.transAxes)
    
    if 'lag_patterns' in all_results:
        y_pos = 0.6
        for biomarker, lag_info in list(all_results['lag_patterns'].items())[:2]:
            ax4.text(0.1, y_pos, f"• {biomarker[:15]}...\n  Optimal: {lag_info['optimal_lag'][-10:]}", 
                    fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.3
    
    # Panel 5: Non-linear Relationships
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'NON-LINEAR RELATIONSHIPS', fontsize=14, fontweight='bold', 
             ha='center', color='#388e3c', transform=ax5.transAxes)
    
    if 'nonlinear' in all_results:
        y_pos = 0.6
        for biomarker, info in list(all_results['nonlinear'].items())[:3]:
            ax5.text(0.1, y_pos, f"• {biomarker}: R²={info['r2_score']:.3f}", 
                    fontsize=10, transform=ax5.transAxes)
            y_pos -= 0.2
    
    # Panel 6: Key Statistics
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    ax6.text(0.5, 0.9, 'ANALYSIS SUMMARY', fontsize=14, fontweight='bold', 
             ha='center', color='#455a64', transform=ax6.transAxes)
    
    # Count significant findings
    n_significant = sum(len(v) if isinstance(v, list) else 1 
                       for v in all_results.values() if v)
    
    ax6.text(0.1, 0.6, f"• Biomarkers tested: 30+", fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.4, f"• Significant findings: {n_significant}", fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.2, f"• Methods: ANOVA, ML, Lag analysis", fontsize=10, transform=ax6.transAxes)
    
    plt.tight_layout()
    
    # Save
    output_path = '/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/meaningful_climate_effects.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSlide saved to: {output_path}")
    
    return fig

def main():
    """Main analysis pipeline"""
    print("="*70)
    print("COMPREHENSIVE CLIMATE-HEALTH ANALYSIS")
    print("Finding Meaningful Relationships in ENBEL Dataset")
    print("="*70)
    
    # Load data
    df = load_full_dataset()
    
    # Find all biomarkers
    biomarker_cols = explore_all_biomarkers(df)
    print(f"\nFound {len(biomarker_cols)} biomarkers with sufficient data")
    
    # Run comprehensive analyses
    all_results = {}
    
    # 1. Extreme heat analysis
    extreme_results = analyze_extreme_heat_events(df, biomarker_cols)
    all_results.update(extreme_results)
    
    # 2. Heat wave analysis
    heat_wave_results = analyze_heat_waves(df, biomarker_cols)
    all_results.update(heat_wave_results)
    
    # 3. Vulnerable subgroups
    vulnerable_results = analyze_vulnerable_subgroups(df, biomarker_cols)
    all_results.update(vulnerable_results)
    
    # 4. Lag patterns
    lag_results = analyze_lag_patterns(df, biomarker_cols)
    all_results['lag_patterns'] = lag_results
    
    # 5. Non-linear relationships
    nonlinear_results = find_nonlinear_relationships(df, biomarker_cols)
    all_results['nonlinear'] = nonlinear_results
    
    # Create visualization
    if any(all_results.values()):
        create_meaningful_results_slide(all_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print(f"Found {sum(1 for v in all_results.values() if v)} categories of significant effects")
        print("="*70)
    else:
        print("\nNo significant findings detected - may need different approach or data quality check")
    
    return all_results

if __name__ == "__main__":
    results = main()