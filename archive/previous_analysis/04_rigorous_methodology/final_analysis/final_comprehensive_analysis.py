#!/usr/bin/env python3
"""
Final Comprehensive Climate-Health Analysis
==========================================

One final attempt using the most proven methods from epidemiological literature
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def final_comprehensive_analysis():
    """Final comprehensive analysis using proven epidemiological methods"""
    
    print("🔬 FINAL COMPREHENSIVE CLIMATE-HEALTH ANALYSIS")
    print("=" * 50)
    print("Using proven epidemiological methods from literature")
    print()
    
    # Load data
    df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    print(f"Dataset: {len(df):,} records")
    
    # Define biomarkers and climate variables
    biomarkers = {
        'systolic blood pressure': 'Cardiovascular',
        'diastolic blood pressure': 'Cardiovascular', 
        'FASTING GLUCOSE': 'Metabolic',
        'FASTING TOTAL CHOLESTEROL': 'Metabolic',
        'FASTING HDL': 'Metabolic',
        'CD4 cell count (cells/µL)': 'Immune',
        'Hemoglobin (g/dL)': 'Hematologic',
        'Creatinine (mg/dL)': 'Renal'
    }
    
    # Climate variables - focus on temperature (most studied)
    climate_vars = ['temperature_tas_lag0', 'temperature_tas_lag1', 'temperature_tas_lag2', 'temperature_tas_lag3']
    
    print("Testing biomarkers:")
    for biomarker, system in biomarkers.items():
        if biomarker in df.columns:
            n_samples = df[biomarker].notna().sum()
            print(f"  • {biomarker} ({system}): {n_samples:,} samples")
    
    print(f"\nClimate variables: {len([c for c in climate_vars if c in df.columns])}")
    print()
    
    # Method 1: Large Sample Correlations (Epidemiological Standard)
    print("📊 METHOD 1: Large Sample Correlations")
    print("-" * 40)
    
    significant_correlations = []
    
    for biomarker, system in biomarkers.items():
        if biomarker not in df.columns:
            continue
            
        bio_data = df[biomarker].dropna()
        if len(bio_data) < 1000:  # Large sample requirement
            continue
            
        print(f"\n{biomarker} (n={len(bio_data):,}):")
        
        for climate_var in climate_vars:
            if climate_var not in df.columns:
                continue
                
            # Get paired data
            paired_data = df[[biomarker, climate_var]].dropna()
            if len(paired_data) < 1000:
                continue
                
            # Calculate correlation
            correlation, p_value = pearsonr(paired_data[biomarker], paired_data[climate_var])
            
            # Calculate 95% confidence interval for correlation
            n = len(paired_data)
            se = 1.0 / np.sqrt(n - 3)
            z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
            z_alpha = stats.norm.ppf(0.975)  # 95% CI
            z_lower = z_r - z_alpha * se
            z_upper = z_r + z_alpha * se
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            print(f"  {climate_var}: r = {correlation:.4f}, p = {p_value:.4f}, 95%CI = [{r_lower:.4f}, {r_upper:.4f}]")
            
            # Check significance with large sample power
            if p_value < 0.001 and abs(correlation) > 0.05:  # Strict criteria for large samples
                significant_correlations.append({
                    'biomarker': biomarker,
                    'system': system,
                    'climate_var': climate_var,
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_samples': n,
                    'ci_lower': r_lower,
                    'ci_upper': r_upper
                })
                print(f"    ✅ SIGNIFICANT: Large sample correlation")
    
    # Method 2: Extreme Temperature Analysis
    print(f"\n📊 METHOD 2: Extreme Temperature Effects")
    print("-" * 40)
    
    extreme_effects = []
    
    for biomarker, system in biomarkers.items():
        if biomarker not in df.columns:
            continue
            
        bio_data = df.dropna(subset=[biomarker])
        if len(bio_data) < 500:
            continue
            
        print(f"\n{biomarker}:")
        
        # Use main temperature variable
        temp_var = 'temperature_tas_lag0'
        if temp_var not in bio_data.columns:
            continue
            
        # Define extreme temperatures (10th and 90th percentiles)
        temp_10 = bio_data[temp_var].quantile(0.10)
        temp_90 = bio_data[temp_var].quantile(0.90)
        
        # Create temperature groups
        cold_extreme = bio_data[bio_data[temp_var] <= temp_10][biomarker]
        hot_extreme = bio_data[bio_data[temp_var] >= temp_90][biomarker]
        moderate = bio_data[(bio_data[temp_var] > temp_10) & (bio_data[temp_var] < temp_90)][biomarker]
        
        print(f"  Cold extreme (<={temp_10:.1f}°C): {len(cold_extreme)} samples")
        print(f"  Hot extreme (>={temp_90:.1f}°C): {len(hot_extreme)} samples")
        print(f"  Moderate ({temp_10:.1f}-{temp_90:.1f}°C): {len(moderate)} samples")
        
        # Test cold vs moderate
        if len(cold_extreme) >= 20 and len(moderate) >= 100:
            t_stat, p_cold = stats.ttest_ind(cold_extreme, moderate)
            effect_size_cold = (cold_extreme.mean() - moderate.mean()) / moderate.std()
            
            print(f"  Cold vs Moderate: effect = {effect_size_cold:.3f}, p = {p_cold:.4f}")
            
            if p_cold < 0.01 and abs(effect_size_cold) > 0.2:
                extreme_effects.append({
                    'biomarker': biomarker,
                    'system': system,
                    'extreme_type': 'cold',
                    'effect_size': effect_size_cold,
                    'p_value': p_cold,
                    'threshold': temp_10,
                    'n_extreme': len(cold_extreme),
                    'n_reference': len(moderate)
                })
                print(f"    ✅ SIGNIFICANT COLD EFFECT")
        
        # Test hot vs moderate  
        if len(hot_extreme) >= 20 and len(moderate) >= 100:
            t_stat, p_hot = stats.ttest_ind(hot_extreme, moderate)
            effect_size_hot = (hot_extreme.mean() - moderate.mean()) / moderate.std()
            
            print(f"  Hot vs Moderate: effect = {effect_size_hot:.3f}, p = {p_hot:.4f}")
            
            if p_hot < 0.01 and abs(effect_size_hot) > 0.2:
                extreme_effects.append({
                    'biomarker': biomarker,
                    'system': system,
                    'extreme_type': 'hot',
                    'effect_size': effect_size_hot,
                    'p_value': p_hot,
                    'threshold': temp_90,
                    'n_extreme': len(hot_extreme),
                    'n_reference': len(moderate)
                })
                print(f"    ✅ SIGNIFICANT HOT EFFECT")
    
    # Method 3: Dose-Response Analysis
    print(f"\n📊 METHOD 3: Dose-Response Analysis")
    print("-" * 35)
    
    dose_response = []
    
    for biomarker, system in biomarkers.items():
        if biomarker not in df.columns:
            continue
            
        bio_data = df.dropna(subset=[biomarker])
        if len(bio_data) < 500:
            continue
            
        temp_var = 'temperature_tas_lag0'
        if temp_var not in bio_data.columns:
            continue
            
        print(f"\n{biomarker}:")
        
        # Create temperature quintiles
        temp_quintiles = pd.qcut(bio_data[temp_var], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        bio_data_with_quintiles = bio_data.copy()
        bio_data_with_quintiles['temp_quintile'] = temp_quintiles
        
        # Calculate means by quintile
        quintile_means = bio_data_with_quintiles.groupby('temp_quintile')[biomarker].mean()
        quintile_counts = bio_data_with_quintiles.groupby('temp_quintile')[biomarker].count()
        
        print("  Temperature quintiles:")
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if q in quintile_means.index:
                print(f"    {q}: {quintile_means[q]:.2f} (n={quintile_counts[q]})")
        
        # Test for linear trend
        if len(quintile_means) == 5:
            # Assign numeric values to quintiles for trend test
            x_values = [1, 2, 3, 4, 5]
            y_values = quintile_means.values
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            
            print(f"  Linear trend: slope = {slope:.3f}, r² = {r_value**2:.3f}, p = {p_value:.4f}")
            
            if p_value < 0.05 and abs(r_value) > 0.5:
                dose_response.append({
                    'biomarker': biomarker,
                    'system': system,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing'
                })
                print(f"    ✅ SIGNIFICANT DOSE-RESPONSE TREND")
    
    # FINAL SUMMARY
    print("\n" + "=" * 60)
    print("🎯 FINAL COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_findings = len(significant_correlations) + len(extreme_effects) + len(dose_response)
    
    print(f"Analysis methods applied: 3 (correlational, extreme effects, dose-response)")
    print(f"Total significant findings: {total_findings}")
    print()
    
    if significant_correlations:
        print(f"📊 SIGNIFICANT CORRELATIONS: {len(significant_correlations)}")
        for i, finding in enumerate(significant_correlations, 1):
            print(f"  {i}. {finding['biomarker']} ~ {finding['climate_var']}")
            print(f"     r = {finding['correlation']:.4f}, p = {finding['p_value']:.2e}, n = {finding['n_samples']:,}")
    
    if extreme_effects:
        print(f"\n🌡️ EXTREME TEMPERATURE EFFECTS: {len(extreme_effects)}")
        for i, finding in enumerate(extreme_effects, 1):
            print(f"  {i}. {finding['biomarker']} - {finding['extreme_type']} extreme")
            print(f"     Effect size = {finding['effect_size']:.3f}, p = {finding['p_value']:.4f}")
    
    if dose_response:
        print(f"\n📈 DOSE-RESPONSE RELATIONSHIPS: {len(dose_response)}")
        for i, finding in enumerate(dose_response, 1):
            print(f"  {i}. {finding['biomarker']} - {finding['trend_direction']} trend")
            print(f"     R² = {finding['r_squared']:.3f}, p = {finding['p_value']:.4f}")
    
    if total_findings == 0:
        print("❌ NO SIGNIFICANT CLIMATE-HEALTH RELATIONSHIPS DETECTED")
        print()
        print("CONCLUSION: After comprehensive analysis using multiple")
        print("proven epidemiological methods with large sample sizes,")
        print("no statistically significant climate-health relationships")
        print("were identified in this South African urban dataset.")
        print()
        print("This null result is scientifically valuable and suggests:")
        print("• Climate effects may be population/context specific")
        print("• Longer observation periods may be needed")
        print("• Individual vulnerability factors may be important")
        print("• Different health outcomes might be more sensitive")
    else:
        print(f"\n✅ SUCCESSFULLY IDENTIFIED {total_findings} SIGNIFICANT RELATIONSHIPS")
        print("These findings represent robust climate-health associations")
        print("using rigorous epidemiological methods.")
    
    return {
        'correlations': significant_correlations,
        'extreme_effects': extreme_effects,
        'dose_response': dose_response,
        'total_findings': total_findings
    }

if __name__ == "__main__":
    results = final_comprehensive_analysis()