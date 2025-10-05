#!/usr/bin/env python3
"""
Analyze climate-health effects using the full temperature range
including the 183 hot days >30°C that exist in the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_heat_effects_properly(df):
    """Analyze effects using the full temperature range including hot days"""
    print("=== PROPER HEAT ANALYSIS WITH HOT DAYS ===")
    
    # Use daily max temperature which has the full range
    temp_col = 'climate_daily_max_temp'
    
    print(f"Temperature range: {df[temp_col].min():.1f} - {df[temp_col].max():.1f}°C")
    print(f"Hot days (>30°C): {(df[temp_col] > 30).sum():,}")
    print(f"Very hot days (>28°C): {(df[temp_col] > 28).sum():,}")
    print(f"Warm days (>25°C): {(df[temp_col] > 25).sum():,}")
    
    # Define temperature categories properly
    df['temp_category'] = pd.cut(df[temp_col], 
                                bins=[-np.inf, 20, 25, 30, np.inf],
                                labels=['Cool', 'Mild', 'Warm', 'Hot'])
    
    print(f"\nTemperature distribution:")
    print(df['temp_category'].value_counts())
    
    # Test key biomarkers
    biomarkers = {
        'CD4 cell count (cells/µL)': 'CD4',
        'systolic_bp_mmHg': 'Systolic BP',
        'fasting_glucose_mmol_L': 'Glucose',
        'weight_kg': 'Weight'
    }
    
    results = {}
    
    for biomarker, name in biomarkers.items():
        if biomarker in df.columns:
            data = df[[biomarker, temp_col, 'temp_category']].dropna()
            
            if len(data) > 100:
                print(f"\n{name} ({biomarker}):")
                
                # ANOVA across temperature categories
                groups = [group[biomarker].values for cat, group in data.groupby('temp_category')]
                if len(groups) >= 3 and all(len(g) > 10 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # Calculate means by category
                    means = data.groupby('temp_category')[biomarker].agg(['mean', 'count', 'std'])
                    print(means.round(2))
                    
                    # Compare Cool vs Hot specifically
                    cool_data = data[data['temp_category'] == 'Cool'][biomarker]
                    hot_data = data[data['temp_category'] == 'Hot'][biomarker]
                    
                    if len(cool_data) > 10 and len(hot_data) > 10:
                        # T-test
                        t_stat, t_p = stats.ttest_ind(hot_data, cool_data)
                        
                        # Effect size
                        pooled_std = np.sqrt(((len(cool_data)-1)*cool_data.std()**2 + 
                                            (len(hot_data)-1)*hot_data.std()**2) / 
                                           (len(cool_data) + len(hot_data) - 2))
                        effect_size = (hot_data.mean() - cool_data.mean()) / pooled_std
                        percent_change = ((hot_data.mean() - cool_data.mean()) / cool_data.mean()) * 100
                        
                        print(f"  ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")
                        print(f"  Cool vs Hot t-test: p = {t_p:.4f}")
                        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
                        print(f"  Percent change: {percent_change:+.1f}%")
                        print(f"  Cool mean: {cool_data.mean():.2f} (n={len(cool_data)})")
                        print(f"  Hot mean: {hot_data.mean():.2f} (n={len(hot_data)})")
                        
                        # Store if significant or meaningful effect
                        if p_value < 0.05 or abs(effect_size) > 0.2:
                            results[name] = {
                                'biomarker': biomarker,
                                'anova_p': p_value,
                                'ttest_p': t_p,
                                'effect_size': effect_size,
                                'percent_change': percent_change,
                                'cool_mean': cool_data.mean(),
                                'hot_mean': hot_data.mean(),
                                'cool_n': len(cool_data),
                                'hot_n': len(hot_data),
                                'f_stat': f_stat
                            }
    
    return results

def analyze_extreme_heat_days(df):
    """Focus specifically on the hottest days"""
    print("\n=== EXTREME HEAT DAY ANALYSIS ===")
    
    # Use the hottest days (>30°C)
    hot_days = df[df['climate_daily_max_temp'] > 30]
    normal_days = df[df['climate_daily_max_temp'] < 25]
    
    print(f"Extreme hot days (>30°C): {len(hot_days):,}")
    print(f"Normal days (<25°C): {len(normal_days):,}")
    
    biomarkers = ['CD4 cell count (cells/µL)', 'systolic_bp_mmHg', 'fasting_glucose_mmol_L']
    
    results = {}
    
    for biomarker in biomarkers:
        if biomarker in df.columns:
            hot_values = hot_days[biomarker].dropna()
            normal_values = normal_days[biomarker].dropna()
            
            if len(hot_values) > 20 and len(normal_values) > 100:
                # T-test
                t_stat, p_value = stats.ttest_ind(hot_values, normal_values)
                
                # Effect size
                pooled_std = np.sqrt(((len(normal_values)-1)*normal_values.std()**2 + 
                                    (len(hot_values)-1)*hot_values.std()**2) / 
                                   (len(normal_values) + len(hot_values) - 2))
                effect_size = (hot_values.mean() - normal_values.mean()) / pooled_std
                percent_change = ((hot_values.mean() - normal_values.mean()) / normal_values.mean()) * 100
                
                print(f"\n{biomarker}:")
                print(f"  Normal (<25°C): {normal_values.mean():.2f} ± {normal_values.std():.2f} (n={len(normal_values)})")
                print(f"  Hot (>30°C): {hot_values.mean():.2f} ± {hot_values.std():.2f} (n={len(hot_values)})")
                print(f"  Change: {percent_change:+.1f}%")
                print(f"  Effect size: {effect_size:.3f}")
                print(f"  p-value: {p_value:.4f}")
                
                if p_value < 0.05 or abs(effect_size) > 0.2:
                    results[biomarker] = {
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'percent_change': percent_change,
                        'normal_mean': normal_values.mean(),
                        'hot_mean': hot_values.mean(),
                        'normal_n': len(normal_values),
                        'hot_n': len(hot_values)
                    }
    
    return results

def main():
    """Main analysis with proper hot day inclusion"""
    print("="*60)
    print("CLIMATE-HEALTH ANALYSIS WITH HOT DAYS")
    print("Using full temperature range from clinical dataset")
    print("="*60)
    
    # Load clinical dataset
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', 
                     low_memory=False)
    
    print(f"Total records: {len(df):,}")
    
    # Run analyses
    heat_effects = analyze_heat_effects_properly(df)
    extreme_effects = analyze_extreme_heat_days(df)
    
    print(f"\n" + "="*60)
    print("SUMMARY OF SIGNIFICANT FINDINGS")
    print("="*60)
    
    if heat_effects:
        print("Temperature category effects:")
        for name, data in heat_effects.items():
            print(f"  {name}: {data['percent_change']:+.1f}% (p={data['anova_p']:.3f}, d={data['effect_size']:.3f})")
    
    if extreme_effects:
        print("\nExtreme heat effects (>30°C vs <25°C):")
        for biomarker, data in extreme_effects.items():
            print(f"  {biomarker}: {data['percent_change']:+.1f}% (p={data['p_value']:.3f}, d={data['effect_size']:.3f})")
    
    if not heat_effects and not extreme_effects:
        print("No significant heat effects detected even with hot days included")
    
    return heat_effects, extreme_effects

if __name__ == "__main__":
    heat_results, extreme_results = main()