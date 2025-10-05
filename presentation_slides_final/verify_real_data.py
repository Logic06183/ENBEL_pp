#!/usr/bin/env python3
"""
Verify that the significant findings are from real data, not artifacts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_examine_raw_data():
    """Load and examine the raw data for quality issues"""
    print("=== VERIFYING REAL DATA QUALITY ===")
    
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', 
                     low_memory=False)
    
    print(f"Total records: {len(df):,}")
    
    return df

def examine_suspicious_findings(df):
    """Examine the specific findings that seemed too extreme"""
    
    print("\n=== EXAMINING HEMATOCRIT VALUES ===")
    
    # Check hematocrit data
    hct_data = df['Hematocrit (%)'].dropna()
    print(f"Hematocrit observations: {len(hct_data):,}")
    print(f"Range: {hct_data.min():.1f} - {hct_data.max():.1f}%")
    print(f"Mean: {hct_data.mean():.1f}%")
    print(f"Median: {hct_data.median():.1f}%")
    
    # Check for outliers
    q25, q75 = hct_data.quantile([0.25, 0.75])
    iqr = q75 - q25
    outliers = hct_data[(hct_data < q25 - 1.5*iqr) | (hct_data > q75 + 1.5*iqr)]
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(hct_data)*100:.1f}%)")
    print(f"Outlier range: {outliers.min():.1f} - {outliers.max():.1f}%")
    
    # Check temperature categories
    if 'climate_daily_max_temp' in df.columns:
        temp_95 = df['climate_daily_max_temp'].quantile(0.95)
        temp_data = df[['Hematocrit (%)', 'climate_daily_max_temp']].dropna()
        
        normal_temp = temp_data[temp_data['climate_daily_max_temp'] < 25]
        extreme_temp = temp_data[temp_data['climate_daily_max_temp'] > temp_95]
        
        print(f"\nTemperature categories:")
        print(f"Normal (<25°C): n={len(normal_temp)}, Hct mean={normal_temp['Hematocrit (%)'].mean():.1f}%")
        print(f"Extreme (>{temp_95:.1f}°C): n={len(extreme_temp)}, Hct mean={extreme_temp['Hematocrit (%)'].mean():.1f}%")
        
        # Show actual distributions
        print(f"\nNormal temp Hct distribution:")
        print(normal_temp['Hematocrit (%)'].describe())
        print(f"\nExtreme temp Hct distribution:")
        print(extreme_temp['Hematocrit (%)'].describe())
    
    print("\n=== EXAMINING CHOLESTEROL VALUES ===")
    
    # Check cholesterol data
    chol_data = df['total_cholesterol_mg_dL'].dropna()
    print(f"Cholesterol observations: {len(chol_data):,}")
    print(f"Range: {chol_data.min():.1f} - {chol_data.max():.1f} mg/dL")
    print(f"Mean: {chol_data.mean():.1f} mg/dL")
    print(f"Median: {chol_data.median():.1f} mg/dL")
    
    # Check for impossible values
    impossible_low = chol_data[chol_data < 50]  # Impossibly low
    impossible_high = chol_data[chol_data > 500]  # Extremely high
    print(f"Impossible low (<50): {len(impossible_low)} values")
    print(f"Impossible high (>500): {len(impossible_high)} values")
    
    if len(impossible_low) > 0:
        print(f"Low values: {impossible_low.values}")
    if len(impossible_high) > 0:
        print(f"High values: {impossible_high.values}")
    
    # Check temperature relationship for cholesterol
    if 'climate_daily_max_temp' in df.columns:
        temp_chol_data = df[['total_cholesterol_mg_dL', 'climate_daily_max_temp']].dropna()
        
        normal_temp_chol = temp_chol_data[temp_chol_data['climate_daily_max_temp'] < 25]
        extreme_temp_chol = temp_chol_data[temp_chol_data['climate_daily_max_temp'] > temp_95]
        
        print(f"\nCholesterol by temperature:")
        print(f"Normal temp: n={len(normal_temp_chol)}, mean={normal_temp_chol['total_cholesterol_mg_dL'].mean():.1f}")
        print(f"Extreme temp: n={len(extreme_temp_chol)}, mean={extreme_temp_chol['total_cholesterol_mg_dL'].mean():.1f}")

def check_data_collection_dates(df):
    """Check if extreme values correlate with specific dates/periods"""
    
    print("\n=== CHECKING DATA COLLECTION PATTERNS ===")
    
    # Convert visit date
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df['year'] = df['visit_date'].dt.year
    df['month'] = df['visit_date'].dt.month
    
    # Check hematocrit by year
    hct_by_year = df.groupby('year')['Hematocrit (%)'].agg(['count', 'mean', 'std']).round(2)
    print("Hematocrit by year:")
    print(hct_by_year)
    
    # Check cholesterol by year
    chol_by_year = df.groupby('year')['total_cholesterol_mg_dL'].agg(['count', 'mean', 'std']).round(2)
    print("\nCholesterol by year:")
    print(chol_by_year)
    
    # Check for specific studies with extreme values
    if 'data_source' in df.columns:
        print("\nHematocrit by data source:")
        hct_by_source = df.groupby('data_source')['Hematocrit (%)'].agg(['count', 'mean', 'std']).round(2)
        print(hct_by_source)

def check_unit_consistency(df):
    """Check if values suggest unit conversion errors"""
    
    print("\n=== CHECKING UNIT CONSISTENCY ===")
    
    # Hematocrit should be 35-45% for normal adults
    hct_data = df['Hematocrit (%)'].dropna()
    normal_range = hct_data[(hct_data >= 30) & (hct_data <= 50)]
    outside_range = hct_data[(hct_data < 30) | (hct_data > 50)]
    
    print(f"Hematocrit in normal range (30-50%): {len(normal_range):,} ({len(normal_range)/len(hct_data)*100:.1f}%)")
    print(f"Hematocrit outside normal range: {len(outside_range):,} ({len(outside_range)/len(hct_data)*100:.1f}%)")
    
    # Check if low values might be in different units (0.35 instead of 35%)
    very_low = hct_data[hct_data < 1]
    if len(very_low) > 0:
        print(f"Very low values (<1%): {len(very_low)} - might be decimal format")
        print(f"Sample low values: {very_low.head().values}")
    
    # Cholesterol should be 150-250 mg/dL typically
    chol_data = df['total_cholesterol_mg_dL'].dropna()
    normal_chol = chol_data[(chol_data >= 100) & (chol_data <= 300)]
    outside_chol = chol_data[(chol_data < 100) | (chol_data > 300)]
    
    print(f"\nCholesterol in reasonable range (100-300): {len(normal_chol):,} ({len(normal_chol)/len(chol_data)*100:.1f}%)")
    print(f"Cholesterol outside range: {len(outside_chol):,} ({len(outside_chol)/len(chol_data)*100:.1f}%)")

def perform_sanity_check_analysis(df):
    """Perform a more conservative analysis to see if effects persist"""
    
    print("\n=== SANITY CHECK ANALYSIS ===")
    
    # Remove obvious outliers and retest
    
    # Hematocrit: keep only reasonable values
    hct_reasonable = df[(df['Hematocrit (%)'] >= 20) & (df['Hematocrit (%)'] <= 60)]
    print(f"Hematocrit after outlier removal: {len(hct_reasonable):,} records")
    
    # Cholesterol: keep only reasonable values  
    chol_reasonable = df[(df['total_cholesterol_mg_dL'] >= 50) & (df['total_cholesterol_mg_dL'] <= 400)]
    print(f"Cholesterol after outlier removal: {len(chol_reasonable):,} records")
    
    # Retest temperature effects with cleaned data
    if 'climate_daily_max_temp' in df.columns:
        temp_95 = df['climate_daily_max_temp'].quantile(0.95)
        
        # Test hematocrit
        hct_temp_clean = hct_reasonable[['Hematocrit (%)', 'climate_daily_max_temp']].dropna()
        normal_hct = hct_temp_clean[hct_temp_clean['climate_daily_max_temp'] < 25]['Hematocrit (%)']
        extreme_hct = hct_temp_clean[hct_temp_clean['climate_daily_max_temp'] > temp_95]['Hematocrit (%)']
        
        if len(normal_hct) > 30 and len(extreme_hct) > 30:
            t_stat, p_val = stats.ttest_ind(extreme_hct, normal_hct)
            effect_size = (extreme_hct.mean() - normal_hct.mean()) / normal_hct.std()
            percent_change = ((extreme_hct.mean() - normal_hct.mean()) / normal_hct.mean()) * 100
            
            print(f"\nCLEANED Hematocrit analysis:")
            print(f"Normal temp (n={len(normal_hct)}): {normal_hct.mean():.2f}% ± {normal_hct.std():.2f}")
            print(f"Extreme temp (n={len(extreme_hct)}): {extreme_hct.mean():.2f}% ± {extreme_hct.std():.2f}")
            print(f"Percent change: {percent_change:.1f}%")
            print(f"Effect size: {effect_size:.3f}")
            print(f"P-value: {p_val:.6f}")
        
        # Test cholesterol
        chol_temp_clean = chol_reasonable[['total_cholesterol_mg_dL', 'climate_daily_max_temp']].dropna()
        normal_chol = chol_temp_clean[chol_temp_clean['climate_daily_max_temp'] < 25]['total_cholesterol_mg_dL']
        extreme_chol = chol_temp_clean[chol_temp_clean['climate_daily_max_temp'] > temp_95]['total_cholesterol_mg_dL']
        
        if len(normal_chol) > 30 and len(extreme_chol) > 30:
            t_stat, p_val = stats.ttest_ind(extreme_chol, normal_chol)
            effect_size = (extreme_chol.mean() - normal_chol.mean()) / normal_chol.std()
            percent_change = ((extreme_chol.mean() - normal_chol.mean()) / normal_chol.mean()) * 100
            
            print(f"\nCLEANED Cholesterol analysis:")
            print(f"Normal temp (n={len(normal_chol)}): {normal_chol.mean():.1f} ± {normal_chol.std():.1f}")
            print(f"Extreme temp (n={len(extreme_chol)}): {extreme_chol.mean():.1f} ± {extreme_chol.std():.1f}")
            print(f"Percent change: {percent_change:.1f}%")
            print(f"Effect size: {effect_size:.3f}")
            print(f"P-value: {p_val:.6f}")

def main():
    """Main verification pipeline"""
    
    df = load_and_examine_raw_data()
    examine_suspicious_findings(df)
    check_data_collection_dates(df)
    check_unit_consistency(df)
    perform_sanity_check_analysis(df)
    
    print("\n" + "="*60)
    print("DATA VERIFICATION COMPLETE")
    print("Check the output above for data quality issues")
    print("="*60)

if __name__ == "__main__":
    main()