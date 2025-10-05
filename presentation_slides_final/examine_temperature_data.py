#!/usr/bin/env python3
"""
Examine the actual temperature data in the clinical trial dataset
to understand why we're not seeing hot days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def examine_all_temperature_columns(df):
    """Examine all temperature-related columns"""
    print("=== ALL TEMPERATURE COLUMNS ===")
    
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    
    for col in temp_cols:
        if df[col].notna().sum() > 100:
            print(f"\n{col}:")
            print(f"  Count: {df[col].notna().sum():,}")
            print(f"  Range: {df[col].min():.1f} - {df[col].max():.1f}")
            print(f"  Mean: {df[col].mean():.1f}")
            print(f"  90th percentile: {df[col].quantile(0.90):.1f}")
            print(f"  95th percentile: {df[col].quantile(0.95):.1f}")
            print(f"  99th percentile: {df[col].quantile(0.99):.1f}")

def examine_dates_and_seasons(df):
    """Look at the date distribution and seasonal patterns"""
    print("\n=== DATE AND SEASONAL ANALYSIS ===")
    
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df['year'] = df['visit_date'].dt.year
    df['month'] = df['visit_date'].dt.month
    df['season'] = df['month'].map({
        12: 'Summer', 1: 'Summer', 2: 'Summer',
        3: 'Autumn', 4: 'Autumn', 5: 'Autumn', 
        6: 'Winter', 7: 'Winter', 8: 'Winter',
        9: 'Spring', 10: 'Spring', 11: 'Spring'
    })
    
    print(f"Date range: {df['visit_date'].min()} to {df['visit_date'].max()}")
    print(f"Years covered: {sorted(df['year'].dropna().unique())}")
    
    # Monthly distribution
    print(f"\nMonthly distribution:")
    monthly = df['month'].value_counts().sort_index()
    for month, count in monthly.items():
        month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]
        print(f"  {month_name}: {count:,}")
    
    # Seasonal distribution
    print(f"\nSeasonal distribution:")
    seasonal = df['season'].value_counts()
    for season, count in seasonal.items():
        print(f"  {season}: {count:,}")

def examine_temperature_by_season(df):
    """Look at temperature patterns by season"""
    print("\n=== TEMPERATURE BY SEASON ===")
    
    # Use the most reliable temperature column
    temp_cols = ['climate_daily_max_temp', 'climate_daily_mean_temp', 'climate_7d_max_temp']
    
    for temp_col in temp_cols:
        if temp_col in df.columns and df[temp_col].notna().sum() > 100:
            print(f"\n{temp_col} by season:")
            
            seasonal_temps = df.groupby('season')[temp_col].agg(['count', 'mean', 'std', 'min', 'max'])
            print(seasonal_temps.round(1))
            
            # Check for hot days (>30°C) by season
            hot_days = df[df[temp_col] > 30].groupby('season').size()
            if len(hot_days) > 0:
                print(f"\nDays >30°C:")
                for season, count in hot_days.items():
                    print(f"  {season}: {count}")
            else:
                print(f"\nNo days >30°C found in {temp_col}")
            
            # Check for very hot days (>35°C)
            very_hot = df[df[temp_col] > 35].groupby('season').size()
            if len(very_hot) > 0:
                print(f"\nDays >35°C:")
                for season, count in very_hot.items():
                    print(f"  {season}: {count}")

def check_unit_issues(df):
    """Check if temperature might be in different units"""
    print("\n=== CHECKING FOR UNIT ISSUES ===")
    
    temp_cols = ['climate_daily_max_temp', 'climate_daily_mean_temp', 'body_temperature_celsius']
    
    for col in temp_cols:
        if col in df.columns and df[col].notna().sum() > 10:
            temps = df[col].dropna()
            
            print(f"\n{col}:")
            print(f"  Min: {temps.min():.2f}")
            print(f"  Max: {temps.max():.2f}")
            print(f"  Range check:")
            
            # Check if in Celsius (should be 15-40°C for Johannesburg)
            celsius_range = temps[(temps >= 10) & (temps <= 45)]
            print(f"    Celsius range (10-45°C): {len(celsius_range):,} ({len(celsius_range)/len(temps)*100:.1f}%)")
            
            # Check if in Kelvin (should be 288-318K for Johannesburg)
            kelvin_range = temps[(temps >= 280) & (temps <= 320)]
            print(f"    Kelvin range (280-320K): {len(kelvin_range):,} ({len(kelvin_range)/len(temps)*100:.1f}%)")
            
            # Check if in Fahrenheit (should be 59-104°F for Johannesburg)
            fahr_range = temps[(temps >= 50) & (temps <= 110)]
            print(f"    Fahrenheit range (50-110°F): {len(fahr_range):,} ({len(fahr_range)/len(temps)*100:.1f}%)")

def check_summer_temperatures(df):
    """Specifically check summer temperatures in Johannesburg"""
    print("\n=== JOHANNESBURG SUMMER ANALYSIS ===")
    
    # Johannesburg summer is Dec-Feb
    summer_months = [12, 1, 2]
    summer_data = df[df['month'].isin(summer_months)]
    
    print(f"Summer records (Dec-Feb): {len(summer_data):,}")
    
    if len(summer_data) > 0:
        temp_cols = ['climate_daily_max_temp', 'climate_daily_mean_temp']
        
        for temp_col in temp_cols:
            if temp_col in summer_data.columns:
                summer_temps = summer_data[temp_col].dropna()
                
                if len(summer_temps) > 0:
                    print(f"\n{temp_col} in summer:")
                    print(f"  Count: {len(summer_temps):,}")
                    print(f"  Mean: {summer_temps.mean():.1f}°C")
                    print(f"  Max: {summer_temps.max():.1f}°C")
                    print(f"  95th percentile: {summer_temps.quantile(0.95):.1f}°C")
                    print(f"  Days >25°C: {(summer_temps > 25).sum()}")
                    print(f"  Days >30°C: {(summer_temps > 30).sum()}")
                    print(f"  Days >35°C: {(summer_temps > 35).sum()}")
                    
                    # Johannesburg typical summer max should be 25-30°C
                    if summer_temps.max() < 20:
                        print("  ⚠️  WARNING: Max summer temp too low for Johannesburg!")
                    if summer_temps.mean() < 15:
                        print("  ⚠️  WARNING: Mean summer temp too low for Johannesburg!")

def compare_with_expected_johannesburg_climate():
    """Compare with expected Johannesburg climate"""
    print("\n=== EXPECTED vs OBSERVED CLIMATE ===")
    
    print("Expected Johannesburg climate:")
    print("  Summer (Dec-Feb): 16-26°C daily, max can reach 30-35°C")
    print("  Winter (Jun-Aug): 4-18°C daily, min can drop to 0°C")
    print("  Altitude: 1,753m (affects temperature)")
    print("  Hot days (>30°C): 10-20 days per year typically")
    print("  Very hot days (>35°C): 1-5 days per year")

def main():
    """Main examination function"""
    print("="*60)
    print("TEMPERATURE DATA EXAMINATION")
    print("Investigating why no hot days detected")
    print("="*60)
    
    # Load the clinical dataset
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv', 
                     low_memory=False)
    
    print(f"Total records: {len(df):,}")
    
    # Add date columns
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df['month'] = df['visit_date'].dt.month
    
    # Run examinations
    examine_all_temperature_columns(df)
    examine_dates_and_seasons(df)
    examine_temperature_by_season(df)
    check_unit_issues(df)
    check_summer_temperatures(df)
    compare_with_expected_johannesburg_climate()
    
    print("\n" + "="*60)
    print("EXAMINATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()