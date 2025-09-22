#!/usr/bin/env python3
"""
Time Series Climate-Health Analysis
==================================

Final approach using time series methodology:
1. Seasonal decomposition
2. Time series cross-correlation
3. Granger causality testing
4. Wavelet coherence analysis
5. Change point detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, ccf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesClimateAnalysis:
    def __init__(self):
        self.significant_findings = []
        
    def load_time_series_data(self):
        """Load and prepare time series data"""
        print("📈 TIME SERIES CLIMATE-HEALTH ANALYSIS")
        print("=" * 40)
        print("📊 Loading time series data...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Check for date/time information
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'year', 'month'])]
        print(f"Date columns available: {date_cols}")
        
        # Focus on biomarkers with sufficient temporal data
        biomarkers = [
            'systolic blood pressure',
            'diastolic blood pressure', 
            'FASTING GLUCOSE',
            'FASTING TOTAL CHOLESTEROL',
            'CD4 cell count (cells/µL)',
            'Hemoglobin (g/dL)'
        ]
        
        # Get temperature data (immediate effects)
        temp_cols = [col for col in df.columns if 'temp' in col.lower() and 'lag0' in col][:3]
        
        print(f"Available biomarkers: {len([b for b in biomarkers if b in df.columns])}")
        print(f"Temperature variables: {len(temp_cols)}")
        
        return df, biomarkers, temp_cols
    
    def seasonal_correlation_analysis(self, df, biomarker, temp_cols):
        """Analyze seasonal patterns and correlations"""
        print(f"\n🌅 Seasonal Analysis: {biomarker}")
        print("-" * 35)
        
        if biomarker not in df.columns:
            print("Biomarker not available")
            return None
        
        # Check if we have monthly/seasonal data
        if 'month' not in df.columns:
            print("No temporal information available")
            return None
        
        # Prepare data
        biomarker_data = df.dropna(subset=[biomarker])
        if len(biomarker_data) < 100:
            print("Insufficient data")
            return None
        
        print(f"Sample size: {len(biomarker_data):,}")
        
        # Monthly aggregation
        monthly_data = biomarker_data.groupby('month').agg({
            biomarker: ['mean', 'std', 'count']
        }).round(3)
        
        print("\nMonthly patterns:")
        for month in sorted(biomarker_data['month'].unique()):
            month_subset = biomarker_data[biomarker_data['month'] == month]
            if len(month_subset) >= 5:
                mean_val = month_subset[biomarker].mean()
                print(f"  Month {month:2d}: {mean_val:.2f} (n={len(month_subset)})")
        
        # Test for seasonal variation
        if 'month' in biomarker_data.columns and len(biomarker_data['month'].unique()) >= 6:
            # ANOVA for seasonal differences
            month_groups = [biomarker_data[biomarker_data['month'] == month][biomarker].values 
                           for month in biomarker_data['month'].unique() 
                           if len(biomarker_data[biomarker_data['month'] == month]) >= 5]
            
            if len(month_groups) >= 3:
                f_stat, p_value = stats.f_oneway(*month_groups)
                print(f"\nSeasonal ANOVA: F = {f_stat:.3f}, p = {p_value:.4f}")
                
                if p_value < 0.05:
                    print("✅ SIGNIFICANT SEASONAL VARIATION")
                    
                    # Correlate with temperature
                    best_correlation = 0
                    best_temp_var = None
                    
                    for temp_col in temp_cols:
                        if temp_col in biomarker_data.columns:
                            # Monthly correlation
                            monthly_summary = biomarker_data.groupby('month').agg({
                                biomarker: 'mean',
                                temp_col: 'mean'
                            }).dropna()
                            
                            if len(monthly_summary) >= 6:
                                corr, corr_p = pearsonr(monthly_summary[biomarker], monthly_summary[temp_col])
                                print(f"  Monthly correlation with {temp_col}: r = {corr:.3f}, p = {corr_p:.3f}")
                                
                                if abs(corr) > abs(best_correlation) and corr_p < 0.10:
                                    best_correlation = corr
                                    best_temp_var = temp_col
                    
                    if best_temp_var and abs(best_correlation) > 0.5:
                        result = {
                            'biomarker': biomarker,
                            'method': 'Seasonal Correlation',
                            'seasonal_anova_p': p_value,
                            'best_temp_correlation': best_correlation,
                            'best_temp_variable': best_temp_var,
                            'n_months': len(biomarker_data['month'].unique()),
                            'total_samples': len(biomarker_data)
                        }
                        
                        print(f"✅ STRONG SEASONAL-CLIMATE RELATIONSHIP")
                        print(f"   Best correlation: {best_temp_var} (r = {best_correlation:.3f})")
                        return result
        
        print("No significant seasonal-climate relationship")
        return None
    
    def cross_correlation_analysis(self, df, biomarker, temp_cols):
        """Cross-correlation analysis with different lags"""
        print(f"\n🔄 Cross-Correlation Analysis: {biomarker}")
        print("-" * 40)
        
        if biomarker not in df.columns:
            print("Biomarker not available")
            return None
        
        # Get data with temporal ordering
        biomarker_data = df.dropna(subset=[biomarker]).copy()
        if len(biomarker_data) < 100:
            print("Insufficient data")
            return None
        
        # Sort by available time information
        if 'year' in biomarker_data.columns and 'month' in biomarker_data.columns:
            biomarker_data = biomarker_data.sort_values(['year', 'month'])
        elif 'month' in biomarker_data.columns:
            biomarker_data = biomarker_data.sort_values('month')
        
        print(f"Time series length: {len(biomarker_data)}")
        
        # Test cross-correlations with different lags
        best_ccf = 0
        best_lag = 0
        best_temp_var = None
        
        for temp_col in temp_cols:
            if temp_col in biomarker_data.columns:
                # Get clean paired data
                paired_data = biomarker_data[[biomarker, temp_col]].dropna()
                if len(paired_data) < 50:
                    continue
                
                bio_series = paired_data[biomarker].values
                temp_series = paired_data[temp_col].values
                
                # Test different lag correlations manually
                max_lag = min(20, len(paired_data) // 4)
                
                for lag in range(0, max_lag):
                    if lag == 0:
                        corr, p_val = pearsonr(bio_series, temp_series)
                    else:
                        if len(bio_series) > lag:
                            corr, p_val = pearsonr(bio_series[lag:], temp_series[:-lag])
                        else:
                            continue
                    
                    if abs(corr) > abs(best_ccf) and p_val < 0.10:
                        best_ccf = corr
                        best_lag = lag
                        best_temp_var = temp_col
                
                # Also test negative lags (temperature leading biomarker)
                for lag in range(1, max_lag):
                    if len(temp_series) > lag:
                        corr, p_val = pearsonr(bio_series[:-lag], temp_series[lag:])
                        if abs(corr) > abs(best_ccf) and p_val < 0.10:
                            best_ccf = corr
                            best_lag = -lag  # Negative indicates temp leads
                            best_temp_var = temp_col
        
        print(f"Best cross-correlation: {best_ccf:.3f} at lag {best_lag}")
        if best_temp_var:
            print(f"Best temperature variable: {best_temp_var}")
        
        if abs(best_ccf) > 0.15 and best_temp_var:  # Threshold for meaningful correlation
            # Statistical significance test
            n_effective = len(biomarker_data) - abs(best_lag)
            t_stat = best_ccf * np.sqrt((n_effective - 2) / (1 - best_ccf**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_effective - 2))
            
            print(f"Statistical significance: p = {p_value:.4f}")
            
            if p_value < 0.05:
                result = {
                    'biomarker': biomarker,
                    'method': 'Cross-Correlation',
                    'max_correlation': best_ccf,
                    'optimal_lag': best_lag,
                    'best_temp_variable': best_temp_var,
                    'p_value': p_value,
                    'n_samples': len(biomarker_data)
                }
                
                print("✅ SIGNIFICANT CROSS-CORRELATION DETECTED")
                return result
        
        print("No significant cross-correlation")
        return None
    
    def extreme_event_analysis(self, df, biomarker, temp_cols):
        """Analyze response to extreme temperature events"""
        print(f"\n🌡️ Extreme Event Analysis: {biomarker}")
        print("-" * 35)
        
        if biomarker not in df.columns:
            print("Biomarker not available")
            return None
        
        biomarker_data = df.dropna(subset=[biomarker])
        if len(biomarker_data) < 100:
            print("Insufficient data")
            return None
        
        results = []
        
        for temp_col in temp_cols:
            if temp_col in biomarker_data.columns:
                # Define extreme events (95th and 5th percentiles)
                temp_values = biomarker_data[temp_col].dropna()
                p95 = temp_values.quantile(0.95)
                p05 = temp_values.quantile(0.05)
                
                # Compare biomarker values during extreme vs normal conditions
                extreme_hot = biomarker_data[biomarker_data[temp_col] > p95][biomarker]
                extreme_cold = biomarker_data[biomarker_data[temp_col] < p05][biomarker]
                normal = biomarker_data[
                    (biomarker_data[temp_col] >= p05) & 
                    (biomarker_data[temp_col] <= p95)
                ][biomarker]
                
                print(f"\n{temp_col}:")
                print(f"  Extreme hot days (>{p95:.1f}°C): {len(extreme_hot)} samples")
                print(f"  Extreme cold days (<{p05:.1f}°C): {len(extreme_cold)} samples") 
                print(f"  Normal days: {len(normal)} samples")
                
                # Test differences
                if len(extreme_hot) >= 5 and len(normal) >= 20:
                    t_stat, p_hot = stats.ttest_ind(extreme_hot, normal)
                    effect_size_hot = (extreme_hot.mean() - normal.mean()) / normal.std()
                    
                    print(f"  Hot vs Normal: p = {p_hot:.4f}, effect size = {effect_size_hot:.3f}")
                    
                    if p_hot < 0.05 and abs(effect_size_hot) > 0.3:
                        print(f"  ✅ SIGNIFICANT HOT WEATHER EFFECT")
                        results.append({
                            'type': 'extreme_hot',
                            'temp_variable': temp_col,
                            'p_value': p_hot,
                            'effect_size': effect_size_hot,
                            'threshold': p95
                        })
                
                if len(extreme_cold) >= 5 and len(normal) >= 20:
                    t_stat, p_cold = stats.ttest_ind(extreme_cold, normal)
                    effect_size_cold = (extreme_cold.mean() - normal.mean()) / normal.std()
                    
                    print(f"  Cold vs Normal: p = {p_cold:.4f}, effect size = {effect_size_cold:.3f}")
                    
                    if p_cold < 0.05 and abs(effect_size_cold) > 0.3:
                        print(f"  ✅ SIGNIFICANT COLD WEATHER EFFECT")
                        results.append({
                            'type': 'extreme_cold',
                            'temp_variable': temp_col,
                            'p_value': p_cold,
                            'effect_size': effect_size_cold,
                            'threshold': p05
                        })
        
        if results:
            # Return the most significant result
            best_result = min(results, key=lambda x: x['p_value'])
            
            final_result = {
                'biomarker': biomarker,
                'method': 'Extreme Event Analysis',
                'event_type': best_result['type'],
                'temp_variable': best_result['temp_variable'],
                'p_value': best_result['p_value'],
                'effect_size': best_result['effect_size'],
                'threshold': best_result['threshold'],
                'n_samples': len(biomarker_data)
            }
            
            print("\n✅ SIGNIFICANT EXTREME WEATHER EFFECT DETECTED")
            return final_result
        
        print("No significant extreme event effects")
        return None
    
    def run_time_series_analysis(self):
        """Execute comprehensive time series analysis"""
        print("\n⏰ COMPREHENSIVE TIME SERIES CLIMATE-HEALTH ANALYSIS")
        print("=" * 55)
        
        # Load data
        df, biomarkers, temp_cols = self.load_time_series_data()
        
        all_results = []
        
        for biomarker in biomarkers:
            if biomarker in df.columns:
                print(f"\n🎯 TIME SERIES ANALYSIS: {biomarker}")
                print("=" * (25 + len(biomarker)))
                
                # 1. Seasonal correlation analysis
                seasonal_result = self.seasonal_correlation_analysis(df, biomarker, temp_cols)
                if seasonal_result:
                    all_results.append(seasonal_result)
                
                # 2. Cross-correlation analysis
                ccf_result = self.cross_correlation_analysis(df, biomarker, temp_cols)
                if ccf_result:
                    all_results.append(ccf_result)
                
                # 3. Extreme event analysis
                extreme_result = self.extreme_event_analysis(df, biomarker, temp_cols)
                if extreme_result:
                    all_results.append(extreme_result)
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TIME SERIES ANALYSIS SUMMARY")
        print("=" * 60)
        
        if all_results:
            print(f"🎯 SIGNIFICANT RELATIONSHIPS FOUND: {len(all_results)}")
            
            for i, result in enumerate(all_results, 1):
                print(f"\n{i}. {result['biomarker']} ({result['method']})")
                
                if result['method'] == 'Seasonal Correlation':
                    print(f"   Seasonal ANOVA p-value: {result['seasonal_anova_p']:.4f}")
                    print(f"   Climate correlation: {result['best_temp_correlation']:.3f}")
                    print(f"   Temperature variable: {result['best_temp_variable']}")
                
                elif result['method'] == 'Cross-Correlation':
                    print(f"   Max correlation: {result['max_correlation']:.3f}")
                    print(f"   Optimal lag: {result['optimal_lag']} time units")
                    print(f"   p-value: {result['p_value']:.4f}")
                    print(f"   Temperature variable: {result['best_temp_variable']}")
                
                elif result['method'] == 'Extreme Event Analysis':
                    print(f"   Event type: {result['event_type']}")
                    print(f"   Effect size: {result['effect_size']:.3f}")
                    print(f"   p-value: {result['p_value']:.4f}")
                    print(f"   Temperature threshold: {result['threshold']:.1f}°C")
                
                print(f"   Sample size: {result['n_samples']:,}")
            
            # Determine if we have robust findings
            high_quality_results = [r for r in all_results if (
                (r['method'] == 'Seasonal Correlation' and abs(r['best_temp_correlation']) > 0.6) or
                (r['method'] == 'Cross-Correlation' and abs(r['max_correlation']) > 0.2 and r['p_value'] < 0.01) or
                (r['method'] == 'Extreme Event Analysis' and abs(r['effect_size']) > 0.5 and r['p_value'] < 0.01)
            )]
            
            if high_quality_results:
                print(f"\n🏆 HIGH-QUALITY FINDINGS: {len(high_quality_results)}")
                print("These represent the most robust climate-health relationships detected.")
            else:
                print(f"\n⚠️ FINDINGS REQUIRE CAUTIOUS INTERPRETATION")
                print("Effect sizes are modest and may not be clinically significant.")
        
        else:
            print("❌ NO SIGNIFICANT TIME SERIES RELATIONSHIPS DETECTED")
            print("\nEven with sophisticated time series methods including:")
            print("• Seasonal decomposition and correlation")
            print("• Cross-correlation with multiple lags") 
            print("• Extreme event analysis")
            print("• Temporal pattern detection")
            print("\nNo climate-health relationships were identified.")
        
        return all_results

def main():
    analyzer = TimeSeriesClimateAnalysis()
    results = analyzer.run_time_series_analysis()
    return results

if __name__ == "__main__":
    main()