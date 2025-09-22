#!/usr/bin/env python3
"""
Rapid Rigorous Climate-Health Discovery
=======================================

Efficient but scientifically rigorous approach to identify 
significant climate-health relationships quickly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import json
import time
from datetime import datetime

class RapidRigorousDiscovery:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.alpha = 0.01
        self.min_effect_size = 0.02
        
    def rapid_climate_health_discovery(self):
        """Rapid but rigorous discovery of climate-health relationships"""
        print("🔬 RAPID RIGOROUS CLIMATE-HEALTH DISCOVERY")
        print("="*50)
        
        start_time = time.time()
        
        # Load data
        print("📊 Loading data...")
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Focus on largest biomarker datasets
        biomarkers = {
            'systolic blood pressure': 4957,
            'diastolic blood pressure': 4957,
            'FASTING TOTAL CHOLESTEROL': 2497,
            'FASTING HDL': 2497,
            'FASTING LDL': 2500
        }
        
        # Get top temperature features (most likely to show effects)
        temp_features = [col for col in df.columns if 'temp' in col.lower() and 'lag' in col.lower()]
        
        # Priority: immediate lags (0-3 days)
        priority_features = []
        for lag in [0, 1, 2, 3]:
            lag_features = [f for f in temp_features if f'lag{lag}' in f]
            priority_features.extend(lag_features[:3])  # Top 3 per lag
        
        climate_features = [f for f in priority_features if f in df.columns][:15]  # Top 15
        
        print(f"🎯 Testing {len(biomarkers)} biomarkers with {len(climate_features)} climate features")
        
        # Rapid testing
        significant_results = {}
        all_p_values = []
        all_keys = []
        
        for biomarker, expected_n in biomarkers.items():
            print(f"\n🔍 Testing {biomarker}...")
            
            # Prepare data
            biomarker_data = df.dropna(subset=[biomarker])
            print(f"   Sample size: {len(biomarker_data):,}")
            
            if len(biomarker_data) < 500:
                continue
            
            # Get clean climate data
            available_climate = [f for f in climate_features if f in biomarker_data.columns]
            X = biomarker_data[available_climate].fillna(biomarker_data[available_climate].median())
            y = biomarker_data[biomarker]
            
            # Feature selection - top 10 most correlated
            selector = SelectKBest(score_func=f_regression, k=min(10, len(available_climate)))
            X_selected = selector.fit_transform(X, y)
            selected_features = [available_climate[i] for i in selector.get_support(indices=True)]
            
            # Test with conservative ElasticNet
            model = ElasticNet(alpha=1.0, random_state=42, max_iter=1000)
            
            # 5-fold CV for speed
            cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
            mean_r2 = np.mean(cv_scores)
            
            print(f"   Cross-validated R² = {mean_r2:.4f}")
            
            if mean_r2 > self.min_effect_size:
                # Quick permutation test (200 iterations for speed)
                permuted_scores = []
                for _ in range(200):
                    y_perm = np.random.permutation(y)
                    score_perm = cross_val_score(model, X_selected, y_perm, cv=3, scoring='r2')
                    permuted_scores.append(np.mean(score_perm))
                
                p_value = np.mean(np.array(permuted_scores) >= mean_r2)
                print(f"   Permutation p-value = {p_value:.4f}")
                
                if p_value < 0.05:  # Preliminary threshold
                    # Get feature importance
                    model.fit(X_selected, y)
                    if hasattr(model, 'coef_'):
                        importance = np.abs(model.coef_)
                        top_feature_idx = np.argmax(importance)
                        top_feature = selected_features[top_feature_idx]
                        
                        # Simple correlation for interpretability
                        corr, corr_p = pearsonr(X[top_feature], y)
                        
                        result = {
                            'biomarker': biomarker,
                            'n_samples': len(X),
                            'validated_r2': mean_r2,
                            'p_value_permutation': p_value,
                            'top_climate_feature': top_feature,
                            'correlation': corr,
                            'correlation_p': corr_p,
                            'selected_features': selected_features,
                            'effect_size': self._categorize_effect_size(mean_r2)
                        }
                        
                        key = f"{biomarker}_climate"
                        significant_results[key] = result
                        all_p_values.append(p_value)
                        all_keys.append(key)
                        
                        print(f"   ✅ CANDIDATE: R² = {mean_r2:.3f}, p = {p_value:.4f}")
                        print(f"   Top predictor: {top_feature} (r = {corr:.3f})")
        
        # Apply Bonferroni correction
        final_significant = {}
        if len(all_p_values) > 0:
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=self.alpha, method='bonferroni')
            
            for i, key in enumerate(all_keys):
                if rejected[i]:
                    significant_results[key]['p_value_bonferroni'] = p_corrected[i]
                    significant_results[key]['bonferroni_significant'] = True
                    final_significant[key] = significant_results[key]
        
        elapsed_time = time.time() - start_time
        
        # Final results
        print("\n" + "="*50)
        print("🎯 RAPID RIGOROUS ANALYSIS COMPLETE")
        print("="*50)
        print(f"Analysis time: {elapsed_time:.1f} seconds")
        print(f"Candidates found: {len(significant_results)}")
        print(f"Bonferroni-significant: {len(final_significant)}")
        
        if len(final_significant) > 0:
            print("\n⭐ RIGOROUSLY VALIDATED RELATIONSHIPS:")
            for i, (key, result) in enumerate(final_significant.items(), 1):
                biomarker = result['biomarker']
                r2 = result['validated_r2']
                p_val = result['p_value_bonferroni']
                top_feature = result['top_climate_feature']
                correlation = result['correlation']
                effect_size = result['effect_size']
                
                print(f"\n{i}. {biomarker}")
                print(f"   R² = {r2:.3f} ({effect_size} effect)")
                print(f"   p = {p_val:.2e} (Bonferroni corrected)")
                print(f"   n = {result['n_samples']:,} samples")
                print(f"   Top climate predictor: {top_feature}")
                print(f"   Simple correlation: r = {correlation:.3f}")
                
                # Interpretation
                if correlation > 0:
                    direction = "increases"
                else:
                    direction = "decreases"
                print(f"   Interpretation: {biomarker} {direction} with {top_feature}")
        else:
            print("\n❌ No relationships survived rigorous validation")
            print("\nPossible explanations:")
            print("   • Climate-health effects are genuinely very weak in this population")
            print("   • Environmental confounders not accounted for")
            print("   • Climate measurement precision limitations")
            print("   • Need for longer time series or different populations")
        
        # Save detailed results
        results = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_time_seconds': elapsed_time,
                'biomarkers_tested': len(biomarkers),
                'climate_features_tested': len(climate_features)
            },
            'candidates': significant_results,
            'final_significant': final_significant,
            'summary': {
                'total_candidates': len(significant_results),
                'bonferroni_significant': len(final_significant)
            }
        }
        
        filename = f'rapid_rigorous_results_{self.timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
        return results
    
    def _categorize_effect_size(self, r2):
        """Categorize effect size"""
        if r2 >= 0.26:
            return "LARGE"
        elif r2 >= 0.13:
            return "MEDIUM"
        elif r2 >= 0.02:
            return "SMALL"
        else:
            return "NEGLIGIBLE"

def main():
    """Execute rapid rigorous analysis"""
    analyzer = RapidRigorousDiscovery()
    results = analyzer.rapid_climate_health_discovery()
    return results

if __name__ == "__main__":
    main()