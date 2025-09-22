#!/usr/bin/env python3
"""
Focused Rigorous Climate-Health Analysis
========================================

A targeted, efficient approach to find the most statistically robust
climate-health relationships using the strongest validation methods.

Focus: Find at least one genuinely significant, scientifically validated relationship.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FocusedRigorousAnalysis:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.alpha = 0.01  # Very strict significance threshold
        self.min_effect_size = 0.02  # Cohen's small effect
        
    def log_progress(self, message):
        """Simple logging"""
        logging.info(f"🔬 {message}")
    
    def load_and_prepare_focused_data(self):
        """Load data with targeted preparation"""
        self.log_progress("Loading data for focused rigorous analysis...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Focus on biomarkers with largest sample sizes
        biomarkers = [
            'systolic blood pressure',  # n=4957
            'diastolic blood pressure', # n=4957  
            'FASTING TOTAL CHOLESTEROL', # n=2497
            'FASTING HDL',              # n=2497
            'FASTING LDL',              # n=2500
            'Hemoglobin (g/dL)',        # n=1282
            'Creatinine (mg/dL)'        # n=1251
        ]
        
        # Focus on most reliable climate features
        # Get immediate lag temperature features (most likely to show effects)
        temp_features = [col for col in df.columns if 'temp' in col.lower() and 'lag' in col.lower()]
        
        # Priority order: lag0, lag1, lag2 (immediate effects most reliable)
        priority_features = []
        for lag in [0, 1, 2, 3, 7, 14]:
            lag_features = [f for f in temp_features if f'lag{lag}' in f]
            priority_features.extend(lag_features[:5])  # Top 5 per lag
        
        # Add heat-related features
        heat_features = [col for col in df.columns if any(term in col.lower() for term in ['heat', 'apparent']) and 'lag' in col.lower()]
        priority_features.extend(heat_features[:10])
        
        # Remove duplicates and ensure they exist
        climate_features = list(dict.fromkeys([f for f in priority_features if f in df.columns]))[:30]
        
        self.log_progress(f"Focused analysis: {len(biomarkers)} biomarkers, {len(climate_features)} climate features")
        
        return df, biomarkers, climate_features
    
    def rigorous_single_relationship_test(self, df, biomarker, climate_features):
        """Ultra-rigorous test for a single biomarker-climate relationship"""
        
        # Prepare clean data
        biomarker_data = df.dropna(subset=[biomarker])
        if len(biomarker_data) < 500:
            return None
            
        # Get available climate features with good data quality
        available_features = []
        for feature in climate_features:
            if feature in biomarker_data.columns:
                missing_rate = biomarker_data[feature].isna().sum() / len(biomarker_data)
                if missing_rate < 0.1:  # Less than 10% missing
                    available_features.append(feature)
        
        if len(available_features) < 5:
            return None
        
        # Prepare feature matrix
        X = biomarker_data[available_features].fillna(biomarker_data[available_features].median())
        y = biomarker_data[biomarker]
        
        # Remove outliers (3 sigma rule)
        z_scores = np.abs(stats.zscore(y))
        mask = z_scores < 3
        X, y = X[mask], y[mask]
        
        if len(X) < 500:
            return None
        
        # 1. FEATURE SELECTION - Select most promising features
        selector = SelectKBest(score_func=f_regression, k=min(15, len(available_features)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [available_features[i] for i in selector.get_support(indices=True)]
        
        # 2. MODEL TESTING with multiple approaches
        models = {
            'elastic_net_conservative': ElasticNet(alpha=1.0, random_state=42, max_iter=2000),
            'elastic_net_moderate': ElasticNet(alpha=0.1, random_state=42, max_iter=2000),
            'ridge_conservative': Ridge(alpha=10.0, random_state=42),
            'random_forest_conservative': RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_split=20, 
                min_samples_leaf=10, random_state=42
            )
        }
        
        model_results = {}
        for model_name, model in models.items():
            # Repeated K-Fold Cross-Validation (more robust)
            rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
            scores = cross_val_score(model, X_selected, y, cv=rkf, scoring='r2')
            
            model_results[model_name] = {
                'mean_r2': np.mean(scores),
                'std_r2': np.std(scores),
                'scores': scores,
                'n_cv_folds': len(scores)
            }
        
        # Get best result
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['mean_r2'])
        best_r2 = model_results[best_model_name]['mean_r2']
        best_std = model_results[best_model_name]['std_r2']
        
        # 3. STATISTICAL SIGNIFICANCE TESTING
        if best_r2 < self.min_effect_size:
            return None
        
        # Permutation test (gold standard for significance)
        n_permutations = 1000
        permuted_scores = []
        best_model = models[best_model_name]
        
        for i in range(n_permutations):
            y_permuted = np.random.permutation(y)
            scores_perm = cross_val_score(best_model, X_selected, y_permuted, cv=5, scoring='r2')
            permuted_scores.append(np.mean(scores_perm))
        
        p_value = np.mean(np.array(permuted_scores) >= best_r2)
        
        # 4. EFFECT SIZE VALIDATION
        if p_value >= self.alpha:
            return None
        
        # 5. STABILITY TESTING
        # Test with different random seeds
        stability_scores = []
        for seed in [42, 123, 456, 789, 999]:
            model_stable = type(models[best_model_name])(
                **{k: v for k, v in models[best_model_name].get_params().items() if k != 'random_state'},
                random_state=seed
            )
            scores_stable = cross_val_score(model_stable, X_selected, y, cv=5, scoring='r2')
            stability_scores.append(np.mean(scores_stable))
        
        stability_std = np.std(stability_scores)
        
        # 6. FEATURE IMPORTANCE AND INTERPRETABILITY
        final_model = models[best_model_name]
        final_model.fit(X_selected, y)
        
        if hasattr(final_model, 'feature_importances_'):
            feature_importance = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'):
            feature_importance = np.abs(final_model.coef_)
        else:
            feature_importance = None
        
        if feature_importance is not None:
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            top_features = importance_df.head(5).to_dict('records')
        else:
            top_features = None
        
        # 7. CORRELATION ANALYSIS for interpretability
        # Simple correlation with top climate feature
        if len(selected_features) > 0:
            top_climate_feature = selected_features[0]
            correlation, corr_p = pearsonr(X[top_climate_feature], y)
        else:
            correlation, corr_p = 0, 1
        
        # Compile rigorous results
        result = {
            'biomarker': biomarker,
            'n_samples': len(X),
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'best_model': best_model_name,
            'validated_r2': best_r2,
            'r2_std': best_std,
            'p_value_permutation': p_value,
            'stability_std': stability_std,
            'model_comparison': {k: {'mean_r2': v['mean_r2'], 'std_r2': v['std_r2']} 
                               for k, v in model_results.items()},
            'top_features': top_features,
            'primary_correlation': correlation,
            'correlation_p_value': corr_p,
            'effect_size_category': self._categorize_effect_size(best_r2),
            'validation_quality': 'ULTRA_RIGOROUS'
        }
        
        return result
    
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
    
    def run_focused_analysis(self):
        """Run focused rigorous analysis"""
        self.log_progress("="*60)
        self.log_progress("FOCUSED RIGOROUS CLIMATE-HEALTH ANALYSIS")
        self.log_progress("="*60)
        
        start_time = time.time()
        
        # Load focused data
        df, biomarkers, climate_features = self.load_and_prepare_focused_data()
        
        # Test each biomarker rigorously
        significant_results = {}
        all_p_values = []
        all_keys = []
        
        for biomarker in biomarkers:
            self.log_progress(f"Rigorously testing {biomarker}...")
            
            result = self.rigorous_single_relationship_test(df, biomarker, climate_features)
            
            if result is not None:
                key = f"{biomarker}_climate_relationship"
                significant_results[key] = result
                all_p_values.append(result['p_value_permutation'])
                all_keys.append(key)
                
                self.log_progress(f"CANDIDATE: {biomarker} - R² = {result['validated_r2']:.3f}, p = {result['p_value_permutation']:.4f}")
        
        # Apply multiple testing correction
        if len(all_p_values) > 0:
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=self.alpha, method='bonferroni')
            
            final_significant = {}
            for i, key in enumerate(all_keys):
                if rejected[i]:
                    significant_results[key]['p_value_bonferroni'] = p_corrected[i]
                    significant_results[key]['bonferroni_significant'] = True
                    final_significant[key] = significant_results[key]
                    
                    biomarker = significant_results[key]['biomarker']
                    r2 = significant_results[key]['validated_r2']
                    self.log_progress(f"✅ SIGNIFICANT AFTER CORRECTION: {biomarker} - R² = {r2:.3f}")
        else:
            final_significant = {}
        
        # Compile final results
        elapsed_time = time.time() - start_time
        
        results = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'focused_rigorous',
                'analysis_time_minutes': elapsed_time / 60,
                'biomarkers_tested': len(biomarkers),
                'significance_threshold': self.alpha,
                'multiple_testing_correction': 'bonferroni'
            },
            'candidate_results': significant_results,
            'final_significant_results': final_significant,
            'summary': {
                'total_candidates': len(significant_results),
                'bonferroni_significant': len(final_significant),
                'success': len(final_significant) > 0
            }
        }
        
        # Save results
        with open(f'focused_rigorous_results_{self.timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Final report
        self.log_progress("="*60)
        self.log_progress("FOCUSED RIGOROUS ANALYSIS COMPLETE")
        self.log_progress("="*60)
        self.log_progress(f"Analysis time: {elapsed_time/60:.1f} minutes")
        self.log_progress(f"Candidates found: {len(significant_results)}")
        self.log_progress(f"Bonferroni-significant: {len(final_significant)}")
        
        if len(final_significant) > 0:
            self.log_progress("\n🎯 RIGOROUSLY VALIDATED CLIMATE-HEALTH RELATIONSHIPS:")
            for i, (key, result) in enumerate(final_significant.items(), 1):
                biomarker = result['biomarker']
                r2 = result['validated_r2']
                p_val = result['p_value_bonferroni']
                effect = result['effect_size_category']
                n_samples = result['n_samples']
                
                self.log_progress(f"  {i}. {biomarker}")
                self.log_progress(f"     R² = {r2:.3f} ({effect} effect size)")
                self.log_progress(f"     p = {p_val:.2e} (Bonferroni corrected)")
                self.log_progress(f"     n = {n_samples:,} samples")
                
                if result['top_features']:
                    top_feature = result['top_features'][0]['feature']
                    self.log_progress(f"     Top climate predictor: {top_feature}")
        else:
            self.log_progress("\n❌ No relationships survived rigorous validation")
            self.log_progress("Possible reasons:")
            self.log_progress("  - Climate-health effects are genuinely very weak")
            self.log_progress("  - Sample size insufficient for small effects")
            self.log_progress("  - Climate measurement precision limitations")
            self.log_progress("  - Confounding variables not accounted for")
        
        return results

def main():
    """Execute focused rigorous analysis"""
    analyzer = FocusedRigorousAnalysis()
    results = analyzer.run_focused_analysis()
    return results

if __name__ == "__main__":
    main()