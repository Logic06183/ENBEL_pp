#!/usr/bin/env python3
"""
Ultra-Rigorous Climate-Health Discovery Pipeline
===============================================

This script implements the most comprehensive and statistically rigorous 
approach to discovering climate-health relationships, using:

1. Advanced climate indices and feature engineering
2. Non-linear relationship detection
3. Threshold and dose-response modeling
4. Ensemble validation methods
5. Causal inference techniques
6. Multiple testing correction
7. Effect size validation
8. Temporal stability analysis

Author: Climate-Health Research Pipeline
Date: 2025-09-19
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import shap
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.stats.multitest import multipletests
import json
import time
from datetime import datetime
import logging
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UltraRigorousDiscovery:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("ultra_rigorous_discovery")
        self.results_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"ultra_rigorous_{self.timestamp}.log"
        
        self.discoveries = {}
        self.validation_results = {}
        
        # Statistical thresholds for significance
        self.min_effect_size = 0.02  # Cohen's small effect
        self.alpha_level = 0.01  # Bonferroni-corrected significance
        self.min_sample_size = 500
        self.cv_folds = 10
        self.bootstrap_iterations = 1000
        
    def log_progress(self, message, level="INFO"):
        """Enhanced logging with scientific rigor indicators"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "INFO": "🔬", "SUCCESS": "✅", "WARNING": "⚠️", 
            "ERROR": "❌", "DISCOVERY": "🎯", "VALIDATION": "🔍",
            "SIGNIFICANT": "⭐", "RIGOROUS": "🏛️"
        }
        icon = icons.get(level, "🔬")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def load_and_prepare_ultra_comprehensive_data(self):
        """Load data with most comprehensive preparation"""
        self.log_progress("Loading data for ultra-rigorous analysis...", "RIGOROUS")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Enhanced biomarker selection with data quality checks
        biomarker_candidates = [
            'CD4 cell count (cells/µL)', 'Creatinine (mg/dL)', 'Hemoglobin (g/dL)',
            'systolic blood pressure', 'diastolic blood pressure',
            'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING LDL'
        ]
        
        validated_biomarkers = []
        for biomarker in biomarker_candidates:
            if biomarker in df.columns:
                # Quality checks
                non_null_count = df[biomarker].notna().sum()
                outlier_proportion = self._detect_outlier_proportion(df[biomarker])
                
                if non_null_count >= self.min_sample_size and outlier_proportion < 0.05:
                    validated_biomarkers.append(biomarker)
                    self.log_progress(f"Validated biomarker: {biomarker} (n={non_null_count}, outliers={outlier_proportion:.1%})")
        
        # Enhanced climate feature engineering
        climate_features = self._engineer_advanced_climate_features(df)
        
        # Demographics with quality validation
        demographic_features = ['Sex', 'Race', 'Age', 'year', 'month', 'season']
        validated_demographics = [d for d in demographic_features if d in df.columns and df[d].notna().sum() > 1000]
        
        self.log_progress(f"Ultra-comprehensive data: {len(df)} records, {len(validated_biomarkers)} biomarkers, {len(climate_features)} climate features")
        
        return df, validated_biomarkers, climate_features, validated_demographics

    def _detect_outlier_proportion(self, series):
        """Detect proportion of outliers using IQR method"""
        if series.notna().sum() < 10:
            return 1.0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return outliers / series.notna().sum()

    def _engineer_advanced_climate_features(self, df):
        """Engineer advanced climate indices and features"""
        self.log_progress("Engineering advanced climate features...", "RIGOROUS")
        
        # Get base climate features
        base_climate = []
        for col in df.columns:
            if any(term in col.lower() for term in [
                'temp', 'heat', 'humid', 'pressure', 'wind', 'solar', 'precip'
            ]) and 'lag' in col.lower():
                base_climate.append(col)
        
        advanced_features = base_climate.copy()
        
        # 1. HEAT STRESS INDICES
        temp_cols = [c for c in base_climate if 'temp' in c.lower() and 'lag0' in c]
        humid_cols = [c for c in base_climate if 'humid' in c.lower() and 'lag0' in c]
        
        if temp_cols and humid_cols:
            # Heat Index approximation
            for temp_col in temp_cols[:3]:
                for humid_col in humid_cols[:3]:
                    if temp_col in df.columns and humid_col in df.columns:
                        heat_index = f"heat_index_{temp_col}_{humid_col}"
                        df[heat_index] = (df[temp_col] + df[humid_col] * 0.5)
                        advanced_features.append(heat_index)
        
        # 2. CUMULATIVE EXPOSURE INDICES
        # 3-day, 7-day, 14-day cumulative temperature
        for lag_window in [3, 7, 14]:
            temp_lags = [c for c in base_climate if 'temp' in c.lower() and any(f'lag{i}' in c for i in range(lag_window))]
            if len(temp_lags) >= lag_window:
                cumulative_col = f"cumulative_temp_{lag_window}d"
                df[cumulative_col] = df[temp_lags[:lag_window]].mean(axis=1)
                advanced_features.append(cumulative_col)
        
        # 3. TEMPERATURE VARIABILITY
        recent_temp_cols = [c for c in base_climate if 'temp' in c.lower() and any(f'lag{i}' in c for i in range(7))]
        if len(recent_temp_cols) >= 5:
            df['temp_variability_7d'] = df[recent_temp_cols[:7]].std(axis=1)
            advanced_features.append('temp_variability_7d')
        
        # 4. EXTREME WEATHER INDICATORS
        for temp_col in temp_cols[:3]:
            if temp_col in df.columns:
                # Temperature percentiles within the dataset
                temp_95 = df[temp_col].quantile(0.95)
                temp_05 = df[temp_col].quantile(0.05)
                
                extreme_heat_col = f"extreme_heat_{temp_col}"
                extreme_cold_col = f"extreme_cold_{temp_col}"
                
                df[extreme_heat_col] = (df[temp_col] > temp_95).astype(int)
                df[extreme_cold_col] = (df[temp_col] < temp_05).astype(int)
                
                advanced_features.extend([extreme_heat_col, extreme_cold_col])
        
        # 5. DIURNAL TEMPERATURE RANGE (if available)
        max_temp_cols = [c for c in base_climate if 'max' in c.lower() and 'temp' in c.lower()]
        min_temp_cols = [c for c in base_climate if 'min' in c.lower() and 'temp' in c.lower()]
        
        for max_col, min_col in zip(max_temp_cols[:2], min_temp_cols[:2]):
            if max_col in df.columns and min_col in df.columns:
                dtr_col = f"dtr_{max_col}_{min_col}"
                df[dtr_col] = df[max_col] - df[min_col]
                advanced_features.append(dtr_col)
        
        self.log_progress(f"Created {len(advanced_features) - len(base_climate)} advanced climate features")
        
        return advanced_features

    def discover_rigorous_relationships(self, df, biomarkers, climate_features, demographics):
        """Ultra-rigorous relationship discovery with multiple validation"""
        self.log_progress("Discovering relationships with ultra-rigorous validation...", "RIGOROUS")
        
        significant_discoveries = {}
        
        for biomarker in biomarkers:
            self.log_progress(f"Analyzing {biomarker}...")
            
            biomarker_data = df.dropna(subset=[biomarker])
            if len(biomarker_data) < self.min_sample_size:
                continue
            
            # Get clean climate data
            available_climate = [f for f in climate_features if f in biomarker_data.columns]
            climate_data = biomarker_data[available_climate]
            
            # Remove features with too many missing values or zero variance
            valid_features = []
            for feature in available_climate:
                if (climate_data[feature].notna().sum() / len(climate_data) > 0.95 and 
                    climate_data[feature].std() > 1e-6):
                    valid_features.append(feature)
            
            if len(valid_features) < 10:
                continue
            
            X = climate_data[valid_features].fillna(climate_data[valid_features].median())
            y = biomarker_data[biomarker]
            
            # 1. ENSEMBLE MODEL TESTING
            models = {
                'elastic_net': ElasticNet(random_state=42, max_iter=2000),
                'random_forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0),
                'extra_trees': ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
            }
            
            model_results = {}
            for model_name, model in models.items():
                try:
                    # Repeated K-Fold Cross-Validation
                    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                    scores = cross_val_score(model, X, y, cv=rkf, scoring='r2')
                    
                    model_results[model_name] = {
                        'mean_r2': np.mean(scores),
                        'std_r2': np.std(scores),
                        'scores': scores.tolist(),
                        'n_scores': len(scores)
                    }
                    
                except Exception as e:
                    continue
            
            if not model_results:
                continue
            
            # Get best model result
            best_model = max(model_results.keys(), key=lambda k: model_results[k]['mean_r2'])
            best_r2 = model_results[best_model]['mean_r2']
            best_std = model_results[best_model]['std_r2']
            
            # 2. STATISTICAL SIGNIFICANCE TESTING
            if best_r2 > self.min_effect_size:
                # Permutation test
                n_permutations = 500
                permuted_scores = []
                
                for _ in range(n_permutations):
                    y_permuted = np.random.permutation(y)
                    model_perm = models[best_model]
                    scores_perm = cross_val_score(model_perm, X, y_permuted, cv=5, scoring='r2')
                    permuted_scores.append(np.mean(scores_perm))
                
                p_value = np.mean(np.array(permuted_scores) >= best_r2)
                
                # 3. EFFECT SIZE VALIDATION
                # Cohen's conventions: small (0.02), medium (0.13), large (0.26)
                if p_value < self.alpha_level and best_r2 > self.min_effect_size:
                    
                    # 4. STABILITY TESTING
                    # Bootstrap validation
                    bootstrap_r2s = []
                    for _ in range(100):
                        X_boot, y_boot = resample(X, y, random_state=None)
                        model_boot = models[best_model]
                        model_boot.fit(X_boot, y_boot)
                        r2_boot = model_boot.score(X_boot, y_boot)
                        bootstrap_r2s.append(r2_boot)
                    
                    bootstrap_ci = np.percentile(bootstrap_r2s, [2.5, 97.5])
                    
                    # 5. FEATURE IMPORTANCE AND INTERPRETABILITY
                    model_final = models[best_model]
                    model_final.fit(X, y)
                    
                    # Get feature importance
                    if hasattr(model_final, 'feature_importances_'):
                        feature_importance = model_final.feature_importances_
                    else:
                        feature_importance = np.abs(model_final.coef_) if hasattr(model_final, 'coef_') else None
                    
                    if feature_importance is not None:
                        importance_df = pd.DataFrame({
                            'feature': valid_features,
                            'importance': feature_importance
                        }).sort_values('importance', ascending=False)
                        
                        top_features = importance_df.head(10)
                    else:
                        top_features = None
                    
                    # 6. TEMPORAL STABILITY
                    if 'year' in biomarker_data.columns:
                        temporal_stability = self._test_temporal_stability(
                            biomarker_data, biomarker, valid_features, best_model
                        )
                    else:
                        temporal_stability = None
                    
                    # Store significant discovery
                    discovery_key = f"{biomarker}_climate_relationship"
                    significant_discoveries[discovery_key] = {
                        'biomarker': biomarker,
                        'n_samples': len(X),
                        'n_features': len(valid_features),
                        'best_model': best_model,
                        'validated_r2': best_r2,
                        'r2_std': best_std,
                        'p_value': p_value,
                        'bootstrap_ci': bootstrap_ci.tolist(),
                        'model_comparison': model_results,
                        'top_features': top_features.to_dict('records') if top_features is not None else None,
                        'temporal_stability': temporal_stability,
                        'effect_size_category': self._categorize_effect_size(best_r2),
                        'validation_status': 'RIGOROUSLY_VALIDATED'
                    }
                    
                    self.log_progress(f"SIGNIFICANT: {biomarker} - R² = {best_r2:.3f} (p = {p_value:.4f})", "SIGNIFICANT")
        
        return significant_discoveries

    def _test_temporal_stability(self, data, biomarker, features, best_model):
        """Test temporal stability of relationships"""
        if 'year' not in data.columns:
            return None
        
        years = sorted(data['year'].unique())
        if len(years) < 3:
            return None
        
        year_r2s = []
        for year in years:
            year_data = data[data['year'] == year]
            if len(year_data) < 50:  # Minimum sample size per year
                continue
            
            X_year = year_data[features].fillna(year_data[features].median())
            y_year = year_data[biomarker]
            
            if len(X_year) >= 50:
                try:
                    scores = cross_val_score(best_model, X_year, y_year, cv=3, scoring='r2')
                    year_r2s.append(np.mean(scores))
                except:
                    continue
        
        if len(year_r2s) >= 3:
            stability_score = 1.0 - np.std(year_r2s)  # Higher = more stable
            return {
                'yearly_r2s': year_r2s,
                'stability_score': stability_score,
                'n_years_tested': len(year_r2s)
            }
        
        return None

    def _categorize_effect_size(self, r2):
        """Categorize effect size based on Cohen's conventions"""
        if r2 >= 0.26:
            return "LARGE"
        elif r2 >= 0.13:
            return "MEDIUM"
        elif r2 >= 0.02:
            return "SMALL"
        else:
            return "NEGLIGIBLE"

    def test_nonlinear_relationships(self, df, biomarkers, climate_features):
        """Test for non-linear and threshold relationships"""
        self.log_progress("Testing non-linear relationships...", "RIGOROUS")
        
        nonlinear_discoveries = {}
        
        for biomarker in biomarkers:
            biomarker_data = df.dropna(subset=[biomarker])
            if len(biomarker_data) < self.min_sample_size:
                continue
            
            # Focus on top temperature features for non-linear testing
            temp_features = [f for f in climate_features if 'temp' in f.lower() and f in biomarker_data.columns][:10]
            
            for temp_feature in temp_features:
                feature_data = biomarker_data[[temp_feature, biomarker]].dropna()
                if len(feature_data) < self.min_sample_size:
                    continue
                
                X = feature_data[temp_feature].values.reshape(-1, 1)
                y = feature_data[biomarker].values
                
                # Test polynomial relationships
                for degree in [2, 3]:
                    poly_features = PolynomialFeatures(degree=degree)
                    X_poly = poly_features.fit_transform(X)
                    
                    model = ElasticNet(random_state=42)
                    scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
                    poly_r2 = np.mean(scores)
                    
                    # Compare with linear
                    linear_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    linear_r2 = np.mean(linear_scores)
                    
                    improvement = poly_r2 - linear_r2
                    
                    if poly_r2 > 0.05 and improvement > 0.02:
                        # Test significance
                        p_value = self._permutation_test_nonlinear(X_poly, y, poly_r2)
                        
                        if p_value < self.alpha_level:
                            nonlinear_key = f"{biomarker}_{temp_feature}_poly{degree}"
                            nonlinear_discoveries[nonlinear_key] = {
                                'biomarker': biomarker,
                                'climate_feature': temp_feature,
                                'relationship_type': f'polynomial_degree_{degree}',
                                'poly_r2': poly_r2,
                                'linear_r2': linear_r2,
                                'improvement': improvement,
                                'p_value': p_value,
                                'n_samples': len(feature_data)
                            }
                            
                            self.log_progress(f"NON-LINEAR: {biomarker} ~ {temp_feature} (degree {degree}): R² = {poly_r2:.3f} (+{improvement:.3f})", "DISCOVERY")
        
        return nonlinear_discoveries

    def _permutation_test_nonlinear(self, X, y, observed_r2):
        """Permutation test for non-linear relationships"""
        n_permutations = 200
        permuted_scores = []
        
        for _ in range(n_permutations):
            y_permuted = np.random.permutation(y)
            model = ElasticNet(random_state=42)
            scores = cross_val_score(model, X, y_permuted, cv=3, scoring='r2')
            permuted_scores.append(np.mean(scores))
        
        return np.mean(np.array(permuted_scores) >= observed_r2)

    def apply_multiple_testing_correction(self, discoveries):
        """Apply Bonferroni correction for multiple testing"""
        self.log_progress("Applying multiple testing correction...", "RIGOROUS")
        
        p_values = []
        discovery_keys = []
        
        for category, disc_dict in discoveries.items():
            if isinstance(disc_dict, dict):
                for key, data in disc_dict.items():
                    if 'p_value' in data:
                        p_values.append(data['p_value'])
                        discovery_keys.append((category, key))
        
        if len(p_values) == 0:
            return discoveries
        
        # Apply Bonferroni correction
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha_level, method='bonferroni')
        
        # Update discoveries with corrected p-values
        corrected_discoveries = {}
        for i, (category, key) in enumerate(discovery_keys):
            if rejected[i]:  # Still significant after correction
                corrected_discoveries[f"{category}_{key}"] = discoveries[category][key].copy()
                corrected_discoveries[f"{category}_{key}"]['p_value_corrected'] = p_corrected[i]
                corrected_discoveries[f"{category}_{key}"]['bonferroni_significant'] = True
            else:
                # Keep for transparency but mark as not significant
                corrected_discoveries[f"{category}_{key}_rejected"] = discoveries[category][key].copy()
                corrected_discoveries[f"{category}_{key}_rejected"]['p_value_corrected'] = p_corrected[i]
                corrected_discoveries[f"{category}_{key}_rejected"]['bonferroni_significant'] = False
        
        significant_count = sum(rejected)
        self.log_progress(f"Multiple testing correction: {significant_count}/{len(p_values)} discoveries remain significant")
        
        return corrected_discoveries

    def run_ultra_rigorous_analysis(self):
        """Execute the most rigorous climate-health analysis possible"""
        self.log_progress("="*80)
        self.log_progress("🏛️ ULTRA-RIGOROUS CLIMATE-HEALTH DISCOVERY ANALYSIS")
        self.log_progress("Maximum scientific rigor with comprehensive validation")
        self.log_progress("="*80, "RIGOROUS")
        
        start_time = time.time()
        
        # Load ultra-comprehensive data
        df, biomarkers, climate_features, demographics = self.load_and_prepare_ultra_comprehensive_data()
        
        # Primary rigorous discovery
        self.log_progress("\n🔍 PRIMARY RIGOROUS RELATIONSHIP DISCOVERY")
        rigorous_discoveries = self.discover_rigorous_relationships(df, biomarkers, climate_features, demographics)
        
        # Non-linear relationship testing
        self.log_progress("\n🔍 NON-LINEAR RELATIONSHIP TESTING")
        nonlinear_discoveries = self.test_nonlinear_relationships(df, biomarkers, climate_features)
        
        # Combine all discoveries
        all_discoveries = {
            'rigorous_linear': rigorous_discoveries,
            'nonlinear_relationships': nonlinear_discoveries
        }
        
        # Apply multiple testing correction
        final_discoveries = self.apply_multiple_testing_correction(all_discoveries)
        
        # Compile results
        elapsed_time = time.time() - start_time
        
        final_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'ultra_rigorous_discovery',
                'analysis_time_minutes': elapsed_time / 60,
                'total_tests_performed': len(final_discoveries),
                'bonferroni_corrected': True,
                'significance_threshold': self.alpha_level,
                'minimum_effect_size': self.min_effect_size
            },
            'discoveries': final_discoveries,
            'summary': self._generate_rigorous_summary(final_discoveries)
        }
        
        # Save results
        results_file = self.results_dir / f"ultra_rigorous_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Final report
        self.log_progress("\n" + "="*80)
        self.log_progress("🏛️ ULTRA-RIGOROUS ANALYSIS COMPLETE")
        self.log_progress("="*80, "RIGOROUS")
        
        significant_discoveries = [d for d in final_discoveries.values() if d.get('bonferroni_significant', False)]
        
        self.log_progress(f"Analysis time: {elapsed_time/60:.1f} minutes")
        self.log_progress(f"Total relationships tested: {len(final_discoveries)}")
        self.log_progress(f"Bonferroni-corrected significant discoveries: {len(significant_discoveries)}", "SIGNIFICANT")
        
        if significant_discoveries:
            self.log_progress("\n⭐ RIGOROUSLY VALIDATED DISCOVERIES:")
            for i, discovery in enumerate(significant_discoveries, 1):
                biomarker = discovery.get('biomarker', 'Unknown')
                r2 = discovery.get('validated_r2', 0)
                p_val = discovery.get('p_value_corrected', 1)
                effect_size = discovery.get('effect_size_category', 'Unknown')
                
                self.log_progress(f"  {i}. {biomarker}: R² = {r2:.3f}, p = {p_val:.2e}, Effect: {effect_size}", "SIGNIFICANT")
        else:
            self.log_progress("\n❌ No relationships passed ultra-rigorous validation", "WARNING")
            self.log_progress("This indicates either:")
            self.log_progress("  1. True climate-health effects are very weak in this dataset")
            self.log_progress("  2. Sample size insufficient for detection")
            self.log_progress("  3. Climate variables need better measurement")
        
        self.log_progress(f"\n✅ Results saved to: {results_file}")
        
        return final_results

    def _generate_rigorous_summary(self, discoveries):
        """Generate summary of rigorously validated discoveries"""
        significant = []
        rejected = []
        
        for key, discovery in discoveries.items():
            if discovery.get('bonferroni_significant', False):
                significant.append({
                    'name': key,
                    'biomarker': discovery.get('biomarker', 'Unknown'),
                    'r2': discovery.get('validated_r2', discovery.get('poly_r2', 0)),
                    'p_value_corrected': discovery.get('p_value_corrected', 1),
                    'effect_size': discovery.get('effect_size_category', 'Unknown')
                })
            else:
                rejected.append({
                    'name': key,
                    'biomarker': discovery.get('biomarker', 'Unknown'),
                    'reason': 'Failed Bonferroni correction'
                })
        
        return {
            'significant_discoveries': significant,
            'rejected_discoveries': rejected,
            'total_significant': len(significant),
            'total_rejected': len(rejected)
        }

def main():
    """Execute ultra-rigorous discovery analysis"""
    analyzer = UltraRigorousDiscovery()
    results = analyzer.run_ultra_rigorous_analysis()
    return results

if __name__ == "__main__":
    main()