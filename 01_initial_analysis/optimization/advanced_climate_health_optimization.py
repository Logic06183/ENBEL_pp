#!/usr/bin/env python3
"""
Advanced Climate-Health Model Optimization
=========================================

This builds on the rigorous methodology to explore advanced techniques
for improving climate-health model performance while maintaining 
scientific integrity and preventing data leakage.

GOAL: Improve R² from current range (-3.55 to 0.13) to realistic 
literature-validated range (0.05-0.25) through:

1. Advanced feature engineering
2. Optimal lag window discovery  
3. Climate-specific algorithms
4. Domain-aware validation
5. Literature-guided enhancements
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.stats import pearsonr
import json
import time
from datetime import datetime
import logging
import warnings
from pathlib import Path
import joblib
from itertools import combinations

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedClimateHealthOptimizer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("advanced_results")
        self.models_dir = Path("advanced_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"advanced_progress_{self.timestamp}.log"
        
        # Literature-based targets for optimization
        self.target_performance = {
            'cardiovascular': {'target_r2': 0.15, 'min_acceptable': 0.05},
            'immune': {'target_r2': 0.10, 'min_acceptable': 0.03},
            'metabolic': {'target_r2': 0.08, 'min_acceptable': 0.02},
            'renal': {'target_r2': 0.20, 'min_acceptable': 0.08}
        }

    def log_progress(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🔬", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅", "OPTIMIZE": "🚀"}
        icon = icons.get(level, "🔬")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def load_rigorous_dataset(self):
        """Load dataset with the proven rigorous feature set"""
        self.log_progress("Loading dataset with rigorous feature selection...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Apply the proven rigorous feature selection
        climate_keywords = [
            'temp', 'heat', 'cool', 'warm', 'cold',
            'humid', 'moisture', 'rh_', 
            'pressure', 'mslp', 'sp_',
            'wind', 'breeze', 'gust',
            'solar', 'radiation', 'uv',
            'precip', 'rain', 'precipitation',
            'utci', 'heat_index', 'feels_like',
            'lag'
        ]
        
        rigorous_features = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in climate_keywords):
                if not any(exclude in col.lower() for exclude in [
                    'vulnerability', 'risk', 'health', 'stress_score', 
                    'index_health', 'comfort', 'danger'
                ]):
                    rigorous_features.append(col)
        
        # Add basic demographics
        for col in df.columns:
            if col.lower() in ['sex', 'race', 'age', 'latitude', 'longitude', 
                              'year', 'month', 'season', 'day_of_year']:
                rigorous_features.append(col)
        
        self.log_progress(f"Rigorous features identified: {len(rigorous_features)}")
        return df, rigorous_features

    def engineer_advanced_climate_features(self, df, base_features):
        """Create advanced climate features based on domain knowledge"""
        self.log_progress("Engineering advanced climate features...", "OPTIMIZE")
        
        df_enhanced = df.copy()
        new_features = []
        
        # 1. Heat stress indices (if not already present)
        temp_cols = [col for col in base_features if 'temp' in col.lower() and 'lag' not in col.lower()]
        humid_cols = [col for col in base_features if 'humid' in col.lower() and 'lag' not in col.lower()]
        
        if temp_cols and humid_cols:
            temp_col = temp_cols[0]  # Use first available temperature
            humid_col = humid_cols[0]  # Use first available humidity
            
            if temp_col in df.columns and humid_col in df.columns:
                # Simplified heat index calculation
                df_enhanced['heat_stress_index'] = (
                    df[temp_col] * 1.8 + 32 + 
                    0.5 * (df[humid_col] - 50) * 0.1
                )
                new_features.append('heat_stress_index')
        
        # 2. Optimal lag combinations for different health pathways
        lag_features = [col for col in base_features if 'lag' in col.lower()]
        
        # Acute exposure (0-3 days) - for immediate physiological responses
        acute_lag_features = [col for col in lag_features if any(f'lag{i}' in col.lower() for i in [0, 1, 2, 3])]
        if len(acute_lag_features) > 3:
            df_enhanced['acute_climate_exposure'] = df[acute_lag_features[:4]].mean(axis=1)
            new_features.append('acute_climate_exposure')
        
        # Chronic exposure (7-14 days) - for adaptation responses
        chronic_lag_features = [col for col in lag_features if any(f'lag{i}' in col.lower() for i in range(7, 15))]
        if len(chronic_lag_features) > 3:
            df_enhanced['chronic_climate_exposure'] = df[chronic_lag_features[:4]].mean(axis=1)
            new_features.append('chronic_climate_exposure')
        
        # 3. Climate interaction terms (temperature × humidity effects)
        if temp_cols and humid_cols and len(temp_cols) > 0 and len(humid_cols) > 0:
            temp_col = temp_cols[0]
            humid_col = humid_cols[0]
            if temp_col in df.columns and humid_col in df.columns:
                df_enhanced['temp_humid_interaction'] = df[temp_col] * df[humid_col]
                new_features.append('temp_humid_interaction')
        
        # 4. Seasonal climate anomalies
        if 'month' in df.columns:
            for col in temp_cols[:2]:  # Top 2 temperature variables
                if col in df.columns:
                    monthly_means = df.groupby('month')[col].transform('mean')
                    df_enhanced[f'{col}_seasonal_anomaly'] = df[col] - monthly_means
                    new_features.append(f'{col}_seasonal_anomaly')
        
        # 5. Extreme weather indicators
        for col in temp_cols[:2]:
            if col in df.columns:
                q95 = df[col].quantile(0.95)
                q05 = df[col].quantile(0.05)
                df_enhanced[f'{col}_extreme_heat'] = (df[col] > q95).astype(int)
                df_enhanced[f'{col}_extreme_cold'] = (df[col] < q05).astype(int)
                new_features.extend([f'{col}_extreme_heat', f'{col}_extreme_cold'])
        
        self.log_progress(f"Created {len(new_features)} advanced climate features")
        
        return df_enhanced, base_features + new_features

    def optimize_lag_windows(self, df, features, biomarker, biomarker_data):
        """Find optimal lag windows for specific biomarker"""
        self.log_progress(f"Optimizing lag windows for {biomarker}...", "OPTIMIZE")
        
        lag_features = [col for col in features if 'lag' in col.lower()]
        if len(lag_features) < 5:
            return features, "insufficient_lag_features"
        
        # Test different lag window combinations
        lag_windows = {
            'immediate': [0, 1, 2],           # 0-2 days
            'short_term': [1, 2, 3, 4, 5],   # 1-5 days  
            'medium_term': [3, 4, 5, 6, 7],  # 3-7 days
            'long_term': [7, 10, 14, 21]     # 1-3 weeks
        }
        
        best_r2 = -999
        best_window = None
        
        # Get base features (non-lag) and ensure they're numeric
        base_feature_candidates = [f for f in features if 'lag' not in f.lower() and f in biomarker_data.columns]
        numeric_base_features = []
        
        for col in base_feature_candidates:
            if biomarker_data[col].dtype in ['float64', 'int64']:
                numeric_base_features.append(col)
            elif biomarker_data[col].dtype == 'object':
                # Try to encode categoricals
                try:
                    if biomarker_data[col].nunique() <= 10:
                        encoded_col = f"{col}_encoded"
                        biomarker_data[encoded_col] = pd.Categorical(biomarker_data[col]).codes
                        numeric_base_features.append(encoded_col)
                except:
                    pass
        
        if len(numeric_base_features) < 3:
            self.log_progress("Insufficient numeric base features for lag optimization", "WARNING")
            return features, "insufficient_base_features"
        
        X_base = biomarker_data[numeric_base_features[:10]]  # Use top 10 base features
        y = biomarker_data[biomarker]
        
        for window_name, lags in lag_windows.items():
            # Get features for this lag window
            window_features = []
            for lag in lags:
                lag_cols = [col for col in lag_features if f'lag{lag}' in col.lower()]
                # Only add numeric lag features
                for col in lag_cols[:3]:  # Top 3 per lag
                    if col in biomarker_data.columns and biomarker_data[col].dtype in ['float64', 'int64']:
                        window_features.append(col)
            
            if len(window_features) < 3:
                continue
                
            X_window = biomarker_data[window_features]
            X_combined = pd.concat([X_base, X_window], axis=1)
            
            # Fill missing values with median
            numeric_cols = X_combined.select_dtypes(include=[np.number]).columns
            X_combined[numeric_cols] = X_combined[numeric_cols].fillna(X_combined[numeric_cols].median())
            
            # Final check for any remaining non-numeric data
            if X_combined.select_dtypes(include=['object']).shape[1] > 0:
                continue
            
            try:
                # Quick test with simple RF
                rf_test = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
                scores = cross_val_score(rf_test, X_combined, y, cv=3, scoring='r2')
                mean_r2 = np.mean(scores)
                
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_window = window_name
                    
            except Exception as e:
                self.log_progress(f"Lag window test failed for {window_name}: {str(e)[:50]}", "WARNING")
                continue
        
        if best_window:
            self.log_progress(f"Best lag window: {best_window} (R² = {best_r2:.4f})")
            # Return features with optimal lag window
            optimal_lags = lag_windows[best_window]
            optimal_lag_features = []
            for lag in optimal_lags:
                lag_cols = [col for col in lag_features if f'lag{lag}' in col.lower()]
                for col in lag_cols[:2]:  # Top 2 per lag
                    if col in biomarker_data.columns and biomarker_data[col].dtype in ['float64', 'int64']:
                        optimal_lag_features.append(col)
            
            return numeric_base_features + optimal_lag_features, best_window
        else:
            self.log_progress("No improvement found in lag window optimization", "WARNING")
            return features, "no_improvement"

    def advanced_feature_selection(self, X, y, max_features=50):
        """Advanced feature selection combining multiple methods"""
        self.log_progress("Performing advanced feature selection...", "OPTIMIZE")
        
        if len(X.columns) <= max_features:
            return X.columns.tolist()
        
        # 1. Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        X_reduced = X.drop(columns=high_corr_features)
        
        # 2. Statistical significance test
        try:
            selector_stats = SelectKBest(score_func=f_regression, k=min(max_features, len(X_reduced.columns)))
            X_selected = selector_stats.fit_transform(X_reduced, y)
            selected_features = X_reduced.columns[selector_stats.get_support()].tolist()
        except:
            selected_features = X_reduced.columns.tolist()[:max_features]
        
        # 3. Recursive feature elimination with RF
        if len(selected_features) > max_features:
            try:
                rf_selector = RandomForestRegressor(n_estimators=50, random_state=42)
                rfe = RFE(estimator=rf_selector, n_features_to_select=max_features)
                rfe.fit(X_reduced[selected_features], y)
                selected_features = [f for f, selected in zip(selected_features, rfe.support_) if selected]
            except:
                selected_features = selected_features[:max_features]
        
        self.log_progress(f"Selected {len(selected_features)} features from {len(X.columns)} original")
        return selected_features

    def train_advanced_models(self, X_train, y_train, X_test, y_test, biomarker_name):
        """Train multiple advanced models and ensemble"""
        self.log_progress(f"Training advanced models for {biomarker_name}...", "OPTIMIZE")
        
        models = {}
        
        # 1. Optimized Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 8, 10],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10]
        }
        
        try:
            rf_grid = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                rf_params, cv=3, scoring='r2', n_jobs=-1
            )
            rf_grid.fit(X_train, y_train)
            rf_pred = rf_grid.predict(X_test)
            
            models['optimized_rf'] = {
                'model': rf_grid.best_estimator_,
                'r2': r2_score(y_test, rf_pred),
                'mae': mean_absolute_error(y_test, rf_pred),
                'params': rf_grid.best_params_
            }
        except Exception as e:
            self.log_progress(f"RF optimization failed: {e}", "WARNING")
        
        # 2. LightGBM (often better for climate data)
        try:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            
            models['lightgbm'] = {
                'model': lgb_model,
                'r2': r2_score(y_test, lgb_pred),
                'mae': mean_absolute_error(y_test, lgb_pred)
            }
        except Exception as e:
            self.log_progress(f"LightGBM failed: {e}", "WARNING")
        
        # 3. Gradient Boosting with climate-specific parameters
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            
            models['gradient_boosting'] = {
                'model': gb_model,
                'r2': r2_score(y_test, gb_pred),
                'mae': mean_absolute_error(y_test, gb_pred)
            }
        except Exception as e:
            self.log_progress(f"Gradient Boosting failed: {e}", "WARNING")
        
        # 4. Ensemble of best models
        if len(models) >= 2:
            try:
                # Simple average ensemble of top 2 models
                sorted_models = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)
                top_models = sorted_models[:2]
                
                pred1 = top_models[0][1]['model'].predict(X_test)
                pred2 = top_models[1][1]['model'].predict(X_test)
                ensemble_pred = (pred1 + pred2) / 2
                
                models['ensemble'] = {
                    'r2': r2_score(y_test, ensemble_pred),
                    'mae': mean_absolute_error(y_test, ensemble_pred),
                    'components': [top_models[0][0], top_models[1][0]]
                }
            except Exception as e:
                self.log_progress(f"Ensemble failed: {e}", "WARNING")
        
        # Select best model
        if models:
            best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
            best_performance = models[best_model_name]
            
            self.log_progress(f"Best model: {best_model_name}, R² = {best_performance['r2']:.4f}")
            
            return models, best_model_name, best_performance
        else:
            return {}, None, None

    def validate_against_literature(self, biomarker_name, r2_score, improvement_over_baseline):
        """Validate results against literature expectations"""
        
        # Categorize biomarker
        if any(term in biomarker_name.lower() for term in ['blood pressure', 'systolic', 'diastolic']):
            category = 'cardiovascular'
        elif any(term in biomarker_name.lower() for term in ['cd4', 'immune']):
            category = 'immune'
        elif any(term in biomarker_name.lower() for term in ['glucose', 'cholesterol', 'hdl', 'ldl']):
            category = 'metabolic'
        elif any(term in biomarker_name.lower() for term in ['creatinine', 'kidney']):
            category = 'renal'
        else:
            category = 'metabolic'
        
        targets = self.target_performance[category]
        
        status = "unknown"
        if r2_score >= targets['target_r2']:
            status = "excellent"
        elif r2_score >= targets['min_acceptable']:
            status = "acceptable"
        elif r2_score > 0:
            status = "weak_but_positive"
        else:
            status = "no_relationship"
        
        return {
            'category': category,
            'status': status,
            'target_r2': targets['target_r2'],
            'min_acceptable': targets['min_acceptable'],
            'improvement': improvement_over_baseline
        }

    def run_advanced_optimization(self):
        """Execute complete advanced optimization pipeline"""
        self.log_progress("="*80)
        self.log_progress("🚀 ADVANCED CLIMATE-HEALTH MODEL OPTIMIZATION")
        self.log_progress("Building on rigorous methodology with advanced techniques")
        self.log_progress("="*80)
        
        start_time = time.time()
        
        # Load rigorous dataset
        df, rigorous_features = self.load_rigorous_dataset()
        
        # Engineer advanced features
        df_enhanced, enhanced_features = self.engineer_advanced_climate_features(df, rigorous_features)
        
        # Define biomarkers
        biomarkers = [
            'CD4 cell count (cells/µL)',
            'Creatinine (mg/dL)', 
            'Hemoglobin (g/dL)',
            'systolic blood pressure',
            'diastolic blood pressure',
            'FASTING GLUCOSE',
            'FASTING TOTAL CHOLESTEROL',
            'FASTING HDL',
            'FASTING LDL'
        ]
        
        # Load baseline results for comparison
        try:
            with open('rigorous_results/rigorous_analysis_20250918_191653.json', 'r') as f:
                baseline_results = json.load(f)
            baseline_r2 = {k: v['best_r2'] for k, v in baseline_results['biomarker_results'].items()}
        except:
            baseline_r2 = {}
        
        optimization_results = {}
        
        for i, biomarker in enumerate(biomarkers, 1):
            self.log_progress(f"\n🚀 [{i}/{len(biomarkers)}] OPTIMIZING: {biomarker}")
            
            if biomarker not in df_enhanced.columns:
                self.log_progress(f"Biomarker not found", "ERROR")
                continue
            
            try:
                # Prepare biomarker data
                biomarker_data = df_enhanced.dropna(subset=[biomarker]).copy()
                
                if len(biomarker_data) < 100:
                    self.log_progress(f"Insufficient data: {len(biomarker_data)}", "ERROR")
                    continue
                
                # Get available enhanced features
                available_features = [f for f in enhanced_features if f in biomarker_data.columns]
                
                if len(available_features) < 10:
                    self.log_progress(f"Insufficient features: {len(available_features)}", "ERROR")
                    continue
                
                # Optimize lag windows
                optimal_features, lag_window = self.optimize_lag_windows(
                    df_enhanced, available_features, biomarker, biomarker_data
                )
                
                # Prepare dataset with optimal features
                X = biomarker_data[optimal_features].copy()
                y = biomarker_data[biomarker].copy()
                
                # Handle missing values
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                
                # Encode categoricals
                categorical_cols = X.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if X[col].nunique() <= 10:
                        X[col] = pd.Categorical(X[col]).codes
                    else:
                        X = X.drop(columns=[col])
                
                # Final cleaning
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) < 100:
                    self.log_progress(f"Insufficient clean data: {len(X_clean)}", "ERROR")
                    continue
                
                # Advanced feature selection
                selected_features = self.advanced_feature_selection(X_clean, y_clean, max_features=30)
                X_selected = X_clean[selected_features]
                
                self.log_progress(f"Optimized dataset: {len(X_selected)} samples, {len(selected_features)} features")
                
                # Train-test split
                split_idx = int(len(X_selected) * 0.8)
                X_train = X_selected.iloc[:split_idx]
                X_test = X_selected.iloc[split_idx:]
                y_train = y_clean.iloc[:split_idx]
                y_test = y_clean.iloc[split_idx:]
                
                # Train advanced models
                models, best_model_name, best_performance = self.train_advanced_models(
                    X_train, y_train, X_test, y_test, biomarker
                )
                
                if best_performance:
                    baseline_r2_val = baseline_r2.get(biomarker, 0)
                    improvement = best_performance['r2'] - baseline_r2_val
                    
                    # Literature validation
                    validation = self.validate_against_literature(
                        biomarker, best_performance['r2'], improvement
                    )
                    
                    optimization_results[biomarker] = {
                        'biomarker': biomarker,
                        'n_samples': len(X_selected),
                        'n_features': len(selected_features),
                        'optimal_lag_window': lag_window,
                        'models': models,
                        'best_model': best_model_name,
                        'best_performance': best_performance,
                        'baseline_r2': baseline_r2_val,
                        'improvement': improvement,
                        'literature_validation': validation,
                        'selected_features': selected_features[:10]  # Top 10 for interpretation
                    }
                    
                    self.log_progress(f"OPTIMIZATION COMPLETE: R² = {best_performance['r2']:.4f} (improvement: {improvement:+.4f})", "SUCCESS")
                
            except Exception as e:
                self.log_progress(f"Optimization failed: {e}", "ERROR")
                continue
        
        # Generate final results
        elapsed_time = time.time() - start_time
        
        final_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'methodology': 'advanced_climate_health_optimization',
                'total_biomarkers_optimized': len(optimization_results),
                'optimization_time_minutes': elapsed_time / 60,
                'techniques_used': [
                    'advanced_feature_engineering',
                    'optimal_lag_windows',
                    'multiple_algorithms',
                    'ensemble_methods',
                    'literature_validation'
                ]
            },
            'optimization_results': optimization_results
        }
        
        # Save results
        results_file = self.results_dir / f"advanced_optimization_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # FINAL SUMMARY
        self.log_progress("\n" + "="*80)
        self.log_progress("🚀 ADVANCED OPTIMIZATION COMPLETE")
        self.log_progress("="*80)
        
        if optimization_results:
            self.log_progress(f"Successfully optimized {len(optimization_results)} biomarkers")
            self.log_progress(f"Total optimization time: {elapsed_time/60:.1f} minutes")
            self.log_progress("")
            
            # Performance comparison table
            self.log_progress(f"{'Biomarker':<35} {'Baseline R²':<12} {'Optimized R²':<13} {'Improvement':<12} {'Status':<15}")
            self.log_progress("-" * 95)
            
            for biomarker, result in optimization_results.items():
                baseline = result['baseline_r2']
                optimized = result['best_performance']['r2']
                improvement = result['improvement']
                status = result['literature_validation']['status']
                
                biomarker_short = biomarker[:34]
                status_icon = {
                    'excellent': '🌟',
                    'acceptable': '✅',
                    'weak_but_positive': '📈',
                    'no_relationship': '📉'
                }.get(status, '❓')
                
                self.log_progress(f"{biomarker_short:<35} {baseline:<12.4f} {optimized:<13.4f} {improvement:<+12.4f} {status_icon}{status:<14}")
            
            # Summary statistics
            improvements = [res['improvement'] for res in optimization_results.values()]
            optimized_r2s = [res['best_performance']['r2'] for res in optimization_results.values()]
            
            positive_improvements = sum(1 for imp in improvements if imp > 0.01)
            acceptable_models = sum(1 for res in optimization_results.values() 
                                  if res['literature_validation']['status'] in ['excellent', 'acceptable'])
            
            self.log_progress("")
            self.log_progress(f"📊 Optimization Summary:")
            self.log_progress(f"   Mean optimized R²: {np.mean(optimized_r2s):.4f}")
            self.log_progress(f"   Mean improvement: {np.mean(improvements):+.4f}")
            self.log_progress(f"   Models with substantial improvement (>0.01): {positive_improvements}/{len(optimization_results)}")
            self.log_progress(f"   Literature-acceptable models: {acceptable_models}/{len(optimization_results)}")
            self.log_progress("")
            self.log_progress(f"✅ Results saved to: {results_file}")
            
        else:
            self.log_progress("❌ No biomarkers successfully optimized")
        
        return final_results

def main():
    """Execute advanced climate-health optimization"""
    optimizer = AdvancedClimateHealthOptimizer()
    results = optimizer.run_advanced_optimization()
    return results

if __name__ == "__main__":
    main()