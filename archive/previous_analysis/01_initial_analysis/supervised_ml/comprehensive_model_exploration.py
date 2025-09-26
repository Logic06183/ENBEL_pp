#!/usr/bin/env python3
"""
Comprehensive Climate-Health Model Exploration
==============================================

Scientific exploration of advanced ML algorithms for climate-health relationships
while maintaining rigorous validation and preventing data leakage.

FOCUS: Test multiple state-of-the-art algorithms on validated biomarkers:
- Hemoglobin (R² = 0.159) ✅
- Systolic BP (R² = 0.220) ✅  
- Creatinine (R² = 0.113) ✅

EXCLUDE: Diastolic BP (R² = 1.0 - data leakage suspected)

ALGORITHMS TO EXPLORE:
1. Advanced ensemble methods
2. Neural networks
3. Bayesian approaches
4. Time-series specific models
5. Domain-adapted algorithms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, VotingRegressor, BaggingRegressor,
                             AdaBoostRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNet, Ridge, Lasso, 
                                 HuberRegressor, SGDRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (cross_val_score, GridSearchCV, 
                                    RandomizedSearchCV, TimeSeriesSplit)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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

class ComprehensiveModelExplorer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("comprehensive_results")
        self.models_dir = Path("comprehensive_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"comprehensive_progress_{self.timestamp}.log"
        
        # Focus on scientifically validated biomarkers
        self.validated_biomarkers = {
            'Hemoglobin (g/dL)': {
                'baseline_r2': 0.124,
                'optimized_r2': 0.159,
                'optimal_lag_window': 'medium_term',
                'expected_range': (0.02, 0.20)
            },
            'systolic blood pressure': {
                'baseline_r2': 0.002,
                'optimized_r2': 0.220,
                'optimal_lag_window': 'immediate',
                'expected_range': (0.05, 0.25)
            },
            'Creatinine (mg/dL)': {
                'baseline_r2': 0.129,
                'optimized_r2': 0.113,
                'optimal_lag_window': 'immediate',
                'expected_range': (0.08, 0.30)
            }
        }

    def log_progress(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🔬", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅", "MODEL": "🤖"}
        icon = icons.get(level, "🔬")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def define_advanced_model_zoo(self):
        """Define comprehensive model zoo with scientific parameter ranges"""
        self.log_progress("Defining advanced model zoo...", "MODEL")
        
        model_zoo = {}
        
        # 1. ENSEMBLE METHODS
        model_zoo['extra_trees'] = {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 8, 12],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10]
            },
            'category': 'ensemble'
        }
        
        model_zoo['hist_gradient_boosting'] = {
            'model': HistGradientBoostingRegressor(random_state=42),
            'params': {
                'max_iter': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 8],
                'min_samples_leaf': [10, 20]
            },
            'category': 'ensemble'
        }
        
        model_zoo['ada_boost'] = {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.5],
                'loss': ['linear', 'square']
            },
            'category': 'ensemble'
        }
        
        # 2. ADVANCED GRADIENT BOOSTING
        model_zoo['lightgbm_advanced'] = {
            'model': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                verbosity=-1,
                random_state=42
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'feature_fraction': [0.6, 0.8, 1.0],
                'bagging_fraction': [0.6, 0.8, 1.0],
                'reg_alpha': [0.0, 0.1, 0.5],
                'reg_lambda': [0.0, 0.1, 0.5]
            },
            'category': 'gradient_boosting'
        }
        
        if CATBOOST_AVAILABLE:
            model_zoo['catboost'] = {
                'model': cb.CatBoostRegressor(
                    random_state=42,
                    verbose=False
                ),
                'params': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128]
                },
                'category': 'gradient_boosting'
            }
        
        # 3. NEURAL NETWORKS
        model_zoo['mlp_small'] = {
            'model': MLPRegressor(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'solver': ['adam', 'lbfgs']
            },
            'category': 'neural_network'
        }
        
        model_zoo['mlp_medium'] = {
            'model': MLPRegressor(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(100, 50), (100, 50, 25), (200, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['constant', 'adaptive']
            },
            'category': 'neural_network'
        }
        
        # 4. BAYESIAN APPROACHES
        model_zoo['bayesian_ridge'] = {
            'model': BayesianRidge(),
            'params': {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            },
            'category': 'bayesian'
        }
        
        # 5. ROBUST REGRESSORS
        model_zoo['huber'] = {
            'model': HuberRegressor(),
            'params': {
                'epsilon': [1.1, 1.35, 1.5, 2.0],
                'alpha': [0.0001, 0.001, 0.01, 0.1]
            },
            'category': 'robust'
        }
        
        # 6. SUPPORT VECTOR MACHINES
        model_zoo['svr'] = {
            'model': SVR(),
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'category': 'svm'
        }
        
        # 7. NEAREST NEIGHBORS
        model_zoo['knn'] = {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'category': 'neighbors'
        }
        
        # 8. REGULARIZED LINEAR MODELS
        model_zoo['elastic_net'] = {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'category': 'linear'
        }
        
        self.log_progress(f"Model zoo defined: {len(model_zoo)} algorithms across {len(set(m['category'] for m in model_zoo.values()))} categories")
        return model_zoo

    def prepare_biomarker_data(self, biomarker_name):
        """Prepare data for specific biomarker with optimal preprocessing"""
        self.log_progress(f"Preparing data for {biomarker_name}...", "MODEL")
        
        # Load data
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        if biomarker_name not in df.columns:
            return None, None, None
        
        # Get biomarker info
        biomarker_info = self.validated_biomarkers[biomarker_name]
        optimal_lag = biomarker_info['optimal_lag_window']
        
        # Load optimal features from previous optimization
        try:
            with open('advanced_results/advanced_optimization_20250918_193208.json', 'r') as f:
                prev_results = json.load(f)
            
            if biomarker_name in prev_results['optimization_results']:
                selected_features = prev_results['optimization_results'][biomarker_name]['selected_features']
                self.log_progress(f"Using {len(selected_features)} pre-optimized features")
            else:
                # Fallback to basic climate features
                selected_features = self._get_basic_climate_features(df, optimal_lag)
        except:
            selected_features = self._get_basic_climate_features(df, optimal_lag)
        
        # Prepare dataset
        biomarker_data = df.dropna(subset=[biomarker_name]).copy()
        
        # Get available features
        available_features = [f for f in selected_features if f in biomarker_data.columns]
        
        if len(available_features) < 5:
            self.log_progress(f"Insufficient features: {len(available_features)}", "ERROR")
            return None, None, None
        
        X = biomarker_data[available_features].copy()
        y = biomarker_data[biomarker_name].copy()
        
        # Robust preprocessing
        # Handle categoricals
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].nunique() <= 10:
                X[col] = pd.Categorical(X[col]).codes
            else:
                X = X.drop(columns=[col])
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Final cleaning
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        self.log_progress(f"Data prepared: {len(X_clean)} samples, {len(X_clean.columns)} features")
        
        return X_clean, y_clean, available_features

    def _get_basic_climate_features(self, df, optimal_lag_window):
        """Get basic climate features based on optimal lag window"""
        climate_keywords = ['temp', 'humid', 'pressure', 'wind', 'solar', 'utci']
        
        # Define lag ranges
        lag_ranges = {
            'immediate': [0, 1, 2],
            'short_term': [1, 2, 3, 4, 5],
            'medium_term': [3, 4, 5, 6, 7],
            'long_term': [7, 10, 14, 21]
        }
        
        target_lags = lag_ranges.get(optimal_lag_window, [0, 1, 2])
        
        features = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in climate_keywords):
                if any(f'lag{lag}' in col.lower() for lag in target_lags):
                    features.append(col)
        
        # Add basic demographics
        basic_demos = ['year', 'month', 'latitude', 'longitude']
        for col in basic_demos:
            if col in df.columns:
                features.append(col)
        
        return features[:30]  # Limit to top 30

    def comprehensive_model_evaluation(self, X, y, biomarker_name):
        """Comprehensive evaluation of all models with rigorous validation"""
        self.log_progress(f"Comprehensive model evaluation for {biomarker_name}...", "MODEL")
        
        model_zoo = self.define_advanced_model_zoo()
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        results = {}
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        total_models = len(model_zoo)
        
        for i, (model_name, model_config) in enumerate(model_zoo.items(), 1):
            self.log_progress(f"[{i}/{total_models}] Training {model_name}...", "MODEL")
            
            try:
                model = model_config['model']
                params = model_config['params']
                category = model_config['category']
                
                # Use scaled data for models that benefit from it
                if category in ['neural_network', 'svm', 'linear']:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train
                    X_test_use = X_test
                
                # Hyperparameter optimization with time constraint
                if len(params) > 0:
                    # Use RandomizedSearchCV for efficiency
                    search = RandomizedSearchCV(
                        model, params, 
                        n_iter=20,  # Limit iterations for efficiency
                        cv=3, 
                        scoring='r2', 
                        random_state=42,
                        n_jobs=-1
                    )
                    search.fit(X_train_use, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    best_model = model
                    best_params = {}
                    best_model.fit(X_train_use, y_train)
                
                # Predict
                y_pred = best_model.predict(X_test_use)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train_use, y_train, cv=3, scoring='r2')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                results[model_name] = {
                    'model': best_model,
                    'category': category,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'best_params': best_params,
                    'n_features': X_train.shape[1],
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
                
                self.log_progress(f"  {model_name}: R² = {r2:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
                
            except Exception as e:
                self.log_progress(f"  {model_name} failed: {str(e)[:50]}", "ERROR")
                continue
        
        return results

    def create_ensemble_of_best_models(self, results, X_train, y_train, X_test, y_test):
        """Create ensemble from top performing models"""
        self.log_progress("Creating ensemble from best models...", "MODEL")
        
        # Sort models by CV performance
        sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_r2_mean'], reverse=True)
        
        # Take top 3-5 models with positive performance
        top_models = [(name, data) for name, data in sorted_models[:5] if data['cv_r2_mean'] > 0]
        
        if len(top_models) < 2:
            return None
        
        # Create voting ensemble
        estimators = [(name, data['model']) for name, data in top_models]
        
        try:
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            
            y_pred_ensemble = ensemble.predict(X_test)
            r2_ensemble = r2_score(y_test, y_pred_ensemble)
            mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
            
            ensemble_result = {
                'model': ensemble,
                'category': 'ensemble',
                'r2': r2_ensemble,
                'mae': mae_ensemble,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
                'cv_r2_mean': r2_ensemble,  # Approximate
                'cv_r2_std': 0.0,
                'best_params': {'component_models': [name for name, _ in top_models]},
                'n_features': X_train.shape[1],
                'n_train': len(X_train),
                'n_test': len(X_test)
            }
            
            self.log_progress(f"Ensemble R² = {r2_ensemble:.4f} from {len(top_models)} models")
            return ensemble_result
            
        except Exception as e:
            self.log_progress(f"Ensemble creation failed: {e}", "ERROR")
            return None

    def validate_against_literature(self, biomarker_name, results):
        """Validate results against literature expectations"""
        biomarker_info = self.validated_biomarkers[biomarker_name]
        expected_min, expected_max = biomarker_info['expected_range']
        baseline_r2 = biomarker_info['baseline_r2']
        
        validated_results = {}
        
        for model_name, model_data in results.items():
            r2 = model_data['r2']
            improvement = r2 - baseline_r2
            
            # Literature validation
            if r2 > expected_max * 1.5:  # 50% above maximum expected
                status = "suspicious_high"
            elif r2 >= expected_max:
                status = "excellent"
            elif r2 >= expected_min:
                status = "acceptable"
            elif r2 > 0:
                status = "weak_positive"
            else:
                status = "no_relationship"
            
            validated_results[model_name] = {
                **model_data,
                'literature_status': status,
                'improvement_over_baseline': improvement,
                'expected_range': (expected_min, expected_max)
            }
        
        return validated_results

    def run_comprehensive_exploration(self):
        """Execute comprehensive model exploration"""
        self.log_progress("="*80)
        self.log_progress("🤖 COMPREHENSIVE CLIMATE-HEALTH MODEL EXPLORATION")
        self.log_progress("Scientific evaluation of advanced algorithms")
        self.log_progress("="*80)
        
        start_time = time.time()
        
        all_biomarker_results = {}
        
        for biomarker_name in self.validated_biomarkers.keys():
            self.log_progress(f"\n🤖 EXPLORING MODELS FOR: {biomarker_name}")
            
            # Prepare data
            X, y, features = self.prepare_biomarker_data(biomarker_name)
            
            if X is None:
                self.log_progress(f"Data preparation failed for {biomarker_name}", "ERROR")
                continue
            
            # Comprehensive model evaluation
            model_results = self.comprehensive_model_evaluation(X, y, biomarker_name)
            
            if len(model_results) == 0:
                self.log_progress(f"No models succeeded for {biomarker_name}", "ERROR")
                continue
            
            # Create ensemble
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            ensemble_result = self.create_ensemble_of_best_models(
                model_results, X_train, y_train, X_test, y_test
            )
            
            if ensemble_result:
                model_results['intelligent_ensemble'] = ensemble_result
            
            # Literature validation
            validated_results = self.validate_against_literature(biomarker_name, model_results)
            
            all_biomarker_results[biomarker_name] = {
                'biomarker_info': self.validated_biomarkers[biomarker_name],
                'n_samples': len(X),
                'n_features': len(features),
                'model_results': validated_results
            }
            
            # Summary for this biomarker
            best_model = max(validated_results.items(), key=lambda x: x[1]['r2'])
            self.log_progress(f"✅ Best model for {biomarker_name}: {best_model[0]} (R² = {best_model[1]['r2']:.4f})", "SUCCESS")
        
        # Generate comprehensive results
        elapsed_time = time.time() - start_time
        
        final_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'methodology': 'comprehensive_model_exploration',
                'total_biomarkers': len(all_biomarker_results),
                'exploration_time_minutes': elapsed_time / 60,
                'models_tested_per_biomarker': len(self.define_advanced_model_zoo()),
                'validation_approach': 'literature_validated_with_cross_validation'
            },
            'biomarker_explorations': all_biomarker_results
        }
        
        # Save results
        results_file = self.results_dir / f"comprehensive_exploration_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # COMPREHENSIVE SUMMARY
        self.log_progress("\n" + "="*80)
        self.log_progress("🤖 COMPREHENSIVE MODEL EXPLORATION COMPLETE")
        self.log_progress("="*80)
        
        if all_biomarker_results:
            self.log_progress(f"Successfully explored {len(all_biomarker_results)} validated biomarkers")
            self.log_progress(f"Total exploration time: {elapsed_time/60:.1f} minutes")
            self.log_progress("")
            
            # Best model summary
            self.log_progress(f"{'Biomarker':<30} {'Best Model':<25} {'R²':<8} {'Status':<15} {'Improvement':<12}")
            self.log_progress("-" * 95)
            
            for biomarker, data in all_biomarker_results.items():
                model_results = data['model_results']
                best_model_name = max(model_results.items(), key=lambda x: x[1]['r2'])[0]
                best_result = model_results[best_model_name]
                
                biomarker_short = biomarker[:29]
                model_short = best_model_name[:24]
                r2 = best_result['r2']
                status = best_result['literature_status']
                improvement = best_result['improvement_over_baseline']
                
                status_icon = {
                    'excellent': '🌟',
                    'acceptable': '✅',
                    'weak_positive': '📈',
                    'no_relationship': '📉',
                    'suspicious_high': '⚠️'
                }.get(status, '❓')
                
                self.log_progress(f"{biomarker_short:<30} {model_short:<25} {r2:<8.4f} {status_icon}{status:<14} {improvement:<+12.4f}")
            
            # Algorithm category analysis
            self.log_progress("\n📊 Algorithm Category Performance:")
            category_performance = {}
            
            for biomarker, data in all_biomarker_results.items():
                for model_name, model_data in data['model_results'].items():
                    category = model_data['category']
                    if category not in category_performance:
                        category_performance[category] = []
                    category_performance[category].append(model_data['r2'])
            
            for category, r2_scores in category_performance.items():
                mean_r2 = np.mean(r2_scores)
                max_r2 = np.max(r2_scores)
                self.log_progress(f"   {category:<20}: Mean R² = {mean_r2:.4f}, Max R² = {max_r2:.4f}")
            
            self.log_progress("")
            self.log_progress(f"✅ Results saved to: {results_file}")
            
        else:
            self.log_progress("❌ No biomarkers successfully explored")
        
        return final_results

def main():
    """Execute comprehensive model exploration"""
    explorer = ComprehensiveModelExplorer()
    results = explorer.run_comprehensive_exploration()
    return results

if __name__ == "__main__":
    main()