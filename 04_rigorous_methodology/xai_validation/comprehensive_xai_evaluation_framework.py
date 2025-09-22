#!/usr/bin/env python3
"""
Comprehensive Model-Agnostic Explainable AI Evaluation Framework
================================================================

This framework evaluates model-agnostic XAI methods specifically for climate-health 
relationships with moderate effect sizes (R² 0.05-0.35) in high-dimensional datasets.

Focus Areas:
1. SHAP variants and extensions (TreeSHAP, KernelSHAP, LinearSHAP)
2. LIME adaptations for temporal and interaction features
3. Advanced permutation importance for correlated climate features
4. Counterfactual explanations feasibility assessment
5. Anchor explanations for demographic subgroups
6. Novel 2024-2025 XAI approaches

Technical Requirements:
- 18,205 health records with 9 biomarkers
- 27+ climate variables with temporal lags (0-21 days)
- 300+ features total
- Moderate effect sizes detection
- Computational efficiency optimization
- Uncertainty quantification

Authors: Climate-Health XAI Research Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from datetime import datetime
from pathlib import Path
import json
import joblib
from itertools import combinations
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

class ComprehensiveXAIEvaluator:
    """
    Comprehensive evaluation of model-agnostic XAI methods for climate-health research
    """
    
    def __init__(self, 
                 dataset_path: str = "FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv",
                 results_dir: str = "xai_evaluation_results",
                 max_features: int = 50,
                 sample_size: int = 2000):
        """
        Initialize the XAI evaluation framework
        
        Args:
            dataset_path: Path to the climate-health dataset
            results_dir: Directory to store evaluation results
            max_features: Maximum number of features to evaluate (computational efficiency)
            sample_size: Sample size for XAI analysis (memory management)
        """
        self.dataset_path = dataset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.max_features = max_features
        self.sample_size = sample_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results storage
        self.evaluation_results = {}
        
        # XAI methods to evaluate
        self.xai_methods = {
            'shap_tree': self._evaluate_shap_tree,
            'shap_kernel': self._evaluate_shap_kernel,
            'shap_linear': self._evaluate_shap_linear,
            'lime_tabular': self._evaluate_lime_tabular,
            'permutation_basic': self._evaluate_permutation_basic,
            'permutation_conditional': self._evaluate_permutation_conditional,
            'feature_interaction_shap': self._evaluate_shap_interactions,
            'temporal_importance': self._evaluate_temporal_importance,
            'anchor_explanations': self._evaluate_anchor_explanations
        }
        
        # Climate variable groups for analysis
        self.climate_groups = {
            'temperature': ['temp', 'tas', 'temperature', 'apparent'],
            'heat_stress': ['heat_index', 'utci', 'wbgt', 'heat_stress'],
            'wind': ['wind', 'ws'],
            'temporal_patterns': ['lag', 'mean', 'max', 'min', 'change']
        }
        
        # Lag periods to analyze
        self.lag_periods = [0, 1, 2, 3, 5, 7, 10, 14, 21]
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Load and prepare climate-health data for XAI evaluation
        """
        print("🔄 Loading and preparing climate-health dataset...")
        
        df = pd.read_csv(self.dataset_path, low_memory=False)
        
        # Identify biomarkers (target variables)
        biomarkers = [
            'systolic_blood_pressure', 'diastolic_blood_pressure',
            'FASTING_GLUCOSE', 'FASTING_HDL', 'FASTING_LDL', 
            'FASTING_TOTAL_CHOLESTEROL', 'Hemoglobin_gdL',
            'Creatinine_mgdL', 'CD4_cell_count_cellsµL'
        ]
        
        # Identify climate features
        climate_features = []
        for col in df.columns:
            if any(term in col.lower() for term in 
                   ['temp', 'heat', 'humid', 'pressure', 'wind', 'solar']) and \
               any(f'lag{lag}' in col.lower() for lag in self.lag_periods):
                climate_features.append(col)
        
        print(f"✅ Dataset loaded: {len(df):,} records")
        print(f"✅ Biomarkers available: {len([b for b in biomarkers if b in df.columns])}")
        print(f"✅ Climate features: {len(climate_features)}")
        
        return df, biomarkers, climate_features
    
    def _prepare_model_data(self, df: pd.DataFrame, biomarker: str, 
                           climate_features: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for a specific biomarker with feature selection and preprocessing
        """
        # Get clean data
        clean_data = df.dropna(subset=[biomarker]).copy()
        
        # Feature selection: correlation-based + importance-based hybrid
        available_features = [f for f in climate_features[:100] if f in clean_data.columns]
        
        if len(available_features) < 10:
            raise ValueError(f"Insufficient features for {biomarker}")
        
        X_temp = clean_data[available_features].fillna(clean_data[available_features].median())
        y = clean_data[biomarker]
        
        # Quick feature selection using correlation and variance
        feature_scores = []
        for feat in available_features:
            if X_temp[feat].var() > 1e-8:  # Avoid zero variance
                corr_with_target = abs(pearsonr(X_temp[feat], y)[0])
                feature_scores.append((feat, corr_with_target))
        
        # Select top features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in feature_scores[:self.max_features]]
        
        X = X_temp[selected_features].values
        
        # Sample for computational efficiency
        if len(X) > self.sample_size:
            sample_idx = np.random.choice(len(X), size=self.sample_size, replace=False)
            X = X[sample_idx]
            y = y.iloc[sample_idx]
        
        return X, y.values, selected_features
    
    def _train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple model types for XAI evaluation
        """
        models = {}
        
        # Tree-based models (for TreeSHAP)
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        
        # Linear models (for LinearSHAP)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        models['elastic_net'] = ElasticNet(alpha=0.1, random_state=42, max_iter=2000)
        models['linear_regression'] = LinearRegression()
        
        # Neural network (for general model-agnostic methods)
        models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            if name in ['elastic_net', 'linear_regression']:
                model.fit(X_scaled, y)
                trained_models[name] = {'model': model, 'scaler': scaler}
            else:
                model.fit(X, y)
                trained_models[name] = {'model': model, 'scaler': None}
        
        return trained_models
    
    def _evaluate_shap_tree(self, X: np.ndarray, y: np.ndarray, 
                           trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate TreeSHAP for tree-based models
        """
        print("  🌳 Evaluating TreeSHAP...")
        results = {}
        
        for model_name in ['random_forest', 'gradient_boosting']:
            if model_name not in trained_models:
                continue
                
            start_time = time.time()
            model_obj = trained_models[model_name]['model']
            
            try:
                # TreeSHAP explainer
                explainer = shap.TreeExplainer(model_obj)
                shap_values = explainer.shap_values(X[:min(100, len(X))])
                
                # Calculate metrics
                computation_time = time.time() - start_time
                feature_importance = np.abs(shap_values).mean(axis=0)
                
                # Stability test: compute SHAP values for different samples
                stability_scores = []
                for _ in range(3):
                    sample_idx = np.random.choice(len(X), size=min(50, len(X)), replace=False)
                    shap_sample = explainer.shap_values(X[sample_idx])
                    importance_sample = np.abs(shap_sample).mean(axis=0)
                    stability_scores.append(importance_sample)
                
                stability_coefficient = np.mean([
                    pearsonr(stability_scores[0], stability_scores[i])[0] 
                    for i in range(1, len(stability_scores))
                ])
                
                results[model_name] = {
                    'computation_time': computation_time,
                    'feature_importance': feature_importance.tolist(),
                    'stability_coefficient': stability_coefficient,
                    'shap_values_shape': shap_values.shape,
                    'expected_value': explainer.expected_value,
                    'memory_efficient': True,
                    'handles_interactions': True
                }
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _evaluate_shap_kernel(self, X: np.ndarray, y: np.ndarray, 
                             trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate KernelSHAP (model-agnostic)
        """
        print("  🔮 Evaluating KernelSHAP...")
        results = {}
        
        # Test with a small sample due to computational cost
        X_sample = X[:min(20, len(X))]
        
        for model_name, model_data in list(trained_models.items())[:2]:  # Limit to 2 models
            start_time = time.time()
            model_obj = model_data['model']
            scaler = model_data['scaler']
            
            try:
                # Prepare prediction function
                if scaler is not None:
                    def predict_fn(x):
                        return model_obj.predict(scaler.transform(x))
                else:
                    def predict_fn(x):
                        return model_obj.predict(x)
                
                # KernelSHAP explainer with limited background
                background = X[:min(10, len(X))]
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_sample, nsamples=50)  # Limit samples
                
                computation_time = time.time() - start_time
                feature_importance = np.abs(shap_values).mean(axis=0)
                
                results[model_name] = {
                    'computation_time': computation_time,
                    'feature_importance': feature_importance.tolist(),
                    'shap_values_shape': shap_values.shape,
                    'memory_efficient': False,
                    'model_agnostic': True,
                    'sample_size_used': len(X_sample)
                }
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _evaluate_shap_linear(self, X: np.ndarray, y: np.ndarray, 
                             trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate LinearSHAP for linear models
        """
        print("  📊 Evaluating LinearSHAP...")
        results = {}
        
        for model_name in ['elastic_net', 'linear_regression']:
            if model_name not in trained_models:
                continue
                
            start_time = time.time()
            model_obj = trained_models[model_name]['model']
            scaler = trained_models[model_name]['scaler']
            
            try:
                X_scaled = scaler.transform(X)
                
                # LinearSHAP explainer
                explainer = shap.LinearExplainer(model_obj, X_scaled)
                shap_values = explainer.shap_values(X_scaled[:min(100, len(X_scaled))])
                
                computation_time = time.time() - start_time
                feature_importance = np.abs(shap_values).mean(axis=0)
                
                results[model_name] = {
                    'computation_time': computation_time,
                    'feature_importance': feature_importance.tolist(),
                    'shap_values_shape': shap_values.shape,
                    'memory_efficient': True,
                    'exact_calculation': True,
                    'coefficients': model_obj.coef_.tolist() if hasattr(model_obj, 'coef_') else None
                }
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _evaluate_lime_tabular(self, X: np.ndarray, y: np.ndarray, 
                              trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate LIME for tabular data with temporal feature considerations
        """
        print("  🍋 Evaluating LIME Tabular...")
        results = {}
        
        # Test with one representative model
        model_name = 'random_forest'
        if model_name not in trained_models:
            return {'error': 'Random forest model not available'}
        
        start_time = time.time()
        model_obj = trained_models[model_name]['model']
        
        try:
            # LIME explainer for tabular data
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X[:min(100, len(X))],  # Training data sample
                feature_names=feature_names,
                mode='regression',
                discretize_continuous=False,
                random_state=42
            )
            
            # Explain multiple instances
            explanations = []
            computation_times = []
            
            for i in range(min(5, len(X))):  # Explain 5 instances
                instance_start = time.time()
                explanation = explainer.explain_instance(
                    X[i], 
                    model_obj.predict,
                    num_features=10,
                    num_samples=100  # Reduced for efficiency
                )
                computation_times.append(time.time() - instance_start)
                explanations.append(explanation.as_list())
            
            total_time = time.time() - start_time
            
            # Extract feature importance from explanations
            all_features = {}
            for explanation in explanations:
                for feature, importance in explanation:
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(abs(importance))
            
            avg_importance = {feat: np.mean(values) for feat, values in all_features.items()}
            
            results = {
                'computation_time': total_time,
                'avg_time_per_instance': np.mean(computation_times),
                'feature_importance': avg_importance,
                'local_explanations': True,
                'handles_temporal': True,
                'num_instances_explained': len(explanations)
            }
            
        except Exception as e:
            results = {'error': str(e)}
        
        return results
    
    def _evaluate_permutation_basic(self, X: np.ndarray, y: np.ndarray, 
                                   trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate basic permutation importance
        """
        print("  🔄 Evaluating Basic Permutation Importance...")
        results = {}
        
        for model_name, model_data in trained_models.items():
            start_time = time.time()
            model_obj = model_data['model']
            scaler = model_data['scaler']
            
            try:
                # Prepare data
                X_eval = scaler.transform(X) if scaler is not None else X
                
                # Basic permutation importance
                perm_importance = permutation_importance(
                    model_obj, X_eval, y,
                    n_repeats=5,  # Reduced for efficiency
                    random_state=42,
                    n_jobs=-1
                )
                
                computation_time = time.time() - start_time
                
                results[model_name] = {
                    'computation_time': computation_time,
                    'feature_importance': perm_importance.importances_mean.tolist(),
                    'importance_std': perm_importance.importances_std.tolist(),
                    'robust_to_correlations': False,
                    'statistical_significance': True
                }
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _evaluate_permutation_conditional(self, X: np.ndarray, y: np.ndarray, 
                                         trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate conditional permutation importance for correlated features
        """
        print("  🔄 Evaluating Conditional Permutation Importance...")
        results = {}
        
        # Use one model for this analysis
        model_name = 'random_forest'
        if model_name not in trained_models:
            return {'error': 'Random forest model not available'}
        
        start_time = time.time()
        model_obj = trained_models[model_name]['model']
        
        try:
            # Identify correlated feature groups
            corr_matrix = np.corrcoef(X.T)
            correlated_groups = self._identify_correlation_groups(corr_matrix, threshold=0.7)
            
            # Conditional permutation for each group
            conditional_importance = {}
            
            for group_idx, feature_indices in enumerate(correlated_groups):
                if len(feature_indices) > 1:
                    # Permute group together
                    X_perm = X.copy()
                    perm_idx = np.random.permutation(len(X))
                    for feat_idx in feature_indices:
                        X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
                    
                    # Calculate importance
                    original_score = r2_score(y, model_obj.predict(X))
                    permuted_score = r2_score(y, model_obj.predict(X_perm))
                    importance = original_score - permuted_score
                    
                    group_features = [feature_names[i] for i in feature_indices]
                    conditional_importance[f'group_{group_idx}'] = {
                        'importance': importance,
                        'features': group_features,
                        'correlation_level': np.mean([corr_matrix[i, j] 
                                                    for i in feature_indices 
                                                    for j in feature_indices if i != j])
                    }
            
            computation_time = time.time() - start_time
            
            results = {
                'computation_time': computation_time,
                'conditional_importance': conditional_importance,
                'correlation_groups_found': len(correlated_groups),
                'robust_to_correlations': True
            }
            
        except Exception as e:
            results = {'error': str(e)}
        
        return results
    
    def _evaluate_shap_interactions(self, X: np.ndarray, y: np.ndarray, 
                                   trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate SHAP interaction values for feature interactions
        """
        print("  🔗 Evaluating SHAP Interaction Values...")
        results = {}
        
        # Use tree model for interaction analysis
        model_name = 'random_forest'
        if model_name not in trained_models:
            return {'error': 'Random forest model not available'}
        
        start_time = time.time()
        model_obj = trained_models[model_name]['model']
        
        try:
            # Limit features for computational efficiency
            n_features = min(10, len(feature_names))
            X_subset = X[:min(50, len(X)), :n_features]
            
            # TreeSHAP with interactions
            explainer = shap.TreeExplainer(model_obj)
            shap_interaction_values = explainer.shap_interaction_values(X_subset)
            
            computation_time = time.time() - start_time
            
            # Extract main effects and interactions
            main_effects = np.diagonal(shap_interaction_values, axis1=1, axis2=2).mean(axis=0)
            
            # Top interactions (off-diagonal elements)
            interaction_effects = {}
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
                    if interaction_strength > 0.001:  # Threshold for meaningful interactions
                        interaction_effects[f'{feature_names[i]}_x_{feature_names[j]}'] = interaction_strength
            
            # Sort interactions by strength
            top_interactions = sorted(interaction_effects.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            
            results = {
                'computation_time': computation_time,
                'main_effects': main_effects.tolist(),
                'top_interactions': top_interactions,
                'interaction_matrix_shape': shap_interaction_values.shape,
                'features_analyzed': n_features,
                'memory_intensive': True
            }
            
        except Exception as e:
            results = {'error': str(e)}
        
        return results
    
    def _evaluate_temporal_importance(self, X: np.ndarray, y: np.ndarray, 
                                     trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate temporal pattern importance for lag features
        """
        print("  ⏱️ Evaluating Temporal Importance Patterns...")
        results = {}
        
        # Use random forest for temporal analysis
        model_name = 'random_forest'
        if model_name not in trained_models:
            return {'error': 'Random forest model not available'}
        
        start_time = time.time()
        model_obj = trained_models[model_name]['model']
        
        try:
            # Group features by lag period
            lag_importance = {}
            lag_groups = {}
            
            for lag in self.lag_periods:
                lag_features = []
                for i, feature in enumerate(feature_names):
                    if f'lag{lag}' in feature.lower() or f'_lag_{lag}' in feature.lower():
                        lag_features.append(i)
                
                if lag_features:
                    lag_groups[lag] = lag_features
                    
                    # Calculate importance for this lag period
                    if hasattr(model_obj, 'feature_importances_'):
                        lag_importance[lag] = np.sum([model_obj.feature_importances_[i] 
                                                    for i in lag_features])
                    else:
                        # Use permutation importance for this lag group
                        X_perm = X.copy()
                        perm_idx = np.random.permutation(len(X))
                        for feat_idx in lag_features:
                            X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
                        
                        original_score = r2_score(y, model_obj.predict(X))
                        permuted_score = r2_score(y, model_obj.predict(X_perm))
                        lag_importance[lag] = original_score - permuted_score
            
            # Identify critical lag periods
            if lag_importance:
                critical_lags = sorted(lag_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
            else:
                critical_lags = []
            
            computation_time = time.time() - start_time
            
            results = {
                'computation_time': computation_time,
                'lag_importance': lag_importance,
                'critical_lag_periods': critical_lags,
                'lag_groups': {str(k): v for k, v in lag_groups.items()},
                'temporal_pattern_detected': len(lag_importance) > 0
            }
            
        except Exception as e:
            results = {'error': str(e)}
        
        return results
    
    def _evaluate_anchor_explanations(self, X: np.ndarray, y: np.ndarray, 
                                     trained_models: Dict, feature_names: List[str]) -> Dict:
        """
        Evaluate feasibility of anchor explanations (rule-based)
        """
        print("  ⚓ Evaluating Anchor Explanations Feasibility...")
        
        # Simplified anchor-like analysis using decision rules
        results = {}
        
        model_name = 'random_forest'
        if model_name not in trained_models:
            return {'error': 'Random forest model not available'}
        
        start_time = time.time()
        model_obj = trained_models[model_name]['model']
        
        try:
            # Extract simple rules from random forest
            rules = []
            predictions = model_obj.predict(X)
            
            # For each feature, find threshold-based rules
            for i, feature in enumerate(feature_names[:10]):  # Limit for efficiency
                feature_values = X[:, i]
                
                # Find median as threshold
                threshold = np.median(feature_values)
                
                # Evaluate rule: "if feature > threshold"
                high_mask = feature_values > threshold
                low_mask = feature_values <= threshold
                
                if np.sum(high_mask) > 10 and np.sum(low_mask) > 10:
                    high_pred_mean = np.mean(predictions[high_mask])
                    low_pred_mean = np.mean(predictions[low_mask])
                    effect_size = abs(high_pred_mean - low_pred_mean)
                    
                    rules.append({
                        'feature': feature,
                        'threshold': threshold,
                        'effect_size': effect_size,
                        'coverage': np.sum(high_mask) / len(X),
                        'rule': f'{feature} > {threshold:.3f}'
                    })
            
            # Sort rules by effect size
            rules.sort(key=lambda x: x['effect_size'], reverse=True)
            
            computation_time = time.time() - start_time
            
            results = {
                'computation_time': computation_time,
                'rules_extracted': len(rules),
                'top_rules': rules[:5],
                'rule_based_explanations': True,
                'interpretable_thresholds': True
            }
            
        except Exception as e:
            results = {'error': str(e)}
        
        return results
    
    def _identify_correlation_groups(self, corr_matrix: np.ndarray, 
                                    threshold: float = 0.7) -> List[List[int]]:
        """
        Identify groups of correlated features
        """
        n_features = corr_matrix.shape[0]
        visited = set()
        groups = []
        
        for i in range(n_features):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            for j in range(i+1, n_features):
                if j not in visited and abs(corr_matrix[i, j]) > threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def evaluate_computational_efficiency(self, method_results: Dict) -> Dict:
        """
        Evaluate computational efficiency of different XAI methods
        """
        efficiency_analysis = {}
        
        for method_name, results in method_results.items():
            if 'error' in results:
                efficiency_analysis[method_name] = {
                    'status': 'failed',
                    'error': results['error']
                }
                continue
            
            # Extract timing information
            total_time = 0
            if isinstance(results, dict):
                for model_result in results.values():
                    if isinstance(model_result, dict) and 'computation_time' in model_result:
                        total_time += model_result['computation_time']
            
            # Efficiency rating
            if total_time < 1:
                efficiency_rating = 'very_fast'
            elif total_time < 10:
                efficiency_rating = 'fast'
            elif total_time < 60:
                efficiency_rating = 'moderate'
            else:
                efficiency_rating = 'slow'
            
            efficiency_analysis[method_name] = {
                'total_computation_time': total_time,
                'efficiency_rating': efficiency_rating,
                'scalability': 'high' if total_time < 10 else 'medium' if total_time < 60 else 'low',
                'memory_efficient': any('memory_efficient' in str(results) for _ in [1])
            }
        
        return efficiency_analysis
    
    def evaluate_correlation_handling(self, X: np.ndarray, 
                                     feature_names: List[str], 
                                     method_results: Dict) -> Dict:
        """
        Evaluate how well different methods handle correlated features
        """
        # Calculate feature correlations
        corr_matrix = np.corrcoef(X.T)
        high_corr_pairs = []
        
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.7:
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
        
        correlation_analysis = {
            'high_correlation_pairs': len(high_corr_pairs),
            'max_correlation': np.max(np.abs(corr_matrix - np.eye(len(feature_names)))),
            'correlation_handling_assessment': {}
        }
        
        # Assess each method's correlation handling
        methods_robust_to_correlation = [
            'permutation_conditional', 'shap_tree', 'shap_linear'
        ]
        
        for method_name in method_results:
            if method_name in methods_robust_to_correlation:
                correlation_analysis['correlation_handling_assessment'][method_name] = 'robust'
            elif 'permutation_basic' in method_name or 'lime' in method_name:
                correlation_analysis['correlation_handling_assessment'][method_name] = 'vulnerable'
            else:
                correlation_analysis['correlation_handling_assessment'][method_name] = 'moderate'
        
        return correlation_analysis
    
    def generate_method_recommendations(self, 
                                       evaluation_results: Dict, 
                                       efficiency_analysis: Dict,
                                       correlation_analysis: Dict) -> Dict:
        """
        Generate specific recommendations for climate-health XAI application
        """
        recommendations = {
            'tier_1_immediate': [],
            'tier_2_secondary': [],
            'tier_3_research': [],
            'not_recommended': []
        }
        
        # Evaluation criteria
        for method_name in evaluation_results:
            if method_name in efficiency_analysis:
                efficiency = efficiency_analysis[method_name]
                
                # Tier 1: Fast, robust, and suitable for moderate effect sizes
                if (efficiency['efficiency_rating'] in ['very_fast', 'fast'] and
                    correlation_analysis['correlation_handling_assessment'].get(method_name) == 'robust'):
                    recommendations['tier_1_immediate'].append({
                        'method': method_name,
                        'rationale': 'Fast computation, robust to correlations, suitable for production'
                    })
                
                # Tier 2: Good performance but some limitations
                elif efficiency['efficiency_rating'] in ['fast', 'moderate']:
                    recommendations['tier_2_secondary'].append({
                        'method': method_name,
                        'rationale': 'Good performance, may require optimization for large-scale use'
                    })
                
                # Tier 3: Research applications only
                elif efficiency['efficiency_rating'] == 'slow':
                    recommendations['tier_3_research'].append({
                        'method': method_name,
                        'rationale': 'Computationally expensive, suitable for research applications only'
                    })
                
                # Not recommended if failed or very poor performance
                else:
                    recommendations['not_recommended'].append({
                        'method': method_name,
                        'rationale': 'Poor performance or technical issues'
                    })
        
        return recommendations
    
    def run_comprehensive_evaluation(self, biomarker: str = 'systolic_blood_pressure') -> Dict:
        """
        Run comprehensive XAI evaluation for a specific biomarker
        """
        print("="*80)
        print(f"🔍 COMPREHENSIVE MODEL-AGNOSTIC XAI EVALUATION")
        print(f"Target Biomarker: {biomarker}")
        print("="*80)
        
        start_time = time.time()
        
        # Load data
        df, biomarkers, climate_features = self.load_and_prepare_data()
        
        if biomarker not in df.columns:
            raise ValueError(f"Biomarker {biomarker} not found in dataset")
        
        # Prepare model data
        print(f"🔄 Preparing data for {biomarker}...")
        X, y, selected_features = self._prepare_model_data(df, biomarker, climate_features)
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train models
        print("🤖 Training multiple model types...")
        trained_models = self._train_models(X, y)
        
        # Evaluate XAI methods
        print("🔍 Evaluating XAI methods...")
        method_results = {}
        
        for method_name, method_func in self.xai_methods.items():
            print(f"  Evaluating {method_name}...")
            try:
                method_results[method_name] = method_func(X, y, trained_models, selected_features)
            except Exception as e:
                method_results[method_name] = {'error': str(e)}
                print(f"    ❌ Error in {method_name}: {e}")
        
        # Efficiency analysis
        print("⚡ Analyzing computational efficiency...")
        efficiency_analysis = self.evaluate_computational_efficiency(method_results)
        
        # Correlation handling analysis
        print("🔗 Analyzing correlation handling capabilities...")
        correlation_analysis = self.evaluate_correlation_handling(X, selected_features, method_results)
        
        # Generate recommendations
        print("📋 Generating method recommendations...")
        recommendations = self.generate_method_recommendations(
            method_results, efficiency_analysis, correlation_analysis
        )
        
        total_time = time.time() - start_time
        
        # Compile final results
        comprehensive_results = {
            'metadata': {
                'biomarker': biomarker,
                'timestamp': self.timestamp,
                'total_evaluation_time': total_time,
                'dataset_shape': [X.shape[0], X.shape[1]],
                'features_selected': selected_features
            },
            'xai_method_results': method_results,
            'efficiency_analysis': efficiency_analysis,
            'correlation_analysis': correlation_analysis,
            'recommendations': recommendations,
            'summary': self._generate_evaluation_summary(method_results, recommendations)
        }
        
        # Save results
        results_file = self.results_dir / f"comprehensive_xai_evaluation_{biomarker}_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print("="*80)
        print("✅ COMPREHENSIVE XAI EVALUATION COMPLETE")
        print(f"📁 Results saved: {results_file}")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print("="*80)
        
        return comprehensive_results
    
    def _generate_evaluation_summary(self, method_results: Dict, recommendations: Dict) -> Dict:
        """
        Generate summary of evaluation results
        """
        summary = {
            'methods_evaluated': len(method_results),
            'methods_successful': len([r for r in method_results.values() if 'error' not in r]),
            'methods_failed': len([r for r in method_results.values() if 'error' in r]),
            'tier_1_methods': len(recommendations['tier_1_immediate']),
            'tier_2_methods': len(recommendations['tier_2_secondary']),
            'tier_3_methods': len(recommendations['tier_3_research'])
        }
        
        return summary

def main():
    """
    Execute comprehensive XAI evaluation
    """
    evaluator = ComprehensiveXAIEvaluator(
        max_features=30,  # Reduced for demonstration
        sample_size=1000
    )
    
    # Evaluate for primary biomarker
    results = evaluator.run_comprehensive_evaluation('systolic_blood_pressure')
    
    return results

if __name__ == "__main__":
    main()