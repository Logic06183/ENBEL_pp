#!/usr/bin/env python3
"""
Practical XAI Evaluation for Climate-Health Data
================================================

This script evaluates the most practical and validated XAI methods 
for climate-health relationships with moderate effect sizes.

Focus: TreeSHAP, Permutation Importance, Basic SHAP methods
"""

import pandas as pd
import numpy as np
import shap
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from datetime import datetime
from pathlib import Path
import json
import joblib

warnings.filterwarnings('ignore')

class PracticalXAIEvaluator:
    """
    Practical evaluation of core XAI methods for climate-health research
    """
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("practical_xai_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Climate variable patterns for grouping
        self.climate_patterns = {
            'temperature': ['temp', 'tas', 'temperature', 'apparent'],
            'heat_stress': ['heat_index', 'utci', 'wbgt'],
            'wind': ['wind', 'ws']
        }
        
        self.lag_periods = [0, 1, 2, 3, 5, 7, 10, 14, 21]
        
    def load_and_prepare_data(self, biomarker='systolic blood pressure'):
        """Load and prepare data for XAI evaluation"""
        print(f"🔄 Loading data for {biomarker}...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Get climate features
        climate_features = []
        for col in df.columns:
            if any(term in col.lower() for term in ['temp', 'heat', 'wind', 'humid']) and \
               any(f'lag{lag}' in col.lower() for lag in self.lag_periods):
                climate_features.append(col)
        
        # Clean data for the biomarker
        clean_data = df.dropna(subset=[biomarker]).copy()
        
        # Select top features by correlation
        available_features = [f for f in climate_features[:60] if f in clean_data.columns]
        
        if len(available_features) < 10:
            raise ValueError(f"Insufficient features for {biomarker}")
        
        X_temp = clean_data[available_features].fillna(clean_data[available_features].median())
        y = clean_data[biomarker]
        
        # Feature selection by correlation with target
        feature_correlations = []
        for feat in available_features:
            if X_temp[feat].var() > 1e-8:
                corr = abs(pearsonr(X_temp[feat], y)[0])
                feature_correlations.append((feat, corr))
        
        # Select top 25 features
        feature_correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in feature_correlations[:25]]
        
        X = X_temp[top_features]
        
        # Sample for efficiency
        if len(X) > 1500:
            sample_idx = np.random.choice(len(X), size=1500, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]
        
        print(f"✅ Data prepared: {len(X)} samples, {len(top_features)} features")
        
        return X, y, top_features
    
    def train_models(self, X, y):
        """Train multiple model types for XAI comparison"""
        print("🤖 Training models...")
        
        models = {}
        
        # Random Forest (for TreeSHAP)
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        models['random_forest'] = {'model': rf, 'type': 'tree'}
        
        # Gradient Boosting (for TreeSHAP)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X, y)
        models['gradient_boosting'] = {'model': gb, 'type': 'tree'}
        
        # Linear model (for LinearSHAP)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = ElasticNet(alpha=0.1, random_state=42, max_iter=2000)
        lr.fit(X_scaled, y)
        models['elastic_net'] = {'model': lr, 'scaler': scaler, 'type': 'linear'}
        
        print("✅ Models trained")
        return models
    
    def evaluate_tree_shap(self, X, y, models):
        """Evaluate TreeSHAP on tree-based models"""
        print("  🌳 Evaluating TreeSHAP...")
        
        results = {}
        
        for model_name in ['random_forest', 'gradient_boosting']:
            if model_name not in models:
                continue
                
            start_time = time.time()
            model = models[model_name]['model']
            
            try:
                # TreeSHAP explainer
                explainer = shap.TreeExplainer(model)
                
                # Compute SHAP values (sample for efficiency)
                X_sample = X.iloc[:min(200, len(X))]
                shap_values = explainer.shap_values(X_sample)
                
                # Feature importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                
                # Model performance
                r2 = model.score(X, y)
                
                computation_time = time.time() - start_time
                
                results[model_name] = {
                    'computation_time': computation_time,
                    'model_r2': r2,
                    'feature_importance': dict(zip(X.columns, feature_importance)),
                    'shap_values_computed': True,
                    'expected_value': explainer.expected_value,
                    'sample_size': len(X_sample)
                }
                
                print(f"    ✅ {model_name}: R² = {r2:.3f}, Time = {computation_time:.1f}s")
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
                print(f"    ❌ {model_name} failed: {e}")
        
        return results
    
    def evaluate_linear_shap(self, X, y, models):
        """Evaluate LinearSHAP on linear models"""
        print("  📊 Evaluating LinearSHAP...")
        
        results = {}
        
        if 'elastic_net' not in models:
            return {'error': 'No linear model available'}
        
        start_time = time.time()
        model_data = models['elastic_net']
        model = model_data['model']
        scaler = model_data['scaler']
        
        try:
            X_scaled = scaler.transform(X)
            
            # LinearSHAP explainer
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled[:min(200, len(X_scaled))])
            
            # Feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Model performance
            r2 = model.score(X_scaled, y)
            
            computation_time = time.time() - start_time
            
            results = {
                'computation_time': computation_time,
                'model_r2': r2,
                'feature_importance': dict(zip(X.columns, feature_importance)),
                'coefficients': dict(zip(X.columns, model.coef_)),
                'exact_calculation': True
            }
            
            print(f"    ✅ Linear model: R² = {r2:.3f}, Time = {computation_time:.1f}s")
            
        except Exception as e:
            results = {'error': str(e)}
            print(f"    ❌ Linear SHAP failed: {e}")
        
        return results
    
    def evaluate_permutation_importance(self, X, y, models):
        """Evaluate permutation importance across models"""
        print("  🔄 Evaluating Permutation Importance...")
        
        results = {}
        
        for model_name, model_data in models.items():
            start_time = time.time()
            model = model_data['model']
            
            try:
                # Prepare data
                if 'scaler' in model_data:
                    X_eval = model_data['scaler'].transform(X)
                else:
                    X_eval = X.values
                
                # Permutation importance
                perm_result = permutation_importance(
                    model, X_eval, y,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Model performance
                r2 = model.score(X_eval, y)
                
                computation_time = time.time() - start_time
                
                results[model_name] = {
                    'computation_time': computation_time,
                    'model_r2': r2,
                    'feature_importance': dict(zip(X.columns, perm_result.importances_mean)),
                    'importance_std': dict(zip(X.columns, perm_result.importances_std)),
                    'robust_to_correlations': True
                }
                
                print(f"    ✅ {model_name}: R² = {r2:.3f}, Time = {computation_time:.1f}s")
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
                print(f"    ❌ {model_name} failed: {e}")
        
        return results
    
    def analyze_temporal_patterns(self, importance_dict, feature_names):
        """Analyze temporal lag patterns in feature importance"""
        print("  ⏱️ Analyzing temporal patterns...")
        
        lag_analysis = {}
        
        for lag in self.lag_periods:
            lag_features = [f for f in feature_names if f'lag{lag}' in f.lower()]
            
            if lag_features:
                total_importance = sum(importance_dict.get(f, 0) for f in lag_features)
                lag_analysis[lag] = {
                    'total_importance': total_importance,
                    'n_features': len(lag_features),
                    'avg_importance': total_importance / len(lag_features) if lag_features else 0,
                    'features': lag_features
                }
        
        # Find most important lag periods
        if lag_analysis:
            critical_lags = sorted(lag_analysis.items(), 
                                 key=lambda x: x[1]['total_importance'], 
                                 reverse=True)[:3]
            
            print(f"    🔍 Critical lag periods: {[f'Day {lag}' for lag, _ in critical_lags]}")
        
        return lag_analysis
    
    def analyze_climate_groups(self, importance_dict, feature_names):
        """Analyze importance by climate variable groups"""
        print("  🌡️ Analyzing climate variable groups...")
        
        group_analysis = {}
        
        for group_name, patterns in self.climate_patterns.items():
            group_features = []
            for feature in feature_names:
                if any(pattern in feature.lower() for pattern in patterns):
                    group_features.append(feature)
            
            if group_features:
                total_importance = sum(importance_dict.get(f, 0) for f in group_features)
                group_analysis[group_name] = {
                    'total_importance': total_importance,
                    'n_features': len(group_features),
                    'avg_importance': total_importance / len(group_features),
                    'top_features': sorted([(f, importance_dict.get(f, 0)) for f in group_features], 
                                         key=lambda x: x[1], reverse=True)[:3]
                }
        
        # Most important climate groups
        if group_analysis:
            top_groups = sorted(group_analysis.items(), 
                              key=lambda x: x[1]['total_importance'], 
                              reverse=True)
            
            print(f"    🔍 Most important climate groups: {[group for group, _ in top_groups]}")
        
        return group_analysis
    
    def create_visualizations(self, results, biomarker):
        """Create visualization of XAI results"""
        print("📊 Creating visualizations...")
        
        # Prepare data for plotting
        all_importance = {}
        
        # Collect feature importance from all methods
        for method_name, method_results in results.items():
            if method_name in ['tree_shap', 'linear_shap']:
                if 'feature_importance' in method_results:
                    all_importance[method_name] = method_results['feature_importance']
            elif method_name == 'permutation':
                for model_name, model_results in method_results.items():
                    if 'feature_importance' in model_results:
                        all_importance[f'perm_{model_name}'] = model_results['feature_importance']
        
        if not all_importance:
            print("  ⚠️ No feature importance data available for visualization")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(1, len(all_importance), figsize=(5 * len(all_importance), 8))
        if len(all_importance) == 1:
            axes = [axes]
        
        for i, (method, importance) in enumerate(all_importance.items()):
            # Top 10 features
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            features, values = zip(*sorted_features)
            
            axes[i].barh(range(len(features)), [abs(v) for v in values])
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in features])
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'{method.upper()}\nTop 10 Features')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"xai_comparison_{biomarker}_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Visualization saved: {plot_file}")
        
        return str(plot_file)
    
    def run_evaluation(self, biomarker='systolic blood pressure'):
        """Run comprehensive practical XAI evaluation"""
        print("="*80)
        print(f"🔍 PRACTICAL XAI EVALUATION FOR CLIMATE-HEALTH DATA")
        print(f"Target Biomarker: {biomarker}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            X, y, feature_names = self.load_and_prepare_data(biomarker)
            
            # Train models
            models = self.train_models(X, y)
            
            # Evaluate XAI methods
            print("\n🔍 Evaluating XAI Methods...")
            
            # TreeSHAP
            tree_shap_results = self.evaluate_tree_shap(X, y, models)
            
            # LinearSHAP
            linear_shap_results = self.evaluate_linear_shap(X, y, models)
            
            # Permutation Importance
            permutation_results = self.evaluate_permutation_importance(X, y, models)
            
            # Compile results
            results = {
                'tree_shap': tree_shap_results,
                'linear_shap': linear_shap_results,
                'permutation': permutation_results
            }
            
            # Temporal and group analysis
            print("\n🔍 Analyzing Patterns...")
            
            # Use the best performing method for pattern analysis
            best_importance = None
            if 'random_forest' in tree_shap_results and 'feature_importance' in tree_shap_results['random_forest']:
                best_importance = tree_shap_results['random_forest']['feature_importance']
            elif 'random_forest' in permutation_results and 'feature_importance' in permutation_results['random_forest']:
                best_importance = permutation_results['random_forest']['feature_importance']
            
            if best_importance:
                temporal_analysis = self.analyze_temporal_patterns(best_importance, feature_names)
                climate_analysis = self.analyze_climate_groups(best_importance, feature_names)
                
                results['temporal_patterns'] = temporal_analysis
                results['climate_groups'] = climate_analysis
            
            # Create visualizations
            plot_file = self.create_visualizations(results, biomarker)
            
            # Compile final results
            final_results = {
                'metadata': {
                    'biomarker': biomarker,
                    'timestamp': self.timestamp,
                    'evaluation_time': time.time() - start_time,
                    'n_samples': len(X),
                    'n_features': len(feature_names),
                    'features': feature_names
                },
                'xai_results': results,
                'visualization': plot_file,
                'summary': self._create_summary(results)
            }
            
            # Save results
            results_file = self.results_dir / f"practical_xai_{biomarker}_{self.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            total_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("✅ PRACTICAL XAI EVALUATION COMPLETE")
            print(f"📁 Results saved: {results_file}")
            print(f"⏱️ Total time: {total_time:.1f} seconds")
            self._print_summary(final_results['summary'])
            print("="*80)
            
            return final_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {'error': str(e)}
    
    def _create_summary(self, results):
        """Create summary of evaluation results"""
        summary = {
            'methods_evaluated': [],
            'best_performing_models': {},
            'computation_times': {},
            'key_findings': []
        }
        
        # Collect method information
        for method_name, method_results in results.items():
            if method_name in ['tree_shap', 'linear_shap']:
                summary['methods_evaluated'].append(method_name)
                for model_name, model_results in method_results.items():
                    if isinstance(model_results, dict) and 'model_r2' in model_results:
                        summary['best_performing_models'][f'{method_name}_{model_name}'] = model_results['model_r2']
                        summary['computation_times'][f'{method_name}_{model_name}'] = model_results.get('computation_time', 0)
            elif method_name == 'permutation':
                summary['methods_evaluated'].append(method_name)
                for model_name, model_results in method_results.items():
                    if isinstance(model_results, dict) and 'model_r2' in model_results:
                        summary['best_performing_models'][f'perm_{model_name}'] = model_results['model_r2']
                        summary['computation_times'][f'perm_{model_name}'] = model_results.get('computation_time', 0)
        
        # Key findings
        if 'temporal_patterns' in results:
            critical_lags = sorted(results['temporal_patterns'].items(), 
                                 key=lambda x: x[1]['total_importance'], 
                                 reverse=True)[:2]
            summary['key_findings'].append(f"Critical lag periods: {[f'Day {lag}' for lag, _ in critical_lags]}")
        
        if 'climate_groups' in results:
            top_groups = sorted(results['climate_groups'].items(), 
                              key=lambda x: x[1]['total_importance'], 
                              reverse=True)[:2]
            summary['key_findings'].append(f"Most important climate types: {[group for group, _ in top_groups]}")
        
        return summary
    
    def _print_summary(self, summary):
        """Print formatted summary"""
        print("\n📋 EVALUATION SUMMARY:")
        print(f"  Methods evaluated: {len(summary['methods_evaluated'])}")
        
        if summary['best_performing_models']:
            best_model = max(summary['best_performing_models'].items(), key=lambda x: x[1])
            print(f"  Best performing: {best_model[0]} (R² = {best_model[1]:.3f})")
        
        if summary['computation_times']:
            avg_time = np.mean(list(summary['computation_times'].values()))
            print(f"  Average computation time: {avg_time:.1f} seconds")
        
        if summary['key_findings']:
            print("  Key findings:")
            for finding in summary['key_findings']:
                print(f"    - {finding}")

def main():
    """Run practical XAI evaluation"""
    evaluator = PracticalXAIEvaluator()
    results = evaluator.run_evaluation('systolic blood pressure')
    return results

if __name__ == "__main__":
    main()