#!/usr/bin/env python3
"""
Rigorous ML Multi-Biomarker Climate-Health Analysis
===================================================

Scientifically rigorous machine learning approach combining:
1. Multi-biomarker composite targets
2. Multi-task learning
3. Ensemble methods with statistical validation
4. Traditional epidemiological validation
5. Causal inference considerations

Building on our successful simple methodology but leveraging ML power.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class RigorousMLMultivariateAnalysis:
    def __init__(self):
        self.alpha = 0.01  # Strict significance threshold
        self.min_effect_size = 0.02  # Minimum R² for meaningful effect
        self.min_sample_size = 500
        self.cv_folds = 10
        self.cv_repeats = 3
        
    def log_analysis(self, message, level="INFO"):
        """Scientific logging"""
        icons = {"INFO": "🔬", "SUCCESS": "✅", "WARNING": "⚠️", "DISCOVERY": "🎯"}
        print(f"{icons.get(level, '🔬')} {message}")
    
    def load_and_prepare_multivariate_data(self):
        """Load data with focus on multi-biomarker analysis"""
        self.log_analysis("RIGOROUS ML MULTI-BIOMARKER CLIMATE-HEALTH ANALYSIS")
        self.log_analysis("=" * 55)
        self.log_analysis("Loading data for multivariate ML analysis...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Define biomarker systems based on our successful findings
        biomarker_systems = {
            'cardiovascular': {
                'biomarkers': ['systolic blood pressure', 'diastolic blood pressure'],
                'rationale': 'Blood pressure regulation system'
            },
            'metabolic': {
                'biomarkers': ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING LDL'],
                'rationale': 'Glucose and lipid metabolism'
            },
            'immune_hematologic': {
                'biomarkers': ['CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)'],
                'rationale': 'Immune and blood cell function'
            },
            'renal_metabolic': {
                'biomarkers': ['Creatinine (mg/dL)', 'FASTING GLUCOSE'],
                'rationale': 'Kidney function and glucose metabolism'
            }
        }
        
        # Temperature variables that showed significance in our previous analysis
        significant_climate_vars = [
            'temperature_tas_lag0', 'temperature_tas_lag1', 
            'temperature_tas_lag2', 'temperature_tas_lag3'
        ]
        
        # Validate availability and sample sizes
        validated_systems = {}
        for system_name, system_info in biomarker_systems.items():
            available_biomarkers = [b for b in system_info['biomarkers'] if b in df.columns]
            
            if len(available_biomarkers) >= 2:
                # Check combined sample size
                combined_data = df.dropna(subset=available_biomarkers)
                if len(combined_data) >= self.min_sample_size:
                    validated_systems[system_name] = {
                        'biomarkers': available_biomarkers,
                        'n_samples': len(combined_data),
                        'rationale': system_info['rationale']
                    }
                    self.log_analysis(f"Validated system: {system_name} ({len(available_biomarkers)} biomarkers, n={len(combined_data):,})")
        
        available_climate = [c for c in significant_climate_vars if c in df.columns]
        self.log_analysis(f"Climate variables: {len(available_climate)}")
        
        return df, validated_systems, available_climate
    
    def create_composite_biomarker_targets(self, df, system_info):
        """Create scientifically meaningful composite biomarker targets"""
        biomarkers = system_info['biomarkers']
        system_data = df.dropna(subset=biomarkers)
        
        if len(system_data) < self.min_sample_size:
            return None, None
        
        # Method 1: Standardized Average (equal weighting)
        scaler = StandardScaler()
        standardized_biomarkers = scaler.fit_transform(system_data[biomarkers])
        composite_average = np.mean(standardized_biomarkers, axis=1)
        
        # Method 2: Principal Component (first PC)
        pca = PCA(n_components=1)
        first_pc = pca.fit_transform(standardized_biomarkers).flatten()
        
        # Method 3: Clinically-weighted composite (if applicable)
        clinical_weights = self._get_clinical_weights(biomarkers)
        if clinical_weights:
            weighted_composite = np.average(standardized_biomarkers, weights=clinical_weights, axis=1)
        else:
            weighted_composite = composite_average
        
        composites = {
            'average': composite_average,
            'pca': first_pc,
            'clinical': weighted_composite
        }
        
        # Add composites to dataframe
        for comp_name, comp_values in composites.items():
            system_data[f'composite_{comp_name}'] = comp_values
        
        self.log_analysis(f"Created {len(composites)} composite targets")
        self.log_analysis(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}")
        
        return system_data, composites
    
    def _get_clinical_weights(self, biomarkers):
        """Get clinical importance weights for biomarkers"""
        # Clinical importance weights based on cardiovascular risk
        weights_map = {
            'systolic blood pressure': 0.6,  # Higher clinical importance
            'diastolic blood pressure': 0.4,
            'FASTING GLUCOSE': 0.4,
            'FASTING TOTAL CHOLESTEROL': 0.25,
            'FASTING HDL': 0.35,  # Protective factor
            'FASTING LDL': 0.3,
            'CD4 cell count (cells/µL)': 0.6,
            'Hemoglobin (g/dL)': 0.4,
            'Creatinine (mg/dL)': 0.5
        }
        
        weights = [weights_map.get(b, 1.0) for b in biomarkers]
        return weights if len(set(weights)) > 1 else None
    
    def rigorous_ml_composite_analysis(self, system_data, composites, climate_vars, system_name):
        """Rigorous ML analysis of composite biomarker targets"""
        self.log_analysis(f"\n🎯 ML Analysis: {system_name.upper()} System")
        self.log_analysis("-" * 45)
        
        # Prepare climate data
        climate_data = system_data[climate_vars].fillna(system_data[climate_vars].median())
        
        # Feature selection based on our successful methodology
        selector = SelectKBest(score_func=f_regression, k=min(len(climate_vars), 4))
        
        results = {}
        
        for comp_name, comp_values in composites.items():
            self.log_analysis(f"\nComposite target: {comp_name}")
            
            # Feature selection
            X_selected = selector.fit_transform(climate_data, comp_values)
            selected_features = [climate_vars[i] for i in selector.get_support(indices=True)]
            
            # ML Models with conservative hyperparameters
            models = {
                'elastic_net': ElasticNet(alpha=1.0, random_state=42, max_iter=2000),
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_split=20,
                    min_samples_leaf=10, random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
                )
            }
            
            # Rigorous cross-validation
            cv = RepeatedKFold(n_splits=self.cv_folds, n_repeats=self.cv_repeats, random_state=42)
            
            model_performance = {}
            for model_name, model in models.items():
                cv_results = cross_validate(
                    model, X_selected, comp_values, cv=cv, 
                    scoring=['r2', 'neg_mean_squared_error'],
                    return_train_score=False
                )
                
                mean_r2 = np.mean(cv_results['test_r2'])
                std_r2 = np.std(cv_results['test_r2'])
                
                model_performance[model_name] = {
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'scores': cv_results['test_r2']
                }
                
                self.log_analysis(f"  {model_name}: R² = {mean_r2:.4f} ± {std_r2:.4f}")
            
            # Select best model
            best_model_name = max(model_performance.keys(), 
                                key=lambda k: model_performance[k]['mean_r2'])
            best_r2 = model_performance[best_model_name]['mean_r2']
            
            # Statistical significance testing
            if best_r2 > self.min_effect_size:
                # Permutation test
                best_model = models[best_model_name]
                null_scores = []
                
                for _ in range(200):  # Reduced for efficiency
                    y_perm = np.random.permutation(comp_values)
                    perm_scores = cross_val_score(best_model, X_selected, y_perm, cv=5, scoring='r2')
                    null_scores.append(np.mean(perm_scores))
                
                p_value = np.mean(np.array(null_scores) >= best_r2)
                
                # Effect size validation
                if p_value < self.alpha:
                    # Model interpretation
                    best_model.fit(X_selected, comp_values)
                    
                    # Feature importance
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = best_model.feature_importances_
                    elif hasattr(best_model, 'coef_'):
                        feature_importance = np.abs(best_model.coef_)
                    else:
                        feature_importance = None
                    
                    if feature_importance is not None:
                        importance_df = pd.DataFrame({
                            'feature': selected_features,
                            'importance': feature_importance
                        }).sort_values('importance', ascending=False)
                        top_feature = importance_df.iloc[0]['feature']
                    else:
                        top_feature = selected_features[0]
                    
                    # Simple correlation for validation
                    simple_corr, simple_p = pearsonr(climate_data[top_feature], comp_values)
                    
                    results[comp_name] = {
                        'best_model': best_model_name,
                        'r2': best_r2,
                        'r2_std': model_performance[best_model_name]['std_r2'],
                        'p_value': p_value,
                        'n_samples': len(comp_values),
                        'n_features': len(selected_features),
                        'selected_features': selected_features,
                        'top_feature': top_feature,
                        'simple_correlation': simple_corr,
                        'simple_p_value': simple_p,
                        'model_performance': model_performance
                    }
                    
                    self.log_analysis(f"  ✅ SIGNIFICANT: {comp_name} - R² = {best_r2:.4f}, p = {p_value:.4f}", "SUCCESS")
                    self.log_analysis(f"     Top predictor: {top_feature} (r = {simple_corr:.3f})")
                else:
                    self.log_analysis(f"  Not significant: {comp_name} - p = {p_value:.4f}")
            else:
                self.log_analysis(f"  Below threshold: {comp_name} - R² = {best_r2:.4f}")
        
        return results
    
    def multi_task_learning_analysis(self, system_data, biomarkers, climate_vars, system_name):
        """Multi-task learning for simultaneous biomarker prediction"""
        self.log_analysis(f"\n🔄 Multi-Task Learning: {system_name.upper()}")
        self.log_analysis("-" * 40)
        
        # Prepare data
        Y = system_data[biomarkers].values  # Multiple targets
        X = system_data[climate_vars].fillna(system_data[climate_vars].median()).values
        
        # Standardize targets for multi-task learning
        target_scaler = StandardScaler()
        Y_scaled = target_scaler.fit_transform(Y)
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(len(climate_vars), 4))
        X_selected = selector.fit_transform(X, Y_scaled[:, 0])  # Use first biomarker for selection
        selected_features = [climate_vars[i] for i in selector.get_support(indices=True)]
        
        # Multi-task models
        multi_models = {
            'multi_task_elastic_net': MultiTaskElasticNet(alpha=1.0, random_state=42, max_iter=2000),
            'multi_output_rf': MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            )
        }
        
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        
        multi_results = {}
        for model_name, model in multi_models.items():
            cv_scores = cross_val_score(model, X_selected, Y_scaled, cv=cv, scoring='r2')
            mean_r2 = np.mean(cv_scores)
            std_r2 = np.std(cv_scores)
            
            self.log_analysis(f"  {model_name}: R² = {mean_r2:.4f} ± {std_r2:.4f}")
            
            if mean_r2 > self.min_effect_size:
                # Permutation test
                null_scores = []
                for _ in range(100):
                    Y_perm = np.random.permutation(Y_scaled)
                    perm_scores = cross_val_score(model, X_selected, Y_perm, cv=3, scoring='r2')
                    null_scores.append(np.mean(perm_scores))
                
                p_value = np.mean(np.array(null_scores) >= mean_r2)
                
                if p_value < self.alpha:
                    multi_results[model_name] = {
                        'r2': mean_r2,
                        'r2_std': std_r2,
                        'p_value': p_value,
                        'n_targets': len(biomarkers),
                        'selected_features': selected_features
                    }
                    
                    self.log_analysis(f"  ✅ SIGNIFICANT: {model_name} - p = {p_value:.4f}", "SUCCESS")
        
        return multi_results
    
    def epidemiological_validation(self, system_data, significant_results, climate_vars):
        """Validate ML findings with traditional epidemiological methods"""
        self.log_analysis(f"\n🔍 Epidemiological Validation")
        self.log_analysis("-" * 35)
        
        validation_results = {}
        
        for comp_name, ml_result in significant_results.items():
            if 'top_feature' not in ml_result:
                continue
                
            top_climate_var = ml_result['top_feature']
            composite_values = system_data[f'composite_{comp_name}']
            climate_values = system_data[top_climate_var]
            
            # Traditional correlation
            corr, p_corr = pearsonr(climate_values, composite_values)
            
            # Extreme temperature analysis
            temp_90 = climate_values.quantile(0.90)
            temp_10 = climate_values.quantile(0.10)
            
            hot_extreme = composite_values[climate_values >= temp_90]
            cold_extreme = composite_values[climate_values <= temp_10]
            moderate = composite_values[(climate_values > temp_10) & (climate_values < temp_90)]
            
            hot_effect = None
            cold_effect = None
            
            if len(hot_extreme) >= 20 and len(moderate) >= 50:
                t_stat, p_hot = stats.ttest_ind(hot_extreme, moderate)
                hot_effect = {
                    'effect_size': (hot_extreme.mean() - moderate.mean()) / moderate.std(),
                    'p_value': p_hot
                }
            
            if len(cold_extreme) >= 20 and len(moderate) >= 50:
                t_stat, p_cold = stats.ttest_ind(cold_extreme, moderate)
                cold_effect = {
                    'effect_size': (cold_extreme.mean() - moderate.mean()) / moderate.std(),
                    'p_value': p_cold
                }
            
            validation_results[comp_name] = {
                'ml_r2': ml_result['r2'],
                'ml_p_value': ml_result['p_value'],
                'correlation': corr,
                'correlation_p': p_corr,
                'hot_extreme_effect': hot_effect,
                'cold_extreme_effect': cold_effect,
                'top_climate_predictor': top_climate_var
            }
            
            self.log_analysis(f"  {comp_name}:")
            self.log_analysis(f"    ML R² = {ml_result['r2']:.4f}, Traditional r = {corr:.4f}")
            self.log_analysis(f"    Consistency: {'High' if abs(corr) > 0.8 * np.sqrt(ml_result['r2']) else 'Moderate'}")
        
        return validation_results
    
    def run_rigorous_ml_analysis(self):
        """Execute comprehensive rigorous ML multi-biomarker analysis"""
        self.log_analysis("EXECUTING RIGOROUS ML MULTI-BIOMARKER ANALYSIS")
        self.log_analysis("=" * 55)
        
        # Load data
        df, validated_systems, climate_vars = self.load_and_prepare_multivariate_data()
        
        all_results = {}
        
        for system_name, system_info in validated_systems.items():
            self.log_analysis(f"\n🏥 ANALYZING: {system_name.upper()} System")
            self.log_analysis(f"Rationale: {system_info['rationale']}")
            self.log_analysis(f"Biomarkers: {system_info['biomarkers']}")
            self.log_analysis(f"Sample size: {system_info['n_samples']:,}")
            
            # Create composite targets
            system_data, composites = self.create_composite_biomarker_targets(df, system_info)
            
            if system_data is None:
                continue
            
            # ML composite analysis
            composite_results = self.rigorous_ml_composite_analysis(
                system_data, composites, climate_vars, system_name
            )
            
            # Multi-task learning
            multi_task_results = self.multi_task_learning_analysis(
                system_data, system_info['biomarkers'], climate_vars, system_name
            )
            
            # Epidemiological validation
            if composite_results:
                validation_results = self.epidemiological_validation(
                    system_data, composite_results, climate_vars
                )
            else:
                validation_results = {}
            
            all_results[system_name] = {
                'system_info': system_info,
                'composite_results': composite_results,
                'multi_task_results': multi_task_results,
                'validation_results': validation_results
            }
        
        # Final summary
        self.log_analysis("\n" + "=" * 60)
        self.log_analysis("🎯 RIGOROUS ML MULTI-BIOMARKER ANALYSIS SUMMARY")
        self.log_analysis("=" * 60)
        
        total_significant = 0
        high_quality_findings = []
        
        for system_name, results in all_results.items():
            system_significant = len(results['composite_results']) + len(results['multi_task_results'])
            total_significant += system_significant
            
            if system_significant > 0:
                self.log_analysis(f"\n🏥 {system_name.upper()}: {system_significant} significant relationships")
                
                # Composite findings
                for comp_name, comp_result in results['composite_results'].items():
                    self.log_analysis(f"  • Composite {comp_name}: R² = {comp_result['r2']:.4f}")
                    
                    # Check validation consistency
                    if comp_name in results['validation_results']:
                        val_result = results['validation_results'][comp_name]
                        consistency = abs(val_result['correlation']) > 0.7 * np.sqrt(comp_result['r2'])
                        if consistency and comp_result['r2'] > 0.05:
                            high_quality_findings.append({
                                'system': system_name,
                                'type': f'composite_{comp_name}',
                                'ml_r2': comp_result['r2'],
                                'correlation': val_result['correlation'],
                                'predictor': comp_result['top_feature']
                            })
                
                # Multi-task findings
                for mt_name, mt_result in results['multi_task_results'].items():
                    self.log_analysis(f"  • Multi-task {mt_name}: R² = {mt_result['r2']:.4f}")
        
        if high_quality_findings:
            self.log_analysis(f"\n🏆 HIGH-QUALITY VALIDATED FINDINGS: {len(high_quality_findings)}")
            for i, finding in enumerate(high_quality_findings, 1):
                self.log_analysis(f"  {i}. {finding['system']} - {finding['type']}")
                self.log_analysis(f"     ML R² = {finding['ml_r2']:.4f}, Validation r = {finding['correlation']:.4f}")
                self.log_analysis(f"     Predictor: {finding['predictor']}")
        
        if total_significant == 0:
            self.log_analysis("\n❌ No significant ML relationships detected")
            self.log_analysis("This suggests multi-biomarker approaches may not add")
            self.log_analysis("substantial value beyond single biomarker analysis.")
        else:
            self.log_analysis(f"\n✅ TOTAL SIGNIFICANT RELATIONSHIPS: {total_significant}")
            self.log_analysis(f"High-quality validated findings: {len(high_quality_findings)}")
        
        return all_results

def main():
    """Execute rigorous ML multi-biomarker analysis"""
    analyzer = RigorousMLMultivariateAnalysis()
    results = analyzer.run_rigorous_ml_analysis()
    return results

if __name__ == "__main__":
    main()