#!/usr/bin/env python3
"""
Rigorous Scientific Validation and Explainable AI Analysis
==========================================================

This script validates the discovered climate-health relationships using:
1. Statistical significance testing
2. Bootstrap confidence intervals
3. Cross-validation with multiple methods
4. SHAP (SHapley Additive exPlanations) for interpretability
5. Permutation importance testing
6. Temporal stability analysis
7. Spurious correlation detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import shap
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import json
import time
from datetime import datetime
import logging
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RigorousValidationXAI:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("rigorous_validation_xai")
        self.results_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"validation_progress_{self.timestamp}.log"
        
        # Findings to validate
        self.findings_to_validate = {
            'short_term': {
                'systolic_bp': {
                    'reported_r2': 0.221,
                    'biomarker': 'systolic blood pressure',
                    'time_scale': 'immediate',
                    'expected_range': (0.05, 0.25)
                }
            },
            'long_term': {
                'glucose': {
                    'reported_r2': 0.732,
                    'biomarker': 'FASTING GLUCOSE',
                    'time_scale': 'annual',
                    'expected_range': (0.02, 0.15)  # Literature expectation
                },
                'cholesterol': {
                    'reported_r2': 0.418,
                    'biomarker': 'FASTING TOTAL CHOLESTEROL',
                    'time_scale': 'annual',
                    'expected_range': (0.02, 0.12)
                },
                'hdl': {
                    'reported_r2': 0.372,
                    'biomarker': 'FASTING HDL',
                    'time_scale': 'annual',
                    'expected_range': (0.02, 0.12)
                }
            }
        }
        
        self.validation_results = {}
        
    def log_progress(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🔬", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "VALIDATION": "🔍"}
        icon = icons.get(level, "🔬")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def load_and_prepare_data(self):
        """Load data for validation"""
        self.log_progress("Loading data for rigorous validation...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Get climate features
        climate_features = []
        for col in df.columns:
            if any(term in col.lower() for term in ['temp', 'heat', 'humid', 'pressure', 'wind', 'solar']) and 'lag' in col.lower():
                climate_features.append(col)
        
        return df, climate_features

    def validate_short_term_relationships(self, df, climate_features):
        """Validate short-term (daily) climate-health relationships"""
        self.log_progress("Validating short-term relationships...", "VALIDATION")
        
        validation_results = {}
        
        for name, config in self.findings_to_validate['short_term'].items():
            biomarker = config['biomarker']
            reported_r2 = config['reported_r2']
            
            self.log_progress(f"Validating {name}: {biomarker} (reported R² = {reported_r2:.3f})")
            
            if biomarker not in df.columns:
                self.log_progress(f"Biomarker {biomarker} not found", "ERROR")
                continue
            
            # Prepare data
            biomarker_data = df.dropna(subset=[biomarker])
            
            # Get immediate lag features (lag 0-2)
            immediate_features = [f for f in climate_features if any(f'lag{i}' in f.lower() for i in [0, 1, 2])]
            available_features = [f for f in immediate_features[:30] if f in biomarker_data.columns]
            
            if len(available_features) < 10 or len(biomarker_data) < 1000:
                self.log_progress(f"Insufficient data for {biomarker}", "WARNING")
                continue
            
            X = biomarker_data[available_features].fillna(biomarker_data[available_features].median())
            y = biomarker_data[biomarker]
            
            # 1. CROSS-VALIDATION WITH MULTIPLE METHODS
            models = {
                'elastic_net': ElasticNet(random_state=42, max_iter=1000),
                'random_forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            }
            
            cv_results = {}
            for model_name, model in models.items():
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                cv_results[model_name] = {
                    'mean_r2': np.mean(scores),
                    'std_r2': np.std(scores),
                    'scores': scores.tolist()
                }
            
            # 2. BOOTSTRAP CONFIDENCE INTERVALS
            n_bootstrap = 100
            bootstrap_r2s = []
            
            for _ in range(n_bootstrap):
                X_boot, y_boot = resample(X, y, random_state=None)
                model_boot = ElasticNet(random_state=42, max_iter=1000)
                model_boot.fit(X_boot, y_boot)
                r2_boot = model_boot.score(X_boot, y_boot)
                bootstrap_r2s.append(r2_boot)
            
            ci_lower = np.percentile(bootstrap_r2s, 2.5)
            ci_upper = np.percentile(bootstrap_r2s, 97.5)
            
            # 3. STATISTICAL SIGNIFICANCE TEST
            # Test if R² is significantly different from 0
            # Using permutation test
            n_permutations = 100
            permuted_scores = []
            
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y)
                model_perm = ElasticNet(random_state=42, max_iter=1000)
                scores_perm = cross_val_score(model_perm, X, y_permuted, cv=3, scoring='r2')
                permuted_scores.append(np.mean(scores_perm))
            
            actual_r2 = cv_results['elastic_net']['mean_r2']
            p_value = np.mean(np.array(permuted_scores) >= actual_r2)
            
            # 4. CHECK FOR SPURIOUS CORRELATIONS
            # Test with random noise features
            n_noise_features = 10
            X_noise = np.random.randn(len(X), n_noise_features)
            noise_model = ElasticNet(random_state=42, max_iter=1000)
            noise_scores = cross_val_score(noise_model, X_noise, y, cv=5, scoring='r2')
            noise_r2 = np.mean(noise_scores)
            
            validation_results[name] = {
                'biomarker': biomarker,
                'n_samples': len(X),
                'n_features': len(available_features),
                'reported_r2': reported_r2,
                'validated_r2': actual_r2,
                'cv_results': cv_results,
                'bootstrap_ci': (ci_lower, ci_upper),
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'noise_baseline_r2': noise_r2,
                'validation_status': self._determine_validation_status(reported_r2, actual_r2, p_value)
            }
            
            self.log_progress(f"Validation: R² = {actual_r2:.3f} (CI: {ci_lower:.3f}-{ci_upper:.3f}), p = {p_value:.4f}")
        
        return validation_results

    def validate_long_term_relationships(self, df, climate_features):
        """Validate long-term (annual) climate-health relationships"""
        self.log_progress("Validating long-term relationships...", "VALIDATION")
        
        validation_results = {}
        
        for name, config in self.findings_to_validate['long_term'].items():
            biomarker = config['biomarker']
            reported_r2 = config['reported_r2']
            
            self.log_progress(f"Validating {name}: {biomarker} (reported R² = {reported_r2:.3f})")
            
            if biomarker not in df.columns:
                self.log_progress(f"Biomarker {biomarker} not found", "ERROR")
                continue
            
            # Check if we have year column for annual aggregation
            if 'year' not in df.columns:
                self.log_progress("No year column for annual aggregation", "ERROR")
                continue
            
            # Prepare annual aggregated data
            biomarker_data = df.dropna(subset=[biomarker])
            
            # Get temperature features
            temp_features = [f for f in climate_features if 'temp' in f.lower() and 'lag0' in f.lower()][:5]
            available_temp = [f for f in temp_features if f in biomarker_data.columns]
            
            if len(available_temp) < 2:
                self.log_progress(f"Insufficient temperature features for {biomarker}", "WARNING")
                continue
            
            # Annual aggregation
            try:
                annual_climate = biomarker_data.groupby('year')[available_temp].mean()
                annual_biomarker = biomarker_data.groupby('year')[biomarker].mean()
                annual_counts = biomarker_data.groupby('year').size()
                
                # Merge and filter for years with sufficient data
                annual_data = annual_climate.join(annual_biomarker).join(annual_counts.rename('n_samples'))
                annual_data = annual_data[annual_data['n_samples'] >= 50]  # Minimum samples per year
                
                if len(annual_data) < 5:  # Need at least 5 years
                    self.log_progress(f"Insufficient annual data for {biomarker}: {len(annual_data)} years", "WARNING")
                    continue
                
                X_annual = annual_data[available_temp]
                y_annual = annual_data[biomarker]
                
                # 1. VALIDATE WITH MULTIPLE METHODS
                models = {
                    'linear': LinearRegression(),
                    'ridge': Ridge(alpha=1.0),
                    'elastic_net': ElasticNet(random_state=42)
                }
                
                fit_results = {}
                for model_name, model in models.items():
                    model.fit(X_annual, y_annual)
                    r2 = model.score(X_annual, y_annual)
                    predictions = model.predict(X_annual)
                    rmse = np.sqrt(mean_squared_error(y_annual, predictions))
                    
                    fit_results[model_name] = {
                        'r2': r2,
                        'rmse': rmse,
                        'n_years': len(X_annual)
                    }
                
                # 2. LEAVE-ONE-OUT CROSS-VALIDATION (for small sample)
                loo_scores = []
                for i in range(len(X_annual)):
                    X_train = X_annual.drop(X_annual.index[i])
                    y_train = y_annual.drop(y_annual.index[i])
                    X_test = X_annual.iloc[[i]]
                    y_test = y_annual.iloc[[i]]
                    
                    model_loo = ElasticNet(random_state=42)
                    model_loo.fit(X_train, y_train)
                    pred = model_loo.predict(X_test)
                    
                    # Calculate R² for single prediction
                    ss_res = (y_test.values[0] - pred[0]) ** 2
                    ss_tot = (y_test.values[0] - y_train.mean()) ** 2
                    r2_single = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    loo_scores.append(r2_single)
                
                loo_mean_r2 = np.mean(loo_scores)
                loo_std_r2 = np.std(loo_scores)
                
                # 3. CORRELATION ANALYSIS
                # Check if it's just a temporal trend
                years = np.array(range(len(y_annual)))
                trend_corr, trend_p = pearsonr(years, y_annual.values)
                climate_corr, climate_p = pearsonr(X_annual.mean(axis=1).values, y_annual.values)
                
                # 4. SPURIOUS CORRELATION CHECK
                # Compare with random walk
                random_walk = np.cumsum(np.random.randn(len(y_annual)))
                random_model = LinearRegression()
                random_model.fit(random_walk.reshape(-1, 1), y_annual)
                random_r2 = random_model.score(random_walk.reshape(-1, 1), y_annual)
                
                validation_results[name] = {
                    'biomarker': biomarker,
                    'n_years': len(X_annual),
                    'n_features': len(available_temp),
                    'samples_per_year': annual_data['n_samples'].mean(),
                    'reported_r2': reported_r2,
                    'validated_r2': fit_results['elastic_net']['r2'],
                    'model_comparison': fit_results,
                    'loo_cv_r2': loo_mean_r2,
                    'loo_cv_std': loo_std_r2,
                    'temporal_trend_corr': trend_corr,
                    'temporal_trend_p': trend_p,
                    'climate_corr': climate_corr,
                    'climate_p': climate_p,
                    'random_baseline_r2': random_r2,
                    'validation_status': self._determine_annual_validation_status(
                        reported_r2, fit_results['elastic_net']['r2'], len(X_annual)
                    )
                }
                
                self.log_progress(f"Annual validation: R² = {fit_results['elastic_net']['r2']:.3f}, LOO-CV R² = {loo_mean_r2:.3f}")
                
                # WARNING if R² seems too high for the sample size
                if fit_results['elastic_net']['r2'] > 0.5 and len(X_annual) < 10:
                    self.log_progress(f"WARNING: Very high R² ({fit_results['elastic_net']['r2']:.3f}) with only {len(X_annual)} data points - likely overfitting!", "WARNING")
                
            except Exception as e:
                self.log_progress(f"Error validating {biomarker}: {e}", "ERROR")
                continue
        
        return validation_results

    def apply_explainable_ai(self, df, climate_features):
        """Apply SHAP and other XAI methods for interpretability"""
        self.log_progress("Applying Explainable AI methods...", "VALIDATION")
        
        xai_results = {}
        
        # Focus on validated relationships
        test_cases = [
            ('systolic_bp', 'systolic blood pressure', 'immediate'),
            ('glucose', 'FASTING GLUCOSE', 'annual')
        ]
        
        for name, biomarker, time_scale in test_cases:
            if biomarker not in df.columns:
                continue
            
            self.log_progress(f"XAI analysis for {biomarker} ({time_scale})")
            
            biomarker_data = df.dropna(subset=[biomarker])
            
            if time_scale == 'immediate':
                # Short-term analysis
                immediate_features = [f for f in climate_features if any(f'lag{i}' in f.lower() for i in [0, 1, 2])]
                available_features = [f for f in immediate_features[:20] if f in biomarker_data.columns]
                
                if len(available_features) < 10 or len(biomarker_data) < 1000:
                    continue
                
                X = biomarker_data[available_features].fillna(biomarker_data[available_features].median())
                y = biomarker_data[biomarker]
                
                # Train model for SHAP
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y)
                
                # SHAP Analysis
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X[:100])  # Sample for efficiency
                
                # Feature importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame({
                    'feature': available_features,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                # Interaction effects
                shap_interaction = None  # Would compute if needed
                
                xai_results[name] = {
                    'biomarker': biomarker,
                    'time_scale': time_scale,
                    'model_r2': model.score(X, y),
                    'top_features': importance_df.head(10).to_dict('records'),
                    'shap_summary': {
                        'mean_abs_shap': np.mean(np.abs(shap_values)),
                        'max_abs_shap': np.max(np.abs(shap_values))
                    }
                }
                
                self.log_progress(f"Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.4f})")
            
            elif time_scale == 'annual' and 'year' in df.columns:
                # Annual analysis - simpler due to small sample size
                temp_features = [f for f in climate_features if 'temp' in f.lower() and 'lag0' in f.lower()][:3]
                available_temp = [f for f in temp_features if f in biomarker_data.columns]
                
                if len(available_temp) < 2:
                    continue
                
                try:
                    annual_climate = biomarker_data.groupby('year')[available_temp].mean()
                    annual_biomarker = biomarker_data.groupby('year')[biomarker].mean()
                    
                    annual_data = annual_climate.join(annual_biomarker).dropna()
                    
                    if len(annual_data) >= 5:
                        X_annual = annual_data[available_temp]
                        y_annual = annual_data[biomarker]
                        
                        # Simple linear model for interpretability
                        model_annual = LinearRegression()
                        model_annual.fit(X_annual, y_annual)
                        
                        # Coefficients as feature importance
                        importance_df = pd.DataFrame({
                            'feature': available_temp,
                            'coefficient': model_annual.coef_,
                            'abs_coefficient': np.abs(model_annual.coef_)
                        }).sort_values('abs_coefficient', ascending=False)
                        
                        xai_results[f"{name}_annual"] = {
                            'biomarker': biomarker,
                            'time_scale': 'annual',
                            'n_years': len(annual_data),
                            'model_r2': model_annual.score(X_annual, y_annual),
                            'feature_coefficients': importance_df.to_dict('records'),
                            'intercept': model_annual.intercept_
                        }
                        
                except Exception as e:
                    self.log_progress(f"Error in annual XAI for {biomarker}: {e}", "ERROR")
        
        return xai_results

    def _determine_validation_status(self, reported_r2, validated_r2, p_value):
        """Determine validation status for short-term relationships"""
        if p_value > 0.05:
            return "NOT_SIGNIFICANT"
        
        discrepancy = abs(reported_r2 - validated_r2)
        relative_discrepancy = discrepancy / reported_r2 if reported_r2 > 0 else 1.0
        
        if relative_discrepancy < 0.2:
            return "VALIDATED"
        elif relative_discrepancy < 0.5:
            return "PARTIALLY_VALIDATED"
        else:
            return "NOT_VALIDATED"

    def _determine_annual_validation_status(self, reported_r2, validated_r2, n_years):
        """Determine validation status for annual relationships"""
        # With few data points, high R² is suspicious
        if n_years < 10:
            if validated_r2 > 0.5:
                return "OVERFITTING_LIKELY"
            elif validated_r2 > 0.3:
                return "CAUTIOUS_INTERPRETATION"
        
        discrepancy = abs(reported_r2 - validated_r2)
        
        if discrepancy < 0.1:
            return "VALIDATED"
        elif discrepancy < 0.3:
            return "PARTIALLY_VALIDATED"
        else:
            return "NOT_VALIDATED"

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        self.log_progress("Generating validation report...")
        
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'validation_methods': [
                    'Cross-validation with multiple algorithms',
                    'Bootstrap confidence intervals',
                    'Permutation significance testing',
                    'Spurious correlation detection',
                    'SHAP analysis for interpretability',
                    'Leave-one-out validation for small samples'
                ]
            },
            'validation_results': self.validation_results,
            'summary': self._generate_summary()
        }
        
        # Save report
        report_file = self.results_dir / f"validation_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_progress(f"Validation report saved to: {report_file}")
        
        return report

    def _generate_summary(self):
        """Generate summary of validation results"""
        summary = {
            'validated_relationships': [],
            'rejected_relationships': [],
            'requires_caution': []
        }
        
        for category, results in self.validation_results.items():
            if not results:
                continue
                
            for name, result in results.items():
                status = result.get('validation_status', 'UNKNOWN')
                relationship = {
                    'name': name,
                    'biomarker': result['biomarker'],
                    'reported_r2': result.get('reported_r2', 'N/A'),
                    'validated_r2': result.get('validated_r2', result.get('validated_r2', 'N/A')),
                    'status': status
                }
                
                if status in ['VALIDATED', 'PARTIALLY_VALIDATED']:
                    summary['validated_relationships'].append(relationship)
                elif status in ['OVERFITTING_LIKELY', 'CAUTIOUS_INTERPRETATION']:
                    summary['requires_caution'].append(relationship)
                else:
                    summary['rejected_relationships'].append(relationship)
        
        return summary

    def run_complete_validation(self):
        """Execute complete validation and XAI pipeline"""
        self.log_progress("="*80)
        self.log_progress("🔬 RIGOROUS SCIENTIFIC VALIDATION AND EXPLAINABLE AI ANALYSIS")
        self.log_progress("="*80)
        
        start_time = time.time()
        
        # Load data
        df, climate_features = self.load_and_prepare_data()
        
        # Validate short-term relationships
        self.log_progress("\n📊 VALIDATING SHORT-TERM RELATIONSHIPS")
        short_term_results = self.validate_short_term_relationships(df, climate_features)
        self.validation_results['short_term'] = short_term_results
        
        # Validate long-term relationships
        self.log_progress("\n📊 VALIDATING LONG-TERM RELATIONSHIPS")
        long_term_results = self.validate_long_term_relationships(df, climate_features)
        self.validation_results['long_term'] = long_term_results
        
        # Apply XAI methods
        self.log_progress("\n🤖 APPLYING EXPLAINABLE AI METHODS")
        xai_results = self.apply_explainable_ai(df, climate_features)
        self.validation_results['xai'] = xai_results
        
        # Generate report
        report = self.generate_validation_report()
        
        elapsed_time = time.time() - start_time
        
        # Summary
        self.log_progress("\n" + "="*80)
        self.log_progress("VALIDATION COMPLETE")
        self.log_progress("="*80)
        self.log_progress(f"Validation time: {elapsed_time/60:.1f} minutes")
        
        # Print summary
        summary = report['summary']
        self.log_progress(f"\n✅ Validated relationships: {len(summary['validated_relationships'])}")
        for rel in summary['validated_relationships']:
            self.log_progress(f"  - {rel['biomarker']}: R² = {rel['validated_r2']:.3f}")
        
        self.log_progress(f"\n⚠️  Requires caution: {len(summary['requires_caution'])}")
        for rel in summary['requires_caution']:
            self.log_progress(f"  - {rel['biomarker']}: {rel['status']}")
        
        self.log_progress(f"\n❌ Rejected relationships: {len(summary['rejected_relationships'])}")
        for rel in summary['rejected_relationships']:
            self.log_progress(f"  - {rel['biomarker']}: {rel['status']}")
        
        return report

def main():
    """Execute rigorous validation and XAI analysis"""
    validator = RigorousValidationXAI()
    report = validator.run_complete_validation()
    return report

if __name__ == "__main__":
    main()