#!/usr/bin/env python3
"""
Climate-Health ML Interpretability Framework
===========================================

This module provides comprehensive interpretability analysis for climate-health
machine learning models, focusing on meaningful climate insights.

Key Components:
1. SHAP analysis for feature importance and interaction effects
2. Lag pattern analysis to identify critical exposure windows
3. Geographic and seasonal effect decomposition
4. Climate threshold analysis
5. Publication-ready visualizations

Authors: Climate-Health Data Science Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
from scipy import stats
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

class ClimateHealthInterpretability:
    """
    Comprehensive interpretability analysis for climate-health ML models
    """
    
    def __init__(self, results_dir="rigorous_results", figures_dir="rigorous_figures"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set publication-ready plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Define climate variable groups for analysis
        self.climate_groups = {
            'temperature': ['temp', 'tas', 'temperature'],
            'heat_stress': ['heat_index', 'utci', 'wbgt', 'apparent_temp'],
            'heat_exposure': ['heat_stress', 'heat_exposure', 'extreme_heat', 'cooling_degree'],
            'wind': ['wind', 'ws'],
            'temporal': ['lag', 'mean', 'max', 'min', 'change', 'variability']
        }
        
        # Define lag periods for analysis
        self.lag_periods = [0, 1, 2, 3, 5, 7, 10, 14, 21]
    
    def load_model_and_data(self, model_path, data_path="FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv"):
        """
        Load trained model and associated data for interpretability analysis
        """
        print(f"📥 Loading model from {model_path}")
        
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data.get('scaler', None)
        self.feature_names = model_data['feature_names']
        self.biomarker = model_data['biomarker']
        self.model_type = model_data['model_type']
        
        # Load data
        print(f"📥 Loading data from {data_path}")
        self.df = pd.read_csv(data_path, low_memory=False)
        
        print(f"✅ Loaded {self.model_type} model for {self.biomarker}")
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"✅ Data: {len(self.df):,} samples")
        
        return self
    
    def prepare_interpretation_data(self, sample_size=1000):
        """
        Prepare data for SHAP analysis with proper sampling
        """
        print(f"🔄 Preparing interpretation data (sample size: {sample_size})")
        
        # Get clean data for the biomarker
        clean_data = self.df.dropna(subset=[self.biomarker]).copy()
        
        # Get available features
        available_features = [f for f in self.feature_names if f in clean_data.columns]
        
        # Prepare feature matrix
        X = clean_data[available_features].copy()
        y = clean_data[self.biomarker]
        
        # Handle missing values (same as in main pipeline)
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                mode_val = X[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                X[col] = X[col].fillna(fill_val)
        
        # Sample for efficient SHAP analysis
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X.iloc[sample_indices].copy()
            y_sample = y.iloc[sample_indices].copy()
        else:
            X_sample = X.copy()
            y_sample = y.copy()
        
        # Apply scaling if needed
        if self.scaler is not None:
            X_sample_scaled = pd.DataFrame(
                self.scaler.transform(X_sample),
                columns=X_sample.columns,
                index=X_sample.index
            )
            self.X_interpretation = X_sample_scaled
        else:
            self.X_interpretation = X_sample
        
        self.y_interpretation = y_sample
        
        print(f"✅ Interpretation data prepared: {len(self.X_interpretation)} samples")
        return self
    
    def compute_shap_values(self, explainer_type='auto'):
        """
        Compute SHAP values for feature importance analysis
        """
        print(f"🔄 Computing SHAP values using {explainer_type} explainer")
        
        # Select appropriate SHAP explainer
        if explainer_type == 'auto':
            if hasattr(self.model, 'predict_proba'):
                explainer_type = 'tree'
            elif hasattr(self.model, 'coef_'):
                explainer_type = 'linear'
            else:
                explainer_type = 'permutation'
        
        try:
            if explainer_type == 'tree' and hasattr(self.model, 'feature_importances_'):
                # Tree-based models (RandomForest, XGBoost)
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X_interpretation)
                
            elif explainer_type == 'linear' and hasattr(self.model, 'coef_'):
                # Linear models
                self.explainer = shap.LinearExplainer(self.model, self.X_interpretation)
                self.shap_values = self.explainer.shap_values(self.X_interpretation)
                
            else:
                # Fallback to permutation explainer
                self.explainer = shap.PermutationExplainer(
                    self.model.predict, 
                    self.X_interpretation.sample(n=min(100, len(self.X_interpretation)))
                )
                self.shap_values = self.explainer.shap_values(
                    self.X_interpretation.sample(n=min(500, len(self.X_interpretation)))
                )
            
            print(f"✅ SHAP values computed: {self.shap_values.shape}")
            
        except Exception as e:
            print(f"⚠️  SHAP computation failed: {e}")
            print("   Using feature importance as fallback")
            self.shap_values = None
        
        return self
    
    def analyze_feature_importance(self, top_n=20):
        """
        Analyze and visualize feature importance
        """
        print(f"📊 Analyzing feature importance (top {top_n})")
        
        importance_data = {}
        
        if self.shap_values is not None:
            # SHAP-based importance
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            importance_data['shap'] = dict(zip(self.X_interpretation.columns, shap_importance))
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based feature importance
            tree_importance = self.model.feature_importances_
            importance_data['tree'] = dict(zip(self.X_interpretation.columns, tree_importance))
        
        if hasattr(self.model, 'coef_'):
            # Linear model coefficients
            coef_importance = np.abs(self.model.coef_)
            importance_data['linear'] = dict(zip(self.X_interpretation.columns, coef_importance))
        
        # Create comprehensive importance plot
        fig, axes = plt.subplots(1, len(importance_data), figsize=(6 * len(importance_data), 8))
        if len(importance_data) == 1:
            axes = [axes]
        
        for i, (method, importances) in enumerate(importance_data.items()):
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            features, values = zip(*sorted_features)
            
            axes[i].barh(range(len(features)), values)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels([self._format_feature_name(f) for f in features])
            axes[i].set_xlabel(f'{method.upper()} Importance')
            axes[i].set_title(f'Top {top_n} Features - {method.upper()}')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        safe_biomarker = self._safe_filename(self.biomarker)
        importance_file = self.figures_dir / f"feature_importance_{safe_biomarker}.png"
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance plot saved: {importance_file}")
        
        # Store importance data
        self.importance_data = importance_data
        return importance_data
    
    def analyze_lag_patterns(self):
        """
        Analyze temporal lag patterns to identify critical exposure windows
        """
        print("📊 Analyzing temporal lag patterns")
        
        if self.shap_values is None:
            print("⚠️  SHAP values not available - using feature importance")
            if hasattr(self.model, 'feature_importances_'):
                importances = dict(zip(self.X_interpretation.columns, self.model.feature_importances_))
            else:
                print("❌ No importance measure available")
                return None
        else:
            importances = dict(zip(self.X_interpretation.columns, np.abs(self.shap_values).mean(axis=0)))
        
        # Group features by lag period
        lag_analysis = {}
        
        for lag in self.lag_periods:
            lag_features = [f for f in importances.keys() if f'lag{lag}' in f or f'_lag_{lag}' in f]
            if lag_features:
                lag_importance = sum(importances[f] for f in lag_features)
                lag_analysis[lag] = {
                    'total_importance': lag_importance,
                    'n_features': len(lag_features),
                    'avg_importance': lag_importance / len(lag_features),
                    'features': lag_features
                }
        
        if not lag_analysis:
            print("⚠️  No lag features found")
            return None
        
        # Create lag pattern visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        lags = list(lag_analysis.keys())
        total_importances = [lag_analysis[lag]['total_importance'] for lag in lags]
        avg_importances = [lag_analysis[lag]['avg_importance'] for lag in lags]
        
        # Total importance by lag
        ax1.plot(lags, total_importances, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Lag Period (days)')
        ax1.set_ylabel('Total Feature Importance')
        ax1.set_title('Climate Impact by Lag Period')
        ax1.grid(True, alpha=0.3)
        
        # Average importance by lag
        ax2.plot(lags, avg_importances, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Lag Period (days)')
        ax2.set_ylabel('Average Feature Importance')
        ax2.set_title('Average Climate Effect by Lag Period')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_biomarker = self._safe_filename(self.biomarker)
        lag_file = self.figures_dir / f"lag_patterns_{safe_biomarker}.png"
        plt.savefig(lag_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Lag pattern analysis saved: {lag_file}")
        
        # Identify most important lag periods
        most_important_lags = sorted(lag_analysis.items(), 
                                   key=lambda x: x[1]['total_importance'], 
                                   reverse=True)[:3]
        
        print("🔍 Most important lag periods:")
        for lag, info in most_important_lags:
            print(f"   Day {lag}: {info['total_importance']:.4f} total importance")
        
        self.lag_analysis = lag_analysis
        return lag_analysis
    
    def analyze_climate_variable_groups(self):
        """
        Analyze importance by climate variable groups
        """
        print("📊 Analyzing climate variable groups")
        
        if self.shap_values is None:
            if hasattr(self.model, 'feature_importances_'):
                importances = dict(zip(self.X_interpretation.columns, self.model.feature_importances_))
            else:
                print("❌ No importance measure available")
                return None
        else:
            importances = dict(zip(self.X_interpretation.columns, np.abs(self.shap_values).mean(axis=0)))
        
        # Group features by climate variable type
        group_analysis = {}
        
        for group_name, patterns in self.climate_groups.items():
            group_features = []
            for feature in importances.keys():
                if any(pattern in feature.lower() for pattern in patterns):
                    group_features.append(feature)
            
            if group_features:
                group_importance = sum(importances[f] for f in group_features)
                group_analysis[group_name] = {
                    'total_importance': group_importance,
                    'n_features': len(group_features),
                    'avg_importance': group_importance / len(group_features),
                    'features': group_features[:5]  # Top 5 for display
                }
        
        # Create group visualization
        if group_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            groups = list(group_analysis.keys())
            total_importances = [group_analysis[g]['total_importance'] for g in groups]
            avg_importances = [group_analysis[g]['avg_importance'] for g in groups]
            
            # Total importance by group
            ax1.bar(groups, total_importances, alpha=0.7)
            ax1.set_xlabel('Climate Variable Group')
            ax1.set_ylabel('Total Feature Importance')
            ax1.set_title('Climate Impact by Variable Type')
            ax1.tick_params(axis='x', rotation=45)
            
            # Average importance by group
            ax2.bar(groups, avg_importances, alpha=0.7, color='orange')
            ax2.set_xlabel('Climate Variable Group')
            ax2.set_ylabel('Average Feature Importance')
            ax2.set_title('Average Climate Effect by Variable Type')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            safe_biomarker = self._safe_filename(self.biomarker)
            groups_file = self.figures_dir / f"climate_groups_{safe_biomarker}.png"
            plt.savefig(groups_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Climate groups analysis saved: {groups_file}")
            
            # Print summary
            most_important_groups = sorted(group_analysis.items(), 
                                         key=lambda x: x[1]['total_importance'], 
                                         reverse=True)
            
            print("🔍 Most important climate variable groups:")
            for group, info in most_important_groups:
                print(f"   {group}: {info['total_importance']:.4f} total importance ({info['n_features']} features)")
        
        self.group_analysis = group_analysis
        return group_analysis
    
    def create_shap_summary_plots(self):
        """
        Create comprehensive SHAP summary visualizations
        """
        if self.shap_values is None:
            print("⚠️  SHAP values not available - skipping SHAP plots")
            return None
        
        print("📊 Creating SHAP summary plots")
        
        # SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_interpretation,
            feature_names=[self._format_feature_name(f) for f in self.X_interpretation.columns],
            show=False,
            max_display=20
        )
        
        safe_biomarker = self._safe_filename(self.biomarker)
        shap_summary_file = self.figures_dir / f"shap_summary_{safe_biomarker}.png"
        plt.savefig(shap_summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP waterfall plot for a representative sample
        plt.figure(figsize=(10, 8))
        sample_idx = len(self.X_interpretation) // 2  # Middle sample
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=self.X_interpretation.iloc[sample_idx].values,
                feature_names=[self._format_feature_name(f) for f in self.X_interpretation.columns]
            ),
            show=False,
            max_display=15
        )
        
        shap_waterfall_file = self.figures_dir / f"shap_waterfall_{safe_biomarker}.png"
        plt.savefig(shap_waterfall_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SHAP summary plots saved: {shap_summary_file}, {shap_waterfall_file}")
        
        return {
            'summary_plot': str(shap_summary_file),
            'waterfall_plot': str(shap_waterfall_file)
        }
    
    def generate_interpretation_report(self):
        """
        Generate comprehensive interpretation report
        """
        print("📝 Generating interpretation report")
        
        report = {
            'biomarker': self.biomarker,
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'n_samples': len(self.X_interpretation),
            'timestamp': datetime.now().isoformat(),
            'analysis_components': {
                'feature_importance': hasattr(self, 'importance_data'),
                'lag_patterns': hasattr(self, 'lag_analysis'),
                'climate_groups': hasattr(self, 'group_analysis'),
                'shap_analysis': self.shap_values is not None
            }
        }
        
        # Add feature importance summary
        if hasattr(self, 'importance_data'):
            report['top_features'] = {}
            for method, importances in self.importance_data.items():
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                report['top_features'][method] = [
                    {'feature': feat, 'importance': float(imp)} for feat, imp in top_features
                ]
        
        # Add lag pattern summary
        if hasattr(self, 'lag_analysis'):
            most_important_lags = sorted(self.lag_analysis.items(), 
                                       key=lambda x: x[1]['total_importance'], 
                                       reverse=True)[:5]
            report['critical_lag_periods'] = [
                {
                    'lag_days': lag,
                    'total_importance': float(info['total_importance']),
                    'n_features': info['n_features']
                }
                for lag, info in most_important_lags
            ]
        
        # Add climate group summary
        if hasattr(self, 'group_analysis'):
            report['climate_variable_groups'] = {
                group: {
                    'total_importance': float(info['total_importance']),
                    'n_features': info['n_features'],
                    'avg_importance': float(info['avg_importance'])
                }
                for group, info in self.group_analysis.items()
            }
        
        # Save report
        safe_biomarker = self._safe_filename(self.biomarker)
        report_file = self.results_dir / f"interpretation_report_{safe_biomarker}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Interpretation report saved: {report_file}")
        
        return report
    
    def _format_feature_name(self, feature_name):
        """Format feature names for better readability in plots"""
        # Truncate long names
        if len(feature_name) > 30:
            return feature_name[:27] + "..."
        return feature_name
    
    def _safe_filename(self, name):
        """Convert biomarker name to safe filename"""
        return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')
    
    def run_complete_interpretation(self, model_path, sample_size=1000):
        """
        Run complete interpretability analysis
        """
        print("="*80)
        print("🔍 CLIMATE-HEALTH ML INTERPRETABILITY ANALYSIS")
        print("="*80)
        
        # Load model and data
        self.load_model_and_data(model_path)
        
        # Prepare interpretation data
        self.prepare_interpretation_data(sample_size)
        
        # Compute SHAP values
        self.compute_shap_values()
        
        # Run all analyses
        self.analyze_feature_importance()
        self.analyze_lag_patterns()
        self.analyze_climate_variable_groups()
        self.create_shap_summary_plots()
        
        # Generate report
        report = self.generate_interpretation_report()
        
        print("="*80)
        print("✅ INTERPRETABILITY ANALYSIS COMPLETE")
        print("="*80)
        
        return report

def interpret_all_models(models_dir="rigorous_models", sample_size=1000):
    """
    Run interpretability analysis for all trained models
    """
    models_dir = Path(models_dir)
    model_files = list(models_dir.glob("rigorous_model_*.joblib"))
    
    print(f"🔍 Found {len(model_files)} models for interpretation")
    
    all_reports = {}
    
    for model_file in model_files:
        print(f"\n📊 Interpreting model: {model_file.name}")
        
        try:
            interpreter = ClimateHealthInterpretability()
            report = interpreter.run_complete_interpretation(str(model_file), sample_size)
            all_reports[str(model_file)] = report
            
        except Exception as e:
            print(f"❌ Failed to interpret {model_file.name}: {e}")
    
    # Save combined report
    combined_report_file = Path("rigorous_results") / f"all_interpretations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_report_file, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    
    print(f"\n✅ All interpretations saved: {combined_report_file}")
    
    return all_reports

if __name__ == "__main__":
    # Example usage
    interpret_all_models()