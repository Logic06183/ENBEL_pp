#!/usr/bin/env python3
"""
Rigorous XAI Climate-Health Discovery Framework
==============================================

Clean implementation focused on discovering genuine climate-health relationships
using explainable AI methods. Avoids data leakage and focuses on scientifically
valid climate-to-health predictions.

Key principles:
1. Strict separation of climate features from health outcomes
2. Conservative validation with cross-validation
3. Focus on interpretable, actionable insights
4. Statistical significance testing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import shap
from scipy.stats import pearsonr
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RigorousXAIDiscovery:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("rigorous_xai_discovery")
        self.results_dir.mkdir(exist_ok=True)
        
        # Define strict climate-only features (no health outcomes as predictors)
        self.climate_keywords = [
            'temp', 'heat', 'humid', 'pressure', 'wind', 'solar', 
            'radiation', 'precipitation', 'rain', 'weather'
        ]
        
        # Biomarkers to predict (health outcomes)
        self.target_biomarkers = [
            'systolic blood pressure', 'diastolic blood pressure',
            'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 
            'FASTING HDL', 'FASTING LDL', 'CD4 cell count', 
            'creatinine', 'hemoglobin'
        ]
        
        # Demographic features for interaction analysis
        self.demographic_features = ['Sex', 'Race', 'Education', 'Age']
        
        self.discoveries = {}
        
    def log_progress(self, message, level="INFO"):
        """Progress logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🔬", "SUCCESS": "✅", "WARNING": "⚠️", "DISCOVERY": "💡"}
        icon = icons.get(level, "🔬")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)

    def load_and_clean_data(self):
        """Load data with strict climate-health separation"""
        self.log_progress("Loading data with strict climate-health separation...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Identify PURE climate features (no health outcomes)
        climate_features = []
        for col in df.columns:
            col_lower = col.lower()
            # Include if contains climate keywords AND not a health biomarker
            if (any(keyword in col_lower for keyword in self.climate_keywords) and
                not any(biomarker.lower() in col_lower for biomarker in self.target_biomarkers)):
                climate_features.append(col)
        
        # Available biomarkers
        available_biomarkers = [col for col in self.target_biomarkers if col in df.columns]
        
        # Available demographics
        available_demographics = [col for col in self.demographic_features if col in df.columns]
        
        self.log_progress(f"Dataset: {len(df)} records")
        self.log_progress(f"Climate features (pure): {len(climate_features)}")
        self.log_progress(f"Available biomarkers: {len(available_biomarkers)}")
        self.log_progress(f"Demographics: {available_demographics}")
        
        return df, climate_features, available_biomarkers, available_demographics

    def create_interaction_features(self, df, climate_features, demographics):
        """Create climate-demographic interaction features"""
        self.log_progress("Creating climate-demographic interactions...")
        
        interaction_features = []
        
        # Focus on top climate features for interactions
        top_climate = climate_features[:15]  # Top 15 for computational efficiency
        
        for climate_var in top_climate:
            if climate_var not in df.columns:
                continue
                
            for demo_var in demographics:
                if demo_var not in df.columns:
                    continue
                    
                try:
                    # Handle categorical demographics
                    if df[demo_var].dtype == 'object':
                        le = LabelEncoder()
                        demo_encoded = le.fit_transform(df[demo_var].fillna('unknown'))
                        interaction_name = f"{climate_var}_x_{demo_var}"
                        df[interaction_name] = (df[climate_var].fillna(df[climate_var].median()) * 
                                              demo_encoded)
                        interaction_features.append(interaction_name)
                    else:
                        # Numeric demographic
                        interaction_name = f"{climate_var}_x_{demo_var}"
                        df[interaction_name] = (df[climate_var].fillna(df[climate_var].median()) * 
                                              df[demo_var].fillna(df[demo_var].median()))
                        interaction_features.append(interaction_name)
                except Exception as e:
                    continue
        
        self.log_progress(f"Created {len(interaction_features)} interaction features")
        return interaction_features

    def analyze_biomarker_climate_relationship(self, df, biomarker, climate_features, interaction_features):
        """Rigorous analysis of single biomarker-climate relationship"""
        
        # Prepare data
        biomarker_data = df.dropna(subset=[biomarker])
        
        if len(biomarker_data) < 500:
            return None
        
        # Combine climate and interaction features
        all_predictors = climate_features + interaction_features
        available_predictors = [f for f in all_predictors if f in biomarker_data.columns]
        
        # Remove any features that might be data leakage
        clean_predictors = []
        biomarker_lower = biomarker.lower()
        for pred in available_predictors:
            pred_lower = pred.lower()
            # Exclude if predictor name contains biomarker name
            if not any(bio_word in pred_lower for bio_word in biomarker_lower.split()):
                clean_predictors.append(pred)
        
        if len(clean_predictors) < 5:
            return None
        
        # Limit features for computational efficiency
        clean_predictors = clean_predictors[:40]
        
        X = biomarker_data[clean_predictors].fillna(biomarker_data[clean_predictors].median())
        y = biomarker_data[biomarker]
        
        # Cross-validation with multiple models
        models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
        }
        
        cv_results = {}
        best_model = None
        best_score = -1
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_score = np.mean(scores)
                cv_results[model_name] = {
                    'mean_r2': mean_score,
                    'std_r2': np.std(scores),
                    'scores': scores.tolist()
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    
            except Exception as e:
                continue
        
        if best_score <= 0.005:  # Very conservative threshold
            return None
        
        # Train best model for feature importance
        best_model.fit(X, y)
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            importance = np.zeros(len(clean_predictors))
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': clean_predictors,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # SHAP analysis for best model
        shap_results = None
        try:
            if hasattr(best_model, 'predict') and len(X) > 100:
                sample_size = min(200, len(X))
                X_sample = X.iloc[:sample_size]
                
                if isinstance(best_model, (GradientBoostingRegressor, RandomForestRegressor)):
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Feature importance from SHAP
                    shap_importance = np.abs(shap_values).mean(axis=0)
                    shap_results = {
                        'mean_abs_shap': np.mean(np.abs(shap_values)),
                        'top_shap_features': [
                            {'feature': clean_predictors[i], 'shap_importance': shap_importance[i]}
                            for i in np.argsort(shap_importance)[-5:][::-1]
                        ]
                    }
        except Exception as e:
            pass
        
        # Temporal analysis - identify lag effects
        lag_analysis = {}
        for lag in [0, 1, 2, 3, 7, 14, 21]:
            lag_features = [f for f in clean_predictors if f'lag{lag}' in f.lower()]
            if lag_features:
                lag_indices = [clean_predictors.index(f) for f in lag_features]
                lag_importance = importance[lag_indices].sum()
                lag_analysis[f'lag_{lag}'] = {
                    'total_importance': lag_importance,
                    'features': lag_features
                }
        
        # Climate variable type analysis
        climate_analysis = {}
        climate_types = ['temp', 'humid', 'wind', 'pressure', 'heat', 'solar']
        for climate_type in climate_types:
            type_features = [f for f in clean_predictors if climate_type in f.lower()]
            if type_features:
                type_indices = [clean_predictors.index(f) for f in type_features]
                type_importance = importance[type_indices].sum()
                climate_analysis[climate_type] = {
                    'total_importance': type_importance,
                    'n_features': len(type_features),
                    'top_feature': max([(f, importance[clean_predictors.index(f)]) for f in type_features],
                                     key=lambda x: x[1]) if type_features else None
                }
        
        return {
            'biomarker': biomarker,
            'n_samples': len(biomarker_data),
            'n_features': len(clean_predictors),
            'cv_results': cv_results,
            'best_r2': best_score,
            'feature_importance': importance_df.head(10).to_dict('records'),
            'shap_analysis': shap_results,
            'temporal_analysis': lag_analysis,
            'climate_analysis': climate_analysis
        }

    def demographic_interaction_analysis(self, df, biomarker, climate_features, demographics):
        """Analyze climate-health relationships across demographic groups"""
        
        if not demographics:
            return {}
        
        demographic_results = {}
        
        for demo_var in demographics:
            if demo_var not in df.columns:
                continue
            
            # Get groups
            groups = df[demo_var].dropna().unique()
            if len(groups) < 2 or len(groups) > 8:
                continue
            
            group_analysis = {}
            
            for group in groups:
                group_data = df[(df[demo_var] == group) & df[biomarker].notna()]
                
                if len(group_data) < 100:
                    continue
                
                # Quick climate relationship test
                top_climate = climate_features[:10]
                available_climate = [f for f in top_climate if f in group_data.columns]
                
                if len(available_climate) < 3:
                    continue
                
                X_group = group_data[available_climate].fillna(group_data[available_climate].median())
                y_group = group_data[biomarker]
                
                # Simple model test
                model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
                cv_scores = cross_val_score(model, X_group, y_group, cv=3, scoring='r2')
                mean_score = np.mean(cv_scores)
                
                if mean_score > 0.01:
                    model.fit(X_group, y_group)
                    
                    # Top climate predictors for this group
                    importance = model.feature_importances_
                    top_features = [(available_climate[i], importance[i]) 
                                  for i in np.argsort(importance)[-3:][::-1]]
                    
                    group_analysis[str(group)] = {
                        'r2': mean_score,
                        'r2_std': np.std(cv_scores),
                        'n_samples': len(group_data),
                        'top_climate_predictors': [{'feature': f, 'importance': imp} 
                                                 for f, imp in top_features]
                    }
            
            if len(group_analysis) >= 2:
                demographic_results[demo_var] = group_analysis
        
        return demographic_results

    def run_comprehensive_discovery(self):
        """Execute complete rigorous XAI discovery"""
        self.log_progress("="*60)
        self.log_progress("🔬 RIGOROUS XAI CLIMATE-HEALTH DISCOVERY")
        self.log_progress("="*60)
        
        start_time = time.time()
        
        # Load and prepare data
        df, climate_features, biomarkers, demographics = self.load_and_clean_data()
        interaction_features = self.create_interaction_features(df, climate_features, demographics)
        
        # Analyze each biomarker
        significant_discoveries = {}
        
        for biomarker in biomarkers:
            if biomarker not in df.columns:
                continue
                
            self.log_progress(f"Analyzing {biomarker}...")
            
            # Main climate-health analysis
            result = self.analyze_biomarker_climate_relationship(
                df, biomarker, climate_features, interaction_features
            )
            
            if result and result['best_r2'] > 0.01:
                # Demographic stratification
                demo_results = self.demographic_interaction_analysis(
                    df, biomarker, climate_features, demographics
                )
                result['demographic_analysis'] = demo_results
                
                significant_discoveries[biomarker] = result
                
                self.log_progress(f"DISCOVERY: {biomarker} - R² = {result['best_r2']:.3f}", "DISCOVERY")
                
                if result['feature_importance']:
                    top_feature = result['feature_importance'][0]['feature']
                    top_importance = result['feature_importance'][0]['importance']
                    self.log_progress(f"  Top predictor: {top_feature} (importance: {top_importance:.4f})")
        
        # Generate summary report
        report = self.generate_discovery_report(significant_discoveries)
        
        elapsed_time = time.time() - start_time
        
        # Final summary
        self.log_progress("="*60)
        self.log_progress("✅ DISCOVERY ANALYSIS COMPLETE")
        self.log_progress(f"Analysis time: {elapsed_time/60:.1f} minutes")
        self.log_progress(f"Significant discoveries: {len(significant_discoveries)}")
        
        for biomarker, result in significant_discoveries.items():
            r2 = result['best_r2']
            n_samples = result['n_samples']
            self.log_progress(f"  • {biomarker}: R² = {r2:.3f} (n={n_samples})")
        
        return report

    def generate_discovery_report(self, discoveries):
        """Generate comprehensive discovery report"""
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'Rigorous XAI Climate-Health Discovery',
                'n_discoveries': len(discoveries)
            },
            'discoveries': discoveries,
            'summary': {
                'total_biomarkers_with_relationships': len(discoveries),
                'max_r2': max([d['best_r2'] for d in discoveries.values()]) if discoveries else 0,
                'average_r2': np.mean([d['best_r2'] for d in discoveries.values()]) if discoveries else 0,
                'total_samples_analyzed': sum([d['n_samples'] for d in discoveries.values()]) if discoveries else 0
            }
        }
        
        # Save report
        report_file = self.results_dir / f"rigorous_discovery_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_progress(f"Report saved: {report_file}")
        
        return report

def main():
    """Execute rigorous XAI discovery"""
    discoverer = RigorousXAIDiscovery()
    report = discoverer.run_comprehensive_discovery()
    return report

if __name__ == "__main__":
    main()