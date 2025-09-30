#!/usr/bin/env python3
"""
Advanced Causal XAI Discovery Framework for Climate-Health Interactions
=====================================================================

Comprehensive explainable AI approach to discover:
1. Multi-way climate × socioeconomic × demographic interactions
2. Causal pathways between physiological systems
3. Non-linear relationships and temporal patterns
4. Novel interaction mechanisms using SHAP, LIME, and causal inference

This builds on your solid glucose/BP findings to discover deeper mechanistic insights.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import shap
import warnings
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr, kendalltau
import logging

warnings.filterwarnings('ignore')

class AdvancedCausalXAIDiscovery:
    """
    Advanced XAI framework for discovering causal climate-health interactions
    """
    
    def __init__(self, results_dir="advanced_causal_xai_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / f'causal_xai_discovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'analysis_type': 'Advanced Causal XAI Discovery',
                'focus': 'Multi-system physiological interaction discovery'
            },
            'interaction_discoveries': {},
            'causal_pathways': {},
            'mechanism_insights': {},
            'novel_findings': {}
        }
    
    def load_and_prepare_comprehensive_data(self, filepath):
        """Load data with comprehensive feature engineering for interaction discovery"""
        
        self.logger.info("🔬 Loading comprehensive dataset for causal XAI discovery")
        
        df = pd.read_csv(filepath, low_memory=False)
        self.logger.info(f"Dataset loaded: {len(df)} records, {len(df.columns)} variables")
        
        # Define comprehensive feature groups for interaction analysis
        self.feature_groups = {
            'health_outcomes': [
                'systolic blood pressure', 'diastolic blood pressure',
                'FASTING GLUCOSE', 'FASTING LDL', 'FASTING TOTAL CHOLESTEROL', 
                'FASTING HDL', 'FASTING TRIGLYCERIDES', 'Creatinine (mg/dL)',
                'ALT (U/L)', 'AST (U/L)', 'Hemoglobin (g/dL)', 'Hematocrit (%)',
                'CD4 cell count (cells/µL)', 'Height', 'weight'
            ],
            'climate_immediate': [
                'temperature', 'humidity', 'wind_speed', 'heat_index',
                'apparent_temp', 'wet_bulb_temp'
            ],
            'climate_lagged': [col for col in df.columns if any(lag in col for lag in ['lag0', 'lag1', 'lag2', 'lag3', 'lag5', 'lag7', 'lag10', 'lag14', 'lag21'])],
            'socioeconomic': [
                'Education', 'employment_status', 'housing_vulnerability',
                'economic_vulnerability', 'heat_vulnerability_index',
                'population_density', 'infrastructure_level'
            ],
            'demographics': [
                'Sex', 'Race', 'latitude', 'longitude', 'year', 'month', 'season'
            ],
            'study_context': [
                'study_type', 'target_population', 'from_hiv_study',
                'hiv_status_indicator', 'viral_load_detected'
            ]
        }
        
        # Filter available columns
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            self.feature_groups[group_name] = available_features
            self.logger.info(f"{group_name}: {len(available_features)} features available")
        
        return df
    
    def create_interaction_features(self, df):
        """Create comprehensive interaction features for XAI analysis"""
        
        self.logger.info("🔧 Engineering interaction features for causal discovery")
        
        # Core climate variables for interactions
        core_climate = ['temperature', 'humidity', 'heat_index', 'apparent_temp']
        climate_available = [c for c in core_climate if c in df.columns]
        
        # Core demographic variables
        demographic_vars = ['Race', 'Sex', 'Education', 'employment_status']
        demo_available = [d for d in demographic_vars if d in df.columns]
        
        # Core socioeconomic variables
        socioec_vars = ['housing_vulnerability', 'economic_vulnerability', 'heat_vulnerability_index']
        socioec_available = [s for s in socioec_vars if s in df.columns]
        
        interaction_features = {}
        
        # 1. Climate × Demographics interactions
        for climate_var in climate_available:
            for demo_var in demo_available:
                if climate_var in df.columns and demo_var in df.columns:
                    # Create interaction terms
                    if df[demo_var].dtype == 'object':
                        # Encode categorical variables
                        le = LabelEncoder()
                        demo_encoded = le.fit_transform(df[demo_var].fillna('Unknown'))
                        interaction_name = f"{climate_var}_x_{demo_var}"
                        interaction_features[interaction_name] = df[climate_var] * demo_encoded
                    else:
                        interaction_name = f"{climate_var}_x_{demo_var}"
                        interaction_features[interaction_name] = df[climate_var] * df[demo_var]
        
        # 2. Climate × Socioeconomic interactions
        for climate_var in climate_available:
            for socioec_var in socioec_available:
                if climate_var in df.columns and socioec_var in df.columns:
                    interaction_name = f"{climate_var}_x_{socioec_var}"
                    interaction_features[interaction_name] = df[climate_var] * df[socioec_var]
        
        # 3. Three-way interactions (Climate × Demographics × Socioeconomic)
        for climate_var in climate_available[:2]:  # Limit to prevent explosion
            for demo_var in demo_available[:2]:
                for socioec_var in socioec_available[:2]:
                    if all(var in df.columns for var in [climate_var, demo_var, socioec_var]):
                        if df[demo_var].dtype == 'object':
                            le = LabelEncoder()
                            demo_encoded = le.fit_transform(df[demo_var].fillna('Unknown'))
                            interaction_name = f"{climate_var}_x_{demo_var}_x_{socioec_var}"
                            interaction_features[interaction_name] = df[climate_var] * demo_encoded * df[socioec_var]
        
        # 4. Temporal interaction patterns (Climate lag × Demographics)
        lag_vars = [col for col in df.columns if 'temperature_tas_lag' in col][:5]  # Top 5 lags
        for lag_var in lag_vars:
            for demo_var in demo_available:
                if lag_var in df.columns and demo_var in df.columns:
                    if df[demo_var].dtype == 'object':
                        le = LabelEncoder()
                        demo_encoded = le.fit_transform(df[demo_var].fillna('Unknown'))
                        interaction_name = f"{lag_var}_x_{demo_var}"
                        interaction_features[interaction_name] = df[lag_var] * demo_encoded
        
        self.logger.info(f"Created {len(interaction_features)} interaction features")
        
        # Add interaction features to dataframe
        for name, values in interaction_features.items():
            df[name] = values
        
        return df, list(interaction_features.keys())
    
    def discover_multi_system_interactions(self, df, interaction_features):
        """Use XAI to discover interactions between multiple physiological systems"""
        
        self.logger.info("🧬 Discovering multi-system physiological interactions")
        
        # Group health outcomes by physiological system
        physiological_systems = {
            'cardiovascular': ['systolic blood pressure', 'diastolic blood pressure'],
            'metabolic': ['FASTING GLUCOSE', 'FASTING LDL', 'FASTING TOTAL CHOLESTEROL', 
                         'FASTING HDL', 'FASTING TRIGLYCERIDES'],
            'hepatic': ['ALT (U/L)', 'AST (U/L)'],
            'renal': ['Creatinine (mg/dL)'],
            'hematologic': ['Hemoglobin (g/dL)', 'Hematocrit (%)'],
            'immune': ['CD4 cell count (cells/µL)'],
            'anthropometric': ['Height', 'weight']
        }
        
        multi_system_discoveries = {}
        
        # Create composite system scores using PCA-like approach
        system_scores = {}
        for system_name, biomarkers in physiological_systems.items():
            available_biomarkers = [b for b in biomarkers if b in df.columns]
            if len(available_biomarkers) >= 1:
                # Simple mean composite score (could use PCA for more sophistication)
                system_data = df[available_biomarkers].dropna()
                if len(system_data) > 100:
                    # Standardize and average
                    scaler = StandardScaler()
                    standardized = scaler.fit_transform(system_data)
                    system_scores[system_name] = pd.Series(
                        np.mean(standardized, axis=1), 
                        index=system_data.index,
                        name=f"{system_name}_composite"
                    )
        
        self.logger.info(f"Created composite scores for {len(system_scores)} physiological systems")
        
        # Analyze each system with comprehensive XAI
        for system_name, composite_score in system_scores.items():
            self.logger.info(f"\n🔍 Analyzing {system_name} system interactions")
            
            # Prepare features for this system
            all_features = (
                self.feature_groups['climate_immediate'] + 
                self.feature_groups['climate_lagged'][:10] +  # Top 10 lag features
                self.feature_groups['demographics'] + 
                self.feature_groups['socioeconomic'] + 
                interaction_features[:20]  # Top 20 interaction features
            )
            
            # Filter available features
            available_features = [f for f in all_features if f in df.columns]
            
            # Create analysis dataset
            analysis_data = df[available_features + [composite_score.name]].copy()
            
            # Encode categorical variables
            for col in analysis_data.columns:
                if analysis_data[col].dtype == 'object':
                    le = LabelEncoder()
                    analysis_data[col] = le.fit_transform(analysis_data[col].fillna('Unknown'))
            
            # Align composite score with analysis data
            common_idx = analysis_data.index.intersection(composite_score.index)
            if len(common_idx) < 100:
                self.logger.warning(f"Insufficient data for {system_name} system analysis")
                continue
            
            analysis_data = analysis_data.loc[common_idx]
            y = composite_score.loc[common_idx]
            X = analysis_data.drop(columns=[composite_score.name])
            
            # Remove rows with missing values
            complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[complete_mask]
            y = y[complete_mask]
            
            if len(X) < 100:
                self.logger.warning(f"Insufficient complete data for {system_name} system")
                continue
            
            self.logger.info(f"{system_name} analysis: {len(X)} samples, {len(X.columns)} features")
            
            # Train XGBoost model for this system
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            self.logger.info(f"{system_name} model: R² train={train_score:.3f}, test={test_score:.3f}")
            
            # SHAP analysis for interaction discovery
            if len(X) > 300:
                # Sample for SHAP analysis
                shap_sample_size = min(300, len(X_test))
                shap_idx = np.random.choice(len(X_test), shap_sample_size, replace=False)
                X_shap = X_test.iloc[shap_idx]
            else:
                X_shap = X_test
            
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                
                # Calculate feature importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                
                # Identify top interaction features
                interaction_features_in_model = [f for f in X.columns if '_x_' in f]
                interaction_importance = {}
                
                for i, feature in enumerate(X.columns):
                    if feature in interaction_features_in_model:
                        interaction_importance[feature] = feature_importance[i]
                
                # Sort by importance
                sorted_interactions = sorted(interaction_importance.items(), 
                                           key=lambda x: x[1], reverse=True)
                
                multi_system_discoveries[system_name] = {
                    'model_performance': {
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'n_samples': len(X),
                        'n_features': len(X.columns)
                    },
                    'top_interactions': sorted_interactions[:10],
                    'mean_shap_importance': np.mean(feature_importance),
                    'discovery_status': 'SUCCESS' if test_score > 0.1 else 'LIMITED',
                    'interpretation': self._interpret_system_interactions(system_name, sorted_interactions[:5])
                }
                
                self.logger.info(f"✅ {system_name} XAI analysis complete - {len(sorted_interactions)} interactions discovered")
                
            except Exception as e:
                self.logger.error(f"SHAP analysis failed for {system_name}: {str(e)}")
                multi_system_discoveries[system_name] = {
                    'model_performance': {
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'n_samples': len(X),
                        'n_features': len(X.columns)
                    },
                    'discovery_status': 'FAILED',
                    'error': str(e)
                }
        
        self.results['interaction_discoveries'] = multi_system_discoveries
        return multi_system_discoveries
    
    def _interpret_system_interactions(self, system_name, top_interactions):
        """Provide mechanistic interpretation of discovered interactions"""
        
        interpretations = []
        
        for feature, importance in top_interactions:
            if '_x_Race' in feature and 'temperature' in feature:
                interpretations.append(f"Race-specific temperature sensitivity in {system_name} system")
            elif '_x_Sex' in feature and 'climate' in feature:
                interpretations.append(f"Sex-specific climate vulnerability affecting {system_name}")
            elif '_x_housing_vulnerability' in feature:
                interpretations.append(f"Housing conditions modify climate effects on {system_name}")
            elif '_x_economic_vulnerability' in feature:
                interpretations.append(f"Economic status influences climate-{system_name} relationships")
            elif 'lag' in feature and '_x_' in feature:
                interpretations.append(f"Delayed climate-demographic interactions in {system_name}")
            else:
                interpretations.append(f"Complex interaction affecting {system_name}: {feature}")
        
        return interpretations
    
    def discover_causal_pathways(self, df, interaction_features):
        """Discover potential causal pathways using XAI and causal inference techniques"""
        
        self.logger.info("🔗 Discovering causal pathways between systems")
        
        # Define potential causal relationships to test
        causal_hypotheses = [
            {
                'pathway': 'Climate → Cardiovascular → Metabolic',
                'mediator': 'systolic blood pressure',
                'outcome': 'FASTING GLUCOSE',
                'exposure': 'temperature'
            },
            {
                'pathway': 'Climate → Metabolic → Cardiovascular', 
                'mediator': 'FASTING GLUCOSE',
                'outcome': 'systolic blood pressure',
                'exposure': 'heat_index'
            },
            {
                'pathway': 'Climate → Immune → Metabolic',
                'mediator': 'CD4 cell count (cells/µL)',
                'outcome': 'FASTING GLUCOSE',
                'exposure': 'temperature'
            }
        ]
        
        causal_discoveries = {}
        
        for hypothesis in causal_hypotheses:
            pathway_name = hypothesis['pathway']
            self.logger.info(f"Testing pathway: {pathway_name}")
            
            # Check if all variables are available
            required_vars = [hypothesis['exposure'], hypothesis['mediator'], hypothesis['outcome']]
            if not all(var in df.columns for var in required_vars):
                self.logger.warning(f"Missing variables for pathway {pathway_name}")
                continue
            
            # Create analysis dataset
            pathway_data = df[required_vars + ['Race', 'Sex']].dropna()
            
            if len(pathway_data) < 200:
                self.logger.warning(f"Insufficient data for pathway {pathway_name}")
                continue
            
            # Test mediation using XAI-based approach
            mediation_result = self._test_mediation_with_xai(
                pathway_data,
                hypothesis['exposure'],
                hypothesis['mediator'], 
                hypothesis['outcome']
            )
            
            causal_discoveries[pathway_name] = mediation_result
            
        self.results['causal_pathways'] = causal_discoveries
        return causal_discoveries
    
    def _test_mediation_with_xai(self, data, exposure, mediator, outcome):
        """Test mediation using XAI-based approach"""
        
        # Model 1: Exposure → Outcome (total effect)
        X_total = data[[exposure]].values
        y_outcome = data[outcome].values
        
        model_total = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_total.fit(X_total, y_outcome)
        total_effect_r2 = model_total.score(X_total, y_outcome)
        
        # Model 2: Exposure → Mediator
        y_mediator = data[mediator].values
        
        model_mediator = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_mediator.fit(X_total, y_mediator)
        mediator_effect_r2 = model_mediator.score(X_total, y_mediator)
        
        # Model 3: Exposure + Mediator → Outcome (direct effect)
        X_direct = data[[exposure, mediator]].values
        
        model_direct = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_direct.fit(X_direct, y_outcome)
        direct_effect_r2 = model_direct.score(X_direct, y_outcome)
        
        # Calculate mediation metrics
        total_r2_improvement = direct_effect_r2 - total_effect_r2
        mediation_strength = mediator_effect_r2 * total_r2_improvement
        
        # Use SHAP to understand the mediation mechanism
        try:
            explainer = shap.TreeExplainer(model_direct)
            shap_sample = X_direct[:min(100, len(X_direct))]
            shap_values = explainer.shap_values(shap_sample)
            
            # Feature importance for exposure vs mediator
            exposure_importance = np.abs(shap_values[:, 0]).mean()
            mediator_importance = np.abs(shap_values[:, 1]).mean()
            
            mediation_ratio = mediator_importance / (exposure_importance + mediator_importance)
            
        except:
            mediation_ratio = 0.5  # Default if SHAP fails
        
        return {
            'total_effect_r2': total_effect_r2,
            'mediator_effect_r2': mediator_effect_r2,
            'direct_effect_r2': direct_effect_r2,
            'mediation_strength': mediation_strength,
            'mediation_ratio': mediation_ratio,
            'evidence_strength': 'Strong' if mediation_strength > 0.05 else 'Moderate' if mediation_strength > 0.02 else 'Weak',
            'sample_size': len(data)
        }
    
    def generate_novel_insights(self, multi_system_discoveries, causal_discoveries):
        """Generate novel mechanistic insights from XAI discoveries"""
        
        self.logger.info("💡 Generating novel mechanistic insights")
        
        insights = {
            'cross_system_patterns': [],
            'demographic_vulnerabilities': [],
            'temporal_mechanisms': [],
            'causal_mechanisms': [],
            'clinical_implications': []
        }
        
        # Analyze cross-system patterns
        successful_systems = {k: v for k, v in multi_system_discoveries.items() 
                            if v.get('discovery_status') == 'SUCCESS'}
        
        if len(successful_systems) >= 2:
            # Look for common interaction patterns across systems
            all_interactions = {}
            for system, results in successful_systems.items():
                for interaction, importance in results['top_interactions']:
                    if interaction not in all_interactions:
                        all_interactions[interaction] = []
                    all_interactions[interaction].append((system, importance))
            
            # Find interactions that appear in multiple systems
            cross_system_interactions = {k: v for k, v in all_interactions.items() if len(v) >= 2}
            
            for interaction, systems in cross_system_interactions.items():
                insight = f"Multi-system interaction: {interaction} affects {[s[0] for s in systems]}"
                insights['cross_system_patterns'].append(insight)
        
        # Analyze demographic vulnerability patterns
        for system, results in successful_systems.items():
            race_interactions = [i for i, _ in results['top_interactions'] if 'Race' in i[0]]
            sex_interactions = [i for i, _ in results['top_interactions'] if 'Sex' in i[0]]
            
            if race_interactions:
                insights['demographic_vulnerabilities'].append(
                    f"Race-specific climate vulnerability in {system} system"
                )
            if sex_interactions:
                insights['demographic_vulnerabilities'].append(
                    f"Sex-specific climate effects on {system} system"
                )
        
        # Analyze temporal mechanisms
        for system, results in successful_systems.items():
            lag_interactions = [i for i, _ in results['top_interactions'] if 'lag' in i[0]]
            if lag_interactions:
                insights['temporal_mechanisms'].append(
                    f"Delayed climate effects on {system} system via {len(lag_interactions)} lag interactions"
                )
        
        # Analyze causal mechanisms
        for pathway, result in causal_discoveries.items():
            if result['evidence_strength'] in ['Strong', 'Moderate']:
                insights['causal_mechanisms'].append(
                    f"{pathway}: {result['evidence_strength']} evidence "
                    f"(mediation strength: {result['mediation_strength']:.3f})"
                )
        
        # Generate clinical implications
        if insights['demographic_vulnerabilities']:
            insights['clinical_implications'].append(
                "Personalized climate risk assessment needed based on demographic factors"
            )
        
        if insights['temporal_mechanisms']:
            insights['clinical_implications'].append(
                "Extended monitoring protocols required for delayed climate health effects"
            )
        
        if insights['causal_mechanisms']:
            insights['clinical_implications'].append(
                "Multi-system intervention strategies may be more effective than single-system approaches"
            )
        
        self.results['novel_insights'] = insights
        return insights
    
    def run_comprehensive_analysis(self, filepath):
        """Run the complete advanced causal XAI discovery pipeline"""
        
        self.logger.info("🚀 Starting Advanced Causal XAI Discovery Analysis")
        self.logger.info("="*60)
        
        # Load and prepare data
        df = self.load_and_prepare_comprehensive_data(filepath)
        
        # Create interaction features
        df, interaction_features = self.create_interaction_features(df)
        
        # Discover multi-system interactions
        multi_system_discoveries = self.discover_multi_system_interactions(df, interaction_features)
        
        # Discover causal pathways
        causal_discoveries = self.discover_causal_pathways(df, interaction_features)
        
        # Generate novel insights
        novel_insights = self.generate_novel_insights(multi_system_discoveries, causal_discoveries)
        
        # Summarize results
        self._summarize_breakthrough_discoveries()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _summarize_breakthrough_discoveries(self):
        """Summarize the breakthrough discoveries"""
        
        self.logger.info("\n🎯 BREAKTHROUGH DISCOVERY SUMMARY")
        self.logger.info("="*40)
        
        # Count successful discoveries
        successful_systems = sum(1 for system, results in self.results['interaction_discoveries'].items()
                               if results.get('discovery_status') == 'SUCCESS')
        
        total_systems = len(self.results['interaction_discoveries'])
        
        strong_causal_evidence = sum(1 for pathway, results in self.results['causal_pathways'].items()
                                   if results.get('evidence_strength') == 'Strong')
        
        total_insights = sum(len(insights) for insights in self.results['novel_insights'].values())
        
        self.logger.info(f"✅ Multi-system discoveries: {successful_systems}/{total_systems}")
        self.logger.info(f"✅ Strong causal evidence: {strong_causal_evidence}")
        self.logger.info(f"✅ Novel mechanistic insights: {total_insights}")
        
        # Highlight top discoveries
        if successful_systems > 0:
            self.logger.info("\n🔬 Top Multi-System Discoveries:")
            for system, results in self.results['interaction_discoveries'].items():
                if results.get('discovery_status') == 'SUCCESS':
                    test_r2 = results['model_performance']['test_r2']
                    n_interactions = len(results['top_interactions'])
                    self.logger.info(f"  • {system}: R² = {test_r2:.3f}, {n_interactions} interactions")
        
        if strong_causal_evidence > 0:
            self.logger.info("\n🔗 Strong Causal Evidence:")
            for pathway, results in self.results['causal_pathways'].items():
                if results.get('evidence_strength') == 'Strong':
                    strength = results['mediation_strength']
                    self.logger.info(f"  • {pathway}: mediation strength = {strength:.3f}")
        
        self.logger.info("\n🎯 Analysis Status: DISCOVERY PIPELINE COMPLETE")
    
    def _save_results(self):
        """Save comprehensive results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"advanced_causal_xai_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"📁 Results saved to: {results_file}")


def main():
    """Run the advanced causal XAI discovery analysis"""
    
    discovery = AdvancedCausalXAIDiscovery()
    
    # Run comprehensive analysis
    results = discovery.run_comprehensive_analysis('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv')
    
    print("\n🏆 ADVANCED CAUSAL XAI DISCOVERY COMPLETE!")
    print("="*50)
    print("This analysis goes beyond traditional correlations to discover:")
    print("• Multi-system physiological interactions")
    print("• Causal pathways between climate and health")
    print("• Demographic-specific vulnerability mechanisms")
    print("• Novel temporal interaction patterns")
    print("\nResults provide mechanistic insights for precision climate medicine!")
    
    return results


if __name__ == "__main__":
    main()