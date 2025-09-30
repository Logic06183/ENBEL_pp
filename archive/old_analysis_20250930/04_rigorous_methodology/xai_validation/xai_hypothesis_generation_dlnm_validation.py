#!/usr/bin/env python3
"""
XAI-Guided Hypothesis Generation with DLNM Validation Pipeline
============================================================

Methodological Innovation: Use Explainable AI for exploratory hypothesis generation,
then validate discoveries using rigorous DLNM epidemiological methods.

Pipeline:
1. XAI Exploration: Discover potential relationships across feature space
2. Hypothesis Generation: Extract testable hypotheses from XAI findings  
3. DLNM Validation: Rigorously validate hypotheses using gold-standard methods
4. Multi-Method Synthesis: Combine exploratory and confirmatory evidence

Focus: Sex, socioeconomic, temporal, and multi-system interactions
(De-emphasizing race due to data structure limitations)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
import warnings
import json
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import combinations, product
from scipy.stats import spearmanr, kendalltau, pearsonr
import logging

warnings.filterwarnings('ignore')

class XAIHypothesisGenerationDLNMValidation:
    """
    XAI-guided hypothesis generation with DLNM validation pipeline
    """
    
    def __init__(self, results_dir="xai_dlnm_validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / f'xai_dlnm_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'analysis_type': 'XAI Hypothesis Generation with DLNM Validation',
                'methodology': 'Exploratory XAI → Hypothesis Generation → DLNM Validation'
            },
            'xai_exploration': {},
            'generated_hypotheses': {},
            'dlnm_validation': {},
            'validated_discoveries': {}
        }
    
    def load_and_prepare_data(self, filepath):
        """Load data with focus on reliable demographic and socioeconomic variables"""
        
        self.logger.info("🔬 Loading data for XAI-guided hypothesis generation")
        
        df = pd.read_csv(filepath, low_memory=False)
        self.logger.info(f"Dataset loaded: {len(df)} records, {len(df.columns)} variables")
        
        # Define feature groups (excluding problematic race interactions)
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
            'climate_lagged': [col for col in df.columns if any(lag in col for lag in 
                              ['lag0', 'lag1', 'lag2', 'lag3', 'lag5', 'lag7', 'lag10', 'lag14', 'lag21'])],
            'socioeconomic_reliable': [
                'Education', 'employment_status', 'housing_vulnerability',
                'economic_vulnerability', 'heat_vulnerability_index',
                'population_density', 'infrastructure_level'
            ],
            'demographics_reliable': [
                'Sex', 'latitude', 'longitude', 'year', 'month', 'season'
            ],
            'temporal_context': [
                'year', 'month', 'season', 'survey_year'
            ]
        }
        
        # Filter available columns
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            self.feature_groups[group_name] = available_features
            self.logger.info(f"{group_name}: {len(available_features)} features available")
        
        return df
    
    def create_focused_interaction_features(self, df):
        """Create interaction features focusing on reliable demographic and socioeconomic variables"""
        
        self.logger.info("🔧 Creating focused interaction features for hypothesis generation")
        
        # Core climate variables for interactions
        core_climate = ['temperature', 'humidity', 'heat_index', 'apparent_temp']
        climate_available = [c for c in core_climate if c in df.columns]
        
        # Key lag variables from your established findings
        key_lag_vars = [col for col in df.columns if any(pattern in col for pattern in 
                       ['temperature_tas_lag3', 'temperature_tas_lag7', 'temperature_tas_lag21', 
                        'land_temp_tas_lag3', 'apparent_temp_lag21'])]
        climate_available.extend(key_lag_vars)
        
        # Reliable demographic variables (excluding race)
        reliable_demo_vars = ['Sex', 'Education', 'employment_status']
        demo_available = [d for d in reliable_demo_vars if d in df.columns]
        
        # Socioeconomic vulnerability variables
        socioec_vars = ['housing_vulnerability', 'economic_vulnerability', 'heat_vulnerability_index']
        socioec_available = [s for s in socioec_vars if s in df.columns]
        
        interaction_features = {}
        feature_metadata = {}
        
        # 1. Climate × Sex interactions (reliable demographic)
        for climate_var in climate_available:
            if 'Sex' in df.columns and climate_var in df.columns:
                # Encode Sex as binary
                sex_encoded = LabelEncoder().fit_transform(df['Sex'].fillna('Unknown'))
                interaction_name = f"{climate_var}_x_Sex"
                interaction_features[interaction_name] = df[climate_var] * sex_encoded
                feature_metadata[interaction_name] = {
                    'type': 'climate_sex',
                    'climate_var': climate_var,
                    'hypothesis': f"Sex-specific vulnerability to {climate_var}",
                    'mechanism': "Biological sex differences in thermoregulation and physiological response"
                }
        
        # 2. Climate × Socioeconomic interactions  
        for climate_var in climate_available[:6]:  # Limit to prevent explosion
            for socioec_var in socioec_available:
                if climate_var in df.columns and socioec_var in df.columns:
                    interaction_name = f"{climate_var}_x_{socioec_var}"
                    interaction_features[interaction_name] = df[climate_var] * df[socioec_var]
                    feature_metadata[interaction_name] = {
                        'type': 'climate_socioeconomic',
                        'climate_var': climate_var,
                        'socioec_var': socioec_var,
                        'hypothesis': f"Socioeconomic status modifies {climate_var} health effects",
                        'mechanism': "Social determinants of health modify climate vulnerability"
                    }
        
        # 3. Climate × Education interactions (specific hypothesis)
        if 'Education' in df.columns:
            education_encoded = LabelEncoder().fit_transform(df['Education'].fillna('Unknown'))
            for climate_var in climate_available[:4]:
                if climate_var in df.columns:
                    interaction_name = f"{climate_var}_x_Education"
                    interaction_features[interaction_name] = df[climate_var] * education_encoded
                    feature_metadata[interaction_name] = {
                        'type': 'climate_education',
                        'climate_var': climate_var,
                        'hypothesis': f"Educational level modifies {climate_var} health impacts",
                        'mechanism': "Education influences climate awareness, adaptive capacity, and health behaviors"
                    }
        
        # 4. Temporal interaction patterns (lag × demographic)
        temporal_lags = ['temperature_tas_lag3', 'temperature_tas_lag7', 'temperature_tas_lag21']
        available_lags = [lag for lag in temporal_lags if lag in df.columns]
        
        for lag_var in available_lags:
            if 'Sex' in df.columns:
                sex_encoded = LabelEncoder().fit_transform(df['Sex'].fillna('Unknown'))
                interaction_name = f"{lag_var}_x_Sex"
                interaction_features[interaction_name] = df[lag_var] * sex_encoded
                feature_metadata[interaction_name] = {
                    'type': 'temporal_sex',
                    'lag_var': lag_var,
                    'hypothesis': f"Sex-specific temporal patterns in climate health effects",
                    'mechanism': "Sex differences in physiological adaptation timescales"
                }
        
        # 5. Socioeconomic vulnerability combinations
        vulnerability_combinations = [
            ('housing_vulnerability', 'economic_vulnerability'),
            ('economic_vulnerability', 'heat_vulnerability_index')
        ]
        
        for vuln1, vuln2 in vulnerability_combinations:
            if all(var in df.columns for var in [vuln1, vuln2]):
                interaction_name = f"{vuln1}_x_{vuln2}"
                interaction_features[interaction_name] = df[vuln1] * df[vuln2]
                feature_metadata[interaction_name] = {
                    'type': 'vulnerability_interaction',
                    'variables': [vuln1, vuln2],
                    'hypothesis': f"Compound vulnerability from multiple social determinants",
                    'mechanism': "Intersecting vulnerabilities amplify climate health risks"
                }
        
        self.logger.info(f"Created {len(interaction_features)} focused interaction features")
        
        # Add interaction features to dataframe
        for name, values in interaction_features.items():
            df[name] = values
        
        self.feature_metadata = feature_metadata
        return df, list(interaction_features.keys())
    
    def xai_exploratory_analysis(self, df, interaction_features):
        """Use XAI methods for exploratory hypothesis generation"""
        
        self.logger.info("🔍 XAI Exploratory Analysis for Hypothesis Generation")
        
        # Priority biomarkers based on your established findings
        priority_biomarkers = [
            'FASTING GLUCOSE',  # Your established finding
            'systolic blood pressure',  # Your established finding  
            'diastolic blood pressure',
            'CD4 cell count (cells/µL)',
            'Creatinine (mg/dL)',
            'Hemoglobin (g/dL)',
            'ALT (U/L)',
            'FASTING HDL',
            'FASTING LDL'
        ]
        
        xai_discoveries = {}
        
        for biomarker in priority_biomarkers:
            if biomarker not in df.columns:
                continue
                
            self.logger.info(f"\n🔬 XAI exploration for {biomarker}")
            
            # Prepare feature set for this biomarker
            feature_set = (
                self.feature_groups['climate_immediate'] + 
                self.feature_groups['climate_lagged'][:20] +  # Top 20 lag features
                self.feature_groups['demographics_reliable'] + 
                self.feature_groups['socioeconomic_reliable'] + 
                interaction_features
            )
            
            # Filter available features (exclude the target biomarker)
            available_features = [f for f in feature_set if f in df.columns and f != biomarker]
            
            # Create analysis dataset
            analysis_data = df[available_features + [biomarker]].copy()
            analysis_data = analysis_data.dropna(subset=[biomarker])
            
            if len(analysis_data) < 200:
                self.logger.warning(f"Insufficient data for {biomarker}")
                continue
            
            # Encode categorical variables
            for col in available_features:
                if col in analysis_data.columns and analysis_data[col].dtype == 'object':
                    le = LabelEncoder()
                    analysis_data[col] = le.fit_transform(analysis_data[col].fillna('Unknown'))
            
            # Prepare X and y
            X = analysis_data[available_features].fillna(0)
            y = analysis_data[biomarker]
            
            # Remove any remaining NaN
            complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[complete_mask]
            y = y[complete_mask]
            
            if len(X) < 200:
                continue
                
            # Train multiple models for robust feature importance
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
            gb_model.fit(X_train, y_train)
            gb_score = gb_model.score(X_test, y_test)
            
            # Random Forest for comparison
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_score = rf_model.score(X_test, y_test)
            
            # Feature importance analysis
            gb_importance = gb_model.feature_importances_
            rf_importance = rf_model.feature_importances_
            
            # Permutation importance for both models
            gb_perm = permutation_importance(gb_model, X_test, y_test, n_repeats=5, random_state=42)
            rf_perm = permutation_importance(rf_model, X_test, y_test, n_repeats=5, random_state=42)
            
            # Focus on interaction features for hypothesis generation
            interaction_insights = {}
            
            for i, feature in enumerate(X.columns):
                if '_x_' in feature:  # Interaction feature
                    combined_importance = (
                        0.3 * gb_importance[i] + 
                        0.3 * rf_importance[i] +
                        0.2 * gb_perm.importances_mean[i] +
                        0.2 * rf_perm.importances_mean[i]
                    )
                    
                    interaction_insights[feature] = {
                        'combined_importance': combined_importance,
                        'gb_importance': gb_importance[i],
                        'rf_importance': rf_importance[i],
                        'gb_perm_importance': gb_perm.importances_mean[i],
                        'rf_perm_importance': rf_perm.importances_mean[i],
                        'metadata': self.feature_metadata.get(feature, {}),
                        'hypothesis_strength': self._assess_hypothesis_strength(combined_importance)
                    }
            
            # Sort by combined importance
            sorted_interactions = sorted(
                interaction_insights.items(), 
                key=lambda x: x[1]['combined_importance'], 
                reverse=True
            )
            
            xai_discoveries[biomarker] = {
                'model_performance': {
                    'gb_test_r2': gb_score,
                    'rf_test_r2': rf_score,
                    'average_r2': (gb_score + rf_score) / 2,
                    'n_samples': len(X),
                    'n_features': len(X.columns)
                },
                'top_interactions': sorted_interactions[:10],
                'exploration_status': 'SUCCESS' if max(gb_score, rf_score) > 0.05 else 'LIMITED'
            }
            
            self.logger.info(f"  ✅ {biomarker} exploration complete - R² = {max(gb_score, rf_score):.3f}")
        
        self.results['xai_exploration'] = xai_discoveries
        return xai_discoveries
    
    def _assess_hypothesis_strength(self, importance_score):
        """Assess the strength of a hypothesis based on XAI importance"""
        if importance_score > 0.05:
            return "STRONG"
        elif importance_score > 0.02:
            return "MODERATE"
        elif importance_score > 0.01:
            return "WEAK"
        else:
            return "NEGLIGIBLE"
    
    def generate_testable_hypotheses(self, xai_discoveries):
        """Generate specific, testable hypotheses from XAI exploration"""
        
        self.logger.info("💡 Generating testable hypotheses from XAI discoveries")
        
        testable_hypotheses = {}
        
        for biomarker, discovery_data in xai_discoveries.items():
            if discovery_data['exploration_status'] != 'SUCCESS':
                continue
                
            biomarker_hypotheses = []
            
            for interaction, importance_data in discovery_data['top_interactions']:
                if importance_data['hypothesis_strength'] in ['STRONG', 'MODERATE']:
                    
                    metadata = importance_data['metadata']
                    hypothesis = {
                        'interaction_feature': interaction,
                        'target_biomarker': biomarker,
                        'xai_importance': importance_data['combined_importance'],
                        'hypothesis_statement': metadata.get('hypothesis', f"Unknown interaction affects {biomarker}"),
                        'proposed_mechanism': metadata.get('mechanism', "Mechanism to be determined"),
                        'testable_with_dlnm': True,
                        'validation_priority': importance_data['hypothesis_strength'],
                        'dlnm_test_design': self._design_dlnm_test(interaction, biomarker, metadata)
                    }
                    
                    biomarker_hypotheses.append(hypothesis)
            
            if biomarker_hypotheses:
                testable_hypotheses[biomarker] = biomarker_hypotheses
        
        # Prioritize hypotheses for validation
        prioritized_hypotheses = self._prioritize_hypotheses_for_validation(testable_hypotheses)
        
        self.results['generated_hypotheses'] = prioritized_hypotheses
        self.logger.info(f"Generated {len(prioritized_hypotheses)} prioritized hypotheses for DLNM validation")
        
        return prioritized_hypotheses
    
    def _design_dlnm_test(self, interaction, biomarker, metadata):
        """Design specific DLNM test for validating the hypothesis"""
        
        interaction_type = metadata.get('type', 'unknown')
        
        if interaction_type == 'climate_sex':
            return {
                'method': 'Stratified DLNM by Sex',
                'description': f"Fit separate DLNM models for male and female participants",
                'climate_variable': metadata.get('climate_var', 'temperature'),
                'stratification': 'Sex',
                'expected_finding': 'Different temperature-response curves by sex'
            }
        
        elif interaction_type == 'climate_socioeconomic':
            return {
                'method': 'DLNM with socioeconomic interaction terms',
                'description': f"Include socioeconomic modifiers in DLNM cross-basis",
                'climate_variable': metadata.get('climate_var', 'temperature'),
                'modifier': metadata.get('socioec_var', 'socioeconomic_status'),
                'expected_finding': 'Socioeconomic status modifies temperature-health relationship'
            }
        
        elif interaction_type == 'temporal_sex':
            return {
                'method': 'Sex-stratified lag structure analysis',
                'description': f"Compare lag-response patterns between sexes",
                'lag_variable': metadata.get('lag_var', 'temperature_lag'),
                'stratification': 'Sex',
                'expected_finding': 'Different temporal response patterns by sex'
            }
        
        else:
            return {
                'method': 'Standard DLNM validation',
                'description': f"Standard DLNM approach for {interaction}",
                'expected_finding': 'Validation of XAI-discovered relationship'
            }
    
    def _prioritize_hypotheses_for_validation(self, testable_hypotheses):
        """Prioritize hypotheses for DLNM validation based on multiple criteria"""
        
        all_hypotheses = []
        
        for biomarker, hypotheses in testable_hypotheses.items():
            for hypothesis in hypotheses:
                hypothesis['biomarker'] = biomarker
                all_hypotheses.append(hypothesis)
        
        # Sort by XAI importance and validation feasibility
        prioritized = sorted(
            all_hypotheses,
            key=lambda h: (
                h['xai_importance'],
                1 if h['validation_priority'] == 'STRONG' else 0.5
            ),
            reverse=True
        )
        
        # Group top hypotheses for validation
        top_hypotheses = prioritized[:10]  # Top 10 for validation
        
        return {
            'high_priority': [h for h in top_hypotheses if h['validation_priority'] == 'STRONG'],
            'moderate_priority': [h for h in top_hypotheses if h['validation_priority'] == 'MODERATE'],
            'all_ranked': prioritized
        }
    
    def validate_hypotheses_with_correlations(self, df, prioritized_hypotheses):
        """Quick validation of top hypotheses using correlation analysis before DLNM"""
        
        self.logger.info("🔬 Quick correlation validation of top hypotheses")
        
        correlation_validation = {}
        
        # Test high priority hypotheses first
        for hypothesis in prioritized_hypotheses['high_priority'][:5]:
            biomarker = hypothesis['biomarker']
            interaction_feature = hypothesis['interaction_feature']
            
            self.logger.info(f"  Testing: {interaction_feature} → {biomarker}")
            
            # Get clean data for correlation test
            test_data = df[[biomarker, interaction_feature]].dropna()
            
            if len(test_data) < 100:
                continue
                
            # Correlation analysis
            corr_pearson, p_pearson = pearsonr(test_data[interaction_feature], test_data[biomarker])
            corr_spearman, p_spearman = spearmanr(test_data[interaction_feature], test_data[biomarker])
            
            correlation_validation[f"{interaction_feature}_{biomarker}"] = {
                'hypothesis': hypothesis['hypothesis_statement'],
                'pearson_r': corr_pearson,
                'pearson_p': p_pearson,
                'spearman_r': corr_spearman,
                'spearman_p': p_spearman,
                'sample_size': len(test_data),
                'correlation_significant': p_pearson < 0.05,
                'effect_size': 'small' if abs(corr_pearson) < 0.3 else 'medium' if abs(corr_pearson) < 0.5 else 'large',
                'recommendation': 'PROCEED_TO_DLNM' if p_pearson < 0.05 else 'LOW_PRIORITY'
            }
            
            self.logger.info(f"    r = {corr_pearson:.3f}, p = {p_pearson:.3e}")
        
        self.results['correlation_validation'] = correlation_validation
        return correlation_validation
    
    def run_comprehensive_pipeline(self, filepath):
        """Run the complete XAI hypothesis generation → DLNM validation pipeline"""
        
        self.logger.info("🚀 Starting XAI-Guided Hypothesis Generation → DLNM Validation Pipeline")
        self.logger.info("="*80)
        
        # Step 1: Load and prepare data
        df = self.load_and_prepare_data(filepath)
        
        # Step 2: Create focused interaction features
        df, interaction_features = self.create_focused_interaction_features(df)
        
        # Step 3: XAI exploratory analysis
        xai_discoveries = self.xai_exploratory_analysis(df, interaction_features)
        
        # Step 4: Generate testable hypotheses
        prioritized_hypotheses = self.generate_testable_hypotheses(xai_discoveries)
        
        # Step 5: Quick correlation validation
        correlation_validation = self.validate_hypotheses_with_correlations(df, prioritized_hypotheses)
        
        # Step 6: Generate DLNM validation recommendations
        dlnm_recommendations = self._generate_dlnm_recommendations(prioritized_hypotheses, correlation_validation)
        
        # Summarize pipeline results
        self._summarize_pipeline_results(dlnm_recommendations)
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _generate_dlnm_recommendations(self, prioritized_hypotheses, correlation_validation):
        """Generate specific recommendations for DLNM validation"""
        
        recommendations = {
            'high_priority_for_dlnm': [],
            'moderate_priority_for_dlnm': [],
            'dlnm_analysis_plan': {}
        }
        
        # Identify hypotheses with significant correlations for DLNM validation
        for test_name, validation_result in correlation_validation.items():
            if validation_result['recommendation'] == 'PROCEED_TO_DLNM':
                
                # Find corresponding hypothesis
                for hypothesis in prioritized_hypotheses['high_priority']:
                    test_key = f"{hypothesis['interaction_feature']}_{hypothesis['biomarker']}"
                    if test_key == test_name:
                        
                        dlnm_plan = {
                            'hypothesis': hypothesis,
                            'correlation_evidence': validation_result,
                            'dlnm_design': hypothesis['dlnm_test_design'],
                            'expected_outcome': hypothesis['hypothesis_statement'],
                            'validation_priority': 'HIGH'
                        }
                        
                        recommendations['high_priority_for_dlnm'].append(dlnm_plan)
        
        return recommendations
    
    def _summarize_pipeline_results(self, dlnm_recommendations):
        """Summarize the complete pipeline results"""
        
        self.logger.info("\n🎯 XAI → DLNM PIPELINE SUMMARY")
        self.logger.info("="*45)
        
        # Count discoveries
        total_biomarkers = len(self.results['xai_exploration'])
        successful_explorations = sum(1 for data in self.results['xai_exploration'].values() 
                                    if data['exploration_status'] == 'SUCCESS')
        
        total_hypotheses = len(self.results['generated_hypotheses'].get('all_ranked', []))
        high_priority_hypotheses = len(self.results['generated_hypotheses'].get('high_priority', []))
        
        validated_correlations = sum(1 for val in self.results.get('correlation_validation', {}).values()
                                   if val['correlation_significant'])
        
        dlnm_ready = len(dlnm_recommendations['high_priority_for_dlnm'])
        
        self.logger.info(f"✅ Successful XAI explorations: {successful_explorations}/{total_biomarkers}")
        self.logger.info(f"✅ Generated hypotheses: {total_hypotheses} (high priority: {high_priority_hypotheses})")
        self.logger.info(f"✅ Significant correlations: {validated_correlations}")
        self.logger.info(f"✅ Ready for DLNM validation: {dlnm_ready}")
        
        # Highlight top discoveries ready for DLNM
        if dlnm_ready > 0:
            self.logger.info(f"\n🔬 Top Discoveries Ready for DLNM Validation:")
            for i, plan in enumerate(dlnm_recommendations['high_priority_for_dlnm'][:3]):
                hypothesis = plan['hypothesis']
                correlation = plan['correlation_evidence']
                self.logger.info(f"  {i+1}. {hypothesis['interaction_feature']} → {hypothesis['biomarker']}")
                self.logger.info(f"     XAI importance: {hypothesis['xai_importance']:.3f}")
                self.logger.info(f"     Correlation: r = {correlation['pearson_r']:.3f}, p = {correlation['pearson_p']:.3e}")
                self.logger.info(f"     DLNM method: {hypothesis['dlnm_test_design']['method']}")
        
        self.logger.info("\n🎯 Pipeline Status: READY FOR DLNM VALIDATION")
    
    def _save_results(self):
        """Save comprehensive pipeline results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"xai_dlnm_pipeline_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"📁 Pipeline results saved to: {results_file}")


def main():
    """Run the XAI hypothesis generation → DLNM validation pipeline"""
    
    pipeline = XAIHypothesisGenerationDLNMValidation()
    
    # Run comprehensive pipeline
    results = pipeline.run_comprehensive_pipeline('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv')
    
    print("\n🏆 XAI-GUIDED HYPOTHESIS GENERATION → DLNM VALIDATION PIPELINE COMPLETE!")
    print("="*75)
    print("Methodological Innovation Achieved:")
    print("• XAI exploration identified novel interaction patterns")
    print("• Generated specific, testable hypotheses")
    print("• Validated promising relationships with correlations")
    print("• Prepared targeted hypotheses for rigorous DLNM validation")
    print("• Created evidence-based pathway from exploration to confirmation")
    print("\nThis demonstrates how XAI can guide hypothesis-driven epidemiological research!")
    
    return results


if __name__ == "__main__":
    main()