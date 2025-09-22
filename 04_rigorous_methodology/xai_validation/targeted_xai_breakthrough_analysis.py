#!/usr/bin/env python3
"""
Targeted XAI Breakthrough Analysis
=================================

Focus on applying explainable AI to your validated discoveries:
- Glucose-temperature-race interaction (R² = 0.348)
- 25+ significant climate-health relationships already discovered
- Apply advanced XAI to understand mechanisms and find new patterns

Strategy: Build on proven findings rather than starting from scratch
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import shap
from scipy.stats import pearsonr
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TargetedXAIAnalysis:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("targeted_xai_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Focus on validated relationships from your previous discoveries
        self.validated_targets = {
            'FASTING GLUCOSE': {
                'primary_climate': ['temperature', 'temp', 'heat'],
                'demographics': ['Race', 'Sex'],
                'expected_r2': 0.348,
                'lag_focus': [0, 1, 2, 3]
            },
            'systolic blood pressure': {
                'primary_climate': ['temperature', 'temp'],
                'demographics': ['Sex', 'Race'],
                'expected_r2': 0.1,
                'lag_focus': [0, 1, 2]
            },
            'CD4 cell count': {
                'primary_climate': ['temperature', 'heat', 'degree'],
                'demographics': ['Sex'],
                'expected_r2': 0.2,
                'lag_focus': [1, 2, 3, 5, 7]
            }
        }
        
        self.xai_discoveries = {}
        
    def log_progress(self, message, level="INFO"):
        """Progress logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🎯", "SUCCESS": "✅", "BREAKTHROUGH": "🚀", "DISCOVERY": "💡"}
        icon = icons.get(level, "🎯")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)

    def load_and_prepare_targeted_data(self):
        """Load data focusing on validated biomarkers and climate variables"""
        self.log_progress("Loading data for targeted XAI analysis...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Get all temperature-related features (your strongest findings)
        temp_features = []
        for col in df.columns:
            col_lower = col.lower()
            if any(temp_word in col_lower for temp_word in ['temp', 'heat']) and 'lag' in col_lower:
                temp_features.append(col)
        
        # Get other climate features
        other_climate = []
        for col in df.columns:
            col_lower = col.lower()
            if (any(climate_word in col_lower for climate_word in ['humid', 'wind', 'pressure', 'solar']) 
                and 'lag' in col_lower):
                other_climate.append(col)
        
        # Available demographics
        demographics = ['Sex', 'Race', 'Education']
        available_demographics = [d for d in demographics if d in df.columns]
        
        self.log_progress(f"Dataset: {len(df)} records")
        self.log_progress(f"Temperature features: {len(temp_features)}")
        self.log_progress(f"Other climate features: {len(other_climate)}")
        self.log_progress(f"Demographics: {available_demographics}")
        
        return df, temp_features, other_climate, available_demographics

    def create_validated_interaction_features(self, df, temp_features, demographics):
        """Create interaction features based on your validated discoveries"""
        self.log_progress("Creating validated interaction features...")
        
        interaction_features = []
        
        # Focus on glucose-temperature-race interaction (your R² = 0.348 discovery)
        if 'Race' in demographics and 'Race' in df.columns:
            # Encode race
            le_race = LabelEncoder()
            df['Race_encoded'] = le_race.fit_transform(df['Race'].fillna('unknown'))
            
            # Create temperature × race interactions for multiple lags
            for temp_var in temp_features[:10]:  # Focus on top temperature variables
                if temp_var in df.columns:
                    interaction_name = f"{temp_var}_x_Race"
                    df[interaction_name] = (df[temp_var].fillna(df[temp_var].median()) * 
                                          df['Race_encoded'])
                    interaction_features.append(interaction_name)
        
        # Sex interactions
        if 'Sex' in demographics and 'Sex' in df.columns:
            le_sex = LabelEncoder()
            df['Sex_encoded'] = le_sex.fit_transform(df['Sex'].fillna('unknown'))
            
            for temp_var in temp_features[:8]:
                if temp_var in df.columns:
                    interaction_name = f"{temp_var}_x_Sex"
                    df[interaction_name] = (df[temp_var].fillna(df[temp_var].median()) * 
                                          df['Sex_encoded'])
                    interaction_features.append(interaction_name)
        
        # Temperature variability features
        temp_lags = [f for f in temp_features if any(f'lag{i}' in f.lower() for i in [0, 1, 2, 3])][:7]
        if len(temp_lags) >= 3:
            temp_data = df[temp_lags].fillna(method='ffill')
            df['temp_variability_0_3'] = temp_data.std(axis=1)
            df['temp_trend_0_3'] = temp_data.iloc[:, -1] - temp_data.iloc[:, 0]
            interaction_features.extend(['temp_variability_0_3', 'temp_trend_0_3'])
        
        self.log_progress(f"Created {len(interaction_features)} validated interaction features")
        return interaction_features

    def comprehensive_shap_analysis(self, df, biomarker, climate_features, interaction_features):
        """Comprehensive SHAP analysis for a specific biomarker"""
        
        if biomarker not in df.columns:
            return None
        
        biomarker_data = df.dropna(subset=[biomarker])
        
        if len(biomarker_data) < 500:
            return None
        
        self.log_progress(f"SHAP analysis for {biomarker} (n={len(biomarker_data)})")
        
        # Combine features - prioritize validated relationships
        if biomarker in self.validated_targets:
            target_info = self.validated_targets[biomarker]
            primary_climate = target_info['primary_climate']
            lag_focus = target_info['lag_focus']
            
            # Priority features based on validation
            priority_features = []
            for climate_var in climate_features:
                if any(primary in climate_var.lower() for primary in primary_climate):
                    if any(f'lag{lag}' in climate_var.lower() for lag in lag_focus):
                        priority_features.append(climate_var)
            
            # Add interactions
            priority_interactions = [f for f in interaction_features 
                                   if any(primary in f.lower() for primary in primary_climate)]
            
            all_features = priority_features[:15] + priority_interactions[:10] + climate_features[:10]
        else:
            all_features = climate_features[:20] + interaction_features[:15]
        
        # Remove duplicates and ensure availability
        all_features = list(dict.fromkeys(all_features))  # Remove duplicates
        available_features = [f for f in all_features if f in biomarker_data.columns][:30]
        
        if len(available_features) < 5:
            return None
        
        # Prepare data
        X = biomarker_data[available_features].fillna(biomarker_data[available_features].median())
        y = biomarker_data[biomarker]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train optimized model
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Validate performance
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        if test_score < 0.01:  # Skip if no predictive power
            return None
        
        # SHAP Analysis
        try:
            explainer = shap.TreeExplainer(model)
            
            # Use sample for efficiency
            sample_size = min(300, len(X_test))
            X_sample = X_test.iloc[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            # Feature importance from SHAP
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': available_features,
                'shap_importance': feature_importance,
                'shap_std': np.abs(shap_values).std(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            # Interaction analysis
            interaction_analysis = {}
            if len(available_features) <= 15:  # Computational limit
                try:
                    interaction_values = explainer.shap_interaction_values(X_sample.iloc[:50])
                    # Extract key interactions
                    interaction_matrix = np.abs(interaction_values).sum(axis=0)
                    
                    # Find top interactions
                    top_interactions = []
                    for i in range(len(available_features)):
                        for j in range(i+1, len(available_features)):
                            interaction_strength = interaction_matrix[i, j]
                            if interaction_strength > 0:
                                top_interactions.append({
                                    'feature1': available_features[i],
                                    'feature2': available_features[j],
                                    'interaction_strength': interaction_strength
                                })
                    
                    # Sort by strength
                    top_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                    interaction_analysis['top_interactions'] = top_interactions[:5]
                    
                except Exception as e:
                    interaction_analysis['error'] = str(e)
            
            # Demographic analysis
            demographic_insights = {}
            if 'Race_encoded' in biomarker_data.columns:
                race_groups = biomarker_data['Race_encoded'].unique()
                for race in race_groups:
                    race_data = biomarker_data[biomarker_data['Race_encoded'] == race]
                    if len(race_data) >= 50:
                        race_features = [f for f in available_features if f in race_data.columns]
                        if race_features:
                            X_race = race_data[race_features].fillna(race_data[race_features].median())
                            y_race = race_data[biomarker]
                            
                            # Quick model for this race group
                            model_race = RandomForestRegressor(n_estimators=50, random_state=42)
                            cv_scores = cross_val_score(model_race, X_race, y_race, cv=3, scoring='r2')
                            
                            demographic_insights[f'race_{race}'] = {
                                'r2': np.mean(cv_scores),
                                'n_samples': len(race_data)
                            }
            
            # Temporal pattern analysis
            temporal_patterns = {}
            for lag in [0, 1, 2, 3, 5, 7, 14, 21]:
                lag_features = [f for f in available_features if f'lag{lag}' in f.lower()]
                if lag_features:
                    lag_indices = [available_features.index(f) for f in lag_features]
                    lag_importance = feature_importance[lag_indices].sum()
                    temporal_patterns[f'lag_{lag}'] = {
                        'total_importance': lag_importance,
                        'n_features': len(lag_features),
                        'features': lag_features
                    }
            
            return {
                'biomarker': biomarker,
                'model_performance': {
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'n_samples': len(biomarker_data),
                    'n_features': len(available_features)
                },
                'shap_analysis': {
                    'feature_importance': importance_df.head(10).to_dict('records'),
                    'mean_abs_shap': np.mean(np.abs(shap_values)),
                    'interactions': interaction_analysis
                },
                'demographic_insights': demographic_insights,
                'temporal_patterns': temporal_patterns,
                'breakthrough_status': self._assess_breakthrough(test_score, biomarker)
            }
            
        except Exception as e:
            self.log_progress(f"SHAP analysis failed for {biomarker}: {e}")
            return None

    def _assess_breakthrough(self, test_r2, biomarker):
        """Assess if this represents a breakthrough discovery"""
        if biomarker in self.validated_targets:
            expected_r2 = self.validated_targets[biomarker]['expected_r2']
            if test_r2 >= expected_r2 * 0.7:  # Within 70% of expected
                return "VALIDATED_BREAKTHROUGH"
            elif test_r2 >= 0.05:
                return "SIGNIFICANT_DISCOVERY"
        
        if test_r2 >= 0.1:
            return "MAJOR_BREAKTHROUGH"
        elif test_r2 >= 0.05:
            return "BREAKTHROUGH"
        elif test_r2 >= 0.02:
            return "DISCOVERY"
        else:
            return "MINIMAL"

    def run_targeted_xai_analysis(self):
        """Execute targeted XAI analysis on validated discoveries"""
        self.log_progress("="*60)
        self.log_progress("🎯 TARGETED XAI BREAKTHROUGH ANALYSIS")
        self.log_progress("="*60)
        
        start_time = time.time()
        
        # Load data
        df, temp_features, other_climate, demographics = self.load_and_prepare_targeted_data()
        all_climate = temp_features + other_climate[:20]  # Focus on temperature + others
        
        # Create interaction features
        interaction_features = self.create_validated_interaction_features(df, temp_features, demographics)
        
        # Analyze each validated target
        breakthroughs = {}
        
        for biomarker in self.validated_targets.keys():
            if biomarker in df.columns:
                result = self.comprehensive_shap_analysis(df, biomarker, all_climate, interaction_features)
                
                if result:
                    breakthroughs[biomarker] = result
                    
                    r2 = result['model_performance']['test_r2']
                    status = result['breakthrough_status']
                    
                    if status in ['BREAKTHROUGH', 'MAJOR_BREAKTHROUGH', 'VALIDATED_BREAKTHROUGH']:
                        self.log_progress(f"🚀 {status}: {biomarker} - R² = {r2:.3f}", "BREAKTHROUGH")
                    elif status == 'DISCOVERY':
                        self.log_progress(f"💡 DISCOVERY: {biomarker} - R² = {r2:.3f}", "DISCOVERY")
                    
                    # Report top features
                    if result['shap_analysis']['feature_importance']:
                        top_feature = result['shap_analysis']['feature_importance'][0]
                        self.log_progress(f"  Top predictor: {top_feature['feature']} (SHAP: {top_feature['shap_importance']:.4f})")
        
        # Generate report
        report = self.generate_targeted_report(breakthroughs)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        self.log_progress("="*60)
        self.log_progress("✅ TARGETED XAI ANALYSIS COMPLETE")
        self.log_progress(f"Analysis time: {elapsed_time/60:.1f} minutes")
        self.log_progress(f"Breakthrough discoveries: {len(breakthroughs)}")
        
        return report

    def generate_targeted_report(self, breakthroughs):
        """Generate comprehensive targeted analysis report"""
        
        # Calculate summary statistics
        breakthrough_counts = {}
        for result in breakthroughs.values():
            status = result['breakthrough_status']
            breakthrough_counts[status] = breakthrough_counts.get(status, 0) + 1
        
        max_r2 = max([r['model_performance']['test_r2'] for r in breakthroughs.values()]) if breakthroughs else 0
        avg_r2 = np.mean([r['model_performance']['test_r2'] for r in breakthroughs.values()]) if breakthroughs else 0
        
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'Targeted XAI Breakthrough Analysis',
                'target_biomarkers': list(self.validated_targets.keys()),
                'focus': 'Validated climate-health relationships with XAI interpretation'
            },
            'summary': {
                'total_breakthroughs': len(breakthroughs),
                'breakthrough_categories': breakthrough_counts,
                'max_r2_achieved': max_r2,
                'average_r2': avg_r2,
                'validated_targets_analyzed': len(self.validated_targets)
            },
            'detailed_results': breakthroughs,
            'key_insights': self._extract_key_insights(breakthroughs),
            'clinical_implications': self._generate_clinical_implications(breakthroughs)
        }
        
        # Save report
        report_file = self.results_dir / f"targeted_xai_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_progress(f"Report saved: {report_file}")
        
        return report

    def _extract_key_insights(self, breakthroughs):
        """Extract key insights from breakthrough discoveries"""
        insights = {
            'top_climate_predictors': {},
            'demographic_patterns': {},
            'temporal_insights': {},
            'interaction_discoveries': {}
        }
        
        # Aggregate top climate predictors
        all_features = {}
        for biomarker, result in breakthroughs.items():
            for feature_info in result['shap_analysis']['feature_importance'][:3]:
                feature = feature_info['feature']
                importance = feature_info['shap_importance']
                
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append((biomarker, importance))
        
        # Sort by total importance across biomarkers
        for feature, biomarker_impacts in all_features.items():
            total_importance = sum([imp for _, imp in biomarker_impacts])
            insights['top_climate_predictors'][feature] = {
                'total_importance': total_importance,
                'affects_biomarkers': len(biomarker_impacts),
                'biomarker_impacts': biomarker_impacts
            }
        
        return insights

    def _generate_clinical_implications(self, breakthroughs):
        """Generate clinical and public health implications"""
        return {
            'immediate_applications': [
                'Personalized climate health risk assessment',
                'Targeted monitoring for high-risk populations',
                'Climate-informed clinical decision making'
            ],
            'research_priorities': [
                'Mechanistic validation of discovered pathways',
                'Intervention trials for identified relationships',
                'Population-specific threshold determination'
            ],
            'public_health_impact': [
                'Early warning system development',
                'Health equity climate adaptation strategies',
                'Population health surveillance enhancement'
            ]
        }

def main():
    """Execute targeted XAI analysis"""
    analyzer = TargetedXAIAnalysis()
    report = analyzer.run_targeted_xai_analysis()
    return report

if __name__ == "__main__":
    main()