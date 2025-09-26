#!/usr/bin/env python3
"""
Robust Alternative Discovery Framework
=====================================

Scientific exploration focusing on robust demographic groups and traditional
epidemiological approaches, avoiding small sample size issues.

Focus areas:
1. Sex-based analysis (more balanced groups)
2. Traditional epidemiological approaches  
3. Climate variable discovery without demographics
4. Temporal pattern analysis
5. Biomarker system interactions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RobustAlternativeDiscovery:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("robust_discovery_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Focus on robust biomarkers
        self.biomarkers = [
            'systolic blood pressure', 'diastolic blood pressure',
            'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 
            'FASTING HDL', 'FASTING LDL', 'CD4 cell count'
        ]
        
        self.discoveries = {}
        
    def log_progress(self, message, level="INFO"):
        """Progress logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🔍", "SUCCESS": "✅", "DISCOVERY": "💡", "WARNING": "⚠️"}
        icon = icons.get(level, "🔍")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)

    def load_and_assess_data_quality(self):
        """Load data with comprehensive quality assessment"""
        self.log_progress("Loading data with quality assessment...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Climate features assessment
        temp_features = [col for col in df.columns if 'temp' in col.lower() and 'lag' in col.lower()]
        humidity_features = [col for col in df.columns if 'humid' in col.lower() and 'lag' in col.lower()]
        wind_features = [col for col in df.columns if 'wind' in col.lower() and 'lag' in col.lower()]
        pressure_features = [col for col in df.columns if 'pressure' in col.lower() and 'lag' in col.lower()]
        
        # Demographic assessment
        sex_viable = False
        if 'Sex' in df.columns:
            sex_counts = df['Sex'].value_counts(dropna=False)
            viable_sex_groups = {k: v for k, v in sex_counts.items() if v >= 200}
            if len(viable_sex_groups) >= 2:
                sex_viable = True
        
        self.log_progress(f"Dataset: {len(df)} records")
        self.log_progress(f"Temperature features: {len(temp_features)}")
        self.log_progress(f"Humidity features: {len(humidity_features)}")
        self.log_progress(f"Wind features: {len(wind_features)}")
        self.log_progress(f"Sex analysis viable: {sex_viable}")
        
        return df, temp_features, humidity_features, wind_features, pressure_features, sex_viable

    def traditional_epidemiological_analysis(self, df, climate_features, biomarker):
        """Traditional epidemiological correlation analysis"""
        
        biomarker_data = df.dropna(subset=[biomarker])
        
        if len(biomarker_data) < 500:
            return None
        
        self.log_progress(f"Traditional analysis for {biomarker} (n={len(biomarker_data)})")
        
        results = {
            'biomarker': biomarker,
            'n_samples': len(biomarker_data),
            'correlations': [],
            'significant_relationships': []
        }
        
        # Test correlations with individual climate variables
        for climate_var in climate_features:
            if climate_var in biomarker_data.columns:
                climate_data = biomarker_data[climate_var].dropna()
                biomarker_subset = biomarker_data.loc[climate_data.index, biomarker]
                
                if len(climate_data) >= 100:
                    # Pearson correlation
                    corr, p_value = pearsonr(climate_data, biomarker_subset)
                    
                    # Effect size assessment
                    effect_size = "negligible"
                    if abs(corr) >= 0.1:
                        effect_size = "small"
                    if abs(corr) >= 0.3:
                        effect_size = "medium"
                    if abs(corr) >= 0.5:
                        effect_size = "large"
                    
                    correlation_result = {
                        'climate_variable': climate_var,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_pairs': len(climate_data),
                        'effect_size': effect_size,
                        'significant': p_value < 0.001 and abs(corr) >= 0.05
                    }
                    
                    results['correlations'].append(correlation_result)
                    
                    if correlation_result['significant']:
                        results['significant_relationships'].append(correlation_result)
        
        # Sort by correlation strength
        results['correlations'].sort(key=lambda x: abs(x['correlation']), reverse=True)
        results['significant_relationships'].sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return results

    def sex_stratified_analysis(self, df, climate_features, biomarker):
        """Robust sex-based stratification (avoiding small sample issues)"""
        
        if 'Sex' not in df.columns:
            return None
        
        biomarker_data = df.dropna(subset=[biomarker])
        
        # Only consider sex groups with adequate sample sizes
        sex_counts = biomarker_data['Sex'].value_counts(dropna=False)
        viable_sex_groups = {k: v for k, v in sex_counts.items() if v >= 200}
        
        if len(viable_sex_groups) < 2:
            return None
        
        self.log_progress(f"Sex-stratified analysis for {biomarker}")
        for sex, count in viable_sex_groups.items():
            self.log_progress(f"  {sex}: n={count}")
        
        sex_results = {}
        
        for sex_group in viable_sex_groups.keys():
            sex_data = biomarker_data[biomarker_data['Sex'] == sex_group]
            
            # Focus on top temperature features
            top_temp_features = [f for f in climate_features if 'temp' in f.lower()][:10]
            available_features = [f for f in top_temp_features if f in sex_data.columns]
            
            if len(available_features) < 3:
                continue
            
            # Traditional approach: simple linear regression
            for climate_var in available_features[:5]:  # Top 5 for computational efficiency
                climate_values = sex_data[climate_var].dropna()
                biomarker_values = sex_data.loc[climate_values.index, biomarker]
                
                if len(climate_values) >= 50:
                    corr, p_value = pearsonr(climate_values, biomarker_values)
                    
                    if abs(corr) >= 0.05 and p_value < 0.05:
                        if str(sex_group) not in sex_results:
                            sex_results[str(sex_group)] = []
                        
                        sex_results[str(sex_group)].append({
                            'climate_variable': climate_var,
                            'correlation': corr,
                            'p_value': p_value,
                            'n_samples': len(climate_values)
                        })
        
        return sex_results if sex_results else None

    def temporal_pattern_discovery(self, df, biomarker):
        """Discover temporal patterns without demographic complications"""
        
        biomarker_data = df.dropna(subset=[biomarker])
        
        if len(biomarker_data) < 1000:
            return None
        
        self.log_progress(f"Temporal pattern discovery for {biomarker}")
        
        # Group climate features by lag
        lag_groups = {}
        for lag in [0, 1, 2, 3, 5, 7, 14, 21]:
            lag_features = [col for col in df.columns 
                          if f'lag{lag}' in col.lower() and 'temp' in col.lower()]
            if lag_features:
                lag_groups[f'lag_{lag}'] = lag_features
        
        temporal_results = {}
        
        for lag_name, lag_features in lag_groups.items():
            available_lag_features = [f for f in lag_features if f in biomarker_data.columns]
            
            if len(available_lag_features) < 2:
                continue
            
            # Test each lag period
            best_correlation = 0
            best_feature = None
            
            for feature in available_lag_features:
                feature_data = biomarker_data[feature].dropna()
                biomarker_subset = biomarker_data.loc[feature_data.index, biomarker]
                
                if len(feature_data) >= 100:
                    corr, p_value = pearsonr(feature_data, biomarker_subset)
                    
                    if abs(corr) > abs(best_correlation) and p_value < 0.01:
                        best_correlation = corr
                        best_feature = feature
            
            if best_feature:
                temporal_results[lag_name] = {
                    'best_feature': best_feature,
                    'correlation': best_correlation,
                    'n_features_tested': len(available_lag_features)
                }
        
        return temporal_results if temporal_results else None

    def climate_variable_discovery(self, df, biomarker):
        """Systematic climate variable discovery without demographics"""
        
        biomarker_data = df.dropna(subset=[biomarker])
        
        if len(biomarker_data) < 500:
            return None
        
        self.log_progress(f"Climate variable discovery for {biomarker}")
        
        # Group by climate type
        climate_groups = {
            'temperature': [col for col in df.columns if 'temp' in col.lower() and 'lag' in col.lower()],
            'humidity': [col for col in df.columns if 'humid' in col.lower() and 'lag' in col.lower()],
            'wind': [col for col in df.columns if 'wind' in col.lower() and 'lag' in col.lower()],
            'pressure': [col for col in df.columns if 'pressure' in col.lower() and 'lag' in col.lower()],
            'heat_index': [col for col in df.columns if 'heat' in col.lower() and 'lag' in col.lower()]
        }
        
        climate_results = {}
        
        for climate_type, features in climate_groups.items():
            available_features = [f for f in features if f in biomarker_data.columns]
            
            if len(available_features) < 2:
                continue
            
            # Test with traditional ML approach
            if len(available_features) >= 5:
                X = biomarker_data[available_features[:10]].fillna(biomarker_data[available_features[:10]].median())
                y = biomarker_data[biomarker]
                
                # Simple Ridge regression for stability
                model = Ridge(alpha=1.0)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_score = np.mean(cv_scores)
                
                if mean_score > 0.01:  # Conservative threshold
                    model.fit(X, y)
                    
                    # Feature importance from coefficients
                    importance = np.abs(model.coef_)
                    feature_importance = list(zip(available_features[:10], importance))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    climate_results[climate_type] = {
                        'cv_r2': mean_score,
                        'cv_std': np.std(cv_scores),
                        'n_features': len(available_features[:10]),
                        'top_features': feature_importance[:3]
                    }
        
        return climate_results if climate_results else None

    def biomarker_interaction_discovery(self, df):
        """Explore relationships between biomarkers and climate collectively"""
        
        self.log_progress("Exploring biomarker interaction patterns...")
        
        # Find biomarkers with sufficient data
        available_biomarkers = []
        for biomarker in self.biomarkers:
            if biomarker in df.columns:
                biomarker_data = df.dropna(subset=[biomarker])
                if len(biomarker_data) >= 500:
                    available_biomarkers.append(biomarker)
        
        if len(available_biomarkers) < 2:
            return None
        
        # Temperature features
        temp_features = [col for col in df.columns if 'temp' in col.lower() and 'lag' in col.lower()][:15]
        
        interaction_results = {}
        
        # Test each biomarker pair
        for i, bio1 in enumerate(available_biomarkers):
            for bio2 in available_biomarkers[i+1:]:
                # Find common samples
                common_data = df.dropna(subset=[bio1, bio2])
                
                if len(common_data) >= 200:
                    # Test biomarker correlation
                    bio_corr, bio_p = pearsonr(common_data[bio1], common_data[bio2])
                    
                    if abs(bio_corr) >= 0.1 and bio_p < 0.01:
                        # Test if climate affects both
                        available_temp = [f for f in temp_features if f in common_data.columns][:5]
                        
                        if len(available_temp) >= 3:
                            combined_biomarker = (common_data[bio1] + common_data[bio2]) / 2
                            
                            best_climate_corr = 0
                            best_climate_feature = None
                            
                            for temp_feature in available_temp:
                                temp_data = common_data[temp_feature].dropna()
                                bio_subset = combined_biomarker.loc[temp_data.index]
                                
                                if len(temp_data) >= 50:
                                    climate_corr, climate_p = pearsonr(temp_data, bio_subset)
                                    
                                    if abs(climate_corr) > abs(best_climate_corr) and climate_p < 0.05:
                                        best_climate_corr = climate_corr
                                        best_climate_feature = temp_feature
                            
                            if best_climate_feature:
                                interaction_results[f"{bio1}_x_{bio2}"] = {
                                    'biomarker_correlation': bio_corr,
                                    'biomarker_p_value': bio_p,
                                    'climate_correlation': best_climate_corr,
                                    'best_climate_feature': best_climate_feature,
                                    'n_samples': len(common_data)
                                }
        
        return interaction_results if interaction_results else None

    def run_comprehensive_robust_discovery(self):
        """Execute comprehensive robust discovery avoiding small sample issues"""
        self.log_progress("="*60)
        self.log_progress("🔍 ROBUST ALTERNATIVE DISCOVERY FRAMEWORK")
        self.log_progress("="*60)
        
        start_time = time.time()
        
        # Load and assess data
        df, temp_features, humidity_features, wind_features, pressure_features, sex_viable = self.load_and_assess_data_quality()
        all_climate = temp_features + humidity_features + wind_features + pressure_features
        
        discoveries = {}
        
        # 1. Traditional epidemiological analysis
        self.log_progress("\n📊 TRADITIONAL EPIDEMIOLOGICAL ANALYSIS")
        for biomarker in self.biomarkers:
            if biomarker in df.columns:
                traditional_result = self.traditional_epidemiological_analysis(df, all_climate, biomarker)
                if traditional_result and traditional_result['significant_relationships']:
                    discoveries[f"traditional_{biomarker}"] = traditional_result
                    n_sig = len(traditional_result['significant_relationships'])
                    if n_sig > 0:
                        top_corr = traditional_result['significant_relationships'][0]['correlation']
                        top_var = traditional_result['significant_relationships'][0]['climate_variable']
                        self.log_progress(f"DISCOVERY: {biomarker} - {n_sig} significant relationships", "DISCOVERY")
                        self.log_progress(f"  Strongest: {top_var} (r={top_corr:.3f})")
        
        # 2. Sex-stratified analysis (if viable)
        if sex_viable:
            self.log_progress("\n👥 SEX-STRATIFIED ANALYSIS")
            for biomarker in self.biomarkers:
                if biomarker in df.columns:
                    sex_result = self.sex_stratified_analysis(df, all_climate, biomarker)
                    if sex_result:
                        discoveries[f"sex_stratified_{biomarker}"] = sex_result
                        self.log_progress(f"DISCOVERY: {biomarker} shows sex-specific patterns", "DISCOVERY")
        
        # 3. Temporal pattern discovery
        self.log_progress("\n⏰ TEMPORAL PATTERN DISCOVERY")
        for biomarker in self.biomarkers:
            if biomarker in df.columns:
                temporal_result = self.temporal_pattern_discovery(df, biomarker)
                if temporal_result:
                    discoveries[f"temporal_{biomarker}"] = temporal_result
                    n_lags = len(temporal_result)
                    self.log_progress(f"DISCOVERY: {biomarker} shows {n_lags} significant lag patterns", "DISCOVERY")
        
        # 4. Climate variable discovery
        self.log_progress("\n🌡️ SYSTEMATIC CLIMATE VARIABLE DISCOVERY")
        for biomarker in self.biomarkers:
            if biomarker in df.columns:
                climate_result = self.climate_variable_discovery(df, biomarker)
                if climate_result:
                    discoveries[f"climate_{biomarker}"] = climate_result
                    climate_types = list(climate_result.keys())
                    self.log_progress(f"DISCOVERY: {biomarker} affected by {climate_types}", "DISCOVERY")
        
        # 5. Biomarker interaction discovery
        self.log_progress("\n🔗 BIOMARKER INTERACTION DISCOVERY")
        interaction_result = self.biomarker_interaction_discovery(df)
        if interaction_result:
            discoveries["biomarker_interactions"] = interaction_result
            n_interactions = len(interaction_result)
            self.log_progress(f"DISCOVERY: {n_interactions} significant biomarker-climate interactions", "DISCOVERY")
        
        # Generate report
        report = self.generate_robust_report(discoveries)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        self.log_progress("="*60)
        self.log_progress("✅ ROBUST DISCOVERY ANALYSIS COMPLETE")
        self.log_progress(f"Analysis time: {elapsed_time/60:.1f} minutes")
        self.log_progress(f"Total discoveries: {len(discoveries)}")
        
        return report

    def generate_robust_report(self, discoveries):
        """Generate comprehensive robust discovery report"""
        
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'Robust Alternative Discovery',
                'focus': 'Traditional epidemiology + robust demographics + systematic exploration'
            },
            'discoveries': discoveries,
            'summary': {
                'total_discovery_categories': len(discoveries),
                'traditional_epidemiology': len([k for k in discoveries.keys() if 'traditional' in k]),
                'sex_stratified': len([k for k in discoveries.keys() if 'sex_stratified' in k]),
                'temporal_patterns': len([k for k in discoveries.keys() if 'temporal' in k]),
                'climate_systematic': len([k for k in discoveries.keys() if 'climate' in k]),
                'biomarker_interactions': 1 if 'biomarker_interactions' in discoveries else 0
            }
        }
        
        # Save report
        report_file = self.results_dir / f"robust_discovery_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_progress(f"Report saved: {report_file}")
        
        return report

def main():
    """Execute robust alternative discovery"""
    discoverer = RobustAlternativeDiscovery()
    report = discoverer.run_comprehensive_robust_discovery()
    return report

if __name__ == "__main__":
    main()