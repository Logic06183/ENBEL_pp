#!/usr/bin/env python3
"""
Expanded Climate-Health Discovery Analysis

This module systematically tests additional climate variables, lagged effects,
and climate indices to discover more climate-health relationships.

Author: Climate-Health Analysis Pipeline
Date: 2025-09-19
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import KMeans
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import json
import logging
from datetime import datetime
import os

class ExpandedDiscoveryAnalysis:
    """
    Expanded analysis to systematically discover more climate-health relationships.
    """
    
    def __init__(self, data_path, results_dir="expanded_results"):
        """Initialize the expanded analysis."""
        self.data_path = data_path
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.results_dir}/expanded_analysis_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'expanded_systematic_discovery'
            },
            'discoveries': [],
            'climate_indices': {},
            'lagged_effects': {},
            'advanced_interactions': {},
            'summary': {}
        }
        
        self.biomarkers = [
            'CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'FASTING LDL',
            'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'Creatinine (mg/dL)',
            'Hemoglobin (g/dL)', 'systolic blood pressure', 'diastolic blood pressure'
        ]
        
        self.load_and_enhance_data()
        
    def load_and_enhance_data(self):
        """Load data and create enhanced climate features."""
        self.logger.info("Loading and enhancing dataset...")
        
        self.df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(self.df)} records")
        
        # Identify all climate features
        climate_patterns = [
            'temperature', 'temp_', 'heat_', 'utci_', 'wbgt_', 'apparent_temp',
            'humidity', 'wind_', 'cooling_degree', 'heating_degree', 'saaqis_era5'
        ]
        
        self.climate_features = []
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in climate_patterns):
                self.climate_features.append(col)
        
        self.logger.info(f"Identified {len(self.climate_features)} climate features")
        
        # Create enhanced climate indices
        self.create_climate_indices()
        
        # Encode categorical variables
        for col in ['Sex', 'Race', 'season']:
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = LabelEncoder().fit_transform(
                    self.df[col].fillna('missing').astype(str)
                )
        
        # Fill missing values
        for feature in self.climate_features:
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].fillna(self.df[feature].median())
    
    def create_climate_indices(self):
        """Create comprehensive climate indices."""
        self.logger.info("Creating climate indices...")
        
        # 1. Heat Stress Index (combining temperature, humidity, heat index)
        heat_vars = []
        for var in ['temperature', 'humidity', 'heat_index', 'utci_lag0', 'wbgt_lag0']:
            if var in self.df.columns:
                heat_vars.append(var)
        
        if len(heat_vars) >= 2:
            heat_data = self.df[heat_vars].fillna(0)
            scaler = StandardScaler()
            heat_scaled = scaler.fit_transform(heat_data)
            self.df['comprehensive_heat_stress'] = np.mean(heat_scaled, axis=1)
            self.climate_features.append('comprehensive_heat_stress')
        
        # 2. Temperature Variability Index
        temp_vars = [col for col in self.climate_features if 'temp' in col.lower() and 'lag' not in col][:5]
        if len(temp_vars) >= 3:
            temp_data = self.df[temp_vars].fillna(method='ffill')
            self.df['temperature_variability_index'] = temp_data.std(axis=1)
            self.climate_features.append('temperature_variability_index')
        
        # 3. Climate Change Indicator (long-term temperature trend)
        if 'year' in self.df.columns and 'temperature' in self.df.columns:
            yearly_temp = self.df.groupby('year')['temperature'].mean()
            year_temp_map = yearly_temp.to_dict()
            self.df['yearly_temp_anomaly'] = self.df['year'].map(year_temp_map) - yearly_temp.mean()
            self.climate_features.append('yearly_temp_anomaly')
        
        # 4. Extreme Weather Composite
        extreme_indicators = []
        for col in self.climate_features:
            if 'extreme' in col.lower() or 'strong_heat_stress' in col:
                extreme_indicators.append(col)
        
        if len(extreme_indicators) >= 2:
            extreme_data = self.df[extreme_indicators].fillna(0)
            self.df['extreme_weather_composite'] = extreme_data.sum(axis=1)
            self.climate_features.append('extreme_weather_composite')
        
        # 5. Diurnal Temperature Range
        if 'temperature_max' in self.df.columns and 'temperature_min' in self.df.columns:
            self.df['diurnal_temp_range'] = self.df['temperature_max'] - self.df['temperature_min']
            self.climate_features.append('diurnal_temp_range')
        
        # 6. Wind-Temperature Interaction
        if 'wind_speed' in self.df.columns and 'temperature' in self.df.columns:
            self.df['wind_temp_interaction'] = self.df['wind_speed'] * self.df['temperature']
            self.climate_features.append('wind_temp_interaction')
        
        self.logger.info(f"Created climate indices. Total features: {len(self.climate_features)}")
    
    def test_comprehensive_relationship(self, target, features, relationship_name, model_type='xgb'):
        """Comprehensive relationship testing with multiple models."""
        
        # Prepare data
        data = pd.concat([target, features], axis=1).dropna()
        if len(data) < 50:
            return None
        
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
        
        if X.shape[1] == 0:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        # Test multiple models
        models = {
            'xgb': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        for model_name, model in models.items():
            try:
                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[model_name] = {
                    'r2': r2,
                    'mae': mae,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                else:
                    importance = np.zeros(X.shape[1])
                
                top_features = []
                for idx in np.argsort(importance)[-5:]:
                    if idx < len(features.columns):
                        top_features.append({
                            'feature': features.columns[idx],
                            'importance': float(importance[idx])
                        })
                
                results[model_name]['top_features'] = top_features[::-1]
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        # Find best model
        best_r2 = -np.inf
        best_model = None
        for model_name, model_results in results.items():
            if 'r2' in model_results and model_results['r2'] > best_r2:
                best_r2 = model_results['r2']
                best_model = model_name
        
        if best_model:
            return {
                'relationship': relationship_name,
                'best_model': best_model,
                'best_r2': best_r2,
                'best_cv_r2': results[best_model]['cv_r2_mean'],
                'cv_r2_std': results[best_model]['cv_r2_std'],
                'mae': results[best_model]['mae'],
                'n_samples': len(data),
                'n_features': X.shape[1],
                'top_features': results[best_model]['top_features'],
                'all_models': results
            }
        
        return None
    
    def systematic_biomarker_climate_testing(self):
        """Systematically test each biomarker against climate features."""
        self.logger.info("Systematic biomarker-climate testing...")
        
        discoveries = []
        
        # Group climate features by type
        climate_groups = {
            'temperature': [col for col in self.climate_features if 'temp' in col.lower()],
            'heat_stress': [col for col in self.climate_features if any(term in col.lower() for term in ['heat', 'utci', 'wbgt'])],
            'humidity_wind': [col for col in self.climate_features if any(term in col.lower() for term in ['humidity', 'wind'])],
            'degree_days': [col for col in self.climate_features if 'degree' in col.lower()],
            'lagged_effects': [col for col in self.climate_features if 'lag' in col.lower()],
            'composites': [col for col in self.climate_features if any(term in col.lower() for term in ['composite', 'index', 'variability'])]
        }
        
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
                
            self.logger.info(f"Testing {biomarker}...")
            
            for group_name, climate_vars in climate_groups.items():
                if not climate_vars:
                    continue
                
                # Test with top variables from each group
                available_vars = [var for var in climate_vars if var in self.df.columns][:15]  # Limit for efficiency
                
                if len(available_vars) < 2:
                    continue
                
                target = self.df[biomarker].dropna()
                features = self.df.loc[target.index, available_vars]
                
                result = self.test_comprehensive_relationship(
                    target, features, f'{biomarker} ~ {group_name}'
                )
                
                if result and result['best_r2'] > 0.05:
                    discoveries.append(result)
                    self.logger.info(f"Found: {result['relationship']}, R² = {result['best_r2']:.3f}")
        
        return discoveries
    
    def advanced_interaction_testing(self):
        """Test advanced interaction effects."""
        self.logger.info("Testing advanced interactions...")
        
        discoveries = []
        
        # Climate-demographic interactions
        key_climate = ['temperature', 'comprehensive_heat_stress', 'temperature_variability_index']
        available_climate = [var for var in key_climate if var in self.df.columns]
        
        if not available_climate:
            available_climate = [col for col in self.climate_features if 'temp' in col.lower()][:3]
        
        demographic_vars = ['Sex_encoded', 'Race_encoded']
        available_demo = [var for var in demographic_vars if var in self.df.columns]
        
        for biomarker in ['FASTING GLUCOSE', 'systolic blood pressure', 'Hemoglobin (g/dL)', 'Creatinine (mg/dL)']:
            if biomarker not in self.df.columns:
                continue
            
            for climate_var in available_climate[:2]:
                for demo_var in available_demo:
                    
                    # Three-way interaction: climate × demo × season
                    if 'season_encoded' in self.df.columns:
                        interaction_data = self.df[[biomarker, climate_var, demo_var, 'season_encoded']].dropna()
                        
                        if len(interaction_data) < 100:
                            continue
                        
                        # Create interaction terms
                        interaction_data = interaction_data.copy()
                        interaction_data['climate_demo'] = interaction_data[climate_var] * interaction_data[demo_var]
                        interaction_data['climate_season'] = interaction_data[climate_var] * interaction_data['season_encoded']
                        interaction_data['three_way'] = interaction_data[climate_var] * interaction_data[demo_var] * interaction_data['season_encoded']
                        
                        target = interaction_data[biomarker]
                        features = interaction_data[[climate_var, demo_var, 'season_encoded', 'climate_demo', 'climate_season', 'three_way']]
                        
                        result = self.test_comprehensive_relationship(
                            target, features, f'{biomarker} ~ {climate_var} × {demo_var} × season'
                        )
                        
                        if result and result['best_r2'] > 0.08:
                            discoveries.append(result)
                            self.logger.info(f"Found three-way interaction: {result['relationship']}, R² = {result['best_r2']:.3f}")
        
        return discoveries
    
    def lagged_effects_analysis(self):
        """Comprehensive lagged effects analysis."""
        self.logger.info("Testing lagged effects...")
        
        discoveries = []
        
        # Group lagged features by lag period
        lag_groups = {}
        for col in self.climate_features:
            if 'lag' in col.lower():
                # Extract lag number
                for lag_num in ['0', '1', '2', '3', '5', '7', '10', '14', '21']:
                    if f'lag{lag_num}' in col or f'lag_{lag_num}' in col:
                        if lag_num not in lag_groups:
                            lag_groups[lag_num] = []
                        lag_groups[lag_num].append(col)
                        break
        
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
            
            for lag_period, lag_features in lag_groups.items():
                if len(lag_features) < 5:  # Need multiple features
                    continue
                
                # Test this lag period
                available_features = [f for f in lag_features if f in self.df.columns][:20]  # Limit features
                
                target = self.df[biomarker].dropna()
                features = self.df.loc[target.index, available_features]
                
                result = self.test_comprehensive_relationship(
                    target, features, f'{biomarker} ~ climate_lag_{lag_period}d'
                )
                
                if result and result['best_r2'] > 0.05:
                    discoveries.append(result)
                    self.logger.info(f"Found lagged effect: {result['relationship']}, R² = {result['best_r2']:.3f}")
        
        return discoveries
    
    def climate_phenotype_analysis(self):
        """Analyze health outcomes by climate phenotypes."""
        self.logger.info("Testing climate phenotype effects...")
        
        discoveries = []
        
        # Create climate phenotypes using clustering
        climate_data = self.df[self.climate_features[:30]].fillna(0)  # Use subset for efficiency
        
        scaler = StandardScaler()
        climate_scaled = scaler.fit_transform(climate_data)
        
        # K-means clustering for climate phenotypes
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        climate_phenotypes = kmeans.fit_predict(climate_scaled)
        
        self.df['climate_phenotype'] = climate_phenotypes
        
        # Test biomarker differences across phenotypes
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
            
            # Collect data for each phenotype
            phenotype_groups = []
            for phenotype in range(n_clusters):
                phenotype_data = self.df[self.df['climate_phenotype'] == phenotype][biomarker].dropna()
                if len(phenotype_data) > 20:
                    phenotype_groups.append(phenotype_data.values)
            
            if len(phenotype_groups) >= 3:
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*phenotype_groups)
                
                if p_value < 0.05:
                    # Effect size (eta-squared)
                    group_means = [np.mean(group) for group in phenotype_groups]
                    overall_mean = np.mean(np.concatenate(phenotype_groups))
                    
                    ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in phenotype_groups)
                    ss_total = sum((val - overall_mean)**2 for group in phenotype_groups for val in group)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    discoveries.append({
                        'relationship': f'{biomarker} ~ climate_phenotype',
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'eta_squared': eta_squared,
                        'n_phenotypes': len(phenotype_groups),
                        'phenotype_means': group_means,
                        'n_total': sum(len(group) for group in phenotype_groups)
                    })
                    
                    self.logger.info(f"Found phenotype effect: {biomarker}, p = {p_value:.3f}, η² = {eta_squared:.3f}")
        
        return discoveries
    
    def run_expanded_analysis(self):
        """Run the complete expanded analysis."""
        self.logger.info("Starting expanded discovery analysis...")
        
        all_discoveries = []
        
        # Run analysis components
        analyses = [
            ('Systematic Climate Testing', self.systematic_biomarker_climate_testing),
            ('Advanced Interactions', self.advanced_interaction_testing),
            ('Lagged Effects', self.lagged_effects_analysis),
            ('Climate Phenotypes', self.climate_phenotype_analysis)
        ]
        
        for analysis_name, analysis_func in analyses:
            self.logger.info(f"Running {analysis_name}...")
            try:
                discoveries = analysis_func()
                all_discoveries.extend(discoveries)
                self.logger.info(f"{analysis_name}: Found {len(discoveries)} relationships")
            except Exception as e:
                self.logger.error(f"Error in {analysis_name}: {str(e)}")
        
        # Compile results
        self.results['discoveries'] = all_discoveries
        
        # Filter significant results
        significant_r2 = [d for d in all_discoveries if d.get('best_r2', 0) > 0.05]
        significant_p = [d for d in all_discoveries if d.get('p_value', 1) < 0.05]
        
        high_quality = [d for d in all_discoveries if 
                       d.get('best_r2', 0) > 0.10 or 
                       (d.get('best_r2', 0) > 0.05 and d.get('best_cv_r2', 0) > 0.03)]
        
        self.results['summary'] = {
            'total_discoveries': len(all_discoveries),
            'significant_r2': len(significant_r2),
            'significant_p': len(significant_p),
            'high_quality': len(high_quality),
            'novel_discoveries': high_quality,
            'climate_features_tested': len(self.climate_features)
        }
        
        # Save results
        results_file = f"{self.results_dir}/expanded_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis complete! Found {len(all_discoveries)} total relationships")
        self.logger.info(f"High-quality discoveries: {len(high_quality)}")
        
        return self.results


def main():
    """Main execution function."""
    print("=" * 80)
    print("EXPANDED CLIMATE-HEALTH DISCOVERY ANALYSIS")
    print("=" * 80)
    
    data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv"
    
    analyzer = ExpandedDiscoveryAnalysis(data_path)
    results = analyzer.run_expanded_analysis()
    
    # Print comprehensive results
    summary = results['summary']
    
    print(f"\nCOMPREHENSIVE RESULTS:")
    print(f"Total discoveries: {summary['total_discoveries']}")
    print(f"Significant (R² > 0.05): {summary['significant_r2']}")
    print(f"Significant (p < 0.05): {summary['significant_p']}")
    print(f"High-quality discoveries: {summary['high_quality']}")
    print(f"Climate features tested: {summary['climate_features_tested']}")
    
    print(f"\nHIGH-QUALITY DISCOVERIES:")
    for i, discovery in enumerate(summary['novel_discoveries'][:15], 1):
        print(f"{i}. {discovery['relationship']}")
        if 'best_r2' in discovery:
            print(f"   R² = {discovery['best_r2']:.3f} (CV: {discovery.get('best_cv_r2', 'N/A'):.3f})")
            print(f"   Model: {discovery['best_model']}, n = {discovery['n_samples']}")
        if 'eta_squared' in discovery:
            print(f"   η² = {discovery['eta_squared']:.3f}, p = {discovery['p_value']:.3f}")
        
        # Show top climate feature
        if 'top_features' in discovery and discovery['top_features']:
            print(f"   Key feature: {discovery['top_features'][0]['feature']}")
        print()
    
    print(f"\nNOVEL CLIMATE-HEALTH RELATIONSHIPS DISCOVERED:")
    strong_relationships = [d for d in results['discoveries'] if d.get('best_r2', 0) > 0.15]
    for rel in strong_relationships:
        print(f"- {rel['relationship']}: R² = {rel['best_r2']:.3f}")
    
    return results


if __name__ == "__main__":
    results = main()