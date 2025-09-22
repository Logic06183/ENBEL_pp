#!/usr/bin/env python3
"""
Focused Alternative Climate-Health Analysis

This is a streamlined version focusing on the most promising strategies
to quickly discover additional climate-health relationships.

Author: Climate-Health Analysis Pipeline
Date: 2025-09-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import json
import logging
from datetime import datetime
import os

class FocusedClimateHealthAnalysis:
    """
    Focused analysis pipeline for discovering climate-health relationships
    using the most promising alternative strategies.
    """
    
    def __init__(self, data_path, results_dir="focused_results"):
        """Initialize the analysis pipeline."""
        self.data_path = data_path
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.results_dir}/focused_analysis_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            'metadata': {
                'timestamp': self.timestamp,
                'focused_strategies': [
                    'composite_indices',
                    'interaction_effects', 
                    'temporal_patterns',
                    'alternative_features',
                    'subpopulation_highlights'
                ]
            },
            'discoveries': []
        }
        
        self.biomarkers = [
            'CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'FASTING LDL',
            'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'Creatinine (mg/dL)',
            'Hemoglobin (g/dL)', 'systolic blood pressure', 'diastolic blood pressure'
        ]
        
        self.load_data()
        
    def load_data(self):
        """Load and prepare the dataset."""
        self.logger.info("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} features")
        
        # Identify climate features
        climate_patterns = ['temperature', 'temp_', 'heat_', 'utci_', 'humidity', 'wind_']
        self.climate_features = [col for col in self.df.columns 
                               if any(pattern in col.lower() for pattern in climate_patterns)][:50]  # Limit for efficiency
        
        # Clean key variables
        for col in ['Sex', 'Race', 'season']:
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = LabelEncoder().fit_transform(
                    self.df[col].fillna('missing').astype(str)
                )
        
        # Fill climate features
        for feature in self.climate_features:
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].fillna(self.df[feature].median())
    
    def test_relationship(self, target, features, relationship_name):
        """Test a climate-health relationship."""
        # Prepare data
        data = pd.concat([target, features], axis=1).dropna()
        if len(data) < 50:
            return None
        
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
        
        # Quick test with XGBoost
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Feature importance
        importance = model.feature_importances_
        top_features = []
        for idx in np.argsort(importance)[-5:]:
            if idx < len(features.columns):
                top_features.append({
                    'feature': features.columns[idx],
                    'importance': float(importance[idx])
                })
        
        return {
            'relationship': relationship_name,
            'r2': r2,
            'mae': mae,
            'n_samples': len(data),
            'top_features': top_features[::-1]
        }
    
    def composite_indices_analysis(self):
        """Strategy 1: Composite health indices."""
        self.logger.info("Testing composite health indices...")
        
        discoveries = []
        
        # 1. Cardiovascular Risk Score
        cv_cols = ['systolic blood pressure', 'diastolic blood pressure']
        cv_data = self.df[cv_cols].dropna()
        
        if len(cv_data) > 100:
            # Create standardized composite
            scaler = StandardScaler()
            cv_scaled = scaler.fit_transform(cv_data)
            cv_composite = pd.Series(np.mean(cv_scaled, axis=1), index=cv_data.index, name='cv_risk')
            
            # Test climate relationships
            climate_subset = self.df.loc[cv_data.index, self.climate_features[:20]]
            result = self.test_relationship(cv_composite, climate_subset, 'Cardiovascular Risk ~ Climate')
            
            if result and result['r2'] > 0.05:
                discoveries.append(result)
                self.logger.info(f"Found CV risk relationship: R² = {result['r2']:.3f}")
        
        # 2. Metabolic Syndrome Index
        metabolic_cols = ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING LDL']
        metabolic_data = self.df[metabolic_cols].dropna()
        
        if len(metabolic_data) > 100:
            scaler = StandardScaler()
            metabolic_scaled = scaler.fit_transform(metabolic_data)
            # Weight HDL inversely (higher HDL is better)
            weights = [1, 1, -1, 1]
            metabolic_composite = pd.Series(
                np.average(metabolic_scaled, axis=1, weights=weights), 
                index=metabolic_data.index, 
                name='metabolic_risk'
            )
            
            climate_subset = self.df.loc[metabolic_data.index, self.climate_features[:20]]
            result = self.test_relationship(metabolic_composite, climate_subset, 'Metabolic Risk ~ Climate')
            
            if result and result['r2'] > 0.05:
                discoveries.append(result)
                self.logger.info(f"Found metabolic risk relationship: R² = {result['r2']:.3f}")
        
        return discoveries
    
    def interaction_effects_analysis(self):
        """Strategy 2: Interaction effects."""
        self.logger.info("Testing interaction effects...")
        
        discoveries = []
        
        # Key climate variables
        key_climate = ['temperature', 'humidity', 'heat_index']
        available_climate = [var for var in key_climate if var in self.df.columns]
        
        if not available_climate:
            available_climate = [col for col in self.climate_features if 'temp' in col.lower()][:2]
        
        # Test interactions with demographics
        for biomarker in self.biomarkers[:3]:  # Limit for efficiency
            if biomarker not in self.df.columns:
                continue
                
            for climate_var in available_climate[:1]:  # Just test one climate variable
                for demo_var in ['Sex_encoded', 'Race_encoded']:
                    if demo_var not in self.df.columns:
                        continue
                    
                    # Create interaction dataset
                    interaction_data = self.df[[biomarker, climate_var, demo_var]].dropna()
                    
                    if len(interaction_data) < 100:
                        continue
                    
                    # Create interaction term
                    interaction_data = interaction_data.copy()
                    interaction_data['interaction'] = interaction_data[climate_var] * interaction_data[demo_var]
                    
                    # Test model with interaction
                    target = interaction_data[biomarker]
                    features = interaction_data[[climate_var, demo_var, 'interaction']]
                    
                    result = self.test_relationship(target, features, 
                                                  f'{biomarker} ~ {climate_var} × {demo_var}')
                    
                    if result and result['r2'] > 0.05:
                        discoveries.append(result)
                        self.logger.info(f"Found interaction: {result['relationship']}, R² = {result['r2']:.3f}")
        
        return discoveries
    
    def temporal_patterns_analysis(self):
        """Strategy 3: Temporal patterns."""
        self.logger.info("Testing temporal patterns...")
        
        discoveries = []
        
        # Seasonal analysis
        if 'season_encoded' in self.df.columns:
            for biomarker in self.biomarkers[:3]:
                if biomarker not in self.df.columns:
                    continue
                
                # Test each season separately
                for season in self.df['season_encoded'].unique():
                    if pd.isna(season):
                        continue
                    
                    season_data = self.df[self.df['season_encoded'] == season]
                    if len(season_data) < 100:
                        continue
                    
                    target = season_data[biomarker].dropna()
                    if len(target) < 50:
                        continue
                    
                    climate_subset = season_data.loc[target.index, self.climate_features[:15]]
                    result = self.test_relationship(target, climate_subset, 
                                                  f'{biomarker} ~ Climate (Season {season})')
                    
                    if result and result['r2'] > 0.08:  # Higher threshold for seasonal
                        discoveries.append(result)
                        self.logger.info(f"Found seasonal pattern: {result['relationship']}, R² = {result['r2']:.3f}")
        
        # Extreme weather analysis
        if 'temperature' in self.df.columns:
            temp_p95 = self.df['temperature'].quantile(0.95)
            extreme_heat = self.df['temperature'] > temp_p95
            
            for biomarker in self.biomarkers[:3]:
                if biomarker not in self.df.columns:
                    continue
                
                heat_values = self.df[extreme_heat][biomarker].dropna()
                normal_values = self.df[~extreme_heat][biomarker].dropna()
                
                if len(heat_values) > 20 and len(normal_values) > 20:
                    t_stat, p_value = stats.ttest_ind(heat_values, normal_values)
                    
                    if p_value < 0.05:
                        effect_size = (heat_values.mean() - normal_values.mean()) / normal_values.std()
                        
                        discoveries.append({
                            'relationship': f'{biomarker} ~ Extreme Heat',
                            'effect_size': effect_size,
                            'p_value': p_value,
                            'heat_mean': heat_values.mean(),
                            'normal_mean': normal_values.mean(),
                            'n_heat': len(heat_values),
                            'n_normal': len(normal_values)
                        })
                        
                        self.logger.info(f"Found extreme heat effect: {biomarker}, p = {p_value:.3f}")
        
        return discoveries
    
    def alternative_features_analysis(self):
        """Strategy 4: Alternative climate features."""
        self.logger.info("Testing alternative climate features...")
        
        discoveries = []
        
        # Create new features
        if 'temperature' in self.df.columns:
            # Temperature variability
            self.df['temp_variability_7d'] = self.df['temperature'].rolling(window=7, min_periods=1).std()
            
            # Temperature change
            self.df['temp_change_1d'] = self.df['temperature'].diff()
        
        # Heat stress composite
        heat_vars = []
        for var in ['heat_index', 'temperature', 'humidity']:
            if var in self.df.columns:
                heat_vars.append(var)
        
        if len(heat_vars) >= 2:
            heat_data = self.df[heat_vars].fillna(0)
            scaler = StandardScaler()
            heat_scaled = scaler.fit_transform(heat_data)
            self.df['heat_stress_composite'] = np.mean(heat_scaled, axis=1)
        
        # Test new features
        new_features = ['temp_variability_7d', 'temp_change_1d', 'heat_stress_composite']
        
        for biomarker in self.biomarkers[:4]:  # Test more biomarkers for alt features
            if biomarker not in self.df.columns:
                continue
                
            for new_feature in new_features:
                if new_feature not in self.df.columns:
                    continue
                
                # Single feature test
                data = self.df[[biomarker, new_feature]].dropna()
                
                if len(data) < 50:
                    continue
                
                correlation = data[biomarker].corr(data[new_feature])
                
                if abs(correlation) > 0.1:  # Meaningful correlation
                    target = data[biomarker]
                    features = data[[new_feature]]
                    
                    result = self.test_relationship(target, features, f'{biomarker} ~ {new_feature}')
                    
                    if result and result['r2'] > 0.02:
                        result['correlation'] = correlation
                        discoveries.append(result)
                        self.logger.info(f"Found alt feature: {result['relationship']}, R² = {result['r2']:.3f}")
        
        return discoveries
    
    def subpopulation_highlights(self):
        """Strategy 5: Key subpopulation differences."""
        self.logger.info("Testing key subpopulations...")
        
        discoveries = []
        
        # Sex-specific effects
        if 'Sex_encoded' in self.df.columns:
            for biomarker in ['systolic blood pressure', 'Hemoglobin (g/dL)', 'Creatinine (mg/dL)']:
                if biomarker not in self.df.columns:
                    continue
                
                for sex in self.df['Sex_encoded'].unique():
                    if pd.isna(sex):
                        continue
                    
                    sex_data = self.df[self.df['Sex_encoded'] == sex]
                    if len(sex_data) < 100:
                        continue
                    
                    target = sex_data[biomarker].dropna()
                    if len(target) < 50:
                        continue
                    
                    climate_subset = sex_data.loc[target.index, self.climate_features[:15]]
                    result = self.test_relationship(target, climate_subset, 
                                                  f'{biomarker} ~ Climate (Sex {sex})')
                    
                    if result and result['r2'] > 0.08:
                        discoveries.append(result)
                        self.logger.info(f"Found sex-specific effect: {result['relationship']}, R² = {result['r2']:.3f}")
        
        return discoveries
    
    def run_focused_analysis(self):
        """Run the focused analysis pipeline."""
        self.logger.info("Starting focused climate-health analysis...")
        
        all_discoveries = []
        
        # Run each strategy
        strategies = [
            ('Composite Indices', self.composite_indices_analysis),
            ('Interaction Effects', self.interaction_effects_analysis),
            ('Temporal Patterns', self.temporal_patterns_analysis),
            ('Alternative Features', self.alternative_features_analysis),
            ('Subpopulation Highlights', self.subpopulation_highlights)
        ]
        
        for strategy_name, strategy_func in strategies:
            self.logger.info(f"Running {strategy_name}...")
            try:
                discoveries = strategy_func()
                all_discoveries.extend(discoveries)
                self.logger.info(f"{strategy_name}: Found {len(discoveries)} relationships")
            except Exception as e:
                self.logger.error(f"Error in {strategy_name}: {str(e)}")
        
        # Summarize results
        self.results['discoveries'] = all_discoveries
        self.results['summary'] = {
            'total_discoveries': len(all_discoveries),
            'strategies_tested': len(strategies),
            'significant_relationships': [d for d in all_discoveries if d.get('r2', 0) > 0.05 or d.get('p_value', 1) < 0.05]
        }
        
        # Save results
        results_file = f"{self.results_dir}/focused_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis complete. Found {len(all_discoveries)} total relationships")
        self.logger.info(f"Significant relationships: {len(self.results['summary']['significant_relationships'])}")
        
        return self.results


def main():
    """Main execution function."""
    print("=" * 80)
    print("FOCUSED ALTERNATIVE CLIMATE-HEALTH ANALYSIS")
    print("=" * 80)
    
    data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv"
    
    analyzer = FocusedClimateHealthAnalysis(data_path)
    results = analyzer.run_focused_analysis()
    
    # Print detailed results
    print(f"\nANALYSIS RESULTS:")
    print(f"Total discoveries: {results['summary']['total_discoveries']}")
    print(f"Significant relationships: {len(results['summary']['significant_relationships'])}")
    
    print(f"\nSIGNIFICANT RELATIONSHIPS:")
    for i, discovery in enumerate(results['summary']['significant_relationships'][:10], 1):
        print(f"{i}. {discovery['relationship']}")
        if 'r2' in discovery:
            print(f"   R² = {discovery['r2']:.3f}, n = {discovery.get('n_samples', 'N/A')}")
        if 'p_value' in discovery:
            print(f"   p-value = {discovery['p_value']:.3f}")
        if 'top_features' in discovery and discovery['top_features']:
            print(f"   Top feature: {discovery['top_features'][0]['feature']}")
        print()
    
    print(f"\nNOVEL DISCOVERIES WITH R² > 0.10:")
    high_r2 = [d for d in results['discoveries'] if d.get('r2', 0) > 0.10]
    for discovery in high_r2:
        print(f"- {discovery['relationship']}: R² = {discovery['r2']:.3f}")
    
    return results


if __name__ == "__main__":
    results = main()