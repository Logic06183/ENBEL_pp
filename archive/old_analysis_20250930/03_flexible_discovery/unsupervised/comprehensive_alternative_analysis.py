#!/usr/bin/env python3
"""
Comprehensive Alternative Climate-Health Analysis Pipeline

This module implements multiple alternative strategies to uncover additional 
climate-health relationships beyond single biomarker approaches.

Strategies implemented:
1. Composite Health Indices
2. Multi-Output Models
3. Subpopulation Analysis
4. Temporal Pattern Analysis
5. Interaction Effects
6. Alternative Feature Engineering
7. Unsupervised Approaches
8. Causal Inference

Author: Climate-Health Analysis Pipeline
Date: 2025-09-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

import joblib
import json
import logging
from datetime import datetime
import os

class ComprehensiveClimateHealthAnalysis:
    """
    Comprehensive analysis pipeline for discovering climate-health relationships
    using multiple alternative strategies.
    """
    
    def __init__(self, data_path, results_dir="alternative_results"):
        """
        Initialize the comprehensive analysis pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the climate-health dataset
        results_dir : str
            Directory to save results
        """
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
                logging.FileHandler(f'{self.results_dir}/comprehensive_analysis_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize results storage
        self.results = {
            'metadata': {
                'timestamp': self.timestamp,
                'strategies_implemented': [
                    'composite_indices',
                    'multi_output_models', 
                    'subpopulation_analysis',
                    'temporal_patterns',
                    'interaction_effects',
                    'alternative_features',
                    'unsupervised_approaches',
                    'causal_inference'
                ]
            },
            'composite_indices': {},
            'multi_output_models': {},
            'subpopulation_analysis': {},
            'temporal_patterns': {},
            'interaction_effects': {},
            'alternative_features': {},
            'unsupervised_approaches': {},
            'causal_inference': {},
            'summary': {}
        }
        
        # Define biomarker groups
        self.biomarker_groups = {
            'cardiovascular': [
                'systolic blood pressure', 
                'diastolic blood pressure'
            ],
            'metabolic': [
                'FASTING GLUCOSE', 
                'FASTING TOTAL CHOLESTEROL', 
                'FASTING HDL', 
                'FASTING LDL'
            ],
            'immune': [
                'CD4 cell count (cells/µL)'
            ],
            'hematologic': [
                'Hemoglobin (g/dL)'
            ],
            'renal': [
                'Creatinine (mg/dL)'
            ]
        }
        
        # Load and prepare data
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for analysis."""
        self.logger.info("Loading and preparing dataset...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded dataset with {len(self.df)} records and {len(self.df.columns)} features")
        
        # Identify biomarkers
        self.biomarkers = [
            'CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'FASTING LDL',
            'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'Creatinine (mg/dL)',
            'Hemoglobin (g/dL)', 'systolic blood pressure', 'diastolic blood pressure'
        ]
        
        # Identify climate features
        climate_patterns = [
            'temperature', 'temp_', 'heat_', 'utci_', 'wbgt_', 'apparent_temp',
            'humidity', 'wind_', 'cooling_degree', 'heating_degree', 'saaqis_era5'
        ]
        
        self.climate_features = []
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in climate_patterns):
                self.climate_features.append(col)
        
        self.logger.info(f"Identified {len(self.climate_features)} climate features")
        
        # Identify demographic and socioeconomic features
        self.demographic_features = ['Sex', 'Race', 'latitude', 'longitude', 'year', 'month', 'season']
        self.socioeconomic_features = [col for col in self.df.columns if 'vuln_' in col or 
                                     col in ['Education', 'employment_status', 'housing_vulnerability', 
                                           'economic_vulnerability', 'heat_vulnerability_index']]
        
        # Clean data
        self.clean_data()
        
    def clean_data(self):
        """Clean and preprocess the data."""
        self.logger.info("Cleaning and preprocessing data...")
        
        # Handle missing values in biomarkers
        for biomarker in self.biomarkers:
            if biomarker in self.df.columns:
                # Remove extreme outliers (beyond 3 standard deviations)
                mean_val = self.df[biomarker].mean()
                std_val = self.df[biomarker].std()
                self.df[biomarker] = self.df[biomarker].where(
                    (self.df[biomarker] >= mean_val - 3*std_val) & 
                    (self.df[biomarker] <= mean_val + 3*std_val)
                )
        
        # Encode categorical variables
        self.label_encoders = {}
        categorical_cols = ['Sex', 'Race', 'season']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                valid_mask = self.df[col].notna()
                # Convert to string first to handle mixed types
                valid_data = self.df.loc[valid_mask, col].astype(str)
                self.df.loc[valid_mask, f'{col}_encoded'] = le.fit_transform(valid_data)
                self.label_encoders[col] = le
        
        # Fill missing climate features with median
        for feature in self.climate_features:
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].fillna(self.df[feature].median())
    
    def create_composite_indices(self):
        """
        Strategy 1: Create composite health indices and test climate relationships.
        """
        self.logger.info("Strategy 1: Creating composite health indices...")
        
        composite_results = {}
        
        # 1. Cardiovascular Risk Score
        cv_biomarkers = ['systolic blood pressure', 'diastolic blood pressure']
        cv_data = self.df[cv_biomarkers].dropna()
        
        if len(cv_data) > 100:
            # Standardize and create composite
            scaler = StandardScaler()
            cv_scaled = scaler.fit_transform(cv_data)
            cv_composite = np.mean(cv_scaled, axis=1)
            
            # Test climate relationships
            cv_climate_data = self.df.loc[cv_data.index, self.climate_features].fillna(0)
            cv_results = self.test_climate_relationships(cv_composite, cv_climate_data, 'cardiovascular_risk')
            composite_results['cardiovascular_risk'] = cv_results
            
        # 2. Metabolic Syndrome Index
        metabolic_biomarkers = ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 
                               'FASTING HDL', 'FASTING LDL']
        metabolic_data = self.df[metabolic_biomarkers].dropna()
        
        if len(metabolic_data) > 100:
            scaler = StandardScaler()
            metabolic_scaled = scaler.fit_transform(metabolic_data)
            # Weight HDL inversely (higher is better)
            weights = [1, 1, -1, 1]
            metabolic_composite = np.average(metabolic_scaled, axis=1, weights=weights)
            
            metabolic_climate_data = self.df.loc[metabolic_data.index, self.climate_features].fillna(0)
            metabolic_results = self.test_climate_relationships(metabolic_composite, metabolic_climate_data, 'metabolic_syndrome')
            composite_results['metabolic_syndrome'] = metabolic_results
            
        # 3. Kidney Function Composite
        renal_biomarkers = ['Creatinine (mg/dL)']
        if 'Hemoglobin (g/dL)' in self.df.columns:
            # Hemoglobin can indicate kidney function
            renal_data = self.df[['Creatinine (mg/dL)', 'Hemoglobin (g/dL)']].dropna()
            
            if len(renal_data) > 100:
                scaler = StandardScaler()
                renal_scaled = scaler.fit_transform(renal_data)
                # Weight creatinine positively (higher is worse), hemoglobin negatively (lower is worse)
                renal_composite = renal_scaled[:, 0] - renal_scaled[:, 1]
                
                renal_climate_data = self.df.loc[renal_data.index, self.climate_features].fillna(0)
                renal_results = self.test_climate_relationships(renal_composite, renal_climate_data, 'kidney_function')
                composite_results['kidney_function'] = renal_results
        
        self.results['composite_indices'] = composite_results
        self.logger.info(f"Completed composite indices analysis with {len(composite_results)} indices")
        
    def test_climate_relationships(self, target, climate_data, target_name):
        """Test climate relationships for a given target variable."""
        
        # Prepare data
        X = climate_data.values
        y = target
        
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 50:
            return {'error': 'Insufficient data after cleaning'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        results = {
            'target_name': target_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'models': {}
        }
        
        for model_name, model in models.items():
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    importances = np.zeros(X.shape[1])
                
                # Get top features
                feature_names = climate_data.columns
                top_features = []
                for idx in np.argsort(importances)[-10:]:
                    top_features.append({
                        'feature': feature_names[idx],
                        'importance': float(importances[idx])
                    })
                
                results['models'][model_name] = {
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'top_features': top_features[::-1]  # Reverse for descending order
                }
                
            except Exception as e:
                results['models'][model_name] = {'error': str(e)}
        
        # Find best model
        best_r2 = -np.inf
        best_model = None
        for model_name, model_results in results['models'].items():
            if 'r2' in model_results and model_results['r2'] > best_r2:
                best_r2 = model_results['r2']
                best_model = model_name
        
        results['best_model'] = best_model
        results['best_r2'] = best_r2
        
        return results
    
    def multi_output_modeling(self):
        """
        Strategy 2: Multi-output models to predict multiple biomarkers simultaneously.
        """
        self.logger.info("Strategy 2: Building multi-output models...")
        
        # Prepare multi-output data
        valid_biomarkers = []
        for biomarker in self.biomarkers:
            if biomarker in self.df.columns and self.df[biomarker].notna().sum() > 100:
                valid_biomarkers.append(biomarker)
        
        if len(valid_biomarkers) < 2:
            self.logger.warning("Insufficient biomarkers for multi-output modeling")
            return
        
        # Get samples with data for multiple biomarkers
        multi_data = self.df[valid_biomarkers].dropna()
        self.logger.info(f"Multi-output data: {len(multi_data)} samples with {len(valid_biomarkers)} biomarkers")
        
        if len(multi_data) < 100:
            self.logger.warning("Insufficient samples for multi-output modeling")
            return
        
        # Get climate features for these samples
        climate_data = self.df.loc[multi_data.index, self.climate_features].fillna(0)
        
        # Prepare targets
        y = multi_data.values
        X = climate_data.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Multi-output models
        models = {
            'multi_rf': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            'multi_ridge': MultiOutputRegressor(Ridge(alpha=1.0))
        }
        
        multi_results = {}
        
        for model_name, model in models.items():
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics for each biomarker
                biomarker_results = {}
                for i, biomarker in enumerate(valid_biomarkers):
                    r2 = r2_score(y_test[:, i], y_pred[:, i])
                    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                    
                    biomarker_results[biomarker] = {
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse
                    }
                
                # Overall performance
                overall_r2 = np.mean([biomarker_results[b]['r2'] for b in valid_biomarkers])
                
                multi_results[model_name] = {
                    'overall_r2': overall_r2,
                    'biomarker_results': biomarker_results,
                    'n_samples': len(X),
                    'n_biomarkers': len(valid_biomarkers)
                }
                
                # Feature importance analysis
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    # Average feature importance across all biomarkers
                    avg_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                    
                    top_features = []
                    feature_names = climate_data.columns
                    for idx in np.argsort(avg_importance)[-15:]:
                        top_features.append({
                            'feature': feature_names[idx],
                            'importance': float(avg_importance[idx])
                        })
                    
                    multi_results[model_name]['shared_climate_features'] = top_features[::-1]
                
            except Exception as e:
                multi_results[model_name] = {'error': str(e)}
        
        self.results['multi_output_models'] = multi_results
        self.logger.info(f"Completed multi-output modeling with {len(multi_results)} models")
    
    def subpopulation_analysis(self):
        """
        Strategy 3: Analyze climate-health relationships in specific subpopulations.
        """
        self.logger.info("Strategy 3: Conducting subpopulation analysis...")
        
        subpop_results = {}
        
        # Define subpopulations
        stratification_vars = {
            'sex': 'Sex_encoded' if 'Sex_encoded' in self.df.columns else 'Sex',
            'race': 'Race_encoded' if 'Race_encoded' in self.df.columns else 'Race',
            'age_group': None,  # Will create from data if available
            'season': 'season_encoded' if 'season_encoded' in self.df.columns else 'season'
        }
        
        # Create age groups if available data suggests it
        if 'year' in self.df.columns:
            # Use year as proxy for temporal cohorts
            self.df['temporal_cohort'] = pd.cut(self.df['year'], bins=3, labels=['early', 'middle', 'late'])
            stratification_vars['temporal_cohort'] = 'temporal_cohort'
        
        # For each biomarker and stratification variable
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
                
            biomarker_results = {}
            
            for strat_name, strat_var in stratification_vars.items():
                if strat_var is None or strat_var not in self.df.columns:
                    continue
                
                strat_results = {}
                unique_values = self.df[strat_var].unique()
                
                for value in unique_values:
                    if pd.isna(value):
                        continue
                    
                    # Get subpopulation
                    subpop_mask = (self.df[strat_var] == value) & self.df[biomarker].notna()
                    subpop_data = self.df[subpop_mask]
                    
                    if len(subpop_data) < 50:  # Minimum sample size
                        continue
                    
                    # Test climate relationships in this subpopulation
                    target = subpop_data[biomarker].values
                    climate_data = subpop_data[self.climate_features].fillna(0)
                    
                    subpop_result = self.test_climate_relationships(target, climate_data, 
                                                                 f"{biomarker}_{strat_name}_{value}")
                    
                    if 'error' not in subpop_result:
                        strat_results[str(value)] = subpop_result
                
                if strat_results:
                    biomarker_results[strat_name] = strat_results
            
            if biomarker_results:
                subpop_results[biomarker] = biomarker_results
        
        self.results['subpopulation_analysis'] = subpop_results
        self.logger.info(f"Completed subpopulation analysis for {len(subpop_results)} biomarkers")
    
    def temporal_pattern_analysis(self):
        """
        Strategy 4: Analyze temporal patterns in climate-health relationships.
        """
        self.logger.info("Strategy 4: Analyzing temporal patterns...")
        
        temporal_results = {}
        
        # 1. Seasonal Analysis
        if 'season' in self.df.columns or 'month' in self.df.columns:
            seasonal_results = {}
            
            for biomarker in self.biomarkers:
                if biomarker not in self.df.columns:
                    continue
                
                biomarker_seasonal = {}
                
                # Monthly patterns
                if 'month' in self.df.columns:
                    monthly_stats = []
                    for month in range(1, 13):
                        month_data = self.df[self.df['month'] == month]
                        if len(month_data) > 10 and biomarker in month_data.columns:
                            month_mean = month_data[biomarker].mean()
                            month_std = month_data[biomarker].std()
                            monthly_stats.append({
                                'month': month,
                                'mean': month_mean,
                                'std': month_std,
                                'n': len(month_data[month_data[biomarker].notna()])
                            })
                    
                    biomarker_seasonal['monthly_patterns'] = monthly_stats
                
                # Seasonal climate sensitivity
                if 'season_encoded' in self.df.columns or 'season' in self.df.columns:
                    season_col = 'season_encoded' if 'season_encoded' in self.df.columns else 'season'
                    season_sensitivity = {}
                    
                    for season in self.df[season_col].unique():
                        if pd.isna(season):
                            continue
                        
                        season_data = self.df[self.df[season_col] == season]
                        if len(season_data) < 30:
                            continue
                        
                        # Test climate relationships in this season
                        target = season_data[biomarker].dropna()
                        if len(target) < 30:
                            continue
                        
                        climate_data = season_data.loc[target.index, self.climate_features].fillna(0)
                        season_result = self.test_climate_relationships(target.values, climate_data, 
                                                                      f"{biomarker}_season_{season}")
                        
                        if 'error' not in season_result:
                            season_sensitivity[str(season)] = season_result
                    
                    biomarker_seasonal['seasonal_sensitivity'] = season_sensitivity
                
                if biomarker_seasonal:
                    seasonal_results[biomarker] = biomarker_seasonal
            
            temporal_results['seasonal_analysis'] = seasonal_results
        
        # 2. Extreme Weather Analysis
        extreme_results = {}
        
        # Identify extreme temperature days
        if 'temperature' in self.df.columns:
            temp_p95 = self.df['temperature'].quantile(0.95)
            temp_p5 = self.df['temperature'].quantile(0.05)
            
            self.df['extreme_heat'] = self.df['temperature'] > temp_p95
            self.df['extreme_cold'] = self.df['temperature'] < temp_p5
            
            for biomarker in self.biomarkers:
                if biomarker not in self.df.columns:
                    continue
                
                biomarker_extreme = {}
                
                # Heat wave effects
                heat_data = self.df[self.df['extreme_heat'] == True]
                normal_data = self.df[self.df['extreme_heat'] == False]
                
                if len(heat_data) > 20 and len(normal_data) > 20:
                    heat_mean = heat_data[biomarker].mean()
                    normal_mean = normal_data[biomarker].mean()
                    
                    # Statistical test
                    if not pd.isna(heat_mean) and not pd.isna(normal_mean):
                        heat_values = heat_data[biomarker].dropna()
                        normal_values = normal_data[biomarker].dropna()
                        
                        if len(heat_values) > 10 and len(normal_values) > 10:
                            t_stat, p_value = stats.ttest_ind(heat_values, normal_values)
                            
                            biomarker_extreme['heat_wave_effect'] = {
                                'heat_mean': heat_mean,
                                'normal_mean': normal_mean,
                                'difference': heat_mean - normal_mean,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'n_heat': len(heat_values),
                                'n_normal': len(normal_values)
                            }
                
                # Cold wave effects
                cold_data = self.df[self.df['extreme_cold'] == True]
                
                if len(cold_data) > 20:
                    cold_mean = cold_data[biomarker].mean()
                    
                    if not pd.isna(cold_mean) and not pd.isna(normal_mean):
                        cold_values = cold_data[biomarker].dropna()
                        
                        if len(cold_values) > 10 and len(normal_values) > 10:
                            t_stat, p_value = stats.ttest_ind(cold_values, normal_values)
                            
                            biomarker_extreme['cold_wave_effect'] = {
                                'cold_mean': cold_mean,
                                'normal_mean': normal_mean,
                                'difference': cold_mean - normal_mean,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'n_cold': len(cold_values),
                                'n_normal': len(normal_values)
                            }
                
                if biomarker_extreme:
                    extreme_results[biomarker] = biomarker_extreme
        
        temporal_results['extreme_weather_analysis'] = extreme_results
        
        self.results['temporal_patterns'] = temporal_results
        self.logger.info("Completed temporal pattern analysis")
    
    def interaction_effects_analysis(self):
        """
        Strategy 5: Test interaction effects between climate and demographic variables.
        """
        self.logger.info("Strategy 5: Analyzing interaction effects...")
        
        interaction_results = {}
        
        # Define key climate variables for interaction testing
        key_climate_vars = ['temperature', 'humidity', 'heat_index']
        available_climate = [var for var in key_climate_vars if var in self.df.columns]
        
        if not available_climate:
            # Use temperature-related variables
            temp_vars = [col for col in self.climate_features if 'temp' in col.lower()][:3]
            available_climate = temp_vars
        
        # Define demographic variables for interactions
        demo_vars = ['Sex_encoded', 'Race_encoded'] if 'Sex_encoded' in self.df.columns else ['Sex', 'Race']
        demo_vars = [var for var in demo_vars if var in self.df.columns]
        
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
            
            biomarker_interactions = {}
            
            for climate_var in available_climate[:2]:  # Limit to avoid excessive computation
                for demo_var in demo_vars:
                    
                    # Prepare interaction data
                    interaction_data = self.df[[biomarker, climate_var, demo_var]].dropna()
                    
                    if len(interaction_data) < 100:
                        continue
                    
                    # Create interaction term
                    interaction_data['interaction'] = interaction_data[climate_var] * interaction_data[demo_var]
                    
                    # Fit model with interaction
                    from sklearn.linear_model import LinearRegression
                    
                    X = interaction_data[[climate_var, demo_var, 'interaction']]
                    y = interaction_data[biomarker]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Model with interaction
                    model_interaction = LinearRegression()
                    model_interaction.fit(X_train, y_train)
                    y_pred_interaction = model_interaction.predict(X_test)
                    r2_interaction = r2_score(y_test, y_pred_interaction)
                    
                    # Model without interaction
                    model_main = LinearRegression()
                    model_main.fit(X_train[[climate_var, demo_var]], y_train)
                    y_pred_main = model_main.predict(X_test[[climate_var, demo_var]])
                    r2_main = r2_score(y_test, y_pred_main)
                    
                    # Interaction effect
                    interaction_effect = r2_interaction - r2_main
                    
                    interaction_key = f"{climate_var}_x_{demo_var}"
                    biomarker_interactions[interaction_key] = {
                        'r2_with_interaction': r2_interaction,
                        'r2_main_effects': r2_main,
                        'interaction_effect': interaction_effect,
                        'interaction_coefficient': model_interaction.coef_[2],
                        'n_samples': len(interaction_data)
                    }
            
            if biomarker_interactions:
                interaction_results[biomarker] = biomarker_interactions
        
        self.results['interaction_effects'] = interaction_results
        self.logger.info(f"Completed interaction effects analysis for {len(interaction_results)} biomarkers")
    
    def alternative_feature_engineering(self):
        """
        Strategy 6: Create alternative climate features and test relationships.
        """
        self.logger.info("Strategy 6: Engineering alternative climate features...")
        
        alternative_results = {}
        
        # 1. Climate Variability Features
        if 'temperature' in self.df.columns:
            # Rolling variability
            self.df['temp_variability_7d'] = self.df['temperature'].rolling(window=7, min_periods=1).std()
            self.df['temp_range_7d'] = self.df['temperature'].rolling(window=7, min_periods=1).max() - \
                                      self.df['temperature'].rolling(window=7, min_periods=1).min()
            
            # Diurnal temperature range if max/min available
            if 'temperature_max' in self.df.columns and 'temperature_min' in self.df.columns:
                self.df['diurnal_temp_range'] = self.df['temperature_max'] - self.df['temperature_min']
        
        # 2. Heat Stress Composite
        heat_stress_vars = []
        for var in ['heat_index', 'temperature', 'humidity']:
            if var in self.df.columns:
                heat_stress_vars.append(var)
        
        if len(heat_stress_vars) >= 2:
            heat_stress_data = self.df[heat_stress_vars].fillna(0)
            scaler = StandardScaler()
            heat_stress_scaled = scaler.fit_transform(heat_stress_data)
            self.df['heat_stress_composite'] = np.mean(heat_stress_scaled, axis=1)
        
        # 3. Climate Change Indicators
        if 'year' in self.df.columns and 'temperature' in self.df.columns:
            # Temperature trend over time
            self.df['temp_deviation_from_period_mean'] = self.df.groupby('year')['temperature'].transform(
                lambda x: x - x.mean()
            )
        
        # New alternative features to test
        alt_features = []
        for col in ['temp_variability_7d', 'temp_range_7d', 'diurnal_temp_range', 
                   'heat_stress_composite', 'temp_deviation_from_period_mean']:
            if col in self.df.columns:
                alt_features.append(col)
        
        # Test relationships with alternative features
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
            
            biomarker_alt_results = {}
            
            for alt_feature in alt_features:
                # Single feature analysis
                feature_data = self.df[[biomarker, alt_feature]].dropna()
                
                if len(feature_data) < 50:
                    continue
                
                # Simple correlation
                correlation = feature_data[biomarker].corr(feature_data[alt_feature])
                
                # Model performance
                X = feature_data[[alt_feature]]
                y = feature_data[biomarker]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Simple linear model
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                biomarker_alt_results[alt_feature] = {
                    'correlation': correlation,
                    'r2': r2,
                    'coefficient': model.coef_[0],
                    'n_samples': len(feature_data)
                }
            
            if biomarker_alt_results:
                alternative_results[biomarker] = biomarker_alt_results
        
        self.results['alternative_features'] = alternative_results
        self.logger.info(f"Completed alternative feature engineering for {len(alternative_results)} biomarkers")
    
    def unsupervised_analysis(self):
        """
        Strategy 7: Unsupervised approaches for pattern discovery.
        """
        self.logger.info("Strategy 7: Conducting unsupervised analysis...")
        
        unsupervised_results = {}
        
        # 1. Climate Phenotyping
        climate_data_clean = self.df[self.climate_features].fillna(0)
        
        # Reduce dimensionality for visualization
        scaler = StandardScaler()
        climate_scaled = scaler.fit_transform(climate_data_clean)
        
        # PCA
        pca = PCA(n_components=min(10, climate_scaled.shape[1]))
        climate_pca = pca.fit_transform(climate_scaled)
        
        # Cluster analysis
        n_clusters = 4  # Climate phenotypes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        climate_clusters = kmeans.fit_predict(climate_pca)
        
        self.df['climate_phenotype'] = climate_clusters
        
        # Analyze biomarker differences across climate phenotypes
        phenotype_results = {}
        
        for biomarker in self.biomarkers:
            if biomarker not in self.df.columns:
                continue
            
            phenotype_means = []
            phenotype_stds = []
            phenotype_ns = []
            
            for cluster in range(n_clusters):
                cluster_data = self.df[self.df['climate_phenotype'] == cluster][biomarker].dropna()
                
                if len(cluster_data) > 5:
                    phenotype_means.append(cluster_data.mean())
                    phenotype_stds.append(cluster_data.std())
                    phenotype_ns.append(len(cluster_data))
                else:
                    phenotype_means.append(np.nan)
                    phenotype_stds.append(np.nan)
                    phenotype_ns.append(0)
            
            # ANOVA test across phenotypes
            valid_groups = []
            for cluster in range(n_clusters):
                cluster_data = self.df[self.df['climate_phenotype'] == cluster][biomarker].dropna()
                if len(cluster_data) > 5:
                    valid_groups.append(cluster_data.values)
            
            if len(valid_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*valid_groups)
                
                phenotype_results[biomarker] = {
                    'phenotype_means': phenotype_means,
                    'phenotype_stds': phenotype_stds,
                    'phenotype_ns': phenotype_ns,
                    'anova_f_statistic': f_stat,
                    'anova_p_value': p_value
                }
        
        unsupervised_results['climate_phenotyping'] = {
            'n_clusters': n_clusters,
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'biomarker_differences': phenotype_results
        }
        
        # 2. Biomarker Clustering
        biomarker_data = self.df[self.biomarkers].dropna()
        
        if len(biomarker_data) > 50:
            # Correlation-based clustering
            biomarker_corr = biomarker_data.corr()
            
            # Hierarchical clustering of biomarkers
            from scipy.cluster.hierarchy import linkage, fcluster
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(biomarker_corr)
            
            # Perform clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Get clusters
            biomarker_clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
            
            biomarker_clustering = {}
            for i, biomarker in enumerate(self.biomarkers):
                if biomarker in biomarker_data.columns:
                    biomarker_clustering[biomarker] = int(biomarker_clusters[biomarker_data.columns.get_loc(biomarker)])
            
            unsupervised_results['biomarker_clustering'] = {
                'clusters': biomarker_clustering,
                'correlation_matrix': biomarker_corr.to_dict()
            }
        
        self.results['unsupervised_approaches'] = unsupervised_results
        self.logger.info("Completed unsupervised analysis")
    
    def causal_inference_analysis(self):
        """
        Strategy 8: Causal inference approaches using natural experiments.
        """
        self.logger.info("Strategy 8: Conducting causal inference analysis...")
        
        causal_results = {}
        
        # 1. Instrumental Variables using Geographic Variation
        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            
            for biomarker in self.biomarkers:
                if biomarker not in self.df.columns:
                    continue
                
                # Use geographic coordinates as instruments for climate exposure
                geo_data = self.df[['latitude', 'longitude', biomarker]].dropna()
                
                if len(geo_data) < 100:
                    continue
                
                # Add main climate variables
                climate_vars = ['temperature'] if 'temperature' in self.df.columns else []
                if 'humidity' in self.df.columns:
                    climate_vars.append('humidity')
                
                if not climate_vars:
                    continue
                
                full_data = self.df[['latitude', 'longitude', biomarker] + climate_vars].dropna()
                
                if len(full_data) < 100:
                    continue
                
                # Simple IV approach: use geographic variation
                # First stage: predict climate from geography
                from sklearn.linear_model import LinearRegression
                
                for climate_var in climate_vars:
                    # First stage
                    X_geo = full_data[['latitude', 'longitude']]
                    y_climate = full_data[climate_var]
                    
                    model_first = LinearRegression()
                    model_first.fit(X_geo, y_climate)
                    climate_predicted = model_first.predict(X_geo)
                    
                    # Second stage: predict biomarker from predicted climate
                    model_second = LinearRegression()
                    model_second.fit(climate_predicted.reshape(-1, 1), full_data[biomarker])
                    
                    # Calculate R-squared for instrument strength
                    r2_first_stage = r2_score(y_climate, climate_predicted)
                    
                    # Reduced form: direct effect of geography on biomarker
                    model_reduced = LinearRegression()
                    model_reduced.fit(X_geo, full_data[biomarker])
                    r2_reduced = r2_score(full_data[biomarker], model_reduced.predict(X_geo))
                    
                    causal_key = f"{biomarker}_{climate_var}_IV"
                    if biomarker not in causal_results:
                        causal_results[biomarker] = {}
                    
                    causal_results[biomarker][f'iv_analysis_{climate_var}'] = {
                        'iv_coefficient': model_second.coef_[0],
                        'first_stage_r2': r2_first_stage,
                        'reduced_form_r2': r2_reduced,
                        'instrument_strength': 'weak' if r2_first_stage < 0.1 else 'moderate' if r2_first_stage < 0.3 else 'strong',
                        'n_samples': len(full_data)
                    }
        
        # 2. Natural Experiments using Weather Shocks
        if 'temperature' in self.df.columns:
            
            # Identify temperature shocks (unusual deviations)
            self.df['temp_shock'] = np.abs(self.df['temperature'] - self.df['temperature'].rolling(30, min_periods=1).mean()) > \
                                   2 * self.df['temperature'].rolling(30, min_periods=1).std()
            
            for biomarker in self.biomarkers:
                if biomarker not in self.df.columns:
                    continue
                
                shock_data = self.df[['temp_shock', biomarker, 'temperature']].dropna()
                
                if len(shock_data) < 50:
                    continue
                
                # Compare outcomes during shocks vs normal periods
                shock_outcomes = shock_data[shock_data['temp_shock'] == True][biomarker]
                normal_outcomes = shock_data[shock_data['temp_shock'] == False][biomarker]
                
                if len(shock_outcomes) > 10 and len(normal_outcomes) > 10:
                    # Difference in means
                    shock_mean = shock_outcomes.mean()
                    normal_mean = normal_outcomes.mean()
                    difference = shock_mean - normal_mean
                    
                    # Statistical test
                    t_stat, p_value = stats.ttest_ind(shock_outcomes, normal_outcomes)
                    
                    if biomarker not in causal_results:
                        causal_results[biomarker] = {}
                    
                    causal_results[biomarker]['weather_shock_analysis'] = {
                        'shock_mean': shock_mean,
                        'normal_mean': normal_mean,
                        'causal_effect': difference,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'n_shock_periods': len(shock_outcomes),
                        'n_normal_periods': len(normal_outcomes)
                    }
        
        self.results['causal_inference'] = causal_results
        self.logger.info(f"Completed causal inference analysis for {len(causal_results)} biomarkers")
    
    def generate_summary(self):
        """Generate comprehensive summary of all analyses."""
        self.logger.info("Generating comprehensive summary...")
        
        summary = {
            'total_strategies_implemented': len(self.results['metadata']['strategies_implemented']),
            'significant_relationships': [],
            'strategy_performance': {},
            'novel_discoveries': [],
            'validation_summary': {}
        }
        
        # Define significance thresholds
        r2_threshold = 0.05  # Minimum meaningful R²
        p_value_threshold = 0.05
        
        # Analyze each strategy
        strategies = ['composite_indices', 'multi_output_models', 'subpopulation_analysis', 
                     'temporal_patterns', 'interaction_effects', 'alternative_features',
                     'unsupervised_approaches', 'causal_inference']
        
        total_relationships = 0
        significant_relationships = 0
        
        for strategy in strategies:
            if strategy not in self.results:
                continue
            
            strategy_data = self.results[strategy]
            strategy_significant = 0
            strategy_total = 0
            best_r2 = 0
            
            # Count relationships based on strategy type
            if strategy == 'composite_indices':
                for composite, results in strategy_data.items():
                    if 'error' not in results:
                        strategy_total += 1
                        total_relationships += 1
                        if results.get('best_r2', 0) > r2_threshold:
                            strategy_significant += 1
                            significant_relationships += 1
                            best_r2 = max(best_r2, results['best_r2'])
                            
                            summary['significant_relationships'].append({
                                'strategy': strategy,
                                'relationship': f"{composite} ~ climate",
                                'r2': results['best_r2'],
                                'model': results.get('best_model', 'unknown')
                            })
            
            elif strategy == 'multi_output_models':
                for model_name, results in strategy_data.items():
                    if 'error' not in results:
                        strategy_total += 1
                        total_relationships += 1
                        overall_r2 = results.get('overall_r2', 0)
                        if overall_r2 > r2_threshold:
                            strategy_significant += 1
                            significant_relationships += 1
                            best_r2 = max(best_r2, overall_r2)
                            
                            summary['significant_relationships'].append({
                                'strategy': strategy,
                                'relationship': f"multi_biomarker ~ climate ({model_name})",
                                'r2': overall_r2,
                                'model': model_name
                            })
            
            elif strategy == 'subpopulation_analysis':
                for biomarker, strat_data in strategy_data.items():
                    for strat_var, subpop_data in strat_data.items():
                        for subpop, results in subpop_data.items():
                            if 'error' not in results:
                                strategy_total += 1
                                total_relationships += 1
                                if results.get('best_r2', 0) > r2_threshold:
                                    strategy_significant += 1
                                    significant_relationships += 1
                                    best_r2 = max(best_r2, results['best_r2'])
                                    
                                    summary['significant_relationships'].append({
                                        'strategy': strategy,
                                        'relationship': f"{biomarker} ~ climate (subpop: {strat_var}={subpop})",
                                        'r2': results['best_r2'],
                                        'model': results.get('best_model', 'unknown')
                                    })
            
            elif strategy == 'temporal_patterns':
                # Check seasonal and extreme weather analyses
                if 'seasonal_analysis' in strategy_data:
                    for biomarker, seasonal_data in strategy_data['seasonal_analysis'].items():
                        if 'seasonal_sensitivity' in seasonal_data:
                            for season, results in seasonal_data['seasonal_sensitivity'].items():
                                if 'error' not in results:
                                    strategy_total += 1
                                    total_relationships += 1
                                    if results.get('best_r2', 0) > r2_threshold:
                                        strategy_significant += 1
                                        significant_relationships += 1
                                        best_r2 = max(best_r2, results['best_r2'])
                
                if 'extreme_weather_analysis' in strategy_data:
                    for biomarker, extreme_data in strategy_data['extreme_weather_analysis'].items():
                        for effect_type, effect_data in extreme_data.items():
                            if 'p_value' in effect_data and effect_data['p_value'] < p_value_threshold:
                                strategy_total += 1
                                total_relationships += 1
                                strategy_significant += 1
                                significant_relationships += 1
                                
                                summary['significant_relationships'].append({
                                    'strategy': strategy,
                                    'relationship': f"{biomarker} ~ {effect_type}",
                                    'effect_size': effect_data.get('difference', 0),
                                    'p_value': effect_data['p_value']
                                })
            
            elif strategy == 'interaction_effects':
                for biomarker, interaction_data in strategy_data.items():
                    for interaction, results in interaction_data.items():
                        strategy_total += 1
                        total_relationships += 1
                        interaction_effect = results.get('interaction_effect', 0)
                        if interaction_effect > 0.02:  # Meaningful interaction effect
                            strategy_significant += 1
                            significant_relationships += 1
                            
                            summary['significant_relationships'].append({
                                'strategy': strategy,
                                'relationship': f"{biomarker} ~ {interaction}",
                                'interaction_effect': interaction_effect,
                                'r2_with_interaction': results.get('r2_with_interaction', 0)
                            })
            
            elif strategy == 'alternative_features':
                for biomarker, alt_data in strategy_data.items():
                    for feature, results in alt_data.items():
                        strategy_total += 1
                        total_relationships += 1
                        if results.get('r2', 0) > r2_threshold:
                            strategy_significant += 1
                            significant_relationships += 1
                            best_r2 = max(best_r2, results['r2'])
                            
                            summary['significant_relationships'].append({
                                'strategy': strategy,
                                'relationship': f"{biomarker} ~ {feature}",
                                'r2': results['r2'],
                                'correlation': results.get('correlation', 0)
                            })
            
            elif strategy == 'unsupervised_approaches':
                if 'climate_phenotyping' in strategy_data:
                    phenotype_data = strategy_data['climate_phenotyping']
                    if 'biomarker_differences' in phenotype_data:
                        for biomarker, results in phenotype_data['biomarker_differences'].items():
                            strategy_total += 1
                            total_relationships += 1
                            if results.get('anova_p_value', 1) < p_value_threshold:
                                strategy_significant += 1
                                significant_relationships += 1
                                
                                summary['significant_relationships'].append({
                                    'strategy': strategy,
                                    'relationship': f"{biomarker} ~ climate_phenotype",
                                    'f_statistic': results.get('anova_f_statistic', 0),
                                    'p_value': results['anova_p_value']
                                })
            
            elif strategy == 'causal_inference':
                for biomarker, causal_data in strategy_data.items():
                    for analysis, results in causal_data.items():
                        strategy_total += 1
                        total_relationships += 1
                        
                        if 'p_value' in results and results['p_value'] < p_value_threshold:
                            strategy_significant += 1
                            significant_relationships += 1
                            
                            summary['significant_relationships'].append({
                                'strategy': strategy,
                                'relationship': f"{biomarker} ~ {analysis}",
                                'causal_effect': results.get('causal_effect', results.get('iv_coefficient', 0)),
                                'p_value': results.get('p_value', np.nan)
                            })
            
            # Strategy performance summary
            summary['strategy_performance'][strategy] = {
                'total_tested': strategy_total,
                'significant_found': strategy_significant,
                'success_rate': strategy_significant / strategy_total if strategy_total > 0 else 0,
                'best_r2': best_r2
            }
        
        # Overall summary
        summary['overall_performance'] = {
            'total_relationships_tested': total_relationships,
            'significant_relationships_found': significant_relationships,
            'overall_success_rate': significant_relationships / total_relationships if total_relationships > 0 else 0,
            'improvement_over_baseline': significant_relationships - 1  # Baseline was 1 significant relationship
        }
        
        # Identify novel discoveries
        for rel in summary['significant_relationships']:
            if rel.get('r2', 0) > 0.1 or rel.get('interaction_effect', 0) > 0.05:
                summary['novel_discoveries'].append(rel)
        
        self.results['summary'] = summary
        self.logger.info(f"Summary: Found {significant_relationships} significant relationships across {len(strategies)} strategies")
        
        return summary
    
    def run_comprehensive_analysis(self):
        """Run all analysis strategies."""
        self.logger.info("Starting comprehensive climate-health analysis...")
        
        try:
            # Update todo
            self.create_composite_indices()
            self.multi_output_modeling() 
            self.subpopulation_analysis()
            self.temporal_pattern_analysis()
            self.interaction_effects_analysis()
            self.alternative_feature_engineering()
            self.unsupervised_analysis()
            self.causal_inference_analysis()
            
            # Generate summary
            summary = self.generate_summary()
            
            # Save results
            results_file = f"{self.results_dir}/comprehensive_analysis_{self.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Analysis complete. Results saved to {results_file}")
            self.logger.info(f"Found {summary['overall_performance']['significant_relationships_found']} significant relationships")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise


def main():
    """Main execution function."""
    print("=" * 80)
    print("COMPREHENSIVE ALTERNATIVE CLIMATE-HEALTH ANALYSIS")
    print("=" * 80)
    
    # Initialize and run analysis
    data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv"
    
    analyzer = ComprehensiveClimateHealthAnalysis(data_path)
    results = analyzer.run_comprehensive_analysis()
    
    # Print summary
    summary = results['summary']
    print(f"\nANALYSIS SUMMARY:")
    print(f"Total relationships tested: {summary['overall_performance']['total_relationships_tested']}")
    print(f"Significant relationships found: {summary['overall_performance']['significant_relationships_found']}")
    print(f"Overall success rate: {summary['overall_performance']['overall_success_rate']:.3f}")
    print(f"Improvement over baseline: +{summary['overall_performance']['improvement_over_baseline']} relationships")
    
    print(f"\nSTRATEGY PERFORMANCE:")
    for strategy, perf in summary['strategy_performance'].items():
        print(f"{strategy}: {perf['significant_found']}/{perf['total_tested']} "
              f"(success rate: {perf['success_rate']:.3f})")
    
    print(f"\nNOVEL DISCOVERIES: {len(summary['novel_discoveries'])}")
    for discovery in summary['novel_discoveries'][:5]:  # Show top 5
        print(f"- {discovery['relationship']}")
    
    return results


if __name__ == "__main__":
    results = main()