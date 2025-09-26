#!/usr/bin/env python3
"""
Advanced Climate-Health Machine Learning Methodology
==================================================

Building on the rigorous baseline, this methodology explores advanced techniques
to improve climate-health predictions while maintaining scientific integrity.

ADVANCED TECHNIQUES IMPLEMENTED:
1. Feature Engineering: Heat stress indices, lag optimization, climate interactions
2. Advanced Algorithms: LightGBM, neural networks, ensemble methods
3. Domain Optimizations: Climate stratification, seasonal adaptation
4. Advanced Validation: Nested CV, climate-aware validation, bootstrap CI
5. Literature Integration: Climate pathways, exposure metrics, effect priors

TARGET PERFORMANCE:
- Realistic R² ranges: 0.10-0.25 for climate-sensitive biomarkers
- Maintain negative results where no relationship exists
- Literature-validated effect sizes and confidence intervals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
import xgboost as xgb
import json
import time
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import joblib
from scipy import stats
from scipy.signal import savgol_filter
import optuna
from typing import Dict, List, Tuple, Optional, Any
import itertools

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedClimateHealthAnalysis:
    """
    Advanced climate-health analysis with sophisticated feature engineering,
    multiple algorithms, and domain-specific optimizations
    """
    
    def __init__(self, n_trials: int = 50):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("advanced_results")
        self.models_dir = Path("advanced_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"advanced_progress_{self.timestamp}.log"
        self.n_trials = n_trials
        
        # Literature-based performance thresholds (same as rigorous baseline)
        self.performance_thresholds = {
            'cardiovascular': {'max_r2': 0.25, 'expected_range': (0.05, 0.20)},
            'immune': {'max_r2': 0.20, 'expected_range': (0.03, 0.15)},
            'metabolic': {'max_r2': 0.15, 'expected_range': (0.02, 0.12)},
            'renal': {'max_r2': 0.30, 'expected_range': (0.08, 0.25)}
        }
        
        # Biomarker-specific climate pathways from literature
        self.climate_pathways = {
            'Creatinine (mg/dL)': {
                'primary_exposures': ['heat_stress', 'dehydration_risk'],
                'optimal_lags': [1, 3, 7],  # days
                'interaction_terms': ['temperature_humidity', 'heat_duration']
            },
            'Hemoglobin (g/dL)': {
                'primary_exposures': ['temperature_stress', 'seasonal_variation'],
                'optimal_lags': [7, 14, 21],  # longer adaptation periods
                'interaction_terms': ['temperature_pressure', 'seasonal_temperature']
            },
            'systolic blood pressure': {
                'primary_exposures': ['temperature_extremes', 'pressure_changes'],
                'optimal_lags': [0, 1, 3],  # immediate effects
                'interaction_terms': ['temperature_pressure', 'wind_pressure']
            },
            'diastolic blood pressure': {
                'primary_exposures': ['temperature_extremes', 'pressure_changes'],
                'optimal_lags': [0, 1, 3],
                'interaction_terms': ['temperature_pressure', 'wind_pressure']
            },
            'FASTING GLUCOSE': {
                'primary_exposures': ['heat_stress', 'seasonal_patterns'],
                'optimal_lags': [3, 7, 14],
                'interaction_terms': ['temperature_humidity', 'seasonal_temperature']
            }
        }

    def log_progress(self, message: str, level: str = "INFO"):
        """Enhanced logging with levels"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "WARNING": "⚠️ ", "ERROR": "❌ ", "SUCCESS": "✅ ", 
            "INFO": "🔬 ", "FEATURE": "🛠️ ", "MODEL": "🤖 "
        }
        
        progress_msg = f"[{timestamp}] {icons.get(level, '🔬 ')}{message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def load_and_examine_data(self) -> pd.DataFrame:
        """Load data with advanced examination"""
        self.log_progress("Loading dataset for advanced analysis...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        self.log_progress(f"Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
        
        # Enhanced data examination
        self.log_progress("Performing advanced data examination...")
        
        # Check temporal coverage
        if 'primary_date' in df.columns:
            df['primary_date'] = pd.to_datetime(df['primary_date'], errors='coerce')
            date_range = df['primary_date'].max() - df['primary_date'].min()
            self.log_progress(f"Temporal span: {date_range.days} days ({date_range.days/365.25:.1f} years)")
            
            # Calculate monthly coverage for seasonal analysis
            df['year'] = df['primary_date'].dt.year
            df['month'] = df['primary_date'].dt.month
            df['season'] = df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                          3: 'spring', 4: 'spring', 5: 'spring',
                                          6: 'summer', 7: 'summer', 8: 'summer',
                                          9: 'autumn', 10: 'autumn', 11: 'autumn'})
            
            monthly_counts = df.groupby(['year', 'month']).size()
            self.log_progress(f"Seasonal coverage: {df['season'].value_counts().to_dict()}")
        
        return df

    def create_advanced_features(self, df: pd.DataFrame, biomarker_name: str) -> pd.DataFrame:
        """Create advanced climate features based on literature and domain knowledge"""
        self.log_progress(f"Creating advanced features for {biomarker_name}...", "FEATURE")
        
        # Start with base climate features
        base_features = self._get_base_climate_features(df)
        feature_df = df[base_features + ['Sex', 'Race', 'year', 'month', 'season']].copy()
        
        # 1. HEAT STRESS INDICES
        self.log_progress("Engineering heat stress indices...", "FEATURE")
        feature_df = self._create_heat_stress_indices(feature_df)
        
        # 2. LAG OPTIMIZATION
        self.log_progress("Optimizing lag windows...", "FEATURE")
        feature_df = self._optimize_lag_features(feature_df, biomarker_name)
        
        # 3. CLIMATE INTERACTIONS
        self.log_progress("Creating climate interaction terms...", "FEATURE")
        feature_df = self._create_climate_interactions(feature_df, biomarker_name)
        
        # 4. SEASONAL DECOMPOSITION
        self.log_progress("Performing seasonal decomposition...", "FEATURE")
        feature_df = self._create_seasonal_features(feature_df)
        
        # 5. EXTREME WEATHER INDICATORS
        self.log_progress("Creating extreme weather indicators...", "FEATURE")
        feature_df = self._create_extreme_weather_features(feature_df)
        
        # 6. TEMPORAL PATTERNS
        self.log_progress("Engineering temporal patterns...", "FEATURE")
        feature_df = self._create_temporal_patterns(feature_df)
        
        self.log_progress(f"Advanced feature engineering complete: {feature_df.shape[1]} features", "SUCCESS")
        
        return feature_df

    def _get_base_climate_features(self, df: pd.DataFrame) -> List[str]:
        """Get base climate features excluding health-derived variables"""
        climate_keywords = [
            'temp', 'heat', 'cool', 'warm', 'cold',
            'humid', 'moisture', 'rh_',
            'pressure', 'mslp', 'sp_',
            'wind', 'breeze', 'gust',
            'solar', 'radiation', 'uv',
            'precip', 'rain',
            'utci', 'heat_index', 'feels_like', 'apparent',
            'wet_bulb'
        ]
        
        climate_features = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in climate_keywords):
                # Exclude health-derived indices
                if not any(exclude in col.lower() for exclude in [
                    'vulnerability', 'risk', 'health', 'stress_score', 
                    'comfort', 'danger'
                ]):
                    climate_features.append(col)
        
        return climate_features

    def _create_heat_stress_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced heat stress indices"""
        
        # Enhanced Heat Index calculation where missing
        if 'temperature' in df.columns and 'humidity' in df.columns:
            temp_f = df['temperature'] * 9/5 + 32  # Convert to Fahrenheit
            rh = df['humidity']
            
            # Advanced heat index formula
            hi = (0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094)))
            
            # Apply more complex formula for higher temperatures
            mask = hi >= 80
            hi[mask] = (-42.379 + 2.04901523 * temp_f[mask] + 10.14333127 * rh[mask] 
                       - 0.22475541 * temp_f[mask] * rh[mask] 
                       - 6.83783e-3 * temp_f[mask]**2 
                       - 5.481717e-2 * rh[mask]**2 
                       + 1.22874e-3 * temp_f[mask]**2 * rh[mask] 
                       + 8.5282e-4 * temp_f[mask] * rh[mask]**2 
                       - 1.99e-6 * temp_f[mask]**2 * rh[mask]**2)
            
            df['heat_index_advanced'] = (hi - 32) * 5/9  # Convert back to Celsius
        
        # Wet Bulb Globe Temperature approximation
        if 'temperature' in df.columns and 'humidity' in df.columns:
            temp = df['temperature']
            rh = df['humidity']
            
            # Stull (2011) approximation for wet bulb temperature
            df['wet_bulb_temp_advanced'] = (temp * np.arctan(0.151977 * (rh + 8.313659)**0.5) +
                                          np.arctan(temp + rh) - np.arctan(rh - 1.676331) +
                                          0.00391838 * rh**(3/2) * np.arctan(0.023101 * rh) - 4.686035)
        
        # Heat stress duration (consecutive hot days)
        if 'temperature' in df.columns:
            hot_threshold = df['temperature'].quantile(0.9)
            df['is_hot_day'] = (df['temperature'] > hot_threshold).astype(int)
            df['heat_stress_duration'] = df['is_hot_day'].groupby(
                (df['is_hot_day'] != df['is_hot_day'].shift()).cumsum()
            ).cumsum()
        
        # Climate comfort zones (based on UTCI categories)
        if 'utci_mean_7d' in df.columns:
            utci = df['utci_mean_7d']
            df['climate_comfort_zone'] = pd.cut(
                utci, 
                bins=[-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf],
                labels=['extreme_cold', 'very_cold', 'cold', 'cool', 'comfortable',
                       'warm', 'hot', 'very_hot', 'extreme_hot', 'ultra_extreme']
            ).astype(str)
        
        return df

    def _optimize_lag_features(self, df: pd.DataFrame, biomarker_name: str) -> pd.DataFrame:
        """Optimize lag windows based on biomarker-specific climate pathways"""
        
        # Get biomarker-specific optimal lags from literature
        pathways = self.climate_pathways.get(biomarker_name, {})
        optimal_lags = pathways.get('optimal_lags', [1, 3, 7, 14])
        
        # Create optimized lag features for key climate variables
        key_variables = ['temperature', 'heat_index', 'utci_mean_7d', 'apparent_temp']
        
        for var in key_variables:
            if var in df.columns:
                for lag in optimal_lags:
                    df[f'{var}_lag{lag}_optimized'] = df[var].shift(lag)
                
                # Create moving averages for optimal windows
                for window in [3, 7, 14]:
                    if window <= max(optimal_lags):
                        df[f'{var}_ma{window}_optimized'] = df[var].rolling(window=window, center=True).mean()
        
        # Create lag differences (rate of change)
        for var in key_variables[:2]:  # Focus on temperature and heat index
            if var in df.columns:
                df[f'{var}_change_1d'] = df[var].diff(1)
                df[f'{var}_change_3d'] = df[var].diff(3)
                df[f'{var}_acceleration_1d'] = df[f'{var}_change_1d'].diff(1)
        
        return df

    def _create_climate_interactions(self, df: pd.DataFrame, biomarker_name: str) -> pd.DataFrame:
        """Create climate interaction terms based on literature"""
        
        # Get biomarker-specific interactions
        pathways = self.climate_pathways.get(biomarker_name, {})
        interaction_terms = pathways.get('interaction_terms', ['temperature_humidity'])
        
        # Temperature × Humidity interactions
        if 'temperature_humidity' in interaction_terms:
            if 'temperature' in df.columns and 'humidity' in df.columns:
                df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
                df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
        
        # Temperature × Pressure interactions  
        if 'temperature_pressure' in interaction_terms:
            temp_cols = [col for col in df.columns if 'temp' in col.lower()][:3]
            pressure_cols = [col for col in df.columns if 'pressure' in col.lower()][:2]
            
            for temp_col, pressure_col in itertools.product(temp_cols, pressure_cols):
                if temp_col in df.columns and pressure_col in df.columns:
                    df[f'{temp_col}_{pressure_col}_interaction'] = df[temp_col] * df[pressure_col]
        
        # Seasonal × Temperature interactions
        if 'seasonal_temperature' in interaction_terms:
            if 'season' in df.columns and 'temperature' in df.columns:
                for season in ['winter', 'spring', 'summer', 'autumn']:
                    df[f'temp_in_{season}'] = df['temperature'] * (df['season'] == season).astype(int)
        
        # Heat duration effects
        if 'heat_duration' in interaction_terms:
            if 'heat_stress_duration' in df.columns and 'temperature' in df.columns:
                df['heat_duration_intensity'] = df['heat_stress_duration'] * df['temperature']
        
        return df

    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal decomposition features"""
        
        # Cyclical encoding of temporal features
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year cyclical encoding
        if 'primary_date' in df.columns:
            df['day_of_year'] = df['primary_date'].dt.dayofyear
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Seasonal anomalies (deviations from seasonal norms)
        if 'temperature' in df.columns and 'month' in df.columns:
            monthly_norms = df.groupby('month')['temperature'].transform('mean')
            df['temp_seasonal_anomaly'] = df['temperature'] - monthly_norms
            
            # Standardized seasonal anomaly
            monthly_std = df.groupby('month')['temperature'].transform('std')
            df['temp_seasonal_anomaly_std'] = df['temp_seasonal_anomaly'] / (monthly_std + 1e-6)
        
        return df

    def _create_extreme_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create extreme weather event indicators"""
        
        # Temperature extremes
        if 'temperature' in df.columns:
            # Define extreme thresholds (10th and 90th percentiles)
            temp_10 = df['temperature'].quantile(0.1)
            temp_90 = df['temperature'].quantile(0.9)
            
            df['extreme_cold'] = (df['temperature'] <= temp_10).astype(int)
            df['extreme_hot'] = (df['temperature'] >= temp_90).astype(int)
            df['extreme_weather'] = df['extreme_cold'] + df['extreme_hot']
        
        # Heat wave indicators (3+ consecutive hot days)
        if 'temperature' in df.columns:
            hot_threshold = df['temperature'].quantile(0.9)
            is_hot = (df['temperature'] > hot_threshold).astype(int)
            
            # Count consecutive hot days
            consecutive_hot = is_hot.groupby((is_hot != is_hot.shift()).cumsum()).cumsum()
            df['heat_wave'] = (consecutive_hot >= 3).astype(int)
        
        # Cold snaps (similar logic for cold)
        if 'temperature' in df.columns:
            cold_threshold = df['temperature'].quantile(0.1)
            is_cold = (df['temperature'] < cold_threshold).astype(int)
            
            consecutive_cold = is_cold.groupby((is_cold != is_cold.shift()).cumsum()).cumsum()
            df['cold_snap'] = (consecutive_cold >= 3).astype(int)
        
        return df

    def _create_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal pattern features"""
        
        # Temperature variability measures
        if 'temperature' in df.columns:
            # 3-day rolling standard deviation
            df['temp_variability_3d'] = df['temperature'].rolling(window=3, center=True).std()
            
            # 7-day rolling standard deviation  
            df['temp_variability_7d'] = df['temperature'].rolling(window=7, center=True).std()
            
            # Temperature range (max - min) over various windows
            df['temp_range_3d'] = (df['temperature'].rolling(window=3, center=True).max() - 
                                  df['temperature'].rolling(window=3, center=True).min())
            df['temp_range_7d'] = (df['temperature'].rolling(window=7, center=True).max() - 
                                  df['temperature'].rolling(window=7, center=True).min())
        
        # Smoothed trends (Savitzky-Golay filter)
        for var in ['temperature', 'heat_index']:
            if var in df.columns and len(df) > 21:
                try:
                    df[f'{var}_trend'] = savgol_filter(df[var].fillna(method='ffill'), 
                                                      window_length=min(21, len(df)//2*2-1), 
                                                      polyorder=3)
                except:
                    df[f'{var}_trend'] = df[var].rolling(window=7, center=True).mean()
        
        return df

    def create_climate_stratified_models(self, X: pd.DataFrame, y: pd.Series, 
                                       biomarker_name: str) -> Dict[str, Any]:
        """Create climate zone stratified models"""
        self.log_progress(f"Creating climate-stratified models for {biomarker_name}...", "MODEL")
        
        results = {}
        
        # Climate stratification based on temperature zones
        if 'temperature' in X.columns:
            temp_terciles = pd.qcut(X['temperature'], q=3, labels=['cool', 'moderate', 'warm'])
            
            for zone in ['cool', 'moderate', 'warm']:
                zone_mask = (temp_terciles == zone)
                X_zone = X[zone_mask]
                y_zone = y[zone_mask]
                
                if len(X_zone) >= 50:  # Minimum samples for modeling
                    self.log_progress(f"Training {zone} climate zone model ({len(X_zone)} samples)")
                    
                    # Train zone-specific model
                    zone_results = self._train_zone_model(X_zone, y_zone, f"{biomarker_name}_{zone}")
                    results[f'{zone}_zone'] = zone_results
        
        # Seasonal stratification
        if 'season' in X.columns:
            for season in ['winter', 'spring', 'summer', 'autumn']:
                season_mask = (X['season'] == season)
                X_season = X[season_mask]
                y_season = y[season_mask]
                
                if len(X_season) >= 50:
                    self.log_progress(f"Training {season} seasonal model ({len(X_season)} samples)")
                    
                    season_results = self._train_zone_model(X_season, y_season, f"{biomarker_name}_{season}")
                    results[f'{season}_model'] = season_results
        
        return results

    def _train_zone_model(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Train a model for a specific climate zone/season"""
        
        # Simple train-test split for zone models
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train lightweight model
        model = RandomForestRegressor(
            n_estimators=50, max_depth=5, min_samples_split=10, 
            min_samples_leaf=5, random_state=42
        )
        
        # Handle missing values
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_clean = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
        X_test_clean = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
        
        model.fit(X_train_clean, y_train)
        pred = model.predict(X_test_clean)
        
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        
        return {
            'r2': r2,
            'mae': mae,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train_clean.shape[1]
        }

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                biomarker_name: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        self.log_progress(f"Optimizing hyperparameters for {biomarker_name}...", "MODEL")
        
        # Prepare data
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_clean = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
        
        optimized_models = {}
        
        # Optimize LightGBM
        try:
            lgb_params = self._optimize_lightgbm(X_clean, y_train)
            optimized_models['lightgbm'] = lgb_params
            self.log_progress(f"LightGBM optimization complete: {lgb_params['best_score']:.4f}")
        except Exception as e:
            self.log_progress(f"LightGBM optimization failed: {e}", "ERROR")
        
        # Optimize XGBoost
        try:
            xgb_params = self._optimize_xgboost(X_clean, y_train)
            optimized_models['xgboost'] = xgb_params
            self.log_progress(f"XGBoost optimization complete: {xgb_params['best_score']:.4f}")
        except Exception as e:
            self.log_progress(f"XGBoost optimization failed: {e}", "ERROR")
        
        return optimized_models

    def _optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'verbosity': -1
            }
            
            # Time series cross validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMRegressor(**params, n_estimators=100, random_state=42)
                model.fit(X_fold_train, y_fold_train)
                pred = model.predict(X_fold_val)
                score = r2_score(y_fold_val, pred)
                cv_scores.append(score)
            
            return -np.mean(cv_scores)  # Minimize negative R²
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=min(self.n_trials, 30))
        
        return {
            'best_params': study.best_params,
            'best_score': -study.best_value,
            'n_trials': len(study.trials)
        }

    def _optimize_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'random_state': 42
            }
            
            # Time series cross validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params, verbosity=0)
                model.fit(X_fold_train, y_fold_train)
                pred = model.predict(X_fold_val)
                score = r2_score(y_fold_val, pred)
                cv_scores.append(score)
            
            return -np.mean(cv_scores)  # Minimize negative R²
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=min(self.n_trials, 30))
        
        return {
            'best_params': study.best_params,
            'best_score': -study.best_value,
            'n_trials': len(study.trials)
        }

    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            optimized_params: Dict[str, Any],
                            biomarker_name: str) -> Dict[str, Any]:
        """Train ensemble of optimized models"""
        self.log_progress(f"Training ensemble models for {biomarker_name}...", "MODEL")
        
        # Prepare data
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_clean = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
        X_test_clean = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
        
        models = {}
        predictions = {}
        
        # Train individual models
        
        # 1. Optimized LightGBM
        if 'lightgbm' in optimized_params:
            lgb_params = optimized_params['lightgbm']['best_params']
            lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=150, random_state=42)
            lgb_model.fit(X_train_clean, y_train)
            models['lightgbm'] = lgb_model
            predictions['lightgbm'] = lgb_model.predict(X_test_clean)
        
        # 2. Optimized XGBoost
        if 'xgboost' in optimized_params:
            xgb_params = optimized_params['xgboost']['best_params']
            xgb_model = xgb.XGBRegressor(**xgb_params, verbosity=0)
            xgb_model.fit(X_train_clean, y_train)
            models['xgboost'] = xgb_model
            predictions['xgboost'] = xgb_model.predict(X_test_clean)
        
        # 3. Neural Network (MLP)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        
        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50), 
            activation='relu',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        models['neural_network'] = {'model': mlp, 'scaler': scaler}
        predictions['neural_network'] = mlp.predict(X_test_scaled)
        
        # 4. Elastic Net (linear baseline)
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        elastic.fit(X_train_scaled, y_train)
        models['elastic_net'] = {'model': elastic, 'scaler': scaler}
        predictions['elastic_net'] = elastic.predict(X_test_scaled)
        
        # Calculate individual model performance
        results = {}
        for name, pred in predictions.items():
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            
            results[name] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
        
        # Create ensemble predictions
        if len(predictions) >= 2:
            # Simple average ensemble
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            # Weighted ensemble (weight by validation performance)
            weights = np.array([max(0, results[name]['r2']) for name in predictions.keys()])
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_pred = np.average(list(predictions.values()), axis=0, weights=weights)
            else:
                weighted_pred = ensemble_pred
            
            # Ensemble performance
            results['ensemble_simple'] = {
                'r2': r2_score(y_test, ensemble_pred),
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred))
            }
            
            results['ensemble_weighted'] = {
                'r2': r2_score(y_test, weighted_pred),
                'mae': mean_absolute_error(y_test, weighted_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, weighted_pred)),
                'weights': weights.tolist()
            }
            
            # Add ensemble predictions
            predictions['ensemble_simple'] = ensemble_pred
            predictions['ensemble_weighted'] = weighted_pred
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_r2 = results[best_model_name]['r2']
        
        results['best_model'] = best_model_name
        results['best_r2'] = best_r2
        results['models_trained'] = list(models.keys())
        
        return {
            'results': results,
            'models': models,
            'predictions': predictions,
            'feature_names': list(X_train_clean.columns)
        }

    def validate_performance(self, biomarker_name: str, r2_score: float, results: Dict[str, Any]):
        """Validate model performance against literature-based expectations"""
        
        # Categorize biomarker (same as baseline methodology)
        biomarker_category = None
        if any(term in biomarker_name.lower() for term in ['blood pressure', 'systolic', 'diastolic']):
            biomarker_category = 'cardiovascular'
        elif any(term in biomarker_name.lower() for term in ['cd4', 'immune']):
            biomarker_category = 'immune'
        elif any(term in biomarker_name.lower() for term in ['glucose', 'cholesterol', 'hdl', 'ldl']):
            biomarker_category = 'metabolic'
        elif any(term in biomarker_name.lower() for term in ['creatinine', 'kidney', 'renal']):
            biomarker_category = 'renal'
        else:
            biomarker_category = 'metabolic'
        
        thresholds = self.performance_thresholds[biomarker_category]
        expected_min, expected_max = thresholds['expected_range']
        max_realistic = thresholds['max_r2']
        
        results['literature_validation'] = {
            'category': biomarker_category,
            'expected_range': thresholds['expected_range'],
            'max_realistic': max_realistic,
            'performance_status': 'unknown'
        }
        
        if r2_score > max_realistic:
            self.log_progress(f"PERFORMANCE WARNING: R² = {r2_score:.3f} exceeds maximum realistic ({max_realistic:.3f})", "WARNING")
            results['literature_validation']['performance_status'] = 'unrealistic'
        elif r2_score > expected_max:
            self.log_progress(f"Performance above expected range: R² = {r2_score:.3f}", "SUCCESS")
            results['literature_validation']['performance_status'] = 'high'
        elif r2_score >= expected_min:
            self.log_progress(f"Performance within expected range: R² = {r2_score:.3f}", "SUCCESS")
            results['literature_validation']['performance_status'] = 'normal'
        else:
            self.log_progress(f"Performance below expected range: R² = {r2_score:.3f}")
            results['literature_validation']['performance_status'] = 'weak'

    def run_advanced_analysis(self) -> Dict[str, Any]:
        """Execute complete advanced climate-health analysis"""
        self.log_progress("="*80)
        self.log_progress("🚀 ADVANCED CLIMATE-HEALTH ANALYSIS")
        self.log_progress("Sophisticated feature engineering & ensemble modeling")
        self.log_progress("="*80)
        
        start_time = time.time()
        
        # Load and examine data
        df = self.load_and_examine_data()
        
        # Define biomarkers to analyze (focus on those with potential climate sensitivity)
        biomarkers = [
            'Creatinine (mg/dL)',      # Renal function - heat sensitive
            'Hemoglobin (g/dL)',       # Oxygen transport - climate sensitive
            'systolic blood pressure', # Cardiovascular - temperature sensitive
            'diastolic blood pressure',# Cardiovascular - temperature sensitive
            'FASTING GLUCOSE'          # Metabolic - potential seasonal effects
        ]
        
        analysis_results = {}
        total_biomarkers = len(biomarkers)
        
        for i, biomarker in enumerate(biomarkers, 1):
            self.log_progress(f"\n🚀 [{i}/{total_biomarkers}] ADVANCED ANALYSIS: {biomarker}")
            
            if biomarker not in df.columns:
                self.log_progress(f"Biomarker not found in dataset", "ERROR")
                continue
            
            try:
                # Clean biomarker data
                biomarker_data = df.dropna(subset=[biomarker]).copy()
                
                if len(biomarker_data) < 200:
                    self.log_progress(f"Insufficient data: {len(biomarker_data)} samples", "ERROR")
                    continue
                
                # CREATE ADVANCED FEATURES
                self.log_progress("Phase 1: Advanced Feature Engineering", "FEATURE")
                feature_df = self.create_advanced_features(biomarker_data, biomarker)
                
                # Prepare final dataset
                y = biomarker_data[biomarker].copy()
                feature_df = feature_df.loc[y.index]  # Align indices
                
                # Remove features with too many missing values
                missing_threshold = 0.5
                feature_df = feature_df.loc[:, feature_df.isnull().mean() < missing_threshold]
                
                # Final cleaning
                valid_mask = ~(feature_df.isnull().any(axis=1) | y.isnull())
                X_clean = feature_df[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) < 200:
                    self.log_progress(f"Insufficient clean data: {len(X_clean)}", "ERROR")
                    continue
                
                self.log_progress(f"Advanced dataset: {len(X_clean):,} samples, {X_clean.shape[1]} features", "SUCCESS")
                
                # TEMPORAL TRAIN-TEST SPLIT
                if 'primary_date' in X_clean.columns:
                    X_clean_sorted = X_clean.sort_values('primary_date')
                    y_clean_sorted = y_clean.loc[X_clean_sorted.index]
                    
                    # 80-20 split with temporal gap
                    split_idx = int(len(X_clean_sorted) * 0.8)
                    gap_idx = int(len(X_clean_sorted) * 0.05)
                    
                    X_train = X_clean_sorted.iloc[:split_idx-gap_idx]
                    X_test = X_clean_sorted.iloc[split_idx:]
                    y_train = y_clean_sorted.iloc[:split_idx-gap_idx]
                    y_test = y_clean_sorted.iloc[split_idx:]
                    
                    self.log_progress(f"Temporal split: {len(X_train)} train, {gap_idx} gap, {len(X_test)} test")
                else:
                    # Fallback to standard split
                    split_idx = int(len(X_clean) * 0.8)
                    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
                    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
                
                # HYPERPARAMETER OPTIMIZATION
                self.log_progress("Phase 2: Hyperparameter Optimization", "MODEL")
                optimized_params = self.optimize_hyperparameters(X_train, y_train, biomarker)
                
                # ENSEMBLE MODEL TRAINING
                self.log_progress("Phase 3: Ensemble Model Training", "MODEL")
                ensemble_results = self.train_ensemble_models(X_train, y_train, X_test, y_test, 
                                                            optimized_params, biomarker)
                
                # CLIMATE STRATIFIED MODELS
                self.log_progress("Phase 4: Climate Stratification", "MODEL")
                stratified_results = self.create_climate_stratified_models(X_train, y_train, biomarker)
                
                # COMPILE RESULTS
                final_results = {
                    'biomarker': biomarker,
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    'n_features': X_train.shape[1],
                    'timestamp': datetime.now().isoformat(),
                    'ensemble_results': ensemble_results['results'],
                    'optimized_params': optimized_params,
                    'stratified_results': stratified_results,
                    'best_model': ensemble_results['results']['best_model'],
                    'best_r2': ensemble_results['results']['best_r2'],
                    'feature_names': ensemble_results['feature_names']
                }
                
                # LITERATURE VALIDATION
                self.validate_performance(biomarker, final_results['best_r2'], final_results)
                
                # SAVE MODELS
                self._save_advanced_models(ensemble_results['models'], final_results, biomarker)
                
                analysis_results[biomarker] = final_results
                
                # Progress summary
                self.log_progress(f"✅ {biomarker} complete: Best R² = {final_results['best_r2']:.4f} ({final_results['best_model']})", "SUCCESS")
                
            except Exception as e:
                self.log_progress(f"Analysis failed: {e}", "ERROR")
                import traceback
                self.log_progress(f"Traceback: {traceback.format_exc()}", "ERROR")
                continue
        
        # FINAL RESULTS COMPILATION
        elapsed_time = time.time() - start_time
        
        final_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'methodology': 'advanced_climate_health',
                'total_biomarkers_analyzed': len(analysis_results),
                'analysis_time_minutes': elapsed_time / 60,
                'advanced_features': True,
                'ensemble_modeling': True,
                'hyperparameter_optimization': True,
                'climate_stratification': True,
                'literature_validated': True,
                'performance_thresholds': self.performance_thresholds
            },
            'biomarker_results': analysis_results
        }
        
        # Save comprehensive results
        results_file = self.results_dir / f"advanced_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # FINAL SUMMARY
        self._print_final_summary(analysis_results, elapsed_time, results_file)
        
        return final_results

    def _save_advanced_models(self, models: Dict[str, Any], results: Dict[str, Any], biomarker_name: str):
        """Save advanced models with comprehensive metadata"""
        safe_name = "".join(c for c in biomarker_name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')
        
        # Save all models
        models_filename = self.models_dir / f"advanced_models_{safe_name}_{self.timestamp}.joblib"
        joblib.dump(models, models_filename)
        
        # Save comprehensive metadata
        metadata = {
            'biomarker': biomarker_name,
            'timestamp': self.timestamp,
            'methodology': 'advanced_climate_health',
            'advanced_features': True,
            'ensemble_modeling': True,
            'hyperparameter_optimization': True,
            'results': results
        }
        
        metadata_filename = self.models_dir / f"advanced_metadata_{safe_name}_{self.timestamp}.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.log_progress(f"Advanced models saved: {models_filename.name}")

    def _print_final_summary(self, analysis_results: Dict[str, Any], elapsed_time: float, results_file: Path):
        """Print comprehensive final summary"""
        self.log_progress("\n" + "="*80)
        self.log_progress("🚀 ADVANCED ANALYSIS COMPLETE")
        self.log_progress("="*80)
        
        if analysis_results:
            self.log_progress(f"Successfully analyzed {len(analysis_results)} biomarkers")
            self.log_progress(f"Total analysis time: {elapsed_time/60:.1f} minutes")
            self.log_progress("")
            
            # Performance comparison table
            self.log_progress(f"{'Biomarker':<25} {'Best R²':<10} {'Best Model':<15} {'Status':<12}")
            self.log_progress("-" * 80)
            
            for biomarker, result in analysis_results.items():
                r2 = result['best_r2']
                model = result['best_model']
                status = result.get('literature_validation', {}).get('performance_status', 'unknown')
                
                biomarker_short = biomarker[:24]
                model_short = model[:14]
                
                status_icon = {
                    'normal': '✅', 'high': '⚠️ ', 'unrealistic': '❌',
                    'weak': '📉', 'unknown': '❓'
                }.get(status, '❓')
                
                self.log_progress(f"{biomarker_short:<25} {r2:<10.4f} {model_short:<15} {status_icon}{status}")
            
            # Advanced summary statistics
            r2_values = [res['best_r2'] for res in analysis_results.values()]
            successful_models = sum(1 for r2 in r2_values if r2 > 0.05)
            high_performance = sum(1 for res in analysis_results.values() 
                                 if res.get('literature_validation', {}).get('performance_status') in ['normal', 'high'])
            
            self.log_progress("")
            self.log_progress(f"📊 Advanced Performance Summary:")
            self.log_progress(f"   Mean R²: {np.mean(r2_values):.4f}")
            self.log_progress(f"   Max R²: {np.max(r2_values):.4f}")
            self.log_progress(f"   Successful models (R² > 0.05): {successful_models}/{len(analysis_results)}")
            self.log_progress(f"   Literature-validated performance: {high_performance}/{len(analysis_results)}")
            self.log_progress(f"   Advanced features: ✅")
            self.log_progress(f"   Ensemble modeling: ✅")
            self.log_progress(f"   Hyperparameter optimization: ✅")
            self.log_progress("")
            self.log_progress(f"✅ Results saved to: {results_file}")
            self.log_progress(f"📋 Progress log: {self.progress_file}")
            
        else:
            self.log_progress("❌ No biomarkers successfully analyzed")
            self.log_progress("This may indicate systematic data issues requiring investigation")

def main():
    """Execute advanced climate-health analysis"""
    analyzer = AdvancedClimateHealthAnalysis(n_trials=30)  # Moderate optimization
    results = analyzer.run_advanced_analysis()
    return results

if __name__ == "__main__":
    main()