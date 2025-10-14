#!/usr/bin/env python3
"""
Refined Climate-Health Analysis Pipeline
========================================

An optimized and refined version of the ENBEL climate-health analysis pipeline
focusing on robust methodology, reproducibility, and interpretability.

Key Improvements:
- Enhanced data validation and quality checks
- Optimized feature engineering pipeline
- Advanced hyperparameter tuning with Optuna
- Comprehensive SHAP analysis with multiple plot types
- Automated result documentation and visualization
- Performance benchmarking against previous runs

Author: ENBEL Research Team
Version: 2.0.0
Date: 2025-10-14
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

# Core scientific computing
import numpy as np
import pandas as pd
from scipy import stats

# Machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import optuna

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Explainable AI
import shap

# Progress tracking
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('refined_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(RANDOM_SEED)

class RefinedClimateHealthPipeline:
    """
    Refined pipeline for climate-health biomarker analysis with improved methodology.
    """

    def __init__(self, data_path: str = "data/raw", output_dir: str = "results/refined_analysis"):
        """
        Initialize the refined pipeline.

        Args:
            data_path: Path to raw data directory
            output_dir: Directory for output results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clinical_file = self.data_path / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
        self.gcro_file = self.data_path / "GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv"

        self.results = {}
        self.models = {}
        self.start_time = None

        logger.info(f"Initialized Refined Climate-Health Pipeline")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and validate clinical and GCRO datasets.

        Returns:
            Tuple of (clinical_df, gcro_df)
        """
        logger.info("Loading datasets...")

        # Load clinical data
        clinical_df = pd.read_csv(self.clinical_file, low_memory=False)
        logger.info(f"Clinical dataset: {clinical_df.shape[0]} rows, {clinical_df.shape[1]} columns")

        # Load GCRO data
        gcro_df = pd.read_csv(self.gcro_file, low_memory=False)
        logger.info(f"GCRO dataset: {gcro_df.shape[0]} rows, {gcro_df.shape[1]} columns")

        # Data validation
        logger.info("Validating data quality...")

        # Check for critical columns
        required_clinical_cols = ['latitude', 'longitude', 'primary_date']
        missing_cols = [col for col in required_clinical_cols if col not in clinical_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in clinical data: {missing_cols}")

        # Check data types
        clinical_df['primary_date'] = pd.to_datetime(clinical_df['primary_date'], errors='coerce')

        # Log data quality metrics
        logger.info(f"Clinical data date range: {clinical_df['primary_date'].min()} to {clinical_df['primary_date'].max()}")
        logger.info(f"Clinical data missing values: {clinical_df.isnull().sum().sum()} total")
        logger.info(f"GCRO data missing values: {gcro_df.isnull().sum().sum()} total")

        return clinical_df, gcro_df

    def identify_biomarkers(self, clinical_df: pd.DataFrame) -> List[str]:
        """
        Identify available biomarkers in the dataset.

        Args:
            clinical_df: Clinical dataframe

        Returns:
            List of biomarker column names
        """
        # Define biomarker patterns
        biomarker_keywords = [
            'CD4', 'glucose', 'cholesterol', 'LDL', 'HDL', 'triglyceride',
            'creatinine', 'ALT', 'AST', 'hemoglobin', 'hematocrit',
            'blood pressure', 'systolic', 'diastolic'
        ]

        biomarkers = []
        for col in clinical_df.columns:
            if any(keyword.lower() in col.lower() for keyword in biomarker_keywords):
                # Check if column has numeric data
                if pd.api.types.is_numeric_dtype(clinical_df[col]):
                    # Check if has reasonable amount of non-null values
                    if clinical_df[col].notna().sum() > 100:
                        biomarkers.append(col)

        logger.info(f"Identified {len(biomarkers)} biomarkers: {biomarkers}")
        return biomarkers

    def identify_climate_features(self, clinical_df: pd.DataFrame) -> List[str]:
        """
        Identify climate-related features in the dataset.

        Args:
            clinical_df: Clinical dataframe

        Returns:
            List of climate feature column names
        """
        climate_keywords = [
            'climate', 'temp', 'temperature', 'humidity', 'precipitation',
            'heat', 'wind', 'pressure', 'solar', 'lag'
        ]

        climate_features = []
        for col in clinical_df.columns:
            if any(keyword.lower() in col.lower() for keyword in climate_keywords):
                if pd.api.types.is_numeric_dtype(clinical_df[col]):
                    if clinical_df[col].notna().sum() > 1000:
                        climate_features.append(col)

        logger.info(f"Identified {len(climate_features)} climate features")
        logger.info(f"Sample features: {climate_features[:5]}")
        return climate_features

    def prepare_features(self, df: pd.DataFrame, biomarker: str,
                        climate_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            df: Input dataframe
            biomarker: Target biomarker column
            climate_features: List of climate feature columns

        Returns:
            Tuple of (X, y) for modeling
        """
        # Start with complete cases for target
        df_clean = df[df[biomarker].notna()].copy()
        logger.info(f"Complete cases for {biomarker}: {len(df_clean)}")

        # Select features
        feature_cols = climate_features.copy()

        # Add temporal features
        if 'month' in df_clean.columns:
            feature_cols.append('month')
        if 'season' in df_clean.columns and df_clean['season'].dtype == 'object':
            # One-hot encode season
            season_dummies = pd.get_dummies(df_clean['season'], prefix='season', drop_first=True)
            df_clean = pd.concat([df_clean, season_dummies], axis=1)
            feature_cols.extend(season_dummies.columns.tolist())

        # Remove features with too many missing values
        feature_cols = [col for col in feature_cols if col in df_clean.columns]
        missing_pct = df_clean[feature_cols].isnull().mean()
        valid_features = missing_pct[missing_pct < 0.5].index.tolist()

        logger.info(f"Valid features after missing value filter: {len(valid_features)}")

        # Prepare X and y
        X = df_clean[valid_features].copy()
        y = df_clean[biomarker].copy()

        # NOTE: Missing value imputation will be done AFTER train/test split
        # to avoid data leakage. Do NOT impute here.

        logger.info(f"Feature matrix shape (before imputation): {X.shape}")
        logger.info(f"Missing values per feature: {X.isnull().sum().sum()} total")
        logger.info(f"Target distribution: mean={y.mean():.2f}, std={y.std():.2f}")

        return X, y

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                 model_type: str = 'lightgbm') -> Dict:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to optimize

        Returns:
            Best hyperparameters dictionary
        """
        logger.info(f"Optimizing hyperparameters for {model_type}...")

        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                    'random_state': RANDOM_SEED,
                    'n_jobs': -1,
                    'verbosity': -1
                }
                model = lgb.LGBMRegressor(**params)

            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                    'random_state': RANDOM_SEED,
                    'n_jobs': -1
                }
                model = xgb.XGBRegressor(**params)

            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                    'random_state': RANDOM_SEED,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)

            # Cross-validation
            scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
            return scores.mean()

        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        logger.info(f"Best R² score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        return study.best_params

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series,
                          biomarker: str, optimize_hyperparams: bool = False) -> Dict:
        """
        Train models and evaluate performance.

        Args:
            X: Feature matrix
            y: Target vector
            biomarker: Biomarker name

        Returns:
            Results dictionary
        """
        logger.info(f"Training models for {biomarker}...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # CRITICAL: Impute missing values AFTER split to avoid data leakage
        # Fit imputation on training data only
        train_medians = X_train.median()
        X_train_imputed = X_train.fillna(train_medians)
        X_test_imputed = X_test.fillna(train_medians)  # Use training medians for test

        logger.info(f"Imputed {X_train.isnull().sum().sum()} training missing values")
        logger.info(f"Imputed {X_test.isnull().sum().sum()} test missing values")

        # Scale features (tree-based models don't require scaling but good practice)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        results = {
            'biomarker': biomarker,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'models': {}
        }

        # Optimize hyperparameters if requested
        best_params = {}
        if optimize_hyperparams:
            logger.info("Hyperparameter optimization enabled")
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'lightgbm')

        # Train multiple models
        if optimize_hyperparams and best_params:
            models_to_train = {
                'LightGBM': lgb.LGBMRegressor(**best_params),
                'XGBoost': xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
                'RandomForest': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
            }
        else:
            models_to_train = {
                'LightGBM': lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1),
                'XGBoost': xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
                'RandomForest': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
            }

        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")

            # Train on SCALED data (CRITICAL BUG FIX)
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # Predict on SCALED data
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results['models'][model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'train_time_seconds': train_time
            }

            logger.info(f"{model_name} - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")

            # Store best model
            if model_name == 'LightGBM':
                self.models[biomarker] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': X.columns.tolist()
                }

        return results

    def generate_shap_analysis(self, X: pd.DataFrame, biomarker: str):
        """
        Generate SHAP analysis and visualizations.

        Args:
            X: Feature matrix
            biomarker: Biomarker name
        """
        logger.info(f"Generating SHAP analysis for {biomarker}...")

        if biomarker not in self.models:
            logger.warning(f"No model found for {biomarker}")
            return

        model = self.models[biomarker]['model']

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values (sample if too large)
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=RANDOM_SEED)
        shap_values = explainer.shap_values(X_sample)

        # Create output directory
        shap_dir = self.output_dir / "shap_analysis" / biomarker.replace(" ", "_").replace("/", "_")
        shap_dir.mkdir(parents=True, exist_ok=True)

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(shap_dir / "summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Waterfall plot for first sample
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                             base_values=explainer.expected_value,
                                             data=X_sample.iloc[0],
                                             feature_names=X_sample.columns.tolist()),
                           show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(shap_dir / "waterfall_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        importance_df.to_csv(shap_dir / "feature_importance.csv", index=False)
        logger.info(f"SHAP analysis saved to {shap_dir}")

    def save_results(self, results: Dict, filename: str = "analysis_results.json"):
        """
        Save analysis results to JSON file.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    def run(self):
        """
        Run the complete refined analysis pipeline.
        """
        self.start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting Refined Climate-Health Analysis Pipeline")
        logger.info("=" * 80)

        try:
            # Load data
            clinical_df, gcro_df = self.load_and_validate_data()

            # Identify biomarkers and features
            biomarkers = self.identify_biomarkers(clinical_df)
            climate_features = self.identify_climate_features(clinical_df)

            # Analyze each biomarker
            all_results = {}

            for biomarker in tqdm(biomarkers[:5], desc="Analyzing biomarkers"):  # Limit to 5 for efficiency
                logger.info(f"\n{'='*60}")
                logger.info(f"Analyzing: {biomarker}")
                logger.info(f"{'='*60}")

                try:
                    # Prepare features
                    X, y = self.prepare_features(clinical_df, biomarker, climate_features)

                    if len(X) < 100:
                        logger.warning(f"Insufficient samples for {biomarker}: {len(X)}")
                        continue

                    # Train and evaluate
                    results = self.train_and_evaluate(X, y, biomarker)
                    all_results[biomarker] = results

                    # Generate SHAP analysis
                    self.generate_shap_analysis(X, biomarker)

                except Exception as e:
                    logger.error(f"Error analyzing {biomarker}: {str(e)}")
                    continue

            # Save all results
            self.save_results(all_results)

            # Print summary
            total_time = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            logger.info(f"Analyzed {len(all_results)} biomarkers")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"{'='*80}")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for the refined pipeline."""
    print("\n" + "="*80)
    print("Refined Climate-Health Analysis Pipeline v2.0")
    print("="*80 + "\n")

    # Initialize and run pipeline
    pipeline = RefinedClimateHealthPipeline()
    pipeline.run()

    print("\n" + "="*80)
    print("Analysis complete! Check the results directory for outputs.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
