#!/usr/bin/env python3
"""
Refined Climate-Health Analysis Pipeline - FIXED VERSION
=========================================================

This version FIXES the critical feature leakage bug where biomarkers
were being used to predict other biomarkers.

KEY FIX: Strict whitelist-based feature selection ensuring ONLY
climate and socioeconomic features are used.

Author: ENBEL Research Team
Version: 2.1.0 (LEAKAGE FIX)
Date: 2025-10-14
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
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
        logging.FileHandler('refined_pipeline_fixed.log'),
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


# ============================================================================
# CRITICAL FIX: STRICT FEATURE WHITELIST
# ============================================================================

# ALLOWED climate features (prefix-based)
ALLOWED_CLIMATE_PREFIXES = ('climate_',)

# ALLOWED socioeconomic features (explicit list)
ALLOWED_SOCIOECONOMIC_FEATURES = {
    'HEAT_VULNERABILITY_SCORE',
    'HEAT_STRESS_RISK_CATEGORY',
}

# ALLOWED temporal features
ALLOWED_TEMPORAL_FEATURES = {
    'month',
    'season',
    'year',
}

# ALLOWED demographic features (non-outcome)
ALLOWED_DEMOGRAPHIC_FEATURES = {
    'Age (at enrolment)',
    'Sex',
    'Race',
}

# COMPREHENSIVE BIOMARKER BLACKLIST - NEVER allow these as features
BIOMARKER_BLACKLIST = {
    # Primary biomarkers
    'CD4 cell count (cells/µL)',
    'hemoglobin_g_dL',
    'Hematocrit (%)',
    'fasting_glucose_mmol_L',
    'creatinine_umol_L',
    'creatinine clearance',
    'total_cholesterol_mg_dL',
    'hdl_cholesterol_mg_dL',
    'ldl_cholesterol_mg_dL',
    'FASTING HDL',
    'FASTING LDL',
    'FASTING TRIGLYCERIDES',
    'Triglycerides (mg/dL)',
    'ALT (U/L)',
    'AST (U/L)',

    # Blood cell counts
    'White blood cell count (×10³/µL)',
    'Red blood cell count (×10⁶/µL)',
    'Platelet count (×10³/µL)',
    'Lymphocyte count (×10³/µL)',
    'Neutrophil count (×10³/µL)',
    'Lymphocyte percentage (%)',
    'Lymphocytes (%)',
    'Neutrophil percentage (%)',
    'Neutrophils (%)',
    'Monocyte percentage (%)',
    'Monocytes (%)',
    'Eosinophil percentage (%)',
    'Eosinophils (%)',
    'Basophil percentage (%)',
    'Basophils (%)',
    'Erythrocytes',
    'MCV (MEAN CELL VOLUME)',
    'Mean corpuscular volume (fL)',
    'mch_pg',
    'mchc_g_dL',
    'RDW',

    # Vital signs
    'systolic_bp_mmHg',
    'diastolic_bp_mmHg',
    'heart_rate_bpm',
    'Respiratory rate (breaths/min)',
    'respiration rate',
    'Oxygen saturation (%)',
    'body_temperature_celsius',  # CRITICAL: This was causing leakage!

    # Anthropometric measures
    'BMI (kg/m²)',
    'height_m',
    'Last height recorded (m)',
    'weight_kg',
    'Last weight recorded (kg)',
    'Waist circumference (cm)',
    'Other measures of obesity',

    # Other clinical
    'Albumin (g/dL)',
    'Total protein (g/dL)',
    'Alkaline phosphatase (U/L)',
    'Total bilirubin (mg/dL)',
    'Sodium (mEq/L)',
    'Potassium (mEq/L)',
    'HIV viral load (copies/mL)',
}


def validate_no_biomarker_leakage(features: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that no biomarkers are in the feature set.

    Args:
        features: List of feature column names

    Returns:
        Tuple of (is_valid, list_of_leaked_biomarkers)
    """
    leaked = [f for f in features if f in BIOMARKER_BLACKLIST]
    is_valid = len(leaked) == 0

    if not is_valid:
        logger.error(f"CRITICAL: {len(leaked)} biomarkers found in feature set!")
        logger.error(f"Leaked biomarkers: {leaked}")

    return is_valid, leaked


class RefinedClimateHealthPipeline:
    """
    Refined pipeline for climate-health biomarker analysis with FIXED feature selection.
    """

    def __init__(self, data_path: str = "data/raw", output_dir: str = "results/refined_analysis_FIXED"):
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

        logger.info(f"Initialized Refined Climate-Health Pipeline (FIXED)")
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

    def identify_safe_features(self, clinical_df: pd.DataFrame) -> List[str]:
        """
        Identify SAFE features using strict whitelist approach.

        ONLY allows:
        - Climate features (climate_* prefix)
        - Heat vulnerability features (HEAT_* prefix)
        - Temporal features (month, season, year)
        - Demographics (Age, Sex, Race)

        NEVER allows biomarkers as features.

        Args:
            clinical_df: Clinical dataframe

        Returns:
            List of safe feature column names
        """
        logger.info("Identifying safe features (climate + socioeconomic only)...")

        safe_features = []

        for col in clinical_df.columns:
            # Skip if not numeric (except some temporal/categorical we'll handle)
            if not pd.api.types.is_numeric_dtype(clinical_df[col]):
                # Allow non-numeric temporal/categorical for later encoding
                if col in ALLOWED_TEMPORAL_FEATURES or col in ALLOWED_DEMOGRAPHIC_FEATURES:
                    if clinical_df[col].notna().sum() > 100:
                        safe_features.append(col)
                continue

            # Check if column has sufficient data
            if clinical_df[col].notna().sum() < 1000:
                continue

            # WHITELIST 1: Climate features (prefix-based)
            if col.startswith(ALLOWED_CLIMATE_PREFIXES):
                safe_features.append(col)
                continue

            # WHITELIST 2: Socioeconomic features (explicit)
            if col in ALLOWED_SOCIOECONOMIC_FEATURES:
                safe_features.append(col)
                continue

            # WHITELIST 3: Temporal features
            if col in ALLOWED_TEMPORAL_FEATURES:
                safe_features.append(col)
                continue

            # WHITELIST 4: Demographics
            if col in ALLOWED_DEMOGRAPHIC_FEATURES:
                safe_features.append(col)
                continue

        # CRITICAL: Validate no biomarker leakage
        is_valid, leaked = validate_no_biomarker_leakage(safe_features)

        if not is_valid:
            raise ValueError(f"BIOMARKER LEAKAGE DETECTED! Features: {leaked}")

        logger.info(f"Selected {len(safe_features)} safe features")
        logger.info(f"Feature breakdown:")
        climate_count = sum(1 for f in safe_features if f.startswith('climate_'))
        heat_count = sum(1 for f in safe_features if f.startswith('HEAT_'))
        temporal_count = sum(1 for f in safe_features if f in ALLOWED_TEMPORAL_FEATURES)
        demo_count = sum(1 for f in safe_features if f in ALLOWED_DEMOGRAPHIC_FEATURES)

        logger.info(f"  - Climate features: {climate_count}")
        logger.info(f"  - Heat vulnerability: {heat_count}")
        logger.info(f"  - Temporal features: {temporal_count}")
        logger.info(f"  - Demographic features: {demo_count}")
        logger.info(f"✓ NO biomarker leakage detected")

        return safe_features

    def prepare_features(self, df: pd.DataFrame, biomarker: str,
                        safe_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            df: Input dataframe
            biomarker: Target biomarker column
            safe_features: List of SAFE feature columns (no biomarkers)

        Returns:
            Tuple of (X, y) for modeling
        """
        # Start with complete cases for target
        df_clean = df[df[biomarker].notna()].copy()
        logger.info(f"Complete cases for {biomarker}: {len(df_clean)}")

        # CRITICAL: Ensure target biomarker is NOT in features
        if biomarker in safe_features:
            raise ValueError(f"Target biomarker {biomarker} found in feature set!")

        # Select features
        feature_cols = safe_features.copy()

        # Handle categorical features
        if 'season' in df_clean.columns and df_clean['season'].dtype == 'object':
            # One-hot encode season
            season_dummies = pd.get_dummies(df_clean['season'], prefix='season', drop_first=True)
            df_clean = pd.concat([df_clean, season_dummies], axis=1)
            feature_cols.extend(season_dummies.columns.tolist())
            feature_cols.remove('season')  # Remove original categorical

        if 'Sex' in df_clean.columns and df_clean['Sex'].dtype == 'object':
            # One-hot encode sex
            sex_dummies = pd.get_dummies(df_clean['Sex'], prefix='sex', drop_first=True)
            df_clean = pd.concat([df_clean, sex_dummies], axis=1)
            feature_cols.extend(sex_dummies.columns.tolist())
            feature_cols.remove('Sex')

        if 'Race' in df_clean.columns and df_clean['Race'].dtype == 'object':
            # One-hot encode race
            race_dummies = pd.get_dummies(df_clean['Race'], prefix='race', drop_first=True)
            df_clean = pd.concat([df_clean, race_dummies], axis=1)
            feature_cols.extend(race_dummies.columns.tolist())
            feature_cols.remove('Race')

        # Remove features with too many missing values
        feature_cols = [col for col in feature_cols if col in df_clean.columns]
        missing_pct = df_clean[feature_cols].isnull().mean()
        valid_features = missing_pct[missing_pct < 0.5].index.tolist()

        logger.info(f"Valid features after missing value filter: {len(valid_features)}")

        # FINAL VALIDATION: No biomarkers in features
        is_valid, leaked = validate_no_biomarker_leakage(valid_features)
        if not is_valid:
            raise ValueError(f"BIOMARKER LEAKAGE in final features: {leaked}")

        # Prepare X and y
        X = df_clean[valid_features].copy()
        y = df_clean[biomarker].copy()

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target: {biomarker}")
        logger.info(f"Target distribution: mean={y.mean():.2f}, std={y.std():.2f}, n={len(y)}")

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

        # Impute missing values AFTER split
        train_medians = X_train.median()
        X_train_imputed = X_train.fillna(train_medians)
        X_test_imputed = X_test.fillna(train_medians)

        logger.info(f"Imputed {X_train.isnull().sum().sum()} training missing values")
        logger.info(f"Imputed {X_test.isnull().sum().sum()} test missing values")

        # Scale features
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
            'feature_list': X.columns.tolist(),
            'models': {}
        }

        # Train multiple models
        models_to_train = {
            'LightGBM': lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1),
            'XGBoost': xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            'RandomForest': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, n_estimators=200)
        }

        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")

            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # Predict
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
                    'feature_names': X.columns.tolist(),
                    'train_medians': train_medians
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
        scaler = self.models[biomarker]['scaler']
        train_medians = self.models[biomarker]['train_medians']

        # Prepare data for SHAP (impute and scale)
        X_imputed = X.fillna(train_medians)
        X_scaled = pd.DataFrame(
            scaler.transform(X_imputed),
            columns=X.columns,
            index=X.index
        )

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values (sample if too large)
        sample_size = min(1000, len(X_scaled))
        X_sample = X_scaled.sample(n=sample_size, random_state=RANDOM_SEED)
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

        # Log top features
        logger.info(f"Top 5 features for {biomarker}:")
        for i, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def save_results(self, results: Dict, filename: str = "analysis_results_FIXED.json"):
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

    def run(self, target_biomarkers: Optional[List[str]] = None):
        """
        Run the complete refined analysis pipeline.

        Args:
            target_biomarkers: Optional list of specific biomarkers to analyze.
                              If None, analyzes all biomarkers.
        """
        self.start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting Refined Climate-Health Analysis Pipeline (FIXED)")
        logger.info("FEATURE LEAKAGE BUG - RESOLVED")
        logger.info("=" * 80)

        try:
            # Load data
            clinical_df, gcro_df = self.load_and_validate_data()

            # Identify biomarkers
            all_biomarkers = self.identify_biomarkers(clinical_df)

            # Use specified biomarkers or all
            if target_biomarkers:
                biomarkers_to_analyze = [b for b in target_biomarkers if b in all_biomarkers]
                logger.info(f"Analyzing {len(biomarkers_to_analyze)} specified biomarkers")
            else:
                biomarkers_to_analyze = all_biomarkers
                logger.info(f"Analyzing all {len(biomarkers_to_analyze)} biomarkers")

            # Identify SAFE features (climate + socioeconomic only)
            safe_features = self.identify_safe_features(clinical_df)

            # Analyze each biomarker
            all_results = {}

            for biomarker in tqdm(biomarkers_to_analyze, desc="Analyzing biomarkers"):
                logger.info(f"\n{'='*60}")
                logger.info(f"Analyzing: {biomarker}")
                logger.info(f"{'='*60}")

                try:
                    # Prepare features
                    X, y = self.prepare_features(clinical_df, biomarker, safe_features)

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
                    import traceback
                    traceback.print_exc()
                    continue

            # Save all results
            self.save_results(all_results)

            # Print summary
            total_time = time.time() - self.start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            logger.info(f"Analyzed {len(all_results)} biomarkers")
            logger.info(f"Results saved to: {self.output_dir}")

            # Print R² summary
            logger.info(f"\n{'='*80}")
            logger.info("MODEL PERFORMANCE SUMMARY (Climate-only features)")
            logger.info(f"{'='*80}")
            for biomarker, result in sorted(all_results.items(),
                                           key=lambda x: x[1]['models']['LightGBM']['test_r2'],
                                           reverse=True):
                r2 = result['models']['LightGBM']['test_r2']
                mae = result['models']['LightGBM']['test_mae']
                logger.info(f"{biomarker:40s} - R²: {r2:6.4f}, MAE: {mae:8.2f}")

            logger.info(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point for the refined pipeline."""
    print("\n" + "="*80)
    print("Refined Climate-Health Analysis Pipeline v2.1 (LEAKAGE FIX)")
    print("="*80 + "\n")

    # Initialize and run pipeline
    pipeline = RefinedClimateHealthPipeline()

    # Analyze specific biomarkers for testing (or None for all)
    # Start with hematocrit to verify fix
    target_biomarkers = ['Hematocrit (%)', 'hemoglobin_g_dL', 'CD4 cell count (cells/µL)']

    pipeline.run(target_biomarkers=target_biomarkers)

    print("\n" + "="*80)
    print("Analysis complete! Check the results directory for outputs.")
    print("Expected: Hematocrit R² should be 0.05-0.30 (NOT 0.93)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
