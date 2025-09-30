"""
Machine Learning Pipeline for Biomarker Analysis with SHAP Explainability
=========================================================================

This module implements a comprehensive ML pipeline for analyzing climate-health
relationships across multiple physiological systems (renal, cardiovascular, 
immune, metabolic) using high-quality models and SHAP values for interpretation.

Key Features:
- System-specific biomarker grouping
- Rigorous model evaluation metrics
- Hyperparameter optimization for model quality
- SHAP analysis for feature importance and interactions
- Comprehensive result visualization and reporting

Author: ENBEL Project Team
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                           mean_absolute_percentage_error, explained_variance_score)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BiomarkerMLPipeline:
    """
    Comprehensive ML pipeline for biomarker analysis with focus on model quality
    and interpretability through SHAP analysis.
    """
    
    def __init__(self, random_state: int = 42, min_model_quality: float = 0.3):
        """
        Initialize the ML pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        min_model_quality : float
            Minimum R² score required for SHAP analysis
        """
        self.random_state = random_state
        self.min_model_quality = min_model_quality
        np.random.seed(random_state)
        
        # Define physiological systems and their biomarkers
        self.physiological_systems = {
            'immune': ['CD4_cell_count', 'CD4', 'lymphocyte', 'white_blood_cell'],
            'metabolic': ['glucose', 'fasting_glucose', 'HbA1c', 'insulin'],
            'cardiovascular': ['systolic_blood_pressure', 'diastolic_blood_pressure', 
                             'heart_rate', 'pulse_pressure'],
            'renal': ['creatinine', 'urea', 'eGFR', 'albumin'],
            'lipid': ['cholesterol', 'HDL', 'LDL', 'triglycerides'],
            'hematological': ['hemoglobin', 'hematocrit', 'red_blood_cell', 'platelet']
        }
        
        # Model configurations for optimization
        self.model_configs = {
            'random_forest': {
                'model_class': RandomForestRegressor,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3]
                }
            },
            'xgboost': {
                'model_class': xgb.XGBRegressor,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            },
            'lightgbm': {
                'model_class': lgb.LGBMRegressor,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'num_leaves': [31, 50, 100],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'feature_fraction': [0.7, 0.8, 0.9],
                    'bagging_fraction': [0.7, 0.8, 0.9]
                }
            }
        }
        
        self.results = {}
        self.models = {}
        self.shap_values = {}
        
        logger.info(f"Initialized Biomarker ML Pipeline (min quality: R²>{min_model_quality})")
    
    def identify_biomarkers(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify available biomarkers in the dataset by physiological system.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
            
        Returns:
        --------
        Dict[str, List[str]]
            Available biomarkers grouped by system
        """
        available_biomarkers = {}
        
        for system, biomarker_patterns in self.physiological_systems.items():
            found_biomarkers = []
            for pattern in biomarker_patterns:
                matching_cols = [col for col in df.columns 
                               if pattern.lower() in col.lower() and 
                               df[col].dtype in [np.float64, np.int64]]
                found_biomarkers.extend(matching_cols)
            
            if found_biomarkers:
                available_biomarkers[system] = list(set(found_biomarkers))
                logger.info(f"{system.capitalize()} system: {len(available_biomarkers[system])} biomarkers found")
        
        return available_biomarkers
    
    def prepare_features(self, 
                        df: pd.DataFrame,
                        target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for ML analysis with careful selection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_col : str
            Target biomarker column
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            Feature matrix, target vector, feature names
        """
        # Identify climate features
        climate_features = [col for col in df.columns 
                          if any(pattern in col.lower() 
                                for pattern in ['temp', 'humid', 'pressure', 'wind', 
                                              'climate', 'heat', 'lag', 'rolling'])]
        
        # Identify demographic features
        demographic_features = [col for col in df.columns
                              if any(pattern in col.lower()
                                    for pattern in ['age', 'sex', 'race', 'gender'])]
        
        # Identify socioeconomic features
        socioeconomic_features = [col for col in df.columns
                                 if any(pattern in col.lower()
                                       for pattern in ['dwelling', 'education', 'income', 
                                                      'employment', 'household'])]
        
        # Combine all features
        all_features = climate_features + demographic_features + socioeconomic_features
        
        # Remove target from features
        all_features = [f for f in all_features if f != target_col and f in df.columns]
        
        # Select only numeric features
        numeric_features = [f for f in all_features 
                          if df[f].dtype in [np.float64, np.int64]]
        
        # Prepare data
        feature_df = df[numeric_features].copy()
        target_series = df[target_col].copy()
        
        # Remove rows with missing target
        valid_mask = target_series.notna()
        feature_df = feature_df[valid_mask]
        target_series = target_series[valid_mask]
        
        # Handle remaining missing values in features
        feature_df = feature_df.fillna(feature_df.median())
        
        logger.info(f"Prepared {len(numeric_features)} features for {target_col}")
        logger.info(f"  - Climate features: {len([f for f in numeric_features if f in climate_features])}")
        logger.info(f"  - Demographic features: {len([f for f in numeric_features if f in demographic_features])}")
        logger.info(f"  - Socioeconomic features: {len([f for f in numeric_features if f in socioeconomic_features])}")
        
        return feature_df, target_series, numeric_features
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        model_name : str
            Name of the model
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
        
        # Add MAPE if no zeros in true values
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        # Calculate correlation
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Model quality assessment
        metrics['quality_assessment'] = self._assess_model_quality(metrics)
        
        logger.info(f"{model_name} - R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def _assess_model_quality(self, metrics: Dict[str, float]) -> str:
        """Assess overall model quality based on metrics."""
        r2 = metrics['r2_score']
        
        if r2 >= 0.7:
            return "Excellent"
        elif r2 >= 0.5:
            return "Good"
        elif r2 >= 0.3:
            return "Moderate"
        elif r2 >= 0.1:
            return "Weak"
        else:
            return "Poor"
    
    def train_optimized_model(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            model_type: str = 'xgboost',
                            cv_folds: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with hyperparameter optimization.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        model_type : str
            Type of model to train
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            Trained model and optimization results
        """
        logger.info(f"Training optimized {model_type} model...")
        
        config = self.model_configs[model_type]
        base_model = config['model_class'](random_state=self.random_state)
        
        # Use a subset of parameter grid for faster optimization
        param_grid = {}
        for param, values in config['param_grid'].items():
            param_grid[param] = values[:2] if len(values) > 2 else values
        
        # Hyperparameter optimization
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Cross-validation evaluation
        cv_scores = cross_validate(
            best_model,
            X_train, y_train,
            cv=cv_folds,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            return_train_score=True
        )
        
        optimization_results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_scores': {
                'r2_mean': np.mean(cv_scores['test_r2']),
                'r2_std': np.std(cv_scores['test_r2']),
                'mae_mean': -np.mean(cv_scores['test_neg_mean_absolute_error']),
                'rmse_mean': -np.mean(cv_scores['test_neg_root_mean_squared_error'])
            }
        }
        
        return best_model, optimization_results
    
    def perform_shap_analysis(self,
                            model: Any,
                            X: pd.DataFrame,
                            feature_names: List[str],
                            biomarker: str) -> Dict[str, Any]:
        """
        Perform SHAP analysis for model interpretability.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature matrix
        feature_names : List[str]
            Feature names
        biomarker : str
            Target biomarker name
            
        Returns:
        --------
        Dict[str, Any]
            SHAP analysis results
        """
        logger.info(f"Performing SHAP analysis for {biomarker}...")
        
        # Create SHAP explainer
        if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X.sample(100))
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Identify top climate predictors
        climate_features = [f for f in feature_names 
                          if any(p in f.lower() for p in ['temp', 'humid', 'climate', 'heat'])]
        climate_importance = importance_df[importance_df['feature'].isin(climate_features)]
        
        # Calculate interaction effects for top features
        top_features = importance_df.head(10)['feature'].tolist()
        
        shap_results = {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'top_features': top_features,
            'climate_importance': climate_importance,
            'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
        }
        
        return shap_results
    
    def analyze_biomarker(self,
                        df: pd.DataFrame,
                        biomarker: str,
                        model_types: List[str] = ['xgboost', 'lightgbm']) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a single biomarker.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        biomarker : str
            Target biomarker
        model_types : List[str]
            Types of models to train
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive analysis results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing biomarker: {biomarker}")
        logger.info(f"{'='*60}")
        
        # Prepare features
        try:
            X, y, feature_names = self.prepare_features(df, biomarker)
        except Exception as e:
            logger.error(f"Feature preparation failed for {biomarker}: {e}")
            return {'error': str(e)}
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for {biomarker}: {len(X)} samples")
            return {'error': 'Insufficient data'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_names,
            index=X_test.index
        )
        
        results = {
            'biomarker': biomarker,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'models': {}
        }
        
        best_score = -float('inf')
        best_model = None
        best_model_type = None
        
        # Train different models
        for model_type in model_types:
            try:
                # Train optimized model
                model, opt_results = self.train_optimized_model(
                    X_train_scaled, y_train, model_type
                )
                
                # Evaluate on test set
                y_pred = model.predict(X_test_scaled)
                test_metrics = self.evaluate_model(y_test, y_pred, model_type)
                
                # Store results
                results['models'][model_type] = {
                    'optimization': opt_results,
                    'test_metrics': test_metrics
                }
                
                # Track best model
                if test_metrics['r2_score'] > best_score:
                    best_score = test_metrics['r2_score']
                    best_model = model
                    best_model_type = model_type
                
            except Exception as e:
                logger.error(f"Training failed for {model_type}: {e}")
                results['models'][model_type] = {'error': str(e)}
        
        # Perform SHAP analysis if model quality is sufficient
        if best_model and best_score >= self.min_model_quality:
            logger.info(f"Best model ({best_model_type}) achieved R²={best_score:.4f}")
            
            shap_results = self.perform_shap_analysis(
                best_model, X_test_scaled, feature_names, biomarker
            )
            
            results['shap_analysis'] = {
                'model_type': best_model_type,
                'model_r2': best_score,
                'top_features': shap_results['top_features'],
                'top_climate_features': shap_results['climate_importance'].head(5)['feature'].tolist()
            }
            
            # Store for later use
            self.models[biomarker] = best_model
            self.shap_values[biomarker] = shap_results
        else:
            logger.warning(f"Model quality insufficient for SHAP analysis (R²={best_score:.4f} < {self.min_model_quality})")
            results['shap_analysis'] = None
        
        results['best_model'] = best_model_type
        results['best_r2'] = best_score
        
        return results
    
    def analyze_all_systems(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze all physiological systems.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
            
        Returns:
        --------
        Dict[str, Any]
            Results for all systems
        """
        logger.info("\nStarting comprehensive biomarker analysis across all systems...")
        
        # Identify available biomarkers
        available_biomarkers = self.identify_biomarkers(df)
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'systems': {},
            'summary': {
                'total_biomarkers': 0,
                'successful_models': 0,
                'shap_analyses': 0
            }
        }
        
        # Analyze each system
        for system, biomarkers in available_biomarkers.items():
            logger.info(f"\nAnalyzing {system.upper()} system...")
            system_results = {}
            
            for biomarker in biomarkers:
                biomarker_results = self.analyze_biomarker(df, biomarker)
                system_results[biomarker] = biomarker_results
                
                comprehensive_results['summary']['total_biomarkers'] += 1
                
                if 'error' not in biomarker_results:
                    comprehensive_results['summary']['successful_models'] += 1
                    
                    if biomarker_results.get('shap_analysis'):
                        comprehensive_results['summary']['shap_analyses'] += 1
            
            comprehensive_results['systems'][system] = system_results
        
        self.results = comprehensive_results
        return comprehensive_results
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.results:
            return "No analysis results available"
        
        report = []
        report.append("=" * 70)
        report.append("BIOMARKER MACHINE LEARNING ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {self.results['timestamp']}")
        report.append("")
        
        # Summary statistics
        summary = self.results['summary']
        report.append("ANALYSIS SUMMARY")
        report.append("-" * 40)
        report.append(f"Total biomarkers analyzed: {summary['total_biomarkers']}")
        report.append(f"Successful models: {summary['successful_models']}")
        report.append(f"SHAP analyses performed: {summary['shap_analyses']}")
        report.append("")
        
        # Results by system
        report.append("RESULTS BY PHYSIOLOGICAL SYSTEM")
        report.append("-" * 40)
        
        for system, system_results in self.results['systems'].items():
            report.append(f"\n{system.upper()} SYSTEM:")
            
            # Collect successful models
            successful = []
            for biomarker, results in system_results.items():
                if 'error' not in results and 'best_r2' in results:
                    successful.append((biomarker, results['best_r2'], results.get('best_model')))
            
            if successful:
                successful.sort(key=lambda x: x[1], reverse=True)
                for biomarker, r2, model in successful:
                    quality = "✓ SHAP" if r2 >= self.min_model_quality else ""
                    report.append(f"  {biomarker[:30]:<30} R²={r2:.3f} ({model}) {quality}")
            else:
                report.append("  No successful models")
        
        # Top climate predictors
        report.append("\n" + "=" * 70)
        report.append("TOP CLIMATE PREDICTORS (from SHAP analysis)")
        report.append("-" * 40)
        
        climate_features_count = {}
        for system_results in self.results['systems'].values():
            for biomarker_results in system_results.values():
                if biomarker_results.get('shap_analysis'):
                    for feature in biomarker_results['shap_analysis'].get('top_climate_features', []):
                        climate_features_count[feature] = climate_features_count.get(feature, 0) + 1
        
        if climate_features_count:
            sorted_features = sorted(climate_features_count.items(), key=lambda x: x[1], reverse=True)
            for feature, count in sorted_features[:10]:
                report.append(f"  {feature[:40]:<40} (appears in {count} models)")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "results"):
        """Save all results and models."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results JSON
        results_path = f"{output_dir}/ml_analysis_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = json.loads(json.dumps(self.results, default=str))
            json.dump(json_results, f, indent=2)
        
        # Save summary report
        report_path = f"{output_dir}/ml_analysis_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_summary_report())
        
        logger.info(f"Results saved to {output_dir}")
        
        return results_path, report_path


def run_biomarker_ml_analysis(data_path: str,
                             output_dir: str = "results",
                             min_model_quality: float = 0.3) -> Dict[str, Any]:
    """
    Run complete biomarker ML analysis pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to imputed dataset
    output_dir : str
        Directory for saving results
    min_model_quality : float
        Minimum R² for SHAP analysis
        
    Returns:
    --------
    Dict[str, Any]
        Analysis results
    """
    # Load data
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Initialize pipeline
    pipeline = BiomarkerMLPipeline(
        random_state=42,
        min_model_quality=min_model_quality
    )
    
    # Run analysis
    results = pipeline.analyze_all_systems(df)
    
    # Save results
    pipeline.save_results(output_dir)
    
    # Print summary
    print("\n" + pipeline.generate_summary_report())
    
    return results


if __name__ == "__main__":
    print("Biomarker Machine Learning Analysis Pipeline")
    print("=" * 50)
    
    # Example usage
    data_file = "data/imputed/clinical_imputed_latest.csv"
    
    if Path(data_file).exists():
        results = run_biomarker_ml_analysis(
            data_path=data_file,
            output_dir="results/ml_analysis",
            min_model_quality=0.3
        )
        print(f"\nAnalysis complete. Analyzed {results['summary']['total_biomarkers']} biomarkers")
    else:
        print("Please run imputation pipeline first to generate imputed dataset.")