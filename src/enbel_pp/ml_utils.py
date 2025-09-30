#!/usr/bin/env python3
"""
Machine Learning Utilities for ENBEL Climate-Health Analysis
============================================================

Common ML functions, model training, validation, and statistical corrections
to ensure reproducibility and eliminate code duplication.

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import shap
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from scipy import stats
import warnings
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLPipelineError(Exception):
    """Custom exception for ML pipeline failures."""
    pass

def prepare_features_safely(df: pd.DataFrame, 
                          feature_columns: List[str],
                          target_column: str,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Safely prepare features and target with proper validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    feature_columns : List[str]
        List of feature column names
    target_column : str
        Target column name
    test_size : float
        Proportion of data for testing
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
        
    Raises:
    -------
    MLPipelineError
        If data preparation fails
    """
    try:
        # Validate columns exist
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise MLPipelineError(f"Missing feature columns: {missing_features}")
        
        if target_column not in df.columns:
            raise MLPipelineError(f"Missing target column: {target_column}")
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column]).copy()
        logger.info(f"Removed {len(df) - len(df_clean)} rows with missing target")
        
        if len(df_clean) == 0:
            raise MLPipelineError("No valid data remaining after removing missing targets")
        
        # Prepare features and target
        X = df_clean[feature_columns].copy()
        y = df_clean[target_column].copy()
        
        # Handle missing values in features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
        
        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            X[col] = pd.Categorical(X[col]).codes
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        raise MLPipelineError(f"Feature preparation failed: {e}")

def train_model_with_cv(X: pd.DataFrame, 
                       y: pd.Series,
                       model_type: str = 'random_forest',
                       model_params: Optional[Dict] = None,
                       cv_folds: int = 5,
                       scoring: List[str] = ['r2', 'neg_mean_absolute_error'],
                       feature_selection: bool = False,
                       max_features: Optional[int] = None) -> Dict[str, Any]:
    """
    Train model with proper cross-validation and feature selection.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model_type : str
        Type of model ('random_forest', 'xgboost')
    model_params : Dict, optional
        Model hyperparameters
    cv_folds : int
        Number of CV folds
    scoring : List[str]
        Scoring metrics for CV
    feature_selection : bool
        Whether to perform feature selection
    max_features : int, optional
        Maximum number of features to select
        
    Returns:
    --------
    Dict[str, Any]
        Training results including model, CV scores, feature importance
    """
    try:
        # Initialize model
        if model_params is None:
            model_params = {}
            
        if model_type == 'random_forest':
            default_params = {'n_estimators': 250, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1}
            default_params.update(model_params)
            model = RandomForestRegressor(**default_params)
        elif model_type == 'xgboost':
            default_params = {'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 200, 'random_state': 42}
            default_params.update(model_params)
            model = xgb.XGBRegressor(**default_params)
        else:
            raise MLPipelineError(f"Unsupported model type: {model_type}")
        
        # Feature selection within CV (to avoid leakage)
        if feature_selection and max_features and max_features < X.shape[1]:
            logger.info(f"Performing feature selection: {X.shape[1]} -> {max_features} features")
            
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
            
            # Perform cross-validation with feature selection inside
            cv_results = cross_validate_with_feature_selection(
                X, y, model, selector, cv_folds, scoring
            )
            
            # Train final model with selected features
            X_selected = selector.fit_transform(X, y)
            model.fit(X_selected, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            cv_results['selected_features'] = selected_features
            
        else:
            # Standard cross-validation
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                      return_train_score=True, n_jobs=-1)
            
            # Train final model
            model.fit(X, y)
            cv_results['selected_features'] = X.columns.tolist()
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': cv_results['selected_features'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = None
        
        results = {
            'model': model,
            'cv_scores': {
                'test_r2_mean': cv_results['test_r2'].mean(),
                'test_r2_std': cv_results['test_r2'].std(),
                'test_mae_mean': -cv_results['test_neg_mean_absolute_error'].mean(),
                'test_mae_std': cv_results['test_neg_mean_absolute_error'].std(),
                'train_r2_mean': cv_results['train_r2'].mean() if 'train_r2' in cv_results else None
            },
            'feature_importance': feature_importance,
            'selected_features': cv_results['selected_features'],
            'n_features': len(cv_results['selected_features'])
        }
        
        logger.info(f"Model trained successfully: R² = {results['cv_scores']['test_r2_mean']:.3f} ± {results['cv_scores']['test_r2_std']:.3f}")
        return results
        
    except Exception as e:
        raise MLPipelineError(f"Model training failed: {e}")

def cross_validate_with_feature_selection(X: pd.DataFrame, 
                                         y: pd.Series,
                                         model,
                                         selector,
                                         cv_folds: int,
                                         scoring: List[str]) -> Dict[str, np.ndarray]:
    """
    Perform cross-validation with feature selection inside each fold to avoid leakage.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model : sklearn estimator
        Machine learning model
    selector : sklearn feature selector
        Feature selection method
    cv_folds : int
        Number of CV folds
    scoring : List[str]
        Scoring metrics
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Cross-validation results
    """
    from sklearn.model_selection import KFold
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Initialize result arrays
    test_r2_scores = []
    test_mae_scores = []
    train_r2_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        # Split data
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        # Feature selection on training data only
        X_train_selected = selector.fit_transform(X_train_fold, y_train_fold)
        X_test_selected = selector.transform(X_test_fold)
        
        # Train model
        model.fit(X_train_selected, y_train_fold)
        
        # Predictions
        y_pred_test = model.predict(X_test_selected)
        y_pred_train = model.predict(X_train_selected)
        
        # Calculate scores
        test_r2_scores.append(r2_score(y_test_fold, y_pred_test))
        test_mae_scores.append(mean_absolute_error(y_test_fold, y_pred_test))
        train_r2_scores.append(r2_score(y_train_fold, y_pred_train))
    
    return {
        'test_r2': np.array(test_r2_scores),
        'test_neg_mean_absolute_error': -np.array(test_mae_scores),
        'train_r2': np.array(train_r2_scores)
    }

def calculate_shap_values(model, X: pd.DataFrame, 
                         sample_size: Optional[int] = 100) -> np.ndarray:
    """
    Calculate SHAP values for model interpretability.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : pd.DataFrame
        Feature matrix
    sample_size : int, optional
        Number of samples for SHAP calculation
        
    Returns:
    --------
    np.ndarray
        SHAP values
    """
    try:
        # Sample data if too large
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create SHAP explainer
        if hasattr(model, 'predict'):
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            return shap_values.values
        else:
            raise MLPipelineError("Model does not support SHAP analysis")
            
    except Exception as e:
        logger.warning(f"SHAP calculation failed: {e}")
        return None

def apply_multiple_testing_correction(p_values: List[float], 
                                    method: str = 'bonferroni',
                                    alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply multiple testing correction to p-values.
    
    Parameters:
    -----------
    p_values : List[float]
        List of p-values to correct
    method : str
        Correction method ('bonferroni', 'fdr_bh')
    alpha : float
        Significance level
        
    Returns:
    --------
    Dict[str, Any]
        Correction results
    """
    p_values = np.array(p_values)
    
    if method == 'bonferroni':
        corrected_alpha = alpha / len(p_values)
        significant = p_values < corrected_alpha
        corrected_p = p_values * len(p_values)
        corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        from statsmodels.stats.multitest import multipletests
        significant, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        corrected_alpha = alpha
        
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return {
        'original_p_values': p_values,
        'corrected_p_values': corrected_p,
        'significant': significant,
        'corrected_alpha': corrected_alpha,
        'n_significant': significant.sum(),
        'method': method
    }

def evaluate_model_performance(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             model_name: str = "Model") -> Dict[str, float]:
    """
    Calculate comprehensive model performance metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Name of the model for logging
        
    Returns:
    --------
    Dict[str, float]
        Performance metrics
    """
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'pearson_r': stats.pearsonr(y_true, y_pred)[0],
        'spearman_r': stats.spearmanr(y_true, y_pred)[0]
    }
    
    logger.info(f"{model_name} performance: R² = {metrics['r2_score']:.3f}, "
               f"MAE = {metrics['mae']:.3f}, RMSE = {metrics['rmse']:.3f}")
    
    return metrics

def save_model_results(results: Dict[str, Any], 
                      output_path: Path,
                      biomarker_name: str) -> None:
    """
    Save model training results to file.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Model training results
    output_path : Path
        Output directory path
    biomarker_name : str
        Name of the biomarker
    """
    import pickle
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_file = output_path / f"{biomarker_name}_model_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(results['model'], f)
    
    # Save results (excluding model object)
    results_to_save = {k: v for k, v in results.items() if k != 'model'}
    
    # Convert numpy arrays and other non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Recursively convert results
    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_convert(v) for v in d]
        else:
            return convert_for_json(d)
    
    results_serializable = recursive_convert(results_to_save)
    
    results_file = output_path / f"{biomarker_name}_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Model saved: {model_file}")
    logger.info(f"Results saved: {results_file}")

if __name__ == "__main__":
    # Test utilities
    logger.info("Testing ML utilities...")
    
    # Create synthetic test data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(X.sum(axis=1) + np.random.randn(100) * 0.1)
    
    # Test model training
    results = train_model_with_cv(X, y, model_type='random_forest')
    print(f"Test R² score: {results['cv_scores']['test_r2_mean']:.3f}")
    
    logger.info("ML utilities test completed successfully!")