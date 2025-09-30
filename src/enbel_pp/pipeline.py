#!/usr/bin/env python3
"""
Climate-Health Analysis Pipeline
===============================

Comprehensive pipeline for analyzing climate impacts on health biomarkers.
This module consolidates and improves upon the various analysis approaches
developed in the ENBEL project.

Key Features:
- Configurable analysis pipelines (simple, improved, state-of-the-art)
- Proper data validation and preprocessing
- Multiple ML algorithms with hyperparameter optimization
- Explainable AI analysis with SHAP
- Comprehensive result reporting and visualization
- Reproducible analysis workflows

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# Local imports
from .config import get_config, set_reproducible_environment
from .data_validation import validate_file_exists, validate_biomarker_data
from .ml_utils import (
    prepare_features_safely,
    train_model_with_cv,
    evaluate_model_performance,
    apply_multiple_testing_correction
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ClimateHealthPipeline:
    """
    Comprehensive Climate-Health Analysis Pipeline
    
    This pipeline provides different analysis modes:
    - 'simple': Basic analysis for quick exploration
    - 'improved': Enhanced analysis with feature engineering
    - 'state_of_the_art': Full analysis with optimization and XAI
    """
    
    def __init__(self, 
                 analysis_mode: str = 'improved',
                 config_file: Optional[str] = None,
                 random_seed: int = 42):
        """
        Initialize the climate-health analysis pipeline.
        
        Parameters:
        -----------
        analysis_mode : str
            Analysis complexity level ('simple', 'improved', 'state_of_the_art')
        config_file : str, optional
            Path to custom configuration file
        random_seed : int
            Random seed for reproducibility
        """
        self.analysis_mode = analysis_mode
        self.random_seed = random_seed
        
        # Set up configuration
        self.config = get_config()
        if config_file:
            self.config.load_config(config_file)
        
        # Set reproducible environment
        set_reproducible_environment(random_seed)
        
        # Initialize components
        self.data = None
        self.results = {}
        self.models = {}
        
        # Configure analysis based on mode
        self._configure_analysis_mode()
        
        logger.info(f"Initialized {analysis_mode} pipeline with seed {random_seed}")
    
    def _configure_analysis_mode(self):
        """Configure pipeline parameters based on analysis mode."""
        mode_configs = {
            'simple': {
                'feature_selection': False,
                'hyperparameter_tuning': False,
                'cross_validation_folds': 3,
                'shap_analysis': False,
                'max_features': 20
            },
            'improved': {
                'feature_selection': True,
                'hyperparameter_tuning': True,
                'cross_validation_folds': 5,
                'shap_analysis': False,
                'max_features': 50
            },
            'state_of_the_art': {
                'feature_selection': True,
                'hyperparameter_tuning': True,
                'cross_validation_folds': 10,
                'shap_analysis': True,
                'max_features': 100
            }
        }
        
        if self.analysis_mode not in mode_configs:
            raise ValueError(f"Unknown analysis mode: {self.analysis_mode}")
        
        self.mode_config = mode_configs[self.analysis_mode]
        logger.info(f"Configured for {self.analysis_mode} analysis")
    
    def load_data(self, data_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load and validate climate-health dataset.
        
        Parameters:
        -----------
        data_file : str, optional
            Path to data file. If None, uses default from config.
            
        Returns:
        --------
        pd.DataFrame
            Loaded and validated dataset
        """
        logger.info("Loading climate-health dataset...")
        
        if data_file is None:
            # Try different data files based on availability
            for data_type in ['full_dataset', 'clinical_imputed', 'clinical_original']:
                try:
                    data_path = self.config.get_data_path(data_type)
                    if data_path.exists():
                        data_file = str(data_path)
                        break
                except:
                    continue
            
            if data_file is None:
                raise FileNotFoundError("No suitable data file found in configured paths")
        
        # Validate file exists
        data_path = Path(data_file)
        validate_file_exists(data_path, "climate-health dataset")
        
        # Load data
        self.data = pd.read_csv(data_file, low_memory=False)
        
        # Validate biomarker data
        biomarker_stats = validate_biomarker_data(self.data)
        
        logger.info(f"Loaded dataset: {self.data.shape[0]:,} records, {self.data.shape[1]:,} features")
        logger.info(f"Found {len(biomarker_stats)} biomarkers with data")
        
        return self.data
    
    def identify_features(self) -> Dict[str, List[str]]:
        """
        Identify and categorize features for analysis.
        
        Returns:
        --------
        Dict[str, List[str]]
            Categorized feature lists
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        features = {
            'climate': [],
            'demographic': [],
            'biomarkers': [],
            'other': []
        }
        
        # Define feature patterns
        climate_patterns = [
            'temp', 'humid', 'heat', 'wind', 'pressure', 'climate',
            'weather', 'lag', 'rolling', 'anomaly'
        ]
        
        demographic_patterns = ['sex', 'race', 'age', 'gender']
        
        biomarker_patterns = [
            'cd4', 'glucose', 'cholesterol', 'hemoglobin', 'creatinine',
            'blood pressure', 'ldl', 'hdl', 'triglycerides', 'alt', 'ast'
        ]
        
        # Categorize features
        for column in self.data.columns:
            column_lower = column.lower()
            
            # Check for biomarkers first
            if any(pattern in column_lower for pattern in biomarker_patterns):
                if self.data[column].dtype in ['float64', 'int64']:
                    features['biomarkers'].append(column)
                continue
            
            # Check for climate features
            if any(pattern in column_lower for pattern in climate_patterns):
                if (self.data[column].dtype in ['float64', 'int64'] and 
                    self.data[column].notna().sum() / len(self.data) > 0.5):
                    features['climate'].append(column)
                continue
            
            # Check for demographic features
            if any(pattern in column_lower for pattern in demographic_patterns):
                features['demographic'].append(column)
                continue
            
            # Everything else
            features['other'].append(column)
        
        logger.info(f"Identified features: "
                   f"{len(features['climate'])} climate, "
                   f"{len(features['demographic'])} demographic, "
                   f"{len(features['biomarkers'])} biomarkers")
        
        return features
    
    def run_biomarker_analysis(self, 
                              biomarker: str,
                              features: List[str]) -> Dict[str, Any]:
        """
        Run complete analysis for a specific biomarker.
        
        Parameters:
        -----------
        biomarker : str
            Target biomarker for analysis
        features : List[str]
            List of feature columns to use as predictors
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive analysis results
        """
        logger.info(f"Analyzing biomarker: {biomarker}")
        
        # Prepare data
        try:
            X_train, X_test, y_train, y_test = prepare_features_safely(
                self.data, features, biomarker,
                test_size=self.config.get_ml_config()['test_size'],
                random_state=self.random_seed
            )
        except Exception as e:
            logger.error(f"Data preparation failed for {biomarker}: {e}")
            return {'error': str(e)}
        
        if len(X_train) < 100:
            logger.warning(f"Insufficient data for {biomarker}: {len(X_train)} samples")
            return {'error': 'Insufficient data'}
        
        results = {
            'biomarker': biomarker,
            'n_samples': len(X_train) + len(X_test),
            'n_features': X_train.shape[1],
            'models': {}
        }
        
        # Train models based on analysis mode
        model_types = ['random_forest']
        if self.analysis_mode in ['improved', 'state_of_the_art']:
            model_types.append('xgboost')
        
        for model_type in model_types:
            try:
                model_results = train_model_with_cv(
                    X_train, y_train,
                    model_type=model_type,
                    cv_folds=self.mode_config['cross_validation_folds'],
                    feature_selection=self.mode_config['feature_selection'],
                    max_features=self.mode_config['max_features'],
                    hyperparameter_tuning=self.mode_config['hyperparameter_tuning']
                )
                
                # Evaluate on test set
                model = model_results['model']
                selected_features = model_results.get('selected_features', features)
                X_test_selected = X_test[selected_features] if selected_features else X_test
                
                test_predictions = model.predict(X_test_selected)
                test_metrics = evaluate_model_performance(
                    y_test, test_predictions, f"{model_type}_{biomarker}"
                )
                
                model_results['test_metrics'] = test_metrics
                results['models'][model_type] = model_results
                
                logger.info(f"{model_type} R² = {test_metrics['r2_score']:.4f}")
                
            except Exception as e:
                logger.error(f"{model_type} training failed for {biomarker}: {e}")
                results['models'][model_type] = {'error': str(e)}
        
        # Select best model
        best_model_type = None
        best_score = -float('inf')
        
        for model_type, model_results in results['models'].items():
            if 'error' not in model_results:
                score = model_results['cv_scores']['test_r2_mean']
                if score > best_score:
                    best_score = score
                    best_model_type = model_type
        
        if best_model_type:
            results['best_model'] = best_model_type
            results['best_score'] = best_score
            logger.info(f"Best model for {biomarker}: {best_model_type} (R² = {best_score:.4f})")
        else:
            logger.warning(f"No successful models for {biomarker}")
        
        return results
    
    def run_comprehensive_analysis(self, 
                                 target_biomarkers: Optional[List[str]] = None,
                                 exclude_demographics: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive analysis across multiple biomarkers.
        
        Parameters:
        -----------
        target_biomarkers : List[str], optional
            List of biomarkers to analyze. If None, uses all available.
        exclude_demographics : bool
            Whether to exclude demographic features from analysis
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive analysis results
        """
        logger.info(f"Starting {self.analysis_mode} comprehensive analysis...")
        
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Identify features
        feature_groups = self.identify_features()
        
        # Select features for analysis
        analysis_features = feature_groups['climate'].copy()
        if not exclude_demographics:
            analysis_features.extend(feature_groups['demographic'])
        
        # Select biomarkers
        if target_biomarkers is None:
            available_biomarkers = self.config.get_biomarkers()
            target_biomarkers = [b for b in available_biomarkers 
                               if b in feature_groups['biomarkers']]
        
        logger.info(f"Analyzing {len(target_biomarkers)} biomarkers with {len(analysis_features)} features")
        
        # Run analysis for each biomarker
        comprehensive_results = {
            'analysis_mode': self.analysis_mode,
            'timestamp': datetime.now().isoformat(),
            'configuration': self.mode_config,
            'feature_summary': {
                'n_climate_features': len(feature_groups['climate']),
                'n_demographic_features': len(feature_groups['demographic']),
                'exclude_demographics': exclude_demographics,
                'total_features': len(analysis_features)
            },
            'biomarker_results': {},
            'summary_statistics': {}
        }
        
        successful_analyses = []
        p_values = []
        
        for i, biomarker in enumerate(target_biomarkers, 1):
            logger.info(f"[{i}/{len(target_biomarkers)}] Analyzing {biomarker}")
            
            biomarker_results = self.run_biomarker_analysis(biomarker, analysis_features)
            comprehensive_results['biomarker_results'][biomarker] = biomarker_results
            
            if 'error' not in biomarker_results and 'best_score' in biomarker_results:
                successful_analyses.append(biomarker_results)
                # Add p-value calculation if available
                if 'test_metrics' in biomarker_results.get('models', {}).get(biomarker_results.get('best_model', ''), {}):
                    # Use a simple significance test based on R²
                    r2 = biomarker_results['best_score']
                    n = biomarker_results['n_samples']
                    p = biomarker_results['n_features']
                    # Simple F-test approximation
                    if r2 > 0 and n > p + 1:
                        f_stat = (r2 / (1 - r2)) * ((n - p - 1) / p)
                        # Approximate p-value (would need scipy.stats for exact)
                        p_value = max(0.001, min(0.999, 1 / (1 + f_stat)))
                        p_values.append(p_value)
        
        # Calculate summary statistics
        if successful_analyses:
            scores = [result['best_score'] for result in successful_analyses]
            comprehensive_results['summary_statistics'] = {
                'n_successful_analyses': len(successful_analyses),
                'n_total_biomarkers': len(target_biomarkers),
                'success_rate': len(successful_analyses) / len(target_biomarkers),
                'mean_r2': np.mean(scores),
                'median_r2': np.median(scores),
                'std_r2': np.std(scores),
                'max_r2': np.max(scores),
                'min_r2': np.min(scores)
            }
            
            # Multiple testing correction if p-values available
            if p_values:
                correction_results = apply_multiple_testing_correction(
                    p_values, method='bonferroni', alpha=0.05
                )
                comprehensive_results['multiple_testing'] = correction_results
        
        # Save results
        self.results = comprehensive_results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.analysis_mode}_analysis_results_{timestamp}.json"
        
        self.save_results(results_file)
        
        logger.info(f"Analysis completed. Results saved to {results_file}")
        return comprehensive_results
    
    def save_results(self, filename: str) -> None:
        """Save analysis results to JSON file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Prepare results for JSON serialization
        json_results = self._prepare_results_for_json(self.results)
        
        results_path = self.config.get_output_path('results', filename)
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization by removing non-serializable objects."""
        json_results = {}
        
        for key, value in results.items():
            if key == 'biomarker_results':
                json_results[key] = {}
                for biomarker, biomarker_results in value.items():
                    json_results[key][biomarker] = self._serialize_biomarker_results(biomarker_results)
            elif isinstance(value, dict):
                json_results[key] = self._prepare_results_for_json(value)
            elif isinstance(value, (list, str, int, float, bool, type(None))):
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        return json_results
    
    def _serialize_biomarker_results(self, biomarker_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize biomarker results for JSON output."""
        serialized = {}
        
        for key, value in biomarker_results.items():
            if key == 'models':
                serialized[key] = {}
                for model_type, model_results in value.items():
                    serialized[key][model_type] = {}
                    for k, v in model_results.items():
                        if k == 'model':
                            # Don't serialize the actual model object
                            serialized[key][model_type][k] = f"<{type(v).__name__} object>"
                        elif isinstance(v, (dict, list, str, int, float, bool, type(None))):
                            serialized[key][model_type][k] = v
                        else:
                            serialized[key][model_type][k] = str(v)
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        if not self.results:
            return "No analysis results available"
        
        report = []
        report.append("ENBEL Climate-Health Analysis Summary Report")
        report.append("=" * 50)
        report.append(f"Analysis Mode: {self.results['analysis_mode']}")
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append("")
        
        # Feature summary
        fs = self.results['feature_summary']
        report.append("Feature Summary:")
        report.append(f"  - Climate features: {fs['n_climate_features']}")
        report.append(f"  - Demographic features: {fs['n_demographic_features']}")
        report.append(f"  - Demographics excluded: {fs['exclude_demographics']}")
        report.append(f"  - Total features used: {fs['total_features']}")
        report.append("")
        
        # Summary statistics
        if 'summary_statistics' in self.results:
            ss = self.results['summary_statistics']
            report.append("Analysis Summary:")
            report.append(f"  - Successful analyses: {ss['n_successful_analyses']}/{ss['n_total_biomarkers']}")
            report.append(f"  - Success rate: {ss['success_rate']:.1%}")
            report.append(f"  - Mean R²: {ss['mean_r2']:.4f}")
            report.append(f"  - Median R²: {ss['median_r2']:.4f}")
            report.append(f"  - Best R²: {ss['max_r2']:.4f}")
            report.append("")
        
        # Top performing biomarkers
        biomarker_scores = []
        for biomarker, results in self.results['biomarker_results'].items():
            if 'best_score' in results:
                biomarker_scores.append((biomarker, results['best_score'], results.get('best_model', 'unknown')))
        
        if biomarker_scores:
            biomarker_scores.sort(key=lambda x: x[1], reverse=True)
            report.append("Top Performing Biomarkers:")
            for i, (biomarker, score, model) in enumerate(biomarker_scores[:5], 1):
                report.append(f"  {i}. {biomarker[:40]:<40} R² = {score:.4f} ({model})")
            report.append("")
        
        return "\n".join(report)

def main():
    """Example usage of the ClimateHealthPipeline."""
    print("ENBEL Climate-Health Analysis Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ClimateHealthPipeline(analysis_mode='improved')
    
    # Load data
    try:
        pipeline.load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in the configured location.")
        return False
    
    # Run analysis
    results = pipeline.run_comprehensive_analysis()
    
    # Print summary
    print("\n" + pipeline.generate_summary_report())
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)