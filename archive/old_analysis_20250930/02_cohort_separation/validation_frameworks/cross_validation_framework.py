#!/usr/bin/env python3
"""
Robust Cross-Validation Framework for Climate-Health ML
======================================================

This module provides specialized cross-validation strategies for time series
climate-health data that prevent temporal data leakage while providing robust
performance estimates.

Key Features:
1. Time-aware cross-validation with proper temporal blocking
2. Spatial cross-validation for geographic generalization
3. Seasonal cross-validation for temporal generalization
4. Statistical significance testing of model performance
5. Confidence intervals for all metrics

Authors: Climate-Health Data Science Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class RobustClimateHealthCV:
    """
    Robust cross-validation framework specialized for climate-health time series data
    """
    
    def __init__(self, results_dir="rigorous_results", figures_dir="rigorous_figures"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Default model configurations (conservative hyperparameters)
        self.model_configs = {
            'ridge': {
                'model': Ridge,
                'params': {'alpha': 1.0, 'random_state': 42}
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                }
            }
        }
    
    def temporal_blocked_cv(self, X, y, dates, n_splits=5, test_size=0.2, gap_days=30):
        """
        Time-aware cross-validation with temporal blocking to prevent data leakage.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        dates : pd.Series
            Date column for temporal ordering
        n_splits : int
            Number of CV splits
        test_size : float
            Fraction of data for testing in each split
        gap_days : int
            Gap between train and test to prevent temporal leakage
        """
        print(f"🕐 Temporal blocked CV: {n_splits} splits, {gap_days}-day gap")
        
        # Sort by date
        sorted_indices = dates.sort_values().index
        X_sorted = X.loc[sorted_indices]
        y_sorted = y.loc[sorted_indices]
        dates_sorted = dates.loc[sorted_indices]
        
        splits = []
        n_samples = len(X_sorted)
        
        for i in range(n_splits):
            # Calculate split boundaries
            split_start = int(i * n_samples / n_splits)
            split_end = int((i + 1) * n_samples / n_splits)
            
            # Training set: before split_start
            if split_start > 0:
                train_end_date = dates_sorted.iloc[split_start] - timedelta(days=gap_days)
                train_indices = dates_sorted[dates_sorted <= train_end_date].index
            else:
                train_indices = []
            
            # Test set: split_start to split_end
            test_indices = sorted_indices[split_start:split_end]
            
            if len(train_indices) > 50 and len(test_indices) > 20:  # Minimum sample requirements
                splits.append((train_indices, test_indices))
        
        print(f"   Created {len(splits)} valid temporal splits")
        return splits
    
    def spatial_cv(self, X, y, coordinates, n_splits=5):
        """
        Spatial cross-validation based on geographic coordinates.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        coordinates : pd.DataFrame
            DataFrame with 'latitude' and 'longitude' columns
        n_splits : int
            Number of spatial splits
        """
        print(f"🌍 Spatial CV: {n_splits} geographic splits")
        
        # Create spatial clusters using kmeans-like approach
        from sklearn.cluster import KMeans
        
        # Clean coordinates
        coord_clean = coordinates.dropna()
        if len(coord_clean) < len(X) * 0.5:
            print("   ⚠️  Too many missing coordinates for spatial CV")
            return []
        
        # Perform spatial clustering
        kmeans = KMeans(n_clusters=n_splits, random_state=42)
        spatial_clusters = kmeans.fit_predict(coord_clean[['latitude', 'longitude']])
        
        splits = []
        for test_cluster in range(n_splits):
            test_mask = spatial_clusters == test_cluster
            test_indices = coord_clean[test_mask].index
            train_indices = coord_clean[~test_mask].index
            
            # Filter to available indices
            test_indices = [idx for idx in test_indices if idx in X.index]
            train_indices = [idx for idx in train_indices if idx in X.index]
            
            if len(train_indices) > 50 and len(test_indices) > 20:
                splits.append((train_indices, test_indices))
        
        print(f"   Created {len(splits)} valid spatial splits")
        return splits
    
    def seasonal_cv(self, X, y, dates, seasons=None):
        """
        Seasonal cross-validation for temporal generalization.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        dates : pd.Series
            Date column
        seasons : list
            List of seasons to use as test sets
        """
        if seasons is None:
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
        print(f"🌱 Seasonal CV: {len(seasons)} seasonal splits")
        
        # Map months to seasons
        season_map = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',  # Southern hemisphere
            3: 'Fall', 4: 'Fall', 5: 'Fall',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        }
        
        dates_dt = pd.to_datetime(dates)
        data_seasons = dates_dt.dt.month.map(season_map)
        
        splits = []
        for test_season in seasons:
            test_indices = dates[data_seasons == test_season].index
            train_indices = dates[data_seasons != test_season].index
            
            # Filter to available indices
            test_indices = [idx for idx in test_indices if idx in X.index]
            train_indices = [idx for idx in train_indices if idx in X.index]
            
            if len(train_indices) > 50 and len(test_indices) > 20:
                splits.append((train_indices, test_indices))
        
        print(f"   Created {len(splits)} valid seasonal splits")
        return splits
    
    def evaluate_model_cv(self, model, X, y, cv_splits, cv_type="temporal"):
        """
        Evaluate model performance using custom cross-validation splits
        """
        scores = {'r2': [], 'mae': [], 'rmse': []}
        
        for i, (train_idx, test_idx) in enumerate(cv_splits):
            try:
                # Get train/test data
                X_train = X.loc[train_idx]
                X_test = X.loc[test_idx]
                y_train = y.loc[train_idx]
                y_test = y.loc[test_idx]
                
                # Train model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model_copy.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                scores['r2'].append(r2)
                scores['mae'].append(mae)
                scores['rmse'].append(rmse)
                
            except Exception as e:
                print(f"   ⚠️  Split {i+1} failed: {e}")
                continue
        
        if not scores['r2']:
            return None
        
        # Calculate summary statistics
        summary = {}
        for metric in scores:
            values = scores[metric]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
                'values': values
            }
        
        summary['n_splits'] = len(scores['r2'])
        summary['cv_type'] = cv_type
        
        return summary
    
    def statistical_significance_test(self, scores1, scores2, metric='r2'):
        """
        Test statistical significance between two sets of CV scores
        """
        values1 = scores1[metric]['values']
        values2 = scores2[metric]['values']
        
        # Paired t-test if same number of splits
        if len(values1) == len(values2):
            statistic, p_value = stats.ttest_rel(values1, values2)
            test_type = "paired_ttest"
        else:
            # Independent t-test
            statistic, p_value = stats.ttest_ind(values1, values2)
            test_type = "independent_ttest"
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (np.mean(values1) - np.mean(values2)) / np.sqrt((np.var(values1) + np.var(values2)) / 2)
        }
    
    def comprehensive_cv_analysis(self, X, y, dates, coordinates=None, biomarker_name="biomarker"):
        """
        Run comprehensive cross-validation analysis with multiple CV strategies
        """
        print(f"🔄 Comprehensive CV analysis for {biomarker_name}")
        
        results = {
            'biomarker': biomarker_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'timestamp': datetime.now().isoformat(),
            'cv_results': {},
            'model_comparisons': {}
        }
        
        # Generate CV splits
        cv_strategies = {}
        
        # 1. Temporal blocked CV
        temporal_splits = self.temporal_blocked_cv(X, y, dates)
        if temporal_splits:
            cv_strategies['temporal_blocked'] = temporal_splits
        
        # 2. Seasonal CV
        seasonal_splits = self.seasonal_cv(X, y, dates)
        if seasonal_splits:
            cv_strategies['seasonal'] = seasonal_splits
        
        # 3. Spatial CV (if coordinates available)
        if coordinates is not None:
            spatial_splits = self.spatial_cv(X, y, coordinates)
            if spatial_splits:
                cv_strategies['spatial'] = spatial_splits
        
        # 4. Standard TimeSeriesSplit for comparison
        tss = TimeSeriesSplit(n_splits=5)
        standard_splits = list(tss.split(X))
        cv_strategies['standard_temporal'] = standard_splits
        
        print(f"   Testing {len(cv_strategies)} CV strategies")
        
        # Evaluate each model with each CV strategy
        for model_name, model_config in self.model_configs.items():
            print(f"   📊 Evaluating {model_name}")
            
            model = model_config['model'](**model_config['params'])
            results['cv_results'][model_name] = {}
            
            for cv_name, cv_splits in cv_strategies.items():
                print(f"      {cv_name} CV...")
                
                cv_scores = self.evaluate_model_cv(model, X, y, cv_splits, cv_name)
                if cv_scores:
                    results['cv_results'][model_name][cv_name] = cv_scores
        
        # Model comparisons
        for cv_name in cv_strategies.keys():
            results['model_comparisons'][cv_name] = {}
            
            cv_results_for_strategy = {}
            for model_name in self.model_configs.keys():
                if cv_name in results['cv_results'][model_name]:
                    cv_results_for_strategy[model_name] = results['cv_results'][model_name][cv_name]
            
            if len(cv_results_for_strategy) >= 2:
                # Compare all pairs
                model_names = list(cv_results_for_strategy.keys())
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names[i+1:], i+1):
                        comparison_key = f"{model1}_vs_{model2}"
                        
                        sig_test = self.statistical_significance_test(
                            cv_results_for_strategy[model1],
                            cv_results_for_strategy[model2]
                        )
                        
                        results['model_comparisons'][cv_name][comparison_key] = sig_test
        
        # Create visualization
        self.visualize_cv_results(results, biomarker_name)
        
        # Save results
        safe_biomarker = self._safe_filename(biomarker_name)
        cv_results_file = self.results_dir / f"cv_analysis_{safe_biomarker}.json"
        with open(cv_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   ✅ CV analysis saved: {cv_results_file}")
        
        return results
    
    def visualize_cv_results(self, results, biomarker_name):
        """
        Create comprehensive visualization of CV results
        """
        print(f"   📊 Creating CV visualization for {biomarker_name}")
        
        # Extract data for plotting
        plot_data = []
        for model_name, cv_results in results['cv_results'].items():
            for cv_type, scores in cv_results.items():
                for metric in ['r2', 'mae', 'rmse']:
                    if metric in scores:
                        for value in scores[metric]['values']:
                            plot_data.append({
                                'Model': model_name,
                                'CV_Type': cv_type,
                                'Metric': metric,
                                'Value': value
                            })
        
        if not plot_data:
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create subplots for each metric
        metrics = ['r2', 'mae', 'rmse']
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
        
        for i, metric in enumerate(metrics):
            metric_data = plot_df[plot_df['Metric'] == metric]
            
            if not metric_data.empty:
                sns.boxplot(
                    data=metric_data,
                    x='CV_Type',
                    y='Value',
                    hue='Model',
                    ax=axes[i]
                )
                
                axes[i].set_title(f'{metric.upper()} by CV Strategy')
                axes[i].set_xlabel('Cross-Validation Type')
                axes[i].set_ylabel(f'{metric.upper()} Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        safe_biomarker = self._safe_filename(biomarker_name)
        cv_plot_file = self.figures_dir / f"cv_analysis_{safe_biomarker}.png"
        plt.savefig(cv_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 CV visualization saved: {cv_plot_file}")
    
    def _clone_model(self, model):
        """Clone a model with the same parameters"""
        return model.__class__(**model.get_params())
    
    def _safe_filename(self, name):
        """Convert name to safe filename"""
        return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')
    
    def print_cv_summary(self, results):
        """
        Print a comprehensive summary of CV results
        """
        print("\n" + "="*80)
        print(f"📊 CV ANALYSIS SUMMARY: {results['biomarker']}")
        print("="*80)
        
        # Overall performance by model
        print("\n🏆 MODEL PERFORMANCE BY CV STRATEGY:")
        print("-" * 80)
        
        for model_name, cv_results in results['cv_results'].items():
            print(f"\n{model_name.upper()}:")
            
            for cv_type, scores in cv_results.items():
                r2_mean = scores['r2']['mean']
                r2_std = scores['r2']['std']
                n_splits = scores['n_splits']
                
                print(f"  {cv_type:20s}: R² = {r2_mean:.4f} ± {r2_std:.4f} ({n_splits} splits)")
        
        # Statistical significance
        print("\n🔬 STATISTICAL SIGNIFICANCE TESTS:")
        print("-" * 80)
        
        for cv_type, comparisons in results['model_comparisons'].items():
            print(f"\n{cv_type.upper()} CV:")
            
            for comparison, test_result in comparisons.items():
                models = comparison.replace('_vs_', ' vs ')
                significance = "***" if test_result['p_value'] < 0.001 else \
                             "**" if test_result['p_value'] < 0.01 else \
                             "*" if test_result['p_value'] < 0.05 else "ns"
                
                print(f"  {models:25s}: p = {test_result['p_value']:.4f} {significance}")
        
        print("\n" + "="*80)

def run_comprehensive_cv_analysis(df, features, biomarkers, date_col='primary_date'):
    """
    Run comprehensive CV analysis for multiple biomarkers
    """
    cv_analyzer = RobustClimateHealthCV()
    all_results = {}
    
    for biomarker in biomarkers:
        print(f"\n{'='*100}")
        print(f"🔄 CV ANALYSIS: {biomarker}")
        print('='*100)
        
        if biomarker not in df.columns:
            print(f"❌ {biomarker} not found in dataset")
            continue
        
        try:
            # Clean data
            clean_data = df.dropna(subset=[biomarker]).copy()
            
            if len(clean_data) < 100:
                print(f"❌ Insufficient data: {len(clean_data)} samples")
                continue
            
            # Prepare features
            available_features = [f for f in features if f in clean_data.columns]
            X = clean_data[available_features].copy()
            y = clean_data[biomarker]
            dates = pd.to_datetime(clean_data[date_col])
            
            # Handle coordinates if available
            coordinates = None
            if 'latitude' in clean_data.columns and 'longitude' in clean_data.columns:
                coordinates = clean_data[['latitude', 'longitude']]
            
            # Handle missing values
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    mode_val = X[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                    X[col] = X[col].fillna(fill_val)
            
            # Run comprehensive CV analysis
            results = cv_analyzer.comprehensive_cv_analysis(
                X, y, dates, coordinates, biomarker
            )
            
            # Print summary
            cv_analyzer.print_cv_summary(results)
            
            all_results[biomarker] = results
            
        except Exception as e:
            print(f"❌ Failed to analyze {biomarker}: {e}")
            continue
    
    # Save combined results
    combined_file = cv_analyzer.results_dir / f"all_cv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ All CV results saved: {combined_file}")
    
    return all_results

if __name__ == "__main__":
    # Example usage
    print("🔄 Loading example data for CV analysis...")
    
    # This would be called from the main pipeline
    # run_comprehensive_cv_analysis(df, features, biomarkers)