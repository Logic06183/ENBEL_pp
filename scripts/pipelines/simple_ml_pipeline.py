#!/usr/bin/env python3
"""
Simple and Clean ENBEL Climate-Health ML Pipeline
================================================

This is a simplified, easy-to-understand version of the ML pipeline that your team
can easily review, understand, and verify. Every step is clearly documented and 
the logic is straightforward.

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SimpleClimateHealthPipeline:
    """
    A simple, transparent ML pipeline for climate-health analysis.
    Every step is clearly documented and easy to understand.
    """
    
    def __init__(self):
        """Initialize the pipeline with clear settings."""
        print("Initializing Simple Climate-Health Pipeline")
        print("="*60)
        
        # Set reproducible random seed
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
        # Define our target biomarkers (health outcomes we want to predict)
        self.biomarkers = [
            'systolic blood pressure',
            'FASTING GLUCOSE', 
            'CD4 cell count (cells/µL)',
            'FASTING LDL',
            'Hemoglobin (g/dL)'
        ]
        
        # Results storage
        self.results = {}
        
        print(f"Random seed set to: {self.random_seed}")
        print(f"Target biomarkers: {len(self.biomarkers)}")
        print()
    
    def load_data(self):
        """
        Step 1: Load the climate-health dataset.
        This is straightforward data loading with basic validation.
        """
        print("Step 1: Loading Climate-Health Dataset")
        print("-" * 40)
        
        # Load the de-identified dataset (safe for sharing)
        data_file = "DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv"
        
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        self.data = pd.read_csv(data_file, low_memory=False)
        
        print(f"Data loaded: {self.data.shape[0]:,} participants")
        print(f"Total features: {self.data.shape[1]:,}")
        print()
        
        return self.data
    
    def select_climate_features(self):
        """
        Step 2: Select interpretable climate features.
        We focus on features that are easy to understand and explain.
        """
        print("Step 2: Selecting Climate Features")
        print("-" * 40)
        
        # Find climate-related features (temperature, humidity, etc.)
        climate_keywords = ['temp', 'humid', 'heat', 'wind', 'pressure']
        
        climate_features = []
        for column in self.data.columns:
            # Check if column contains climate keywords
            if any(keyword in column.lower() for keyword in climate_keywords):
                # Only include if it's numeric and has reasonable data
                if (self.data[column].dtype in ['float64', 'int64'] and 
                    self.data[column].notna().sum() / len(self.data) > 0.5):
                    climate_features.append(column)
        
        # Add demographic features for control
        demographic_features = []
        for column in self.data.columns:
            if column.lower() in ['sex', 'race', 'age']:
                demographic_features.append(column)
        
        # Combine all features
        self.features = climate_features + demographic_features
        
        print(f"Selected {len(climate_features)} climate features")
        print(f"Added {len(demographic_features)} demographic controls")
        print(f"Total features for analysis: {len(self.features)}")
        print()
        
        return self.features
    
    def prepare_data_for_biomarker(self, biomarker):
        """
        Step 3: Prepare clean data for a specific biomarker.
        This removes missing values and prepares features.
        """
        print(f"Step 3: Preparing Data for {biomarker}")
        print("-" * 40)
        
        # Check if biomarker exists in data
        if biomarker not in self.data.columns:
            print(f"ERROR: Biomarker '{biomarker}' not found in dataset")
            return None, None
        
        # Start with rows that have the target biomarker
        clean_data = self.data.dropna(subset=[biomarker]).copy()
        
        # Get available features (only those that exist in the clean data)
        available_features = [f for f in self.features if f in clean_data.columns]
        
        if len(available_features) < 5:
            print(f"ERROR: Too few features available: {len(available_features)}")
            return None, None
        
        # Prepare feature matrix (X) and target vector (y)
        X = clean_data[available_features].copy()
        y = clean_data[biomarker].copy()
        
        # Handle categorical variables (like Sex, Race)
        for column in X.columns:
            if X[column].dtype == 'object':
                if column.lower() == 'sex':
                    # Convert Male/Female to 1/0
                    X[column] = X[column].map({'Male': 1, 'Female': 0})
                elif len(X[column].unique()) <= 10:
                    # For other categorical variables with few categories
                    unique_values = X[column].dropna().unique()
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    X[column] = X[column].map(mapping)
        
        # Fill missing values with median (for numeric) or 0 (for categorical)
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64']:
                X[column] = X[column].fillna(X[column].median())
            else:
                X[column] = X[column].fillna(0)
        
        # Remove any remaining rows with missing values
        complete_rows = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[complete_rows]
        y_clean = y[complete_rows]
        
        print(f"Clean dataset: {len(X_clean):,} participants")
        print(f"Features: {X_clean.shape[1]}")
        print(f"Target range: {y_clean.min():.1f} to {y_clean.max():.1f}")
        print()
        
        return X_clean, y_clean
    
    def train_models(self, X, y, biomarker):
        """
        Step 4: Train machine learning models.
        We use two standard models: Random Forest and XGBoost.
        """
        print(f"Step 4: Training Models for {biomarker}")
        print("-" * 40)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        
        print(f"Training set: {len(X_train):,} participants")
        print(f"Test set: {len(X_test):,} participants")
        
        results = {}
        
        # Model 1: Random Forest (interpretable tree-based model)
        print("\n🌲 Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,           # Number of trees
            max_depth=10,               # Maximum tree depth
            min_samples_split=20,       # Minimum samples to split
            random_state=self.random_seed,
            n_jobs=-1                   # Use all CPU cores
        )
        
        # Train the model
        rf_model.fit(X_train, y_train)
        
        # Test the model
        rf_predictions = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_predictions)
        rf_mae = mean_absolute_error(y_test, rf_predictions)
        
        # Cross-validation for more robust performance estimate
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
        
        results['random_forest'] = {
            'model': rf_model,
            'test_r2': rf_r2,
            'test_mae': rf_mae,
            'cv_r2_mean': rf_cv_scores.mean(),
            'cv_r2_std': rf_cv_scores.std()
        }
        
        print(f"   Test R²: {rf_r2:.4f}")
        print(f"   Cross-validation R²: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
        
        # Model 2: XGBoost (gradient boosting model)
        try:
            print("\n🚀 Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_seed,
                n_jobs=-1,
                verbosity=0  # Suppress XGBoost output
            )
            
            # Train the model
            xgb_model.fit(X_train, y_train)
            
            # Test the model
            xgb_predictions = xgb_model.predict(X_test)
            xgb_r2 = r2_score(y_test, xgb_predictions)
            xgb_mae = mean_absolute_error(y_test, xgb_predictions)
            
            # Cross-validation
            xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
            
            results['xgboost'] = {
                'model': xgb_model,
                'test_r2': xgb_r2,
                'test_mae': xgb_mae,
                'cv_r2_mean': xgb_cv_scores.mean(),
                'cv_r2_std': xgb_cv_scores.std()
            }
            
            print(f"   Test R²: {xgb_r2:.4f}")
            print(f"   Cross-validation R²: {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")
            
        except Exception as e:
            print(f"   ⚠️ XGBoost training failed: {e}")
            results['xgboost'] = {'error': str(e)}
        
        # Select best model
        if 'xgboost' in results and 'error' not in results['xgboost']:
            if results['xgboost']['cv_r2_mean'] > results['random_forest']['cv_r2_mean']:
                best_model = 'xgboost'
            else:
                best_model = 'random_forest'
        else:
            best_model = 'random_forest'
        
        results['best_model'] = best_model
        results['best_score'] = results[best_model]['cv_r2_mean']
        
        print(f"\nBest model: {best_model} (R² = {results['best_score']:.4f})")
        print()
        
        return results
    
    def analyze_feature_importance(self, model_results, X, biomarker):
        """
        Step 5: Understand which features are most important.
        This helps us interpret what climate factors affect health.
        """
        print(f"Step 5: Feature Importance for {biomarker}")
        print("-" * 40)
        
        best_model_name = model_results['best_model']
        best_model = model_results[best_model_name]['model']
        
        # Get feature importance from the model
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X.columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Show top 10 most important features
            print("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:<40} {row['importance']:.4f}")
            
            print()
            return importance_df
        else:
            print("   ⚠️ Feature importance not available for this model")
            return None
    
    def run_analysis(self):
        """
        Main function: Run the complete analysis pipeline.
        This coordinates all the steps in a clear sequence.
        """
        print("🚀 Starting Climate-Health Analysis Pipeline")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Select features
        self.select_climate_features()
        
        # Step 3-5: Analyze each biomarker
        print("🧬 Analyzing Health Biomarkers")
        print("="*60)
        
        for i, biomarker in enumerate(self.biomarkers, 1):
            print(f"\n[{i}/{len(self.biomarkers)}] ANALYZING: {biomarker}")
            print("="*60)
            
            # Prepare data for this biomarker
            X, y = self.prepare_data_for_biomarker(biomarker)
            
            if X is None or len(X) < 100:
                print(f"SKIPPING {biomarker}: insufficient data")
                continue
            
            # Train models
            model_results = self.train_models(X, y, biomarker)
            
            # Analyze feature importance
            importance_df = self.analyze_feature_importance(model_results, X, biomarker)
            
            # Store results
            self.results[biomarker] = {
                'model_results': model_results,
                'feature_importance': importance_df.to_dict('records') if importance_df is not None else None,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
        
        # Final summary
        self.print_final_summary()
        
        return self.results
    
    def print_final_summary(self):
        """Print a clear summary of all results."""
        print("\n" + "="*60)
        print("📋 FINAL RESULTS SUMMARY")
        print("="*60)
        
        if not self.results:
            print("ERROR: No successful analyses completed")
            return
        
        print(f"{'Biomarker':<35} {'Samples':<10} {'Best R²':<10} {'Model':<15}")
        print("-" * 70)
        
        total_r2 = 0
        count = 0
        
        for biomarker, result in self.results.items():
            n_samples = result['n_samples']
            best_score = result['model_results']['best_score']
            best_model = result['model_results']['best_model']
            
            print(f"{biomarker[:34]:<35} {n_samples:<10} {best_score:<10.4f} {best_model:<15}")
            
            total_r2 += best_score
            count += 1
        
        if count > 0:
            avg_r2 = total_r2 / count
            print("-" * 70)
            print(f"{'AVERAGE PERFORMANCE':<35} {'':<10} {avg_r2:<10.4f}")
        
        print("\n✅ Analysis completed successfully!")
        print(f"   - Biomarkers analyzed: {len(self.results)}")
        print(f"   - Average R² score: {avg_r2:.4f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_analysis_results_{timestamp}.json"
        
        # Prepare results for JSON (remove sklearn objects)
        json_results = {}
        for biomarker, result in self.results.items():
            json_results[biomarker] = {
                'best_model': result['model_results']['best_model'],
                'best_r2': result['model_results']['best_score'],
                'n_samples': result['n_samples'],
                'n_features': result['n_features'],
                'feature_importance': result['feature_importance']
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"   - Results saved to: {results_file}")

def main():
    """
    Run the simple climate-health analysis.
    This is the main entry point that anyone can run.
    """
    print("🌟 Simple Climate-Health ML Pipeline")
    print("====================================")
    print("This pipeline analyzes how climate affects health biomarkers.")
    print("Every step is documented and easy to understand.\n")
    
    try:
        # Create and run the pipeline
        pipeline = SimpleClimateHealthPipeline()
        results = pipeline.run_analysis()
        
        print("\nSUCCESS: Analysis completed!")
        print("You can now review the results and model performance.")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Analysis failed - {e}")
        return False

if __name__ == "__main__":
    # Run the analysis when script is executed directly
    success = main()
    exit(0 if success else 1)