#!/usr/bin/env python3
"""
Improved Climate-Health ML Pipeline for ENBEL
=============================================

Addresses key issues from previous analysis:
1. Removes race confounding by excluding demographic predictors
2. Focuses purely on climate-health relationships
3. Implements hyperparameter optimization
4. Better feature engineering for climate variables
5. Proper cross-validation and model evaluation
6. Targets systolic blood pressure specifically

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime
import shap

warnings.filterwarnings('ignore')

class ImprovedClimateHealthPipeline:
    """
    Improved climate-health ML pipeline focusing on pure climate effects
    without demographic confounding.
    """
    
    def __init__(self, target_biomarker='systolic blood pressure'):
        """Initialize the improved pipeline"""
        print("Initializing Improved Climate-Health Pipeline")
        print("="*60)
        
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
        self.target_biomarker = target_biomarker
        self.results = {}
        
        print(f"Target biomarker: {target_biomarker}")
        print(f"Focus: Pure climate effects (no demographic confounding)")
        print()
    
    def load_and_clean_data(self):
        """Load and clean the climate-health dataset"""
        print("Step 1: Loading and Cleaning Data")
        print("-" * 40)
        
        # Load data
        self.data = pd.read_csv("DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv", low_memory=False)
        
        # Focus on clinical participants only
        clinical_data = self.data[self.data['data_source'].notna()].copy()
        
        print(f"Total dataset: {len(self.data):,} participants")
        print(f"Clinical participants: {len(clinical_data):,}")
        
        # Focus on participants with target biomarker
        clean_data = clinical_data.dropna(subset=[self.target_biomarker]).copy()
        
        print(f"Participants with {self.target_biomarker}: {len(clean_data):,}")
        
        if len(clean_data) < 1000:
            raise ValueError(f"Insufficient data for {self.target_biomarker}: {len(clean_data)} participants")
        
        self.clean_data = clean_data
        return clean_data
    
    def engineer_climate_features(self):
        """Engineer sophisticated climate features"""
        print("\nStep 2: Climate Feature Engineering")
        print("-" * 40)
        
        # Identify pure climate features (NO demographics)
        climate_keywords = [
            'temp', 'humid', 'heat', 'wind', 'pressure', 'apparent',
            'degree_day', 'cooling', 'heating', 'precipitation', 'solar'
        ]
        
        # Exclude demographic variables explicitly
        exclude_keywords = ['sex', 'race', 'age', 'gender', 'ethnicity']
        
        climate_features = []
        for column in self.clean_data.columns:
            # Include if contains climate keywords and not demographic
            if (any(keyword in column.lower() for keyword in climate_keywords) and
                not any(exclude in column.lower() for exclude in exclude_keywords)):
                
                # Only include numeric columns with sufficient data
                if (self.clean_data[column].dtype in ['float64', 'int64'] and 
                    self.clean_data[column].notna().sum() / len(self.clean_data) > 0.7):
                    climate_features.append(column)
        
        print(f"Pure climate features selected: {len(climate_features)}")
        
        # Create additional engineered features
        engineered_features = []
        base_temp_cols = [col for col in climate_features if 'temperature' in col.lower() and 'lag' not in col.lower()]
        
        if len(base_temp_cols) >= 2:
            # Temperature range and variability
            for i, temp_col in enumerate(base_temp_cols[:3]):  # Limit to avoid too many features
                if temp_col in self.clean_data.columns:
                    # Daily temperature range
                    range_col = f"{temp_col}_daily_range"
                    if 'max' in temp_col.lower():
                        min_col = temp_col.replace('max', 'min').replace('MAX', 'MIN')
                        if min_col in self.clean_data.columns:
                            self.clean_data[range_col] = self.clean_data[temp_col] - self.clean_data[min_col]
                            engineered_features.append(range_col)
                    
                    # Temperature acceleration (rate of change)
                    accel_col = f"{temp_col}_acceleration"
                    if temp_col in self.clean_data.columns:
                        self.clean_data[accel_col] = self.clean_data[temp_col].diff().fillna(0)
                        engineered_features.append(accel_col)
        
        all_features = climate_features + engineered_features
        
        print(f"Additional engineered features: {len(engineered_features)}")
        print(f"Total climate features: {len(all_features)}")
        
        self.climate_features = all_features
        return all_features
    
    def prepare_modeling_data(self):
        """Prepare clean data for modeling"""
        print("\nStep 3: Preparing Modeling Data")
        print("-" * 40)
        
        # Get available features
        available_features = [f for f in self.climate_features if f in self.clean_data.columns]
        
        # Prepare feature matrix and target
        X = self.clean_data[available_features].copy()
        y = self.clean_data[self.target_biomarker].copy()
        
        # Handle missing values
        print(f"Handling missing values...")
        X = X.fillna(X.median())
        
        # Remove features with zero variance
        feature_vars = X.var()
        non_zero_var_features = feature_vars[feature_vars > 1e-6].index.tolist()
        X = X[non_zero_var_features]
        
        # Remove highly correlated features (>0.95)
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X = X.drop(columns=high_corr_features)
        
        print(f"Features after correlation filtering: {X.shape[1]}")
        print(f"Final dataset: {len(X):,} participants, {X.shape[1]} features")
        print(f"Target range: {y.min():.1f} to {y.max():.1f}")
        
        self.X = X
        self.y = y
        return X, y
    
    def optimize_models(self):
        """Optimize models with hyperparameter tuning"""
        print("\nStep 4: Model Optimization")
        print("-" * 40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_seed
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # 1. Optimized Random Forest
        print("🌲 Optimizing Random Forest...")
        rf_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_base = RandomForestRegressor(random_state=self.random_seed, n_jobs=-1)
        rf_grid = GridSearchCV(
            rf_base, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        rf_grid.fit(X_train, y_train)
        
        rf_predictions = rf_grid.predict(X_test)
        rf_r2 = r2_score(y_test, rf_predictions)
        rf_mae = mean_absolute_error(y_test, rf_predictions)
        rf_cv_scores = cross_val_score(rf_grid.best_estimator_, X_train, y_train, cv=5, scoring='r2')
        
        results['random_forest'] = {
            'model': rf_grid.best_estimator_,
            'best_params': rf_grid.best_params_,
            'test_r2': rf_r2,
            'test_mae': rf_mae,
            'cv_r2_mean': rf_cv_scores.mean(),
            'cv_r2_std': rf_cv_scores.std(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"   Best R²: {rf_r2:.4f}")
        print(f"   CV R²: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
        print(f"   Best params: {rf_grid.best_params_}")
        
        # 2. Optimized XGBoost
        print("\n🚀 Optimizing XGBoost...")
        xgb_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_base = xgb.XGBRegressor(random_state=self.random_seed, n_jobs=-1)
        xgb_grid = GridSearchCV(
            xgb_base, xgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        xgb_grid.fit(X_train, y_train)
        
        xgb_predictions = xgb_grid.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_predictions)
        xgb_mae = mean_absolute_error(y_test, xgb_predictions)
        xgb_cv_scores = cross_val_score(xgb_grid.best_estimator_, X_train, y_train, cv=5, scoring='r2')
        
        results['xgboost'] = {
            'model': xgb_grid.best_estimator_, 
            'best_params': xgb_grid.best_params_,
            'test_r2': xgb_r2,
            'test_mae': xgb_mae,
            'cv_r2_mean': xgb_cv_scores.mean(),
            'cv_r2_std': xgb_cv_scores.std(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"   Best R²: {xgb_r2:.4f}")
        print(f"   CV R²: {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")
        print(f"   Best params: {xgb_grid.best_params_}")
        
        # Select best model
        if results['xgboost']['cv_r2_mean'] > results['random_forest']['cv_r2_mean']:
            best_model_name = 'xgboost'
        else:
            best_model_name = 'random_forest'
        
        results['best_model_name'] = best_model_name
        results['best_model'] = results[best_model_name]['model']
        results['best_score'] = results[best_model_name]['cv_r2_mean']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   CV R²: {results['best_score']:.4f} ± {results[best_model_name]['cv_r2_std']:.4f}")
        
        self.model_results = results
        return results
    
    def analyze_feature_importance(self):
        """Analyze feature importance for the best model"""
        print("\nStep 5: Feature Importance Analysis")
        print("-" * 40)
        
        best_model = self.model_results['best_model']
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = self.X.columns
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Climate Features:")
            print("-" * 60)
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<35} {row['importance']:.4f}")
            
            self.feature_importance = importance_df
            return importance_df
        else:
            print("⚠️ Feature importance not available for this model")
            return None
    
    def generate_shap_analysis(self, max_samples=500):
        """Generate SHAP analysis for the best model"""
        print("\nStep 6: SHAP Analysis")
        print("-" * 40)
        
        best_model = self.model_results['best_model']
        X_test = self.model_results[self.model_results['best_model_name']]['X_test']
        
        # Sample for SHAP (computationally intensive)
        sample_size = min(max_samples, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        print(f"Calculating SHAP values for {sample_size} participants...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(0)
        shap_feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        print("Top 10 Features by SHAP Importance:")
        print("-" * 50)
        for i, (_, row) in enumerate(shap_feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<35} {row['shap_importance']:.4f}")
        
        self.shap_values = shap_values
        self.shap_sample = X_sample
        self.explainer = explainer
        self.shap_feature_importance = shap_feature_importance
        
        return shap_values, X_sample, explainer
    
    def create_shap_visualizations(self):
        """Create publication-quality SHAP visualizations"""
        print("\nStep 7: Creating SHAP Visualizations")
        print("-" * 40)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'serif'
        
        model_name = self.model_results['best_model_name']
        r2_score = self.model_results['best_score']
        
        # 1. SHAP Summary Plot (Beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.shap_sample, max_display=15, show=False)
        plt.title(f'SHAP Summary Plot: {self.target_biomarker}\n' + 
                 f'{model_name.title()} Model (R² = {r2_score:.3f})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('improved_shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig('improved_shap_summary_plot.svg', bbox_inches='tight')
        plt.close()
        print("✓ SHAP summary plot saved")
        
        # 2. SHAP Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.shap_sample, plot_type="bar", max_display=15, show=False)
        plt.title(f'SHAP Feature Importance: {self.target_biomarker}\n' + 
                 f'Mean Absolute SHAP Values ({model_name.title()})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('improved_shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig('improved_shap_bar_plot.svg', bbox_inches='tight')
        plt.close()
        print("✓ SHAP bar plot saved")
        
        # 3. SHAP Dependency Plot for top feature
        top_feature_idx = np.argmax(np.abs(self.shap_values).mean(0))
        top_feature_name = self.shap_sample.columns[top_feature_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature_idx, self.shap_values, self.shap_sample, show=False)
        plt.title(f'SHAP Dependency Plot: {top_feature_name}\n' + 
                 f'Most Important Climate Feature for {self.target_biomarker}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('improved_shap_dependency_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig('improved_shap_dependency_plot.svg', bbox_inches='tight')
        plt.close()
        print("✓ SHAP dependency plot saved")
        
        # 4. SHAP Waterfall Plot
        plt.figure(figsize=(10, 8))
        try:
            shap.plots.waterfall(shap.Explanation(
                values=self.shap_values[0], 
                base_values=self.explainer.expected_value,
                data=self.shap_sample.iloc[0]
            ), max_display=10, show=False)
            plt.title(f'SHAP Waterfall Plot: Individual Prediction\n' + 
                     f'{self.target_biomarker} - Single Participant Example', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('improved_shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
            plt.savefig('improved_shap_waterfall_plot.svg', bbox_inches='tight')
            plt.close()
            print("✓ SHAP waterfall plot saved")
        except Exception as e:
            print(f"⚠️ Waterfall plot failed: {e}")
        
        print(f"\n🎯 Pure climate-focused SHAP analysis complete!")
        print(f"   Model performance: R² = {r2_score:.4f}")
        print(f"   No demographic confounding (race/sex excluded)")
    
    def run_complete_analysis(self):
        """Run the complete improved analysis pipeline"""
        print("🚀 Starting Improved Climate-Health Analysis")
        print("="*60)
        
        # Load and clean data
        self.load_and_clean_data()
        
        # Engineer climate features
        self.engineer_climate_features()
        
        # Prepare modeling data
        self.prepare_modeling_data()
        
        # Optimize models
        self.optimize_models()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Generate SHAP analysis
        self.generate_shap_analysis()
        
        # Create visualizations
        self.create_shap_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*60)
        print("✅ IMPROVED ANALYSIS COMPLETE")
        print("="*60)
        print(f"Target: {self.target_biomarker}")
        print(f"Model: {self.model_results['best_model_name']}")
        print(f"Performance: R² = {self.model_results['best_score']:.4f}")
        print(f"Features: {len(self.X.columns)} pure climate variables")
        print("No demographic confounding!")
        
        return self.model_results
    
    def save_results(self):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_summary = {
            'target_biomarker': self.target_biomarker,
            'sample_size': len(self.clean_data),
            'n_features': len(self.X.columns),
            'best_model': self.model_results['best_model_name'],
            'best_r2': self.model_results['best_score'],
            'best_r2_std': self.model_results[self.model_results['best_model_name']]['cv_r2_std'],
            'feature_importance': self.feature_importance.head(20).to_dict('records') if hasattr(self, 'feature_importance') else None,
            'shap_importance': self.shap_feature_importance.head(20).to_dict('records') if hasattr(self, 'shap_feature_importance') else None,
            'timestamp': timestamp
        }
        
        with open(f'improved_analysis_results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Results saved to: improved_analysis_results_{timestamp}.json")

def main():
    """Main function"""
    # Focus on systolic blood pressure
    pipeline = ImprovedClimateHealthPipeline(target_biomarker='systolic blood pressure')
    results = pipeline.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()