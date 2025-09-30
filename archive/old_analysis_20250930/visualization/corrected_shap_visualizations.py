#!/usr/bin/env python3
"""
Corrected SHAP Visualizations for ENBEL Climate-Health Analysis
===============================================================

This script creates SHAP visualizations using REAL model outputs instead
of simulated data, addressing the critical issue identified in the quality review.

FIXES IMPLEMENTED:
- ✅ Uses actual trained models and real SHAP values
- ✅ Proper SHAP explainer initialization
- ✅ Accurate feature importance ranking
- ✅ Real biomarker predictions and explanations
- ✅ Publication-ready visualization styling

Author: ENBEL Project Team
Version: 2.0 (Production Ready)
"""

import sys
from pathlib import Path
import warnings

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import custom utilities
from config import ENBELConfig, set_reproducible_environment
from data_validation import validate_data_files
from ml_utils import prepare_features_safely, calculate_shap_values

# Configure matplotlib for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class CorrectedSHAPVisualizer:
    """
    Production-ready SHAP visualization class using real model outputs.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize visualizer with configuration."""
        set_reproducible_environment(42)
        self.config = ENBELConfig(config_file)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("🎨 Corrected SHAP Visualizer initialized")
    
    def load_trained_models(self, results_dir: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Load trained models and their metadata.
        
        Parameters:
        -----------
        results_dir : Path, optional
            Directory containing trained models
            
        Returns:
        --------
        Dict[str, Dict]
            Dictionary of loaded models and metadata
        """
        if results_dir is None:
            results_dir = self.config.config['paths']['models_dir']
        
        logger.info(f"📂 Loading trained models from: {results_dir}")
        
        models = {}
        
        # Find all model files
        model_files = list(results_dir.glob("*_model_*.joblib"))
        feature_files = list(results_dir.glob("features_*.json"))
        
        logger.info(f"Found {len(model_files)} model files and {len(feature_files)} feature files")
        
        for model_file in model_files:
            try:
                # Extract biomarker name from filename
                # Format: rf_model_biomarker_timestamp.joblib
                parts = model_file.stem.split('_')
                model_type = parts[0]  # rf or xgb
                biomarker_parts = parts[2:-1]  # biomarker name parts
                biomarker = '_'.join(biomarker_parts)
                
                # Find corresponding feature file
                feature_file = None
                for f_file in feature_files:
                    if biomarker in f_file.stem:
                        feature_file = f_file
                        break
                
                if feature_file is None:
                    logger.warning(f"⚠️  No feature file found for {biomarker}")
                    continue
                
                # Load model and features
                model = joblib.load(model_file)
                with open(feature_file, 'r') as f:
                    feature_info = json.load(f)
                
                if biomarker not in models:
                    models[biomarker] = {}
                
                models[biomarker][model_type] = {
                    'model': model,
                    'feature_names': feature_info['feature_names'],
                    'model_file': model_file,
                    'feature_file': feature_file
                }
                
                logger.info(f"✅ Loaded {model_type} model for {biomarker}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {model_file}: {e}")
        
        logger.info(f"🎯 Successfully loaded models for {len(models)} biomarkers")
        return models
    
    def prepare_shap_data(self, biomarker: str, 
                         models: Dict[str, Dict],
                         sample_size: int = 200) -> Tuple[Optional[Any], Optional[pd.DataFrame], Optional[np.ndarray]]:
        """
        Prepare data for SHAP analysis using real trained models.
        
        Parameters:
        -----------
        biomarker : str
            Target biomarker name
        models : Dict[str, Dict]
            Loaded models dictionary
        sample_size : int
            Sample size for SHAP calculation
            
        Returns:
        --------
        Tuple[model, DataFrame, ndarray]
            Trained model, feature data, and SHAP values
        """
        if biomarker not in models:
            logger.error(f"❌ No model found for biomarker: {biomarker}")
            return None, None, None
        
        try:
            # Load data
            data_files = validate_data_files()
            df = pd.read_csv(data_files['full_dataset'], low_memory=False)
            
            # Choose best model (prefer RF if available)
            if 'rf' in models[biomarker]:
                model_info = models[biomarker]['rf']
                model_type = 'Random Forest'
            elif 'xgb' in models[biomarker]:
                model_info = models[biomarker]['xgb']
                model_type = 'XGBoost'
            else:
                logger.error(f"❌ No valid model found for {biomarker}")
                return None, None, None
            
            model = model_info['model']
            feature_names = model_info['feature_names']
            
            logger.info(f"🧬 Preparing SHAP data for {biomarker} using {model_type}")
            
            # Prepare features using the same features as the trained model
            X_train, X_test, y_train, y_test = prepare_features_safely(
                df, feature_names, biomarker, test_size=0.2, random_state=42
            )
            
            # Use test set for SHAP (to avoid overfitting in explanations)
            if len(X_test) > sample_size:
                X_shap = X_test.sample(n=sample_size, random_state=42)
            else:
                X_shap = X_test
            
            logger.info(f"📊 SHAP data prepared: {len(X_shap)} samples, {len(feature_names)} features")
            
            # Calculate SHAP values
            logger.info("🔍 Calculating SHAP values...")
            shap_values = calculate_shap_values(model, X_shap, sample_size=len(X_shap))
            
            if shap_values is None:
                logger.error(f"❌ SHAP calculation failed for {biomarker}")
                return None, None, None
            
            logger.info(f"✅ SHAP values calculated: {shap_values.shape}")
            
            return model, X_shap, shap_values
            
        except Exception as e:
            logger.error(f"❌ SHAP data preparation failed for {biomarker}: {e}")
            return None, None, None
    
    def create_shap_beeswarm_plot(self, biomarker: str, 
                                 X_data: pd.DataFrame,
                                 shap_values: np.ndarray,
                                 max_features: int = 15) -> plt.Figure:
        """
        Create SHAP beeswarm plot with real data.
        
        Parameters:
        -----------
        biomarker : str
            Target biomarker name
        X_data : pd.DataFrame
            Feature data
        shap_values : np.ndarray
            Real SHAP values
        max_features : int
            Maximum number of features to display
            
        Returns:
        --------
        plt.Figure
            Created figure
        """
        logger.info(f"🐝 Creating SHAP beeswarm plot for {biomarker}")
        
        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        importance_order = np.argsort(feature_importance)[::-1][:max_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Plot beeswarm
        for i, feature_idx in enumerate(importance_order):
            feature_name = X_data.columns[feature_idx]
            feature_shap = shap_values[:, feature_idx]
            feature_values = X_data.iloc[:, feature_idx].values
            
            # Normalize feature values for color mapping
            if feature_values.std() > 0:
                feature_values_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
            else:
                feature_values_norm = np.zeros_like(feature_values)
            
            # Add some jitter to y-coordinate
            y_pos = i + np.random.normal(0, 0.1, len(feature_shap))
            
            # Create scatter plot with color mapping
            scatter = ax.scatter(feature_shap, y_pos, c=feature_values_norm, 
                               cmap='coolwarm', alpha=0.6, s=30, edgecolors='none')
        
        # Customize plot
        ax.set_yticks(range(len(importance_order)))
        ax.set_yticklabels([X_data.columns[i] for i in importance_order], fontsize=10)
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Beeswarm Plot: {biomarker}\n(Real Model Explanations)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Feature Value\n(Low → High)', rotation=270, labelpad=20, fontsize=10)
        
        # Style improvements
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add annotation
        ax.text(0.02, 0.98, f'n = {len(X_data)} samples', 
               transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        
        logger.info(f"✅ SHAP beeswarm plot created for {biomarker}")
        return fig
    
    def create_shap_waterfall_plot(self, biomarker: str,
                                  X_data: pd.DataFrame, 
                                  shap_values: np.ndarray,
                                  model,
                                  sample_idx: int = 0) -> plt.Figure:
        """
        Create SHAP waterfall plot for a specific sample.
        
        Parameters:
        -----------
        biomarker : str
            Target biomarker name
        X_data : pd.DataFrame
            Feature data
        shap_values : np.ndarray
            Real SHAP values
        model
            Trained model
        sample_idx : int
            Index of sample to explain
            
        Returns:
        --------
        plt.Figure
            Created figure
        """
        logger.info(f"💧 Creating SHAP waterfall plot for {biomarker}")
        
        if sample_idx >= len(X_data):
            sample_idx = 0
        
        # Get data for specific sample
        sample_shap = shap_values[sample_idx]
        sample_features = X_data.iloc[sample_idx]
        
        # Calculate base value (mean prediction)
        all_predictions = model.predict(X_data)
        base_value = np.mean(all_predictions)
        
        # Get prediction for this sample
        sample_prediction = model.predict(sample_features.values.reshape(1, -1))[0]
        
        # Sort features by absolute SHAP value
        importance_order = np.argsort(np.abs(sample_shap))[::-1][:10]  # Top 10 features
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Calculate waterfall positions
        running_total = base_value
        y_positions = []
        x_positions = []
        widths = []
        colors = []
        feature_labels = []
        
        # Add base value
        y_positions.append(0)
        x_positions.append(base_value - 0.01)
        widths.append(0.02)
        colors.append('gray')
        feature_labels.append('Base')
        
        # Add each feature contribution
        for i, feature_idx in enumerate(importance_order):
            feature_name = X_data.columns[feature_idx]
            shap_val = sample_shap[feature_idx]
            feature_value = sample_features.iloc[feature_idx]
            
            y_positions.append(i + 1)
            
            if shap_val >= 0:
                x_positions.append(running_total)
                colors.append('#ff4444')  # Red for positive
            else:
                x_positions.append(running_total + shap_val)
                colors.append('#4444ff')  # Blue for negative
            
            widths.append(abs(shap_val))
            running_total += shap_val
            
            # Create feature label with value
            if isinstance(feature_value, (int, float)):
                if abs(feature_value) >= 1000:
                    value_str = f"{feature_value:.0f}"
                elif abs(feature_value) >= 10:
                    value_str = f"{feature_value:.1f}"
                else:
                    value_str = f"{feature_value:.2f}"
            else:
                value_str = str(feature_value)
            
            feature_labels.append(f"{feature_name}\n= {value_str}")
        
        # Create horizontal bar chart
        bars = ax.barh(y_positions, widths, left=x_positions, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, shap_val) in enumerate(zip(bars[1:], [sample_shap[j] for j in importance_order])):
            if abs(shap_val) > max(widths) * 0.1:  # Only label significant contributions
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                       f'{shap_val:+.2f}', ha='center', va='center', 
                       fontweight='bold', fontsize=9, color='white')
        
        # Add final prediction
        y_positions.append(len(importance_order) + 1)
        ax.axvline(x=sample_prediction, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(sample_prediction, len(importance_order) + 1, 
               f'Prediction\n{sample_prediction:.2f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Customize plot
        ax.set_yticks(y_positions[:-1])
        ax.set_yticklabels(feature_labels, fontsize=9)
        ax.set_xlabel(f'{biomarker} Prediction', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Waterfall Plot: {biomarker}\n(Sample {sample_idx+1} Explanation)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Style improvements
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff4444', label='Increases Prediction'),
            Patch(facecolor='#4444ff', label='Decreases Prediction'),
            Patch(facecolor='gray', label='Base Value')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        logger.info(f"✅ SHAP waterfall plot created for {biomarker}")
        return fig
    
    def create_shap_summary_plot(self, biomarker: str,
                                X_data: pd.DataFrame,
                                shap_values: np.ndarray,
                                max_features: int = 15) -> plt.Figure:
        """
        Create SHAP summary bar plot.
        
        Parameters:
        -----------
        biomarker : str
            Target biomarker name
        X_data : pd.DataFrame
            Feature data
        shap_values : np.ndarray
            Real SHAP values
        max_features : int
            Maximum number of features to display
            
        Returns:
        --------
        plt.Figure
            Created figure
        """
        logger.info(f"📊 Creating SHAP summary plot for {biomarker}")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort by importance
        importance_order = np.argsort(mean_abs_shap)[::-1][:max_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # Create horizontal bar plot
        y_pos = np.arange(len(importance_order))
        feature_names = [X_data.columns[i] for i in importance_order]
        importances = mean_abs_shap[importance_order]
        
        bars = ax.barh(y_pos, importances, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Feature Importance: {biomarker}\n(Mean Absolute SHAP Values)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Style improvements
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        logger.info(f"✅ SHAP summary plot created for {biomarker}")
        return fig
    
    def generate_all_visualizations(self, biomarkers: Optional[List[str]] = None,
                                  max_biomarkers: int = 5) -> Dict[str, Dict]:
        """
        Generate all SHAP visualizations for specified biomarkers.
        
        Parameters:
        -----------
        biomarkers : List[str], optional
            List of biomarkers to visualize. If None, uses top biomarkers.
        max_biomarkers : int
            Maximum number of biomarkers to process
            
        Returns:
        --------
        Dict[str, Dict]
            Generated visualization information
        """
        logger.info("🎨 Generating all SHAP visualizations...")
        
        # Load trained models
        models = self.load_trained_models()
        
        if not models:
            logger.error("❌ No trained models found. Please run the ML pipeline first.")
            return {}
        
        # Select biomarkers to visualize
        if biomarkers is None:
            biomarkers = list(models.keys())[:max_biomarkers]
        
        logger.info(f"🎯 Creating visualizations for {len(biomarkers)} biomarkers")
        
        visualization_results = {}
        
        for biomarker in biomarkers:
            logger.info(f"🧬 Processing {biomarker}...")
            
            try:
                # Prepare SHAP data
                model, X_data, shap_values = self.prepare_shap_data(biomarker, models)
                
                if model is None or X_data is None or shap_values is None:
                    logger.warning(f"⚠️  Skipping {biomarker} due to data preparation failure")
                    continue
                
                biomarker_results = {}
                
                # Create beeswarm plot
                try:
                    beeswarm_fig = self.create_shap_beeswarm_plot(biomarker, X_data, shap_values)
                    beeswarm_path = self.config.get_output_path('figures', 
                                                               f'shap_beeswarm_{biomarker.replace(" ", "_")}_{self.timestamp}.svg')
                    beeswarm_fig.savefig(beeswarm_path, format='svg', bbox_inches='tight', facecolor='white')
                    plt.close(beeswarm_fig)
                    
                    biomarker_results['beeswarm'] = str(beeswarm_path)
                    logger.info(f"   ✅ Beeswarm plot saved: {beeswarm_path.name}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Beeswarm plot failed: {e}")
                
                # Create waterfall plot
                try:
                    waterfall_fig = self.create_shap_waterfall_plot(biomarker, X_data, shap_values, model)
                    waterfall_path = self.config.get_output_path('figures', 
                                                                f'shap_waterfall_{biomarker.replace(" ", "_")}_{self.timestamp}.svg')
                    waterfall_fig.savefig(waterfall_path, format='svg', bbox_inches='tight', facecolor='white')
                    plt.close(waterfall_fig)
                    
                    biomarker_results['waterfall'] = str(waterfall_path)
                    logger.info(f"   ✅ Waterfall plot saved: {waterfall_path.name}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Waterfall plot failed: {e}")
                
                # Create summary plot
                try:
                    summary_fig = self.create_shap_summary_plot(biomarker, X_data, shap_values)
                    summary_path = self.config.get_output_path('figures', 
                                                              f'shap_summary_{biomarker.replace(" ", "_")}_{self.timestamp}.svg')
                    summary_fig.savefig(summary_path, format='svg', bbox_inches='tight', facecolor='white')
                    plt.close(summary_fig)
                    
                    biomarker_results['summary'] = str(summary_path)
                    logger.info(f"   ✅ Summary plot saved: {summary_path.name}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Summary plot failed: {e}")
                
                if biomarker_results:
                    visualization_results[biomarker] = biomarker_results
                    logger.info(f"✅ {biomarker} visualizations complete")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {biomarker}: {e}")
        
        # Save visualization index
        if visualization_results:
            index_path = self.config.get_output_path('figures', f'shap_visualization_index_{self.timestamp}.json')
            with open(index_path, 'w') as f:
                json.dump(visualization_results, f, indent=2)
            
            logger.info(f"📋 Visualization index saved: {index_path}")
        
        logger.info(f"🎉 SHAP visualization generation complete: {len(visualization_results)} biomarkers")
        
        return visualization_results

def main():
    """Main function to generate corrected SHAP visualizations."""
    print("🎨 Starting Corrected SHAP Visualization Generation...")
    print("="*80)
    
    try:
        visualizer = CorrectedSHAPVisualizer()
        results = visualizer.generate_all_visualizations(max_biomarkers=3)  # Limit for demo
        
        if results:
            print(f"\n✅ SUCCESS: Generated SHAP visualizations for {len(results)} biomarkers")
            print("\nGenerated files:")
            for biomarker, files in results.items():
                print(f"\n{biomarker}:")
                for plot_type, file_path in files.items():
                    print(f"   - {plot_type}: {Path(file_path).name}")
        else:
            print("\n❌ No visualizations generated. Please check logs for errors.")
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)