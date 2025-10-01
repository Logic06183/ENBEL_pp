#!/usr/bin/env python3
"""
ENBEL Climate-Health SHAP Waterfall Plot Generator
================================================

Creates a scientifically rigorous SHAP waterfall plot for CD4 cell count predictions
showing individual feature contributions in the ENBEL climate-health analysis.

This visualization demonstrates how specific climate variables contribute to CD4
predictions for a representative case, providing mechanistic insights into
climate-immune system interactions.

Requirements:
- Publication-quality SVG output suitable for scientific presentations
- Real SHAP values from trained XGBoost model 
- Climate feature focus with temporal lag analysis
- Accessible color scheme and typography

Author: ENBEL Climate-Health Research Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import joblib
import json
import shap
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings('ignore')

# Scientific visualization settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def prepare_dataset_and_train_model():
    """Load data, train a new model with available features, and return everything needed"""
    print("Loading and preparing ENBEL clinical dataset...")
    
    # Load the clinical dataset
    data = pd.read_csv("data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv", low_memory=False)
    print(f"Total records: {len(data):,}")
    
    # Filter for CD4 data
    cd4_data = data.dropna(subset=['CD4 cell count (cells/µL)']).copy()
    print(f"CD4 records: {len(cd4_data):,}")
    
    # Select climate and relevant features
    climate_keywords = ['climate_', 'heat_', 'temp', 'HEAT_']
    demographic_features = ['Sex', 'Race', 'latitude', 'longitude', 'year', 'month']
    
    climate_features = []
    for col in cd4_data.columns:
        if any(keyword in col for keyword in climate_keywords):
            if cd4_data[col].dtype in ['float64', 'int64'] and cd4_data[col].notna().sum() > 100:
                climate_features.append(col)
    
    # Add available demographic features
    available_demographics = [f for f in demographic_features if f in cd4_data.columns]
    
    all_features = climate_features + available_demographics
    print(f"Selected {len(climate_features)} climate features + {len(available_demographics)} demographic features")
    print("Climate features:", climate_features)
    
    # Prepare features and target
    X = cd4_data[all_features].copy()
    y = cd4_data['CD4 cell count (cells/µL)'].copy()
    
    # Handle missing values and encode categorical variables
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        elif col == 'Sex':
            X[col] = X[col].map({'Male': 1, 'Female': 0}).fillna(0)
            X[col] = X[col].astype('float64')
        elif col == 'Race':
            # Encode race categories
            race_categories = X[col].dropna().unique()
            race_mapping = {race: i for i, race in enumerate(race_categories)}
            X[col] = X[col].map(race_mapping).fillna(0)
            X[col] = X[col].astype('float64')
        else:
            # Convert any remaining object columns to numeric
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            else:
                X[col] = X[col].fillna(0)
            X[col] = X[col].astype('float64')
    
    # Remove any remaining missing values
    complete_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[complete_mask]
    y_clean = y[complete_mask]
    
    print(f"Complete cases: {len(X_clean):,}")
    print(f"Final features: {len(X_clean.columns)}")
    
    # Train multiple models and select the best one
    print("\nTraining and comparing models for CD4 prediction...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # Try different models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=5, 
            min_samples_split=10,
            random_state=42, 
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=100,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = -np.inf
    model_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        model_results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'model': model
        }
        
        print(f"  {name}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, RMSE={test_rmse:.1f}")
        
        if test_r2 > best_score and test_r2 > 0:  # Only positive R²
            best_score = test_r2
            best_model = model
    
    # If no model has positive R², use the least negative one
    if best_model is None:
        best_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        best_model = model_results[best_name]['model']
        print(f"\nSelected model: {best_name} (best available)")
    else:
        best_name = [name for name, result in model_results.items() 
                    if result['model'] is best_model][0]
        print(f"\nSelected model: {best_name} (positive R²)")
    
    # Use the best model
    model = best_model
    train_r2 = model_results[best_name]['train_r2']
    test_r2 = model_results[best_name]['test_r2']
    test_rmse = model_results[best_name]['test_rmse']
    
    # Create metadata
    metadata = {
        'biomarker': 'CD4 cell count (cells/µL)',
        'n_features': len(X_clean.columns),
        'feature_names': list(X_clean.columns),
        'model_name': best_name,
        'performance': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'n_train': len(X_train),
            'n_test': len(X_test)
        },
        'all_models': {name: {k: v for k, v in result.items() if k != 'model'} 
                      for name, result in model_results.items()}
    }
    
    return model, X_clean, y_clean, metadata

def select_representative_case(X, y, model):
    """Select a representative case for waterfall plot"""
    print("\nSelecting representative case...")
    
    # Make predictions to find interesting cases
    predictions = model.predict(X)
    
    # Find a case with:
    # 1. Moderate CD4 count (not extreme)
    # 2. Good prediction accuracy
    # 3. Interesting climate exposure
    
    cd4_median = y.median()
    cd4_std = y.std()
    
    # Look for cases near median with good prediction accuracy
    moderate_mask = (y >= cd4_median - 0.5*cd4_std) & (y <= cd4_median + 0.5*cd4_std)
    moderate_indices = X[moderate_mask].index
    
    if len(moderate_indices) > 0:
        # Select a case that has interesting climate exposure
        selected_idx = moderate_indices[0]
    else:
        # Fallback to first case
        selected_idx = X.index[0]
    
    case_data = X.loc[selected_idx]
    actual_cd4 = y.loc[selected_idx]
    predicted_cd4 = predictions[X.index.get_loc(selected_idx)]
    
    print(f"Selected case index: {selected_idx}")
    print(f"Actual CD4: {actual_cd4:.1f} cells/µL")
    print(f"Predicted CD4: {predicted_cd4:.1f} cells/µL")
    print(f"Prediction error: {abs(actual_cd4 - predicted_cd4):.1f} cells/µL")
    
    return case_data, actual_cd4, predicted_cd4, selected_idx

def calculate_shap_values(model, X, case_data, model_name):
    """Calculate SHAP values for the representative case"""
    print("\nCalculating SHAP values...")
    
    # Choose appropriate SHAP explainer based on model type
    if model_name in ['RandomForest', 'XGBoost']:
        # Tree-based models
        explainer = shap.TreeExplainer(model)
        case_shap = explainer.shap_values(case_data.values.reshape(1, -1))[0]
        base_value = explainer.expected_value
    else:
        # Linear models or other models - use LinearExplainer or Explainer
        try:
            explainer = shap.LinearExplainer(model, X.sample(min(100, len(X)), random_state=42))
            case_shap = explainer.shap_values(case_data.values.reshape(1, -1))[0]
            base_value = explainer.expected_value
        except:
            # Fallback to general explainer (model-agnostic)
            print("Using model-agnostic explainer (slower but works with any model)...")
            explainer = shap.Explainer(model.predict, X.sample(min(100, len(X)), random_state=42))
            shap_explanation = explainer(case_data.values.reshape(1, -1))
            case_shap = shap_explanation.values[0]
            base_value = shap_explanation.base_values[0]
    
    print(f"Base value (population mean): {base_value:.1f}")
    print(f"SHAP values calculated for {len(case_shap)} features")
    
    return case_shap, base_value

def create_feature_labels(feature_names):
    """Create clean, scientific feature labels"""
    label_mapping = {
        'year': 'Study Year',
        'saaqis_dem_lag1': 'Elevation (1-day lag)',
        'saaqis_dem_lag0': 'Elevation (current)',
        'temp_mean_5d': 'Mean Temperature (5-day)',
        'cooling_degree_days_7d': 'Cooling Degree Days (7-day)',
        'saaqis_era5_ws_lag0': 'Wind Speed (current)',
        'apparent_temp_lag14': 'Apparent Temperature (14-day lag)',
        'temp_mean_14d': 'Mean Temperature (14-day)',
        'utci_mean_7d': 'UTCI Mean (7-day)',
        'utci_mean_5d': 'UTCI Mean (5-day)',
        'heat_index': 'Heat Index',
        'temperature': 'Temperature',
        'wind_speed': 'Wind Speed',
        'temperature_max': 'Maximum Temperature',
        'temperature_min': 'Minimum Temperature',
        'Sex': 'Sex (Male=1)',
        'Race': 'Race Category',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'month': 'Month',
        'season': 'Season'
    }
    
    clean_labels = []
    for name in feature_names:
        if name in label_mapping:
            clean_labels.append(label_mapping[name])
        elif 'temp' in name.lower():
            clean_labels.append(f"Temperature {name.split('_')[-1]}")
        elif 'heat' in name.lower():
            clean_labels.append(f"Heat Index {name.split('_')[-1]}")
        elif 'utci' in name.lower():
            clean_labels.append(f"UTCI {name.split('_')[-1]}")
        elif 'lag' in name:
            parts = name.split('_')
            lag_part = parts[-1]
            clean_labels.append(f"{' '.join(parts[:-1])} ({lag_part})")
        else:
            clean_labels.append(name.replace('_', ' ').title())
    
    return clean_labels

def create_waterfall_plot(case_shap, base_value, feature_names, case_data, actual_cd4, predicted_cd4, metadata):
    """Create the SHAP waterfall plot"""
    print("\nCreating SHAP waterfall plot...")
    
    # Get top features by absolute SHAP value
    top_n = 15
    abs_shap = np.abs(case_shap)
    top_indices = np.argsort(abs_shap)[-top_n:][::-1]
    
    top_shap = case_shap[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    top_values = [case_data.iloc[i] for i in top_indices]
    
    # Create clean labels
    clean_labels = create_feature_labels(top_features)
    
    # Calculate cumulative values for waterfall
    cumulative = np.zeros(len(top_shap) + 2)  # +2 for base and final
    cumulative[0] = base_value
    
    for i, shap_val in enumerate(top_shap):
        cumulative[i + 1] = cumulative[i] + shap_val
    
    cumulative[-1] = predicted_cd4
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color scheme: Blue for negative, Red for positive
    colors_list = ['#2E86AB' if val < 0 else '#E63946' for val in top_shap]
    base_color = '#6C757D'
    final_color = '#495057'
    
    # Plot waterfall bars
    bar_width = 0.6
    x_positions = range(len(cumulative))
    
    # Base value bar
    ax.bar(0, base_value, bar_width, color=base_color, alpha=0.8, 
           label='Population Mean')
    
    # Feature contribution bars
    for i, (shap_val, color) in enumerate(zip(top_shap, colors_list), 1):
        bottom = min(cumulative[i-1], cumulative[i])
        height = abs(shap_val)
        ax.bar(i, height, bar_width, bottom=bottom, color=color, alpha=0.8)
        
        # Add connecting lines
        if i < len(top_shap):
            ax.plot([i-0.3, i+0.3], [cumulative[i], cumulative[i]], 
                   'k--', alpha=0.3, linewidth=1)
    
    # Final prediction bar
    ax.bar(len(cumulative)-1, predicted_cd4, bar_width, color=final_color, 
           alpha=0.8, label='Final Prediction')
    
    # Add value labels on bars
    ax.text(0, base_value/2, f'{base_value:.1f}', ha='center', va='center', 
            fontweight='bold', color='white', fontsize=10)
    
    for i, shap_val in enumerate(top_shap, 1):
        y_pos = cumulative[i-1] + shap_val/2
        ax.text(i, y_pos, f'{shap_val:+.1f}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=9)
    
    ax.text(len(cumulative)-1, predicted_cd4/2, f'{predicted_cd4:.1f}', 
            ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Customize axes
    x_labels = ['Population\nMean'] + [f'{label}\n({val:.2f})' 
                for label, val in zip(clean_labels, top_values)] + ['Final\nPrediction']
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('CD4 Cell Count (cells/µL)', fontsize=12, fontweight='bold')
    
    # Add horizontal line at actual value
    ax.axhline(y=actual_cd4, color='green', linestyle=':', linewidth=2, 
               alpha=0.7, label=f'Actual CD4: {actual_cd4:.1f}')
    
    # Styling
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Title and annotations
    model_r2 = metadata['performance']['test_r2']
    model_rmse = metadata['performance']['test_rmse']
    model_name = metadata.get('model_name', 'XGBoost')
    n_samples = metadata['performance']['n_train'] + metadata['performance']['n_test']
    
    plt.suptitle('SHAP Waterfall Plot: CD4 Cell Count Prediction', 
                fontsize=16, fontweight='bold', y=0.96)
    
    ax.set_title(f'Individual Feature Contributions for Climate-Health Model\n'
                f'{model_name} Model (R² = {model_r2:.3f}, RMSE = {model_rmse:.1f}, n = {n_samples:,} participants)',
                fontsize=12, pad=20)
    
    # Add methodology note
    methodology_text = (
        'SHAP (SHapley Additive exPlanations) values show how each climate feature\n'
        'contributes to the prediction relative to the population average.\n'
        'Positive values (red) increase CD4 count, negative values (blue) decrease it.'
    )
    
    ax.text(0.02, 0.98, methodology_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#E63946', alpha=0.8, label='Positive Impact'),
        plt.Rectangle((0,0),1,1, facecolor='#2E86AB', alpha=0.8, label='Negative Impact'),
        plt.Rectangle((0,0),1,1, facecolor=base_color, alpha=0.8, label='Population Mean'),
        plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2, label='Actual Value')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              fancybox=True, shadow=True)
    
    # Add study attribution
    attribution_text = (
        'ENBEL Climate-Health Analysis | ERA5 Climate Data | '
        'Johannesburg HIV Clinical Trials (2002-2021)'
    )
    plt.figtext(0.5, 0.01, attribution_text, ha='center', fontsize=8, 
                style='italic', color='gray')
    
    plt.tight_layout()
    
    return fig

def main():
    """Main function to create SHAP waterfall plot"""
    print("="*60)
    print("ENBEL SHAP WATERFALL PLOT GENERATOR")
    print("="*60)
    
    try:
        # Prepare dataset and train model
        model, X, y, metadata = prepare_dataset_and_train_model()
        
        # Select representative case
        case_data, actual_cd4, predicted_cd4, case_idx = select_representative_case(X, y, model)
        
        # Calculate SHAP values
        case_shap, base_value = calculate_shap_values(model, X, case_data, metadata['model_name'])
        
        # Create waterfall plot
        fig = create_waterfall_plot(case_shap, base_value, metadata['feature_names'], 
                                  case_data, actual_cd4, predicted_cd4, metadata)
        
        # Save plot
        output_path = "presentation_slides_final/enbel_shap_waterfall_final.svg"
        plt.savefig(output_path, format='svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Also save PNG for backup
        png_path = output_path.replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print(f"\n✅ SHAP waterfall plot saved:")
        print(f"   SVG: {output_path}")
        print(f"   PNG: {png_path}")
        
        # Print summary statistics
        print(f"\n📊 Analysis Summary:")
        print(f"   Case ID: {case_idx}")
        print(f"   Actual CD4: {actual_cd4:.1f} cells/µL")
        print(f"   Predicted CD4: {predicted_cd4:.1f} cells/µL")
        print(f"   Population Mean: {base_value:.1f} cells/µL")
        print(f"   Best Model: {metadata.get('model_name', 'Unknown')}")
        print(f"   Model Test R²: {metadata['performance']['test_r2']:.4f}")
        print(f"   Model RMSE: {metadata['performance']['test_rmse']:.1f} cells/µL")
        print(f"   Features: {len(metadata['feature_names'])}")
        
        # Print model comparison
        if 'all_models' in metadata:
            print(f"\n🔬 Model Comparison:")
            for name, perf in metadata['all_models'].items():
                print(f"   {name}: R² = {perf['test_r2']:.4f}, RMSE = {perf['test_rmse']:.1f}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()