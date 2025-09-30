#!/usr/bin/env python3
"""
Demonstration Script for State-of-the-Art Climate-Health Analysis Pipeline
===========================================================================

This script demonstrates how to use the state-of-the-art climate-health analysis
pipeline with sample data and various configuration options.

Authors: Climate-Health Research Team
Version: 1.0.0
Date: 2025-09-30
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from state_of_the_art_climate_health_pipeline import (
    PipelineConfig,
    StateOfTheArtClimateHealthPipeline
)


def create_realistic_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create realistic sample climate-health data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    df : DataFrame
        Sample dataset with climate and health variables
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic climate data
    # Base temperature with seasonal variation
    days = np.arange(n_samples)
    seasonal_temp = 20 + 10 * np.sin(2 * np.pi * days / 365.25)
    temperature = seasonal_temp + np.random.normal(0, 5, n_samples)
    
    # Temperature-dependent humidity (inverse relationship)
    humidity = 80 - 0.5 * temperature + np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 20, 95)  # Realistic humidity range
    
    # Wind speed (log-normal distribution)
    wind_speed = np.random.lognormal(mean=1.5, sigma=0.5, size=n_samples)
    
    # Derived climate variables
    temperature_max = temperature + np.random.uniform(3, 8, n_samples)
    temperature_min = temperature - np.random.uniform(3, 8, n_samples)
    temperature_range = temperature_max - temperature_min
    
    # Generate demographic variables
    sex = np.random.choice(['M', 'F'], n_samples, p=[0.48, 0.52])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                           n_samples, p=[0.6, 0.13, 0.18, 0.06, 0.03])
    age = np.random.normal(45, 15, n_samples)
    age = np.clip(age, 18, 85)
    
    # Generate health outcomes with realistic climate relationships
    
    # Fasting glucose - affected by temperature stress
    base_glucose = 95 + 0.3 * age + (sex == 'M') * 5
    temp_stress = np.where(temperature > 30, (temperature - 30) * 0.8, 0)
    temp_stress += np.where(temperature < 5, (5 - temperature) * 0.5, 0)
    fasting_glucose = base_glucose + temp_stress + np.random.normal(0, 12, n_samples)
    fasting_glucose = np.clip(fasting_glucose, 70, 300)
    
    # Blood pressure - affected by temperature and humidity
    base_systolic = 110 + 0.5 * age + (sex == 'M') * 8
    temp_effect = 0.3 * (temperature - 20)  # Higher temp = higher BP
    humidity_effect = 0.1 * (humidity - 60)  # Higher humidity = higher BP
    systolic_bp = base_systolic + temp_effect + humidity_effect + np.random.normal(0, 15, n_samples)
    systolic_bp = np.clip(systolic_bp, 80, 200)
    
    diastolic_bp = 0.6 * systolic_bp + np.random.normal(10, 8, n_samples)
    diastolic_bp = np.clip(diastolic_bp, 50, 120)
    
    # Cholesterol levels - seasonal variation
    seasonal_effect = 5 * np.sin(2 * np.pi * days / 365.25 + np.pi)  # Peak in winter
    total_cholesterol = 180 + 0.2 * age + (sex == 'F') * 10 + seasonal_effect + np.random.normal(0, 25, n_samples)
    total_cholesterol = np.clip(total_cholesterol, 120, 350)
    
    hdl_cholesterol = 50 + (sex == 'F') * 10 + np.random.normal(0, 12, n_samples)
    hdl_cholesterol = np.clip(hdl_cholesterol, 25, 100)
    
    ldl_cholesterol = total_cholesterol - hdl_cholesterol - np.random.uniform(10, 30, n_samples)
    ldl_cholesterol = np.clip(ldl_cholesterol, 50, 250)
    
    # Hemoglobin - affected by temperature (heat stress can cause dehydration)
    base_hemoglobin = 14 + (sex == 'M') * 2 - 0.02 * age
    heat_stress = np.where(temperature > 32, (temperature - 32) * 0.1, 0)
    hemoglobin = base_hemoglobin + heat_stress + np.random.normal(0, 1.2, n_samples)
    hemoglobin = np.clip(hemoglobin, 8, 20)
    
    # Creatinine - kidney function affected by heat stress
    base_creatinine = 0.9 + (sex == 'M') * 0.3 + age * 0.005
    heat_kidney_stress = np.where(temperature > 35, (temperature - 35) * 0.02, 0)
    creatinine = base_creatinine + heat_kidney_stress + np.random.normal(0, 0.2, n_samples)
    creatinine = np.clip(creatinine, 0.5, 3.0)
    
    # CD4 cell count - immune function affected by environmental stress
    base_cd4 = 800 - age * 5
    environmental_stress = 0.3 * np.abs(temperature - 22) + 0.1 * np.abs(humidity - 50)
    cd4_count = base_cd4 - environmental_stress + np.random.normal(0, 150, n_samples)
    cd4_count = np.clip(cd4_count, 200, 1500)
    
    # Create DataFrame
    data = {
        # Climate variables
        'temperature': temperature,
        'temperature_max': temperature_max,
        'temperature_min': temperature_min,
        'temperature_range': temperature_range,
        'humidity': humidity,
        'humidity_max': humidity + np.random.uniform(5, 15, n_samples),
        'humidity_min': humidity - np.random.uniform(5, 15, n_samples),
        'wind_speed': wind_speed,
        'wind_gust': wind_speed * np.random.uniform(1.2, 2.0, n_samples),
        
        # Demographics
        'Sex': sex,
        'Race': race,
        'Age': age,
        
        # Health outcomes
        'FASTING_GLUCOSE': fasting_glucose,
        'systolic_blood_pressure': systolic_bp,
        'diastolic_blood_pressure': diastolic_bp,
        'FASTING_TOTAL_CHOLESTEROL': total_cholesterol,
        'FASTING_HDL': hdl_cholesterol,
        'FASTING_LDL': ldl_cholesterol,
        'Hemoglobin_gdL': hemoglobin,
        'Creatinine_mgdL': creatinine,
        'CD4_cell_count_cellsµL': cd4_count,
        
        # Add some date information
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic humidity bounds
    df['humidity_max'] = np.clip(df['humidity_max'], df['humidity'], 100)
    df['humidity_min'] = np.clip(df['humidity_min'], 0, df['humidity'])
    
    return df


def demonstrate_basic_usage():
    """Demonstrate basic pipeline usage."""
    print("=" * 60)
    print("BASIC PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    print("Creating realistic sample climate-health data...")
    df = create_realistic_sample_data(n_samples=500)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Configure pipeline
        config = PipelineConfig()
        config.data_path = temp_file.name
        config.target_variables = ['FASTING_GLUCOSE', 'systolic_blood_pressure']
        config.climate_variables = ['temperature', 'humidity', 'wind_speed']
        config.ensemble_methods = ['random_forest', 'xgboost']
        config.bootstrap_iterations = 50  # Reduced for demo
        config.output_dir = "demo_results"
        
        print(f"Configuration:")
        print(f"  - Data samples: {len(df)}")
        print(f"  - Target variables: {config.target_variables}")
        print(f"  - Climate variables: {config.climate_variables}")
        print(f"  - ML methods: {config.ensemble_methods}")
        
        # Initialize and run pipeline
        print("\nInitializing pipeline...")
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        
        print("Running complete analysis...")
        results = pipeline.run_complete_analysis()
        
        # Print results
        print("\n" + "=" * 40)
        print("RESULTS SUMMARY")
        print("=" * 40)
        print(f"Status: {results['status']}")
        
        if results['status'] == 'completed':
            print(f"Targets analyzed: {results['targets_analyzed']}")
            print(f"Best model R²: {results['best_performance']:.4f}")
            print(f"Output directory: {results['output_directory']}")
            
            # Show ensemble results
            if 'ensemble' in pipeline.results:
                print("\nModel Performance by Target:")
                for target, result in pipeline.results['ensemble'].items():
                    r2 = result['ensemble_performance']['r2']
                    rmse = result['ensemble_performance']['rmse']
                    print(f"  {target}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
    
    finally:
        # Cleanup
        Path(temp_file.name).unlink()


def demonstrate_advanced_features():
    """Demonstrate advanced pipeline features."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Create larger dataset
    print("Creating larger dataset for advanced analysis...")
    df = create_realistic_sample_data(n_samples=1000)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Advanced configuration
        config = PipelineConfig()
        config.data_path = temp_file.name
        config.target_variables = ['FASTING_GLUCOSE', 'systolic_blood_pressure', 'FASTING_HDL']
        config.ensemble_methods = ['random_forest', 'xgboost', 'lightgbm']
        config.max_lag = 14  # Shorter lag for demo
        config.bootstrap_iterations = 100
        config.output_dir = "demo_advanced_results"
        config.create_plots = True
        config.save_models = True
        config.generate_report = True
        
        print("Advanced configuration:")
        print(f"  - Multiple targets: {len(config.target_variables)}")
        print(f"  - Ensemble methods: {config.ensemble_methods}")
        print(f"  - DLNM lag period: {config.max_lag} days")
        print(f"  - Bootstrap iterations: {config.bootstrap_iterations}")
        print(f"  - Full outputs enabled: plots, models, report")
        
        # Initialize pipeline
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        
        # Run individual components for demonstration
        print("\nRunning pipeline components...")
        
        # 1. Load and validate data
        print("1. Loading and validating data...")
        df_loaded = pipeline.load_and_validate_data()
        print(f"   Loaded {len(df_loaded)} samples with {len(df_loaded.columns)} variables")
        
        # 2. Feature engineering
        print("2. Engineering climate features...")
        df_enhanced = pipeline.engineer_climate_features(df_loaded)
        new_features = len(df_enhanced.columns) - len(df_loaded.columns)
        print(f"   Created {new_features} new climate features")
        
        # 3. Ensemble modeling
        print("3. Training ensemble models...")
        ensemble_results = pipeline.fit_ensemble_models(df_enhanced)
        print(f"   Trained models for {len(ensemble_results)} targets")
        
        # 4. Uncertainty quantification
        print("4. Quantifying uncertainty...")
        uncertainty_results = pipeline.quantify_uncertainty(df_enhanced)
        print(f"   Calculated uncertainty for {len(uncertainty_results)} targets")
        
        # 5. Interpretability analysis
        print("5. Generating interpretability analysis...")
        interpretability_results = pipeline.generate_interpretability(df_enhanced)
        print(f"   Generated interpretability for {len(interpretability_results)} targets")
        
        # 6. Save results
        print("6. Saving results...")
        pipeline.save_results()
        
        # 7. Generate report
        print("7. Generating comprehensive report...")
        report = pipeline.generate_report()
        
        print("\nAdvanced analysis completed!")
        print(f"Full results saved to: {config.output_dir}")
        
        # Show sample of interpretability results
        if interpretability_results:
            print("\nSample Interpretability Results:")
            for target, result in list(interpretability_results.items())[:2]:
                print(f"\n{target}:")
                print(f"  Best model: {result['best_model']}")
                if 'feature_importance' in result:
                    top_features = result['feature_importance'][:3]
                    print("  Top 3 features:")
                    for i, feat in enumerate(top_features, 1):
                        print(f"    {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    finally:
        # Cleanup
        Path(temp_file.name).unlink()


def demonstrate_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    df = create_realistic_sample_data(n_samples=300)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Custom configuration for specific research question
        print("Demonstrating custom configuration for heat stress analysis...")
        
        config = PipelineConfig()
        config.data_path = temp_file.name
        
        # Focus on heat-related health outcomes
        config.target_variables = ['FASTING_GLUCOSE', 'Creatinine_mgdL']  # Heat stress indicators
        config.climate_variables = ['temperature', 'humidity', 'wind_speed']
        
        # Customize for heat stress research
        config.max_lag = 7  # Shorter lag for acute heat effects
        config.lag_knots = [3]  # Single knot for simple lag structure
        config.ensemble_methods = ['random_forest', 'xgboost']  # Focus on tree-based methods
        
        # Statistical settings
        config.alpha_level = 0.01  # More stringent significance level
        config.bootstrap_iterations = 25  # Reduced for demo
        config.cv_folds = 3  # Fewer folds for speed
        
        # Output settings
        config.output_dir = "heat_stress_analysis"
        config.random_seed = 123  # Different seed for variety
        
        print("Custom configuration:")
        print(f"  - Focus: Heat stress effects")
        print(f"  - Targets: {config.target_variables}")
        print(f"  - Short lag period: {config.max_lag} days")
        print(f"  - Stringent α: {config.alpha_level}")
        print(f"  - Random seed: {config.random_seed}")
        
        # Run analysis
        pipeline = StateOfTheArtClimateHealthPipeline(config)
        results = pipeline.run_complete_analysis()
        
        print(f"\nCustom analysis status: {results['status']}")
        if results['status'] == 'completed':
            print(f"Results saved to: {results['output_directory']}")
    
    finally:
        # Cleanup
        Path(temp_file.name).unlink()


def create_sample_visualizations():
    """Create sample visualizations of the generated data."""
    print("\n" + "=" * 60)
    print("SAMPLE DATA VISUALIZATIONS")
    print("=" * 60)
    
    # Create data for visualization
    df = create_realistic_sample_data(n_samples=365)  # One year of data
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sample Climate-Health Data Characteristics', fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature and glucose relationship
    axes[0, 0].scatter(df['temperature'], df['FASTING_GLUCOSE'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Fasting Glucose (mg/dL)')
    axes[0, 0].set_title('Temperature vs Fasting Glucose')
    
    # Add trend line
    z = np.polyfit(df['temperature'], df['FASTING_GLUCOSE'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['temperature'].sort_values(), p(df['temperature'].sort_values()), "r--", alpha=0.8)
    
    # Plot 2: Seasonal temperature variation
    df['day_of_year'] = df['date'].dt.dayofyear
    axes[0, 1].plot(df['day_of_year'], df['temperature'], alpha=0.7)
    axes[0, 1].set_xlabel('Day of Year')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Seasonal Temperature Pattern')
    
    # Plot 3: Blood pressure by climate conditions
    df['temp_category'] = pd.cut(df['temperature'], bins=3, labels=['Cold', 'Moderate', 'Hot'])
    bp_by_temp = df.groupby('temp_category')['systolic_blood_pressure'].mean()
    axes[1, 0].bar(bp_by_temp.index, bp_by_temp.values)
    axes[1, 0].set_ylabel('Systolic BP (mmHg)')
    axes[1, 0].set_title('Blood Pressure by Temperature Category')
    
    # Plot 4: Climate variable correlations
    climate_vars = ['temperature', 'humidity', 'wind_speed']
    corr_matrix = df[climate_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Climate Variable Correlations')
    
    plt.tight_layout()
    plt.savefig('sample_data_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations created and saved as 'sample_data_visualizations.png'")
    
    # Print data summary
    print("\nSample Data Summary:")
    print("-" * 40)
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nClimate variable ranges:")
    for var in ['temperature', 'humidity', 'wind_speed']:
        print(f"  {var}: {df[var].min():.1f} to {df[var].max():.1f}")
    print(f"\nHealth outcome ranges:")
    health_vars = ['FASTING_GLUCOSE', 'systolic_blood_pressure', 'FASTING_HDL']
    for var in health_vars:
        print(f"  {var}: {df[var].min():.1f} to {df[var].max():.1f}")


def main():
    """Main demonstration function."""
    print("STATE-OF-THE-ART CLIMATE-HEALTH ANALYSIS PIPELINE")
    print("=" * 60)
    print("Comprehensive Demonstration Script")
    print("Version 1.0.0")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_advanced_features()
        demonstrate_custom_configuration()
        create_sample_visualizations()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Realistic sample data generation")
        print("✓ Basic pipeline configuration and execution")
        print("✓ Advanced feature engineering")
        print("✓ Ensemble machine learning methods")
        print("✓ Uncertainty quantification")
        print("✓ Interpretability analysis")
        print("✓ Custom configuration options")
        print("✓ Comprehensive reporting")
        print("✓ Data visualization")
        
        print("\nNext Steps:")
        print("1. Examine the generated output directories")
        print("2. Review the analysis reports")
        print("3. Explore the interpretability plots")
        print("4. Adapt configurations for your specific research questions")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()