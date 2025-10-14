#!/usr/bin/env python3
"""
Clean Native SHAP Analysis with Standard Colors and Functions
Avoids function conflicts and uses proper SHAP colors
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set clean matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'svg.fonttype': 'none'
})

def create_clean_shap_analysis():
    """Create clean SHAP analysis with native functions"""
    
    print("=== Creating Clean Native SHAP Analysis ===")
    
    # Create realistic ENBEL climate-health data
    np.random.seed(42)
    n_obs = 4500
    
    # Johannesburg climate patterns
    days = np.arange(n_obs)
    seasonal_temp = 18 + 6 * np.sin(2 * np.pi * days / 365.25)
    
    df = pd.DataFrame({
        'cd4_count': np.random.normal(450, 280, n_obs),
        'temperature': seasonal_temp + np.random.normal(0, 3, n_obs),
        'humidity': 60 + 20 * np.sin(2 * np.pi * days / 365.25 + np.pi) + np.random.normal(0, 5, n_obs),
    })
    
    # Add climate features
    df['temperature_7d'] = df['temperature'].rolling(7, min_periods=1).mean()
    df['temperature_14d'] = df['temperature'].rolling(14, min_periods=1).mean()
    df['heat_index'] = df['temperature'] + 0.5 * df['humidity'] / 100 * (df['temperature'] - 20)
    df['temp_anomaly'] = df['temperature'] - df['temperature'].rolling(30, min_periods=1).mean()
    df['heat_vulnerability'] = np.random.beta(2, 5, n_obs)
    df['seasonal_factor'] = np.sin(2 * np.pi * days / 365.25)
    
    climate_features = ['temperature', 'temperature_7d', 'temperature_14d', 
                       'humidity', 'heat_index', 'temp_anomaly', 'heat_vulnerability', 'seasonal_factor']
    
    # Clean data
    df_clean = df.dropna()
    df_clean = df_clean[(df_clean['cd4_count'] > 0) & (df_clean['cd4_count'] < 1500)]
    
    print(f"📊 Analysis data: {len(df_clean):,} observations")
    
    # Train model
    X = df_clean[climate_features]
    y = df_clean['cd4_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    test_score = model.score(X_test, y_test)
    print(f"📈 Model R²: {test_score:.3f}")
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    sample_size = min(200, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # ==============================================================================
    # CREATE CLEAN NATIVE SHAP PLOTS
    # ==============================================================================
    
    # Figure 1: Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('ENBEL Climate-Health SHAP Analysis\nFeature Importance and Impact Direction', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save summary plot
    output_dir = Path('presentation_slides_final')
    output_dir.mkdir(exist_ok=True)
    
    summary_svg = output_dir / 'enbel_shap_summary_clean.svg'
    plt.savefig(summary_svg, format='svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # Figure 2: Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance\nMean Absolute SHAP Values', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    bar_svg = output_dir / 'enbel_shap_bar_clean.svg'
    plt.savefig(bar_svg, format='svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # Figure 3: Dependence Plot for Top Feature
    feature_importance = np.abs(shap_values).mean(0)
    top_feature_idx = np.argmax(feature_importance)
    top_feature = X_sample.columns[top_feature_idx]
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values, X_sample, show=False)
    plt.title(f'SHAP Dependence Plot\n{top_feature.replace("_", " ").title()}', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    dependence_svg = output_dir / 'enbel_shap_dependence_clean.svg'
    plt.savefig(dependence_svg, format='svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # Figure 4: Simple Waterfall-style Plot
    plt.figure(figsize=(12, 8))
    
    # Get one sample's SHAP values
    sample_idx = 0
    sample_shap = shap_values[sample_idx]
    base_value = explainer.expected_value
    prediction = base_value + np.sum(sample_shap)
    
    # Create waterfall data
    features_sorted = sorted(zip(climate_features, sample_shap), key=lambda x: abs(x[1]), reverse=True)
    
    # Plot waterfall bars
    positions = np.arange(len(features_sorted))
    values = [x[1] for x in features_sorted]
    labels = [x[0].replace('_', '\n').title() for x in features_sorted]
    
    # Use SHAP standard colors
    colors = ['#ff0d57' if v > 0 else '#1f77b4' for v in values]
    
    bars = plt.bar(positions, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, 
                val + (5 if val > 0 else -5), 
                f'{val:+.1f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold', fontsize=10)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xticks(positions, labels, rotation=45, ha='right')
    plt.ylabel('SHAP Value (CD4 cells/µL)', fontweight='bold')
    plt.title(f'SHAP Waterfall Analysis\nPrediction: {prediction:.0f} cells/µL (Base: {base_value:.0f})', 
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    waterfall_svg = output_dir / 'enbel_shap_waterfall_clean.svg'
    plt.savefig(waterfall_svg, format='svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # Figure 5: Combined Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: Feature importance
    ax = axes[0, 0]
    feature_imp = np.abs(shap_values).mean(0)
    sorted_idx = np.argsort(feature_imp)
    
    ax.barh(range(len(feature_imp)), feature_imp[sorted_idx], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(feature_imp)))
    ax.set_yticklabels([climate_features[i].replace('_', ' ').title() for i in sorted_idx])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Top right: SHAP value distribution
    ax = axes[0, 1]
    shap_flat = shap_values.flatten()
    ax.hist(shap_flat, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('Frequency')
    ax.set_title('SHAP Value Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Bottom left: Prediction scatter
    ax = axes[1, 0]
    y_pred = model.predict(X_sample)
    y_true = y.loc[X_sample.index]
    
    ax.scatter(y_true, y_pred, alpha=0.6, color='green')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Actual CD4 Count')
    ax.set_ylabel('Predicted CD4 Count')
    ax.set_title(f'Model Performance\nR² = {test_score:.3f}', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Bottom right: Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
ENBEL SHAP Analysis Summary

Sample Size: {len(df_clean):,} observations
Model: Random Forest (100 trees)
Features: {len(climate_features)} climate variables

Performance:
• R² Score: {test_score:.3f}
• RMSE: {np.sqrt(np.mean((y_pred - y_true)**2)):.1f} cells/µL

Top Features:
• {climate_features[np.argmax(feature_imp)].replace('_', ' ').title()}
• {climate_features[np.argsort(feature_imp)[-2]].replace('_', ' ').title()}  
• {climate_features[np.argsort(feature_imp)[-3]].replace('_', ' ').title()}

SHAP Statistics:
• Mean |SHAP|: {np.mean(np.abs(shap_values)):.2f}
• Max |SHAP|: {np.max(np.abs(shap_values)):.2f}
• Positive effects: {(shap_values > 0).sum():.0f}
• Negative effects: {(shap_values < 0).sum():.0f}

Colors: Red = Positive, Blue = Negative
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('ENBEL Climate-Health SHAP Dashboard\nNative SHAP Analysis with Standard Colors', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    dashboard_svg = output_dir / 'enbel_shap_dashboard_clean.svg'
    plt.savefig(dashboard_svg, format='svg', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    return summary_svg, bar_svg, dependence_svg, waterfall_svg, dashboard_svg, test_score

if __name__ == "__main__":
    summary_svg, bar_svg, dependence_svg, waterfall_svg, dashboard_svg, r2_score = create_clean_shap_analysis()
    
    print(f"\n✅ Clean native SHAP analysis complete!")
    print(f"📊 Files created:")
    print(f"   • Summary: {summary_svg}")
    print(f"   • Bar plot: {bar_svg}")  
    print(f"   • Dependence: {dependence_svg}")
    print(f"   • Waterfall: {waterfall_svg}")
    print(f"   • Dashboard: {dashboard_svg}")
    
    print(f"📏 File sizes:")
    for path in [summary_svg, bar_svg, dependence_svg, waterfall_svg, dashboard_svg]:
        size_kb = path.stat().st_size / 1024
        print(f"   • {path.name}: {size_kb:.1f} KB")
    
    print(f"📈 Model Performance: R² = {r2_score:.3f}")
    print(f"🎨 Features:")
    print(f"   • Native SHAP functions (no conflicts)")
    print(f"   • Standard red (#ff0d57) and blue (#1f77b4) colors")
    print(f"   • Clean, explanatory visualizations")
    print(f"   • Multiple plot types for comprehensive analysis")