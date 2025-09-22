#!/usr/bin/env python3
"""
Publication-Quality Visualizations for Climate-Health Analysis
===========================================================

Creates publication-ready figures for the climate-health study in African cities.

Author: Climate-Health Data Science Team
Date: 2025-09-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def setup_publication_style():
    """Set up publication-quality matplotlib parameters"""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'font.family': 'Arial',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })

def create_figure_1_climate_health_associations():
    """Figure 1: Temperature-Health Associations in Clinical Cohort"""
    
    # Load data
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    clinical = df[df['dataset_group'] == 'clinical'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Temperature-Health Associations in African Cities\n(Clinical Cohort Analysis)', fontsize=16, fontweight='bold')
    
    # Panel A: Glucose vs Temperature
    ax1 = axes[0, 0]
    glucose_temp_data = clinical[['FASTING GLUCOSE', 'temperature_tas_lag0']].dropna()
    
    if len(glucose_temp_data) > 100:
        # Create temperature bins for clearer visualization
        glucose_temp_data['temp_bin'] = pd.cut(glucose_temp_data['temperature_tas_lag0'], bins=10)
        glucose_by_temp = glucose_temp_data.groupby('temp_bin')['FASTING GLUCOSE'].agg(['mean', 'std', 'count'])
        glucose_by_temp = glucose_by_temp[glucose_by_temp['count'] >= 10]
        
        # Plot with error bars
        x_labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in glucose_by_temp.index]
        x_pos = range(len(x_labels))
        
        ax1.errorbar(x_pos, glucose_by_temp['mean'], yerr=glucose_by_temp['std']/np.sqrt(glucose_by_temp['count']),
                     fmt='o-', color='red', capsize=5, capthick=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Fasting Glucose (mg/dL)')
        ax1.set_title('A. Glucose Response to Temperature', fontweight='bold')
        ax1.set_xticks(x_pos[::2])
        ax1.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], rotation=45)
        
        # Add correlation coefficient
        corr = glucose_temp_data['FASTING GLUCOSE'].corr(glucose_temp_data['temperature_tas_lag0'])
        ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Panel B: Blood Pressure vs Temperature
    ax2 = axes[0, 1]
    bp_temp_data = clinical[['systolic blood pressure', 'temperature_tas_lag0']].dropna()
    
    if len(bp_temp_data) > 100:
        bp_temp_data['temp_bin'] = pd.cut(bp_temp_data['temperature_tas_lag0'], bins=10)
        bp_by_temp = bp_temp_data.groupby('temp_bin')['systolic blood pressure'].agg(['mean', 'std', 'count'])
        bp_by_temp = bp_by_temp[bp_by_temp['count'] >= 10]
        
        x_labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in bp_by_temp.index]
        x_pos = range(len(x_labels))
        
        ax2.errorbar(x_pos, bp_by_temp['mean'], yerr=bp_by_temp['std']/np.sqrt(bp_by_temp['count']),
                     fmt='o-', color='blue', capsize=5, capthick=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Systolic Blood Pressure (mmHg)')
        ax2.set_title('B. Blood Pressure Response to Temperature', fontweight='bold')
        ax2.set_xticks(x_pos[::2])
        ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], rotation=45)
        
        corr = bp_temp_data['systolic blood pressure'].corr(bp_temp_data['temperature_tas_lag0'])
        ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Panel C: Extreme Temperature Effects
    ax3 = axes[1, 0]
    
    if len(glucose_temp_data) > 100:
        temp_q90 = glucose_temp_data['temperature_tas_lag0'].quantile(0.9)
        temp_q10 = glucose_temp_data['temperature_tas_lag0'].quantile(0.1)
        
        glucose_temp_data['temp_category'] = 'Normal'
        glucose_temp_data.loc[glucose_temp_data['temperature_tas_lag0'] > temp_q90, 'temp_category'] = 'Hot'
        glucose_temp_data.loc[glucose_temp_data['temperature_tas_lag0'] < temp_q10, 'temp_category'] = 'Cold'
        
        # Box plot of glucose by temperature category
        sns.boxplot(data=glucose_temp_data, x='temp_category', y='FASTING GLUCOSE', ax=ax3,
                   order=['Cold', 'Normal', 'Hot'])
        ax3.set_xlabel('Temperature Category')
        ax3.set_ylabel('Fasting Glucose (mg/dL)')
        ax3.set_title('C. Extreme Temperature Effects on Glucose', fontweight='bold')
        
        # Add sample sizes
        for i, category in enumerate(['Cold', 'Normal', 'Hot']):
            n = len(glucose_temp_data[glucose_temp_data['temp_category'] == category])
            ax3.text(i, ax3.get_ylim()[1]*0.9, f'n={n}', ha='center', fontweight='bold')
    
    # Panel D: Sex Differences
    ax4 = axes[1, 1]
    
    if 'Sex' in clinical.columns:
        sex_temp_data = clinical[clinical['Sex'].isin(['Male', 'Female'])][['FASTING GLUCOSE', 'temperature_tas_lag0', 'Sex']].dropna()
        
        if len(sex_temp_data) > 100:
            # Scatter plot with regression lines by sex
            for sex, color in [('Male', 'blue'), ('Female', 'red')]:
                subset = sex_temp_data[sex_temp_data['Sex'] == sex]
                if len(subset) > 20:
                    ax4.scatter(subset['temperature_tas_lag0'], subset['FASTING GLUCOSE'], 
                              alpha=0.3, color=color, label=f'{sex} (n={len(subset)})')
                    
                    # Add regression line
                    z = np.polyfit(subset['temperature_tas_lag0'], subset['FASTING GLUCOSE'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(subset['temperature_tas_lag0'].min(), subset['temperature_tas_lag0'].max(), 100)
                    ax4.plot(x_range, p(x_range), color=color, linewidth=2)
                    
                    # Add correlation
                    corr = subset['FASTING GLUCOSE'].corr(subset['temperature_tas_lag0'])
                    ax4.text(0.05, 0.9 - (0 if sex == 'Male' else 0.1), f'{sex}: r={corr:.3f}', 
                            transform=ax4.transAxes, color=color, fontweight='bold')
            
            ax4.set_xlabel('Temperature (°C)')
            ax4.set_ylabel('Fasting Glucose (mg/dL)')
            ax4.set_title('D. Sex Differences in Temperature Response', fontweight='bold')
            ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/Figure1_Temperature_Health_Associations.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 1: Temperature-Health Associations saved")

def create_figure_2_socioeconomic_patterns():
    """Figure 2: Socioeconomic Vulnerability and Climate Exposure Patterns"""
    
    # Load data
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    socioeconomic = df[df['dataset_group'] == 'socioeconomic'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Socioeconomic Vulnerability and Climate Exposure in African Cities\n(Socioeconomic Cohort Analysis)', 
                 fontsize=16, fontweight='bold')
    
    # Panel A: Education and Heat Exposure
    ax1 = axes[0, 0]
    
    if 'Education' in socioeconomic.columns:
        edu_heat_data = socioeconomic[['Education', 'heat_index_lag0']].dropna()
        
        if len(edu_heat_data) > 100:
            # Box plot
            sns.boxplot(data=edu_heat_data, x='Education', y='heat_index_lag0', ax=ax1)
            ax1.set_xlabel('Education Level')
            ax1.set_ylabel('Heat Index')
            ax1.set_title('A. Heat Exposure by Education Level', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add sample sizes
            for i, edu in enumerate(edu_heat_data['Education'].unique()):
                n = len(edu_heat_data[edu_heat_data['Education'] == edu])
                ax1.text(i, ax1.get_ylim()[1]*0.95, f'n={n}', ha='center', fontsize=9)
    
    # Panel B: Employment and Heat Exposure
    ax2 = axes[0, 1]
    
    if 'employment_status' in socioeconomic.columns:
        emp_heat_data = socioeconomic[['employment_status', 'heat_index_lag0']].dropna()
        
        if len(emp_heat_data) > 50:
            sns.boxplot(data=emp_heat_data, x='employment_status', y='heat_index_lag0', ax=ax2)
            ax2.set_xlabel('Employment Status')
            ax2.set_ylabel('Heat Index')
            ax2.set_title('B. Heat Exposure by Employment Status', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add sample sizes
            for i, emp in enumerate(emp_heat_data['employment_status'].unique()):
                n = len(emp_heat_data[emp_heat_data['employment_status'] == emp])
                ax2.text(i, ax2.get_ylim()[1]*0.95, f'n={n}', ha='center', fontsize=9)
    
    # Panel C: Geographic Climate Patterns
    ax3 = axes[1, 0]
    
    if all(col in socioeconomic.columns for col in ['latitude', 'longitude', 'heat_index_lag0']):
        geo_data = socioeconomic[['latitude', 'longitude', 'heat_index_lag0']].dropna()
        
        if len(geo_data) > 100:
            # Create scatter plot with color representing heat index
            scatter = ax3.scatter(geo_data['longitude'], geo_data['latitude'], 
                                c=geo_data['heat_index_lag0'], cmap='Reds', alpha=0.6, s=1)
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title('C. Geographic Heat Exposure Patterns', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Heat Index')
    
    # Panel D: Education Distribution
    ax4 = axes[1, 1]
    
    if 'Education' in socioeconomic.columns:
        edu_counts = socioeconomic['Education'].value_counts()
        
        # Pie chart of education distribution
        ax4.pie(edu_counts.values, labels=edu_counts.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('D. Education Distribution\n(Socioeconomic Cohort)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/Figure2_Socioeconomic_Climate_Patterns.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 2: Socioeconomic-Climate Patterns saved")

def create_figure_3_methodological_overview():
    """Figure 3: Study Design and Methodological Framework"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Methodological Framework for Climate-Health Analysis in African Cities', 
                 fontsize=16, fontweight='bold')
    
    # Panel A: Study Design Flow Chart (Text-based)
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.9, 'STUDY DESIGN', ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax1.transAxes)
    
    design_text = """
Total Dataset: 18,205 participants
    ↓
Clinical Cohort (n=9,103)
• Biomarkers available
• Complete climate data
• Limited socioeconomic data

Socioeconomic Cohort (n=9,102)  
• Complete socioeconomic data
• Complete climate data
• No biomarkers

    ↓
Ecological Aggregation
Geographic units for integration
"""
    
    ax1.text(0.05, 0.8, design_text, ha='left', va='top', fontsize=10,
            transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax1.set_title('A. Study Design Overview', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Data Availability Heatmap
    ax2 = axes[0, 1]
    
    # Create data availability matrix
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    variables = ['FASTING GLUCOSE', 'systolic blood pressure', 'Education', 'employment_status', 
                'temperature_tas_lag0', 'heat_index_lag0']
    cohorts = ['clinical', 'socioeconomic']
    
    availability_matrix = []
    for cohort in cohorts:
        cohort_data = df[df['dataset_group'] == cohort]
        availability_row = []
        for var in variables:
            if var in cohort_data.columns:
                availability = cohort_data[var].notna().sum() / len(cohort_data) * 100
            else:
                availability = 0
            availability_row.append(availability)
        availability_matrix.append(availability_row)
    
    availability_df = pd.DataFrame(availability_matrix, columns=variables, index=cohorts)
    
    sns.heatmap(availability_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Data Availability (%)'}, ax=ax2)
    ax2.set_title('B. Data Availability by Cohort', fontweight='bold')
    ax2.set_ylabel('Cohort')
    ax2.set_xlabel('Variables')
    
    # Panel C: Climate Variable Distributions
    ax3 = axes[1, 0]
    
    climate_data = df[['temperature_tas_lag0', 'heat_index_lag0', 'dataset_group']].dropna()
    
    # Create violin plots for climate variables by cohort
    climate_melted = climate_data.melt(id_vars=['dataset_group'], 
                                     value_vars=['temperature_tas_lag0', 'heat_index_lag0'],
                                     var_name='climate_variable', value_name='value')
    
    sns.violinplot(data=climate_melted, x='climate_variable', y='value', 
                  hue='dataset_group', ax=ax3)
    ax3.set_title('C. Climate Variable Distributions', fontweight='bold')
    ax3.set_xlabel('Climate Variable')
    ax3.set_ylabel('Value')
    ax3.legend(title='Cohort')
    
    # Panel D: Geographic Coverage
    ax4 = axes[1, 1]
    
    if all(col in df.columns for col in ['latitude', 'longitude', 'dataset_group']):
        geo_data = df[['latitude', 'longitude', 'dataset_group']].dropna()
        
        colors = {'clinical': 'red', 'socioeconomic': 'blue'}
        for cohort in ['clinical', 'socioeconomic']:
            subset = geo_data[geo_data['dataset_group'] == cohort]
            ax4.scatter(subset['longitude'], subset['latitude'], 
                       c=colors[cohort], alpha=0.3, s=1, label=f'{cohort} (n={len(subset):,})')
        
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('D. Geographic Coverage', fontweight='bold')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/Figure3_Methodological_Framework.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Figure 3: Methodological Framework saved")

def create_supplementary_tables():
    """Create supplementary tables with detailed statistics"""
    
    # Load data
    df = pd.read_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    print("\n=== SUPPLEMENTARY TABLES ===")
    
    # Table S1: Descriptive Statistics
    print("\nTable S1: Descriptive Statistics by Cohort")
    print("-" * 60)
    
    variables_of_interest = {
        'Clinical Cohort': ['FASTING GLUCOSE', 'systolic blood pressure', 'diastolic blood pressure'],
        'Socioeconomic Cohort': ['Education', 'employment_status'],
        'Both Cohorts': ['temperature_tas_lag0', 'heat_index_lag0']
    }
    
    for category, variables in variables_of_interest.items():
        print(f"\n{category}:")
        
        for var in variables:
            if var in df.columns:
                if category == 'Clinical Cohort':
                    subset = df[df['dataset_group'] == 'clinical'][var]
                elif category == 'Socioeconomic Cohort':
                    subset = df[df['dataset_group'] == 'socioeconomic'][var]
                else:  # Both cohorts
                    subset = df[var]
                
                if subset.dtype in ['float64', 'int64']:
                    stats_summary = {
                        'n': subset.notna().sum(),
                        'mean': subset.mean(),
                        'std': subset.std(),
                        'min': subset.min(),
                        'max': subset.max(),
                        'missing': subset.isna().sum()
                    }
                    print(f"  {var}:")
                    print(f"    n={stats_summary['n']:,}, mean={stats_summary['mean']:.2f} ± {stats_summary['std']:.2f}")
                    print(f"    range: {stats_summary['min']:.1f} - {stats_summary['max']:.1f}, missing: {stats_summary['missing']:,}")
                else:
                    value_counts = subset.value_counts()
                    print(f"  {var}:")
                    for value, count in value_counts.head().items():
                        print(f"    {value}: {count:,} ({count/subset.notna().sum()*100:.1f}%)")
    
    # Table S2: Climate-Health Correlations
    print("\n\nTable S2: Climate-Health Correlations (Clinical Cohort)")
    print("-" * 60)
    
    clinical = df[df['dataset_group'] == 'clinical'].copy()
    
    health_vars = ['FASTING GLUCOSE', 'systolic blood pressure', 'diastolic blood pressure']
    climate_vars = ['temperature_tas_lag0', 'heat_index_lag0', 'temperature_tas_lag1', 'heat_index_lag1']
    
    correlation_table = []
    
    for health_var in health_vars:
        if health_var in clinical.columns:
            for climate_var in climate_vars:
                if climate_var in clinical.columns:
                    subset = clinical[[health_var, climate_var]].dropna()
                    if len(subset) > 50:
                        corr = subset[health_var].corr(subset[climate_var])
                        correlation_table.append({
                            'Health Variable': health_var,
                            'Climate Variable': climate_var,
                            'Correlation': corr,
                            'n': len(subset)
                        })
    
    corr_df = pd.DataFrame(correlation_table)
    print(corr_df.to_string(index=False, float_format='%.3f'))
    
    # Save tables to CSV
    corr_df.to_csv('/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/MLPaper/Table_S2_Climate_Health_Correlations.csv', index=False)
    print("\n✓ Supplementary tables saved")

def main():
    """Create all publication figures and tables"""
    
    setup_publication_style()
    
    print("Creating publication-quality visualizations...")
    print("=" * 60)
    
    # Create figures
    create_figure_1_climate_health_associations()
    create_figure_2_socioeconomic_patterns()
    create_figure_3_methodological_overview()
    
    # Create supplementary materials
    create_supplementary_tables()
    
    print("\n" + "=" * 60)
    print("All publication materials created successfully!")
    print("\nFiles created:")
    print("• Figure1_Temperature_Health_Associations.png")
    print("• Figure2_Socioeconomic_Climate_Patterns.png")
    print("• Figure3_Methodological_Framework.png")
    print("• Table_S2_Climate_Health_Correlations.csv")

if __name__ == "__main__":
    main()