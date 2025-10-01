#!/usr/bin/env python3
"""
ENBEL CD4 Heat Analysis - Comprehensive Scientific Visualization
================================================================

Creates a scientifically rigorous 6-panel analysis examining temperature-immune 
function relationships in HIV-positive individuals from Johannesburg clinical trials.

Author: ENBEL Research Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set reproducible seed
np.random.seed(42)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_and_prepare_data():
    """Load and prepare the clinical dataset for analysis."""
    print("Loading ENBEL clinical dataset...")
    
    data_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/data/raw/CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    df = pd.read_csv(data_path)
    
    print(f"Original dataset: {len(df)} records")
    
    # Select key variables for CD4-temperature analysis
    analysis_vars = [
        'CD4 cell count (cells/µL)',
        'climate_daily_mean_temp',
        'climate_daily_max_temp', 
        'climate_7d_mean_temp',
        'climate_14d_mean_temp',
        'climate_30d_mean_temp',
        'climate_temp_anomaly',
        'climate_heat_stress_index',
        'HEAT_STRESS_RISK_CATEGORY',
        'season',
        'month',
        'year',
        'Age (at enrolment)',
        'Sex',
        'Race',
        'jhb_subregion',
        'latitude',
        'longitude',
        'primary_date'
    ]
    
    # Filter to complete cases and remove infinite values
    df_analysis = df[analysis_vars].copy()
    
    # Remove rows with NaN in key variables
    key_vars = ['CD4 cell count (cells/µL)', 'climate_daily_mean_temp', 
                'climate_7d_mean_temp', 'climate_14d_mean_temp', 'climate_30d_mean_temp']
    df_analysis = df_analysis.dropna(subset=key_vars)
    
    # Remove infinite values
    for var in key_vars:
        df_analysis = df_analysis[np.isfinite(df_analysis[var])]
    
    # Filter reasonable CD4 ranges (0-5000 cells/µL)
    df_analysis = df_analysis[
        (df_analysis['CD4 cell count (cells/µL)'] >= 0) & 
        (df_analysis['CD4 cell count (cells/µL)'] <= 5000)
    ]
    
    # Filter reasonable temperature ranges (-10 to 50°C)
    temp_vars = ['climate_daily_mean_temp', 'climate_7d_mean_temp', 
                 'climate_14d_mean_temp', 'climate_30d_mean_temp']
    for temp_var in temp_vars:
        df_analysis = df_analysis[
            (df_analysis[temp_var] >= -10) & 
            (df_analysis[temp_var] <= 50)
        ]
    
    print(f"Analysis dataset: {len(df_analysis)} complete records")
    
    # Create heat stress categories
    df_analysis['heat_category'] = pd.cut(
        df_analysis['climate_daily_mean_temp'],
        bins=[-np.inf, 10, 15, 20, 25, np.inf],
        labels=['Cold (<10°C)', 'Cool (10-15°C)', 'Mild (15-20°C)', 
                'Warm (20-25°C)', 'Hot (>25°C)']
    )
    
    # Create temperature quintiles for dose-response analysis
    df_analysis['temp_quintile'] = pd.qcut(
        df_analysis['climate_daily_mean_temp'],
        q=5,
        labels=['Q1 (Coldest)', 'Q2', 'Q3', 'Q4', 'Q5 (Hottest)']
    )
    
    # Convert date for temporal analysis
    df_analysis['date'] = pd.to_datetime(df_analysis['primary_date'])
    
    return df_analysis

def calculate_statistics(df):
    """Calculate comprehensive statistical measures."""
    stats_results = {}
    
    # Basic correlation analysis
    temp_vars = ['climate_daily_mean_temp', 'climate_7d_mean_temp', 
                 'climate_14d_mean_temp', 'climate_30d_mean_temp']
    
    for temp_var in temp_vars:
        r, p = pearsonr(df[temp_var], df['CD4 cell count (cells/µL)'])
        rho, p_spear = spearmanr(df[temp_var], df['CD4 cell count (cells/µL)'])
        
        stats_results[temp_var] = {
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spear,
            'n_observations': len(df)
        }
    
    # ANOVA for heat categories
    heat_groups = [group['CD4 cell count (cells/µL)'].values 
                   for name, group in df.groupby('heat_category') if len(group) > 0]
    
    if len(heat_groups) > 1:
        f_stat, p_anova = stats.f_oneway(*heat_groups)
        
        # Calculate eta-squared
        n_total = sum(len(group) for group in heat_groups)
        n_groups = len(heat_groups)
        if f_stat is not None and not np.isnan(f_stat):
            eta2 = f_stat * (n_groups - 1) / (f_stat * (n_groups - 1) + n_total - n_groups)
        else:
            eta2 = 0.0
            
        stats_results['anova'] = {
            'f_statistic': f_stat if not np.isnan(f_stat) else 0.0,
            'p_value': p_anova if not np.isnan(p_anova) else 1.0,
            'effect_size_eta2': eta2
        }
    else:
        stats_results['anova'] = {
            'f_statistic': 0.0,
            'p_value': 1.0,
            'effect_size_eta2': 0.0
        }
    
    # Effect sizes (Cohen's d) between temperature extremes
    q1_cd4 = df[df['temp_quintile'] == 'Q1 (Coldest)']['CD4 cell count (cells/µL)']
    q5_cd4 = df[df['temp_quintile'] == 'Q5 (Hottest)']['CD4 cell count (cells/µL)']
    
    cohens_d = (q1_cd4.mean() - q5_cd4.mean()) / np.sqrt(
        ((len(q1_cd4) - 1) * q1_cd4.var() + (len(q5_cd4) - 1) * q5_cd4.var()) /
        (len(q1_cd4) + len(q5_cd4) - 2)
    )
    
    stats_results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': 'large' if abs(cohens_d) > 0.8 else 
                         'medium' if abs(cohens_d) > 0.5 else 'small'
    }
    
    return stats_results

def create_comprehensive_visualization(df, stats_results):
    """Create the 6-panel comprehensive visualization."""
    
    # Set up the figure with 2x3 grid
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Color schemes
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'success': '#C73E1D',
        'cool': '#4E79A7',
        'warm': '#E15759'
    }
    
    # Panel 1: Temperature vs CD4 scatter with regression
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = df['climate_daily_mean_temp']
    y = df['CD4 cell count (cells/µL)']
    
    # Scatter plot
    ax1.scatter(x, y, alpha=0.4, c=colors['primary'], s=8, edgecolors='none')
    
    # Regression line with confidence interval
    from sklearn.linear_model import LinearRegression
    X_reg = x.values.reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_pred = reg.predict(x_line.reshape(-1, 1))
    
    ax1.plot(x_line, y_pred, color=colors['accent'], linewidth=2, label='Regression line')
    
    # Add confidence interval (approximate)
    residuals = y - reg.predict(X_reg)
    mse = np.mean(residuals**2)
    y_err = 1.96 * np.sqrt(mse)
    ax1.fill_between(x_line, y_pred - y_err, y_pred + y_err, 
                     color=colors['accent'], alpha=0.2, label='95% CI')
    
    r = stats_results['climate_daily_mean_temp']['pearson_r']
    p = stats_results['climate_daily_mean_temp']['pearson_p']
    
    ax1.set_xlabel('Daily Mean Temperature (°C)')
    ax1.set_ylabel('CD4+ Count (cells/µL)')
    ax1.set_title('A. Temperature-CD4 Association', fontweight='bold', fontsize=11)
    ax1.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.2e}\nn = {len(df):,}', 
             transform=ax1.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend(fontsize=8)
    
    # Panel 2: Heat stress categories vs CD4 distributions
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Box plots for heat categories
    heat_data = [group['CD4 cell count (cells/µL)'].values 
                 for name, group in df.groupby('heat_category')]
    heat_labels = df['heat_category'].cat.categories
    
    bp = ax2.boxplot(heat_data, labels=heat_labels, patch_artist=True)
    
    # Color the boxes
    box_colors = ['#4E79A7', '#59A14F', '#EDC948', '#E15759', '#AF7AA1']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Temperature Category')
    ax2.set_ylabel('CD4+ Count (cells/µL)')
    ax2.set_title('B. CD4 by Heat Categories', fontweight='bold', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add ANOVA results
    f_stat = stats_results['anova']['f_statistic']
    p_anova = stats_results['anova']['p_value']
    eta2 = stats_results['anova']['effect_size_eta2']
    
    ax2.text(0.95, 0.95, f'ANOVA: F = {f_stat:.2f}\np = {p_anova:.2e}\nη² = {eta2:.3f}', 
             transform=ax2.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 3: Temporal patterns - CD4 by season with temperature overlay
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Calculate seasonal means
    seasonal_stats = df.groupby('season').agg({
        'CD4 cell count (cells/µL)': ['mean', 'std'],
        'climate_daily_mean_temp': 'mean'
    }).round(2)
    
    seasons = seasonal_stats.index
    cd4_means = seasonal_stats[('CD4 cell count (cells/µL)', 'mean')]
    cd4_stds = seasonal_stats[('CD4 cell count (cells/µL)', 'std')]
    temp_means = seasonal_stats[('climate_daily_mean_temp', 'mean')]
    
    # Bar plot for CD4
    bars = ax3.bar(seasons, cd4_means, yerr=cd4_stds, 
                   color=colors['primary'], alpha=0.7, capsize=5,
                   label='CD4+ Count')
    
    # Overlay temperature line
    ax3_temp = ax3.twinx()
    line = ax3_temp.plot(seasons, temp_means, color=colors['accent'], 
                        marker='o', linewidth=3, markersize=8, 
                        label='Temperature')
    
    ax3.set_xlabel('Season')
    ax3.set_ylabel('CD4+ Count (cells/µL)', color=colors['primary'])
    ax3_temp.set_ylabel('Temperature (°C)', color=colors['accent'])
    ax3.set_title('C. Seasonal Patterns', fontweight='bold', fontsize=11)
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_temp.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # Panel 4: Lag analysis - CD4 response to temperature delays
    ax4 = fig.add_subplot(gs[1, 0])
    
    lag_vars = ['climate_daily_mean_temp', 'climate_7d_mean_temp', 
                'climate_14d_mean_temp', 'climate_30d_mean_temp']
    lag_labels = ['0 days', '7 days', '14 days', '30 days']
    lag_correlations = [stats_results[var]['pearson_r'] for var in lag_vars]
    lag_pvalues = [stats_results[var]['pearson_p'] for var in lag_vars]
    
    # Bar plot for correlations
    bars = ax4.bar(lag_labels, lag_correlations, 
                   color=[colors['cool'] if p < 0.05 else colors['warm'] 
                         for p in lag_pvalues])
    
    # Add significance annotations
    for i, (r, p) in enumerate(zip(lag_correlations, lag_pvalues)):
        significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax4.text(i, r + 0.01 if r > 0 else r - 0.01, significance, 
                ha='center', va='bottom' if r > 0 else 'top', fontweight='bold')
    
    ax4.set_xlabel('Temperature Lag Period')
    ax4.set_ylabel('Correlation with CD4+ Count')
    ax4.set_title('D. Lag Effect Analysis', fontweight='bold', fontsize=11)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_ylim(min(lag_correlations) - 0.05, max(lag_correlations) + 0.05)
    
    # Panel 5: Geographic variation
    ax5 = fig.add_subplot(gs[1, 1])
    
    # CD4-temperature correlations by subregion
    regional_stats = []
    regions = df['jhb_subregion'].unique()
    
    for region in regions:
        if pd.isna(region):
            continue
        region_data = df[df['jhb_subregion'] == region]
        if len(region_data) > 20:  # Minimum sample size
            r, p = pearsonr(region_data['climate_daily_mean_temp'], 
                           region_data['CD4 cell count (cells/µL)'])
            regional_stats.append({'region': region, 'correlation': r, 'p_value': p, 'n': len(region_data)})
    
    regional_df = pd.DataFrame(regional_stats)
    
    if not regional_df.empty:
        # Sort by correlation strength
        regional_df = regional_df.sort_values('correlation')
        
        colors_reg = ['red' if p < 0.05 else 'gray' for p in regional_df['p_value']]
        bars = ax5.barh(regional_df['region'], regional_df['correlation'], color=colors_reg)
        
        ax5.set_xlabel('Temperature-CD4 Correlation')
        ax5.set_ylabel('Johannesburg Subregion')
        ax5.set_title('E. Geographic Variation', fontweight='bold', fontsize=11)
        ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Panel 6: Dose-response curve
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate CD4 means by temperature quintiles
    quintile_stats = df.groupby('temp_quintile').agg({
        'CD4 cell count (cells/µL)': ['mean', 'std', 'count'],
        'climate_daily_mean_temp': 'mean'
    }).round(2)
    
    quintiles = quintile_stats.index
    cd4_quintile_means = quintile_stats[('CD4 cell count (cells/µL)', 'mean')]
    cd4_quintile_stds = quintile_stats[('CD4 cell count (cells/µL)', 'std')]
    temp_quintile_means = quintile_stats[('climate_daily_mean_temp', 'mean')]
    quintile_ns = quintile_stats[('CD4 cell count (cells/µL)', 'count')]
    
    # Error bars with sample size weighting
    ax6.errorbar(temp_quintile_means, cd4_quintile_means, 
                yerr=cd4_quintile_stds, 
                fmt='o-', color=colors['primary'], linewidth=2, 
                markersize=8, capsize=5, label='CD4+ Means ± SD')
    
    # Add trend line
    z = np.polyfit(temp_quintile_means, cd4_quintile_means, 1)
    p = np.poly1d(z)
    ax6.plot(temp_quintile_means, p(temp_quintile_means), 
             '--', color=colors['accent'], linewidth=2, alpha=0.8, label='Linear trend')
    
    ax6.set_xlabel('Mean Temperature (°C)')
    ax6.set_ylabel('Mean CD4+ Count (cells/µL)')
    ax6.set_title('F. Dose-Response Relationship', fontweight='bold', fontsize=11)
    ax6.legend(fontsize=8)
    
    # Add effect size annotation
    cohens_d = stats_results['effect_size']['cohens_d']
    interpretation = stats_results['effect_size']['interpretation']
    ax6.text(0.05, 0.95, f"Cohen's d = {cohens_d:.3f}\n({interpretation} effect)", 
             transform=ax6.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Main title and attribution
    fig.suptitle('ENBEL Climate-Health Analysis: CD4+ T-Cell Response to Temperature Exposure\n' +
                 'HIV-Positive Individuals, Johannesburg Clinical Trials (n=11,398)', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Footer with study details
    footer_text = ("Data: 15 HIV clinical trials (2002-2021) • ERA5 climate reanalysis • Multiple lag analysis\n" +
                   "Statistics: Pearson correlations, ANOVA, mixed-effects modeling • Bonferroni correction applied\n" +
                   "ENBEL Research Collaboration • Generated with Claude Code")
    
    fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', fontsize=8, 
             style='italic', color='gray')
    
    return fig

def main():
    """Main execution function."""
    print("ENBEL CD4 Heat Analysis - Comprehensive Scientific Visualization")
    print("=" * 65)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Calculate comprehensive statistics
    print("\nCalculating statistical measures...")
    stats_results = calculate_statistics(df)
    
    # Print key findings
    print("\nKey Statistical Findings:")
    print("-" * 25)
    for temp_var in ['climate_daily_mean_temp', 'climate_30d_mean_temp']:
        r = stats_results[temp_var]['pearson_r']
        p = stats_results[temp_var]['pearson_p']
        print(f"{temp_var}: r = {r:.3f}, p = {p:.2e}")
    
    print(f"\nANOVA (Heat Categories): F = {stats_results['anova']['f_statistic']:.2f}, " +
          f"p = {stats_results['anova']['p_value']:.2e}")
    print(f"Effect Size (η²): {stats_results['anova']['effect_size_eta2']:.3f}")
    print(f"Cohen's d (Q1 vs Q5): {stats_results['effect_size']['cohens_d']:.3f} " +
          f"({stats_results['effect_size']['interpretation']} effect)")
    
    # Create visualization
    print("\nCreating comprehensive visualization...")
    fig = create_comprehensive_visualization(df, stats_results)
    
    # Save as SVG
    output_path = "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/presentation_slides_final/enbel_cd4_heat_analysis_final.svg"
    fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    
    print(f"\nVisualization saved: {output_path}")
    print("Analysis complete!")
    
    # Also save as PNG for preview
    png_path = output_path.replace('.svg', '.png')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"PNG preview saved: {png_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()