#!/usr/bin/env python3
"""
Enhanced Climate-Health Discovery Analysis
=========================================

Implementation of multiple alternative strategies to discover additional
climate-health relationships beyond single-biomarker approaches.

STRATEGIES IMPLEMENTED:
1. Composite health indices (cardiovascular, metabolic, immune)
2. Subpopulation analysis (age, sex, race stratification) 
3. Interaction effects (climate × demographics)
4. Temporal patterns (seasonal, long-term trends)
5. Multi-output modeling
6. Alternative feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import json
import time
from datetime import datetime
import logging
import warnings
from pathlib import Path
import joblib
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedDiscoveryAnalyzer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("enhanced_discovery")
        self.results_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"enhanced_discovery_{self.timestamp}.log"
        
        self.discoveries = {}
        
    def log_progress(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "🔍", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "DISCOVERY": "🎯"}
        icon = icons.get(level, "🔍")
        
        progress_msg = f"[{timestamp}] {icon} {message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def load_and_prepare_comprehensive_data(self):
        """Load data with comprehensive feature preparation"""
        self.log_progress("Loading comprehensive dataset for discovery analysis...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Identify all available biomarkers
        biomarker_candidates = [
            'CD4 cell count (cells/µL)', 'Creatinine (mg/dL)', 'Hemoglobin (g/dL)',
            'systolic blood pressure', 'diastolic blood pressure',
            'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING LDL'
        ]
        
        available_biomarkers = [b for b in biomarker_candidates if b in df.columns]
        
        # Get climate features
        climate_features = []
        for col in df.columns:
            if any(term in col.lower() for term in [
                'temp', 'heat', 'humid', 'pressure', 'wind', 'solar', 'precip', 'utci'
            ]) and 'lag' in col.lower():
                climate_features.append(col)
        
        # Get demographic features
        demographic_features = ['Sex', 'Race', 'Age', 'year', 'month', 'season', 'latitude', 'longitude']
        available_demographics = [d for d in demographic_features if d in df.columns]
        
        self.log_progress(f"Data loaded: {len(df)} records, {len(available_biomarkers)} biomarkers, {len(climate_features)} climate features")
        
        return df, available_biomarkers, climate_features, available_demographics

    def create_composite_health_indices(self, df, biomarkers):
        """Create composite health indices from multiple biomarkers"""
        self.log_progress("Creating composite health indices...", "DISCOVERY")
        
        composite_indices = {}
        
        # 1. CARDIOVASCULAR COMPOSITE
        cardio_markers = ['systolic blood pressure', 'diastolic blood pressure']
        available_cardio = [m for m in cardio_markers if m in biomarkers and m in df.columns]
        
        if len(available_cardio) >= 2:
            # Create cardiovascular risk index (higher = worse)
            cardio_data = df[available_cardio].copy()
            
            # Standardize each component
            cardio_standardized = (cardio_data - cardio_data.mean()) / cardio_data.std()
            
            # Simple average (could be weighted by clinical importance)
            df['cardiovascular_composite'] = cardio_standardized.mean(axis=1)
            composite_indices['cardiovascular_composite'] = {
                'components': available_cardio,
                'description': 'Standardized cardiovascular risk composite'
            }
        
        # 2. METABOLIC COMPOSITE  
        metabolic_markers = ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING LDL']
        available_metabolic = [m for m in metabolic_markers if m in biomarkers and m in df.columns]
        
        if len(available_metabolic) >= 2:
            metabolic_data = df[available_metabolic].copy()
            
            # For HDL, higher is better, so invert
            if 'FASTING HDL' in available_metabolic:
                metabolic_data['FASTING HDL'] = -metabolic_data['FASTING HDL']
            
            metabolic_standardized = (metabolic_data - metabolic_data.mean()) / metabolic_data.std()
            df['metabolic_composite'] = metabolic_standardized.mean(axis=1)
            composite_indices['metabolic_composite'] = {
                'components': available_metabolic,
                'description': 'Standardized metabolic dysfunction composite'
            }
        
        # 3. IMMUNE COMPOSITE (if we have multiple immune markers)
        immune_markers = ['CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)']
        available_immune = [m for m in immune_markers if m in biomarkers and m in df.columns]
        
        if len(available_immune) >= 2:
            immune_data = df[available_immune].copy()
            # Higher values are generally better for immune markers
            immune_standardized = (immune_data - immune_data.mean()) / immune_data.std()
            df['immune_composite'] = immune_standardized.mean(axis=1)
            composite_indices['immune_composite'] = {
                'components': available_immune,
                'description': 'Standardized immune function composite'
            }
        
        self.log_progress(f"Created {len(composite_indices)} composite indices")
        return df, composite_indices

    def analyze_subpopulation_effects(self, df, biomarkers, climate_features, demographics):
        """Analyze climate-health relationships within demographic subpopulations"""
        self.log_progress("Analyzing subpopulation-specific effects...", "DISCOVERY")
        
        subpop_discoveries = {}
        
        # Focus on biomarkers with sufficient data
        target_biomarkers = []
        for biomarker in biomarkers:
            if df[biomarker].notna().sum() >= 500:  # Minimum sample size
                target_biomarkers.append(biomarker)
        
        # Define subpopulations
        subpopulations = {}
        
        # Age groups
        if 'Age' in demographics and 'Age' in df.columns:
            df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Older'])
            subpopulations['age_group'] = ['Young', 'Middle', 'Older']
        
        # Sex
        if 'Sex' in demographics and 'Sex' in df.columns:
            subpopulations['Sex'] = df['Sex'].dropna().unique()
        
        # Race
        if 'Race' in demographics and 'Race' in df.columns:
            race_counts = df['Race'].value_counts()
            # Only include races with sufficient sample size
            sufficient_races = race_counts[race_counts >= 200].index.tolist()
            if len(sufficient_races) >= 2:
                subpopulations['Race'] = sufficient_races
        
        # Season
        if 'season' in demographics and 'season' in df.columns:
            subpopulations['season'] = df['season'].dropna().unique()
        
        self.log_progress(f"Analyzing {len(subpopulations)} demographic stratifications")
        
        # Test each subpopulation
        for demo_var, groups in subpopulations.items():
            for group in groups:
                if demo_var == 'age_group':
                    subpop_data = df[df['age_group'] == group]
                else:
                    subpop_data = df[df[demo_var] == group]
                
                if len(subpop_data) < 100:  # Skip small subgroups
                    continue
                
                # Test each biomarker in this subpopulation
                for biomarker in target_biomarkers:
                    biomarker_data = subpop_data.dropna(subset=[biomarker])
                    
                    if len(biomarker_data) < 100:
                        continue
                    
                    # Get available climate features
                    available_climate = [f for f in climate_features[:50] if f in biomarker_data.columns]  # Top 50 features
                    if len(available_climate) < 10:
                        continue
                    
                    try:
                        X = biomarker_data[available_climate]
                        y = biomarker_data[biomarker]
                        
                        # Fill missing values
                        X = X.fillna(X.median())
                        
                        # Quick model test
                        if len(X) >= 100:
                            model = ElasticNet(random_state=42, max_iter=1000)
                            scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                            mean_r2 = np.mean(scores)
                            
                            if mean_r2 > 0.10:  # Threshold for interesting relationships
                                subpop_key = f"{biomarker}_in_{demo_var}_{group}"
                                subpop_discoveries[subpop_key] = {
                                    'biomarker': biomarker,
                                    'demographic': demo_var,
                                    'subgroup': group,
                                    'r2': mean_r2,
                                    'r2_std': np.std(scores),
                                    'n_samples': len(X),
                                    'n_features': len(available_climate)
                                }
                                
                                self.log_progress(f"DISCOVERY: {biomarker} in {demo_var}={group}: R² = {mean_r2:.3f} (n={len(X)})", "DISCOVERY")
                    
                    except Exception as e:
                        continue
        
        return subpop_discoveries

    def analyze_interaction_effects(self, df, biomarkers, climate_features, demographics):
        """Analyze climate × demographic interaction effects"""
        self.log_progress("Analyzing interaction effects...", "DISCOVERY")
        
        interaction_discoveries = {}
        
        # Focus on promising biomarkers
        target_biomarkers = ['systolic blood pressure', 'FASTING GLUCOSE', 'CD4 cell count (cells/µL)']
        available_targets = [b for b in target_biomarkers if b in biomarkers and df[b].notna().sum() >= 1000]
        
        # Focus on key demographics and climate variables
        key_demographics = []
        if 'Sex' in df.columns and df['Sex'].notna().sum() > 1000:
            key_demographics.append('Sex')
        if 'Race' in df.columns and df['Race'].notna().sum() > 1000:
            key_demographics.append('Race')
        
        # Get top climate features (temperature-related for interpretability)
        temp_features = [f for f in climate_features if 'temp' in f.lower()][:10]
        
        for biomarker in available_targets:
            biomarker_data = df.dropna(subset=[biomarker])
            
            for demo in key_demographics:
                demo_data = biomarker_data.dropna(subset=[demo])
                
                if len(demo_data) < 500:
                    continue
                
                # Encode demographic variable
                if demo_data[demo].dtype == 'object':
                    le = LabelEncoder()
                    demo_encoded = le.fit_transform(demo_data[demo])
                else:
                    demo_encoded = demo_data[demo].values
                
                # Get available temperature features
                available_temp = [f for f in temp_features if f in demo_data.columns]
                if len(available_temp) < 3:
                    continue
                
                try:
                    # Create interaction features (Climate × Demographic)
                    X_climate = demo_data[available_temp].fillna(demo_data[available_temp].median())
                    
                    # Add main effects
                    X_main = X_climate.copy()
                    X_main[f'{demo}_encoded'] = demo_encoded
                    
                    # Add interaction effects
                    X_interact = X_main.copy()
                    for temp_col in available_temp[:5]:  # Top 5 to avoid too many features
                        X_interact[f'{temp_col}_x_{demo}'] = X_climate[temp_col] * demo_encoded
                    
                    y = demo_data[biomarker]
                    
                    # Compare models with and without interactions
                    model_main = ElasticNet(random_state=42, max_iter=1000)
                    model_interact = ElasticNet(random_state=42, max_iter=1000)
                    
                    scores_main = cross_val_score(model_main, X_main, y, cv=3, scoring='r2')
                    scores_interact = cross_val_score(model_interact, X_interact, y, cv=3, scoring='r2')
                    
                    r2_main = np.mean(scores_main)
                    r2_interact = np.mean(scores_interact)
                    improvement = r2_interact - r2_main
                    
                    if r2_interact > 0.10 and improvement > 0.02:  # Meaningful interaction
                        interaction_key = f"{biomarker}_x_{demo}_interaction"
                        interaction_discoveries[interaction_key] = {
                            'biomarker': biomarker,
                            'demographic': demo,
                            'r2_main_effects': r2_main,
                            'r2_with_interactions': r2_interact,
                            'interaction_improvement': improvement,
                            'n_samples': len(X_interact),
                            'n_features': len(X_interact.columns)
                        }
                        
                        self.log_progress(f"INTERACTION: {biomarker} × {demo}: R² = {r2_interact:.3f} (+{improvement:.3f} from interactions)", "DISCOVERY")
                
                except Exception as e:
                    continue
        
        return interaction_discoveries

    def analyze_temporal_patterns(self, df, biomarkers, climate_features):
        """Analyze temporal patterns and seasonal effects"""
        self.log_progress("Analyzing temporal patterns...", "DISCOVERY")
        
        temporal_discoveries = {}
        
        # Focus on biomarkers with good temporal coverage
        target_biomarkers = []
        for biomarker in biomarkers:
            if df[biomarker].notna().sum() >= 1000:
                target_biomarkers.append(biomarker)
        
        for biomarker in target_biomarkers:
            biomarker_data = df.dropna(subset=[biomarker])
            
            # 1. SEASONAL ANALYSIS
            if 'season' in df.columns and 'month' in df.columns:
                # Test if biomarker varies significantly by season
                seasonal_data = biomarker_data.groupby('season')[biomarker].agg(['mean', 'std', 'count'])
                
                if len(seasonal_data) >= 3:  # At least 3 seasons
                    seasonal_variance = seasonal_data['mean'].var()
                    within_variance = (seasonal_data['std'] ** 2).mean()
                    
                    if seasonal_variance > within_variance * 0.1:  # Meaningful seasonal variation
                        temporal_discoveries[f"{biomarker}_seasonal_variation"] = {
                            'biomarker': biomarker,
                            'pattern_type': 'seasonal',
                            'seasonal_variance': seasonal_variance,
                            'within_variance': within_variance,
                            'seasonal_effect_size': seasonal_variance / within_variance,
                            'seasonal_means': seasonal_data['mean'].to_dict()
                        }
                        
                        self.log_progress(f"SEASONAL: {biomarker} shows seasonal variation (effect size: {seasonal_variance/within_variance:.2f})", "DISCOVERY")
            
            # 2. LONG-TERM CLIMATE EXPOSURE
            if 'year' in df.columns and len(biomarker_data['year'].unique()) >= 3:
                # Create long-term climate exposure features (annual averages)
                temp_features = [f for f in climate_features if 'temp' in f.lower() and 'lag0' in f.lower()][:5]
                available_temp = [f for f in temp_features if f in biomarker_data.columns]
                
                if len(available_temp) >= 2:
                    # Calculate annual climate averages
                    annual_climate = biomarker_data.groupby('year')[available_temp].mean()
                    annual_biomarker = biomarker_data.groupby('year')[biomarker].mean()
                    
                    # Merge for analysis
                    annual_data = annual_climate.join(annual_biomarker).dropna()
                    
                    if len(annual_data) >= 5:  # At least 5 years
                        try:
                            X_annual = annual_data[available_temp]
                            y_annual = annual_data[biomarker]
                            
                            model = ElasticNet(random_state=42)
                            model.fit(X_annual, y_annual)
                            r2_annual = model.score(X_annual, y_annual)
                            
                            if r2_annual > 0.15:  # Meaningful long-term relationship
                                temporal_discoveries[f"{biomarker}_longterm_climate"] = {
                                    'biomarker': biomarker,
                                    'pattern_type': 'long_term_annual',
                                    'r2': r2_annual,
                                    'n_years': len(annual_data),
                                    'climate_features': available_temp
                                }
                                
                                self.log_progress(f"LONG-TERM: {biomarker} shows long-term climate relationship (R² = {r2_annual:.3f})", "DISCOVERY")
                        
                        except Exception:
                            continue
        
        return temporal_discoveries

    def multi_output_analysis(self, df, biomarkers, climate_features):
        """Multi-output modeling of related biomarkers"""
        self.log_progress("Analyzing multi-output relationships...", "DISCOVERY")
        
        multi_output_discoveries = {}
        
        # Define related biomarker groups
        biomarker_groups = {
            'cardiovascular': ['systolic blood pressure', 'diastolic blood pressure'],
            'metabolic': ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING LDL'],
            'renal_immune': ['Creatinine (mg/dL)', 'CD4 cell count (cells/µL)']
        }
        
        for group_name, group_biomarkers in biomarker_groups.items():
            available_group = [b for b in group_biomarkers if b in biomarkers and b in df.columns]
            
            if len(available_group) >= 2:
                # Get data with all biomarkers in group
                group_data = df.dropna(subset=available_group)
                
                if len(group_data) >= 200:  # Minimum sample size
                    # Get climate features
                    available_climate = [f for f in climate_features[:30] if f in group_data.columns]
                    
                    if len(available_climate) >= 10:
                        try:
                            X = group_data[available_climate].fillna(group_data[available_climate].median())
                            y = group_data[available_group]
                            
                            # Multi-output model
                            multi_model = MultiOutputRegressor(ElasticNet(random_state=42, max_iter=1000))
                            
                            # Cross-validation for multi-output
                            scores = cross_val_score(multi_model, X, y, cv=3, scoring='r2')
                            mean_r2 = np.mean(scores)
                            
                            if mean_r2 > 0.05:  # Threshold for multi-output
                                multi_output_discoveries[f"{group_name}_multioutput"] = {
                                    'group_name': group_name,
                                    'biomarkers': available_group,
                                    'r2': mean_r2,
                                    'r2_std': np.std(scores),
                                    'n_samples': len(X),
                                    'n_features': len(available_climate),
                                    'n_outputs': len(available_group)
                                }
                                
                                self.log_progress(f"MULTI-OUTPUT: {group_name} group (R² = {mean_r2:.3f}, {len(available_group)} biomarkers)", "DISCOVERY")
                        
                        except Exception:
                            continue
        
        return multi_output_discoveries

    def run_enhanced_discovery_analysis(self):
        """Execute comprehensive enhanced discovery analysis"""
        self.log_progress("="*80)
        self.log_progress("🎯 ENHANCED CLIMATE-HEALTH DISCOVERY ANALYSIS")
        self.log_progress("Multiple strategies for extracting additional insights")
        self.log_progress("="*80)
        
        start_time = time.time()
        
        # Load comprehensive data
        df, biomarkers, climate_features, demographics = self.load_and_prepare_comprehensive_data()
        
        # Strategy 1: Composite Health Indices
        self.log_progress("\n🎯 STRATEGY 1: Composite Health Indices")
        df, composite_indices = self.create_composite_health_indices(df, biomarkers)
        
        # Test composite indices
        composite_results = {}
        for composite_name in composite_indices.keys():
            if composite_name in df.columns:
                composite_data = df.dropna(subset=[composite_name])
                if len(composite_data) >= 500:
                    available_climate = [f for f in climate_features[:30] if f in composite_data.columns]
                    
                    if len(available_climate) >= 10:
                        try:
                            X = composite_data[available_climate].fillna(composite_data[available_climate].median())
                            y = composite_data[composite_name]
                            
                            model = ElasticNet(random_state=42, max_iter=1000)
                            scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                            mean_r2 = np.mean(scores)
                            
                            if mean_r2 > 0.05:
                                composite_results[composite_name] = {
                                    'r2': mean_r2,
                                    'r2_std': np.std(scores),
                                    'n_samples': len(X),
                                    'components': composite_indices[composite_name]['components']
                                }
                                self.log_progress(f"COMPOSITE: {composite_name} (R² = {mean_r2:.3f})", "DISCOVERY")
                        except:
                            continue
        
        self.discoveries['composite_indices'] = composite_results
        
        # Strategy 2: Subpopulation Analysis
        self.log_progress("\n🎯 STRATEGY 2: Subpopulation Analysis")
        subpop_discoveries = self.analyze_subpopulation_effects(df, biomarkers, climate_features, demographics)
        self.discoveries['subpopulation_effects'] = subpop_discoveries
        
        # Strategy 3: Interaction Effects
        self.log_progress("\n🎯 STRATEGY 3: Interaction Effects")
        interaction_discoveries = self.analyze_interaction_effects(df, biomarkers, climate_features, demographics)
        self.discoveries['interaction_effects'] = interaction_discoveries
        
        # Strategy 4: Temporal Patterns
        self.log_progress("\n🎯 STRATEGY 4: Temporal Patterns")
        temporal_discoveries = self.analyze_temporal_patterns(df, biomarkers, climate_features)
        self.discoveries['temporal_patterns'] = temporal_discoveries
        
        # Strategy 5: Multi-output Analysis
        self.log_progress("\n🎯 STRATEGY 5: Multi-output Analysis")
        multi_output_discoveries = self.multi_output_analysis(df, biomarkers, climate_features)
        self.discoveries['multi_output'] = multi_output_discoveries
        
        # Compile final results
        elapsed_time = time.time() - start_time
        
        final_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'analysis_type': 'enhanced_discovery_analysis',
                'strategies_implemented': 5,
                'analysis_time_minutes': elapsed_time / 60,
                'total_discoveries': sum(len(disc) for disc in self.discoveries.values())
            },
            'discoveries': self.discoveries
        }
        
        # Save results
        results_file = self.results_dir / f"enhanced_discoveries_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Summary report
        self.log_progress("\n" + "="*80)
        self.log_progress("🎯 ENHANCED DISCOVERY ANALYSIS COMPLETE")
        self.log_progress("="*80)
        
        total_discoveries = sum(len(disc) for disc in self.discoveries.values())
        self.log_progress(f"Total discoveries: {total_discoveries}")
        self.log_progress(f"Analysis time: {elapsed_time/60:.1f} minutes")
        self.log_progress("")
        
        # Summary by strategy
        for strategy, discoveries in self.discoveries.items():
            if discoveries:
                self.log_progress(f"{strategy.replace('_', ' ').title()}: {len(discoveries)} discoveries")
                
                # Show top discoveries for each strategy
                if isinstance(discoveries, dict):
                    sorted_discoveries = sorted(discoveries.items(), 
                                              key=lambda x: x[1].get('r2', x[1].get('r2_main_effects', 0)), 
                                              reverse=True)
                    
                    for i, (name, data) in enumerate(sorted_discoveries[:3]):  # Top 3
                        r2_value = data.get('r2', data.get('r2_with_interactions', data.get('seasonal_effect_size', 0)))
                        self.log_progress(f"  {i+1}. {name}: R² = {r2_value:.3f}")
        
        self.log_progress(f"\n✅ Results saved to: {results_file}")
        
        return final_results

def main():
    """Execute enhanced discovery analysis"""
    analyzer = EnhancedDiscoveryAnalyzer()
    results = analyzer.run_enhanced_discovery_analysis()
    return results

if __name__ == "__main__":
    main()