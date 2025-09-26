#!/usr/bin/env python3
"""
Flexible Climate-Health Discovery Framework
===========================================

Multiple approaches to find climate-health insights:
1. Separate cohort analyses with flexible targets
2. Ecological aggregation (following Labib et al.)
3. Unsupervised learning to discover patterns
4. Climate as target (reverse the typical approach)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import warnings
import json
from datetime import datetime
from pathlib import Path
import logging

warnings.filterwarnings('ignore')


class FlexibleClimateHealthDiscovery:
    """
    Flexible framework for discovering climate-health relationships
    """
    
    def __init__(self, results_dir="flexible_discovery_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'approaches': {}
        }
    
    def load_and_prepare_data(self, filepath):
        """Load data and separate cohorts"""
        
        self.logger.info("📊 Loading and preparing data with flexible approach")
        
        df = pd.read_csv(filepath, low_memory=False)
        
        # Separate cohorts
        if 'dataset_group' in df.columns:
            self.clinical_df = df[df['dataset_group'] == 'clinical'].copy()
            self.socioeconomic_df = df[df['dataset_group'] == 'socioeconomic'].copy()
        else:
            # Use biomarker presence
            has_biomarkers = df[['FASTING GLUCOSE', 'systolic blood pressure']].notna().any(axis=1)
            self.clinical_df = df[has_biomarkers].copy()
            self.socioeconomic_df = df[~has_biomarkers].copy()
        
        self.logger.info(f"Clinical cohort: {len(self.clinical_df)} participants")
        self.logger.info(f"Socioeconomic cohort: {len(self.socioeconomic_df)} participants")
        
        # Store full dataset for ecological analysis
        self.full_df = df
        
        return self.clinical_df, self.socioeconomic_df
    
    # ==========================================
    # APPROACH 1A: Clinical Cohort - Multiple Targets
    # ==========================================
    
    def approach_1a_clinical_flexible_targets(self):
        """Analyze clinical cohort with various target configurations"""
        
        self.logger.info("\n🔬 APPROACH 1A: Clinical Cohort - Flexible Targets")
        self.logger.info("="*50)
        
        results = {}
        
        # Configuration 1: Biomarkers as targets (traditional)
        self.logger.info("\n1. Traditional: Climate → Biomarkers")
        biomarker_results = self._analyze_biomarkers_as_targets()
        results['biomarkers_as_targets'] = biomarker_results
        
        # Configuration 2: Climate extremes as targets (reverse)
        self.logger.info("\n2. Reverse: Biomarkers → Climate Vulnerability")
        climate_results = self._analyze_climate_as_target()
        results['climate_as_target'] = climate_results
        
        # Configuration 3: Create composite health risk score
        self.logger.info("\n3. Composite: Climate → Health Risk Score")
        composite_results = self._analyze_composite_health_score()
        results['composite_health_score'] = composite_results
        
        # Configuration 4: Unsupervised clustering
        self.logger.info("\n4. Unsupervised: Climate-Health Clusters")
        cluster_results = self._discover_clinical_clusters()
        results['clustering'] = cluster_results
        
        self.results['approaches']['clinical_flexible'] = results
        return results
    
    def _analyze_biomarkers_as_targets(self):
        """Traditional approach with biomarkers as targets"""
        
        biomarkers = ['systolic blood pressure', 'diastolic blood pressure', 
                     'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL']
        
        climate_features = ['temperature', 'humidity', 'heat_index', 'apparent_temp']
        lag_features = [col for col in self.clinical_df.columns if 'lag' in col.lower()][:15]
        
        results = {}
        
        for biomarker in biomarkers:
            if biomarker not in self.clinical_df.columns:
                continue
            
            valid_data = self.clinical_df[self.clinical_df[biomarker].notna()].copy()
            
            if len(valid_data) < 100:
                continue
            
            # Prepare features
            features = [f for f in climate_features + lag_features if f in valid_data.columns]
            
            X = valid_data[features].fillna(valid_data[features].median())
            y = valid_data[biomarker]
            
            # Remove any issues
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X, y = X[mask], y[mask]
            
            if len(X) < 100:
                continue
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            model.fit(X, y)
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results[biomarker] = {
                'n_samples': len(X),
                'r2_mean': scores.mean(),
                'r2_std': scores.std(),
                'top_features': importance.head(5).to_dict('records'),
                'climate_importance_total': importance[importance['feature'].str.contains('temp|humid|heat', case=False, na=False)]['importance'].sum()
            }
            
            self.logger.info(f"  {biomarker}: R²={scores.mean():.3f}, n={len(X)}")
        
        return results
    
    def _analyze_climate_as_target(self):
        """Reverse approach: predict climate vulnerability from health status"""
        
        # Create climate extremes indicator
        self.clinical_df['extreme_heat'] = (
            self.clinical_df['temperature'] > self.clinical_df['temperature'].quantile(0.9)
        ).astype(int)
        
        # Use biomarkers and demographics to predict exposure to extreme heat
        biomarkers = ['systolic blood pressure', 'diastolic blood pressure', 
                     'FASTING GLUCOSE', 'Height', 'weight']
        
        available_biomarkers = [b for b in biomarkers if b in self.clinical_df.columns]
        
        # Prepare data
        valid_data = self.clinical_df[self.clinical_df['extreme_heat'].notna()].copy()
        
        for biomarker in available_biomarkers:
            valid_data[f'{biomarker}_available'] = valid_data[biomarker].notna().astype(int)
            valid_data[biomarker] = valid_data[biomarker].fillna(valid_data[biomarker].median())
        
        features = available_biomarkers + [f'{b}_available' for b in available_biomarkers]
        
        # Add demographics if available
        if 'Sex' in valid_data.columns:
            valid_data['is_male'] = (valid_data['Sex'] == 'Male').astype(int)
            features.append('is_male')
        
        X = valid_data[features].fillna(0)
        y = valid_data['extreme_heat']
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        if len(X) > 100:
            # Classification model
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            model.fit(X, y)
            
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result = {
                'target': 'extreme_heat_exposure',
                'n_samples': len(X),
                'auc_mean': scores.mean(),
                'auc_std': scores.std(),
                'top_predictors': importance.head(5).to_dict('records'),
                'interpretation': 'Health status predicting climate exposure patterns'
            }
            
            self.logger.info(f"  Extreme heat prediction: AUC={scores.mean():.3f}")
            
            return result
        
        return None
    
    def _analyze_composite_health_score(self):
        """Create and analyze composite health risk score"""
        
        # Create composite health score from available biomarkers
        biomarkers = ['systolic blood pressure', 'diastolic blood pressure', 
                     'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL']
        
        available = [b for b in biomarkers if b in self.clinical_df.columns]
        
        if len(available) < 2:
            return None
        
        # Standardize biomarkers and create composite score
        valid_data = self.clinical_df.copy()
        
        for biomarker in available:
            if biomarker in valid_data.columns:
                # Standardize (z-score)
                mean = valid_data[biomarker].mean()
                std = valid_data[biomarker].std()
                valid_data[f'{biomarker}_zscore'] = (valid_data[biomarker] - mean) / std
        
        # Create composite score (average of z-scores)
        zscore_cols = [f'{b}_zscore' for b in available]
        valid_data['health_risk_score'] = valid_data[zscore_cols].mean(axis=1)
        
        # Remove rows with no health score
        valid_data = valid_data[valid_data['health_risk_score'].notna()].copy()
        
        if len(valid_data) < 100:
            return None
        
        # Predict composite score from climate
        climate_features = ['temperature', 'humidity', 'heat_index', 'apparent_temp']
        lag_features = [col for col in valid_data.columns if 'lag' in col.lower()][:10]
        
        features = [f for f in climate_features + lag_features if f in valid_data.columns]
        
        X = valid_data[features].fillna(valid_data[features].median())
        y = valid_data['health_risk_score']
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'target': 'composite_health_risk_score',
            'n_samples': len(X),
            'r2_mean': scores.mean(),
            'r2_std': scores.std(),
            'top_features': importance.head(5).to_dict('records'),
            'biomarkers_included': available
        }
        
        self.logger.info(f"  Composite health score: R²={scores.mean():.3f}")
        
        return result
    
    def _discover_clinical_clusters(self):
        """Unsupervised clustering to discover patterns"""
        
        # Select features for clustering
        features = []
        
        # Add biomarkers
        biomarkers = ['systolic blood pressure', 'diastolic blood pressure', 
                     'FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL']
        features.extend([b for b in biomarkers if b in self.clinical_df.columns])
        
        # Add climate
        climate = ['temperature', 'humidity', 'heat_index']
        features.extend([c for c in climate if c in self.clinical_df.columns])
        
        if len(features) < 3:
            return None
        
        # Prepare data
        cluster_data = self.clinical_df[features].copy()
        
        # Fill missing values with median
        for col in features:
            cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
        
        # Remove any remaining NaN rows
        cluster_data = cluster_data.dropna()
        
        if len(cluster_data) < 100:
            return None
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)
        
        # Find optimal number of clusters (2-8)
        silhouette_scores = []
        for k in range(2, min(9, len(cluster_data)//50)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append((k, score))
        
        # Use best k
        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Characterize clusters
        cluster_profiles = []
        for cluster_id in range(best_k):
            cluster_mask = cluster_data['cluster'] == cluster_id
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'characteristics': {}
            }
            
            for feature in features:
                profile['characteristics'][feature] = {
                    'mean': cluster_data[cluster_mask][feature].mean(),
                    'std': cluster_data[cluster_mask][feature].std()
                }
            
            cluster_profiles.append(profile)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        result = {
            'n_clusters': best_k,
            'silhouette_score': max(silhouette_scores, key=lambda x: x[1])[1],
            'n_samples': len(cluster_data),
            'features_used': features,
            'cluster_profiles': cluster_profiles,
            'pca_variance_explained': pca.explained_variance_ratio_.tolist()
        }
        
        self.logger.info(f"  Found {best_k} distinct climate-health clusters")
        
        return result
    
    # ==========================================
    # APPROACH 1B: Socioeconomic Cohort Analysis
    # ==========================================
    
    def approach_1b_socioeconomic_flexible(self):
        """Analyze socioeconomic cohort with flexible approaches"""
        
        self.logger.info("\n🏘️ APPROACH 1B: Socioeconomic Cohort - Flexible Analysis")
        self.logger.info("="*50)
        
        results = {}
        
        # Configuration 1: Vulnerability indices as targets
        self.logger.info("\n1. Climate → Vulnerability Indices")
        vuln_results = self._analyze_vulnerability_indices()
        results['vulnerability_prediction'] = vuln_results
        
        # Configuration 2: Socioeconomic clustering
        self.logger.info("\n2. Socioeconomic-Climate Clusters")
        socio_clusters = self._discover_socioeconomic_clusters()
        results['socioeconomic_clusters'] = socio_clusters
        
        # Configuration 3: Climate exposure patterns by socioeconomic status
        self.logger.info("\n3. Socioeconomic → Climate Exposure Patterns")
        exposure_results = self._analyze_climate_exposure_by_ses()
        results['climate_exposure_patterns'] = exposure_results
        
        self.results['approaches']['socioeconomic_flexible'] = results
        return results
    
    def _analyze_vulnerability_indices(self):
        """Predict vulnerability indices from climate and socioeconomic factors"""
        
        vulnerability_indices = ['heat_vulnerability_index', 'housing_vulnerability', 
                               'economic_vulnerability']
        
        results = {}
        
        for vuln_index in vulnerability_indices:
            if vuln_index not in self.socioeconomic_df.columns:
                continue
            
            valid_data = self.socioeconomic_df[self.socioeconomic_df[vuln_index].notna()].copy()
            
            if len(valid_data) < 100:
                continue
            
            # Features
            features = []
            
            # Climate features - check for non-null values
            climate = ['temperature', 'humidity', 'heat_index']
            for c in climate:
                if c in valid_data.columns and valid_data[c].notna().sum() > 0:
                    features.append(c)
            
            # Socioeconomic features
            if 'Education' in valid_data.columns:
                valid_data['education_encoded'] = pd.Categorical(valid_data['Education']).codes
                features.append('education_encoded')
            
            if 'employment_status' in valid_data.columns:
                valid_data['employment_encoded'] = pd.Categorical(valid_data['employment_status']).codes
                features.append('employment_encoded')
            
            # Geographic features
            if 'latitude' in valid_data.columns:
                features.extend(['latitude', 'longitude'])
            
            if len(features) < 2:
                continue
            
            X = valid_data[features].fillna(valid_data[features].median())
            y = valid_data[vuln_index]
            
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X, y = X[mask], y[mask]
            
            if len(X) < 10:
                continue
            
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            
            # Adjust CV folds based on sample size
            n_folds = min(5, len(X))
            if n_folds < 2:
                continue
            
            scores = cross_val_score(model, X, y, cv=n_folds, scoring='r2')
            
            model.fit(X, y)
            
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results[vuln_index] = {
                'n_samples': len(X),
                'r2_mean': scores.mean(),
                'r2_std': scores.std(),
                'top_features': importance.head(5).to_dict('records')
            }
            
            self.logger.info(f"  {vuln_index}: R²={scores.mean():.3f}, n={len(X)}")
        
        return results
    
    def _discover_socioeconomic_clusters(self):
        """Discover socioeconomic-climate clusters"""
        
        features = []
        
        # Vulnerability indices
        vuln = ['heat_vulnerability_index', 'housing_vulnerability', 'economic_vulnerability']
        features.extend([v for v in vuln if v in self.socioeconomic_df.columns])
        
        # Climate
        climate = ['temperature', 'humidity', 'heat_index']
        features.extend([c for c in climate if c in self.socioeconomic_df.columns])
        
        # Demographics
        if 'Education' in self.socioeconomic_df.columns:
            self.socioeconomic_df['education_level'] = pd.Categorical(self.socioeconomic_df['Education']).codes
            features.append('education_level')
        
        if len(features) < 3:
            return None
        
        # Prepare data
        cluster_data = self.socioeconomic_df[features].copy()
        
        for col in features:
            cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
        
        cluster_data = cluster_data.dropna()
        
        if len(cluster_data) < 100:
            return None
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)
        
        # Clustering
        best_k = 4  # Or determine optimally
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Characterize clusters
        cluster_profiles = []
        for cluster_id in range(best_k):
            cluster_mask = cluster_data['cluster'] == cluster_id
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'vulnerability_level': 'High' if cluster_data[cluster_mask]['heat_vulnerability_index'].mean() > cluster_data['heat_vulnerability_index'].median() else 'Low'
                if 'heat_vulnerability_index' in cluster_data.columns else 'Unknown'
            }
            cluster_profiles.append(profile)
        
        result = {
            'n_clusters': best_k,
            'n_samples': len(cluster_data),
            'cluster_profiles': cluster_profiles
        }
        
        self.logger.info(f"  Found {best_k} socioeconomic-climate clusters")
        
        return result
    
    def _analyze_climate_exposure_by_ses(self):
        """Analyze how socioeconomic status affects climate exposure"""
        
        if 'Education' not in self.socioeconomic_df.columns:
            return None
        
        # Group by education level
        exposure_by_education = {}
        
        for education_level in self.socioeconomic_df['Education'].dropna().unique():
            subset = self.socioeconomic_df[self.socioeconomic_df['Education'] == education_level]
            
            exposure_by_education[education_level] = {
                'n': len(subset),
                'avg_temperature': subset['temperature'].mean() if 'temperature' in subset.columns else None,
                'avg_heat_index': subset['heat_index'].mean() if 'heat_index' in subset.columns else None,
                'heat_vulnerability': subset['heat_vulnerability_index'].mean() if 'heat_vulnerability_index' in subset.columns else None
            }
        
        self.logger.info(f"  Analyzed climate exposure for {len(exposure_by_education)} education levels")
        
        return exposure_by_education
    
    # ==========================================
    # APPROACH 2: Ecological Aggregation
    # ==========================================
    
    def approach_2_ecological_aggregation(self):
        """Ecological analysis following Labib et al. methodology"""
        
        self.logger.info("\n🌍 APPROACH 2: Ecological (Neighborhood-Level) Analysis")
        self.logger.info("="*50)
        
        # Check for geographic variables
        if 'latitude' not in self.full_df.columns or 'longitude' not in self.full_df.columns:
            self.logger.warning("No geographic variables for ecological aggregation")
            return None
        
        # Create geographic units (grid cells)
        n_lat_bins = 15
        n_lon_bins = 15
        
        self.logger.info(f"Creating {n_lat_bins}x{n_lon_bins} geographic grid")
        
        # Create bins
        self.full_df['lat_bin'] = pd.qcut(
            self.full_df['latitude'].dropna(), 
            n_lat_bins, 
            duplicates='drop', 
            labels=False
        )
        self.full_df['lon_bin'] = pd.qcut(
            self.full_df['longitude'].dropna(), 
            n_lon_bins, 
            duplicates='drop', 
            labels=False
        )
        self.full_df['geo_unit'] = self.full_df['lat_bin'].astype(str) + '_' + self.full_df['lon_bin'].astype(str)
        
        # Aggregate clinical data
        clinical_agg = self._aggregate_clinical_to_ecological()
        
        # Aggregate socioeconomic data
        socioeconomic_agg = self._aggregate_socioeconomic_to_ecological()
        
        # Merge ecological datasets
        if clinical_agg is not None and socioeconomic_agg is not None:
            ecological_df = pd.merge(
                clinical_agg, 
                socioeconomic_agg, 
                on='geo_unit', 
                how='inner',
                suffixes=('_clinical', '_socio')
            )
            
            self.logger.info(f"Created ecological dataset with {len(ecological_df)} geographic units")
            
            # Run ecological models
            ecological_results = self._run_ecological_models(ecological_df)
            
            self.results['approaches']['ecological'] = ecological_results
            return ecological_results
        
        return None
    
    def _aggregate_clinical_to_ecological(self):
        """Aggregate clinical data to geographic units"""
        
        clinical_with_geo = self.full_df[self.full_df['dataset_group'] == 'clinical'].copy()
        
        if 'geo_unit' not in clinical_with_geo.columns:
            return None
        
        # Variables to aggregate
        agg_dict = {
            'systolic blood pressure': 'mean',
            'diastolic blood pressure': 'mean',
            'FASTING GLUCOSE': 'mean',
            'FASTING TOTAL CHOLESTEROL': 'mean',
            'temperature': 'mean',
            'humidity': 'mean',
            'heat_index': 'mean'
        }
        
        # Filter available columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in clinical_with_geo.columns}
        
        # Add sample size
        agg_dict['clinical_n'] = ('unified_id', 'count')
        
        # Aggregate
        clinical_agg = clinical_with_geo.groupby('geo_unit').agg(agg_dict).reset_index()
        
        # Rename columns for clarity
        clinical_agg.columns = [col[0] if col[1] == '' else f'{col[0]}_{col[1]}' 
                                for col in clinical_agg.columns]
        clinical_agg.rename(columns={'unified_id_count': 'clinical_n'}, inplace=True)
        
        # Filter units with sufficient data
        clinical_agg = clinical_agg[clinical_agg['clinical_n'] >= 5]
        
        self.logger.info(f"  Aggregated clinical data to {len(clinical_agg)} geographic units")
        
        return clinical_agg
    
    def _aggregate_socioeconomic_to_ecological(self):
        """Aggregate socioeconomic data to geographic units"""
        
        socio_with_geo = self.full_df[self.full_df['dataset_group'] == 'socioeconomic'].copy()
        
        if 'geo_unit' not in socio_with_geo.columns:
            return None
        
        # Variables to aggregate
        agg_dict = {
            'heat_vulnerability_index': 'mean',
            'housing_vulnerability': 'mean',
            'economic_vulnerability': 'mean'
        }
        
        # Filter available columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in socio_with_geo.columns}
        
        # Add education mode if available
        if 'Education' in socio_with_geo.columns:
            # Most common education level
            agg_dict['Education'] = lambda x: x.value_counts().index[0] if len(x) > 0 else None
        
        # Add sample size
        agg_dict['socio_n'] = ('unified_id', 'count')
        
        # Aggregate
        socio_agg = socio_with_geo.groupby('geo_unit').agg(agg_dict).reset_index()
        
        # Rename columns
        if ('socio_n', 'unified_id') in socio_agg.columns:
            socio_agg.rename(columns={('socio_n', 'unified_id'): 'socio_n'}, inplace=True)
        
        # Filter units with sufficient data
        if 'socio_n' in socio_agg.columns:
            socio_agg = socio_agg[socio_agg['socio_n'] >= 5]
        
        self.logger.info(f"  Aggregated socioeconomic data to {len(socio_agg)} geographic units")
        
        return socio_agg
    
    def _run_ecological_models(self, ecological_df):
        """Run models on ecological (neighborhood-level) data"""
        
        results = {}
        
        # Model 1: Socioeconomic factors → Average health outcomes
        self.logger.info("\n  Model 1: Neighborhood SES → Health Outcomes")
        
        health_outcomes = ['systolic blood pressure', 'diastolic blood pressure', 'FASTING GLUCOSE']
        ses_predictors = ['heat_vulnerability_index', 'housing_vulnerability', 'economic_vulnerability']
        
        for outcome in health_outcomes:
            if outcome not in ecological_df.columns:
                continue
            
            available_predictors = [p for p in ses_predictors if p in ecological_df.columns]
            
            if len(available_predictors) < 1:
                continue
            
            valid_data = ecological_df[ecological_df[outcome].notna()].copy()
            
            if len(valid_data) < 10:
                continue
            
            # Add climate variables
            climate_vars = ['temperature', 'humidity', 'heat_index']
            available_predictors.extend([c for c in climate_vars if c in valid_data.columns])
            
            X = valid_data[available_predictors].fillna(valid_data[available_predictors].median())
            y = valid_data[outcome]
            
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X, y = X[mask], y[mask]
            
            if len(X) < 10:
                continue
            
            # Use simpler model for small ecological datasets
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
            # Leave-one-out CV for small samples
            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
            
            model.fit(X, y)
            
            results[f'ecological_{outcome}'] = {
                'n_neighborhoods': len(X),
                'predictors': available_predictors,
                'r2_mean': scores.mean(),
                'r2_std': scores.std(),
                'coefficients': dict(zip(available_predictors, model.coef_))
            }
            
            self.logger.info(f"    {outcome}: R²={scores.mean():.3f}, n={len(X)} neighborhoods")
        
        # Model 2: Identify vulnerable neighborhoods
        self.logger.info("\n  Model 2: Identifying Vulnerable Neighborhoods")
        
        if 'heat_vulnerability_index' in ecological_df.columns and 'systolic blood pressure' in ecological_df.columns:
            # Create vulnerability categories
            ecological_df['high_vulnerability'] = (
                ecological_df['heat_vulnerability_index'] > ecological_df['heat_vulnerability_index'].median()
            ).astype(int)
            
            ecological_df['high_bp'] = (
                ecological_df['systolic blood pressure'] > 130  # Hypertension threshold
            ).astype(int)
            
            # Cross-tabulation
            vuln_health_crosstab = pd.crosstab(
                ecological_df['high_vulnerability'],
                ecological_df['high_bp'],
                normalize='index'
            )
            
            results['vulnerability_health_relationship'] = {
                'crosstab': vuln_health_crosstab.to_dict(),
                'n_neighborhoods': len(ecological_df),
                'interpretation': 'Proportion of high BP neighborhoods by vulnerability status'
            }
            
            self.logger.info(f"    Analyzed vulnerability-health patterns across {len(ecological_df)} neighborhoods")
        
        return results
    
    def generate_insights_summary(self):
        """Generate summary of key insights discovered"""
        
        self.logger.info("\n" + "="*70)
        self.logger.info("KEY INSIGHTS DISCOVERED")
        self.logger.info("="*70)
        
        insights = []
        
        # Clinical insights
        if 'clinical_flexible' in self.results['approaches']:
            clinical = self.results['approaches']['clinical_flexible']
            
            # Check clustering results
            if 'clustering' in clinical and clinical['clustering']:
                n_clusters = clinical['clustering']['n_clusters']
                insights.append(f"• Found {n_clusters} distinct climate-health phenotypes in clinical data")
            
            # Check composite score
            if 'composite_health_score' in clinical and clinical['composite_health_score']:
                r2 = clinical['composite_health_score']['r2_mean']
                if r2 > 0.05:
                    insights.append(f"• Composite health risk score shows R²={r2:.3f} with climate predictors")
        
        # Socioeconomic insights
        if 'socioeconomic_flexible' in self.results['approaches']:
            socio = self.results['approaches']['socioeconomic_flexible']
            
            if 'climate_exposure_patterns' in socio and socio['climate_exposure_patterns']:
                insights.append(f"• Identified differential climate exposure patterns by education level")
        
        # Ecological insights
        if 'ecological' in self.results['approaches']:
            ecological = self.results['approaches']['ecological']
            
            for key, value in ecological.items():
                if 'ecological_' in key and 'r2_mean' in value:
                    if value['r2_mean'] > 0.1:
                        outcome = key.replace('ecological_', '')
                        insights.append(f"• Neighborhood-level {outcome} associated with socioeconomic factors (R²={value['r2_mean']:.3f})")
        
        self.logger.info("\nDiscovered Insights:")
        for insight in insights:
            self.logger.info(insight)
        
        self.results['insights_summary'] = insights
        
        return insights
    
    def save_results(self):
        """Save all results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"flexible_discovery_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"\n📁 Results saved to: {results_file}")
        
        # Create summary report
        summary_file = self.results_dir / f"discovery_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write("# Flexible Climate-Health Discovery Results\n\n")
            
            f.write("## Approach Summary\n\n")
            f.write("1. **Clinical Cohort Analysis**\n")
            f.write("   - Multiple target configurations\n")
            f.write("   - Unsupervised clustering\n")
            f.write("   - Composite health scores\n\n")
            
            f.write("2. **Socioeconomic Cohort Analysis**\n")
            f.write("   - Vulnerability prediction\n")
            f.write("   - Climate exposure patterns\n")
            f.write("   - Socioeconomic clustering\n\n")
            
            f.write("3. **Ecological Analysis**\n")
            f.write("   - Neighborhood-level aggregation\n")
            f.write("   - Linked socioeconomic-health patterns\n\n")
            
            f.write("## Key Insights\n\n")
            if 'insights_summary' in self.results:
                for insight in self.results['insights_summary']:
                    f.write(f"{insight}\n")
        
        self.logger.info(f"📄 Summary saved to: {summary_file}")
    
    def run_complete_analysis(self, filepath):
        """Run all approaches"""
        
        # Load data
        self.load_and_prepare_data(filepath)
        
        # Approach 1A: Clinical cohort
        self.approach_1a_clinical_flexible_targets()
        
        # Approach 1B: Socioeconomic cohort
        self.approach_1b_socioeconomic_flexible()
        
        # Approach 2: Ecological aggregation
        self.approach_2_ecological_aggregation()
        
        # Generate insights
        self.generate_insights_summary()
        
        # Save results
        self.save_results()
        
        return self.results


def main():
    """Run the flexible discovery analysis"""
    
    discovery = FlexibleClimateHealthDiscovery()
    
    results = discovery.run_complete_analysis('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv')
    
    print("\n" + "="*70)
    print("FLEXIBLE DISCOVERY ANALYSIS COMPLETE")
    print("="*70)
    print("\nExplored multiple approaches:")
    print("✓ Clinical cohort with flexible targets")
    print("✓ Socioeconomic cohort with various analyses")
    print("✓ Ecological aggregation following Labib methodology")
    print("\nResults saved to flexible_discovery_results/ directory")
    
    return results


if __name__ == "__main__":
    main()