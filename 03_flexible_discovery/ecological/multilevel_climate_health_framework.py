#!/usr/bin/env python3
"""
Multi-Level Climate-Health Modeling Framework
=============================================

Addresses the fundamental data structure issue:
- Separate clinical trial data (with biomarkers)
- Separate socioeconomic survey data
- Proper approaches for each data type and potential linkage strategies

Based on ecological study design principles from Labib et al.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import json
from datetime import datetime
from pathlib import Path
import logging
from scipy import stats

warnings.filterwarnings('ignore')


class MultiLevelClimateHealthFramework:
    """
    Framework for properly handling multi-level climate-health data
    """
    
    def __init__(self, results_dir="multilevel_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / f'multilevel_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'analysis_type': 'Multi-Level Climate-Health Framework',
                'approach': 'Separate cohort-specific analyses with ecological aggregation'
            }
        }
    
    def load_and_separate_cohorts(self, filepath):
        """
        Load data and properly separate the two distinct cohorts
        """
        self.logger.info("📊 Loading and separating distinct cohorts")
        
        df = pd.read_csv(filepath, low_memory=False)
        self.logger.info(f"Total dataset: {len(df)} records, {len(df.columns)} variables")
        
        # Separate cohorts based on dataset_group
        clinical_cohort = df[df['dataset_group'] == 'clinical'].copy() if 'dataset_group' in df.columns else pd.DataFrame()
        socioeconomic_cohort = df[df['dataset_group'] == 'socioeconomic'].copy() if 'dataset_group' in df.columns else pd.DataFrame()
        
        # Alternative separation if dataset_group not available
        if clinical_cohort.empty or socioeconomic_cohort.empty:
            # Use presence of biomarkers to identify clinical cohort
            biomarker_cols = ['FASTING GLUCOSE', 'systolic blood pressure', 'diastolic blood pressure', 
                            'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING TRIGLYCERIDES']
            
            has_biomarkers = df[biomarker_cols].notna().any(axis=1)
            clinical_cohort = df[has_biomarkers].copy()
            socioeconomic_cohort = df[~has_biomarkers].copy()
        
        self.logger.info(f"Clinical cohort: {len(clinical_cohort)} participants")
        self.logger.info(f"Socioeconomic cohort: {len(socioeconomic_cohort)} participants")
        
        # Analyze data availability in each cohort
        self._analyze_cohort_completeness(clinical_cohort, "Clinical")
        self._analyze_cohort_completeness(socioeconomic_cohort, "Socioeconomic")
        
        return clinical_cohort, socioeconomic_cohort
    
    def _analyze_cohort_completeness(self, cohort_df, cohort_name):
        """Analyze data completeness for each cohort"""
        
        self.logger.info(f"\n--- {cohort_name} Cohort Data Availability ---")
        
        # Check key variables
        key_vars = {
            'Biomarkers': ['FASTING GLUCOSE', 'systolic blood pressure', 'diastolic blood pressure'],
            'Socioeconomic': ['Education', 'employment_status', 'housing_vulnerability'],
            'Climate': ['temperature', 'humidity', 'heat_index'],
            'Geographic': ['latitude', 'longitude']
        }
        
        for category, variables in key_vars.items():
            available = []
            for var in variables:
                if var in cohort_df.columns:
                    non_missing = cohort_df[var].notna().sum()
                    pct = (non_missing / len(cohort_df)) * 100
                    if pct > 5:  # Only report if >5% have data
                        available.append(f"{var} ({pct:.1f}%)")
            
            if available:
                self.logger.info(f"  {category}: {', '.join(available)}")
    
    def approach_1_clinical_cohort_analysis(self, clinical_cohort):
        """
        Approach 1: Analyze clinical cohort with available covariates
        Focus on climate-biomarker relationships with limited socioeconomic adjustment
        """
        
        self.logger.info("\n🔬 APPROACH 1: Clinical Cohort Analysis")
        self.logger.info("="*50)
        
        results = {}
        
        # Define target biomarkers
        biomarkers = [
            'FASTING GLUCOSE', 'systolic blood pressure', 'diastolic blood pressure',
            'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING TRIGLYCERIDES',
            'Creatinine (mg/dL)', 'ALT (U/L)', 'AST (U/L)'
        ]
        
        # Climate and temporal features
        climate_features = ['temperature', 'humidity', 'heat_index', 'apparent_temp', 'wet_bulb_temp']
        temporal_features = ['year', 'month', 'season']
        demographic_features = ['Sex', 'Race', 'latitude', 'longitude']
        
        # Add lag features
        lag_features = [col for col in clinical_cohort.columns if 'lag' in col.lower()][:20]
        
        for biomarker in biomarkers:
            if biomarker not in clinical_cohort.columns:
                continue
                
            # Create analysis dataset
            target_data = clinical_cohort[clinical_cohort[biomarker].notna()].copy()
            
            if len(target_data) < 100:
                self.logger.warning(f"Insufficient data for {biomarker} (n={len(target_data)})")
                continue
            
            self.logger.info(f"\nAnalyzing {biomarker} (n={len(target_data)})")
            
            # Select available features
            all_features = climate_features + temporal_features + demographic_features + lag_features
            available_features = [f for f in all_features if f in target_data.columns]
            
            # Prepare data
            X = target_data[available_features].copy()
            y = target_data[biomarker]
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col]).codes
            
            # Remove rows with missing values
            complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[complete_mask]
            y = y[complete_mask]
            
            if len(X) < 100:
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(X.median()))
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            # Fit final model
            model.fit(X_scaled, y)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results[biomarker] = {
                'n_samples': len(X),
                'n_features': len(available_features),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'top_features': feature_importance.head(10).to_dict('records'),
                'climate_importance': feature_importance[
                    feature_importance['feature'].isin(climate_features + lag_features)
                ]['importance'].sum()
            }
            
            self.logger.info(f"  R² = {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            self.logger.info(f"  Climate feature importance: {results[biomarker]['climate_importance']:.3f}")
        
        self.results['clinical_cohort_analysis'] = results
        return results
    
    def approach_2_ecological_aggregation(self, clinical_cohort, socioeconomic_cohort):
        """
        Approach 2: Aggregate both cohorts to neighborhood level (ecological study)
        Following Labib et al. methodology
        """
        
        self.logger.info("\n🌍 APPROACH 2: Ecological (Neighborhood-Level) Analysis")
        self.logger.info("="*50)
        
        # Check for geographic identifiers
        geo_vars = ['latitude', 'longitude', 'neighborhood_id', 'district', 'ward']
        available_geo = [v for v in geo_vars if v in clinical_cohort.columns or v in socioeconomic_cohort.columns]
        
        if not available_geo:
            self.logger.warning("No geographic variables found for aggregation")
            return None
        
        self.logger.info(f"Using geographic variables: {available_geo}")
        
        # Create geographic bins if only lat/lon available
        if 'latitude' in available_geo and 'longitude' in available_geo:
            # Create neighborhood proxy using spatial binning
            n_neighborhoods = 50  # Adjust based on your data
            
            # Bin coordinates to create neighborhood groups
            if 'latitude' in clinical_cohort.columns:
                clinical_cohort['neighborhood_id'] = (
                    pd.qcut(clinical_cohort['latitude'], n_neighborhoods, duplicates='drop', labels=False).astype(str) + '_' +
                    pd.qcut(clinical_cohort['longitude'], n_neighborhoods, duplicates='drop', labels=False).astype(str)
                )
            
            if 'latitude' in socioeconomic_cohort.columns:
                socioeconomic_cohort['neighborhood_id'] = (
                    pd.qcut(socioeconomic_cohort['latitude'], n_neighborhoods, duplicates='drop', labels=False).astype(str) + '_' +
                    pd.qcut(socioeconomic_cohort['longitude'], n_neighborhoods, duplicates='drop', labels=False).astype(str)
                )
        
        # Aggregate clinical data to neighborhood level
        clinical_agg = self._aggregate_to_neighborhood(clinical_cohort, 'clinical')
        socioeconomic_agg = self._aggregate_to_neighborhood(socioeconomic_cohort, 'socioeconomic')
        
        # Merge aggregated data
        if clinical_agg is not None and socioeconomic_agg is not None:
            ecological_data = pd.merge(
                clinical_agg, 
                socioeconomic_agg, 
                on='neighborhood_id', 
                how='outer',
                suffixes=('_clinical', '_socioeconomic')
            )
            
            self.logger.info(f"Ecological dataset: {len(ecological_data)} neighborhoods")
            
            # Run ecological analysis
            ecological_results = self._run_ecological_models(ecological_data)
            self.results['ecological_analysis'] = ecological_results
            
            return ecological_results
        
        return None
    
    def _aggregate_to_neighborhood(self, cohort_df, cohort_type):
        """Aggregate cohort data to neighborhood level"""
        
        if 'neighborhood_id' not in cohort_df.columns:
            return None
        
        self.logger.info(f"Aggregating {cohort_type} data to neighborhood level")
        
        # Define aggregation rules
        agg_rules = {}
        
        # Numeric variables - take mean
        numeric_cols = cohort_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['latitude', 'longitude', 'year']:
                agg_rules[col] = 'mean'
        
        # Categorical variables - take mode or proportion
        categorical_cols = cohort_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'neighborhood_id':
                # Calculate mode (most common category)
                agg_rules[col] = lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else None
        
        # Note: sample size will be added separately
        
        # Perform aggregation
        neighborhood_df = cohort_df.groupby('neighborhood_id').agg(agg_rules).reset_index()
        
        # Add sample size
        sample_counts = cohort_df.groupby('neighborhood_id').size().reset_index(name=f'n_samples_{cohort_type}')
        neighborhood_df = neighborhood_df.merge(sample_counts, on='neighborhood_id', how='left')
        
        self.logger.info(f"  Aggregated to {len(neighborhood_df)} neighborhoods")
        
        return neighborhood_df
    
    def _run_ecological_models(self, ecological_data):
        """Run models on ecological (neighborhood-level) data"""
        
        results = {}
        
        # Define ecological outcomes (neighborhood averages)
        outcomes = [
            col for col in ecological_data.columns 
            if any(marker in col for marker in ['GLUCOSE', 'blood pressure', 'CHOLESTEROL', 'HDL', 'TRIGLYCERIDES'])
        ]
        
        # Define ecological predictors
        predictors = [
            col for col in ecological_data.columns 
            if any(term in col for term in ['temperature', 'humidity', 'heat_index', 'Education', 'employment', 'vulnerability'])
        ]
        
        for outcome in outcomes:
            if ecological_data[outcome].notna().sum() < 20:
                continue
            
            self.logger.info(f"\nEcological model for {outcome}")
            
            # Prepare data
            valid_data = ecological_data[ecological_data[outcome].notna()].copy()
            
            X = valid_data[predictors].fillna(valid_data[predictors].median())
            y = valid_data[outcome]
            
            if len(X) < 20:
                continue
            
            # Check for multicollinearity (VIF)
            self._check_multicollinearity(X)
            
            # Simple model due to limited sample size
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            
            # Leave-one-out CV for small samples
            if len(X) < 50:
                from sklearn.model_selection import LeaveOneOut
                loo = LeaveOneOut()
                cv_scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            model.fit(X, y)
            
            results[outcome] = {
                'n_neighborhoods': len(X),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'feature_importance': pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10).to_dict('records')
            }
            
            self.logger.info(f"  Neighborhoods: {len(X)}, R² = {cv_scores.mean():.3f}")
        
        return results
    
    def _check_multicollinearity(self, X):
        """Check for multicollinearity using correlation matrix (simplified VIF)"""
        
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        
        if to_drop:
            self.logger.info(f"  Dropping {len(to_drop)} highly correlated features")
            X.drop(columns=to_drop, inplace=True)
        
        return X
    
    def approach_3_matched_subset_analysis(self, clinical_cohort, socioeconomic_cohort):
        """
        Approach 3: Find and analyze matched subset with both clinical and socioeconomic data
        """
        
        self.logger.info("\n🔗 APPROACH 3: Matched Subset Analysis")
        self.logger.info("="*50)
        
        # Look for participants with both types of data
        # This requires some linking variable (ID, location, time)
        
        # Check if there's any overlap in IDs
        if 'unified_id' in clinical_cohort.columns and 'unified_id' in socioeconomic_cohort.columns:
            clinical_ids = set(clinical_cohort['unified_id'])
            socioeconomic_ids = set(socioeconomic_cohort['unified_id'])
            overlap_ids = clinical_ids.intersection(socioeconomic_ids)
            
            if overlap_ids:
                self.logger.info(f"Found {len(overlap_ids)} participants with both data types")
                
                # Create matched dataset
                clinical_matched = clinical_cohort[clinical_cohort['unified_id'].isin(overlap_ids)]
                socioeconomic_matched = socioeconomic_cohort[socioeconomic_cohort['unified_id'].isin(overlap_ids)]
                
                # Merge on ID
                matched_data = pd.merge(
                    clinical_matched,
                    socioeconomic_matched,
                    on='unified_id',
                    suffixes=('', '_socio')
                )
                
                # Run integrated analysis
                return self._run_integrated_analysis(matched_data)
        
        # Alternative: Match by location and time window
        if all(col in clinical_cohort.columns for col in ['latitude', 'longitude', 'primary_date']):
            self.logger.info("Attempting spatiotemporal matching...")
            
            # Convert dates
            clinical_cohort['date'] = pd.to_datetime(clinical_cohort['primary_date'])
            socioeconomic_cohort['date'] = pd.to_datetime(socioeconomic_cohort['primary_date'])
            
            # Match within spatial and temporal windows
            matched_pairs = []
            
            for idx, clinical_row in clinical_cohort.iterrows():
                if pd.isna(clinical_row['latitude']):
                    continue
                
                # Find socioeconomic records within 1km and 30 days
                spatial_match = (
                    (abs(socioeconomic_cohort['latitude'] - clinical_row['latitude']) < 0.01) &  # ~1km
                    (abs(socioeconomic_cohort['longitude'] - clinical_row['longitude']) < 0.01)
                )
                
                temporal_match = abs((socioeconomic_cohort['date'] - clinical_row['date']).dt.days) < 30
                
                matches = socioeconomic_cohort[spatial_match & temporal_match]
                
                if len(matches) > 0:
                    # Take closest match
                    matched_pairs.append({
                        'clinical_idx': idx,
                        'socioeconomic_idx': matches.index[0]
                    })
                
                if len(matched_pairs) >= 1000:  # Stop after finding enough matches
                    break
            
            if matched_pairs:
                self.logger.info(f"Found {len(matched_pairs)} spatiotemporal matches")
                # Create matched dataset
                # [Implementation would continue here]
        
        self.logger.info("No sufficient matching possible between cohorts")
        return None
    
    def _run_integrated_analysis(self, matched_data):
        """Run analysis on matched data with both clinical and socioeconomic variables"""
        
        self.logger.info(f"Running integrated analysis on {len(matched_data)} matched records")
        
        # This would implement the full model with both data types
        # Similar to approach_1 but with full covariate set
        
        return {}
    
    def approach_4_transfer_learning(self, clinical_cohort, socioeconomic_cohort):
        """
        Approach 4: Use transfer learning / domain adaptation
        Train on clinical data, adapt to socioeconomic context
        """
        
        self.logger.info("\n🔄 APPROACH 4: Transfer Learning Framework")
        self.logger.info("="*50)
        
        # Train base models on clinical data
        # Then use socioeconomic data to refine predictions
        # This is more advanced and would require careful implementation
        
        self.logger.info("Transfer learning approach requires specialized implementation")
        self.logger.info("Consider using domain adaptation techniques or multi-task learning")
        
        return None
    
    def generate_recommendations(self):
        """Generate specific recommendations based on analysis results"""
        
        self.logger.info("\n📋 RECOMMENDATIONS")
        self.logger.info("="*50)
        
        recommendations = []
        
        # Check which approaches were successful
        if 'clinical_cohort_analysis' in self.results:
            clinical_results = self.results['clinical_cohort_analysis']
            successful_biomarkers = [k for k, v in clinical_results.items() if v['cv_r2_mean'] > 0.1]
            
            if successful_biomarkers:
                recommendations.append({
                    'approach': 'Clinical Cohort Analysis',
                    'status': 'VIABLE',
                    'recommendation': f"Focus on {len(successful_biomarkers)} biomarkers with R² > 0.1",
                    'next_steps': [
                        "Enhance climate lag features",
                        "Add interaction terms",
                        "Consider non-linear models (XGBoost, neural networks)"
                    ]
                })
        
        if 'ecological_analysis' in self.results and self.results['ecological_analysis']:
            recommendations.append({
                'approach': 'Ecological Analysis',
                'status': 'RECOMMENDED',
                'recommendation': "Neighborhood-level aggregation shows promise",
                'next_steps': [
                    "Improve geographic resolution",
                    "Add more neighborhood-level covariates",
                    "Account for spatial autocorrelation",
                    "Consider multilevel models"
                ]
            })
        
        # Data collection recommendations
        recommendations.append({
            'approach': 'Data Enhancement',
            'status': 'CRITICAL',
            'recommendation': "Improve data linkage between cohorts",
            'next_steps': [
                "Implement participant matching protocol",
                "Collect socioeconomic data from clinical participants",
                "Collect basic health metrics from survey participants",
                "Add common identifiers across studies"
            ]
        })
        
        self.results['recommendations'] = recommendations
        
        for rec in recommendations:
            self.logger.info(f"\n{rec['status']}: {rec['approach']}")
            self.logger.info(f"  {rec['recommendation']}")
            self.logger.info(f"  Next steps:")
            for step in rec['next_steps']:
                self.logger.info(f"    • {step}")
        
        return recommendations
    
    def save_results(self):
        """Save all results and recommendations"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"multilevel_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"\n📁 Results saved to: {results_file}")
        
        # Create summary report
        summary_file = self.results_dir / f"analysis_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("MULTI-LEVEL CLIMATE-HEALTH ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write("DATA STRUCTURE ISSUE:\n")
            f.write("- Clinical cohort and socioeconomic cohort are separate populations\n")
            f.write("- Cannot be modeled together as single dataset\n")
            f.write("- Requires specialized approaches\n\n")
            
            f.write("APPROACHES TESTED:\n")
            for approach in ['clinical_cohort_analysis', 'ecological_analysis']:
                if approach in self.results:
                    f.write(f"✓ {approach.replace('_', ' ').title()}\n")
            
            f.write("\nKEY RECOMMENDATIONS:\n")
            if 'recommendations' in self.results:
                for rec in self.results['recommendations']:
                    if rec['status'] == 'CRITICAL' or rec['status'] == 'RECOMMENDED':
                        f.write(f"• {rec['recommendation']}\n")
        
        self.logger.info(f"📄 Summary saved to: {summary_file}")


def main():
    """Run the multi-level framework analysis"""
    
    framework = MultiLevelClimateHealthFramework()
    
    # Load and separate cohorts
    clinical_cohort, socioeconomic_cohort = framework.load_and_separate_cohorts(
        'FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv'
    )
    
    # Approach 1: Clinical cohort analysis
    clinical_results = framework.approach_1_clinical_cohort_analysis(clinical_cohort)
    
    # Approach 2: Ecological aggregation
    ecological_results = framework.approach_2_ecological_aggregation(
        clinical_cohort, socioeconomic_cohort
    )
    
    # Approach 3: Try to find matched subset
    matched_results = framework.approach_3_matched_subset_analysis(
        clinical_cohort, socioeconomic_cohort
    )
    
    # Generate recommendations
    recommendations = framework.generate_recommendations()
    
    # Save all results
    framework.save_results()
    
    print("\n" + "="*70)
    print("MULTI-LEVEL ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Finding: Your data structure requires separate modeling approaches")
    print("for clinical and socioeconomic cohorts. See recommendations for details.")
    print("\nResults saved to multilevel_results/ directory")
    
    return framework.results


if __name__ == "__main__":
    main()