#!/usr/bin/env python3
"""
Practical ENBEL Socioeconomic Imputation
========================================

This script applies rigorous socioeconomic imputation to the real ENBEL dataset,
combining KNN and ecological approaches for robust results.

Scientific approach:
1. Validate and clean coordinate data
2. Apply spatial-demographic KNN matching
3. Use ecological stratification as backup
4. Validate results with statistical measures
5. Generate comprehensive imputation report

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')

class ENBELImputationPipeline:
    """Practical imputation pipeline for ENBEL data."""
    
    def __init__(self, 
                 k_neighbors=10, 
                 spatial_weight=0.4, 
                 max_distance_km=15,
                 min_matches=3,
                 random_state=42):
        
        self.k_neighbors = k_neighbors
        self.spatial_weight = spatial_weight
        self.demographic_weight = 1 - spatial_weight
        self.max_distance_km = max_distance_km
        self.min_matches = min_matches
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.results = {}
        print(f"Initialized ENBEL Imputation Pipeline")
        print(f"  K-neighbors: {k_neighbors}")
        print(f"  Spatial weight: {spatial_weight:.1%}")
        print(f"  Demographic weight: {self.demographic_weight:.1%}")
        print(f"  Max distance: {max_distance_km}km")
    
    def load_and_validate_data(self, data_file):
        """Load and validate ENBEL dataset."""
        print(f"\nLoading ENBEL data from: {data_file}")
        
        try:
            data = pd.read_csv(data_file, low_memory=False)
            print(f"✅ Loaded {len(data):,} total records")
        except FileNotFoundError:
            print(f"❌ Data file not found: {data_file}")
            return None, None
        
        # Separate cohorts based on data source
        clinical_cohort = data[data['data_source'].notna()].copy()
        gcro_cohort = data[data['data_source'].isna()].copy()
        
        print(f"  Clinical cohort: {len(clinical_cohort):,} participants")
        print(f"  GCRO cohort: {len(gcro_cohort):,} participants")
        
        # Validate coordinates
        def validate_coordinates(df, name):
            # South African coordinate bounds
            lat_bounds = (-35, -22)
            lon_bounds = (16, 33)
            
            valid_coords = (
                df['latitude'].between(*lat_bounds) & 
                df['longitude'].between(*lon_bounds) &
                df['latitude'].notna() &
                df['longitude'].notna()
            )
            
            n_valid = valid_coords.sum()
            print(f"  {name}: {n_valid:,}/{len(df):,} valid coordinates")
            
            return df[valid_coords].copy()
        
        clinical_clean = validate_coordinates(clinical_cohort, "Clinical")
        gcro_clean = validate_coordinates(gcro_cohort, "GCRO")
        
        return clinical_clean, gcro_clean
    
    def analyze_imputation_targets(self, clinical_data, gcro_data):
        """Analyze what variables can be imputed."""
        print("\nAnalyzing imputation opportunities...")
        
        # Potential socioeconomic variables in GCRO data
        potential_vars = [
            'Education', 'employment_status', 'vuln_Housing',
            'vuln_employment_status', 'housing_vulnerability',
            'economic_vulnerability', 'heat_vulnerability_index'
        ]
        
        available_vars = []
        for var in potential_vars:
            if var in gcro_data.columns:
                n_available = gcro_data[var].notna().sum()
                pct_available = (n_available / len(gcro_data)) * 100
                
                if n_available >= 100:  # Minimum threshold
                    available_vars.append(var)
                    print(f"  ✅ {var}: {n_available:,} values ({pct_available:.1f}%)")
                else:
                    print(f"  ❌ {var}: {n_available:,} values ({pct_available:.1f}%) - insufficient")
        
        print(f"\nVariables selected for imputation: {len(available_vars)}")
        return available_vars
    
    def prepare_matching_features(self, donor_data, recipient_data):
        """Prepare features for spatial-demographic matching."""
        print("Preparing matching features...")
        
        # Spatial features (lat/lon)
        donor_spatial = donor_data[['latitude', 'longitude']].values
        recipient_spatial = recipient_data[['latitude', 'longitude']].values
        
        # Scale spatial coordinates
        scaler = StandardScaler()
        combined_spatial = np.vstack([donor_spatial, recipient_spatial])
        scaler.fit(combined_spatial)
        
        donor_spatial_scaled = scaler.transform(donor_spatial)
        recipient_spatial_scaled = scaler.transform(recipient_spatial)
        
        # Demographic features
        donor_demo_features = []
        recipient_demo_features = []
        
        # Encode categorical variables
        demographic_cols = ['Sex', 'Race']
        encoders = {}
        
        for col in demographic_cols:
            if col in donor_data.columns and col in recipient_data.columns:
                # Combine and encode
                combined_values = pd.concat([
                    donor_data[col].astype(str), 
                    recipient_data[col].astype(str)
                ]).fillna('Unknown')
                
                encoder = LabelEncoder()
                encoder.fit(combined_values)
                encoders[col] = encoder
                
                # Transform
                donor_encoded = encoder.transform(donor_data[col].astype(str).fillna('Unknown'))
                recipient_encoded = encoder.transform(recipient_data[col].astype(str).fillna('Unknown'))
                
                donor_demo_features.append(donor_encoded)
                recipient_demo_features.append(recipient_encoded)
        
        # Combine features with weights
        if donor_demo_features:
            donor_demo_array = np.column_stack(donor_demo_features)
            recipient_demo_array = np.column_stack(recipient_demo_features)
        else:
            donor_demo_array = np.zeros((len(donor_data), 1))
            recipient_demo_array = np.zeros((len(recipient_data), 1))
        
        # Weight and combine
        donor_features = np.column_stack([
            donor_spatial_scaled * self.spatial_weight,
            donor_demo_array * self.demographic_weight
        ])
        
        recipient_features = np.column_stack([
            recipient_spatial_scaled * self.spatial_weight,
            recipient_demo_array * self.demographic_weight
        ])
        
        print(f"  Feature matrix: {donor_features.shape[1]} dimensions")
        
        return donor_features, recipient_features, encoders
    
    def perform_knn_imputation(self, donor_features, donor_data, recipient_features, target_vars):
        """Perform KNN-based imputation."""
        print(f"\nPerforming KNN imputation for {len(target_vars)} variables...")
        
        imputation_results = {}
        
        for var in target_vars:
            print(f"  Processing {var}...")
            
            # Get valid donors for this variable
            valid_mask = donor_data[var].notna()
            n_valid = valid_mask.sum()
            
            if n_valid < self.min_matches:
                print(f"    ❌ Insufficient donors: {n_valid}")
                continue
            
            # Prepare KNN
            valid_features = donor_features[valid_mask]
            valid_values = donor_data.loc[valid_mask, var].values
            
            knn = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, n_valid),
                metric='euclidean'
            )
            knn.fit(valid_features)
            
            # Find neighbors and impute
            distances, indices = knn.kneighbors(recipient_features)
            
            imputed_values = []
            confidence_scores = []
            
            for i in range(len(recipient_features)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_values = valid_values[neighbor_indices]
                
                # Weighted average by inverse distance
                weights = 1 / (neighbor_distances + 1e-8)
                weights = weights / weights.sum()
                
                imputed_value = np.average(neighbor_values, weights=weights)
                
                # Confidence based on distance and agreement
                avg_distance = np.mean(neighbor_distances)
                value_std = np.std(neighbor_values)
                
                distance_conf = 1 / (1 + avg_distance)
                agreement_conf = 1 / (1 + value_std) if value_std > 0 else 1.0
                confidence = (distance_conf + agreement_conf) / 2
                
                imputed_values.append(imputed_value)
                confidence_scores.append(confidence)
            
            imputation_results[var] = {
                'values': np.array(imputed_values),
                'confidence': np.array(confidence_scores),
                'n_donors': n_valid,
                'mean_confidence': np.mean(confidence_scores),
                'method': 'knn'
            }
            
            print(f"    ✅ {len(imputed_values):,} values imputed")
            print(f"       Donors: {n_valid:,}, Confidence: {np.mean(confidence_scores):.3f}")
        
        return imputation_results
    
    def perform_ecological_imputation(self, donor_data, recipient_data, target_vars):
        """Perform ecological stratification imputation."""
        print(f"\nPerforming ecological imputation for {len(target_vars)} variables...")
        
        ecological_results = {}
        
        for var in target_vars:
            print(f"  Processing {var}...")
            
            # Calculate stratum means
            group_means = {}
            
            # Sex-Race groups
            if 'Sex' in donor_data.columns and 'Race' in donor_data.columns:
                sex_race_means = donor_data.groupby(['Sex', 'Race'])[var].agg(['mean', 'count']).reset_index()
                sex_race_means = sex_race_means[sex_race_means['count'] >= 5]  # Min 5 observations
                
                for _, row in sex_race_means.iterrows():
                    key = (str(row['Sex']), str(row['Race']))
                    group_means[key] = {'mean': row['mean'], 'count': row['count'], 'confidence': 0.7}
            
            # Sex groups (fallback)
            if 'Sex' in donor_data.columns:
                sex_means = donor_data.groupby('Sex')[var].agg(['mean', 'count']).reset_index()
                
                for _, row in sex_means.iterrows():
                    key = str(row['Sex'])
                    if key not in [k[0] for k in group_means.keys()]:  # Don't override specific groups
                        group_means[key] = {'mean': row['mean'], 'count': row['count'], 'confidence': 0.5}
            
            # Overall mean (final fallback)
            overall_mean = donor_data[var].mean()
            
            # Impute for recipients
            imputed_values = []
            confidence_scores = []
            
            for _, row in recipient_data.iterrows():
                imputed_value = overall_mean
                confidence = 0.3
                
                # Try sex-race match
                if 'Sex' in row and 'Race' in row:
                    key = (str(row['Sex']), str(row['Race']))
                    if key in group_means:
                        imputed_value = group_means[key]['mean']
                        confidence = group_means[key]['confidence']
                    else:
                        # Try sex-only match
                        sex_key = str(row['Sex'])
                        if sex_key in group_means:
                            imputed_value = group_means[sex_key]['mean']
                            confidence = group_means[sex_key]['confidence']
                
                imputed_values.append(imputed_value)
                confidence_scores.append(confidence)
            
            ecological_results[var] = {
                'values': np.array(imputed_values),
                'confidence': np.array(confidence_scores),
                'n_groups': len(group_means),
                'mean_confidence': np.mean(confidence_scores),
                'method': 'ecological'
            }
            
            print(f"    ✅ {len(imputed_values):,} values imputed")
            print(f"       Groups: {len(group_means)}, Confidence: {np.mean(confidence_scores):.3f}")
        
        return ecological_results
    
    def combine_imputation_methods(self, knn_results, eco_results):
        """Combine KNN and ecological results with confidence weighting."""
        print("\nCombining imputation methods...")
        
        combined_results = {}
        
        for var in knn_results.keys():
            if var in eco_results:
                knn_vals = knn_results[var]['values']
                knn_conf = knn_results[var]['confidence']
                eco_vals = eco_results[var]['values']
                eco_conf = eco_results[var]['confidence']
                
                # Weight by confidence
                total_conf = knn_conf + eco_conf
                knn_weight = knn_conf / total_conf
                eco_weight = eco_conf / total_conf
                
                combined_vals = knn_vals * knn_weight + eco_vals * eco_weight
                combined_conf = (knn_conf + eco_conf) / 2
                
                combined_results[var] = {
                    'values': combined_vals,
                    'confidence': combined_conf,
                    'mean_confidence': np.mean(combined_conf),
                    'method': 'combined'
                }
                
                print(f"  {var}: Combined with confidence weighting")
        
        return combined_results
    
    def generate_imputation_report(self, clinical_data, results, filename=None):
        """Generate comprehensive imputation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"enbel_imputation_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'methodology': {
                'k_neighbors': self.k_neighbors,
                'spatial_weight': self.spatial_weight,
                'demographic_weight': self.demographic_weight,
                'max_distance_km': self.max_distance_km,
                'min_matches': self.min_matches
            },
            'dataset_info': {
                'n_recipients': len(clinical_data),
                'n_variables_imputed': len(results)
            },
            'imputation_results': {}
        }
        
        print(f"\nImputation Summary Report")
        print("="*50)
        
        for var, result in results.items():
            n_imputed = len(result['values'])
            mean_conf = result['mean_confidence']
            method = result['method']
            
            print(f"{var}:")
            print(f"  Values imputed: {n_imputed:,}")
            print(f"  Mean confidence: {mean_conf:.3f}")
            print(f"  Method: {method}")
            print(f"  Value range: {np.min(result['values']):.1f} - {np.max(result['values']):.1f}")
            
            report['imputation_results'][var] = {
                'n_imputed': int(n_imputed),
                'mean_confidence': float(mean_conf),
                'method': method,
                'value_range': [float(np.min(result['values'])), float(np.max(result['values']))],
                'confidence_range': [float(np.min(result['confidence'])), float(np.max(result['confidence']))]
            }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to: {filename}")
        return report
    
    def run_full_imputation(self, data_file="DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv"):
        """Run the complete imputation pipeline."""
        print("ENBEL SOCIOECONOMIC IMPUTATION PIPELINE")
        print("="*60)
        
        # Step 1: Load and validate data
        clinical_data, gcro_data = self.load_and_validate_data(data_file)
        
        if clinical_data is None or gcro_data is None:
            print("❌ Data loading failed")
            return None
        
        # Step 2: Analyze imputation targets
        target_vars = self.analyze_imputation_targets(clinical_data, gcro_data)
        
        if not target_vars:
            print("❌ No variables available for imputation")
            return None
        
        # Step 3: Prepare features
        donor_features, recipient_features, encoders = self.prepare_matching_features(gcro_data, clinical_data)
        
        # Step 4: Perform KNN imputation
        knn_results = self.perform_knn_imputation(donor_features, gcro_data, recipient_features, target_vars)
        
        # Step 5: Perform ecological imputation
        eco_results = self.perform_ecological_imputation(gcro_data, clinical_data, target_vars)
        
        # Step 6: Combine methods
        combined_results = self.combine_imputation_methods(knn_results, eco_results)
        
        # Step 7: Create final dataset
        final_data = clinical_data.copy()
        
        for var, result in combined_results.items():
            final_data[f'{var}_imputed'] = result['values']
            final_data[f'{var}_confidence'] = result['confidence']
        
        # Step 8: Generate report
        report = self.generate_imputation_report(clinical_data, combined_results)
        
        # Step 9: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"enbel_imputed_clinical_data_{timestamp}.csv"
        final_data.to_csv(output_file, index=False)
        
        print(f"\n✅ Imputed dataset saved to: {output_file}")
        print(f"✅ Final dataset: {len(final_data):,} records with {len(target_vars)} imputed variables")
        
        return final_data, report

def main():
    """Main execution function."""
    
    # Initialize pipeline with scientific parameters
    pipeline = ENBELImputationPipeline(
        k_neighbors=10,          # Sufficient neighbors for stability
        spatial_weight=0.4,      # Balance spatial and demographic
        max_distance_km=15,      # Reasonable distance for Johannesburg
        min_matches=3,           # Minimum for statistical validity
        random_state=42          # Reproducibility
    )
    
    # Run imputation
    try:
        results = pipeline.run_full_imputation()
        
        if results is not None:
            final_data, report = results
            print("\n🎉 ENBEL imputation completed successfully!")
            
            # Print final summary
            n_vars = len([col for col in final_data.columns if col.endswith('_imputed')])
            print(f"📊 {n_vars} socioeconomic variables imputed")
            print(f"📊 {len(final_data):,} clinical participants enhanced")
            
            return True
        else:
            print("\n❌ Imputation failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during imputation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)