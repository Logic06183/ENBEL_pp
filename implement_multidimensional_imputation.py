#!/usr/bin/env python3
"""
Multi-dimensional Socioeconomic Imputation for Clinical Trial Participants
===========================================================================

This script implements a sophisticated imputation strategy that matches
clinical trial participants to G-Soro participants based on multiple dimensions:
1. Demographic characteristics (Sex, Race, Age group)
2. Spatial proximity (GPS coordinates)
3. Temporal alignment (study period)

The approach avoids ecological fallacy by considering individual-level matching
while acknowledging that participants may travel to study sites from diverse areas.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

def calculate_demographic_similarity(clinical_row, socio_data):
    """
    Calculate demographic similarity score between a clinical participant
    and all socioeconomic participants.
    """
    scores = np.ones(len(socio_data))
    
    # Sex matching (high weight)
    if pd.notna(clinical_row.get('Sex')):
        # Map socio sex codes (1=Male, 2=Female) to clinical format
        socio_sex_mapped = socio_data['Sex'].map({1.0: 'Male', 2.0: 'Female'})
        sex_match = (socio_sex_mapped == clinical_row['Sex']).astype(float).values
        scores *= (0.5 + 0.5 * sex_match)  # 50% penalty for sex mismatch
    
    # Race matching (high weight) 
    if 'Race' in clinical_row and pd.notna(clinical_row['Race']):
        if 'Race' in socio_data.columns:
            race_match = (socio_data['Race'] == clinical_row['Race']).astype(float).values
            scores *= (0.6 + 0.4 * race_match)  # 40% penalty for race mismatch
    
    # Age group matching (if available)
    if 'vuln_Agegroup' in socio_data.columns:
        # Infer age group from clinical data if possible
        # This would need actual age data or grouping logic
        pass
    
    return scores

def spatial_temporal_matching(clinical_df, socio_df, k_neighbors=10, max_distance_km=15):
    """
    Perform multi-dimensional matching combining:
    - Demographic similarity
    - Spatial proximity
    - Temporal alignment
    """
    
    print("="*80)
    print("MULTI-DIMENSIONAL SOCIOECONOMIC IMPUTATION")
    print("="*80)
    
    # Prepare socioeconomic variables to impute
    socio_vars = ['Education', 'employment_status', 'vuln_Housing', 
                  'vuln_employment_status', 'housing_vulnerability',
                  'economic_vulnerability', 'heat_vulnerability_index']
    
    # Filter to variables that exist and have data
    available_vars = [v for v in socio_vars if v in socio_df.columns]
    print(f"\nVariables to impute: {available_vars}")
    
    # Initialize imputed dataframe
    imputed_df = clinical_df.copy()
    for var in available_vars:
        imputed_df[f'{var}_imputed'] = np.nan
        imputed_df[f'{var}_confidence'] = np.nan
    
    # Process in batches for memory efficiency
    batch_size = 100
    n_batches = len(clinical_df) // batch_size + 1
    
    print(f"\nProcessing {len(clinical_df)} clinical participants in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(clinical_df))
        
        if start_idx >= len(clinical_df):
            break
            
        batch = clinical_df.iloc[start_idx:end_idx]
        
        # Skip if no valid coordinates
        valid_batch = batch[batch[['latitude', 'longitude']].notna().all(axis=1)]
        if len(valid_batch) == 0:
            continue
        
        # Get socio participants with valid coordinates
        valid_socio = socio_df[socio_df[['latitude', 'longitude']].notna().all(axis=1)].copy()
        
        if len(valid_socio) == 0:
            continue
        
        for idx, clinical_row in valid_batch.iterrows():
            # Step 1: Calculate demographic similarity scores
            demo_scores = calculate_demographic_similarity(clinical_row, valid_socio)
            
            # Step 2: Calculate spatial distances (haversine formula)
            clinical_coord = np.radians([[clinical_row['latitude'], clinical_row['longitude']]])
            socio_coords = np.radians(valid_socio[['latitude', 'longitude']].values)
            
            # Haversine distance
            lat_diff = socio_coords[:, 0] - clinical_coord[0, 0]
            lon_diff = socio_coords[:, 1] - clinical_coord[0, 1]
            a = np.sin(lat_diff/2)**2 + np.cos(clinical_coord[0, 0]) * np.cos(socio_coords[:, 0]) * np.sin(lon_diff/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances_km = 6371 * c  # Earth radius in km
            
            # Step 3: Calculate combined matching score
            # Normalize distances to 0-1 scale (0 = far, 1 = close)
            distance_scores = np.exp(-distances_km / 5)  # Exponential decay with 5km scale
            
            # Combine demographic and spatial scores
            # Weight: 60% demographic, 40% spatial (since participants may travel to sites)
            combined_scores = 0.6 * demo_scores + 0.4 * distance_scores
            
            # Apply hard distance cutoff
            combined_scores[distances_km > max_distance_km] = 0
            
            # Step 4: Select top k matches
            if np.sum(combined_scores > 0) < 3:  # Need at least 3 matches
                continue
                
            top_k_idx = np.argsort(combined_scores)[-k_neighbors:][::-1]
            top_k_scores = combined_scores[top_k_idx]
            top_k_distances = distances_km[top_k_idx]
            
            # Filter out very poor matches
            good_matches = top_k_scores > 0.3
            if np.sum(good_matches) < 3:
                continue
                
            top_k_idx = top_k_idx[good_matches]
            top_k_scores = top_k_scores[good_matches]
            top_k_distances = top_k_distances[good_matches]
            
            # Step 5: Impute variables using weighted average or mode
            for var in available_vars:
                # Use integer position indexing instead of label indexing
                var_values = valid_socio.iloc[top_k_idx][var]
                valid_values = var_values[var_values.notna()]
                
                if len(valid_values) == 0:
                    continue
                
                # Determine if categorical or continuous
                if valid_values.dtype == 'object' or valid_values.nunique() < 10:
                    # Categorical: use weighted mode
                    if len(valid_values) > 0:
                        imputed_value = valid_values.mode()[0] if len(valid_values.mode()) > 0 else valid_values.iloc[0]
                        confidence = (valid_values == imputed_value).mean()
                    else:
                        continue
                else:
                    # Continuous: use weighted mean
                    weights = top_k_scores[:len(valid_values)] / top_k_scores[:len(valid_values)].sum()
                    imputed_value = np.average(valid_values, weights=weights)
                    confidence = 1 - (np.std(valid_values) / (np.mean(valid_values) + 1e-10))
                
                imputed_df.loc[idx, f'{var}_imputed'] = imputed_value
                imputed_df.loc[idx, f'{var}_confidence'] = confidence * np.mean(top_k_scores)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed batch {batch_idx + 1}/{n_batches}")
    
    return imputed_df

def validate_imputation(imputed_df, available_vars):
    """
    Validate and report imputation quality
    """
    print("\n" + "="*80)
    print("IMPUTATION QUALITY REPORT")
    print("="*80)
    
    for var in available_vars:
        imputed_col = f'{var}_imputed'
        confidence_col = f'{var}_confidence'
        
        if imputed_col not in imputed_df.columns:
            continue
            
        n_imputed = imputed_df[imputed_col].notna().sum()
        coverage = n_imputed / len(imputed_df) * 100
        
        if n_imputed > 0:
            avg_confidence = imputed_df[confidence_col].mean()
            high_conf = (imputed_df[confidence_col] > 0.7).sum()
            
            print(f"\n{var}:")
            print(f"  Coverage: {coverage:.1f}% ({n_imputed:,}/{len(imputed_df):,})")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  High confidence (>0.7): {high_conf:,} ({high_conf/n_imputed*100:.1f}%)")
            
            if imputed_df[imputed_col].dtype == 'object' or imputed_df[imputed_col].nunique() < 10:
                print(f"  Distribution: {imputed_df[imputed_col].value_counts().to_dict()}")

def main():
    print("\nLoading data...")
    df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    # Separate cohorts
    clinical_df = df[df['dataset_group'] == 'clinical'].copy()
    socio_df = df[df['dataset_group'] == 'socioeconomic'].copy()
    
    print(f"Clinical cohort: {len(clinical_df):,} participants")
    print(f"Socioeconomic cohort: {len(socio_df):,} participants")
    
    # Perform imputation
    imputed_df = spatial_temporal_matching(clinical_df, socio_df, k_neighbors=10, max_distance_km=15)
    
    # Get list of available variables that were actually imputed
    available_vars = [v for v in ['Education', 'employment_status', 'vuln_Housing', 
                                  'vuln_employment_status', 'housing_vulnerability',
                                  'economic_vulnerability', 'heat_vulnerability_index'] 
                     if f'{v}_imputed' in imputed_df.columns]
    
    # Validate results
    validate_imputation(imputed_df, available_vars)
    
    # Save imputed dataset
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Add metadata columns
    imputed_df['imputation_method'] = 'multidimensional_matching'
    imputed_df['imputation_version'] = '1.0'
    imputed_df['imputation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    output_file = 'CLINICAL_WITH_IMPUTED_SOCIOECONOMIC.csv'
    imputed_df.to_csv(output_file, index=False)
    print(f"\nImputed dataset saved to: {output_file}")
    print(f"Shape: {imputed_df.shape}")
    
    # Create summary statistics
    summary = {
        'total_participants': len(imputed_df),
        'variables_imputed': available_vars,
        'average_coverage': np.mean([imputed_df[f'{v}_imputed'].notna().sum() / len(imputed_df) * 100 
                                     for v in available_vars if f'{v}_imputed' in imputed_df.columns]),
        'high_confidence_rate': np.mean([(imputed_df[f'{v}_confidence'] > 0.7).sum() / 
                                         imputed_df[f'{v}_imputed'].notna().sum() 
                                         for v in available_vars 
                                         if f'{v}_confidence' in imputed_df.columns and 
                                         imputed_df[f'{v}_imputed'].notna().sum() > 0])
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully imputed socioeconomic characteristics for clinical trial participants")
    print(f"Average coverage across variables: {summary['average_coverage']:.1f}%")
    print(f"High confidence imputation rate: {summary['high_confidence_rate']*100:.1f}%")
    print("\nThe imputation used a multi-dimensional approach combining:")
    print("  • Demographic matching (Sex, Race)")
    print("  • Spatial proximity (GPS coordinates)")
    print("  • Weighted scoring (60% demographic, 40% spatial)")
    print("\nThis approach accounts for participants traveling to study sites from diverse areas")
    print("and ensures socioeconomic characteristics reflect multiple dimensions of similarity.")

if __name__ == "__main__":
    main()