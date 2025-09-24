#!/usr/bin/env python3
"""
FIXED Multi-dimensional Socioeconomic Imputation
=================================================

Fixes identified in the original matching:
1. G-SORO Sex codes are strings: '1.0' = Male, '2.0' = Female
2. G-SORO Race codes need mapping: 1.0=Black, 2.0=White, 3.0=Coloured, 4.0=Asian
3. Many studies use clinic coordinates instead of participant home addresses
4. Need to relax spatial constraints and rely more on demographic matching

This revised approach uses:
- 80% demographic matching, 20% spatial (since coordinates may be clinic-based)
- Proper data type handling and code mapping
- More lenient distance thresholds
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def calculate_demographic_similarity_fixed(clinical_row, socio_data):
    """
    Fixed demographic similarity calculation with proper data type handling
    """
    scores = np.ones(len(socio_data))
    
    # Sex matching with proper string handling
    if pd.notna(clinical_row.get('Sex')) and clinical_row['Sex'] in ['Male', 'Female']:
        # Map G-SORO sex codes: '1.0' = Male, '2.0' = Female
        socio_sex_mapped = socio_data['Sex'].map({'1.0': 'Male', '2.0': 'Female'})
        sex_match = (socio_sex_mapped == clinical_row['Sex']).fillna(False).astype(float).values
        scores *= (0.3 + 0.7 * sex_match)  # 70% penalty for sex mismatch
    
    # Race matching with proper code mapping
    if 'Race' in clinical_row and pd.notna(clinical_row['Race']):
        # Map G-SORO race codes based on South African demographics
        race_mapping = {
            '1.0': 'Black',      # Most common in G-SORO
            '2.0': 'White', 
            '3.0': 'Coloured',
            '4.0': 'Asian'
        }
        
        if 'Race' in socio_data.columns:
            socio_race_mapped = socio_data['Race'].map(race_mapping)
            race_match = (socio_race_mapped == clinical_row['Race']).fillna(False).astype(float).values
            scores *= (0.4 + 0.6 * race_match)  # 60% penalty for race mismatch
    
    return scores

def fixed_spatial_temporal_matching(clinical_df, socio_df, k_neighbors=15, max_distance_km=25):
    """
    Fixed matching algorithm with proper data handling
    """
    
    print("="*80)
    print("FIXED MULTI-DIMENSIONAL SOCIOECONOMIC IMPUTATION")
    print("="*80)
    
    # Prepare socioeconomic variables to impute
    socio_vars = ['Education', 'employment_status', 'vuln_Housing', 
                  'vuln_employment_status', 'housing_vulnerability',
                  'economic_vulnerability', 'heat_vulnerability_index']
    
    available_vars = [v for v in socio_vars if v in socio_df.columns]
    print(f"\nVariables to impute: {available_vars}")
    
    # Initialize imputed dataframe
    imputed_df = clinical_df.copy()
    for var in available_vars:
        imputed_df[f'{var}_imputed'] = np.nan
        imputed_df[f'{var}_confidence'] = np.nan
    
    # Debug data types
    print("\nData type verification:")
    print(f"Clinical Sex sample: {clinical_df['Sex'].dropna().iloc[0]} (type: {type(clinical_df['Sex'].dropna().iloc[0])})")
    print(f"G-SORO Sex sample: {socio_df['Sex'].dropna().iloc[0]} (type: {type(socio_df['Sex'].dropna().iloc[0])})")
    
    # Process in smaller batches
    batch_size = 50
    n_batches = len(clinical_df) // batch_size + 1
    
    print(f"\nProcessing {len(clinical_df)} clinical participants in {n_batches} batches...")
    print("Using 80% demographic + 20% spatial weighting (accounting for clinic coordinates)")
    
    matched_count = 0
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(clinical_df))
        
        if start_idx >= len(clinical_df):
            break
            
        batch = clinical_df.iloc[start_idx:end_idx]
        
        # Get socio participants with valid coordinates
        valid_socio = socio_df[socio_df[['latitude', 'longitude']].notna().all(axis=1)].copy()
        
        if len(valid_socio) == 0:
            continue
        
        for idx, clinical_row in batch.iterrows():
            # Skip if no coordinates
            if pd.isna(clinical_row['latitude']) or pd.isna(clinical_row['longitude']):
                continue
            
            # Step 1: Calculate demographic similarity scores
            demo_scores = calculate_demographic_similarity_fixed(clinical_row, valid_socio)
            
            # Step 2: Calculate spatial distances
            clinical_coord = np.radians([[clinical_row['latitude'], clinical_row['longitude']]])
            socio_coords = np.radians(valid_socio[['latitude', 'longitude']].values)
            
            # Haversine distance
            lat_diff = socio_coords[:, 0] - clinical_coord[0, 0]
            lon_diff = socio_coords[:, 1] - clinical_coord[0, 1]
            a = np.sin(lat_diff/2)**2 + np.cos(clinical_coord[0, 0]) * np.cos(socio_coords[:, 0]) * np.sin(lon_diff/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances_km = 6371 * c
            
            # Step 3: Combined scoring (80% demographic, 20% spatial)
            distance_scores = np.exp(-distances_km / 8)  # More lenient distance decay
            combined_scores = 0.8 * demo_scores + 0.2 * distance_scores
            
            # Apply hard distance cutoff (more lenient)
            combined_scores[distances_km > max_distance_km] = 0
            
            # Step 4: Select top k matches with lower threshold
            if np.sum(combined_scores > 0.2) < 3:  # Lower threshold
                continue
                
            top_k_idx = np.argsort(combined_scores)[-k_neighbors:][::-1]
            top_k_scores = combined_scores[top_k_idx]
            
            # Filter out poor matches
            good_matches = top_k_scores > 0.2  # Lower threshold
            if np.sum(good_matches) < 3:
                continue
                
            top_k_idx = top_k_idx[good_matches]
            top_k_scores = top_k_scores[good_matches]
            
            # Step 5: Impute variables
            matched_count += 1
            
            for var in available_vars:
                var_values = valid_socio.iloc[top_k_idx][var]
                valid_values = var_values[var_values.notna()]
                
                if len(valid_values) == 0:
                    continue
                
                # Determine if categorical or continuous
                if var_values.dtype == 'object' or valid_values.nunique() < 10:
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
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed batch {batch_idx + 1}/{n_batches} - Matched so far: {matched_count}")
    
    print(f"\nTotal successfully matched: {matched_count:,}")
    return imputed_df

def main():
    print("\nLoading data...")
    df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    # Separate cohorts
    clinical_df = df[df['dataset_group'] == 'clinical'].copy()
    socio_df = df[df['dataset_group'] == 'socioeconomic'].copy()
    
    print(f"Clinical cohort: {len(clinical_df):,} participants")
    print(f"Socioeconomic cohort: {len(socio_df):,} participants")
    
    # Test the mapping first
    print("\nTesting fixed mappings:")
    test_sex_map = socio_df['Sex'].map({'1.0': 'Male', '2.0': 'Female'})
    print("Sex mapping result:", test_sex_map.value_counts())
    
    race_mapping = {'1.0': 'Black', '2.0': 'White', '3.0': 'Coloured', '4.0': 'Asian'}
    test_race_map = socio_df['Race'].map(race_mapping)
    print("Race mapping result:", test_race_map.value_counts())
    
    # Perform fixed imputation
    imputed_df = fixed_spatial_temporal_matching(clinical_df, socio_df, 
                                               k_neighbors=15, max_distance_km=25)
    
    # Validate results
    available_vars = [v for v in ['Education', 'employment_status', 'vuln_Housing', 
                                  'vuln_employment_status', 'housing_vulnerability',
                                  'economic_vulnerability', 'heat_vulnerability_index'] 
                     if f'{v}_imputed' in imputed_df.columns]
    
    print("\n" + "="*80)
    print("IMPROVED IMPUTATION RESULTS")
    print("="*80)
    
    total_coverage = 0
    for var in available_vars:
        imputed_col = f'{var}_imputed'
        confidence_col = f'{var}_confidence'
        
        n_imputed = imputed_df[imputed_col].notna().sum()
        coverage = n_imputed / len(imputed_df) * 100
        total_coverage += coverage
        
        if n_imputed > 0:
            avg_confidence = imputed_df[confidence_col].mean()
            high_conf = (imputed_df[confidence_col] > 0.5).sum()
            
            print(f"\n{var}:")
            print(f"  Coverage: {coverage:.1f}% ({n_imputed:,}/{len(imputed_df):,})")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  High confidence (>0.5): {high_conf:,} ({high_conf/n_imputed*100:.1f}%)")
    
    avg_coverage = total_coverage / len(available_vars)
    print(f"\nOverall average coverage: {avg_coverage:.1f}%")
    
    # Save results
    output_file = 'CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv'
    imputed_df['imputation_method'] = 'fixed_multidimensional_matching'
    imputed_df['imputation_version'] = '2.0_fixed'
    imputed_df['imputation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    imputed_df.to_csv(output_file, index=False)
    print(f"\nFixed imputed dataset saved to: {output_file}")

if __name__ == "__main__":
    main()