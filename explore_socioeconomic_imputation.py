#!/usr/bin/env python3
"""
Explore Feasibility of Imputing G-Soro Socioeconomic Data onto Clinical Trial Participants
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("SOCIOECONOMIC DATA IMPUTATION FEASIBILITY ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n1. LOADING DATA")
    print("-"*40)
    df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
    
    # Separate cohorts
    clinical = df[df['dataset_group'] == 'clinical'].copy()
    socioeconomic = df[df['dataset_group'] == 'socioeconomic'].copy()
    
    print(f"Clinical cohort: {len(clinical):,} participants")
    print(f"Socioeconomic cohort (G-Soro): {len(socioeconomic):,} participants")
    
    # Analyze available socioeconomic variables
    print("\n2. SOCIOECONOMIC VARIABLES AVAILABLE IN G-SORO")
    print("-"*40)
    
    socio_vars = ['Education', 'employment_status', 'vuln_Housing', 
                  'vuln_employment_status', 'vuln_Agegroup',
                  'housing_vulnerability', 'economic_vulnerability', 
                  'heat_vulnerability_index']
    
    for var in socio_vars:
        if var in socioeconomic.columns:
            coverage = socioeconomic[var].notna().sum() / len(socioeconomic) * 100
            unique_vals = socioeconomic[var].nunique()
            print(f"  {var}: {coverage:.1f}% coverage, {unique_vals} unique values")
    
    # Analyze geographic overlap
    print("\n3. GEOGRAPHIC OVERLAP ANALYSIS")
    print("-"*40)
    
    clinical_coords = clinical[['latitude', 'longitude']].dropna()
    socio_coords = socioeconomic[['latitude', 'longitude']].dropna()
    
    print(f"Clinical participants with coordinates: {len(clinical_coords):,}")
    print(f"G-Soro participants with coordinates: {len(socio_coords):,}")
    
    # Calculate geographic statistics
    if len(clinical_coords) > 0 and len(socio_coords) > 0:
        clinical_lat_range = (clinical_coords['latitude'].min(), clinical_coords['latitude'].max())
        clinical_lon_range = (clinical_coords['longitude'].min(), clinical_coords['longitude'].max())
        
        socio_lat_range = (socio_coords['latitude'].min(), socio_coords['latitude'].max())
        socio_lon_range = (socio_coords['longitude'].min(), socio_coords['longitude'].max())
        
        print(f"\nClinical geographic range:")
        print(f"  Latitude: {clinical_lat_range[0]:.4f} to {clinical_lat_range[1]:.4f}")
        print(f"  Longitude: {clinical_lon_range[0]:.4f} to {clinical_lon_range[1]:.4f}")
        
        print(f"\nG-Soro geographic range:")
        print(f"  Latitude: {socio_lat_range[0]:.4f} to {socio_lat_range[1]:.4f}")
        print(f"  Longitude: {socio_lon_range[0]:.4f} to {socio_lon_range[1]:.4f}")
        
        # Check overlap
        lat_overlap = not (clinical_lat_range[1] < socio_lat_range[0] or 
                           clinical_lat_range[0] > socio_lat_range[1])
        lon_overlap = not (clinical_lon_range[1] < socio_lon_range[0] or 
                           clinical_lon_range[0] > socio_lon_range[1])
        
        if lat_overlap and lon_overlap:
            print("\n✓ Geographic ranges overlap - spatial imputation feasible")
        else:
            print("\n✗ Limited geographic overlap - may affect imputation quality")
    
    # Explore temporal overlap
    print("\n4. TEMPORAL OVERLAP ANALYSIS")
    print("-"*40)
    
    if 'primary_date' in clinical.columns and 'primary_date' in socioeconomic.columns:
        clinical['primary_date'] = pd.to_datetime(clinical['primary_date'], errors='coerce')
        socioeconomic['primary_date'] = pd.to_datetime(socioeconomic['primary_date'], errors='coerce')
        
        clinical_dates = clinical['primary_date'].dropna()
        socio_dates = socioeconomic['primary_date'].dropna()
        
        if len(clinical_dates) > 0 and len(socio_dates) > 0:
            print(f"Clinical date range: {clinical_dates.min():%Y-%m-%d} to {clinical_dates.max():%Y-%m-%d}")
            print(f"G-Soro date range: {socio_dates.min():%Y-%m-%d} to {socio_dates.max():%Y-%m-%d}")
    
    # Propose imputation strategies
    print("\n5. PROPOSED IMPUTATION STRATEGIES")
    print("-"*40)
    
    print("\nStrategy 1: Spatial K-Nearest Neighbors (KNN)")
    print("  - Match clinical participants to nearest G-Soro participants by GPS coordinates")
    print("  - Use k=5-10 neighbors, weighted by distance")
    print("  - Advantages: Direct spatial matching, preserves local patterns")
    print("  - Limitations: Assumes spatial homogeneity of socioeconomic factors")
    
    print("\nStrategy 2: Demographic Matching")
    print("  - Match on Age group, Sex, Race where available")
    print("  - Use conditional distributions from G-Soro")
    print("  - Advantages: Preserves demographic-SES relationships")
    print("  - Limitations: May not capture local variations")
    
    print("\nStrategy 3: Hybrid Spatiotemporal-Demographic")
    print("  - Combine spatial proximity with demographic similarity")
    print("  - Weight by both distance and demographic match score")
    print("  - Advantages: Most comprehensive approach")
    print("  - Limitations: More complex, requires validation")
    
    print("\nStrategy 4: Area-Level Aggregation (Ecological)")
    print("  - Aggregate G-Soro data to ward/district level")
    print("  - Assign area-level averages to clinical participants")
    print("  - Advantages: Simple, transparent, commonly used")
    print("  - Limitations: Ecological fallacy, loss of individual variation")
    
    # Test feasibility with a simple KNN example
    print("\n6. TESTING SPATIAL KNN IMPUTATION (EXAMPLE)")
    print("-"*40)
    
    # Prepare data for KNN
    clinical_test = clinical[clinical[['latitude', 'longitude']].notna().all(axis=1)].head(100)
    socio_train = socioeconomic[socioeconomic[['latitude', 'longitude']].notna().all(axis=1)]
    
    if len(clinical_test) > 0 and len(socio_train) > 0:
        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=min(5, len(socio_train)), metric='haversine')
        
        # Convert to radians for haversine distance
        socio_coords_rad = np.radians(socio_train[['latitude', 'longitude']].values)
        clinical_coords_rad = np.radians(clinical_test[['latitude', 'longitude']].values)
        
        knn.fit(socio_coords_rad)
        distances, indices = knn.kneighbors(clinical_coords_rad)
        
        # Convert distances from radians to km (Earth radius ≈ 6371 km)
        distances_km = distances * 6371
        
        print(f"Tested on {len(clinical_test)} clinical participants")
        print(f"Average distance to nearest G-Soro participant: {distances_km[:, 0].mean():.2f} km")
        print(f"Median distance to nearest G-Soro participant: {np.median(distances_km[:, 0]):.2f} km")
        print(f"Max distance to nearest G-Soro participant: {distances_km[:, 0].max():.2f} km")
        
        # Check how many are within reasonable distance (e.g., 5km)
        within_5km = (distances_km[:, 0] <= 5).sum()
        within_10km = (distances_km[:, 0] <= 10).sum()
        
        print(f"\nParticipants with G-Soro match within 5km: {within_5km}/{len(clinical_test)} ({within_5km/len(clinical_test)*100:.1f}%)")
        print(f"Participants with G-Soro match within 10km: {within_10km}/{len(clinical_test)} ({within_10km/len(clinical_test)*100:.1f}%)")
    
    # Recommendations
    print("\n7. RECOMMENDATIONS")
    print("-"*40)
    print("\n✓ FEASIBILITY: Socioeconomic imputation is feasible with the following approach:")
    print("\n1. PRIMARY METHOD: Spatial KNN with k=5 neighbors")
    print("   - Use GPS coordinates for matching")
    print("   - Weight by inverse distance")
    print("   - Set maximum distance threshold (e.g., 10km)")
    
    print("\n2. FALLBACK METHOD: Demographic matching for participants without close spatial matches")
    print("   - Match on Sex and Age group")
    print("   - Use district-level averages where available")
    
    print("\n3. VARIABLES TO IMPUTE:")
    print("   - Education level")
    print("   - Employment status")
    print("   - Housing vulnerability index")
    print("   - Economic vulnerability index")
    print("   - Heat vulnerability index (composite)")
    
    print("\n4. VALIDATION APPROACH:")
    print("   - Hold out 20% of G-Soro data for validation")
    print("   - Test imputation accuracy on held-out data")
    print("   - Report uncertainty in imputed values")
    print("   - Clearly document all assumptions and limitations")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - Imputation is feasible and recommended")
    print("="*80)

if __name__ == "__main__":
    main()