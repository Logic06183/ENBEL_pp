#!/usr/bin/env python3
"""
CORRECTED Multi-dimensional Socioeconomic Imputation Methodology
================================================================

The correct approach for ENBEL imputation:

1. GCRO Cohort (~9,102 participants): Has socioeconomic variables like Education, 
   employment_status, vuln_Housing but NO health biomarkers

2. Clinical Cohort (~9,103 participants): Has health biomarkers like fasting glucose,
   blood pressure, CD4 count but MISSING socioeconomic variables

3. Imputation Goal: Transfer socioeconomic data from GCRO participants to 
   Clinical participants using spatial proximity and demographic similarity

This is NOT simple sex/race matching - it's sophisticated spatial-demographic 
matching using actual GCRO survey data.

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_imputation_structure():
    """Analyze the actual data structure for imputation"""
    print("="*80)
    print("CORRECTED ENBEL IMPUTATION METHODOLOGY ANALYSIS")
    print("="*80)
    
    # Load data
    data = pd.read_csv("DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv", low_memory=False)
    
    # Separate cohorts
    clinical_cohort = data[data['data_source'].notna()].copy()  # Has health data
    gcro_cohort = data[data['data_source'].isna()].copy()      # Has socioeconomic data
    
    print(f"Total participants: {len(data):,}")
    print(f"Clinical cohort (health biomarkers): {len(clinical_cohort):,}")
    print(f"GCRO cohort (socioeconomic data): {len(gcro_cohort):,}")
    
    # Analyze what each cohort has
    print("\n" + "="*50)
    print("GCRO COHORT VARIABLES (Source for Imputation)")
    print("="*50)
    
    gcro_socioeconomic_vars = [
        'Education', 'employment_status', 'vuln_Housing'
    ]
    
    for var in gcro_socioeconomic_vars:
        n_available = gcro_cohort[var].notna().sum()
        pct_available = (n_available / len(gcro_cohort)) * 100
        print(f"{var:25} {n_available:5,} participants ({pct_available:5.1f}%)")
    
    print("\n" + "="*50)
    print("CLINICAL COHORT BIOMARKERS (Target for Imputation)")
    print("="*50)
    
    clinical_biomarkers = [
        'FASTING GLUCOSE', 'systolic blood pressure', 'CD4 cell count (cells/µL)',
        'FASTING LDL', 'Hemoglobin (g/dL)'
    ]
    
    for var in clinical_biomarkers:
        if var in clinical_cohort.columns:
            n_available = clinical_cohort[var].notna().sum()
            pct_available = (n_available / len(clinical_cohort)) * 100
            print(f"{var:25} {n_available:5,} participants ({pct_available:5.1f}%)")
    
    return clinical_cohort, gcro_cohort

def design_correct_imputation_method(clinical_cohort, gcro_cohort):
    """Design the correct imputation methodology"""
    print("\n" + "="*60)
    print("CORRECTED IMPUTATION METHODOLOGY")
    print("="*60)
    
    # Variables to impute (from GCRO to Clinical)
    imputation_vars = ['Education', 'employment_status', 'vuln_Housing']
    
    # Matching variables (available in both cohorts)
    demographic_vars = ['Sex', 'Race']
    spatial_vars = ['latitude', 'longitude']
    
    print("STEP 1: Multi-dimensional Matching Strategy")
    print("-" * 40)
    print("Target Variables (GCRO → Clinical):")
    for var in imputation_vars:
        gcro_available = gcro_cohort[var].notna().sum()
        clinical_missing = clinical_cohort[var].isna().sum()
        print(f"  • {var}: {gcro_available:,} GCRO donors → {clinical_missing:,} Clinical recipients")
    
    print("\nMatching Dimensions:")
    print("  • Spatial: Geographic proximity (lat/lon coordinates)")
    print("  • Demographic: Sex and Race similarity")
    print("  • Temporal: Collection date alignment (if available)")
    
    print("\nSTEP 2: Multi-dimensional Distance Calculation")
    print("-" * 40)
    print("  • Geographic Distance: Haversine distance between coordinates")
    print("  • Demographic Distance: Categorical mismatch penalties")
    print("  • Combined Weighting: Optimize spatial vs demographic importance")
    
    print("\nSTEP 3: K-Nearest Neighbors Imputation")
    print("-" * 40)
    print("  • For each Clinical participant:")
    print("    - Find k=5-15 most similar GCRO participants")
    print("    - Weight by multi-dimensional similarity")
    print("    - Impute socioeconomic variables using weighted average/mode")
    print("    - Calculate confidence scores based on match quality")
    
    print("\nSTEP 4: Quality Control and Validation")
    print("-" * 40)
    print("  • Spatial constraint: Maximum distance threshold (e.g., 50km)")
    print("  • Demographic penalty: Higher weight for exact sex/race matches")
    print("  • Missing data handling: Graceful degradation for sparse areas")
    print("  • Confidence scoring: Lower confidence for poor matches")

def demonstrate_correct_workflow():
    """Demonstrate the correct imputation workflow"""
    print("\n" + "="*60)
    print("CORRECT IMPUTATION WORKFLOW EXAMPLE")
    print("="*60)
    
    # Load actual data
    clinical_cohort, gcro_cohort = analyze_imputation_structure()
    
    # Example: Find GCRO participants who can donate Education data
    gcro_with_education = gcro_cohort[gcro_cohort['Education'].notna()].copy()
    clinical_need_education = clinical_cohort[clinical_cohort['Education'].isna()].copy()
    
    print(f"\nExample: Education Variable Imputation")
    print(f"  • GCRO participants with Education data: {len(gcro_with_education):,}")
    print(f"  • Clinical participants needing Education: {len(clinical_need_education):,}")
    
    # Check spatial coverage
    if 'latitude' in gcro_with_education.columns and 'longitude' in gcro_with_education.columns:
        gcro_spatial = gcro_with_education[['latitude', 'longitude']].dropna()
        clinical_spatial = clinical_need_education[['latitude', 'longitude']].dropna()
        
        print(f"  • GCRO participants with coordinates: {len(gcro_spatial):,}")
        print(f"  • Clinical participants with coordinates: {len(clinical_spatial):,}")
        
        if len(gcro_spatial) > 0 and len(clinical_spatial) > 0:
            # Calculate spatial coverage
            gcro_lat_range = (gcro_spatial['latitude'].min(), gcro_spatial['latitude'].max())
            gcro_lon_range = (gcro_spatial['longitude'].min(), gcro_spatial['longitude'].max())
            
            print(f"  • GCRO spatial coverage:")
            print(f"    - Latitude: {gcro_lat_range[0]:.3f} to {gcro_lat_range[1]:.3f}")
            print(f"    - Longitude: {gcro_lon_range[0]:.3f} to {gcro_lon_range[1]:.3f}")
    
    # Check demographic distribution
    print(f"\nDemographic Matching Potential:")
    for var in ['Sex', 'Race']:
        if var in gcro_with_education.columns and var in clinical_need_education.columns:
            gcro_dist = gcro_with_education[var].value_counts(dropna=False)
            clinical_dist = clinical_need_education[var].value_counts(dropna=False)
            
            print(f"  • {var} distribution:")
            print(f"    GCRO donors: {dict(gcro_dist)}")
            print(f"    Clinical recipients: {dict(clinical_dist)}")

def main():
    """Main analysis function"""
    # Analyze the correct structure
    clinical_cohort, gcro_cohort = analyze_imputation_structure()
    
    # Design correct methodology
    design_correct_imputation_method(clinical_cohort, gcro_cohort)
    
    # Demonstrate workflow
    demonstrate_correct_workflow()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR IMPUTATION METHODOLOGY")
    print("="*80)
    print("1. This is NOT simple demographic matching")
    print("2. GCRO cohort provides rich socioeconomic data (~9,100 participants)")
    print("3. Clinical cohort needs this data for ~9,100 health participants")
    print("4. Spatial proximity is crucial for accurate imputation")
    print("5. Multi-dimensional matching ensures robust imputation")
    print("6. Quality control and confidence scoring are essential")
    print("\nThis methodology enables combining health biomarkers with")
    print("socioeconomic vulnerabilities for comprehensive climate-health analysis!")

if __name__ == "__main__":
    main()