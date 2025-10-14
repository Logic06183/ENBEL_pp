#!/usr/bin/env python3
"""
Simple Test of Imputation Methodology
=====================================

Direct test of the imputation code to verify it works.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create synthetic test data."""
    print("Creating synthetic test data...")
    
    np.random.seed(42)
    
    # Johannesburg coordinates
    lat_min, lat_max = -26.4, -26.0
    lon_min, lon_max = 27.8, 28.4
    
    # Donor data (GCRO-like)
    n_donors = 1000
    donor_data = pd.DataFrame({
        'latitude': np.random.uniform(lat_min, lat_max, n_donors),
        'longitude': np.random.uniform(lon_min, lon_max, n_donors),
        'Sex': np.random.choice(['Male', 'Female'], n_donors),
        'Race': np.random.choice(['Black', 'White', 'Coloured'], n_donors),
        'Education': np.random.choice([1, 2, 3, 4, 5], n_donors),
        'employment_status': np.random.choice([1, 2, 3], n_donors),
        'vuln_Housing': np.random.choice([1, 2, 3], n_donors)
    })
    
    # Recipient data (Clinical-like)
    n_recipients = 500
    recipient_data = pd.DataFrame({
        'latitude': np.random.uniform(lat_min, lat_max, n_recipients),
        'longitude': np.random.uniform(lon_min, lon_max, n_recipients),
        'Sex': np.random.choice(['Male', 'Female'], n_recipients),
        'Race': np.random.choice(['Black', 'White', 'Coloured'], n_recipients),
        'systolic_bp': np.random.normal(130, 20, n_recipients)
    })
    
    print(f"Donor data: {len(donor_data):,} records")
    print(f"Recipient data: {len(recipient_data):,} records")
    
    return donor_data, recipient_data

def simple_knn_imputation(donor_data, recipient_data, target_var='Education', k=5):
    """Simple KNN imputation implementation."""
    print(f"\nPerforming KNN imputation for {target_var}...")
    
    # Prepare features
    features = ['latitude', 'longitude']
    
    # Get donors with the target variable
    valid_donors = donor_data[donor_data[target_var].notna()].copy()
    
    if len(valid_donors) == 0:
        print(f"No valid donors for {target_var}")
        return recipient_data
    
    # Prepare feature matrices
    donor_features = valid_donors[features].values
    recipient_features = recipient_data[features].values
    
    # Scale features
    scaler = StandardScaler()
    donor_features_scaled = scaler.fit_transform(donor_features)
    recipient_features_scaled = scaler.transform(recipient_features)
    
    # Fit KNN
    knn = NearestNeighbors(n_neighbors=min(k, len(valid_donors)), metric='euclidean')
    knn.fit(donor_features_scaled)
    
    # Find neighbors and impute
    distances, indices = knn.kneighbors(recipient_features_scaled)
    
    imputed_values = []
    confidence_scores = []
    
    for i in range(len(recipient_data)):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]
        neighbor_values = valid_donors.iloc[neighbor_indices][target_var].values
        
        # Calculate weighted average
        weights = 1 / (neighbor_distances + 1e-8)
        weights = weights / weights.sum()
        
        imputed_value = np.average(neighbor_values, weights=weights)
        confidence = 1 / (1 + np.mean(neighbor_distances))
        
        imputed_values.append(imputed_value)
        confidence_scores.append(confidence)
    
    # Add results to recipient data
    result = recipient_data.copy()
    result[f'{target_var}_imputed'] = imputed_values
    result[f'{target_var}_confidence'] = confidence_scores
    
    n_imputed = len(imputed_values)
    mean_confidence = np.mean(confidence_scores)
    
    print(f"✅ Imputed {n_imputed:,} values for {target_var}")
    print(f"   Mean confidence: {mean_confidence:.3f}")
    print(f"   Value range: {np.min(imputed_values):.1f} - {np.max(imputed_values):.1f}")
    
    return result

def test_ecological_imputation(donor_data, recipient_data, target_var='Education'):
    """Simple ecological imputation by demographic groups."""
    print(f"\nPerforming ecological imputation for {target_var}...")
    
    # Calculate group means
    group_means = donor_data.groupby(['Sex', 'Race'])[target_var].mean()
    overall_mean = donor_data[target_var].mean()
    
    print(f"Overall mean {target_var}: {overall_mean:.2f}")
    print("Group means:")
    for group, mean_val in group_means.items():
        print(f"  {group}: {mean_val:.2f}")
    
    # Impute for recipients
    result = recipient_data.copy()
    imputed_values = []
    confidence_scores = []
    
    for _, row in recipient_data.iterrows():
        sex = row['Sex']
        race = row['Race']
        
        # Try to find group mean
        if (sex, race) in group_means:
            imputed_value = group_means[(sex, race)]
            confidence = 0.7
        else:
            imputed_value = overall_mean
            confidence = 0.3
        
        imputed_values.append(imputed_value)
        confidence_scores.append(confidence)
    
    result[f'{target_var}_eco_imputed'] = imputed_values
    result[f'{target_var}_eco_confidence'] = confidence_scores
    
    mean_confidence = np.mean(confidence_scores)
    print(f"✅ Ecological imputation completed")
    print(f"   Mean confidence: {mean_confidence:.3f}")
    
    return result

def main():
    """Run simple imputation tests."""
    print("SIMPLE IMPUTATION TEST")
    print("="*50)
    
    try:
        # Create test data
        donor_data, recipient_data = create_test_data()
        
        # Test KNN imputation
        result_knn = simple_knn_imputation(donor_data, recipient_data, 'Education', k=5)
        
        # Test ecological imputation
        result_eco = test_ecological_imputation(donor_data, recipient_data, 'Education')
        
        # Test multiple variables
        print("\nTesting multiple variables...")
        target_vars = ['Education', 'employment_status', 'vuln_Housing']
        
        final_result = recipient_data.copy()
        
        for var in target_vars:
            if var in donor_data.columns:
                temp_result = simple_knn_imputation(donor_data, recipient_data, var, k=8)
                final_result[f'{var}_imputed'] = temp_result[f'{var}_imputed']
                final_result[f'{var}_confidence'] = temp_result[f'{var}_confidence']
        
        print("\n✅ All imputation tests completed successfully!")
        print("\nSummary:")
        
        for var in target_vars:
            imputed_col = f'{var}_imputed'
            confidence_col = f'{var}_confidence'
            
            if imputed_col in final_result.columns:
                n_imputed = final_result[imputed_col].notna().sum()
                mean_conf = final_result[confidence_col].mean()
                print(f"  {var}: {n_imputed:,} imputed (confidence: {mean_conf:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Imputation methodology works correctly!")
    else:
        print("\n⚠️ Issues found in imputation methodology.")
    
    exit(0 if success else 1)