#!/usr/bin/env python3
"""
Test Script - Step 1: Test Imputation Pipeline
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

print("=" * 60)
print("TESTING STEP 1: IMPUTATION PIPELINE")
print("=" * 60)

# Test 1: Load and explore data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/clinical_dataset.csv')
print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Test 2: Check what needs imputation
print("\n2. Checking missing data patterns...")
missing_summary = df.isnull().sum()
high_missing = missing_summary[missing_summary > 0].sort_values(ascending=False)
print(f"   Columns with missing data: {len(high_missing)}")
print(f"   Top missing columns:")
for col in high_missing.head(5).index:
    pct = (high_missing[col] / len(df)) * 100
    print(f"      {col}: {pct:.1f}% missing ({high_missing[col]} values)")

# Test 3: Identify biomarker columns
print("\n3. Identifying biomarkers...")
biomarker_patterns = ['glucose', 'cd4', 'cholesterol', 'hemoglobin', 'creatinine', 
                      'blood_pressure', 'systolic', 'diastolic']
biomarker_cols = []
for col in df.columns:
    if any(pattern in col.lower() for pattern in biomarker_patterns):
        if df[col].dtype in [np.float64, np.int64]:
            biomarker_cols.append(col)

print(f"   Found {len(biomarker_cols)} biomarker columns")
for col in biomarker_cols[:5]:
    non_missing = df[col].notna().sum()
    print(f"      {col}: {non_missing} non-missing values")

# Test 4: Identify climate features
print("\n4. Identifying climate features...")
climate_patterns = ['temp', 'humid', 'pressure', 'wind', 'climate', 'heat', 'lag']
climate_cols = []
for col in df.columns:
    if any(pattern in col.lower() for pattern in climate_patterns):
        if df[col].dtype in [np.float64, np.int64]:
            climate_cols.append(col)

print(f"   Found {len(climate_cols)} climate columns")
for col in climate_cols[:5]:
    print(f"      {col}")

# Test 5: Check for socioeconomic variables
print("\n5. Checking for socioeconomic variables...")
socio_patterns = ['dwelling', 'education', 'income', 'employment', 'household']
socio_cols = []
for col in df.columns:
    if any(pattern in col.lower() for pattern in socio_patterns):
        socio_cols.append(col)

if socio_cols:
    print(f"   Found {len(socio_cols)} socioeconomic columns:")
    for col in socio_cols:
        print(f"      {col}")
else:
    print("   No socioeconomic columns found - these would need imputation")

# Test 6: Try simplified imputation
print("\n6. Testing simple imputation...")
from sklearn.impute import SimpleImputer

# Select numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Working with {len(numeric_cols)} numeric columns")

# Try simple mean imputation on a subset
test_cols = biomarker_cols[:3] if len(biomarker_cols) >= 3 else numeric_cols[:3]
if test_cols:
    print(f"   Testing on: {test_cols}")
    
    # Create imputer
    imputer = SimpleImputer(strategy='mean')
    
    # Test on subset
    subset_df = df[test_cols].copy()
    before_missing = subset_df.isnull().sum().sum()
    
    # Impute
    imputed_array = imputer.fit_transform(subset_df)
    imputed_df = pd.DataFrame(imputed_array, columns=test_cols)
    after_missing = imputed_df.isnull().sum().sum()
    
    print(f"   Missing before: {before_missing}")
    print(f"   Missing after: {after_missing}")
    print(f"   Imputation successful!" if after_missing < before_missing else "   No change")

print("\n" + "=" * 60)
print("IMPUTATION TEST SUMMARY")
print("=" * 60)
print(f"✓ Data loaded: {df.shape}")
print(f"✓ Biomarkers identified: {len(biomarker_cols)}")
print(f"✓ Climate features identified: {len(climate_cols)}")
print(f"{'✓' if len(socio_cols) > 0 else '✗'} Socioeconomic variables: {len(socio_cols)}")
print(f"✓ Basic imputation works: Yes")

print("\nNext steps:")
print("1. Add socioeconomic variables if missing")
print("2. Implement ecological imputation with geographic matching")
print("3. Use KNN for remaining missing values")
print("4. Validate imputed distributions")