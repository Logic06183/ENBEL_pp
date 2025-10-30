#!/usr/bin/env python3
"""
Feature Selection Verification Script
======================================

This script verifies what features are being selected by the pipeline
and checks for biomarker leakage.

Usage:
    python scripts/verification/verify_feature_selection.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def identify_climate_features_current(clinical_df: pd.DataFrame):
    """Current implementation from refined_analysis_pipeline.py"""
    climate_keywords = [
        'climate', 'temp', 'temperature', 'humidity', 'precipitation',
        'heat', 'wind', 'pressure', 'solar', 'lag'
    ]

    climate_features = []
    for col in clinical_df.columns:
        if any(keyword.lower() in col.lower() for keyword in climate_keywords):
            if pd.api.types.is_numeric_dtype(clinical_df[col]):
                if clinical_df[col].notna().sum() > 1000:
                    climate_features.append(col)

    return climate_features


def identify_biomarkers(clinical_df: pd.DataFrame):
    """Identify biomarker columns."""
    biomarker_keywords = [
        'CD4', 'glucose', 'cholesterol', 'LDL', 'HDL', 'triglyceride',
        'creatinine', 'ALT', 'AST', 'hemoglobin', 'hematocrit',
        'blood pressure', 'systolic', 'diastolic', 'White blood cell',
        'Red blood cell', 'Platelet', 'Lymphocyte', 'Neutrophil',
        'Monocyte', 'Eosinophil', 'Basophil', 'BMI', 'weight', 'height',
        'Waist', 'Albumin', 'protein', 'Alkaline phosphatase',
        'bilirubin', 'Sodium', 'Potassium', 'HIV viral load',
        'heart_rate', 'Respiratory rate', 'Oxygen saturation',
        'body_temperature', 'MCV', 'mch', 'RDW', 'Erythrocyte'
    ]

    biomarkers = []
    for col in clinical_df.columns:
        if any(keyword.lower() in col.lower() for keyword in biomarker_keywords):
            if pd.api.types.is_numeric_dtype(clinical_df[col]):
                if clinical_df[col].notna().sum() > 100:
                    biomarkers.append(col)

    return biomarkers


def main():
    print("\n" + "="*80)
    print("FEATURE SELECTION VERIFICATION")
    print("="*80 + "\n")

    # Load data
    data_path = project_root / "data" / "raw" / "CLINICAL_DATASET_COMPLETE_CLIMATE.csv"
    print(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Get current feature selection
    print("-" * 80)
    print("CURRENT FEATURE SELECTION (identify_climate_features)")
    print("-" * 80)
    current_features = identify_climate_features_current(df)
    print(f"\nTotal features selected: {len(current_features)}\n")
    print("Selected features:")
    for i, feat in enumerate(sorted(current_features), 1):
        print(f"  {i:2d}. {feat}")

    # Get biomarkers
    print("\n" + "-" * 80)
    print("IDENTIFIED BIOMARKERS (should NOT be in features)")
    print("-" * 80)
    biomarkers = identify_biomarkers(df)
    print(f"\nTotal biomarkers: {len(biomarkers)}\n")
    print("Biomarkers:")
    for i, bio in enumerate(sorted(biomarkers), 1):
        print(f"  {i:2d}. {bio}")

    # Check for overlap (LEAKAGE)
    print("\n" + "="*80)
    print("LEAKAGE DETECTION")
    print("="*80 + "\n")

    leakage = set(current_features) & set(biomarkers)

    if leakage:
        print(f"CRITICAL WARNING: {len(leakage)} BIOMARKERS FOUND IN FEATURE SET!")
        print("\nLeaked biomarkers:")
        for i, leak in enumerate(sorted(leakage), 1):
            print(f"  {i:2d}. {leak}")
        print("\nThis causes feature leakage and invalidates model results!")
    else:
        print("✓ No biomarker leakage detected")
        print("✓ All features are climate/socioeconomic variables")

    # Categorize features
    print("\n" + "="*80)
    print("FEATURE CATEGORIZATION")
    print("="*80 + "\n")

    climate_only = []
    heat_vulnerability = []
    temporal = []
    suspicious = []

    for feat in current_features:
        if feat.startswith('climate_'):
            climate_only.append(feat)
        elif feat.startswith('HEAT_'):
            heat_vulnerability.append(feat)
        elif feat in ['month', 'season', 'year']:
            temporal.append(feat)
        else:
            suspicious.append(feat)

    print(f"Climate features (climate_*): {len(climate_only)}")
    for feat in sorted(climate_only):
        print(f"  - {feat}")

    print(f"\nHeat vulnerability features (HEAT_*): {len(heat_vulnerability)}")
    for feat in sorted(heat_vulnerability):
        print(f"  - {feat}")

    print(f"\nTemporal features: {len(temporal)}")
    for feat in sorted(temporal):
        print(f"  - {feat}")

    if suspicious:
        print(f"\nSUSPICIOUS FEATURES (need review): {len(suspicious)}")
        for feat in sorted(suspicious):
            print(f"  - {feat}")
            # Check if it's a biomarker
            if feat in biomarkers:
                print(f"    ^^^ BIOMARKER LEAKAGE!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print(f"Total columns in dataset: {df.shape[1]}")
    print(f"Features selected by current method: {len(current_features)}")
    print(f"Biomarkers identified: {len(biomarkers)}")
    print(f"Biomarker leakage detected: {len(leakage)}")
    print(f"Climate-only features: {len(climate_only)}")
    print(f"Heat vulnerability features: {len(heat_vulnerability)}")
    print(f"Suspicious features: {len(suspicious)}")

    if leakage:
        print("\n" + "!"*80)
        print("CRITICAL: FEATURE LEAKAGE DETECTED - FIX REQUIRED")
        print("!"*80 + "\n")
        return 1
    else:
        print("\n✓ No leakage detected - feature selection is clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
