#!/usr/bin/env python3
"""
Data Validation Utilities for ENBEL Climate-Health Analysis
===========================================================

Comprehensive data validation and integrity checking functions to ensure
reproducibility and catch data issues early in the pipeline.

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, Union
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass

def validate_file_exists(file_path: Union[str, Path], description: str = "") -> Path:
    """
    Validate that a required file exists.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file to check
    description : str
        Description of the file for error messages
        
    Returns:
    --------
    Path
        Validated Path object
        
    Raises:
    -------
    DataValidationError
        If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise DataValidationError(
            f"Required {description} file not found: {file_path}\n"
            f"Please ensure the file exists in the expected location."
        )
    logger.info(f"Validated {description} file: {file_path}")
    return file_path

def validate_data_files() -> Dict[str, Path]:
    """
    Validate all required data files for the ENBEL analysis pipeline.
    
    Returns:
    --------
    Dict[str, Path]
        Dictionary mapping data file types to their validated paths
        
    Raises:
    -------
    DataValidationError
        If any required files are missing
    """
    required_files = {
        'full_dataset': 'FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv',
        'clinical_imputed': 'CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv', 
        'clinical_original': 'CLINICAL_WITH_IMPUTED_SOCIOECONOMIC.csv'
    }
    
    validated_paths = {}
    
    for file_type, filename in required_files.items():
        try:
            validated_paths[file_type] = validate_file_exists(filename, file_type)
        except DataValidationError as e:
            logger.error(f"Missing required data file: {filename}")
            raise e
            
    logger.info("All required data files validated successfully")
    return validated_paths

def validate_dataframe_schema(df: pd.DataFrame, 
                            required_columns: List[str],
                            dataset_name: str = "dataset") -> bool:
    """
    Validate that a DataFrame has the required column structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
    dataset_name : str
        Name of dataset for error messages
        
    Returns:
    --------
    bool
        True if validation passes
        
    Raises:
    -------
    DataValidationError
        If required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns in {dataset_name}: {missing_columns}\n"
            f"Available columns: {list(df.columns)[:10]}..."  # Show first 10
        )
    
    logger.info(f"Schema validation passed for {dataset_name}")
    return True

def validate_biomarker_data(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Validate biomarker data quality and completeness.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing biomarker data
        
    Returns:
    --------
    Dict[str, Dict]
        Biomarker quality statistics
    """
    biomarkers = [
        'CD4 cell count (cells/µL)', 'FASTING GLUCOSE', 'FASTING LDL', 
        'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 'FASTING TRIGLYCERIDES',
        'Creatinine (mg/dL)', 'ALT (U/L)', 'AST (U/L)', 'Hemoglobin (g/dL)',
        'Hematocrit (%)', 'systolic blood pressure', 'diastolic blood pressure'
    ]
    
    biomarker_stats = {}
    
    for biomarker in biomarkers:
        if biomarker in df.columns:
            stats = {
                'total_count': len(df),
                'non_null_count': df[biomarker].notna().sum(),
                'null_count': df[biomarker].isna().sum(),
                'completeness_pct': (df[biomarker].notna().sum() / len(df)) * 100,
                'outliers_count': 0,
                'valid_range': True
            }
            
            # Check for obvious outliers (values outside physiologically plausible ranges)
            if not df[biomarker].isna().all():
                values = df[biomarker].dropna()
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (values < (Q1 - 3 * IQR)) | (values > (Q3 + 3 * IQR))
                stats['outliers_count'] = outlier_mask.sum()
                
                # Basic range validation
                if biomarker == 'systolic blood pressure':
                    stats['valid_range'] = (values >= 60).all() and (values <= 300).all()
                elif biomarker == 'diastolic blood pressure':
                    stats['valid_range'] = (values >= 30).all() and (values <= 150).all()
                elif 'GLUCOSE' in biomarker:
                    stats['valid_range'] = (values >= 30).all() and (values <= 800).all()
            
            biomarker_stats[biomarker] = stats
            
            # Log warnings for low completeness
            if stats['completeness_pct'] < 50:
                logger.warning(f"Low data completeness for {biomarker}: {stats['completeness_pct']:.1f}%")
                
    logger.info(f"Biomarker validation completed for {len(biomarker_stats)} biomarkers")
    return biomarker_stats

def validate_climate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate climate data integrity and lag structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
        
    Returns:
    --------
    Dict[str, any]
        Climate data validation results
    """
    # Check for expected climate variables
    expected_climate_vars = [
        'temperature', 'humidity', 'wind_speed', 'heat_index',
        'apparent_temp', 'wet_bulb_temp'
    ]
    
    # Check lag structure
    lag_patterns = ['lag0', 'lag1', 'lag2', 'lag3', 'lag5', 'lag7', 'lag10', 'lag14', 'lag21']
    
    climate_stats = {
        'base_climate_vars': {},
        'lag_structure': {},
        'missing_data_patterns': {},
        'temporal_coverage': {}
    }
    
    # Validate base climate variables
    for var in expected_climate_vars:
        if var in df.columns:
            climate_stats['base_climate_vars'][var] = {
                'completeness_pct': (df[var].notna().sum() / len(df)) * 100,
                'mean': df[var].mean() if df[var].notna().any() else None,
                'std': df[var].std() if df[var].notna().any() else None
            }
    
    # Validate lag structure
    for lag in lag_patterns:
        lag_cols = [col for col in df.columns if f'_{lag}' in col]
        climate_stats['lag_structure'][lag] = {
            'column_count': len(lag_cols),
            'variables': lag_cols[:5]  # Show first 5 examples
        }
    
    # Check for temporal patterns in missing data
    if 'primary_date' in df.columns:
        try:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['primary_date'])
            climate_stats['temporal_coverage'] = {
                'date_range': (df_temp['date'].min(), df_temp['date'].max()),
                'total_days': (df_temp['date'].max() - df_temp['date'].min()).days
            }
        except Exception as e:
            logger.warning(f"Could not parse temporal coverage: {e}")
    
    logger.info("Climate data validation completed")
    return climate_stats

def validate_imputation_quality(original_df: pd.DataFrame, 
                               imputed_df: pd.DataFrame,
                               imputed_vars: List[str]) -> Dict[str, Dict]:
    """
    Validate the quality of imputation results.
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataset before imputation
    imputed_df : pd.DataFrame
        Dataset after imputation
    imputed_vars : List[str]
        List of variables that were imputed
        
    Returns:
    --------
    Dict[str, Dict]
        Imputation quality metrics
    """
    imputation_stats = {}
    
    for var in imputed_vars:
        if f'{var}_imputed' in imputed_df.columns:
            original_missing = original_df[var].isna().sum() if var in original_df.columns else len(original_df)
            imputed_count = imputed_df[f'{var}_imputed'].notna().sum()
            
            stats = {
                'original_missing_count': original_missing,
                'imputed_count': imputed_count,
                'imputation_rate': (imputed_count / original_missing * 100) if original_missing > 0 else 0,
                'mean_confidence': imputed_df[f'{var}_confidence'].mean() if f'{var}_confidence' in imputed_df.columns else None
            }
            
            imputation_stats[var] = stats
            
            if stats['imputation_rate'] < 70:
                logger.warning(f"Low imputation rate for {var}: {stats['imputation_rate']:.1f}%")
    
    logger.info(f"Imputation quality validation completed for {len(imputation_stats)} variables")
    return imputation_stats

def generate_validation_report(data_files: Dict[str, Path],
                             biomarker_stats: Dict[str, Dict],
                             climate_stats: Dict[str, any],
                             output_path: str = "data_validation_report.txt") -> None:
    """
    Generate a comprehensive data validation report.
    
    Parameters:
    -----------
    data_files : Dict[str, Path]
        Validated data file paths
    biomarker_stats : Dict[str, Dict]
        Biomarker validation statistics
    climate_stats : Dict[str, any]
        Climate data validation statistics
    output_path : str
        Path for output report
    """
    with open(output_path, 'w') as f:
        f.write("ENBEL CLIMATE-HEALTH DATA VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Data files section
        f.write("1. DATA FILES VALIDATION\n")
        f.write("-" * 25 + "\n")
        for file_type, path in data_files.items():
            f.write(f"VALIDATED {file_type}: {path}\n")
        f.write("\n")
        
        # Biomarker statistics
        f.write("2. BIOMARKER DATA QUALITY\n")
        f.write("-" * 25 + "\n")
        for biomarker, stats in biomarker_stats.items():
            f.write(f"{biomarker}:\n")
            f.write(f"  - Completeness: {stats['completeness_pct']:.1f}%\n")
            f.write(f"  - Non-null count: {stats['non_null_count']:,}\n")
            f.write(f"  - Outliers: {stats['outliers_count']}\n")
            f.write(f"  - Valid range: {stats['valid_range']}\n\n")
        
        # Climate data statistics
        f.write("3. CLIMATE DATA VALIDATION\n")
        f.write("-" * 25 + "\n")
        f.write("Base climate variables:\n")
        for var, stats in climate_stats['base_climate_vars'].items():
            f.write(f"  - {var}: {stats['completeness_pct']:.1f}% complete\n")
        
        f.write("\nLag structure:\n")
        for lag, info in climate_stats['lag_structure'].items():
            f.write(f"  - {lag}: {info['column_count']} variables\n")
        
        f.write(f"\nValidation report generated: {output_path}\n")
    
    logger.info(f"Data validation report saved to: {output_path}")

def run_full_validation() -> bool:
    """
    Run complete data validation pipeline.
    
    Returns:
    --------
    bool
        True if all validations pass
    """
    try:
        logger.info("Starting comprehensive data validation...")
        
        # Validate file existence
        data_files = validate_data_files()
        
        # Load and validate main dataset
        df = pd.read_csv(data_files['full_dataset'], low_memory=False)
        logger.info(f"Loaded main dataset: {df.shape}")
        
        # Validate biomarker data
        biomarker_stats = validate_biomarker_data(df)
        
        # Validate climate data
        climate_stats = validate_climate_data(df)
        
        # Generate comprehensive report
        generate_validation_report(data_files, biomarker_stats, climate_stats)
        
        logger.info("All data validation checks passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise DataValidationError(f"Validation pipeline failed: {e}")

if __name__ == "__main__":
    run_full_validation()