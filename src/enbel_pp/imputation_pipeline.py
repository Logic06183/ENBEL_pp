"""
Methodologically Sound Imputation Pipeline for Climate-Health Analysis
======================================================================

This module implements a rigorous imputation strategy for the clinical trial dataset,
incorporating socioeconomic variables from GCRO data using ecological and KNN methods.

Key Features:
- Ecological imputation based on geographic and temporal matching
- KNN imputation with careful feature selection
- Preservation of data structure and relationships
- Full documentation of imputation decisions
- Validation of imputed values against known distributions

Author: ENBEL Project Team
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MethodologicalImputation:
    """
    Implements methodologically sound imputation for climate-health datasets.
    
    This class provides both ecological (group-based) and KNN (similarity-based)
    imputation methods, with careful handling of different variable types.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the imputation pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.imputation_log = []
        self.validation_metrics = {}
        
        # Define variable categories for imputation strategy
        self.variable_categories = {
            'demographic': ['age', 'sex', 'race', 'gender'],
            'socioeconomic': ['dwelling_type', 'education_level', 'income_category', 
                             'employment_status', 'household_size'],
            'geographic': ['ward', 'region', 'latitude', 'longitude'],
            'temporal': ['year', 'month', 'season', 'date'],
            'clinical': ['CD4', 'glucose', 'hemoglobin', 'creatinine', 'cholesterol',
                        'blood_pressure', 'HDL', 'LDL', 'triglycerides']
        }
        
        logger.info("Initialized Methodological Imputation Pipeline")
    
    def load_datasets(self, 
                     clinical_path: str,
                     gcro_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load clinical and GCRO datasets with validation.
        
        Parameters:
        -----------
        clinical_path : str
            Path to clinical dataset
        gcro_path : str, optional
            Path to GCRO socioeconomic dataset
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            Clinical and GCRO dataframes
        """
        logger.info("Loading datasets...")
        
        # Load clinical data
        clinical_df = pd.read_csv(clinical_path, low_memory=False)
        logger.info(f"Loaded clinical data: {clinical_df.shape}")
        
        # Load GCRO data if provided
        gcro_df = None
        if gcro_path and Path(gcro_path).exists():
            gcro_df = pd.read_csv(gcro_path, low_memory=False)
            logger.info(f"Loaded GCRO data: {gcro_df.shape}")
        
        # Validate data structure
        self._validate_data_structure(clinical_df, gcro_df)
        
        return clinical_df, gcro_df
    
    def _validate_data_structure(self, clinical_df: pd.DataFrame, gcro_df: Optional[pd.DataFrame]):
        """Validate the structure and content of loaded datasets."""
        
        # Check for essential columns in clinical data
        essential_clinical = ['participant_id', 'date', 'latitude', 'longitude']
        missing_clinical = [col for col in essential_clinical 
                           if col not in clinical_df.columns and 
                           not any(col in c.lower() for c in clinical_df.columns)]
        
        if missing_clinical:
            logger.warning(f"Missing essential clinical columns: {missing_clinical}")
        
        # Check data completeness
        completeness = (clinical_df.notna().sum() / len(clinical_df)) * 100
        logger.info(f"Clinical data completeness: {completeness.mean():.1f}% average")
        
        # Log variable counts by category
        for category, patterns in self.variable_categories.items():
            matching_cols = [col for col in clinical_df.columns 
                           if any(p.lower() in col.lower() for p in patterns)]
            if matching_cols:
                logger.info(f"Found {len(matching_cols)} {category} variables")
    
    def ecological_imputation(self,
                             clinical_df: pd.DataFrame,
                             gcro_df: pd.DataFrame,
                             match_variables: List[str] = ['ward', 'year'],
                             target_variables: List[str] = None) -> pd.DataFrame:
        """
        Perform ecological imputation based on geographic and temporal matching.
        
        This method imputes missing socioeconomic variables in clinical data
        using aggregate statistics from GCRO data matched by location and time.
        
        Parameters:
        -----------
        clinical_df : pd.DataFrame
            Clinical dataset with missing socioeconomic variables
        gcro_df : pd.DataFrame
            GCRO dataset with complete socioeconomic information
        match_variables : List[str]
            Variables to use for matching (e.g., ward, year)
        target_variables : List[str]
            Socioeconomic variables to impute
            
        Returns:
        --------
        pd.DataFrame
            Clinical dataset with ecologically imputed values
        """
        logger.info("Starting ecological imputation...")
        
        if target_variables is None:
            target_variables = self.variable_categories['socioeconomic']
        
        # Find available target variables in GCRO data
        available_targets = [var for var in target_variables 
                           if any(var.lower() in col.lower() for col in gcro_df.columns)]
        
        if not available_targets:
            logger.warning("No matching target variables found for ecological imputation")
            return clinical_df
        
        logger.info(f"Imputing {len(available_targets)} variables using ecological method")
        
        imputed_df = clinical_df.copy()
        
        for target_var in available_targets:
            # Find the actual column name in GCRO
            gcro_col = [col for col in gcro_df.columns 
                       if target_var.lower() in col.lower()][0]
            
            # Calculate ecological estimates (mode for categorical, mean for numeric)
            if gcro_df[gcro_col].dtype == 'object':
                ecological_values = gcro_df.groupby(match_variables)[gcro_col].agg(
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan
                )
            else:
                ecological_values = gcro_df.groupby(match_variables)[gcro_col].mean()
            
            # Apply ecological imputation
            n_imputed = 0
            for idx, row in imputed_df.iterrows():
                if pd.isna(row.get(target_var)):
                    match_key = tuple(row[var] for var in match_variables if var in row.index)
                    if match_key in ecological_values.index:
                        imputed_df.loc[idx, target_var] = ecological_values[match_key]
                        n_imputed += 1
            
            logger.info(f"  - {target_var}: imputed {n_imputed} values")
            
            # Log imputation details
            self.imputation_log.append({
                'method': 'ecological',
                'variable': target_var,
                'n_imputed': n_imputed,
                'match_variables': match_variables,
                'timestamp': datetime.now().isoformat()
            })
        
        return imputed_df
    
    def knn_imputation(self,
                      df: pd.DataFrame,
                      feature_columns: List[str],
                      target_columns: List[str],
                      n_neighbors: int = 5,
                      weights: str = 'distance') -> pd.DataFrame:
        """
        Perform KNN imputation for specified variables.
        
        This method uses K-Nearest Neighbors to impute missing values based on
        similarity in feature space, with careful handling of mixed data types.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with missing values
        feature_columns : List[str]
            Columns to use as features for finding neighbors
        target_columns : List[str]
            Columns with missing values to impute
        n_neighbors : int
            Number of neighbors to use for imputation
        weights : str
            Weight function for neighbors ('uniform' or 'distance')
            
        Returns:
        --------
        pd.DataFrame
            Dataset with KNN-imputed values
        """
        logger.info(f"Starting KNN imputation with {n_neighbors} neighbors...")
        
        imputed_df = df.copy()
        
        # Prepare feature matrix
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) < 3:
            logger.warning("Insufficient features for KNN imputation")
            return imputed_df
        
        # Handle mixed data types
        numeric_features = df[available_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            logger.warning("No numeric features available for KNN imputation")
            return imputed_df
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix = df[numeric_features].values
        
        # Handle missing values in features
        feature_imputer = KNNImputer(n_neighbors=min(3, n_neighbors))
        feature_matrix_complete = feature_imputer.fit_transform(feature_matrix)
        feature_matrix_scaled = scaler.fit_transform(feature_matrix_complete)
        
        # Impute each target column
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
                
            target_data = df[target_col].values.reshape(-1, 1)
            n_missing = pd.isna(target_data).sum()
            
            if n_missing == 0:
                continue
            
            # Combine features with target for imputation
            combined_data = np.hstack([feature_matrix_scaled, target_data])
            
            # Perform KNN imputation
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputed_combined = imputer.fit_transform(combined_data)
            
            # Extract imputed target values
            imputed_values = imputed_combined[:, -1]
            imputed_df[target_col] = imputed_values
            
            n_imputed = n_missing - pd.isna(imputed_values).sum()
            logger.info(f"  - {target_col}: imputed {n_imputed}/{n_missing} missing values")
            
            # Log imputation details
            self.imputation_log.append({
                'method': 'knn',
                'variable': target_col,
                'n_imputed': int(n_imputed),
                'n_neighbors': n_neighbors,
                'n_features': len(numeric_features),
                'timestamp': datetime.now().isoformat()
            })
        
        return imputed_df
    
    def validate_imputation(self,
                          original_df: pd.DataFrame,
                          imputed_df: pd.DataFrame,
                          validation_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate the quality of imputed values.
        
        Parameters:
        -----------
        original_df : pd.DataFrame
            Original dataset before imputation
        imputed_df : pd.DataFrame
            Dataset after imputation
        validation_columns : List[str]
            Columns to validate
            
        Returns:
        --------
        Dict[str, Any]
            Validation metrics and diagnostics
        """
        logger.info("Validating imputation quality...")
        
        if validation_columns is None:
            validation_columns = [col for col in imputed_df.columns 
                                if imputed_df[col].dtype in [np.float64, np.int64]]
        
        validation_results = {}
        
        for col in validation_columns:
            if col not in original_df.columns:
                continue
            
            # Calculate statistics
            original_values = original_df[col].dropna()
            imputed_values = imputed_df[col].dropna()
            newly_imputed_mask = original_df[col].isna() & imputed_df[col].notna()
            newly_imputed_values = imputed_df.loc[newly_imputed_mask, col]
            
            if len(newly_imputed_values) == 0:
                continue
            
            validation_results[col] = {
                'n_imputed': len(newly_imputed_values),
                'original_mean': float(original_values.mean()) if len(original_values) > 0 else np.nan,
                'original_std': float(original_values.std()) if len(original_values) > 0 else np.nan,
                'imputed_mean': float(newly_imputed_values.mean()),
                'imputed_std': float(newly_imputed_values.std()),
                'distribution_test': None
            }
            
            # Kolmogorov-Smirnov test for distribution similarity
            if len(original_values) > 20 and len(newly_imputed_values) > 20:
                ks_stat, ks_pval = stats.ks_2samp(original_values, newly_imputed_values)
                validation_results[col]['distribution_test'] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(ks_pval),
                    'similar_distribution': ks_pval > 0.05
                }
        
        self.validation_metrics = validation_results
        return validation_results
    
    def create_imputation_report(self, output_path: str = None) -> str:
        """
        Create a comprehensive report of the imputation process.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report
            
        Returns:
        --------
        str
            Formatted imputation report
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("METHODOLOGICAL IMPUTATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Imputation summary
        report_lines.append("IMPUTATION SUMMARY")
        report_lines.append("-" * 40)
        
        methods_used = {}
        for log_entry in self.imputation_log:
            method = log_entry['method']
            if method not in methods_used:
                methods_used[method] = {'variables': [], 'total_imputed': 0}
            methods_used[method]['variables'].append(log_entry['variable'])
            methods_used[method]['total_imputed'] += log_entry['n_imputed']
        
        for method, details in methods_used.items():
            report_lines.append(f"\n{method.upper()} Imputation:")
            report_lines.append(f"  Variables: {', '.join(details['variables'])}")
            report_lines.append(f"  Total values imputed: {details['total_imputed']:,}")
        
        # Validation results
        if self.validation_metrics:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("VALIDATION METRICS")
            report_lines.append("-" * 40)
            
            for var, metrics in self.validation_metrics.items():
                report_lines.append(f"\n{var}:")
                report_lines.append(f"  Values imputed: {metrics['n_imputed']}")
                report_lines.append(f"  Original mean ± std: {metrics['original_mean']:.3f} ± {metrics['original_std']:.3f}")
                report_lines.append(f"  Imputed mean ± std: {metrics['imputed_mean']:.3f} ± {metrics['imputed_std']:.3f}")
                
                if metrics['distribution_test']:
                    test = metrics['distribution_test']
                    report_lines.append(f"  Distribution test p-value: {test['p_value']:.4f}")
                    report_lines.append(f"  Similar distribution: {'Yes' if test['similar_distribution'] else 'No'}")
        
        # Methodological notes
        report_lines.append("\n" + "=" * 70)
        report_lines.append("METHODOLOGICAL NOTES")
        report_lines.append("-" * 40)
        report_lines.append("1. Ecological imputation used ward-level aggregates from GCRO data")
        report_lines.append("2. KNN imputation used distance-weighted neighbors in standardized space")
        report_lines.append("3. Distribution similarity assessed using Kolmogorov-Smirnov test")
        report_lines.append("4. All imputation decisions logged for full reproducibility")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def save_imputation_log(self, output_path: str):
        """Save detailed imputation log as JSON."""
        with open(output_path, 'w') as f:
            json.dump({
                'imputation_log': self.imputation_log,
                'validation_metrics': self.validation_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Imputation log saved to: {output_path}")


def run_full_imputation_pipeline(clinical_path: str,
                                gcro_path: str = None,
                                output_dir: str = "data/imputed",
                                random_state: int = 42) -> pd.DataFrame:
    """
    Run the complete imputation pipeline with both ecological and KNN methods.
    
    Parameters:
    -----------
    clinical_path : str
        Path to clinical dataset
    gcro_path : str
        Path to GCRO socioeconomic dataset
    output_dir : str
        Directory to save imputed data and reports
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Fully imputed dataset
    """
    # Initialize pipeline
    imputer = MethodologicalImputation(random_state=random_state)
    
    # Load datasets
    clinical_df, gcro_df = imputer.load_datasets(clinical_path, gcro_path)
    
    # Step 1: Ecological imputation for socioeconomic variables
    if gcro_df is not None:
        imputed_df = imputer.ecological_imputation(
            clinical_df, gcro_df,
            match_variables=['ward', 'year'],
            target_variables=['dwelling_type', 'education_level', 'income_category']
        )
    else:
        imputed_df = clinical_df.copy()
    
    # Step 2: KNN imputation for remaining missing values
    # Use demographic and climate features for biomarker imputation
    feature_cols = [col for col in imputed_df.columns 
                   if any(pattern in col.lower() 
                         for pattern in ['age', 'sex', 'temp', 'humid', 'pressure'])]
    
    biomarker_cols = [col for col in imputed_df.columns
                     if any(pattern in col.lower()
                           for pattern in ['glucose', 'cd4', 'hemoglobin', 'creatinine'])]
    
    if feature_cols and biomarker_cols:
        imputed_df = imputer.knn_imputation(
            imputed_df,
            feature_columns=feature_cols,
            target_columns=biomarker_cols,
            n_neighbors=5
        )
    
    # Validate imputation
    validation_results = imputer.validate_imputation(clinical_df, imputed_df)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save imputed dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/clinical_imputed_{timestamp}.csv"
    imputed_df.to_csv(output_path, index=False)
    logger.info(f"Imputed dataset saved to: {output_path}")
    
    # Generate and save reports
    report = imputer.create_imputation_report(f"{output_dir}/imputation_report_{timestamp}.txt")
    imputer.save_imputation_log(f"{output_dir}/imputation_log_{timestamp}.json")
    
    print("\n" + report)
    
    return imputed_df


if __name__ == "__main__":
    # Example usage
    print("Running Methodological Imputation Pipeline")
    print("=" * 50)
    
    # You would replace these with actual file paths
    clinical_file = "data/raw/clinical_dataset.csv"
    gcro_file = "data/raw/gcro_socioeconomic.csv"
    
    if Path(clinical_file).exists():
        imputed_data = run_full_imputation_pipeline(
            clinical_path=clinical_file,
            gcro_path=gcro_file,
            output_dir="data/imputed",
            random_state=42
        )
        print(f"\nImputation complete. Final dataset shape: {imputed_data.shape}")
    else:
        print("Please provide valid data file paths to run the imputation pipeline.")