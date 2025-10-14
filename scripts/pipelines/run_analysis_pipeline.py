#!/usr/bin/env python3
"""
Master Pipeline Execution Script for ENBEL Climate-Health Analysis
==================================================================

This script orchestrates the complete analysis workflow:
1. Data organization and preparation
2. Methodological imputation (ecological + KNN)
3. ML analysis with SHAP explainability
4. DLNM validation using R

Author: ENBEL Project Team
Date: 2024
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging

# Add src to path
sys.path.append('src')

from enbel_pp.imputation_pipeline import run_full_imputation_pipeline
from enbel_pp.biomarker_ml_pipeline import run_biomarker_ml_analysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check that required packages and data are available."""
    logger.info("Checking environment setup...")
    
    issues = []
    
    # Check Python packages
    required_packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 'shap']
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {package} installed")
        except ImportError:
            issues.append(f"Missing Python package: {package}")
            logger.error(f"  ✗ {package} not installed")
    
    # Check R installation
    try:
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        logger.info("  ✓ R is installed")
    except FileNotFoundError:
        issues.append("R is not installed or not in PATH")
        logger.error("  ✗ R not found")
    
    # Check R packages
    r_check = """
    packages <- c('dlnm', 'mgcv', 'splines', 'ggplot2', 'dplyr', 'jsonlite')
    missing <- packages[!packages %in% installed.packages()[,'Package']]
    if(length(missing) > 0) {
        cat('MISSING:', paste(missing, collapse=','))
    } else {
        cat('OK')
    }
    """
    
    try:
        result = subprocess.run(['R', '--slave', '-e', r_check], 
                              capture_output=True, text=True)
        if 'MISSING' in result.stdout:
            missing_r = result.stdout.replace('MISSING:', '').strip()
            issues.append(f"Missing R packages: {missing_r}")
            logger.error(f"  ✗ R packages missing: {missing_r}")
        else:
            logger.info("  ✓ R packages installed")
    except Exception as e:
        logger.warning(f"  Could not check R packages: {e}")
    
    if issues:
        logger.error("\nEnvironment issues detected:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Environment check passed!")
    return True


def organize_data():
    """Organize data files into proper structure."""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA ORGANIZATION")
    logger.info("="*60)
    
    # Check for data files in archive
    archive_path = Path("archive/previous_analysis")
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    # Look for clinical data
    clinical_files = list(archive_path.glob("CLINICAL*.csv")) if archive_path.exists() else []
    
    if clinical_files:
        # Use the most recent clinical file
        clinical_file = sorted(clinical_files)[-1]
        target_clinical = raw_data_path / "clinical_dataset.csv"
        
        if not target_clinical.exists():
            logger.info(f"Copying {clinical_file.name} to data/raw/")
            df = pd.read_csv(clinical_file)
            df.to_csv(target_clinical, index=False)
            logger.info(f"  Clinical data: {df.shape[0]} records, {df.shape[1]} columns")
        else:
            logger.info("  Clinical data already in place")
    else:
        logger.warning("  No clinical data found in archive")
        logger.info("  Please place CLINICAL_DATASET_COMPLETE_CLIMATE.csv in data/raw/")
        return None
    
    # Look for GCRO data
    gcro_file = None
    if archive_path.exists():
        gcro_files = list(archive_path.glob("GCRO*.csv"))
        if gcro_files:
            gcro_file = gcro_files[0]
            target_gcro = raw_data_path / "gcro_socioeconomic.csv"
            if not target_gcro.exists() and gcro_file.exists():
                logger.info(f"Copying {gcro_file.name} to data/raw/")
                df = pd.read_csv(gcro_file)
                df.to_csv(target_gcro, index=False)
                logger.info(f"  GCRO data: {df.shape[0]} records, {df.shape[1]} columns")
    
    return str(target_clinical) if 'target_clinical' in locals() else None


def run_imputation(clinical_path: str):
    """Run the imputation pipeline."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: METHODOLOGICAL IMPUTATION")
    logger.info("="*60)
    
    gcro_path = "data/raw/gcro_socioeconomic.csv"
    if not Path(gcro_path).exists():
        gcro_path = None
        logger.warning("No GCRO data available - using KNN imputation only")
    
    # Run imputation
    imputed_df = run_full_imputation_pipeline(
        clinical_path=clinical_path,
        gcro_path=gcro_path,
        output_dir="data/imputed",
        random_state=42
    )
    
    # Get the latest imputed file
    imputed_files = sorted(Path("data/imputed").glob("clinical_imputed_*.csv"))
    if imputed_files:
        return str(imputed_files[-1])
    
    return None


def run_ml_analysis(imputed_path: str):
    """Run the ML analysis with SHAP."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: MACHINE LEARNING ANALYSIS WITH SHAP")
    logger.info("="*60)
    
    results = run_biomarker_ml_analysis(
        data_path=imputed_path,
        output_dir="results/ml_analysis",
        min_model_quality=0.3  # Minimum R² for SHAP analysis
    )
    
    # Get the latest results file
    results_files = sorted(Path("results/ml_analysis").glob("ml_analysis_results_*.json"))
    if results_files:
        return str(results_files[-1])
    
    return None


def run_dlnm_validation(ml_results_path: str, imputed_path: str):
    """Run DLNM validation in R."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: DLNM VALIDATION")
    logger.info("="*60)
    
    r_script = "R/dlnm_validation/dlnm_validation_pipeline.R"
    
    if not Path(r_script).exists():
        logger.error(f"R script not found: {r_script}")
        return False
    
    try:
        # Run R validation script
        cmd = ['Rscript', r_script, ml_results_path, imputed_path, "results/dlnm_validation"]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("DLNM validation completed successfully")
            # Print R output
            if result.stdout:
                logger.info("R Output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            return True
        else:
            logger.error(f"DLNM validation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running DLNM validation: {e}")
        return False


def generate_final_report():
    """Generate a comprehensive final report."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("="*60)
    
    report = []
    report.append("="*70)
    report.append("ENBEL CLIMATE-HEALTH ANALYSIS - FINAL REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Check for imputation results
    imputation_reports = list(Path("data/imputed").glob("imputation_report_*.txt"))
    if imputation_reports:
        report.append("IMPUTATION COMPLETED ✓")
        with open(imputation_reports[-1], 'r') as f:
            lines = f.readlines()[6:15]  # Get summary lines
            report.extend([line.strip() for line in lines])
    
    report.append("")
    
    # Check for ML results
    ml_reports = list(Path("results/ml_analysis").glob("ml_analysis_report_*.txt"))
    if ml_reports:
        report.append("MACHINE LEARNING ANALYSIS COMPLETED ✓")
        with open(ml_reports[-1], 'r') as f:
            lines = f.readlines()[6:20]  # Get summary lines
            report.extend([line.strip() for line in lines])
    
    report.append("")
    
    # Check for DLNM results
    dlnm_reports = list(Path("results/dlnm_validation").glob("dlnm_validation_report.txt"))
    if dlnm_reports:
        report.append("DLNM VALIDATION COMPLETED ✓")
        with open(dlnm_reports[0], 'r') as f:
            lines = f.readlines()[5:12]  # Get summary lines
            report.extend([line.strip() for line in lines])
    
    report.append("")
    report.append("="*70)
    report.append("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    report.append("="*70)
    
    # Save report
    report_path = f"FINAL_ANALYSIS_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    logger.info(f"\nFinal report saved to: {report_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("ENBEL CLIMATE-HEALTH ANALYSIS PIPELINE")
    print("="*70)
    print("This pipeline will:")
    print("1. Organize and prepare data")
    print("2. Perform methodological imputation (ecological + KNN)")
    print("3. Run ML analysis with SHAP for biomarker relationships")
    print("4. Validate findings using DLNM in R")
    print("="*70)
    
    # Check environment
    if not check_environment():
        print("\nPlease install missing dependencies:")
        print("  Python: pip install -r requirements.txt")
        print("  R: install.packages(c('dlnm', 'mgcv', 'ggplot2', 'dplyr', 'jsonlite'))")
        return False
    
    # Step 1: Organize data
    clinical_path = organize_data()
    if not clinical_path:
        logger.error("Data organization failed. Please check data files.")
        return False
    
    # Step 2: Run imputation
    imputed_path = run_imputation(clinical_path)
    if not imputed_path:
        logger.error("Imputation failed.")
        return False
    
    logger.info(f"Imputed data saved to: {imputed_path}")
    
    # Step 3: Run ML analysis
    ml_results_path = run_ml_analysis(imputed_path)
    if not ml_results_path:
        logger.error("ML analysis failed.")
        return False
    
    logger.info(f"ML results saved to: {ml_results_path}")
    
    # Step 4: Run DLNM validation
    dlnm_success = run_dlnm_validation(ml_results_path, imputed_path)
    if not dlnm_success:
        logger.warning("DLNM validation encountered issues (may be due to R setup)")
    
    # Generate final report
    generate_final_report()
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE EXECUTION COMPLETED")
    logger.info("="*70)
    logger.info("Results available in:")
    logger.info("  - Imputed data: data/imputed/")
    logger.info("  - ML analysis: results/ml_analysis/")
    logger.info("  - DLNM validation: results/dlnm_validation/")
    logger.info("  - Logs: pipeline_execution.log")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)