"""
Automated Feature Leakage Detection
====================================

This module provides utilities to detect and prevent feature leakage in
climate-health biomarker modeling. Ensures that biomarkers are never used
to predict other biomarkers, and that highly correlated features are flagged.

Author: ENBEL Team
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    """Container for leakage detection results."""
    is_safe: bool
    biomarker_leakage: List[str]
    high_correlation_pairs: List[Tuple[str, str, float]]
    circular_predictions: List[Tuple[str, str]]
    warnings: List[str]


class LeakageChecker:
    """
    Automated feature leakage detection for biomarker modeling.

    This class identifies three types of leakage:
    1. Biomarker-to-biomarker prediction (e.g., hemoglobin → hematocrit)
    2. High correlation features (r > threshold)
    3. Circular predictions (e.g., BMI → Weight → BMI)
    """

    # Comprehensive list of ALL biomarker columns (MUST BE EXCLUDED)
    BIOMARKER_COLUMNS = {
        # Primary biomarkers
        'CD4 cell count (cells/µL)',
        'HIV viral load (copies/mL)',
        'hemoglobin_g_dL',
        'Hematocrit (%)',
        'fasting_glucose_mmol_L',
        'creatinine_umol_L',
        'creatinine clearance',
        'total_cholesterol_mg_dL',
        'hdl_cholesterol_mg_dL',
        'ldl_cholesterol_mg_dL',
        'FASTING HDL',
        'FASTING LDL',
        'FASTING TRIGLYCERIDES',
        'Triglycerides (mg/dL)',
        'ALT (U/L)',
        'AST (U/L)',

        # Blood cell counts & indices
        'White blood cell count (×10³/µL)',
        'Red blood cell count (×10⁶/µL)',
        'Platelet count (×10³/µL)',
        'Lymphocyte count (×10³/µL)',
        'Neutrophil count (×10³/µL)',
        'Lymphocyte percentage (%)',
        'Lymphocytes (%)',
        'Neutrophil percentage (%)',
        'Neutrophils (%)',
        'Monocyte percentage (%)',
        'Monocytes (%)',
        'Eosinophil percentage (%)',
        'Eosinophils (%)',
        'Basophil percentage (%)',
        'Basophils (%)',
        'Erythrocytes',
        'MCV (MEAN CELL VOLUME)',
        'Mean corpuscular volume (fL)',
        'mch_pg',
        'mchc_g_dL',
        'RDW',

        # Vital signs
        'systolic_bp_mmHg',
        'diastolic_bp_mmHg',
        'heart_rate_bpm',
        'Respiratory rate (breaths/min)',
        'respiration rate',
        'Oxygen saturation (%)',
        'body_temperature_celsius',

        # Anthropometric (with exceptions)
        'Last height recorded (m)',
        'Last weight recorded (kg)',
        'Waist circumference (cm)',

        # Liver function
        'Albumin (g/dL)',
        'Alkaline phosphatase (U/L)',
        'Total bilirubin (mg/dL)',
        'Total protein (g/dL)',

        # Electrolytes
        'Potassium (mEq/L)',
        'Sodium (mEq/L)',
    }

    # Circular prediction groups (features that can predict each other)
    CIRCULAR_GROUPS = [
        {'BMI (kg/m²)', 'weight_kg', 'Last weight recorded (kg)'},
        {'systolic_bp_mmHg', 'diastolic_bp_mmHg'},
        {'hemoglobin_g_dL', 'Hematocrit (%)'},
        {'ALT (U/L)', 'AST (U/L)'},
        {'FASTING LDL', 'ldl_cholesterol_mg_dL'},
        {'FASTING HDL', 'hdl_cholesterol_mg_dL'},
        {'FASTING TRIGLYCERIDES', 'Triglycerides (mg/dL)'},
        {'total_cholesterol_mg_dL', 'FASTING TOTAL CHOLESTEROL'},
        {'creatinine_umol_L', 'creatinine clearance'},
    ]

    # Safe feature categories
    SAFE_CLIMATE_KEYWORDS = [
        'climate', 'temp', 'temperature', 'heat', 'anomaly',
        'lag', 'stress', 'threshold', 'season'
    ]

    SAFE_SOCIOECONOMIC_FEATURES = {
        'HEAT_VULNERABILITY_SCORE',
        'heat_vulnerability_index',
        'economic_vulnerability_indicator',
        'employment_vulnerability_indicator',
        'education_adaptive_capacity',
        'age_vulnerability_indicator',
        'dwelling_type_enhanced',
        'DwellingType',
        'EmploymentStatus',
        'Education',
        'std_education',
        'Q15_20_income',
        'q15_3_income_recode',
        'Q1_03_households',
        'dwelling_count',
        'Q2_02_dwelling_dissatisfaction',
        'Q2_14_Drainage',
        'Ward',
        'std_race',
        'Race',
    }

    SAFE_DEMOGRAPHIC_FEATURES = {
        'Age (at enrolment)',
        'Sex',
        'Race',
        'Antiretroviral Therapy Status',
        'HIV_status',
        'month',
        'season',
        'year',
        'study_source',
    }

    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize the leakage checker.

        Args:
            correlation_threshold: Pearson r threshold for high correlation warning
        """
        self.correlation_threshold = correlation_threshold

    def check_features(
        self,
        target_biomarker: str,
        feature_list: List[str],
        data: pd.DataFrame = None
    ) -> LeakageReport:
        """
        Comprehensive leakage check for a feature set.

        Args:
            target_biomarker: The biomarker being predicted
            feature_list: List of features to be used as predictors
            data: Optional dataframe for correlation analysis

        Returns:
            LeakageReport with detailed findings
        """
        biomarker_leakage = []
        high_correlation_pairs = []
        circular_predictions = []
        warnings = []

        # Check 1: Biomarker-to-biomarker leakage
        for feature in feature_list:
            if feature in self.BIOMARKER_COLUMNS and feature != target_biomarker:
                biomarker_leakage.append(feature)
                logger.error(
                    f"🚨 BIOMARKER LEAKAGE: Using '{feature}' to predict "
                    f"'{target_biomarker}'"
                )

        # Check 2: Circular predictions
        target_group = self._find_circular_group(target_biomarker)
        if target_group:
            for feature in feature_list:
                if feature in target_group and feature != target_biomarker:
                    circular_predictions.append((target_biomarker, feature))
                    logger.warning(
                        f"⚠️  CIRCULAR PREDICTION: '{feature}' and "
                        f"'{target_biomarker}' are in the same circular group"
                    )

        # Check 3: High correlation (if data provided)
        if data is not None:
            available_features = [f for f in feature_list if f in data.columns]
            if len(available_features) > 1:
                corr_matrix = data[available_features].corr(method='pearson')

                for i in range(len(available_features)):
                    for j in range(i + 1, len(available_features)):
                        feat1 = available_features[i]
                        feat2 = available_features[j]
                        corr = abs(corr_matrix.iloc[i, j])

                        if corr > self.correlation_threshold:
                            high_correlation_pairs.append((feat1, feat2, corr))
                            logger.warning(
                                f"⚠️  HIGH CORRELATION: '{feat1}' and '{feat2}' "
                                f"(r = {corr:.3f})"
                            )

        # Check 4: Feature categorization validation
        for feature in feature_list:
            is_safe = (
                self._is_climate_feature(feature) or
                feature in self.SAFE_SOCIOECONOMIC_FEATURES or
                feature in self.SAFE_DEMOGRAPHIC_FEATURES
            )

            if not is_safe and feature not in self.BIOMARKER_COLUMNS:
                warnings.append(
                    f"Unrecognized feature '{feature}' - not in safe categories"
                )

        # Determine if feature set is safe
        is_safe = (
            len(biomarker_leakage) == 0 and
            len(circular_predictions) == 0
        )

        return LeakageReport(
            is_safe=is_safe,
            biomarker_leakage=biomarker_leakage,
            high_correlation_pairs=high_correlation_pairs,
            circular_predictions=circular_predictions,
            warnings=warnings
        )

    def _is_climate_feature(self, feature: str) -> bool:
        """Check if a feature is a climate variable."""
        return any(
            keyword.lower() in feature.lower()
            for keyword in self.SAFE_CLIMATE_KEYWORDS
        )

    def _find_circular_group(self, biomarker: str) -> Set[str]:
        """Find the circular prediction group containing the biomarker."""
        for group in self.CIRCULAR_GROUPS:
            if biomarker in group:
                return group
        return None

    def generate_safe_feature_set(
        self,
        target_biomarker: str,
        all_features: List[str],
        data: pd.DataFrame = None
    ) -> Dict[str, List[str]]:
        """
        Generate a safe feature set by filtering out leakage sources.

        Args:
            target_biomarker: The biomarker being predicted
            all_features: All available features
            data: Optional dataframe for validation

        Returns:
            Dictionary with 'safe_features' and 'excluded_features'
        """
        safe_features = []
        excluded_features = []

        for feature in all_features:
            # Skip the target itself
            if feature == target_biomarker:
                continue

            # Exclude all other biomarkers
            if feature in self.BIOMARKER_COLUMNS:
                excluded_features.append(feature)
                continue

            # Exclude circular prediction features
            target_group = self._find_circular_group(target_biomarker)
            if target_group and feature in target_group:
                excluded_features.append(feature)
                continue

            # Include if safe
            if (self._is_climate_feature(feature) or
                feature in self.SAFE_SOCIOECONOMIC_FEATURES or
                feature in self.SAFE_DEMOGRAPHIC_FEATURES):
                safe_features.append(feature)
            else:
                excluded_features.append(feature)

        logger.info(
            f"Generated safe feature set for '{target_biomarker}': "
            f"{len(safe_features)} included, {len(excluded_features)} excluded"
        )

        return {
            'safe_features': safe_features,
            'excluded_features': excluded_features
        }

    def print_report(self, report: LeakageReport):
        """Print a formatted leakage report."""
        print("\n" + "="*80)
        print("FEATURE LEAKAGE CHECK REPORT")
        print("="*80)

        if report.is_safe:
            print("✅ SAFE: No biomarker leakage or circular predictions detected")
        else:
            print("🚨 UNSAFE: Leakage detected!")

        if report.biomarker_leakage:
            print(f"\n🚨 Biomarker Leakage ({len(report.biomarker_leakage)} features):")
            for feature in report.biomarker_leakage:
                print(f"   - {feature}")

        if report.circular_predictions:
            print(f"\n⚠️  Circular Predictions ({len(report.circular_predictions)}):")
            for target, feature in report.circular_predictions:
                print(f"   - {target} ↔ {feature}")

        if report.high_correlation_pairs:
            print(f"\n⚠️  High Correlation Pairs ({len(report.high_correlation_pairs)}):")
            for feat1, feat2, corr in report.high_correlation_pairs[:10]:
                print(f"   - {feat1} ↔ {feat2} (r = {corr:.3f})")

        if report.warnings:
            print(f"\n⚠️  Warnings ({len(report.warnings)}):")
            for warning in report.warnings[:5]:
                print(f"   - {warning}")

        print("="*80 + "\n")


def example_usage():
    """Example usage of the LeakageChecker."""

    # Initialize checker
    checker = LeakageChecker(correlation_threshold=0.95)

    # Example 1: UNSAFE feature set (contains biomarker leakage)
    print("Example 1: Testing UNSAFE feature set")
    unsafe_features = [
        'climate_daily_mean_temp',
        'climate_7d_mean_temp',
        'HEAT_VULNERABILITY_SCORE',
        'Age (at enrolment)',
        'Sex',
        'hemoglobin_g_dL',  # ❌ Biomarker leakage!
    ]

    report = checker.check_features(
        target_biomarker='Hematocrit (%)',
        feature_list=unsafe_features
    )
    checker.print_report(report)

    # Example 2: SAFE feature set
    print("\nExample 2: Testing SAFE feature set")
    safe_features = [
        'climate_daily_mean_temp',
        'climate_7d_mean_temp',
        'climate_14d_mean_temp',
        'climate_30d_mean_temp',
        'climate_temp_anomaly',
        'climate_heat_stress_index',
        'HEAT_VULNERABILITY_SCORE',
        'Age (at enrolment)',
        'Sex',
        'month',
        'season',
    ]

    report = checker.check_features(
        target_biomarker='Hematocrit (%)',
        feature_list=safe_features
    )
    checker.print_report(report)

    # Example 3: Generate safe feature set
    print("\nExample 3: Generating safe feature set")
    all_available_features = [
        'climate_daily_mean_temp',
        'climate_7d_mean_temp',
        'HEAT_VULNERABILITY_SCORE',
        'Age (at enrolment)',
        'Sex',
        'hemoglobin_g_dL',  # Will be excluded
        'CD4 cell count (cells/µL)',  # Will be excluded
        'Hematocrit (%)',  # Target - will be excluded
    ]

    result = checker.generate_safe_feature_set(
        target_biomarker='Hematocrit (%)',
        all_features=all_available_features
    )

    print("Safe features:", result['safe_features'])
    print("Excluded features:", result['excluded_features'])


if __name__ == '__main__':
    example_usage()
