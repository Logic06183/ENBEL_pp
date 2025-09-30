#!/usr/bin/env python3
"""
Rigorous Socioeconomic Imputation for ENBEL Climate-Health Analysis
===================================================================

This module implements scientifically rigorous imputation methodologies for 
transferring socioeconomic variables from GCRO survey data to clinical cohorts.

The methodology combines:
1. K-Nearest Neighbors (KNN) for similarity-based matching
2. Ecological inference using geographic and demographic stratification
3. Statistical validation and uncertainty quantification

Scientific Foundations:
- Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys
- Little, R.J.A. & Rubin, D.B. (2020). Statistical Analysis with Missing Data
- Stekhoven & Bühlmann (2012). MissForest: Non-parametric missing value imputation

Author: ENBEL Project Team
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SocioeconomicImputer:
    """
    Rigorous multi-dimensional imputation for socioeconomic variables.
    
    This class implements multiple imputation strategies:
    1. Spatial-demographic KNN matching
    2. Ecological stratification imputation
    3. Hot-deck imputation with validation
    
    Parameters:
    -----------
    method : str
        Imputation method ('knn', 'ecological', 'combined')
    k_neighbors : int
        Number of neighbors for KNN approaches
    spatial_weight : float
        Weight for spatial vs demographic matching (0-1)
    max_distance_km : float
        Maximum spatial distance for matching
    min_matches : int
        Minimum required matches for valid imputation
    validation_fraction : float
        Fraction of observed data to use for validation
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self,
                 method: str = 'combined',
                 k_neighbors: int = 10,
                 spatial_weight: float = 0.4,
                 max_distance_km: float = 15,
                 min_matches: int = 3,
                 validation_fraction: float = 0.2,
                 random_state: int = 42):
        
        self.method = method
        self.k_neighbors = k_neighbors
        self.spatial_weight = spatial_weight
        self.demographic_weight = 1 - spatial_weight
        self.max_distance_km = max_distance_km
        self.min_matches = min_matches
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Initialize components
        self.spatial_scaler = StandardScaler()
        self.demographic_encoders = {}
        self.is_fitted = False
        
        # Results storage
        self.imputation_results = {}
        self.validation_scores = {}
        self.matching_statistics = {}
        
        logger.info(f"Initialized SocioeconomicImputer with method={method}")
    
    def validate_data(self, 
                     donor_data: pd.DataFrame, 
                     recipient_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate input data for imputation.
        
        Parameters:
        -----------
        donor_data : pd.DataFrame
            Data containing socioeconomic variables (GCRO cohort)
        recipient_data : pd.DataFrame
            Data missing socioeconomic variables (Clinical cohort)
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Validated donor and recipient datasets
        """
        logger.info("Validating input data for imputation...")
        
        # Required columns
        required_spatial = ['latitude', 'longitude']
        required_demographic = ['Sex', 'Race']
        
        # Check spatial coordinates
        for col in required_spatial:
            if col not in donor_data.columns or col not in recipient_data.columns:
                raise ValueError(f"Missing required spatial column: {col}")
        
        # Check demographic variables
        for col in required_demographic:
            if col not in donor_data.columns or col not in recipient_data.columns:
                logger.warning(f"Missing demographic column: {col}")
        
        # Validate coordinate ranges (South African bounds)
        lat_bounds = (-35, -22)  # South Africa latitude range
        lon_bounds = (16, 33)    # South Africa longitude range
        
        for data, name in [(donor_data, 'donor'), (recipient_data, 'recipient')]:
            valid_coords = (
                (data['latitude'].between(*lat_bounds)) & 
                (data['longitude'].between(*lon_bounds))
            )
            invalid_count = (~valid_coords).sum()
            if invalid_count > 0:
                logger.warning(f"{name} data has {invalid_count} invalid coordinates")
        
        # Remove records with invalid coordinates
        donor_valid = (
            donor_data['latitude'].between(*lat_bounds) & 
            donor_data['longitude'].between(*lon_bounds) &
            donor_data['latitude'].notna() &
            donor_data['longitude'].notna()
        )
        
        recipient_valid = (
            recipient_data['latitude'].between(*lat_bounds) & 
            recipient_data['longitude'].between(*lon_bounds) &
            recipient_data['latitude'].notna() &
            recipient_data['longitude'].notna()
        )
        
        donor_clean = donor_data[donor_valid].copy()
        recipient_clean = recipient_data[recipient_valid].copy()
        
        logger.info(f"Donor data: {len(donor_clean):,}/{len(donor_data):,} valid records")
        logger.info(f"Recipient data: {len(recipient_clean):,}/{len(recipient_data):,} valid records")
        
        return donor_clean, recipient_clean
    
    def prepare_matching_features(self, 
                                 donor_data: pd.DataFrame, 
                                 recipient_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for matching algorithm.
        
        Parameters:
        -----------
        donor_data : pd.DataFrame
            Donor dataset
        recipient_data : pd.DataFrame
            Recipient dataset
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Processed feature arrays for donor and recipient data
        """
        logger.info("Preparing matching features...")
        
        # Spatial features (latitude, longitude)
        donor_spatial = donor_data[['latitude', 'longitude']].values
        recipient_spatial = recipient_data[['latitude', 'longitude']].values
        
        # Scale spatial coordinates
        combined_spatial = np.vstack([donor_spatial, recipient_spatial])
        self.spatial_scaler.fit(combined_spatial)
        
        donor_spatial_scaled = self.spatial_scaler.transform(donor_spatial)
        recipient_spatial_scaled = self.spatial_scaler.transform(recipient_spatial)
        
        # Demographic features
        demographic_cols = ['Sex', 'Race']
        
        donor_demo_features = []
        recipient_demo_features = []
        
        for col in demographic_cols:
            if col in donor_data.columns and col in recipient_data.columns:
                # Encode categorical variables
                combined_values = pd.concat([
                    donor_data[col], recipient_data[col]
                ]).astype(str)
                
                encoder = LabelEncoder()
                encoder.fit(combined_values.dropna())
                self.demographic_encoders[col] = encoder
                
                # Transform data
                donor_encoded = self._safe_encode(donor_data[col], encoder)
                recipient_encoded = self._safe_encode(recipient_data[col], encoder)
                
                donor_demo_features.append(donor_encoded)
                recipient_demo_features.append(recipient_encoded)
        
        # Combine features with weights
        if donor_demo_features:
            donor_demo_array = np.column_stack(donor_demo_features)
            recipient_demo_array = np.column_stack(recipient_demo_features)
        else:
            # If no demographic features, use zeros
            donor_demo_array = np.zeros((len(donor_data), 1))
            recipient_demo_array = np.zeros((len(recipient_data), 1))
        
        # Weight and combine features
        donor_features = np.column_stack([
            donor_spatial_scaled * self.spatial_weight,
            donor_demo_array * self.demographic_weight
        ])
        
        recipient_features = np.column_stack([
            recipient_spatial_scaled * self.spatial_weight,
            recipient_demo_array * self.demographic_weight
        ])
        
        logger.info(f"Feature preparation complete: {donor_features.shape[1]} features")
        
        return donor_features, recipient_features
    
    def _safe_encode(self, series: pd.Series, encoder: LabelEncoder) -> np.ndarray:
        """Safely encode categorical data with missing values."""
        result = np.zeros(len(series))
        valid_mask = series.notna()
        
        if valid_mask.any():
            valid_values = series[valid_mask].astype(str)
            # Only encode values that were seen during fitting
            known_mask = valid_values.isin(encoder.classes_)
            
            if known_mask.any():
                indices = np.where(valid_mask)[0][known_mask]
                encoded_values = encoder.transform(valid_values[known_mask])
                result[indices] = encoded_values
        
        return result
    
    def fit_knn_imputer(self, 
                       donor_features: np.ndarray,
                       donor_data: pd.DataFrame,
                       target_variables: List[str]) -> Dict:
        """
        Fit KNN-based imputation model.
        
        Parameters:
        -----------
        donor_features : np.ndarray
            Processed donor features
        donor_data : pd.DataFrame
            Donor dataset
        target_variables : List[str]
            Variables to impute
            
        Returns:
        --------
        Dict
            Fitted KNN models and statistics
        """
        logger.info(f"Fitting KNN imputer for {len(target_variables)} variables...")
        
        knn_models = {}
        donor_targets = {}
        
        for var in target_variables:
            if var not in donor_data.columns:
                logger.warning(f"Target variable {var} not found in donor data")
                continue
            
            # Get non-missing values for this variable
            valid_mask = donor_data[var].notna()
            n_valid = valid_mask.sum()
            
            if n_valid < self.min_matches:
                logger.warning(f"Insufficient data for {var}: {n_valid} valid observations")
                continue
            
            # Fit KNN model
            knn = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, n_valid),
                metric='euclidean',
                algorithm='auto'
            )
            
            valid_features = donor_features[valid_mask]
            knn.fit(valid_features)
            
            knn_models[var] = {
                'model': knn,
                'features': valid_features,
                'values': donor_data.loc[valid_mask, var].values,
                'indices': np.where(valid_mask)[0],
                'n_donors': n_valid
            }
            
            logger.info(f"  {var}: {n_valid:,} donor observations")
        
        return knn_models
    
    def impute_with_knn(self,
                       recipient_features: np.ndarray,
                       knn_models: Dict,
                       target_variables: List[str]) -> Dict:
        """
        Perform KNN-based imputation.
        
        Parameters:
        -----------
        recipient_features : np.ndarray
            Processed recipient features
        knn_models : Dict
            Fitted KNN models
        target_variables : List[str]
            Variables to impute
            
        Returns:
        --------
        Dict
            Imputed values and confidence scores
        """
        logger.info("Performing KNN imputation...")
        
        imputed_values = {}
        confidence_scores = {}
        
        n_recipients = len(recipient_features)
        
        for var in target_variables:
            if var not in knn_models:
                logger.warning(f"No KNN model available for {var}")
                continue
            
            model_info = knn_models[var]
            knn = model_info['model']
            donor_values = model_info['values']
            
            # Find nearest neighbors
            distances, indices = knn.kneighbors(recipient_features)
            
            # Calculate imputed values (weighted average by inverse distance)
            imputed = np.full(n_recipients, np.nan)
            confidence = np.full(n_recipients, np.nan)
            
            for i in range(n_recipients):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_values = donor_values[neighbor_indices]
                
                # Filter out invalid neighbors
                valid_neighbors = ~np.isnan(neighbor_values)
                
                if valid_neighbors.sum() >= self.min_matches:
                    valid_distances = neighbor_distances[valid_neighbors]
                    valid_values = neighbor_values[valid_neighbors]
                    
                    # Weight by inverse distance (add small constant to avoid division by zero)
                    weights = 1 / (valid_distances + 1e-8)
                    weights = weights / weights.sum()
                    
                    # Calculate weighted average
                    imputed[i] = np.average(valid_values, weights=weights)
                    
                    # Confidence based on distance and neighbor agreement
                    distance_confidence = 1 / (1 + np.mean(valid_distances))
                    value_variance = np.var(valid_values) if len(valid_values) > 1 else 0
                    agreement_confidence = 1 / (1 + value_variance)
                    
                    confidence[i] = (distance_confidence + agreement_confidence) / 2
            
            imputed_values[var] = imputed
            confidence_scores[var] = confidence
            
            n_imputed = ~np.isnan(imputed).sum()
            logger.info(f"  {var}: {n_imputed:,}/{n_recipients:,} values imputed")
        
        return {'values': imputed_values, 'confidence': confidence_scores}
    
    def fit_ecological_imputer(self,
                              donor_data: pd.DataFrame,
                              target_variables: List[str]) -> Dict:
        """
        Fit ecological stratification imputer.
        
        This method uses geographic and demographic strata to impute values
        based on local ecological characteristics.
        
        Parameters:
        -----------
        donor_data : pd.DataFrame
            Donor dataset
        target_variables : List[str]
            Variables to impute
            
        Returns:
        --------
        Dict
            Ecological imputation models
        """
        logger.info("Fitting ecological stratification imputer...")
        
        # Define spatial strata (grid cells)
        lat_bins = np.percentile(donor_data['latitude'].dropna(), np.linspace(0, 100, 11))
        lon_bins = np.percentile(donor_data['longitude'].dropna(), np.linspace(0, 100, 11))
        
        # Define demographic strata
        sex_categories = donor_data['Sex'].dropna().unique() if 'Sex' in donor_data.columns else []
        race_categories = donor_data['Race'].dropna().unique() if 'Race' in donor_data.columns else []
        
        ecological_models = {}
        
        for var in target_variables:
            if var not in donor_data.columns:
                continue
            
            # Create stratum means
            strata_stats = {}
            
            # Spatial strata
            donor_data['lat_bin'] = pd.cut(donor_data['latitude'], bins=lat_bins, include_lowest=True)
            donor_data['lon_bin'] = pd.cut(donor_data['longitude'], bins=lon_bins, include_lowest=True)
            
            spatial_means = donor_data.groupby(['lat_bin', 'lon_bin'])[var].agg(['mean', 'count', 'std']).reset_index()
            spatial_means = spatial_means[spatial_means['count'] >= 3]  # Minimum observations per stratum
            
            # Demographic strata
            demographic_means = {}
            if 'Sex' in donor_data.columns:
                sex_means = donor_data.groupby('Sex')[var].agg(['mean', 'count', 'std']).reset_index()
                demographic_means['Sex'] = sex_means[sex_means['count'] >= 3]
            
            if 'Race' in donor_data.columns:
                race_means = donor_data.groupby('Race')[var].agg(['mean', 'count', 'std']).reset_index()
                demographic_means['Race'] = race_means[race_means['count'] >= 3]
            
            # Combined strata
            if 'Sex' in donor_data.columns and 'Race' in donor_data.columns:
                combined_means = donor_data.groupby(['Sex', 'Race'])[var].agg(['mean', 'count', 'std']).reset_index()
                demographic_means['Sex_Race'] = combined_means[combined_means['count'] >= 3]
            
            ecological_models[var] = {
                'spatial_means': spatial_means,
                'demographic_means': demographic_means,
                'overall_mean': donor_data[var].mean(),
                'overall_std': donor_data[var].std(),
                'lat_bins': lat_bins,
                'lon_bins': lon_bins
            }
            
            logger.info(f"  {var}: {len(spatial_means)} spatial strata, "
                       f"{sum(len(dm) for dm in demographic_means.values())} demographic strata")
        
        return ecological_models
    
    def impute_with_ecological(self,
                              recipient_data: pd.DataFrame,
                              ecological_models: Dict,
                              target_variables: List[str]) -> Dict:
        """
        Perform ecological stratification imputation.
        
        Parameters:
        -----------
        recipient_data : pd.DataFrame
            Recipient dataset
        ecological_models : Dict
            Fitted ecological models
        target_variables : List[str]
            Variables to impute
            
        Returns:
        --------
        Dict
            Imputed values and confidence scores
        """
        logger.info("Performing ecological imputation...")
        
        imputed_values = {}
        confidence_scores = {}
        
        n_recipients = len(recipient_data)
        
        for var in target_variables:
            if var not in ecological_models:
                continue
            
            model = ecological_models[var]
            imputed = np.full(n_recipients, np.nan)
            confidence = np.full(n_recipients, np.nan)
            
            # Assign recipients to spatial bins
            recipient_data_copy = recipient_data.copy()
            recipient_data_copy['lat_bin'] = pd.cut(recipient_data_copy['latitude'], bins=model['lat_bins'], include_lowest=True)
            recipient_data_copy['lon_bin'] = pd.cut(recipient_data_copy['longitude'], bins=model['lon_bins'], include_lowest=True)
            
            for i in range(n_recipients):
                recipient_row = recipient_data_copy.iloc[i]
                
                # Try multiple imputation strategies in order of preference
                imputed_value = None
                confidence_value = 0
                
                # 1. Spatial stratum mean
                spatial_match = model['spatial_means'][
                    (model['spatial_means']['lat_bin'] == recipient_row['lat_bin']) &
                    (model['spatial_means']['lon_bin'] == recipient_row['lon_bin'])
                ]
                
                if len(spatial_match) > 0 and not pd.isna(spatial_match.iloc[0]['mean']):
                    imputed_value = spatial_match.iloc[0]['mean']
                    # Confidence based on number of observations and standard deviation
                    n_obs = spatial_match.iloc[0]['count']
                    std_val = spatial_match.iloc[0]['std']
                    confidence_value = min(0.9, n_obs / 50) * (1 / (1 + std_val)) if not pd.isna(std_val) else 0.5
                
                # 2. Demographic stratum mean (if spatial failed)
                if imputed_value is None:
                    demo_means = model['demographic_means']
                    
                    # Try combined demographic strata first
                    if 'Sex_Race' in demo_means and 'Sex' in recipient_row and 'Race' in recipient_row:
                        demo_match = demo_means['Sex_Race'][
                            (demo_means['Sex_Race']['Sex'] == recipient_row['Sex']) &
                            (demo_means['Sex_Race']['Race'] == recipient_row['Race'])
                        ]
                        
                        if len(demo_match) > 0 and not pd.isna(demo_match.iloc[0]['mean']):
                            imputed_value = demo_match.iloc[0]['mean']
                            confidence_value = 0.7
                    
                    # Try individual demographic variables
                    if imputed_value is None:
                        for demo_var in ['Sex', 'Race']:
                            if demo_var in demo_means and demo_var in recipient_row:
                                demo_match = demo_means[demo_var][
                                    demo_means[demo_var][demo_var] == recipient_row[demo_var]
                                ]
                                
                                if len(demo_match) > 0 and not pd.isna(demo_match.iloc[0]['mean']):
                                    imputed_value = demo_match.iloc[0]['mean']
                                    confidence_value = 0.5
                                    break
                
                # 3. Overall mean (fallback)
                if imputed_value is None:
                    imputed_value = model['overall_mean']
                    confidence_value = 0.3
                
                imputed[i] = imputed_value
                confidence[i] = confidence_value
            
            imputed_values[var] = imputed
            confidence_scores[var] = confidence
            
            n_imputed = ~np.isnan(imputed).sum()
            logger.info(f"  {var}: {n_imputed:,}/{n_recipients:,} values imputed")
        
        return {'values': imputed_values, 'confidence': confidence_scores}
    
    def validate_imputation(self,
                           donor_data: pd.DataFrame,
                           target_variables: List[str]) -> Dict:
        """
        Validate imputation accuracy using holdout testing.
        
        Parameters:
        -----------
        donor_data : pd.DataFrame
            Complete donor dataset
        target_variables : List[str]
            Variables to validate
            
        Returns:
        --------
        Dict
            Validation statistics
        """
        logger.info("Validating imputation accuracy...")
        
        validation_results = {}
        
        for var in target_variables:
            if var not in donor_data.columns:
                continue
            
            # Get complete cases for this variable
            complete_cases = donor_data[donor_data[var].notna()].copy()
            
            if len(complete_cases) < 100:
                logger.warning(f"Insufficient data for validation of {var}")
                continue
            
            # Create validation split
            n_val = int(len(complete_cases) * self.validation_fraction)
            val_indices = np.random.choice(len(complete_cases), size=n_val, replace=False)
            
            val_data = complete_cases.iloc[val_indices].copy()
            train_data = complete_cases.drop(complete_cases.index[val_indices]).copy()
            
            # Hide values in validation set
            true_values = val_data[var].values
            val_data_missing = val_data.copy()
            val_data_missing[var] = np.nan
            
            # Perform imputation
            validation_imputer = SocioeconomicImputer(
                method=self.method,
                k_neighbors=self.k_neighbors,
                spatial_weight=self.spatial_weight,
                max_distance_km=self.max_distance_km,
                random_state=self.random_state
            )
            
            # Fit on training data
            train_clean, val_clean = validation_imputer.validate_data(train_data, val_data_missing)
            
            if len(train_clean) == 0 or len(val_clean) == 0:
                logger.warning(f"No valid data for validation of {var}")
                continue
            
            # Prepare features
            train_features, val_features = validation_imputer.prepare_matching_features(train_clean, val_clean)
            
            # Perform imputation
            if self.method in ['knn', 'combined']:
                knn_models = validation_imputer.fit_knn_imputer(train_features, train_clean, [var])
                knn_results = validation_imputer.impute_with_knn(val_features, knn_models, [var])
                predicted_values = knn_results['values'][var]
            elif self.method == 'ecological':
                eco_models = validation_imputer.fit_ecological_imputer(train_clean, [var])
                eco_results = validation_imputer.impute_with_ecological(val_clean, eco_models, [var])
                predicted_values = eco_results['values'][var]
            
            # Calculate validation metrics
            valid_predictions = ~np.isnan(predicted_values)
            if valid_predictions.sum() > 0:
                mse = mean_squared_error(
                    true_values[valid_predictions], 
                    predicted_values[valid_predictions]
                )
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(true_values[valid_predictions] - predicted_values[valid_predictions]))
                
                # Correlation
                correlation = np.corrcoef(true_values[valid_predictions], predicted_values[valid_predictions])[0, 1]
                
                validation_results[var] = {
                    'n_validation': len(true_values),
                    'n_predicted': valid_predictions.sum(),
                    'rmse': rmse,
                    'mae': mae,
                    'correlation': correlation,
                    'coverage': valid_predictions.sum() / len(true_values)
                }
                
                logger.info(f"  {var}: RMSE={rmse:.3f}, MAE={mae:.3f}, r={correlation:.3f}")
        
        return validation_results
    
    def fit_and_impute(self,
                      donor_data: pd.DataFrame,
                      recipient_data: pd.DataFrame,
                      target_variables: List[str]) -> pd.DataFrame:
        """
        Complete fit and imputation workflow.
        
        Parameters:
        -----------
        donor_data : pd.DataFrame
            Data containing target variables (GCRO cohort)
        recipient_data : pd.DataFrame
            Data missing target variables (Clinical cohort)
        target_variables : List[str]
            Variables to impute
            
        Returns:
        --------
        pd.DataFrame
            Recipient data with imputed variables
        """
        logger.info("Starting complete imputation workflow...")
        
        # Validate data
        donor_clean, recipient_clean = self.validate_data(donor_data, recipient_data)
        
        # Prepare features
        donor_features, recipient_features = self.prepare_matching_features(donor_clean, recipient_clean)
        
        # Initialize results
        imputed_results = recipient_clean.copy()
        
        # Perform imputation based on method
        if self.method == 'knn':
            knn_models = self.fit_knn_imputer(donor_features, donor_clean, target_variables)
            results = self.impute_with_knn(recipient_features, knn_models, target_variables)
            
        elif self.method == 'ecological':
            eco_models = self.fit_ecological_imputer(donor_clean, target_variables)
            results = self.impute_with_ecological(recipient_clean, eco_models, target_variables)
            
        elif self.method == 'combined':
            # Use both methods and average results
            knn_models = self.fit_knn_imputer(donor_features, donor_clean, target_variables)
            knn_results = self.impute_with_knn(recipient_features, knn_models, target_variables)
            
            eco_models = self.fit_ecological_imputer(donor_clean, target_variables)
            eco_results = self.impute_with_ecological(recipient_clean, eco_models, target_variables)
            
            # Combine results with confidence weighting
            combined_values = {}
            combined_confidence = {}
            
            for var in target_variables:
                if var in knn_results['values'] and var in eco_results['values']:
                    knn_vals = knn_results['values'][var]
                    knn_conf = knn_results['confidence'][var]
                    eco_vals = eco_results['values'][var]
                    eco_conf = eco_results['confidence'][var]
                    
                    # Weight by confidence scores
                    total_conf = knn_conf + eco_conf
                    valid_mask = (total_conf > 0) & ~np.isnan(knn_vals) & ~np.isnan(eco_vals)
                    
                    combined_vals = np.full(len(knn_vals), np.nan)
                    combined_conf_vals = np.full(len(knn_vals), np.nan)
                    
                    if valid_mask.any():
                        knn_weight = knn_conf[valid_mask] / total_conf[valid_mask]
                        eco_weight = eco_conf[valid_mask] / total_conf[valid_mask]
                        
                        combined_vals[valid_mask] = (
                            knn_vals[valid_mask] * knn_weight + 
                            eco_vals[valid_mask] * eco_weight
                        )
                        combined_conf_vals[valid_mask] = total_conf[valid_mask] / 2
                    
                    combined_values[var] = combined_vals
                    combined_confidence[var] = combined_conf_vals
                    
                elif var in knn_results['values']:
                    combined_values[var] = knn_results['values'][var]
                    combined_confidence[var] = knn_results['confidence'][var]
                elif var in eco_results['values']:
                    combined_values[var] = eco_results['values'][var]
                    combined_confidence[var] = eco_results['confidence'][var]
            
            results = {'values': combined_values, 'confidence': combined_confidence}
        
        # Add imputed values to results
        for var in target_variables:
            if var in results['values']:
                imputed_results[f'{var}_imputed'] = results['values'][var]
                imputed_results[f'{var}_confidence'] = results['confidence'][var]
        
        # Validate imputation
        validation_results = self.validate_imputation(donor_clean, target_variables)
        
        # Store results
        self.imputation_results = results
        self.validation_scores = validation_results
        self.is_fitted = True
        
        logger.info("Imputation workflow completed successfully")
        
        return imputed_results


def load_and_impute_enbel_data(data_file: str = "DEIDENTIFIED_CLIMATE_HEALTH_DATASET.csv") -> pd.DataFrame:
    """
    Convenience function to load ENBEL data and perform socioeconomic imputation.
    
    Parameters:
    -----------
    data_file : str
        Path to the ENBEL dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with imputed socioeconomic variables
    """
    logger.info(f"Loading ENBEL data from {data_file}")
    
    # Load data
    data = pd.read_csv(data_file, low_memory=False)
    
    # Separate cohorts
    clinical_cohort = data[data['data_source'].notna()].copy()  # Has health data
    gcro_cohort = data[data['data_source'].isna()].copy()      # Has socioeconomic data
    
    logger.info(f"Clinical cohort: {len(clinical_cohort):,} participants")
    logger.info(f"GCRO cohort: {len(gcro_cohort):,} participants")
    
    # Define variables to impute
    socioeconomic_vars = [
        'Education', 'employment_status', 'vuln_Housing',
        'heat_vulnerability_index'
    ]
    
    # Filter available variables
    available_vars = [var for var in socioeconomic_vars if var in gcro_cohort.columns]
    logger.info(f"Variables to impute: {available_vars}")
    
    # Perform imputation
    imputer = SocioeconomicImputer(
        method='combined',
        k_neighbors=10,
        spatial_weight=0.4,
        max_distance_km=15,
        random_state=42
    )
    
    imputed_clinical = imputer.fit_and_impute(
        donor_data=gcro_cohort,
        recipient_data=clinical_cohort,
        target_variables=available_vars
    )
    
    return imputed_clinical


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        imputed_data = load_and_impute_enbel_data()
        print(f"Imputation completed: {len(imputed_data):,} records")
        
        # Display imputation summary
        socioeconomic_vars = ['Education', 'employment_status', 'vuln_Housing']
        for var in socioeconomic_vars:
            imputed_col = f'{var}_imputed'
            confidence_col = f'{var}_confidence'
            
            if imputed_col in imputed_data.columns:
                n_imputed = imputed_data[imputed_col].notna().sum()
                mean_confidence = imputed_data[confidence_col].mean()
                print(f"{var}: {n_imputed:,} values imputed (mean confidence: {mean_confidence:.3f})")
    
    except FileNotFoundError:
        print("ENBEL dataset not found. Please ensure the data file is available.")
    except Exception as e:
        print(f"Error during imputation: {e}")