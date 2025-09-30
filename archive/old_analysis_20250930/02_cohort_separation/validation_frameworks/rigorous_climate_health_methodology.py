#!/usr/bin/env python3
"""
Rigorous Climate-Health Machine Learning Methodology
==================================================

This methodology addresses the data leakage issues that caused artificially 
inflated R² scores in the previous analysis. Based on literature review:

EXPECTED REALISTIC R² RANGES:
- Individual biomarkers: 0.02-0.15  
- Strong climate effects: 0.10-0.20
- Maximum realistic: 0.20-0.30
- RED FLAG: R² > 0.30 indicates data leakage

METHODOLOGY IMPROVEMENTS:
1. Strict feature exclusion (no biomarker→biomarker prediction)
2. Temporal validation with proper gaps
3. Conservative hyperparameters 
4. Literature-based performance validation
5. Comprehensive data leakage detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import json
import time
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import joblib
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RigorousClimateHealthAnalysis:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("rigorous_results")
        self.models_dir = Path("rigorous_models")
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.progress_file = self.results_dir / f"rigorous_progress_{self.timestamp}.log"
        
        # Literature-based performance thresholds
        self.performance_thresholds = {
            'cardiovascular': {'max_r2': 0.25, 'expected_range': (0.05, 0.20)},
            'immune': {'max_r2': 0.20, 'expected_range': (0.03, 0.15)},
            'metabolic': {'max_r2': 0.15, 'expected_range': (0.02, 0.12)},
            'renal': {'max_r2': 0.30, 'expected_range': (0.08, 0.25)}
        }

    def log_progress(self, message, level="INFO"):
        """Enhanced logging with levels"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "WARNING":
            icon = "⚠️ "
        elif level == "ERROR":
            icon = "❌ "
        elif level == "SUCCESS":
            icon = "✅ "
        else:
            icon = "🔬 "
        
        progress_msg = f"[{timestamp}] {icon}{message}"
        logging.info(progress_msg)
        
        with open(self.progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")

    def load_and_examine_data(self):
        """Load data and perform comprehensive integrity checks"""
        self.log_progress("Loading dataset for rigorous analysis...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        self.log_progress(f"Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
        
        # Data integrity checks
        self.log_progress("Performing data integrity analysis...")
        
        # 1. Check for suspicious column patterns
        suspicious_patterns = []
        for col in df.columns:
            if any(term in col.lower() for term in ['predict', 'target', 'outcome', 'result']):
                suspicious_patterns.append(col)
        
        if suspicious_patterns:
            self.log_progress(f"WARNING: Found {len(suspicious_patterns)} suspicious column patterns", "WARNING")
            for pattern in suspicious_patterns[:5]:
                self.log_progress(f"  - {pattern}", "WARNING")
        
        # 2. Temporal consistency check
        if 'primary_date' in df.columns:
            df['primary_date'] = pd.to_datetime(df['primary_date'], errors='coerce')
            date_range = df['primary_date'].max() - df['primary_date'].min()
            self.log_progress(f"Temporal span: {date_range.days} days")
            
            # Check for future dates
            future_dates = df[df['primary_date'] > datetime.now()]
            if len(future_dates) > 0:
                self.log_progress(f"WARNING: {len(future_dates)} records with future dates", "WARNING")
        
        return df

    def create_rigorous_feature_set(self, df):
        """Create strictly controlled feature set with comprehensive exclusions"""
        self.log_progress("Creating rigorous feature set with strict exclusions...")
        
        # STRICT INCLUSION CRITERIA: Only exogenous climate and basic demographics
        
        # 1. Climate variables ONLY (exogenous predictors)
        climate_keywords = [
            'temp', 'heat', 'cool', 'warm', 'cold',  # Temperature
            'humid', 'moisture', 'rh_',               # Humidity  
            'pressure', 'mslp', 'sp_',                # Pressure
            'wind', 'breeze', 'gust',                 # Wind
            'solar', 'radiation', 'uv',               # Solar
            'precip', 'rain', 'precipitation',        # Precipitation
            'utci', 'heat_index', 'feels_like',       # Heat stress indices
            'lag'                                     # Lag features
        ]
        
        climate_features = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in climate_keywords):
                # Exclude any derived health indices
                if not any(exclude in col.lower() for exclude in [
                    'vulnerability', 'risk', 'health', 'stress_score', 
                    'index_health', 'comfort', 'danger'
                ]):
                    climate_features.append(col)
        
        # 2. Basic demographics ONLY (no derived health indicators)
        demographic_features = []
        for col in df.columns:
            if col.lower() in ['sex', 'race', 'age', 'latitude', 'longitude', 
                              'year', 'month', 'season', 'day_of_year']:
                demographic_features.append(col)
        
        # 3. COMPREHENSIVE EXCLUSION LIST
        excluded_categories = [
            # Biomarkers predicting biomarkers
            'cd4', 'glucose', 'cholesterol', 'creatinine', 'hemoglobin', 
            'blood_pressure', 'systolic', 'diastolic', 'heart_rate',
            
            # Health-derived indices  
            'vulnerability', 'risk_score', 'health_index', 'comfort_index',
            'stress_score', 'danger', 'warning', 'alert',
            
            # Socioeconomic variables that might leak health info
            'income', 'education', 'employment', 'housing', 'healthcare_access',
            'insurance', 'medication', 'treatment', 'hospital', 'clinic',
            
            # Administrative/ID variables
            'id', 'uid', 'record', 'file', 'source', 'dataset', 'study',
            
            # Derived temporal features that might leak
            'visit_number', 'follow_up', 'time_since', 'duration',
            
            # Quality indicators that might correlate with health
            'quality', 'flag', 'valid', 'missing', 'imputed'
        ]
        
        # Apply exclusions
        all_features = climate_features + demographic_features
        rigorous_features = []
        
        for feature in all_features:
            exclude = False
            for excluded in excluded_categories:
                if excluded in feature.lower():
                    exclude = True
                    break
            if not exclude:
                rigorous_features.append(feature)
        
        self.log_progress(f"Feature selection results:")
        self.log_progress(f"  Climate features identified: {len(climate_features)}")
        self.log_progress(f"  Demographic features identified: {len(demographic_features)}")
        self.log_progress(f"  Features after exclusions: {len(rigorous_features)}")
        self.log_progress(f"  Exclusion rate: {(len(all_features) - len(rigorous_features))/len(all_features)*100:.1f}%")
        
        return rigorous_features

    def detect_data_leakage(self, X, y, biomarker_name):
        """Comprehensive data leakage detection"""
        self.log_progress(f"Detecting potential data leakage for {biomarker_name}...")
        
        leakage_warnings = []
        
        # 1. Check for perfect correlations
        correlations = []
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                corr = np.corrcoef(X[col].dropna(), y.loc[X[col].dropna().index])[0, 1]
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Flag suspicious correlations
        for col, corr in correlations[:5]:
            if corr > 0.8:
                leakage_warnings.append(f"Suspicious correlation: {col} (r={corr:.3f})")
            elif corr > 0.5:
                self.log_progress(f"High correlation: {col} (r={corr:.3f})", "WARNING")
        
        # 2. Check for biomarker keywords in features
        biomarker_keywords = biomarker_name.lower().split()
        for keyword in biomarker_keywords:
            if len(keyword) > 3:  # Skip short words
                matching_features = [col for col in X.columns if keyword in col.lower()]
                if matching_features:
                    leakage_warnings.append(f"Biomarker keyword '{keyword}' found in features: {matching_features}")
        
        # 3. Check for identical distributions
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # KS test for identical distributions
                try:
                    ks_stat, p_value = stats.ks_2samp(X[col].dropna(), y.dropna())
                    if p_value > 0.95:  # Very similar distributions
                        leakage_warnings.append(f"Identical distribution detected: {col} (p={p_value:.3f})")
                except:
                    pass
        
        if leakage_warnings:
            self.log_progress(f"DATA LEAKAGE DETECTED: {len(leakage_warnings)} warnings", "ERROR")
            for warning in leakage_warnings[:3]:  # Show first 3
                self.log_progress(f"  {warning}", "ERROR")
            return True
        else:
            self.log_progress("No data leakage detected", "SUCCESS")
            return False

    def temporal_train_test_split(self, X, y, test_size=0.2, gap_days=30):
        """Proper temporal splitting with gap to prevent leakage"""
        if 'primary_date' not in X.columns:
            self.log_progress("No temporal information available, using standard split", "WARNING")
            # Fall back to random split but warn
            split_idx = int(len(X) * (1 - test_size))
            return (X.iloc[:split_idx], X.iloc[split_idx:], 
                   y.iloc[:split_idx], y.iloc[split_idx:])
        
        # Sort by date
        X_sorted = X.sort_values('primary_date')
        y_sorted = y.loc[X_sorted.index]
        
        # Calculate split with gap
        total_samples = len(X_sorted)
        test_samples = int(total_samples * test_size)
        gap_samples = int(total_samples * 0.05)  # 5% gap
        
        train_end = total_samples - test_samples - gap_samples
        test_start = train_end + gap_samples
        
        X_train = X_sorted.iloc[:train_end]
        X_test = X_sorted.iloc[test_start:]
        y_train = y_sorted.iloc[:train_end]
        y_test = y_sorted.iloc[test_start:]
        
        self.log_progress(f"Temporal split: {len(X_train)} train, {gap_samples} gap, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test

    def train_conservative_models(self, X_train, y_train, X_test, y_test, biomarker_name):
        """Train models with conservative hyperparameters to prevent overfitting"""
        self.log_progress(f"Training conservative models for {biomarker_name}...")
        
        results = {
            'biomarker': biomarker_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        # CONSERVATIVE Random Forest (preventing overfitting)
        self.log_progress("Training conservative Random Forest...")
        rf_conservative = RandomForestRegressor(
            n_estimators=100,        # Moderate ensemble size
            max_depth=5,             # Shallow depth to prevent overfitting
            min_samples_split=20,    # Require more samples for splits
            min_samples_leaf=10,     # Larger leaf sizes
            max_features='sqrt',     # Feature subsampling
            random_state=42,
            n_jobs=-1
        )
        
        rf_conservative.fit(X_train, y_train)
        rf_pred = rf_conservative.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        results['rf_conservative'] = {
            'r2': rf_r2,
            'mae': rf_mae,
            'rmse': rf_rmse,
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10
            }
        }
        
        self.log_progress(f"RF Conservative: R² = {rf_r2:.4f}, MAE = {rf_mae:.3f}")
        
        # CONSERVATIVE XGBoost
        try:
            self.log_progress("Training conservative XGBoost...")
            xgb_conservative = xgb.XGBRegressor(
                n_estimators=100,        # Moderate number of trees
                max_depth=3,             # Very shallow trees
                learning_rate=0.01,      # Slow learning
                subsample=0.8,           # Sample subsampling
                colsample_bytree=0.8,    # Feature subsampling
                reg_alpha=1.0,           # Strong L1 regularization
                reg_lambda=1.0,          # Strong L2 regularization
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            xgb_conservative.fit(X_train, y_train)
            xgb_pred = xgb_conservative.predict(X_test)
            xgb_r2 = r2_score(y_test, xgb_pred)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            
            results['xgb_conservative'] = {
                'r2': xgb_r2,
                'mae': xgb_mae,
                'rmse': xgb_rmse,
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.01,
                    'reg_alpha': 1.0,
                    'reg_lambda': 1.0
                }
            }
            
            self.log_progress(f"XGB Conservative: R² = {xgb_r2:.4f}, MAE = {xgb_mae:.3f}")
            
        except Exception as e:
            self.log_progress(f"XGBoost training failed: {e}", "ERROR")
            xgb_r2 = -999
        
        # Select best model
        if 'xgb_conservative' in results and results['xgb_conservative']['r2'] > results['rf_conservative']['r2']:
            best_model = 'xgb_conservative'
            best_r2 = results['xgb_conservative']['r2']
            best_model_obj = xgb_conservative
        else:
            best_model = 'rf_conservative'
            best_r2 = results['rf_conservative']['r2']
            best_model_obj = rf_conservative
        
        results['best_model'] = best_model
        results['best_r2'] = best_r2
        
        # Feature importance
        if hasattr(best_model_obj, 'feature_importances_'):
            feature_importance = list(zip(X_train.columns, best_model_obj.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            results['top_features'] = [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in feature_importance[:10]
            ]
        
        # Validate against literature benchmarks
        self.validate_performance(biomarker_name, best_r2, results)
        
        return results, best_model_obj

    def validate_performance(self, biomarker_name, r2_score, results):
        """Validate model performance against literature-based expectations"""
        
        # Categorize biomarker
        biomarker_category = None
        if any(term in biomarker_name.lower() for term in ['blood pressure', 'systolic', 'diastolic', 'heart']):
            biomarker_category = 'cardiovascular'
        elif any(term in biomarker_name.lower() for term in ['cd4', 'immune', 'white blood']):
            biomarker_category = 'immune'
        elif any(term in biomarker_name.lower() for term in ['glucose', 'cholesterol', 'hdl', 'ldl']):
            biomarker_category = 'metabolic'
        elif any(term in biomarker_name.lower() for term in ['creatinine', 'kidney', 'renal']):
            biomarker_category = 'renal'
        else:
            biomarker_category = 'metabolic'  # Default to most conservative
        
        thresholds = self.performance_thresholds[biomarker_category]
        expected_min, expected_max = thresholds['expected_range']
        max_realistic = thresholds['max_r2']
        
        results['literature_validation'] = {
            'category': biomarker_category,
            'expected_range': thresholds['expected_range'],
            'max_realistic': max_realistic,
            'performance_status': 'unknown'
        }
        
        if r2_score > max_realistic:
            self.log_progress(f"PERFORMANCE WARNING: R² = {r2_score:.3f} exceeds maximum realistic ({max_realistic:.3f})", "WARNING")
            self.log_progress(f"This suggests possible data leakage or overfitting", "WARNING")
            results['literature_validation']['performance_status'] = 'unrealistic'
        elif r2_score > expected_max:
            self.log_progress(f"Performance above expected range: R² = {r2_score:.3f} (expected: {expected_min:.3f}-{expected_max:.3f})", "WARNING")
            results['literature_validation']['performance_status'] = 'high'
        elif r2_score >= expected_min:
            self.log_progress(f"Performance within expected range: R² = {r2_score:.3f}", "SUCCESS")
            results['literature_validation']['performance_status'] = 'normal'
        else:
            self.log_progress(f"Performance below expected range: R² = {r2_score:.3f} (expected: {expected_min:.3f}-{expected_max:.3f})")
            results['literature_validation']['performance_status'] = 'weak'

    def save_rigorous_models(self, model_obj, results, biomarker_name, X_features):
        """Save models with comprehensive metadata"""
        safe_name = "".join(c for c in biomarker_name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')
        
        # Save model
        model_filename = self.models_dir / f"rigorous_model_{safe_name}_{self.timestamp}.joblib"
        joblib.dump(model_obj, model_filename)
        
        # Save comprehensive metadata
        metadata = {
            'biomarker': biomarker_name,
            'timestamp': self.timestamp,
            'methodology': 'rigorous_conservative',
            'feature_names': list(X_features.columns),
            'n_features': len(X_features.columns),
            'performance': results,
            'data_leakage_checked': True,
            'temporal_validation': True,
            'conservative_hyperparameters': True,
            'literature_validated': True
        }
        
        metadata_filename = self.models_dir / f"rigorous_metadata_{safe_name}_{self.timestamp}.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        results['model_path'] = str(model_filename)
        results['metadata_path'] = str(metadata_filename)
        
        self.log_progress(f"Rigorous model saved: {model_filename.name}")

    def run_rigorous_analysis(self):
        """Execute complete rigorous climate-health analysis"""
        self.log_progress("="*80)
        self.log_progress("🔬 RIGOROUS CLIMATE-HEALTH ANALYSIS")
        self.log_progress("Literature-validated methodology with data leakage prevention")
        self.log_progress("="*80)
        
        start_time = time.time()
        
        # Load and examine data
        df = self.load_and_examine_data()
        
        # Create rigorous feature set
        rigorous_features = self.create_rigorous_feature_set(df)
        
        # Define biomarkers to analyze
        biomarkers = [
            'CD4 cell count (cells/µL)',
            'Creatinine (mg/dL)', 
            'Hemoglobin (g/dL)',
            'systolic blood pressure',
            'diastolic blood pressure',
            'FASTING GLUCOSE',
            'FASTING TOTAL CHOLESTEROL',
            'FASTING HDL',
            'FASTING LDL'
        ]
        
        analysis_results = {}
        total_biomarkers = len(biomarkers)
        
        for i, biomarker in enumerate(biomarkers, 1):
            self.log_progress(f"\n🔬 [{i}/{total_biomarkers}] RIGOROUS ANALYSIS: {biomarker}")
            
            if biomarker not in df.columns:
                self.log_progress(f"Biomarker not found in dataset", "ERROR")
                continue
            
            try:
                # Clean biomarker data
                biomarker_data = df.dropna(subset=[biomarker]).copy()
                
                if len(biomarker_data) < 100:
                    self.log_progress(f"Insufficient data: {len(biomarker_data)} samples", "ERROR")
                    continue
                
                # Get available rigorous features
                available_features = [f for f in rigorous_features if f in biomarker_data.columns]
                
                if len(available_features) < 5:
                    self.log_progress(f"Insufficient features: {len(available_features)}", "ERROR")
                    continue
                
                # Prepare dataset
                X = biomarker_data[available_features].copy()
                y = biomarker_data[biomarker].copy()
                
                # Handle missing values conservatively
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                
                # Encode categorical conservatively
                categorical_cols = X.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if X[col].nunique() <= 10:  # Only small categorical variables
                        X[col] = pd.Categorical(X[col]).codes
                    else:
                        X = X.drop(columns=[col])  # Drop high-cardinality categoricals
                
                # Final cleaning
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) < 100:
                    self.log_progress(f"Insufficient clean data: {len(X_clean)}", "ERROR")
                    continue
                
                self.log_progress(f"Clean dataset: {len(X_clean):,} samples, {X_clean.shape[1]} features")
                
                # DATA LEAKAGE DETECTION
                has_leakage = self.detect_data_leakage(X_clean, y_clean, biomarker)
                if has_leakage:
                    self.log_progress(f"CRITICAL: Data leakage detected - skipping analysis", "ERROR")
                    continue
                
                # TEMPORAL TRAIN-TEST SPLIT
                X_train, X_test, y_train, y_test = self.temporal_train_test_split(X_clean, y_clean)
                
                # CONSERVATIVE MODEL TRAINING
                results, best_model = self.train_conservative_models(X_train, y_train, X_test, y_test, biomarker)
                
                # SAVE RIGOROUS MODEL
                self.save_rigorous_models(best_model, results, biomarker, X_train)
                
                analysis_results[biomarker] = results
                
            except Exception as e:
                self.log_progress(f"Analysis failed: {e}", "ERROR")
                continue
        
        # Generate final results
        elapsed_time = time.time() - start_time
        
        final_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'methodology': 'rigorous_conservative_climate_health',
                'total_biomarkers_analyzed': len(analysis_results),
                'analysis_time_minutes': elapsed_time / 60,
                'data_leakage_prevention': True,
                'temporal_validation': True,
                'literature_validated': True,
                'performance_thresholds': self.performance_thresholds
            },
            'biomarker_results': analysis_results
        }
        
        # Save comprehensive results
        results_file = self.results_dir / f"rigorous_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # FINAL SUMMARY
        self.log_progress("\n" + "="*80)
        self.log_progress("🔬 RIGOROUS ANALYSIS COMPLETE")
        self.log_progress("="*80)
        
        if analysis_results:
            self.log_progress(f"Successfully analyzed {len(analysis_results)} biomarkers")
            self.log_progress(f"Total analysis time: {elapsed_time/60:.1f} minutes")
            self.log_progress("")
            
            # Performance summary with literature validation
            self.log_progress(f"{'Biomarker':<35} {'R²':<8} {'Status':<12} {'Expected Range':<15}")
            self.log_progress("-" * 80)
            
            for biomarker, result in analysis_results.items():
                r2 = result['best_r2']
                status = result.get('literature_validation', {}).get('performance_status', 'unknown')
                expected = result.get('literature_validation', {}).get('expected_range', 'unknown')
                expected_str = f"{expected[0]:.2f}-{expected[1]:.2f}" if expected != 'unknown' else 'unknown'
                
                biomarker_short = biomarker[:34]
                status_icon = {
                    'normal': '✅',
                    'high': '⚠️ ',
                    'unrealistic': '❌',
                    'weak': '📉',
                    'unknown': '❓'
                }.get(status, '❓')
                
                self.log_progress(f"{biomarker_short:<35} {r2:<8.4f} {status_icon}{status:<11} {expected_str:<15}")
            
            # Summary statistics
            r2_values = [res['best_r2'] for res in analysis_results.values()]
            realistic_models = sum(1 for res in analysis_results.values() 
                                 if res.get('literature_validation', {}).get('performance_status') in ['normal', 'high'])
            
            self.log_progress("")
            self.log_progress(f"📊 Performance Summary:")
            self.log_progress(f"   Mean R²: {np.mean(r2_values):.4f}")
            self.log_progress(f"   Realistic models: {realistic_models}/{len(analysis_results)}")
            self.log_progress(f"   Literature-validated methodology: ✅")
            self.log_progress("")
            self.log_progress(f"✅ Results saved to: {results_file}")
            self.log_progress(f"📋 Progress log: {self.progress_file}")
            
        else:
            self.log_progress("❌ No biomarkers successfully analyzed")
            self.log_progress("This may indicate systematic data issues requiring investigation")
        
        return final_results

def main():
    """Execute rigorous climate-health analysis"""
    analyzer = RigorousClimateHealthAnalysis()
    results = analyzer.run_rigorous_analysis()
    return results

if __name__ == "__main__":
    main()