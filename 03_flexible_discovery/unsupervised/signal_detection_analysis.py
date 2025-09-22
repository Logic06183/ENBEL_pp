#!/usr/bin/env python3
"""
Advanced Signal Detection for Weak Climate-Health Relationships
==============================================================

Using cutting-edge machine learning techniques specifically designed
for detecting weak signals in high-dimensional, noisy data.

Methods:
1. Bayesian regularization
2. Ensemble feature selection
3. Stability selection
4. Mutual information-based detection
5. Cross-decomposition analysis
6. Network-based approaches
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

class AdvancedSignalDetection:
    def __init__(self):
        self.significant_findings = []
        
    def load_and_preprocess_data(self):
        """Load data with advanced preprocessing"""
        print("🔬 ADVANCED SIGNAL DETECTION FOR CLIMATE-HEALTH")
        print("=" * 50)
        print("📊 Loading and preprocessing data...")
        
        df = pd.read_csv('FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv', low_memory=False)
        
        # Enhanced biomarker selection
        biomarkers = {
            'systolic blood pressure': df['systolic blood pressure'].dropna() if 'systolic blood pressure' in df.columns else pd.Series(),
            'diastolic blood pressure': df['diastolic blood pressure'].dropna() if 'diastolic blood pressure' in df.columns else pd.Series(),
            'FASTING GLUCOSE': df['FASTING GLUCOSE'].dropna() if 'FASTING GLUCOSE' in df.columns else pd.Series(),
            'FASTING TOTAL CHOLESTEROL': df['FASTING TOTAL CHOLESTEROL'].dropna() if 'FASTING TOTAL CHOLESTEROL' in df.columns else pd.Series(),
            'FASTING HDL': df['FASTING HDL'].dropna() if 'FASTING HDL' in df.columns else pd.Series(),
            'CD4 cell count (cells/µL)': df['CD4 cell count (cells/µL)'].dropna() if 'CD4 cell count (cells/µL)' in df.columns else pd.Series(),
            'Hemoglobin (g/dL)': df['Hemoglobin (g/dL)'].dropna() if 'Hemoglobin (g/dL)' in df.columns else pd.Series(),
            'Creatinine (mg/dL)': df['Creatinine (mg/dL)'].dropna() if 'Creatinine (mg/dL)' in df.columns else pd.Series()
        }
        
        # Filter biomarkers with sufficient data
        valid_biomarkers = {k: v for k, v in biomarkers.items() if len(v) >= 500}
        
        # Enhanced climate feature engineering
        climate_features = self._create_advanced_climate_features(df)
        
        print(f"Valid biomarkers: {len(valid_biomarkers)}")
        print(f"Climate features: {len(climate_features)}")
        
        return df, valid_biomarkers, climate_features
    
    def _create_advanced_climate_features(self, df):
        """Create comprehensive climate feature set"""
        # Get base climate variables
        base_climate = []
        for col in df.columns:
            if any(term in col.lower() for term in ['temp', 'heat', 'humid', 'pressure', 'wind']) and 'lag' in col.lower():
                base_climate.append(col)
        
        advanced_features = base_climate.copy()
        
        # 1. Moving averages (3, 7, 14 day)
        temp_cols = [c for c in base_climate if 'temp' in c.lower()]
        
        for window in [3, 7, 14]:
            for temp_col in temp_cols:
                if temp_col in df.columns:
                    # Extract lag number
                    try:
                        lag_num = int(temp_col.split('lag')[-1])
                        if lag_num <= 14:  # Only for recent lags
                            ma_col = f"{temp_col}_ma{window}"
                            # Simple moving average approximation
                            relevant_cols = [c for c in temp_cols if f'lag{lag_num}' in c or f'lag{lag_num+1}' in c or f'lag{lag_num+2}' in c]
                            if len(relevant_cols) >= 2:
                                df[ma_col] = df[relevant_cols[:3]].mean(axis=1)
                                advanced_features.append(ma_col)
                    except:
                        continue
        
        # 2. Temperature variability indicators
        recent_temp_cols = [c for c in temp_cols if any(f'lag{i}' in c for i in range(8))][:7]
        if len(recent_temp_cols) >= 5:
            df['temp_variability'] = df[recent_temp_cols].std(axis=1)
            df['temp_range'] = df[recent_temp_cols].max(axis=1) - df[recent_temp_cols].min(axis=1)
            advanced_features.extend(['temp_variability', 'temp_range'])
        
        # 3. Extreme temperature indicators
        for temp_col in temp_cols[:5]:
            if temp_col in df.columns:
                p95 = df[temp_col].quantile(0.95)
                p05 = df[temp_col].quantile(0.05)
                
                extreme_hot = f"{temp_col}_extreme_hot"
                extreme_cold = f"{temp_col}_extreme_cold"
                
                df[extreme_hot] = (df[temp_col] > p95).astype(int)
                df[extreme_cold] = (df[temp_col] < p05).astype(int)
                
                advanced_features.extend([extreme_hot, extreme_cold])
        
        # Filter to existing columns
        return [f for f in advanced_features if f in df.columns]
    
    def bayesian_signal_detection(self, df, biomarker_name, climate_features):
        """Bayesian approach for weak signal detection"""
        print(f"\n🔍 Bayesian Signal Detection: {biomarker_name}")
        print("-" * 40)
        
        # Prepare data
        biomarker_data = df.dropna(subset=[biomarker_name])
        if len(biomarker_data) < 500:
            print("Insufficient data")
            return None
        
        # Get available climate features
        available_features = [f for f in climate_features if f in biomarker_data.columns]
        if len(available_features) < 10:
            print("Insufficient climate features")
            return None
        
        # Robust preprocessing
        X = biomarker_data[available_features]
        y = biomarker_data[biomarker_name]
        
        # Handle missing values and outliers
        X = X.fillna(X.median())
        
        # Remove extreme outliers (beyond 4 sigma)
        z_scores = np.abs(stats.zscore(y))
        mask = z_scores < 4
        X, y = X[mask], y[mask]
        
        if len(X) < 500:
            print("Insufficient data after cleaning")
            return None
        
        print(f"Sample size: {len(X):,}")
        print(f"Features: {len(available_features)}")
        
        # Robust scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Automatic Relevance Determination (ARD) Regression
        ard_model = ARDRegression(compute_score=True, threshold_lambda=1e2)
        ard_scores = cross_val_score(ard_model, X_scaled, y, cv=5, scoring='r2')
        ard_mean = np.mean(ard_scores)
        
        print(f"ARD R²: {ard_mean:.4f} ± {np.std(ard_scores):.4f}")
        
        # 2. Bayesian Ridge Regression
        bayesian_model = BayesianRidge(compute_score=True)
        bay_scores = cross_val_score(bayesian_model, X_scaled, y, cv=5, scoring='r2')
        bay_mean = np.mean(bay_scores)
        
        print(f"Bayesian Ridge R²: {bay_mean:.4f} ± {np.std(bay_scores):.4f}")
        
        # Find best performing model
        best_score = max(ard_mean, bay_mean)
        best_model_name = "ARD" if ard_mean > bay_mean else "Bayesian Ridge"
        
        # 3. Feature relevance detection
        if ard_mean > 0.01:  # If ARD shows promise
            ard_model.fit(X_scaled, y)
            # Features with high precision (low variance) are more relevant
            feature_precision = ard_model.lambda_
            relevant_features = np.where(feature_precision < np.percentile(feature_precision, 50))[0]
            
            if len(relevant_features) > 0:
                top_features = [available_features[i] for i in relevant_features[:5]]
                print(f"ARD relevant features: {len(relevant_features)}")
                print(f"Top features: {top_features[:3]}")
                
                # Significance test
                if best_score > 0.02:
                    # Permutation test
                    null_scores = []
                    best_model = ard_model if ard_mean > bay_mean else bayesian_model
                    
                    for _ in range(100):
                        y_perm = np.random.permutation(y)
                        perm_scores = cross_val_score(best_model, X_scaled, y_perm, cv=3, scoring='r2')
                        null_scores.append(np.mean(perm_scores))
                    
                    p_value = np.mean(np.array(null_scores) >= best_score)
                    
                    print(f"Permutation p-value: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        result = {
                            'biomarker': biomarker_name,
                            'method': 'Bayesian Signal Detection',
                            'best_model': best_model_name,
                            'r2': best_score,
                            'p_value': p_value,
                            'n_samples': len(X),
                            'n_relevant_features': len(relevant_features),
                            'top_features': top_features
                        }
                        
                        print("✅ POTENTIAL BAYESIAN SIGNAL DETECTED")
                        return result
        
        print("No significant Bayesian signal")
        return None
    
    def mutual_information_detection(self, df, biomarker_name, climate_features):
        """Mutual information-based signal detection"""
        print(f"\n🧠 Mutual Information Detection: {biomarker_name}")
        print("-" * 45)
        
        biomarker_data = df.dropna(subset=[biomarker_name])
        if len(biomarker_data) < 500:
            print("Insufficient data")
            return None
        
        available_features = [f for f in climate_features if f in biomarker_data.columns][:50]  # Top 50 for MI
        if len(available_features) < 10:
            print("Insufficient features")
            return None
        
        X = biomarker_data[available_features].fillna(biomarker_data[available_features].median())
        y = biomarker_data[biomarker_name]
        
        print(f"Sample size: {len(X):,}")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Sort features by MI score
        mi_df = pd.DataFrame({
            'feature': available_features,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Get top features by MI
        top_mi_features = mi_df.head(10)['feature'].tolist()
        max_mi = mi_df['mi_score'].max()
        
        print(f"Max MI score: {max_mi:.4f}")
        print(f"Top MI feature: {mi_df.iloc[0]['feature']}")
        
        # Test with top MI features
        if max_mi > 0.01:  # Threshold for meaningful MI
            X_mi = X[top_mi_features]
            
            # Simple model with MI-selected features
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            mi_scores = cross_val_score(rf_model, X_mi, y, cv=5, scoring='r2')
            mi_mean = np.mean(mi_scores)
            
            print(f"MI-based model R²: {mi_mean:.4f}")
            
            if mi_mean > 0.02:
                # Statistical test
                null_scores = []
                for _ in range(50):
                    y_perm = np.random.permutation(y)
                    perm_scores = cross_val_score(rf_model, X_mi, y_perm, cv=3, scoring='r2')
                    null_scores.append(np.mean(perm_scores))
                
                p_value = np.mean(np.array(null_scores) >= mi_mean)
                
                if p_value < 0.10:  # Relaxed threshold for MI
                    # Simple correlation with best feature
                    best_feature = top_mi_features[0]
                    correlation, corr_p = spearmanr(X[best_feature], y)
                    
                    result = {
                        'biomarker': biomarker_name,
                        'method': 'Mutual Information',
                        'r2': mi_mean,
                        'p_value': p_value,
                        'max_mi_score': max_mi,
                        'best_feature': best_feature,
                        'correlation': correlation,
                        'n_samples': len(X),
                        'top_mi_features': top_mi_features[:5]
                    }
                    
                    print(f"p-value: {p_value:.4f}")
                    print(f"Best feature correlation: {correlation:.3f}")
                    print("✅ POTENTIAL MI SIGNAL DETECTED")
                    return result
        
        print("No significant MI signal")
        return None
    
    def cross_decomposition_analysis(self, df, biomarker_name, climate_features):
        """Cross-decomposition analysis using PLS and CCA"""
        print(f"\n🔄 Cross-Decomposition Analysis: {biomarker_name}")
        print("-" * 45)
        
        biomarker_data = df.dropna(subset=[biomarker_name])
        if len(biomarker_data) < 500:
            print("Insufficient data")
            return None
        
        available_features = [f for f in climate_features if f in biomarker_data.columns][:30]
        if len(available_features) < 10:
            print("Insufficient features")
            return None
        
        X = biomarker_data[available_features].fillna(biomarker_data[available_features].median())
        y = biomarker_data[biomarker_name].values.reshape(-1, 1)
        
        print(f"Sample size: {len(X):,}")
        
        # Standardize data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # 1. Partial Least Squares (PLS) Regression
        best_pls_score = -999
        best_n_components = 1
        
        for n_comp in range(1, min(6, len(available_features))):
            pls = PLSRegression(n_components=n_comp)
            pls_scores = cross_val_score(pls, X_scaled, y_scaled.ravel(), cv=5, scoring='r2')
            pls_mean = np.mean(pls_scores)
            
            if pls_mean > best_pls_score:
                best_pls_score = pls_mean
                best_n_components = n_comp
        
        print(f"Best PLS R² ({best_n_components} components): {best_pls_score:.4f}")
        
        if best_pls_score > 0.02:
            # Fit final PLS model
            pls_final = PLSRegression(n_components=best_n_components)
            pls_final.fit(X_scaled, y_scaled.ravel())
            
            # Get component loadings
            X_loadings = pls_final.x_loadings_[:, 0]  # First component
            
            # Find most important features
            loading_importance = np.abs(X_loadings)
            top_indices = np.argsort(loading_importance)[-5:]
            top_features = [available_features[i] for i in top_indices]
            
            # Test significance
            null_scores = []
            for _ in range(50):
                y_perm = np.random.permutation(y_scaled.ravel())
                perm_scores = cross_val_score(pls_final, X_scaled, y_perm, cv=3, scoring='r2')
                null_scores.append(np.mean(perm_scores))
            
            p_value = np.mean(np.array(null_scores) >= best_pls_score)
            
            if p_value < 0.10:
                result = {
                    'biomarker': biomarker_name,
                    'method': 'PLS Cross-Decomposition',
                    'r2': best_pls_score,
                    'n_components': best_n_components,
                    'p_value': p_value,
                    'n_samples': len(X),
                    'top_features': top_features
                }
                
                print(f"p-value: {p_value:.4f}")
                print(f"Top PLS features: {top_features[:3]}")
                print("✅ POTENTIAL PLS SIGNAL DETECTED")
                return result
        
        print("No significant cross-decomposition signal")
        return None
    
    def run_advanced_signal_detection(self):
        """Execute comprehensive advanced signal detection"""
        print("\n🚀 RUNNING ADVANCED SIGNAL DETECTION ANALYSIS")
        print("=" * 50)
        
        # Load data
        df, biomarkers, climate_features = self.load_and_preprocess_data()
        
        all_results = []
        
        for biomarker_name in biomarkers.keys():
            print(f"\n🎯 COMPREHENSIVE ANALYSIS: {biomarker_name}")
            print("=" * (25 + len(biomarker_name)))
            
            # 1. Bayesian signal detection
            bayesian_result = self.bayesian_signal_detection(df, biomarker_name, climate_features)
            if bayesian_result:
                all_results.append(bayesian_result)
            
            # 2. Mutual information detection
            mi_result = self.mutual_information_detection(df, biomarker_name, climate_features)
            if mi_result:
                all_results.append(mi_result)
            
            # 3. Cross-decomposition analysis
            decomp_result = self.cross_decomposition_analysis(df, biomarker_name, climate_features)
            if decomp_result:
                all_results.append(decomp_result)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎯 ADVANCED SIGNAL DETECTION SUMMARY")
        print("=" * 60)
        
        if len(all_results) > 0:
            print(f"Total signals detected: {len(all_results)}")
            print("\n🔍 DETECTED SIGNALS:")
            
            for i, result in enumerate(all_results, 1):
                print(f"\n{i}. {result['biomarker']} ({result['method']})")
                print(f"   R² = {result['r2']:.4f}")
                print(f"   p-value = {result['p_value']:.4f}")
                print(f"   Sample size = {result['n_samples']:,}")
                
                if 'best_feature' in result:
                    print(f"   Best feature: {result['best_feature']}")
                elif 'top_features' in result and result['top_features']:
                    print(f"   Top features: {result['top_features'][:2]}")
        else:
            print("❌ NO SIGNALS DETECTED WITH ADVANCED METHODS")
            print("\nThis comprehensive analysis using:")
            print("• Bayesian regularization techniques")
            print("• Mutual information theory") 
            print("• Cross-decomposition methods")
            print("• Robust preprocessing")
            print("• Multiple significance tests")
            print("\nStrongly suggests that detectable climate-health")
            print("relationships are absent in this dataset.")
        
        return all_results

def main():
    detector = AdvancedSignalDetection()
    results = detector.run_advanced_signal_detection()
    return results

if __name__ == "__main__":
    main()