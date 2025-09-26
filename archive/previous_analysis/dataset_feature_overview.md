# ENBEL Climate-Health Dataset: Comprehensive Feature Overview

## Dataset Summary
- **Total Records**: 18,205 participants
- **Total Features**: 343 columns
- **Geographic Scope**: Johannesburg, South Africa
- **Time Period**: Clinical trials and observational studies
- **Data Sources**: Clinical trials, weather stations, satellite data (ERA5, SAAQIS)

## Target Variables (Biomarkers)

### Primary Health Outcomes
| Biomarker | Description | Non-null Count | Unit |
|-----------|-------------|----------------|------|
| **CD4 cell count** | Immune system function | 1,283 | cells/µL |
| **FASTING GLUCOSE** | Blood sugar/metabolic health | 2,731 | mg/dL |
| **FASTING LDL** | "Bad" cholesterol | 2,500 | mg/dL |
| **FASTING TOTAL CHOLESTEROL** | Total cholesterol | 2,497 | mg/dL |
| **FASTING HDL** | "Good" cholesterol | 2,497 | mg/dL |
| **FASTING TRIGLYCERIDES** | Blood lipids | 972 | mg/dL |
| **Creatinine** | Kidney function | 1,251 | mg/dL |
| **ALT** | Liver enzyme | 1,254 | U/L |
| **AST** | Liver enzyme | 1,254 | U/L |
| **Hemoglobin** | Oxygen transport | 1,282 | g/dL |
| **Hematocrit** | Blood cell volume | 1,066 | % |
| **Systolic BP** | Blood pressure (high) | 4,957 | mmHg |
| **Diastolic BP** | Blood pressure (low) | 4,957 | mmHg |

## Predictor Features

### 1. Base Climate Variables (Same-day, lag 0)
| Feature Type | Count | Description | Example |
|--------------|-------|-------------|---------|
| **Temperature** | 4 | Basic temperature measures | temperature, temperature_max, temperature_min |
| **Humidity** | 2 | Moisture content | humidity, humidity_max, humidity_min |
| **Wind** | 2 | Wind speed and gusts | wind_speed, wind_gust |
| **Heat Index** | 4 | Apparent/feels-like temperature | heat_index, apparent_temp, wet_bulb_temp |

### 2. Climate Lag Structure
**Lag Days**: 0, 1, 2, 3, 5, 7, 10, 14, 21 (up to 3 weeks prior exposure)

| Lag Period | Features per Source | Total Features | Interpretation |
|------------|-------------------|----------------|----------------|
| **lag0** (same day) | 21 | 21 | Immediate exposure effects |
| **lag1** (1 day prior) | 21 | 63 | Short-term delayed effects |
| **lag2** (2 days prior) | 21 | 42 | Early physiological response |
| **lag3** (3 days prior) | 21 | 21 | Metabolic adaptation period |
| **lag5** (5 days prior) | 21 | 21 | Intermediate biological effects |
| **lag7** (1 week prior) | 21 | 21 | Weekly pattern effects |
| **lag10** (10 days prior) | 9 | - | Extended exposure period |
| **lag14** (2 weeks prior) | 9 | - | Chronic adaptation period |
| **lag21** (3 weeks prior) | 9 | - | Long-term cumulative effects |

### 3. Advanced Climate Indices (with lags)
| Index Type | Count | Description | Health Relevance |
|------------|-------|-------------|------------------|
| **Heat Index** | 63 | Apparent temperature combining heat + humidity | Heat stress, dehydration |
| **UTCI** | 63 | Universal Thermal Climate Index | Human thermal comfort |
| **WBGT** | 63 | Wet Bulb Globe Temperature | Heat stress risk |
| **Heat Stress Categories** | 49 | Binary indicators (moderate, strong, extreme) | Clinical heat thresholds |

### 4. Multi-day Climate Aggregations
| Time Window | Features | Variables | Purpose |
|-------------|----------|-----------|---------|
| **3-day aggregates** | 12 | mean, max, min, range, variability | Short-term patterns |
| **5-day aggregates** | 12 | mean, max, min, range, variability | Work-week patterns |
| **7-day aggregates** | 12 | mean, max, min, range, variability | Weekly exposure cycles |
| **14-day aggregates** | 12 | mean, max, min, range, variability | Bi-weekly trends |

**Derived Indicators**:
- **Cooling/Heating Degree Days**: Energy demand proxies
- **Heat Exposure Days**: Days above health thresholds
- **Temperature Variability**: Climate stability measures
- **Temperature Change/Acceleration**: Rate of climate change

### 5. SAAQIS (South African Air Quality) Features
| Variable | Lags | Total | Description |
|----------|------|-------|-------------|
| **ERA5 Temperature** | 0,1,2,3,5,7,10,14,21 | 9 | Reanalysis temperature |
| **Land Temperature** | 0,1,2,3,5,7,10,14,21 | 9 | Surface temperature |
| **Wind Speed** | 0,1,2,3,5,7,10,14,21 | 9 | Wind patterns |
| **Land Surface Temp** | 0,1,2,3,5,7,10,14,21 | 9 | Satellite-derived temperature |
| **Day of Year** | 0,1,2,3,5,7,10,14,21 | 9 | Seasonal patterns |
| **Digital Elevation** | 0,1,2,3,5,7,10,14,21 | 9 | Topographic effects |
| **Time of Day** | 0,1,2,3,5,7,10,14,21 | 9 | Diurnal patterns |
| **Week** | 0,1,2,3,5,7,10,14,21 | 9 | Weekly cycles |
| **Station ID** | 0,1,2,3,5,7,10,14,21 | 9 | Spatial location |

### 6. Demographic & Socioeconomic Features
| Category | Features | Description |
|----------|----------|-------------|
| **Basic Demographics** | Sex, Race, Age groups | Individual characteristics |
| **Geographic** | latitude, longitude | Spatial location |
| **Temporal** | year, month, season | Time-based patterns |
| **Socioeconomic** | Education, employment_status | Sociodemographic factors |
| **Vulnerability Indices** | housing_vulnerability, economic_vulnerability, heat_vulnerability_index | Composite risk scores |
| **Physical** | Height, weight | Anthropometric measures |

### 7. Study Design Features
| Category | Features | Purpose |
|----------|----------|---------|
| **Study Identifiers** | study_code, study_type, data_source | Study provenance |
| **HIV-specific** | from_hiv_study, hiv_status_indicator, viral_load | HIV research context |
| **Data Quality** | has_biomarkers, has_socioeconomic, spatial_precision | Missing data patterns |

## Feature Engineering Details

### Lag Construction
- **Point-in-time lags**: Single day values at 0, 1, 2, 3, 5, 7, 10, 14, 21 days prior
- **Aggregation windows**: Multi-day means, maxima, minima for 3, 5, 7, 14-day periods
- **Rate of change**: Temperature change and acceleration over 1, 3, 7, 14-day periods

### Interaction Terms
The dataset includes derived interaction terms:
- **Heat + Humidity**: Combined in heat index calculations
- **Temperature + Time**: Seasonal and diurnal interactions
- **Geographic + Climate**: Spatial climate variations
- **Vulnerability + Exposure**: Risk-weighted climate exposure

### Heat Stress Thresholds
Binary indicators for clinically relevant heat stress levels:
- **Moderate Heat Stress**: 26-32°C UTCI
- **Strong Heat Stress**: 32-38°C UTCI  
- **Very Strong Heat Stress**: 38-46°C UTCI
- **Above Health Threshold**: >28°C wet bulb temperature
- **Extreme Heat Event**: >95th percentile temperature

## Data Sources Integration
1. **Clinical Data**: Hospital/clinic biomarker measurements
2. **ERA5 Reanalysis**: European Centre for Medium-Range Weather Forecasts
3. **SAAQIS**: South African Air Quality Information System
4. **Local Weather Stations**: Ground-based temperature, humidity, wind
5. **Satellite Data**: Land surface temperature, elevation
6. **Census Data**: Demographics, socioeconomic indicators

## Model Input Specifications

### Feature Selection for ML Models
- **Total potential predictors**: ~266 climate features + ~20 demographic features
- **Typical model input**: 200-300 features after preprocessing
- **Feature importance ranking**: Based on Random Forest and XGBoost feature importance
- **Lag pattern focus**: 0-7 day lags show strongest associations
- **Multicollinearity handling**: Correlation filtering applied (r > 0.95 threshold)

### Target Variable Handling
- **Continuous outcomes**: All biomarkers treated as regression targets
- **Missing data**: Complete case analysis per biomarker (varies by outcome)
- **Outlier treatment**: Conservative IQR-based outlier detection
- **Scaling**: StandardScaler applied to features, targets kept in original units for interpretability

This comprehensive feature set enables analysis of immediate, short-term, and long-term climate health relationships across multiple biological systems and vulnerable populations.