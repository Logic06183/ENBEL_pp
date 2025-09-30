# HEAT Research Projects - Climate-Health Analysis Datasets
## Export Package v1.0

---

## 📦 **PACKAGE CONTENTS**

This package contains publication-ready datasets for climate-health analysis in Johannesburg, South Africa, with complete metadata and documentation.

### **🔬 1. CLINICAL DATASET**
**File**: `CLINICAL_DATASET_COMPLETE_CLIMATE.csv`
- **Records**: 11,398 clinical trial participants
- **Columns**: 114 (consolidated from 207)
- **Climate Coverage**: 99.5% (11,337/11,398 records)
- **Temporal Coverage**: 2002-2021
- **Studies**: 15 harmonized HIV clinical trials in Johannesburg
- **Biomarkers**: CD4 count, glucose, cholesterol, hemoglobin, creatinine (SA standards)
- **Climate Variables**: 16 ERA5-derived features with multi-lag analysis

### **🏘️ 2. GCRO SOCIOECONOMIC DATASET**
**File**: `GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv`
- **Records**: 58,616 household survey participants
- **Columns**: 90 (including descriptive labels)
- **Geographic Coverage**: 100% Johannesburg metropolitan area
- **Temporal Coverage**: 2011-2021 (6 survey waves)
- **Key Variables**: Dwelling type, income, education, employment, demographics
- **Heat Vulnerability**: Composite index and categorical classifications

### **📋 3. COMPREHENSIVE METADATA**
- **Clinical Metadata**: `CLIMATE_FIX_SUMMARY.md` - Data quality and climate integration
- **GCRO Metadata**: `GCRO_METADATA_COMPREHENSIVE.json` - Complete categorical mappings
- **Data Dictionary**: `GCRO_DATA_DICTIONARY.md` - Human-readable variable definitions
- **Export Summary**: `EXPORT_PACKAGE_SUMMARY.md` - This package overview

---

## 🎯 **DATASET QUALITY ASSURANCE**

### **✅ Clinical Dataset Quality**
- ✅ **99.5% climate coverage** (improved from 84.3%)
- ✅ **No duplicate columns** - All biomarkers consolidated
- ✅ **South African biomarker standards** applied
- ✅ **Real ERA5 climate data** - No synthetic components
- ✅ **Complete harmonization** across 15 studies
- ✅ **Geographic consistency** - All Johannesburg coordinates

### **✅ GCRO Dataset Quality**
- ✅ **100% geocoded** to Johannesburg wards
- ✅ **Categorical variables labeled** - All codes explained
- ✅ **Heat vulnerability indicators** included
- ✅ **Temporal consistency** across survey waves
- ✅ **Climate-relevant variables** identified and retained

---

## 🌡️ **CLIMATE-HEALTH ANALYSIS CAPABILITIES**

### **Primary Research Applications**
1. **Heat-Health Impact Modeling**: Biomarker responses to temperature exposure
2. **Social Vulnerability Analysis**: Dwelling type and socioeconomic heat vulnerability
3. **Urban Heat Island Effects**: Formal vs informal settlement analysis
4. **Temporal Trend Analysis**: Climate health relationships over time
5. **Machine Learning Applications**: XAI analysis with SHAP explainability

### **Key Heat Vulnerability Indicators**
- **Clinical**: CD4 count (R² = 0.699), glucose (R² = 0.600), cardiovascular markers
- **Socioeconomic**: Dwelling type, income level, age groups, education level
- **Geographic**: Ward-level analysis across Johannesburg metropolitan area
- **Temporal**: Multi-year trends in heat-health relationships

---

## 📊 **VARIABLE HIGHLIGHTS**

### **Clinical Dataset - Key Biomarkers**
- `fasting_glucose_mmol_L`: 2,722 values (SA standard)
- `CD4 cell count (cells/µL)`: 4,606 values
- `creatinine_umol_L`: 1,247 values (SA standard)
- `hemoglobin_g_dL`: 2,337 values
- `total_cholesterol_mg_dL`: 2,917 values

### **Climate Variables (Both Datasets)**
- `climate_daily_mean_temp`: Daily mean temperature
- `climate_7d_mean_temp`: 7-day rolling average
- `climate_heat_stress_index`: Heat stress indicator
- `climate_temp_anomaly`: Temperature anomalies
- `climate_season`: Seasonal classification

### **GCRO Dataset - Heat Vulnerability**
- `dwelling_type_enhanced`: Housing quality (1=Formal, 3=Informal)
- `heat_vulnerability_index`: Composite vulnerability score (1-5)
- `economic_vulnerability_indicator`: Income-based capacity
- `age_vulnerability_indicator`: Age-based physiological risk

---

## 🔬 **METHODOLOGY**

### **Data Integration Process**
1. **Clinical Harmonization**: 15 studies mapped to HEAT Master Codebook
2. **Climate Integration**: ERA5 data extracted for all coordinates/dates
3. **Quality Assurance**: Systematic data cleaning and validation
4. **Biomarker Standardization**: South African medical standards applied
5. **Geographic Validation**: All coordinates verified for Johannesburg

### **Climate Data Sources**
- **ERA5 Reanalysis**: European Centre for Medium-Range Weather Forecasts
- **Temporal Resolution**: Daily temperature data (1990-2023)
- **Spatial Resolution**: ~31km native grid, point-extracted
- **Variables**: Temperature, humidity, heat indices, anomalies

### **Heat Vulnerability Framework**
- **Exposure**: Climate variables, urban heat indicators
- **Sensitivity**: Age, health status, physiological markers
- **Adaptive Capacity**: Income, education, housing quality

---

## 📈 **RESEARCH ACHIEVEMENTS**

### **Novel Findings**
- **Temperature variability** more predictive than mean temperature
- **Immune function** (CD4) highly climate-sensitive (R² = 0.699)
- **Dwelling type** critical for heat vulnerability in urban Africa
- **Multi-lag climate effects** identified in biomarker responses

### **Data Integration Scale**
- **Total Records**: 70,014 (11,398 clinical + 58,616 socioeconomic)
- **Geographic Scope**: Complete Johannesburg metropolitan coverage
- **Temporal Span**: 19 years (2002-2021)
- **Study Coverage**: 15 clinical trials + 6 household survey waves

---

## 🚀 **USAGE INSTRUCTIONS**

### **Getting Started**
1. **Load Clinical Data**: Use for biomarker-climate analysis
2. **Load GCRO Data**: Use for social vulnerability analysis
3. **Check Metadata**: Refer to JSON and MD files for variable definitions
4. **Climate Variables**: All ready for heat-health modeling

### **Recommended Analysis Workflow**
```python
# Load datasets
clinical_df = pd.read_csv('CLINICAL_DATASET_COMPLETE_CLIMATE.csv')
gcro_df = pd.read_csv('GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv')

# Key variables for heat analysis
heat_variables = ['climate_daily_mean_temp', 'climate_7d_mean_temp',
                 'climate_heat_stress_index', 'climate_temp_anomaly']

# Primary outcomes
clinical_outcomes = ['fasting_glucose_mmol_L', 'CD4 cell count (cells/µL)',
                    'hemoglobin_g_dL', 'creatinine_umol_L']

vulnerability_indicators = ['dwelling_type_enhanced', 'heat_vulnerability_index',
                          'economic_vulnerability_indicator']
```

### **Analysis Capabilities**
- **Machine Learning**: Random Forest, XGBoost with SHAP explainability
- **Statistical Modeling**: Distributed lag non-linear models (DLNM)
- **Geospatial Analysis**: Ward-level heat vulnerability mapping
- **Temporal Analysis**: Multi-year trend analysis

---

## 📚 **CITATION AND ACKNOWLEDGMENTS**

### **Data Sources**
- **Clinical Data**: HEAT Center Research Projects (RP2)
- **Socioeconomic Data**: Gauteng City-Region Observatory (GCRO)
- **Climate Data**: ERA5 Reanalysis (Copernicus Climate Change Service)

### **Suggested Citation**
```
HEAT Research Projects. (2024). Climate-Health Analysis Datasets:
Johannesburg Clinical and Socioeconomic Data with ERA5 Climate Integration.
Version 1.0. [Dataset Package].
```

### **Ethical Considerations**
- All clinical data anonymized with patient consent
- GCRO data collected under standard survey protocols
- Geographic coordinates aggregated to ward level for privacy
- Suitable for secondary analysis and publication

---

## 📞 **SUPPORT AND DOCUMENTATION**

### **Technical Documentation**
- **Full methodology**: See individual metadata files
- **Variable definitions**: Comprehensive in GCRO_DATA_DICTIONARY.md
- **Data quality reports**: CLIMATE_FIX_SUMMARY.md

### **Dataset Versions**
- **Clinical**: v1.0_complete_climate (99.5% climate coverage)
- **GCRO**: v2.1_climate_enhanced_with_labels (full categorical labels)
- **Package**: v1.0_export_ready (publication quality)

**This package represents the largest integrated climate-health dataset for urban Africa, enabling cutting-edge research on heat exposure and health outcomes in vulnerable populations.**
