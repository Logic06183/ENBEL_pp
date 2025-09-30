# Simple Climate-Health Analysis Pipeline

This repository contains a **clean, simple, and transparent** machine learning pipeline for analyzing how climate affects health biomarkers.

## 🎯 Purpose

We want to understand: **Do climate conditions (temperature, humidity) predict health outcomes (blood pressure, glucose, etc.)?**

## 📁 Key Files (Simple Version)

### Main Analysis
- **`simple_ml_pipeline.py`** - The main analysis script (easy to understand!)
- **`setup_and_validate.py`** - Checks if everything is working
- **`requirements.txt`** - Lists all needed software packages

### Data Files (Must Have These)
- `FULL_DATASET_WITH_REAL_CLIMATE_LAGS.csv` - Main dataset (18,205 participants)
- `CLINICAL_WITH_FIXED_IMPUTED_SOCIOECONOMIC.csv` - Clinical data
- `CLINICAL_WITH_IMPUTED_SOCIOECONOMIC.csv` - Original clinical data

### Utility Files
- `utils/config.py` - Configuration settings
- `utils/data_validation.py` - Data checking functions
- `utils/ml_utils.py` - Machine learning helper functions

## 🚀 How to Run

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Validate Setup
```bash
python setup_and_validate.py
```

### Step 3: Run Simple Analysis
```bash
python simple_ml_pipeline.py
```

## 📊 What the Analysis Does

### Simple 5-Step Process:

1. **Load Data** - Reads the climate-health dataset
2. **Select Features** - Picks interpretable climate variables (temperature, humidity, etc.)
3. **Prepare Data** - Cleans data for each health biomarker
4. **Train Models** - Uses Random Forest and XGBoost to predict health outcomes
5. **Analyze Results** - Shows which climate factors are most important

### Health Biomarkers We Analyze:
- Systolic blood pressure
- Fasting glucose
- CD4 cell count
- LDL cholesterol  
- Hemoglobin

### Climate Features We Use:
- Temperature (various lag periods)
- Humidity
- Heat index
- Wind patterns
- Atmospheric pressure

## 📈 Understanding Results

The pipeline produces **R² scores** that tell us how well climate predicts health:
- **R² = 0.00**: Climate doesn't predict health at all
- **R² = 0.05**: Climate explains 5% of health variation (weak but meaningful)
- **R² = 0.10**: Climate explains 10% of health variation (moderate)
- **R² = 0.20**: Climate explains 20% of health variation (strong)

## 🔍 Quality Assurance

### Statistical Rigor:
- ✅ **Cross-validation** prevents overfitting
- ✅ **Train/test splits** for unbiased performance
- ✅ **Reproducible seeds** for consistent results
- ✅ **Multiple testing correction** for biomarker comparisons

### Code Quality:
- ✅ **Clear documentation** for every function
- ✅ **Simple logic** that's easy to follow
- ✅ **Error handling** for robust execution
- ✅ **Comprehensive testing** with `tests/test_ml_pipeline.py`

## 📋 Example Output

```
🌟 Simple Climate-Health ML Pipeline
====================================

📁 Step 1: Loading Climate-Health Dataset
✓ Data loaded: 18,205 participants
✓ Total features: 343

🌡️ Step 2: Selecting Climate Features  
✓ Selected 64 climate features
✓ Added 3 demographic controls
✓ Total features for analysis: 67

🧬 Analyzing Health Biomarkers
[1/5] ANALYZING: systolic blood pressure
✓ Clean dataset: 4,956 participants
✓ Training Random Forest...
✓ Test R²: 0.0234
✓ Cross-validation R²: 0.0198 ± 0.0056

📋 FINAL RESULTS SUMMARY
Biomarker                          Samples    Best R²    Model          
systolic blood pressure           4,956      0.0234     random_forest  
FASTING GLUCOSE                   3,891      0.0156     xgboost        
✅ Analysis completed successfully!
```

## 🔧 Troubleshooting

### Common Issues:

1. **Missing Data Files**
   - Make sure the CSV files are in the main directory
   - Check file names match exactly

2. **Package Installation Errors**
   - Try: `pip install --upgrade pip`
   - Then: `pip install -r requirements.txt`

3. **Memory Issues**
   - The dataset is large (62MB)
   - Close other programs if needed
   - Reduce `n_estimators` in models if necessary

### Getting Help:
- Run `python setup_and_validate.py` to check your setup
- Check the generated log files for detailed error messages
- All functions have clear documentation

## 📝 Validation Checklist

Before sharing results, verify:
- [ ] `setup_and_validate.py` passes all checks
- [ ] Results are reproducible (same R² scores each run)
- [ ] Performance seems reasonable (R² between -0.1 and 0.3)
- [ ] No error messages in the output
- [ ] Results saved to JSON file successfully

## 🎯 For Your Team

This pipeline is designed to be:
- **Transparent** - Every step is clearly documented
- **Simple** - Straightforward logic, no complex algorithms
- **Verifiable** - Easy to check and validate results
- **Reproducible** - Same results every time with fixed random seeds

Your team can confidently review this code and understand exactly what it's doing!