# Heat-Health Analysis Pipeline - Quick Start Guide

## 🚀 Quick Setup (M4 Mac Optimized)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
```bash
python optimized_interpretable_ml_pipeline.py
```

## 📊 What It Does

Analyzes relationships between climate variables (temperature, humidity, pressure) and health biomarkers using:
- **Optimized Random Forest**: 250 trees, max_depth=15
- **Optimized XGBoost**: learning_rate=0.05, max_depth=8
- **266 climate features** with multi-day lags (0-21 days)
- **Real clinical data** from Johannesburg trials

## 📈 Expected Performance

- R² improvements: +0.02 to +0.10 over baseline
- Analysis time: ~5-10 minutes on M4 Mac
- Outputs saved to `optimized_results/` folder

## 🎯 Biomarkers Analyzed

1. CD4 cell count (immune system)
2. Creatinine (kidney function)
3. Hemoglobin (oxygen transport)
4. Blood pressure (cardiovascular)
5. Glucose levels (metabolic)
6. Cholesterol markers (HDL, LDL, Total)

## 📝 Output Files

- `optimized_results/optimized_analysis_[timestamp].json` - Full results with feature importance
- `optimized_results/progress_[timestamp].log` - Detailed progress tracking

## 💡 M4 Mac Optimization Tips

The script automatically uses:
- All CPU cores (`n_jobs=-1`)
- Optimized tree methods for Apple Silicon
- Efficient memory management

## 🔍 Interpreting Results

Look for:
- **R² > 0.05**: Clinically meaningful relationship
- **Top features**: Which climate variables matter most
- **Lag patterns**: Delayed effects (e.g., heat exposure 3-7 days before)

## ⚡ Performance Notes

- Dataset: 66MB, 18,205 rows × 343 columns
- Typical runtime: 5-10 minutes
- Memory usage: ~2-4GB peak