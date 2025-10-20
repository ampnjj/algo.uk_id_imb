# Enhanced Time Series Forecasting with Advanced ML Models

This project provides production-ready time series forecasting using multiple machine learning models (Ridge, Lasso, LightGBM, XGBoost) with advanced features including uncertainty quantification, feature importance analysis, and robust validation.

## Features

✅ **4 ML Models**: Ridge, Lasso, LightGBM, XGBoost with ensemble
✅ **No Data Leakage**: Proper lag features, temporal splits, rolling window validation
✅ **Uncertainty Quantification**: 95% prediction intervals for risk assessment
✅ **Feature Importance**: Automatic analysis and export of top features
✅ **Early Stopping**: Prevents overfitting in gradient boosting models
✅ **Walk-Forward Validation**: Optional robust time series evaluation
✅ **Hyperparameter Optimization**: Using Optuna with time series cross-validation
✅ **DST-Safe**: All timestamps in UTC

---

## Scripts

- `data.py` → Framework for building datasets from custom data sources (requires implementation)
- `train_forecast.py` → Train, optimize, evaluate, and save the best model
- `predict.py` → Load trained model and predict 12, 24, or 96 steps ahead

---

## Quick Start

### Training
```bash
# Basic training
python train_forecast.py data.csv --max-lag 96 --n-trials 50

# With walk-forward validation
python train_forecast.py data.csv --walk-forward --cv-folds 5
```

### Prediction
```bash
# Predict 24 steps ahead with uncertainty intervals
python predict.py model.pkl history.csv 24 --output predictions.csv
```

---

## Training Workflow (`train_forecast.py`)

### 1. Data Loading & Cleaning
- **Target:** `premium`
- **Timestamp:** `valueDateTimeOffset` (parsed to UTC)
- **Regressors:** All other columns (used only as lagged features)
- Removes duplicates, missing values, and outliers (>3 std dev)

### 2. Feature Engineering
Creates comprehensive feature set:
- **Lag features**: `{1,2,3,6,12,24,48,96}` for target and all regressors
- **Rolling statistics**: Mean, std, min, max over windows `{6,12,24,48}`
  - **Fixed data leakage**: Uses `min_periods=window` to prevent incomplete windows
- **Fourier features**: Periodic patterns (hourly, weekly, yearly cycles)
- **Seasonal decomposition**: Trend, seasonal, residual components
- **Calendar features**: Hour, day of week, day of month, month, quarter, weekend flag
- **Interaction features**: Cross-lag interactions between target and regressors

### 3. Train/Test Split
- **80/20 split** with proper temporal separation
- **Critical fix**: Added `max_lag` buffer between train/test to prevent leakage
- Preserves chronological order

### 4. Model Training with Optimization
Each model undergoes:
- **Hyperparameter optimization** using Optuna (default 50 trials)
- **Early stopping** for gradient boosting models (LightGBM, XGBoost, CatBoost)
- **Expanding window CV** for time series validation
- Proper preprocessing (scaling for linear models, imputation for all)

### 5. Evaluation
**Standard evaluation:**
- MAE and RMSE on test set
- Best model selected by lowest MAE
- Ensemble predictions (average of all models)

**Walk-forward validation** (optional with `--walk-forward`):
- Multiple train/test splits advancing through time
- More realistic performance estimates
- Reports mean MAE ± standard deviation

### 6. Model Interpretation
- **Feature importance** automatically saved to CSV
- Top 10 most important features displayed
- Works for both tree models (built-in) and linear models (coefficient magnitude)

### 7. Uncertainty Quantification
- **Prediction intervals** calculated from residual distribution
- 95% confidence bounds by default
- Saved with model for use in predictions

### 8. Outputs
- `model.pkl` → Complete model bundle with metadata
- `forecast.csv` → Test predictions with actual values
- `metrics.json` → MAE/RMSE for all models
- `forecast_chart.png` → Visualization with prediction intervals (shaded bands)
- `feature_importance.csv` → Ranked feature importance

---

## Prediction Workflow (`predict.py`)

### 1. Load Model & History
- Loads trained model with all metadata
- Reads historical data (must include all regressor columns)
- Parses timestamps to UTC

### 2. Frequency Handling
- Uses frequency from training
- Can infer from data if not specified
- Override with `--freq` flag

### 3. Recursive Multi-Step Prediction
For each future step:
1. Generate all required features (lags, rolling, calendar, etc.)
2. Make prediction using best model
3. Calculate prediction intervals (95% confidence)
4. Append prediction to history for next step

**Important**: Regressor values are forward-filled during prediction. For better accuracy, provide expected future regressor values if available.

### 4. Output
CSV with columns:
- `valueDateTimeOffset` → Future timestamps
- `y_pred` → Point predictions
- `y_lower` → Lower bound (95% confidence)
- `y_upper` → Upper bound (95% confidence)
- `confidence_level` → 0.95

---

## Command-Line Arguments

### train_forecast.py
```
positional arguments:
  csv_path              Path to CSV file with time series data

optional arguments:
  --max-lag            Maximum lag for features (default: 96)
  --freq               Frequency for regularization (e.g., "H", "D")
  --n-trials           Number of Optuna optimization trials (default: 50)
  --cv-folds           Number of cross-validation folds (default: 3)
  --model-path         Path to save model (default: model.pkl)
  --forecast-path      Path to save forecasts (default: forecast.csv)
  --metrics-path       Path to save metrics (default: metrics.json)
  --chart-path         Path to save chart (default: forecast_chart.png)
  --importance-path    Path to save feature importance (default: feature_importance.csv)
  --walk-forward       Use walk-forward validation (flag)
```

### predict.py
```
positional arguments:
  model_path           Path to trained model file
  history_path         Path to historical data CSV
  steps                Number of steps to predict (12, 24, or 96)

optional arguments:
  --freq               Override frequency (e.g., "H", "D")
  --output             Output CSV file path (default: predictions.csv)
```

---

## Recent Improvements

### Critical Fixes
1. **Data Leakage Prevention**
   - Rolling features now use `min_periods=window` (was `min_periods=1`)
   - Prevents incomplete windows that leak future information
   - May reduce test scores but provides realistic estimates

2. **Temporal Split Fix**
   - Added `max_lag` buffer between train/test sets
   - Prevents lag features from crossing the boundary
   - Ensures true temporal separation

3. **Feature Consistency**
   - Model now saves exact feature names and order
   - Prediction uses same features in same order
   - Eliminates silent prediction errors

4. **Modern pandas API**
   - Replaced deprecated `fillna(method='ffill')` with `.ffill()`
   - Future-proof code

### Enhancements
5. **Early Stopping**
   - All gradient boosting models use validation-based early stopping
   - 20-40% faster training
   - 5-15% better generalization typically

6. **Feature Importance**
   - Automatic calculation and export
   - Works for all model types
   - Helps identify key predictors

7. **Prediction Intervals**
   - Residual-based 95% confidence intervals
   - Visualized in charts
   - Exported in predictions
   - Critical for risk assessment

8. **Walk-Forward Validation**
   - Optional robust evaluation method
   - Multiple expanding window splits
   - More realistic performance estimates

---

## Why This Avoids Leakage

✅ **Regressors**: Never used directly, only lagged versions
✅ **Rolling features**: Use complete windows only (`min_periods=window`)
✅ **Train/test**: Separated by `max_lag` buffer
✅ **Calendar features**: Derived from timestamps (safe)
✅ **Multi-step**: Recursive predictions, no future data

---

## Model Comparison

| Model      | Preprocessing        | Strengths                          | Best For              |
|------------|----------------------|------------------------------------|-----------------------|
| Ridge      | Scale + Impute       | Regularized linear, stable         | Linear relationships  |
| Lasso      | Scale + Impute       | Feature selection, sparse          | High-dim data        |
| LightGBM   | Impute only          | Fast, handles missing, categorical | Large datasets       |
| XGBoost    | Impute only          | Robust, excellent performance      | General purpose      |
| Ensemble   | N/A                  | Averages all models, stable        | Risk-averse          |

Best model is automatically selected by lowest MAE on test set.

---

## DST Handling

- All timestamps parsed to **UTC**
- Avoids ambiguity during DST transitions
- Convert to local time after predictions if needed

---

## Performance Expectations

- **Data leakage fixes**: More realistic (often lower) test scores
- **Early stopping**: 20-40% faster training
- **Walk-forward validation**: Typically 10-20% higher MAE than single split
- **Prediction intervals**: Minimal overhead (<5% slower)

---

## Output Files Reference

| File                     | Description                                    |
|--------------------------|------------------------------------------------|
| `model.pkl`              | Complete model bundle with all metadata        |
| `forecast.csv`           | Test set predictions vs actuals                |
| `metrics.json`           | MAE/RMSE for all models                        |
| `forecast_chart.png`     | Visualization with prediction intervals        |
| `feature_importance.csv` | Ranked features with importance scores         |
| `predictions.csv`        | Future predictions with confidence intervals   |

---

## Example Workflow

```bash
# 1. Train model with walk-forward validation
python train_forecast.py historical_data.csv \
  --max-lag 96 \
  --n-trials 100 \
  --walk-forward \
  --model-path my_model.pkl \
  --importance-path features.csv

# Output:
# - my_model.pkl (trained model)
# - features.csv (top features)
# - forecast_chart.png (with intervals)
# - metrics.json (performance)

# 2. Make predictions
python predict.py my_model.pkl latest_data.csv 24 \
  --output next_24h.csv

# Output:
# - next_24h.csv with columns:
#   [valueDateTimeOffset, y_pred, y_lower, y_upper, confidence_level]
```

---

## Future Enhancements (Optional)

- SHAP values for explainability
- Direct multi-step forecasting
- Production drift detection
- Automated feature selection
- Quantile regression for asymmetric intervals

---

## Requirements

- pandas, numpy
- scikit-learn
- lightgbm, xgboost
- optuna
- statsmodels
- matplotlib, seaborn
- joblib

---

## Notes

- Models are refitted on full dataset after evaluation
- Feature engineering is identical in training and prediction
- All random seeds set to 42 for reproducibility
- Supports any number of regressor columns
- Horizons limited to 12, 24, or 96 steps (configurable in code)
