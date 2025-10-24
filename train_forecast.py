#!/usr/bin/env python3
"""
Enhanced Time Series Forecasting with Advanced Models and Features
Supports Ridge, Lasso, LightGBM, XGBoost, and Ensemble methods
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from pathlib import Path
import json
import joblib
from datetime import datetime, timezone
import pytz

# ML imports
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
# import xgboost as xgb  # Disabled: slow training, similar performance to Lasso/LightGBM
import optuna

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class EnhancedTimeSeriesForecaster:
    def __init__(self, max_lag=96, freq=None, n_trials=100, cv_folds=3, use_walk_forward=False):
        self.max_lag = max_lag
        self.freq = freq
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_walk_forward = use_walk_forward
        self.lag_set = [1, 2, 4, 8, 16, 24, 48, 96, 336]
        self.lag_set = [lag for lag in self.lag_set if lag <= max_lag]

        # Features that are pre-shifted by 2 periods in data.py
        # These features need adjusted lags/windows to avoid redundancy
        self.shifted_features = ['indo', 'indo_ndf', 'modelError', 'itsdo', 'itsdo_tsdf', 'HH_NET_SUM']
        self.shifted_lag_set = [1, 2, 6, 14, 22, 46, 94, 334]  # Adjusted lags (subtract 2 from standard)
        self.shifted_lag_set = [lag for lag in self.shifted_lag_set if lag <= max_lag]
        self.shifted_windows = [2, 6, 14, 22, 46, 94, 334]  # Adjusted rolling windows

        self.feature_columns = []
        self.feature_names = []  # Store actual feature names used during training
        self.target_col = 'premium'
        self.timestamp_col = 'valueDateTimeOffset'
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.best_model_name = None
        self.ensemble_model = None
        self.residuals = None  # For prediction intervals
        self.best_params = {}  # Store best hyperparameters for walk-forward retraining

    def load_data(self, csv_path):
        """Load and initial processing of data"""
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Parse timestamps to UTC
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)

        # Sort by time
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        # Drop duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=[self.timestamp_col], keep='last')
        print(f"Final dataset shape after deduplication: {df.shape}")
        print(f"NOTE: NaN rows will be removed AFTER feature engineering to preserve temporal alignment")

        return df

    def regularize_frequency(self, df):
        """Regularize to a specific frequency if requested"""
        if self.freq:
            print(f"Regularizing to frequency: {self.freq}")
            df_indexed = df.set_index(self.timestamp_col)

            # Create regular grid
            start_time = df_indexed.index.min()
            end_time = df_indexed.index.max()
            regular_index = pd.date_range(start=start_time, end=end_time, freq=self.freq, tz=timezone.utc)

            # Reindex and forward fill
            df_regular = df_indexed.reindex(regular_index, method='ffill')
            df = df_regular.reset_index().rename(columns={'index': self.timestamp_col})

        return df

    def remove_outliers(self, df, std_threshold=3):
        """Remove outliers from target variable"""
        mean_val = df[self.target_col].mean()
        std_val = df[self.target_col].std()

        outlier_mask = np.abs(df[self.target_col] - mean_val) > (std_threshold * std_val)
        outliers_removed = outlier_mask.sum()

        if outliers_removed > 0:
            print(f"Removing {outliers_removed} outliers (>{std_threshold} std from mean)")
            df = df[~outlier_mask].reset_index(drop=True)

        return df

    def create_fourier_features(self, df, periods=[4, 8, 12, 24, 48, 168]):  # 4h, 8h, 12h, daily, 2-day, weekly
        """Create Fourier features for periodic patterns"""
        fourier_features = pd.DataFrame(index=df.index)

        # Create time index in hours from start
        time_idx = (df[self.timestamp_col] - df[self.timestamp_col].min()).dt.total_seconds() / 3600

        for period in periods:
            for k in range(1, 3):  # First 2 harmonics
                fourier_features[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * time_idx / period)
                fourier_features[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * time_idx / period)

        return fourier_features

    def create_rolling_features(self, df, windows=[4, 8, 16, 24, 48, 96, 336], shifted_windows=[2, 6, 14, 22, 46, 94, 334]):
        """Create rolling statistical features"""
        rolling_features = pd.DataFrame(index=df.index)

        # Only create rolling features for numeric columns (EXCLUDING target to prevent leakage)
        numeric_cols = [col for col in self.feature_columns
                       if pd.api.types.is_numeric_dtype(df[col])]

        for col in numeric_cols:
            # Check if this is a shift(2) feature from data.py
            if col in self.shifted_features:
                # Use adjusted windows for features already shifted by 2 periods
                window_set = shifted_windows
                print(f"Creating adjusted rolling windows for shift(2) feature: {col}")
            else:
                # Use standard windows for normal features
                window_set = windows

            for window in window_set:
                if window <= len(df):
                    # Fix data leakage: use min_periods=window to ensure full window
                    rolling_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=window).mean()
                    rolling_features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=window).std()
                    rolling_features[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=window).min()
                    rolling_features[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=window).max()

        return rolling_features

    def create_seasonal_features(self, df):
        """Create seasonal decomposition features"""
        # DISABLED: Seasonal decomposition of target causes data leakage
        # The decomposition uses the entire time series including future values
        # which would not be available at prediction time
        seasonal_features = pd.DataFrame(index=df.index)
        print("Skipping seasonal decomposition features to prevent target leakage")
        return seasonal_features

    def create_lag_features(self, df):
        """Create lag features for target and regressors"""
        feature_df = df[[self.timestamp_col, self.target_col] + self.feature_columns].copy()

        # Only create lag features for numeric columns
        numeric_feature_cols = [col for col in self.feature_columns
                               if pd.api.types.is_numeric_dtype(df[col])]

        # Create lag features with appropriate lag set based on feature type
        for col in numeric_feature_cols:
            # Check if this is a shift(2) feature from data.py
            if col in self.shifted_features:
                # Use adjusted lag set for features already shifted by 2 periods
                lag_set = self.shifted_lag_set
                print(f"Creating adjusted lags for shift(2) feature: {col}")
            else:
                # Use standard lag set for normal features
                lag_set = self.lag_set

            # Create lag features
            for lag in lag_set:
                feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)

        return feature_df

    def create_interaction_features(self, df, max_interactions=20):
        """Create interaction features between important features (excluding premium/target)"""
        interaction_features = pd.DataFrame(index=df.index)

        # Get ALL numeric feature columns (excluding target/premium and timestamp)
        # This includes: original features, lag features, rolling features, calendar features, etc.
        all_feature_cols = [col for col in df.columns
                           if col not in [self.timestamp_col, self.target_col]
                           and not col.startswith(f'{self.target_col}_')  # Exclude any premium-derived columns
                           and pd.api.types.is_numeric_dtype(df[col])]

        print(f"Found {len(all_feature_cols)} numeric features for potential interactions")

        # Create interactions between top features (by variance)
        if len(all_feature_cols) >= 2:
            # Calculate variance for each feature to identify most informative ones
            feature_vars = []
            for col in all_feature_cols:
                var = df[col].var()
                if not pd.isna(var) and var > 0:  # Exclude constant or NaN features
                    feature_vars.append((col, var))

            # Sort by variance (higher variance = potentially more informative)
            feature_vars.sort(key=lambda x: x[1], reverse=True)
            top_features = [col for col, _ in feature_vars[:15]]  # Top 15 features by variance

            print(f"Creating interactions from top {len(top_features)} features by variance")

            # Create pairwise interactions
            interaction_count = 0
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:  # Avoid duplicate pairs and self-interaction
                    if interaction_count >= max_interactions:
                        break
                    interaction_features[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    interaction_count += 1
                if interaction_count >= max_interactions:
                    break

            print(f"Created {interaction_count} feature interactions (premium completely excluded)")
        else:
            print("Not enough features to create interactions")

        return interaction_features

    def create_calendar_features(self, df):
        """Create calendar-based features"""
        calendar_features = pd.DataFrame(index=df.index)

        dt = df[self.timestamp_col]
        calendar_features['hour'] = dt.dt.hour
        calendar_features['dow'] = dt.dt.dayofweek  # 0=Monday
        calendar_features['dom'] = dt.dt.day
        calendar_features['month'] = dt.dt.month
        calendar_features['quarter'] = dt.dt.quarter
        calendar_features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        calendar_features['is_month_start'] = dt.dt.is_month_start.astype(int)
        calendar_features['is_month_end'] = dt.dt.is_month_end.astype(int)

        return calendar_features

    def feature_engineering(self, df):
        """Complete feature engineering pipeline"""
        print("Starting feature engineering...")

        # Identify regressor columns (all except timestamp and target)
        self.feature_columns = [col for col in df.columns if col not in [self.timestamp_col, self.target_col]]
        print(f"Found {len(self.feature_columns)} regressor columns")

        # Create lag features
        print("Creating lag features...")
        feature_df = self.create_lag_features(df)

        # Create rolling features
        print("Creating rolling statistical features...")
        rolling_features = self.create_rolling_features(df)
        feature_df = pd.concat([feature_df, rolling_features], axis=1)

        # Create Fourier features
        print("Creating Fourier features...")
        fourier_features = self.create_fourier_features(df)
        feature_df = pd.concat([feature_df, fourier_features], axis=1)

        # Create seasonal features
        print("Creating seasonal decomposition features...")
        seasonal_features = self.create_seasonal_features(df)
        if not seasonal_features.empty:
            feature_df = pd.concat([feature_df, seasonal_features], axis=1)

        # Create calendar features
        print("Creating calendar features...")
        calendar_features = self.create_calendar_features(df)
        feature_df = pd.concat([feature_df, calendar_features], axis=1)

        # Create interaction features
        print("Creating interaction features...")
        interaction_features = self.create_interaction_features(feature_df)
        if not interaction_features.empty:
            feature_df = pd.concat([feature_df, interaction_features], axis=1)

        # Drop rows with NaNs caused by lag/rolling features
        print("=" * 60)
        print("Removing NaN Rows (Post Feature Engineering)")
        print("=" * 60)

        initial_rows = len(feature_df)

        # Analyze NaN distribution before dropping
        nan_counts = feature_df.isnull().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

        if len(nan_counts) > 0:
            print(f"\nNaN counts by column (before removal):")
            # Show top 10 columns with most NaNs
            for col, count in nan_counts.head(10).items():
                print(f"  {col}: {count} NaNs ({count/initial_rows*100:.2f}%)")
            if len(nan_counts) > 10:
                print(f"  ... and {len(nan_counts) - 10} more columns with NaNs")
        else:
            print("\nNo NaN values found in any column")

        # Drop rows with NaNs
        feature_df = feature_df.dropna()

        removed_rows = initial_rows - len(feature_df)
        print(f"\nRows before: {initial_rows}")
        print(f"Rows after: {len(feature_df)}")
        print(f"Rows removed: {removed_rows}")
        print(f"Final shape: {feature_df.shape}")
        print("=" * 60)

        return feature_df

    def expanding_window_split(self, df, min_train_size=0.6):
        """Create expanding window splits for time series validation"""
        n = len(df)
        min_train = int(n * min_train_size)

        splits = []
        # Create multiple expanding windows
        for i in range(self.cv_folds):
            train_end = min_train + int((n - min_train) * (i + 1) / (self.cv_folds + 1))
            test_start = train_end
            test_end = min_train + int((n - min_train) * (i + 2) / (self.cv_folds + 1))

            if test_end <= n:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, min(test_end, n))
                if len(test_idx) > 0:
                    splits.append((train_idx, test_idx))

        return splits

    def optimize_hyperparameters(self, X_train, y_train, model_type):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters for {model_type}...")

        def objective(trial):
            if model_type == 'ridge':
                alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
                model = Ridge(alpha=alpha, random_state=42)

            elif model_type == 'lasso':
                alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
                model = Lasso(alpha=alpha, random_state=42, max_iter=500)

            elif model_type == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'n_estimators': 1000,  # Use early stopping
                }
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

            # elif model_type == 'xgboost':  # Disabled: slow training, similar performance
            #     params = {
            #         'n_estimators': 1000,  # Use early stopping
            #         'max_depth': trial.suggest_int('max_depth', 3, 10),
            #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            #         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            #         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            #         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            #         'early_stopping_rounds': 50,
            #     }
            #     model = xgb.XGBRegressor(**params, random_state=42, verbose=0)

            # Cross-validation with expanding windows
            splits = self.expanding_window_split(pd.DataFrame(X_train))
            scores = []

            for train_idx, val_idx in splits:
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Apply preprocessing for linear models
                if model_type in ['ridge', 'lasso']:
                    # Use temporary scalers for CV
                    temp_scaler = StandardScaler()
                    temp_imputer = SimpleImputer(strategy='median')

                    X_tr_processed = temp_scaler.fit_transform(temp_imputer.fit_transform(X_tr))
                    X_val_processed = temp_scaler.transform(temp_imputer.transform(X_val))
                else:
                    temp_imputer = SimpleImputer(strategy='median')
                    X_tr_processed = temp_imputer.fit_transform(X_tr)
                    X_val_processed = temp_imputer.transform(X_val)

                # Fit with early stopping for gradient boosting models
                if model_type == 'lightgbm':
                    model.fit(X_tr_processed, y_tr,
                            eval_set=[(X_val_processed, y_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
                # elif model_type == 'xgboost':  # Disabled
                #     model.fit(X_tr_processed, y_tr,
                #             eval_set=[(X_val_processed, y_val)],
                #             verbose=False)
                else:
                    model.fit(X_tr_processed, y_tr)

                y_pred = model.predict(X_val_processed)
                mae = mean_absolute_error(y_val, y_pred)
                scores.append(mae)

            return np.mean(scores)

        # Run optimization with model-specific trials
        if model_type == 'lasso':
            n_trials = 10  # Lasso only has 1 hyperparameter, 10 trials is sufficient
        else:
            n_trials = min(self.n_trials, 50)  # Keep 50 for other models

        study = optuna.create_study(direction='minimize')

        # Parallelize Optuna trials to utilize multiple CPU cores
        if model_type == 'lightgbm':
            # LightGBM already uses all cores internally, limit parallel trials
            n_jobs = 4
        else:
            # Ridge/Lasso are single-threaded, can run many trials in parallel
            n_jobs = 8

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        return study.best_params

    def train_models(self, X_train, y_train):
        """Train all models with hyperparameter optimization"""
        print("Training models with hyperparameter optimization...")

        # Store feature names from training data
        self.feature_names = list(X_train.columns)

        model_configs = {
            'ridge': Ridge,
            'lasso': Lasso,
            'lightgbm': lgb.LGBMRegressor,
            # 'xgboost': xgb.XGBRegressor  # Disabled: slow training, similar performance
        }

        for model_name, model_class in model_configs.items():
            print(f"\nTraining {model_name}...")

            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_train, y_train, model_name)
            print(f"Best parameters for {model_name}: {best_params}")

            # Store best hyperparameters for walk-forward retraining
            self.best_params[model_name] = best_params

            # Prepare data based on model type
            if model_name in ['ridge', 'lasso']:
                # Linear models need scaling and imputation
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()

                X_processed = scaler.fit_transform(imputer.fit_transform(X_train))

                self.imputers[model_name] = imputer
                self.scalers[model_name] = scaler

                if model_name == 'ridge':
                    model = Ridge(**best_params, random_state=42)
                else:  # lasso
                    model = Lasso(**best_params, random_state=42, max_iter=500)

            else:
                # Tree-based models only need imputation
                imputer = SimpleImputer(strategy='median')
                X_processed = imputer.fit_transform(X_train)

                self.imputers[model_name] = imputer

                if model_name == 'lightgbm':
                    model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
                # elif model_name == 'xgboost':  # Disabled
                #     model = xgb.XGBRegressor(**best_params, random_state=42, verbose=0)

            # Train the model (with early stopping for tree models)
            if model_name == 'lightgbm':
                # Create a validation split for early stopping
                val_size = int(len(X_train) * 0.2)
                X_tr, X_val = X_processed[:-val_size], X_processed[-val_size:]
                y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

                model.fit(X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
                # elif model_name == 'xgboost':  # Disabled
                #     model.fit(X_tr, y_tr,
                #             eval_set=[(X_val, y_val)],
                #             verbose=False)
            else:
                model.fit(X_processed, y_train)

            self.models[model_name] = model
            print(f"{model_name} training completed")

    def create_ensemble(self, X_train, y_train):
        """Create ensemble model from individual models"""
        print("Creating ensemble model...")

        # Create voting regressor with all models
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))

        # We can't directly use VotingRegressor due to different preprocessing needs
        # Instead, we'll create a custom ensemble that averages predictions
        self.ensemble_model = 'weighted_average'  # Flag for custom ensemble
        print("Ensemble model created using weighted average")

    def walk_forward_validation(self, X, y, n_splits=5):
        """Perform TRUE walk-forward validation with retraining at each split"""
        print(f"\nPerforming TRUE walk-forward validation with {n_splits} splits...")
        print("NOTE: Models will be retrained at each split (no data leakage)")

        n = len(X)
        min_train = int(n * 0.6)  # Minimum 60% for initial training
        step_size = (n - min_train) // n_splits

        results = {model_name: [] for model_name in self.best_params.keys()}
        results['ensemble'] = []

        for i in range(n_splits):
            train_end = min_train + i * step_size
            test_start = train_end
            test_end = min(train_end + step_size, n)

            if test_end <= test_start:
                break

            print(f"\n  Split {i+1}/{n_splits}: Train[0:{train_end}], Test[{test_start}:{test_end}]")

            X_train_wf = X.iloc[:train_end]
            y_train_wf = y.iloc[:train_end]
            X_test_wf = X.iloc[test_start:test_end]
            y_test_wf = y.iloc[test_start:test_end]

            # Store predictions for ensemble
            split_predictions = {}

            # Retrain and evaluate each model with best hyperparameters
            for model_name in self.best_params.keys():
                print(f"    Retraining {model_name}...")
                best_params = self.best_params[model_name]

                # Create and preprocess based on model type
                if model_name == 'ridge':
                    # Ridge: impute + scale
                    imputer_wf = SimpleImputer(strategy='median')
                    scaler_wf = StandardScaler()

                    X_train_processed = scaler_wf.fit_transform(
                        imputer_wf.fit_transform(X_train_wf)
                    )
                    X_test_processed = scaler_wf.transform(
                        imputer_wf.transform(X_test_wf)
                    )

                    model = Ridge(**best_params, random_state=42)
                    model.fit(X_train_processed, y_train_wf)

                elif model_name == 'lasso':
                    # Lasso: impute + scale
                    imputer_wf = SimpleImputer(strategy='median')
                    scaler_wf = StandardScaler()

                    X_train_processed = scaler_wf.fit_transform(
                        imputer_wf.fit_transform(X_train_wf)
                    )
                    X_test_processed = scaler_wf.transform(
                        imputer_wf.transform(X_test_wf)
                    )

                    model = Lasso(**best_params, random_state=42, max_iter=500)
                    model.fit(X_train_processed, y_train_wf)

                elif model_name == 'lightgbm':
                    # LightGBM: impute only
                    imputer_wf = SimpleImputer(strategy='median')

                    X_train_processed = imputer_wf.fit_transform(X_train_wf)
                    X_test_processed = imputer_wf.transform(X_test_wf)

                    # Use early stopping with validation split
                    val_size = int(len(X_train_processed) * 0.2)
                    X_tr, X_val = X_train_processed[:-val_size], X_train_processed[-val_size:]
                    y_tr, y_val = y_train_wf.iloc[:-val_size], y_train_wf.iloc[-val_size:]

                    model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
                    model.fit(X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)])

                # Make predictions
                y_pred = model.predict(X_test_processed)
                mae = mean_absolute_error(y_test_wf, y_pred)
                results[model_name].append(mae)
                split_predictions[model_name] = y_pred

                print(f"      {model_name} MAE: {mae:.4f}")

            # Calculate ensemble prediction (simple average)
            ensemble_pred = np.mean([split_predictions[m] for m in split_predictions.keys()], axis=0)
            mae_ensemble = mean_absolute_error(y_test_wf, ensemble_pred)
            results['ensemble'].append(mae_ensemble)
            print(f"      ensemble MAE: {mae_ensemble:.4f}")

        # Calculate average results
        print("\n" + "="*60)
        print("Walk-Forward Validation Results (with retraining):")
        print("="*60)

        avg_results = {}
        for model_name, scores in results.items():
            if len(scores) > 0:
                avg_mae = np.mean(scores)
                std_mae = np.std(scores)
                avg_results[model_name] = {'mae': avg_mae, 'mae_std': std_mae}
                print(f"{model_name:10s} | MAE: {avg_mae:.4f} ± {std_mae:.4f}")

        # Select best model
        best_model = min(avg_results.keys(), key=lambda x: avg_results[x]['mae'])
        self.best_model_name = best_model
        print(f"\nBest model (walk-forward): {best_model} (MAE: {avg_results[best_model]['mae']:.4f})")

        return avg_results

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and select the best one"""
        print("\nEvaluating models...")
        results = {}

        for model_name, model in self.models.items():
            # Apply same preprocessing as training
            if model_name in ['ridge', 'lasso']:
                X_processed = self.scalers[model_name].transform(
                    self.imputers[model_name].transform(X_test)
                )
            else:
                X_processed = self.imputers[model_name].transform(X_test)

            # Make predictions
            y_pred = model.predict(X_processed)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred,
                'residuals': y_test.values - y_pred
            }

            print(f"{model_name:10s} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Evaluate ensemble
        ensemble_pred = np.zeros(len(y_test))
        for model_name in self.models.keys():
            ensemble_pred += results[model_name]['predictions']
        ensemble_pred /= len(self.models)

        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

        results['ensemble'] = {
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'predictions': ensemble_pred,
            'residuals': y_test.values - ensemble_pred
        }

        print(f"{'ensemble':10s} | MAE: {ensemble_mae:.4f} | RMSE: {ensemble_rmse:.4f}")

        # Select best model by MAE
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        self.best_model_name = best_model

        # Store residuals for prediction intervals
        self.residuals = results[best_model]['residuals']

        print(f"\nBest model: {best_model} (MAE: {results[best_model]['mae']:.4f})")

        return results

    def calculate_prediction_intervals(self, y_pred, confidence_level=0.95):
        """Calculate prediction intervals using residual distribution"""
        if self.residuals is None:
            print("Warning: No residuals available for prediction intervals")
            return y_pred, y_pred

        # Calculate quantiles from residuals
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - (alpha / 2)

        # Use residual quantiles to estimate intervals
        lower_bound = np.percentile(self.residuals, lower_quantile * 100)
        upper_bound = np.percentile(self.residuals, upper_quantile * 100)

        # Add to predictions
        y_lower = y_pred + lower_bound
        y_upper = y_pred + upper_bound

        return y_lower, y_upper

    def create_prediction_chart(self, timestamps, y_true, y_pred, save_path='forecast_chart.png',
                               show_intervals=True, confidence_level=0.95):
        """Create and save prediction vs actual chart with optional prediction intervals"""
        plt.figure(figsize=(15, 8))

        plt.plot(timestamps, y_true, label='Actual', alpha=0.7, linewidth=1)
        plt.plot(timestamps, y_pred, label='Predicted', alpha=0.8, linewidth=1)

        # Add prediction intervals if available
        if show_intervals and self.residuals is not None:
            y_lower, y_upper = self.calculate_prediction_intervals(y_pred, confidence_level)
            plt.fill_between(timestamps, y_lower, y_upper, alpha=0.2,
                           label=f'{int(confidence_level*100)}% Prediction Interval')

        plt.title('Actual vs Predicted Values', fontsize=16, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Premium', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Improve layout
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save chart
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Prediction chart saved to {save_path}")

    def get_feature_importance(self):
        """Get feature importance from the best model"""
        feature_importance = {}

        if self.best_model_name == 'ensemble':
            # Average importance across all tree models
            for model_name, model in self.models.items():
                if model_name == 'lightgbm':  # Only lightgbm now (xgboost disabled)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance[model_name] = model.feature_importances_
        else:
            # Get importance from best model
            model = self.models[self.best_model_name]
            if hasattr(model, 'feature_importances_'):
                feature_importance[self.best_model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):  # Linear models
                feature_importance[self.best_model_name] = np.abs(model.coef_)

        return feature_importance

    def save_feature_importance(self, feature_names=None, output_path='feature_importance.csv'):
        """Save feature importance to CSV"""
        importance_dict = self.get_feature_importance()

        if not importance_dict:
            print("No feature importance available for this model")
            return

        # Use stored feature names if not provided
        if feature_names is None:
            feature_names = self.feature_names

        # Verify length matches
        first_importance = next(iter(importance_dict.values()))
        if len(feature_names) != len(first_importance):
            print(f"Warning: Feature name count ({len(feature_names)}) doesn't match importance count ({len(first_importance)})")
            print("Using stored feature names from training data")
            feature_names = self.feature_names

        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({'feature': feature_names})

        for model_name, importances in importance_dict.items():
            importance_df[f'{model_name}_importance'] = importances

        # Sort by importance (use first model's importance)
        first_col = importance_df.columns[1]
        importance_df = importance_df.sort_values(first_col, ascending=False)

        importance_df.to_csv(output_path, index=False)
        print(f"Feature importance saved to {output_path}")

        # Print top 10 features
        print("\nTop 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row[first_col]:.6f}")

    def refit_best_model(self, X_full, y_full):
        """Refit the best model on full dataset"""
        print(f"\nRefitting {self.best_model_name} on full dataset...")

        if self.best_model_name == 'ensemble':
            # Refit all models for ensemble
            for model_name, model in self.models.items():
                if model_name in ['ridge', 'lasso']:
                    X_processed = self.scalers[model_name].fit_transform(
                        self.imputers[model_name].fit_transform(X_full)
                    )
                else:
                    X_processed = self.imputers[model_name].fit_transform(X_full)

                model.fit(X_processed, y_full)
        else:
            # Refit single best model
            if self.best_model_name in ['ridge', 'lasso']:
                X_processed = self.scalers[self.best_model_name].fit_transform(
                    self.imputers[self.best_model_name].fit_transform(X_full)
                )
            else:
                X_processed = self.imputers[self.best_model_name].fit_transform(X_full)

            self.models[self.best_model_name].fit(X_processed, y_full)

        print("Model refitting completed")

    def save_model(self, filepath='model.pkl', feature_names=None):
        """Save the complete model bundle"""
        model_bundle = {
            'best_model_name': self.best_model_name,
            'models': self.models,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'feature_columns': self.feature_columns,
            'feature_names': feature_names,  # Save exact feature order from training
            'lag_set': self.lag_set,
            'max_lag': self.max_lag,
            'freq': self.freq,
            'target_col': self.target_col,
            'timestamp_col': self.timestamp_col,
            'ensemble_model': self.ensemble_model,
            'residuals': self.residuals  # For prediction intervals
        }

        joblib.dump(model_bundle, filepath)
        print(f"Model bundle saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Time Series Forecasting')
    parser.add_argument('csv_path', help='Path to CSV file with time series data')
    parser.add_argument('--max-lag', type=int, default=96, help='Maximum lag for features')
    parser.add_argument('--freq', type=str, help='Frequency for regularization (e.g., "H", "D")')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of hyperparameter optimization trials')
    parser.add_argument('--cv-folds', type=int, default=3, help='Number of cross-validation folds')
    parser.add_argument('--model-path', type=str, default='model.pkl', help='Path to save model')
    parser.add_argument('--forecast-path', type=str, default='forecast.csv', help='Path to save forecasts')
    parser.add_argument('--metrics-path', type=str, default='metrics.json', help='Path to save metrics')
    parser.add_argument('--chart-path', type=str, default='forecast_chart.png', help='Path to save chart')
    parser.add_argument('--importance-path', type=str, default='feature_importance.csv', help='Path to save feature importance')
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward validation instead of single train/test split')
    parser.add_argument('--train-end-date', type=str, help='End date for training (format: YYYY-MM-DD). If provided, uses custom date-based split.')
    parser.add_argument('--test-start-date', type=str, help='Start date for testing (format: YYYY-MM-DD). If provided, uses custom date-based split.')

    args = parser.parse_args()

    # Initialize forecaster
    forecaster = EnhancedTimeSeriesForecaster(
        max_lag=args.max_lag,
        freq=args.freq,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        use_walk_forward=args.walk_forward
    )

    # Load and process data
    df = forecaster.load_data(args.csv_path)
    df = forecaster.regularize_frequency(df)
    df = forecaster.remove_outliers(df)

    # Feature engineering
    feature_df = forecaster.feature_engineering(df)

    # Prepare features and target
    # Filter out non-numeric columns (except timestamp and target)
    feature_cols = [col for col in feature_df.columns
                   if col not in [forecaster.timestamp_col, forecaster.target_col]
                   and pd.api.types.is_numeric_dtype(feature_df[col])]

    print(f"Selected {len(feature_cols)} numeric feature columns for modeling")

    X = feature_df[feature_cols]
    y = feature_df[forecaster.target_col]
    timestamps = feature_df[forecaster.timestamp_col]

    # Time-based split - use custom dates if provided, otherwise default 80/20
    if args.train_end_date or args.test_start_date:
        # Custom date-based split
        print("Using custom date-based train/test split...")

        if not args.train_end_date or not args.test_start_date:
            raise ValueError("Both --train-end-date and --test-start-date must be provided for custom split")

        # Parse dates (assume end of day for train_end_date, start of day for test_start_date)
        train_end = pd.to_datetime(args.train_end_date, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        test_start = pd.to_datetime(args.test_start_date, utc=True)

        print(f"Train end date: {train_end}")
        print(f"Test start date: {test_start}")

        # Find split indices based on timestamps
        train_mask = timestamps <= train_end
        test_mask = timestamps >= test_start

        if train_mask.sum() == 0:
            raise ValueError(f"No training data found before {train_end}")
        if test_mask.sum() == 0:
            raise ValueError(f"No test data found after {test_start}")

        # Get indices
        train_indices = timestamps[train_mask].index
        test_indices = timestamps[test_mask].index

        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]
        timestamps_test = timestamps.loc[test_indices]

        print(f"Train size: {len(X_train)} (ends {timestamps.loc[train_indices[-1]]})")
        print(f"Test size: {len(X_test)} (starts {timestamps.loc[test_indices[0]]})")

        # Calculate gap between train and test
        gap_days = (timestamps.loc[test_indices[0]] - timestamps.loc[train_indices[-1]]).days
        print(f"Gap between train and test: {gap_days} days")

    else:
        # Default percentage-based split (80/20) with proper temporal separation
        print("Using default 80/20 train/test split...")

        # Add max_lag buffer to prevent data leakage from training into test
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        # Start test set after max_lag to prevent leakage from lag features
        test_start_idx = min(split_idx + forecaster.max_lag, len(X))
        X_test = X.iloc[test_start_idx:]
        y_test = y.iloc[test_start_idx:]
        timestamps_test = timestamps.iloc[test_start_idx:]

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)} (with {forecaster.max_lag}-step buffer)")

    # Train models
    forecaster.train_models(X_train, y_train)

    # Create ensemble
    forecaster.create_ensemble(X_train, y_train)

    # Evaluate models - use walk-forward if enabled
    if forecaster.use_walk_forward:
        print("\nUsing walk-forward validation...")
        wf_results = forecaster.walk_forward_validation(X, y, n_splits=5)
        # Still evaluate on test set for comparison
        results = forecaster.evaluate_models(X_test, y_test)
    else:
        results = forecaster.evaluate_models(X_test, y_test)

    # Create prediction chart
    best_predictions = results[forecaster.best_model_name]['predictions']
    forecaster.create_prediction_chart(timestamps_test, y_test, best_predictions, args.chart_path)

    # Save feature importance before refitting
    forecaster.save_feature_importance(feature_cols, args.importance_path)

    # Refit best model on full data
    forecaster.refit_best_model(X, y)

    # Save results with feature names for proper prediction
    forecaster.save_model(args.model_path, feature_names=feature_cols)

    # Save forecast results
    forecast_df = pd.DataFrame({
        'timestamp': timestamps_test,
        'y_true': y_test,
        'y_pred': best_predictions
    })
    forecast_df.to_csv(args.forecast_path, index=False)
    print(f"Forecast results saved to {args.forecast_path}")

    # Save metrics
    metrics = {model: {'mae': results[model]['mae'], 'rmse': results[model]['rmse']}
              for model in results.keys()}

    with open(args.metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.metrics_path}")

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()