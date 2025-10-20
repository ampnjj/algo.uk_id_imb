#!/usr/bin/env python3
"""
Enhanced Time Series Prediction Script
Load trained model and predict future values with advanced features
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import joblib
from datetime import datetime, timezone
import pytz
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

class EnhancedTimeSeriesPredictor:
    def __init__(self):
        self.model_bundle = None
        self.best_model = None
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        self.lag_set = []
        self.max_lag = 96
        self.freq = None
        self.target_col = 'premium'
        self.timestamp_col = 'valueDateTimeOffset'
        self.ensemble_model = None
        self.residuals = None  # For prediction intervals

    def load_model(self, model_path):
        """Load the trained model bundle"""
        print(f"Loading model from {model_path}")
        self.model_bundle = joblib.load(model_path)

        self.best_model_name = self.model_bundle['best_model_name']
        self.models = self.model_bundle['models']
        self.scalers = self.model_bundle.get('scalers', {})
        self.imputers = self.model_bundle.get('imputers', {})
        self.feature_columns = self.model_bundle['feature_columns']
        self.lag_set = self.model_bundle['lag_set']
        self.max_lag = self.model_bundle['max_lag']
        self.freq = self.model_bundle.get('freq')
        self.target_col = self.model_bundle.get('target_col', 'premium')
        self.timestamp_col = self.model_bundle.get('timestamp_col', 'valueDateTimeOffset')
        self.ensemble_model = self.model_bundle.get('ensemble_model')
        self.residuals = self.model_bundle.get('residuals')

        print(f"Loaded model: {self.best_model_name}")
        print(f"Feature columns: {len(self.feature_columns)}")
        print(f"Lag set: {self.lag_set}")

    def load_history(self, csv_path):
        """Load historical data"""
        print(f"Loading history from {csv_path}")
        df = pd.read_csv(csv_path)

        # Parse timestamps to UTC
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)

        # Sort by time and remove duplicates
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        df = df.drop_duplicates(subset=[self.timestamp_col], keep='last')

        print(f"Loaded {len(df)} historical records")
        print(f"Date range: {df[self.timestamp_col].min()} to {df[self.timestamp_col].max()}")

        return df

    def infer_frequency(self, df):
        """Infer frequency from historical data"""
        if self.freq:
            return self.freq

        # Calculate time differences
        time_diffs = df[self.timestamp_col].diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0]

        # Map to pandas frequency strings
        seconds = most_common_diff.total_seconds()
        if seconds <= 60:
            freq = 'T'  # Minute
        elif seconds <= 3600:
            freq = 'H'  # Hour
        elif seconds <= 86400:
            freq = 'D'  # Day
        else:
            freq = 'D'  # Default to daily

        print(f"Inferred frequency: {freq}")
        return freq

    def create_fourier_features(self, df, periods=[24, 168, 8760]):
        """Create Fourier features for periodic patterns"""
        fourier_features = pd.DataFrame(index=df.index)

        # Create time index in hours from start
        time_idx = (df[self.timestamp_col] - df[self.timestamp_col].min()).dt.total_seconds() / 3600

        for period in periods:
            for k in range(1, 3):  # First 2 harmonics
                fourier_features[f'fourier_sin_{period}_{k}'] = np.sin(2 * np.pi * k * time_idx / period)
                fourier_features[f'fourier_cos_{period}_{k}'] = np.cos(2 * np.pi * k * time_idx / period)

        return fourier_features

    def create_rolling_features(self, df, windows=[6, 12, 24, 48]):
        """Create rolling statistical features"""
        rolling_features = pd.DataFrame(index=df.index)

        for col in [self.target_col] + self.feature_columns:
            if col in df.columns:
                for window in windows:
                    if window <= len(df):
                        # Fix data leakage: use min_periods=window to ensure full window
                        rolling_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=window).mean()
                        rolling_features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=window).std()
                        rolling_features[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=window).min()
                        rolling_features[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=window).max()

        return rolling_features

    def create_seasonal_features(self, df):
        """Create seasonal decomposition features"""
        seasonal_features = pd.DataFrame(index=df.index)

        # Only if we have enough data points
        if len(df) >= 48:  # At least 2 days for hourly data
            try:
                # Use a reasonable period for decomposition
                period = min(24, len(df) // 4)  # Daily cycle or quarter of data
                decomposition = seasonal_decompose(df[self.target_col].ffill(),
                                                 model='additive', period=period, extrapolate_trend='freq')

                seasonal_features[f'{self.target_col}_trend'] = decomposition.trend
                seasonal_features[f'{self.target_col}_seasonal'] = decomposition.seasonal
                seasonal_features[f'{self.target_col}_residual'] = decomposition.resid
            except:
                # If decomposition fails, create dummy features
                seasonal_features[f'{self.target_col}_trend'] = df[self.target_col]
                seasonal_features[f'{self.target_col}_seasonal'] = 0
                seasonal_features[f'{self.target_col}_residual'] = 0

        return seasonal_features

    def create_lag_features(self, df):
        """Create lag features for target and regressors"""
        feature_df = df.copy()

        # Standard lag features
        for lag in self.lag_set:
            # Target lags
            feature_df[f'{self.target_col}_lag_{lag}'] = feature_df[self.target_col].shift(lag)

            # Regressor lags
            for col in self.feature_columns:
                if col in feature_df.columns:
                    feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)

        return feature_df

    def create_interaction_features(self, df, max_interactions=10):
        """Create cross-lag interaction features"""
        interaction_features = pd.DataFrame(index=df.index)

        # Get lag columns
        target_lags = [col for col in df.columns if col.startswith(f'{self.target_col}_lag_')]
        regressor_lags = [col for col in df.columns if '_lag_' in col and not col.startswith(f'{self.target_col}_lag_')]

        # Create interactions between target lags and regressor lags
        interaction_count = 0
        for target_lag in target_lags[:5]:  # Limit to first 5 target lags
            for reg_lag in regressor_lags:
                if interaction_count >= max_interactions:
                    break
                if target_lag in df.columns and reg_lag in df.columns:
                    interaction_features[f'{target_lag}_x_{reg_lag}'] = df[target_lag] * df[reg_lag]
                    interaction_count += 1
            if interaction_count >= max_interactions:
                break

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

    def build_feature_row(self, history_df, target_timestamp):
        """Build a single feature row for prediction"""
        # Create a temporary dataframe with the target timestamp
        temp_df = history_df.copy()

        # Add the prediction timestamp with NaN target (will be filled by prediction)
        new_row = pd.DataFrame({
            self.timestamp_col: [target_timestamp],
            self.target_col: [np.nan]
        })

        # Add regressor columns with forward-filled values (assuming they're known)
        for col in self.feature_columns:
            if col in temp_df.columns:
                last_value = temp_df[col].iloc[-1] if not temp_df[col].empty else 0
                new_row[col] = [last_value]
            else:
                new_row[col] = [0]

        # Append to history
        extended_df = pd.concat([temp_df, new_row], ignore_index=True)

        # Create all features
        feature_df = self.create_lag_features(extended_df)

        # Add rolling features
        rolling_features = self.create_rolling_features(extended_df)
        feature_df = pd.concat([feature_df, rolling_features], axis=1)

        # Add Fourier features
        fourier_features = self.create_fourier_features(extended_df)
        feature_df = pd.concat([feature_df, fourier_features], axis=1)

        # Add seasonal features
        seasonal_features = self.create_seasonal_features(extended_df)
        if not seasonal_features.empty:
            feature_df = pd.concat([feature_df, seasonal_features], axis=1)

        # Add calendar features
        calendar_features = self.create_calendar_features(extended_df)
        feature_df = pd.concat([feature_df, calendar_features], axis=1)

        # Add interaction features
        interaction_features = self.create_interaction_features(feature_df)
        if not interaction_features.empty:
            feature_df = pd.concat([feature_df, interaction_features], axis=1)

        # Get the last row (corresponding to our prediction timestamp)
        feature_row = feature_df.iloc[-1:].copy()

        # Get all feature columns that were used in training
        all_feature_cols = [col for col in feature_df.columns
                           if col not in [self.timestamp_col, self.target_col]]

        # Ensure we have all required features, fill missing with 0
        feature_vector = []
        training_features = [col for col in self.model_bundle.get('feature_names', all_feature_cols)]

        for col in training_features:
            if col in feature_row.columns:
                feature_vector.append(feature_row[col].iloc[0])
            else:
                feature_vector.append(0.0)

        return np.array(feature_vector).reshape(1, -1)

    def predict_single(self, feature_vector, model_name=None):
        """Make a single prediction using specified model or best model"""
        if model_name is None:
            model_name = self.best_model_name

        if model_name == 'ensemble':
            # Make ensemble prediction
            predictions = []
            for name, model in self.models.items():
                # Apply preprocessing
                if name in ['ridge', 'lasso']:
                    X_processed = self.scalers[name].transform(
                        self.imputers[name].transform(feature_vector)
                    )
                else:
                    X_processed = self.imputers[name].transform(feature_vector)

                pred = model.predict(X_processed)[0]
                predictions.append(pred)

            return np.mean(predictions)
        else:
            # Single model prediction
            model = self.models[model_name]

            # Apply preprocessing
            if model_name in ['ridge', 'lasso']:
                X_processed = self.scalers[model_name].transform(
                    self.imputers[model_name].transform(feature_vector)
                )
            else:
                X_processed = self.imputers[model_name].transform(feature_vector)

            return model.predict(X_processed)[0]

    def predict_future(self, history_df, n_steps):
        """Predict n steps into the future"""
        print(f"Predicting {n_steps} steps into the future using {self.best_model_name}")

        # Determine frequency
        freq = self.infer_frequency(history_df)

        # Get the last timestamp
        last_timestamp = history_df[self.timestamp_col].iloc[-1]

        # Generate future timestamps
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),  # Assuming hourly for now
            periods=n_steps,
            freq=freq,
            tz=timezone.utc
        )

        # Keep a working copy of history
        working_history = history_df.copy()
        predictions = []

        for i, future_timestamp in enumerate(future_timestamps):
            print(f"Predicting step {i+1}/{n_steps}: {future_timestamp}")

            # Build feature vector for this timestamp
            feature_vector = self.build_feature_row(working_history, future_timestamp)

            # Make prediction
            prediction = self.predict_single(feature_vector)
            predictions.append(prediction)

            # Add prediction to working history for next iteration
            new_row = pd.DataFrame({
                self.timestamp_col: [future_timestamp],
                self.target_col: [prediction]
            })

            # Add regressor values (forward-fill from last known values)
            for col in self.feature_columns:
                if col in working_history.columns:
                    last_value = working_history[col].iloc[-1] if not working_history[col].empty else 0
                    new_row[col] = [last_value]
                else:
                    new_row[col] = [0]

            working_history = pd.concat([working_history, new_row], ignore_index=True)

        return future_timestamps, predictions

    def calculate_prediction_intervals(self, predictions, confidence_level=0.95):
        """Calculate prediction intervals using residual distribution"""
        if self.residuals is None:
            print("Warning: No residuals available for prediction intervals")
            return None, None

        # Calculate quantiles from residuals
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - (alpha / 2)

        # Use residual quantiles to estimate intervals
        lower_bound = np.percentile(self.residuals, lower_quantile * 100)
        upper_bound = np.percentile(self.residuals, upper_quantile * 100)

        # Add to predictions
        y_lower = predictions + lower_bound
        y_upper = predictions + upper_bound

        return y_lower, y_upper

    def save_predictions(self, timestamps, predictions, output_path, confidence_level=0.95):
        """Save predictions to CSV with optional prediction intervals"""
        results_df = pd.DataFrame({
            'valueDateTimeOffset': timestamps,
            'y_pred': predictions
        })

        # Add prediction intervals if available
        if self.residuals is not None:
            y_lower, y_upper = self.calculate_prediction_intervals(predictions, confidence_level)
            if y_lower is not None:
                results_df['y_lower'] = y_lower
                results_df['y_upper'] = y_upper
                results_df['confidence_level'] = confidence_level

        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Time Series Prediction')
    parser.add_argument('model_path', help='Path to trained model file')
    parser.add_argument('history_path', help='Path to historical data CSV')
    parser.add_argument('steps', type=int, choices=[12, 24, 96],
                       help='Number of steps to predict (12, 24, or 96)')
    parser.add_argument('--freq', type=str, help='Override frequency (e.g., "H", "D")')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    # Initialize predictor
    predictor = EnhancedTimeSeriesPredictor()

    # Load model
    predictor.load_model(args.model_path)

    # Override frequency if provided
    if args.freq:
        predictor.freq = args.freq

    # Load historical data
    history_df = predictor.load_history(args.history_path)

    # Make predictions
    timestamps, predictions = predictor.predict_future(history_df, args.steps)

    # Save results
    predictor.save_predictions(timestamps, predictions, args.output)

    print(f"\nPrediction completed!")
    print(f"Predicted {len(predictions)} future values")
    print(f"Results saved to {args.output}")

    # Display first few predictions
    print("\nFirst few predictions:")
    for i, (ts, pred) in enumerate(zip(timestamps[:5], predictions[:5])):
        print(f"  {ts}: {pred:.4f}")


if __name__ == '__main__':
    main()