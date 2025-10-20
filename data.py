#!/usr/bin/env python3
"""
Data Pipeline for Time Series Forecasting
Builds the dataset required by train_forecast.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DataBuilder:
    """
    Build and prepare time series data for forecasting.

    Output format required:
    - valueDateTimeOffset: UTC timestamp column
    - Additional feature columns
    - Optionally: premium (target variable)
    """

    def __init__(self, start_date, end_date):
        """
        Initialize DataBuilder with global date range.

        Args:
            start_date: Start date for data collection (YYYY-MM-DD or datetime.date)
            end_date: End date for data collection (YYYY-MM-DD or datetime.date)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.sources = []
        self.logger = logging.getLogger(__name__)

    def merge_sources(self, df1, df2, merge_type='left'):
        """
        Merge multiple data sources on timestamp and settlement period.

        Args:
            df1: Primary dataframe with valueDateTimeOffset
            df2: Secondary dataframe with additional features
            merge_type: Type of merge ('left', 'inner', 'outer')

        Returns:
            pd.DataFrame: Merged dataset
        """
        self.logger.info(f"Merging data sources with {merge_type} join...")

        # Determine join keys (use both timestamp and settlementPeriod if available)
        join_keys = ['valueDateTimeOffset']
        if 'settlementPeriod' in df1.columns and 'settlementPeriod' in df2.columns:
            join_keys.append('settlementPeriod')
            self.logger.info(f"Merging on: {join_keys}")

        # Merge on join keys
        merged_df = pd.merge(
            df1,
            df2,
            on=join_keys,
            how=merge_type,
            suffixes=('', '_source2')
        )

        self.logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df

    def validate_data(self, df, require_premium=False):
        """
        Validate the dataset has required columns and proper format.

        Args:
            df: DataFrame to validate
            require_premium: If True, validate that 'premium' column exists and is numeric.
                           If False, only validate valueDateTimeOffset (features-only mode)

        Returns:
            bool: True if valid, raises error otherwise
        """
        self.logger.info("Validating dataset...")

        # Check required timestamp column
        required_cols = ['valueDateTimeOffset']
        if require_premium:
            required_cols.append('premium')

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check timestamp format
        if not pd.api.types.is_datetime64_any_dtype(df['valueDateTimeOffset']):
            raise ValueError("valueDateTimeOffset must be datetime type")

        # Check for timezone awareness
        if df['valueDateTimeOffset'].dt.tz is None:
            raise ValueError("valueDateTimeOffset must be timezone-aware (UTC)")

        # Check premium is numeric (only if required)
        if require_premium:
            if not pd.api.types.is_numeric_dtype(df['premium']):
                raise ValueError("premium must be numeric type")

        # Check for duplicates
        duplicates = df.duplicated(subset=['valueDateTimeOffset']).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate timestamps")

        # Check for missing values
        missing_timestamp = df['valueDateTimeOffset'].isna().sum()

        if missing_timestamp > 0:
            self.logger.warning(f"Found {missing_timestamp} missing timestamps")

        if require_premium:
            missing_premium = df['premium'].isna().sum()
            if missing_premium > 0:
                self.logger.warning(f"Found {missing_premium} missing premium values")

        self.logger.info("✓ Dataset validation passed")
        self.logger.info(f"  - Shape: {df.shape}")
        self.logger.info(f"  - Time range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")
        self.logger.info(f"  - Columns: {list(df.columns)}")

        return True

    def clean_data(self, df, require_premium=False):
        """
        Clean the dataset before saving.

        Args:
            df: DataFrame to clean
            require_premium: If True, drop rows with missing premium values.
                           If False, only drop rows with missing timestamps.

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        self.logger.info("Cleaning dataset...")

        initial_rows = len(df)

        # Remove rows with missing timestamps
        drop_cols = ['valueDateTimeOffset']
        if require_premium and 'premium' in df.columns:
            drop_cols.append('premium')

        df = df.dropna(subset=drop_cols)

        # Remove duplicate timestamps (keep last)
        df = df.drop_duplicates(subset=['valueDateTimeOffset'], keep='last')

        # Sort by timestamp
        df = df.sort_values('valueDateTimeOffset').reset_index(drop=True)

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"  - Removed {removed_rows} rows during cleaning")

        self.logger.info(f"  - Final shape: {df.shape}")

        return df

    def save_dataset(self, df, output_path, require_premium=False):
        """
        Save the final dataset to CSV.

        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
            require_premium: If True, validate that premium column exists
        """
        # Validate before saving
        self.validate_data(df, require_premium=require_premium)

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        self.logger.info(f"✓ Dataset saved to {output_path}")
        self.logger.info(f"  - {len(df)} rows, {len(df.columns)} columns")

    def build_dataset(self, output_path, load_sources=None, require_premium=False):
        """
        Main pipeline to build the dataset from multiple sources.

        Args:
            output_path: Path to save the final CSV
            load_sources: List of source numbers to load (e.g., [1, 2])
                         Note: You need to implement load_source_N methods for your data sources
            require_premium: If True, validate that premium column exists in final dataset

        Returns:
            pd.DataFrame: Final dataset
        """
        self.logger.info("=" * 60)
        self.logger.info("Building Dataset for Time Series Forecasting")
        self.logger.info("=" * 60)
        self.logger.info(f"Date range: {self.start_date} to {self.end_date}")

        if not load_sources:
            raise ValueError(
                "No data sources specified. Please implement load_source_N methods "
                "and pass the source numbers via --sources argument."
            )

        # Load first source as primary
        first_source = load_sources[0]
        loader_method = getattr(self, f"load_source_{first_source}", None)

        if loader_method is None:
            raise ValueError(
                f"Source {first_source} not implemented. "
                f"Please implement load_source_{first_source}() method in DataBuilder class."
            )

        self.logger.info(f"\nLoading primary source {first_source}...")
        df = loader_method()

        # Load and merge additional sources if specified
        for source_num in load_sources[1:]:
            self.logger.info(f"\nLoading source {source_num}...")
            loader_method = getattr(self, f"load_source_{source_num}", None)

            if loader_method is None:
                self.logger.warning(f"Source {source_num} not implemented, skipping...")
                continue

            df_additional = loader_method()
            df = self.merge_sources(df, df_additional, merge_type='left')

        # Clean the data
        df = self.clean_data(df, require_premium=require_premium)

        # Save the final dataset
        self.save_dataset(df, output_path, require_premium=require_premium)

        self.df = df
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Build dataset for time series forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Note: You must implement load_source_N methods in the DataBuilder class before using this script

  # Fetch data from source 1 for a week
  python data.py --start 2025-01-01 --end 2025-01-07 --sources 1 --output data/my_data.csv

  # Fetch data for multiple sources
  python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 --output data/combined_data.csv

  # Require premium column validation (for complete datasets)
  python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 --output data/full_data.csv --require-premium
        """
    )

    # Required arguments
    parser.add_argument('--start', type=str, required=True,
                       help='Start date for data collection (format: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='End date for data collection (format: YYYY-MM-DD, inclusive)')

    # Optional arguments
    parser.add_argument('--output', type=str, default='data/processed_data.csv',
                       help='Path to save the processed dataset (default: data/processed_data.csv)')
    parser.add_argument('--sources', type=int, nargs='+', required=True,
                       help='List of data sources to load. Example: --sources 1 2 (You must implement load_source_N methods)')
    parser.add_argument('--require-premium', action='store_true',
                       help='Validate that premium column exists in final dataset')

    args = parser.parse_args()

    try:
        # Parse dates
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

        # Validate date range
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")

        # Initialize builder with date range
        builder = DataBuilder(start_date, end_date)

        # Build dataset
        df = builder.build_dataset(
            output_path=args.output,
            load_sources=args.sources,
            require_premium=args.require_premium
        )

        logging.info("\n" + "=" * 60)
        logging.info("Dataset building completed successfully!")
        logging.info("=" * 60)
        logging.info(f"\nDataset saved to: {args.output}")
        logging.info(f"Total rows: {len(df)}")
        logging.info(f"Columns: {list(df.columns)}")

        if not args.require_premium:
            logging.info("\nNote: This is a features-only dataset (no premium/target column).")
            logging.info("To train a model, you'll need to add a 'premium' column as the target variable.")
        else:
            logging.info(f"\nNext step: Train the model using:")
            logging.info(f"  python train_forecast.py {args.output}")

    except ValueError as e:
        logging.error(f"Error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
