# Data Pipeline Documentation

This document explains how to use `data.py` to build datasets for the time series forecasting model in `train_forecast.py`.

## Overview

The `data.py` script provides a framework for collecting data from multiple sources and combining them into a single dataset suitable for time series forecasting. All data sources use a **global date range** specified via `--start` and `--end` arguments.

## Current Status

⚠️ **No data sources are currently implemented.** You will need to implement your own data source loaders based on your project requirements.

## Quick Start

### Basic Usage

You must first implement at least one data source loader before using this script. See [Adding New Data Sources](#adding-new-data-sources) below.

```bash
# After implementing load_source_1(), fetch data for a date range
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 --output data/my_data.csv
```

### Common Options

```bash
# Required arguments:
--start YYYY-MM-DD    # Start date (inclusive)
--end YYYY-MM-DD      # End date (inclusive)
--sources N [N ...]   # List of data source numbers (e.g., 1 2 3)

# Optional arguments:
--output PATH         # Output CSV path (default: data/processed_data.csv)
--require-premium     # Validate that 'premium' column exists (for complete datasets)
```

## Output Format

The generated dataset follows this structure required by `train_forecast.py`:

| Column Name | Type | Description | Required |
|------------|------|-------------|----------|
| `valueDateTimeOffset` | datetime (UTC) | Timestamp column | ✅ Yes |
| `premium` | float | Target variable to forecast | Only with `--require-premium` |
| Additional columns | various | Feature columns from data sources | No |

### Example Output Structure

```csv
valueDateTimeOffset,feature1,feature2,premium
2025-01-01 00:00:00+00:00,23.5,100.2,1.45
2025-01-01 01:00:00+00:00,24.1,98.7,1.52
2025-01-01 02:00:00+00:00,22.8,102.3,1.38
...
```

## Adding New Data Sources

To add a new data source, follow these steps:

### 1. Implement the Loader Method

Add a method to the `DataBuilder` class in `data.py`:

```python
def load_source_1(self):
    """
    Load data from your first data source.

    Returns:
        pd.DataFrame with columns: valueDateTimeOffset and your feature columns
    """
    self.logger.info("=" * 60)
    self.logger.info("Loading Source 1: Your Data Source Name")
    self.logger.info("=" * 60)

    # Your data fetching logic here
    # Example: fetch data for self.start_date to self.end_date

    # Create DataFrame with your data
    df = pd.DataFrame({
        'timestamp': [...],  # Your timestamps
        'feature1': [...],
        'feature2': [...],
    })

    # Convert timestamps to UTC as valueDateTimeOffset
    df['valueDateTimeOffset'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.drop(columns=['timestamp'])

    # Ensure valueDateTimeOffset is first column
    cols = ['valueDateTimeOffset'] + [col for col in df.columns if col != 'valueDateTimeOffset']
    df = df[cols]

    self.logger.info(f"Source 1 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

    return df
```

### 2. Use Your Data Source

```bash
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 --output data/my_data.csv
```

### 3. Multiple Data Sources

You can implement multiple sources (e.g., `load_source_1`, `load_source_2`, `load_source_3`) and merge them:

```bash
# This will merge sources 1, 2, and 3 on valueDateTimeOffset
python data.py --start 2025-01-01 --end 2025-01-31 --sources 1 2 3 --output data/combined.csv
```

The sources will be merged using a left join on `valueDateTimeOffset` (and `settlementPeriod` if both sources have it).

## Data Processing Pipeline

The `data.py` script follows this processing pipeline:

```
1. Parse date range (--start and --end)
   ↓
2. Initialize DataBuilder with date range
   ↓
3. Load primary source (first number in --sources)
   ↓
4. [Optional] Load and merge additional sources
   - Merge on valueDateTimeOffset
   ↓
5. Clean data
   - Remove missing timestamps
   - Remove duplicates
   - Sort by timestamp
   ↓
6. Validate data
   - Check required columns exist
   - Check timestamp format (UTC)
   - Check for missing values
   ↓
7. Save to CSV
```

## Data Quality Checks

The script performs the following validation:

### Automatic Checks

1. ✅ **Timestamp validation**
   - Must be datetime type
   - Must be timezone-aware (UTC)
   - No missing timestamps

2. ✅ **Duplicate detection**
   - Warns if duplicates found
   - Automatically removes duplicates

3. ✅ **Column validation**
   - Checks required columns exist
   - Checks data types are correct

4. ✅ **Sorting**
   - Sorts by timestamp (ascending)

### Optional Checks (with `--require-premium`)

5. ✅ **Target variable validation**
   - Premium column exists
   - Premium column is numeric
   - No missing premium values

## Integration with train_forecast.py

Once you have a dataset with features and a target column:

```bash
# 1. Build the dataset
python data.py --start 2025-01-01 --end 2025-03-31 --sources 1 --output data/q1_data.csv --require-premium

# 2. Train the forecasting model
python train_forecast.py data/q1_data.csv --max-lag 96 --n-trials 100
```

## Command Line Reference

### Full Command Syntax

```bash
python data.py --start YYYY-MM-DD --end YYYY-MM-DD --sources N [N ...] [OPTIONS]
```

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--start` | string | Start date (format: YYYY-MM-DD, inclusive) |
| `--end` | string | End date (format: YYYY-MM-DD, inclusive) |
| `--sources` | int [int ...] | List of data source numbers to load |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | string | `data/processed_data.csv` | Output file path |
| `--require-premium` | flag | `False` | Validate that 'premium' column exists |

### Help

```bash
python data.py --help
```

## Example Implementation

Here's a complete example of implementing a simple CSV data source:

```python
def load_source_1(self):
    """
    Load data from a CSV file.

    Returns:
        pd.DataFrame with valueDateTimeOffset and feature columns
    """
    self.logger.info("=" * 60)
    self.logger.info("Loading Source 1: CSV Data")
    self.logger.info("=" * 60)

    # Read CSV file
    df = pd.read_csv('input_data.csv')

    # Convert timestamp column to UTC
    df['valueDateTimeOffset'] = pd.to_datetime(df['timestamp'], utc=True)

    # Filter to date range
    df = df[
        (df['valueDateTimeOffset'] >= pd.Timestamp(self.start_date, tz='UTC')) &
        (df['valueDateTimeOffset'] <= pd.Timestamp(self.end_date, tz='UTC'))
    ]

    # Select columns
    df = df[['valueDateTimeOffset', 'feature1', 'feature2', 'premium']]

    self.logger.info(f"Source 1 loaded: {len(df)} rows")

    return df
```

## Available Framework Methods

The `DataBuilder` class provides these helper methods:

- **`merge_sources(df1, df2, merge_type='left')`**: Merge two DataFrames on valueDateTimeOffset
- **`validate_data(df, require_premium=False)`**: Validate dataset structure and quality
- **`clean_data(df, require_premium=False)`**: Remove duplicates, sort, and clean data
- **`save_dataset(df, output_path, require_premium=False)`**: Save to CSV with validation

## Notes

- All timestamps must be in UTC timezone
- The first source specified in `--sources` is treated as the primary source
- Additional sources are merged using left joins
- Duplicate timestamps are automatically removed (keeps last occurrence)

---

**Last Updated:** 2025-10-20
