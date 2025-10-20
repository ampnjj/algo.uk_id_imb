#!/usr/bin/env python3
"""
Data Pipeline for Time Series Forecasting
Builds the dataset required by train_forecast.py
"""

import pandas as pd
import numpy as np
import polars as pl
from numba import njit
from datetime import datetime, timezone, timedelta
import argparse
from pathlib import Path
import requests
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ElexonSystemPricesFetcher:
    """
    Fetch imbalance settlement system prices from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date}?format=json
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon System Prices fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_system_prices(self, settlement_date):
        """
        Fetch system prices for a specific settlement date.

        Args:
            settlement_date: datetime.date object for the settlement date

        Returns:
            pd.DataFrame with columns: startTime, settlementDate, settlementPeriod, netImbalanceVolume
        """
        settlement_date_str = settlement_date.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching system prices for settlement date {settlement_date_str}")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build URL with settlement date
                url = f"{self.BASE_URL}/{settlement_date_str}"

                # Build request parameters
                params = {
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {settlement_date_str}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # Extract data array
                if 'data' not in data or not isinstance(data['data'], list):
                    self.logger.warning(f"No data array found in response for {settlement_date_str}")
                    return pd.DataFrame()

                records = data['data']

                if not records:
                    self.logger.warning(f"Empty data array for {settlement_date_str}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Check required columns exist
                required_cols = ['startTime', 'settlementDate', 'settlementPeriod', 'netImbalanceVolume']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {settlement_date_str}")
                    return pd.DataFrame()

                # Select required columns
                df_result = df[required_cols].copy()

                self.logger.info(f"Successfully fetched {len(df_result)} rows for {settlement_date_str}")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {settlement_date_str} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {settlement_date_str}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch system prices for a date range.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame with columns: startTime, settlementDate, settlementPeriod, netImbalanceVolume
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon System Prices data from {start_date} to {end_date}")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            df_day = self.fetch_system_prices(current_date)

            if not df_day.empty:
                all_dfs.append(df_day)

            current_date += timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any date in range")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod', 'netImbalanceVolume'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class ElexonDemandForecastFetcher:
    """
    Fetch day-ahead demand forecast data from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/history
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/history"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon Demand Forecast fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_for_settlement_period(self, start_time):
        """
        Fetch demand forecast for a specific settlement period.

        Args:
            start_time: datetime object for the settlement period start time (UTC)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod, publishTime,
                                      transmissionSystemDemand, nationalDemand
        """
        # Calculate publish time: 1 hour before startTime
        publish_time = start_time - timedelta(hours=1)
        publish_time_str = publish_time.strftime('%Y-%m-%dT%H:%M')

        self.logger.info(f"Fetching demand forecast for startTime {start_time.strftime('%Y-%m-%d %H:%M')} "
                        f"(publishTime={publish_time_str})")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'publishTime': publish_time_str,
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {start_time}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # Extract data array
                if 'data' not in data or not isinstance(data['data'], list):
                    self.logger.warning(f"No data array found in response for {start_time}")
                    return pd.DataFrame()

                records = data['data']

                if not records:
                    self.logger.warning(f"Empty data array for {start_time}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Convert startTime to datetime for filtering
                df['startTime'] = pd.to_datetime(df['startTime'], utc=True)

                # Filter to the exact startTime we're looking for
                df_filtered = df[df['startTime'] == start_time].copy()

                if df_filtered.empty:
                    self.logger.warning(f"No matching startTime {start_time} in response")
                    return pd.DataFrame()

                # Check required columns exist
                required_cols = ['startTime', 'settlementPeriod', 'publishTime',
                               'transmissionSystemDemand', 'nationalDemand']
                missing_cols = [col for col in required_cols if col not in df_filtered.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {start_time}")
                    return pd.DataFrame()

                # Select required columns
                df_result = df_filtered[required_cols].copy()

                # Take the first row if there are multiple matches (should be only one)
                if len(df_result) > 1:
                    self.logger.info(f"Multiple rows found for {start_time}, taking first")
                    df_result = df_result.iloc[:1]

                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {start_time} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {start_time}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch demand forecasts for a date range.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod, publishTime,
                                      transmissionSystemDemand, nationalDemand
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon Demand Forecast data from {start_date} to {end_date}")

        # Calculate total API calls
        num_days = (end_date - start_date).days + 1
        total_calls = num_days * 48  # 48 settlement periods per day
        self.logger.info(f"Expected API calls: {total_calls} ({num_days} days × 48 settlement periods)")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            # Generate all 48 settlement periods for this date
            # Settlement period 1 starts at 23:00 on previous day
            base_time = datetime.combine(current_date, datetime.min.time())
            base_time = base_time.replace(tzinfo=timezone.utc) - timedelta(hours=1)  # Start at 23:00 previous day

            for period in range(1, 49):  # Periods 1-48
                # Calculate start time for this settlement period (30-minute intervals)
                start_time = base_time + timedelta(minutes=(period - 1) * 30)

                # Fetch data for this settlement period
                df_period = self.fetch_for_settlement_period(start_time)

                if not df_period.empty:
                    all_dfs.append(df_period)

                # Small delay to avoid overwhelming the API
                time.sleep(0.1)

            current_date += timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any settlement period")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod', 'publishTime',
                                        'transmissionSystemDemand', 'nationalDemand'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod (should not be any)
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class ElexonDemandOutturnFetcher:
    """
    Fetch actual demand outturn data from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/demand/outturn/stream
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn/stream"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon Demand Outturn fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_date_batch(self, start_date, end_date):
        """
        Fetch demand outturn data for a date batch.

        Args:
            start_date: datetime.date object for the batch start date
            end_date: datetime.date object for the batch end date

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      initialDemandOutturn, initialTransmissionSystemDemandOutturn
        """
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching demand outturn for {start_date_str} to {end_date_str}")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'settlementDateFrom': start_date_str,
                    'settlementDateTo': end_date_str,
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=60)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {start_date_str} to {end_date_str}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # The streaming API returns a JSON array directly (not wrapped in {"data": []})
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and 'data' in data:
                    records = data['data']
                else:
                    self.logger.warning(f"Unexpected response format for {start_date_str} to {end_date_str}")
                    return pd.DataFrame()

                if not records:
                    self.logger.warning(f"Empty data array for {start_date_str} to {end_date_str}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Check required columns exist
                required_cols = ['startTime', 'settlementPeriod', 'initialDemandOutturn',
                               'initialTransmissionSystemDemandOutturn']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {start_date_str} to {end_date_str}")
                    return pd.DataFrame()

                # Select required columns
                df_result = df[required_cols].copy()

                self.logger.info(f"Successfully fetched {len(df_result)} rows for {start_date_str} to {end_date_str}")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {start_date_str} to {end_date_str} "
                                  f"(attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {start_date_str} to {end_date_str}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date, batch_size=10):
        """
        Fetch demand outturn data for a date range in batches.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)
            batch_size: Number of days to fetch per API call (default: 10)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      initialDemandOutturn, initialTransmissionSystemDemandOutturn
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon Demand Outturn data from {start_date} to {end_date}")

        # Calculate number of batches
        total_days = (end_date - start_date).days + 1
        num_batches = (total_days + batch_size - 1) // batch_size  # Ceiling division
        self.logger.info(f"Expected API calls: {num_batches} (fetching {total_days} days in batches of {batch_size})")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range in batches
        current_date = start_date
        while current_date <= end_date:
            # Calculate batch end date (min of batch_size days or remaining days)
            batch_end = min(current_date + timedelta(days=batch_size - 1), end_date)

            # Fetch data for this batch
            df_batch = self.fetch_date_batch(current_date, batch_end)

            if not df_batch.empty:
                all_dfs.append(df_batch)

            # Move to next batch
            current_date = batch_end + timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any batch")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod',
                                        'initialDemandOutturn', 'initialTransmissionSystemDemandOutturn'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined['startTime'] = pd.to_datetime(df_combined['startTime'], utc=True)
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class ElexonIndicatedImbalanceFetcher:
    """
    Fetch day-ahead indicated imbalance forecast data from Elexon BMRS API.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/forecast/indicated/day-ahead/history
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/indicated/day-ahead/history"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon Indicated Imbalance fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_for_settlement_period(self, start_time):
        """
        Fetch indicated imbalance forecast for a specific settlement period.

        Args:
            start_time: datetime object for the settlement period start time (UTC)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      indicatedGeneration, indicatedImbalance
        """
        # Calculate publish time: 1 hour before startTime
        publish_time = start_time - timedelta(hours=1)
        publish_time_str = publish_time.strftime('%Y-%m-%dT%H:%M')

        self.logger.info(f"Fetching indicated imbalance for startTime {start_time.strftime('%Y-%m-%d %H:%M')} "
                        f"(publishTime={publish_time_str})")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'publishTime': publish_time_str,
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {start_time}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # Extract data array
                if 'data' not in data or not isinstance(data['data'], list):
                    self.logger.warning(f"No data array found in response for {start_time}")
                    return pd.DataFrame()

                records = data['data']

                if not records:
                    self.logger.warning(f"Empty data array for {start_time}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Convert startTime to datetime for filtering
                df['startTime'] = pd.to_datetime(df['startTime'], utc=True)

                # Filter to the exact startTime we're looking for
                df_filtered = df[df['startTime'] == start_time].copy()

                if df_filtered.empty:
                    self.logger.warning(f"No matching startTime {start_time} in response")
                    return pd.DataFrame()

                # Check required columns exist
                required_cols = ['startTime', 'settlementPeriod', 'indicatedGeneration', 'indicatedImbalance']
                missing_cols = [col for col in required_cols if col not in df_filtered.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {start_time}")
                    return pd.DataFrame()

                # Select required columns
                df_result = df_filtered[required_cols].copy()

                # Take the first row if there are multiple matches (should be only one)
                if len(df_result) > 1:
                    self.logger.info(f"Multiple rows found for {start_time}, taking first")
                    df_result = df_result.iloc[:1]

                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {start_time} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {start_time}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch indicated imbalance forecasts for a date range.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)

        Returns:
            pd.DataFrame with columns: startTime, settlementPeriod,
                                      indicatedGeneration, indicatedImbalance
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon Indicated Imbalance data from {start_date} to {end_date}")

        # Calculate total API calls
        num_days = (end_date - start_date).days + 1
        total_calls = num_days * 48  # 48 settlement periods per day
        self.logger.info(f"Expected API calls: {total_calls} ({num_days} days × 48 settlement periods)")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range (inclusive)
        current_date = start_date
        while current_date <= end_date:
            # Generate all 48 settlement periods for this date
            # Settlement period 1 starts at 23:00 on previous day
            base_time = datetime.combine(current_date, datetime.min.time())
            base_time = base_time.replace(tzinfo=timezone.utc) - timedelta(hours=1)  # Start at 23:00 previous day

            for period in range(1, 49):  # Periods 1-48
                # Calculate start time for this settlement period (30-minute intervals)
                start_time = base_time + timedelta(minutes=(period - 1) * 30)

                # Fetch data for this settlement period
                df_period = self.fetch_for_settlement_period(start_time)

                if not df_period.empty:
                    all_dfs.append(df_period)

                # Small delay to avoid overwhelming the API
                time.sleep(0.1)

            current_date += timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any settlement period")
            return pd.DataFrame(columns=['startTime', 'settlementPeriod',
                                        'indicatedGeneration', 'indicatedImbalance'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Drop duplicates by startTime and settlementPeriod (should not be any)
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['startTime', 'settlementPeriod'], keep='last')
        duplicates_removed = initial_rows - len(df_combined)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Sort by startTime
        df_combined = df_combined.sort_values('startTime').reset_index(drop=True)

        self.logger.info(f"Total rows fetched: {len(df_combined)}")

        return df_combined


class ElexonDISBSADFetcher:
    """
    Fetch DISBSAD (Disaggregated Balancing Services Adjustment Data) from Elexon BMRS API.

    This source aggregates volume by settlementDate and settlementPeriod.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/datasets/DISBSAD/stream
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/DISBSAD/stream"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon DISBSAD fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_date_batch(self, start_date, end_date):
        """
        Fetch DISBSAD data for a date range (single API call).

        Args:
            start_date: datetime.date object for start date
            end_date: datetime.date object for end date

        Returns:
            pd.DataFrame with columns: settlementDate, settlementPeriod, volume
        """
        # Format dates as ISO 8601 with time
        start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

        start_str = start_datetime.strftime('%Y-%m-%dT%H:%MZ')
        end_str = end_datetime.strftime('%Y-%m-%dT%H:%MZ')

        self.logger.info(f"Fetching DISBSAD data for {start_date} to {end_date}")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'from': start_str,
                    'to': end_str
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=60)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed for {start_date} to {end_date}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # The response is a list of records
                if not isinstance(data, list):
                    self.logger.warning(f"Unexpected response format for {start_date} to {end_date}")
                    return pd.DataFrame()

                if not data:
                    self.logger.warning(f"Empty data array for {start_date} to {end_date}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(data)

                # Check required columns exist
                required_cols = ['settlementDate', 'settlementPeriod', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {start_date} to {end_date}")
                    return pd.DataFrame()

                # Select required columns
                df_result = df[required_cols].copy()

                self.logger.info(f"Successfully fetched {len(df_result)} rows for {start_date} to {end_date}")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {start_date} to {end_date} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {start_date} to {end_date}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date, batch_size=10):
        """
        Fetch DISBSAD data for a date range with batching and volume aggregation.

        Args:
            start_date: datetime.date or string (YYYY-MM-DD) for start date (inclusive)
            end_date: datetime.date or string (YYYY-MM-DD) for end date (inclusive)
            batch_size: Number of days to fetch per API call (default: 10)

        Returns:
            pd.DataFrame with columns: settlementDate, settlementPeriod, totalVolume
                                      (aggregated by settlementDate + settlementPeriod)
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon DISBSAD data from {start_date} to {end_date}")

        # Calculate total API calls
        num_days = (end_date - start_date).days + 1
        num_batches = (num_days + batch_size - 1) // batch_size  # Ceiling division
        self.logger.info(f"Expected API calls: {num_batches} (fetching {num_days} days in batches of {batch_size})")

        # Collect all DataFrames
        all_dfs = []

        # Iterate over date range in batches
        current_date = start_date
        while current_date <= end_date:
            batch_end = min(current_date + timedelta(days=batch_size - 1), end_date)

            # Fetch batch
            df_batch = self.fetch_date_batch(current_date, batch_end)

            if not df_batch.empty:
                all_dfs.append(df_batch)

            # Small delay to avoid overwhelming the API
            time.sleep(0.5)

            current_date = batch_end + timedelta(days=1)

        # Concatenate all results
        if not all_dfs:
            self.logger.warning("No data fetched for any batch")
            return pd.DataFrame(columns=['settlementDate', 'settlementPeriod', 'totalVolume'])

        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Aggregate volume by settlementDate + settlementPeriod
        self.logger.info(f"Aggregating {len(df_combined)} rows by settlementDate + settlementPeriod...")
        df_aggregated = df_combined.groupby(['settlementDate', 'settlementPeriod'], as_index=False)['volume'].sum()
        df_aggregated = df_aggregated.rename(columns={'volume': 'totalVolume'})

        # Sort by settlementDate and settlementPeriod
        df_aggregated = df_aggregated.sort_values(['settlementDate', 'settlementPeriod']).reset_index(drop=True)

        self.logger.info(f"Total aggregated rows: {len(df_aggregated)}")

        return df_aggregated


class BmUnitsReferenceFetcher:
    """
    Fetch BM Units (Balancing Mechanism Units) reference data from Elexon BMRS API.

    This is a static reference table, not time-series data.
    Used as a lookup table for enriching other data sources.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/reference/bmunits/all
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/reference/bmunits/all"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the BM Units reference fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_bmunits(self):
        """
        Fetch all BM Units reference data.

        Returns:
            pd.DataFrame with columns: nationalGridBmUnit, bmUnitType
        """
        self.logger.info("Fetching BM Units reference data from Elexon BMRS API...")

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    'format': 'json'
                }

                # Make GET request
                response = requests.get(self.BASE_URL, params=params, timeout=60)
                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except ValueError as e:
                    self.logger.warning(f"JSON parsing failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                    return pd.DataFrame()

                # The response might be a list or have a 'data' key
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and 'data' in data:
                    records = data['data']
                else:
                    self.logger.warning(f"Unexpected response format: {type(data)}")
                    return pd.DataFrame()

                if not records:
                    self.logger.warning("Empty data array in response")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Check required columns exist
                required_cols = ['nationalGridBmUnit', 'bmUnitType']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} in response")
                    # Show available columns for debugging
                    self.logger.info(f"Available columns: {list(df.columns)}")
                    return pd.DataFrame()

                # Select only required columns
                df_result = df[required_cols].copy()

                # Remove duplicates (keep first occurrence)
                initial_rows = len(df_result)
                df_result = df_result.drop_duplicates(subset=['nationalGridBmUnit'], keep='first')
                duplicates_removed = initial_rows - len(df_result)

                if duplicates_removed > 0:
                    self.logger.info(f"Removed {duplicates_removed} duplicate BM Units")

                # Sort by nationalGridBmUnit for consistency
                df_result = df_result.sort_values('nationalGridBmUnit').reset_index(drop=True)

                self.logger.info(f"Successfully fetched {len(df_result)} BM Units")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("All retry attempts failed")
                    return pd.DataFrame()

        return pd.DataFrame()


# ==============================================================================
# FPN (Physical Notification) Processing Functions with Numba Acceleration
# ==============================================================================

def build_schedule_dataframe_numba(schedule_df, full_index):
    """
    Build full 1-minute schedules for all BMUs using Numba-accelerated interpolation.
    Accurate linear interpolation, ~20-50x faster than pandas loops.

    Args:
        schedule_df: DataFrame with columns ['bmu', 'timeFrom', 'timeTo', 'levelFrom', 'levelTo']
        full_index: DatetimeIndex for the full time range

    Returns:
        DataFrame with 1-minute resolution, one column per BMU
    """
    # Prepare lookup tables
    n_times = len(full_index)
    time_to_idx = pd.Series(np.arange(n_times), index=full_index)

    bmus = schedule_df['bmu'].unique()
    n_bmus = len(bmus)
    bmu_to_idx = {bmu: i for i, bmu in enumerate(bmus)}

    # Initialize result matrix
    result = np.full((n_times, n_bmus), np.nan, dtype=np.float64)

    # Convert columns to numpy arrays for speed
    bmu_idx_arr = schedule_df['bmu'].map(bmu_to_idx).to_numpy()
    start_idx = schedule_df['timeFrom'].map(time_to_idx).to_numpy()
    end_idx = schedule_df['timeTo'].map(time_to_idx).to_numpy()
    level_from = schedule_df['levelFrom'].to_numpy(dtype=np.float64)
    level_to = schedule_df['levelTo'].to_numpy(dtype=np.float64)

    # Numba-accelerated fill
    _fill_segments_numba(result, bmu_idx_arr, start_idx, end_idx, level_from, level_to)

    # Create DataFrame with named index for consistency
    df = pd.DataFrame(result, index=full_index, columns=bmus)
    df.index.name = 'startTime'
    return df


@njit
def _fill_segments_numba(result, bmu_idx_arr, start_idx, end_idx, level_from, level_to):
    """
    Vectorized interpolation filling, JIT-compiled.
    This is where the 20-50x speedup comes from.
    """
    for i in range(len(bmu_idx_arr)):
        bi = bmu_idx_arr[i]
        s, e = start_idx[i], end_idx[i]
        if np.isnan(s) or np.isnan(e):
            continue
        s = int(s)
        e = int(e)
        n = e - s + 1
        if n <= 0:
            continue

        lf = level_from[i]
        lt = level_to[i]

        if n == 1:
            result[s, bi] = lf
        else:
            step = (lt - lf) / (n - 1)
            for j in range(n):
                result[s + j, bi] = lf + step * j


def process_fpn_to_30min(bmu_data_pd, pn_data_pd):
    """
    Process FPN data: interpolate to 1-min resolution, then aggregate to 30-min.

    This is adapted from prepare_fpn_pl() but simplified for this use case:
    - Returns only production data (prepared_prod_pl equivalent)
    - Aggregates all BMU columns into totalProduction
    - Resamples to 30-min intervals
    - Returns DataFrame with valueDateTimeOffset, totalProduction (no settlementPeriod)

    Args:
        bmu_data_pd: pandas DataFrame with BMU reference (nationalGridBmUnit, bmUnitType)
        pn_data_pd: pandas DataFrame with PN data

    Returns:
        pandas DataFrame with columns: valueDateTimeOffset, totalProduction
    """
    try:
        if bmu_data_pd.empty or pn_data_pd.empty:
            logger.warning("process_fpn_to_30min: Missing data")
            return pd.DataFrame()

        # Merge PN data with BMU reference
        schedule = pn_data_pd.copy()
        schedule = pd.merge(schedule, bmu_data_pd, on="nationalGridBmUnit", how="left")
        schedule = schedule.drop_duplicates()
        schedule['bmUnitType'] = schedule['bmUnitType'].fillna('M')

        # Keep required columns
        columns_to_keep = ['timeFrom', 'timeTo', 'levelFrom', 'levelTo',
                          'nationalGridBmUnit', 'bmUnitType']
        schedule = schedule[columns_to_keep]
        schedule = schedule.rename(columns={'nationalGridBmUnit': 'bmu'})
        schedule = schedule.dropna(subset=['bmu'])

        # Convert timestamps
        schedule['timeFrom'] = pd.to_datetime(schedule['timeFrom'], utc=True, errors='coerce')
        schedule['timeTo'] = pd.to_datetime(schedule['timeTo'], utc=True, errors='coerce') - pd.Timedelta(minutes=1)

        global_start = schedule['timeFrom'].min()
        global_end = schedule['timeTo'].max()
        full_index = pd.date_range(global_start, global_end, freq='1min', tz='UTC')

        # Filter for production units only (T, E, I, V, M)
        production_prefixes = ('T', 'E', 'I', 'V', 'M')
        prod_schedule = schedule[schedule['bmUnitType'].str.startswith(production_prefixes)]

        # Build 1-minute interpolated schedule using Numba
        logger.info("Building 1-minute interpolated schedule (this may take time)...")
        prepared_prod_pd = build_schedule_dataframe_numba(prod_schedule, full_index)

        if prepared_prod_pd.empty:
            return pl.DataFrame()

        # Reset index to make startTime a column
        prepared_prod_pd = prepared_prod_pd.reset_index()

        # Aggregate all BMU columns into totalProduction
        bmu_columns = [col for col in prepared_prod_pd.columns if col != 'startTime']
        prepared_prod_pd['totalProduction'] = prepared_prod_pd[bmu_columns].sum(axis=1, skipna=True)

        # Keep only startTime and totalProduction
        prepared_prod_pd = prepared_prod_pd[['startTime', 'totalProduction']]

        # Resample to 30-minute intervals (mean)
        logger.info("Resampling to 30-minute intervals...")
        prepared_prod_pd = prepared_prod_pd.set_index('startTime')
        resampled_pd = prepared_prod_pd.resample('30min').mean()
        resampled_pd = resampled_pd.reset_index()
        resampled_pd = resampled_pd.rename(columns={'startTime': 'valueDateTimeOffset'})

        # Note: We don't add settlementPeriod here because:
        # 1. Settlement period calculation is complex (period 1 starts at 23:00 previous day)
        # 2. The merge logic will use valueDateTimeOffset only if settlementPeriod is missing
        # 3. This avoids mismatches with other sources that use API-provided settlementPeriod

        # Return pandas DataFrame (keep compatible with existing infrastructure)
        logger.info(f"FPN processing complete: {len(resampled_pd)} 30-min intervals")
        return resampled_pd

    except Exception as e:
        logger.warning(f"Error in process_fpn_to_30min: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return pd.DataFrame()


class ElexonPNFetcher:
    """
    Fetch Physical Notification (PN) data from Elexon BMRS API.

    This is a heavy endpoint with many records per settlement period.
    Returns raw PN data for FPN processing.

    API Endpoint: https://data.elexon.co.uk/bmrs/api/v1/datasets/PN/stream
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/PN/stream"

    def __init__(self, max_retries=3, backoff_factor=1):
        """
        Initialize the Elexon PN fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Base factor for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def fetch_pn_day(self, date):
        """
        Fetch PN data for a single day.

        Note: The Elexon API requires querying across day boundaries to get complete data.
        We fetch from 'date' to 'date+1' and filter by settlementDate to get all 48 periods.

        Args:
            date: datetime.date object

        Returns:
            pd.DataFrame with columns: timeFrom, timeTo, levelFrom, levelTo,
                                      nationalGridBmUnit, settlementDate, settlementPeriod
        """
        # Format date as YYYY-MM-DD
        date_str = date.strftime('%Y-%m-%d')

        # Add 1-day buffer to get complete data (settlement period 1 starts at 23:00 previous day)
        next_date = date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching PN data for {date_str} (querying to {next_date_str} for complete coverage)")

        for attempt in range(self.max_retries):
            try:
                params = {
                    'from': date_str,
                    'to': next_date_str,  # Query next day to get all 48 periods
                    'format': 'json'
                }

                response = requests.get(self.BASE_URL, params=params, timeout=120)
                response.raise_for_status()

                data = response.json()

                if not isinstance(data, list):
                    self.logger.warning(f"Unexpected response format for {date_str}")
                    return pd.DataFrame()

                if not data:
                    self.logger.warning(f"Empty data for {date_str}")
                    return pd.DataFrame()

                df = pd.DataFrame(data)

                required_cols = ['timeFrom', 'timeTo', 'levelFrom', 'levelTo',
                                'nationalGridBmUnit', 'settlementDate', 'settlementPeriod']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols} for {date_str}")
                    return pd.DataFrame()

                df_result = df[required_cols].copy()

                # Filter to only include records for the requested settlementDate
                df_result = df_result[df_result['settlementDate'] == date_str]

                self.logger.info(f"Successfully fetched {len(df_result)} PN records for {date_str} (all 48 periods)")
                return df_result

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for {date_str} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed for {date_str}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_date_range(self, start_date, end_date):
        """
        Fetch PN data for a date range (1 day at a time).

        Args:
            start_date: datetime.date or string (YYYY-MM-DD)
            end_date: datetime.date or string (YYYY-MM-DD)

        Returns:
            pd.DataFrame with all PN records for the date range
        """
        # Convert strings to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.logger.info(f"Fetching Elexon PN data from {start_date} to {end_date}")

        num_days = (end_date - start_date).days + 1
        self.logger.info(f"Expected API calls: {num_days} (1 call per day)")

        all_dfs = []
        current_date = start_date

        while current_date <= end_date:
            df_day = self.fetch_pn_day(current_date)

            if not df_day.empty:
                all_dfs.append(df_day)

            # Small delay between requests
            time.sleep(0.5)
            current_date += timedelta(days=1)

        if not all_dfs:
            self.logger.warning("No PN data fetched for any day")
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"Total PN records fetched: {len(df_combined)}")

        return df_combined


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

    def load_source_1(self):
        """
        Load data from Elexon BMRS System Prices API (Source 1).

        Fetches net imbalance volume data for the date range specified
        in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementDate, settlementPeriod, netImbalanceVolume
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 1: Elexon BMRS System Prices (Net Imbalance Volume)")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonSystemPricesFetcher()

        # Fetch data for date range
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon System Prices API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementDate', 'settlementPeriod', 'netImbalanceVolume']]

        self.logger.info(f"Source 1 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def load_source_2(self):
        """
        Load data from Elexon BMRS Demand Forecast API (Source 2).

        Fetches day-ahead demand forecasts (TSDF and NDF) for the date range specified
        in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod, publishTime,
                                      transmissionSystemDemand, nationalDemand
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 2: Elexon BMRS Day-Ahead Demand Forecast (TSDF & NDF)")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonDemandForecastFetcher()

        # Fetch data for date range
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon Demand Forecast API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Also convert publishTime to datetime for consistency
        df['publishTime'] = pd.to_datetime(df['publishTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementPeriod', 'publishTime',
                'transmissionSystemDemand', 'nationalDemand']]

        self.logger.info(f"Source 2 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def load_source_3(self):
        """
        Load data from Elexon BMRS Demand Outturn API (Source 3).

        Fetches actual demand outturn data for the date range specified
        in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod,
                                      initialDemandOutturn, initialTransmissionSystemDemandOutturn
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 3: Elexon BMRS Demand Outturn (Actual Demand)")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonDemandOutturnFetcher()

        # Fetch data for date range (with 10-day batching)
        df = fetcher.fetch_date_range(self.start_date, self.end_date, batch_size=10)

        if df.empty:
            raise ValueError("No data fetched from Elexon Demand Outturn API")

        # startTime is already converted to datetime in fetch_date_range
        # Convert to valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = df['startTime']

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementPeriod',
                'initialDemandOutturn', 'initialTransmissionSystemDemandOutturn']]

        self.logger.info(f"Source 3 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def load_source_4(self):
        """
        Load data from Elexon BMRS Indicated Imbalance API (Source 4).

        Fetches day-ahead indicated imbalance forecasts for the date range specified
        in __init__ (self.start_date to self.end_date).

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, settlementPeriod,
                                      indicatedGeneration, indicatedImbalance
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 4: Elexon BMRS Indicated Imbalance Forecast")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonIndicatedImbalanceFetcher()

        # Fetch data for date range
        df = fetcher.fetch_date_range(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("No data fetched from Elexon Indicated Imbalance API")

        # Convert startTime to UTC datetime as valueDateTimeOffset
        self.logger.info("Converting startTime to valueDateTimeOffset (UTC datetime)...")
        df['valueDateTimeOffset'] = pd.to_datetime(df['startTime'], utc=True)

        # Drop the original startTime column (keep valueDateTimeOffset)
        df = df.drop(columns=['startTime'])

        # Reorder columns: valueDateTimeOffset first, then features
        df = df[['valueDateTimeOffset', 'settlementPeriod', 'indicatedGeneration', 'indicatedImbalance']]

        self.logger.info(f"Source 4 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

    def load_source_5(self):
        """
        Load data from Elexon BMRS DISBSAD API (Source 5).

        Fetches Disaggregated Balancing Services Adjustment Data for the date range
        specified in __init__ (self.start_date to self.end_date).

        Volume is aggregated (summed) by settlementDate + settlementPeriod.

        Returns:
            pd.DataFrame with columns: settlementDate, settlementPeriod, totalVolume
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 5: Elexon BMRS DISBSAD (Aggregated Volume)")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = ElexonDISBSADFetcher()

        # Fetch data for date range (with 10-day batching and aggregation)
        df = fetcher.fetch_date_range(self.start_date, self.end_date, batch_size=10)

        if df.empty:
            raise ValueError("No data fetched from Elexon DISBSAD API")

        self.logger.info(f"Source 5 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Settlement date range: {df['settlementDate'].min()} to {df['settlementDate'].max()}")

        return df

    def load_source_6(self):
        """
        Load BM Units reference data from Elexon BMRS API (Source 6).

        This is a REFERENCE TABLE, not time-series data.
        Returns a static lookup table with BM Unit metadata.

        NOTE: This source is NOT merged with time-series data (Sources 1-5).
              It is saved separately as a reference file for use in Source 7.

        Returns:
            pd.DataFrame with columns: nationalGridBmUnit, bmUnitType
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 6: BM Units Reference Data")
        self.logger.info("=" * 60)

        # Initialize fetcher
        fetcher = BmUnitsReferenceFetcher()

        # Fetch BM Units reference data
        df = fetcher.fetch_bmunits()

        if df.empty:
            raise ValueError("No data fetched from BM Units Reference API")

        self.logger.info(f"Source 6 loaded successfully: {len(df)} BM Units, {len(df.columns)} columns")
        self.logger.info(f"Sample BM Units: {df['nationalGridBmUnit'].head(5).tolist()}")

        return df

    def load_source_7(self):
        """
        Load Physical Notification (PN) data from Elexon BMRS API (Source 7).

        Fetches PN data, processes FPN with 1-minute interpolation,
        then aggregates to 30-minute intervals for merging with other sources.

        Requires: Source 6 (BMU reference) must be downloaded first.

        Returns:
            pd.DataFrame with columns: valueDateTimeOffset, totalProduction
            Note: settlementPeriod is NOT included to avoid merge conflicts
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading Source 7: Physical Notification (FPN) Data")
        self.logger.info("=" * 60)

        # Load BMU reference from Source 6
        bmu_path = Path("data/reference/bmunits.csv")
        if not bmu_path.exists():
            raise ValueError(
                "BMU reference data not found. Please download Source 6 first:\n"
                "python data.py --start 2025-01-01 --end 2025-01-01 --sources 6 --output data/reference/bmunits.csv"
            )

        self.logger.info("Loading BMU reference data...")
        bmu_data_pd = pd.read_csv(bmu_path)
        self.logger.info(f"Loaded {len(bmu_data_pd)} BMU units")

        # Fetch PN data
        fetcher = ElexonPNFetcher()
        pn_data_pd = fetcher.fetch_date_range(self.start_date, self.end_date)

        if pn_data_pd.empty:
            raise ValueError("No PN data fetched from Elexon API")

        # Process FPN data: interpolate to 1-min, aggregate to 30-min
        self.logger.info("Processing FPN data (interpolation + aggregation)...")
        df = process_fpn_to_30min(bmu_data_pd, pn_data_pd)

        if df.empty:
            raise ValueError("FPN processing returned empty DataFrame")

        self.logger.info(f"Source 7 loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['valueDateTimeOffset'].min()} to {df['valueDateTimeOffset'].max()}")

        return df

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

        # Check if df2 is Source 5 (DISBSAD) which uses settlementDate instead of valueDateTimeOffset
        if 'valueDateTimeOffset' not in df2.columns and 'settlementDate' in df2.columns:
            # Source 5: merge on settlementDate + settlementPeriod
            join_keys = ['settlementDate', 'settlementPeriod']
            self.logger.info(f"Merging on: {join_keys} (Source 5 detected)")

            # Merge on join keys
            merged_df = pd.merge(
                df1,
                df2,
                on=join_keys,
                how=merge_type,
                suffixes=('', '_source2')
            )
        else:
            # Standard merge: use valueDateTimeOffset + settlementPeriod
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

        # Determine which timestamp columns to check
        if 'valueDateTimeOffset' in df.columns:
            # Standard dataset with valueDateTimeOffset
            required_cols = ['valueDateTimeOffset']
            timestamp_col = 'valueDateTimeOffset'
            duplicate_check_cols = ['valueDateTimeOffset']
        elif 'settlementDate' in df.columns and 'settlementPeriod' in df.columns:
            # Source 5 only dataset
            required_cols = ['settlementDate', 'settlementPeriod']
            timestamp_col = None
            duplicate_check_cols = ['settlementDate', 'settlementPeriod']
        else:
            raise ValueError("DataFrame must have either 'valueDateTimeOffset' or 'settlementDate'+'settlementPeriod'")

        if require_premium:
            required_cols.append('premium')

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check timestamp format (only for valueDateTimeOffset)
        if timestamp_col:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                raise ValueError(f"{timestamp_col} must be datetime type")

            # Check for timezone awareness
            if df[timestamp_col].dt.tz is None:
                raise ValueError(f"{timestamp_col} must be timezone-aware (UTC)")

        # Check premium is numeric (only if required)
        if require_premium:
            if not pd.api.types.is_numeric_dtype(df['premium']):
                raise ValueError("premium must be numeric type")

        # Check for duplicates
        duplicates = df.duplicated(subset=duplicate_check_cols).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate timestamps")

        # Check for missing values
        if timestamp_col:
            missing_timestamp = df[timestamp_col].isna().sum()
            if missing_timestamp > 0:
                self.logger.warning(f"Found {missing_timestamp} missing timestamps")
        else:
            missing_date = df['settlementDate'].isna().sum()
            missing_period = df['settlementPeriod'].isna().sum()
            if missing_date > 0 or missing_period > 0:
                self.logger.warning(f"Found {missing_date} missing settlementDates, {missing_period} missing settlementPeriods")

        if require_premium:
            missing_premium = df['premium'].isna().sum()
            if missing_premium > 0:
                self.logger.warning(f"Found {missing_premium} missing premium values")

        self.logger.info("✓ Dataset validation passed")
        self.logger.info(f"  - Shape: {df.shape}")

        if timestamp_col:
            self.logger.info(f"  - Time range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
        else:
            self.logger.info(f"  - Settlement date range: {df['settlementDate'].min()} to {df['settlementDate'].max()}")

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

        # Determine which timestamp column to use
        if 'valueDateTimeOffset' in df.columns:
            timestamp_col = 'valueDateTimeOffset'
            sort_cols = ['valueDateTimeOffset']
        elif 'settlementDate' in df.columns and 'settlementPeriod' in df.columns:
            # Source 5 only - use settlementDate + settlementPeriod
            timestamp_col = None
            sort_cols = ['settlementDate', 'settlementPeriod']
        else:
            raise ValueError("DataFrame must have either 'valueDateTimeOffset' or 'settlementDate'+'settlementPeriod'")

        # Remove rows with missing timestamps
        drop_cols = []
        if timestamp_col:
            drop_cols.append(timestamp_col)
        else:
            drop_cols.extend(['settlementDate', 'settlementPeriod'])

        if require_premium and 'premium' in df.columns:
            drop_cols.append('premium')

        df = df.dropna(subset=drop_cols)

        # Remove duplicate timestamps (keep last)
        if timestamp_col:
            df = df.drop_duplicates(subset=[timestamp_col], keep='last')
        else:
            df = df.drop_duplicates(subset=['settlementDate', 'settlementPeriod'], keep='last')

        # Sort by timestamp
        df = df.sort_values(sort_cols).reset_index(drop=True)

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
                         Special: Source 6 is a reference table and is handled separately
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

        # Special handling for Source 6 (BM Units reference data)
        # Source 6 is a reference table, not time-series data, so handle it separately
        if load_sources == [6]:
            self.logger.info("\nLoading Source 6 (Reference Data Only)...")
            loader_method = getattr(self, "load_source_6", None)

            if loader_method is None:
                raise ValueError("Source 6 not implemented.")

            df = loader_method()

            # For Source 6, skip cleaning/validation and just save directly
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            self.logger.info(f"✓ Reference data saved to {output_path}")
            self.logger.info(f"  - {len(df)} BM Units, {len(df.columns)} columns")
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Reference data download completed successfully!")
            self.logger.info("=" * 60)

            self.df = df
            return df

        # Filter out Source 6 from regular time-series processing
        timeseries_sources = [s for s in load_sources if s != 6]

        if not timeseries_sources:
            raise ValueError("No time-series sources specified (only Source 6 was requested)")

        # Load first time-series source as primary
        first_source = timeseries_sources[0]
        loader_method = getattr(self, f"load_source_{first_source}", None)

        if loader_method is None:
            raise ValueError(
                f"Source {first_source} not implemented. "
                f"Please implement load_source_{first_source}() method in DataBuilder class."
            )

        self.logger.info(f"\nLoading primary source {first_source}...")
        df = loader_method()

        # Load and merge additional time-series sources if specified
        for source_num in timeseries_sources[1:]:
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
