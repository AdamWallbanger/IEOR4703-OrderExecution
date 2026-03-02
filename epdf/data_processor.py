"""
Data Processor Module

Handles data loading, cleaning, resampling, and range computation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings


class DataProcessor:
    """
    Data preprocessing pipeline for OHLCV data.

    Responsibilities:
    - Load raw 1-minute OHLCV data
    - Filter sparse trading days
    - Resample to τ-minute bars
    - Compute tick-normalized ranges
    """

    @staticmethod
    def load_raw_data(filepath: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from CSV file.

        Expected CSV format: time,open,high,low,close,volume

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        # Load CSV
        df = pd.read_csv(filepath)

        # Parse datetime
        df['time'] = pd.to_datetime(df['time'])

        # Set time as index and sort
        df = df.set_index('time').sort_index()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Keep only OHLCV columns
        df = df[required_cols]

        return df

    @staticmethod
    def filter_sparse_days(df: pd.DataFrame, min_completeness: float = 0.9) -> pd.DataFrame:
        """
        Filter out sparse trading days with insufficient data.

        Strategy:
        1. Calculate actual minutes per trading day
        2. Use median of historical trading days as "expected minutes"
        3. Filter out days with actual minutes < expected × min_completeness

        Args:
            df: DataFrame with datetime index
            min_completeness: Minimum completeness ratio (default: 0.9 = 90%)

        Returns:
            Filtered DataFrame
        """
        # Count minutes per day
        df['date'] = df.index.date
        daily_counts = df.groupby('date').size()

        # Calculate expected minutes (median of historical days)
        expected_minutes = daily_counts.median()

        # Filter days with sufficient data
        min_minutes = expected_minutes * min_completeness
        valid_dates = daily_counts[daily_counts >= min_minutes].index

        # Filter dataframe
        df_filtered = df[df['date'].isin(valid_dates)].copy()
        df_filtered = df_filtered.drop(columns=['date'])

        n_removed = len(daily_counts) - len(valid_dates)
        if n_removed > 0:
            warnings.warn(
                f"Filtered out {n_removed} sparse trading days "
                f"(expected: {expected_minutes:.0f} min, threshold: {min_minutes:.0f} min)"
            )

        return df_filtered

    @staticmethod
    def _identify_day_boundaries(df: pd.DataFrame, gap_threshold_hours: float = 4.0) -> pd.Series:
        """
        Identify trading day boundaries by detecting large time gaps.

        Args:
            df: DataFrame with datetime index
            gap_threshold_hours: Time gap threshold to consider as new day (default: 4 hours)

        Returns:
            Boolean Series indicating start of new trading day
        """
        time_diff = df.index.to_series().diff()
        is_new_day = time_diff > pd.Timedelta(hours=gap_threshold_hours)
        is_new_day.iloc[0] = True  # First row is always start of a day
        return is_new_day

    @staticmethod
    def resample_to_tau(df: pd.DataFrame, tau: int) -> pd.DataFrame:
        """
        Resample 1-minute bars to τ-minute bars.

        Key handling:
        1. Window completeness: Discard bars with <80% data (missing >20%)
        2. Day boundaries: Force new window at each trading day start
        3. End-of-day: Discard incomplete bars at end of each day

        Args:
            df: DataFrame with 1-minute OHLCV data
            tau: Target bar length in minutes

        Returns:
            DataFrame with τ-minute bars
        """
        if tau == 1:
            return df.copy()

        # Identify day boundaries
        is_new_day = DataProcessor._identify_day_boundaries(df)
        df['day_id'] = is_new_day.cumsum()

        resampled_bars = []

        # Process each trading day separately
        for day_id, day_df in df.groupby('day_id'):
            day_df = day_df.drop(columns=['day_id'])

            # Create τ-minute windows
            n_bars = len(day_df)
            n_windows = n_bars // tau

            for i in range(n_windows):
                window_start = i * tau
                window_end = (i + 1) * tau
                window = day_df.iloc[window_start:window_end]

                # Calculate window completeness
                actual_minutes = len(window)
                completeness = actual_minutes / tau

                # Discard if completeness < 80%
                if completeness < 0.8:
                    continue

                # Compute OHLCV for this window
                bar = {
                    'open': window['open'].iloc[0],
                    'high': window['high'].max(),
                    'low': window['low'].min(),
                    'close': window['close'].iloc[-1],
                    'volume': window['volume'].sum(),
                    'completeness': completeness
                }

                # Use the timestamp of the window start
                bar_time = window.index[0]
                resampled_bars.append((bar_time, bar))

        # Convert to DataFrame
        if not resampled_bars:
            raise ValueError("No valid bars after resampling. Check data quality.")

        times, bars = zip(*resampled_bars)
        df_resampled = pd.DataFrame(bars, index=pd.DatetimeIndex(times))

        # Check for consecutive invalid bars (would have been filtered out)
        # This is already handled by the completeness check above

        n_original = len(df)
        n_resampled = len(df_resampled)
        warnings.warn(
            f"Resampled from {n_original} 1-min bars to {n_resampled} {tau}-min bars "
            f"(expected ~{n_original//tau}, actual {n_resampled})"
        )

        return df_resampled

    @staticmethod
    def compute_tick_normalized_ranges(df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
        """
        Compute tick-normalized ranges.

        Formulas:
            R = (H - L) / epsilon
            R_up = (H - O) / epsilon
            R_dn = (O - L) / epsilon

        All ranges are rounded to nearest integer (number of ticks).

        Args:
            df: DataFrame with OHLC columns
            tick_size: Tick size (epsilon) for the instrument

        Returns:
            DataFrame with added columns: R, R_up, R_dn
        """
        df = df.copy()

        # Compute ranges (normalized to ticks)
        df['R'] = np.round((df['high'] - df['low']) / tick_size).astype(int)
        df['R_up'] = np.round((df['high'] - df['open']) / tick_size).astype(int)
        df['R_dn'] = np.round((df['open'] - df['low']) / tick_size).astype(int)

        # Sanity check: R should equal R_up + R_dn (approximately, due to rounding)
        # Allow small discrepancies due to rounding
        discrepancy = np.abs(df['R'] - (df['R_up'] + df['R_dn']))
        if (discrepancy > 1).any():
            n_issues = (discrepancy > 1).sum()
            warnings.warn(
                f"Found {n_issues} bars where R != R_up + R_dn (discrepancy > 1 tick). "
                f"This may indicate data quality issues."
            )

        return df

    @staticmethod
    def process_pipeline(filepath: str, tick_size: float, tau: int = 1,
                        min_completeness: float = 0.9) -> pd.DataFrame:
        """
        Complete data processing pipeline.

        Args:
            filepath: Path to raw CSV file
            tick_size: Tick size for the instrument
            tau: Target bar length in minutes (default: 1)
            min_completeness: Minimum completeness for sparse day filtering (default: 0.9)

        Returns:
            Processed DataFrame with OHLCV and range columns
        """
        # Load data
        df = DataProcessor.load_raw_data(filepath)

        # Filter sparse days
        df = DataProcessor.filter_sparse_days(df, min_completeness)

        # Resample to tau minutes
        df = DataProcessor.resample_to_tau(df, tau)

        # Compute ranges
        df = DataProcessor.compute_tick_normalized_ranges(df, tick_size)

        return df
