"""
State Classifier Module

Implements EWMA computation (Algorithm 1) and state classification via binning.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings


class StateClassifier:
    """
    Market state classification based on EWMA-smoothed features.

    Responsibilities:
    - Compute EWMA features (Algorithm 1 from specification)
    - Bin continuous features into discrete states
    - Classify timestamps into state combinations (m, n, k)
    """

    def __init__(self):
        """Initialize state classifier."""
        self.bin_boundaries = None  # Will store binning thresholds
        self.M = None  # Number of volume states
        self.N = None  # Number of volatility states
        self.K = None  # Number of price change states
        self.halflife = None

    @staticmethod
    def compute_ewma(eta_series: pd.Series, halflife: int) -> Tuple[pd.Series, pd.Series]:
        """
        Compute Exponentially Weighted Moving Average and Variance.

        Implements Algorithm 1 from specification document.

        CRITICAL: At step j, uses η_{j-1} to avoid forward-looking bias.

        Args:
            eta_series: Input time series (e.g., volume, range, price change)
            halflife: Half-life parameter m (λ = 2^(-1/m))

        Returns:
            Tuple of (ewma_series, ewmv_series)
            - ewma_series: Exponentially weighted moving average
            - ewmv_series: Exponentially weighted moving standard deviation
        """
        eta = eta_series.values
        n = len(eta)

        # Initialize arrays
        sumW = np.zeros(n)
        sumWX = np.zeros(n)
        ewma = np.zeros(n)
        sumWSS = np.zeros(n)
        ewmv = np.zeros(n)

        # Calculate lambda parameter
        lambda_param = 2 ** (-1 / halflife)

        # Algorithm 1 implementation
        for j in range(n):
            if j == 0:
                # Initial values (all zeros)
                sumW[j] = 0
                sumWX[j] = 0
                ewma[j] = 0
                sumWSS[j] = 0
                ewmv[j] = 0

            elif j == 1:
                # First update: use η_{j-1} = η[0]
                sumW[j] = 1
                sumWX[j] = eta[j-1]  # Use η_{j-1}, not η[j]
                ewma[j] = sumWX[j] / sumW[j]
                sumWSS[j] = (eta[j-1] - ewma[j]) ** 2
                ewmv[j] = np.sqrt(sumWSS[j] / sumW[j])

            else:  # j >= 2
                # Recursive update: use η_{j-1}
                sumW[j] = lambda_param * sumW[j-1] + 1
                sumWX[j] = lambda_param * sumWX[j-1] + eta[j-1]  # Use η_{j-1}, not η[j]
                ewma[j] = sumWX[j] / sumW[j]
                sumWSS[j] = lambda_param * sumWSS[j-1] + (eta[j-1] - ewma[j]) ** 2
                ewmv[j] = np.sqrt(sumWSS[j] / sumW[j])

        # Convert back to pandas Series with original index
        ewma_series = pd.Series(ewma, index=eta_series.index)
        ewmv_series = pd.Series(ewmv, index=eta_series.index)

        return ewma_series, ewmv_series

    @staticmethod
    def compute_all_ewma_features(df: pd.DataFrame, halflife: int) -> pd.DataFrame:
        """
        Compute all three EWMA features required for state classification.

        Features:
        1. Volume EWMA: smoothed volume
        2. Volatility EWMA: smoothed range (as volatility proxy)
        3. Price Change EWMA: smoothed price change (using Open prices)

        Args:
            df: DataFrame with columns: open, volume, R (range)
            halflife: EWMA half-life parameter

        Returns:
            DataFrame with added columns: v_ewma, sigma_ewma, delta_x_ewma, delta_x
        """
        df = df.copy()

        # 1. Volume EWMA
        v_ewma, _ = StateClassifier.compute_ewma(df['volume'], halflife)
        df['v_ewma'] = v_ewma

        # 2. Volatility EWMA (using range as proxy)
        sigma_ewma, _ = StateClassifier.compute_ewma(df['R'], halflife)
        df['sigma_ewma'] = sigma_ewma

        # 3. Price Change EWMA
        # delta_x[j] = O[j] - O[j-1]
        df['delta_x'] = df['open'].diff()

        # Apply EWMA to delta_x
        delta_x_ewma, _ = StateClassifier.compute_ewma(df['delta_x'].fillna(0), halflife)
        df['delta_x_ewma'] = delta_x_ewma

        return df

    @staticmethod
    def determine_J_s(df: pd.DataFrame, halflife: int, M: int, N: int, K: int) -> int:
        """
        Determine J_s: the starting position for building ePDF.

        Formula: J_s = max(100, 3 × halflife, 50 × max(M, N, K))

        Rationale:
        - Need sufficient data for EWMA to stabilize (3 × halflife)
        - Need sufficient data for reliable quantile estimation (50 × max(M, N, K))
        - Minimum of 100 bars for statistical reliability

        Args:
            df: DataFrame with data
            halflife: EWMA half-life parameter
            M: Number of volume states
            N: Number of volatility states
            K: Number of price change states

        Returns:
            J_s: Starting index for ePDF construction
        """
        J_s = max(100, 3 * halflife, 50 * max(M, N, K))

        # Ensure J_s doesn't exceed data length
        if J_s >= len(df):
            warnings.warn(
                f"Calculated J_s ({J_s}) exceeds data length ({len(df)}). "
                f"Using J_s = {len(df) // 2} instead."
            )
            J_s = len(df) // 2

        return J_s

    def bin_into_states(self, df: pd.DataFrame, M: int, N: int, K: int, J_s: int) -> pd.DataFrame:
        """
        Bin continuous EWMA features into discrete states.

        Uses quantile-based binning on the full sample [J_s:] to determine boundaries.

        Note: Using full-sample quantiles is acceptable (see specification document).
        The bin boundaries use historical distribution to define state standards,
        but ePDF construction still respects no-forward-looking principle.

        Args:
            df: DataFrame with EWMA features (v_ewma, sigma_ewma, delta_x_ewma)
            M: Number of volume states
            N: Number of volatility states
            K: Number of price change states
            J_s: Starting index for binning

        Returns:
            DataFrame with added columns: state_m, state_n, state_k
        """
        df = df.copy()
        self.M = M
        self.N = N
        self.K = K

        # Use data from J_s onwards for quantile calculation
        df_train = df.iloc[J_s:]

        # Initialize bin boundaries dictionary
        self.bin_boundaries = {}

        # 1. Volume states (m)
        volume_quantiles = [i / M for i in range(1, M)]
        volume_boundaries = df_train['v_ewma'].quantile(volume_quantiles).values
        self.bin_boundaries['volume'] = volume_boundaries

        df['state_m'] = pd.cut(
            df['v_ewma'],
            bins=[-np.inf] + list(volume_boundaries) + [np.inf],
            labels=range(1, M + 1),
            include_lowest=True
        ).astype(int)

        # 2. Volatility states (n)
        volatility_quantiles = [i / N for i in range(1, N)]
        volatility_boundaries = df_train['sigma_ewma'].quantile(volatility_quantiles).values
        self.bin_boundaries['volatility'] = volatility_boundaries

        df['state_n'] = pd.cut(
            df['sigma_ewma'],
            bins=[-np.inf] + list(volatility_boundaries) + [np.inf],
            labels=range(1, N + 1),
            include_lowest=True
        ).astype(int)

        # 3. Price change states (k)
        price_change_quantiles = [i / K for i in range(1, K)]
        price_change_boundaries = df_train['delta_x_ewma'].quantile(price_change_quantiles).values
        self.bin_boundaries['price_change'] = price_change_boundaries

        df['state_k'] = pd.cut(
            df['delta_x_ewma'],
            bins=[-np.inf] + list(price_change_boundaries) + [np.inf],
            labels=range(1, K + 1),
            include_lowest=True
        ).astype(int)

        # Report state distribution
        state_counts = df.iloc[J_s:].groupby(['state_m', 'state_n', 'state_k']).size()
        n_states = M * N * K
        n_observed = len(state_counts)
        min_samples = state_counts.min() if len(state_counts) > 0 else 0

        warnings.warn(
            f"State distribution: {n_observed}/{n_states} states observed. "
            f"Min samples per state: {min_samples}"
        )

        return df

    def get_current_state(self, volume: float, volatility: float, price_change: float) -> Tuple[int, int, int]:
        """
        Classify current market conditions into a state tuple.

        Args:
            volume: Current EWMA volume
            volatility: Current EWMA volatility
            price_change: Current EWMA price change

        Returns:
            Tuple (m, n, k) representing the state
        """
        if self.bin_boundaries is None:
            raise ValueError("Must call bin_into_states() before get_current_state()")

        # Find volume state
        m = 1
        for boundary in self.bin_boundaries['volume']:
            if volume >= boundary:
                m += 1
            else:
                break

        # Find volatility state
        n = 1
        for boundary in self.bin_boundaries['volatility']:
            if volatility >= boundary:
                n += 1
            else:
                break

        # Find price change state
        k = 1
        for boundary in self.bin_boundaries['price_change']:
            if price_change >= boundary:
                k += 1
            else:
                break

        return (m, n, k)

    def get_state_boundaries(self) -> Dict:
        """
        Get the bin boundaries for all features.

        Returns:
            Dictionary with keys: 'volume', 'volatility', 'price_change'
        """
        if self.bin_boundaries is None:
            raise ValueError("Must call bin_into_states() first")
        return self.bin_boundaries.copy()
