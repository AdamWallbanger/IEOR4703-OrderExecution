"""
Probability Estimator Module

Builds conditional empirical PDFs and CDFs for range distributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict
import warnings


class PDFEstimator:
    """
    Conditional probability distribution estimator.

    Responsibilities:
    - Build conditional PDFs for each state (m, n, k)
    - Support raw and smoothed estimation methods
    - Compute CDFs for fill probability queries
    - Provide query interfaces for strategy use
    """

    def __init__(self):
        """Initialize PDF estimator."""
        self.pdf_dict = {}  # {(m,n,k): {'range': {ell: prob}, 'range_up': ..., 'range_dn': ...}}
        self.cdf_dict = {}  # {(m,n,k): {'range_up': {ell: cum_prob}, 'range_dn': ...}}
        self.state_sample_counts = {}  # {(m,n,k): count}
        self.method = None
        self.alpha = None

    def build_conditional_pdf(self, df: pd.DataFrame, J_s: int,
                             method: str = 'raw', alpha: float = 0.5):
        """
        Build conditional PDFs for all state combinations.

        For each state (m, n, k):
        1. Filter historical data matching that state
        2. Count frequency of each range value
        3. Normalize to get probability distribution

        Args:
            df: DataFrame with columns: state_m, state_n, state_k, R, R_up, R_dn
            J_s: Starting index for ePDF construction
            method: 'raw' (empirical frequencies) or 'smoothed' (Laplace smoothing)
            alpha: Smoothing parameter for Laplace smoothing (default: 0.5)
        """
        self.method = method
        self.alpha = alpha

        # Use data from J_s onwards
        df_train = df.iloc[J_s:].copy()

        # Get all unique states
        states = df_train.groupby(['state_m', 'state_n', 'state_k']).size()

        # Build PDF for each state
        for (m, n, k), count in states.items():
            self.state_sample_counts[(m, n, k)] = count

            # Filter data for this state
            state_mask = (df_train['state_m'] == m) & \
                        (df_train['state_n'] == n) & \
                        (df_train['state_k'] == k)
            state_data = df_train[state_mask]

            # Initialize PDF dictionary for this state
            self.pdf_dict[(m, n, k)] = {
                'range': {},
                'range_up': {},
                'range_dn': {}
            }

            # Build PDF for each range type
            for range_col, range_key in [('R', 'range'),
                                         ('R_up', 'range_up'),
                                         ('R_dn', 'range_dn')]:

                # Count frequencies
                range_counts = state_data[range_col].value_counts().to_dict()
                total_count = len(state_data)

                # Get all possible range values (0 to max observed)
                if range_counts:
                    max_range = int(max(range_counts.keys()))
                else:
                    max_range = 0

                # Compute probabilities (sparse: only store observed values + fill gaps efficiently)
                if method == 'raw':
                    # Raw empirical frequencies — only store observed values
                    for ell, count_ell in range_counts.items():
                        prob = count_ell / total_count if total_count > 0 else 0
                        self.pdf_dict[(m, n, k)][range_key][int(ell)] = prob
                    # Ensure 0 is present
                    if 0 not in self.pdf_dict[(m, n, k)][range_key]:
                        self.pdf_dict[(m, n, k)][range_key][0] = 0.0

                elif method == 'smoothed':
                    # Laplace smoothing — use sparse storage
                    num_bins = max_range + 1
                    denominator = total_count + alpha * num_bins
                    base_prob = alpha / denominator  # probability for unobserved values

                    # Store observed values with smoothed probability
                    for ell, count_ell in range_counts.items():
                        prob = (count_ell + alpha) / denominator
                        self.pdf_dict[(m, n, k)][range_key][int(ell)] = prob

                    # Fill unobserved values only up to a reasonable limit
                    # For large ranges, store base_prob as metadata instead of all values
                    if num_bins <= 1000:
                        # Small range: fill all gaps explicitly
                        for ell in range(0, max_range + 1):
                            if ell not in self.pdf_dict[(m, n, k)][range_key]:
                                self.pdf_dict[(m, n, k)][range_key][ell] = base_prob
                    else:
                        # Large range: store base probability as metadata
                        # Unobserved values will be handled at query time
                        if 0 not in self.pdf_dict[(m, n, k)][range_key]:
                            self.pdf_dict[(m, n, k)][range_key][0] = base_prob

                    # Store metadata for query-time smoothing
                    if not hasattr(self, '_smoothing_meta'):
                        self._smoothing_meta = {}
                    self._smoothing_meta[((m, n, k), range_key)] = {
                        'base_prob': base_prob,
                        'max_range': max_range,
                        'num_bins': num_bins,
                        'denominator': denominator,
                    }

                else:
                    raise ValueError(f"Unknown method: {method}. Use 'raw' or 'smoothed'.")

        # Report statistics
        n_states = len(self.pdf_dict)
        min_samples = min(self.state_sample_counts.values()) if self.state_sample_counts else 0
        max_samples = max(self.state_sample_counts.values()) if self.state_sample_counts else 0

        warnings.warn(
            f"Built PDFs for {n_states} states. "
            f"Sample counts: min={min_samples}, max={max_samples}, method={method}"
        )

    def compute_cdf(self):
        """
        Compute cumulative distribution functions from PDFs.

        For trading applications:
        - CDF for 'range_up': P(R_up >= ℓ) = sell order fill probability
        - CDF for 'range_dn': P(R_dn >= ℓ) = buy order fill probability

        CDF is computed as: F(ℓ) = sum_{i=ℓ}^{ℓ_max} P(R = i)
        Uses reverse cumulative sum for O(n) performance.
        """
        for state, pdfs in self.pdf_dict.items():
            self.cdf_dict[state] = {
                'range_up': {},
                'range_dn': {}
            }

            for direction in ['range_up', 'range_dn']:
                pdf = pdfs[direction]
                if not pdf:
                    continue

                max_ell = max(pdf.keys())
                meta_key = (state, direction)
                base_prob = 0.0
                if hasattr(self, '_smoothing_meta') and meta_key in self._smoothing_meta:
                    base_prob = self._smoothing_meta[meta_key]['base_prob']

                # Build CDF using reverse cumulative sum
                # Start from max_ell and accumulate backwards
                cum_prob = 0.0
                cdf = {}
                for ell in range(max_ell, -1, -1):
                    cum_prob += pdf.get(ell, base_prob)
                    cdf[ell] = cum_prob

                self.cdf_dict[state][direction] = cdf

    def query_pdf(self, ell: int, direction: str, state: Tuple[int, int, int]) -> float:
        """
        Query probability mass function.

        Args:
            ell: Range value (number of ticks)
            direction: 'range', 'range_up', or 'range_dn'
            state: State tuple (m, n, k)

        Returns:
            P(R = ℓ | state)
        """
        if state not in self.pdf_dict:
            warnings.warn(f"State {state} not found in training data. Returning 0.")
            return 0.0

        if direction not in ['range', 'range_up', 'range_dn']:
            raise ValueError(f"Invalid direction: {direction}. Use 'range', 'range_up', or 'range_dn'.")

        pdf = self.pdf_dict[state][direction]

        # If ell is in the stored PDF, return it directly
        if ell in pdf:
            return pdf[ell]

        # For smoothed method, check if ell is within the valid range
        if self.method == 'smoothed':
            meta_key = (state, direction)
            if hasattr(self, '_smoothing_meta') and meta_key in self._smoothing_meta:
                meta = self._smoothing_meta[meta_key]
                if 0 <= ell <= meta['max_range'] + 10:
                    return meta['base_prob']

        return 0.0

    def query_cdf(self, ell: int, direction: str, state: Tuple[int, int, int]) -> float:
        """
        Query cumulative distribution function (fill probability).

        Args:
            ell: Range value (number of ticks)
            direction: 'range_up' (sell order) or 'range_dn' (buy order)
            state: State tuple (m, n, k)

        Returns:
            P(R >= ℓ | state) - probability that price moves at least ℓ ticks
        """
        if state not in self.cdf_dict:
            warnings.warn(f"State {state} not found in training data. Returning 0.")
            return 0.0

        if direction not in ['range_up', 'range_dn']:
            raise ValueError(f"Invalid direction: {direction}. Use 'range_up' or 'range_dn'.")

        return self.cdf_dict[state][direction].get(ell, 0.0)

    def get_full_distribution(self, direction: str, state: Tuple[int, int, int]) -> Dict[int, float]:
        """
        Get complete probability distribution for a state.

        Args:
            direction: 'range', 'range_up', or 'range_dn'
            state: State tuple (m, n, k)

        Returns:
            Dictionary {ell: probability} for all observed range values
        """
        if state not in self.pdf_dict:
            warnings.warn(f"State {state} not found in training data. Returning empty dict.")
            return {}

        if direction not in ['range', 'range_up', 'range_dn']:
            raise ValueError(f"Invalid direction: {direction}. Use 'range', 'range_up', or 'range_dn'.")

        return self.pdf_dict[state][direction].copy()

    def get_state_statistics(self, state: Tuple[int, int, int]) -> Dict:
        """
        Get statistics for a specific state.

        Args:
            state: State tuple (m, n, k)

        Returns:
            Dictionary with sample count and distribution statistics
        """
        if state not in self.pdf_dict:
            return {'sample_count': 0, 'exists': False}

        sample_count = self.state_sample_counts.get(state, 0)

        # Get max range values
        max_range = max(self.pdf_dict[state]['range'].keys()) if self.pdf_dict[state]['range'] else 0
        max_range_up = max(self.pdf_dict[state]['range_up'].keys()) if self.pdf_dict[state]['range_up'] else 0
        max_range_dn = max(self.pdf_dict[state]['range_dn'].keys()) if self.pdf_dict[state]['range_dn'] else 0

        return {
            'exists': True,
            'sample_count': sample_count,
            'max_range': max_range,
            'max_range_up': max_range_up,
            'max_range_dn': max_range_dn
        }

    def validate_probabilities(self) -> Dict[str, bool]:
        """
        Validate that all PDFs sum to 1 and CDFs are monotonic.

        Returns:
            Dictionary with validation results
        """
        results = {
            'pdf_sums_valid': True,
            'cdf_monotonic': True,
            'issues': []
        }

        for state, pdfs in self.pdf_dict.items():
            # Check PDF sums
            for direction in ['range', 'range_up', 'range_dn']:
                pdf = pdfs[direction]
                if pdf:
                    total = sum(pdf.values())

                    # Account for unobserved values in sparse storage
                    meta_key = (state, direction)
                    if hasattr(self, '_smoothing_meta') and meta_key in self._smoothing_meta:
                        meta = self._smoothing_meta[meta_key]
                        n_stored = len(pdf)
                        n_total = meta['num_bins']
                        n_missing = n_total - n_stored
                        if n_missing > 0:
                            total += n_missing * meta['base_prob']

                    if not np.isclose(total, 1.0, atol=1e-6):
                        results['pdf_sums_valid'] = False
                        results['issues'].append(
                            f"State {state}, {direction}: PDF sum = {total:.6f} (expected 1.0)"
                        )

            # Check CDF monotonicity
            if state in self.cdf_dict:
                for direction in ['range_up', 'range_dn']:
                    cdf = self.cdf_dict[state][direction]
                    if cdf:
                        sorted_ells = sorted(cdf.keys())
                        for i in range(len(sorted_ells) - 1):
                            ell1, ell2 = sorted_ells[i], sorted_ells[i + 1]
                            if cdf[ell1] < cdf[ell2]:
                                results['cdf_monotonic'] = False
                                results['issues'].append(
                                    f"State {state}, {direction}: CDF not monotonic at ℓ={ell1},{ell2}"
                                )

        return results
