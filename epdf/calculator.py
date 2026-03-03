"""
ePDF Calculator - Main Interface

Conditional empirical PDF calculator for futures trading.
"""

import pandas as pd
import pickle
from typing import Tuple, Dict, Optional
from datetime import datetime
import warnings

from .instrument_config import InstrumentConfig
from .data_processor import DataProcessor
from .state_classifier import StateClassifier
from .probability_estimator import PDFEstimator


class ePDFCalculator:
    """
    Conditional Empirical PDF Calculator for Futures Trading.

    Estimates probability distributions of price movements over a future time window (τ)
    conditioned on current market state (volume, volatility, price change).

    Usage:
        # Initialize
        calc = ePDFCalculator(
            instrument='VG',
            tau=5,
            M=3, N=3, K=2,
            ewma_halflife=10,
            estimation_method='smoothed'
        )

        # Train on historical data
        calc.fit('data/VGH22.csv')

        # Query fill probability
        state = calc.get_current_state(volume=1000, volatility=5, price_change=0.5)
        fill_prob = calc.query_cdf(ell=3, direction='range_dn', state=state)

        # Save/load model
        calc.save('models/epdf_VG_tau5.pkl')
        calc = ePDFCalculator.load('models/epdf_VG_tau5.pkl')
    """

    def __init__(self,
                 instrument: str,
                 tau: int,
                 M: int,
                 N: int,
                 K: int,
                 ewma_halflife: int,
                 estimation_method: str = 'raw',
                 smoothing_alpha: float = 0.5,
                 tick_size: Optional[float] = None):
        """
        Initialize ePDF Calculator.

        Args:
            instrument: Instrument symbol (e.g., 'ES', 'VG', 'NQ')
            tau: Holding period in minutes (e.g., 5, 10, 15, 30, 60)
            M: Number of volume states (recommended: 3-5)
            N: Number of volatility states (recommended: 3-5)
            K: Number of price change states (recommended: 3-5)
            ewma_halflife: EWMA half-life parameter (λ = 2^(-1/halflife))
            estimation_method: 'raw' (empirical) or 'smoothed' (Laplace)
            smoothing_alpha: Laplace smoothing parameter (default: 0.5)
            tick_size: Optional manual tick size override
        """
        self.instrument = instrument.upper()
        self.tau = tau
        self.M = M
        self.N = N
        self.K = K
        self.ewma_halflife = ewma_halflife
        self.estimation_method = estimation_method
        self.smoothing_alpha = smoothing_alpha

        # Get tick size
        if tick_size is not None:
            self.tick_size = tick_size
        else:
            self.tick_size = InstrumentConfig.get_tick_size(self.instrument)

        # Initialize components
        self.data_processor = DataProcessor()
        self.state_classifier = StateClassifier()
        self.pdf_estimator = PDFEstimator()

        # Training metadata
        self.J_s = None
        self.training_data_path = None
        self.training_timestamp = None
        self.train_end_date = None
        self.data_shape = None
        self.is_fitted = False

    def fit(self, filepath: str, min_completeness: float = 0.9,
            train_end_date: Optional[str] = None):
        """
        Train the ePDF calculator on historical data.

        Pipeline:
        1. Load and preprocess data (DataProcessor)
        2. Compute EWMA features (StateClassifier)
        3. Determine J_s and bin into states (StateClassifier)
        4. Build conditional PDFs (PDFEstimator)
        5. Compute CDFs (PDFEstimator)

        Args:
            filepath: Path to CSV file with OHLCV data
            min_completeness: Minimum completeness for sparse day filtering (default: 0.9)
            train_end_date: Optional cutoff date for training data (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').
                           If specified, only data before this date will be used for training.
                           This prevents forward-looking bias in backtesting scenarios.
                           Example: calc.fit('data.csv', train_end_date='2020-06-15')
        """
        print(f"Training ePDF Calculator for {self.instrument}, τ={self.tau} min...")
        print(f"Parameters: M={self.M}, N={self.N}, K={self.K}, halflife={self.ewma_halflife}")
        if train_end_date is not None:
            print(f"Training cutoff date: {train_end_date}")

        # Stage 1: Data preprocessing
        print("\n[1/5] Loading and preprocessing data...")
        df = self.data_processor.process_pipeline(
            filepath=filepath,
            tick_size=self.tick_size,
            tau=self.tau,
            min_completeness=min_completeness,
            train_end_date=train_end_date
        )
        print(f"  Loaded {len(df)} bars after preprocessing")

        # Stage 2: Compute EWMA features
        print("\n[2/5] Computing EWMA features...")
        df = self.state_classifier.compute_all_ewma_features(df, self.ewma_halflife)
        print(f"  Computed v_ewma, sigma_ewma, delta_x_ewma")

        # Stage 3: Determine J_s and bin into states
        print("\n[3/5] Determining J_s and binning into states...")
        self.J_s = self.state_classifier.determine_J_s(df, self.ewma_halflife, self.M, self.N, self.K)
        print(f"  J_s = {self.J_s} (using data from index {self.J_s} onwards)")

        df = self.state_classifier.bin_into_states(df, self.M, self.N, self.K, self.J_s)
        print(f"  Binned into {self.M}×{self.N}×{self.K} = {self.M*self.N*self.K} states")

        # Stage 4: Build conditional PDFs
        print("\n[4/5] Building conditional PDFs...")
        self.pdf_estimator.build_conditional_pdf(
            df=df,
            J_s=self.J_s,
            method=self.estimation_method,
            alpha=self.smoothing_alpha
        )
        print(f"  Built PDFs using method: {self.estimation_method}")

        # Stage 5: Compute CDFs
        print("\n[5/5] Computing CDFs...")
        self.pdf_estimator.compute_cdf()
        print(f"  Computed CDFs for fill probability queries")

        # Validate
        print("\n[Validation] Checking probability properties...")
        validation = self.pdf_estimator.validate_probabilities()
        if validation['pdf_sums_valid'] and validation['cdf_monotonic']:
            print("  ✓ All PDFs sum to 1.0")
            print("  ✓ All CDFs are monotonic")
        else:
            print("  ⚠ Validation issues found:")
            for issue in validation['issues'][:5]:  # Show first 5 issues
                print(f"    - {issue}")

        # Store metadata
        self.training_data_path = filepath
        self.training_timestamp = datetime.now()
        self.train_end_date = train_end_date
        self.data_shape = df.shape
        self.is_fitted = True

        print(f"\n✓ Training complete! Model ready for queries.")

    def get_current_state(self, volume: float, volatility: float, price_change: float) -> Tuple[int, int, int]:
        """
        Classify current market conditions into a state.

        Args:
            volume: Current EWMA volume
            volatility: Current EWMA volatility (range)
            price_change: Current EWMA price change

        Returns:
            State tuple (m, n, k)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.state_classifier.get_current_state(volume, volatility, price_change)

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
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.pdf_estimator.query_pdf(ell, direction, state)

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
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.pdf_estimator.query_cdf(ell, direction, state)

    def get_full_distribution(self, direction: str, state: Tuple[int, int, int]) -> Dict[int, float]:
        """
        Get complete probability distribution for a state.

        Args:
            direction: 'range', 'range_up', or 'range_dn'
            state: State tuple (m, n, k)

        Returns:
            Dictionary {ell: probability}
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.pdf_estimator.get_full_distribution(direction, state)

    def get_state_statistics(self, state: Tuple[int, int, int]) -> Dict:
        """
        Get statistics for a specific state.

        Args:
            state: State tuple (m, n, k)

        Returns:
            Dictionary with sample count and distribution statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.pdf_estimator.get_state_statistics(state)

    def get_model_info(self) -> Dict:
        """
        Get model configuration and training metadata.

        Returns:
            Dictionary with model information
        """
        return {
            'instrument': self.instrument,
            'tau': self.tau,
            'M': self.M,
            'N': self.N,
            'K': self.K,
            'ewma_halflife': self.ewma_halflife,
            'estimation_method': self.estimation_method,
            'smoothing_alpha': self.smoothing_alpha,
            'tick_size': self.tick_size,
            'J_s': self.J_s,
            'training_data_path': self.training_data_path,
            'training_timestamp': self.training_timestamp,
            'train_end_date': self.train_end_date,
            'data_shape': self.data_shape,
            'is_fitted': self.is_fitted,
            'n_states': self.M * self.N * self.K,
            'n_states_observed': len(self.pdf_estimator.pdf_dict) if self.is_fitted else 0
        }

    def save(self, filepath: str):
        """
        Save trained model to file.

        Args:
            filepath: Path to save pickle file
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Model saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ePDFCalculator':
        """
        Load trained model from file.

        Args:
            filepath: Path to pickle file

        Returns:
            Loaded ePDFCalculator instance
        """
        with open(filepath, 'rb') as f:
            calc = pickle.load(f)

        if not isinstance(calc, ePDFCalculator):
            raise ValueError(f"Loaded object is not an ePDFCalculator instance")

        print(f"Model loaded from: {filepath}")
        print(f"  Instrument: {calc.instrument}, τ={calc.tau} min")
        print(f"  States: {calc.M}×{calc.N}×{calc.K} = {calc.M*calc.N*calc.K}")
        print(f"  Trained on: {calc.training_timestamp}")

        return calc
