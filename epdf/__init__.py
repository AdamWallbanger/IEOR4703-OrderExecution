"""
ePDF Calculator for Futures Trading

A conditional empirical PDF calculator for estimating price movement distributions
in futures markets over a future time window (τ minutes) based on historical data
and current market state.
"""

from .calculator import ePDFCalculator

__version__ = "1.0.0"
__all__ = ["ePDFCalculator"]
