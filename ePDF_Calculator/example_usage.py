"""
Example Usage: Training and Using the ePDF Calculator

This script demonstrates how to:
1. Train an ePDF calculator on historical data
2. Query fill probabilities for different market states
3. Save and load trained models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epdf import ePDFCalculator
import numpy as np


def example_1_train_model():
    """Example 1: Train a model on VG (EuroStoxx 50) data."""
    print("=" * 70)
    print("EXAMPLE 1: Training ePDF Calculator")
    print("=" * 70)

    # Initialize calculator
    calc = ePDFCalculator(
        instrument='VG',           # EuroStoxx 50
        tau=5,                     # 5-minute holding period
        M=3,                       # 3 volume states (low, medium, high)
        N=3,                       # 3 volatility states
        K=2,                       # 2 price change states (down, up)
        ewma_halflife=10,          # EWMA half-life = 10 bars
        estimation_method='smoothed',  # Use Laplace smoothing
        smoothing_alpha=0.5
    )

    # Train on historical data
    data_path = 'data/EuroStoxx/VGH22.csv'
    calc.fit(data_path)

    # Save trained model
    model_path = 'models/epdf_VG_tau5_M3N3K2.pkl'
    os.makedirs('models', exist_ok=True)
    calc.save(model_path)

    print(f"\n✓ Model saved to: {model_path}")
    return calc


def example_2_query_probabilities(calc):
    """Example 2: Query fill probabilities for different scenarios."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Querying Fill Probabilities")
    print("=" * 70)

    # Scenario 1: Low volume, low volatility, neutral price change
    print("\nScenario 1: Quiet market (low vol, low volatility)")
    state_1 = (1, 1, 1)  # (low volume, low volatility, down trend)

    # Query: If I place a buy order 3 ticks below current price, what's the fill probability?
    fill_prob_buy_3 = calc.query_cdf(ell=3, direction='range_dn', state=state_1)
    print(f"  State: {state_1}")
    print(f"  Buy order 3 ticks below: Fill probability = {fill_prob_buy_3:.2%}")

    # Query: If I place a sell order 3 ticks above current price, what's the fill probability?
    fill_prob_sell_3 = calc.query_cdf(ell=3, direction='range_up', state=state_1)
    print(f"  Sell order 3 ticks above: Fill probability = {fill_prob_sell_3:.2%}")

    # Scenario 2: High volume, high volatility
    print("\nScenario 2: Active market (high vol, high volatility)")
    state_2 = (3, 3, 2)  # (high volume, high volatility, up trend)

    fill_prob_buy_3 = calc.query_cdf(ell=3, direction='range_dn', state=state_2)
    fill_prob_sell_3 = calc.query_cdf(ell=3, direction='range_up', state=state_2)
    print(f"  State: {state_2}")
    print(f"  Buy order 3 ticks below: Fill probability = {fill_prob_buy_3:.2%}")
    print(f"  Sell order 3 ticks above: Fill probability = {fill_prob_sell_3:.2%}")

    # Get full distribution
    print("\nScenario 3: Full distribution for state (2, 2, 1)")
    state_3 = (2, 2, 1)
    dist_dn = calc.get_full_distribution('range_dn', state_3)

    print(f"  State: {state_3}")
    print(f"  Range_dn distribution (first 10 values):")
    for ell in sorted(dist_dn.keys())[:10]:
        prob = dist_dn[ell]
        cum_prob = calc.query_cdf(ell, 'range_dn', state_3)
        print(f"    ℓ={ell:2d}: P(R_dn={ell}) = {prob:.4f}, P(R_dn≥{ell}) = {cum_prob:.4f}")


def example_3_state_classification():
    """Example 3: Classify current market state."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Real-time State Classification")
    print("=" * 70)

    # Load trained model
    model_path = 'models/epdf_VG_tau5_M3N3K2.pkl'
    calc = ePDFCalculator.load(model_path)

    # Simulate current market conditions (EWMA values)
    print("\nCurrent market conditions (EWMA values):")
    current_volume = 150.0
    current_volatility = 8.5
    current_price_change = 2.3

    print(f"  Volume EWMA: {current_volume}")
    print(f"  Volatility EWMA: {current_volatility}")
    print(f"  Price Change EWMA: {current_price_change}")

    # Classify into state
    state = calc.get_current_state(current_volume, current_volatility, current_price_change)
    print(f"\n  → Classified state: {state}")

    # Get state statistics
    stats = calc.get_state_statistics(state)
    print(f"  → Historical samples in this state: {stats['sample_count']}")
    print(f"  → Max observed range: {stats['max_range']} ticks")


def example_4_compare_methods():
    """Example 4: Compare raw vs smoothed estimation methods."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Comparing Raw vs Smoothed Methods")
    print("=" * 70)

    data_path = 'data/EuroStoxx/VGH22.csv'

    # Train with raw method
    print("\nTraining with RAW method...")
    calc_raw = ePDFCalculator(
        instrument='VG', tau=5, M=3, N=3, K=2,
        ewma_halflife=10, estimation_method='raw'
    )
    calc_raw.fit(data_path)

    # Train with smoothed method
    print("\nTraining with SMOOTHED method...")
    calc_smooth = ePDFCalculator(
        instrument='VG', tau=5, M=3, N=3, K=2,
        ewma_halflife=10, estimation_method='smoothed', smoothing_alpha=1.0
    )
    calc_smooth.fit(data_path)

    # Compare probabilities for a sparse state
    print("\nComparing probabilities for state (1, 1, 1):")
    state = (1, 1, 1)

    for ell in range(0, 10):
        prob_raw = calc_raw.query_pdf(ell, 'range', state)
        prob_smooth = calc_smooth.query_pdf(ell, 'range', state)
        print(f"  ℓ={ell:2d}: Raw={prob_raw:.4f}, Smoothed={prob_smooth:.4f}")


def example_5_model_info():
    """Example 5: Inspect model configuration and metadata."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Model Information")
    print("=" * 70)

    model_path = 'models/epdf_VG_tau5_M3N3K2.pkl'
    calc = ePDFCalculator.load(model_path)

    info = calc.get_model_info()

    print("\nModel Configuration:")
    print(f"  Instrument: {info['instrument']}")
    print(f"  Holding period (τ): {info['tau']} minutes")
    print(f"  Tick size: {info['tick_size']}")
    print(f"  State space: {info['M']}×{info['N']}×{info['K']} = {info['n_states']} states")
    print(f"  EWMA half-life: {info['ewma_halflife']}")
    print(f"  Estimation method: {info['estimation_method']}")

    print("\nTraining Metadata:")
    print(f"  Training data: {info['training_data_path']}")
    print(f"  Training timestamp: {info['training_timestamp']}")
    print(f"  Data shape: {info['data_shape']}")
    print(f"  J_s (start index): {info['J_s']}")
    print(f"  States observed: {info['n_states_observed']}/{info['n_states']}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ePDF CALCULATOR - USAGE EXAMPLES")
    print("=" * 70)

    # Check if data exists
    data_path = 'data/EuroStoxx/VGH22.csv'
    if not os.path.exists(data_path):
        print(f"\n⚠ Data file not found: {data_path}")
        print("Please ensure the data file exists before running examples.")
        return

    # Run examples
    calc = example_1_train_model()
    example_2_query_probabilities(calc)
    example_3_state_classification()
    example_4_compare_methods()
    example_5_model_info()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
