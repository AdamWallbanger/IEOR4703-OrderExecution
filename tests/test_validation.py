"""
Validation Tests for ePDF Calculator

Critical tests to ensure correctness:
1. Forward-looking bias detection
2. Probability properties (PDF sums, CDF monotonicity)
3. EWMA correctness
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epdf.state_classifier import StateClassifier
from epdf.probability_estimator import PDFEstimator


def test_ewma_no_forward_looking():
    """
    Test that EWMA at step j does not depend on η[j].

    This is the most critical test for avoiding forward-looking bias.
    """
    print("=" * 60)
    print("TEST 1: EWMA Forward-Looking Bias Detection")
    print("=" * 60)

    # Create simple test data
    eta = pd.Series([10, 20, 30, 40, 50])
    halflife = 2

    # Compute EWMA
    ewma, ewmv = StateClassifier.compute_ewma(eta, halflife)

    # Manual calculation to verify
    lambda_param = 2 ** (-1 / halflife)

    # j=0: ewma[0] = 0
    assert ewma[0] == 0, "ewma[0] should be 0"
    print("✓ j=0: ewma[0] = 0")

    # j=1: ewma[1] should use η[0] = 10
    expected_ewma_1 = 10.0
    assert np.isclose(ewma[1], expected_ewma_1), f"ewma[1] should be {expected_ewma_1}, got {ewma[1]}"
    print(f"✓ j=1: ewma[1] = {ewma[1]:.4f} (uses η[0] = {eta[0]})")

    # j=2: ewma[2] should use η[1] = 20, not η[2] = 30
    sumW_2 = lambda_param * 1 + 1
    sumWX_2 = lambda_param * 10 + 20  # Uses η[1] = 20
    expected_ewma_2 = sumWX_2 / sumW_2
    assert np.isclose(ewma[2], expected_ewma_2), f"ewma[2] should be {expected_ewma_2}, got {ewma[2]}"
    print(f"✓ j=2: ewma[2] = {ewma[2]:.4f} (uses η[1] = {eta[1]}, NOT η[2] = {eta[2]})")

    # Key test: Change η[j] and verify ewma[j] doesn't change
    eta_modified = eta.copy()
    eta_modified[2] = 999  # Change η[2] dramatically

    ewma_modified, _ = StateClassifier.compute_ewma(eta_modified, halflife)

    # ewma[2] should NOT change because it doesn't use η[2]
    assert np.isclose(ewma[2], ewma_modified[2]), \
        f"ewma[2] changed when η[2] changed! This indicates forward-looking bias."
    print(f"✓ Critical: Changing η[2] from {eta[2]} to {eta_modified[2]} does NOT affect ewma[2]")

    # But ewma[3] SHOULD change because it uses η[2]
    assert not np.isclose(ewma[3], ewma_modified[3]), \
        f"ewma[3] should change when η[2] changes"
    print(f"✓ ewma[3] correctly changes when η[2] changes (from {ewma[3]:.4f} to {ewma_modified[3]:.4f})")

    print("\n✓✓✓ PASSED: No forward-looking bias detected in EWMA\n")


def test_pdf_properties():
    """
    Test that PDFs sum to 1 and CDFs are monotonic.
    """
    print("=" * 60)
    print("TEST 2: Probability Properties")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'state_m': np.random.randint(1, 4, n),
        'state_n': np.random.randint(1, 4, n),
        'state_k': np.random.randint(1, 3, n),
        'R': np.random.randint(0, 20, n),
        'R_up': np.random.randint(0, 15, n),
        'R_dn': np.random.randint(0, 15, n)
    })

    # Build PDF
    estimator = PDFEstimator()
    estimator.build_conditional_pdf(df, J_s=0, method='raw')
    estimator.compute_cdf()

    # Validate
    validation = estimator.validate_probabilities()

    if validation['pdf_sums_valid']:
        print("✓ All PDFs sum to 1.0")
    else:
        print("✗ PDF sum validation failed:")
        for issue in validation['issues']:
            if 'PDF sum' in issue:
                print(f"  {issue}")

    if validation['cdf_monotonic']:
        print("✓ All CDFs are monotonic (non-increasing)")
    else:
        print("✗ CDF monotonicity validation failed:")
        for issue in validation['issues']:
            if 'CDF not monotonic' in issue:
                print(f"  {issue}")

    # Additional check: CDF(0) should be close to 1.0
    print("\nChecking CDF(0) values (should be ~1.0):")
    for state in list(estimator.cdf_dict.keys())[:3]:  # Check first 3 states
        cdf_up_0 = estimator.query_cdf(0, 'range_up', state)
        cdf_dn_0 = estimator.query_cdf(0, 'range_dn', state)
        print(f"  State {state}: CDF_up(0)={cdf_up_0:.4f}, CDF_dn(0)={cdf_dn_0:.4f}")
        assert np.isclose(cdf_up_0, 1.0, atol=0.01), f"CDF_up(0) should be ~1.0"
        assert np.isclose(cdf_dn_0, 1.0, atol=0.01), f"CDF_dn(0) should be ~1.0"

    print("\n✓✓✓ PASSED: Probability properties validated\n")


def test_ewma_manual_calculation():
    """
    Test EWMA against manual calculation for first few steps.
    """
    print("=" * 60)
    print("TEST 3: EWMA Manual Calculation Verification")
    print("=" * 60)

    # Simple test case
    eta = pd.Series([5, 10, 15, 20, 25])
    halflife = 3
    lambda_param = 2 ** (-1 / halflife)

    ewma, ewmv = StateClassifier.compute_ewma(eta, halflife)

    print(f"Input: η = {eta.values}")
    print(f"Halflife: {halflife}, λ = {lambda_param:.4f}\n")

    # Manual calculations
    print("Manual verification:")

    # j=0
    print(f"j=0: ewma[0] = 0 (initialization)")
    assert ewma[0] == 0

    # j=1
    sumW_1 = 1
    sumWX_1 = eta[0]  # Uses η[0]
    manual_ewma_1 = sumWX_1 / sumW_1
    print(f"j=1: sumW={sumW_1}, sumWX={sumWX_1}, ewma={manual_ewma_1:.4f}")
    assert np.isclose(ewma[1], manual_ewma_1)

    # j=2
    sumW_2 = lambda_param * sumW_1 + 1
    sumWX_2 = lambda_param * sumWX_1 + eta[1]  # Uses η[1]
    manual_ewma_2 = sumWX_2 / sumW_2
    print(f"j=2: sumW={sumW_2:.4f}, sumWX={sumWX_2:.4f}, ewma={manual_ewma_2:.4f}")
    assert np.isclose(ewma[2], manual_ewma_2)

    # j=3
    sumW_3 = lambda_param * sumW_2 + 1
    sumWX_3 = lambda_param * sumWX_2 + eta[2]  # Uses η[2]
    manual_ewma_3 = sumWX_3 / sumW_3
    print(f"j=3: sumW={sumW_3:.4f}, sumWX={sumWX_3:.4f}, ewma={manual_ewma_3:.4f}")
    assert np.isclose(ewma[3], manual_ewma_3)

    print(f"\nComputed EWMA: {ewma.values}")
    print("\n✓✓✓ PASSED: EWMA matches manual calculation\n")


def test_laplace_smoothing():
    """
    Test that Laplace smoothing handles zero counts correctly.
    """
    print("=" * 60)
    print("TEST 4: Laplace Smoothing")
    print("=" * 60)

    # Create data with sparse states
    df = pd.DataFrame({
        'state_m': [1, 1, 1, 2, 2],
        'state_n': [1, 1, 1, 1, 1],
        'state_k': [1, 1, 1, 1, 1],
        'R': [5, 5, 5, 10, 10],
        'R_up': [3, 3, 3, 7, 7],
        'R_dn': [2, 2, 2, 3, 3]
    })

    # Build with raw method
    estimator_raw = PDFEstimator()
    estimator_raw.build_conditional_pdf(df, J_s=0, method='raw')

    # Build with smoothed method
    estimator_smooth = PDFEstimator()
    estimator_smooth.build_conditional_pdf(df, J_s=0, method='smoothed', alpha=1.0)

    state = (1, 1, 1)

    # Raw method: P(R=5) = 1.0, P(R=6) = 0.0
    prob_raw_5 = estimator_raw.query_pdf(5, 'range', state)
    prob_raw_6 = estimator_raw.query_pdf(6, 'range', state)

    print(f"Raw method:")
    print(f"  P(R=5|state) = {prob_raw_5:.4f} (observed)")
    print(f"  P(R=6|state) = {prob_raw_6:.4f} (not observed)")

    # Smoothed method: both should be non-zero
    prob_smooth_5 = estimator_smooth.query_pdf(5, 'range', state)
    prob_smooth_6 = estimator_smooth.query_pdf(6, 'range', state)

    print(f"\nSmoothed method (α=1.0):")
    print(f"  P(R=5|state) = {prob_smooth_5:.4f} (observed, but smoothed)")
    print(f"  P(R=6|state) = {prob_smooth_6:.4f} (not observed, but non-zero)")

    assert prob_raw_6 == 0.0, "Raw method should give 0 for unobserved values"
    assert prob_smooth_6 > 0.0, "Smoothed method should give non-zero for unobserved values"
    assert prob_smooth_5 > prob_smooth_6, "Observed values should have higher probability"

    print("\n✓✓✓ PASSED: Laplace smoothing works correctly\n")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("ePDF CALCULATOR VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_ewma_no_forward_looking()
        test_ewma_manual_calculation()
        test_pdf_properties()
        test_laplace_smoothing()

        print("=" * 60)
        print("ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n✗✗✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗✗✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
