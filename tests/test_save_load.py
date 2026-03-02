"""Test model persistence (save/load)."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from epdf import ePDFCalculator


def test_save_load():
    """Test that saved model can be loaded and produces identical results."""
    
    print("Testing save/load functionality...")
    
    # Train a model
    print("  Training model...")
    calc = ePDFCalculator(
        instrument='VG',
        tau=5,
        M=3, N=3, K=2,
        ewma_halflife=10,
        estimation_method='smoothed'
    )
    calc.fit('data/EuroStoxx/VGH22.csv')
    
    # Save
    model_path = '/tmp/test_epdf_model.pkl'
    print(f"  Saving to {model_path}...")
    calc.save(model_path)
    
    # Load
    print(f"  Loading from {model_path}...")
    calc_loaded = ePDFCalculator.load(model_path)
    
    # Verify model info matches
    info_orig = calc.get_model_info()
    info_loaded = calc_loaded.get_model_info()
    
    assert info_orig['instrument'] == info_loaded['instrument']
    assert info_orig['tau'] == info_loaded['tau']
    assert info_orig['n_states'] == info_loaded['n_states']
    print("  ✓ Model metadata matches")
    
    # Verify state classification matches
    test_inputs = [
        (100.0, 5.0, 1.0),
        (200.0, 10.0, -2.0),
        (150.0, 7.5, 0.5),
    ]
    
    for vol, sigma, delta in test_inputs:
        state_orig = calc.get_current_state(vol, sigma, delta)
        state_loaded = calc_loaded.get_current_state(vol, sigma, delta)
        assert state_orig == state_loaded, \
            f"State mismatch: {state_orig} != {state_loaded}"
    print("  ✓ State classification matches")
    
    # Verify query results match
    test_states = list(calc.pdf_estimator.pdf_dict.keys())[:5]
    for state in test_states:
        for ell in range(0, 10):
            prob_orig = calc.query_cdf(ell, 'range_dn', state)
            prob_loaded = calc_loaded.query_cdf(ell, 'range_dn', state)
            assert abs(prob_orig - prob_loaded) < 1e-10, \
                f"CDF mismatch at state={state}, ell={ell}"
    print("  ✓ Query results match")
    
    # Clean up
    os.remove(model_path)
    print("  ✓ Cleanup complete")
    
    print("\n✓✓✓ PASSED: Save/load works correctly\n")


if __name__ == '__main__':
    test_save_load()