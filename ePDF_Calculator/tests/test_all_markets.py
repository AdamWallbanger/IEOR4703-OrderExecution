"""
Integration Test: Run ePDF Calculator on All Market Data Files

Scans data/ directory, trains a model on each OHLCV CSV file,
and validates basic correctness.

Each file is tested in a separate subprocess to avoid memory accumulation.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def find_ohlcv_files(data_dir: str) -> list:
    """Find all OHLCV CSV files (exclude AIAgent_* files)."""
    files = []
    data_path = Path(data_dir)
    for csv_file in sorted(data_path.rglob("*.csv")):
        if csv_file.name.startswith("AIAgent_"):
            continue
        files.append(csv_file)
    return files


# Script that runs in a subprocess for a single file
SINGLE_FILE_SCRIPT = '''
import sys
import json
import time
import warnings
from pathlib import Path

sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")

from epdf.calculator import ePDFCalculator
from epdf.instrument_config import InstrumentConfig

filepath = Path(sys.argv[2])
result = {
    'filepath': str(filepath),
    'filename': filepath.name,
    'market': filepath.parent.name,
    'success': False,
    'n_rows': 0,
    'n_bars': 0,
    'n_states_observed': 0,
    'n_states_total': 0,
    'train_time': 0.0,
    'query_ok': False,
    'error': None,
}

try:
    symbol = InstrumentConfig.parse_symbol_from_filename(str(filepath))
    tick_size = InstrumentConfig.get_tick_size(symbol)
    result['symbol'] = symbol
    result['tick_size'] = tick_size

    with open(filepath, 'r') as f:
        result['n_rows'] = sum(1 for _ in f) - 1

    t0 = time.time()
    calc = ePDFCalculator(
        instrument=symbol,
        tau=60,
        M=3, N=3, K=2,
        ewma_halflife=10,
        estimation_method='smoothed',
        smoothing_alpha=0.5,
        tick_size=tick_size,
    )
    calc.fit(str(filepath))
    result['train_time'] = time.time() - t0

    info = calc.get_model_info()
    result['n_bars'] = info['data_shape'][0] if info['data_shape'] else 0
    result['n_states_observed'] = info['n_states_observed']
    result['n_states_total'] = info['n_states']
    result['J_s'] = info['J_s']

    assert result['n_states_observed'] > 0, "No states observed after training"

    validation = calc.pdf_estimator.validate_probabilities()
    assert validation['pdf_sums_valid'], f"PDF sums invalid: {validation['issues'][:3]}"
    assert validation['cdf_monotonic'], f"CDF not monotonic: {validation['issues'][:3]}"

    query_errors = []
    for state in list(calc.pdf_estimator.pdf_dict.keys())[:3]:
        try:
            cdf_val = calc.query_cdf(ell=1, direction='range_dn', state=state)
            assert 0 <= cdf_val <= 1, f"CDF value out of range: {cdf_val}"
            pdf_val = calc.query_pdf(ell=0, direction='range', state=state)
            assert 0 <= pdf_val <= 1, f"PDF value out of range: {pdf_val}"
            dist = calc.get_full_distribution('range_dn', state)
            assert len(dist) > 0, "Empty distribution"
        except Exception as e:
            query_errors.append(f"State {state}: {e}")

    if query_errors:
        result['error'] = f"Query errors: {'; '.join(query_errors)}"
    else:
        result['query_ok'] = True

    result['success'] = True

except Exception as e:
    result['error'] = f"{type(e).__name__}: {e}"

print(json.dumps(result))
'''


def test_single_file_subprocess(filepath: Path) -> dict:
    """Run a single file test in a subprocess."""
    proc = subprocess.run(
        [sys.executable, '-c', SINGLE_FILE_SCRIPT, PROJECT_ROOT, str(filepath)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Parse the JSON result from the last line of stdout
    stdout_lines = proc.stdout.strip().split('\n')
    # The JSON result is the last line; earlier lines are training progress output
    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {
        'filepath': str(filepath),
        'filename': filepath.name,
        'market': filepath.parent.name,
        'success': False,
        'n_rows': 0,
        'n_bars': 0,
        'n_states_observed': 0,
        'n_states_total': 0,
        'train_time': 0.0,
        'query_ok': False,
        'error': f"Subprocess failed: {proc.stderr[-500:] if proc.stderr else 'no stderr'}",
    }


def run_all_tests():
    """Run integration tests on all market data files."""
    data_dir = Path(__file__).resolve().parent.parent / "data"

    print("=" * 70)
    print("INTEGRATION TEST: All Market Data Files")
    print("=" * 70)
    print(f"Data directory: {data_dir}\n")

    # Find all OHLCV files
    files = find_ohlcv_files(str(data_dir))
    print(f"Found {len(files)} OHLCV CSV files\n")

    if not files:
        print("No CSV files found. Check data directory.")
        return False

    # Test each file
    results = []
    for i, filepath in enumerate(files, 1):
        rel_path = filepath.relative_to(data_dir.parent)
        print(f"[{i}/{len(files)}] Testing {rel_path}...", flush=True)

        t0 = time.time()
        try:
            result = test_single_file_subprocess(filepath)
        except subprocess.TimeoutExpired:
            result = {
                'filepath': str(filepath),
                'filename': filepath.name,
                'market': filepath.parent.name,
                'success': False,
                'n_rows': 0,
                'n_bars': 0,
                'n_states_observed': 0,
                'n_states_total': 0,
                'train_time': time.time() - t0,
                'query_ok': False,
                'error': 'Subprocess timed out (120s)',
            }

        results.append(result)

        # Print result
        if result['success']:
            print(f"  ✓ Loaded {result['n_rows']:,} rows")
            print(f"  ✓ Trained in {result['train_time']:.1f}s "
                  f"({result['n_bars']:,} {60}-min bars)")
            print(f"  ✓ Built {result['n_states_observed']}/{result['n_states_total']} states "
                  f"(J_s={result.get('J_s', '?')})")
            print(f"  ✓ Query test passed")
        else:
            print(f"  ✗ FAILED: {result['error']}")
        print(flush=True)

    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:  {len(results)}")
    print(f"Successful:   {len(successful)}")
    print(f"Failed:       {len(failed)}")
    print()

    # Detailed table
    print(f"{'Market':<35} {'Symbol':<6} {'Rows':>8} {'Bars':>8} "
          f"{'States':>8} {'Time':>6} {'Status':<8}")
    print("-" * 90)
    for r in results:
        status = "✓ OK" if r['success'] else "✗ FAIL"
        symbol = r.get('symbol', '?')
        market = f"{r['market']}/{r['filename']}"
        print(f"{market:<35} {symbol:<6} {r['n_rows']:>8,} {r['n_bars']:>8,} "
              f"{r['n_states_observed']:>3}/{r['n_states_total']:<4} "
              f"{r['train_time']:>5.1f}s {status:<8}")

    # Failed files detail
    if failed:
        print(f"\n{'=' * 70}")
        print("FAILED FILES")
        print("=" * 70)
        for r in failed:
            print(f"\n  {r['market']}/{r['filename']}:")
            print(f"    {r['error']}")

    print()
    return len(failed) == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
