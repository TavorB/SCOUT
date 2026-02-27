"""
Minimal smoke tests for SCOUT_cleaned.py.

Run with pytest:
    cd github_code/
    pytest test_scout.py -v

Or standalone:
    python test_scout.py
"""

import os
import sys
import numpy as np

# Ensure tests run from the github_code/ directory so that figures/ is created there.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Suppress matplotlib display in headless environments.
import matplotlib
matplotlib.use('Agg')


# ---------------------------------------------------------------------------
# Helpers imported from the module under test
# ---------------------------------------------------------------------------
from github_code.SCOUT import (
    logistic,
    draw_unit_ball_vector,
    generate_synthetic_data,
    compute_theta_est_cvx,
    compute_theta_mle,
    compute_tau_opt,
    SCOUTAlgorithm,
    run_scout_experiment,
    eval_on_real_data,
    validate_on_synthetic_data,
    generate_paper_figures,
)


# ---------------------------------------------------------------------------
# Test 1: public API imports
# ---------------------------------------------------------------------------

def test_imports():
    """All public API symbols are importable."""
    # If we reached this point, all imports above succeeded.
    assert callable(generate_paper_figures)
    assert callable(eval_on_real_data)
    assert callable(generate_synthetic_data)
    assert callable(validate_on_synthetic_data)
    assert callable(run_scout_experiment)


# ---------------------------------------------------------------------------
# Test 2: utility functions
# ---------------------------------------------------------------------------

def test_utility_functions():
    """Utility math and data-generation functions return correct types/shapes."""
    # logistic
    assert abs(logistic(0) - 0.5) < 1e-10
    assert logistic(100) > 0.999
    assert logistic(-100) < 0.001

    # draw_unit_ball_vector
    v = draw_unit_ball_vector(4)
    assert v.shape == (4,)
    assert np.linalg.norm(v) <= 1.0 + 1e-9

    # generate_synthetic_data
    X, Y, theta = generate_synthetic_data(d=3, n=100, true_theta=None, S=1, seed=0)
    assert X.shape == (100, 3)
    assert Y.shape == (100,)
    assert set(np.unique(Y)).issubset({0, 1})
    assert theta.shape == (3,)

    # compute_theta_est_cvx — non-empty input
    theta_est = compute_theta_est_cvx(X[:20], Y[:20].astype(float), t=20,
                                      looseness_factor_theta_est=1.0)
    assert theta_est is not None, "Expected ndarray, got None for non-empty input"
    assert theta_est.shape == (3,)

    # compute_theta_est_cvx — empty input (tests the bug-fix: must return None, not a tuple)
    theta_est_empty = compute_theta_est_cvx(np.empty((0, 3)), np.empty(0), t=0,
                                            looseness_factor_theta_est=1.0)
    assert theta_est_empty is None, "Expected None for empty input"

    # compute_theta_mle
    theta_mle = compute_theta_mle(X[:50], Y[:50].astype(float))
    assert theta_mle is not None
    assert theta_mle.shape == (3,)

    # compute_tau_opt
    tau = compute_tau_opt(theta, X[:50], alpha=0.05)
    assert np.isscalar(tau) or (isinstance(tau, np.ndarray) and tau.ndim == 0)
    assert float(tau) >= 0


# ---------------------------------------------------------------------------
# Test 3: SCOUTAlgorithm end-to-end
# ---------------------------------------------------------------------------

def test_scout_algorithm_run():
    """SCOUTAlgorithm runs T rounds and returns a well-formed result dict."""
    d, T = 2, 300
    true_theta = np.array([1.0, 0.0])
    scout = SCOUTAlgorithm(d=d, T=T, alpha=0.1, delta=0.1,
                           true_theta=true_theta, S=1)
    scout.run()
    results = scout.evaluate()

    expected_keys = {'error_rate', 'num_tests', 'test_rate',
                     'error_threshold', 'final_theta', 'true_theta'}
    assert expected_keys.issubset(results.keys()), \
        f"Missing keys: {expected_keys - results.keys()}"

    assert 0.0 <= results['error_rate'] <= 1.0
    assert 0.0 <= results['test_rate'] <= 1.0
    assert 0 <= results['num_tests'] <= T
    assert results['error_threshold'] == 0.1
    assert results['final_theta'] is not None
    assert results['final_theta'].shape == (d,)
    assert results['true_theta'].shape == (d,)

    # Trajectory arrays have the right length
    assert len(scout.all_decisions) == T
    assert len(scout.all_predictions) == T
    assert len(scout.all_labels) == T


# ---------------------------------------------------------------------------
# Test 4: run_scout_experiment
# ---------------------------------------------------------------------------

def test_run_scout_experiment():
    """run_scout_experiment returns the expected dict and saves a .npz file."""
    d, T = 2, 300
    true_theta = np.array([1.0, 0.0])
    results = run_scout_experiment(d=d, T=T, alpha=0.1, delta=0.1, S=1,
                                   true_theta=true_theta, num_runs=1)

    expected_keys = {'error_rate', 'error_std', 'test_rate', 'test_rate_std',
                     'num_tests', 'num_tests_std', 'error_threshold'}
    assert expected_keys.issubset(results.keys()), \
        f"Missing keys: {expected_keys - results.keys()}"

    assert 0.0 <= results['error_rate'] <= 1.0
    assert 0.0 <= results['test_rate'] <= 1.0
    assert results['error_threshold'] == 0.1

    # .npz file should have been created
    npz_path = f"figures/SCOUT_results_d={d}_T={T}_alpha=0.1.npz"
    assert os.path.isfile(npz_path), f"Expected .npz output at {npz_path}"


# ---------------------------------------------------------------------------
# Test 5: eval_on_real_data
# ---------------------------------------------------------------------------

def test_eval_on_real_data():
    """eval_on_real_data returns correct dict with right array shapes."""
    X, Y, _ = generate_synthetic_data(d=2, n=300, true_theta=None, S=1, seed=0)
    results = eval_on_real_data(X, Y, alpha=0.1, num_perms=2, delta=0.1, S=1)

    expected_keys = {'theta_star', 'tau_star', 'opt_test_rate',
                     'all_cum_tests', 'all_cum_errors',
                     'all_cum_tests_opt', 'all_cum_errors_opt'}
    assert expected_keys.issubset(results.keys()), \
        f"Missing keys: {expected_keys - results.keys()}"

    assert results['theta_star'].shape == (2,)
    assert float(results['tau_star']) >= 0
    assert 0.0 <= float(results['opt_test_rate']) <= 1.0
    assert results['all_cum_tests'].shape == (2, 300)
    assert results['all_cum_errors'].shape == (2, 300)
    assert results['all_cum_tests_opt'].shape == (2, 300)
    assert results['all_cum_errors_opt'].shape == (2, 300)


# ---------------------------------------------------------------------------
# Test 6: validate_on_synthetic_data
# ---------------------------------------------------------------------------

def test_validate_on_synthetic_data():
    """validate_on_synthetic_data runs the full real-data pipeline on known data."""
    results = validate_on_synthetic_data(d=2, n=300, alpha=0.1,
                                         num_perms=2, delta=0.1)
    assert 'true_theta' in results, "'true_theta' key missing from result"
    assert results['true_theta'].shape == (2,)
    # Inherits all keys from eval_on_real_data
    assert 'theta_star' in results
    assert 'tau_star' in results


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

_TEST_OUTPUT_FILES = [
    "figures/SCOUT_results_d=2_T=300_alpha=0.1.npz",
]

def cleanup_test_outputs():
    for path in _TEST_OUTPUT_FILES:
        if os.path.isfile(path):
            os.remove(path)


# pytest session-scoped fixture: runs cleanup after all tests finish.
try:
    import pytest

    @pytest.fixture(scope="session", autouse=True)
    def _cleanup_after_session():
        yield
        cleanup_test_outputs()

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_imports,
        test_utility_functions,
        test_scout_algorithm_run,
        test_run_scout_experiment,
        test_eval_on_real_data,
        test_validate_on_synthetic_data,
    ]

    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            print(f"FAIL  {test.__name__}: {exc}")
            failures.append(test.__name__)

    cleanup_test_outputs()

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} test(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
