"""
SCOUT: Safe Classification Under Online Uncertainty Testing
===========================================================

Code to reproduce all figures from:
  "The Good, the Bad, and the Sampled: a No-Regret Approach to Safe Online Classification"
  AISTATS 2026.

Public API
----------
generate_paper_figures()
    Reproduce the three synthetic experiments from the paper.

eval_on_real_data(X, Y, alpha, num_perms, delta, S)
    Evaluate SCOUT on a real labelled dataset by treating it as an online stream.

generate_synthetic_data(d, n, true_theta, S, seed)
    Generate synthetic logistic regression data (contexts + binary labels).

validate_on_synthetic_data(d, n, alpha, true_theta, S, num_perms, delta, seed)
    Generate synthetic data and run eval_on_real_data on it for end-to-end validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def logistic(x):
    """Return the logistic (sigmoid) function value 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def draw_2d_unit_ball_vector():
    """
    Draw a random 2-D vector from the unit disk, heavily biased toward the boundary.

    The radius is drawn as U^0.01 where U ~ Uniform(0, 1), which concentrates
    almost all mass near r = 1 (i.e. on the unit circle).  This mimics a
    distribution whose support is essentially the circle boundary.
    """
    r = np.random.uniform(0, 1) ** 0.01
    theta = np.random.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def draw_unit_ball_vector(d, bias_towards_edge=True):
    """
    Draw a random d-dimensional vector from the unit ball.

    Parameters
    ----------
    d : int
        Dimensionality.
    bias_towards_edge : bool
        If True, use radius r = U^0.01 (strong bias toward the unit sphere
        surface).  If False, use r = U^(1/d) which gives a uniform distribution
        over the ball volume.

    Returns
    -------
    np.ndarray, shape (d,)
    """
    u = np.random.normal(0, 1, d)
    u = u / np.linalg.norm(u)
    if bias_towards_edge:
        r = np.random.uniform(0, 1) ** 0.01
    else:
        r = np.random.uniform(0, 1) ** (1.0 / d)
    return r * u


def log_loss(theta, X, y, lambd):
    """
    Regularised logistic log-loss.

    Parameters
    ----------
    theta : np.ndarray, shape (d,)
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)  — binary labels in {0, 1}
    lambd : float — L2 regularisation coefficient

    Returns
    -------
    float
    """
    mu = 1 / (1 + np.exp(-X @ theta))
    loss = -np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)) + lambd * np.linalg.norm(theta) ** 2
    return loss


def gradient_log_loss(theta, X, y, lambd):
    """Gradient of the regularised log-loss with respect to theta."""
    mu = 1 / (1 + np.exp(-X @ theta))
    return X.T @ (mu - y) + lambd * theta


def hessian_log_loss(theta, X, lambd):
    """Hessian of the regularised log-loss with respect to theta."""
    mu = 1 / (1 + np.exp(-X @ theta))
    W = np.diag(mu * (1 - mu))
    return X.T @ W @ X + lambd * np.eye(len(theta))


def compute_theta_est_cvx(X, y, t, looseness_factor_theta_est=1):
    """
    Estimate theta by minimising the regularised log-loss (L-BFGS-B).

    The regularisation parameter follows the schedule lambda_t = d * log(t+1),
    consistent with the theoretical analysis in the paper.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)  — binary labels
    t : int — current round (controls regularisation strength)
    looseness_factor_theta_est : float
        Divides lambda_t; values > 1 reduce regularisation (default 1).

    Returns
    -------
    np.ndarray, shape (d,)  — estimated theta, or zeros if X is empty.
    """
    if X.shape[0] == 0:
        return None

    d = X.shape[1]
    lambd = d * np.log(t + 1) / looseness_factor_theta_est

    try:
        result = minimize(log_loss, np.zeros(d), args=(X, y, lambd),
                          method='L-BFGS-B', jac=gradient_log_loss)
        theta_hat_t = result.x
    except Exception:
        theta_hat_t = np.zeros(d)

    return theta_hat_t


def compute_theta_mle(X, y):
    """
    Fit the exact (unregularized) logistic regression MLE via L-BFGS-B.

    Uses lambd=1e-8 for numerical stability (effectively zero regularization).
    Intended for oracle / reference fits where bias from regularization is
    undesirable (e.g. computing theta_star in eval_on_real_data).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,) — binary labels in {0, 1}

    Returns
    -------
    np.ndarray, shape (d,) — MLE estimate, or None if X is empty.
    """
    if X.shape[0] == 0:
        return None
    d = X.shape[1]
    lambd = 1e-8
    try:
        result = minimize(log_loss, np.zeros(d), args=(X, y, lambd),
                          method='L-BFGS-B', jac=gradient_log_loss)
        return result.x
    except Exception:
        return np.zeros(d)


def compute_tau_opt(theta, contexts, alpha):
    """
    Compute the smallest threshold tau such that the empirical error rate is <= alpha.

    For a threshold policy with parameter tau, a sample x is *not* tested when
    |x^T theta| > tau, and the prediction error probability is 1/(1+exp(|x^T theta|)).
    This function finds the smallest tau such that the cumulative error from
    untested samples does not exceed alpha * n.

    Parameters
    ----------
    theta : np.ndarray, shape (d,)
    contexts : np.ndarray, shape (n, d)
    alpha : float — target error rate

    Returns
    -------
    float — optimal threshold tau
    """
    if len(contexts) == 0 or alpha <= 0:
        return 1.0

    t = contexts.shape[0]
    dot_products = np.abs(contexts @ theta)
    error_probs = 1 / (1 + np.exp(dot_products))

    sorting = np.argsort(dot_products)[::-1]
    error_probs_sorted = error_probs[sorting]
    error_probs_cumsum = np.cumsum(error_probs_sorted)

    indices = np.where(error_probs_cumsum > t * alpha)[0]
    if len(indices) == 0:
        tau_opt = np.min(dot_products)
    else:
        idx = indices[0]
        tau_opt = dot_products[sorting[idx - 1]] if idx > 0 else np.min(dot_products)

    return tau_opt


def generate_synthetic_data(d, n, true_theta=None, S=1, seed=None):
    """
    Generate synthetic logistic regression data.

    Parameters
    ----------
    d : int — feature dimensionality
    n : int — number of samples
    true_theta : np.ndarray, shape (d,) or None
        True logistic parameter.  If None, defaults to ones(d)/sqrt(d) * S.
    S : float — norm of true_theta when auto-generated (default 1)
    seed : int or None — random seed

    Returns
    -------
    X : np.ndarray, shape (n, d)
    Y : np.ndarray, shape (n,)
    true_theta : np.ndarray, shape (d,)
    """
    if seed is not None:
        np.random.seed(seed)
    if true_theta is None:
        true_theta = np.ones(d) / np.sqrt(d) * S
    if d == 2:
        X = np.array([draw_2d_unit_ball_vector() for _ in range(n)])
    else:
        X = np.array([draw_unit_ball_vector(d) for _ in range(n)])
    probs = logistic(X @ true_theta)
    Y = np.random.binomial(1, probs)
    return X, Y, true_theta


# ---------------------------------------------------------------------------
# SCOUT algorithm
# ---------------------------------------------------------------------------

class SCOUTAlgorithm:
    """
    Online safe classification via SCOUT.

    SCOUT maintains confidence sets for the unknown logistic parameter theta*
    and distribution P, and at each round decides whether to test a patient
    (observe the true label) or predict without testing.  It guarantees that
    the cumulative misclassification rate stays below alpha with probability
    at least 1 - delta.

    This class is designed for *simulated* experiments where true_theta is
    known.  For real data use eval_on_real_data() instead.
    """

    def __init__(self, d, T, alpha, delta, true_theta, S=1,
                 looseness_factor_beta=500, looseness_factor_theta_est=1):
        """
        Parameters
        ----------
        d : int — feature dimensionality
        T : int — total number of rounds
        alpha : float — target misclassification rate
        delta : float — failure probability (safety holds with prob >= 1-delta)
        true_theta : np.ndarray, shape (d,) — true parameter (simulation only)
        S : float — assumed upper bound on ||theta*||_2 (default 1)
        looseness_factor_beta : float
            Divisor applied to the theoretical confidence radius beta_t.
            Larger values → tighter radius → fewer tests (default 500).
        looseness_factor_theta_est : float
            Divisor applied to the regularisation strength in compute_theta_est_cvx.
            Larger values → weaker regularisation → faster convergence of theta_est
            (default 1).
        """
        self.d = d
        self.T = T
        self.alpha = alpha
        self.delta = delta
        self.S = S
        self.looseness_factor_beta = looseness_factor_beta
        self.looseness_factor_theta_est = looseness_factor_theta_est

        self.CS_P = []
        self.CS_theta_X = []
        self.CS_theta_y = []
        self.N_P = 0
        self.N_theta = 0

        self.all_contexts = []
        self.all_labels = []
        self.all_decisions = []
        self.all_predictions = []

        self.all_theta_ests = []
        self.all_lambds = []
        self.all_diam_bounds = []

        self.true_theta = true_theta

    def compute_on_round(self, t):
        """
        Return True if theta/tau estimates should be refreshed on round t.

        Uses a tapering schedule: updates are frequent early (every round for
        t < 50, every 10 for t < 3000) and become sparser as t grows (every
        500 rounds for t >= 20000).  This amortises the cost of re-solving the
        convex optimisation problem.

        Parameters
        ----------
        t : int — current round (0-indexed)

        Returns
        -------
        bool
        """
        if t < 50:
            return True
        if t < 3000:
            return t % 10 == 0
        if t < 6000:
            return t % 40 == 0
        if t < 10000:
            return t % 100 == 0
        if t < 20000:
            return t % 200 == 0
        return t % 500 == 0

    def run(self):
        """Run the SCOUT algorithm for T rounds."""
        theta_est = np.zeros(self.d)
        tau_est = 0.0

        for t in tqdm(range(self.T)):
            Xt = self._generate_context()
            self.all_contexts.append(Xt)

            if t < 50:
                # Warmup: always test.
                Zt = 1
                diam_bound = np.inf
            else:
                lambd = self.d * np.log(t + 1)
                gamma_t = (np.sqrt(lambd)
                           + np.sqrt(self.d * np.log(1 + t / (16 * self.d * lambd))
                                     + np.log(4 * self.T / self.delta)))
                beta_t = (gamma_t + gamma_t ** 2 / np.sqrt(lambd)) / self.looseness_factor_beta
                diam_bound = np.sqrt(2 * beta_t ** 2 / self.N_theta) if self.N_theta > 0 else np.inf

                if (theta_est == 0).all() or self.compute_on_round(t):
                    theta_est = compute_theta_est_cvx(
                        np.array(self.CS_theta_X),
                        np.array(self.CS_theta_y),
                        t,
                        self.looseness_factor_theta_est,
                    )
                    tau_est = compute_tau_opt(
                        theta_est, np.array(self.CS_P), self.alpha - diam_bound
                    )

                Zt = 1 if np.abs(Xt @ theta_est) < tau_est + diam_bound else 0

            self.all_decisions.append(Zt)

            Yt = self._generate_label(Xt)
            self.all_labels.append(Yt)

            if Zt == 1:
                self.all_predictions.append(Yt)
                if t % 2 == 0:  # even tested rounds → theta estimation set
                    self.CS_theta_X.append(Xt)
                    self.CS_theta_y.append(Yt)
                    self.N_theta += 1
            else:
                Y_hat_t = 1 if np.dot(Xt, theta_est) > 0 else 0
                self.all_predictions.append(Y_hat_t)

            if t % 2 == 1:  # odd rounds → distribution estimation set
                self.CS_P.append(Xt)
                self.N_P += 1

            self.all_theta_ests.append(theta_est.copy() if not (theta_est == 0).all() else None)
            self.all_lambds.append(self.d * np.log(t + 1) / self.looseness_factor_theta_est)
            self.all_diam_bounds.append(diam_bound)

    def _generate_context(self):
        """Sample a context vector from the synthetic distribution (simulation only)."""
        if self.d == 2:
            return draw_2d_unit_ball_vector()
        else:
            return draw_unit_ball_vector(self.d)

    def _generate_label(self, x):
        """Sample a binary label from the logistic model (simulation only)."""
        p = logistic(np.dot(x, self.true_theta))
        return np.random.binomial(1, p)

    def compute_opt_testing_frac(self, theta_true, alpha, num_contexts_to_sim=100_000):
        """
        Estimate the oracle threshold tau* and optimal testing fraction p*.

        Uses a large fresh sample from the context distribution to approximate
        the continuous integrals in the paper's definition of tau* and p*.

        Parameters
        ----------
        theta_true : np.ndarray, shape (d,)
        alpha : float — target error rate
        num_contexts_to_sim : int — number of Monte-Carlo contexts (default 100 000)

        Returns
        -------
        tau_opt : float
        optimal_testing_fraction : float — fraction of contexts with |x^T theta| <= tau_opt
        """
        if alpha <= 0:
            return 1.0, 1.0

        contexts = np.array([self._generate_context() for _ in range(num_contexts_to_sim)])
        t = num_contexts_to_sim

        dot_products = np.abs(contexts @ theta_true)
        error_probs = 1 / (1 + np.exp(dot_products))

        sorting = np.argsort(dot_products)[::-1]
        error_probs_sorted = error_probs[sorting]
        error_probs_cumsum = np.cumsum(error_probs_sorted)

        indices = np.where(error_probs_cumsum > t * alpha)[0]

        if len(indices) == 0:
            tau_opt = np.min(dot_products)
            optimal_testing_fraction = 0.0
        else:
            idx = indices[0]
            tau_opt = dot_products[sorting[idx - 1]] if idx > 0 else np.min(dot_products)
            optimal_testing_fraction = np.mean(dot_products < tau_opt)

        return tau_opt, optimal_testing_fraction

    def evaluate(self):
        """
        Summarise algorithm performance after run() has completed.

        Returns
        -------
        dict with keys:
            error_rate, num_tests, test_rate, error_threshold,
            final_theta, true_theta
        """
        errors = sum(pred != true for pred, true in zip(self.all_predictions, self.all_labels))
        error_rate = errors / self.T
        num_tests = int(np.sum(self.all_decisions))
        test_rate = num_tests / self.T

        if self.N_theta > 0:
            final_theta = compute_theta_est_cvx(
                np.array(self.CS_theta_X),
                np.array(self.CS_theta_y),
                self.N_theta,
                self.looseness_factor_theta_est,
            )
        else:
            final_theta = np.zeros(self.d)

        return {
            'error_rate': error_rate,
            'num_tests': num_tests,
            'test_rate': test_rate,
            'error_threshold': self.alpha,
            'final_theta': final_theta,
            'true_theta': self.true_theta,
        }


# ---------------------------------------------------------------------------
# Real-data adapter (private)
# ---------------------------------------------------------------------------

class _RealDataSCOUT(SCOUTAlgorithm):
    """
    SCOUTAlgorithm variant that streams from a pre-loaded dataset.

    Overrides _generate_context and _generate_label so that run() iterates
    through the provided arrays X_perm, Y_perm in order.  All other algorithm
    logic (confidence radii, sample splitting, decision rule) is inherited
    unchanged from SCOUTAlgorithm.
    """

    def __init__(self, X_perm, Y_perm, alpha, delta, S=1,
                 looseness_factor_beta=500, looseness_factor_theta_est=1):
        n, d = X_perm.shape
        # true_theta=None: the algorithm does not know the ground-truth theta
        super().__init__(d=d, T=n, alpha=alpha, delta=delta, true_theta=None, S=S,
                         looseness_factor_beta=looseness_factor_beta,
                         looseness_factor_theta_est=looseness_factor_theta_est)
        self._stream_X = X_perm
        self._stream_Y = Y_perm
        self._stream_idx = 0

    def _generate_context(self):
        """Return the next context from the pre-shuffled stream."""
        return self._stream_X[self._stream_idx]

    def _generate_label(self, x):
        """Return the next label from the pre-shuffled stream and advance the index."""
        lbl = self._stream_Y[self._stream_idx]
        self._stream_idx += 1
        return lbl


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aggregate_results(all_cum_tests, all_cum_errors, all_cum_tests_opt,
                            all_cum_errors_opt, alpha, opt_test_rate,
                            label_prefix="SCOUT", fig_fname=None, out_dir="figures"):
    """
    Plot mean and 10-90 percentile bands for cumulative test/error rates.

    Creates a 3-panel figure:
      Left   — cumulative test rate (SCOUT vs oracle)
      Middle — excess tests over oracle (regret)
      Right  — cumulative error rate vs target alpha

    The figure is saved to {out_dir}/{fig_fname or label_prefix}_Results_aggregate.pdf.

    Parameters
    ----------
    all_cum_tests : array-like, shape (num_runs, T)
        Cumulative test rate per round for each run.
    all_cum_errors : array-like, shape (num_runs, T)
        Cumulative error rate per round for each run.
    all_cum_tests_opt : array-like, shape (num_runs, T)
        Oracle cumulative test rate per round for each run.
    all_cum_errors_opt : array-like, shape (num_runs, T)
        Oracle cumulative error rate per round for each run.
    alpha : float — target error rate (drawn as horizontal reference line)
    opt_test_rate : float — oracle steady-state test fraction (reference line)
    label_prefix : str — prefix used for legend labels and default filename
    fig_fname : str or None — base filename (without extension) for the saved PDF;
        defaults to ``{label_prefix}_Results_aggregate`` if None
    out_dir : str — directory where the PDF is saved (default "figures")
    """
    all_cum_tests = np.array(all_cum_tests)
    all_cum_errors = np.array(all_cum_errors)
    all_cum_tests_opt = np.array(all_cum_tests_opt)
    all_cum_errors_opt = np.array(all_cum_errors_opt)
    num_runs, T = all_cum_tests.shape

    mean_cum_tests = np.mean(all_cum_tests, axis=0)
    q10_cum_tests = np.quantile(all_cum_tests, 0.10, axis=0)
    q90_cum_tests = np.quantile(all_cum_tests, 0.90, axis=0)

    mean_cum_errors = np.mean(all_cum_errors, axis=0)
    q10_cum_errors = np.quantile(all_cum_errors, 0.10, axis=0)
    q90_cum_errors = np.quantile(all_cum_errors, 0.90, axis=0)

    mean_cum_tests_opt = np.mean(all_cum_tests_opt, axis=0)
    q10_cum_tests_opt = np.quantile(all_cum_tests_opt, 0.10, axis=0)
    q90_cum_tests_opt = np.quantile(all_cum_tests_opt, 0.90, axis=0)

    mean_cum_errors_opt = np.mean(all_cum_errors_opt, axis=0)
    q10_cum_errors_opt = np.quantile(all_cum_errors_opt, 0.10, axis=0)
    q90_cum_errors_opt = np.quantile(all_cum_errors_opt, 0.90, axis=0)

    all_num_tests = all_cum_tests * np.arange(1, T + 1)
    all_num_tests_opt = all_cum_tests_opt * np.arange(1, T + 1)
    mean_excess = np.mean(all_num_tests - all_num_tests_opt, axis=0)
    q10_excess = np.quantile(all_num_tests - all_num_tests_opt, 0.10, axis=0)
    q90_excess = np.quantile(all_num_tests - all_num_tests_opt, 0.90, axis=0)

    x = np.arange(T)
    scout_color = "C0"
    oracle_color = "C2"

    plt.figure(figsize=(18, 6))

    # Left: cumulative test rate
    plt.subplot(1, 3, 1)
    plt.plot(x, mean_cum_tests, label=f"{label_prefix} mean", color=scout_color)
    plt.fill_between(x, q10_cum_tests, q90_cum_tests, color=scout_color, alpha=0.3,
                     label="10-90% quantile")
    plt.plot(x, mean_cum_tests_opt, label="Oracle mean", color=oracle_color)
    plt.fill_between(x, q10_cum_tests_opt, q90_cum_tests_opt, color=oracle_color, alpha=0.3,
                     label="Oracle 10-90% quantile")
    plt.axhline(y=opt_test_rate, color="orange", linestyle='--',
                label=f'Oracle test rate={opt_test_rate:.2f}')
    plt.xlabel("Time")
    plt.ylabel("Cumulative Test Rate")
    plt.title("Cumulative Test Rate Across Runs")
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0.9 * opt_test_rate, top=1)

    # Middle: excess tests (regret)
    plt.subplot(1, 3, 2)
    plt.plot(x, mean_excess, label="Mean Excess Tests", color="C4")
    plt.fill_between(x, q10_excess, q90_excess, color="C4", alpha=0.3,
                     label="10-90% quantile")
    plt.xlabel("Time")
    plt.ylabel("Excess Tests (Regret)")
    plt.title("Excess Tests Over Oracle Baseline")
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Right: cumulative error rate
    plt.subplot(1, 3, 3)
    plt.plot(x, mean_cum_errors, label=f"{label_prefix} mean", color=scout_color)
    plt.fill_between(x, q10_cum_errors, q90_cum_errors, color=scout_color, alpha=0.3,
                     label="10-90% quantile")
    plt.plot(x, mean_cum_errors_opt, label="Oracle mean", color=oracle_color)
    plt.fill_between(x, q10_cum_errors_opt, q90_cum_errors_opt, color=oracle_color, alpha=0.3,
                     label="Oracle 10-90% quantile")
    plt.axhline(y=alpha, color="orange", linestyle='--', label=f'Target α={alpha}')
    plt.xlabel("Time")
    plt.ylabel("Cumulative Error Rate")
    plt.title("Cumulative Error Rate Across Runs")
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0,
             top=max(np.max(mean_cum_errors), np.max(mean_cum_errors_opt), alpha) * 1.1)

    plt.tight_layout()
    fig_title = f"{fig_fname}.pdf" if fig_fname is not None else f"{label_prefix}_Results_aggregate.pdf"
    plt.savefig(str(Path(out_dir) / fig_title), bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_scout_experiment(d=2, T=1000, alpha=0.05, delta=0.01, S=1,
                          true_theta=None, num_runs=1, aggressiveness=10.0, debug=False,
                          out_dir="figures"):
    """
    Run the SCOUT synthetic experiment and aggregate results across runs.

    Parameters
    ----------
    d : int — feature dimensionality
    T : int — number of rounds
    alpha : float — target error rate
    delta : float — failure probability
    S : float — assumed upper bound on ||theta*||_2
    true_theta : np.ndarray, shape (d,) or None
        Ground-truth parameter.  If None, defaults to the unit vector along
        the first axis.
    num_runs : int — number of independent runs (seeds 0, 1, ..., num_runs-1)
    aggressiveness : float - (default 1.0).  Higher values → more aggressive testing (tighter confidence sets, weaker regularisation)
    debug : bool - if True, track per-round theta estimates, confidence radii, and lambdas for debugging
    out_dir : str — directory for .npz and .pdf outputs (default "figures")

    Returns
    -------
    dict with keys: error_rate, error_std, test_rate, test_rate_std,
                    num_tests, num_tests_std, error_threshold
    """
    if true_theta is None:
        true_theta = np.zeros(d)
        true_theta[0] = S

    print(f"Running SCOUT experiment: d={d}, T={T}, alpha={alpha}, "
          f"delta={delta}, S={S}, num_runs={num_runs}...")

    all_results = []
    all_cum_tests = []
    all_cum_errors = []
    all_cum_tests_opt = []
    all_cum_errors_opt = []
    if debug:
        all_run_theta_ests = []
        all_run_lambds = []
        all_run_diam_bounds = []

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        np.random.seed(run)

        scout = SCOUTAlgorithm(d, T, alpha, delta, true_theta, S,
                               looseness_factor_beta=500 * aggressiveness,
                               looseness_factor_theta_est=aggressiveness)
        scout.run()
        results = scout.evaluate()
        all_results.append(results)

        print(f"    Error Rate: {results['error_rate']:.4f} (Target: {results['error_threshold']})")
        print(f"    Test Rate: {results['test_rate'] * 100:.1f}%")
        print(f"    Tau (final): "
              f"{compute_tau_opt(results['final_theta'], np.array(scout.all_contexts), alpha):.4f}")

        Z = np.array(scout.all_decisions)
        preds = np.array(scout.all_predictions)
        labels = np.array(scout.all_labels)
        errors = (preds != labels).astype(float)
        cum_tests = np.cumsum(Z) / np.arange(1, T + 1)
        cum_errors = np.cumsum(errors) / np.arange(1, T + 1)
        all_cum_tests.append(cum_tests)
        all_cum_errors.append(cum_errors)

        # Oracle baseline using ground-truth theta*
        tau_opt, _ = scout.compute_opt_testing_frac(scout.true_theta, scout.alpha)
        Z_opt = (np.abs(np.array(scout.all_contexts) @ scout.true_theta) < tau_opt).astype(int)
        Y = np.array(scout.all_labels)
        Y_opt = np.array([1 if np.dot(scout.all_contexts[i], scout.true_theta) > 0 else 0
                          for i in range(len(scout.all_contexts))])
        Y_opt[Z_opt == 1] = Y[Z_opt == 1]
        errors_opt = (Y_opt != Y).astype(float)
        cum_tests_opt = np.cumsum(Z_opt) / np.arange(1, T + 1)
        cum_errors_opt = np.cumsum(errors_opt) / np.arange(1, T + 1)
        all_cum_tests_opt.append(cum_tests_opt)
        all_cum_errors_opt.append(cum_errors_opt)

        if debug:
            theta_ests = np.array([t if t is not None else np.full(d, np.nan)
                                   for t in scout.all_theta_ests])
            all_run_theta_ests.append(theta_ests)
            all_run_lambds.append(np.array(scout.all_lambds))
            all_run_diam_bounds.append(np.array(scout.all_diam_bounds))

    avg_results = {
        'error_rate': np.mean([r['error_rate'] for r in all_results]),
        'error_std': np.std([r['error_rate'] for r in all_results]),
        'test_rate': np.mean([r['test_rate'] for r in all_results]),
        'test_rate_std': np.std([r['test_rate'] for r in all_results]),
        'num_tests': np.mean([r['num_tests'] for r in all_results]),
        'num_tests_std': np.std([r['num_tests'] for r in all_results]),
        'error_threshold': alpha,
    }

    print("\nOverall Results:")
    print(f"  Error Rate : {avg_results['error_rate']:.4f} ± {avg_results['error_std']:.4f} "
          f"(Target: {avg_results['error_threshold']})")
    print(f"  Test Rate  : {avg_results['test_rate'] * 100:.1f}% "
          f"± {avg_results['test_rate_std'] * 100:.1f}%")
    print(f"  Tests      : {avg_results['num_tests']:.1f} "
          f"± {avg_results['num_tests_std']:.1f} out of {T}")

    title = f"SCOUT_results_d={d}_T={T}_alpha={alpha}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = str(Path(out_dir) / f"{title}.npz")
    debug_kwargs = dict(
        all_theta_ests=np.array(all_run_theta_ests),
        all_lambds=np.array(all_run_lambds),
        all_diam_bounds=np.array(all_run_diam_bounds),
    ) if debug else {}
    np.savez(fname,
             all_cum_tests=all_cum_tests,
             all_cum_errors=all_cum_errors,
             all_cum_tests_opt=all_cum_tests_opt,
             all_cum_errors_opt=all_cum_errors_opt,
             **debug_kwargs,
             alpha=alpha, num_runs=num_runs, d=d, T=T, S=S, true_theta=true_theta)

    opt_test_rate = scout.compute_opt_testing_frac(scout.true_theta, alpha)[1]
    plot_aggregate_results(all_cum_tests, all_cum_errors, all_cum_tests_opt,
                           all_cum_errors_opt, alpha, opt_test_rate,
                           label_prefix=f"SCOUT (d={d}, T={T}, α={alpha})",
                           fig_fname=title, out_dir=out_dir)

    return avg_results


def eval_on_real_data(X, Y, alpha=0.05, num_perms=10, delta=0.1, S=1, aggressiveness=10.0,
                      title=None, debug=False, out_dir="figures"):
    """
    Evaluate SCOUT on a real labelled dataset.

    The dataset is treated as a finite population that arrives in a random
    order (simulating an online stream).  To obtain confidence intervals,
    the experiment is repeated over ``num_perms`` independent random shuffles.

    The oracle ("opt") baseline is constructed from theta_star fitted on the
    *full* dataset and the corresponding optimal threshold tau_star.  This
    represents the best possible fixed threshold policy that has access to all
    data up-front — SCOUT must learn to match it online.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.  Columns should be scaled to a comparable range;
        the algorithm does not perform internal normalisation.
    Y : np.ndarray, shape (n,)
        Binary disease labels in {0, 1}.
    alpha : float
        Target cumulative misclassification rate (default 0.05).
    num_perms : int
        Number of random data permutations used to estimate variability
        (default 10).
    delta : float
        SCOUT failure probability; safety guarantee holds with prob >= 1-delta
        (default 0.1).
    S : float
        Assumed upper bound on ||theta*||_2 (default 1).  Does not affect the
        core algorithm in the current implementation but is passed through for
        consistency with the theoretical framework.
    aggressiveness : float
        Rescales the confidence radius and regularisation strength.
        With a value of 1.0: extremely conservative and safe, 
        but slow convergence to optimal testing rate.
        Default value of 10.0, higher values → more aggressive testing.
    title : str or None
        Base filename (without extension) for .npz and .pdf outputs.
        Defaults to ``SCOUT_results_n={n}_d={d}_alpha={alpha}`` if None.
    debug : bool
        If True, records per-permutation theta estimates, confidence radii,
        and lambda values in the saved .npz (default False).
    out_dir : str — directory for .npz and .pdf outputs (default "figures")

    Returns
    -------
    dict with keys:
        theta_star      : np.ndarray, shape (d,) — full-data unregularized logistic MLE
        tau_star        : float — oracle threshold
        opt_test_rate   : float — fraction of samples tested by the oracle
        all_cum_tests   : np.ndarray, shape (num_perms, n) — SCOUT test rates
        all_cum_errors  : np.ndarray, shape (num_perms, n) — SCOUT error rates
        all_cum_tests_opt  : np.ndarray, shape (num_perms, n) — oracle test rates
        all_cum_errors_opt : np.ndarray, shape (num_perms, n) — oracle error rates
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=int)
    n, d = X.shape

    if title is None:
        title = f"SCOUT_results_n={n}_d={d}_alpha={alpha}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"eval_on_real_data: n={n}, d={d}, alpha={alpha}, "
          f"num_perms={num_perms}, delta={delta}")

    print("Fitting oracle theta_star on full dataset (unregularized MLE)...")
    theta_star = compute_theta_mle(X, Y)
    print("Fitted theta_star:", theta_star)
    if theta_star is None:
        raise ValueError("Could not fit theta_star: dataset appears to be empty.")

    tau_star = compute_tau_opt(theta_star, X, alpha)
    opt_test_rate = float(np.mean(np.abs(X @ theta_star) < tau_star))
    print(f"  tau_star={tau_star:.4f}, opt_test_rate={opt_test_rate:.4f}")

    all_cum_tests = []
    all_cum_errors = []
    all_cum_tests_opt = []
    all_cum_errors_opt = []
    if debug:
        all_perm_theta_ests = []
        all_perm_lambds = []
        all_perm_diam_bounds = []

    for perm_idx in range(num_perms):
        print(f"  Permutation {perm_idx + 1}/{num_perms}...")
        np.random.seed(perm_idx)
        perm = np.random.permutation(n)
        X_perm = X[perm]
        Y_perm = Y[perm]

        scout = _RealDataSCOUT(X_perm, Y_perm, alpha, delta, S,
                               looseness_factor_beta=500 * aggressiveness,
                               looseness_factor_theta_est=aggressiveness)
        scout.run()

        Z = np.array(scout.all_decisions)
        preds = np.array(scout.all_predictions)
        labels = np.array(scout.all_labels)
        errors = (preds != labels).astype(float)
        cum_tests = np.cumsum(Z) / np.arange(1, n + 1)
        cum_errors = np.cumsum(errors) / np.arange(1, n + 1)
        all_cum_tests.append(cum_tests)
        all_cum_errors.append(cum_errors)

        Z_opt = (np.abs(X_perm @ theta_star) < tau_star).astype(int)
        Y_opt = np.where(X_perm @ theta_star > 0, 1, 0)
        Y_opt[Z_opt == 1] = Y_perm[Z_opt == 1]
        errors_opt = (Y_opt != Y_perm).astype(float)
        cum_tests_opt = np.cumsum(Z_opt) / np.arange(1, n + 1)
        cum_errors_opt = np.cumsum(errors_opt) / np.arange(1, n + 1)
        all_cum_tests_opt.append(cum_tests_opt)
        all_cum_errors_opt.append(cum_errors_opt)

        if debug:
            theta_ests = np.array([t if t is not None else np.full(d, np.nan)
                                   for t in scout.all_theta_ests])
            all_perm_theta_ests.append(theta_ests)
            all_perm_lambds.append(np.array(scout.all_lambds))
            all_perm_diam_bounds.append(np.array(scout.all_diam_bounds))

    debug_kwargs = dict(
        all_theta_ests=np.array(all_perm_theta_ests),
        all_lambds=np.array(all_perm_lambds),
        all_diam_bounds=np.array(all_perm_diam_bounds),
    ) if debug else {}
    fname = str(Path(out_dir) / f"{title}.npz")
    np.savez(fname,
             all_cum_tests=all_cum_tests,
             all_cum_errors=all_cum_errors,
             all_cum_tests_opt=all_cum_tests_opt,
             all_cum_errors_opt=all_cum_errors_opt,
             **debug_kwargs,
             alpha=alpha, num_perms=num_perms, d=d, n=n, S=S,
             theta_star=theta_star, tau_star=tau_star)
    print(f"Results saved to {fname}")

    plot_aggregate_results(
        all_cum_tests, all_cum_errors,
        all_cum_tests_opt, all_cum_errors_opt,
        alpha, opt_test_rate,
        label_prefix="SCOUT (real data)",
        fig_fname=title, out_dir=out_dir
    )

    return {
        'theta_star': theta_star,
        'tau_star': tau_star,
        'opt_test_rate': opt_test_rate,
        'all_cum_tests': np.array(all_cum_tests),
        'all_cum_errors': np.array(all_cum_errors),
        'all_cum_tests_opt': np.array(all_cum_tests_opt),
        'all_cum_errors_opt': np.array(all_cum_errors_opt),
    }


def validate_on_synthetic_data(d=2, n=5000, alpha=0.05, true_theta=None, S=1,
                                num_perms=5, delta=0.1, seed=42, out_dir="figures"):
    """
    Validate eval_on_real_data using known synthetic data.

    Generates synthetic data from the logistic model, then passes it to
    eval_on_real_data so the full real-data pipeline can be tested against
    a known ground truth.

    Parameters
    ----------
    d : int — feature dimensionality (default 2)
    n : int — number of samples (default 5000)
    alpha : float — target error rate (default 0.05)
    true_theta : np.ndarray or None — true parameter (auto-generated if None)
    S : float — norm of true_theta when auto-generated (default 1)
    num_perms : int — permutations passed to eval_on_real_data (default 5)
    delta : float — failure probability (default 0.1)
    seed : int — random seed (default 42)
    out_dir : str — directory for .npz and .pdf outputs (default "figures")

    Returns
    -------
    dict — eval_on_real_data result dict, with an added 'true_theta' key
    """
    X, Y, true_theta = generate_synthetic_data(d, n, true_theta, S, seed=seed)
    print(f"Synthetic data: n={n}, d={d}, true_theta={true_theta}")
    results = eval_on_real_data(
        X, Y, alpha=alpha, num_perms=num_perms, delta=delta, S=S,
        title=f"SCOUT_synthetic_validate_d={d}_n={n}_alpha={alpha}",
        out_dir=out_dir,
    )
    results['true_theta'] = true_theta
    return results


# ---------------------------------------------------------------------------
# Paper figure reproduction
# ---------------------------------------------------------------------------

def generate_paper_figures(out_dir="figures"):
    """
    Reproduce all three synthetic experiments from the AISTATS 2026 paper.

    Runs SCOUT with each of the three (d, alpha) configurations reported in
    the paper, saves outputs to ``out_dir``, and returns a list of
    per-experiment result dicts.

    Experiment configurations
    -------------------------
    1. d=2,  alpha=0.05
    2. d=2,  alpha=0.10
    3. d=8,  alpha=0.10

    All experiments use T=100 000 rounds, delta=0.1, S=1, 100 independent
    runs, and true_theta = (1/sqrt(d), ..., 1/sqrt(d)) * S.

    Parameters
    ----------
    out_dir : str — directory for .npz and .pdf outputs (default "figures")
    """
    np.random.seed(42)

    T = 100_000
    delta = 0.1
    S = 1
    num_runs = 100

    configs = [
        {"d": 2, "alpha": 0.05},
        {"d": 2, "alpha": 0.10},
        {"d": 8, "alpha": 0.10},
    ]

    all_results = []
    for cfg in configs:
        d, alpha = cfg["d"], cfg["alpha"]
        true_theta = np.ones(d) / np.sqrt(d) * S
        results = run_scout_experiment(d, T, alpha, delta, S, true_theta,
                                       num_runs=num_runs, out_dir=out_dir)
        all_results.append(results)

    return all_results


if __name__ == "__main__":
    generate_paper_figures()
