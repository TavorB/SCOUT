# SCOUT: Safe Classification Under Online Uncertainty Testing

Code for the paper
**"The Good, the Bad, and the Sampled: a No-Regret Approach to Safe Online Classification"**
*AISTATS 2026*

---

## Problem Setting

Patients arrive one at a time with a feature vector **x_t** and an unknown disease label **Y_t**.
At each round the decision maker must choose:
- **Test** (Z_t = 1): observe the true label at some cost, then record it as the prediction.
- **Predict** (Z_t = 0): predict without testing, incurring a potential misclassification.

The goal is to **minimise the total number of tests** while ensuring the cumulative misclassification rate never exceeds a user-supplied tolerance **α** (with high probability 1 − δ).

The disease label follows a logistic model: **Y_t ~ Bernoulli(σ(x_t^T θ*))**, where **θ*** and the context distribution **P** are both unknown.

---

## Key Concepts

### θ* and τ*

**θ*** is the true (unknown) logistic regression parameter.  SCOUT estimates it online from tested patients.

**τ*** is the *optimal decision threshold*: the smallest τ such that the fraction of mispredictions made when not testing satisfies the α constraint.


Intuitively, τ* is the confidence score below which a patient is uncertain enough to warrant testing.

### Optimal Baseline ("opt")

The opt baseline assumes full knowledge of θ* and P.  It tests patient t if and only if |x_t^T θ*| < τ*, achieving test fraction **p* = P(|x^T θ*| ≤ τ*)**.
SCOUT achieves **sublinear regret** relative to this oracle: the excess number of tests grows as O(√T), so the per-round gap vanishes.

### Safety Guarantee

With probability at least 1 − δ, SCOUT achieves **anytime safety**: for all t, the cumulative misclassification rate up to time t is at most α.

---

## Requirements

```
numpy
scipy
matplotlib
tqdm
```

Install with:
```bash
pip install numpy scipy matplotlib tqdm
```

---

## Usage

### Reproduce paper figures

```bash
python SCOUT_cleaned.py
```

This calls `generate_paper_figures()`, which runs three experiments (d=2/α=0.05, d=2/α=0.10, d=8/α=0.10) and saves PDF figures to `figures/`.

### Evaluate on your own dataset

```python
import numpy as np
from SCOUT_cleaned import eval_on_real_data

# X: (n_samples, n_features) float array
# Y: (n_samples,) binary int array {0, 1}
results = eval_on_real_data(X, Y, alpha=0.05, num_perms=10)

print("Oracle theta_star:", results['theta_star'])
print("Oracle threshold tau_star:", results['tau_star'])
print("Oracle test fraction:", results['opt_test_rate'])
```

`eval_on_real_data` fits θ* on the full dataset (the oracle), then runs SCOUT over `num_perms` random orderings of the data, comparing SCOUT's online decisions against the oracle baseline.  Results are plotted and saved to `figures/`.

### Run a single synthetic experiment

```python
from SCOUT_cleaned import run_scout_experiment
import numpy as np

d, alpha = 4, 0.05
true_theta = np.ones(d) / np.sqrt(d)
results = run_scout_experiment(d=d, T=10_000, alpha=alpha, delta=0.1,
                               S=1, true_theta=true_theta, num_runs=20)
```

### Generate synthetic data

```python
from SCOUT_cleaned import generate_synthetic_data
import numpy as np

# Returns feature matrix X, binary labels Y, and the true_theta used.
X, Y, true_theta = generate_synthetic_data(d=4, n=5000, S=1, seed=42)
```

### Validate on synthetic data (end-to-end check)

`validate_on_synthetic_data` generates a synthetic logistic dataset and immediately passes it through the full `eval_on_real_data` pipeline, so you can verify correctness against a known ground truth.

```python
from SCOUT_cleaned import validate_on_synthetic_data

results = validate_on_synthetic_data(d=2, n=5000, alpha=0.05, num_perms=5)
print("True theta:", results['true_theta'])
print("Fitted theta_star:", results['theta_star'])
```

---

## Testing

A minimal smoke-test suite is included in `test_scout.py`.  It covers all public API functions with small parameters (T=300, n=300) so it completes in under a minute.

```bash
cd github_code/
# with pytest
pytest test_scout.py -v
# or standalone
python test_scout.py
```

---

## Output

Each experiment produces:
- A **3-panel PDF figure** in `figures/` showing cumulative test rate, excess tests (regret), and cumulative error rate — all with 10–90 percentile bands across runs.
- A **`.npz` file** in `figures/` containing the raw per-run arrays for further analysis.
