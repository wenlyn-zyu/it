# rd-semantic-repro

This project reproduces numerical results from **A Rate-Distortion Framework for Characterizing Semantic Information** for the EE142 course project.

## Project focus

The paper models an information source as a pair `(S, X)`:
- `S` is the hidden intrinsic state, interpreted as semantic content;
- `X` is the observable extrinsic observation, interpreted as appearance.

The encoder observes only `X`, while the decoder reconstructs both `S_hat` and `X_hat`. The central object is the state-observation rate-distortion function

```text
R(D_s, D_a) = min I(X; S_hat, X_hat)
```

under a semantic distortion constraint `D_s` and an appearance distortion constraint `D_a`.

## Implemented reproduction experiments

The current implementation focuses on the paper's binary classification case in Section V:

```text
S ~ Bernoulli(1/2)
X | S=0 ~ N(A, sigma^2)
X | S=1 ~ N(-A, sigma^2)
```

Implemented artifacts:

| Paper result | Script | Output |
| --- | --- | --- |
| Proposition 5, Figure 4 right: `R(D_s, infinity)` | `experiments/plot_binary_curve.py` | `figures/binary_rd_infinite_a1_sigma1.png`, `.csv` |
| Proposition 5, Figure 4 left: optimal soft decision `g(x)` | `experiments/plot_binary_soft_decision.py` | `figures/binary_soft_decision_a1_sigma1.png`, `.csv` |
| Class-separation comparison for `A/sigma` | `experiments/plot_binary_curve_family.py` | `figures/binary_rd_infinite_curve_family.png`, multiple `.csv` files |
| Proposition 6, Figure 5 style upper bound | `experiments/plot_binary_upper_bound.py` | `figures/binary_upper_bound_a2sqrt10_sigma1.png`, `.csv` |

## Reproduction commands

Run these commands on `idata2` in `~/zhuwl2022/it/rd-semantic-repro`.

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 experiments/run_binary_curve.py
python3 experiments/plot_binary_curve.py
python3 experiments/plot_binary_curve_family.py
python3 experiments/plot_binary_soft_decision.py
python3 experiments/plot_binary_upper_bound.py
```

A successful verification currently reports 15 passing tests.

## Theory-to-code mapping

- `src/binary_case.py::gaussian_q` implements the Bayes-error boundary `Q(A/sigma)`.
- `src/binary_case.py::classification_rd_infinite` implements Proposition 5 by solving for the Lagrange multiplier in Eq. (34) and evaluating the rate integral in Eq. (32).
- `src/binary_case.py::sample_soft_decision_curve` evaluates the optimal soft decision function `g(x)` from Eq. (33).
- `src/binary_case.py::rd_upper_bound` implements the achievable upper bound in Proposition 6.

Tests in `tests/test_binary_case.py` check theorem-level behavior: infeasible semantic distortion below Bayes error, zero-rate behavior at `D_s = 1/2`, monotonicity, curve endpoints, and finite-`D_a` upper-bound behavior.

## Remaining work

The binary classification reproduction is now the main completed track. Useful next steps are:

1. implement the Gaussian scalar case from Proposition 3;
2. implement the Gaussian vector five-region case from Proposition 4 and reproduce Figure 2/Figure 3 style plots;
3. add a modest extension, such as a multiclass semantic state, only after the paper reproduction is stable.