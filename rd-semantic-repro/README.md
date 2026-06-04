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

Implemented artifacts:

| Paper result | Script | Output |
| --- | --- | --- |
| Proposition 5, Figure 4 right: binary `R(D_s, infinity)` | `experiments/plot_binary_curve.py` | `figures/binary_rd_infinite_a1_sigma1.png`, `.csv` |
| Proposition 5, Figure 4 left: binary soft decision `g(x)` | `experiments/plot_binary_soft_decision.py` | `figures/binary_soft_decision_a1_sigma1.png`, `.csv` |
| Binary class-separation comparison | `experiments/plot_binary_curve_family.py` | `figures/binary_rd_infinite_curve_family.png`, multiple `.csv` files |
| Proposition 6, Figure 5 style binary upper bound | `experiments/plot_binary_upper_bound.py` | `figures/binary_upper_bound_a2sqrt10_sigma1.png`, `.csv` |
| Proposition 3 scalar Gaussian SORDF | `experiments/plot_gaussian_scalar.py` | `figures/gaussian_scalar_sordf_contour.png`, `.csv` |
| Proposition 4 vector Gaussian SORDF, Figure 2/3 style | `experiments/plot_gaussian_vector.py` | `figures/gaussian_vector_sordf_m2_contour.png`, `.csv` |

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
python3 experiments/plot_gaussian_scalar.py
python3 experiments/plot_gaussian_vector.py
```

A successful verification currently reports 26 passing tests.

## Theory-to-code mapping

- `src/binary_case.py::classification_rd_infinite` implements Proposition 5 by solving for the Lagrange multiplier in Eq. (34) and evaluating the rate integral in Eq. (32).
- `src/binary_case.py::sample_soft_decision_curve` evaluates the optimal soft decision function `g(x)` from Eq. (33).
- `src/binary_case.py::rd_upper_bound` implements the achievable upper bound in Proposition 6.
- `src/gaussian_case.py::gaussian_scalar_sordf` implements the closed-form scalar Gaussian SORDF in Proposition 3.
- `src/gaussian_case.py::gaussian_vector_sordf` implements the five-region vector Gaussian SORDF in Proposition 4.

Tests in `tests/` check theorem-level behavior: infeasible semantic distortion below Bayes error, zero-rate boundaries, monotonicity, curve endpoints, finite-`D_a` upper-bound behavior, scalar Gaussian dominance cases, and all five vector Gaussian regions.

## Remaining work

The main paper reproduction tracks are now implemented. Useful next steps are:

1. compare generated Gaussian plots numerically/visually against the exact paper figure parameters if the full-version parameters are available;
2. add a modest extension, such as a multiclass semantic state or finite-sample SORDF estimation;
3. integrate the figures and theory mapping into the final IEEE-format report and poster.