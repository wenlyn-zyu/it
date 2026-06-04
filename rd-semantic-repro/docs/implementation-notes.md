# Implementation Notes for rd-semantic-repro

## 1. Core problem

The paper models a source as `(S, X)` with joint law `p(s, x)`.

- `S` is the hidden intrinsic state, representing semantic content.
- `X` is the extrinsic observation, representing appearance.

The encoder observes `X^n` and the decoder reconstructs both `S_hat^n` and `X_hat^n`. The paper's state-observation rate-distortion function is

```text
R(D_s, D_a) = min I(X; S_hat, X_hat)
```

subject to semantic distortion `D_s` and appearance distortion `D_a`.

## 2. Binary classification source model

Section V uses

```text
S ~ Bernoulli(1/2)
X | S=0 ~ N(A, sigma^2)
X | S=1 ~ N(-A, sigma^2)
```

Semantic distortion is Hamming loss. Appearance distortion is squared error. The Bayes error `Q(A/sigma)` is the minimum feasible semantic distortion.

## 3. Proposition 5: `R(D_s, infinity)`

For `Q(A/sigma) <= D_s <= 1/2`, the paper gives

```text
R(D_s, infinity) = 1 - 1/2 * int (N^+(x) + N^-(x)) h_2(g(x)) dx
```

where `g(x)` is the optimal soft decision function and `lambda < 0` is chosen to satisfy the semantic distortion constraint. The implementation uses midpoint integration and bisection on `lambda`.

Scripts:

- `plot_binary_soft_decision.py`: Figure 4 left style `g(x)` curves;
- `plot_binary_curve.py`: Figure 4 right style `R(D_s, infinity)` curve;
- `plot_binary_curve_family.py`: effect of class separation `A/sigma`.

## 4. Proposition 6: finite appearance distortion upper bound

For active semantic and appearance constraints, the paper gives

```text
R(D_s, D_a) <= min_D R(D, infinity) + 1/2 log^+(eta(D) / D_a),
D in [Q(A/sigma), D_s].
```

`rd_upper_bound` samples the feasible `D` interval, evaluates the semantic rate from Proposition 5, and adds the Gaussian appearance-refinement term. `plot_binary_upper_bound.py` generates a Figure 5 style plot for `A^2/sigma^2 = 10`.

## 5. Proposition 3: scalar Gaussian SORDF

For scalar jointly Gaussian `S` and `X`, the paper gives

```text
R(D_s, D_a) = 1/2 max{
  log^+(Sigma_X / D_a),
  log^+(Sigma_SX^2 / (Sigma_X (D_s - mmse)))
}
```

`gaussian_scalar_sordf` implements this closed form. `plot_gaussian_scalar.py` samples the `(D_s, D_a)` plane and produces a contour plot showing whether semantic or appearance distortion dominates.

## 6. Proposition 4: vector Gaussian SORDF

For `X = 1_m S + Z`, with `S ~ N(0, sigma_S^2)` and `Z ~ N(0, sigma_Z^2 I)`, the paper divides the `(D_s, D_a)` plane into five regions. `gaussian_vector_sordf` implements the five region expressions. `plot_gaussian_vector.py` overlays the two slanted region boundaries that expose the semantic-appearance tradeoff.

## 7. Tests as theorem checks

The tests in `tests/` encode mathematical properties rather than software-only behavior:

- binary Bayes-error feasibility and zero-rate boundaries;
- binary soft-decision symmetry `g(0)=1/2` and `g(-x)+g(x)=1`;
- Proposition 6 upper-bound monotonicity as `D_a` relaxes;
- Proposition 3 appearance-dominant, semantic-dominant, and zero-rate cases;
- Proposition 4 all five vector Gaussian regions.

## 8. Reproduction workflow

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

The generated CSV and PNG files under `figures/` are tracked in Git for report and poster use.

## 9. Remaining work

The main paper reproduction is implemented. Remaining project work should focus on report integration and one modest extension, such as multiclass semantics or finite-sample estimation.