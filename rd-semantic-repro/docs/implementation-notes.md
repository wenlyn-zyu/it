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

The implemented reproduction uses Section V of the paper:

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

where

```text
g(x) = [1 + exp(lambda * (1 - exp(-2Ax/sigma^2)) / (1 + exp(-2Ax/sigma^2)))]^(-1)
```

and `lambda < 0` satisfies

```text
int (N^+(x) - N^-(x)) g(x) dx = 1 - 2D_s.
```

The implementation in `src/binary_case.py` uses midpoint integration and bisection on `lambda`. The corresponding scripts reproduce both panels of the paper's Figure 4 style result:

- `plot_binary_soft_decision.py`: `g(x)` for several `D_s` values;
- `plot_binary_curve.py`: `R(D_s, infinity)` for `A=1, sigma=1`.

## 4. Proposition 6: finite appearance distortion upper bound

For active semantic and appearance constraints, the paper gives the achievable upper bound

```text
R(D_s, D_a) <= min_D R(D, infinity) + 1/2 log^+(eta(D) / D_a),
D in [Q(A/sigma), D_s].
```

The implementation in `rd_upper_bound` samples the feasible `D` interval, evaluates the semantic rate from Proposition 5, and adds the Gaussian appearance-refinement term. `plot_binary_upper_bound.py` generates a Figure 5 style plot for `A^2/sigma^2 = 10`, including the two reference curves from the paper.

## 5. Tests as theorem checks

The tests in `tests/test_binary_case.py` encode mathematical properties rather than software-only behavior:

- `Q(1) = 0.158655...`;
- symmetry gives `P(S=0|X=0)=1/2`;
- `D_s < Q(A/sigma)` is infeasible;
- `D_s = 1/2` gives zero semantic rate;
- `R(D_s, infinity)` is nonincreasing in `D_s`;
- soft-decision curves satisfy `g(0)=1/2` and `g(-x)+g(x)=1`;
- the finite-`D_a` upper bound decreases when appearance distortion is relaxed.

## 6. Reproduction workflow

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 experiments/run_binary_curve.py
python3 experiments/plot_binary_curve.py
python3 experiments/plot_binary_curve_family.py
python3 experiments/plot_binary_soft_decision.py
python3 experiments/plot_binary_upper_bound.py
```

The generated CSV and PNG files under `figures/` are tracked in Git for report and poster use.

## 7. Remaining paper reproduction

The current code does not yet implement Section IV Gaussian case studies. The next paper-aligned tasks are:

1. Proposition 3 scalar Gaussian closed-form SORDF;
2. Proposition 4 vector Gaussian five-region SORDF and Figure 2/Figure 3 style plots;
3. an optional small extension after the paper reproduction is complete.