# Implementation Notes for rd-semantic-repro

## 1. Core philosophy
This project is implemented as a mathematics-first reproduction because the original paper is itself mathematics-first. The goal is not to loosely imitate its intuition, but to preserve a direct line from theorem to code to figure.

## 2. Mathematical problem from the paper
The paper models a source as (S, X) with joint law p(s, x).
- S is the intrinsic state, i.e. the semantic aspect.
- X is the extrinsic observation, i.e. the visible appearance.

The encoder sees X^n, compresses it, and the decoder outputs both S_hat^n and X_hat^n.

Two distortions are imposed:
- semantic distortion d_s;
- appearance distortion d_a.

The rate-distortion tradeoff is therefore two-dimensional.

## 3. Why SORDF matters
The paper derives a state-observation rate-distortion function of the form
R(D_s, D_a) = min I(X; S_hat, X_hat)
subject to the two distortion constraints.

This matters conceptually because it formalizes a central theme of semantic communication: the destination may care about both meaning and appearance, but not necessarily equally.

## 4. Binary classification case: source model
The binary classification case (Section V of the paper) is:

- S uniform Bernoulli on {0, 1}
- X|S=0 ~ N(A, sigma^2)
- X|S=1 ~ N(-A, sigma^2)

Semantic distortion: Hamming loss d_s(s, s_hat) = 1{s != s_hat}
Appearance distortion: squared error d_a(x, x_hat) = (x - x_hat)^2

The class separability is controlled by the ratio A/sigma. The Bayes classification error is Q(A/sigma), which sets the minimum achievable D_s.

### 4.1 Gaussian Q-function and Bayesian inference

We implement the Gaussian Q-function and posterior probability P(S=0|x):

```python
gaussian_q(x) = 0.5 * erfc(x / sqrt(2))
posterior_probability_s0(x) = N(A,sigma^2) / (N(A,sigma^2) + N(-A,sigma^2))
```

The function posterior_probability_s0 computes the Bayes posterior. As shown in the paper, this posterior is used to construct the optimal soft decision function g(x) = P(S_hat=0|x).

## 5. Full numerical implementation of R(D_s, inf) (Proposition 5)

### 5.1 What the theorem says

For the binary classification case, when there is no appearance distortion constraint (D_a = inf), the paper's Proposition 5 gives:

**Theorem (Proposition 5):**
R(D_s, inf) = 1 - (1/2) * integral [N^+(x) + N^-(x)] * h_2(g(x)) dx

for Q(A/sigma) <= D_s <= 1/2, and R(D_s, inf) = 0 for D_s > 1/2.

where:
- N^+(x) ~ N(A, sigma^2) and N^-(x) ~ N(-A, sigma^2) are the class-conditional densities
- h_2(p) = -p log_2(p) - (1-p) log_2(1-p) is the binary entropy function
- g(x) = P(S_hat = 0 | x) is the optimal soft decision function

### 5.2 The optimal soft decision function g(x)

The paper derives the optimal g(x) (Eq. 33 in the paper):

```
g(x) = [1 + exp(lambda * (1 - exp(-2Ax/sigma^2)) / (1 + exp(-2Ax/sigma^2)))]^(-1)
```

where lambda < 0 is a Lagrange multiplier determined by the distortion constraint.

The parameter lambda controls the "softness" of the decision:
- As lambda -> -inf, g(x) approaches a hard 0/1 threshold decision
- As lambda -> 0, g(x) approaches the Bayes posterior 1/2 (completely soft)

In our implementation, `_compute_g(x, lam, A, sigma)` evaluates Eq. (33) with numerical safeguards against overflow in the exponential.

### 5.3 The distortion constraint equation

The constraint that determines lambda is (Eq. 34 in the paper):

```
integral [N^+(x) - N^-(x)] * g(x) dx = 1 - 2 * D_s
```

The left-hand side is monotonic in lambda (since g(x) is monotonic in lambda), which makes bisection a reliable root-finding method.

Our implementation in `_dist_constraint(lam, A, sigma, Ds)` computes:
```
F(lambda) = integral [N^+ - N^-] * g_lambda(x) dx - (1 - 2 * D_s)
```
and seeks F(lambda) = 0.

### 5.4 Numerical techniques

**Integration:** Both the constraint integral and the rate integral use the midpoint rule with 2000 points over the interval [-10*sigma, 10*sigma]. This interval captures essentially all probability mass of the Gaussian mixtures (10*sigma covers >5 standard deviations from both means when A is not too large relative to sigma).

**Bisection for lambda:** The `_find_lambda` function searches lambda in [-50, -1e-6]:
- lambda = -1e-6 approximates the "no compression" limit (g(x) -> 1/2, rate -> 1 bit)
- lambda = -50 approximates the "maximum compression" limit (g(x) -> hard threshold, rate -> 0)
- The algorithm first expands the search range if needed, then performs up to 80 bisection iterations until the bracket width is below 1e-12

**Rate computation:** Once lambda is known, the rate integral is computed as:

```
R = 1 - 0.5 * integral (N^+(x) + N^-(x)) * h_2(g(x)) dx
```

### 5.5 Boundary conditions

The implementation explicitly checks three boundary cases:

| Condition | D_s value (A=sigma=1) | R(D_s, inf) | Meaning |
|-----------|----------------------|-------------|---------|
| D_s < Q(A/sigma) | -- | Error | Infeasible -- below Bayes error |
| D_s = Q(A/sigma) | ~0.1587 | 1 bit | Lossless classification |
| D_s >= 1/2 | >= 0.5 | 0 bits | Random guessing suffices |

These match the paper's analysis in Figure 4.

### 5.6 Relationship to the previous staged implementation

The previous implementation used R(D_s, inf) = 1 - h_2(D_s) as a placeholder. While this gives correct boundary behavior (R=1 at D_s=0, R=0 at D_s=0.5) and monotonicity, it does not capture the key information-theoretic structure:

- The true R(D_s, inf) depends on A/sigma through the class-conditional densities
- The full numerical routine accounts for the actual distribution of X
- The new implementation reproduces the paper's actual curve in Figure 4

## 6. The R(D_s, D_a) achievable upper bound (Proposition 6)

When both semantic and appearance distortion constraints are active, the paper provides an achievable upper bound in Proposition 6:

```
R(D_s, D_a) <= min_{D in [Q(A/sigma), D_s]} R(D, inf) + (1/2) * log^+(eta(D) / D_a)
```

where eta(D) is the conditional variance of X given S_hat when operating at semantic distortion D.

**Interpretation:** The encoder first allocates rate to satisfy the semantic constraint (producing S_hat), then adds extra rate to refine the appearance reconstruction. The total rate is the sum of:
1. R(D, inf): rate needed for semantic distortion D
2. (1/2) * log^+(eta / D_a): additional rate to achieve appearance distortion D_a given the semantic side information S_hat

## 7. Gaussian case studies (Section IV)

### 7.1 Jointly Gaussian source

When S and X are jointly Gaussian with covariance matrix and squared error distortions:

**Proposition 2:** R(D_s, inf) = R_1(D_s - mmse), where R_1 is the rate-distortion function of the MMSE estimate and mmse is the MMSE of estimating S from X.

**Proposition 3 (scalar case):** For scalar S and X:
```
R(D_s, D_a) = (1/2) * max{ log^+(Sigma_X / D_a), log^+(Sigma_SX^2 / (Sigma_X * (D_s - mmse))) }
```

**Proposition 4 (vector case):** For the specific vector model X = 1_m S + Z, S ~ N(0, sigma_S^2), Z ~ N(0, sigma_Z^2 I), the (D_s, D_a) plane is divided into five regions with different rate expressions.

### 7.2 Vector Gaussian case: five regions

The (D_s, D_a) plane is divided into five regions (S1-S5 in Figure 2 of the paper):

- S1: D_s >= mmse, moderate D_a -- tradeoff region with two sub-bands
- S2: D_a < m*sigma_Z^2, tight appearance -- appearance dominates
- S3: Loose appearance -- semantic distortion dominates
- S4: D_a moderate but D_s near maximum -- hybrid
- S5: Both large -- zero rate

## 8. Implementation structure

```
src/
  binary_case.py        # Binary classification: R(Ds,inf), R(Ds,Da) upper bound
  gaussian_case.py      # Gaussian source: scalar and vector SORDF (Proposition 2, 3, 4)
  multiclass_case.py    # Extension: K-class classification

tests/
  test_binary_case.py   # Boundary conditions and monotonicity for binary case
  test_gaussian_case.py # Gaussian SORDF boundary cases
  test_multiclass_case.py

experiments/
  run_binary_curve.py          # Generate R(Ds,inf) CSV
  plot_binary_curve.py         # Plot single R(Ds,inf) curve
  plot_binary_curve_family.py  # Plot multiple curves for different A/sigma
  run_gaussian_sordf.py        # Generate Gaussian SORDF surfaces
  run_multiclass_curves.py     # Generate multiclass comparisons

docs/
  implementation-notes.md      # This file: theory-to-code mapping
```

## 9. Test-driven development philosophy

In this project, tests act as mathematical assertions rather than just software checks. Each test encodes a theorem consequence:

- test_gaussian_q_matches_known_value_at_one: Verifies Q(1) = 0.1587, a known constant
- test_posterior_is_half_at_zero: By symmetry, P(S=0|X=0) = 0.5
- test_rejects_semantic_distortion_below_bayes_error: D_s < Q(A/sigma) is impossible
- test_rate_decreases_as_allowed_semantic_distortion_increases: R(D_s, inf) is monotonically nonincreasing
- test_better_class_separation_reduces_left_endpoint_distortion: Larger A/sigma shifts Bayes error left

These tests lock down the theorem domain and qualitative behavior independently of the numerical accuracy of the interior formula.
