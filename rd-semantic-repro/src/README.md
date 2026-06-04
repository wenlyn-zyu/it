# Source modules

## `binary_case.py`

Numerical routines for the binary classification reproduction in Section V of the paper.

Implemented functions:

- `gaussian_q`: Gaussian Q-function used for the Bayes-error boundary.
- `posterior_probability_s0`: Bayes posterior for the symmetric Gaussian-mixture model.
- `classification_rd_infinite`: Proposition 5 numerical computation of `R(D_s, infinity)`.
- `sample_rd_infinite_curve`: sampling helper for Figure 4 right style curves.
- `sample_soft_decision_curve`: sampling helper for Figure 4 left style `g(x)` curves.
- `rd_upper_bound`: Proposition 6 achievable upper bound for finite appearance distortion.
- `sample_rd_finite_curve`: helper for finite-`D_a` semantic-distortion sweeps.

## `gaussian_case.py`

Closed-form routines for the Gaussian case studies in Section IV.

Implemented functions:

- `gaussian_scalar_sordf`: Proposition 3 scalar Gaussian SORDF.
- `sample_scalar_gaussian_surface`: grid sampler for scalar Gaussian contour plots.
- `gaussian_vector_parameters`: MMSE and alpha parameters for Proposition 4.
- `gaussian_vector_sordf`: Proposition 4 five-region vector Gaussian SORDF.
- `sample_vector_gaussian_surface`: grid sampler for vector Gaussian contour plots.

Planned extension module:

- A small extension such as multiclass semantic states or finite-sample SORDF estimation.