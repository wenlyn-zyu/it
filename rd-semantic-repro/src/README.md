# Source modules

`binary_case.py` contains the implemented numerical routines for the binary classification reproduction in Section V of the paper.

Implemented functions:

- `gaussian_q`: Gaussian Q-function used for the Bayes-error boundary.
- `posterior_probability_s0`: Bayes posterior for the symmetric Gaussian-mixture model.
- `classification_rd_infinite`: Proposition 5 numerical computation of `R(D_s, infinity)`.
- `sample_rd_infinite_curve`: sampling helper for Figure 4 right style curves.
- `sample_soft_decision_curve`: sampling helper for Figure 4 left style `g(x)` curves.
- `rd_upper_bound`: Proposition 6 achievable upper bound for finite appearance distortion.
- `sample_rd_finite_curve`: helper for finite-`D_a` semantic-distortion sweeps.

Planned but not yet implemented modules:

- Gaussian scalar/vector case routines for Section IV.
- A small extension module, such as multiclass semantic states.