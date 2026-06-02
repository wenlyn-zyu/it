# Implementation Notes for 
d-semantic-repro

## 1. Core philosophy
This project is implemented as a mathematics-first reproduction because the original paper is itself mathematics-first. The goal is not to loosely imitate its intuition, but to preserve a direct line from theorem to code to figure.

## 2. Mathematical problem from the paper
The paper models a source as (S, X) with joint law p(s, x).
- S is the intrinsic state, i.e. the semantic aspect.
- X is the extrinsic observation, i.e. the visible appearance.

The encoder sees X^n, compresses it, and the decoder outputs both \hat S^n and \hat X^n.

Two distortions are imposed:
- semantic distortion d_s;
- appearance distortion d_a.

The rate-distortion tradeoff is therefore two-dimensional.

## 3. Why SORDF matters
The paper derives a state-observation rate-distortion function of the form
R(D_s, D_a) = min I(X; \hat S, \hat X)
subject to the two distortion constraints.

This matters conceptually because it formalizes a central theme of semantic communication: the destination may care about both meaning and appearance, but not necessarily equally.

## 4. Why the binary case comes first
The binary classification case is numerically smaller and conceptually sharper than the Gaussian vector case.

Model:
- S uniform on {0,1};
- X|S=0 ~ N(A, \sigma^2);
- X|S=1 ~ N(-A, \sigma^2).

The paper shows that the feasible distortion region for R(D_s, \infty) is bounded below by the Bayes error Q(A/\sigma) and above by 1/2.

That gives us immediate executable checks:
- values below Q(A/\sigma) should be rejected;
- value 1/2 should give zero rate.

## 5. Why TDD is especially useful here
In this project, tests act as mathematical assertions.

Examples:
- gaussian_q(1) has a known numeric value;
- infeasible distortion requests must be rejected;
- allowing more semantic distortion should not increase the required rate.

These tests are useful even before the full numerical machinery is built because they lock down the theorem's domain and qualitative behavior.

## 6. Current staged implementation
The current code is intentionally staged.

Stage 1:
- implement exact boundary behavior;
- verify monotonic direction;
- keep the interface stable.

Stage 2:
- replace the staged interior formula by a fuller numerical routine closer to the paper's expression for R(D_s, \infty).

This staged approach prevents us from jumping straight into opaque numerical integration without first validating the logical skeleton of the model.

## 7. How the current code maps to the paper
- gaussian_q corresponds to the Bayes classification boundary used in the binary model.
- classification_rd_infinite is our evolving numerical entry point for the paper's R(D_s, \infty) quantity.
- the tests encode boundary facts and monotonicity suggested by the theory.

## 8. Why the extension should be multiclass
The multiclass extension is mathematically natural because it preserves the same semantic idea:
- a hidden semantic state;
- an observable signal drawn from a class-conditional distribution;
- a task-driven distortion measure.

It is also practical for a course report because it adds novelty without breaking the continuity with the original paper.


## 9. First generated artifact
We now generate a first CSV for the binary-case R(D_s, \infty) curve using experiments/run_binary_curve.py.

This CSV is not yet the final reproduction claim of the paper. It is a staged numerical artifact built on the current interface and boundary-consistent implementation. Its purpose is to stabilize the data pipeline early:
- tested rate function;
- tested curve sampler;
- reproducible experiment script;
- saved curve data in igures/.

This makes it much easier to refine the interior formula later without changing how figures are produced.

## 10. First plotted artifact
The next artifact after the sampled CSV is a direct plot of the binary-case `R(D_s, \infty)` curve.

The plotting script uses the saved CSV rather than recomputing the curve inline. This is intentional:
- data generation and plotting stay separated;
- figure regeneration becomes simple;
- later improvements to the interior formula can reuse the same output contract.

This makes the workflow closer to a reproducible research pipeline rather than a one-off script.