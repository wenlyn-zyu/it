# rd-semantic-repro

Main theory-first project for reproducing the paper **A Rate-Distortion Framework for Characterizing Semantic Information**.

## Goal
This project is the main EE142 course-project track. It aims to:
1. numerically reproduce the key results in the paper;
2. generate publication-ready figures for the report and poster;
3. build one small extension beyond the original paper.

## Why this project exists
The original paper is elegant but compact. It gives the information-theoretic formulation and several closed-form or semi-closed-form case studies, but it does not provide public code in this repository. For a course project, reproduction is valuable because it turns the abstract rate-distortion statements into visible numerical behavior.

We prioritize this track because the EE142 guideline asks for a professional English report, clear technical proof, and encourages new ideas. A clean numerical reproduction satisfies the proof-oriented part, while a controlled extension can provide the new idea.

## Planned phases
### Phase 1: Binary classification case
Reproduce:
- the soft semantic decision function `g(x)`;
- the curve `R(D_s, \infty)`;
- the achievable upper bound on `R(D_s, D_a)`.

This is the best entry point because the numerical problem is lower-dimensional and directly shows the semantic-vs-appearance tradeoff.

### Phase 2: Gaussian cases
Reproduce:
- scalar Gaussian closed-form `R(D_s, D_a)`;
- contour or region plots that visualize the two-distortion tradeoff;
- selected vector-Gaussian behavior if computation remains stable.

### Phase 3: Extension
Preferred extension: multiclass semantic state model.
Instead of `S in {0,1}`, consider `S in {1,2,...,K}` with class-conditional Gaussian observations. This keeps the intrinsic-state / extrinsic-observation philosophy of the original paper while adding a small but meaningful generalization.

## Project structure
- `src/`: reusable numerical routines
- `experiments/`: runnable scripts for each figure or table
- `figures/`: generated plots
- `docs/`: detailed implementation notes

## First implementation target
Start from the binary classification case, because it provides the fastest path to a credible reproduction figure.
