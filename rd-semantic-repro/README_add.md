# Rate-Distortion Framework for Semantic Information — Figure Reproduction

This repository contains a reproduction of key figures from the paper:

&gt; **"A Rate-Distortion Framework for Characterizing Semantic Information"**  
&gt; Jiakun Liu, Wenyi Zhang, H. Vincent Poor  
&gt; IEEE International Symposium on Information Theory (ISIT), 2021

---

## Table of Contents

- [Rate-Distortion Framework for Semantic Information — Figure Reproduction](#rate-distortion-framework-for-semantic-information--figure-reproduction)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Figure 3: Scalar Gaussian SORDF](#figure-3-scalar-gaussian-sordf)
    - [Mathematical Model](#mathematical-model)
    - [Closed-Form Solution](#closed-form-solution)
    - [Key Features of the Plot](#key-features-of-the-plot)
    - [Parameters Used](#parameters-used)
    - [Files](#files)
  - [Figure 4: Binary Classification Case](#figure-4-binary-classification-case)
    - [Mathematical Model](#mathematical-model-1)
    - [Key Innovation: Soft Decision vs. Hard Decision](#key-innovation-soft-decision-vs-hard-decision)
    - [The Optimal Soft Decision Function](#the-optimal-soft-decision-function)
    - [Left Panel: $g(x)$ for Different $D\_s$](#left-panel-gx-for-different-d_s)
    - [Right Panel: Rate-Distortion Tradeoff](#right-panel-rate-distortion-tradeoff)
    - [Parameters Used](#parameters-used-1)
    - [Files](#files-1)
  - [Figure 5: Joint Rate-Distortion with Double Constraints](#figure-5-joint-rate-distortion-with-double-constraints)
    - [Mathematical Model](#mathematical-model-2)

---

## Overview

The paper proposes a **State-Observation Rate-Distortion Function (SORDF)** that characterizes the tradeoff between:

- **Semantic distortion** $D_s$: fidelity of the intrinsic state $S$ (unobservable)
- **Appearance distortion** $D_a$: fidelity of the extrinsic observation $X$ (observable, encoded)
- **Rate** $R$: number of bits required for encoding

The source is modeled as a pair $(S, X)$ with joint distribution $p(s, x)$. The encoder only sees $X$, but the decoder reconstructs both $\hat{S}$ and $\hat{X}$.

---

## Figure 3: Scalar Gaussian SORDF

**Corresponds to: Proposition 3 in the paper**

### Mathematical Model

- $S$ and $X$ are **jointly Gaussian scalars** with zero mean
- Covariance structure
- Distortion metrics: squared error for both semantic and appearance

### Closed-Form Solution

The SORDF has a closed form:

$$R(D_s, D_a) = \frac{1}{2} \max\left\{ \left(\log\frac{\Sigma_X}{D_a}\right)^+, \left(\log\frac{\Sigma_{SX}^2}{\Sigma_X(D_s - \text{mmse})}\right)^+ \right\}$$

where $\text{mmse} = \Sigma_S - \frac{\Sigma_{SX}^2}{\Sigma_X}$ is the minimum mean-square error of estimating $S$ from $X$.

### Key Features of the Plot

| Feature | Description |
|---------|-------------|
| **Vertical contour lines** (left of boundary) | $R$ is dominated by semantic distortion $D_s$ |
| **Horizontal contour lines** (right of boundary) | $R$ is dominated by appearance distortion $D_a$ |
| **Red boundary line** | Where $R(D_s, \infty) = R(\infty, D_a)$, i.e., both constraints are equally tight |
| **No plateaus** | Unlike the vector case, the scalar case has no "flat" regions |

### Parameters Used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\Sigma_S$ | 1.0 | Variance of intrinsic state $S$ |
| $\Sigma_X$ | 2.0 | Variance of extrinsic observation $X$ |
| $\Sigma_{SX}$ | 1.0 | Covariance between $S$ and $X$ |
| $mmse$ | 0.5 | Minimum estimation error $\Sigma_S - \Sigma_{SX}^2/\Sigma_X$ |

### Files

- `src/scalar_gaussian.py` — Implementation and plotting
- `figures/scalar_sordf_3d.png` — 3D surface plot
- `figures/scalar_sordf_contour.png` — Contour plot with boundary analysis

---

## Figure 4: Binary Classification Case

**Corresponds to: Proposition 5 in the paper**

### Mathematical Model

- $S \in \{0, 1\}$: Binary semantic state (uniform prior $P(S=0) = P(S=1) = 0.5$)
- $X \in \mathbb{R}$: Extrinsic observation, conditionally Gaussian:
- $X|S=0 \sim \mathcal{N}(A, \sigma^2)$
- $X|S=1 \sim \mathcal{N}(-A, \sigma^2)$
- Semantic distortion: Hamming distance (classification error rate)
- Appearance distortion: not constrained ($D_a = \infty$)

### Key Innovation: Soft Decision vs. Hard Decision

The paper reveals a **counter-intuitive result**:

> **"Local optimal (Bayesian) classification and encoding the binary classification is suboptimal"**

| Scheme | Strategy | Optimality |
|--------|----------|------------|
| **Naive (Hard)** | Encoder first makes hard Bayesian decision, then compresses the binary result | **Suboptimal** except at extreme $D_s$ |
| **Optimal (Soft)** | Encoder uses probabilistic $g(x) = P(\hat{S}=0\|X=x)$, preserving uncertainty | **Optimal** |

### The Optimal Soft Decision Function

$$g(x) = \left[1 + \exp\left(\lambda \frac{1 - e^{-2Ax/\sigma^2}}{1 + e^{-2Ax/\sigma^2}}\right)\right]^{-1}$$

where $\lambda < 0$ is chosen to satisfy the distortion constraint.

### Left Panel: $g(x)$ for Different $D_s$

| Curve | $D_s$ | $\lambda$ | Behavior |
|-------|-------|-----------|----------|
| Blue | 0.50 | $\approx 0$ | Flat at $g(x) = 0.5$ (no information) |
| Yellow | 0.39 | $\approx -0.8$ | Slightly curved, moderate confidence |
| Green | 0.27 | $\approx -2.1$ | More steep, higher confidence |
| Red | 0.16 | $\approx -17$ | Nearly step function (hard decision) |

**Key observation**: As $D_s \to Q(A/\sigma)$ (Bayes error rate), $g(x)$ approaches a step function. As $D_s \to 0.5$ (random guessing), $g(x)$ flattens to 0.5.

### Right Panel: Rate-Distortion Tradeoff

| Curve | Description |
|-------|-------------|
| **Blue** | Optimal soft decision (Proposition 5) |
| **Orange** | Naive hard decision + compression |

**Gap between curves**: The soft decision achieves **lower rate** for the same $D_s$, except at $D_s = Q(A/\sigma)$ where both coincide.

### Parameters Used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $A$ | 1.0 | Signal amplitude |
| $\sigma^2$ | 1.0 | Noise variance |
| $Q(A/\sigma)$ | $\approx 0.1587$ | Bayes optimal error rate |

### Files

- `src/baseline2.py` — Core implementation (soft decision, numerical solver for $\lambda$)
- `experiments/plot_fig4.py` — Figure generation
- `figures/fig4_left_gx.png` — Soft decision functions
- `figures/fig4_right_rd.png` — Rate-distortion comparison

---

## Figure 5: Joint Rate-Distortion with Double Constraints

**Corresponds to: Proposition 6 in the paper**

### Mathematical Model

Same binary classification model as Figure 4, but now **both constraints are active**:
- Semantic distortion: $D_s \leq$ target
- Appearance distortion: $D_a \leq$ target
