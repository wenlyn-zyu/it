---
presentation:
  theme: white.css # 替换solarized，无自动大写、标准商务白底
  mouseWheel: true
  width: 1280  # 标准16:9横向PPT宽
  height: 720  # 标准16:9高，横向比例
---
<style>
/* 彻底关闭标题自动大写 */
.reveal h1, .reveal h2, .reveal h3, .reveal h4 {
  text-transform: none !important;
}
</style>
<!-- slide -->
### A Rate-Distortion Framework for Characterizing Semantic Information
Presenter: 朱雯琳  章彦琪

<!-- slide -->
## 1. Motivation & Core Problem
* **Traditional Shannon Theory**: Explicitly excludes semantic aspects to focus on engineering problem of transmitting symbols.
* **The Challenge of Semantics**: A task-independent, universal characterization remains elusive.
* **Core Proposal**: For many applications, semantic aspects correspond to accomplishing specific **inference goals**.
* **Key Concept (Decomposition)**:
  * **Intrinsic State $S$**: Unobservable, capturing the "semantic" aspect (features).
  * **Extrinsic Observation $X$**: Observable, capturing the "appearance" (raw data).
* **Objective**: Efficiently encode $X$ to satisfy distortion constraints for **both** $S$ and $X$ simultaneously.

<!-- slide -->
## 2. Related Work
* **Information Bottleneck & Privacy Funnel**:
  * Address similar trade-offs but rely on information measures (e.g., mutual information) rather than explicit distortion metrics.
* **Task-based Compression**:
  * Focuses on quantizer design and shows performance benefits for task-specific goals, but does not jointly model intrinsic-extrinsic state behaviors.
* **Indirect Rate-Distortion Problem**:
  * Deals with unobservable sources under a single constraint. This work extends the paradigm to **multiple joint distortion constraints**.

<!-- slide -->
## 3. Problem Formulation
* **System Model**:
  * Memoryless source: $(S, X) \sim p(s, x)$
  * Encoder: $f_n(X^n) = W \in \{1, 2, \dots, 2^{nR}\}$ (only observes and encodes $X^n$)
  * Decoder: $g_n(W) = (\hat{S}^n, \hat{X}^n)$
* **Distortion Constraints**:
  * Semantic distortion: $d_s(s, \hat{s})$, s.t. $\lim_{n\to\infty} \mathbb{E} d_s(S^n, \hat{S}^n) \le D_s$
  * Appearance distortion: $d_a(x, \hat{x})$, s.t. $\lim_{n\to\infty} \mathbb{E} d_a(X^n, \hat{X}^n) \le D_a$
* **Goal**: Characterize the boundary of achievable rate-distortion triples $(R, D_s, D_a)$, defined as the **State-Observation Rate-Distortion Function (SORDF)**.

<!-- slide -->
## 4. Mathematical Characterization & Derivation
* **Theorem 1 (The SORDF)**:
  $$R(D_s, D_a) = \min_{p(\hat{s}, \hat{x}|x)} I(X; \hat{S}, \hat{X})$$
  $$\text{s.t. } \mathbb{E} \hat{d}_s(X, \hat{S}) \le D_s, \quad \mathbb{E} d_a(X, \hat{X}) \le D_a$$
  where $\hat{d}_s(x, \hat{s}) = \mathbb{E}[d_s(S, \hat{s}) | X = x] = \frac{1}{p(x)}\sum_{s} p(s,x) d_s(s, \hat{s})$

* **Proof Sketch (Markov Chain Relationship)**:
  * Show the one-shot expected distortion $\mathbb{E}d_s(S^n, \hat{S}^n)$ is equivalent to $\mathbb{E}\hat{d}_s(X^n, \hat{S}^n)$.
  * This is validated because $S^n \leftrightarrow X^n \leftrightarrow \hat{S}^n$ forms a Markov chain (given $X^n$, the reproduction $\hat{S}^n$ is conditionally independent of the true state $S^n$).
  * Expanding the expectation:
    $$\mathbb{E}d_s(S^n, \hat{S}^n) = \sum p(x^n, \hat{s}^n) \left[ \sum p(s^n|x^n) d_s(s^n, \hat{s}^n) \right] = \mathbb{E}\hat{d}_s(X^n, \hat{S}^n)$$
  * This reduces the problem to a standard lossy source coding problem with multiple distortion constraints.

<!-- slide -->
## 5. Case Study 1: Jointly Gaussian Model
* **Setting**: $S$ and $X$ are jointly Gaussian vectors, and distortion metrics are squared errors.
* **Scalar Case Result**:
  $$R(D_s, D_a) = \frac{1}{2} \max \left\{ \log^+\frac{\Sigma_X}{D_a}, \, \log^+\frac{\Sigma_{SX}^2}{\Sigma_X(D_s - \text{mmse})} \right\}$$
* **Physical Interpretation**:
  * Since $S$ and $X$ are scalars, their directions are aligned.
  * The required rate is determined strictly by whichever constraint is more demanding (i.e., requires a lower distortion).
  * In both regimes, the optimal reconstructions $\hat{X}$ and $\hat{S}$ are proportional to each other.

<!-- slide -->
## 6. Case Study 2: Binary Classification
* **Setting**:
  * $S \in \{0, 1\}$ (uniform Bernoulli), representing class labels.
  * $X \sim \mathcal{N}(A, \sigma^2)$ if $S=0$, and $X \sim \mathcal{N}(-A, \sigma^2)$ if $S=1$.
  * Semantic distortion: Hamming distortion; Appearance distortion: Squared error.
* **Key Finding**:
  * The study derives $R(D_s, \infty)$ for the unconstrained appearance case.
  * **Critical Insight**: The naive approach of performing local (Bayesian) classification first and then compressing the binary label is **suboptimal** compared to the joint rate-distortion optimization proposed here.

<!-- slide -->
## 7. Conclusion & Future Work
* **Contributions**:
  * Proposed a unified framework modeling information sources via unobservable semantic states and observable appearances.
  * Characterized the trade-off boundaries (SORDF) and highlighted the suboptimality of isolated classification-then-compression schemes.
* **Future Directions**:
  * Development of practical numerical algorithms to compute the SORDF for general sources.
  * Estimating the SORDF when only finite training data of the $(S, X)$ pair is available.

<!-- slide -->
# Thank You!