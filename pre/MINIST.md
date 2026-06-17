# Validating the Semantic Rate-Distortion Framework on MNIST
Presenter: [Your Name]

---

## 1. Mapping MNIST to Paper 1 Formulation
* **Extrinsic Observation $X$**:
  * The $28 \times 28$ pixel image of a handwritten digit (e.g., the appearance of "5").
* **Intrinsic State $S$**:
  * The true category label $S \in \{0, 1, \dots, 9\}$ (the semantic meaning).
* **The Constraint**:
  * The encoder only observes image $X$. It must compress $X$ such that the decoder can reconstruct both image $\hat{X}$ and predict label $\hat{S}$.

---

## 2. Key Formulas Applied
* **Appearance Distortion $d_a(x, \hat{x})$**:
  * Measured by Mean Squared Error (MSE):
    $$d_a(x, \hat{x}) = \|x - \hat{x}\|_2^2$$
* **Semantic Distortion $d_s(s, \hat{s})$**:
  * Measured by Hamming Distortion (1 if incorrect, 0 if correct).
* **Equivalent Semantic Distortion (Eq. 8 in Paper)**:
  $$\hat{d}_s(x, \hat{s}) = \sum_{s} p(s|x) d_s(s, \hat{s}) = 1 - p(S = \hat{s} | X = x)$$
  * *Meaning*: The equivalent semantic loss is exactly the conditional error probability of classification!

---

## 3. Deep Learning Architecture
* **Encoder $f_{\phi}$**:
  * Input: $X \in \mathbb{R}^{28 \times 28} \to$ Bottleneck Latent $Z \in \mathbb{R}^k$.
  * (Rate $R$ is controlled by latent dimension $k$).
* **Appearance Decoder $g_{\theta_a}$**:
  * Input: $Z \to$ Reconstructed Image $\hat{X} \in \mathbb{R}^{28 \times 28}$.
* **Semantic Decoder $g_{\theta_s}$**:
  * Input: $Z \to$ Label Probability distribution $\hat{S} \in \mathbb{R}^{10}$.

---

## 4. Training Pipeline & Objective Loss
* **Multi-Task Optimization**:
  * We train the network jointly using the following loss function:
    $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(X, \hat{X}) + \beta \cdot \mathcal{L}_{\text{CE}}(S, \hat{S})$$
  * **$\mathcal{L}_{\text{MSE}}$**: Binds physical reconstruction quality ($D_a$).
  * **$\mathcal{L}_{\text{CE}}$ (Cross-Entropy)**: Binds semantic classification error ($D_s$).
  * **$\beta$**: Lagrange multiplier controlling the trade-off.

---

## 5. Expected Results & Verification
* **Trade-off Curve**:
  * By varying $\beta$, we trace the trade-off between reconstruction MSE ($D_a$) and classification error ($D_s$).
* **Verifying Suboptimality**:
  * Compare our joint encoder with the "naive classification-first" scheme.
  * *Result*: The joint scheme achieves a better Rate-Distortion-Inference boundary, especially at low code rates.