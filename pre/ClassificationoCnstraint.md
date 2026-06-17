# Innovation Idea: Rate-Distortion-Perception-Classification (RDPC) Framework
Presenter: [Your Name]

---

## 1. Background & The Gap
* **Our Baseline (2021 Paper)**:
  * Balances **Semantic State $D_s$** and **Appearance $D_a$** [2021].
  * *The Limitation*: Relies strictly on pixel-level MSE. Reconstructed images are often **blurry and look unrealistic**.
* **The Inspiration (2025 JSAC Paper)**:
  * Introduces **Perceptual Quality ($P$)** alongside **Classification ($C$)** and **Distortion ($D$)** .
  * Uses GAN-based distribution matching to ensure sharp, realistic reconstructed images .

---

## 2. Our Proposed Fusion: S-RDP Framework
* **Objective**: Introduce the **Perception Constraint $P$** into our baseline semantic communication system.
* **The S-RDP Function $R(D, P, C)$**:
  * We compress $X$. The decoder reconstructs $\hat{X}$ and infers class label $S$ .
  * We minimize the code rate $R$ under three explicit constraints :
    1. **Reconstruction Distortion ($D$)**: $\mathbb{E}[\Delta(X, \hat{X})] \le D$ (Pixel accuracy)
    2. **Perception Quality ($P$)**: $d(p_X, p_{\hat{X}}) \le P$ (Visual realism)
    3. **Classification Uncertainty ($C$)**: $H(S|\hat{X}) \le C$ (Semantic accuracy)

---

## 3. Mathematical Formulation
* **The Unified RDPC Optimization Problem**:
  $$R(D, P, C) = \min_{p(\hat{x}|x)} I(X; \hat{X})$$
  $$\text{s.t. } \mathbb{E}[\Delta(X, \hat{X})] \le D \quad \text{(Reconstruction Constraint)}$$
  $$d_{\text{W}}(p_X, p_{\hat{X}}) \le P \quad \text{(Perception Constraint)}$$
  $$H(S|\hat{X}) \le C \quad \text{(Classification Constraint)}$$
* We use **Wasserstein distance ($d_{\text{W}}$)** to bound perception quality $P$ .
* We use **Conditional Entropy ($H$)** to bound classification accuracy $C$ .

---

## 4. Practical Implementation (Deep Learning Pipeline)
* We can implement this 3-constraint trade-off via **Multi-Task Learning**:
* **Network Components**:
  * **Autoencoder**: Compresses image $X$ and decodes it to $\hat{X}$ [10].
  * **Discriminator**: Binds **Perception ($P$)** by evaluating if $\hat{X}$ looks real (GAN loss) .
  * **Pre-trained Classifier**: Binds **Classification ($C$)** via cross-entropy loss on $\hat{X}$ .
* **Loss Function in PyTorch**:
  $$\mathcal{L} = \lambda_d \text{MSE}(x, \hat{x}) + \lambda_p \text{GAN\_Loss}(p_x, p_{\hat{x}}) + \lambda_c \text{CrossEntropy}(s, \hat{s})$$

---

## 5. Expected Insights & Research Questions
* **Research Question 1: Mapping the 3D Trade-off Surfaces**
  * How do perception weight $\lambda_p$ and classification weight $\lambda_c$ interact under a constrained rate $R$ ?
* **Research Question 2: Suboptimality under Perceptual Realism**
  * Does the "suboptimality of local classification + compression" (from 2021 Paper) worsen or shrink when we enforce high perceptual realism (small $P$) ?
* **Datasets**: Evaluation on MNIST and SVHN datasets .