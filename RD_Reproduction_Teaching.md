# A Rate-Distortion Framework for Characterizing Semantic Information —— 复现教学文档

> 本文档结合论文原文、数学推导和工程实现，详细解释 RD-Semantic 框架中每个命题的复现过程。
> 源代码位于 `rd-semantic-repro/` 目录。

---

## 目录

1. [问题建模 (Problem Formulation)](#1-问题建模)
2. [SORDF 的一般特征 (Theorem 1)](#2-sordf-的一般特征)
3. [联合高斯模型：标量情形 (Proposition 3)](#3-标量高斯-sordf)
4. [联合高斯模型：向量情形 (Proposition 4)](#4-向量高斯-sordf)
5. [二分类情形：无限外观失真 (Proposition 5)](#5-二分类无限外观失真)
6. [二分类情形：有限外观失真上界 (Proposition 6)](#6-二分类有限外观失真上界)
7. [测试即定理验证](#7-测试即定理验证)

---

## 1. 问题建模

### 1.1 论文原文

> We describe a memoryless information source as a tuple of random variables, $(S, X)$ with joint probability distribution $p(s, x)$ in product alphabet $\mathcal{S} \times \mathcal{X}$. We interpret $S$ as the **intrinsic state**, which captures the "semantic" aspect of the source and is not observable, and $X$ as the **extrinsic observation** of the source, which captures the "appearance" of the source to an observer.
>
> For a length-$n$ i.i.d. sequence from the source, $(S^n, X^n)$, a source encoder $f_n$ of rate $R$ is a mapping that maps $X^n$ into an index $W$ within $\{1, 2, \ldots, 2^{nR}\}$, and a corresponding decoder $g_n$ is a mapping that maps $W$ into a pair $(\hat{S}^n, \hat{X}^n)$.
>
> We consider two distortion metrics: $d_s(s, \hat{s}): \mathcal{S} \times \hat{\mathcal{S}} \mapsto \mathbb{R}^+ \cup \{0\}$ that models the **semantic distortion**, and $d_a(x, \hat{x}): \mathcal{X} \times \hat{\mathcal{X}} \mapsto \mathbb{R}^+ \cup \{0\}$ that models the **appearance distortion**.

> A rate-distortion triple $(R, D_s, D_a)$ is achievable if there exists a sequence of encoders $\{f_n\}$ and decoders $\{g_n\}$ at rate $R$ such that as $n$ grows without bound:
> $$\lim_{n \to \infty} \mathbb{E} d_s(S^n, \hat{S}^n) \leq D_s, \quad \lim_{n \to \infty} \mathbb{E} d_a(X^n, \hat{X}^n) \leq D_a$$

> The boundary of the set of all achievable rate-distortion triples is defined as the **state-observation rate-distortion function (SORDF)**.

### 1.2 数学解释

这是一个**间接率失真问题 (indirect rate-distortion problem)**。关键点：

- 编码器只能看到 $X$（外观），不能看到 $S$（语义状态）
- 但解码器需要同时重建 $\hat{S}$ 和 $\hat{X}$
- 存在两个失真约束：语义失真 $D_s$ 和外观失真 $D_a$

模型的核心马尔可夫链：

$$S \leftrightarrow X \leftrightarrow (\hat{S}, \hat{X})$$

这意味着 $\hat{S}$ 和 $\hat{X}$ 只能通过 $X$ 与 $S$ 关联。

**速率-失真函数 (SORDF)** 定义为在满足两个失真约束下所需的最小压缩速率：

$$R(D_s, D_a) = \min I(X; \hat{S}, \hat{X})$$
$$\text{s.t. } \mathbb{E} d_s(S, \hat{S}) \leq D_s, \quad \mathbb{E} d_a(X, \hat{X}) \leq D_a$$

### 1.3 工程实现

工程实现中不直接优化 $I(X; \hat{S}, \hat{X})$，而是对每个具体命题给出闭式解或数值求解方案。

**代码结构概览：**

```
rd-semantic-repro/
├── src/
│   ├── __init__.py
│   ├── gaussian_case.py    # 命题3和命题4的实现
│   └── binary_case.py      # 命题5和命题6的实现
├── experiments/             # 绘图脚本
│   ├── plot_gaussian_scalar.py
│   ├── plot_gaussian_vector.py
│   ├── plot_binary_curve.py
│   ├── plot_binary_soft_decision.py
│   ├── plot_binary_curve_family.py
│   └── plot_binary_upper_bound.py
├── tests/                   # 测试 = 定理验证
│   ├── test_gaussian_case.py
│   └── test_binary_case.py
└── figures/                 # 输出图表和CSV数据
```

---

## 2. SORDF 的一般特征

### 2.1 论文原文 (Theorem 1)

> **Theorem 1:** The SORDF of the problem setup is
> $$R(D_s, D_a) = \min I(X; \hat{S}, \hat{X})$$
> $$\text{s.t. } \mathbb{E} \hat{d}_s(X, \hat{S}) \leq D_s, \quad \mathbb{E} d_a(X, \hat{X}) \leq D_a$$
> where
> $$\hat{d}_s(x, \hat{s}) = \frac{1}{p(x)} \sum_{s \in \mathcal{S}} p(s, x) d_s(s, \hat{s})$$

### 2.2 数学解释

**核心技巧：将不可观测的语义失真转化为可观测的等价失真。**

原始语义失真约束：
$$\mathbb{E} d_s(S, \hat{S}) \leq D_s$$

由于 $S \leftrightarrow X \leftrightarrow \hat{S}$ 是马尔可夫链，可以证明：

$$\mathbb{E} d_s(S^n, \hat{S}^n) = \mathbb{E} \hat{d}_s(X^n, \hat{S}^n)$$

其中 $\hat{d}_s(x, \hat{s}) = \mathbb{E}[d_s(S, \hat{s}) | X = x]$ 是条件期望。

这意味着——编码器虽然看不到 $S$，但可以通过 $X$ 和条件分布 $p(s|x)$ 计算出满足语义失真约束所需的等价条件。问题因而转化为标准的**多约束率失真问题**。

**基本性质**（命题1）：

1. **单调非增性**：$R(D_s, D_a)$ 随 $D_s$、$D_a$ 增大而减小——放宽约束→需要更少比特
2. **联合凸性**：率失真函数是 $(D_s, D_a)$ 的凸函数——体现了"边际效用递减"
3. **等值集凸性**：对于任意 $R \geq 0$，使得 $R(D_s, D_a) \leq R$ 的 $(D_s, D_a)$ 集合是凸的

### 2.3 工程实现

Theorem 1 本身是理论结果，不直接实现。它在工程上体现为：

- **每个具体命题的求解都遵循 Theorem 1 的框架**
- **测试代码中编码了单调性和凸性的验证**（例如：放宽 $D_s$ 时 rate 必须单调非增）
- 等价失真度量 $\hat{d}_s$ 在命题 3/4（高斯）和命题 5（二分类）中分别取不同形式

## 3. 标量高斯 SORDF

### 3.1 论文模型

考虑 $S$ 和 $X$ 是标量且联合高斯，均值为零，协方差矩阵为：

$$\begin{bmatrix} \Sigma_S & \Sigma_{SX} \ \Sigma_{SX} & \Sigma_X \end{bmatrix}$$

失真度量均为平方误差：$d_s(s, \hat{s}) = (s - \hat{s})^2$，$d_a(x, \hat{x}) = (x - \hat{x})^2$。

### 3.2 数学解释

#### 等价语义失真

条件于 $X=x$，$S$ 是条件高斯的：

$$S|x \sim \mathcal{N}\left(\Sigma_{SX}\Sigma_X^{-1} x,\; \Sigma_S - \Sigma_{SX}^2/\Sigma_X\right)$$

因此等价语义失真为：

$$\hat{d}_s(x, \hat{s}) = \mathbb{E}_{S|x}[(S - \hat{s})^2] = \underbrace{(\Sigma_S - \Sigma_{SX}^2/\Sigma_X)}_{\text{mmse}} + (\Sigma_{SX}\Sigma_X^{-1} x - \hat{s})^2$$

**关键观察：** 第一项是给定 $X$ 时估计 $S$ 的最小均方误差 (MMSE)，为不可压缩的常数。只有第二项受编码影响。因此语义失真约束变为：

$$\mathbb{E}[(\Sigma_{SX}\Sigma_X^{-1} X - \hat{S})^2] \leq D_s - \text{mmse}$$

要求 $D_s > \text{mmse}$，即语义失真约束必须大于最优估计器本身的误差。

#### 标量 SORDF 的闭式解

**论文原文 (Proposition 3):**

> For the jointly Gaussian source model where both $S$ and $X$ are scalar, its SORDF is given by
> $$R(D_s, D_a) = \frac{1}{2} \max\left\{ \left[\log \frac{\Sigma_X}{D_a}\right]^+, \; \left[\log \frac{\Sigma_{SX}^2}{\Sigma_X(D_s - \text{mmse})}\right]^+ \right\}$$
> for $D_s > \text{mmse}, D_a > 0$, where $(x)^+$ denotes $\max\{x, 0\}$.

#### 数学推导思路

**下界 (converse):** $R(D_s, D_a)$ 必须同时满足两个独立的下界：
- 仅约束外观：标准高斯率失真 $\frac{1}{2}\left[\log \frac{\Sigma_X}{D_a}\right]^+$
- 仅约束语义：由 Proposition 2，等价于压缩 $\Sigma_{SX}\Sigma_X^{-1}X$，其方差为 $\Sigma_{SX}^2/\Sigma_X$，因此率为 $\frac{1}{2}\left[\log \frac{\Sigma_{SX}^2}{\Sigma_X(D_s - \text{mmse})}\right]^+$

故 $R(D_s, D_a) \geq \max\{\text{两个下界}\}$。

**上界 (achievability):** 考虑两种情形：

| 情形 | 条件 | 策略 | 实现速率 |
|------|------|------|----------|
| 语义主导 | $\frac{D_a}{\Sigma_X} \geq \frac{D_s - \text{mmse}}{\Sigma_{SX}^2/\Sigma_X}$ | 以语义为优化目标压缩，令 $\hat{X} = \frac{\Sigma_X}{\Sigma_{SX}}\hat{S}$ | $\frac{1}{2}\log\frac{\Sigma_{SX}^2}{\Sigma_X(D_s-\text{mmse})}$ |
| 外观主导 | $\frac{D_a}{\Sigma_X} < \frac{D_s - \text{mmse}}{\Sigma_{SX}^2/\Sigma_X}$ | 以外观为优化目标压缩，令 $\hat{S} = \frac{\Sigma_{SX}}{\Sigma_X}\hat{X}$ | $\frac{1}{2}\log\frac{\Sigma_X}{D_a}$ |

在标量情况下，$\hat{S}$ 和 $\hat{X}$ 总是成比例的——因为它们方向完全"对齐"。两个约束中较紧的那个决定最终的速率。

### 3.3 工程实现

#### 源码：`src/gaussian_case.py`

```python
from math import log2


def log_plus(value: float) -> float:
    """(x)^+ = max{0, log2(x)}"""
    return max(0.0, log2(value))


def gaussian_scalar_sordf(sigma_x: float, sigma_sx: float, mmse: float,
                          Ds: float, Da: float) -> float:
    if Ds <= mmse:
        raise ValueError("Ds must be larger than mmse.")
    if Da <= 0.0:
        raise ValueError("Da must be positive.")

    # 外观主导项
    appearance_rate = log_plus(sigma_x / Da)
    # 语义主导项
    semantic_rate = log_plus((sigma_sx * sigma_sx) / (sigma_x * (Ds - mmse)))

    return 0.5 * max(appearance_rate, semantic_rate)
```

**逐行对应论文：**

| 代码 | 论文公式 |
|------|----------|
| `sigma_x / Da` | $\Sigma_X / D_a$ |
| `(sigma_sx * sigma_sx) / (sigma_x * (Ds - mmse))` | $\Sigma_{SX}^2 / (\Sigma_X (D_s - \text{mmse}))$ |
| `log_plus(...)` | $[\log(\cdot)]^+$ |
| `0.5 * max(...)` | $\frac{1}{2}\max\{\text{appearance}, \text{semantic}\}$ |

#### 验证测试：`tests/test_gaussian_case.py`

测试编码了命题 3 的三种核心行为：

```python
def test_scalar_gaussian_sordf_matches_proposition_3_appearance_dominates():
    # σ_x=2, σ_sx=1, mmse=0.25
    # 外观项: 0.5*log2(2/0.5) = 0.5*2 = 1.0
    # 语义项: 0.5*log2(1/(2*(2-0.25))) = 0.5*log2(1/3.5) < 0 → 0
    # max = 1.0
    rate = gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=2.0, Da=0.5)
    assert math.isclose(rate, 1.0, rel_tol=1e-12)

def test_scalar_gaussian_sordf_matches_proposition_3_semantics_dominates():
    # σ_x=2, σ_sx=1, mmse=0.25
    # 外观项: 0.5*log2(2/2) = 0.5*0 = 0
    # 语义项: 0.5*log2(1/(2*(0.5-0.25))) = 0.5*log2(1/0.5) = 0.5*1 = 0.5
    # max = 0.5
    rate = gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=0.5, Da=2.0)
    assert math.isclose(rate, 0.5, rel_tol=1e-12)

def test_scalar_gaussian_sordf_is_zero_when_both_distortions_are_loose():
    # 两个 log 参数都 < 1 → log 为负 → log_plus 返回 0
    rate = gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=2.0, Da=3.0)
    assert math.isclose(rate, 0.0, abs_tol=1e-12)
```

#### 可视化：`experiments/plot_gaussian_scalar.py`

生成 $(D_s, D_a)$ 平面上的等高线图，展示哪些区域由外观约束主导、哪些由语义约束主导。

## 4. 向量高斯 SORDF

### 4.1 论文模型

考虑特殊向量情形：$S$ 是标量，$X$ 是 $m$ 维向量。

$$X = \mathbf{1}_m S + Z$$

其中：
- $S \sim \mathcal{N}(0, \sigma_S^2)$ — 标量语义状态
- $\mathbf{1}_m$ — 全1向量（长度 $m$）
- $Z \sim \mathcal{N}(0, \sigma_Z^2 I)$ — $m$ 维独立同分布噪声

定义 MMSE 和辅助参数 $\alpha$：

$$\text{mmse} = \frac{\sigma_S^2 \sigma_Z^2}{m\sigma_S^2 + \sigma_Z^2}, \quad \alpha = \frac{m\sigma_S^2 + \sigma_Z^2}{\sqrt{m\sigma_S^2}}$$

### 4.2 数学解释

#### 核心思想：坐标变换→独立化→并行高斯水注入

**关键观察：** $\Sigma_{SX}\Sigma_X^{-1} \propto \mathbf{b}_1 = \frac{1}{\sqrt{m}}\mathbf{1}_m$，这是 $\Sigma_X$ 的一个单位特征向量，对应特征值 $m\sigma_S^2 + \sigma_Z^2$。

其余 $m-1$ 个特征向量与 $\mathbf{b}_1$ 正交，且全部对应同一特征值 $\sigma_Z^2$。构造正交矩阵 $B = [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_m]^T$ 后，$BX$ 的 $m$ 个分量**互相独立**：

- $b_1^T X \sim \mathcal{N}(0, m\sigma_S^2 + \sigma_Z^2)$ — **含语义信息的通道**
- $b_i^T X \sim \mathcal{N}(0, \sigma_Z^2), i=2,\ldots,m$ — **纯噪声通道**

问题转化为对 $m$ 个独立高斯分量的**失真分配问题 (distortion allocation)**：

$$R(D_s, D_a) = \min_{(D_1,\ldots,D_m) \in \mathcal{A}(D_s,D_a)} \left\{ R\left(\frac{D_1}{m\sigma_S^2 + \sigma_Z^2}\right) + \sum_{i=2}^{m} R\left(\frac{D_i}{\sigma_Z^2}\right) \right\}$$

$$\mathcal{A}(D_s,D_a) = \left\{ (D_1,\ldots,D_m) : D_1 \leq \alpha^2(D_s - \text{mmse}), \; \sum_{i=1}^{m} D_i \leq D_a, \; D_i \geq 0 \right\}$$

其中 $R(x) = \frac{1}{2}\left[\log \frac{1}{x}\right]^+$ 是标准高斯率失真函数。

**关键约束：**
- $D_1$ 受语义约束限制：$D_1 \leq \alpha^2(D_s - \text{mmse})$
- $\sum D_i$ 受外观约束限制：$\sum D_i \leq D_a$
- $D_i \geq 0$：失真不能为负

#### 五区域划分

优化问题的解将 $(D_s, D_a)$ 平面分为五个区域（见论文 Figure 2 和 Figure 3）：

| 区域 | 条件 | 物理意义 |
|------|------|----------|
| S1 (零速率) | $D_a \geq m\sigma_S^2 + m\sigma_Z^2$ 且语义带 $\geq m\sigma_S^2 + \sigma_Z^2$ | 两个约束都极松，完全无需编码 |
| S2 (外观主导) | 语义带 $\geq D_a/m$ 且 $D_a < m\sigma_Z^2$ | 外观失真约束主导，需严格编码所有分量 |
| S3 (语义主导) | 语义带 $< m\sigma_S^2 + \sigma_Z^2$ 且 $D_a >$ 语义带 $+ (m-1)\sigma_Z^2$ | 语义失真约束主导，外观约束可自然满足 |
| S4 (混合边界) | $m\sigma_Z^2 \leq D_a < m\sigma_S^2 + m\sigma_Z^2$ 且语义带 $\geq D_a - (m-1)\sigma_Z^2$ | 两约束在边界处相互制约 |
| TRADEOFF | $m\alpha^2(D_s - \text{mmse}) \leq D_a \leq \alpha^2(D_s - \text{mmse}) + (m-1)\sigma_Z^2$ | **语义-外观之间存在真正的权衡：优化语义vs外观的比特分配相互竞争** |

**论文原文 (Proposition 4):**

> - **TRADEOFF**: $m\alpha^2(D_s - \text{mmse}) \leq D_a \leq \alpha^2(D_s - \text{mmse}) + (m-1)\sigma_Z^2$
>   $$R = \frac{1}{2}\log\frac{\lambda_1}{\alpha^2(D_s - \text{mmse})} + \frac{m-1}{2}\log\frac{(m-1)\sigma_Z^2}{D_a - \alpha^2(D_s - \text{mmse})}$$
>
> - **外观主导 (S2)**: $0 \leq D_a < m\sigma_Z^2$ 且 $\alpha^2(D_s - \text{mmse}) \geq D_a/m$
>   $$R = \frac{1}{2}\log\frac{m^2\sigma_S^2 + m\sigma_Z^2}{D_a} + \frac{m-1}{2}\log\frac{m\sigma_Z^2}{D_a}$$
>
> - **语义主导 (S3)**: $0 \leq \alpha^2(D_s - \text{mmse}) < m\sigma_S^2 + \sigma_Z^2$ 且 $D_a > \alpha^2(D_s - \text{mmse}) + (m-1)\sigma_Z^2$
>   $$R = \frac{1}{2}\log\frac{m\sigma_S^2 + \sigma_Z^2}{\alpha^2(D_s - \text{mmse})}$$
>
> - **混合边界 (S4)**: $m\sigma_Z^2 \leq D_a < m\sigma_S^2 + m\sigma_Z^2$ 且 $\alpha^2(D_s - \text{mmse}) \geq D_a - (m-1)\sigma_Z^2$
>   $$R = \frac{1}{2}\log\frac{m\sigma_S^2 + \sigma_Z^2}{D_a - (m-1)\sigma_Z^2}$$
>
> - **零速率 (S5)**: $D_a \geq m\sigma_S^2 + m\sigma_Z^2$ 且 $\alpha^2(D_s - \text{mmse}) \geq m\sigma_S^2 + \sigma_Z^2$
>   $$R = 0$$

其中 $\lambda_1 = m\sigma_S^2 + \sigma_Z^2$，且用 $\alpha^2(D_s - \text{mmse})$ 表示"语义带"。

### 4.3 工程实现

#### 源码：`src/gaussian_case.py`

**辅助参数计算：**

```python
def gaussian_vector_parameters(m: int, sigma_s2: float, sigma_z2: float):
    """计算 MMSE 和 alpha 参数"""
    mmse = sigma_s2 * sigma_z2 / (m * sigma_s2 + sigma_z2)
    alpha = (m * sigma_s2 + sigma_z2) / ((m * sigma_s2) ** 0.5)
    return mmse, alpha
```

| 代码 | 论文公式 |
|------|----------|
| `sigma_s2 * sigma_z2 / (m * sigma_s2 + sigma_z2)` | $\text{mmse} = \frac{\sigma_S^2 \sigma_Z^2}{m\sigma_S^2 + \sigma_Z^2}$ |
| `(m * sigma_s2 + sigma_z2) / sqrt(m * sigma_s2)` | $\alpha = \frac{m\sigma_S^2 + \sigma_Z^2}{\sqrt{m\sigma_S^2}}$ |

**五区域 SORDF 实现：**

```python
def gaussian_vector_sordf(m: int, sigma_s2: float, sigma_z2: float,
                          Ds: float, Da: float) -> float:
    if m < 2:
        raise ValueError("m must be at least 2.")  # 标量情况已有 Prop 3
    if Da < 0.0:
        raise ValueError("Da must be nonnegative.")

    mmse, alpha = gaussian_vector_parameters(m, sigma_s2, sigma_z2)
    if Ds < mmse:
        raise ValueError("Ds must be at least mmse.")  # 低于贝叶斯最优误差

    semantic_band = alpha * alpha * (Ds - mmse)
    lambda_1 = m * sigma_s2 + sigma_z2      # Σ_X 的最大特征值
    lambda_noise = sigma_z2                  # Σ_X 的其余 m-1 个特征值

    # 区域 S5：零速率
    if Da >= m * sigma_s2 + m * sigma_z2 and semantic_band >= lambda_1:
        return 0.0

    # 区域 S2：外观主导
    if semantic_band >= Da / m and Da < m * sigma_z2:
        return (0.5 * log_plus((m*m*sigma_s2 + m*sigma_z2) / Da) +
                (m-1) * 0.5 * log_plus((m*sigma_z2) / Da))

    # 区域 S3：语义主导
    if semantic_band < lambda_1 and Da > semantic_band + (m-1) * sigma_z2:
        return 0.5 * log_plus(lambda_1 / semantic_band)

    # 区域 S4：混合边界
    if (m * sigma_z2 <= Da < m * sigma_s2 + m * sigma_z2
            and semantic_band >= Da - (m-1) * sigma_z2):
        return 0.5 * log_plus(lambda_1 / (Da - (m-1) * sigma_z2))

    # 区域 TRADEOFF（未显式列出但数学上最后命中）
    if (m * semantic_band <= Da <= semantic_band + (m-1) * sigma_z2):
        return (0.5 * log_plus(lambda_1 / semantic_band) +
                (m-1) * 0.5 * log_plus(
                    ((m-1) * sigma_z2) / (Da - semantic_band)))

    return 0.0  # fallback
```

**逐行对应论文公式（以 tradeoff 区域为例）：**

| 代码表达式 | 论文公式 |
|------------|----------|
| `lambda_1 / semantic_band` | $\frac{m\sigma_S^2 + \sigma_Z^2}{\alpha^2(D_s - \text{mmse})}$ |
| `((m-1) * sigma_z2) / (Da - semantic_band)` | $\frac{(m-1)\sigma_Z^2}{D_a - \alpha^2(D_s - \text{mmse})}$ |
| `0.5 * log_plus(...) + (m-1) * 0.5 * log_plus(...)` | $\frac{1}{2}\log(\cdot) + \frac{m-1}{2}\log(\cdot)$ |

**条件判断顺序说明：**
代码中按 S5 → S2 → S3 → S4 → TRADEOFF 的顺序判断，因为各区域的条件互斥。最先匹配到的返回。理论上也可以用 if-elif 链，但此处用了独立的 if+return 模式。

#### 验证测试：`tests/test_gaussian_case.py`

```python
def test_vector_gaussian_sordf_region_1_tradeoff_band():
    """TRADEOFF 区域：m=2, σ_S²=1, σ_Z²=1, mmse=1/3, α²=4.5
       Ds=4/9 (刚好>mmse), Da=1
       semantic_band = 4.5*(4/9-1/3) = 4.5*(1/9) = 0.5
       λ₁=3
       R = 0.5*log₂(3/0.5) + 0.5*log₂(1/(1-0.5)) = 0.5*log₂(6) + 0.5*log₂(2)"""
    rate = gaussian_vector_sordf(m=2, sigma_s2=1.0, sigma_z2=1.0,
                                 Ds=4.0/9.0, Da=1.0)
    expected = 0.5 * math.log2(6.0) + 0.5 * math.log2(2.0)
    assert math.isclose(rate, expected, rel_tol=1e-12)

# 类似地测试区域 2/3/4/5，每个使用从论文条件精选的参数确保命中目标区域
```

**测试设计原则：**
- 每个区域一个测试函数
- 参数经过手工计算验证，确保落在目标区域且产生精确可预期的速率值
- `rel_tol=1e-12` 保证浮点运算下仍能匹配理论闭式解

#### 可视化：`experiments/plot_gaussian_vector.py`

```python
M = 2
SIGMA_S2 = 1.0
SIGMA_Z2 = 1.0
MMSE, ALPHA = gaussian_vector_parameters(M, SIGMA_S2, SIGMA_Z2)

# 在 (Ds, Da) 平面上采样 90×90 = 8100 个点
DS_VALUES = np.linspace(MMSE + 0.01, 1.15, 90)
DA_VALUES = np.linspace(0.05, M * SIGMA_S2 + M * SIGMA_Z2 + 0.4, 90)

# 对每个点计算 gaussian_vector_sordf
for i, Ds in enumerate(DS_VALUES):
    for j, Da in enumerate(DA_VALUES):
        rate = gaussian_vector_sordf(M, SIGMA_S2, SIGMA_Z2, float(Ds), float(Da))
        rates[i, j] = rate

# 绘制等高线 + 两条斜边界线
plt.contourf(ds_grid, da_grid, rates, levels=24, cmap="magma")
# 斜边界 1: Da = m * α² * (Ds - mmse)
plt.plot(ds_boundary, M * semantic_band, "c--", ...)
# 斜边界 2: Da = α² * (Ds - mmse) + (m-1) * σ_Z²
plt.plot(ds_boundary, semantic_band + (M-1) * SIGMA_Z2, "w--", ...)
```

**两条斜边界线是 TRADEOFF 区域的上下界**，在图中清晰可见，它们标志着语义和外观约束之间从"独立"到"竞争"的转变。

## 5. 二分类：无限外观失真 $R(D_s, \infty)$

### 5.1 论文模型 (Section V)

**论文原文：**

> Consider the case where $S$ is a binary state, i.e., a Bernoulli random variable drawn from $\{0, 1\}$ with prior probability $1/2$ uniformly. The extrinsic observation $X$ is conditionally Gaussian, as
> $$X \sim \mathcal{N}(A, \sigma^2), \text{ if } S = 0; \quad X \sim \mathcal{N}(-A, \sigma^2), \text{ if } S = 1.$$
> So the marginal distribution of $X$ is a Gaussian mixture. We adopt a Hamming distortion between $S$ and $\hat{S}$, i.e., $d_s(s, \hat{s}) = 0$ if $s = \hat{s}$ and $1$ otherwise; and a squared error distortion between $X$ and $\hat{X}$.

### 5.2 数学解释

#### 信源模型的结构

$X$ 的边缘分布是**两个高斯分布的等权混合**。参数 $A$ 控制类别分离度：
- $A/\sigma$ 大 → 两类分得开 → 分类容易 → 贝叶斯误差小
- $A/\sigma$ 小 → 两类混在一起 → 分类困难 → 贝叶斯误差大

贝叶斯误差（最小可达到的分类错误率）为：

$$Q(A/\sigma) = \frac{1}{2}\text{erfc}\left(\frac{A/\sigma}{\sqrt{2}}\right)$$

**$D_s$ 必须满足：** $Q(A/\sigma) \leq D_s \leq 1/2$
- 下界：不可能比最优贝叶斯分类器做得更好
- 上界：$D_s = 1/2$ 意味着随机猜测（零信息），速率自然为 0

#### 等价语义失真

由于使用汉明失真，$\hat{d}_s(x, \hat{s})$ 简化为：

$$\mathbb{E} \hat{d}_s(X, \hat{S}) = \frac{1}{2} \int_{-\infty}^{\infty} \left[ N^-(x) g(x) + N^+(x) (1 - g(x)) \right] dx$$

其中：
- $N^+(x)$：$\mathcal{N}(A, \sigma^2)$ 的 PDF → "S=0 类的外观分布"
- $N^-(x)$：$\mathcal{N}(-A, \sigma^2)$ 的 PDF → "S=1 类的外观分布"
- $g(x) = \Pr(\hat{S}=0 | X=x)$：**软判决函数**，解码器在看到 $x$ 后输出 $\hat{S}=0$ 的概率

这是问题的核心优化变量。

#### 论文原文 (Proposition 5):

> For the source model (31), we have
> $$R(D_s, \infty) = 1 - \frac{1}{2} \int_{-\infty}^{\infty} \left[ N^+(x) + N^-(x) \right] h_2(g(x)) dx$$
> for $Q(A/\sigma) \leq D_s \leq 1/2$, and $R(D_s, \infty) = 0$ for $D_s > 1/2$, where
> $$g(x) = \left[ 1 + \exp\left( \lambda \frac{1 - e^{-2Ax/\sigma^2}}{1 + e^{-2Ax/\sigma^2}} \right) \right]^{-1}$$
> wherein $\lambda < 0$ is chosen so as to satisfy
> $$\int_{-\infty}^{\infty} \left[ N^+(x) - N^-(x) \right] g(x) dx = 1 - 2D_s.$$

#### 数学推导思路

**最优软判决的形式：**

优化问题：$\min I(X; \hat{S})$ s.t. 语义失真约束。

通过变分法或拉格朗日乘子法，可以得到最优 $g(x)$ 的隐式解——它是一个 sigmoid 形式的函数，其输入的尺度由拉格朗日乘子 $\lambda$ 控制：

$$g(x) = \frac{1}{1 + \exp(\lambda \cdot m(x))}$$

其中 $m(x) = \frac{1 - e^{-2Ax/\sigma^2}}{1 + e^{-2Ax/\sigma^2}}$ 是 $x$ 的**奇函数**，刻画了两类之间的对比度：

- $x \to +\infty$: $m(x) \to 1$ → $g(x) \to \frac{1}{1+e^{\lambda}} < 1$ （$\lambda < 0$ 时 > 0.5）
- $x \to -\infty$: $m(x) \to -1$ → $g(x) \to \frac{1}{1+e^{-\lambda}} > 0.5$
- $x = 0$: $m(0) = 0$ → $g(0) = 1/2$ → 最大不确定性

由于模型对称，$g(x) + g(-x) = 1$ 自然满足。

**$\lambda$ 与 $D_s$ 的关系：**
- $\lambda \to 0^-$: $g(x) \to 1/2$ 恒常 → $D_s = 1/2$（无信息，全随机猜测）
- $\lambda \to -\infty$: $g(x)$ 趋近硬判决 → $D_s \to Q(A/\sigma)$（贝叶斯最优分类器）

$\lambda$ 无法直接求解——它是通过**数值根搜索**从失真约束方程中找到的。

**速率的计算：**

一旦找到 $\lambda$ 和 $g(x)$，速率由以下熵积分给出：

$$R(D_s, \infty) = 1 - \frac{1}{2} \int_{-\infty}^{\infty} \left[ N^+(x) + N^-(x) \right] h_2(g(x)) dx$$

其中 $h_2(p) = -p\log_2 p - (1-p)\log_2(1-p)$ 是二元熵函数。

**直觉：** 先验熵 $H(S) = 1$ bit，减去"从 $X$ 可以平均提取出多少关于 $\hat{S}$ 的确定性的信息"。

---

### 5.3 工程实现

#### 源码：`src/binary_case.py`

**核心函数调用链（自底向上）：**

```
classification_rd_infinite(A, sigma, Ds)
  ├── _find_lambda(A, sigma, Ds)         # 二分法求 λ
  │     └── _dist_constraint(lam, A, sigma, Ds)  # ∫ (N⁺-N⁻)g dx - (1-2Ds) = 0
  │           └── _compute_g(x, lam, A, sigma)    # g(x) 的计算
  └── _compute_rate_integral(lam, A, sigma)        # R = 1 - ½∫(N⁺+N⁻)h₂(g) dx
        └── _integrand_for_rate(x, lam, A, sigma)  # (N⁺+N⁻)h₂(g(x))
```

**Step 1: $g(x)$ 的计算 — 公式(33)**

```python
def _compute_g(x: float, lam: float, A: float, sigma: float) -> float:
    r = exp(-2.0 * A * x / (sigma * sigma))          # e^{-2Ax/σ²}
    numerator = 1.0 - r                               # 1 - e^{-2Ax/σ²}
    denominator = 1.0 + r                             # 1 + e^{-2Ax/σ²}
    if denominator == 0.0:
        return 0.5
    exponent = lam * numerator / denominator          # λ · (1-r)/(1+r)
    if exponent > 100.0:
        return 0.0                                    # exp 溢出保护
    if exponent < -100.0:
        return 1.0
    return 1.0 / (1.0 + exp(exponent))               # sigmoid
```

逐行对应论文：
| 代码 | 论文 |
|------|------|
| `r = exp(-2Ax/σ²)` | $e^{-2Ax/\sigma^2}$ |
| `numerator = 1 - r` | $1 - e^{-2Ax/\sigma^2}$ |
| `denominator = 1 + r` | $1 + e^{-2Ax/\sigma^2}$ |
| `numerator/denominator` | $\frac{1 - e^{-2Ax/\sigma^2}}{1 + e^{-2Ax/\sigma^2}}$ |
| `1/(1+exp(exponent))` | $\left[1 + \exp(\lambda \cdot \frac{1-e^{-2Ax/\sigma^2}}{1+e^{-2Ax/\sigma^2}})\right]^{-1}$ |

**Step 2: $\lambda$ 的根搜索 — 公式(34)的数值求解**

```python
def _dist_constraint(lam: float, A: float, sigma: float, Ds: float) -> float:
    """计算 ∫(N⁺-N⁻)g dx - (1-2Ds)，根搜索的目标函数"""
    n_pts = 2000
    x_min = -10.0 * sigma        # 积分截断：[-10σ, 10σ] 覆盖 99.9999% 概率质量
    x_max = 10.0 * sigma
    dx = (x_max - x_min) / n_pts
    integral = 0.0
    for i in range(n_pts):
        x = x_min + (i + 0.5) * dx     # 中点积分法
        n_plus = gaussian_pdf(x, mean=A, sigma=sigma)    # N⁺(x)
        n_minus = gaussian_pdf(x, mean=-A, sigma=sigma)  # N⁻(x)
        g = _compute_g(x, lam, A, sigma)
        integral += (n_plus - n_minus) * g * dx          # (N⁺-N⁻)g
    return integral - (1.0 - 2.0 * Ds)                    # = 0 时满足约束
```

**目标函数：** $f(\lambda) = \int (N^+(x) - N^-(x)) g(x; \lambda) dx - (1 - 2D_s) = 0$

**物理意义：** $\int (N^+ - N^-)g dx$ 是"正确分类概率 − 错误分类概率"的加权形式。当它等于 $1-2D_s$ 时，语义失真恰好为 $D_s$。

```python
def _find_lambda(A: float, sigma: float, Ds: float) -> float:
    """二分法求解 λ ∈ (-∞, 0)"""
    hi = -1e-6          # 上界：接近0（对应 Ds 接近 1/2）
    lo = -50.0          # 下界：很负（对应 Ds 接近贝叶斯误差）
    f_hi = _dist_constraint(hi, A, sigma, Ds)
    f_lo = _dist_constraint(lo, A, sigma, Ds)

    # 扩展搜索区间直到 f(hi)*f(lo) < 0（保证根在区间内）
    for _ in range(20):
        if f_hi * f_lo < 0.0:
            break
        lo *= 2.0        # 下界继续下移
        f_lo = _dist_constraint(lo, A, sigma, Ds)
        if f_hi * f_lo < 0.0:
            break
        hi /= 2.0        # 上界继续上移
        f_hi = _dist_constraint(hi, A, sigma, Ds)

    # 标准二分法，80 次迭代达到极高精度
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = _dist_constraint(mid, A, sigma, Ds)
        if f_mid == 0.0 or (hi - lo) < 1e-12:
            return mid
        if f_mid * f_lo < 0.0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return (lo + hi) / 2.0
```

**$\lambda$ 的物理含义：**

| $\lambda$ 值 | $g(x)$ 行为 | $D_s$ 含义 |
|-------------|------------|------------|
| $\lambda \to 0^-$ | $g(x) \approx 0.5$（平坦，无判断力） | $D_s \to 0.5$（随机猜测） |
| $\lambda \approx -1$ | 轻微倾斜（软判决） | $D_s$ 中等 |
| $\lambda \to -\infty$ | $g(x) \to$ 阶跃函数（硬判决） | $D_s \to Q(A/\sigma)$（贝叶斯最优） |

**Step 3: 速率积分 — 公式(32)**

```python
def _integrand_for_rate(x: float, lam: float, A: float, sigma: float) -> float:
    n_plus = gaussian_pdf(x, mean=A, sigma=sigma)
    n_minus = gaussian_pdf(x, mean=-A, sigma=sigma)
    g = _compute_g(x, lam, A, sigma)
    return (n_plus + n_minus) * binary_entropy(g)    # (N⁺+N⁻) · h₂(g(x))


def binary_entropy(p: float) -> float:
    """h₂(p) = -p·log₂(p) - (1-p)·log₂(1-p)"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def _compute_rate_integral(lam: float, A: float, sigma: float) -> float:
    n_pts = 2000
    x_min = -10.0 * sigma
    x_max = 10.0 * sigma
    dx = (x_max - x_min) / n_pts
    integral = 0.0
    for i in range(n_pts):
        x = x_min + (i + 0.5) * dx
        integral += _integrand_for_rate(x, lam, A, sigma) * dx
    return 1.0 - 0.5 * integral      # R = 1 - ½∫(N⁺+N⁻)h₂(g) dx
```

**数值积分细节：**
- 积分区间：$[-10\sigma, 10\sigma]$，这个范围覆盖了高斯分布超过 99.9999% 的概率质量
- 方法：复合中点规则 (composite midpoint rule)，2000 个等距节点的黎曼和
- 精度：对于论文级别的绘图需求完全足够

**顶层接口：**

```python
def classification_rd_infinite(A: float, sigma: float, Ds: float) -> float:
    bayes_error = gaussian_q(A / sigma)                # Q(A/σ)
    if Ds < bayes_error - 1e-12:
        raise ValueError("Ds below minimum achievable classification error")
    if Ds >= 0.5 - 1e-12:
        return 0.0                                      # 随机猜测无需速率
    if isclose(Ds, bayes_error, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0                                      # 在贝叶斯边界，R = 1 bit
    lam = _find_lambda(A, sigma, Ds)                    # 求解 λ
    return _compute_rate_integral(lam, A, sigma)       # 计算速率
```

三个特殊情况：
| $D_s$ 值 | 返回值 | 理由 |
|----------|--------|------|
| $D_s = Q(A/\sigma)$ | $R = 1$ | 最佳可能分类仍需1 bit（两类等概率需要1 bit区分） |
| $D_s = 0.5$ | $R = 0$ | 允许50%错误率→随机猜测即可→零速率 |
| $D_s < Q(A/\sigma)$ | 抛出错误 | 物理上不可能达到低于贝叶斯误差的分类精度 |

## 6. 二分类：有限外观失真上界

### 6.1 论文原文 (Proposition 6)

> Based upon Proposition 5, we have the following achievability result.
>
> For the source model (31), we have
> $$R(D_s, D_a) \leq \min_{D \in [Q(A/\sigma), D_s]} \left\{ R(D, \infty) + \frac{1}{2} \left[ \log \frac{\eta}{D_a} \right]^+ \right\}$$
> where
> $$\eta = \int_{-\infty}^{\infty} (x - \gamma)^2 \left[ N^+(x) + N^-(x) \right] g_D(x) dx,$$
> $$\gamma = \int_{-\infty}^{\infty} x \left[ N^+(x) + N^-(x) \right] g_D(x) dx,$$
> where $g_D(x)$ is given by (33) satisfying (34) whose right hand side is now replaced by $1 - 2D$.

### 6.2 数学解释

这是一个**可达到的上界**（不是精确的 SORDF）。构造策略是一个**两级编码方案**：

**编码方案：**

```
Step 1: 语义编码
  X → (用 Proposition 5 的方法) → Ŝ
  速率: R(D, ∞)，满足语义失真 D（其中 D ∈ [Q(A/σ), Ds]）

Step 2: 外观精炼
  条件于 Ŝ，计算残差 X - E[X|Ŝ]
  用 i.i.d. 高斯码本编码残差 → X̃
  速率: ½[log(η/Da)]⁺，其中 η = Var(X - E[X|Ŝ])
  这是条件高斯率失真函数的标准结果

Step 3: 解码器重建
  X̂ = X̃ + E[X|Ŝ]  （条件期望 + 精炼残差）
```

**为什么是最小化而不是固定 D：**
- 语义编码的速率 $R(D, \infty)$ 随 $D$ 增大而减小
- 但外观精炼的速率 $\frac{1}{2}[\log \eta/D_a]^+$ 随 $D$ 变化（通过 $g_D$ 影响 $\eta$）
- 因此需要在 $D$ 上做一维搜索，找到总速率最小的点

**条件方差 $\eta$ 的计算：**
- $\gamma = \mathbb{E}[X | \hat{S}]$：条件均值，由 $g_D(x)$ 加权
- $\eta = \text{Var}(X | \hat{S})$：条件方差，$X$ 中不能被 $\hat{S}$ 解释的剩余变异

论文 Figure 5 展示了三条参考线：
| 曲线 | 公式 | 含义 |
|------|------|------|
| $\frac{1}{2}[\log(\sigma^2/D_a)]^+$ | 最优情况 | 编码器和解码器都**完美知道 S** |
| 上界 (37) | 本文方案 | 仅编码器知道 X、解码器不知道 S |
| $\frac{1}{2}[\log((A^2+\sigma^2)/D_a)]^+$ | 朴素方案 | 直接压缩 X 而忽略语义结构 |

### 6.3 工程实现

#### 源码：`src/binary_case.py`

**条件矩的计算：**

```python
def compute_conditional_moments(A: float, sigma: float, lam: float) -> tuple:
    """计算 E[X|Ŝ] 和 Var(X|Ŝ)，用于外观精炼项"""
    n_pts = 2000
    x_min = -10.0 * sigma
    x_max = 10.0 * sigma
    dx = (x_max - x_min) / n_pts

    # 第一遍：计算归一化常数和条件均值 γ = E[X|Ŝ]
    norm = 0.0
    mean = 0.0
    for i in range(n_pts):
        x = x_min + (i + 0.5) * dx
        n_plus = gaussian_pdf(x, mean=A, sigma=sigma)
        n_minus = gaussian_pdf(x, mean=-A, sigma=sigma)
        g = _compute_g(x, lam, A, sigma)
        w = (n_plus + n_minus) * g     # 联合权重 ∝ p(x, ŝ=0)
        norm += w * dx                  # 归一化常数
        mean += x * w * dx              # 一阶矩累加

    if norm > 0.0:
        mean /= norm                    # γ = E[X|Ŝ]

    # 第二遍：计算条件方差 η = Var(X|Ŝ)
    var = 0.0
    for i in range(n_pts):
        x = x_min + (i + 0.5) * dx
        n_plus = gaussian_pdf(x, mean=A, sigma=sigma)
        n_minus = gaussian_pdf(x, mean=-A, sigma=sigma)
        g = _compute_g(x, lam, A, sigma)
        w = (n_plus + n_minus) * g
        var += (x - mean) ** 2 * w * dx
    if norm > 0.0:
        var /= norm                     # η = Var(X|Ŝ)
    return mean, var
```

| 变量 | 论文符号 | 物理含义 |
|------|----------|----------|
| `mean` | $\gamma$ | $\mathbb{E}[X \mid \hat{S}]$，基于语义推断的外观期望 |
| `var` | $\eta$ | $\text{Var}(X \mid \hat{S})$，语义推断后剩余的外观不确定性 |

**上界计算：**

```python
def rd_upper_bound(A: float, sigma: float, Ds: float, Da: float,
                   num_D_samples: int = 50) -> float:
    bayes_error = gaussian_q(A / sigma)
    if Ds < bayes_error - 1e-12:
        raise ValueError("Ds is below Bayes error.")
    if Da <= 0.0:
        raise ValueError("Da must be positive.")

    best_rate = float('inf')
    D_min = max(bayes_error, 1e-10)
    D_max = Ds

    if D_max <= D_min:
        D_vals = [D_min]
    else:
        # 在 [Q(A/σ), Ds] 区间均匀采样 50 个 D 值
        D_vals = [D_min + i * (D_max - D_min) / (num_D_samples - 1)
                  for i in range(num_D_samples)]

    for D in D_vals:
        if D >= 0.5:
            # 无信息语义编码
            rate_semantic = 0.0
            eta_val = (A*A + sigma*sigma)   # X 的总方差
        else:
            # 用 Proposition 5 求语义速率
            rate_semantic = classification_rd_infinite(A, sigma, D)

            if isclose(D, bayes_error, rel_tol=1e-9, abs_tol=1e-9):
                # 贝叶斯边界附近，λ 太小不稳定，轻微扰动
                lam = _find_lambda(A, sigma, D + 0.001)
            else:
                lam = _find_lambda(A, sigma, D)

            _, eta_val = compute_conditional_moments(A, sigma, lam)

        if eta_val <= 0.0:
            eta_val = 0.001  # 防止数值下溢

        # 外观精炼项：½[log(η/Da)]⁺
        rate_appearance = 0.5 * max(0.0, log2(eta_val / Da))

        total_rate = rate_semantic + rate_appearance
        if total_rate < best_rate:
            best_rate = total_rate

    return best_rate
```

**逐行对应论文公式(37)：**

| 代码 | 论文 |
|------|------|
| `classification_rd_infinite(A, sigma, D)` | $R(D, \infty)$ |
| `compute_conditional_moments(A, sigma, lam)` → $\eta$ | $\int (x-\gamma)^2 (N^+ + N^-) g_D(x) dx$ |
| `0.5 * max(0.0, log2(eta_val / Da))` | $\frac{1}{2} \left[ \log \frac{\eta}{D_a} \right]^+$ |
| `min(best_rate over D_vals)` | $\min_{D \in [Q(A/\sigma), D_s]}$ |

**D 值采样策略：**
- 在 $[Q(A/\sigma), D_s]$ 区间均匀采样 50 个点
- 对每个点先解 $\lambda$、再算语义速率、再算条件方差、最后加外观项
- 返回使得总速率最小的 D 值对应的速率

## 7. 测试即定理验证

测试文件不验证"软件正确性"，而是编码**论文中的数学性质作为可执行的检查**。

### 7.1 二分类测试：`tests/test_binary_case.py`

| 测试 | 验证的定理性质 | 实现 |
|------|--------------|------|
| `test_gaussian_q_matches_known_value` | $Q(1) = 0.158655...$ | 基线数值一致性 |
| `test_posterior_is_half_at_zero` | $x=0$ 时 $P(S=0|x) = 0.5$ | 模型对称性 |
| `test_rate_is_one_bit_at_bayes_error` | $D_s=Q(A/\sigma)$ 时 $R=1$ | 命题5边界条件 |
| `test_rate_decreases_as_Ds_increases` | $D_s \uparrow \implies R \downarrow$ | 命题1 单调性 |
| `test_rate_is_zero_when_Ds_is_half` | $D_s=0.5 \implies R=0$ | 命题5边界条件 |
| `test_rejects_Ds_below_bayes_error` | $D_s < Q \implies$ 不可行 | 物理约束 |
| `test_curve_sampler_includes_endpoints` | 端点 = $(Q, 0.5)$ | 数值一致性 |
| `test_soft_decision_symmetry` | $g(0)=0.5, g(x)+g(-x)=1$ | 命题5对称性 |
| `test_soft_decision_increases` | $g(x)$ 随 $x$ 单调递增 | 感知单调性 |
| `test_upper_bound_nonpositive_Da` | $D_a \leq 0$ 为非法 | 约束验证 |
| `test_finite_Da_nonincreasing` | $D_a \uparrow \implies R \downarrow$ | 命题1 单调性 |

### 7.2 高斯测试：`tests/test_gaussian_case.py`

| 测试 | 验证的定理性质 | 实现 |
|------|--------------|------|
| `test_scalar_appearance_dominates` | 外观约束主导时 $R=0.5\log(\Sigma_X/D_a)$ | 命题3情形1 |
| `test_scalar_semantics_dominates` | 语义约束主导时 $R=0.5\log(\Sigma_{SX}^2/(\Sigma_X(D_s-mmse)))$ | 命题3情形2 |
| `test_scalar_zero_rate` | 两约束都松时 $R=0$ | 命题3的 $(x)^+$ |
| `test_rejects_Ds_at_mmse` | $D_s = mmse \implies$ 不可行 | 物理约束 |
| `test_vector_region_1_tradeoff` | TRADEOFF区域公式 | 命题4 公式(20) |
| `test_vector_region_2_appearance` | 外观主导区域公式 | 命题4 公式(21) |
| `test_vector_region_3_semantics` | 语义主导区域公式 | 命题4 公式(22) |
| `test_vector_region_4_hybrid` | 混合边界区域公式 | 命题4 公式(23) |
| `test_vector_region_5_zero_rate` | 零速率区域 | 命题4 公式(24) |

### 7.3 测试运行

```bash
cd ~/zhuwl2022/it/rd-semantic-repro
python3 -m pytest -q
# 输出：26 passed
```

---

## 8. 完整复现流程

### 8.1 环境准备

```bash
ssh idata2
cd ~/zhuwl2022/it/rd-semantic-repro
python3 -m pip install -r requirements.txt
```

### 8.2 运行所有实验

```bash
# 1. 先运行需要计算的实验
python3 experiments/run_binary_curve.py

# 2. 生成所有图表
python3 experiments/plot_binary_curve.py          # 命题5: R(Ds,∞) 曲线
python3 experiments/plot_binary_soft_decision.py   # 命题5: g(x) 软判决函数
python3 experiments/plot_binary_curve_family.py    # 不同A/σ 分离度对比
python3 experiments/plot_binary_upper_bound.py     # 命题6: 有限Da 上界
python3 experiments/plot_gaussian_scalar.py        # 命题3: 标量高斯等高线
python3 experiments/plot_gaussian_vector.py        # 命题4: 向量高斯五区域
```

### 8.3 输出文件一览

| 输出文件 | 内容 | 对应论文 |
|----------|------|----------|
| `figures/binary_rd_infinite_a1_sigma1.png` | $R(D_s, \infty)$ 曲线 | Figure 4 右 |
| `figures/binary_soft_decision_a1_sigma1.png` | $g(x)$ 软判决函数 | Figure 4 左 |
| `figures/binary_rd_infinite_curve_family.png` | 多 $A/\sigma$ 曲线族 | 拓展分析 |
| `figures/binary_upper_bound_a2sqrt10_sigma1.png` | $R(D_s, D_a)$ 上界 | Figure 5 |
| `figures/gaussian_scalar_sordf_contour.png` | 标量高斯 SORDF 等高线 | 命题3 |
| `figures/gaussian_vector_sordf_m2_contour.png` | 向量高斯五区域等高线 | Figure 2/3 |

---

## 9. 理论到代码的完整映射总结

| 论文概念 | 数学表达式 | 代码位置 | 关键工程决策 |
|----------|-----------|---------|------------|
| 贝叶斯误差 | $Q(A/\sigma)$ | `binary_case.py::gaussian_q` | 用 `erfc` 实现 $Q$ 函数 |
| 二元熵 | $h_2(p)$ | `binary_case.py::binary_entropy` | 边界处理 $p=0,1$ |
| 软判决 $g(x)$ | 公式(33) | `_compute_g` | Sigmoid + 溢出保护 |
| $\lambda$ 根搜索 | $\int(N^+-N^-)gdx = 1-2D_s$ | `_find_lambda` | 二分法 80 次迭代 |
| $R(D_s,\infty)$ 积分 | 公式(32) | `_compute_rate_integral` | 中点法 2000 点 |
| 条件方差 $\eta$ | $\text{Var}(X\mid\hat{S})$ | `compute_conditional_moments` | 数值积分 |
| Prop 6 上界 | $\min_D R(D,\infty) + \frac{1}{2}[\log\eta/D_a]^+$ | `rd_upper_bound` | 50 点网格搜索 |
| 标量 SORDF | $\frac{1}{2}\max\{\log\Sigma_X/D_a, \log\Sigma_{SX}^2/(\Sigma_X(D_s-mmse))\}$ | `gaussian_scalar_sordf` | `log_plus` 辅助 |
| 向量五区域 | 公式(20)-(24) | `gaussian_vector_sordf` | 条件链判断 |
| $\alpha$ 参数 | $\frac{m\sigma_S^2+\sigma_Z^2}{\sqrt{m\sigma_S^2}}$ | `gaussian_vector_parameters` | 闭式计算 |
| 语义带 | $\alpha^2(D_s-mmse)$ | `semantic_band` | 区域判断核心变量 |

---

## 10. 关键数值技巧

### 10.1 中点积分法

所有积分使用复合中点规则：

$$\int_a^b f(x)dx \approx \sum_{i=0}^{n-1} f\left(a + \left(i+\frac{1}{2}\right)\Delta x\right) \Delta x$$

其中 $\Delta x = (b-a)/n$，$n=2000$，截断 $[-10\sigma, 10\sigma]$。

### 10.2 指数溢出保护

```python
if exponent > 100.0:
    return 0.0     # exp(100) 极大 → sigmoid → 0
if exponent < -100.0:
    return 1.0     # exp(-100) ≈ 0 → sigmoid → 1
```

$e^{100} \approx 2.7 \times 10^{43}$，远超浮点范围，直接调用 `exp` 会溢出。

### 10.3 二分法的鲁棒性

- 初始区间通过 auto-scaling 自动调整（最多 20 次扩大/缩小）
- 80 次迭代保证精度达到 $(b-a)/2^{80} \approx 8 \times 10^{-25}$
- 对 $\lambda \to 0^-$ 和 $\lambda \to -\infty$ 的极端情况都保持稳定

---

*文档完。所有代码位于 `~/zhuwl2022/it/rd-semantic-repro/`。运行 `python3 -m pytest -q` 验证 26 项测试通过。*
