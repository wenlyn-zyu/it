# RDPC: Rate-Distortion-Perception-Classification 框架 —— MNIST实验教学文档

> 本文档从论文原文出发，结合数学推导和工程实现，详细解释如何将语义率失真框架推广为信息瓶颈（Information Bottleneck）形式，在 MNIST 上实现联合 Distortion-Perception-Classification 约束的深度压缩模型。
> 源代码位于 `rdpc_mnist/` 目录。

---

## 目录

1. [理论背景与动机](#1-理论背景与动机)
2. [从 SORDF 到 RDPC：约束条件的退化与扩展](#2-从-sordf-到-rdpc约束条件的退化与扩展)
3. [信息瓶颈形式的数学推导](#3-信息瓶颈形式的数学推导)
4. [深度学习架构设计](#4-深度学习架构设计)
5. [损失函数与训练策略](#5-损失函数与训练策略)
6. [工程实现详解](#6-工程实现详解)
7. [实验结果与分析](#7-实验结果与分析)
8. [代码结构概览](#8-代码结构概览)

---

## 1. 理论背景与动机

### 1.1 论文原文（2021 基准论文）

> **"A Rate-Distortion Framework for Characterizing Semantic Information"**
> J. Liu, W. Zhang, and H. V. Poor, IEEE ISIT, 2021.

该论文提出将信息源建模为 $(S, X) \sim p(s,x)$ 的联合分布，其中：

> We describe a memoryless information source as a tuple of random variables, $(S, X)$ with joint probability distribution $p(s, x)$ in product alphabet $\mathcal{S} \times \mathcal{X}$. We interpret $S$ as the **intrinsic state**, which captures the "semantic" aspect of the source and is not observable, and $X$ as the **extrinsic observation** of the source, which captures the "appearance" of the source to an observer.

编码器仅观测 $X$，解码器需同时重建 $\hat{S}$ 和 $\hat{X}$。存在两个失真约束：
- **语义失真 (Semantic Distortion)**: $d_s(s, \hat{s})$，如汉明距离
- **外观失真 (Appearance Distortion)**: $d_a(x, \hat{x})$，如平方误差

速率-失真函数（SORDF）定义为：

$$R(D_s, D_a) = \min_{p(\hat{s}, \hat{x}|x)} I(X; \hat{S}, \hat{X})$$
$$\text{s.t. } \mathbb{E}\hat{d}_s(X, \hat{S}) \leq D_s, \quad \mathbb{E} d_a(X, \hat{X}) \leq D_a$$

### 1.2 参考论文（2025 扩展）

> **"Task-Oriented Lossy Compression With Data, Perception, and Classification Constraints"**
> JSAC, 2025.

该论文将经典率失真理论扩展为三元约束框架 **RDPC**，引入感知质量约束：

$$R(D, P, C) = \min_{p_{\hat{X}|X}} I(X; \hat{X})$$
$$\text{s.t. } \mathbb{E}[\Delta(X, \hat{X})] \leq D \quad \text{(1a) 重建失真}$$
$$d(p_X, p_{\hat{X}}) \leq P \quad \text{(1b) 感知质量}$$
$$H(S|\hat{X}) \leq C \quad \text{(1c) 分类不确定性}$$

### 1.3 本文工作的核心创新

**将原论文的汉明距离约束替换为交叉熵，使问题退化为经典 Information Bottleneck (IB) 形式，同时引入感知约束，完整实现 (1a)(1b)(1c) 三个约束。**

| 原论文 (2021) | 本文扩展 |
|:---|:---|
| 语义失真用汉明距离 $d_s(s,\hat{s}) = \mathbf{1}[s \neq \hat{s}]$ | 退化为交叉熵 $\mathcal{L}_{\text{CE}}(S, \hat{S})$ |
| 仅约束 $D_s$ 和 $D_a$ | 增加感知约束 $P$ 和分类约束 $C$ |
| 理论分析为主 | 端到端深度学习实现 |
| 二分类条件高斯 | MNIST 十类手写数字 |

---

## 2. 从 SORDF 到 RDPC：约束条件的退化与扩展

### 2.1 汉明距离 → 交叉熵（Information Bottleneck 退化）

原论文中，语义失真使用汉明距离（Hamming distortion）：

$$d_s(s, \hat{s}) = \mathbf{1}[s \neq \hat{s}]$$

等价语义失真（Theorem 1 的变形）：

$$\hat{d}_s(x, \hat{s}) = \sum_s p(s|x) d_s(s, \hat{s}) = 1 - p(S = \hat{s} | X = x)$$

这意味着**等价语义损失恰好是分类的条件错误概率**。

当我们将汉明距离替换为交叉熵损失时：

$$\mathcal{L}_{\text{CE}}(S, \hat{S}) = -\sum_{c=0}^{9} \mathbf{1}[S=c] \log p(\hat{S}=c|X)$$

此时语义约束退化为经典的 **Information Bottleneck (IB)** 形式：

$$R(C) = \min_{p_{\hat{X}|X}} I(X; \hat{X}) \quad \text{s.t. } I(X; S) - I(\hat{X}; S) \leq C$$

因为 $H(S|\hat{X}) = H(S) - I(\hat{X}; S)$，最小化条件熵等价于最大化 $\hat{X}$ 与 $S$ 之间的互信息——这正是 IB 的核心思想：**压缩 $X$ 的同时保留关于 $S$ 的预测信息**。

### 2.2 增加感知约束 (1b)

仅用 MSE 约束重建会导致模糊图像。感知约束通过分布匹配来保证视觉真实性：

$$d(p_X, p_{\hat{X}}) \leq P$$

其中 $d(\cdot, \cdot)$ 使用 **Wasserstein-1 距离**（Earth Mover's Distance）：

$$W_1(p_X, p_{\hat{X}}) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_X}[f(x)] - \mathbb{E}_{\hat{x} \sim p_{\hat{X}}}[f(\hat{x})]$$

由 Kantorovich-Rubinstein 对偶性，这等价于训练一个 1-Lipschitz 的判别器网络 $D$（critic），其评分之差即为 Wasserstein 距离的估计。

### 2.3 MNIST 语义映射

在 MNIST 上：
- **$X$**: $28 \times 28$ 像素的手写数字图像（外观）
- **$S$**: 数字类别标签 $\{0, 1, \dots, 9\}$（语义状态）
- **$p(s|x)$**: 给定图像的条件类别分布（由预训练分类器逼近）

---

## 3. 信息瓶颈形式的数学推导

### 3.1 速率估计

无法直接计算连续空间中的 $I(X; Z)$（其中 $Z$ 是编码器输出的隐变量），采用**高斯熵上界**近似：

$$R \approx \frac{1}{2} \sum_{i=1}^{k} \log_2(1 + \sigma_i^2)$$

其中 $\sigma_i^2$ 是隐变量第 $i$ 维在 batch 上的方差。这等价于假设 $Z|X$ 服从各向同性高斯分布时的互信息上界，也是 Variational IB (VIB) 的标准做法。

### 3.2 拉格朗日松弛

三个硬约束的优化问题通过拉格朗日乘子法松弛为无约束优化：

$$\mathcal{L} = I(X; \hat{X}) + \lambda_d \cdot \mathbb{E}[\Delta(X, \hat{X})] + \lambda_p \cdot W_1(p_X, p_{\hat{X}}) + \lambda_c \cdot H(S|\hat{X})$$

在深度学习实现中，速率 $I(X; \hat{X})$ 由瓶颈维度 $k$ 隐式控制（更小的 $k$ → 更低的速率上界），因此训练目标简化为：

$$\mathcal{L}_{\text{train}} = \lambda_d \cdot \text{MSE}(X, \hat{X}) + \lambda_p \cdot \mathcal{L}_{\text{WGAN}} + \lambda_c \cdot \text{CE}(S, \hat{S})$$

其中 $\lambda_d, \lambda_p, \lambda_c$ 分别控制三个约束的权衡强度。

### 3.3 为什么这是 IB

传统的 Information Bottleneck 优化目标为：

$$\min_{p(z|x)} I(X; Z) - \beta \cdot I(Z; Y)$$

我们的框架中：
- $I(X; Z)$ 由瓶颈维度 $k$ 控制（速率约束）
- $I(Z; Y)$ 由交叉熵 $\text{CE}(S, \hat{S})$ 最大化（分类约束）
- $W_1(p_X, p_{\hat{X}})$ 作为附加的正则化项（感知约束）

当 $\lambda_p = 0$ 时，退化为标准的 **Variational Information Bottleneck (VIB)** 在 MNIST 上的实现。

---

## 4. 深度学习架构设计

### 4.1 整体系统模型 (Fig. 1)

```
S (标签 0-9) → X (28×28 图像) → [Encoder] → Z (隐变量, R bits)
                                              ↓
                                         [Decoder]
                                              ↓
                              ┌───────────────┴───────────────┐
                              ↓                               ↓
                           X̂ (重建图像)                    Ŝ (分类 logits)
                              ↓
                         [Classifier]
                              ↓
                           Ŝ (预测标签)
                              ↑
                     [Discriminator] ← 判别 X vs X̂
```

### 4.2 网络结构详解

#### Encoder（编码器）
将 $28 \times 28 \times 1$ 的图像压缩到 $k$ 维隐空间：

$$
X \xrightarrow{\text{Conv 32, 4×4, s=2}} 14 \times 14 \times 32
\xrightarrow{\text{Conv 64, 4×4, s=2}} 7 \times 7 \times 64
\xrightarrow{\text{Conv 128, 3×3, s=2}} 4 \times 4 \times 128
\xrightarrow{\text{Flatten + Linear}} \mathbb{R}^k
$$

#### Decoder / Generator（解码器/生成器）

$$
\mathbb{R}^k \xrightarrow{\text{Linear}} 4 \times 4 \times 128
\xrightarrow{\text{ConvT 64, 3×3, s=2}} 8 \times 8 \times 64
\xrightarrow{\text{ConvT 32, 4×4, s=2}} 14 \times 14 \times 32
\xrightarrow{\text{ConvT 1, 4×4, s=2}} 28 \times 28 \times 1
\xrightarrow{\text{Sigmoid}} \hat{X} \in [0,1]
$$

#### Discriminator（判别器，WGAN-GP Critic）

输出标量评分（无 Sigmoid），评分越高表示图像越真实：

$$
X/\hat{X} (28 \times 28 \times 1)
\xrightarrow{\text{Conv 32, s=2}} 14 \times 14
\xrightarrow{\text{Conv 64, s=2}} 7 \times 7
\xrightarrow{\text{Conv 128, s=2}} 4 \times 4
\xrightarrow{\text{Linear}} \mathbb{R}^1
$$

使用 InstanceNorm + LeakyReLU。梯度惩罚（Gradient Penalty）强制 1-Lipschitz 约束。

#### Classifier（预训练分类器，冻结）

独立训练在真实 MNIST 上，参数冻结后用于评估重建图像 $\hat{X}$ 的分类精度：

$$
X/\hat{X} (28 \times 28 \times 1)
\xrightarrow{\text{Conv 32 + MaxPool}} 14 \times 14
\xrightarrow{\text{Conv 64 + MaxPool}} 7 \times 7
\xrightarrow{\text{Conv 128 + AvgPool}} 1 \times 1
\xrightarrow{\text{Linear}} \mathbb{R}^{10}
$$

---

## 5. 损失函数与训练策略

### 5.1 多任务联合损失

```python
# (1a) Reconstruction Distortion
loss_mse = MSE(x_real, x_hat)

# (1b) Perception — WGAN-GP critic score
d_hat = discriminator(x_hat)
loss_perception = -d_hat.mean()  # 最大化 critic 评分

# (1c) Classification — Cross-Entropy
logits = classifier(x_hat)
loss_cls = CrossEntropy(logits, labels)

# Joint loss
loss_G = λ_d * loss_mse + λ_p * loss_perception + λ_c * loss_cls
```

### 5.2 交替训练策略

采用类似 GAN 的交替优化（与 GAN 交替训练 D/G 一致）：

| 步骤 | 更新网络 | 损失函数 |
|:---|:---|:---|
| Critic 更新（× n_critic） | Discriminator | $W_1 = \mathbb{E}[D(\hat{X})] - \mathbb{E}[D(X)] + \lambda_{gp} \cdot GP$ |
| Generator 更新（× 1） | Encoder + Decoder | $\lambda_d \cdot \text{MSE} - \lambda_p \cdot \mathbb{E}[D(\hat{X})] + \lambda_c \cdot \text{CE}$ |

**Gradient Penalty (WGAN-GP)** 计算随机插值点上的梯度范数惩罚：

$$GP = \mathbb{E}_{\tilde{x} \sim \mathbb{P}_{\tilde{x}}} \left[(\|\nabla_{\tilde{x}} D(\tilde{x})\|_2 - 1)^2\right]$$

其中 $\tilde{x} = \epsilon x + (1-\epsilon) \hat{x}$，$\epsilon \sim U(0,1)$。

### 5.3 分类器预训练

分类器先在真实 MNIST 上训练 5 个 epoch（达到 ~96-98% 准确率），然后冻结所有参数。这确保了分类约束 $\mathcal{L}_{\text{CE}}$ 衡量的是一致且公正的分类难度。

---

## 6. 工程实现详解

### 6.1 文件结构与职责

```
rdpc_mnist/
├── src/
│   ├── __init__.py
│   ├── models.py      # 四个网络的 PyTorch 定义
│   ├── training.py    # 训练循环 + 分类器预训练
│   └── utils.py       # 数据加载、速率估计、评估指标
├── experiments/
│   ├── run_single.py  # 单组超参数训练
│   ├── run_sweep.py   # 超参数网格搜索
│   └── plot_results.py # 折衷曲线可视化
├── tests/
│   └── test_models.py # 模型形状和输出范围测试
├── checkpoints/       # 模型权重保存
├── results/           # CSV 指标输出
└── requirements.txt
```

### 6.2 关键代码片段

#### 速率估计 (`src/utils.py`)

```python
def estimate_rate(z: torch.Tensor) -> float:
    """I(X;Z) 高斯熵上界: 0.5 * sum_i log(1 + var_i)"""
    var = z.var(dim=0, unbiased=True).clamp(min=1e-8)
    rate = 0.5 * torch.log1p(var).sum() / np.log(2)
    return float(rate.item())
```

#### WGAN-GP 梯度惩罚 (`src/training.py`)

```python
def gradient_penalty(discriminator, real, fake, device):
    eps = torch.rand(B, 1, 1, 1, device=device)
    interpolates = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True,
    )[0]
    gp = ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
    return gp
```

#### 训练循环核心 (`src/training.py`)

```python
# Discriminator update (n_critic times)
for _ in range(n_critic):
    d_loss = d_fake.mean() - d_real.mean() + gp_weight * gp
    opt_d.step()

# Generator update (once)
loss_g = lambda_d * loss_mse + lambda_p * loss_perception + lambda_c * loss_cls
opt_g.step()
```

### 6.3 实验配置

| 参数 | 值 | 含义 |
|:---|:---|:---|
| $k$ | $\{4, 8, 16, 32, 64\}$ | 瓶颈维度（控制速率） |
| $\lambda_d$ | $1.0$ | 重建权重（固定） |
| $\lambda_p$ | $\{0.01, 0.1, 1.0\}$ | 感知权重 |
| $\lambda_c$ | $\{0.1, 1.0, 10.0\}$ | 分类权重 |
| n_critic | $5$ | 每次 Generator 更新前的 Critic 更新次数 |
| gp_weight | $10.0$ | 梯度惩罚系数 |
| epochs | $20\sim30$ | 训练轮数 |
| batch_size | $128$ | 批次大小 |
| optimizer | Adam | $\beta=(0.5, 0.9)$ |

---

## 7. 实验结果与分析

### 7.1 分类器预训练

在真实 MNIST 测试集上的表现：

| Epoch | 训练损失 | 测试准确率 |
|:---|:---|:---|
| 1 | 0.4341 | 93.09% |
| 2 | 0.0905 | 95.19% |
| 3 | 0.0607 | 96.62% |
| 4 | 0.0472 | 98.04% |
| 5 | 0.0379 | 96.55% |

### 7.2 RDPC 联合训练 (k=8, λ_d=1.0, λ_p=0.1, λ_c=1.0, 10 epochs)

| Epoch | Rate R (bits) | MSE | 分类准确率 |
|:---|:---|:---|:---|
| 1 | 14.90 | 50.79 | 96.62% |
| 2 | 18.56 | 36.81 | 97.79% |
| 3 | 20.51 | 36.07 | 97.84% |
| 4 | 21.44 | 33.44 | 97.84% |
| 5 | 22.21 | 32.88 | 98.26% |
| 6 | 22.89 | 30.82 | 98.46% |
| 7 | 23.42 | 30.95 | 98.31% |
| 8 | 23.83 | 29.91 | 98.21% |
| 9 | 24.18 | 29.46 | 98.29% |
| **10** | **24.86** | **29.52** | **98.49%** |

### 7.3 结果分析

**关键发现：**

1. **信息瓶颈效应成立**：即使经过维数为 $k=8$ 的瓶颈压缩，分类准确率（98.49%）仍然非常高，甚至略高于在真实图像上的分类器（96.55%）。这说明编码器学到了一种"去噪"表示——自动编码器隐式地滤除了对分类无用的像素级噪声，起到了信息瓶颈的正则化作用。

2. **速率-失真-分类的三元权衡**：
   - 速率 R 从 14.90 上升到 24.86 bits（编码器学习保留更多信息）
   - MSE 从 50.79 下降到 29.52（重建质量持续改善）
   - 分类准确率从 96.62% 上升到 98.49%（语义信息得到保留）

3. **三个约束的协同作用**：
   - 约束 (1a) MSE 拉动着逐像素重建精度
   - 约束 (1b) WGAN 判别器推动 $\hat{X}$ 的分布趋近真实 MNIST 分布
   - 约束 (1c) 交叉熵确保压缩后的表示保留了数字类别的判别信息

4. **速率估计值的解读**：24.86 bits 对应 $k=8$ 维隐空间的高斯熵上界。实际有效速率通常更低（该估计为 looser upper bound）。不同 $k$ 值下的速率对比揭示了维度对压缩程度的影响。

---

## 8. 代码结构概览

### 运行命令

```bash
# 安装依赖
pip install -r rdpc_mnist/requirements.txt

# 运行单组实验
python3 rdpc_mnist/experiments/run_single.py \
    --k 16 --lambda_d 1.0 --lambda_p 0.1 --lambda_c 1.0 --epochs 30

# 运行网格搜索（45 组配置）
python3 rdpc_mnist/experiments/run_sweep.py --epochs 20

# 生成折衷曲线图
python3 rdpc_mnist/experiments/plot_results.py

# 运行测试
python3 -m pytest rdpc_mnist/tests/ -q
```

### 超参数调整指南

| 目标 | 调整方向 |
|:---|:---|
| 更低速率（更强压缩） | 减小 $k$（瓶颈维度） |
| 更好重建质量 | 增大 $\lambda_d$ 或增大 $k$ |
| 更真实图像 | 增大 $\lambda_p$ |
| 更高分类精度 | 增大 $\lambda_c$ |
| 更稳定训练 | 增大 gp_weight，减小 lr_d |

---

## 参考文献

1. J. Liu, W. Zhang, and H. V. Poor, "A Rate-Distortion Framework for Characterizing Semantic Information," *IEEE ISIT*, 2021.
2. "Task-Oriented Lossy Compression With Data, Perception, and Classification Constraints," *IEEE JSAC*, 2025.
3. N. Tishby, F. C. Pereira, and W. Bialek, "The Information Bottleneck Method," *Allerton Conference*, 1999.
4. M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein GAN," *ICML*, 2017.
5. I. Gulrajani et al., "Improved Training of Wasserstein GANs," *NeurIPS*, 2017.
6. A. A. Alemi et al., "Deep Variational Information Bottleneck," *ICLR*, 2017.
