# Rate Distortion Theory（率失真）

### $(2^{nR},n)$ 有损编码

* Encoder
$$
f_n: \mathcal{X}^n \to \{1,2,\dots,2^{nR}\}
$$
* Decoder
$$
g_n: \{1,2,\dots,2^{nR}\} \to \widehat{\mathcal{X}}^n
$$

$$
\widehat{X}^n = g_n\big(f_n(X^n)\big)
$$
* 失真 D
$$
D = \mathbb{E}\big[d(X^n,\widehat{X}^n)\big]
$$

$$
D = \sum_{x^n} p(x^n) \, d\big(x^n,\,g_n(f_n(x^n))\big)
$$


- $D=0$：无损压缩，最低码率 $R=H(X)$
- $D$ 越大（允许的误差越大），需要的编码比特越少，码率 $R$ 越小；
- 给定允许失真上限 $D$，我们把能实现的**最小码率**记作率失真函数 $R(D)$。

---

## Main Results
### 通用率失真函数基本表达式（i.i.d.）
设信源 $X$ i.i.d.，分布 $p(x)$，单样本失真 $d(x,\hat{x})$ 有界，则率失真函数等于信息率失真函数：
$$
R(D) = R^{(I)}(D) = \min_{\substack{p(\hat{x}|x):\\ \sum_{x,\hat{x}} p(x)p(\hat{x}|x)d(x,\hat{x}) \le D}} I(X;\widehat{X})
$$

固定信源分布 $p(x)$，互信息 $I(X;\hat{X})$ 关于条件分布 $p(\hat{x}|x)$ 是**凹函数（concave）**，保证最小值存在、优化可解。
---
### 伯努利信源 + 汉明失真
$X\sim \text{Bernoulli}(p)$，汉明失真 $d(x,\hat{x})=\boldsymbol{1}_{x\ne \hat{x}}$（错一位失真+1），不妨设 $p\le \tfrac12$：
$$
R(D) =
\begin{cases}
H(p) - H(D), & 0 \le D \le \min\{p,1-p\} \\
0, & D > \min\{p,1-p\}
\end{cases}
$$


### 高斯信源 + 平方误差失真（连续值有损压缩，如图像/语音）
$X\sim \mathcal{N}(0,\sigma^2)$，平方失真 $d(x,\hat{x})=(x-\hat{x})^2$：
$$
R(D) =
\begin{cases}
\frac12 \log \frac{\sigma^2}{D}, & 0 \le D \le \sigma^2 \\
0, & D > \sigma^2
\end{cases}
$$
#### 变形关系（码率-失真互推）
$$
R = \frac12\log\frac{\sigma^2}{D} \quad \Longleftrightarrow \quad D = \sigma^2 \cdot 2^{-2R}
$$
- 含义：
  1. $D\le \sigma^2$：码率随允许失真D对数下降；码率R翻倍，失真D变为原来1/4；
  2. $D>\sigma^2$：失真容忍度极高，无需传输信息，码率为0。

---
