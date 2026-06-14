import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from math import isclose

# 找到项目根目录（向上退一级到 experiments 的父目录）
ROOT = Path(__file__).resolve().parents[1]  # experiments/ 的上一级
SRC_DIR = ROOT / "src"

# 把 src/ 加入 Python 搜索路径
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 现在可以导入 baseline2.py 里的函数
from baseline2 import (
    gaussian_q,
    g_soft,
    solve_lambda,
    compute_rd_infinite_soft,
    gaussian_pdf
)

# ============ 计算 η 和 γ（命题6） ============
def compute_eta_gamma(A: float, sigma: float, D: float) -> tuple:
    """计算命题6中的 γ 和 η，带溢出保护"""
    lam = solve_lambda(A, sigma, D)
    
    N_plus = lambda x: gaussian_pdf(x, A, sigma)
    N_minus = lambda x: gaussian_pdf(x, -A, sigma)
    
    limit = abs(A) + 6.0 * sigma
    
    # 计算 γ
    def integrand_gamma(x):
        g_val = g_soft(x, A, sigma, lam)
        # 如果 g_val 太小，贡献可以忽略
        if g_val < 1e-10:
            return 0.0
        return x * (N_plus(x) + N_minus(x)) * g_val
    
    gamma, _ = quad(integrand_gamma, -limit, limit, 
                    limit=200, epsabs=1e-8, epsrel=1e-8)
    
    # 计算 η
    def integrand_eta(x):
        g_val = g_soft(x, A, sigma, lam)
        if g_val < 1e-10:
            return 0.0
        return (x - gamma)**2 * (N_plus(x) + N_minus(x)) * g_val
    
    eta, _ = quad(integrand_eta, -limit, limit,
                  limit=200, epsabs=1e-8, epsrel=1e-8)
    
    # 确保 eta 为正
    eta = max(eta, 1e-10)
    
    return gamma, eta

def compute_rd_joint(A: float, sigma: float, Ds: float, Da: float, 
                     num_D_candidates: int = 20) -> float:  # 减少点数加速
    """计算命题6的可达上界"""
    q = gaussian_q(A / sigma)
    
    # 快速检查：如果两个约束都很松
    if Ds >= 0.5 and Da >= A**2 + sigma**2:
        return 0.0
    
    best_rate = float('inf')
    
    # D 的搜索范围：从 q 到 Ds，但避免太接近 q（那里 λ→-∞，数值不稳定）
    D_min = max(q + 1e-6, q)
    D_candidates = np.linspace(D_min, Ds, num_D_candidates)
    
    for D in D_candidates:
        try:
            # 阶段1：语义编码
            R_semantic = compute_rd_infinite_soft(A, sigma, D)
            
            # 计算 η(D)
            _, eta = compute_eta_gamma(A, sigma, D)
            
            # 阶段2：表象残差编码
            if Da >= eta:
                R_appearance = 0.0
            else:
                R_appearance = 0.5 * max(0.0, np.log2(eta / Da))
            
            total_rate = R_semantic + R_appearance
            
            if total_rate < best_rate:
                best_rate = total_rate
                
        except Exception as e:
            # 跳过数值失败的点
            continue
    
    return best_rate if best_rate != float('inf') else 0.0


# ============ 图5：R(D_s, D_a) 上界 ============

def plot_figure5(A: float = np.sqrt(10), sigma: float = 1.0, 
                 output_dir: Path = Path("figures")):
    """
    复现论文图5
    
    参数: A²/σ² = 10, 取 A=√10, σ=1
    """
    output_dir.mkdir(exist_ok=True)
    
    Ds_settings = [0.50, 0.30, 0.10]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
    
    # D_a 范围
    Da_vals = np.linspace(0.1, 12, 200)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 画三条 D_s 曲线
    for Ds, color in zip(Ds_settings, colors):
        R_vals = []
        for Da in Da_vals:
            R = compute_rd_joint(A, sigma, Ds, Da)
            R_vals.append(R)
        ax.plot(Da_vals, R_vals, color=color, linewidth=2, label=f"$D_s = {Ds:.2f}$")
        print(f"Done D_s={Ds}")
    
    # 基准线1：直接编码 X（忽略语义）
    # X 的方差 ≈ A² + σ² = 11
    var_X = A**2 + sigma**2
    R_baseline1 = [0.5 * max(0, np.log2(var_X / Da)) for Da in Da_vals]
    ax.plot(Da_vals, R_baseline1, 'r--', linewidth=1.5, 
            label=r"$\frac{1}{2}\log_2\left(\frac{A^2+\sigma^2}{D_a}\right)$")
    
    # 基准线2：理想情况（知道 S）
    # 残差方差 = σ² = 1
    var_residual = sigma**2
    R_baseline2 = [0.5 * max(0, np.log2(var_residual / Da)) for Da in Da_vals]
    ax.plot(Da_vals, R_baseline2, 'm--', linewidth=1.5, 
            label=r"$\frac{1}{2}\log_2\left(\frac{\sigma^2}{D_a}\right)$")
    
    ax.set_xlabel("Appearance distortion $D_a$", fontsize=12)
    ax.set_ylabel("$R(D_s, D_a)$ [bits]", fontsize=12)
    ax.set_title(rf"Achievable upper bound of $R(D_s, D_a)$, $A^2/\sigma^2 = {A**2/sigma**2:.0f}$", 
                 fontsize=12)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.1, 3.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_joint_rd.png", dpi=200)
    plt.close()
    print("Saved fig5_joint_rd.png")


# ============ 运行 ============
if __name__ == "__main__":
    plot_figure5()