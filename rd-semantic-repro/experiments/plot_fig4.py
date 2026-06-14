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
    compute_rd_infinite_naive_correct,
)

# ============ 复用你已有的函数 ============
# gaussian_q, gaussian_pdf, binary_entropy, g_soft, solve_lambda, 
# compute_rd_infinite_soft, compute_rd_infinite_naive_correct

# ============ 图4：左图 g(x) ============

def plot_figure4_left(A: float = 1.0, sigma: float = 1.0, output_dir: Path = Path("figures")):
    output_dir.mkdir(exist_ok=True)
    
    Ds_list = [0.50, 0.39, 0.27, 0.16]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    x_vals = np.linspace(-3, 3, 500)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for Ds, color in zip(Ds_list, colors):
        if Ds < gaussian_q(A / sigma):
            continue
        if isclose(Ds, 0.5, rel_tol=1e-12):
            # D_s=0.5 时 g(x)=0.5
            g_vals = np.full_like(x_vals, 0.5)
        else:
            lam = solve_lambda(A, sigma, Ds)
            g_vals = [g_soft(x, A, sigma, lam) for x in x_vals]
        
        ax.plot(x_vals, g_vals, color=color, linewidth=2, label=f"$D_s = {Ds:.2f}$")
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$g(x)$", fontsize=12)
    ax.set_title("Soft decision function $g(x)$ under different $D_s$", fontsize=12)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_left_gx.png", dpi=200)
    plt.close()
    print("Saved fig4_left_gx.png")


# ============ 图4：右图 R(D_s, ∞) ============

def plot_figure4_right(A: float = 1.0, sigma: float = 1.0, num_points: int = 100,
                       output_dir: Path = Path("figures")):
    output_dir.mkdir(exist_ok=True)
    
    q = gaussian_q(A / sigma)
    Ds_vals = np.linspace(q, 0.5, num_points)
    
    R_soft_vals = []
    R_naive_vals = []
    
    for Ds in Ds_vals:
        # 软判决（本文最优）
        R_soft = compute_rd_infinite_soft(A, sigma, Ds)
        R_soft_vals.append(R_soft)
        
        # 朴素方案（本地分类+压缩）
        R_naive = compute_rd_infinite_naive_correct(A, sigma, Ds)
        R_naive_vals.append(R_naive)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(Ds_vals, R_soft_vals, 'b-', linewidth=2, label=r"$R(D_s, \infty)$ (optimal soft)")
    ax.plot(Ds_vals, R_naive_vals, color='orange', linewidth=2, 
            label="Local classification + Compression")
    
    # 标记关键点
    ax.axvline(q, color='gray', linestyle=':', alpha=0.5)
    ax.text(q + 0.01, 0.9, f"$Q(A/\\sigma)={q:.4f}$", fontsize=9, rotation=90, va='top')
    
    ax.set_xlabel("Semantic distortion $D_s$", fontsize=12)
    ax.set_ylabel("$R(D_s, \\infty)$ [bits]", fontsize=12)
    ax.set_title("Rate-distortion tradeoff (A=1, $\\sigma$=1)", fontsize=12)
    ax.set_xlim(q, 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_right_rd.png", dpi=200)
    plt.close()
    print("Saved fig4_right_rd.png")


# ============ 运行 ============
if __name__ == "__main__":
    plot_figure4_left()
    plot_figure4_right()