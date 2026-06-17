import csv
import sys
from dataclasses import dataclass
from math import erfc, exp, isclose, log2, pi, sqrt
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, newton


# ============ 基础工具函数 ============

def gaussian_q(x: float) -> float:
    """标准正态右尾概率 Q(x) = P(Z > x)"""
    return 0.5 * erfc(x / sqrt(2.0))


def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """高斯概率密度函数"""
    z = (x - mu) / sigma
    # 限制 z 范围防止 exp 溢出
    if abs(z) > 200:
        return 0.0
    return (1.0 / (sigma * sqrt(2.0 * pi))) * exp(-0.5 * z ** 2)

def binary_entropy(p: float) -> float:
    """二元熵函数 h_2(p)"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


# ============ 核心：软判决 g(x) ============

def g_soft(x: float, A: float, sigma: float, lam: float) -> float:
    """
    论文公式 (33): 最优软判决函数
    
    g(x) = [1 + exp(λ * (1 - e^{-2Ax/σ²}) / (1 + e^{-2Ax/σ²}))]^{-1}
    
    参数:
        x: 观测值
        A: 信号幅度
        sigma: 噪声标准差
        lam: 拉格朗日乘子 λ (注意: 论文中 λ < 0)
    """
    # 避免数值溢出
    exponent = -2.0 * A * x / (sigma ** 2)
    
    # 限制 exponent 范围防止 exp 溢出
    if exponent > 300:
        tanh_term = 1.0
    elif exponent < -300:
        tanh_term = -1.0
    else:
        tanh_term = (1.0 - exp(exponent)) / (1.0 + exp(exponent))
    
    # 计算 sigmoid 参数
    z = lam * tanh_term
    
    # 限制 z 范围
    if z > 700:
        return 0.0
    if z < -700:
        return 1.0
    
    return 1.0 / (1.0 + exp(z))


# ============ 约束方程 (34) ============

def constraint_equation(lam: float, A: float, sigma: float, D: float) -> float:
    """
    论文公式 (34): 约束方程
    
    ∫ [N⁺(x) - N⁻(x)] g(x) dx = 1 - 2D
    
    返回: 左边 - 右边 = ∫[N⁺-N⁻]g dx - (1-2D)
           = 0 时 λ 正确
    
    参数:
        lam: 待求解的 λ
        A, sigma: 模型参数
        D: 目标语义失真 (注意: 这里用 D 表示阶段1的约束，D ≤ D_s)
    """
    N_plus = lambda x: gaussian_pdf(x, A, sigma)      # S=0 时的分布
    N_minus = lambda x: gaussian_pdf(x, -A, sigma)      # S=1 时的分布
    
    integrand = lambda x: (N_plus(x) - N_minus(x)) * g_soft(x, A, sigma, lam)
    
    # 数值积分，积分限根据 A 和 sigma 自适应
    # 高斯分布的 99.7% 质量在 ±3σ 内
    integration_limit = abs(A) + 5.0 * sigma
    
    result, _ = quad(integrand, -integration_limit, integration_limit, 
                     limit=100, epsabs=1e-10, epsrel=1e-10)
    
    target = 1.0 - 2.0 * D
    return result - target


# ============ 求解 λ ============

def solve_lambda(A: float, sigma: float, D: float, 
                 lam_min: float = -50.0, lam_max: float = -1e-6) -> float:
    """
    对给定 D，求解满足约束方程的 λ
    
    参数:
        A, sigma: 模型参数
        D: 目标语义失真
        lam_min, lam_max: λ 的搜索范围 (λ < 0)
    
    返回:
        满足约束的 λ 值
    """
    # 检查边界条件
    bayes_error = gaussian_q(A / sigma)
    
    if D < bayes_error:
        raise ValueError(f"D={D} < Bayes error={bayes_error}, impossible")
    
    if isclose(D, 0.5, rel_tol=1e-12, abs_tol=1e-12):
        return 0.0  # D=0.5 时 λ=0
    
    if isclose(D, bayes_error, rel_tol=1e-12, abs_tol=1e-12):
        return float('-inf')  # 极限情况
    
    # 验证函数在边界有符号变化（保证有根）
    f_min = constraint_equation(lam_min, A, sigma, D)
    f_max = constraint_equation(lam_max, A, sigma, D)
    
    # 如果符号相同，扩大搜索范围
    if f_min * f_max > 0:
        # 尝试更宽的边界
        test_lams = [-100.0, -200.0, -500.0, -1000.0]
        for test_lam in test_lams:
            f_test = constraint_equation(test_lam, A, sigma, D)
            if f_test * f_max < 0:
                lam_min = test_lam
                break
    
    # 使用 Brent 方法求根（比二分法更快）
    try:
        lam_opt = brentq(
            lambda lam: constraint_equation(lam, A, sigma, D),
            lam_min, lam_max,
            xtol=1e-12, rtol=1e-12, maxiter=100
        )
        return lam_opt
    except ValueError as e:
        # 如果 brentq 失败，尝试 Newton 方法
        try:
            lam_opt = newton(
                lambda lam: constraint_equation(lam, A, sigma, D),
                x0=-1.0,  # 初始猜测
                tol=1e-12, maxiter=100
            )
            return lam_opt
        except Exception as e2:
            raise RuntimeError(f"Failed to solve λ for D={D}: {e}, {e2}")


# ============ 计算 R(D, ∞) ============

def compute_rd_infinite_soft(A: float, sigma: float, D: float) -> float:
    """
    计算论文命题5的最优 R(D, ∞)（软判决方案）
    
    R(D, ∞) = 1 - ½ ∫ [N⁺(x) + N⁻(x)] h₂(g(x)) dx
    
    参数:
        A, sigma: 模型参数
        D: 语义失真约束
    
    返回:
        最优码率 R(D, ∞)
    """
    bayes_error = gaussian_q(A / sigma)
    
    if D < bayes_error:
        raise ValueError(f"D={D} < Bayes error={bayes_error}")
    
    if D >= 0.5:
        return 0.0
    
    if isclose(D, bayes_error, rel_tol=1e-12, abs_tol=1e-12):
        return 1.0  # 或返回一个很大的数表示无限码率
    
    # 求解 λ
    lam = solve_lambda(A, sigma, D)
    
    # 定义被积函数
    N_plus = lambda x: gaussian_pdf(x, A, sigma)
    N_minus = lambda x: gaussian_pdf(x, -A, sigma)
    
    def integrand(x: float) -> float:
        g_val = g_soft(x, A, sigma, lam)
        return (N_plus(x) + N_minus(x)) * binary_entropy(g_val)
    
    # 数值积分
    integration_limit = abs(A) + 5.0 * sigma
    
    result, err = quad(integrand, -integration_limit, integration_limit,
                       limit=100, epsabs=1e-10, epsrel=1e-10)
    
    # 计算 R = 1 - ½ * 积分
    R = 1.0 - 0.5 * result
    
    # 确保非负
    return max(0.0, R)


# ============ 采样函数（软判决版本） ============

def sample_rd_infinite_curve_soft(A: float, sigma: float, num_points: int) -> list:
    """
    采样 R(D, ∞) 曲线（软判决最优方案）
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    
    bayes_error = gaussian_q(A / sigma)
    left = bayes_error
    right = 0.5
    step = (right - left) / (num_points - 1)
    
    points = []
    for idx in range(num_points):
        D = left + idx * step
        if idx == num_points - 1:
            D = right
        
        try:
            rate = compute_rd_infinite_soft(A=A, sigma=sigma, D=D)
            points.append((D, rate))
            print(f"  D={D:.6f}, λ={solve_lambda(A, sigma, D):.4f}, R={rate:.6f}")
        except Exception as e:
            print(f"  Error at D={D}: {e}")
            points.append((D, float('nan')))
    
    return points


# ============ 保存函数 ============

def save_rd_infinite_curve_soft_csv(output_path: Path, A: float, sigma: float, 
                                    num_points: int) -> None:
    points = sample_rd_infinite_curve_soft(A=A, sigma=sigma, num_points=num_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "rate_bits"])
        writer.writerows(points)


# ============ 与朴素方案对比 ============

def compute_rd_infinite_naive_correct(A: float, sigma: float, D: float) -> float:
    q = gaussian_q(A / sigma)  # 贝叶斯最优错误率
    if D < q:
        raise ValueError("D below Bayes error")
    if D >= 0.5:
        return 0.0
    p = (D - q) / (1 - 2 * q)  # 有损压缩的翻转概率
    return 1.0 - binary_entropy(p)


# ============ 测试和验证 ============

def verify_proposition5(A: float = 1.0, sigma: float = 1.0):
    """
    验证软判决实现，与论文图4对比
    """
    print(f"\n=== Verification: A={A}, sigma={sigma} ===")
    print(f"Bayes error Q(A/σ) = {gaussian_q(A/sigma):.6f}")
    
    # 测试几个 D_s 值
    test_Ds = [0.50, 0.39, 0.27, 0.16]
    
    for D in test_Ds:
        if D < gaussian_q(A/sigma):
            continue
        print(f"\nD_s = {D}:")
        try:
            lam = solve_lambda(A, sigma, D)
            R_soft = compute_rd_infinite_soft(A, sigma, D)
            R_naive = compute_rd_infinite_naive_correct(A, sigma, D)
            print(f"  λ = {lam:.6f}")
            print(f"  R_soft = {R_soft:.6f}")
            print(f"  R_naive = {R_naive:.6f}")
            print(f"  gap = {R_naive - R_soft:.6f} (naive - soft)")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # 运行验证
    verify_proposition5(A=1.0, sigma=1.0)
    
    # 生成完整曲线数据
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    save_rd_infinite_curve_soft_csv(
        output_path=output_dir / "binary_rd_soft_a1_sigma1.csv",
        A=1.0, sigma=1.0, num_points=41
    )