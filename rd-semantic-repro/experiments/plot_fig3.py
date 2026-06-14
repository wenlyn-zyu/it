import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def scalar_gaussian_sordf(Sigma_S: float, Sigma_X: float, Sigma_SX: float,
                          Ds: float, Da: float) -> float:
    """
    命题3：标量高斯 SORDF 闭式解
    
    R(D_s, D_a) = 0.5 * max{ (log(Sigma_X/D_a))^+, 
                               (log(Sigma_SX^2 / (Sigma_X*(D_s-mmse))))^+ }
    
    其中 mmse = Sigma_S - Sigma_SX^2 / Sigma_X
    """
    # 计算 MMSE
    mmse = Sigma_S - (Sigma_SX ** 2) / Sigma_X
    
    # 检查可行性
    if Ds <= mmse or Da <= 0:
        return np.nan  # 不可行区域
    
    # 第一项：表象编码的率失真
    # (log(Sigma_X / D_a))^+
    if Da >= Sigma_X:
        term1 = 0.0
    else:
        term1 = np.log2(Sigma_X / Da)
    
    # 第二项：语义编码的率失真
    # (log(Sigma_SX^2 / (Sigma_X * (D_s - mmse))))^+
    ds_residual = Ds - mmse  # 剩余的语义失真预算
    threshold = (Sigma_SX ** 2) / Sigma_X  # 当 ds_residual = 这个值时，term2 = 0
    
    if ds_residual >= threshold:
        term2 = 0.0
    else:
        term2 = np.log2((Sigma_SX ** 2) / (Sigma_X * ds_residual))
    
    return 0.5 * max(term1, term2)


def plot_scalar_sordf_3d(Sigma_S: float = 1.0, 
                         Sigma_X: float = 2.0, 
                         Sigma_SX: float = 1.0,
                         output_dir: Path = Path("figures")):
    """
    标量高斯 SORDF 三维曲面
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    output_dir.mkdir(exist_ok=True)
    mmse = Sigma_S - (Sigma_SX ** 2) / Sigma_X
    
    # 关键：标量情形的范围
    # D_s 从 mmse 到 Sigma_S (或稍大)
    # D_a 从 0 到 Sigma_X (或稍大)
    Ds_min = mmse + 0.001
    Ds_max = Sigma_S * 1.1
    Da_min = 0.01
    Da_max = Sigma_X * 1.1
    
    num_Ds = 100
    num_Da = 100
    Ds_vals = np.linspace(Ds_min, Ds_max, num_Ds)
    Da_vals = np.linspace(Da_min, Da_max, num_Da)
    
    Ds_grid, Da_grid = np.meshgrid(Ds_vals, Da_vals, indexing='ij')
    R_grid = np.zeros_like(Ds_grid)
    
    for i in range(num_Ds):
        for j in range(num_Da):
            R_grid[i, j] = scalar_gaussian_sordf(Sigma_S, Sigma_X, Sigma_SX, 
                                                   Ds_vals[i], Da_vals[j])
    
    # 处理 nan
    R_grid = np.nan_to_num(R_grid, nan=0.0)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 限制Z范围避免尖峰
    R_plot = np.clip(R_grid, 0, 3)
    
    surf = ax.plot_surface(Ds_grid, Da_grid, R_plot, 
                           cmap='viridis', alpha=0.85,
                           linewidth=0, antialiased=True)
    
    # 标记分界线在曲面上
    # term1 = term2 的分界线
    # log(Sigma_X/Da) = log(Sigma_SX^2 / (Sigma_X*(Ds-mmse)))
    # Sigma_X/Da = Sigma_SX^2 / (Sigma_X*(Ds-mmse))
    # Da = Sigma_X^2 * (Ds-mmse) / Sigma_SX^2
    
    Ds_b = np.linspace(mmse + 0.001, Sigma_S, 50)
    Da_b = (Sigma_X ** 2) * (Ds_b - mmse) / (Sigma_SX ** 2)
    
    # 只画在有效范围内的部分
    valid = (Da_b >= Da_min) & (Da_b <= Da_max)
    R_b = [scalar_gaussian_sordf(Sigma_S, Sigma_X, Sigma_SX, ds, da) 
           for ds, da in zip(Ds_b[valid], Da_b[valid])]
    
    ax.plot(Ds_b[valid], Da_b[valid], np.clip(R_b, 0, 3), 
            'r-', linewidth=3, label='Boundary: term1=term2')
    
    ax.set_xlabel('$D_s$', fontsize=11)
    ax.set_ylabel('$D_a$', fontsize=11)
    ax.set_zlabel('$R(D_s, D_a)$', fontsize=11)
    ax.set_title(f'Scalar Gaussian SORDF\n$\\Sigma_S={Sigma_S}, \\Sigma_X={Sigma_X}, \\Sigma_{{SX}}={Sigma_SX}$')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    
    png_path = output_dir / "scalar_sordf_3d.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path}")
    print(f"mmse = {mmse:.4f}")


def plot_scalar_sordf_contour(Sigma_S: float = 1.0, 
                               Sigma_X: float = 2.0, 
                               Sigma_SX: float = 1.0,
                               output_dir: Path = Path("figures")):
    """
    标量高斯 SORDF 等高线图
    """
    output_dir.mkdir(exist_ok=True)
    mmse = Sigma_S - (Sigma_SX ** 2) / Sigma_X
    
    # 范围
    Ds_min = mmse + 0.001
    Ds_max = Sigma_S * 1.2
    Da_min = 0.01
    Da_max = Sigma_X * 1.2
    
    num_Ds = 200
    num_Da = 200
    Ds_vals = np.linspace(Ds_min, Ds_max, num_Ds)
    Da_vals = np.linspace(Da_min, Da_max, num_Da)
    
    Ds_grid, Da_grid = np.meshgrid(Ds_vals, Da_vals, indexing='ij')
    R_grid = np.zeros_like(Ds_grid)
    
    for i in range(num_Ds):
        for j in range(num_Da):
            R_grid[i, j] = scalar_gaussian_sordf(Sigma_S, Sigma_X, Sigma_SX, 
                                                   Ds_vals[i], Da_vals[j])
    
    R_grid = np.nan_to_num(R_grid, nan=0.0)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 画等高线
    levels = np.linspace(0, np.max(R_grid), 15)
    cs = ax.contour(Ds_grid, Da_grid, R_grid, levels=levels, cmap='viridis', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
    
    # 分界线：term1 = term2
    # Da = Sigma_X^2 * (Ds - mmse) / Sigma_SX^2
    Ds_b = np.linspace(mmse + 0.001, Sigma_S, 100)
    Da_b = (Sigma_X ** 2) * (Ds_b - mmse) / (Sigma_SX ** 2)
    
    valid = (Da_b >= Da_min) & (Da_b <= Da_max)
    ax.plot(Ds_b[valid], Da_b[valid], 'r-', linewidth=2.5, 
            label='Boundary: $R(D_s,\\infty)=R(\\infty,D_a)$')
    
    # 标记关键位置
    ax.axvline(mmse, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
               label=f'$D_s = mmse = {mmse:.3f}$')
    
    ax.axhline(Sigma_X, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, 
               label=f'$D_a = \\Sigma_X = {Sigma_X}$')
    
    # term2 = 0 的位置
    threshold = (Sigma_SX ** 2) / Sigma_X
    ax.axvline(mmse + threshold, color='green', linestyle=':', alpha=0.5, linewidth=1.5,
               label=f'$D_s = mmse + \\Sigma_{{SX}}^2/\\Sigma_X = {mmse + threshold:.3f}$')
    
    ax.set_xlabel('$D_s$', fontsize=13)
    ax.set_ylabel('$D_a$', fontsize=13)
    ax.set_title(f'Scalar Gaussian SORDF Contour\n$\\Sigma_S={Sigma_S}, \\Sigma_X={Sigma_X}, \\Sigma_{{SX}}={Sigma_SX}$', 
                 fontsize=12)
    ax.set_xlim(Ds_min, Ds_max)
    ax.set_ylim(Da_min, Da_max)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    png_path = output_dir / "scalar_sordf_contour.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path}")


if __name__ == "__main__":
    # 参数：Sigma_S=1, Sigma_X=2, Sigma_SX=1
    # 注意：Sigma_SX^2 <= Sigma_S * Sigma_X (Cauchy-Schwarz)
    # 这里 1^2 = 1 <= 1*2 = 2，满足
    
    plot_scalar_sordf_3d(Sigma_S=1.0, Sigma_X=2.0, Sigma_SX=1.0)
    plot_scalar_sordf_contour(Sigma_S=1.0, Sigma_X=2.0, Sigma_SX=1.0)