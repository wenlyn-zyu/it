import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置图表字体与样式以接近学术论文风格
rcParams['font.family'] = 'serif'
rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 参数定义 (以 m=3, sigma_S^2=1, sigma_Z^2=1 为例)
# ==========================================
m = 3
sigma_S2 = 1.0
sigma_Z2 = 1.0

mmse = (sigma_S2 * sigma_Z2) / (m * sigma_S2 + sigma_Z2)      # 0.25
alpha2 = (m * sigma_S2 + sigma_Z2)**2 / (m * sigma_S2**2)      # 16/3
intersection_ds = mmse + sigma_Z2 / alpha2                     # 0.4375

# ==========================================
# 2. 状态-观测率失真函数 (SORDF) 计算函数
# ==========================================
def compute_R(Ds, Da):
    """
    根据论文 Proposition 4 矢量情况下的分段公式计算 R(Ds, Da)
    """
    Ds = np.asarray(Ds, dtype=float)
    Da = np.asarray(Da, dtype=float)
    
    R = np.zeros_like(Ds, dtype=float)
    x = Ds - mmse
    y = Da
    
    # 过滤无效区域 (失真小于 MMSE 时不可达，速率设为 np.nan)
    invalid_mask = (x <= 0) | (y < 0)
    R[invalid_mask] = np.nan
    
    # --- 区域 5 与 区域 4 (当 y 较大时) ---
    cond1 = (y >= m * sigma_S2 + m * sigma_Z2) & (~invalid_mask)
    cond5 = cond1 & (alpha2 * x >= m * sigma_S2 + sigma_Z2)
    cond4_high = cond1 & (~cond5)
    
    R[cond5] = 0.0
    val_r4_high = (m * sigma_S2 + sigma_Z2) / (alpha2 * x[cond4_high])
    R[cond4_high] = 0.5 * np.log(np.maximum(1.0, val_r4_high))
    
    # --- 区域 3, 4, 1 (当中等 y 时) ---
    cond2 = (y >= m * sigma_Z2) & (y < m * sigma_S2 + m * sigma_Z2) & (~invalid_mask)
    cond3 = cond2 & (alpha2 * x >= y - (m - 1) * sigma_Z2)
    cond4_mid = cond2 & (alpha2 * x < y / m)
    cond1_mid = cond2 & (~cond3) & (~cond4_mid)
    
    val_r3 = (m * sigma_S2 + sigma_Z2) / (y[cond3] - (m - 1) * sigma_Z2)
    R[cond3] = 0.5 * np.log(np.maximum(1.0, val_r3))
    
    # 修正：删除了之前的错误冗余行，统一使用正确掩码 cond4_mid
    val_r4_mid = (m * sigma_S2 + sigma_Z2) / (alpha2 * x[cond4_mid])
    R[cond4_mid] = 0.5 * np.log(np.maximum(1.0, val_r4_mid))
    
    val1_r1_mid = (m * sigma_S2 + sigma_Z2) / (alpha2 * x[cond1_mid])
    val2_r1_mid = ((m - 1) * sigma_Z2) / (y[cond1_mid] - alpha2 * x[cond1_mid])
    R[cond1_mid] = 0.5 * np.log(np.maximum(1.0, val1_r1_mid)) + 0.5 * (m - 1) * np.log(np.maximum(1.0, val2_r1_mid))
    
    # --- 区域 2, 1 (当小 y 时) ---
    cond3_low = (y < m * sigma_Z2) & (~invalid_mask)
    cond2_low = cond3_low & (alpha2 * x >= y / m)
    cond1_low = cond3_low & (~cond2_low)
    
    val1_r2 = (m**2 * sigma_S2 + m * sigma_Z2) / y[cond2_low]
    val2_r2 = (m * sigma_Z2) / y[cond2_low]
    R[cond2_low] = 0.5 * np.log(np.maximum(1.0, val1_r2)) + 0.5 * (m - 1) * np.log(np.maximum(1.0, val2_r2))
    
    val1_r1_low = (m * sigma_S2 + sigma_Z2) / (alpha2 * x[cond1_low])
    val2_r1_low = ((m - 1) * sigma_Z2) / (y[cond1_low] - alpha2 * x[cond1_low])
    R[cond1_low] = 0.5 * np.log(np.maximum(1.0, val1_r1_low)) + 0.5 * (m - 1) * np.log(np.maximum(1.0, val2_r1_low))
    
    return R

# ==========================================
# 3. 绘制 Figure 2 (五区域示意图)
# ==========================================
def draw_figure_2():
    plt.figure(figsize=(7, 6))
    
    # 定义绘图范围
    ds_max = 1.5
    da_max = 10.0
    
    # 绘制各分界线
    # 1. 下限倾斜线: y = m * alpha2 * (Ds - mmse)
    ds_line1 = np.linspace(mmse, intersection_ds, 100)
    da_line1 = m * alpha2 * (ds_line1 - mmse)
    plt.plot(ds_line1, da_line1, 'k-', linewidth=1.5)
    
    # 2. 上限倾斜线: y = alpha2 * (Ds - mmse) + (m-1)*sigma_Z2
    ds_line2 = np.linspace(mmse, sigma_S2, 100)
    da_line2 = alpha2 * (ds_line2 - mmse) + (m - 1) * sigma_Z2
    plt.plot(ds_line2, da_line2, 'k-', linewidth=1.5)
    
    # 3. 水平分割线1: y = m*sigma_Z2 (x >= intersection_ds)
    plt.hlines(m * sigma_Z2, intersection_ds, ds_max, colors='k', linestyles='-', linewidth=1.5)
    
    # 4. 水平分割线2: y = m*sigma_S2 + m*sigma_Z2 (x >= sigma_S2)
    plt.hlines(m * sigma_S2 + m * sigma_Z2, sigma_S2, ds_max, colors='k', linestyles='-', linewidth=1.5)
    
    # 5. 垂直分割线: x = sigma_S2 (y >= m*sigma_S2 + m*sigma_Z2)
    plt.vlines(sigma_S2, m * sigma_S2 + m * sigma_Z2, da_max, colors='k', linestyles='-', linewidth=1.5)
    
    # 标示区域名称 S1 ~ S5
    plt.text(0.32, 1.2, r'$S_1$', fontsize=13, ha='center')
    plt.text(0.65, 1.2, r'$S_2$', fontsize=13, ha='center')
    plt.text(0.90, 4.5, r'$S_3$', fontsize=13, ha='center')
    plt.text(0.50, 7.5, r'$S_4$', fontsize=13, ha='center')
    plt.text(1.20, 8.5, r'$S_5$', fontsize=13, ha='center')
    
    # 轴刻度与标签设置
    plt.xlim(0.1, ds_max)
    plt.ylim(-0.2, da_max)
    
    plt.xticks(
        [mmse, intersection_ds, sigma_S2], 
        [r'$\mathrm{mmse}$', r'$\mathrm{mmse}+\sigma_Z^2/\alpha^2$', r'$\sigma_S^2$'],
        fontsize=10
    )
    plt.yticks(
        [0, (m - 1) * sigma_Z2, m * sigma_Z2, m * sigma_S2 + m * sigma_Z2], 
        ['0', r'$(m-1)\sigma_Z^2$', r'$m\sigma_Z^2$', r'$m\sigma_S^2+m\sigma_Z^2$'],
        fontsize=10
    )
    
    plt.xlabel(r'$D_{\mathrm{s}}$', fontsize=12)
    plt.ylabel(r'$D_{\mathrm{a}}$', fontsize=12)
    plt.title(r'Fig. 2. Illustration of the five regions of $(D_{\mathrm{s}}, D_{\mathrm{a}})$-plane', fontsize=11, y=-0.18)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

# ==========================================
# 4. 绘制 Figure 3 (3D 表面与等高线图)
# ==========================================
def draw_figure_3():
    # 准备格点数据
    ds_grid = np.linspace(mmse + 0.005, 1.2, 100)
    da_grid = np.linspace(0.05, 9.0, 100)
    Ds, Da = np.meshgrid(ds_grid, da_grid)
    
    R_val = compute_R(Ds, Da)
    
    fig = plt.figure(figsize=(12, 5.5))
    
    # --- 左子图：3D 表面图 ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 为避免 3D 画图出现遮挡，将过高的极值截断展示
    R_display = np.minimum(R_val, 4.0) 
    
    surf = ax1.plot_surface(
        Ds, Da, R_display, 
        cmap='viridis', 
        edgecolor='none', 
        alpha=0.85, 
        rcount=50, ccount=50
    )
    ax1.set_xlabel(r'$D_{\mathrm{s}}$', fontsize=10)
    ax1.set_ylabel(r'$D_{\mathrm{a}}$', fontsize=10)
    ax1.set_zlabel(r'$R(D_{\mathrm{s}}, D_{\mathrm{a}})$', fontsize=10)
    ax1.set_xlim(mmse, 1.2)
    ax1.set_ylim(0, 9.0)
    ax1.set_zlim(0, 4.0)
    ax1.view_init(elev=25, azim=-135) # 调整视角以匹配论文中的3D图
    
    # --- 右子图：2D 等高线图 ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    # 绘制特定电平的等高线以匹配原图质感
    levels = [0.1, 0.3, 0.6, 1.0, 1.5, 2.2]
    contours = ax2.contour(Ds, Da, R_val, levels=levels, cmap='viridis', linewidths=1.5)
    ax2.clabel(contours, inline=True, fmt='%.1f', fontsize=8)
    
    # 绘制无失真约束的分区界线（作为辅助参考）
    ds_b1 = np.linspace(mmse, intersection_ds, 100)
    da_b1 = m * alpha2 * (ds_b1 - mmse)
    ax2.plot(ds_b1, da_b1, 'k-', linewidth=1.0, alpha=0.7)
    
    ds_b2 = np.linspace(mmse, sigma_S2, 100)
    da_b2 = alpha2 * (ds_b2 - mmse) + (m - 1) * sigma_Z2
    ax2.plot(ds_b2, da_b2, 'k-', linewidth=1.0, alpha=0.7)
    
    ax2.set_xlim(mmse, 1.2)
    ax2.set_ylim(0, 9.0)
    ax2.set_xlabel(r'$D_{\mathrm{s}}$', fontsize=11)
    ax2.set_ylabel(r'$D_{\mathrm{a}}$', fontsize=11)
    
    ax2.set_xticks([mmse, mmse + sigma_Z2/alpha2])
    ax2.set_xticklabels([r'$\mathrm{mmse}$', r'$\mathrm{mmse}+\sigma_Z^2/\alpha^2$'])
    ax2.set_yticks([0, (m-1)*sigma_Z2, m*sigma_Z2])
    ax2.set_yticklabels(['0', r'$(m-1)\sigma_Z^2$', r'$m\sigma_Z^2$'])
    
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle(r'Fig. 3. The SORDF $R(D_{\mathrm{s}}, D_{\mathrm{a}})$ (left) and its contour plot (right)', fontsize=12, y=0.05)
    plt.tight_layout()

# ==========================================
# 5. 执行绘图
# ==========================================
if __name__ == '__main__':
    draw_figure_2()
    draw_figure_3()
    plt.show()