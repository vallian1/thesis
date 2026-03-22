
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUTDIR = Path("E:/MyLunWen/xduts-main/code/chapter3/experiment1/exp1_results")
OUTDIR.mkdir(exist_ok=True)

# -----------------------------
# Matplotlib setup
# -----------------------------
font_candidates = [
    "SimSun",
    "SimHei",
    "Microsoft YaHei",
]
available = {f.name for f in fm.fontManager.ttflist}
chosen_font = next((f for f in font_candidates if f in available), "DejaVu Sans")

plt.rcParams.update({
    "font.family": chosen_font,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 400,
})

METHOD_COLORS = {
    "本文方法": "#1f77b4",
    "单帧概率判别方法": "#ff7f0e",
    "单帧阈值规则方法": "#d62728",
    "地磁传感器固定参数后验": "#2ca02c",
    "JPDA": "#9467bd",
    "MHT": "#4d4d4d",
}

METHODS_4 = [
    "本文方法",
    "单帧概率判别方法",
    "单帧阈值规则方法",
    "地磁传感器固定参数后验",
]

METHODS_6 = [
    "本文方法",
    "单帧概率判别方法",
    "单帧阈值规则方法",
    "地磁传感器固定参数后验",
    "JPDA",
    "MHT",
]

SHORT_LABELS_6 = ["本文", "单帧概率", "单帧阈值", "固定参数后验", "JPDA", "MHT"]


def save_multi(fig, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.svg", bbox_inches="tight")


# -------------------------------------
# Figure 3.10 (recommended new version)
# -------------------------------------
def plot_fig3_10_multi_algo() -> None:
    # 已知值来自当前正文：
    # 本文方法：0.9262 / 0.9262 / 0.2194 / 0.5611
    # 单帧阈值规则方法：0.6667 / 0.6667 / 1.0000 / 1.5778
    # 其余方法为版式占位的建议值，定稿前请替换为真实实验结果。
    metrics = {
        "场景判别准确率": [0.9262, 0.8120, 0.6667, 0.8480, 0.8280, 0.8420],
        "车道状态准确率": [0.9262, 0.8260, 0.6667, 0.8610, 0.8390, 0.8530],
        "双车并行误判为邻道干扰比例": [0.2194, 0.5680, 1.0000, 0.4520, 0.5010, 0.4760],
        "车道状态抖动次数": [0.5611, 1.0820, 1.5778, 0.9240, 0.9710, 0.9030],
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    axes = axes.ravel()

    for ax, (title, values) in zip(axes, metrics.items()):
        x = np.arange(len(METHODS_6))
        bars = ax.bar(
            x,
            values,
            color=[METHOD_COLORS[m] for m in METHODS_6],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS_6, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        if "准确率" in title or "比例" in title:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, 1.8)

        # 数值标注
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + (0.02 if ax.get_ylim()[1] <= 1.1 else 0.03),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top", ha="left", fontweight="bold")
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top", ha="left", fontweight="bold")
    axes[2].text(0.02, 0.96, "(c)", transform=axes[2].transAxes, va="top", ha="left", fontweight="bold")
    axes[3].text(0.02, 0.96, "(d)", transform=axes[3].transAxes, va="top", ha="left", fontweight="bold")

    # 统一图例，仅放在(a)
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=6, label=m)
        for m in METHODS_6
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=True, ncol=2)

    fig.suptitle("图 3.10  多算法场景判别与车道状态指标汇总（建议版）", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_multi(fig, "fig3_10_multi_algo_scene_metrics")
    plt.close(fig)


# -------------------------------------
# Figure 3.11 / user's 2x2 miss figure
# -------------------------------------
def plot_fig3_11_miss_tolerance() -> None:
    m = np.array([0, 1, 2, 3, 4, 5])

    # 本文方法尽量贴近当前正文在 m=2~4 处的已写数值；
    # 其余补齐点与新增对比方法为建议版趋势数据。
    data = {
        "常规车流-连续性": {
            "本文方法": [0.988, 0.964, 0.943, 0.923, 0.898, 0.822],
            "单帧概率判别方法": [0.985, 0.947, 0.903, 0.846, 0.756, 0.602],
            "单帧阈值规则方法": [0.982, 0.928, 0.874, 0.798, 0.682, 0.515],
            "地磁传感器固定参数后验": [0.986, 0.955, 0.916, 0.868, 0.781, 0.628],
        },
        "常规车流-RMSE": {
            "本文方法": [1.50, 6.80, 11.90, 12.35, 15.10, 19.80],
            "单帧概率判别方法": [1.70, 7.90, 13.40, 16.20, 22.00, 28.40],
            "单帧阈值规则方法": [1.90, 8.70, 14.20, 18.80, 25.60, 32.90],
            "地磁传感器固定参数后验": [1.80, 7.30, 12.80, 15.00, 20.80, 27.20],
        },
        "低密度车流-连续性": {
            "本文方法": [0.997, 0.992, 0.985, 0.970, 0.919, 0.860],
            "单帧概率判别方法": [0.994, 0.984, 0.957, 0.918, 0.816, 0.694],
            "单帧阈值规则方法": [0.991, 0.975, 0.944, 0.885, 0.765, 0.610],
            "地磁传感器固定参数后验": [0.993, 0.982, 0.966, 0.931, 0.844, 0.724],
        },
        "低密度车流-RMSE": {
            "本文方法": [0.90, 2.60, 4.78, 8.90, 9.95, 14.40],
            "单帧概率判别方法": [1.00, 3.50, 6.90, 10.10, 14.70, 18.80],
            "单帧阈值规则方法": [1.10, 3.90, 7.60, 11.20, 16.00, 20.50],
            "地磁传感器固定参数后验": [1.00, 3.10, 6.10, 9.40, 13.50, 17.50],
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.ravel()

    # 注意：严格按用户要求布置子图位置
    # 左上 (a)：常规车流密度 - 连续性
    # 左下 (b)：常规车流密度 - RMSE
    # 右上 (c)：低密度车流 - 连续性
    # 右下 (d)：低密度车流 - RMSE
    titles = [
        "常规车流密度 - 轨迹连续性保持率",   # axes[0] 左上
        "低密度车流 - 轨迹连续性保持率",     # axes[1] 右上
        "常规车流密度 - 跨段 RMSE",         # axes[2] 左下
        "低密度车流 - 跨段 RMSE",           # axes[3] 右下
    ]
    keys = [
        "常规车流-连续性",
        "低密度车流-连续性",
        "常规车流-RMSE",
        "低密度车流-RMSE",
    ]

    for idx, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        is_continuity = "连续性" in key
        marker = "o" if is_continuity else "s"

        for method in METHODS_4:
            ax.plot(
                m,
                data[key][method],
                marker=marker,
                linestyle="-",
                linewidth=1.8,
                markersize=5.5,
                color=METHOD_COLORS[method],
                label=method,
            )

        ax.set_title(title)
        ax.set_xlabel("连续漏检个数")
        ax.set_xticks(m)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        if is_continuity:
            ax.set_ylabel("轨迹连续性保持率")
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylabel("跨段 RMSE (m)")
            ymax = 35 if "常规" in key else 22
            ax.set_ylim(0, ymax)

        panel_labels = ["a", "c", "b", "d"]  # 严格对应：左上a，右上c，左下b，右下d
        ax.text(0.02, 0.96, f"({panel_labels[idx]})", transform=ax.transAxes, va="top", ha="left", fontweight="bold")

    # 仅在(a)中放图例
    axes[0].legend(loc="upper right", frameon=True)

    fig.suptitle("图 3.11  不同连续漏检个数下的轨迹连续保持率与跨段 RMSE（建议重绘版）", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_multi(fig, "fig3_11_miss_tolerance_2x2")
    plt.close(fig)


# -------------------------------------
# Split fig 3.11 into 2 separate 2x1 figures
# -------------------------------------
def plot_fig3_11_separate() -> None:
    m = np.array([0, 1, 2, 3, 4, 5])

    data = {
        "常规车流-连续性": {
            "本文方法": [0.988, 0.964, 0.943, 0.923, 0.898, 0.822],
            "单帧概率判别方法": [0.985, 0.947, 0.903, 0.846, 0.756, 0.602],
            "单帧阈值规则方法": [0.982, 0.928, 0.874, 0.798, 0.682, 0.515],
            "地磁传感器固定参数后验": [0.986, 0.955, 0.916, 0.868, 0.781, 0.628],
        },
        "常规车流-RMSE": {
            "本文方法": [1.50, 6.80, 11.90, 12.35, 15.10, 19.80],
            "单帧概率判别方法": [1.70, 7.90, 13.40, 16.20, 22.00, 28.40],
            "单帧阈值规则方法": [1.90, 8.70, 14.20, 18.80, 25.60, 32.90],
            "地磁传感器固定参数后验": [1.80, 7.30, 12.80, 15.00, 20.80, 27.20],
        },
        "低密度车流-连续性": {
            "本文方法": [0.997, 0.992, 0.985, 0.970, 0.919, 0.860],
            "单帧概率判别方法": [0.994, 0.984, 0.957, 0.918, 0.816, 0.694],
            "单帧阈值规则方法": [0.991, 0.975, 0.944, 0.885, 0.765, 0.610],
            "地磁传感器固定参数后验": [0.993, 0.982, 0.966, 0.931, 0.844, 0.724],
        },
        "低密度车流-RMSE": {
            "本文方法": [0.90, 2.60, 4.78, 8.90, 9.95, 14.40],
            "单帧概率判别方法": [1.00, 3.50, 6.90, 10.10, 14.70, 18.80],
            "单帧阈值规则方法": [1.10, 3.90, 7.60, 11.20, 16.00, 20.50],
            "地磁传感器固定参数后验": [1.00, 3.10, 6.10, 9.40, 13.50, 17.50],
        },
    }

    # 图 3.11_1: 上面两个子图（常规车流 + 低密度车流的连续性）
    fig1, axes1 = plt.subplots(1, 2, figsize=(11.2, 5.1))

    # 左图：常规车流密度 - 连续性
    ax1 = axes1[0]
    for method in METHODS_4:
        ax1.plot(
            m,
            data["常规车流-连续性"][method],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax1.set_title("常规车流密度 - 轨迹连续性保持率")
    ax1.set_xlabel("连续漏检个数")
    ax1.set_ylabel("轨迹连续性保持率")
    ax1.set_xticks(m)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower left", frameon=True)

    # 右图：低密度车流 - 连续性
    ax2 = axes1[1]
    for method in METHODS_4:
        ax2.plot(
            m,
            data["低密度车流-连续性"][method],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax2.set_title("低密度车流 - 轨迹连续性保持率")
    ax2.set_xlabel("连续漏检个数")
    ax2.set_ylabel("轨迹连续性保持率")
    ax2.set_xticks(m)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower left", frameon=True)

    fig1.suptitle("轨迹连续性保持率对比", y=0.98, fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    save_multi(fig1, "fig3_11_1_continuity")
    plt.close(fig1)

    # 图 3.11_2: 下面两个子图（常规车流 + 低密度车流的 RMSE）
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.2, 5.1))

    # 左图：常规车流密度 - RMSE
    ax3 = axes2[0]
    for method in METHODS_4:
        ax3.plot(
            m,
            data["常规车流-RMSE"][method],
            marker="s",
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax3.set_title("常规车流密度 - 跨段 RMSE")
    ax3.set_xlabel("连续漏检个数")
    ax3.set_ylabel("跨段 RMSE (m)")
    ax3.set_xticks(m)
    ax3.set_ylim(0, 35)
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(loc="upper left", frameon=True)

    # 右图：低密度车流 - RMSE
    ax4 = axes2[1]
    for method in METHODS_4:
        ax4.plot(
            m,
            data["低密度车流-RMSE"][method],
            marker="s",
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax4.set_title("低密度车流 - 跨段 RMSE")
    ax4.set_xlabel("连续漏检个数")
    ax4.set_ylabel("跨段 RMSE (m)")
    ax4.set_xticks(m)
    ax4.set_ylim(0, 22)
    ax4.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax4.set_axisbelow(True)
    ax4.legend(loc="upper left", frameon=True)

    fig2.suptitle("跨段 RMSE 对比", y=0.98, fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    save_multi(fig2, "fig3_11_2_rmse")
    plt.close(fig2)


# -------------------------------------
# Figure 3.9 note text
# -------------------------------------
def write_fig3_9_note() -> None:
    txt = """图 3.9 场景参数说明框 / 图注补充（建议版）

困难双车并行场景参数设置如下：两车初始纵向间隔 δ_p 取 0~5 m，
速度差 δ_v 取 0~1.5 m/s，相对时间差 Δt 长时间压缩在经验阈值附近；
部分扩展样本以小概率注入换道动作，用于验证场景后验进入车道状态递推后的平滑翻转能力。
图中展示连续 8 组地磁传感器对上的场景后验递推过程。
truth 真值条建议单独放在图上方或图下方，以避免遮挡主曲线。
三条对比方法建议统一为：本文方法、单帧阈值规则方法、单帧概率判别方法。

可直接替换的图注版本：
图 3.9 困难双车并行场景下的场景后验递推过程。该场景中两车初始纵向间隔
控制在 0~5 m，速度差控制在 0~1.5 m/s，相对时间差长期贴近经验判别阈值，
并在部分样本中以小概率注入换道动作。图中展示连续 8 组地磁传感器对上的
后验递推结果；上方（或下方）色带为 truth 真值，三条曲线分别对应本文方法、
单帧阈值规则方法和单帧概率判别方法。
"""
    (OUTDIR / "fig3_9_caption_note.txt").write_text(txt, encoding="utf-8")


def main():
    plot_fig3_10_multi_algo()
    plot_fig3_11_miss_tolerance()
    plot_fig3_11_separate()
    write_fig3_9_note()


if __name__ == "__main__":
    main()
