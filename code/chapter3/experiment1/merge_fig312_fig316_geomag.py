
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 图 3.12 + 图 3.16 重构版
# 说明：
# 1) L 扫描中的迟到量测吸收比例与一致率，直接采用论文正文给出的数值；
# 2) 平均缓冲事件数依据原图 3.16 的端点（1.5517 -> 14.6682）和曲线形状重建；
# 3) ΔT 扫描中“跟踪准确率”“平均输出时延”在当前论文摘录里没有明确列出具体数值，
#    这里先保留为可替换数组，便于后续直接替换成你的真实统计结果。
# 4) 全部术语统一写成“地磁传感器”，不再出现“节点”。
# -------------------------------------------------------------------

def set_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

def build_data():
    # ΔT 扫描（原脚本扫描点来自 chapter3_experiments_345.py）
    delta_t_s = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3])

    # 由论文图 3.12 左图重建的“同一扫描内相邻地磁传感器同车触发比例”
    same_vehicle_ratio = np.array([0.00, 0.00, 0.065, 0.252, 0.389, 0.480])

    # 按用户截图要求补充的两条可替换曲线（论文风格示意值）
    tracking_accuracy = np.array([0.832, 0.861, 0.892, 0.914, 0.921, 0.915])
    avg_output_delay_s = np.array([0.18, 0.27, 0.37, 0.50, 0.63, 0.77])

    # L 扫描（论文正文给出的真实数值）
    lag_window_L = np.array([0, 1, 2, 3, 4, 5])
    late_absorb_ratio = np.array([0.2970, 0.3937, 0.5250, 0.6272, 0.7194, 0.7824])
    consistency = np.array([0.4620, 0.5121, 0.5878, 0.6482, 0.7064, 0.7456])

    # 依据图 3.16 形状重建，端点与正文一致
    avg_buffer_events = np.array([1.5517, 2.95, 5.15, 7.70, 11.10, 14.6682])

    return {
        "delta_t_s": delta_t_s,
        "same_vehicle_ratio": same_vehicle_ratio,
        "tracking_accuracy": tracking_accuracy,
        "avg_output_delay_s": avg_output_delay_s,
        "lag_window_L": lag_window_L,
        "late_absorb_ratio": late_absorb_ratio,
        "consistency": consistency,
        "avg_buffer_events": avg_buffer_events,
    }

def save_data_csv(data, out_csv):
    dt_df = pd.DataFrame({
        "delta_T_s": data["delta_t_s"],
        "same_vehicle_ratio": data["same_vehicle_ratio"],
        "tracking_accuracy_placeholder": data["tracking_accuracy"],
        "avg_output_delay_s_placeholder": data["avg_output_delay_s"],
    })
    dt_df["group"] = "delta_T_scan"

    l_df = pd.DataFrame({
        "L": data["lag_window_L"],
        "late_absorb_ratio": data["late_absorb_ratio"],
        "consistency": data["consistency"],
        "avg_buffer_events_reconstructed": data["avg_buffer_events"],
    })
    l_df["group"] = "L_scan"

    # 对齐列后写出
    union_cols = sorted(set(dt_df.columns).union(set(l_df.columns)))
    dt_df = dt_df.reindex(columns=union_cols)
    l_df = l_df.reindex(columns=union_cols)
    pd.concat([dt_df, l_df], ignore_index=True).to_csv(out_csv, index=False, encoding="utf-8-sig")

def plot_merged_figure(data, out_png, out_pdf=None):
    set_chinese_font()

    # fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1))
    fig, ax = plt.subplots(figsize=(6.8, 5.1))

    # # ---------------- (a) ΔT 扫描 ----------------
    # ax = axes[0]
    # ax_r = ax.twinx()

    # # l1, = ax.plot(
    # #     data["delta_t_s"],
    # #     data["same_vehicle_ratio"],
    # #     marker="o",
    # #     linewidth=2,
    # #     label="相邻地磁传感器同车触发比例",
    # # )
    # l2, = ax.plot(
    #     data["delta_t_s"],
    #     data["tracking_accuracy"],
    #     marker="s",
    #     linewidth=2,
    #     color='#2ca02c',
    #     label="跟踪准确率",
    # )
    # l3, = ax_r.plot(
    #     data["delta_t_s"],
    #     data["avg_output_delay_s"],
    #     marker="^",
    #     linewidth=2,
    #     linestyle="--",
    #     color='#ff7f0e',
    #     label="平均输出时延",
    # )

    # ax.set_title("ΔT 扫描：时间组织收益与时延")
    # ax.set_xlabel("时间片长度 ΔT / s")
    # ax.set_ylabel("比例 / 准确率")
    # ax_r.set_ylabel("平均输出时延 / s")
    # ax.set_ylim(-0.02, 1.00)
    # ax_r.set_ylim(0.0, 0.85)
    # ax.grid(alpha=0.30)

    # lines = [l2, l3]
    # ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=18)

    # ---------------- (b) L 扫描 ----------------
    # ax = axes[1]
    # ax_r = ax.twinx()

    l4, = ax.plot(
        data["lag_window_L"],
        data["late_absorb_ratio"],
        marker="o",
        linewidth=2,
        label="迟到量测吸收比例",
    )
    l5, = ax.plot(
        data["lag_window_L"],
        data["consistency"],
        marker="s",
        linewidth=2,
        label="在线与离线结果一致率",
    )
    # l6, = ax_r.plot(
    #     data["lag_window_L"],
    #     data["avg_buffer_events"],
    #     marker="^",
    #     linewidth=2,
    #     linestyle="--",
    #     label="平均缓冲事件数",
    # )

    ax.axvline(4, linestyle=":", linewidth=1.6, zorder=10)
    ax.annotate(
        "L = 4 折中点",
        xy=(4, 0.7064),
        xytext=(3.15, 0.55),
        arrowprops=dict(arrowstyle="->", linewidth=1.2, zorder=10),
        fontsize=14,
        zorder=10,
    )

    ax.set_title("L 扫描：回放收益与缓冲开销")
    ax.set_xlabel("固定滞后窗口长度 L")
    ax.set_ylabel("比例")
    # ax_r.set_ylabel("平均缓冲事件数")
    # ax_r.set_ylabel("")
    # ax_r.set_yticks([])
    ax.set_ylim(0.25, 0.82)
    # ax_r.set_ylim(0.8, 15.5)
    ax.set_xticks(data["lag_window_L"])
    ax.grid(alpha=0.30)

    # lines = [l4, l5, l6]
    lines = [l4, l5]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=14)

    # fig.suptitle("时间组织参数与固定滞后窗口的收益—开销折中", fontsize=28, y=1.02)
    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    base = 'E:/MyLunWen/xduts-main/code/chapter3/experiment1/exp1_results'
    out_png = os.path.join(base, "fig312_316_merged_geomag.png")
    out_pdf = os.path.join(base, "fig312_316_merged_geomag.pdf")
    out_csv = os.path.join(base, "fig312_316_merged_geomag_data.csv")

    data = build_data()
    save_data_csv(data, out_csv)
    plot_merged_figure(data, out_png, out_pdf)
    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
