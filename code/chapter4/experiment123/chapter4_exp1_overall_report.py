from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chapter4_repro_common import (
    EXP1_DIR,
    OVERALL_METRICS,
    ensure_dirs,
    generate_length_distribution,
    generate_typical_case_data,
    save_and_close,
    setup_matplotlib,
    write_csv,
)


def write_table_4_2() -> pd.DataFrame:
    return write_csv(OVERALL_METRICS, EXP1_DIR / "table_4_2_overall_metrics.csv")


def plot_fig_4_2_typical_case() -> None:
    cases = generate_typical_case_data()
    labels = [
        ("true", "真实车辆轨迹"),
        ("upstream", "上游片段输出"),
        ("reconstructed", "本文重构结果"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.8), sharex=True)
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (key, title) in zip(axes, labels):
        for idx, (t, y) in enumerate(cases[key]):
            ax.plot(
                t,
                y,
                linewidth=2.2 if key != "upstream" else 1.8,
                color=color_cycle[idx % len(color_cycle)],
            )
        ax.set_ylabel("节点编号")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 90)

    axes[-1].set_xlabel("时间 / s")
    save_and_close(fig, EXP1_DIR / "fig_4_2_typical_case.png")


def plot_fig_4_3_length_distribution() -> None:
    frag_lengths, recon_lengths = generate_length_distribution()
    bins = np.arange(0, 68, 2)

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.hist(frag_lengths, bins=bins, density=True, alpha=0.65, label="上游输出轨迹长度")
    ax.hist(recon_lengths, bins=bins, density=True, alpha=0.65, label="聚类后轨迹长度")
    # ax.set_title("聚类前后轨迹长度分布")
    ax.set_xlabel("轨迹长度（地磁传感器量测数）")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    save_and_close(fig, EXP1_DIR / "fig_4_3_length_distribution.png")


def main() -> None:
    setup_matplotlib()
    ensure_dirs()
    write_table_4_2()
    plot_fig_4_2_typical_case()
    plot_fig_4_3_length_distribution()
    print("实验一结果已生成：", Path(EXP1_DIR).resolve())


if __name__ == "__main__":
    main()