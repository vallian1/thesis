from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chapter4_repro_common import (
    EXP2_DIR,
    F1_FCEDAS_LMISS,
    F1_FCEDAS_MISS,
    F1_HEATMAP_FULL,
    F1_UPSTREAM_LMISS,
    F1_UPSTREAM_MISS,
    LMISS_LEVELS,
    MISS_LEVELS,
    TOLERANCE_BOUNDARY,
    ensure_dirs,
    save_and_close,
    setup_matplotlib,
    write_csv,
)


def write_table_4_3() -> pd.DataFrame:
    return write_csv(TOLERANCE_BOUNDARY, EXP2_DIR / "table_4_3_tolerance_boundary.csv")


def plot_fig_4_4_f1_vs_miss() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.plot(MISS_LEVELS, F1_UPSTREAM_MISS, marker="o", linewidth=2, label="仅上游片段输出")
    ax.plot(MISS_LEVELS, F1_FCEDAS_MISS, marker="s", linewidth=2.5, label="本文片段级fCEDAS")
    ax.set_title("片段拼接F1值随总体漏检率变化")
    ax.set_xlabel("总体漏检率")
    ax.set_ylabel("片段拼接F1值")
    ax.set_xticks(MISS_LEVELS)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    save_and_close(fig, EXP2_DIR / "fig_4_4_f1_vs_miss.png")


def plot_fig_4_5_f1_vs_lmiss() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.plot(LMISS_LEVELS, F1_UPSTREAM_LMISS, marker="o", linewidth=2, label="仅上游片段输出")
    ax.plot(LMISS_LEVELS, F1_FCEDAS_LMISS, marker="s", linewidth=2.5, label="本文片段级fCEDAS")
    ax.set_title("片段拼接F1值随连续缺测长度变化")
    ax.set_xlabel("最大连续缺测长度")
    ax.set_ylabel("片段拼接F1值")
    ax.set_xticks(LMISS_LEVELS)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    save_and_close(fig, EXP2_DIR / "fig_4_5_f1_vs_lmiss.png")


def plot_fig_4_6_heatmap_full() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    im = ax.imshow(F1_HEATMAP_FULL, aspect="auto", origin="upper", vmin=0.75, vmax=0.96)
    ax.set_title("本文方法片段拼接F1热力图")
    ax.set_xlabel("最大连续缺测长度")
    ax.set_ylabel("总体漏检率")
    ax.set_xticks(np.arange(len(LMISS_LEVELS)))
    ax.set_xticklabels(LMISS_LEVELS)
    ax.set_yticks(np.arange(len(MISS_LEVELS)))
    ax.set_yticklabels([f"{x:.1f}" for x in MISS_LEVELS])

    for i in range(F1_HEATMAP_FULL.shape[0]):
        for j in range(F1_HEATMAP_FULL.shape[1]):
            value = F1_HEATMAP_FULL[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("片段拼接F1值")
    save_and_close(fig, EXP2_DIR / "fig_4_6_heatmap_full.png")


def main() -> None:
    setup_matplotlib()
    ensure_dirs()
    write_table_4_3()
    plot_fig_4_4_f1_vs_miss()
    plot_fig_4_5_f1_vs_lmiss()
    plot_fig_4_6_heatmap_full()
    print("实验二结果已生成：", Path(EXP2_DIR).resolve())


if __name__ == "__main__":
    main()