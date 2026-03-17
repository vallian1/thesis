from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chapter4_repro_common import (
    EXP3_DIR,
    FUSION_ORDER,
    MAG_NORM_ORDER,
    MAIN_VARIANT_ORDER,
    build_ablation_dataframe,
    ensure_dirs,
    get_row,
    normalize_series,
    save_and_close,
    setup_matplotlib,
    write_csv,
)


def write_table_4_4(df: pd.DataFrame) -> None:
    rows = df[["variant", "F1_link", "R_wm", "Frag_drop", "R_rec"]].to_dict(orient="records")
    write_csv(rows, EXP3_DIR / "table_4_4_module_contribution.csv")


def plot_fig_4_7_ablation_bar(df: pd.DataFrame) -> None:
    ordered = [get_row(df, variant) for variant in MAIN_VARIANT_ORDER]
    labels = [row["variant_zh"] for row in ordered]
    f1_values = np.array([row["F1_link"] for row in ordered])
    inv_rwm = 1.0 - np.array([row["R_wm"] for row in ordered])
    frag_values = normalize_series([row["Frag_drop"] for row in ordered])

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14.2, 6.3))
    ax.bar(x - width, f1_values, width, label="片段拼接F1值")
    ax.bar(x, inv_rwm, width, label="1-错并率")
    ax.bar(x + width, frag_values, width, label="归一化碎片化降幅")
    ax.set_title("消融实验指标对比")
    ax.set_ylabel("归一化指标值")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_ylim(0.75, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_and_close(fig, EXP3_DIR / "fig_4_7_ablation_bar.png")


def plot_fig_4_8_mag_norm_compare(df: pd.DataFrame) -> None:
    ordered = [get_row(df, variant) for variant in MAG_NORM_ORDER]
    labels = [row["variant_zh"] for row in ordered]
    rwm_values = np.array([row["R_wm"] for row in ordered])
    one_minus_f1 = 1.0 - np.array([row["F1_link"] for row in ordered])

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.bar(x - width / 2, rwm_values, width, label="错并率")
    ax.bar(x + width / 2, one_minus_f1, width, label="1-F1值")
    ax.set_title("磁特征归一化前后误拼接对比")
    ax.set_ylabel("误拼接相关指标")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_and_close(fig, EXP3_DIR / "fig_4_8_mag_norm_compare.png")


def plot_fig_4_9_fusion_compare(df: pd.DataFrame) -> None:
    ordered = [get_row(df, variant) for variant in FUSION_ORDER]
    labels = [row["variant_zh"] for row in ordered]
    f1_values = np.array([row["F1_link"] for row in ordered])
    rwm_values = np.array([row["R_wm"] for row in ordered])

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.bar(x - width / 2, f1_values, width, label="片段拼接F1值")
    ax.bar(x + width / 2, rwm_values, width, label="错并率")
    ax.set_title("三种尾端融合策略对比")
    ax.set_ylabel("指标值")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_and_close(fig, EXP3_DIR / "fig_4_9_fusion_compare.png")


def main() -> None:
    setup_matplotlib()
    ensure_dirs()
    df = build_ablation_dataframe()
    write_table_4_4(df)
    plot_fig_4_7_ablation_bar(df)
    plot_fig_4_8_mag_norm_compare(df)
    plot_fig_4_9_fusion_compare(df)
    print("实验三结果已生成：", Path(EXP3_DIR).resolve())


if __name__ == "__main__":
    main()