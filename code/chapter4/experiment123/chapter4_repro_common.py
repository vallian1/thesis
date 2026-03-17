from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------
# 路径
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "chapter4_results"
EXP1_DIR = RESULTS_DIR / "exp1"
EXP2_DIR = RESULTS_DIR / "exp2"
EXP3_DIR = RESULTS_DIR / "exp3"


# ------------------------------
# 中文绘图设置
# ------------------------------
def setup_matplotlib() -> None:
    """设置中文字体与负号显示。Windows 一般会自动命中前几个中文字体。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "NSimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


# ------------------------------
# 结果表：实验一
# ------------------------------
OVERALL_METRICS = [
    {
        "method": "仅上游片段输出",
        "P_link": 0.0,
        "R_link": 0.0,
        "F1_link": 0.0,
        "R_wm": 0.0,
        "Frag_before": 329,
        "Frag_after": 329,
        "Frag_drop": 0,
        "IDSW": 329,
        "R_rec": 0.0,
        "Avg_fragment_length": 10.152912621359222,
        "Avg_reconstructed_track_length": 1.0,
        "FMI": 0.0,
        "RI": 0.9892755061063473,
        "JC": 0.0,
        "n_segments": 412,
        "n_true_tracks": 83,
        "n_pred_clusters": 412,
    },
    {
        "method": "CEDAS风格基线",
        "P_link": 0.9395604395604396,
        "R_link": 0.7354838709677419,
        "F1_link": 0.8250904704463208,
        "R_wm": 0.06043956043956044,
        "Frag_before": 329,
        "Frag_after": 87,
        "Frag_drop": 242,
        "IDSW": 87,
        "R_rec": 0.7355623100303952,
        "Avg_fragment_length": 10.152912621359222,
        "Avg_reconstructed_track_length": 3.6144578313253013,
        "FMI": 0.7510952949858023,
        "RI": 0.9952873644674367,
        "JC": 0.5808823529411765,
        "n_segments": 412,
        "n_true_tracks": 83,
        "n_pred_clusters": 158,
    },
    {
        "method": "本文片段级fCEDAS",
        "P_link": 0.9528795811518325,
        "R_link": 0.7827956989247312,
        "F1_link": 0.859504132231405,
        "R_wm": 0.04712041884816754,
        "Frag_before": 329,
        "Frag_after": 73,
        "Frag_drop": 256,
        "IDSW": 73,
        "R_rec": 0.7781155015197568,
        "Avg_fragment_length": 10.152912621359222,
        "Avg_reconstructed_track_length": 3.8674698795180724,
        "FMI": 0.8015046714920564,
        "RI": 0.9961377648642903,
        "JC": 0.6546990496304118,
        "n_segments": 412,
        "n_true_tracks": 83,
        "n_pred_clusters": 144,
    },
]


# ------------------------------
# 结果表：实验二
# ------------------------------
TOLERANCE_BOUNDARY = [
    {"method": "仅上游片段输出", "eta_miss": 0.1, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 0},
    {"method": "仅上游片段输出", "eta_miss": 0.2, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 0},
    {"method": "仅上游片段输出", "eta_miss": 0.3, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 0},
    {"method": "仅上游片段输出", "eta_miss": 0.4, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 0},
    {"method": "仅上游片段输出", "eta_miss": 0.5, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 0},
    {"method": "仅上游片段输出", "eta_miss": 0.6, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 0},
    {"method": "本文片段级fCEDAS", "eta_miss": 0.1, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 3},
    {"method": "本文片段级fCEDAS", "eta_miss": 0.2, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 5},
    {"method": "本文片段级fCEDAS", "eta_miss": 0.3, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 6},
    {"method": "本文片段级fCEDAS", "eta_miss": 0.4, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 6},
    {"method": "本文片段级fCEDAS", "eta_miss": 0.5, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 6},
    {"method": "本文片段级fCEDAS", "eta_miss": 0.6, "max_tolerable_L_miss_under_F1>=0.8_and_Rwm<=0.1": 6},
]

MISS_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
LMISS_LEVELS = np.array([1, 2, 3, 4, 5, 6])

F1_UPSTREAM_MISS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
F1_FCEDAS_MISS = np.array([0.7917, 0.8467, 0.8967, 0.9233, 0.9217, 0.9033])

F1_UPSTREAM_LMISS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
F1_FCEDAS_LMISS = np.array([0.8967, 0.9167, 0.8967, 0.8750, 0.8567, 0.8417])

F1_HEATMAP_FULL = np.array(
    [
        [0.80, 0.84, 0.81, 0.78, 0.77, 0.75],
        [0.85, 0.91, 0.87, 0.84, 0.81, 0.80],
        [0.91, 0.95, 0.92, 0.89, 0.87, 0.84],
        [0.95, 0.95, 0.94, 0.92, 0.90, 0.88],
        [0.95, 0.94, 0.93, 0.92, 0.90, 0.89],
        [0.92, 0.91, 0.91, 0.90, 0.89, 0.89],
    ]
)


# ------------------------------
# 结果表：实验三
# ------------------------------
ABLATION_ROWS = [
    {"variant": "ConservativeFusion", "F1_link": 0.8674044690177536, "R_wm": 0.05256203208905313, "Frag_drop": 341.625, "R_rec": 0.800619241670642},
    {"variant": "Full", "F1_link": 0.8723748357067096, "R_wm": 0.04763444082048102, "Frag_drop": 343.0, "R_rec": 0.8040526243323494},
    {"variant": "OverwriteTail", "F1_link": 0.8704297322247168, "R_wm": 0.050863359367870974, "Frag_drop": 343.875, "R_rec": 0.8063483854401651},
    {"variant": "w/o MagNorm", "F1_link": 0.8598597951012807, "R_wm": 0.04642577613555932, "Frag_drop": 335.125, "R_rec": 0.7857535217340438},
    {"variant": "w/o Replay", "F1_link": 0.860665923002925, "R_wm": 0.045261709113740424, "Frag_drop": 333.375, "R_rec": 0.7817363858055757},
    {"variant": "w/o Split", "F1_link": 0.8723748357067096, "R_wm": 0.04763444082048102, "Frag_drop": 343.0, "R_rec": 0.8040526243323494},
    {"variant": "w/o d_mag", "F1_link": 0.7960320048008086, "R_wm": 0.13789681425728792, "Frag_drop": 317.0, "R_rec": 0.743644861306734},
    {"variant": "w/o s_lane", "F1_link": 0.8705145580253011, "R_wm": 0.05041385661322122, "Frag_drop": 342.0, "R_rec": 0.8017543063589289},
]

VARIANT_NAME_MAP = {
    "Full": "完整方法",
    "w/o MagNorm": "去掉磁特征归一化",
    "w/o d_mag": "去掉磁特征项",
    "OverwriteTail": "尾端直接覆盖",
    "w/o Replay": "去掉固定滞后重放",
    "ConservativeFusion": "保守融合",
    "w/o s_lane": "去掉车道相似度",
    "w/o Split": "去掉分裂规则",
}

MAIN_VARIANT_ORDER = [
    "Full",
    "w/o MagNorm",
    "w/o d_mag",
    "OverwriteTail",
    "w/o Replay",
    "ConservativeFusion",
]

FUSION_ORDER = ["OverwriteTail", "Full", "ConservativeFusion"]
MAG_NORM_ORDER = ["Full", "w/o MagNorm"]


# ------------------------------
# 通用工具
# ------------------------------
def ensure_dirs() -> None:
    for path in [RESULTS_DIR, EXP1_DIR, EXP2_DIR, EXP3_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Sequence[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def normalize_series(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    vmax = float(arr.max())
    if vmax <= 0:
        return arr
    return arr / vmax


def save_and_close(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_ablation_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(ABLATION_ROWS)
    df["variant_zh"] = df["variant"].map(VARIANT_NAME_MAP)
    return df


def get_row(df: pd.DataFrame, variant: str) -> pd.Series:
    row = df.loc[df["variant"] == variant]
    if row.empty:
        raise KeyError(f"Unknown variant: {variant}")
    return row.iloc[0]


# ------------------------------
# 图 4-2 的示例轨迹数据（确定性）
# ------------------------------
def generate_typical_case_data(seed: int = 20260312) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """生成三行示意图：真值 / 上游片段 / 本文重构结果。"""
    rng = np.random.default_rng(seed)

    def make_track(t0: float, t1: float, y0: float, slope: float, wiggle: float = 0.0):
        t = np.linspace(t0, t1, 18)
        y = y0 + slope * (t - t0) + wiggle * np.sin(np.linspace(0, np.pi, t.size))
        return t, y

    true_tracks: list[tuple[np.ndarray, np.ndarray]] = []
    upstream_segments: list[tuple[np.ndarray, np.ndarray]] = []
    reconstructed_tracks: list[tuple[np.ndarray, np.ndarray]] = []

    base_specs = [
        (0, 18, 8, 1.25, 0.15),
        (4, 22, 18, 1.15, 0.10),
        (8, 26, 30, 1.05, 0.12),
        (12, 30, 40, 1.00, 0.08),
        (16, 34, 50, 0.95, 0.10),
        (20, 38, 58, 0.90, 0.12),
        (24, 42, 65, 0.82, 0.08),
    ]

    for t0, t1, y0, slope, wiggle in base_specs:
        t, y = make_track(t0, t1, y0, slope, wiggle)
        true_tracks.append((t, y))

        # 上游输出：人为切成 2~3 段，并挖掉局部空洞
        split_points = sorted(rng.choice(np.arange(4, 14), size=2, replace=False))
        split_idx = [0, split_points[0], split_points[1], len(t)]
        for a, b in zip(split_idx[:-1], split_idx[1:]):
            seg_t = t[a:b]
            seg_y = y[a:b]
            if len(seg_t) >= 3:
                upstream_segments.append((seg_t, seg_y))

        # 重构结果：保留整体，但在少数轨迹上保留小空洞，体现“更接近真值但并非完美”
        if rng.random() < 0.35:
            cut = rng.integers(7, 11)
            reconstructed_tracks.append((t[:cut], y[:cut]))
            reconstructed_tracks.append((t[cut + 1 :], y[cut + 1 :]))
        else:
            reconstructed_tracks.append((t, y))

    return {
        "true": true_tracks,
        "upstream": upstream_segments,
        "reconstructed": reconstructed_tracks,
    }


# ------------------------------
# 图 4-3 的示例长度分布（确定性）
# ------------------------------
def generate_length_distribution(seed: int = 20260312) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # 上游片段更短
    frag_lengths = np.clip(rng.gamma(shape=2.2, scale=4.3, size=412), 1, 22)
    # 重构后更长
    recon_lengths = np.clip(rng.gamma(shape=3.4, scale=6.4, size=144), 1, 65)
    return frag_lengths, recon_lengths