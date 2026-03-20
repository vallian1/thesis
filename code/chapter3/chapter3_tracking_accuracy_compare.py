#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
第3章：跟踪准确度对比实验（统一 CV-KF 框架）
------------------------------------------------
输出：
1. 聚合结果 CSV：code/chapter3/tracking_accuracy_results/ch3_tracking_accuracy_aggregate.csv
2. 单次仿真结果 CSV：code/chapter3/tracking_accuracy_results/ch3_tracking_accuracy_per_seed.csv
3. 结果图：code/chapter3/tracking_accuracy_results/ch3_tracking_accuracy_rmse.png

说明：
- JPDA/KF：事件流条件下的软关联 KF 基线；
- TOMHT(KF)：事件流条件下的轨迹导向硬判决多假设近似基线；
- Proposed：本文第三章完整方法的可运行原型，融合节点状态自适应与磁特征一致性。
"""

import json
import math
import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# basic utilities
# ------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'code', 'chapter3', 'tracking_accuracy_results')
os.makedirs(OUT_DIR, exist_ok=True)

STREAM_FILES = [
    os.path.join(BASE_DIR, 'streamIn1215.txt'),
    os.path.join(BASE_DIR, 'streamIn1216.txt'),
]


# ------------------------------------------------------------
# real-data calibration
# ------------------------------------------------------------


def load_real_bank(files: List[str]) -> Dict:
    peaks = {1: [], 2: []}
    lengths = {1: [], 2: []}
    delays_ms = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                lane = int(d['lane'])
                peaks[lane].append(float(d['geomPeak']))
                lengths[lane].append(float(d['timeLength']))
                if 'inQueueTime' in d and 'date' in d:
                    delays_ms.append(max(0.0, float(d['inQueueTime']) - float(d['date'])))
    return {
        'peaks': {k: np.array(v, dtype=float) for k, v in peaks.items()},
        'lengths': {k: np.array(v, dtype=float) for k, v in lengths.items()},
        'delays_ms': np.array(delays_ms, dtype=float),
    }


REAL_BANK = load_real_bank(STREAM_FILES)


# ------------------------------------------------------------
# data structures
# ------------------------------------------------------------


@dataclass
class VehicleGT:
    vid: int
    lane: int
    start_ms: float
    speed: float
    base_peak: float
    base_tlen: float


@dataclass
class Meas:
    eid: int
    lane: int
    dev: int
    time_ms: float
    z: float
    peak: float
    tlen: float
    true_vid: Optional[int]
    is_false: bool = False


@dataclass
class TrackState:
    tid: int
    lane: int
    x: np.ndarray
    P: np.ndarray
    t_ms: float
    mean_log_peak: float
    mean_tlen: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    last_dev: int = 1
    history: List[Tuple[int, float, float, float, Optional[int], bool]] = field(default_factory=list)


# ------------------------------------------------------------
# scenarios
# ------------------------------------------------------------


SCENARIOS = {
    '场景A': dict(
        horizon_s=160,
        lam=0.070,
        follower_prob=0.32,
        parallel_pairs=28,
        false_rate=0.006,
        adj_false_prob=0.20,
        deg_blocks=[(1, 20, 22, 0.60), (2, 44, 47, 0.63)],
        jitter_ms=34,
    ),
    '场景B': dict(
        horizon_s=160,
        lam=0.082,
        follower_prob=0.40,
        parallel_pairs=40,
        false_rate=0.010,
        adj_false_prob=0.28,
        deg_blocks=[(1, 19, 23, 0.46), (2, 43, 48, 0.50)],
        jitter_ms=44,
    ),
    '场景C': dict(
        horizon_s=160,
        lam=0.090,
        follower_prob=0.44,
        parallel_pairs=46,
        false_rate=0.012,
        adj_false_prob=0.34,
        deg_blocks=[(1, 18, 24, 0.40), (2, 42, 49, 0.44)],
        jitter_ms=50,
    ),
}

SEEDS = [11, 22, 33]
N_NODES = 72
SPACING_M = 15.0


# ------------------------------------------------------------
# simulation
# ------------------------------------------------------------


def sample_base_feature(bank: Dict, lane: int, rng: np.random.Generator) -> Tuple[float, float]:
    gp = float(rng.choice(bank['peaks'][lane]))
    tl = float(rng.choice(bank['lengths'][lane]))
    return gp, tl


def simulate_scenario(cfg: Dict, seed: int, scenario_name: str) -> Dict:
    rng = np.random.default_rng(seed)
    horizon_s = int(cfg['horizon_s'])
    vehicles: List[VehicleGT] = []
    vid = 1

    def node_pd(lane: int, dev: int) -> float:
        for ln, a, b, pd in cfg['deg_blocks']:
            if lane == ln and a <= dev <= b:
                return pd
        return 0.95

    def node_pf_mult(lane: int, dev: int) -> float:
        mult = 1.0
        for ln, a, b, pd in cfg['deg_blocks']:
            if lane == ln and a <= dev <= b:
                mult = max(mult, 2.8 if pd < 0.45 else 2.2)
        return mult

    # independent arrivals + short-headway followers
    for lane in (1, 2):
        t = 0.0
        while t < horizon_s - 55.0:
            t += rng.exponential(1.0 / cfg['lam'])
            if t >= horizon_s - 55.0:
                break
            speed = float(clamp(rng.normal(24.0 if lane == 1 else 22.0, 2.5), 15.0, 33.0))
            gp, tl = sample_base_feature(REAL_BANK, lane, rng)
            vehicles.append(VehicleGT(vid, lane, t * 1000.0, speed, gp, tl))
            vid += 1
            if rng.random() < cfg['follower_prob']:
                headway = float(rng.uniform(450.0, 1100.0))
                speed2 = float(clamp(speed + rng.normal(0.0, 0.7), 15.0, 33.0))
                gp2, tl2 = sample_base_feature(REAL_BANK, lane, rng)
                vehicles.append(VehicleGT(vid, lane, t * 1000.0 + headway, speed2, gp2, tl2))
                vid += 1

    # cross-lane parallel pairs
    for _ in range(cfg['parallel_pairs']):
        base_t = float(rng.uniform(8.0, horizon_s - 55.0)) * 1000.0
        base_speed = float(clamp(rng.normal(23.0, 1.0), 16.0, 31.0))
        delta = float(rng.uniform(-120.0, 120.0))
        gp1, tl1 = sample_base_feature(REAL_BANK, 1, rng)
        gp2, tl2 = sample_base_feature(REAL_BANK, 2, rng)
        vehicles.append(VehicleGT(vid, 1, base_t, clamp(base_speed + rng.normal(0.0, 0.6), 15.0, 31.0), gp1, tl1))
        vid += 1
        vehicles.append(VehicleGT(vid, 2, base_t + delta, clamp(base_speed + rng.normal(0.0, 0.6), 15.0, 31.0), gp2, tl2))
        vid += 1

    vehicles.sort(key=lambda v: (v.start_ms, v.lane, v.vid))

    events: List[Meas] = []
    detection_trials = []
    false_bin_presence = defaultdict(int)
    eid = 1

    for v in vehicles:
        for dev in range(1, N_NODES + 1):
            x = (dev - 1) * SPACING_M
            t_true = v.start_ms + x / v.speed * 1000.0
            if t_true > horizon_s * 1000.0:
                break

            pd_true = node_pd(v.lane, dev)
            detected = 1 if rng.random() < pd_true else 0
            detection_trials.append((v.lane, dev, t_true, detected))

            # adjacent-lane weak trigger
            if rng.random() < cfg['adj_false_prob']:
                ghost_lane = 2 if v.lane == 1 else 1
                dt = float(clamp(rng.normal(120.0, 90.0), -300.0, 300.0))
                t_ghost = t_true + dt
                peak = float(clamp(v.base_peak * np.exp(rng.normal(np.log(0.32), 0.30)), 60.0, 2800.0))
                tlen = float(clamp(v.base_tlen * np.exp(rng.normal(np.log(0.62), 0.28)), 25.0, 1800.0))
                z = x + rng.normal(0.0, 1.5)
                events.append(Meas(eid, ghost_lane, dev, t_ghost, z, peak, tlen, None, True))
                eid += 1
                false_bin_presence[(ghost_lane, dev, int(max(0.0, t_ghost) // 1000))] = 1

            if detected:
                t_obs = t_true + rng.normal(0.0, cfg['jitter_ms'])
                degraded_here = any((ln == v.lane and a <= dev <= b) for ln, a, b, _ in cfg['deg_blocks'])
                z = x + rng.normal(0.0, 2.1 if degraded_here else 1.6)
                peak = float(clamp(v.base_peak * np.exp(rng.normal(0.0, 0.13)), 80.0, 6000.0))
                tlen = float(clamp(v.base_tlen * np.exp(rng.normal(0.0, 0.13)), 40.0, 2500.0))
                events.append(Meas(eid, v.lane, dev, t_obs, z, peak, tlen, v.vid, False))
                eid += 1

    # random clutter
    for lane in (1, 2):
        for dev in range(1, N_NODES + 1):
            rate = cfg['false_rate'] * node_pf_mult(lane, dev)
            n_false = rng.poisson(rate * horizon_s)
            for _ in range(n_false):
                t = float(rng.uniform(0.0, horizon_s * 1000.0))
                x = (dev - 1) * SPACING_M
                peak_med = 320.0 if lane == 1 else 170.0
                tlen_med = 210.0 if lane == 1 else 120.0
                peak = float(clamp(np.exp(rng.normal(np.log(peak_med), 0.55)), 60.0, 3600.0))
                tlen = float(clamp(np.exp(rng.normal(np.log(tlen_med), 0.40)), 25.0, 1800.0))
                z = x + rng.normal(0.0, 2.0)
                events.append(Meas(eid, lane, dev, t, z, peak, tlen, None, True))
                eid += 1
                false_bin_presence[(lane, dev, int(t // 1000))] = 1

    false_bin_trials = [
        (lane, dev, sec, false_bin_presence.get((lane, dev, sec), 0))
        for lane in (1, 2)
        for dev in range(1, N_NODES + 1)
        for sec in range(horizon_s)
    ]

    events.sort(key=lambda e: (e.time_ms, e.lane, e.dev, e.eid))
    return {
        'scenario': scenario_name,
        'events': events,
        'vehicles': vehicles,
        'detection_trials': detection_trials,
        'false_bin_trials': false_bin_trials,
        'horizon_s': horizon_s,
        'n_nodes': N_NODES,
        'spacing': SPACING_M,
    }


# ------------------------------------------------------------
# posterior estimators
# ------------------------------------------------------------


class NodePosterior:
    def __init__(self, lam=0.992, aD=9.5, bD=0.5, aF=0.5, bF=19.5, shrink=0.30):
        self.lam = lam
        self.shrink = shrink
        self.aD = defaultdict(lambda: float(aD))
        self.bD = defaultdict(lambda: float(bD))
        self.aF = defaultdict(lambda: float(aF))
        self.bF = defaultdict(lambda: float(bF))
        self.g_aD = float(aD)
        self.g_bD = float(bD)
        self.g_aF = float(aF)
        self.g_bF = float(bF)

    def update_det(self, lane, dev, outcome):
        k = (lane, dev)
        self.aD[k] = self.lam * self.aD[k] + outcome
        self.bD[k] = self.lam * self.bD[k] + (1 - outcome)
        self.g_aD = self.lam * self.g_aD + outcome
        self.g_bD = self.lam * self.g_bD + (1 - outcome)

    def update_false(self, lane, dev, outcome):
        k = (lane, dev)
        self.aF[k] = self.lam * self.aF[k] + outcome
        self.bF[k] = self.lam * self.bF[k] + (1 - outcome)
        self.g_aF = self.lam * self.g_aF + outcome
        self.g_bF = self.lam * self.g_bF + (1 - outcome)

    def mean_pd(self, lane, dev):
        a, b = self.aD[(lane, dev)], self.bD[(lane, dev)]
        node = a / (a + b)
        glob = self.g_aD / (self.g_aD + self.g_bD)
        return (1 - self.shrink) * node + self.shrink * glob

    def mean_pf(self, lane, dev):
        a, b = self.aF[(lane, dev)], self.bF[(lane, dev)]
        node = a / (a + b)
        glob = self.g_aF / (self.g_aF + self.g_bF)
        return (1 - self.shrink) * node + self.shrink * glob


class FixedPosterior:
    def mean_pd(self, lane, dev):
        return 0.90

    def mean_pf(self, lane, dev):
        return 0.05

    def update_det(self, lane, dev, outcome):
        return None

    def update_false(self, lane, dev, outcome):
        return None


def build_reliability_maps(sim: Dict, mode: str = 'node') -> Dict[int, Dict[Tuple[int, int], float]]:
    est = NodePosterior() if mode == 'node' else FixedPosterior()
    det_by_sec = defaultdict(list)
    false_by_sec = defaultdict(list)
    for lane, dev, t_ms, outcome in sim['detection_trials']:
        det_by_sec[int(t_ms // 1000)].append((lane, dev, outcome))
    for lane, dev, sec, outcome in sim['false_bin_trials']:
        false_by_sec[sec].append((lane, dev, outcome))

    out = {}
    for sec in range(sim['horizon_s'] + 1):
        for lane, dev, outcome in det_by_sec.get(sec, []):
            est.update_det(lane, dev, outcome)
        for lane, dev, outcome in false_by_sec.get(sec, []):
            est.update_false(lane, dev, outcome)
        rel_map = {}
        for lane in (1, 2):
            for dev in range(1, sim['n_nodes'] + 1):
                pd_hat = float(est.mean_pd(lane, dev))
                pf_hat = float(est.mean_pf(lane, dev))
                rel_map[(lane, dev)] = max(0.05, min(0.995, pd_hat * (1.0 - pf_hat)))
        out[sec] = rel_map
    return out


# ------------------------------------------------------------
# KF utilities
# ------------------------------------------------------------


def init_track(e: Meas, tid: int, v0: float) -> TrackState:
    x = np.array([e.z, v0], dtype=float)
    P = np.diag([9.0, 16.0])
    return TrackState(
        tid=tid,
        lane=e.lane,
        x=x,
        P=P,
        t_ms=e.time_ms,
        mean_log_peak=math.log(e.peak + 1.0),
        mean_tlen=e.tlen,
        hits=1,
        misses=0,
        confirmed=False,
        last_dev=e.dev,
        history=[(e.eid, float(x[0]), float(x[1]), float(e.time_ms), e.true_vid, False)],
    )


def predict_track(tr: TrackState, t_ms: float, qa: float) -> Tuple[np.ndarray, np.ndarray]:
    dt = max(0.0, (t_ms - tr.t_ms) / 1000.0)
    if dt <= 0:
        return tr.x.copy(), tr.P.copy()
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    Q = qa * np.array(
        [[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]], dtype=float
    ) + np.array([[0.25 * dt ** 2, 0.0], [0.0, 0.02 * dt]], dtype=float)
    x = F @ tr.x
    P = F @ tr.P @ F.T + Q
    return x, P


def kf_update(x: np.ndarray, P: np.ndarray, z: float, R: float) -> Tuple[np.ndarray, np.ndarray]:
    H = np.array([[1.0, 0.0]], dtype=float)
    S = float((H @ P @ H.T)[0, 0] + R)
    K = (P @ H.T) / S
    y = float(z - x[0])
    x_new = x + K[:, 0] * y
    P_new = (np.eye(2) - K @ H) @ P
    return x_new, P_new


def gaussian_loglik(residual: float, S: float) -> float:
    return -0.5 * (residual * residual / S + math.log(2.0 * math.pi * S))


def update_track_with_event(tr: TrackState, e: Meas, x_pred: np.ndarray, P_pred: np.ndarray, R: float) -> None:
    x_upd, P_upd = kf_update(x_pred, P_pred, e.z, R)
    tr.x = x_upd
    tr.P = P_upd
    tr.t_ms = e.time_ms
    tr.last_dev = e.dev
    tr.mean_log_peak = 0.84 * tr.mean_log_peak + 0.16 * math.log(e.peak + 1.0)
    tr.mean_tlen = 0.84 * tr.mean_tlen + 0.16 * e.tlen
    tr.hits += 1
    tr.misses = 0
    tr.confirmed = tr.confirmed or tr.hits >= 3
    tr.history.append((e.eid, float(tr.x[0]), float(tr.x[1]), float(e.time_ms), e.true_vid, tr.confirmed))


# ------------------------------------------------------------
# event likelihoods
# ------------------------------------------------------------


def event_score(tr: TrackState, e: Meas, rel: float, mode: str):
    if e.dev <= tr.last_dev:
        return None
    dd = e.dev - tr.last_dev

    if mode == 'proposed':
        max_skip = 4 + (1 if rel < 0.45 else 0)
        if dd > max_skip + 1:
            return None
        x_pred, P_pred = predict_track(tr, e.time_ms, qa=3.6)
        R = 3.2 / (0.45 + 0.55 * rel)
        S = float(P_pred[0, 0] + R)
        residual = float(e.z - x_pred[0])
        gate = 4.5 + 2.1 * (dd - 1) + 2.2 * (1.0 - rel)
        if abs(residual) > gate:
            return None
        peak_diff = abs(math.log(e.peak + 1.0) - tr.mean_log_peak)
        tlen_diff = abs(e.tlen - tr.mean_tlen) / max(50.0, tr.mean_tlen)
        score = (
            gaussian_loglik(residual, S)
            - 0.90 * peak_diff
            - 0.60 * tlen_diff
            + math.log(0.45 + 0.55 * rel)
            - 0.10 * (dd - 1)
        )
        return score, x_pred, P_pred, R

    if mode == 'jpda':
        if dd > 4:
            return None
        x_pred, P_pred = predict_track(tr, e.time_ms, qa=3.0)
        R = 4.5
        S = float(P_pred[0, 0] + R)
        residual = float(e.z - x_pred[0])
        gate = 5.5 + 2.0 * (dd - 1)
        if abs(residual) > gate:
            return None
        score = gaussian_loglik(residual, S) - 0.12 * (dd - 1)
        return score, x_pred, P_pred, R

    if mode == 'tomht':
        if dd > 4:
            return None
        x_pred, P_pred = predict_track(tr, e.time_ms, qa=3.1)
        R = 4.4
        S = float(P_pred[0, 0] + R)
        residual = float(e.z - x_pred[0])
        gate = 5.4 + 2.0 * (dd - 1)
        if abs(residual) > gate:
            return None
        score = gaussian_loglik(residual, S) - 0.10 * (dd - 1)
        return score, x_pred, P_pred, R

    raise ValueError(mode)


# ------------------------------------------------------------
# trackers
# ------------------------------------------------------------


def retire_stale(active_by_lane: Dict[int, List[TrackState]], current_t: float, base_gap_ms: float) -> None:
    for lane in (1, 2):
        kept = []
        for tr in active_by_lane[lane]:
            if current_t - tr.t_ms <= base_gap_ms:
                kept.append(tr)
        active_by_lane[lane] = kept



def run_proposed(sim: Dict) -> List[TrackState]:
    rel_maps = build_reliability_maps(sim, mode='node')
    active_by_lane = {1: [], 2: []}
    finished: List[TrackState] = []
    next_tid = 1

    for e in sim['events']:
        rel = rel_maps[min(int(e.time_ms // 1000), sim['horizon_s'])][(e.lane, e.dev)]
        retire_gap = 3600.0 + (900.0 if rel < 0.5 else 0.0)

        # retire stale tracks lane-wise
        for lane in (1, 2):
            kept = []
            for tr in active_by_lane[lane]:
                if e.time_ms - tr.t_ms <= retire_gap:
                    kept.append(tr)
                else:
                    finished.append(tr)
            active_by_lane[lane] = kept

        cands = []
        for tr in active_by_lane[e.lane]:
            out = event_score(tr, e, rel=rel, mode='proposed')
            if out is not None:
                score, x_pred, P_pred, R = out
                cands.append((score, tr, x_pred, P_pred, R))

        if cands:
            cands.sort(key=lambda x: x[0], reverse=True)
            best_score, best_tr, x_pred, P_pred, R = cands[0]
            thresh = -3.7 if rel > 0.45 else -4.2
            if best_score > thresh:
                update_track_with_event(best_tr, e, x_pred, P_pred, R)
            else:
                v0 = 23.5 if e.lane == 1 else 22.0
                active_by_lane[e.lane].append(init_track(e, next_tid, v0))
                next_tid += 1
        else:
            v0 = 23.5 if e.lane == 1 else 22.0
            active_by_lane[e.lane].append(init_track(e, next_tid, v0))
            next_tid += 1

    for lane in (1, 2):
        finished.extend(active_by_lane[lane])
    return finished



def run_jpda(sim: Dict) -> List[TrackState]:
    active_by_lane = {1: [], 2: []}
    finished: List[TrackState] = []
    next_tid = 1
    clutter_c = 0.03

    for e in sim['events']:
        for lane in (1, 2):
            kept = []
            for tr in active_by_lane[lane]:
                if e.time_ms - tr.t_ms <= 3600.0:
                    kept.append(tr)
                else:
                    finished.append(tr)
            active_by_lane[lane] = kept

        cands = []
        for tr in active_by_lane[e.lane]:
            out = event_score(tr, e, rel=0.9, mode='jpda')
            if out is not None:
                score, x_pred, P_pred, R = out
                cands.append((score, tr, x_pred, P_pred, R))

        if not cands:
            v0 = 23.0 if e.lane == 1 else 22.0
            active_by_lane[e.lane].append(init_track(e, next_tid, v0))
            next_tid += 1
            continue

        loglikes = np.array([c[0] for c in cands], dtype=float)
        likes = np.exp(loglikes - np.max(loglikes))
        betas = likes / (clutter_c + likes.sum())
        best_idx = int(np.argmax(betas))
        best_beta = float(betas[best_idx])

        for beta, cand in zip(betas, cands):
            score, tr, x_pred, P_pred, R = cand
            x_kf, P_kf = kf_update(x_pred, P_pred, e.z, R)
            beta = float(beta)
            tr.x = x_pred + beta * (x_kf - x_pred)
            tr.P = P_pred - beta * (P_pred - P_kf) + np.diag([0.18 * (1.0 - beta), 0.05 * (1.0 - beta)])
            tr.t_ms = e.time_ms
            if beta > 0.05:
                tr.last_dev = max(tr.last_dev, e.dev)
            if beta > 0.12:
                tr.hits += 1
                tr.misses = 0
                tr.confirmed = tr.confirmed or tr.hits >= 3
            else:
                tr.misses += 1

        if best_beta > 0.20:
            best_tr = cands[best_idx][1]
            best_tr.history.append((e.eid, float(best_tr.x[0]), float(best_tr.x[1]), float(e.time_ms), e.true_vid, best_tr.confirmed))
        else:
            v0 = 23.0 if e.lane == 1 else 22.0
            active_by_lane[e.lane].append(init_track(e, next_tid, v0))
            next_tid += 1

        active_by_lane[e.lane] = [tr for tr in active_by_lane[e.lane] if not (tr.hits == 1 and tr.misses > 3)]

    for lane in (1, 2):
        finished.extend(active_by_lane[lane])
    return finished



def run_tomht(sim: Dict) -> List[TrackState]:
    """
    事件流条件下的 TOMHT(KF) 近似：
    使用轨迹导向的硬判决与延迟确认思想，但不引入节点自适应与磁特征项。
    """
    active_by_lane = {1: [], 2: []}
    finished: List[TrackState] = []
    next_tid = 1

    for e in sim['events']:
        for lane in (1, 2):
            kept = []
            for tr in active_by_lane[lane]:
                if e.time_ms - tr.t_ms <= 3800.0:
                    kept.append(tr)
                else:
                    finished.append(tr)
            active_by_lane[lane] = kept

        cands = []
        for tr in active_by_lane[e.lane]:
            out = event_score(tr, e, rel=0.9, mode='tomht')
            if out is not None:
                score, x_pred, P_pred, R = out
                cands.append((score, tr, x_pred, P_pred, R))

        if cands:
            cands.sort(key=lambda x: x[0], reverse=True)
            best_score, best_tr, x_pred, P_pred, R = cands[0]
            if best_score > -4.0:
                update_track_with_event(best_tr, e, x_pred, P_pred, R)
            else:
                v0 = 23.0 if e.lane == 1 else 22.0
                active_by_lane[e.lane].append(init_track(e, next_tid, v0))
                next_tid += 1
        else:
            v0 = 23.0 if e.lane == 1 else 22.0
            active_by_lane[e.lane].append(init_track(e, next_tid, v0))
            next_tid += 1

    for lane in (1, 2):
        finished.extend(active_by_lane[lane])
    return finished


# ------------------------------------------------------------
# evaluation
# ------------------------------------------------------------


def eval_rmse(sim: Dict, tracks: List[TrackState]) -> Dict:
    veh_map = {v.vid: v for v in sim['vehicles']}
    rows = []
    for tr in tracks:
        if tr.hits < 3:
            continue
        for eid, est_p, est_v, t_ms, true_vid, confirmed in tr.history:
            if true_vid is None:
                continue
            v = veh_map[true_vid]
            true_p = v.speed * ((t_ms - v.start_ms) / 1000.0)
            true_v = v.speed
            rows.append({
                'eid': eid,
                'track_id': tr.tid,
                'true_vid': true_vid,
                't_ms': t_ms,
                'est_p': est_p,
                'est_v': est_v,
                'true_p': true_p,
                'true_v': true_v,
                'err_p': est_p - true_p,
                'err_v': est_v - true_v,
            })

    if not rows:
        return {'RMSE_p': np.nan, 'RMSE_v': np.nan, 'match_rate': 0.0, 'n_samples': 0}

    df = pd.DataFrame(rows).sort_values('eid').drop_duplicates('eid', keep='last')
    n_true = sum(1 for e in sim['events'] if e.true_vid is not None)
    rmse_p = float(np.sqrt(np.mean(np.square(df['err_p'].values))))
    rmse_v = float(np.sqrt(np.mean(np.square(df['err_v'].values))))
    match_rate = float(len(df) / max(n_true, 1))
    return {'RMSE_p': rmse_p, 'RMSE_v': rmse_v, 'match_rate': match_rate, 'n_samples': int(len(df))}


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------


def plot_rmse(agg_df: pd.DataFrame, out_path: str) -> None:
    methods = ['JPDA/KF', 'TOMHT(KF)', 'Proposed']
    scenarios = list(agg_df['scenario'].drop_duplicates())
    xlabels = {'场景A': 'A', '场景B': 'B', '场景C': 'C'}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=180)
    width = 0.24
    x = np.arange(len(scenarios))

    for idx, method in enumerate(methods):
        sub = agg_df[agg_df['method'] == method].set_index('scenario').loc[scenarios]
        axes[0].bar(x + (idx - 1) * width, sub['RMSE_p_mean'].values, width=width, label=method)
        axes[1].bar(x + (idx - 1) * width, sub['RMSE_v_mean'].values, width=width, label=method)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([xlabels.get(s, s) for s in scenarios])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([xlabels.get(s, s) for s in scenarios])
    axes[0].set_ylabel('Position RMSE / m')
    axes[1].set_ylabel('Velocity RMSE / (m/s)')
    axes[0].set_title('(a) Position RMSE')
    axes[1].set_title('(b) Velocity RMSE')
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)
    axes[1].legend(frameon=False, ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------


def run_subset(selected_scenarios=None) -> pd.DataFrame:
    per_seed_rows = []
    scenario_items = SCENARIOS.items() if selected_scenarios is None else [(k, SCENARIOS[k]) for k in selected_scenarios]
    for scenario_name, cfg in scenario_items:
        for seed in SEEDS:
            sim = simulate_scenario(cfg, seed=seed, scenario_name=scenario_name)
            metrics_prop = eval_rmse(sim, run_proposed(sim))
            metrics_jpda = eval_rmse(sim, run_jpda(sim))
            metrics_tomht = eval_rmse(sim, run_tomht(sim))

            for method, m in [
                ('JPDA/KF', metrics_jpda),
                ('TOMHT(KF)', metrics_tomht),
                ('Proposed', metrics_prop),
            ]:
                per_seed_rows.append({
                    'scenario': scenario_name,
                    'seed': seed,
                    'method': method,
                    'RMSE_p': m['RMSE_p'],
                    'RMSE_v': m['RMSE_v'],
                    'match_rate': m['match_rate'],
                    'n_samples': m['n_samples'],
                })
                print(
                    f"[{scenario_name}][seed={seed:02d}][{method}] "
                    f"RMSE_p={m['RMSE_p']:.4f}, RMSE_v={m['RMSE_v']:.4f}, match={m['match_rate']:.4f}"
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(OUT_DIR, 'ch3_tracking_accuracy_per_seed.csv')
    per_seed_df.to_csv(per_seed_path, index=False, encoding='utf-8-sig')

    agg_df = (
        per_seed_df.groupby(['scenario', 'method'])
        .agg(
            RMSE_p_mean=('RMSE_p', 'mean'),
            RMSE_p_std=('RMSE_p', 'std'),
            RMSE_v_mean=('RMSE_v', 'mean'),
            RMSE_v_std=('RMSE_v', 'std'),
            match_rate_mean=('match_rate', 'mean'),
            n_samples_mean=('n_samples', 'mean'),
        )
        .reset_index()
    )
    agg_path = os.path.join(OUT_DIR, 'ch3_tracking_accuracy_aggregate.csv')
    agg_df.to_csv(agg_path, index=False, encoding='utf-8-sig')

    fig_path = os.path.join(OUT_DIR, 'ch3_tracking_accuracy_rmse.png')
    plot_rmse(agg_df, fig_path)

    md_lines = []
    md_lines.append('# Chapter 3 Tracking Accuracy Comparison')
    md_lines.append('')
    for scenario_name in SCENARIOS.keys():
        sub = agg_df[agg_df['scenario'] == scenario_name]
        md_lines.append(f'## {scenario_name}')
        md_lines.append(sub[['method', 'RMSE_p_mean', 'RMSE_v_mean', 'match_rate_mean']].to_markdown(index=False))
        md_lines.append('')
    md_path = os.path.join(OUT_DIR, 'ch3_tracking_accuracy_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print('\nSaved:')
    print(per_seed_path)
    print(agg_path)
    print(fig_path)
    print(md_path)
    return per_seed_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', nargs='*', default=None)
    args = parser.parse_args()
    run_subset(args.scenario)


if __name__ == '__main__':
    main()
