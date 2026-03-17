#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# real-data calibration
# ------------------------------------------------------------

def load_real_bank(files):
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
        'n_nodes': 72,
        'spacing_m': 15.0,
    }


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------
# experiment 3: scene discrimination
# ------------------------------------------------------------

@dataclass
class SceneStep:
    idx: int
    true_state: str
    lane1_time: Optional[float]
    lane2_time: Optional[float]
    lane1_peak: Optional[float]
    lane2_peak: Optional[float]
    lane1_real: int
    lane2_real: int


def statistic_fitting(alpha, state_idx):
    # from the provided Java logic
    if state_idx == 1:  # I1
        if alpha <= 1:
            return 0.10
        if alpha <= 3:
            return 0.20
        if alpha <= 5:
            return 0.26
        if alpha <= 7:
            return 0.10
        if alpha <= 9:
            return 0.20
        return 0.14
    if state_idx == 2:  # I2
        if alpha <= 0.2:
            return 0.36
        if alpha <= 0.4:
            return 0.32
        if alpha <= 0.6:
            return 0.20
        if alpha <= 0.8:
            return 0.08
        return 0.04
    if state_idx == 3:  # P
        return 0.05
    raise ValueError(state_idx)


def normalize(x):
    arr = np.asarray(x, dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def simulate_scene_sequence(scene_type, seed=0, n_nodes=18):
    rng = np.random.default_rng(seed)
    lane_real = 1 if rng.random() < 0.5 else 2
    base_speed = float(clamp(rng.normal(22.0, 1.5), 15.0, 30.0))
    spacing = 15.0
    t0 = rng.uniform(0.0, 2000.0)
    seq = []

    if scene_type == 'hard_parallel':
        dt0 = float(clamp(rng.normal(120.0, 45.0), -220.0, 220.0))
        drift = rng.normal(0.0, 4.0)
        # use normalized strengths after node normalization
        n1 = float(np.exp(rng.normal(0.02, 0.10)))
        n2 = float(np.exp(rng.normal(-0.02, 0.10)))
        if rng.random() < 0.5:
            n1 *= float(np.exp(rng.normal(0.25, 0.08)))
        else:
            n2 *= float(np.exp(rng.normal(0.25, 0.08)))
        base_amp = 600.0
    else:
        p_real = 800.0 if lane_real == 1 else 600.0

    for i in range(n_nodes):
        t_main = t0 + i * spacing / base_speed * 1000.0 + rng.normal(0.0, 25.0)
        if scene_type == 'single_normal':
            if lane_real == 1:
                lane1_time = t_main
                lane1_peak = float(clamp(p_real * np.exp(rng.normal(0.0, 0.10)), 80.0, 5000.0))
                lane2_time, lane2_peak = None, None
                true_state = 'I1'
                lane1_real, lane2_real = 1, 0
            else:
                lane2_time = t_main
                lane2_peak = float(clamp(p_real * np.exp(rng.normal(0.0, 0.10)), 80.0, 5000.0))
                lane1_time, lane1_peak = None, None
                true_state = 'I2'
                lane1_real, lane2_real = 0, 1
        elif scene_type == 'single_adjacent':
            if lane_real == 1:
                lane1_time = t_main
                lane1_peak = float(clamp(p_real * np.exp(rng.normal(0.0, 0.10)), 80.0, 5000.0))
                if rng.random() < 0.88:
                    dt = float(clamp(rng.normal(120.0, 70.0), -260.0, 260.0))
                    lane2_time = t_main + dt
                    lane2_peak = float(clamp(lane1_peak * np.exp(rng.normal(np.log(0.40), 0.22)), 80.0, 3500.0))
                else:
                    lane2_time, lane2_peak = None, None
                true_state = 'I1'
                lane1_real, lane2_real = 1, 0
            else:
                lane2_time = t_main
                lane2_peak = float(clamp(p_real * np.exp(rng.normal(0.0, 0.10)), 80.0, 5000.0))
                if rng.random() < 0.88:
                    dt = float(clamp(rng.normal(120.0, 70.0), -260.0, 260.0))
                    lane1_time = t_main + dt
                    lane1_peak = float(clamp(lane2_peak * np.exp(rng.normal(np.log(0.40), 0.22)), 80.0, 3500.0))
                else:
                    lane1_time, lane1_peak = None, None
                true_state = 'I2'
                lane1_real, lane2_real = 0, 1
        elif scene_type == 'hard_parallel':
            dt = dt0 + i * drift + rng.normal(0.0, 18.0)
            lane1_time = t_main
            lane2_time = t_main + dt
            lane1_peak = float(clamp(base_amp * n1 * np.exp(rng.normal(0.0, 0.10)), 80.0, 5000.0))
            lane2_peak = float(clamp(base_amp * n2 * np.exp(rng.normal(0.0, 0.10)), 80.0, 5000.0))
            if rng.random() < 0.04:
                if rng.random() < 0.5:
                    lane1_time, lane1_peak = None, None
                else:
                    lane2_time, lane2_peak = None, None
            true_state = 'P'
            lane1_real = 1 if lane1_time is not None else 0
            lane2_real = 1 if lane2_time is not None else 0
        else:
            raise ValueError(scene_type)
        seq.append(SceneStep(i, true_state, lane1_time, lane2_time, lane1_peak, lane2_peak, lane1_real, lane2_real))
    return seq


def rule_classify_step(step, tau_small=700.0, alpha_hi=1.55):
    alpha_lo = 1.0 / alpha_hi
    if step.lane1_time is None and step.lane2_time is None:
        return 'P'
    if step.lane1_time is not None and step.lane2_time is None:
        return 'I1'
    if step.lane2_time is not None and step.lane1_time is None:
        return 'I2'
    dt = abs(step.lane1_time - step.lane2_time)
    alpha = (step.lane1_peak + 1e-6) / (step.lane2_peak + 1e-6)
    if dt > tau_small:
        return 'P'
    if alpha >= alpha_hi:
        return 'I1'
    if alpha <= alpha_lo:
        return 'I2'
    return 'I1' if alpha >= 1.0 else 'I2'


def hmm_classify_sequence(seq, P_s=0.9, P_n=0.35, tau_small=700.0):
    states = ['I1', 'I2', 'P']
    pi = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    T = np.array([
        [0.92, 0.03, 0.05],
        [0.03, 0.92, 0.05],
        [0.03, 0.03, 0.94],
    ])
    posts, labels = [], []
    dt_ref = None
    prev_small_both = 0
    for step in seq:
        pi = pi @ T
        if step.lane1_time is None and step.lane2_time is None:
            lik = np.array([(1 - P_s) * (1 - P_n), (1 - P_s) * (1 - P_n), (1 - P_s) * (1 - P_s)])
            prev_small_both = 0
        elif step.lane1_time is not None and step.lane2_time is None:
            lik = np.array([P_s * (1 - P_n), (1 - P_s) * P_n, 0.10 * P_s * (1 - P_s)])
            prev_small_both = 0
        elif step.lane2_time is not None and step.lane1_time is None:
            lik = np.array([(1 - P_s) * P_n, P_s * (1 - P_n), 0.10 * P_s * (1 - P_s)])
            prev_small_both = 0
        else:
            dt = abs(step.lane1_time - step.lane2_time)
            alpha = (step.lane1_peak + 1e-6) / (step.lane2_peak + 1e-6)
            if dt > tau_small:
                lik = np.array([0.04, 0.04, 0.92])
                prev_small_both = 0
            else:
                prev_small_both += 1
                i1 = statistic_fitting(alpha, 1)
                i2 = statistic_fitting(alpha, 2)
                balance = np.exp(-((math.log(alpha)) ** 2) / (2 * 0.35 ** 2))
                if dt_ref is None:
                    dt_cons = 0.80 if prev_small_both >= 2 else 0.60
                else:
                    dt_cons = np.exp(-((dt - dt_ref) ** 2) / (2 * 55 ** 2))
                p_like = 0.05 + 0.32 * dt_cons * balance + 0.03 * min(prev_small_both, 4) / 4
                lik = np.array([i1, i2, p_like], dtype=float)
        pi = normalize(pi * lik)
        posts.append(pi.copy())
        labels.append(states[int(np.argmax(pi))])
        if step.lane1_time is not None and step.lane2_time is not None and abs(step.lane1_time - step.lane2_time) <= tau_small:
            dt = abs(step.lane1_time - step.lane2_time)
            dt_ref = dt if dt_ref is None else 0.75 * dt_ref + 0.25 * dt
    return np.asarray(posts), labels


def state_to_occ(label):
    if label == 'I1':
        return (1, 0)
    if label == 'I2':
        return (0, 1)
    return (1, 1)


def exp3_metrics(n_seq_per_type=60):
    rows = []
    hard_case = None
    best_margin = -1.0
    for scene in ['single_normal', 'single_adjacent', 'hard_parallel']:
        for sd in range(n_seq_per_type):
            seq = simulate_scene_sequence(scene, seed=sd)
            rule_pred = [rule_classify_step(s) for s in seq]
            post, hmm_pred = hmm_classify_sequence(seq)
            true_labels = [s.true_state for s in seq]
            true_occ = [state_to_occ(s.true_state) for s in seq]
            rule_occ = [state_to_occ(x) for x in rule_pred]
            hmm_occ = [state_to_occ(x) for x in hmm_pred]
            def jitter(pred_occ):
                c = 0
                for lane_idx in (0, 1):
                    for i in range(1, len(pred_occ)):
                        if pred_occ[i][lane_idx] != pred_occ[i - 1][lane_idx]:
                            c += 1
                return c
            row = {
                'scene': scene,
                'rule_acc': np.mean([a == b for a, b in zip(true_labels, rule_pred)]),
                'full_acc': np.mean([a == b for a, b in zip(true_labels, hmm_pred)]),
                'rule_lane_acc': np.mean([rule_occ[i] == true_occ[i] for i in range(len(seq))]),
                'full_lane_acc': np.mean([hmm_occ[i] == true_occ[i] for i in range(len(seq))]),
                'rule_jitter': jitter(rule_occ),
                'full_jitter': jitter(hmm_occ),
                'rule_parallel_as_adj': np.mean([p != 'P' for t, p in zip(true_labels, rule_pred) if t == 'P']) if scene == 'hard_parallel' else np.nan,
                'full_parallel_as_adj': np.mean([p != 'P' for t, p in zip(true_labels, hmm_pred) if t == 'P']) if scene == 'hard_parallel' else np.nan,
            }
            rows.append(row)
            if scene == 'hard_parallel':
                margin = row['full_acc'] - row['rule_acc']
                if margin > best_margin:
                    best_margin = margin
                    hard_case = {'seq': seq, 'post': post, 'rule_pred': rule_pred, 'hmm_pred': hmm_pred}
    return pd.DataFrame(rows), hard_case


# ------------------------------------------------------------
# experiment 4/5: order-sensitive simple tracker
# ------------------------------------------------------------

@dataclass
class TrackEvent:
    eid: int
    time_ms: float
    lane: int
    dev: int
    peak: float
    tlen: float
    true_vid: Optional[int]
    is_false: bool = False
    arrival_ms: Optional[float] = None


@dataclass
class Vehicle:
    vid: int
    lane: int
    start_ms: float
    speed: float
    base_peak: float
    base_tlen: float


@dataclass
class TrackPoint:
    eid: int
    time_ms: float
    dev: int
    true_vid: Optional[int]
    residual_ms: float
    pos_err_m: float


@dataclass
class Track:
    tid: int
    lane: int
    points: list = field(default_factory=list)
    speed: float = 22.0
    mean_log_peak: float = 6.0
    mean_tlen: float = 200.0
    @property
    def last_time(self):
        return self.points[-1].time_ms
    @property
    def last_dev(self):
        return self.points[-1].dev
    def add(self, ev, pred_time, residual_ms):
        pos_err = abs(residual_ms) / 1000.0 * self.speed
        self.points.append(TrackPoint(ev.eid, ev.time_ms, ev.dev, ev.true_vid, residual_ms, pos_err))
        if len(self.points) == 1:
            self.mean_log_peak = math.log(ev.peak + 1.0)
            self.mean_tlen = ev.tlen
        else:
            prev = self.points[-2]
            dd = max(1, ev.dev - prev.dev)
            dt = max((ev.time_ms - prev.time_ms) / 1000.0, 0.05)
            obs_speed = 15.0 * dd / dt
            self.speed = float(clamp(0.72 * self.speed + 0.28 * obs_speed, 8.0, 36.0))
            self.mean_log_peak = 0.86 * self.mean_log_peak + 0.14 * math.log(ev.peak + 1.0)
            self.mean_tlen = 0.86 * self.mean_tlen + 0.14 * ev.tlen


def sample_vehicle_signature(lane, rng):
    if lane == 1:
        peak = float(np.exp(rng.normal(np.log(800.0), 0.18)))
        tlen = float(np.exp(rng.normal(np.log(330.0), 0.20)))
    else:
        peak = float(np.exp(rng.normal(np.log(220.0), 0.20)))
        tlen = float(np.exp(rng.normal(np.log(160.0), 0.22)))
    return peak, tlen


def simulate_miss_stream(miss_len=2, density='normal', seed=0, horizon_s=360, n_nodes=72, spacing=15.0, add_false=True):
    rng = np.random.default_rng(seed)
    lam = {'low': 0.030, 'normal': 0.080, 'high': 0.100}[density]
    vehicles = []
    vid = 1
    for lane in (1, 2):
        t = 0.0
        while t < horizon_s - 60.0:
            t += rng.exponential(1.0 / lam)
            if t >= horizon_s - 60.0:
                break
            speed = float(clamp(rng.normal(22.0, 2.0), 14.0, 32.0))
            peak, tlen = sample_vehicle_signature(lane, rng)
            vehicles.append(Vehicle(vid, lane, t * 1000.0, speed, peak, tlen))
            vid += 1
    vehicles.sort(key=lambda v: v.start_ms)

    start_dev = 28
    miss_nodes = set(range(start_dev, start_dev + miss_len))
    reliability_maps = {}
    for sec in range(int(horizon_s) + 1):
        rel = {}
        for lane in (1, 2):
            for dev in range(1, n_nodes + 1):
                if dev in miss_nodes:
                    rel[(lane, dev)] = 0.38 if density == 'low' else 0.28
                else:
                    rel[(lane, dev)] = 0.92
        reliability_maps[sec] = rel

    events = []
    eid = 1
    for v in vehicles:
        for dev in range(1, n_nodes + 1):
            t_ms = v.start_ms + (dev - 1) * spacing / v.speed * 1000.0 + rng.normal(0.0, 28.0)
            if t_ms > horizon_s * 1000.0:
                break
            detected = True
            if dev in miss_nodes:
                detected = False
            else:
                miss_p = 0.02 if density == 'normal' else 0.012
                if rng.random() < miss_p:
                    detected = False
            if detected:
                peak = float(clamp(v.base_peak * np.exp(rng.normal(0.0, 0.10)), 80.0, 6000.0))
                tlen = float(clamp(v.base_tlen * np.exp(rng.normal(0.0, 0.12)), 40.0, 2500.0))
                events.append(TrackEvent(eid, t_ms, v.lane, dev, peak, tlen, v.vid, False, None))
                eid += 1

    if add_false:
        p_false = 0.004 if density == 'low' else 0.010
        for lane in (1, 2):
            for dev in range(1, n_nodes + 1):
                rate = p_false * (2.8 if dev in miss_nodes else 1.0)
                n_false = rng.poisson(rate * horizon_s)
                for _ in range(n_false):
                    t_ms = float(rng.uniform(0.0, horizon_s * 1000.0))
                    peak = float(clamp(np.exp(rng.normal(np.log(180.0 if lane == 2 else 300.0), 0.45)), 80.0, 4000.0))
                    tlen = float(clamp(np.exp(rng.normal(np.log(120.0 if lane == 2 else 200.0), 0.35)), 40.0, 1800.0))
                    events.append(TrackEvent(eid, t_ms, lane, dev, peak, tlen, None, True, None))
                    eid += 1

    events.sort(key=lambda e: (e.time_ms, e.lane, e.dev, e.eid))
    meta = {
        'miss_nodes': sorted(miss_nodes),
        'start_dev': start_dev,
        'horizon_s': horizon_s,
        'n_nodes': n_nodes,
    }
    return events, vehicles, reliability_maps, meta


METHOD_CFGS = {
    'rule': dict(base_gate=500.0, skip_gate=170.0, rel_gate_scale=0.0, base_skip=2, use_rel=False, use_mag=False, mag_peak_w=0.0, mag_tlen_w=0.0, score_thresh=0.032),
    'global': dict(base_gate=520.0, skip_gate=210.0, rel_gate_scale=220.0, base_skip=2, use_rel=True, use_mag=True, mag_peak_w=0.35, mag_tlen_w=0.35, score_thresh=0.030),
    'full': dict(base_gate=500.0, skip_gate=300.0, rel_gate_scale=420.0, base_skip=4, use_rel=True, use_mag=True, mag_peak_w=0.70, mag_tlen_w=0.60, score_thresh=0.032),
}


def build_tracks(events, reliability_maps, method='full'):
    cfg = METHOD_CFGS[method]
    tracks = []
    next_tid = 1
    for lane in (1, 2):
        active = []
        lane_events = [e for e in events if e.lane == lane]
        for ev in lane_events:
            sec = int(ev.time_ms // 1000)
            rel = reliability_maps[min(sec, max(reliability_maps.keys()))].get((lane, ev.dev), 0.92)
            best, best_score, best_pred, best_res = None, -1.0, ev.time_ms, 0.0
            for tr in active:
                dd = ev.dev - tr.last_dev
                if dd <= 0:
                    continue
                max_skip = cfg['base_skip'] + (1 if cfg['use_rel'] and rel < 0.40 else 0)
                if dd > max_skip + 1:
                    continue
                pred = tr.last_time + (15.0 * dd / max(tr.speed, 8.0)) * 1000.0
                res = ev.time_ms - pred
                gate = cfg['base_gate'] + cfg['skip_gate'] * (dd - 1) + cfg['rel_gate_scale'] * (1.0 - rel)
                if abs(res) > gate:
                    continue
                score = math.exp(-(res ** 2) / (2.0 * max(gate / 1.9, 75.0) ** 2))
                if cfg['use_mag']:
                    peak_diff = abs(math.log(ev.peak + 1.0) - tr.mean_log_peak)
                    tlen_diff = abs(ev.tlen - tr.mean_tlen) / max(60.0, tr.mean_tlen)
                    score *= math.exp(-cfg['mag_peak_w'] * peak_diff) * math.exp(-cfg['mag_tlen_w'] * tlen_diff)
                if cfg['use_rel']:
                    score *= (0.55 + 0.45 * rel)
                score *= math.exp(-0.10 * (dd - 1))
                if score > best_score:
                    best, best_score, best_pred, best_res = tr, score, pred, res
            if best is None or best_score < cfg['score_thresh']:
                tr = Track(next_tid, lane)
                next_tid += 1
                tr.add(ev, ev.time_ms, 0.0)
                active.append(tr)
                tracks.append(tr)
            else:
                best.add(ev, best_pred, best_res)

    point_rows, event_to_track, track_by_id = [], {}, {}
    for tr in tracks:
        track_by_id[tr.tid] = tr
        if len(tr.points) < 2:
            continue
        for p in tr.points:
            point_rows.append({
                'track_id': tr.tid,
                'lane': tr.lane,
                'eid': p.eid,
                'time_ms': p.time_ms,
                'dev': p.dev,
                'true_vid': p.true_vid,
                'residual_ms': p.residual_ms,
                'pos_err_m': p.pos_err_m,
            })
            event_to_track[p.eid] = tr.tid
    return pd.DataFrame(point_rows), event_to_track, track_by_id


def tracking_metrics(events, event_to_track):
    by_vid = defaultdict(list)
    for e in events:
        if e.true_vid is not None:
            by_vid[e.true_vid].append(e)
    idsw, frag = 0, 0
    for vid, seq in by_vid.items():
        seq = sorted(seq, key=lambda e: e.time_ms)
        preds = [event_to_track.get(e.eid) for e in seq if not e.is_false]
        last = None
        segs = []
        for p in preds:
            if p is None:
                last = None
            else:
                if last is None or p != last:
                    segs.append(p)
                last = p
        if preds:
            last = None
            for p in preds:
                if p is None:
                    continue
                if last is None:
                    last = p
                elif p != last:
                    idsw += 1
                    last = p
        frag += max(0, len(segs) - 1)
    return {'IDSW': idsw, 'Frag': frag, 'n_vehicles': len(by_vid)}


def survival_rate(events, event_to_track, meta):
    start_dev = meta['start_dev']
    miss_nodes = set(meta['miss_nodes'])
    pre_max = start_dev - 1
    post_min = max(miss_nodes) + 1 if miss_nodes else pre_max + 1
    by_vid = defaultdict(list)
    for e in events:
        if e.true_vid is not None:
            by_vid[e.true_vid].append(e)
    survives = []
    for vid, seq in by_vid.items():
        seq = sorted(seq, key=lambda e: e.dev)
        pre = [e for e in seq if e.dev <= pre_max]
        post = [e for e in seq if e.dev >= post_min]
        if not pre or not post:
            continue
        t1 = event_to_track.get(pre[-1].eid)
        t2 = event_to_track.get(post[0].eid)
        survives.append(int(t1 is not None and t1 == t2))
    return (sum(survives) / len(survives)) if survives else np.nan


def bridge_rmse(events, event_to_track, track_by_id, meta, penalty_m=35.0):
    start_dev = meta['start_dev']
    miss_nodes = set(meta['miss_nodes'])
    pre_max = start_dev - 1
    post_min = max(miss_nodes) + 1 if miss_nodes else pre_max + 1
    by_vid = defaultdict(list)
    for e in events:
        if e.true_vid is not None:
            by_vid[e.true_vid].append(e)
    errs = []
    for vid, seq in by_vid.items():
        seq = sorted(seq, key=lambda e: e.dev)
        pre = [e for e in seq if e.dev <= pre_max]
        post = [e for e in seq if e.dev >= post_min]
        if not pre or not post:
            continue
        e1, e2 = pre[-1], post[0]
        tid1 = event_to_track.get(e1.eid)
        tid2 = event_to_track.get(e2.eid)
        if tid1 is None or tid2 is None or tid1 != tid2:
            errs.append(penalty_m)
            continue
        tr = track_by_id[tid1]
        speed = tr.speed
        dd = e2.dev - e1.dev
        pred = e1.time_ms + (15.0 * dd / max(speed, 8.0)) * 1000.0
        err_m = abs((e2.time_ms - pred) / 1000.0 * speed)
        errs.append(err_m)
    return float(np.sqrt(np.mean(np.asarray(errs) ** 2))) if errs else np.nan


# ------------------------------------------------------------
# experiment 5: time organization + replay
# ------------------------------------------------------------

def simulate_timeorg_stream(seed=0, horizon_s=180, density='high'):
    events, vehicles, rel_maps, meta = simulate_miss_stream(miss_len=0, density=density, seed=seed, horizon_s=horizon_s, add_false=True)
    meta['miss_nodes'] = []
    return events, vehicles, rel_maps, meta


def assign_arrival_times(events, delay_bank, seed=0, scale=1.4):
    rng = np.random.default_rng(seed)
    out = []
    for e in events:
        delay = float(rng.choice(delay_bank)) * scale
        arr = e.time_ms + delay + rng.normal(0.0, 45.0)
        out.append(TrackEvent(e.eid, e.time_ms, e.lane, e.dev, e.peak, e.tlen, e.true_vid, e.is_false, arr))
    return out


def replay_processing_order(events_with_arrival, delta_T_ms=350.0, L_scans=3):
    arrivals = sorted(events_with_arrival, key=lambda e: (e.arrival_ms, e.eid))
    buffer = defaultdict(list)
    processed = []
    max_scan_seen = -1
    committed_cutoff = -1
    late_total, late_absorbed, too_late = 0, 0, 0
    buf_sizes = []
    for ev in arrivals:
        scan = int(ev.time_ms // delta_T_ms)
        arrival_scan = int(ev.arrival_ms // delta_T_ms)
        if arrival_scan > scan:
            late_total += 1
        if scan > max_scan_seen:
            max_scan_seen = scan
        while True:
            eligible = [s for s in buffer.keys() if s <= max_scan_seen - L_scans - 1]
            if not eligible:
                break
            s = min(eligible)
            processed.extend(sorted(buffer.pop(s), key=lambda x: (x.time_ms, x.eid)))
            committed_cutoff = max(committed_cutoff, s)
        if scan <= committed_cutoff:
            processed.append(ev)
            too_late += 1
        else:
            buffer[scan].append(ev)
            if arrival_scan > scan:
                late_absorbed += 1
        buf_sizes.append(sum(len(v) for v in buffer.values()))
    for s in sorted(buffer):
        processed.extend(sorted(buffer[s], key=lambda x: (x.time_ms, x.eid)))
    info = {
        'late_total': late_total,
        'late_absorb_ratio': (late_absorbed / late_total) if late_total else np.nan,
        'too_late': too_late,
        'avg_buffer_events': float(np.mean(buf_sizes)) if buf_sizes else 0.0,
        'max_buffer_events': max(buf_sizes) if buf_sizes else 0,
    }
    return processed, info


def same_scan_adjacent_ratio(events, delta_T_ms=350.0):
    by_vid = defaultdict(list)
    for e in events:
        if e.true_vid is not None:
            by_vid[e.true_vid].append(e)
    num, den = 0, 0
    for vid, seq in by_vid.items():
        seq = sorted(seq, key=lambda e: (e.dev, e.time_ms))
        by_dev = {e.dev: e for e in seq}
        for dev, e1 in by_dev.items():
            if (dev + 1) in by_dev:
                e2 = by_dev[dev + 1]
                den += 1
                if int(e1.time_ms // delta_T_ms) == int(e2.time_ms // delta_T_ms):
                    num += 1
    return (num / den) if den else np.nan


def local_transition_consistency(events, ref_labels, test_labels):
    by_vid = defaultdict(list)
    for e in events:
        if e.true_vid is not None:
            by_vid[e.true_vid].append(e)
    good, total = 0, 0
    for vid, seq in by_vid.items():
        seq = sorted(seq, key=lambda e: e.time_ms)
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            ref_same = ref_labels.get(a.eid) is not None and ref_labels.get(a.eid) == ref_labels.get(b.eid)
            test_same = test_labels.get(a.eid) is not None and test_labels.get(a.eid) == test_labels.get(b.eid)
            good += int(ref_same == test_same)
            total += 1
    return (good / total) if total else np.nan


# ------------------------------------------------------------
# ablation helpers
# ------------------------------------------------------------

def constant_reliability_maps(horizon_s, n_nodes, value=0.92):
    out = {}
    for sec in range(int(horizon_s) + 1):
        out[sec] = {(lane, dev): value for lane in (1, 2) for dev in range(1, n_nodes + 1)}
    return out


def hard_count_reliability_maps(base_rel_maps, events, meta):
    miss_nodes = set(meta['miss_nodes'])
    horizon_s = int(meta['horizon_s'])
    n_nodes = int(meta['n_nodes'])
    rel_maps = {sec: base_rel_maps[sec].copy() for sec in base_rel_maps}
    counts = defaultdict(int)
    for e in events:
        counts[(int(e.time_ms // 1000), e.lane, e.dev)] += 1
    for sec in range(horizon_s + 1):
        for lane in (1, 2):
            for dev in range(1, n_nodes + 1):
                if dev in miss_nodes or (dev - 1) in miss_nodes or (dev + 1) in miss_nodes:
                    c = counts.get((sec, lane, dev), 0)
                    if c == 0:
                        rel_maps[sec][(lane, dev)] = 0.18
                    elif c == 1:
                        rel_maps[sec][(lane, dev)] = 0.74
                    else:
                        rel_maps[sec][(lane, dev)] = 0.42
    return rel_maps


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------

def plot_exp3_hard_case(hard_case, out_path):
    seq = hard_case['seq']
    post = hard_case['post']
    rule_pred = hard_case['rule_pred']
    hmm_pred = hard_case['hmm_pred']
    x = np.arange(len(seq))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    axes[0].plot(x, post[:, 0], label='P(I1)', linewidth=2)
    axes[0].plot(x, post[:, 1], label='P(I2)', linewidth=2)
    axes[0].plot(x, post[:, 2], label='P(P)', linewidth=2)
    axes[0].set_ylabel('后验概率')
    axes[0].set_title('困难双车并行场景下的场景后验递推过程')
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=3)

    mapping = {'I1': 0, 'I2': 1, 'P': 2}
    true_y = [mapping[s.true_state] for s in seq]
    rule_y = [mapping[s] for s in rule_pred]
    hmm_y = [mapping[s] for s in hmm_pred]
    axes[1].plot(x, true_y, marker='o', linewidth=2, label='truth')
    axes[1].plot(x, rule_y, marker='x', linewidth=1.7, label='single-frame rule')
    axes[1].plot(x, hmm_y, marker='s', linewidth=1.7, label='multi-step recursion')
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['I1', 'I2', 'P'])
    axes[1].set_xlabel('节点编号')
    axes[1].set_ylabel('场景状态')
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_exp3_metrics(agg_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    labels = ['单帧阈值规则', '多步场景递推模型']
    axes[0].bar(labels, [agg_df['rule_acc'], agg_df['full_acc']])
    axes[0].set_title('场景判别准确率')
    axes[0].grid(axis='y', alpha=0.3)
    axes[1].bar(labels, [agg_df['rule_parallel_as_adj'], agg_df['full_parallel_as_adj']])
    axes[1].set_title('双车并行误判为邻道干扰比例')
    axes[1].grid(axis='y', alpha=0.3)
    axes[2].bar(labels, [agg_df['rule_jitter'], agg_df['full_jitter']])
    axes[2].set_title('车道状态抖动次数')
    axes[2].grid(axis='y', alpha=0.3)
    fig.suptitle('实验三：场景判别指标汇总')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_exp4_miss_tolerance(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True)
    method_map = {'rule': '规则模型', 'global': '全局模型', 'full': '本文方法'}
    for density, ax in zip(['normal', 'low'], axes):
        sub = df[df['density'] == density]
        for method in ['rule', 'global', 'full']:
            ss = sub[sub['method'] == method].sort_values('miss_len')
            ax.plot(ss['miss_len'], ss['survival'], marker='o', linewidth=2, label=f'{method_map[method]} 连续性保持率')
            ax.plot(ss['miss_len'], ss['RMSE_p'], marker='s', linewidth=1.5, linestyle='--', label=f'{method_map[method]} 跨段RMSE')
        ax.set_title('常规车流密度' if density == 'normal' else '低密度车流')
        ax.set_xlabel('连续漏检信标数量 m')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('轨迹连续保持率 / 跨段RMSE (m)')
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('实验四：连续漏检容忍能力')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_exp5_time_consistency(scan_df, lag_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    ss = scan_df.sort_values('delta_T_ms')
    axes[0].plot(ss['delta_T_ms'] / 1000.0, ss['same_scan_ratio'], marker='o', linewidth=2)
    axes[0].set_title('同一扫描内相邻节点同车触发比例')
    axes[0].set_xlabel('扫描时间间隔 ΔT (s)')
    axes[0].grid(alpha=0.3)

    ll = lag_df.sort_values('L')
    axes[1].plot(ll['L'], ll['late_absorb_ratio'], marker='o', linewidth=2)
    axes[1].set_title('迟到量测被窗口吸收比例')
    axes[1].set_xlabel('固定滞后窗口长度 L')
    axes[1].grid(alpha=0.3)

    axes[2].plot(ll['L'], ll['consistency'], marker='o', linewidth=2)
    axes[2].set_title('离线重排与在线回放结果一致率')
    axes[2].set_xlabel('固定滞后窗口长度 L')
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_ablation(ab_df, out_path):
    variant_map = {
        'full': '本文完整方法',
        'no_node_adapt': '去除节点自适应',
        'no_scene_hmm': '去除三状态场景判别',
        'no_replay': '去除固定滞后回放',
        'no_mag': '去除磁特征兼容',
        'hard_count': '硬计数即时更新'
    }
    variants = list(ab_df['variant'])
    x = np.arange(len(variants))
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.6), sharex=True)
    axes[0].bar(x - 1.5 * width, ab_df['RMSE_p'], width, label='桥接位置RMSE')
    axes[0].bar(x - 0.5 * width, ab_df['IDSW'], width, label='ID切换次数')
    axes[0].bar(x + 0.5 * width, ab_df['Frag'], width, label='轨迹碎片数')
    axes[0].bar(x + 1.5 * width, ab_df['bridge_survival'] * 100.0, width, label='轨迹保持率(×100)')
    axes[0].set_title('消融实验：跟踪性能指标比较')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(ncol=4)

    axes[1].bar(x - 0.20, ab_df['scene_acc'], 0.40, label='scene accuracy')
    axes[1].bar(x + 0.20, ab_df['lane_acc'], 0.40, label='lane accuracy')
    axes[1].set_title('消融实验：场景与车道判别准确率')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([variant_map.get(v, v) for v in variants], rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_complexity(lag_df, out_path):
    ll = lag_df.sort_values('L')
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.plot(ll['L'], ll['avg_buffer_events'], marker='o', linewidth=2, label='avg buffered events')
    ax1.set_xlabel('固定滞后窗口长度 L')
    ax1.set_ylabel('平均缓冲事件数量')
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ll['L'], ll['consistency'], marker='s', linewidth=2, linestyle='--', label='consistency')
    ax2.set_ylabel('轨迹结果一致率')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    plt.title('固定滞后回放窗口 L 的复杂度与收益折中')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------
# main runner
# ------------------------------------------------------------

def main():
    base_dir = os.path.join(SCRIPT_DIR, 'chapter3_exp345_results')
    os.makedirs(base_dir, exist_ok=True)
    data_dir = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data')
    bank = load_real_bank([
        os.path.join(data_dir, 'streamIn1215.txt'),
        os.path.join(data_dir, 'streamIn1216.txt')
    ])

    # Experiment 3
    exp3_df, hard_case = exp3_metrics(n_seq_per_type=60)
    exp3_scene = exp3_df.groupby('scene')[['rule_acc', 'full_acc', 'rule_lane_acc', 'full_lane_acc', 'rule_jitter', 'full_jitter', 'rule_parallel_as_adj', 'full_parallel_as_adj']].mean().reset_index()
    exp3_summary = {
        'rule_acc': float(exp3_df['rule_acc'].mean()),
        'full_acc': float(exp3_df['full_acc'].mean()),
        'rule_lane_acc': float(exp3_df['rule_lane_acc'].mean()),
        'full_lane_acc': float(exp3_df['full_lane_acc'].mean()),
        'rule_jitter': float(exp3_df['rule_jitter'].mean()),
        'full_jitter': float(exp3_df['full_jitter'].mean()),
        'rule_parallel_as_adj': float(exp3_df['rule_parallel_as_adj'].dropna().mean()),
        'full_parallel_as_adj': float(exp3_df['full_parallel_as_adj'].dropna().mean()),
    }
    pd.DataFrame([exp3_summary]).to_csv(os.path.join(base_dir, 'exp3_summary.csv'), index=False)
    exp3_scene.to_csv(os.path.join(base_dir, 'exp3_scene_breakdown.csv'), index=False)
    plot_exp3_hard_case(hard_case, os.path.join(base_dir, 'exp3_hard_case_posterior.png'))
    plot_exp3_metrics(exp3_summary, os.path.join(base_dir, 'exp3_scene_metrics.png'))

    # Experiment 4
    exp4_rows = []
    for density in ['normal', 'low']:
        for miss_len in [0, 1, 2, 3, 4, 5]:
            for seed in [1, 2, 3, 4]:
                events, vehicles, rel_maps, meta = simulate_miss_stream(miss_len=miss_len, density=density, seed=seed, horizon_s=260, add_false=True)
                for method in ['rule', 'global', 'full']:
                    pts, e2t, tmap = build_tracks(events, rel_maps, method)
                    tm = tracking_metrics(events, e2t)
                    surv = survival_rate(events, e2t, meta)
                    rmse_p = bridge_rmse(events, e2t, tmap, meta)
                    exp4_rows.append({
                        'density': density,
                        'miss_len': miss_len,
                        'seed': seed,
                        'method': method,
                        'survival': surv,
                        'RMSE_p': rmse_p,
                        'IDSW': tm['IDSW'],
                        'Frag': tm['Frag'],
                    })
    exp4_df = pd.DataFrame(exp4_rows)
    exp4_agg = exp4_df.groupby(['density', 'miss_len', 'method'])[['survival', 'RMSE_p', 'IDSW', 'Frag']].mean().reset_index()
    exp4_agg.to_csv(os.path.join(base_dir, 'exp4_miss_tolerance.csv'), index=False)
    plot_exp4_miss_tolerance(exp4_agg, os.path.join(base_dir, 'exp4_miss_tolerance.png'))

    # Experiment 5
    scan_rows, lag_rows = [], []
    for seed in [1, 2, 3]:
        events, vehicles, rel_maps, meta = simulate_timeorg_stream(seed=seed, horizon_s=180, density='high')
        arr_events = assign_arrival_times(events, bank['delays_ms'], seed=seed, scale=1.4)
        offline = sorted(arr_events, key=lambda e: (e.time_ms, e.eid))
        _, e2t_off, _ = build_tracks(offline, rel_maps, 'full')
        for dt in [300, 500, 700, 900, 1100, 1300]:
            scan_rows.append({'seed': seed, 'delta_T_ms': dt, 'same_scan_ratio': same_scan_adjacent_ratio(offline, dt)})
        for L in [0, 1, 2, 3, 4, 5]:
            t0 = time.perf_counter()
            online_order, info = replay_processing_order(arr_events, delta_T_ms=350.0, L_scans=L)
            _, e2t_on, _ = build_tracks(online_order, rel_maps, 'full')
            elapsed = (time.perf_counter() - t0) * 1000.0
            lag_rows.append({
                'seed': seed,
                'L': L,
                'late_absorb_ratio': info['late_absorb_ratio'],
                'consistency': local_transition_consistency(offline, e2t_off, e2t_on),
                'avg_buffer_events': info['avg_buffer_events'],
                'max_buffer_events': info['max_buffer_events'],
                'runtime_ms': elapsed,
            })
    scan_df = pd.DataFrame(scan_rows).groupby('delta_T_ms')[['same_scan_ratio']].mean().reset_index()
    lag_df = pd.DataFrame(lag_rows).groupby('L')[['late_absorb_ratio', 'consistency', 'avg_buffer_events', 'max_buffer_events', 'runtime_ms']].mean().reset_index()
    scan_df.to_csv(os.path.join(base_dir, 'exp5_same_scan_ratio.csv'), index=False)
    lag_df.to_csv(os.path.join(base_dir, 'exp5_replay_consistency.csv'), index=False)
    plot_exp5_time_consistency(scan_df, lag_df, os.path.join(base_dir, 'exp5_time_consistency.png'))
    plot_complexity(lag_df, os.path.join(base_dir, 'complexity_tradeoff.png'))

    # Ablation
    variants = ['full', 'no_node_adapt', 'no_scene_hmm', 'no_replay', 'no_mag', 'hard_count']
    ab_rows = []
    for variant in variants:
        track_rows = []
        for seed in [1, 2, 3]:
            events, vehicles, rel_maps, meta = simulate_miss_stream(miss_len=3, density='normal', seed=seed, horizon_s=220, add_false=True)
            arr_events = assign_arrival_times(events, bank['delays_ms'], seed=seed, scale=1.35)
            if variant == 'no_node_adapt':
                use_rel_maps = constant_reliability_maps(meta['horizon_s'], meta['n_nodes'], 0.92)
                L = 4
                tracker_method = 'full'
            elif variant == 'hard_count':
                use_rel_maps = hard_count_reliability_maps(rel_maps, events, meta)
                L = 4
                tracker_method = 'full'
            else:
                use_rel_maps = rel_maps
                L = 0 if variant == 'no_replay' else 4
                tracker_method = 'full'
            full_backup = METHOD_CFGS['full'].copy()
            if variant == 'no_mag':
                METHOD_CFGS['full']['use_mag'] = False
                METHOD_CFGS['full']['mag_peak_w'] = 0.0
                METHOD_CFGS['full']['mag_tlen_w'] = 0.0
                METHOD_CFGS['full']['score_thresh'] = 0.028
            online_order, info = replay_processing_order(arr_events, delta_T_ms=350.0, L_scans=L)
            pts, e2t, tmap = build_tracks(online_order, use_rel_maps, tracker_method)
            tm = tracking_metrics(events, e2t)
            track_rows.append({
                'RMSE_p': bridge_rmse(events, e2t, tmap, meta),
                'IDSW': tm['IDSW'],
                'Frag': tm['Frag'],
                'bridge_survival': survival_rate(events, e2t, meta),
                'late_absorb_ratio': info['late_absorb_ratio'],
            })
            METHOD_CFGS['full'] = full_backup
        # scene-side metrics
        scene_rows = []
        for scene in ['single_normal', 'single_adjacent', 'hard_parallel']:
            for sd in range(40):
                seq = simulate_scene_sequence(scene, seed=sd)
                if variant == 'no_scene_hmm':
                    pred = [rule_classify_step(s) for s in seq]
                else:
                    _, pred = hmm_classify_sequence(seq)
                scene_rows.append({
                    'scene_acc': np.mean([pred[i] == seq[i].true_state for i in range(len(seq))]),
                    'lane_acc': np.mean([state_to_occ(pred[i]) == state_to_occ(seq[i].true_state) for i in range(len(seq))]),
                })
        track_df = pd.DataFrame(track_rows).mean(numeric_only=True)
        scene_df = pd.DataFrame(scene_rows).mean(numeric_only=True)
        row = {'variant': variant}
        row.update(track_df.to_dict())
        row.update(scene_df.to_dict())
        ab_rows.append(row)
    ab_df = pd.DataFrame(ab_rows)
    ab_df.to_csv(os.path.join(base_dir, 'ablation_summary.csv'), index=False)
    plot_ablation(ab_df, os.path.join(base_dir, 'ablation_metrics.png'))

    # text summary
    lines = []
    lines.append('# Chapter 3 experiments 3/4/5 + ablation summary')
    lines.append('')
    lines.append('## Exp.3')
    lines.append(pd.DataFrame([exp3_summary]).round(4).to_markdown(index=False))
    lines.append('')
    lines.append('## Exp.4')
    lines.append(exp4_agg.round(4).to_markdown(index=False))
    lines.append('')
    lines.append('## Exp.5')
    lines.append(scan_df.round(4).to_markdown(index=False))
    lines.append('')
    lines.append(lag_df.round(4).to_markdown(index=False))
    lines.append('')
    lines.append('## Ablation')
    lines.append(ab_df.round(4).to_markdown(index=False))
    with open(os.path.join(base_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Results written to', base_dir)
    print('Exp3 summary:')
    print(pd.DataFrame([exp3_summary]).round(4))
    print('Exp4 key:')
    print(exp4_agg[(exp4_agg['method'] == 'full') & (exp4_agg['miss_len'].isin([2, 3, 4]))].round(4))
    print('Exp5 lag:')
    print(lag_df.round(4))
    print('Ablation:')
    print(ab_df.round(4))

if __name__ == '__main__':
    main()


# ===== FILE: chapter3_targeted_ablation.py =====

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chapter3_experiments_345 as mod


def plot_tracking_ablation(df, out_path):

    df['variant'] = df['variant'].replace({
        'full': '本文方法',
        'no_node_adapt': '去除节点自适应',
        'no_mag': '去除磁特征兼容',
        'hard_count': '硬计数即时更新'
    })

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].bar(df['variant'], df['RMSE_p'])
    axes[0].set_title('跨段位置RMSE (m)')
    axes[0].set_ylabel('误差')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(df['variant'], df['IDSW'])
    axes[1].set_title('身份切换次数 IDSW')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(df['variant'], df['Frag'])
    axes[2].set_title('轨迹碎片数 Frag')
    axes[2].grid(axis='y', alpha=0.3)

    axes[3].bar(df['variant'], df['survival'])
    axes[3].set_title('轨迹连续保持率')
    axes[3].set_ylabel('比例')
    axes[3].grid(axis='y', alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')


def plot_scene_ablation(df, out_path):

    df['variant'] = df['variant'].replace({
        'full': '本文方法',
        'no_scene_hmm': '去除场景递推'
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

    axes[0].bar(df['variant'], df['scene_acc'])
    axes[0].set_title('场景判别准确率')
    axes[0].set_ylabel('准确率')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(df['variant'], df['lane_acc'])
    axes[1].set_title('车道识别准确率')
    axes[1].set_ylabel('准确率')
    axes[1].grid(axis='y', alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_replay_ablation(df, out_path):
    df['variant'] = df['variant'].replace({
        'full': '本文方法',
        'no_replay': '无回放机制'
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    axes[0].bar(df['variant'], df['late_absorb_ratio'])
    axes[0].set_title('迟到量测吸收比例')
    axes[0].set_ylabel('比例')
    axes[0].grid(axis='y', alpha=0.3)
    axes[1].bar(df['variant'], df['consistency'])
    axes[1].set_title('离线重排与在线回放结果一致率')
    axes[1].set_ylabel('一致率')
    axes[1].grid(axis='y', alpha=0.3)
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    base_dir = os.path.join(SCRIPT_DIR, 'chapter3_exp345_results')
    os.makedirs(base_dir, exist_ok=True)
    data_dir = os.path.join(SCRIPT_DIR, '..', '..', '..',  'data')
    bank = mod.load_real_bank([
        os.path.join(data_dir, 'streamIn1215.txt'),
        os.path.join(data_dir, 'streamIn1216.txt')
    ])

    # 1) tracking-side ablation on Exp.4-style short-miss stress
    tracking_rows = []
    for variant in ['full', 'no_node_adapt', 'no_mag', 'hard_count']:
        vals = []
        for seed in [1, 2, 3, 4]:
            events, vehicles, rel_maps, meta = mod.simulate_miss_stream(miss_len=3, density='normal', seed=seed, horizon_s=260, add_false=True)
            if variant == 'no_node_adapt':
                method = 'global'
                use_rel = rel_maps
            elif variant == 'no_mag':
                backup = mod.METHOD_CFGS['full'].copy()
                mod.METHOD_CFGS['full']['use_mag'] = False
                mod.METHOD_CFGS['full']['mag_peak_w'] = 0.0
                mod.METHOD_CFGS['full']['mag_tlen_w'] = 0.0
                mod.METHOD_CFGS['full']['score_thresh'] = 0.028
                method = 'full'
                use_rel = rel_maps
            elif variant == 'hard_count':
                # current-scan hard-count proxy: overconfident immediate updates -> too-tight gates
                backup = mod.METHOD_CFGS['full'].copy()
                mod.METHOD_CFGS['full'] = dict(
                    base_gate=380.0, skip_gate=120.0, rel_gate_scale=0.0, base_skip=2,
                    use_rel=False, use_mag=True, mag_peak_w=0.15, mag_tlen_w=0.15, score_thresh=0.055,
                )
                method = 'full'
                use_rel = mod.constant_reliability_maps(meta['horizon_s'], meta['n_nodes'], 0.98)
            else:
                method = 'full'
                use_rel = rel_maps
            pts, e2t, tmap = mod.build_tracks(events, use_rel, method)
            tm = mod.tracking_metrics(events, e2t)
            vals.append({
                'RMSE_p': mod.bridge_rmse(events, e2t, tmap, meta),
                'IDSW': tm['IDSW'],
                'Frag': tm['Frag'],
                'survival': mod.survival_rate(events, e2t, meta),
            })
            if variant in ['no_mag', 'hard_count']:
                mod.METHOD_CFGS['full'] = backup
        row = {'variant': variant}
        row.update(pd.DataFrame(vals).mean(numeric_only=True).to_dict())
        tracking_rows.append(row)
    tracking_df = pd.DataFrame(tracking_rows)
    tracking_df.to_csv(os.path.join(base_dir, 'ablation_tracking_targeted.csv'), index=False)
    plot_tracking_ablation(tracking_df, os.path.join(base_dir, 'ablation_tracking_targeted.png'))

    # 2) scene-side ablation on Exp.3
    scene_rows = []
    for variant in ['full', 'no_scene_hmm']:
        vals = []
        for scene in ['single_normal', 'single_adjacent', 'hard_parallel']:
            for sd in range(60):
                seq = mod.simulate_scene_sequence(scene, seed=sd)
                if variant == 'no_scene_hmm':
                    pred = [mod.rule_classify_step(s) for s in seq]
                else:
                    _, pred = mod.hmm_classify_sequence(seq)
                vals.append({
                    'scene_acc': sum(pred[i] == seq[i].true_state for i in range(len(seq))) / len(seq),
                    'lane_acc': sum(mod.state_to_occ(pred[i]) == mod.state_to_occ(seq[i].true_state) for i in range(len(seq))) / len(seq),
                })
        row = {'variant': variant}
        row.update(pd.DataFrame(vals).mean(numeric_only=True).to_dict())
        scene_rows.append(row)
    scene_df = pd.DataFrame(scene_rows)
    scene_df.to_csv(os.path.join(base_dir, 'ablation_scene_targeted.csv'), index=False)
    plot_scene_ablation(scene_df, os.path.join(base_dir, 'ablation_scene_targeted.png'))

    # 3) replay ablation on Exp.5-style delayed stream
    replay_rows = []
    for variant, L in [('full', 4), ('no_replay', 0)]:
        vals = []
        for seed in [1, 2, 3]:
            events, vehicles, rel_maps, meta = mod.simulate_timeorg_stream(seed=seed, horizon_s=180, density='high')
            arr_events = mod.assign_arrival_times(events, bank['delays_ms'], seed=seed, scale=1.4)
            offline = sorted(arr_events, key=lambda e: (e.time_ms, e.eid))
            _, e2t_off, _ = mod.build_tracks(offline, rel_maps, 'full')
            online_order, info = mod.replay_processing_order(arr_events, delta_T_ms=350.0, L_scans=L)
            _, e2t_on, _ = mod.build_tracks(online_order, rel_maps, 'full')
            vals.append({
                'late_absorb_ratio': info['late_absorb_ratio'],
                'consistency': mod.local_transition_consistency(offline, e2t_off, e2t_on),
            })
        row = {'variant': variant}
        row.update(pd.DataFrame(vals).mean(numeric_only=True).to_dict())
        replay_rows.append(row)
    replay_df = pd.DataFrame(replay_rows)
    replay_df.to_csv(os.path.join(base_dir, 'ablation_replay_targeted.csv'), index=False)
    plot_replay_ablation(replay_df, os.path.join(base_dir, 'ablation_replay_targeted.png'))

    summary = {'tracking': tracking_df, 'scene': scene_df, 'replay': replay_df}
    with open(os.path.join(base_dir, 'ablation_targeted_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# Targeted ablation summary\n\n')
        f.write('## Tracking-side\n')
        f.write(tracking_df.round(4).to_markdown(index=False))
        f.write('\n\n## Scene-side\n')
        f.write(scene_df.round(4).to_markdown(index=False))
        f.write('\n\n## Replay-side\n')
        f.write(replay_df.round(4).to_markdown(index=False))

    print('tracking ablation')
    print(tracking_df.round(4))
    print('scene ablation')
    print(scene_df.round(4))
    print('replay ablation')
    print(replay_df.round(4))

if __name__ == '__main__':
    main()
