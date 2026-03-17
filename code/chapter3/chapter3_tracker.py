
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于第三章“节点状态自适应 + 场景概率判别”思路的可运行原型。
输入：streamIn*.txt（每行一条JSON记录）
输出：轨迹摘要CSV、轨迹点CSV、场景判别CSV

说明：
1. 原始 TwoLaneBeaconAssociationV2_1122 依赖大量工程域模型类，当前上传文件中并未包含，
   因而这里给出的是一个可独立运行的章节原型代码；
2. 代码保留了“状态预测-关联-概率更新-轨迹管理”的主骨架，但将原先固定的
   P_s / P_n / 700ms 规则改写为：
   - 节点级检测率/多检率后验估计；
   - 三状态场景概率判别（I1/I2/P）；
   - 统一关联评分；
3. 输出的轨迹适用于后续第四章轨迹段重构接口。
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from collections import defaultdict

DISS = 15.0                      # 相邻信标组距离，单位m
DEFAULT_SPEED = 22.0             # 缺少先验时的初始速度，约79km/h
MAX_SKIP = 4                     # 第三章建议的中等漏检最大容忍观测组数
BASE_GATE_MS = 650.0             # 单步门控阈值
SKIP_GATE_MS = 300.0             # 每多缺失一个观测组，门控扩张量
MIN_TRACK_POINTS = 4             # 进入确认状态的最小关联点数
PAIR_WINDOW_MS = 900.0           # 邻道同时触发的最大时间差
HMM_DT_SIGMA = 550.0             # 场景概率中时间差项的尺度

@dataclass
class Event:
    date: int
    devSort: int
    geomPeak: float
    timeLength: float
    lane: int
    rotate: str
    devEUI: str
    raw: dict

@dataclass
class Point:
    event: Event
    pred_time: float
    residual_ms: float
    node_reliability: float

@dataclass
class Track:
    tid: int
    lane: int
    points: list[Point] = field(default_factory=list)
    speed: float = DEFAULT_SPEED
    confidence: float = 0.5
    scene_hint: str = "unknown"

    @property
    def last(self) -> Event:
        return self.points[-1].event

    @property
    def last_dev(self) -> int:
        return self.last.devSort

    @property
    def last_time(self) -> int:
        return self.last.date

    @property
    def start(self) -> Event:
        return self.points[0].event

    def add(self, ev: Event, pred_time: float, residual_ms: float, node_r: float) -> None:
        self.points.append(Point(ev, pred_time, residual_ms, node_r))
        if len(self.points) >= 2:
            e1 = self.points[-2].event
            e2 = self.points[-1].event
            dd = max(1, e2.devSort - e1.devSort)
            dt_s = max(1.0, (e2.date - e1.date) / 1000.0)
            obs_speed = DISS * dd / dt_s
            self.speed = max(3.0, min(45.0, 0.70 * self.speed + 0.30 * obs_speed))
        c = math.exp(-(residual_ms ** 2) / (2.0 * (800.0 ** 2)))
        self.confidence = 0.80 * self.confidence + 0.20 * c

def load_events(path: str) -> list[Event]:
    events: list[Event] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events.append(
                Event(
                    date=int(d["date"]),
                    devSort=int(d["devSort"]),
                    geomPeak=float(d["geomPeak"]),
                    timeLength=float(d["timeLength"]),
                    lane=int(d["lane"]),
                    rotate=str(d.get("rotate", "")),
                    devEUI=str(d.get("devEUI", "")),
                    raw=d,
                )
            )
    events.sort(key=lambda e: (e.date, e.devSort, e.lane))
    return events

def predicted_time_ms(track: Track, dev_sort: int) -> float:
    dd = dev_sort - track.last_dev
    return track.last_time + (DISS * dd / max(track.speed, 3.0)) * 1000.0

def build_raw_tracks(events: list[Event], reliability: dict[tuple[int, int], float] | None = None) -> list[Track]:
    next_tid = 1
    all_tracks: list[Track] = []
    for lane in (1, 2):
        lane_events = [e for e in events if e.lane == lane]
        lane_tracks: list[Track] = []
        for ev in lane_events:
            best_track = None
            best_score = -1.0
            best_pred = float(ev.date)
            best_res = 0.0
            for tr in lane_tracks:
                dd = ev.devSort - tr.last_dev
                if dd <= 0 or dd > MAX_SKIP:
                    continue
                pred = predicted_time_ms(tr, ev.devSort)
                res = abs(ev.date - pred)
                gate = BASE_GATE_MS + SKIP_GATE_MS * (dd - 1)
                if res > gate:
                    continue
                node_r = 0.8 if reliability is None else reliability.get((ev.lane, ev.devSort), 0.8)
                last_gp = tr.last.geomPeak
                last_tl = tr.last.timeLength
                gdiff = abs(math.log((ev.geomPeak + 1.0) / (last_gp + 1.0)))
                tldiff = abs(ev.timeLength - last_tl) / max(80.0, last_tl)
                score = (
                    math.exp(-(res ** 2) / (2.0 * (gate / 1.8) ** 2))
                    * math.exp(-0.40 * gdiff)
                    * math.exp(-0.60 * tldiff)
                    * node_r
                    * (1.05 if dd == 1 else 1.0)
                )
                if score > best_score:
                    best_track = tr
                    best_score = score
                    best_pred = pred
                    best_res = res
            if best_track is None:
                tr = Track(tid=next_tid, lane=lane)
                next_tid += 1
                tr.add(ev, float(ev.date), 0.0, 0.8 if reliability is None else reliability.get((lane, ev.devSort), 0.8))
                lane_tracks.append(tr)
            else:
                best_track.add(ev, best_pred, best_res, 0.8 if reliability is None else reliability.get((lane, ev.devSort), 0.8))
        all_tracks.extend(lane_tracks)
    return all_tracks

def estimate_sensor_reliability(tracks: list[Track], events: list[Event]) -> dict[tuple[int, int], float]:
    alpha_d = defaultdict(lambda: 4.0)
    beta_d = defaultdict(lambda: 1.0)
    alpha_f = defaultdict(lambda: 1.0)
    beta_f = defaultdict(lambda: 6.0)

    for tr in tracks:
        for p in tr.points:
            alpha_d[(p.event.lane, p.event.devSort)] += 1.0
        for p1, p2 in zip(tr.points[:-1], tr.points[1:]):
            e1, e2 = p1.event, p2.event
            if e2.devSort > e1.devSort + 1:
                for miss_dev in range(e1.devSort + 1, e2.devSort):
                    beta_d[(tr.lane, miss_dev)] += 1.0
        if len(tr.points) <= 1:
            for p in tr.points:
                alpha_f[(p.event.lane, p.event.devSort)] += 1.0

    reliability: dict[tuple[int, int], float] = {}
    for lane in (1, 2):
        dev_sorts = sorted({e.devSort for e in events if e.lane == lane})
        for dev in dev_sorts:
            p_d = alpha_d[(lane, dev)] / (alpha_d[(lane, dev)] + beta_d[(lane, dev)])
            p_f = alpha_f[(lane, dev)] / (alpha_f[(lane, dev)] + beta_f[(lane, dev)])
            reliability[(lane, dev)] = max(0.15, min(0.99, p_d * (1.0 - p_f)))
    return reliability

def match_points_same_dev(t1: Track, t2: Track, max_dt_ms: float = PAIR_WINDOW_MS):
    by_dev = defaultdict(list)
    for p in t2.points:
        by_dev[p.event.devSort].append(p.event)
    pairs = []
    for p in t1.points:
        cands = by_dev.get(p.event.devSort, [])
        if not cands:
            continue
        best = min(cands, key=lambda e: abs(e.date - p.event.date))
        dt = abs(best.date - p.event.date)
        if dt <= max_dt_ms:
            pairs.append((p.event, best))
    return pairs

def scene_probability_for_pair(t1: Track, t2: Track):
    pairs = match_points_same_dev(t1, t2)
    if len(pairs) < 2:
        return None

    speed_diff = abs(t1.speed - t2.speed)
    pi = [0.33, 0.33, 0.34]  # I1, I2, P
    T = [
        [0.93, 0.02, 0.05],
        [0.02, 0.93, 0.05],
        [0.04, 0.04, 0.92],
    ]

    dts = []
    for e1, e2 in pairs:
        dt = abs(e1.date - e2.date)
        dts.append(dt)
        lr = math.log((e1.geomPeak + 1.0) / (e2.geomPeak + 1.0))
        like = [
            math.exp(-(dt ** 2) / (2.0 * HMM_DT_SIGMA ** 2)) * math.exp(-max(0.0, lr) / 1.8),
            math.exp(-(dt ** 2) / (2.0 * HMM_DT_SIGMA ** 2)) * math.exp(-max(0.0, -lr) / 1.8),
            math.exp(-((dt - 120.0) ** 2) / (2.0 * (250.0 ** 2))) * math.exp(-(speed_diff ** 2) / (2.0 * (5.0 ** 2))),
        ]
        pred = [
            T[0][0] * pi[0] + T[0][1] * pi[1] + T[0][2] * pi[2],
            T[1][0] * pi[0] + T[1][1] * pi[1] + T[1][2] * pi[2],
            T[2][0] * pi[0] + T[2][1] * pi[1] + T[2][2] * pi[2],
        ]
        pi = [pred[i] * like[i] for i in range(3)]
        s = sum(pi)
        if s > 0:
            pi = [x / s for x in pi]

    mean_dt = sum(dts) / len(dts)
    std_dt = (sum((x - mean_dt) ** 2 for x in dts) / len(dts)) ** 0.5
    strength1 = sum(math.log(p.event.geomPeak + 1.0) for p in t1.points) / len(t1.points)
    strength2 = sum(math.log(p.event.geomPeak + 1.0) for p in t2.points) / len(t2.points)

    # 全局修正：长期稳定小时间差 + 速度一致 更偏向真实并行
    Lp = math.exp(-(std_dt ** 2) / (2.0 * 180.0 ** 2)) * math.exp(-(speed_diff ** 2) / (2.0 * 5.0 ** 2)) * math.exp(-abs(strength1 - strength2) / 1.8)
    Li1 = math.exp(-(mean_dt ** 2) / (2.0 * 500.0 ** 2)) * math.exp(-max(0.0, strength2 - strength1) / 2.0)
    Li2 = math.exp(-(mean_dt ** 2) / (2.0 * 500.0 ** 2)) * math.exp(-max(0.0, strength1 - strength2) / 2.0)
    pi = [pi[0] * Li1, pi[1] * Li2, pi[2] * Lp]
    s = sum(pi)
    if s > 0:
        pi = [x / s for x in pi]

    label = ("I1", "I2", "P")[max(range(3), key=lambda i: pi[i])]
    return {
        "tid1": t1.tid,
        "tid2": t2.tid,
        "label": label,
        "p_I1": pi[0],
        "p_I2": pi[1],
        "p_P": pi[2],
        "pair_count": len(pairs),
        "mean_dt_ms": mean_dt,
        "std_dt_ms": std_dt,
        "speed_diff": speed_diff,
    }

def suppress_interference_by_scene(tracks: list[Track]):
    lane1 = [t for t in tracks if t.lane == 1 and len(t.points) >= 2]
    lane2 = [t for t in tracks if t.lane == 2 and len(t.points) >= 2]

    scene_rows = []
    suppressed = set()
    for t1 in lane1:
        for t2 in lane2:
            if t1.last_time < t2.start.date - 4000 or t2.last_time < t1.start.date - 4000:
                continue
            info = scene_probability_for_pair(t1, t2)
            if info is None:
                continue
            scene_rows.append(info)
            if info["label"] == "P":
                t1.scene_hint = "parallel"
                t2.scene_hint = "parallel"
                continue
            if info["label"] == "I1" and info["p_I1"] > 0.72 and info["pair_count"] >= 3:
                # lane2更像邻道干扰
                if len(t2.points) <= len(t1.points) + 2:
                    suppressed.add(t2.tid)
            if info["label"] == "I2" and info["p_I2"] > 0.72 and info["pair_count"] >= 3:
                # lane1更像邻道干扰
                if len(t1.points) <= len(t2.points) + 2:
                    suppressed.add(t1.tid)

    kept = [t for t in tracks if t.tid not in suppressed]
    return kept, scene_rows, suppressed

def merge_segments(tracks: list[Track], reliability: dict[tuple[int, int], float]) -> list[Track]:
    tracks = sorted(tracks, key=lambda t: (t.lane, t.start.date))
    out = []
    used = set()
    for i, tr in enumerate(tracks):
        if i in used:
            continue
        cur = tr
        for j in range(i + 1, len(tracks)):
            if j in used:
                continue
            nxt = tracks[j]
            if nxt.lane != cur.lane:
                continue
            dd = nxt.start.devSort - cur.last_dev
            dt = nxt.start.date - cur.last_time
            if dd <= 0 or dd > 3 or dt <= 0 or dt > 2500:
                continue
            pred = predicted_time_ms(cur, nxt.start.devSort)
            res = abs(nxt.start.date - pred)
            avg_r = 0.8
            if dd >= 1:
                rs = [reliability.get((cur.lane, dev), 0.8) for dev in range(cur.last_dev + 1, nxt.start.devSort + 1)]
                avg_r = sum(rs) / len(rs)
            gate = 700.0 + 300.0 * (dd - 1) + (1.0 - avg_r) * 400.0
            if res <= gate:
                for p in nxt.points:
                    pred2 = predicted_time_ms(cur, p.event.devSort)
                    cur.add(p.event, pred2, abs(p.event.date - pred2), reliability.get((cur.lane, p.event.devSort), 0.8))
                used.add(j)
        out.append(cur)
    return out

def final_filter(tracks: list[Track]) -> list[Track]:
    return [t for t in tracks if len(t.points) >= MIN_TRACK_POINTS]

def summarize_tracks(tracks: list[Track]) -> list[dict]:
    rows = []
    for t in tracks:
        mean_peak = sum(p.event.geomPeak for p in t.points) / len(t.points)
        rows.append({
            "track_id": t.tid,
            "sensor_lane": t.lane,
            "n_points": len(t.points),
            "start_time": t.start.date,
            "end_time": t.last_time,
            "start_devSort": t.start.devSort,
            "end_devSort": t.last_dev,
            "duration_s": round((t.last_time - t.start.date) / 1000.0, 3),
            "avg_speed_mps": round(t.speed, 3),
            "mean_geomPeak": round(mean_peak, 3),
            "scene_hint": t.scene_hint,
            "confidence": round(t.confidence, 4),
        })
    rows.sort(key=lambda r: (r["start_time"], r["sensor_lane"], r["track_id"]))
    return rows

def trajectory_points(tracks: list[Track]) -> list[dict]:
    rows = []
    for t in tracks:
        for idx, p in enumerate(t.points, start=1):
            rows.append({
                "track_id": t.tid,
                "sensor_lane": t.lane,
                "seq": idx,
                "date": p.event.date,
                "devSort": p.event.devSort,
                "geomPeak": p.event.geomPeak,
                "timeLength": p.event.timeLength,
                "rotate": p.event.rotate,
                "pred_time": round(p.pred_time, 3),
                "residual_ms": round(p.residual_ms, 3),
                "node_reliability": round(p.node_reliability, 4),
                "scene_hint": t.scene_hint,
            })
    rows.sort(key=lambda r: (r["track_id"], r["seq"]))
    return rows

def sensor_reliability_rows(reliability: dict[tuple[int, int], float]) -> list[dict]:
    rows = []
    for (lane, dev), r in sorted(reliability.items()):
        rows.append({"lane": lane, "devSort": dev, "reliability": round(r, 5)})
    return rows

def write_csv(path: str, rows: list[dict]):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def run_pipeline(events: list[Event]):
    raw_tracks = build_raw_tracks(events, reliability=None)
    reliability = estimate_sensor_reliability(raw_tracks, events)
    raw_tracks = build_raw_tracks(events, reliability=reliability)
    reliability = estimate_sensor_reliability(raw_tracks, events)
    kept_tracks, scene_rows, suppressed = suppress_interference_by_scene(raw_tracks)
    merged_tracks = merge_segments(kept_tracks, reliability)
    final_tracks = final_filter(merged_tracks)
    return raw_tracks, final_tracks, reliability, scene_rows, suppressed

def process_file(input_path: str, output_dir: str):
    events = load_events(input_path)
    raw_tracks, final_tracks, reliability, scene_rows, suppressed = run_pipeline(events)
    name = os.path.splitext(os.path.basename(input_path))[0]

    summary_rows = summarize_tracks(final_tracks)
    point_rows = trajectory_points(final_tracks)
    rel_rows = sensor_reliability_rows(reliability)

    write_csv(os.path.join(output_dir, f"{name}_track_summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, f"{name}_track_points.csv"), point_rows)
    write_csv(os.path.join(output_dir, f"{name}_sensor_reliability.csv"), rel_rows)
    write_csv(os.path.join(output_dir, f"{name}_scene_pairs.csv"), scene_rows)

    stats = {
        "file": input_path,
        "n_events": len(events),
        "n_raw_tracks": len(raw_tracks),
        "n_final_tracks": len(final_tracks),
        "n_suppressed_tracks": len(suppressed),
        "min_track_points": MIN_TRACK_POINTS,
        "max_skip_supported": MAX_SKIP,
    }
    return stats

def main():
    ap = argparse.ArgumentParser(description="双车道地磁事件流在线轨迹生成（第三章原型）")
    ap.add_argument("inputs", nargs="+", help="输入的streamIn文本文件，可同时传入多个")
    ap.add_argument("-o", "--output-dir", default="./tracker_output", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stat_rows = []
    for input_path in args.inputs:
        stat_rows.append(process_file(input_path, args.output_dir))
    write_csv(os.path.join(args.output_dir, "run_summary.csv"), stat_rows)
    print("处理完成，输出目录：", os.path.abspath(args.output_dir))
    for row in stat_rows:
        print(row)

if __name__ == "__main__":
    main()
