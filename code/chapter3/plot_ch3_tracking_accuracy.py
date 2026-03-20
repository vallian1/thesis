#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只根据聚合 CSV 重画第3章跟踪准确度对比图。"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'ch3_tracking_accuracy_aggregate.csv')
OUT_DIR = r'E:\MyLunWen\xduts-main\code\chapter3\experiment1\exp1_results'
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    methods = ['JPDA', 'TOMHT', 'Proposed']
    scenarios = ['场景A', '场景B', '场景C']
    labels = ['A', 'B', 'C']

    width = 0.24
    x = np.arange(len(scenarios))
    
    fig1, ax1 = plt.subplots(figsize=(5.6, 4.3), dpi=180)
    for idx, method in enumerate(methods):
        sub = df[df['method'] == method].set_index('scenario').loc[scenarios]
        ax1.bar(x + (idx - 1) * width, sub['RMSE_p_mean'].values, width=width, label=method)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('位置 RMSE / m')
    # ax1.set_title('(a) Position RMSE')
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    
    out_path1 = os.path.join(OUT_DIR, 'ch3_tracking_accuracy_rmse_p.png')
    fig1.savefig(out_path1, bbox_inches='tight')
    plt.close(fig1)
    print(out_path1)
    
    fig2, ax2 = plt.subplots(figsize=(5.6, 4.3), dpi=180)
    for idx, method in enumerate(methods):
        sub = df[df['method'] == method].set_index('scenario').loc[scenarios]
        ax2.bar(x + (idx - 1) * width, sub['RMSE_v_mean'].values, width=width, label=method)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('速度 RMSE / (m/s)')
    # ax2.set_title('(b) Velocity RMSE')
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(frameon=False, fontsize=9)
    fig2.tight_layout()
    
    out_path2 = os.path.join(OUT_DIR, 'ch3_tracking_accuracy_rmse_v.png')
    fig2.savefig(out_path2, bbox_inches='tight')
    plt.close(fig2)
    print(out_path2)


if __name__ == '__main__':
    main()
