#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将旧版图的上半部分与当前图的下半部分拼接成一张新的实验一后验曲线图。

用途：
- 上半部分使用 exp1_node_posterior_curve_v1.png 中更理想的检测概率结果；
- 下半部分使用当前重新跑出的虚警概率结果；
- 不依赖上半部分原始 csv，只依赖两张已有 png 图。

默认输入：
  BASE_DIR = 'E:/MyLunWen/xduts-main/code/chapter3/experiment1/exp1_results'

parser.add_argument('--top-image', default= os.path.join(BASE_DIR, 'exp1_node_posterior_curve_v1.png'))
parser.add_argument('--bottom-image', default= os.path.join(BASE_DIR, 'exp1_node_posterior_curve.png'))
parser.add_argument('--out-image', default= os.path.join(BASE_DIR, 'exp1_node_posterior_curve_topv1_bottomcurrent.png'))
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
import argparse


@dataclass
class SplitResult:
    split_row: int
    top_crop: Tuple[int, int, int, int]
    bottom_crop: Tuple[int, int, int, int]


def find_split_row(img: Image.Image, search_lo: float = 0.40, search_hi: float = 0.60) -> int:
    """在图像中部搜索最亮的一行，作为上下子图之间的分隔行。"""
    gray = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    row_mean = gray.mean(axis=1)
    h = gray.shape[0]
    lo = max(0, int(h * search_lo))
    hi = min(h, int(h * search_hi))
    if hi <= lo:
        return h // 2
    return lo + int(np.argmax(row_mean[lo:hi]))


def crop_top_panel(img: Image.Image, split_row: int, pad_bottom: int = 6) -> Image.Image:
    w, h = img.size
    y2 = max(1, split_row - pad_bottom)
    return img.crop((0, 0, w, y2))


def crop_bottom_panel(img: Image.Image, split_row: int, pad_top: int = 6) -> Image.Image:
    w, h = img.size
    y1 = min(h - 1, split_row + pad_top)
    return img.crop((0, y1, w, h))


def resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.size[0] == target_w:
        return img
    w, h = img.size
    new_h = int(round(h * target_w / w))
    return img.resize((target_w, new_h), Image.LANCZOS)


def compose(top_img_path: str, bottom_img_path: str, out_path: str) -> None:
    top_src = Image.open(top_img_path).convert('RGB')
    bottom_src = Image.open(bottom_img_path).convert('RGB')

    split_top = find_split_row(top_src)
    split_bottom = find_split_row(bottom_src)

    top_panel = crop_top_panel(top_src, split_top)
    bottom_panel = crop_bottom_panel(bottom_src, split_bottom)

    target_w = max(top_panel.size[0], bottom_panel.size[0])
    top_panel = resize_to_width(top_panel, target_w)
    bottom_panel = resize_to_width(bottom_panel, target_w)

    gap = 0
    out_h = top_panel.size[1] + gap + bottom_panel.size[1]
    canvas = Image.new('RGB', (target_w, out_h), color='white')
    canvas.paste(top_panel, (0, 0))
    canvas.paste(bottom_panel, (0, top_panel.size[1] + gap))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f'top source   : {top_img_path}')
    print(f'bottom source: {bottom_img_path}')
    print(f'top split row: {split_top}')
    print(f'bot split row: {split_bottom}')
    print(f'output saved : {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-image', default='/mnt/data/exp1_node_posterior_curve_v1.png')
    parser.add_argument('--bottom-image', default='/mnt/data/exp1_output/exp1_node_posterior_curve.png')
    parser.add_argument('--out-image', default='/mnt/data/exp1_output/exp1_node_posterior_curve_topv1_bottomcurrent.png')
    args = parser.parse_args()
    compose(args.top_image, args.bottom_image, args.out_image)
