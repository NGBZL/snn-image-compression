# -*- coding: utf-8 -*-
"""make_wiring_figure.py — 生成"参考接线方式"对比图

左图：两种接线的数据流示意（输出端加偏置 vs 输入端拼接）
右图：实测的 残差/锚图 比值。比值 < 1 才意味着"用参考真的省了比特"。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

# ---------------------------------------------------------------- 左：接线示意
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec="#333", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x1, y1, x2, y2, color="#333", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=1.4, color=color,
                                 connectionstyle="arc3,rad=0"))


ax.text(5, 9.55, "旧：参考加在编码器输出端", ha="center", fontsize=12, fontweight="bold")
box(0.3, 7.2, 1.5, 0.9, "x (img)", "#dbeafe")
box(2.2, 7.2, 2.0, 0.9, "conv1-4", "#e5e7eb")
box(4.6, 7.2, 1.2, 0.9, "GDN", "#e5e7eb")
box(6.2, 7.2, 1.1, 0.9, "⊕", "#fecaca")
box(7.7, 7.2, 1.8, 0.9, "残差 r", "#fde68a")
box(6.2, 5.4, 1.1, 0.9, "ref_proj", "#fee2e2")
box(7.7, 5.4, 1.8, 0.9, "参考 y_j", "#dbeafe")
for a, b in [((1.8, 7.65), (2.2, 7.65)), ((4.2, 7.65), (4.6, 7.65)),
             ((5.8, 7.65), (6.2, 7.65)), ((7.3, 7.65), (7.7, 7.65)),
             ((8.6, 6.3), (7.0, 5.4 + 0.9))]:
    arrow(a[0], a[1], b[0], b[1])
arrow(6.75, 6.3, 6.75, 7.2, color="#dc2626")
ax.text(6.95, 6.75, "加法", fontsize=8, color="#dc2626")
ax.text(5.0, 4.55,
        "同一张图时  r = a + y_j = 2a\n残差恒为锚图两倍 —— 起点比不用参考还差",
        ha="center", fontsize=9.5, color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef2f2", ec="#fca5a5"))

ax.text(5, 3.35, "新：参考拼接到输入端", ha="center", fontsize=12, fontweight="bold")
box(0.3, 1.1, 1.5, 0.9, "x (img)", "#dbeafe")
box(0.3, 2.3, 1.5, 0.9, "参考 y_j\n上采样", "#dbeafe")
box(2.2, 1.7, 1.5, 0.9, "cat\n3+C 通道", "#dcfce7")
box(4.0, 1.7, 2.0, 0.9, "conv1-4", "#e5e7eb")
box(6.3, 1.7, 1.2, 0.9, "GDN", "#e5e7eb")
box(7.8, 1.7, 1.7, 0.9, "残差 r", "#fde68a")
for a, b in [((1.8, 1.55), (2.2, 2.0)), ((1.8, 2.75), (2.2, 2.3)),
             ((3.7, 2.15), (4.0, 2.15)), ((6.0, 2.15), (6.3, 2.15)),
             ((7.5, 2.15), (7.8, 2.15))]:
    arrow(a[0], a[1], b[0], b[1])
ax.text(5.0, 0.35,
        "编码器从 (x_i, y_j) 联合算特征，直接学“两张图差在哪”\n不需要在输出端做精确抵消",
        ha="center", fontsize=9.5, color="#15803d",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0fdf4", ec="#86efac"))

# ---------------------------------------------------------------- 右：实测比值
ax2 = fig.add_subplot(gs[0, 1])
labels, vals, colors = [], [], []

def add(label, path, key, color):
    if os.path.isfile(path):
        d = json.load(open(path, encoding="utf-8"))
        if key in d:
            labels.append(label); vals.append(d[key]); colors.append(color)

# 残差/锚图 比值（>1 = 用参考反而更费比特）
add("旧接线\nstar 训练", "anchor_allpairs.json", "ratio_placeholder", "#fca5a5")
# 从各次评估日志里硬编码实测值（这些数字来自日志，JSON 里没存比值）
measured = [("旧接线・64²\nstar", 1.035, "#f87171"),
            ("旧接线・64²\n全对全", 1.182, "#ef4444"),
            ("新接线・64²\n60ep", 1.022, "#fbbf24"),
            ("新接线・96²\n60ep", 1.033, "#4ade80"),
            ("新接线・256²\n130ep (批量)", 1.070, "#22c55e")]

ax2.axhline(1.0, color="#16a34a", ls="--", lw=2)
ax2.text(4.45, 1.005, "1.0 = 打平", fontsize=9,
         color="#16a34a", ha="right", va="bottom")
xs = list(range(len(measured)))
ax2.bar(xs, [m[1] for m in measured], color=[m[2] for m in measured],
        width=0.62, edgecolor="#333")
for x, m in zip(xs, measured):
    ax2.text(x, m[1] + 0.016, f"{m[1]:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(xs)
ax2.set_xticklabels([m[0] for m in measured], fontsize=8)
ax2.set_ylabel("残差比特 / 锚图比特")
ax2.set_title("用参考到底省不省比特？  比值 < 1 才算有效", fontsize=12)
ax2.set_ylim(0, 1.42)
ax2.grid(axis="y", alpha=.3)
ax2.text(0.02, 0.965,
         "MST 相对独立编码的节省率随基座增强单调上升：\n"
         "  +0.9%  ->  +2.4%  ->  +4.6%  ->  +6.7%\n"
         "但 256² 那版的均值比值反而回升到 1.07 ——\n"
         "因为模型 7/8 的训练样本是残差，纯锚图编码成了分布外",
         transform=ax2.transAxes, fontsize=8.0, va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f0fdf4", ec="#86efac"))

plt.suptitle("锚图参考接线方式：输出端加偏置  vs  输入端拼接", fontsize=13.5, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("wiring_compare.png", dpi=125)
print("已保存 wiring_compare.png")
