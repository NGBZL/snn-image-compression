# -*- coding: utf-8 -*-
"""show_effect.py — 展示当前模型的重建效果，训练集 / 测试集各一组

四行网格：训练集原图 / 训练集重建 / 测试集原图 / 测试集重建
直观暴露过拟合程度。

    python show_effect.py --weights anchor_v5.pth --n 6
"""
from __future__ import annotations

import argparse
import math
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from anchor import SAE_Anchor
from sae_model import human_bytes
from snntorch import utils

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def to_np(t):
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="anchor_v5.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="effect_train_test.png")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                       image_size=obj["image_size"], latent_mode="gaussian",
                       use_hyperprior=obj.get("use_hyperprior", True),
                       z_channels=obj.get("z_channels", 48),
                       ref_wiring=obj.get("ref_wiring", "inp")).to(dev)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True
    print(f"权重 {a.weights}   epoch={obj.get('epoch','?')}   "
          f"{model.image_size}x{model.image_size}  C={model.latent_channels}  "
          f"T={model.num_steps}  hyperprior={model.use_hyperprior}  "
          f"接线={model.ref_wiring}")

    tf = transforms.Compose([transforms.Resize(model.image_size),
                             transforms.CenterCrop(model.image_size),
                             transforms.ToTensor()])
    tr = datasets.Flowers102(a.data_dir, split="train", download=False, transform=tf)
    te = datasets.Flowers102(a.data_dir, split="test", download=False, transform=tf)

    rows, stats = [], {}
    for name, ds in [("训练集", tr), ("测试集", te)]:
        orig, recs, psnrs, byts = [], [], [], []
        with torch.no_grad():
            for i in range(a.n):
                x = ds[i][0].to(dev)
                utils.reset(model)
                _r, rate, rec = model(x.unsqueeze(0), ref=None,
                                      checkpoints={model.num_steps})
                r = rec[model.num_steps].clamp(0, 1)[0]
                orig.append(to_np(x))
                recs.append(to_np(r))
                mse = torch.nn.functional.mse_loss(r, x).item()
                psnrs.append(10 * math.log10(1.0 / max(1e-12, mse)))
                byts.append(rate["bits_y"].sum().item() + rate["bits_z"].item())
        stats[name] = dict(psnr=float(np.mean(psnrs)), mse=10 ** (-np.mean(psnrs) / 10),
                           byte=float(np.mean(byts)) / 8)
        rows.append((orig, recs, name, psnrs))

    print(f"\n{'集合':<8}{'平均 PSNR':>12}{'等效 MSE':>14}{'码率':>12}")
    for k, v in stats.items():
        print(f"{k:<8}{v['psnr']:>10.2f} dB{v['mse']:>14.5f}{v['byte']:>10.0f} B")
    gap = stats['训练集']['mse'] / max(1e-9, stats['测试集']['mse'])
    print(f"\n训练集/测试集 MSE 之比 = {gap:.1f} 倍   "
          f"({'过拟合明显' if gap > 3 else '泛化尚可'})")

    # ---------------- 画图 ----------------
    n = a.n
    fig, axes = plt.subplots(4, n, figsize=(2.15 * n, 9.6))
    titles = ["训练集 · 原图", "训练集 · 重建", "测试集 · 原图", "测试集 · 重建"]
    order = [rows[0][0], rows[0][1], rows[1][0], rows[1][1]]
    ps = [None, rows[0][3], None, rows[1][3]]
    for r in range(4):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(order[r][c]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(titles[r], fontsize=11, rotation=90, labelpad=10)
            if ps[r] is not None:
                ax.set_xlabel(f"{ps[r][c]:.1f} dB", fontsize=8)
    fig.suptitle(
        f"锚图压缩模型重建效果（{model.image_size}×{model.image_size}, 钥匙 "
        f"{stats['测试集']['byte']:.0f}-{stats['训练集']['byte']:.0f} 字节/张）   "
        f"训练集 {stats['训练集']['psnr']:.1f} dB  vs  测试集 {stats['测试集']['psnr']:.1f} dB"
        f"（MSE 差 {gap:.1f} 倍）", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(a.out, dpi=115)
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
