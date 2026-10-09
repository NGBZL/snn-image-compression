# -*- coding: utf-8 -*-
"""
visualize.py —— 生成对比图，回答"我们离 JPEG 有多远"

产出两张图：
  1. compare_snn_vs_jpeg.png   三排对比：原图 / SNN 重建 / 同码率 JPEG
                               三排用的是**完全相同**的字节预算（我们的钥匙大小），这才公平
  2. rate_distortion.png       JPEG 的率失真曲线 + 我们的单个工作点
                               要"战胜 JPEG"，我们的点必须落在 JPEG 曲线上方

用法
    python visualize.py --weights sae_cifar_best.pth --n 8
    python visualize.py --weights sae_cifar.pth --n 8 --out-prefix final
"""

from __future__ import annotations

import argparse
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from sae_model import (
    SSIM, bpp, build_model_from_checkpoint, human_bytes, jpeg_size_and_quality, ms_psnr,
)
from snntorch import utils

# 中文字体（Windows 上这些都有；找不到就退回默认，只是标签变方块，不影响数据）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="sae_cifar_best.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n", type=int, default=8, help="对比图用几张")
    p.add_argument("--rd-n", type=int, default=24, help="率失真曲线用几张（越多越平滑）")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-prefix", default="")
    return p.parse_args()


def pick_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_np(t):
    """[3,H,W] in [0,1] -> [H,W,3] uint8"""
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def main():
    args = parse_args()
    dev = pick_device(args.device)
    prefix = args.out_prefix

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
    ds = datasets.Flowers102(args.data_dir, split="test", download=False, transform=tf)

    model, cfg = build_model_from_checkpoint(args.weights, device=dev, strict=True)
    model.eval()
    ssim_fn = SSIM().to(dev)
    key_bytes = model.key_bytes
    print(f"权重 {args.weights}  image_size={model.image_size} C={model.latent_channels} "
          f"T={model.num_steps}")
    print(f"钥匙 {key_bytes} 字节 = {bpp(key_bytes, model.image_size):.3f} bit/像素\n")

    # ---------------------------------------------------------------- 1. 三排对比图
    n = args.n
    orig, snn, jpg = [], [], []
    snn_ps, jpg_ps = [], []
    for i in range(n):
        x = ds[i][0]
        with torch.no_grad():
            utils.reset(model)
            r = model.decode(model.encode(x.unsqueeze(0).to(dev))).clamp(0, 1)[0]
        jb, q, jr, matched = jpeg_size_and_quality(x, key_bytes)
        orig.append(to_np(x)); snn.append(to_np(r)); jpg.append(to_np(jr))
        snn_ps.append(ms_psnr(r.unsqueeze(0).cpu(), x.unsqueeze(0)).item())
        jpg_ps.append(ms_psnr(jr.unsqueeze(0), x.unsqueeze(0)).item())

    print(f"三排对比（每张的字节预算都是 {key_bytes}）:")
    print(f"  SNN  平均 PSNR {np.mean(snn_ps):.2f} dB")
    print(f"  JPEG 平均 PSNR {np.mean(jpg_ps):.2f} dB   (质量因子 {q})")
    print(f"  差值 {np.mean(snn_ps) - np.mean(jpg_ps):+.2f} dB\n")

    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 7.0))
    row_titles = [
        "Original",
        f"SNN  ({key_bytes} B, {np.mean(snn_ps):.1f} dB)",
        f"JPEG ({jb} B, {np.mean(jpg_ps):.1f} dB)",
    ]
    for row, (imgs, title) in enumerate(zip([orig, snn, jpg], row_titles)):
        per_img = [None, snn_ps, jpg_ps][row]
        for c in range(n):
            ax = axes[row, c]
            ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=11, rotation=90, labelpad=12)
            if per_img is not None:
                ax.set_xlabel(f"{per_img[c]:.1f} dB", fontsize=8)
    fig.suptitle(f"SNN vs JPEG at matched rate ({key_bytes} bytes = "
                 f"{bpp(key_bytes, model.image_size):.3f} bpp)", fontsize=13)
    plt.tight_layout()
    f1 = f"{prefix}compare_snn_vs_jpeg.png" if prefix else "compare_snn_vs_jpeg.png"
    plt.savefig(f1, dpi=110); plt.close()
    print(f"已保存 {f1}")

    # ---------------------------------------------------------------- 2. 率失真曲线
    m = min(args.rd_n, len(ds))
    qualities = [3, 5, 8, 12, 16, 22, 30, 40, 50, 60, 70, 80, 90, 95]
    acc = {q: {"b": 0.0, "p": 0.0} for q in qualities}
    snn_b, snn_p = 0.0, 0.0
    for i in range(m):
        x = ds[i][0]
        with torch.no_grad():
            utils.reset(model)
            r = model.decode(model.encode(x.unsqueeze(0).to(dev))).clamp(0, 1)[0]
        snn_b += key_bytes
        snn_p += ms_psnr(r.unsqueeze(0).cpu(), x.unsqueeze(0)).item()
        for q in qualities:
            jb, _, jr, _ = jpeg_size_and_quality(x, 10**9, quality_lo=q, quality_hi=q)
            acc[q]["b"] += jb
            acc[q]["p"] += ms_psnr(jr.unsqueeze(0), x.unsqueeze(0)).item()
    snn_b /= m; snn_p /= m
    jb_pts = [bpp(acc[q]["b"] / m, 256) for q in qualities]
    jp_pts = [acc[q]["p"] / m for q in qualities]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(jb_pts, jp_pts, "o-", color="tab:blue", label="JPEG (标准库, 扫质量因子)", lw=2)
    ax.plot([bpp(snn_b, 256)], [snn_p], "*", color="tab:red", ms=22,
            label=f"SNN SAE (ours)  {snn_p:.1f} dB @ {bpp(snn_b,256):.3f} bpp")
    # 标出同码率对比
    j_at = np.interp(bpp(snn_b, 256), jb_pts, jp_pts)
    ax.annotate("", xy=(bpp(snn_b, 256), j_at), xytext=(bpp(snn_b, 256), snn_p),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(bpp(snn_b, 256) * 1.15, (j_at + snn_p) / 2, f"{j_at - snn_p:.1f} dB\ngap",
            color="gray", fontsize=10, va="center")
    ax.set_xlabel("Rate (bits per pixel)"); ax.set_ylabel("MS-PSNR (dB)")
    ax.set_title(f"Rate-Distortion: SNN autoencoder vs JPEG  (Flowers102 test, {m} images)")
    ax.grid(alpha=.3); ax.legend(loc="lower right")
    plt.tight_layout()
    f2 = f"{prefix}rate_distortion.png" if prefix else "rate_distortion.png"
    plt.savefig(f2, dpi=130); plt.close()
    print(f"已保存 {f2}")
    print(f"\n率失真: 我们的点 ({bpp(snn_b,256):.3f} bpp, {snn_p:.1f} dB); "
          f"JPEG 在同码率处 {j_at:.1f} dB -> 落后 {j_at - snn_p:.1f} dB")
    print(f"        JPEG 要达到我们这 {snn_p:.1f} dB, 只需 "
          f"{np.interp(snn_p, jp_pts, jb_pts):.3f} bpp "
          f"-> 比我们省 {bpp(snn_b,256)/max(1e-9, np.interp(snn_p, jp_pts, jb_pts)):.1f} 倍")


if __name__ == "__main__":
    main()
