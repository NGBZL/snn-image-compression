# -*- coding: utf-8 -*-
"""rate_control.py — 不重训模型，纯推理期调码率

原理
    潜层是被量化的连续值。推理时把量化步长放大 s 倍：
        y_hat = round(y / (s*step)) * (s*step)
    符号的取值范围随之缩小 s 倍。**同时把先验的 sigma 也除以 s**
        sigma_eff = sigma / s
    这样熵模型仍然精确匹配新的符号分布，算术编码不会因为模型失配而浪费比特。
    （不缩放 sigma 也能降码率，但会白扔一部分效率。）

    另一个旋钮是截断时间步：只用前 k 步解码，码率按 k/T 线性下降。

两者都不需要重新训练，是纯粹的推理期权衡。

用法
    python rate_control.py --weights anchor_v5.pth --n 8
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
import torch.nn.functional as F
from torchvision import datasets, transforms

from anchor import SAE_Anchor
from sae_rd import encode_latent_bytes, encode_z_bytes
from snntorch import utils

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def measure(model, x, s: float, steps: int, ch: int = 0, scale_sigma: bool = True,
            want_img: bool = False):
    """返回 (实际字节数, PSNR[, 重建图])。s=步长倍数, steps=用前几步, ch=只传前几通道。"""
    orig_log_step = model.quant.log_step.data.clone()
    orig_log_sigma = model.prior.log_sigma.data.clone()
    orig_mu = model.prior.mu.data.clone()
    C = model.latent_channels
    ch = ch or C
    try:
        model.quant.log_step.data = torch.tensor(math.log(float(s)))
        if scale_sigma and s != 1.0:
            model.prior.log_sigma.data = orig_log_sigma - math.log(float(s))
        utils.reset(model)
        y_hat, rate, rec = model(x.unsqueeze(0), ref=None,
                                 checkpoints={model.num_steps})
        # 通道截断要放在 forward **之后**：forward 内部要用完整的先验算 32 通道，
        # 提前切掉会让 s(16) * extra(32) 形状不匹配。
        if ch < C:
            model.prior.log_sigma.data = model.prior.log_sigma.data[:, :ch]
            model.prior.mu.data = model.prior.mu.data[:, :ch]
        # 只传前 ch 个通道：其余通道在解码端置零
        y_full = y_hat.clone()
        if ch < C:
            y_full[:, :, ch:] = 0.0
        y_use = y_hat[:, :steps, :ch]
        es = rate.get("extra_sigma")
        if es is not None:
            es = es[:, :steps, :ch]
        b_y, _ns, _p, _y = encode_latent_bytes(y_use, model.quant.step, model.prior,
                                               extra_sigma=es)
        total = b_y
        n_zb = 0
        if rate.get("z_hat") is not None:
            # z 的形状是 [B*T, Cz, h/4, w/4]。截断时间步时必须**同时截断 z**，
            # 否则 z 仍按完整 T 步传输 —— 实测它占了大头（~7 kB），
            # 会让所有只作用在 y 上的降码率手段全部撞在同一个地板上。
            z_use = rate["z_hat"][:steps * x.shape[0]]
            b_z, _ = encode_z_bytes(z_use, rate["z_step"], model.hyperprior.z_prior)
            total += b_z
            n_zb = b_z / 8.0
        # 用（可能被截断/置零的）潜层重新解码
        if steps < model.num_steps or ch < C:
            utils.reset(model)
            r = model.decode(y_full[:, :steps], checkpoints={steps})[steps].clamp(0, 1)[0]
        else:
            r = rec[model.num_steps].clamp(0, 1)[0]
        psnr = 10 * math.log10(1.0 / max(1e-12, F.mse_loss(r, x).item()))
        return (total / 8.0, psnr, r.cpu()) if want_img else (total / 8.0, psnr)
    finally:
        model.quant.log_step.data = orig_log_step
        model.prior.log_sigma.data = orig_log_sigma
        model.prior.mu.data = orig_mu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="anchor_v5.pth")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--target-kb", type=float, default=5.0)
    p.add_argument("--out", default="rate_control.png")
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

    tf = transforms.Compose([transforms.Resize(model.image_size),
                             transforms.CenterCrop(model.image_size),
                             transforms.ToTensor()])
    te = datasets.Flowers102(a.data_dir, split="test", download=False, transform=tf)
    xs = [te[i][0] for i in range(a.n)]
    T = model.num_steps
    C = model.latent_channels
    print(f"模型 {a.weights}  {model.image_size}x{model.image_size}  C={model.latent_channels}"
          f"  T={T}  hyperprior={model.use_hyperprior}")
    print(f"测试 {a.n} 张，目标 {a.target_kb} kB\n")

    # 扫两维：步长倍数 x 时间步
    step_mults = [1, 1.5, 2, 3, 4, 6, 8, 12, 16]
    step_choices = sorted({T, max(1, T // 2), max(1, T // 3), max(1, T // 5), 1})

    print(f"{'步长倍数':>8}{'时间步':>7}{'字节':>10}{'PSNR':>10}   备注")
    print("-" * 56)
    rows = []
    for s in step_mults:
        for st in [T]:
            b = pp = 0.0
            for x in xs:
                bb, pv = measure(model, x.to(dev), s, st)
                b += bb; pp += pv
            b /= a.n; pp /= a.n
            rows.append(dict(s=s, steps=st, bytes=b, psnr=pp))
            mark = ""
            if abs(b - a.target_kb * 1024) < 1500:
                mark = f"<-- 接近 {a.target_kb:.0f} kB"
            print(f"{s:>8.1f}{st:>7d}{b:>10.0f}{pp:>9.2f}dB   {mark}")

    print("\n--- 只截断时间步（步长倍数=1）---")
    rows_t = []
    for st in step_choices:
        b = pp = 0.0
        for x in xs:
            bb, pv = measure(model, x.to(dev), 1, st)
            b += bb; pp += pv
        b /= a.n; pp /= a.n
        rows_t.append(dict(s=1, steps=st, bytes=b, psnr=pp))
        print(f"{1:>8.1f}{st:>7d}{b:>10.0f}{pp:>9.2f}dB")

    # 找最接近目标字节数的组合
    allr = rows + rows_t
    best = min(allr, key=lambda r: abs(r["bytes"] - a.target_kb * 1024))
    base = [r for r in allr if r["s"] == 1 and r["steps"] == T][0]
    print("\n" + "=" * 56)
    print(f"基线      : {base['bytes']:.0f} B   {base['psnr']:.2f} dB")
    print(f"最接近 {a.target_kb:.0f} kB: {best['bytes']:.0f} B   {best['psnr']:.2f} dB   "
          f"(步长x{best['s']:g}, 用前 {best['steps']} 步)")

    # ---------- 冲目标码率：联合截断时间步 + 通道 ----------
    print(f"\n--- 联合截断（时间步 x 通道），目标 {a.target_kb:.0f} kB ---")
    cands = []
    for st in sorted({T, max(1, T // 2), max(1, T // 3), 2, 1}):
        for ch in sorted({C, C // 2, C // 4, C // 8, max(1, C // 16)}, reverse=True):
            if st == T and ch == C:
                continue
            b = pp = 0.0
            for x in xs:
                bb, pv = measure(model, x.to(dev), 1, st, ch)
                b += bb; pp += pv
            b /= a.n; pp /= a.n
            cands.append(dict(s=1, steps=st, ch=ch, bytes=b, psnr=pp))
            print(f"   步{st:>2d} 通道{ch:>3d}: {b:>8.0f} B  {pp:>6.2f} dB")

    pool = allr + cands
    hit = min(pool, key=lambda r: abs(r["bytes"] - a.target_kb * 1024))
    print("\n" + "=" * 56)
    print(f"★ 最接近 {a.target_kb:.0f} kB 的纯推理配置：")
    print(f"   字节 {hit['bytes']:.0f} B ({hit['bytes']/1024:.2f} kB)   "
          f"PSNR {hit['psnr']:.2f} dB")
    print(f"   配置：时间步 {hit['steps']}/{T}，通道 {hit.get('ch', C)}/{C}")
    print(f"   相对原工作点：压缩 {base['bytes']/max(1,hit['bytes']):.1f} 倍，"
          f"质量 {base['psnr']-hit['psnr']:.2f} dB")

    # 生成该配置下的重建对比图
    origs, recs = [], []
    for i in range(min(5, a.n)):
        x = xs[i].to(dev)
        _b, _p, r = measure(model, x, 1, hit["steps"], hit.get("ch", C), want_img=True)
        origs.append((x.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        recs.append((r.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    fig, axes = plt.subplots(2, len(origs), figsize=(2.3 * len(origs), 5.4))
    for c in range(len(origs)):
        axes[0, c].imshow(origs[c]); axes[1, c].imshow(recs[c])
        for r_ in (0, 1):
            axes[r_, c].set_xticks([]); axes[r_, c].set_yticks([])
    axes[0, 0].set_ylabel("原图", fontsize=11)
    axes[1, 0].set_ylabel(f"{hit['bytes']/1024:.1f} kB\n{hit['psnr']:.1f} dB", fontsize=11)
    fig.suptitle(f"纯推理降码率结果：{hit['bytes']/1024:.2f} kB/张，{hit['psnr']:.1f} dB"
                 f"（时间步 {hit['steps']}/{T}，通道 {hit.get('ch', C)}/{C}）", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("rate_control_target.png", dpi=120)
    print("   已保存 rate_control_target.png")

    # 画 RD 曲线
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs_ = [r["bytes"] / 1024 for r in rows]
    ys_ = [r["psnr"] for r in rows]
    ax.plot(xs_, ys_, "o-", color="tab:red", lw=2, label="放大步长（无效）")
    if rows_t:
        ax.plot([r["bytes"] / 1024 for r in rows_t], [r["psnr"] for r in rows_t],
                "s--", color="tab:orange", label="截断时间步")
    if cands:
        ax.plot([r["bytes"] / 1024 for r in cands], [r["psnr"] for r in cands],
                "^:", color="tab:purple", label="截断时间步 + 通道")
    ax.axvline(a.target_kb, color="tab:green", ls=":", lw=2)
    ax.text(a.target_kb * 1.05, min(ys_), f"{a.target_kb:.0f} kB", color="tab:green")
    b0 = [r for r in allr if r["s"] == 1 and r["steps"] == T][0]
    ax.plot([b0["bytes"] / 1024], [b0["psnr"]], "k*", ms=20, label="原模型工作点")
    ax.plot([hit["bytes"] / 1024], [hit["psnr"]], "g*", ms=18,
            label=f"目标 {hit['bytes']/1024:.1f} kB / {hit['psnr']:.1f} dB")
    ax.set_xlabel("每张图字节数 (kB)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("纯推理期降码率（不重训）：三个旋钮的实测效果")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(a.out, dpi=125)
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
