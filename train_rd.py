# -*- coding: utf-8 -*-
"""
train_rd.py —— 训练率失真版脉冲自编码器（L = D + lambda * R）

实现需求里的 5 项改进中的 1/2/3/4：
  改进1  率项进损失（真实熵模型算 R）+ 加性均匀噪声量化 + 噪声退火 + 可学步长
  改进2  编码器最后一层 BN -> GDN
  改进3  Scale Hyperprior（逐像素 sigma）
  改进4  时间维渐进式 + 多尺度监督

损失
    D = MSE + 0.1 * (1 - MS_SSIM)          （多尺度监督时对若干检查点加权求和）
    R = E[-log2 p(y_hat)]                  （真实熵模型给的概率，单位 bpp）
    L = D + lambda * R

验证输出（每次训练都会打印/保存）
    - 潜层 0/1 比例（bernoulli 模式，应被率项推离 0.5）或零值占比（gaussian 模式）
    - R 随 epoch 的变化曲线
    - step 随 epoch 的变化（可学步长）
    - 潜层分布直方图（确认不是均匀分布）        -> rd_{tag}_hist.png
    - sigma 空间分布（确认平坦区小、复杂区大）  -> rd_{tag}_sigma.png
    - 质量 vs 收到步数（渐进式）                -> rd_{tag}_progressive.png

用法
    # 单项验证（跑 2 轮看看机制是否生效）
    python train_rd.py --tag probe --epochs 2 --latent-mode bernoulli --lam 0.1 --max-batches 6

    # 扫 lambda 得到 RD 曲线
    python train_rd.py --tag gauss --lam-sweep 0.001,0.01,0.1,1.0 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sae_model import human_bytes, jpeg_size_and_quality
from sae_rd import SAE_RD
from snntorch import utils

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    from pytorch_msssim import ms_ssim as _ms_ssim
except Exception:
    _ms_ssim = None


# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", default="rd", help="输出文件前缀")
    p.add_argument("--lam", type=float, default=0.01, help="率项权重 lambda")
    p.add_argument("--lam-sweep", default="", help="逗号分隔的多个 lambda，逐个训练后画 RD 曲线")
    p.add_argument("--latent-mode", choices=["gaussian", "bernoulli"], default="gaussian")
    p.add_argument("--use-hyperprior", action="store_true", default=True)
    p.add_argument("--no-hyperprior", dest="use_hyperprior", action="store_false")
    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--z-channels", type=int, default=48)
    p.add_argument("--learn-step", action="store_true",
                   help="学习量化步长；默认固定 step=1，码率完全由 sigma 控制（避免尺度退化）")
    p.add_argument("--no-sigma-warmup", dest="sigma_warmup", action="store_false", default=True,
                   help="不做 sigma 预热（默认做：把 sigma 初始化成潜层实际 std）")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--threshold", type=float, default=0.75)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--noise-anneal-epochs", type=int, default=0,
                   help="噪声从 1.0 退火到 0 用多少 epoch；0=整个训练期的 80%%")
    p.add_argument("--time-budget-min", type=float, default=0.0)

    p.add_argument("--prog-checkpoints", default="1,3,6,10",
                   help="多尺度监督的时间步检查点（改进4）")
    p.add_argument("--prog-weights", default="0.05,0.15,0.30,0.50",
                   help="对应权重，递增；会归一化")

    p.add_argument("--dataset", choices=["flowers", "stl10", "fake"], default="flowers")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hot-start", default="", help="用旧模型热启动编码器前三层，如 sae_cifar_best.pth")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-jpeg", type=int, default=24)
    p.add_argument("--real-bytes", type=int, default=3,
                   help="用 constriction 真实算术编码测几张图的实际字节数；0=跳过")
    return p.parse_args()


def pick_device(c):
    if c == "cpu":
        return torch.device("cpu")
    if c == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(a):
    tf = transforms.Compose([transforms.Resize(a.image_size),
                             transforms.CenterCrop(a.image_size),
                             transforms.ToTensor()])
    if a.dataset == "flowers":
        tr = datasets.Flowers102(a.data_dir, split="train", download=False, transform=tf)
        tr = torch.utils.data.ConcatDataset(
            [tr, datasets.Flowers102(a.data_dir, split="val", download=False, transform=tf)])
        te = datasets.Flowers102(a.data_dir, split="test", download=False, transform=tf)
    elif a.dataset == "stl10":
        tr = datasets.STL10(a.data_dir, split="train", download=True, transform=tf)
        te = datasets.STL10(a.data_dir, split="test", download=True, transform=tf)
    else:
        from train import FakeImageDataset
        tr = FakeImageDataset(256, a.image_size, seed=a.seed)
        te = FakeImageDataset(64, a.image_size, seed=a.seed + 9)
    if a.subset > 0:
        tr = Subset(tr, range(min(a.subset, len(tr))))
    if a.eval_limit > 0 and len(te) > a.eval_limit:
        te = Subset(te, range(a.eval_limit))
    pin = torch.cuda.is_available()
    return (DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                       pin_memory=pin, drop_last=False),
            DataLoader(te, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                       pin_memory=pin, drop_last=False))


# ======================================================================================
def distortion(pred, target):
    """D = MSE + 0.1*(1 - MS_SSIM)。尺寸太小时 MS-SSIM 用不了，退回纯 MSE。"""
    d = F.mse_loss(pred, target)
    if _ms_ssim is not None and pred.shape[-1] >= 160 and pred.shape[1] == 3:
        d = d + 0.1 * (1.0 - _ms_ssim(pred.clamp(0, 1), target, data_range=1.0))
    return d


def latent_stats(y_hat, mode):
    """返回 (0/1比例 或 零值占比, 量化后不同取值的个数, 潜层标准差)。"""
    with torch.no_grad():
        if mode == "bernoulli":
            frac = float(y_hat.mean())          # 真正的 0/1 比例
            uniq = 2
        else:
            q = y_hat.detach()
            frac = float((q.abs() < 1e-6).float().mean())   # 量化到 0 的占比
            uniq = int(torch.unique(q.round()).numel())
        std = float(y_hat.detach().std())
    return frac, uniq, std


def run_epoch(model, loader, opt, device, args, ckpts, ws, lam, train):
    model.train(train)
    agg = {k: 0.0 for k in ["loss", "D", "R", "mse", "psnr", "frac", "uniq", "std", "n"]}
    maxb = args.max_batches if args.max_batches > 0 else len(loader)
    for i, (x, _) in enumerate(loader):
        if i >= maxb:
            break
        x = x.to(device, non_blocking=True)
        utils.reset(model)
        y_hat, rate, recon = model(x, checkpoints=ckpts)

        D = 0.0
        for k, w in zip(ckpts, ws):
            D = D + w * distortion(recon[k], x)
        loss = D + lam * rate["bpp"]

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        with torch.no_grad():
            final = recon[max(ckpts)]
            frac, uniq, std = latent_stats(y_hat, args.latent_mode)
            b = x.shape[0]
            agg["loss"] += loss.item() * b
            agg["D"] += D.item() * b
            agg["R"] += rate["bpp"].item() * b
            agg["mse"] += F.mse_loss(final, x).item() * b
            agg["psnr"] += (10 * torch.log10(1.0 / F.mse_loss(final, x).clamp_min(1e-12))).item() * b
            agg["frac"] += frac * b
            agg["uniq"] += uniq * b
            agg["std"] += std * b
            agg["n"] += b
    n = max(1, agg.pop("n"))
    return {k: v / n for k, v in agg.items()}


# ======================================================================================
def save_visuals(model, batch, device, args, ckpts, tag):
    """保存三张诊断图：潜层分布直方图 / sigma 空间图 / 渐进式质量曲线。"""
    model.eval()
    x = batch.to(device)
    with torch.no_grad():
        utils.reset(model)
        y_hat, rate, recon = model(x, checkpoints=set(range(1, model.num_steps + 1)))

        # ---- 1. 潜层分布直方图 ----
        v = y_hat.detach().cpu().flatten().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(v, bins=80, color="tab:blue")
        ax[0].set_title(f"潜层取值分布 ({args.latent_mode})  std={v.std():.3f}")
        ax[0].set_xlabel("value"); ax[0].set_ylabel("count"); ax[0].grid(alpha=.3)
        if args.latent_mode == "bernoulli":
            ax[0].axvline(0.5, color="r", ls="--", label="0/1 中点")
            ax[0].legend()
        # 与"同范围均匀分布"对比：均匀分布下直方图是平的
        ax[1].hist(v, bins=80, density=True, color="tab:blue", alpha=.8, label="实际潜层")
        lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
        ax[1].hlines(1.0 / max(1e-9, (hi - lo)), lo, hi, color="r", ls="--",
                     label="同范围均匀分布")
        ax[1].set_xlim(lo, hi); ax[1].set_title("是否接近均匀分布？")
        ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{tag}_hist.png", dpi=110); plt.close()

        # ---- 2. sigma 空间分布（仅 gaussian + hyperprior）----
        sigma_img = None
        if model.latent_mode == "gaussian" and model.hyperprior is not None:
            y = model.encode_raw(x)
            sig, _, _, _ = model.hyperprior(y, False)
            s = sig[0].mean(dim=(0, 1)).cpu().numpy()      # 对 T 和 C 求均值 -> [h,w]
            sigma_img = s
            x0 = x[0].detach().cpu().permute(1, 2, 0).numpy()
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
            ax[0].imshow(np.clip(x0, 0, 1)); ax[0].set_title("原图")
            im = ax[1].imshow(s, cmap="viridis"); ax[1].set_title("sigma (逐像素)")
            plt.colorbar(im, ax=ax[1], fraction=0.046)
            # 用原图的局部梯度当"复杂度"参照
            g = x[0].mean(0).detach().cpu()
            gx = g[:, 1:] - g[:, :-1]
            gy = g[1:, :] - g[:-1, :]
            comp = torch.zeros_like(g); comp[:, :-1] += gx.abs(); comp[:-1, :] += gy.abs()
            comp = F.avg_pool2d(comp[None, None], 16, 16)[0, 0].numpy()
            im2 = ax[2].imshow(comp, cmap="magma"); ax[2].set_title("局部梯度(复杂度参照)")
            plt.colorbar(im2, ax=ax[2], fraction=0.046)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            plt.tight_layout(); plt.savefig(f"{tag}_sigma.png", dpi=110); plt.close()
            if comp.std() > 1e-6:
                cc = np.corrcoef(s.flatten(), comp.flatten())[0, 1]
                print(f"  [sigma] 与局部复杂度的相关系数 = {cc:+.3f} "
                      f"({'平坦区 sigma 小、复杂区大 OK' if cc > 0.15 else '相关性弱，hyperprior 可能还没学到'})")

        # ---- 3. 渐进式质量曲线（改进4）----
        # bits_y: [B,T,C,h,w] -> 每个时间步的比特 -> 累计
        per_step = rate["bits_y"].sum(dim=(2, 3, 4)).mean(0).detach().cpu().numpy()   # [T]
        cum = np.cumsum(per_step)
        zc = float(rate["bits_z"].item()) / max(1, x.shape[0])
        ks, psnrs, bpps = [], [], []
        for k in sorted(recon.keys()):
            ks.append(k)
            psnrs.append(10 * torch.log10(1.0 / F.mse_loss(recon[k], x).clamp_min(1e-12)).item())
            bpps.append((cum[k - 1] + zc * k / model.num_steps) / (model.image_size ** 2))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks, psnrs, "o-", color="tab:red")
        for k, p, b in zip(ks, psnrs, bpps):
            ax.annotate(f"{b:.3f}bpp", (k, p), fontsize=7, textcoords="offset points",
                        xytext=(0, 7))
        ax.set_xlabel("接收端收到的脉冲步数 k"); ax.set_ylabel("PSNR (dB)")
        ax.set_title("渐进式：质量 vs 收到步数")
        ax.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{tag}_progressive.png", dpi=110); plt.close()
        prog = list(zip(ks, psnrs, bpps))
    model.train(True)
    return prog, sigma_img


def train_one(args, lam, device, train_loader, test_loader, tag):
    print("=" * 78)
    print(f"训练 RD 模型  tag={tag}  lambda={lam}  mode={args.latent_mode}  "
          f"hyperprior={args.use_hyperprior}")
    print("=" * 78)
    torch.manual_seed(args.seed)
    model = SAE_RD(latent_channels=args.latent_channels, num_steps=args.num_steps,
                   image_size=args.image_size, latent_mode=args.latent_mode,
                   use_hyperprior=args.use_hyperprior, z_channels=args.z_channels,
                   beta=args.beta, threshold=args.threshold,
                   learn_step=args.learn_step).to(device)
    if args.hot_start and os.path.isfile(args.hot_start):
        nk, nt = model.load_encoder_front(args.hot_start)
        print(f"热启动: 从 {args.hot_start} 载入编码器前 3 层 {nk}/{nt} 个参数")
    if args.sigma_warmup and args.latent_mode == "gaussian":
        s0 = model.warmup_sigma(train_loader, device, n_batches=4)
        print(f"sigma 预热: 潜层实际 std = {s0:.4f}  ->  log_sigma 已按此初始化")
    print(f"量化步长: {'可学习' if args.learn_step else '固定为 1.0（码率由 sigma 控制）'}")

    nparam = sum(p.numel() for p in model.parameters())
    print(f"参数量 {nparam:,}  潜层 {model.num_steps}x{model.latent_channels}x"
          f"{model.latent_size}x{model.latent_size} = {model.key_bits()} 符号/图")
    print(f"损失 L = D + {lam} * R,  D = MSE + 0.1*(1-MS_SSIM)")

    ck = [int(v) for v in args.prog_checkpoints.split(",") if v.strip()]
    ck = sorted({min(max(1, k), args.num_steps) for k in ck})
    ws = [float(v) for v in args.prog_weights.split(",")]
    ws = ws[:len(ck)] + [1.0] * max(0, len(ck) - len(ws))
    tot = sum(ws); ws = [w / tot for w in ws]
    print(f"多尺度监督检查点 {ck} 权重 {[round(w,3) for w in ws]}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    fixed, _ = next(iter(test_loader))
    fixed = fixed[:8].to(device)

    hist = []
    anneal = args.noise_anneal_epochs or max(1, int(args.epochs * 0.8))
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if args.time_budget_min > 0 and ep > 1:
            el = (time.time() - t0) / 60
            if el > args.time_budget_min:
                print(f"[时间预算] 已用 {el:.1f} 分钟，停止。")
                break
        ns = max(0.0, 1.0 - (ep - 1) / max(1, anneal))
        model.quant.noise_scale = ns
        if model.hyperprior is not None:
            model.hyperprior.z_quant.noise_scale = ns

        te0 = time.time()
        tr = run_epoch(model, train_loader, opt, device, args, ck, ws, lam, True)
        with torch.no_grad():
            te = run_epoch(model, test_loader, None, device, args, ck, ws, lam, False)
        sched.step()
        dt = time.time() - te0
        hist.append(dict(ep=ep, **{f"tr_{k}": v for k, v in tr.items()},
                         **{f"te_{k}": v for k, v in te.items()},
                         noise=ns, step=float(model.quant.step.detach())))
        print(f"ep {ep:3d}/{args.epochs} | L {tr['loss']:.4f} = D {tr['D']:.4f} + {lam}*R "
              f"| train R {tr['R']:.4f} bpp | test R {te['R']:.4f} bpp PSNR {te['psnr']:.2f} dB "
              f"| 0/1比例 {te['frac']:.3f} | 取值数 {te['uniq']:.0f} "
              f"| step {float(model.quant.step.detach()):.3f} "
              f"sigma {float(model.prior.sigma().mean()) if args.latent_mode=='gaussian' else float('nan'):.3f} "
              f"| 噪声 {ns:.2f} | {dt:.1f}s")

    prog, sigma_img = save_visuals(model, fixed, device, args, ck, tag)
    print(f"\n  渐进式（收到 k 步 -> 质量）:")
    for k, p, b in prog:
        print(f"     k={k:2d}  累计码率 {b:.4f} bpp  PSNR {p:5.2f} dB")

    ckpt = f"{tag}_{args.latent_mode}_lam{lam}.pth"
    torch.save({"state_dict": model.state_dict(), "image_size": model.image_size,
                "latent_channels": model.latent_channels, "num_steps": model.num_steps,
                "latent_mode": model.latent_mode, "use_hyperprior": model.use_hyperprior,
                "z_channels": args.z_channels, "beta": model.beta,
                "threshold": model.threshold, "lam": lam, "history": hist,
                "progressive": prog}, ckpt)
    print(f"  已保存 {ckpt} ({human_bytes(os.path.getsize(ckpt))})")
    return model, hist, prog, ckpt


# ======================================================================================
def eval_rd_point(model, loader, device, n, args):
    """在测试集上算一个工作点的 (bpp, PSNR)，以及同码率 JPEG 的 PSNR。"""
    if n <= 0:
        return dict(bpp=float("nan"), psnr=float("nan"), jpeg_bytes=float("nan"),
                    jpeg_psnr=float("nan"), n=0)
    model.eval()
    tot_b = tot_p = tot_jb = tot_jp = 0.0
    got = 0
    with torch.no_grad():
        for x, _ in loader:
            for i in range(x.shape[0]):
                if got >= n:
                    break
                img = x[i:i + 1].to(device)
                utils.reset(model)
                y_hat, rate, recon = model(img, checkpoints={model.num_steps})
                r = recon[model.num_steps].clamp(0, 1)
                bpp = rate["bpp"].item()
                nb = int(round(bpp * model.image_size ** 2 / 8)) + 14
                tot_b += bpp
                tot_p += (10 * torch.log10(1 / F.mse_loss(r, img).clamp_min(1e-12))).item()
                jb, _q, jr, _m = jpeg_size_and_quality(x[i], nb)
                tot_jb += jb
                tot_jp += (10 * torch.log10(
                    1 / F.mse_loss(jr.unsqueeze(0).to(device), img).clamp_min(1e-12))).item()
                got += 1
            if got >= n:
                break
    return dict(bpp=tot_b / got, psnr=tot_p / got,
                jpeg_bytes=tot_jb / got, jpeg_psnr=tot_jp / got, n=got)


@torch.no_grad()
def measure_real_bytes(model, loader, device, n=3):
    """用 constriction 真实算术编码，量实际字节数，并和熵模型的估计对比。

    注意：y 用 hyperprior 的逐像素 sigma，z 也单独编码 ——
    否则真实字节会明显大于估计（实测 +42.9%）。
    """
    from sae_rd import encode_latent_bytes, encode_binary_bytes, encode_z_bytes
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True
    est_bits = real_bits = n_sym = 0
    got = 0
    for x, _ in loader:
        for i in range(x.shape[0]):
            if got >= n:
                break
            img = x[i:i + 1].to(device)
            utils.reset(model)
            y_hat, rate, _ = model(img, checkpoints={1})
            if model.latent_mode == "gaussian":
                b, ns, _p, _y = encode_latent_bytes(
                    y_hat, model.quant.step, model.prior,
                    extra_sigma=rate.get("extra_sigma"), verify=(got == 0))
                real_bits += b
                if rate.get("z_hat") is not None:
                    bz, _ = encode_z_bytes(rate["z_hat"], rate["z_step"],
                                           model.hyperprior.z_prior, verify=(got == 0))
                    real_bits += bz
            else:
                b, ns, _p, _y = encode_binary_bytes(y_hat, model.prior, verify=(got == 0))
                real_bits += b
            est_bits += rate["bits_y"].sum().item() + rate["bits_z"].item()
            n_sym += ns
            got += 1
        if got >= n:
            break
    model.train(True)
    return dict(est_bytes=est_bits / 8 / max(1, got), real_bytes=real_bits / 8 / max(1, got),
                n_sym=n_sym / max(1, got), n=got)


def main():
    args = parse_args()
    device = pick_device(args.device)
    train_loader, test_loader = build_loaders(args)
    print(f"设备 {device}  训练 {len(train_loader.dataset)} 张 / 测试 {len(test_loader.dataset)} 张")

    lams = ([float(v) for v in args.lam_sweep.split(",")] if args.lam_sweep else [args.lam])
    results = []
    for lam in lams:
        tag = args.tag if len(lams) == 1 else f"{args.tag}_lam{lam}"
        model, hist, prog, ckpt = train_one(args, lam, device, train_loader, test_loader, tag)
        r = eval_rd_point(model, test_loader, device, args.compare_jpeg, args)
        r["lam"] = lam
        results.append(r)
        if r["n"]:
            print(f"  -> 工作点 lambda={lam}: {r['bpp']:.4f} bpp, {r['psnr']:.2f} dB   "
                  f"(同码率 JPEG {r['jpeg_bytes']:.0f} B -> {r['jpeg_psnr']:.2f} dB, "
                  f"差 {r['psnr']-r['jpeg_psnr']:+.2f} dB)")
        else:
            print(f"  -> 工作点 lambda={lam}: 已跳过 JPEG 对比 (--compare-jpeg 0)")
        if args.real_bytes:
            mb = measure_real_bytes(model, test_loader, device, n=args.real_bytes)
            r["real_bytes"] = mb["real_bytes"]
            r["est_bytes"] = mb["est_bytes"]
            print(f"  -> 真实算术编码: 估计 {mb['est_bytes']:.0f} 字节 vs 实际 "
                  f"{mb['real_bytes']:.0f} 字节  (每个脉冲{'位' if args.latent_mode=='bernoulli' else '符号'}"
                  f" {mb['n_sym']:.0f} 个, 偏差 "
                  f"{100*(mb['real_bytes']/max(1e-9,mb['est_bytes'])-1):+.1f}%)")
        del model
        torch.cuda.empty_cache()

    # ---------------- RD 曲线 ----------------
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        order = np.argsort([r["bpp"] for r in results])
        bp = [results[i]["bpp"] for i in order]
        pp = [results[i]["psnr"] for i in order]
        ax.plot(bp, pp, "*-", color="tab:red", ms=16, lw=2, label="SNN SAE (ours, RD-trained)")
        jb = [results[i]["jpeg_bytes"] * 8 / (args.image_size ** 2 * 3) for i in order]
        jp = [results[i]["jpeg_psnr"] for i in order]
        ax.plot(jb, jp, "o--", color="tab:blue", label="JPEG @ same byte budget")
        for r in results:
            ax.annotate(f"λ={r['lam']}", (r["bpp"], r["psnr"]), fontsize=8,
                        textcoords="offset points", xytext=(6, -12))
        ax.set_xlabel("Rate (bpp)"); ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"RD curve: {args.dataset}, {args.image_size}x{args.image_size}, "
                     f"mode={args.latent_mode}, hyperprior={args.use_hyperprior}")
        ax.grid(alpha=.3); ax.legend()
        plt.tight_layout(); plt.savefig(f"{args.tag}_rd_curve.png", dpi=130); plt.close()
        print(f"\n已保存 {args.tag}_rd_curve.png")
    with open(f"{args.tag}_rd_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n汇总:")
    for r in results:
        print(f"  lam={r['lam']:<8} {r['bpp']:.4f} bpp  PSNR {r['psnr']:.2f} dB  "
              f"vs JPEG {r['jpeg_psnr']:.2f} dB  ({r['psnr']-r['jpeg_psnr']:+.2f} dB)")


if __name__ == "__main__":
    main()
