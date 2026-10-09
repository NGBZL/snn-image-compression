# -*- coding: utf-8 -*-
"""recalibrate_prior.py — 给**已经训练好的** checkpoint 重标定熵模型先验。

修的是什么
    `warmup_sigma` 原来只碰 y 先验（self.prior），**从不碰 z_prior**。于是
    hyperprior 的边信息码流（z）用的概率模型是训练初期随手学的那个 sigma，再也没人校准。
    实测 anchor_v5 @256：z_prior.sigma() = 0.0566，而逐通道拟合的 z std 均值 ≈ 0.28、
    pooled 1.26 —— 差 5~20 倍。z 流占 v5 整个文件的 ~31%，所以这一项单独就值 ~20% 体积，
    而且**不需要任何额外边信息**（编解码两端都能从权重算出来）。

怎么用
    python recalibrate_prior.py --weights anchor_v5.pth --out anchor_v5_rcal.pth
    python recalibrate_prior.py --weights anchor_v5.pth --mle

默认只重标定 z_prior（`--fit-y` 可连 y 先验一起，口径与 warmup_sigma 完全相同）。
全程 CPU 也可以跑（一分钟内），不会碰 GPU 上正在跑的训练。
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from anchor import SAE_Anchor, build_data, _norm_of
from sae_rd import _bin_probs, encode_z_bytes

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def z_bits_with(z_sym: np.ndarray, sig: np.ndarray, mu: np.ndarray,
                k_max: int = 128) -> float:
    """z_sym: [Cz, P] 整数符号 -> 按逐通道 (mu, sigma) 高斯算总码长（比特）。

    用 `sae_rd._bin_probs`，也就是算术编码器真正用的那张概率表，
    所以这里的数字和实际码流是同一个口径。
    """
    tot = 0.0
    for c in range(z_sym.shape[0]):
        p = _bin_probs(float(sig[c]), float(mu[c]), k_max)
        idx = (z_sym[c].astype(np.int64) + k_max).clip(0, 2 * k_max)
        tot += float(-np.log(np.clip(p[idx], 1e-300, None)).sum())
    return tot / math.log(2.0)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="", help="写出重标定后的 checkpoint（留空则只测量）")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n-images", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "auto"],
                   help="默认 cpu：这台机器上 GPU 可能正被训练占着")
    p.add_argument("--fit-y", action="store_true", help="连 y 先验一起重标定")
    p.add_argument("--mle", action="store_true",
                   help="额外报告逐通道离散 MLE sigma 能达到的 z 码长（诊断用）")
    a = p.parse_args()

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    obj = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                       image_size=obj["image_size"], latent_mode="gaussian",
                       use_hyperprior=obj.get("use_hyperprior", True),
                       z_channels=obj.get("z_channels", 48), norm_type=_norm_of(obj),
                       ref_wiring=obj.get("ref_wiring", "out")).to(dev)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True
    if model.hyperprior is None:
        print("这个 checkpoint 没有 hyperprior，没有 z 流可标定。")
        return

    args = argparse.Namespace(image_size=int(obj["image_size"]), augment="none",
                              data_dir=a.data_dir, holdout_frac=0.02, seed=42,
                              eval_limit=max(a.n_images, a.batch))
    _, te = build_data(args)
    n = min(a.n_images, len(te))
    xs = torch.stack([te[i][0] for i in range(n)]).to(dev)
    print(f"用 {n} 张留出集图片 @ {model.image_size}px 拟合（设备 {dev}）")

    # ---- 拿真实的 z 符号 ----
    # rate["z_hat"] 形状是 [B*T, Cz, hz, wz]（B 主序、t 次序）；码流里每个通道是把
    # (T, hz, wz) 一起编码的，所以先还原成"每张图"的 [Cz, T*hz*wz] 再统计，
    # 否则所有 per-image 数字都会差一个 T 倍。
    T = model.num_steps
    z_syms, z_analytic = [], []
    with torch.no_grad():
        for i in range(0, n, a.batch):
            x = xs[i:i + a.batch]
            _yh, rate, _rec = model(x, ref=None, checkpoints={1})
            z = rate["z_hat"].detach().cpu()
            b = x.shape[0]
            z = z.reshape(b, T, z.shape[1], z.shape[2], z.shape[3])
            for k in range(b):
                z_syms.append(torch.round(z[k]).permute(1, 0, 2, 3).reshape(z.shape[2], -1))
            z_analytic.append(float(rate["bits_z"].item()) / b)
    zsym = torch.stack(z_syms)                            # [N, Cz, T*hz*wz]
    zc = zsym.shape[1]
    per_ch = [zsym[:, c].reshape(-1).numpy().astype(np.float64) for c in range(zc)]
    pooled_std = float(np.concatenate(per_ch).std())
    ch_std = np.array([v.std() for v in per_ch])

    old_sig = model.hyperprior.z_prior.sigma().detach().cpu().numpy().reshape(-1)
    old_mu = model.hyperprior.z_prior.mu.detach().cpu().numpy().reshape(-1)
    old_bits = float(np.mean([z_bits_with(zsym[i].numpy(), old_sig, old_mu)
                              for i in range(zsym.shape[0])]))
    old_analytic = float(sum(z_analytic) / len(z_analytic))

    print(f"\nz 符号: {zsym.shape[0]} 张图 x {zc} 通道 x {zsym.shape[2]} 位置"
          f"（T={T}, hz*wz={zsym.shape[2] // T}）")
    print(f"  z_prior.sigma()  旧值 : 均值 {old_sig.mean():.4f}  "
          f"[{old_sig.min():.4f}, {old_sig.max():.4f}]")
    print(f"  z std（实测）        : pooled {pooled_std:.4f}")
    print(f"  逐通道 z std         : {np.array2string(ch_std, precision=4, max_line_width=100)}")
    print(f"  z 码流（旧先验）     : {old_bits:.1f} bits/图 "
          f"(模型自己的解析估计 {old_analytic:.1f} bits)")

    # ---- 重标定：和 warmup_sigma 完全同一条代码路径 ----
    loader = DataLoader(Subset(te, list(range(n))), batch_size=a.batch,
                        shuffle=False, num_workers=0)
    s0 = model.warmup_sigma(loader, dev, n_batches=max(1, math.ceil(n / a.batch)),
                            fit_y=a.fit_y, fit_z=True, verbose=True, train_mode=False)

    new_sig = model.hyperprior.z_prior.sigma().detach().cpu().numpy().reshape(-1)
    new_mu = model.hyperprior.z_prior.mu.detach().cpu().numpy().reshape(-1)
    new_bits = float(np.mean([z_bits_with(zsym[i].numpy(), new_sig, new_mu)
                              for i in range(zsym.shape[0])]))
    print(f"\n  z_prior.sigma()  新值 : 均值 {new_sig.mean():.4f}  "
          f"[{new_sig.min():.4f}, {new_sig.max():.4f}]")
    print(f"  z 码流（新先验）     : {new_bits:.1f} bits/图  "
          f"({100*(1-new_bits/max(1e-9,old_bits)):+.1f} % 相对旧先验)")
    # 诊断：如果沿用旧代码的"方差当 sigma"单位错，会差多少
    var_bits = float(np.mean([
        z_bits_with(zsym[i].numpy(), np.clip(ch_std ** 2, 1e-3, None), np.zeros(zc))
        for i in range(zsym.shape[0])]))
    print(f"  [诊断] 若沿用旧的 var 口径（sigma=方差，不在此次修复内）: "
          f"{var_bits:.1f} bits/图")
    print(f"  逐通道 log2(sigma_new/sigma_old): "
          f"{np.array2string(np.log2(new_sig/old_sig), precision=2, max_line_width=100)}")

    if a.mle:
        cand = np.exp(np.linspace(math.log(1e-3), math.log(30.0), 300))
        best, mle_total = [], 0.0
        for c in range(zc):
            syms_c = zsym[:, c, :].reshape(1, -1).numpy()
            bs = [z_bits_with(syms_c, np.array([s]), np.array([0.0])) for s in cand]
            k = int(np.argmin(bs))
            best.append(float(cand[k]))
            mle_total += bs[k]
        ent = 0.0
        for c in range(zc):
            v = per_ch[c].astype(np.int64)
            _, cnt = np.unique(v, return_counts=True)
            pr = cnt / cnt.sum()
            ent += float(-(pr * np.log2(pr)).sum() * cnt.sum())
        print(f"  [诊断] 逐通道离散 MLE sigma 上界 : {mle_total/zsym.shape[0]:.1f} bits/图")
        print(f"  [诊断] 逐通道经验直方图（oracle）: {ent/zsym.shape[0]:.1f} bits/图")

    # 真正走一遍算术编码器，确认先验换了之后码流确实变短（不只是解析估计）
    with torch.no_grad():
        _yh, rate, _rec = model(xs[:1], ref=None, checkpoints={1})
        zq = rate["z_hat"][:T].detach()
    print(f"  [实测] constriction z 码流（新先验）: "
          f"{encode_z_bytes(zq, rate['z_step'], model.hyperprior.z_prior, k_max=128)[0]} bits/图")

    if a.out:
        obj["state_dict"] = model.state_dict()
        obj["recalibrated"] = dict(z_sigma=model.hyperprior.z_prior.sigma()
                                   .detach().cpu().numpy().reshape(-1).tolist(),
                                   fit_y=bool(a.fit_y), n_images=int(n),
                                   pooled_z_std=pooled_std)
        torch.save(obj, a.out)
        print(f"\n已写出 {a.out}（z 先验{'和 y 先验' if a.fit_y else ''}已重标定；"
              f"y std 参考值 {s0:.4f}）")


if __name__ == "__main__":
    main()
