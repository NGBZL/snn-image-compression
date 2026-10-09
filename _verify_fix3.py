# -*- coding: utf-8 -*-
"""Verification for fix #3: _SIGMA_GRID range."""
import argparse
import math
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
import sae_rd  # noqa: E402
from sae_rd import SAE_RD, encode_latent_bytes  # noqa: E402
from anchor import SAE_Anchor, build_data, _norm_of  # noqa: E402

OLD_GRID = np.exp(np.linspace(np.log(0.01), np.log(100.0), 48))
NEW_GRID = sae_rd._SIGMA_GRID
print("old grid: %d points, [%.6g, %.6g], step %.4f decade"
      % (len(OLD_GRID), OLD_GRID[0], OLD_GRID[-1], math.log10(OLD_GRID[1] / OLD_GRID[0])))
print("new grid: %d points, [%.6g, %.6g], step %.4f decade"
      % (len(NEW_GRID), NEW_GRID[0], NEW_GRID[-1], math.log10(NEW_GRID[1] / NEW_GRID[0])))
print("SIGMA_MIN/SIGMA_MAX = %g / %g, LOG_SIGMA_CLAMP = %g"
      % (sae_rd.SIGMA_MIN, sae_rd.SIGMA_MAX, sae_rd.LOG_SIGMA_CLAMP))
assert np.isclose(NEW_GRID[0], sae_rd.SIGMA_MIN, rtol=1e-9) and \
       np.isclose(NEW_GRID[-1], sae_rd.SIGMA_MAX, rtol=1e-9)


def pick_kmax(yq, floor=128, cap=4096):
    m = int(yq.abs().max().item()) if yq.numel() else 0
    k = max(floor, 1)
    while k < min(m, cap):
        k *= 2
    return int(min(k, cap))


DEV = torch.device("cpu")
for CKPT in (sys.argv[1:] or ["anchor_v5.pth", "anchor_v7.pth", "normab_gn_best.pth"]):
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    args = argparse.Namespace(image_size=int(obj["image_size"]), augment="none",
                              data_dir="./data", holdout_frac=0.02, seed=42, eval_limit=8)
    _, te = build_data(args)
    model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                       image_size=obj["image_size"], latent_mode="gaussian",
                       use_hyperprior=obj.get("use_hyperprior", True),
                       z_channels=obj.get("z_channels", 48), norm_type=_norm_of(obj),
                       ref_wiring=obj.get("ref_wiring", "out")).to(DEV)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True

    above_o = above_n = below_o = below_n = 0.0
    tot = 0
    ybits_o = ybits_n = 0
    qerr_o = qerr_n = 0.0
    for i in range(4):
        x = te[i][0].unsqueeze(0).to(DEV)
        with torch.no_grad():
            y_hat, rate, _ = model(x, ref=None, checkpoints={1})
            es = rate["extra_sigma"]
            sig = model.prior.sigma(es).detach().cpu().numpy().astype(np.float64).reshape(-1)
        tot += sig.size
        # float32 的 1e-4 = 9.9999997e-05，比 float64 的网格端点低 2.5e-12；
        # 这种像素照样落进 0 号桶，所以用相对容差判"越界"。
        eps = 1e-6
        above_o += float((sig > OLD_GRID[-1] * (1 + eps)).sum())
        below_o += float((sig < OLD_GRID[0] * (1 - eps)).sum())
        above_n += float((sig > NEW_GRID[-1] * (1 + eps)).sum())
        below_n += float((sig < NEW_GRID[0] * (1 - eps)).sum())
        # quantization error of the nearest grid point, in log2
        qerr_o += float(np.abs(np.log2(sig / OLD_GRID[np.abs(
            np.log(sig)[:, None] - np.log(OLD_GRID)[None, :]).argmin(1)])).sum())
        qerr_n += float(np.abs(np.log2(sig / NEW_GRID[np.abs(
            np.log(sig)[:, None] - np.log(NEW_GRID)[None, :]).argmin(1)])).sum())
        kmax = pick_kmax(torch.round(y_hat.detach()).long())
        sae_rd._SIGMA_GRID = OLD_GRID
        try:
            with torch.no_grad():
                b_o = encode_latent_bytes(y_hat, model.quant.step, model.prior,
                                          extra_sigma=es, k_max=kmax)[0]
        finally:
            sae_rd._SIGMA_GRID = NEW_GRID
        with torch.no_grad():
            b_n = encode_latent_bytes(y_hat, model.quant.step, model.prior,
                                      extra_sigma=es, k_max=kmax)[0]
        ybits_o += b_o
        ybits_n += b_n
    n = 4
    print(f"\n{CKPT}: 4 images @ {obj['image_size']}px, effective sigma, {tot} values")
    print(f"  OLD grid: above top {100*above_o/tot:6.3f}%   below bottom {100*below_o/tot:6.3f}%"
          f"   mean |log2(sig/grid)| = {qerr_o/tot:.4f}")
    print(f"  NEW grid: above top {100*above_n/tot:6.3f}%   below bottom {100*below_n/tot:6.3f}%"
          f"   mean |log2(sig/grid)| = {qerr_n/tot:.4f}")
    print(f"  y-stream arithmetic-coded: OLD grid {ybits_o/n:11.1f} bits/img"
          f"  ->  NEW grid {ybits_n/n:11.1f} bits/img  ({100*(ybits_n/ybits_o-1):+.2f} %)")
    assert above_n == 0 and below_n == 0, "new grid must cover the whole clamped range"

print("\nFIX-3 CHECKS PASSED (0 pixels outside the new grid; grid == [SIGMA_MIN, SIGMA_MAX])")
