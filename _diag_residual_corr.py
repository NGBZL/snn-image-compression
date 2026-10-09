# -*- coding: utf-8 -*-
"""_diag_residual_corr.py -- why is the structural residual not smaller?

Pure diagnosis, NO training.  For same-class groups it answers:

  1. pixel-space correlation between the target and its anchor (are the images
     actually similar?)
  2. latent-space correlation / R^2 of y_i explained by y_ref (before any
     quantisation), and what a per-channel mean-shift-only code would cost
  3. the same for a random out-of-class pair, as a control
"""
from __future__ import annotations

import argparse
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F

from anchor import SAE_Anchor, build_data, _labels_of, _norm_of

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="anchor_v5.pth")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--eval-limit", type=int, default=400)
    a = p.parse_args()

    obj = torch.load(a.weights, map_location="cpu", weights_only=False)
    m = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                   image_size=obj["image_size"], latent_mode="gaussian",
                   use_hyperprior=obj.get("use_hyperprior", True),
                   z_channels=obj.get("z_channels", 48), norm_type=_norm_of(obj),
                   ref_wiring=obj.get("ref_wiring", "out"), residual_mode="sub").to(DEV)
    m.load_state_dict(obj["state_dict"])
    m.eval()
    for mm in m.modules():
        if hasattr(mm, "hard"):
            mm.hard = True

    da = argparse.Namespace(image_size=a.image_size, augment="none", data_dir="./data",
                            holdout_frac=0.15, seed=42, eval_limit=a.eval_limit)
    _, te = build_data(da)
    labels = _labels_of(te)
    by_cls = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    groups = [np.array(v) for v in by_cls.values() if len(v) >= a.group_size][:a.n_groups]
    g = a.group_size

    print(f"\n{'grp':>3} {'pix corr':>9} {'pix mae':>8} | {'lat corr':>9} {'lat R2':>7} "
          f"{'std(y_i)':>9} {'std(diff)':>10} {'ratio':>6} | {'shift+res':>10} {'0-shift':>8}"
          f" | {'rand pair corr':>15}")
    rows = []
    with torch.no_grad():
        for gi, idx in enumerate(groups):
            xs = torch.stack([te[int(i)][0] for i in idx[:g]]).to(DEV)
            ys = []
            for i in range(g):
                ys.append(m.encode_raw(xs[i:i + 1], None))
            Y = torch.cat(ys)                                  # [g,T,C,h,w]
            y_a = Y[0]
            pc, lc, r2, ratios, mae = [], [], [], [], []
            sh, zs = [], []
            for i in range(1, g):
                xi, xa = xs[i], xs[0]
                pc.append(float(F.cosine_similarity(xi.flatten(), xa.flatten(), dim=0)))
                mae.append(float((xi - xa).abs().mean()))
                yi = Y[i]
                lc.append(float(F.cosine_similarity(yi.flatten(), y_a.flatten(), dim=0)))
                # R^2 of projecting y_i onto y_a (global least squares)
                num = float((yi * y_a).sum())
                den = float((y_a * y_a).sum())
                alpha = num / max(1e-12, den)
                ss_res = float(((yi - alpha * y_a) ** 2).sum())
                ss_tot = float(((yi - yi.mean()) ** 2).sum())
                r2.append(1.0 - ss_res / max(1e-12, ss_tot))
                d = yi - y_a
                ratios.append(float(d.std()) / max(1e-9, float(yi.std())))
                # per-channel mean shift only
                sh.append(float((yi - y_a.mean(dim=(0, 1), keepdim=True)).std()) /
                          max(1e-9, float(yi.std())))
                zs.append(float(yi.std()))
            # out-of-class control: an image from a different group
            other = torch.stack([te[int(i)][0] for i in groups[(gi + 1) % len(groups)][:g]]).to(DEV)
            y_o = m.encode_raw(other[0:1], None)
            rcorr = float(F.cosine_similarity(Y[1].flatten(), y_o.flatten(), dim=0))
            ao = float((Y[1] * y_o).sum()) / max(1e-12, float((y_o * y_o).sum()))
            r2o = 1.0 - float(((Y[1] - ao * y_o) ** 2).sum()) / max(
                1e-12, float(((Y[1] - Y[1].mean()) ** 2).sum()))
            print(f"{gi+1:>3} {np.mean(pc):>9.4f} {np.mean(mae):>8.4f} | {np.mean(lc):>9.4f} "
                  f"{np.mean(r2):>7.4f} {np.mean(zs):>9.2f} "
                  f"{np.mean([float((Y[i] - y_a).std()) for i in range(1, g)]):>10.2f} "
                  f"{np.mean(ratios):>6.3f} | {np.mean(sh):>10.3f} {np.mean(zs):>8.2f} "
                  f"| rand: corr {rcorr:+.3f} R2 {r2o:.4f}")
            rows.append((np.mean(pc), np.mean(r2), np.mean(ratios)))
    print(f"\nmean over {len(groups)} groups: pixel corr {np.mean([r[0] for r in rows]):.4f}  "
          f"latent R^2(proj) {np.mean([r[1] for r in rows]):.4f}  "
          f"std(y_i-y_ref)/std(y_i) {np.mean([r[2] for r in rows]):.3f}")
    print("interpretation: if lat R2 ~ 0 the two latents are essentially uncorrelated, so")
    print("std(diff) ~ sqrt(2)*std(y) ~ 1.41x and NO structural subtraction can win.")


if __name__ == "__main__":
    main()
