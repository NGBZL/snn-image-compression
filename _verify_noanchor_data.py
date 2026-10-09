# -*- coding: utf-8 -*-
"""Verify build_data for --data flowers / div2k / both (no training). ASCII prints only."""
import argparse
import time

import torch

import anchor


def ns(**kw):
    base = dict(image_size=128, latent_channels=24, num_steps=6, z_channels=48,
                use_hyperprior=True, data_dir="./data", data="both",
                div2k_limit=6000, augment="crop", norm="bn", holdout_frac=0.15,
                workers=4, eval_limit=300, group_size=4, residual_mode="", seed=42,
                weights="anchor_rd.pth", out="anchor_rd.pth",
                out_json="anchor_eval.json", n_groups=12)
    base.update(kw)
    return argparse.Namespace(**base)


for d in ("flowers", "div2k", "both"):
    t0 = time.time()
    tr, te = anchor.build_data(ns(data=d))
    x, y = tr[0]
    print(f"  -> --data {d}: train={len(tr)} holdout={len(te)} "
          f"sample={tuple(x.shape)} dtype={x.dtype} label={y} "
          f"x[min,max]=[{float(x.min()):.3f},{float(x.max()):.3f}] "
          f"({time.time() - t0:.1f}s)")
    # labels must be available for the anchor path
    lb = anchor._labels_of(tr)
    print(f"     labels: n={len(lb)} uniq={len(set(lb.tolist()))} "
          f"min={int(lb.min())} max={int(lb.max())}")

# augment=none must be identical ordering (holdout index alignment)
tr_a, te_a = anchor.build_data(ns(data="both", augment="none"))
xa, _ = te_a[0]
print(f"  -> augment=none sanity: holdout={len(te_a)} sample={tuple(xa.shape)}")

# div2k only, no labels in the file system sense -> anchor sampler must still build
tr_d, _ = anchor.build_data(ns(data="div2k"))
s = anchor.GroupedBatchSampler(anchor._labels_of(tr_d), 4, seed=42)
print(f"  -> div2k GroupedBatchSampler: len={len(s)} first_batch_len={len(next(iter(s)))}")
print("BUILD_DATA OK")
