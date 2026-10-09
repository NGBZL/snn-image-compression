# -*- coding: utf-8 -*-
"""Verification for fix #1: anchor.py::evaluate units / z stream / data size."""
import argparse
import json
import math
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
import anchor as A  # noqa: E402
from sae_rd import encode_latent_bytes, encode_z_bytes  # noqa: E402
from snntorch import utils  # noqa: E402

DEV = torch.device("cpu")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "anchor_v5.pth"
obj = torch.load(CKPT, map_location="cpu", weights_only=False)
args = argparse.Namespace(weights=CKPT, image_size=128, augment="none", data_dir="./data",
                          holdout_frac=0.02, seed=42, eval_limit=64, group_size=4,
                          n_groups=2, out_json="_verify_fix1.json")
obj_size = int(obj["image_size"])

print("=== A) data size now follows the checkpoint ===")
ck_size = int(obj.get("image_size", args.image_size))
if ck_size != args.image_size:
    print(f"  evaluate 会把命令行 image_size={args.image_size} 改成 {ck_size}")
    args.image_size = ck_size
_, te = A.build_data(args)
model = A.build_model(args, obj, DEV)
n_pix = model.image_size ** 2
print(f"  model.image_size={model.image_size}  model.n_pix={model.n_pix}  "
      f"actual pixels={n_pix}  match={model.n_pix == n_pix}")
print(f"  data tensor shape {tuple(te[0][0].shape)} -> "
      f"{'OK' if te[0][0].shape[-1] == model.image_size else 'MISMATCH'}")

labels = A._labels_of(te)
g = args.group_size
by_cls = {}
for i, y in enumerate(labels):
    by_cls.setdefault(int(y), []).append(i)
groups = [np.array(v) for v in by_cls.values() if len(v) >= g][:args.n_groups]

print("\n=== B) per-image accounting, old (what evaluate printed) vs new (container) ===")
print(f"{'grp/img':>9} {'OLD num_bits':>13} {'OLD nib (k32,nz)':>17} {'NEW y bits':>11} "
      f"{'NEW z bits':>11} {'NEW total bits':>15} {'NEW total B':>12}")
sums = dict(old=0, newy=0, newz=0, new=0, k32=0)
kmax_used = set()
with torch.no_grad():
    for gi, idx in enumerate(groups):
        xs = torch.stack([te[int(i)][0] for i in idx[:g]]).to(DEV)
        for i in range(g):
            utils.reset(model)
            r, rate, rec = model(xs[i:i + 1], ref=None, checkpoints={model.num_steps})
            rq = A._quantize(model, r)
            # --- exactly what the old evaluate did ---
            old_bits, _n, _p, _y = encode_latent_bytes(rq, model.quant.step, model.prior)
            # --- what a naive "no z, no extra_sigma" version would give at k=128 ---
            k32, _n, _p, _y = encode_latent_bytes(rq, model.quant.step, model.prior,
                                                  k_max=128)
            tot, yb, zb = A.container_bits(model, rq, rate)
            from codec.rd_codec import _pick_kmax
            kmax_used.add(_pick_kmax(torch.round(rq.detach()).long()))
            sums["old"] += old_bits
            sums["k32"] += k32
            sums["newy"] += yb
            sums["newz"] += zb
            sums["new"] += tot
            print(f"{gi}/{i:<7d} {old_bits:13d} {k32:17d} {yb:11d} {zb:11d} {tot:15d} "
                  f"{tot/8:12.1f}")
n_img = len(groups) * g
print(f"\n  sum over {n_img} images:")
print(f"    OLD (bits, mislabelled 'B'): {sums['old']} bits = {sums['old']/8:.0f} B "
      f"-> reported as {sums['old']/8:.0f} 'bytes' per image ({sums['old']/n_img:.0f} "
      f"bits/img)")
print(f"    NEW (container)            : {sums['new']} bits = {sums['new']/8:.0f} B "
      f"({sums['new']/n_img/8:.1f} B/img, {sums['new']/n_img/n_pix:.3f} bpp)")
print(f"    y/z split                  : {sums['newy']} + {sums['newz']} bits "
      f"(z = {100*sums['newz']/sums['new']:.1f}% of the file)")
print(f"    adaptive alphabet used     : {sorted(kmax_used)}")
print(f"    k_max=32+no-z variant      : {sums['k32']} bits = {sums['k32']/8:.0f} B "
      f"-> underestimates the container by {100*(1-sums['k32']/sums['new']):.1f}%")

print("\n=== C) evaluate's accounting matches pair_bits_matrix (same code path) ===")
with torch.no_grad():
    xs = torch.stack([te[int(i)][0] for i in groups[0][:g]]).to(DEV)
    anchors, pair = A.pair_bits_matrix(model, xs)
    direct = []
    for i in range(g):
        utils.reset(model)
        r, rate, _rec = model(xs[i:i + 1], ref=None, checkpoints={1})
        direct.append(A.container_bits(model, A._quantize(model, r), rate)[0])
print(f"  pair_bits_matrix diagonal : {anchors.astype(int).tolist()}")
print(f"  container_bits            : {direct}")
print(f"  identical: {bool(np.array_equal(anchors.astype(np.int64), np.array(direct)))}")

print("\n=== D) evaluate end-to-end JSON units ===")
a2 = argparse.Namespace(**vars(args))
a2.out_json = "_verify_fix1.json"
A.evaluate(a2)
with open(a2.out_json, encoding="utf-8") as f:
    j = json.load(f)
print(f"  independent_bytes={j['independent_bytes']:.0f}  "
      f"independent_bits={j['independent_bits']}  "
      f"ratio={j['independent_bits']/j['independent_bytes']:.3f} (must be 8.0)")
print(f"  independent_bpp={j['independent_bpp']:.4f} -> bits/(n_img*n_pix)="
      f"{j['independent_bits']/(j['n_groups']*j['group_size']*j['image_size']**2):.4f}")
ok = abs(j["independent_bits"] / j["independent_bytes"] - 8.0) < 1e-9
print("  UNIT CHECK:", "PASS" if ok else "FAIL")
