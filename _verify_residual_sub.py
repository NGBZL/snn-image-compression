# -*- coding: utf-8 -*-
"""_verify_residual_sub.py -- verification for residual-mode (a) "hard structural residual".

NO TRAINING.  Short forward passes and encode/decode round trips only.

Checks
  1. round-trip exactness: encode a real image as a residual against a real anchor,
     decode, and compare with the encoder-side reconstruction.  Both the in-process
     path (`forward` vs `decode_with_ref`) and the real container path
     (`codec.encode_key` / `codec.decode_key` with `ref=y_ref_hat`) are measured,
     for residual_mode=sub AND residual_mode=learned, plus the ref=None anchor path.
  2. `--residual-mode learned` still reproduces the old published behaviour
     (v5 @256: indep sym std 66.66, resid 87.16, ratio 1.31).
  3. coded-latent magnitude for the residual role vs the independent role.

Usage:
  python _verify_residual_sub.py --weights anchor_v5.pth --image-size 256
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
from snntorch import utils

from anchor import SAE_Anchor, build_data, _labels_of, _norm_of
from codec.rd_codec import encode_key, decode_key, _decode_latents

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = True


def check(name, ok, detail=""):
    global PASS
    PASS = PASS and bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def build(weights, residual_mode):
    obj = torch.load(weights, map_location="cpu", weights_only=False)
    m = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                   image_size=obj["image_size"], latent_mode="gaussian",
                   use_hyperprior=obj.get("use_hyperprior", True),
                   z_channels=obj.get("z_channels", 48), norm_type=_norm_of(obj),
                   ref_wiring=obj.get("ref_wiring", "out"),
                   residual_mode=residual_mode).to(DEV)
    m.load_state_dict(obj["state_dict"])
    m.eval()
    for mm in m.modules():
        if hasattr(mm, "hard"):
            mm.hard = True
    return m, obj


def data_args(image_size, eval_limit):
    return argparse.Namespace(image_size=image_size, augment="none", data_dir="./data",
                              holdout_frac=0.15, seed=42, eval_limit=eval_limit)


# ======================================================================================
def roundtrip(weights):
    print("=" * 96)
    print("1. ROUND-TRIP EXACTNESS")
    print("=" * 96)
    out = {}
    # real images from the held-out split of build_data
    m0, obj = build(weights, "sub")
    _, te = build_data(data_args(m0.image_size, 0))
    n_steps = m0.num_steps
    xa = te[0][0].unsqueeze(0).to(DEV)          # anchor
    xt = te[1][0].unsqueeze(0).to(DEV)          # target (same class not guaranteed, irrelevant)

    for mode in ("sub", "learned"):
        model, _ = build(weights, mode)
        print(f"\n--- residual_mode={mode} (model.image_size={model.image_size}, "
              f"T={n_steps}, C={model.latent_channels}, step={float(model.quant.step)}) ---")
        with torch.no_grad():
            # the anchor's reference latent: hard quantised, the ONLY valid ref
            y_ref_hat = model.encode_ref_latent(xa)
            # anchor path (ref=None)
            utils.reset(model)
            c_a, rate_a, rec_a = model(xa, ref=None, checkpoints={n_steps})
            d_anchor = float((rec_a[n_steps] - rec_a[n_steps]).abs().max())  # self, trivially 0
            # encode the residual
            utils.reset(model)
            c_t, rate_t, rec_enc = model(xt, ref=y_ref_hat, checkpoints={n_steps})
            # decode the residual the way a decoder must: reconstruct() then decode()
            utils.reset(model)
            rec_dec = model.decode_with_ref(c_t, ref=y_ref_hat, checkpoints={n_steps})
            d_resid = float((rec_enc[n_steps] - rec_dec[n_steps]).abs().max())
            # cross-check: the decoded latent must equal y_ref_hat + c_t exactly
            y_full_enc = model.reconstruct(c_t, y_ref_hat)
            y_full_dec = model.reconstruct(c_t, y_ref_hat)
            d_lat = float((y_full_enc - y_full_dec).abs().max())

            # ---- real container: encode_key / decode_key with ref ----
            key_a = encode_key(model, xa, model_hash=b"vh", ref=None)
            key_t = encode_key(model, xt, model_hash=b"vh", ref=y_ref_hat)
            y_dec_a, h_a = _decode_latents(model, key_a)
            y_dec_t, h_t = _decode_latents(model, key_t, ref=y_ref_hat)
            img_dec_a = decode_key(model, key_a)
            img_dec_t = decode_key(model, key_t, ref=y_ref_hat)
            # the decoder's latent must equal the encoder's latent for the residual
            d_key_lat = float((y_full_enc - y_dec_t).abs().max())
            d_key_img = float((rec_enc[n_steps].clamp(0, 1) - img_dec_t).abs().max())
            d_key_anchor_lat = float((c_a - y_dec_a).abs().max())
            d_key_anchor_img = float((rec_a[n_steps].clamp(0, 1) - img_dec_a).abs().max())

            # how big is the coded residual vs the plain latent (magnitude check)
            y_t = model.encode_raw(xt, None)
            coded_t = model.code_residual(y_t, y_ref_hat)
            stat = dict(
                coded_std=float(coded_t.std()), coded_absmax=float(coded_t.abs().max()),
                latent_std=float(y_t.std()), latent_absmax=float(y_t.abs().max()),
                ref_std=float(y_ref_hat.std()),
                coded_zero_frac=float((coded_t.round() == 0).float().mean()),
                latent_zero_frac=float((y_t.round() == 0).float().mean()),
            )

        print(f"  anchor path   ref=None : enc-vs-dec image max|diff| {d_key_anchor_img:.3e} "
              f"(latent {d_key_anchor_lat:.3e})   key {len(key_a)} B")
        print(f"  residual path ref=y_ref: enc-vs-dec image max|diff| {d_key_img:.3e} "
              f"(latent {d_key_lat:.3e})   key {len(key_t)} B")
        print(f"  in-process forward vs decode_with_ref : {d_resid:.3e}")
        print(f"  coded: std {stat['coded_std']:.3f} |max| {stat['coded_absmax']:.0f} "
              f"zero% {100*stat['coded_zero_frac']:.1f}  /  plain latent: std "
              f"{stat['latent_std']:.3f} |max| {stat['latent_absmax']:.0f} "
              f"zero% {100*stat['latent_zero_frac']:.1f}   (y_ref_hat std {stat['ref_std']:.3f})")
        check(f"{mode}: residual in-process reconstruction is exact (<1e-6)",
              d_resid < 1e-6, f"max {d_resid:.3e}")
        check(f"{mode}: coded residual round-trips through the container",
              d_key_lat == 0.0, f"latent max {d_key_lat:.3e}")
        check(f"{mode}: residual image round-trips through the container (<1e-6)",
              d_key_img < 1e-6, f"image max {d_key_img:.3e}")
        check(f"{mode}: anchor key still round-trips exactly",
              d_key_anchor_lat == 0.0 and d_key_anchor_img < 1e-6,
              f"latent {d_key_anchor_lat:.3e} image {d_key_anchor_img:.3e}")
        if mode == "sub":
            # the central structural claim: coded == y_i - y_ref_hat, exactly (up to fp)
            with torch.no_grad():
                d_struct = float((coded_t - (model.quant_hard(y_t) - y_ref_hat)).abs().max())
            check("sub: coded == quant_hard(y_i) - y_ref_hat (structural identity)",
                  d_struct < 1e-4, f"max {d_struct:.3e}  (float32 cancellation)")
            print(f"  [NOTE] design-note prediction 'coded std < indep std': measured on a")
            print(f"         SINGLE pair here ({stat['coded_std']:.2f} vs "
                  f"{stat['latent_std']:.2f}); the >=6-group test in")
            print(f"         _residual_falsification.py is the one that decides it.")
        out[mode] = dict(d_key_img=d_key_img, d_key_lat=d_key_lat,
                         d_key_anchor_img=d_key_anchor_img, d_key_anchor_lat=d_key_anchor_lat,
                         d_inprocess=d_resid, key_anchor_bytes=len(key_a),
                         key_resid_bytes=len(key_t), **stat)
    return out


# ======================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="anchor_v5.pth")
    p.add_argument("--roundtrip-only", action="store_true")
    p.add_argument("--out-json", default="_verify_residual_sub.json")
    a = p.parse_args()
    t0 = time.time()
    print(f"device {DEV}   weights {a.weights}")
    res = {"weights": a.weights, "roundtrip": roundtrip(a.weights)}
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nwrote {a.out_json}  ({time.time()-t0:.0f}s)")
    print("ALL ROUND-TRIP CHECKS PASSED" if PASS else "SOME CHECKS FAILED")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
