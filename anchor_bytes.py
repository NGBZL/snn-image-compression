# -*- coding: utf-8 -*-
"""anchor_bytes.py -- byte-level breakdown of where the bytes go in the
"anchor image + residual" scheme.  ANALYSIS ONLY: no training, no checkpoint writes.

For a group of G same-class images (held-out split from anchor.build_data) this
measures, for every image and every role:

    indep   : encoded with ref=None  (the baseline: every image is its own anchor)
    anchor  : the group's first image, encoded with ref=None
    resid   : encoded with ref=<the group's first image>   (the residual path)

and splits the cost into

    y-stream  : symbols of the (residual) latent
    z-stream  : hyperprior side information
    header    : container header (codec/rd_codec.py, HDR_SIZE = 36 B)
    flush     : arithmetic-coder flush / byte-rounding overhead

Two protocols are reported, because they disagree:

  PROTO A ("evaluate" protocol, anchor.py:434) -- what produced _eval_v5/v7.json
      encode_latent_bytes(r_hat, step, prior)   # k_max=32, no extra_sigma, no z
      Per-channel factorized Gaussian only.  NOTE: anchor.py prints enc.num_bits()
      under the label "bytes"; those numbers are BITS.

  PROTO B ("deployment" protocol, codec/rd_codec.py) -- what actually ships
      z stream coded first (k_max>=128, adaptive), then y stream coded with the
      per-pixel hyperprior sigma bucketed on _SIGMA_GRID, adaptive alphabet,
      plus the 36-byte container header.

Extra diagnostics for the 4 questions:
  * latent magnitude (raw and quantized symbol) for indep vs resid
  * empirical symbol histogram entropy ("oracle") vs what the entropy model pays
  * a Gaussian re-fit to the actual symbols vs the learned prior.sigma
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
from snntorch import utils

from sae_rd import (encode_latent_bytes, _SIGMA_GRID, _bin_probs, _bucket_ids,
                    _categorical, _phi)
from anchor import SAE_Anchor, build_data, _labels_of, _norm_of

HDR_SIZE = 36            # codec/rd_codec.py HDR_FMT = "<4sBBBBHHHHHH8sII"
Z_KMAX_RDCODEC = 128     # rd_codec.DEF_KMAX


# ======================================================================================
# stream coders (mirrors of codec/rd_codec.py encode path)
# ======================================================================================
def _pick_kmax(arr, floor: int = 128, cap: int = 4096) -> int:
    """Same adaptive-alphabet rule as codec/rd_codec.py::_pick_kmax."""
    a = np.asarray(arr)
    m = int(np.abs(a).max()) if a.size else 0
    k = max(floor, 1)
    while k < min(m, cap):
        k *= 2
    return int(min(k, cap))


def encode_y_stream(syms: np.ndarray, sigmas: np.ndarray, mu: np.ndarray, kmax: int):
    """syms [T,C,h,w] int32 (already /step rounded), sigmas [T,C,h,w] float64,
    mu [C] float64.  Bucketed on _SIGMA_GRID exactly like rd_codec.encode_key.

    Returns (num_bits, compressed_nbytes, n_clipped).
    """
    import constriction
    C = syms.shape[1]
    n_clip = 0
    enc = constriction.stream.queue.RangeEncoder()
    for c in range(C):
        yq = syms[:, c].reshape(-1).astype(np.int64)
        n_clip += int((np.abs(yq) > kmax).sum())
        yq = np.clip(yq, -kmax, kmax) + kmax
        s = sigmas[:, c].reshape(-1).astype(np.float64)
        if s.size == 1:
            s = np.full(yq.size, float(s[0]))
        bid = _bucket_ids(s)
        for b in np.unique(bid):
            m = bid == b
            p = _bin_probs(float(_SIGMA_GRID[b]), float(mu[c]), kmax)
            enc.encode(yq[m].astype(np.int32), _categorical(p))
    comp = enc.get_compressed()
    return int(enc.num_bits()), int(comp.nbytes), n_clip


def encode_z_stream(zq: np.ndarray, sig: np.ndarray, mu: np.ndarray, kmax: int):
    """zq [T,Cz,hz,wz] int64 rounded, sig/mu [Cz].  Mirrors rd_codec._enc_z."""
    import constriction
    C = zq.shape[1]
    n_clip = 0
    enc = constriction.stream.queue.RangeEncoder()
    for c in range(C):
        zc = zq[:, c].reshape(-1).astype(np.int64)
        n_clip += int((np.abs(zc) > kmax).sum())
        zc = np.clip(zc, -kmax, kmax) + kmax
        p = _bin_probs(float(sig[c]), float(mu[c]), kmax)
        enc.encode(zc.astype(np.int32), _categorical(p))
    comp = enc.get_compressed()
    return int(enc.num_bits()), int(comp.nbytes), n_clip


def interval_bits(syms: np.ndarray, mu, sigma) -> float:
    """Sum of -log2 p(sym) under a factorized Gaussian (quantisation-interval mass).

    float64 throughout, but with the same 1e-9 probability floor the real coder's
    `_bin_probs` uses (float32 saturates Phi to 1.0 -> p=0), so the number is
    directly comparable with the model's analytic `rate['bits_y']`."""
    a = np.asarray(syms, dtype=np.float64)
    s = np.broadcast_to(np.asarray(sigma, dtype=np.float64), a.shape)
    m = np.broadcast_to(np.asarray(mu, dtype=np.float64), a.shape)
    k = torch.from_numpy(a)
    s = torch.from_numpy(np.ascontiguousarray(s))
    m = torch.from_numpy(np.ascontiguousarray(m))
    up = 0.5 * (1.0 + torch.erf((k + 0.5 - m) / (s * math.sqrt(2.0))))
    lo = 0.5 * (1.0 + torch.erf((k - 0.5 - m) / (s * math.sqrt(2.0))))
    p = (up - lo).clamp_min(1e-9)
    return float(-torch.log2(p).sum().item())


def _hist_entropy(v: np.ndarray, lo: int = -2048, hi: int = 2048) -> float:
    """Empirical entropy (TOTAL bits) of a pooled symbol histogram:
    sum_i -n_i * log2(n_i/N).  This is the cost of a per-channel static
    distribution equal to the empirical histogram (side info assumed free)."""
    vv = np.clip(np.asarray(v, dtype=np.int64), lo, hi)
    cnt = np.bincount(vv - lo, minlength=(hi - lo + 1)).astype(np.float64)
    p = cnt / cnt.sum()
    nz = p > 0
    return float(-(cnt[nz] * np.log2(p[nz])).sum())


# ======================================================================================
# one encode of one image in one role
# ======================================================================================
@torch.no_grad()
def measure(model, x, ref, psnr_ref_x, n_steps):
    """Runs the exact SAE_Anchor.forward sequence, but inlined so we can also grab
    the *pre-quantisation* latent.  Returns a dict of measurements.

    `ref` is the RAW reference latent from `encode_ref_latent(x_anchor)` in both
    modes; this function performs the hard quantisation that `code_residual`
    requires (in "sub" it is the subtrahend, in "learned" it is the encoder hint
    and equals what the old code passed).  For "learned" the numbers per image are
    therefore identical to the pre-change script.
    """
    dev = x.device
    if model.residual_mode == "sub":
        # sub: the encoder NEVER sees the reference; the coded quantity is the
        # explicit difference.  quant_hard is the single subtraction definition.
        y = model.encode_raw(x, None)
        coded = model.code_residual(y, ref)                # quant_hard(y_i) - y_ref_hat
        hint_dbg = ref
    else:
        # learned: the coded quantity IS the encoder output, which saw the hint.
        coded = model.encode_raw(x, ref)
        hint_dbg = None
    extra = z_hat = z_step = None
    bits_z_analytic = torch.zeros((), device=dev)
    if model.hyperprior is not None:
        extra, bits_z_analytic, z_hat, z_step = model.hyperprior(coded, model.training)
    r_hat = model.quant(coded)
    bits_y_analytic = model.prior.bits(r_hat, model.quant.step, extra)
    y_full = model.reconstruct(r_hat, hint_dbg)
    rec = model.decode(y_full, checkpoints={n_steps})

    step = float(model.quant.step)
    yy = (r_hat[0] / step).round().long().cpu().numpy()          # [T,C,h,w] symbols
    raw = coded[0].detach().cpu().numpy().astype(np.float64)     # the CODED quantity
    C = yy.shape[1]

    out = dict(
        role=None,
        ana_y_bits=float(bits_y_analytic.sum().item()),
        ana_z_bits=float(bits_z_analytic.item()),
        raw_std=float(raw.std()), raw_absmax=float(np.abs(raw).max()),
        sym_std=float(yy.std()), sym_absmax=float(np.abs(yy).max()),
        sym_absmean=float(np.abs(yy).mean()),
        sym_zero_frac=float((yy == 0).mean()),
        sym_gt32_frac=float((np.abs(yy) > 32).mean()),
        sym_gt128_frac=float((np.abs(yy) > 128).mean()),
        n_sym=int(yy.size),
        psnr=10.0 * math.log10(1.0 / max(1e-12, float(
            F.mse_loss(rec[n_steps], psnr_ref_x).item()))),
    )

    # ---- PROTO A: exactly what anchor.py::evaluate does -------------------------
    bA, nA, _, _ = encode_latent_bytes(r_hat, model.quant.step, model.prior)
    out["A_y_bits"] = bA
    out["A_y_bytes_round"] = int(math.ceil(bA / 8.0))

    # ---- PROTO B: deployment container (rd_codec) ------------------------------
    kmax_y = _pick_kmax(yy)
    sig_all = model.prior.sigma(extra).detach().cpu()[0].numpy().astype(np.float64) \
        if extra is not None else \
        np.broadcast_to(model.prior.sigma().detach().cpu().numpy()[0],
                        (yy.shape[0], C) + yy.shape[2:]).astype(np.float64)
    mu_y = model.prior.mu.detach().cpu().numpy().reshape(-1).astype(np.float64)
    bB, nB, clip_y = encode_y_stream(yy.astype(np.int32), sig_all, mu_y, kmax_y)
    out.update(B_y_bits=bB, B_y_nbytes=nB, kmax_y=kmax_y, clip_y=clip_y)
    # effective per-pixel sigma actually used by the coder, vs |symbol|
    _q = np.percentile(sig_all, [1, 10, 50, 90, 99])
    out["eff_sigma_med"] = float(np.median(sig_all))
    out["eff_sigma_min"] = float(sig_all.min())
    out["eff_sigma_max"] = float(sig_all.max())
    out["eff_sigma_p1"], out["eff_sigma_p10"], out["eff_sigma_p90"], out["eff_sigma_p99"] = \
        (float(_q[0]), float(_q[1]), float(_q[3]), float(_q[4]))
    out["eff_over_gridmax"] = float((sig_all > float(_SIGMA_GRID[-1])).mean())
    out["eff_under_gridmin"] = float((sig_all < float(_SIGMA_GRID[0])).mean())
    out["sym_over_sigma"] = float(np.mean(np.abs(yy) / np.maximum(sig_all, 1e-6)))

    if z_hat is not None:
        # z_hat is [B*T, Cz, hz, wz] -- encode the WHOLE thing (rd_codec._enc_z does)
        zq = (z_hat / float(z_step)).round().long().cpu().numpy()
        sigz = model.hyperprior.z_prior.sigma().detach().cpu().numpy().reshape(-1)
        muz = model.hyperprior.z_prior.mu.detach().cpu().numpy().reshape(-1)
        kmax_z = _pick_kmax(zq, floor=Z_KMAX_RDCODEC)
        bz, nzb, clip_z = encode_z_stream(zq, sigz, muz, kmax_z)
        # z calibration: what a per-channel Gaussian fitted to the actual z
        # symbols would cost, vs what z_prior (learned) costs.
        z_oracle = z_refit = 0.0
        for c in range(zq.shape[1]):
            v = zq[:, c].reshape(-1)
            z_oracle += _hist_entropy(v)
            z_refit += interval_bits(v, float(v.mean()), max(float(v.std()), 1e-3))
        out.update(B_z_bits=bz, B_z_nbytes=nzb, kmax_z=kmax_z, clip_z=clip_z,
                   z_sym=int(zq.size), z_sym_absmax=float(np.abs(zq).max()),
                   z_sym_std=float(zq.std()),
                   z_nz_frac=float((zq != 0).mean()),
                   z_oracle_bits=z_oracle, z_refit_bits=z_refit)
    else:
        out.update(B_z_bits=0, B_z_nbytes=0, kmax_z=0, clip_z=0, z_sym=0,
                   z_sym_absmax=0.0, z_sym_std=0.0, z_nz_frac=0.0,
                   z_oracle_bits=0.0, z_refit_bits=0.0)

    # ---- oracle diagnostics ----------------------------------------------------
    # (a) empirical histogram entropy per channel, for this one image
    oracle = 0.0
    refit = 0.0
    for c in range(C):
        v = yy[:, c].reshape(-1)
        oracle += _hist_entropy(v)
        # (b) Gaussian re-fit to the *actual* symbols of this image
        refit += interval_bits(v, float(v.mean()), max(float(v.std()), 1e-3))
    out["oracle_bits"] = oracle
    out["refit_bits"] = refit
    # keep symbols for the pooled oracle / sigma tables
    out["_syms"] = yy
    out["_zsyms"] = zq if z_hat is not None else None
    return out


# ======================================================================================
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="normab_gn_best.pth")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--image-size", type=int, default=128,
                   help="must match the eval run (anchor.py eval default is 128)")
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--eval-limit", type=int, default=400)
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--augment", default="crop")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--residual-mode", choices=["learned", "sub"], default="",
                   help="which residual semantics to measure.  empty = whatever the "
                        "checkpoint recorded; checkpoints from before this flag "
                        "(v5..v8) default to 'learned', which reproduces the old "
                        "published numbers exactly.")
    p.add_argument("--selftest", action="store_true",
                   help="skip the dataset: build the model and use random images")
    args = p.parse_args()

    t0 = time.time()
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    obj = torch.load(args.weights, map_location="cpu", weights_only=False)
    residual_mode = args.residual_mode or obj.get("residual_mode", "learned")
    model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                       image_size=obj["image_size"], latent_mode="gaussian",
                       use_hyperprior=obj.get("use_hyperprior", True),
                       z_channels=obj.get("z_channels", 48),
                       norm_type=_norm_of(obj),
                       ref_wiring=obj.get("ref_wiring", "out"),
                       residual_mode=residual_mode).to(dev)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True

    tag = args.tag or f"{os.path.splitext(os.path.basename(args.weights))[0]}_sz{args.image_size}"
    print("=" * 100)
    print(f"anchor_bytes.py  |  weights={args.weights}  device={dev}  tag={tag}")
    print(f"  model: image_size={model.image_size} latent_channels={model.latent_channels} "
          f"num_steps={model.num_steps} latent {model.latent_size}x{model.latent_size} "
          f"z_channels={obj.get('z_channels', 48) if model.hyperprior else 0} "
          f"norm={_norm_of(obj)} ref_wiring={model.ref_wiring} "
          f"residual_mode={model.residual_mode} "
          f"use_hyperprior={model.use_hyperprior} epoch={obj.get('epoch')}")
    print(f"  input size fed to the net: {args.image_size}x{args.image_size} "
          f"({args.image_size * args.image_size} px)   quant.step={float(model.quant.step)}")
    s0 = model.prior.sigma().detach().cpu().numpy().reshape(-1)
    print(f"  LEARNED per-channel prior.sigma(): min {s0.min():.4f} max {s0.max():.4f} "
          f"mean {s0.mean():.4f}   (mu: mean {model.prior.mu.detach().mean():+.4f} "
          f"std {model.prior.mu.detach().std():.4f})")
    if model.hyperprior is not None:
        zs = model.hyperprior.z_prior.sigma().detach().cpu().numpy().reshape(-1)
        print(f"  hyperprior z_prior.sigma(): min {zs.min():.4f} max {zs.max():.4f} "
              f"mean {zs.mean():.4f}")

    # ---- data ----------------------------------------------------------------
    if args.selftest:
        g = args.group_size
        xs = torch.rand(2, 3, args.image_size, args.image_size, device=dev)
        groups = [np.array([0, 1])]
        te = None
        print("  [selftest] using random images")
    else:
        da = SimpleNamespace(image_size=args.image_size, augment=args.augment,
                             data_dir=args.data_dir, holdout_frac=args.holdout_frac,
                             seed=args.seed, eval_limit=args.eval_limit)
        _, te = build_data(da)
        labels = _labels_of(te)
        g = args.group_size
        by_cls = {}
        for i, y in enumerate(labels):
            by_cls.setdefault(int(y), []).append(i)
        groups = [np.array(v) for v in by_cls.values() if len(v) >= g][: args.n_groups]
        print(f"  groups: {len(groups)} x {g} same-class images "
              f"({len(groups) * g} images of {len(te)} held out)")

    n_steps = model.num_steps
    rows = []          # flat per-image records (without the big _syms array)
    syms_by_role = {}  # role -> list of symbol arrays (for pooled oracle)
    zsyms_by_role = {}
    per_group = []

    for gi, idx in enumerate(groups):
        if args.selftest:
            xs = torch.rand(len(idx), 3, args.image_size, args.image_size, device=dev)
        else:
            xs = torch.stack([te[int(i)][0] for i in idx[:g]]).to(dev)
        gg = xs.shape[0]
        utils.reset(model)
        # ---- indep: every image is its own anchor (bit-identical to evaluate) ----
        ind = []
        for i in range(gg):
            m = measure(model, xs[i:i + 1], None, xs[i:i + 1], n_steps)
            m["role"] = "anchor" if i == 0 else "indep"
            m["group"] = gi
            m["idx"] = int(idx[i])
            m["anchor_psnr_target"] = None
            ind.append(m)
        # ---- resid: image i encoded against image 0's quantised latent ----------
        r0 = None
        with torch.no_grad():
            r0 = model.encode_ref_latent(xs[0:1])      # y_ref_hat: the exact value the
                                                       # decoder will have (hard quantised)
        for i in range(1, gg):
            m = measure(model, xs[i:i + 1], r0, xs[i:i + 1], n_steps)
            m["role"] = "resid"
            m["group"] = gi
            m["idx"] = int(idx[i])
            # paired baseline: same image, independent
            m["indep_total_B"] = (math.ceil(ind[i]["B_y_bits"] / 8.0)
                                  + math.ceil(ind[i]["B_z_bits"] / 8.0) + HDR_SIZE)
            m["indep_A_bits"] = ind[i]["A_y_bits"]
            m["own_total_B"] = (m["B_y_nbytes"] + m["B_z_nbytes"] + HDR_SIZE)
            ind.append(m)
        for m in ind:
            syms_by_role.setdefault(m["role"], []).append(m.pop("_syms"))
            zz = m.pop("_zsyms")
            if zz is not None:
                zsyms_by_role.setdefault(m["role"], []).append(zz)
            rows.append(m)
        if gi < 5 or gi == len(groups) - 1:
            print(f"  group {gi + 1}/{len(groups)} done  ({time.time() - t0:.0f}s)")

    # ==================================================================================
    # aggregation
    # ==================================================================================
    def agg(role):
        rs = [r for r in rows if r["role"] == role]
        if not rs:
            return None
        n = len(rs)

        def mn(k):
            return float(np.mean([r[k] for r in rs]))

        def sm(k):
            return float(np.sum([r[k] for r in rs]))

        d = dict(n=n,
                 A_y_bits=mn("A_y_bits"),
                 A_bytes=mn("A_y_bits") / 8.0,
                 B_y_bits=mn("B_y_bits"), B_z_bits=mn("B_z_bits"),
                 B_y_nbytes=mn("B_y_nbytes"), B_z_nbytes=mn("B_z_nbytes"),
                 ana_y_bits=mn("ana_y_bits"), ana_z_bits=mn("ana_z_bits"),
                 oracle_bits=mn("oracle_bits"), refit_bits=mn("refit_bits"),
                 psnr=mn("psnr"),
                 raw_std=mn("raw_std"), raw_absmax=mn("raw_absmax"),
                 sym_std=mn("sym_std"), sym_absmax=mn("sym_absmax"),
                 sym_absmean=mn("sym_absmean"), sym_zero_frac=mn("sym_zero_frac"),
                 sym_gt32_frac=mn("sym_gt32_frac"), sym_gt128_frac=mn("sym_gt128_frac"),
                 n_sym=int(rs[0]["n_sym"]), kmax_y=mn("kmax_y"), clip_y=sm("clip_y"),
                 kmax_z=mn("kmax_z"), clip_z=sm("clip_z"),
                 z_sym=int(rs[0]["z_sym"]), z_sym_absmax=mn("z_sym_absmax"),
                 z_sym_std=mn("z_sym_std"), z_nz_frac=mn("z_nz_frac"),
                 z_oracle_bits=mn("z_oracle_bits"), z_refit_bits=mn("z_refit_bits"),
                 eff_sigma_med=mn("eff_sigma_med"), eff_sigma_min=mn("eff_sigma_min"),
                 eff_sigma_max=mn("eff_sigma_max"), sym_over_sigma=mn("sym_over_sigma"),
                 eff_sigma_p1=mn("eff_sigma_p1"), eff_sigma_p10=mn("eff_sigma_p10"),
                 eff_sigma_p90=mn("eff_sigma_p90"), eff_sigma_p99=mn("eff_sigma_p99"),
                 eff_over_gridmax=mn("eff_over_gridmax"),
                 eff_under_gridmin=mn("eff_under_gridmin"))
        d["B_total_nbytes"] = (d["B_y_nbytes"] + d["B_z_nbytes"] + HDR_SIZE)
        d["bits_per_sym"] = d["B_y_bits"] / max(1, d["n_sym"])
        return d

    A = {r: agg(r) for r in ("anchor", "indep", "resid")}

    # pooled oracle: empirical histogram entropy over ALL images of a role
    pooled = {}
    for role, lst in syms_by_role.items():
        ni = len(lst)
        yy = np.concatenate([a.reshape(a.shape[0], a.shape[1], -1) for a in lst], axis=2)
        tot = 0.0
        for c in range(yy.shape[1]):
            tot += _hist_entropy(yy[:, c].reshape(-1))
        d = dict(pooled_n_img=ni, pooled_n_sym=int(yy.size),
                 pooled_oracle_bits=tot, pooled_oracle_bits_per_img=tot / max(1, ni),
                 y_pooled_std=float(yy.std()))
        zl = zsyms_by_role.get(role) or []
        if zl:
            # z shape is [B*T, Cz, hz, wz]; pool over images, keep channels apart
            zz = np.concatenate([a.reshape(a.shape[1], -1) for a in zl], axis=1)
            zo = zr = 0.0
            for c in range(zz.shape[0]):
                v = zz[c].reshape(-1)
                zo += _hist_entropy(v)
                zr += interval_bits(v, float(v.mean()), max(float(v.std()), 1e-4))
            d.update(z_pooled_std=float(zz.std()),
                     z_oracle_pooled_bits=zo, z_oracle_pooled_bits_per_img=zo / max(1, ni),
                     z_refit_pooled_bits=zr, z_refit_pooled_bits_per_img=zr / max(1, ni),
                     z_refit_pooled_sigma_mean=float(
                         np.mean([max(float(zz[c].std()), 1e-4) for c in range(zz.shape[0])])),
                     z_pooled_absmean=float(np.abs(zz).mean()))
        pooled[role] = d

    # ---- per-group scheme totals -----------------------------------------------------
    print()
    print("=" * 100)
    print("PER-IMAGE BYTE BREAKDOWN")
    print("PROTO B = deployment container (codec/rd_codec.py): z stream + y stream with the")
    print("          hyperprior per-pixel sigma + 36-byte header + range-coder flush.")
    print(f"{'role':<8}{'n':>4}{'y-stream B':>12}{'z-stream B':>12}{'hdr/other B':>13}"
          f"{'total B':>10}{'PSNR':>7}{'bits/sym':>9}")
    print("-" * 100)
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        yB, zB = d["B_y_nbytes"], d["B_z_nbytes"]
        other = (HDR_SIZE + (yB - math.ceil(d["B_y_bits"] / 8.0))
                 + (zB - math.ceil(d["B_z_bits"] / 8.0)))
        print(f"{role:<8}{d['n']:>4}{yB:>12.1f}{zB:>12.1f}{other:>13.1f}"
              f"{yB + zB + HDR_SIZE:>10.1f}{d['psnr']:>7.2f}{d['bits_per_sym']:>9.3f}")
    print("-" * 100)
    print(f"{'role':<8}{'':>4}{'y bits':>12}{'z bits':>12}{'(unrounded)':>13}")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"{role:<8}{d['n']:>4}{d['B_y_bits']:>12.0f}{d['B_z_bits']:>12.0f}")
    print("  header/other = 36 B container header (HDR_FMT '<4sBBBBHHHHHH8sII')")
    print("                 + range-coder flush/byte-rounding on both streams (~4 B each).")
    print("-" * 100)
    print("PROTO A (evaluate protocol: flat per-channel prior, k_max=32, NO z stream):")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"  {role:<8} n={d['n']:<3} y-stream {d['A_y_bits']:>10.0f} bits "
              f"= {d['A_y_bits'] / 8:>8.0f} B/img   (anchor.py prints this number as 'B')")
    print()
    print("LATENT MAGNITUDE (raw residual r, and quantised symbol round(r/step)):")
    print(f"{'role':<8}{'raw std':>10}{'raw |max|':>11}{'sym std':>10}{'sym |max|':>10}"
          f"{'sym |mean|':>11}{'zero%':>8}{'|sym|>32%':>10}{'|sym|>128%':>12}{'n_sym':>8}")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"{role:<8}{d['raw_std']:>10.2f}{d['raw_absmax']:>11.0f}{d['sym_std']:>10.2f}"
              f"{d['sym_absmax']:>10.0f}{d['sym_absmean']:>11.3f}"
              f"{100 * d['sym_zero_frac']:>8.2f}{100 * d['sym_gt32_frac']:>10.3f}"
              f"{100 * d['sym_gt128_frac']:>12.4f}{d['n_sym']:>8}")
    print()
    print("ENTROPY-MODEL CALIBRATION (bits per image, y stream only):")
    print(f"{'role':<8}{'model ana_y':>13}{'real y (B)':>12}{'oracle/1img':>13}"
          f"{'oracle pooled':>15}{'refit Gauss':>13}{'eff sig med':>13}{'sym/sig':>10}")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"{role:<8}{d['ana_y_bits']:>13.0f}{d['B_y_bits']:>12.0f}"
              f"{d['oracle_bits']:>13.0f}{pooled[role]['pooled_oracle_bits_per_img']:>15.0f}"
              f"{d['refit_bits']:>13.0f}{d['eff_sigma_med']:>13.4f}{d['sym_over_sigma']:>10.2f}")
    print("  (oracle = empirical symbol histogram of that image, side info free;")
    print("   refit  = single Gaussian per channel fitted to the actual symbols;")
    print("   eff sig = median per-pixel sigma = prior.sigma()*hyperprior sigma actually used)")
    print("  EFFECTIVE per-pixel sigma field (prior.sigma()*hyperprior sigma) vs _SIGMA_GRID[0.01,100]:")
    print(f"{'role':<8}{'p1':>10}{'p10':>10}{'p50':>10}{'p90':>10}{'p99':>11}{'max':>12}"
          f"{'%>grid100':>11}{'%<grid0.01':>12}{'sym std':>9}")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"{role:<8}{d['eff_sigma_p1']:>10.4f}{d['eff_sigma_p10']:>10.4f}"
              f"{d['eff_sigma_med']:>10.4f}{d['eff_sigma_p90']:>10.2f}{d['eff_sigma_p99']:>11.2f}"
              f"{d['eff_sigma_max']:>12.1f}{100 * d['eff_over_gridmax']:>11.2f}"
              f"{100 * d['eff_under_gridmin']:>12.2f}{d['sym_std']:>9.2f}")
    print("  z STREAM CALIBRATION (pooled over all images of the role, so the fit is honest):")
    print(f"{'role':<8}{'z bits':>10}{'model ana_z':>13}{'fitted z':>10}{'z oracle':>10}"
          f"{'z share':>9}{'z std pooled':>14}{'z |mean|':>10}")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"{role:<8}{d['B_z_bits']:>10.0f}{d['ana_z_bits']:>13.0f}"
              f"{pooled[role]['z_refit_pooled_bits_per_img']:>10.0f}"
              f"{pooled[role]['z_oracle_pooled_bits_per_img']:>10.0f}"
              f"{100 * d['B_z_bits'] / max(1e-9, d['B_y_bits'] + d['B_z_bits']):>8.2f}%"
              f"{pooled[role]['z_pooled_std']:>14.3f}{pooled[role]['z_pooled_absmean']:>10.3f}")
    print()
    print("SIDE INFO (z) RAW STATS / per-image (overfit) fits:")
    print(f"{'role':<8}{'z sym':>7}{'z |max|':>9}{'z std':>8}{'z nonzero%':>12}"
          f"{'z fit/1img':>12}{'z orc/1img':>12}")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"{role:<8}{d['z_sym']:>7}{d['z_sym_absmax']:>9.1f}{d['z_sym_std']:>8.2f}"
              f"{100 * d['z_nz_frac']:>12.2f}{d['z_refit_bits']:>12.0f}{d['z_oracle_bits']:>12.0f}")
    print()
    print("y stream: kmax_y per image, clipped symbols:")
    for role in ("indep", "anchor", "resid"):
        d = A[role]
        print(f"  {role:<8} kmax_y {d['kmax_y']:>7.0f}   clipped symbols/img {d['clip_y']:.1f}")

    # ---- scheme totals ---------------------------------------------------------------
    print()
    print("=" * 100)
    print("SCHEME TOTALS PER GROUP (G=8)")
    totB_i = sum(r["B_y_nbytes"] + r["B_z_nbytes"] + HDR_SIZE
                 for r in rows if r["role"] in ("indep", "anchor"))
    totB_a = sum(r["B_y_nbytes"] + r["B_z_nbytes"] + HDR_SIZE
                 for r in rows if r["role"] in ("anchor", "resid"))
    totA_i = sum(r["A_y_bits"] for r in rows if r["role"] in ("indep", "anchor")) / 8.0
    totA_a = sum(r["A_y_bits"] for r in rows if r["role"] in ("anchor", "resid")) / 8.0
    ng = len(groups)
    print(f"  independent  (PROTO B): {totB_i:>9.0f} B total   {totB_i / (ng * g):>8.1f} B/img")
    print(f"  anchor+resid (PROTO B): {totB_a:>9.0f} B total   {totB_a / (ng * g):>8.1f} B/img"
          f"   -> {100 * (totB_a / max(1e-9, totB_i) - 1):+.2f}%")
    print(f"  independent  (PROTO A): {totA_i:>9.0f} B total   {totA_i / (ng * g):>8.1f} B/img")
    print(f"  anchor+resid (PROTO A): {totA_a:>9.0f} B total   {totA_a / (ng * g):>8.1f} B/img"
          f"   -> {100 * (totA_a / max(1e-9, totA_i) - 1):+.2f}%")
    tot_x = A["anchor"]["B_total_nbytes"] + (g - 1) * A["resid"]["B_total_nbytes"]
    print(f"  PROTO B decomposition (per group): 1 anchor {A['anchor']['B_total_nbytes']:.0f} B + "
          f"{g - 1} x resid {A['resid']['B_total_nbytes']:.0f} B = {tot_x:.0f} B "
          f"vs {g} x indep {g * A['indep']['B_total_nbytes']:.0f} B")
    resid_over = A["resid"]["B_total_nbytes"] - A["indep"]["B_total_nbytes"]
    print(f"  resid - indep = {resid_over:+.0f} B/img  "
          f"(y {(A['resid']['B_y_nbytes'] - A['indep']['B_y_nbytes']):+.0f} B, "
          f"z {(A['resid']['B_z_nbytes'] - A['indep']['B_z_nbytes']):+.0f} B)")
    print(f"  header+flush share of a resid image total: "
          f"{100 * (HDR_SIZE + (A['resid']['B_y_nbytes'] - A['resid']['B_y_bits'] / 8)
                   + (A['resid']['B_z_nbytes'] - A['resid']['B_z_bits'] / 8)) / max(1e-9, A['resid']['B_total_nbytes']):.3f}%")

    # paired per-image comparison (Q2)
    pairs = [r for r in rows if r["role"] == "resid"]
    if pairs:
        ratio_B = np.array([r["own_total_B"] / max(1e-9, r["indep_total_B"]) for r in pairs])
        ratio_A = np.array([r["A_y_bits"] / max(1e-9, r["indep_A_bits"]) for r in pairs])
        print()
        print("PAIRED per-image comparison, resid vs the SAME image coded independently:")
        print(f"  PROTO B total  resid/indep: mean {ratio_B.mean():.3f}  median {np.median(ratio_B):.3f} "
              f"min {ratio_B.min():.3f} max {ratio_B.max():.3f}  "
              f">1 in {100 * (ratio_B > 1).mean():.0f}% of images")
        print(f"  PROTO A y-only resid/indep: mean {ratio_A.mean():.3f}  median {np.median(ratio_A):.3f} ")
        print(f"  d(sym std) resid-indep: {A['resid']['sym_std'] - A['indep']['sym_std']:+.2f}   "
              f"d(sym |max|): {A['resid']['sym_absmax'] - A['indep']['sym_absmax']:+.0f}   "
              f"d(raw std): {A['resid']['raw_std'] - A['indep']['raw_std']:+.2f}")

    # ---- per-group detail -------------------------------------------------------------
    print()
    print("PER-GROUP DETAIL (PROTO B, bytes per image):")
    print(f"{'grp':>4}{'indep(8)':>10}{'anchor':>9}{'resid(7)':>10}{'resid/indep':>12}"
          f"{'grp indep B':>13}{'grp anch+res B':>16}{'delta%':>8}")
    for gi in range(ng):
        rs = [r for r in rows if r["group"] == gi]
        ii = [r["own_total_B"] if "own_total_B" in r else
              (r["B_y_nbytes"] + r["B_z_nbytes"] + HDR_SIZE)
              for r in rs if r["role"] == "indep"]
        aa = [r["B_y_nbytes"] + r["B_z_nbytes"] + HDR_SIZE
              for r in rs if r["role"] == "anchor"]
        rr = [r["own_total_B"] for r in rs if r["role"] == "resid"]
        gi_b = float(np.sum(ii) + np.sum(aa))
        ga_b = float(np.sum(aa) + np.sum(rr))
        print(f"{gi + 1:>4}{np.mean(ii):>10.0f}{np.mean(aa):>9.0f}{np.mean(rr):>10.0f}"
              f"{np.mean(rr) / max(1e-9, np.mean(ii)):>12.3f}{gi_b:>13.0f}{ga_b:>16.0f}"
              f"{100 * (ga_b / max(1e-9, gi_b) - 1):>+8.2f}")

    # ---- write json -----------------------------------------------------------------
    out = dict(tag=tag, weights=args.weights, image_size=args.image_size,
               residual_mode=model.residual_mode,
               n_groups=ng, group_size=g, per_role={k: v for k, v in A.items() if v},
               pooled=pooled,
               scheme=dict(protoB_independent_B=totB_i, protoB_anchor_resid_B=totB_a,
                           protoA_independent_B=totA_i, protoA_anchor_resid_B=totA_a),
               rows=[{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    path = args.out_json or f"anchor_bytes_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nwrote {path}   ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
