# -*- coding: utf-8 -*-
"""_residual_falsification.py -- the experiment from design_note_anchor_residual.md,
run for residual_mode in {sub, learned} on the SAME checkpoint.

NO TRAINING.  Forward passes + arithmetic coding only.

For >= 6 groups of 8 same-class held-out images it measures, per image:

    indep : encoded with ref=None            (baseline: every image codes its own latent)
    resid : encoded with ref=y_ref_hat       (the anchor's hard-quantised latent)

and compares bits(resid)/bits(indep) under four entropy models:
    (i)   model   -- the checkpoint's own prior + hyperprior (what actually ships)
    (ii)  gauss   -- a per-channel Gaussian fitted to the measured symbols
    (iii) oracle  -- per-image empirical per-channel histogram (free side info)
    (iv)  oracleP -- per-ROLE pooled empirical histogram (free side info; the
                    protocol used in anchor_bytes_report.md / anchor_bytes_summary.py)

Reported as the PAIRED PER-IMAGE MEDIAN ratio (the design note asks for this, not the
group total) plus the per-group spread.  y-stream and (y+z)-stream are both given.

Also reports the coded-latent std: for a structural residual it should drop BELOW the
independent role's std (today it is 1.11x-10.1x ABOVE).

Usage:
  python _residual_falsification.py --weights anchor_v5.pth --image-size 256 \
         --n-groups 6 --group-size 8 --out-json _residual_falsification.json
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
from sae_rd import encode_latent_bytes, _bin_probs, _categorical
from anchor_bytes import (_pick_kmax, encode_y_stream, encode_z_stream,
                          interval_bits, _hist_entropy, HDR_SIZE, Z_KMAX_RDCODEC)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PriorOverride:
    """Stand-in for FactorizedGaussianPrior with externally supplied mu/sigma.

    Used to answer "what would this stream cost if the factorized prior were fitted to
    the actual coded symbols instead of inherited from the checkpoint" -- the
    recalibrate_prior.py question, with NO training and NO side information
    (a per-channel fit over ONE image is decodable: the moments are what an
    encoder-side two-pass fit would send, and the per-image fit is what a decoder-side
    fit of the same stream cannot do -- so the per-image number is a lower bound;
    the pooled fit is the honest "one model for the role" number).
    """

    def __init__(self, mu, sigma):
        self._mu = torch.as_tensor(mu, dtype=torch.float32).reshape(1, -1, 1, 1)
        self._sigma = torch.as_tensor(sigma, dtype=torch.float32).reshape(1, -1, 1, 1)

    @property
    def mu(self):
        return self._mu

    def sigma(self, extra=None):
        s = self._sigma
        return s if extra is None else (s * extra).clamp(1e-4, 1e6)

    def bits(self, y_hat, step, extra_sigma=None):
        from sae_rd import _log_bin_prob
        yn = y_hat / step
        sig = self.sigma(extra_sigma)
        t = (yn - self.mu) / sig
        h = 0.5 / sig
        return -_log_bin_prob(t, h) / math.log(2.0)


@torch.no_grad()
def measure(model, x, ref, n_steps, fit_prior=None):
    """One encode of one image in one role.  Returns the flat per-image record plus the
    integer symbol arrays (kept out of the JSON).

    fit_prior: if not None, ALSO arithmetically code the same symbols with this prior
    (used for the fitted-Gaussian column).
    """
    if model.residual_mode == "sub":
        y = model.encode_raw(x, None)
        coded = model.code_residual(y, ref)                 # quant_hard(y_i) - y_ref_hat
        hint = ref
    else:
        coded = model.encode_raw(x, ref)                    # encoder saw the reference
        hint = None
    extra = z_hat = z_step = None
    bits_z_ana = torch.zeros((), device=x.device)
    if model.hyperprior is not None:
        extra, bits_z_ana, z_hat, z_step = model.hyperprior(coded, model.training)
    coded_hat = model.quant(coded)
    bits_y_ana = model.prior.bits(coded_hat, model.quant.step, extra)
    rec = model.decode(model.reconstruct(coded_hat, hint), checkpoints={n_steps})

    step = float(model.quant.step)
    yy = (coded_hat[0] / step).round().long().cpu().numpy()      # [T,C,h,w] int symbols
    raw = coded[0].detach().cpu().numpy().astype(np.float64)
    C = yy.shape[1]

    # ---- (i) model's own entropy model, real container ----
    kmax_y = _pick_kmax(yy)
    sig_all = model.prior.sigma(extra).detach().cpu()[0].numpy().astype(np.float64) \
        if extra is not None else \
        np.broadcast_to(model.prior.sigma().detach().cpu().numpy()[0],
                        (yy.shape[0], C) + yy.shape[2:]).astype(np.float64)
    mu_y = model.prior.mu.detach().cpu().numpy().reshape(-1).astype(np.float64)
    b_model, n_model, clip_y = encode_y_stream(yy.astype(np.int32), sig_all, mu_y, kmax_y)

    # ---- (ii) per-image per-channel Gaussian fitted to the actual symbols ----
    # Sum of -log2 interval-mass under the fitted prior.  Same convention as
    # encode_latent_bytes' probability tables (float64, 1e-9 floor), so it is the
    # range-coder cost up to the coder flush; computed in numpy on the CPU because
    # the per-pixel sigma field lives on the GPU while the fit happens here.
    fit_mu = np.zeros(C, dtype=np.float64)
    fit_sig = np.zeros(C, dtype=np.float64)
    # ---- (iii) per-image per-channel empirical histogram ----
    oracle_img = 0.0
    for c in range(C):
        v = yy[:, c].reshape(-1)
        fit_mu[c] = float(v.mean())
        fit_sig[c] = max(float(v.std()), 1e-3)
        oracle_img += _hist_entropy(v)
    b_fit = int(round(interval_bits(yy, fit_mu.reshape(1, C, 1, 1),
                                  fit_sig.reshape(1, C, 1, 1))))

    # ---- z stream (the side information), coded the deployment way ----
    if z_hat is not None:
        zq = (z_hat / float(z_step)).round().long().cpu().numpy()
        sigz = model.hyperprior.z_prior.sigma().detach().cpu().numpy().reshape(-1)
        muz = model.hyperprior.z_prior.mu.detach().cpu().numpy().reshape(-1)
        kmax_z = _pick_kmax(zq, floor=Z_KMAX_RDCODEC)
        b_z, n_zb, clip_z = encode_z_stream(zq, sigz, muz, kmax_z)
    else:
        zq, b_z, n_zb, clip_z = None, 0, 0, 0

    return dict(
        model_y_bits=b_model, fit_y_bits=b_fit, oracle_img_y_bits=oracle_img,
        model_z_bits=b_z, kmax_y=kmax_y, clip_y=clip_y,
        psnr=10.0 * math.log10(1.0 / max(1e-12, float(
            F.mse_loss(rec[n_steps], x).item()))),
        coded_std=float(coded.std()), coded_absmax=float(coded.abs().max()),
        coded_quant_std=float(coded_hat.std()),                   # after quantisation
        coded_zero_frac=float((yy == 0).mean()), n_sym=int(yy.size),
        sym_std=float(yy.std()), sym_absmax=float(np.abs(yy).max()),
        sym_absmean=float(np.abs(yy).mean()),
        ana_y_bits=float(bits_y_ana.sum().item()), ana_z_bits=float(bits_z_ana.item()),
    ), yy, zq


def run_mode(model, groups, te, g, tag):
    rows, syms_by_role, zsyms_by_role = [], {}, {}
    t0 = time.time()
    print(f"\n{'='*104}\nresidual_mode={tag}\n{'='*104}")
    with torch.no_grad():
        for gi, idx in enumerate(groups):
            xs = torch.stack([te[int(i)][0] for i in idx[:g]]).to(DEV)
            for i in range(g):
                utils.reset(model)
                m, yy, zq = measure(model, xs[i:i + 1], None, model.num_steps)
                m.update(role="anchor" if i == 0 else "indep", group=gi, idx=int(idx[i]))
                syms_by_role.setdefault(m["role"], []).append(yy)
                if zq is not None:
                    zsyms_by_role.setdefault(m["role"], []).append(zq)
                rows.append(m)
            y_ref_hat = model.encode_ref_latent(xs[0:1])
            for i in range(1, g):
                utils.reset(model)
                m, yy, zq = measure(model, xs[i:i + 1], y_ref_hat, model.num_steps)
                m.update(role="resid", group=gi, idx=int(idx[i]))
                syms_by_role.setdefault("resid", []).append(yy)
                if zq is not None:
                    zsyms_by_role.setdefault("resid", []).append(zq)
                rows.append(m)
            if gi < 3 or gi == len(groups) - 1:
                print(f"  group {gi+1}/{len(groups)} done ({time.time()-t0:.0f}s)")
    return rows, syms_by_role, zsyms_by_role


@torch.no_grad()
def pooled_oracle(sym_list):
    """Per-ROLE pooled empirical-histogram entropy (bits per image) + pooled std."""
    n = len(sym_list)
    yy = np.concatenate([a.reshape(a.shape[0], a.shape[1], -1) for a in sym_list], axis=2)
    tot = 0.0
    for c in range(yy.shape[1]):
        tot += _hist_entropy(yy[:, c].reshape(-1))
    return tot / n, float(yy.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="anchor_v5.pth")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--eval-limit", type=int, default=400)
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--modes", default="sub,learned")
    p.add_argument("--out-json", default="_residual_falsification.json")
    a = p.parse_args()

    obj = torch.load(a.weights, map_location="cpu", weights_only=False)
    print(f"weights={a.weights}  image_size(ckpt)={obj['image_size']}  "
          f"ref_wiring={obj.get('ref_wiring')}  epoch={obj.get('epoch')}  device={DEV}")
    da = argparse.Namespace(image_size=a.image_size, augment="none", data_dir="./data",
                            holdout_frac=a.holdout_frac, seed=42, eval_limit=a.eval_limit)
    _, te = build_data(da)
    labels = _labels_of(te)
    by_cls = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    groups = [np.array(v) for v in by_cls.values() if len(v) >= a.group_size][:a.n_groups]
    g = a.group_size
    print(f"groups: {len(groups)} x {g} same-class held-out images")

    out = dict(weights=a.weights, image_size=a.image_size, n_groups=len(groups),
               group_size=g, modes={})
    for mode in [m for m in a.modes.split(",") if m]:
        # 固定随机种子：两个模式各自新建模型，BatchNorm 的随机初始化会扰动先验，
        # 而**锚图/独立**那一列在两种模式下数学上完全相同，只有同种子才能逐位对上。
        torch.manual_seed(1234)
        model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                           image_size=obj["image_size"], latent_mode="gaussian",
                           use_hyperprior=obj.get("use_hyperprior", True),
                           z_channels=obj.get("z_channels", 48), norm_type=_norm_of(obj),
                           ref_wiring=obj.get("ref_wiring", "out"),
                           residual_mode=mode).to(DEV)
        model.load_state_dict(obj["state_dict"])
        model.eval()
        for mm in model.modules():
            if hasattr(mm, "hard"):
                mm.hard = True
        rows, syms_by_role, zsyms_by_role = run_mode(model, groups, te, g, mode)
        pooled = {r: pooled_oracle(v) for r, v in syms_by_role.items()}
        # ---- bit-exact cross-check of the roles that must not depend on the mode ----
        # (Floating-point std() reductions can wobble in the last digits between two
        #  freshly built models, so compare the actual INT symbols, which is what the
        #  coder sees.)
        snap = {r: np.stack([a.reshape(-1) for a in v])[:4]
                for r, v in syms_by_role.items() if r in ("indep", "anchor")}
        if "_crosscheck" not in out:
            out["_crosscheck"] = snap
        else:
            for r, cur in snap.items():
                ref = out["_crosscheck"][r]
                same = cur.shape == ref.shape and bool((cur == ref).all())
                print(f"  [cross-check] {r}: coded symbol arrays bit-identical across "
                      f"modes: {'YES' if same else 'NO'} ({cur.shape[0]} images)")
                out.setdefault("cross_check_symbols_identical", {})[r] = same
        syms_by_role.pop("_crosscheck", None)
        if "indep_anchor_check" not in out:
            out["indep_anchor_check"] = {r: dict(
                coded_quant_std=float(np.mean([x["coded_quant_std"] for x in rows
                                               if x["role"] == r])),
                sym_std=float(np.mean([x["sym_std"] for x in rows if x["role"] == r])),
                sym_absmax=float(np.mean([x["sym_absmax"] for x in rows
                                          if x["role"] == r])),
                oracle_pooled_y_bits=pooled[r][0],
                coded_std=float(np.mean([x["coded_std"] for x in rows
                                         if x["role"] == r])))
                for r in ("indep", "anchor")}
        else:
            # The anchor/independent roles use ref=None in BOTH modes, so the CODED
            # latent and everything derived from it must be identical with a fixed
            # seed.  The analytic ana_y/ana_z bits and the y-stream byte count are NOT
            # expected to match: in learned mode the hyperprior consumes the raw
            # encoder output while in sub mode it consumes the already-hard-quantised
            # difference -- that asymmetry is the original code's behaviour
            # (`forward` fed `r` to `hyperprior`, not `r_hat`) and is part of the A/B.
            for r in ("indep", "anchor"):
                new = dict(
                    coded_quant_std=float(np.mean([x["coded_quant_std"] for x in rows
                                                   if x["role"] == r])),
                    sym_std=float(np.mean([x["sym_std"] for x in rows
                                           if x["role"] == r])),
                    sym_absmax=float(np.mean([x["sym_absmax"] for x in rows
                                              if x["role"] == r])),
                    oracle_pooled_y_bits=pooled[r][0],
                    coded_std=float(np.mean([x["coded_std"] for x in rows
                                             if x["role"] == r])))
                old = out["indep_anchor_check"][r]
                d = max(abs(new[k] - old[k]) for k in new)
                print(f"  [cross-check] {r} role derived stats across modes: "
                      f"max|delta| {d:.3e}")
                out.setdefault("cross_check_max_delta", {})[r] = d

        # ---------------- paired per-image ratios -------------------------------------
        ind = {(r["group"], r["idx"]): r for r in rows if r["role"] == "indep"}
        res = [r for r in rows if r["role"] == "resid"]
        pairs = []
        for r in res:
            b = ind[(r["group"], r["idx"])]
            pairs.append(dict(group=r["group"], idx=r["idx"],
                              model_y=r["model_y_bits"] / max(1e-9, b["model_y_bits"]),
                              model_tot=((r["model_y_bits"] + r["model_z_bits"])
                                         / max(1e-9, b["model_y_bits"] + b["model_z_bits"])),
                              fit_y=r["fit_y_bits"] / max(1e-9, b["fit_y_bits"]),
                              oracle_img_y=(r["oracle_img_y_bits"]
                                            / max(1e-9, b["oracle_img_y_bits"])),
                              oracle_pooled_y=(pooled["resid"][0]
                                               / max(1e-9, pooled["indep"][0]))))
        def stat(key):
            v = np.array([p[key] for p in pairs])
            return dict(median=float(np.median(v)), mean=float(v.mean()),
                        q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                        min=float(v.min()), max=float(v.max()),
                        frac_gt1=float((v > 1).mean()))
        per_group = {}
        for gi in range(len(groups)):
            v = np.array([p["model_tot"] for p in pairs if p["group"] == gi])
            per_group[str(gi)] = dict(median=float(np.median(v)), min=float(v.min()),
                                      max=float(v.max()))
        agg = {r: dict(
            n=len([x for x in rows if x["role"] == r]),
            model_y_bits=float(np.mean([x["model_y_bits"] for x in rows if x["role"] == r])),
            model_z_bits=float(np.mean([x["model_z_bits"] for x in rows if x["role"] == r])),
            fit_y_bits=float(np.mean([x["fit_y_bits"] for x in rows if x["role"] == r])),
            oracle_img_y_bits=float(np.mean([x["oracle_img_y_bits"] for x in rows
                                             if x["role"] == r])),
            oracle_pooled_y_bits=pooled[r][0],
            pooled_sym_std=pooled[r][1],
            coded_std=float(np.mean([x["coded_std"] for x in rows if x["role"] == r])),
            coded_quant_std=float(np.mean([x["coded_quant_std"] for x in rows
                                           if x["role"] == r])),
            coded_absmax=float(np.mean([x["coded_absmax"] for x in rows if x["role"] == r])),
            sym_std=float(np.mean([x["sym_std"] for x in rows if x["role"] == r])),
            sym_absmax=float(np.mean([x["sym_absmax"] for x in rows if x["role"] == r])),
            coded_zero_frac=float(np.mean([x["coded_zero_frac"] for x in rows
                                           if x["role"] == r])),
            psnr=float(np.mean([x["psnr"] for x in rows if x["role"] == r])),
            ana_y_bits=float(np.mean([x["ana_y_bits"] for x in rows if x["role"] == r])),
        ) for r in ("indep", "anchor", "resid")}
        for r in agg:
            agg[r]["total_B"] = (agg[r]["model_y_bits"] + agg[r]["model_z_bits"]) / 8 + HDR_SIZE

        # ---------------- print ------------------------------------------------------
        print(f"\n--- residual_mode={mode}: means per role ---")
        print(f"{'role':<7}{'n':>3}{'coded std':>11}{'sym std':>9}{'|max|':>7}{'zero%':>7}"
              f"{'model y':>10}{'fit y':>10}{'orc/img':>10}{'orc pooled':>12}{'model z':>9}"
              f"{'tot B':>8}{'PSNR':>7}")
        for r in ("indep", "anchor", "resid"):
            d = agg[r]
            print(f"{r:<7}{d['n']:>3}{d['coded_std']:>11.2f}{d['sym_std']:>9.2f}"
                  f"{d['sym_absmax']:>7.0f}{100*d['coded_zero_frac']:>7.1f}"
                  f"{d['model_y_bits']:>10.0f}{d['fit_y_bits']:>10.0f}"
                  f"{d['oracle_img_y_bits']:>10.0f}{d['oracle_pooled_y_bits']:>12.0f}"
                  f"{d['model_z_bits']:>9.0f}{d['total_B']:>8.1f}{d['psnr']:>7.2f}")
        print(f"\n  PAIRED per-image bits(resid)/bits(indep), {len(pairs)} pairs "
              f"({len(groups)} groups x {g-1}):")
        rowsp = [("(i)   model prior (y only)", "model_y"),
                 ("(i)   model prior (y+z, ships)", "model_tot"),
                 ("(ii)  fitted Gaussian (y only)", "fit_y"),
                 ("(iii) ORACLE per-image hist (y)", "oracle_img_y"),
                 ("(iv)  ORACLE pooled hist (y)", "oracle_pooled_y")]
        print(f"    {'entropy model':<34}{'median':>9}{'mean':>9}{'q25':>8}{'q75':>8}"
              f"{'min':>8}{'max':>8}{'>1':>7}")
        for label, key in rowsp:
            s = stat(key)
            print(f"    {label:<34}{s['median']:>9.3f}{s['mean']:>9.3f}{s['q25']:>8.3f}"
                  f"{s['q75']:>8.3f}{s['min']:>8.3f}{s['max']:>8.3f}"
                  f"{100*s['frac_gt1']:>6.0f}%")
        print(f"  per-group spread of (i) y+z ratio: " + "  ".join(
            f"g{k}:{v['median']:.3f}" for k, v in per_group.items()))
        std_i, std_r = agg["indep"]["sym_std"], agg["resid"]["sym_std"]
        print(f"  coded-latent std: indep {agg['indep']['coded_std']:.2f} -> "
              f"resid {agg['resid']['coded_std']:.2f} "
              f"(ratio {agg['resid']['coded_std']/max(1e-9, agg['indep']['coded_std']):.3f}x)"
              f"   sym std {std_i:.2f} -> {std_r:.2f} (ratio {std_r/max(1e-9,std_i):.3f}x)")
        out["modes"][mode] = dict(
            agg=agg, paired={k: stat(k) for _, k in rowsp}, pairs=pairs,
            per_group_model_tot=per_group,
            scheme=dict(
                indep_B=((agg["anchor"]["model_y_bits"] + agg["anchor"]["model_z_bits"])
                         + (g - 1) * (agg["indep"]["model_y_bits"] + agg["indep"]["model_z_bits"])
                         ) / 8 + g * HDR_SIZE,
                anchor_res_B=((agg["anchor"]["model_y_bits"] + agg["anchor"]["model_z_bits"])
                              + (g - 1) * (agg["resid"]["model_y_bits"]
                                           + agg["resid"]["model_z_bits"])) / 8 + g * HDR_SIZE,
                indep_oracle_B=(agg["anchor"]["oracle_pooled_y_bits"]
                                + (g - 1) * agg["indep"]["oracle_pooled_y_bits"]) / 8,
                anchor_res_oracle_B=(agg["anchor"]["oracle_pooled_y_bits"]
                                     + (g - 1) * agg["resid"]["oracle_pooled_y_bits"]) / 8,
            ))

    # ---------------- cross-mode scheme table ------------------------------------------
    print(f"\n{'='*104}\nSCHEME TOTALS (per group of {g}, real container: y with hyperprior "
          f"sigma + z + {HDR_SIZE}B header)\n{'='*104}")
    print(f"{'mode':<9}{'indep B':>10}{'anch+res B':>12}{'delta':>9}"
          f"{'oracle indep':>14}{'oracle a+r':>12}{'oracle delta':>14}")
    for mode, d in out["modes"].items():
        s = d["scheme"]
        print(f"{mode:<9}{s['indep_B']:>10.0f}{s['anchor_res_B']:>12.0f}"
              f"{100*(s['anchor_res_B']/s['indep_B']-1):>+8.2f}%"
              f"{s['indep_oracle_B']:>14.0f}{s['anchor_res_oracle_B']:>12.0f}"
              f"{100*(s['anchor_res_oracle_B']/s['indep_oracle_B']-1):>+13.2f}%")
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if not k.startswith("_")}, f,
                  ensure_ascii=False, indent=2, default=float)
    print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
