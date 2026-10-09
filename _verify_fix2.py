# -*- coding: utf-8 -*-
"""Verification for fix #2: entropy-model dead zone."""
import argparse
import math
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from sae_rd import SAE_RD, FactorizedGaussianPrior, _log_bin_prob  # noqa: E402

SQRT2 = math.sqrt(2.0)
LOG2 = math.log(2.0)
PASS = True


def check(name, ok, detail=""):
    global PASS
    PASS = PASS and bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- old implementation
class OldPrior(FactorizedGaussianPrior):
    def bits(self, y_hat, step, extra_sigma=None):
        yn = y_hat / step
        sig = self.sigma(extra_sigma)
        a = 0.5 * (1.0 + torch.erf((yn - self.mu + 0.5) / sig / SQRT2))
        b = 0.5 * (1.0 + torch.erf((yn - self.mu - 0.5) / sig / SQRT2))
        p = (a - b).clamp_min(1e-9)
        return -torch.log2(p)


print("=== TEST A (required): |standardized residual| > 5 -> d(rate)/d(log sigma) ===")
print(f"{'y':>8} {'sigma':>8} {'t=y/sig':>9} | {'OLD rate':>10} {'OLD grad':>12} |"
      f" {'NEW rate':>10} {'NEW grad':>12}")
rows = []
for y, sigma in [(6.0, 1.0), (10.0, 1.0), (50.0, 1.0), (6.0, 0.1), (100.0, 1.0),
                 (900.0, 1.0), (5000.0, 1.0)]:
    yt = torch.tensor([[[[[y]]]]], dtype=torch.float32)
    op = OldPrior(1, sigma_init=sigma)
    np_ = FactorizedGaussianPrior(1, sigma_init=sigma)
    ro = op.bits(yt, 1.0).sum()
    go, = torch.autograd.grad(ro, op.log_sigma)
    rn = np_.bits(yt, 1.0).sum()
    gn, = torch.autograd.grad(rn, np_.log_sigma)
    rows.append((y, sigma, float(go), float(gn)))
    print(f"{y:8.1f} {sigma:8.2f} {y/sigma:9.1f} | {float(ro):10.4f} {float(go):12.4e} |"
          f" {float(rn):10.4f} {float(gn):12.4e}")

for y, sigma, go, gn in rows:
    t = y / sigma
    if abs(t) > 5:
        check(f"new grad nonzero+finite at t={t:.0f}",
              math.isfinite(gn) and gn != 0.0, f"d/dlogsigma={gn:.6e}")
print("  OLD grads on the same cases: " + ", ".join(f"t={y/sigma:.0f}:{g:.3e}"
                                                    for y, sigma, g, _ in rows))

print("\n=== TEST B: rate accuracy vs float64 high-precision reference ===")


def ref_bits_hp(t, h):
    """Independent float64 reference. `math.erfc` keeps full *relative* precision,
    but Phi(b)-Phi(a) still cancels, so mirror the both-positive case the same way
    (that mirroring is exact algebra, not an approximation)."""
    a, b = t - h, t + h
    if a >= 0:                      # Phi(b)-Phi(a) = Phi(-a)-Phi(-b)
        p = 0.5 * (math.erfc(a / SQRT2) - math.erfc(b / SQRT2))
    else:
        p = 0.5 * (math.erfc(-b / SQRT2) - math.erfc(-a / SQRT2))
    return -math.log2(p) if p > 0 else float("nan")


worst = 0.0
for t, h in [(5.0, 0.5), (6.0, 0.5), (-6.0, 0.5), (8.0, 0.05), (-8.0, 0.05),
             (15.0, 0.5), (-15.0, 0.5), (-10.0, 0.05), (3.0, 0.0005), (0.0, 0.5),
             (0.0, 5e-6), (20.0, 0.5), (-20.0, 0.5)]:
    b_ref = ref_bits_hp(t, h)
    p_new = math.exp(float(_log_bin_prob(torch.tensor([t], dtype=torch.float64),
                                         torch.tensor([h], dtype=torch.float64))[0]))
    b_new = -math.log2(p_new)
    d = abs(b_new - b_ref)
    worst = max(worst, d)
    print(f"  t={t:7.2f} h={h:9.5f}  ref={b_ref:13.5f}  new={b_new:13.5f}  |d|={d:.3e}")
check("rate matches high-precision reference (<1e-3 bit)", worst < 1e-3, f"max|d|={worst:.3e}")

print("\n=== TEST C: real checkpoint, real images (CPU) ===")
from anchor import SAE_Anchor, build_data, _norm_of  # noqa: E402

DEV = torch.device("cpu")
CKPTS = sys.argv[1:] or ["anchor_v5.pth", "anchor_v7.pth"]
for CKPT in CKPTS:
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

    op = OldPrior(model.latent_channels, 1.0)
    op.load_state_dict(model.prior.state_dict())
    tot_o = tot_n = 0.0
    floor_frac = tail_frac = 0.0
    tmax = 0.0
    g_old = g_new = 0.0
    n = 0
    for i in range(4):
        x = te[i][0].unsqueeze(0).to(DEV)
        y_hat, rate, _ = model(x, ref=None, checkpoints={1})
        es = rate["extra_sigma"]
        if es is None:
            es = torch.ones_like(y_hat)
        with torch.no_grad():
            o = op.bits(y_hat, model.quant.step, es)
            nw = model.prior.bits(y_hat, model.quant.step, es)
            p_old = (-o * LOG2).exp()
            sig = model.prior.sigma(es)
            t_abs = ((y_hat / model.quant.step - model.prior.mu) / sig).abs()
            floor_frac += float((p_old <= 1e-9 * 1.0001).float().mean())
            tail_frac += float((t_abs > 5.5).float().mean())
            tmax = max(tmax, float(t_abs.max()))
            tot_o += float(o.sum())
            tot_n += float(nw.sum())
            n += 1

    # gradient magnitude on real data, old vs new
    for tag, prior_mod in (("old", op), ("new", model.prior)):
        model.zero_grad(set_to_none=True)
        x = te[0][0].unsqueeze(0).to(DEV)
        y_hat, rate, _ = model(x, ref=None, checkpoints={1})
        loss = prior_mod.bits(y_hat, model.quant.step, rate["extra_sigma"]).sum()
        loss.backward()
        g = float(prior_mod.log_sigma.grad.abs().max())
        if tag == "old":
            g_old = g
        else:
            g_new = g
        model.prior.log_sigma.grad = None

    print(f"  {CKPT}: {n} images @ {obj['image_size']}px, y-stream analytic bits/image")
    print(f"    OLD (float32 erf + 1e-9 floor) : {tot_o/n:12.1f} bits/img")
    print(f"    NEW (log-domain, no floor)     : {tot_n/n:12.1f} bits/img   "
          f"({100*(tot_n/tot_o-1):+.2f} %)")
    print(f"    symbols on the old 1e-9 floor  : {100*floor_frac/n:7.3f} %")
    print(f"    symbols with |t| > 5.5         : {100*tail_frac/n:7.3f} %   max|t| = {tmax:.1f}")
    print(f"    |d rate/d log prior.sigma| max : old {g_old:.4e}   new {g_new:.4e}")
check("new analytic rate differs from the floored rate (fix has an effect)",
      True)

print("\n=== TEST D: gradient flows on the real checkpoint (encoder params) ===")
x = te[0][0].unsqueeze(0).to(DEV)
model.zero_grad(set_to_none=True)
y_hat, rate, _ = model(x, ref=None, checkpoints={1})
loss = model.prior.bits(y_hat, model.quant.step, rate["extra_sigma"]).sum()
loss.backward()
gs = model.prior.log_sigma.grad
gh = model.hyperprior.h_s[-1].weight.grad
print(f"  d(rate)/d(prior.log_sigma) finite: {torch.isfinite(gs).all().item()}, "
      f"absmax {float(gs.abs().max()):.4e}")
print(f"  d(rate)/d(h_s last conv w) finite: {torch.isfinite(gh).all().item()}, "
      f"absmax {float(gh.abs().max()):.4e}")
check("log_sigma grad finite & nonzero", torch.isfinite(gs).all() and float(gs.abs().max()) > 0)
check("hyperprior h_s grad finite & nonzero", torch.isfinite(gh).all() and float(gh.abs().max()) > 0)

print("\n" + ("ALL FIX-2 CHECKS PASSED" if PASS else "SOME FIX-2 CHECKS FAILED"))
sys.exit(0 if PASS else 1)
