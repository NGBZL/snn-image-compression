# -*- coding: utf-8 -*-
"""Fix #4 extras: training-path smoke test (train_mode=True) + y-prior std convention."""
import argparse
import math
import sys
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
from anchor import SAE_Anchor, build_data, _norm_of  # noqa: E402
from sae_rd import encode_latent_bytes  # noqa: E402
from codec.rd_codec import _pick_kmax  # noqa: E402

DEV = torch.device("cpu")
obj = torch.load("anchor_v5.pth", map_location="cpu", weights_only=False)
args = argparse.Namespace(image_size=256, augment="none", data_dir="./data",
                          holdout_frac=0.02, seed=42, eval_limit=12)
_, te = build_data(args)


def build():
    m = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                   image_size=obj["image_size"], latent_mode="gaussian",
                   use_hyperprior=True, z_channels=obj.get("z_channels", 48),
                   norm_type=_norm_of(obj), ref_wiring=obj.get("ref_wiring", "out")).to(DEV)
    m.load_state_dict(obj["state_dict"])
    m.eval()
    for mm in m.modules():
        if hasattr(mm, "hard"):
            mm.hard = True
    return m


loader = DataLoader(Subset(te, list(range(8))), batch_size=4, shuffle=False, num_workers=0)

print("=== A) training-path smoke test: warmup_sigma(train_mode=True, n_batches=2) ===")
m = build()
before = m.state_dict()["bn1.running_mean"].clone()
s0 = m.warmup_sigma(loader, DEV, n_batches=2, fit_y=True, fit_z=True, train_mode=True)
after = m.state_dict()["bn1.running_mean"]
print(f"  returned y std {s0:.4f};  y prior sigma mean {float(m.prior.sigma().mean()):.4f};"
      f"  z prior sigma mean {float(m.hyperprior.z_prior.sigma().mean()):.4f}")
print(f"  bn1 running_mean moved (train mode really used): "
      f"{not bool(torch.equal(before, after))}")
print(f"  z prior log_sigma finite: {bool(torch.isfinite(m.hyperprior.z_prior.log_sigma).all())}")

print("\n=== B) y stream: sigma=std (new warmup) vs sigma=variance (old warmup) ===")
m2 = build()
with torch.no_grad():
    x4 = torch.stack([te[i][0] for i in range(4)]).to(DEV)
    y, rate, _ = m2(x4, ref=None, checkpoints={1})
    es = rate["extra_sigma"]
    kmax = _pick_kmax(torch.round(y.detach()).long())
    yf = y.permute(0, 1, 3, 4, 2).reshape(-1, m2.latent_channels)
    var = yf.var(dim=0, unbiased=False)
    res = {}
    for tag, sig in (("old warmup: sigma=variance", var),
                     ("new warmup: sigma=std", var.sqrt())):
        with torch.no_grad():
            m2.prior.log_sigma.copy_(sig.clamp_min(1e-8).log().view(1, -1, 1, 1))
        b = encode_latent_bytes(y, m2.quant.step, m2.prior, extra_sigma=es, k_max=kmax)[0]
        res[tag] = b / 4
    for k, v in res.items():
        print(f"  {k:28s}: y {v:11.0f} bits/img")
    print(f"  (anchor path, 4 images; y std per-channel mean "
          f"{float(var.sqrt().mean()):.3f}, variance mean {float(var.mean()):.3f})")
