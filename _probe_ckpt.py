# -*- coding: utf-8 -*-
"""CPU-only: print checkpoint metadata (no CUDA context created)."""
import torch

for p in ["anchor_v5.pth", "anchor_v7.pth", "normab_gn_best.pth", "normab_gn.pth"]:
    try:
        o = torch.load(p, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(p, "MISSING")
        continue
    keys = ["image_size", "latent_channels", "num_steps", "z_channels",
            "use_hyperprior", "ref_wiring", "norm_type", "latent_mode", "epoch",
            "ref_mode", "group_size"]
    meta = {k: o.get(k, "<none>") for k in keys}
    sd = o.get("state_dict", {})
    bn_keys = [k for k in sd if k.startswith("bn_") and k.endswith("running_mean")]
    print("=" * 70)
    print(p, " top-level keys:", sorted(o.keys()))
    print("  meta:", meta)
    print("  n_state_dict:", len(sd), " bn running_mean keys:", len(bn_keys))
    for k in ("quant.log_step", "prior.log_sigma", "prior.mu", "ref_proj.weight"):
        if k in sd:
            t = sd[k].float()
            print(f"  {k}: shape {tuple(t.shape)} mean {t.mean():.5f} "
                  f"std {t.std():.5f} min {t.min():.5f} max {t.max():.5f}")
    sig = sd.get("prior.log_sigma")
    if sig is not None:
        s = sig.float().exp()
        print(f"  prior.sigma() -> per-channel: "
              f"min {s.min():.4f} max {s.max():.4f} mean {s.mean():.4f}")
