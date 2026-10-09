# -*- coding: utf-8 -*-
"""Verification for fix #5: bit-exact encode/decode round-trip (CPU only)."""
import argparse
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from anchor import SAE_Anchor, build_data, _norm_of  # noqa: E402
from codec.rd_codec import encode_key, decode_key, _decode_latents, unpack_header  # noqa: E402
from sae_rd import _SIGMA_GRID, _bucket_ids  # noqa: E402

DEV = torch.device("cpu")
PASS = True


def check(name, ok, detail=""):
    global PASS
    PASS = PASS and bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def build(ckpt="anchor_v11.pth" if False else "anchor_v5.pth"):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                   image_size=obj["image_size"], latent_mode="gaussian",
                   use_hyperprior=obj.get("use_hyperprior", True),
                   z_channels=obj.get("z_channels", 48), norm_type=_norm_of(obj),
                   ref_wiring=obj.get("ref_wiring", "out")).to(DEV)
    m.load_state_dict(obj["state_dict"])
    m.eval()
    for mm in m.modules():
        if hasattr(mm, "hard"):
            mm.hard = True
    return m, obj


model, obj = build()
args = argparse.Namespace(image_size=256, augment="none", data_dir="./data",
                          holdout_frac=0.02, seed=42, eval_limit=8)
_, te = build_data(args)

for tag, size in (("native 256px", 256), ("mismatched 128px (the old eval case)", 128)):
    x = torch.nn.functional.interpolate(te[0][0].unsqueeze(0), size=(size, size),
                                        mode="bilinear", align_corners=False).to(DEV)
    print(f"\n=== {tag} (model.image_size={model.image_size}, latent_size={model.latent_size}) ===")
    with torch.no_grad():
        y_enc, rate, rec_enc = model(x, ref=None, checkpoints={model.num_steps})
        recon_enc = rec_enc[model.num_steps]
        key = encode_key(model, x, model_hash=b"testhash")
        y_dec, h = _decode_latents(model, key)
        recon_dec = decode_key(model, key)
    lat_hw = int(h["image_size"]) // 16
    print(f"  header: image_size={h['image_size']} steps={h['steps_used']} "
          f"channels={h['channels_used']} kmax={h['kmax']} "
          f"z_len={h['z_len']} y_len={h['y_len']} key={len(key)} B")
    print(f"  latent shapes: enc {tuple(y_enc.shape)}  dec {tuple(y_dec.shape)}  "
          f"(expected latent_hw={lat_hw}, hz={max(1, lat_hw // 4)})")
    d_lat = float((y_enc - y_dec).abs().max())
    d_img = float((recon_enc - recon_dec).abs().max())
    print(f"  max |latent diff| = {d_lat:.3e}   max |image diff| = {d_img:.3e}")
    check(f"{tag}: decoded latent == encoder latent", d_lat == 0.0, f"max {d_lat:.3e}")
    check(f"{tag}: decoded image == encoder reconstruction (<1e-6)", d_img < 1e-6,
          f"max {d_img:.3e}")

    # what the OLD decode-side sigma formula would have produced
    with torch.no_grad():
        zq = torch.round(rate["z_hat"][:h["steps_used"]].detach())
        zq = zq.clamp(-128, 128).long()
        new_extra = model.hyperprior.sigma_from_z(zq.float())
        old_extra = model.hyperprior.h_s(zq.float()).clamp(-8.0, 8.0).exp().clamp(1e-3, 1e3)
        sig_new = model.prior.sigma(new_extra).detach().cpu()
        sig_old = (model.prior.sigma().detach().cpu() * old_extra.detach().cpu())
        sig_old = sig_old.clamp(1e-3, 1e3)
        a = _bucket_ids(sig_new[:h["steps_used"], :h["channels_used"]]
                        .reshape(-1).numpy().astype(np.float64))
        b = _bucket_ids(sig_old.reshape(-1).numpy().astype(np.float64))
    n = min(a.size, b.size)
    mism = int((a[:n] != b[:n]).sum())
    print(f"  OLD decode formula vs encode: {mism}/{n} pixels land in a different bucket "
          f"({100*mism/n:.2f}%)  [now 0 after the fix]")
    print(f"  sigma field: p99={float(np.percentile(sig_new.numpy(), 99)):.1f} "
          f"max={float(sig_new.max()):.1f}; h_s out-of-±8 would clamp "
          f"{float((model.hyperprior.h_s(zq.float()).abs() > 8).float().mean())*100:.2f}% of pixels")

print("\n=== truncation knobs (steps / channels) still round-trip on the coded part ===")
x = te[1][0].unsqueeze(0).to(DEV)
with torch.no_grad():
    y_enc, _r, _rec = model(x, ref=None, checkpoints={model.num_steps})
    key = encode_key(model, x, steps=6, channels=20)
    y_dec, h = _decode_latents(model, key)
d = float((y_enc[:, :6, :20] - y_dec[:, :6, :20]).abs().max())
print(f"  steps=6 channels=20: coded-part max |latent diff| = {d:.3e}")
check("partial key round-trips the coded sub-tensor", d == 0.0, f"max {d:.3e}")

print("\n" + ("ALL FIX-5 CHECKS PASSED" if PASS else "SOME FIX-5 CHECKS FAILED"))
sys.exit(0 if PASS else 1)
