# -*- coding: utf-8 -*-
"""model_io.py — 模型切分与半区加载（A 机只拿编码器，B 机只拿解码器）

切分规则
    A 机（编码器半区）
        conv1-4 / bn1-3 / gdn / quant / prior(mu,log_sigma)
        hyperprior: h_a, h_s, z_quant, z_prior
        -> 注意 h_s 两边都要：编码器要用 sigma 做熵编码，解码器要用 sigma 做熵解码。
           这是 scale-hyperprior 的固有性质，h_s 很小、也不泄露解码器信息。
    B 机（解码器半区）
        prior(mu,log_sigma) / hyperprior: h_s, z_prior / igdgn / stem / deconv1-4 / bn_d1-3

    切完之后：A 机**没有解码器**（无法重建），B 机**没有编码器和 h_a**（无法编码）。

命令行切分
    python -m codec.model_io --ckpt anchor_v5.pth --out ./models
    -> models/a/encoder.pth, models/b/decoder.pth, models/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

ENCODER_PREFIXES = (
    "conv1.", "conv2.", "conv3.", "conv4.",
    "bn1.", "bn2.", "bn3.", "gdn.", "quant.",
    "prior.",
    "hyperprior.h_a.", "hyperprior.h_s.", "hyperprior.z_quant.", "hyperprior.z_prior.",
    "latent_bias",
)
DECODER_PREFIXES = (
    "prior.",
    "hyperprior.h_s.", "hyperprior.z_prior.",
    "igdgn.", "stem.", "bn_stem.",
    "deconv1.", "deconv2.", "deconv3.", "deconv4.",
    "bn_d1.", "bn_d2.", "bn_d3.",
)


def model_hash_of(path: str, n: int = 8) -> bytes:
    """模型文件指纹（前 n 字节）。写进钥匙头，B 机用来确认配对。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()[:n]


def split_state_dict(sd: dict):
    enc = {k: v for k, v in sd.items() if k.startswith(ENCODER_PREFIXES)}
    dec = {k: v for k, v in sd.items() if k.startswith(DECODER_PREFIXES)}
    return enc, dec


def load_side(ckpt_path: str, side: str, device="cpu"):
    """side: 'a'（只加载编码器半区）或 'b'（只加载解码器半区）。"""
    from anchor import SAE_Anchor
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    enc, dec = split_state_dict(sd)
    want = enc if side == "a" else dec
    model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                       image_size=obj["image_size"], latent_mode=obj.get("latent_mode", "gaussian"),
                       use_hyperprior=obj.get("use_hyperprior", True),
                       z_channels=obj.get("z_channels", 48),
                       ref_wiring=obj.get("ref_wiring", "inp")).to(device)
    missing, unexpected = model.load_state_dict(want, strict=False)
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True
    meta = dict(side=side, loaded=len(want), total=len(sd),
                missing=len(missing), unexpected=len(unexpected),
                image_size=model.image_size, latent_channels=model.latent_channels,
                num_steps=model.num_steps, z_channels=obj.get("z_channels", 48),
                ref_wiring=model.ref_wiring, use_hyperprior=model.use_hyperprior)
    return model, meta


def manifest_of(models_dir: str) -> dict:
    p = os.path.join(models_dir, "manifest.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser(description="把训练好的 checkpoint 切成 A/B 两个半区")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="./models")
    ap.add_argument("--codec-id", type=int, default=3, help="3=RD+锚图输入端拼接")
    a = ap.parse_args()

    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    enc, dec = split_state_dict(sd)
    mh = model_hash_of(a.ckpt)

    os.makedirs(os.path.join(a.out, "a"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "b"), exist_ok=True)
    torch.save({"state_dict": enc, "image_size": obj["image_size"],
                "latent_channels": obj["latent_channels"], "num_steps": obj["num_steps"],
                "z_channels": obj.get("z_channels", 48),
                "latent_mode": obj.get("latent_mode", "gaussian"),
                "use_hyperprior": obj.get("use_hyperprior", True),
                "ref_wiring": obj.get("ref_wiring", "inp")},
               os.path.join(a.out, "a", "encoder.pth"))
    torch.save({"state_dict": dec, "image_size": obj["image_size"],
                "latent_channels": obj["latent_channels"], "num_steps": obj["num_steps"],
                "z_channels": obj.get("z_channels", 48),
                "latent_mode": obj.get("latent_mode", "gaussian"),
                "use_hyperprior": obj.get("use_hyperprior", True),
                "ref_wiring": obj.get("ref_wiring", "inp")},
               os.path.join(a.out, "b", "decoder.pth"))

    man = {"source": os.path.basename(a.ckpt), "model_hash": mh.hex(),
           "codec_id": a.codec_id, "image_size": obj["image_size"],
           "latent_channels": obj["latent_channels"], "num_steps": obj["num_steps"],
           "z_channels": obj.get("z_channels", 48),
           "ref_wiring": obj.get("ref_wiring", "inp"),
           "encoder_params": len(enc), "decoder_params": len(dec)}
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    print(f"模型指纹 {mh.hex()}")
    print(f"A 机 encoder.pth : {len(enc)} 个参数张量")
    print(f"B 机 decoder.pth : {len(dec)} 个参数张量")
    only_a = sorted(set(enc) - set(dec))
    only_b = sorted(set(dec) - set(enc))
    print(f"A 独有 {len(only_a)} 个（B 拿不到），例如 {only_a[:3]}")
    print(f"B 独有 {len(only_b)} 个（A 拿不到），例如 {only_b[:3]}")
    print(f"已写入 {a.out}/manifest.json")


if __name__ == "__main__":
    main()
