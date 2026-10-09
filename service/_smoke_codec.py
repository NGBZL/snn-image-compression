# -*- coding: utf-8 -*-
"""_smoke_codec.py —— 直接调用 codec，不经过 HTTP，先确认链路本身是对的。

    python service\\_smoke_codec.py [图片路径]
"""
from __future__ import annotations

import base64
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MODELS_DIR", os.path.join(ROOT, "models"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from PIL import Image                                       # noqa: E402
from torchvision import transforms                          # noqa: E402

import codec                                                # noqa: E402

IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "data", "flowers-102", "jpg", "image_00001.jpg")
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "runs", "_smoke")

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])


def main():
    man = codec.manifest_of(MODELS)
    print("manifest:", man)
    mh = bytes.fromhex(man["model_hash"])
    x = tf(Image.open(IMG).convert("RGB")).unsqueeze(0)
    print("输入张量", tuple(x.shape), "dtype", x.dtype, "范围", float(x.min()), float(x.max()))

    ma, meta_a = codec.load_side(os.path.join(MODELS, "a", "encoder.pth"), "a", "cpu")
    print("A 机 meta:", meta_a)

    for steps, ch in ((10, 32), (5, 16), (3, 8)):
        t0 = time.perf_counter()
        key = codec.encode_key(ma, x, steps=steps, channels=ch, model_hash=mh,
                               codec_id=codec.CODEC_RD_ANCHOR)
        enc_ms = (time.perf_counter() - t0) * 1000
        h = codec.unpack_header(key)
        print(f"steps={steps:2d} ch={ch:2d} -> {len(key):6d} B  bpp={len(key)*8/256/256:.4f}  "
              f"头={h['steps_used']}/{h['channels_used']} hash={h['model_hash'].hex()}  "
              f"encode={enc_ms:.0f} ms")
        np.save(os.path.join(OUT + f"_{steps}_{ch}.npy"), np.frombuffer(key, dtype=np.uint8))

    # 用满配钥匙走一次解码
    key = codec.encode_key(ma, x, steps=10, channels=32, model_hash=mh,
                           codec_id=codec.CODEC_RD_ANCHOR)
    mb, meta_b = codec.load_side(os.path.join(MODELS, "b", "decoder.pth"), "b", "cpu")
    print("B 机 meta:", meta_b)
    t0 = time.perf_counter()
    img = codec.decode_key(mb, key, verify_hash=mh)
    dec_ms = (time.perf_counter() - t0) * 1000
    print("重建张量", tuple(img.shape), "范围", float(img.min()), float(img.max()),
          f"decode={dec_ms:.0f} ms")

    o = x[0].permute(1, 2, 0).numpy()
    r = img[0].permute(1, 2, 0).numpy()
    mse = float(np.mean((o - r) ** 2))
    p = 99.0 if mse <= 1e-12 else 10 * np.log10(1 / mse)
    print(f"MSE={mse:.5f}  PSNR={p:.2f} dB  （anchor_v5 质量差，13 dB 上下是正常的）")

    # 故意用错 hash 的钥匙（模拟配错模型）
    bad = bytearray(key)
    bad[18:26] = bytes.fromhex("deadbeefdeadbeef")
    try:
        codec.decode_key(mb, bytes(bad), verify_hash=mh)
        print("!!! 错误钥匙竟然没被拒绝 —— 有问题")
        return 1
    except ValueError as e:
        print("错误钥匙被正确拒绝：", e)

    # 截断的钥匙
    try:
        codec.decode_key(mb, bytes(key[:10]), verify_hash=mh)
        print("!!! 截断钥匙竟然没被拒绝 —— 有问题")
        return 1
    except ValueError as e:
        print("截断钥匙被正确拒绝：", e)

    Image.fromarray((r * 255).round().clip(0, 255).astype(np.uint8)).save(
        os.path.join(ROOT, "runs", "_smoke_recon.png"))
    print("已写出 runs/_smoke_recon.png")
    return 0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    raise SystemExit(main())
