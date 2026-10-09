# -*- coding: utf-8 -*-
"""_calib_knobs.py —— 量一遍 (steps, channels) 候选格子的钥匙字节数。

用来给 A 机的码率控制挑一条"探测阶梯"：阶梯上的点越大越好用，
因为渲染一条阶梯给前端下拉框/滑块用很直观，也决定了 target_kb 能被覆盖到什么程度。

    python service\\_calib_knobs.py [图片路径 ...]
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MODELS_DIR", os.path.join(ROOT, "models"))

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402
from torchvision import transforms                          # noqa: E402

import codec                                                # noqa: E402

MODELS = os.path.join(ROOT, "models")
STEPS = (10, 5, 3, 2, 1)
CHANNELS = (32, 24, 16, 8)

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])

imgs = sys.argv[1:]
if not imgs:
    d = os.path.join(ROOT, "data", "flowers-102", "jpg")
    imgs = [os.path.join(d, f"image_{i:05d}.jpg") for i in (1, 2, 3)]

man = codec.manifest_of(MODELS)
mh = bytes.fromhex(man["model_hash"])
ma, _ = codec.load_side(os.path.join(MODELS, "a", "encoder.pth"), "a", "cpu")

print(f"{'steps':>5} {'ch':>3} " + " ".join(f"{os.path.basename(p)[:12]:>13}" for p in imgs)
      + f" {'mean_kB':>8} {'enc_ms':>7}")
rows = {}
for s in STEPS:
    for c in CHANNELS:
        sizes, ms = [], []
        for p in imgs:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            t0 = time.perf_counter()
            k = codec.encode_key(ma, x, steps=s, channels=c, model_hash=mh,
                                 codec_id=codec.CODEC_RD_ANCHOR)
            ms.append((time.perf_counter() - t0) * 1000)
            sizes.append(len(k))
        rows[(s, c)] = float(np.mean(sizes))
        print(f"{s:>5} {c:>3} " + " ".join(f"{v/1024:>13.2f}" for v in sizes)
              + f" {np.mean(sizes)/1024:>8.2f} {np.mean(ms):>7.0f}")

print("\n按平均大小排序（kB）：")
for (s, c), v in sorted(rows.items(), key=lambda kv: kv[1]):
    print(f"  steps={s:2d} ch={c:2d}  {v/1024:7.2f} kB")
