# -*- coding: utf-8 -*-
"""_show_history.py —— 通过 Gateway 拉一次历史记录，打印成表（验证用）。

    python service\\_show_history.py [gateway_url]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

GW = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows = json.loads(urllib.request.urlopen(f"{GW}/api/history?limit=50", timeout=30).read())
print(f"Gateway {GW} 返回 {len(rows)} 条：")
print(f"{'时间(带时区)':30s} {'字节':>7s} {'bpp':>7s} {'PSNR':>6s} {'SSIM':>7s} "
      f"{'编码ms':>8s} {'解码ms':>8s}  steps×ch")
for r in rows:
    print(f"{r['ts']:30s} {r['key_bytes']:7d} {r['bpp']:7.4f} {r['psnr']:6.2f} "
          f"{r['ssim']:7.4f} {r['encode_ms']:8.1f} {r['decode_ms']:8.1f}  "
          f"{r['steps_used']}×{r['channels_used']}")

idx = os.path.join(ROOT, "runs", "index.jsonl")
n = sum(1 for ln in open(idx, encoding="utf-8") if ln.strip())
print(f"\nruns/index.jsonl 共 {n} 行")
