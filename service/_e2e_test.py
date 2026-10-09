# -*- coding: utf-8 -*-
"""_e2e_test.py —— 端到端验收脚本（宿主机直测和 Docker 都跑同一份）

    # 宿主机
    python service\\_e2e_test.py --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002 ^
        --gw-url http://127.0.0.1:8000
    # Docker
    python service\\_e2e_test.py --a-url http://127.0.0.1:8080/..

检查项
    1. A/B/Gateway 的 /health
    2. 直接打 A 机 /encode（不传 target_kb 与传 target_kb 各一次）
    3. 直接把钥匙丢给 B 机 /decode，验证 PNG 与响应头
    4. 走 Gateway /api/run 全链路，验证 PSNR/SSIM 落盘
    5. 故意传坏钥匙（改指纹 / 截断 / 乱 base64）必须 400
    6. '/' 返回 HTML、'/api/history' 有记录、runs 目录有文件
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import httpx
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def png_size(b: bytes):
    im = Image.open(io.BytesIO(b))
    return im.size, im.mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-url", default=os.environ.get("A_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--b-url", default=os.environ.get("B_URL", "http://127.0.0.1:8002"))
    ap.add_argument("--gw-url", default=os.environ.get("GW_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    a = ap.parse_args()

    img = a.image or os.path.join(ROOT, "data", "flowers-102", "jpg", "image_00042.jpg")
    raw = open(img, "rb").read()
    print(f"测试图片: {img}  ({len(raw)} B)\n")

    c = httpx.Client(timeout=180.0)

    # ---------------- 1. health
    print("== 1. /health ==")
    ha = c.get(f"{a.a_url}/health").json()
    hb = c.get(f"{a.b_url}/health").json()
    hg = c.get(f"{a.gw_url}/health").json()
    print("  A:", json.dumps(ha, ensure_ascii=False))
    print("  B:", json.dumps(hb, ensure_ascii=False))
    print("  GW:", json.dumps(hg, ensure_ascii=False))
    check("A /health ok", ha.get("ok") is True and ha.get("side") == "a")
    check("B /health ok", hb.get("ok") is True and hb.get("side") == "b")
    check("GW /health 两个上游都 ok", hg.get("ok") is True)
    mh = ha.get("model_hash")
    check("A/B model_hash 一致", mh == hb.get("model_hash"), f"{mh} vs {hb.get('model_hash')}")
    check("model_hash == manifest", mh == "1a5f6470c5592f19", str(mh))

    # ---------------- 2. A /encode
    print("\n== 2. A 机 /encode ==")
    t0 = time.perf_counter()
    r = c.post(f"{a.a_url}/encode", files={"image": ("t.jpg", raw, "image/jpeg")})
    wall = (time.perf_counter() - t0) * 1000
    check("不带 target_kb -> 200", r.status_code == 200, f"HTTP {r.status_code} {r.text[:200]}")
    j = r.json()
    print("  返回:", json.dumps({k: v for k, v in j.items() if k != "key_b64"},
                                ensure_ascii=False))
    for k in ("key_b64", "key_bytes", "bpp", "steps_used", "channels_used",
              "model_hash", "encode_ms", "image_size"):
        check(f"字段 {k} 存在", k in j)
    check("不传 target_kb 时 steps=10 channels=32",
          j["steps_used"] == 10 and j["channels_used"] == 32,
          f"{j['steps_used']}x{j['channels_used']}")
    check("image_size == 256", j["image_size"] == 256)
    check("key_bytes 与 base64 长度自洽",
          len(j["key_b64"]) * 3 // 4 - j["key_bytes"] in (0, 1, 2),
          f"b64={len(j['key_b64'])} bytes={j['key_bytes']}")
    check("bpp 自洽 (bytes*8/256^2)",
          abs(j["bpp"] - j["key_bytes"] * 8 / 256 / 256) < 5e-4,
          f"bpp={j['bpp']} 期望={j['key_bytes']*8/256/256:.4f}")
    check("encode_ms 合理 (<60 s)", 0 < j["encode_ms"] < 60000, f"{j['encode_ms']} ms")
    key_full = j["key_b64"]

    # 带 target_kb
    targets = {}
    for kb in (2, 5, 10):
        rr = c.post(f"{a.a_url}/encode", params={"target_kb": kb},
                    files={"image": ("t.jpg", raw, "image/jpeg")})
        check(f"target_kb={kb} -> 200", rr.status_code == 200, rr.text[:200])
        jj = rr.json()
        targets[kb] = jj
        err = abs(jj["key_bytes"] - kb * 1024) / (kb * 1024) * 100
        print(f"  target_kb={kb:5.1f} -> {jj['key_bytes']:6d} B ({jj['key_bytes']/1024:6.2f} kB) "
              f"steps={jj['steps_used']:2d} ch={jj['channels_used']:2d} "
              f"bpp={jj['bpp']:.4f} 误差={err:5.1f}% 编码={jj['encode_ms']}ms "
              f"探测={[(p['steps'], p['channels'], p['bytes']) for p in jj.get('probes', [])]}")
        check(f"target_kb={kb} 误差 < 30%", err < 30, f"{err:.1f}%")

    # 缓存命中
    rr = c.post(f"{a.a_url}/encode", params={"target_kb": 5},
                files={"image": ("t.jpg", raw, "image/jpeg")})
    check("同一 target_kb 第二次命中缓存",
          any(p.get("cached") for p in rr.json().get("probes", [])),
          str(rr.json().get("probes")))

    # ---------------- 3. B /decode
    print("\n== 3. B 机 /decode ==")
    r = c.post(f"{a.b_url}/decode", json={"key_b64": key_full})
    check("合法钥匙 -> 200", r.status_code == 200, f"HTTP {r.status_code} {r.text[:300]}")
    check("Content-Type 是 image/png", r.headers.get("content-type", "").startswith("image/png"),
          r.headers.get("content-type"))
    for h in ("X-Model-Hash", "X-Decode-Ms", "X-Key-Bytes", "X-Steps-Used", "X-Channels-Used"):
        check(f"响应头 {h}", h in r.headers, r.headers.get(h))
    check("X-Key-Bytes 与 A 机报告一致",
          int(r.headers["X-Key-Bytes"]) == j["key_bytes"],
          f"{r.headers['X-Key-Bytes']} vs {j['key_bytes']}")
    check("X-Steps-Used/Channels == 10/32",
          r.headers["X-Steps-Used"] == "10" and r.headers["X-Channels-Used"] == "32")
    sz, mode = png_size(r.content)
    check("PNG 是 256x256 RGB", sz == (256, 256) and mode == "RGB", f"{sz} {mode}")
    check("X-Model-Hash 与 manifest 一致", r.headers["X-Model-Hash"] == mh)
    recon_b = r.content
    dd = np.asarray(Image.open(io.BytesIO(recon_b)).convert("RGB"), dtype=np.float64)
    check("重建图不是全黑/全白（像素有变化）", float(dd.std()) > 1.0, f"std={dd.std():.2f}")

    # ---------------- 4. 坏钥匙必须 400
    print("\n== 4. 坏钥匙必须被 400 拒绝 ==")
    import base64 as b64
    kb_ = bytearray(b64.b64decode(key_full))
    bad_hash = bytearray(kb_)
    bad_hash[18:26] = bytes.fromhex("deadbeefdeadbeef")
    r = c.post(f"{a.b_url}/decode",
               json={"key_b64": b64.b64encode(bytes(bad_hash)).decode()})
    check("指纹对不上的钥匙 -> 400", r.status_code == 400, f"HTTP {r.status_code}")
    check("400 里带 detail 且说明指纹不一致",
          r.headers.get("content-type", "").startswith("application/json")
          and "指纹" in r.json().get("detail", ""),
          r.text[:200])

    r = c.post(f"{a.b_url}/decode", json={"key_b64": b64.b64encode(bytes(kb_[:10])).decode()})
    check("截断的钥匙 -> 400", r.status_code == 400, f"HTTP {r.status_code} {r.text[:150]}")

    r = c.post(f"{a.b_url}/decode", json={"key_b64": "!!!not-base64!!!"})
    check("非法 base64 -> 400", r.status_code == 400, f"HTTP {r.status_code} {r.text[:150]}")

    r = c.post(f"{a.b_url}/decode", json={})
    check("缺 key_b64 -> 400", r.status_code == 400, f"HTTP {r.status_code} {r.text[:150]}")

    r = c.post(f"{a.b_url}/decode", content=b"not json",
               headers={"content-type": "application/json"})
    check("请求体不是 JSON -> 400", r.status_code == 400, f"HTTP {r.status_code}")

    # ---------------- 5. Gateway /api/run
    print("\n== 5. Gateway /api/run 全链路 ==")
    for kb in (5, None):
        files = {"image": ("t.jpg", raw, "image/jpeg")}
        data = {} if kb is None else {"target_kb": str(kb)}
        t0 = time.perf_counter()
        r = c.post(f"{a.gw_url}/api/run", files=files, data=data)
        wall = (time.perf_counter() - t0) * 1000
        check(f"/api/run target_kb={kb} -> 200", r.status_code == 200,
              f"HTTP {r.status_code} {r.text[:300]}")
        if r.status_code != 200:
            continue
        g = r.json()
        print("  返回:", json.dumps(g, ensure_ascii=False)[:600])
        for k in ("key_bytes", "bpp", "steps_used", "channels_used", "model_hash",
                  "encode_ms", "decode_ms", "psnr", "ssim", "run_id",
                  "recon_url", "orig_url"):
            check(f"/api/run 字段 {k}", k in g)
        check("PSNR 在合理区间 (8~40 dB)", 8 < g["psnr"] < 40, f"{g['psnr']} dB")
        check("SSIM 在 0~1", 0.0 <= g["ssim"] <= 1.0, str(g["ssim"]))
        check("model_hash 一致", g["model_hash"] == mh, str(g["model_hash"]))
        check(f"往返耗时 < 120 s", wall < 120000, f"{wall:.0f} ms")

        o = c.get(f"{a.gw_url}{g['orig_url']}")
        rr = c.get(f"{a.gw_url}{g['recon_url']}")
        check("orig.png 可取到 256x256", o.status_code == 200 and png_size(o.content)[0] == (256, 256),
              f"HTTP {o.status_code} {png_size(o.content)[0] if o.status_code == 200 else ''}")
        check("recon.png 可取到 256x256",
              rr.status_code == 200 and png_size(rr.content)[0] == (256, 256),
              f"HTTP {rr.status_code}")

        # 独立复算 PSNR，确认 Gateway 报的数没错
        A = np.asarray(Image.open(io.BytesIO(o.content)).convert("RGB"), dtype=np.float64) / 255.0
        B = np.asarray(Image.open(io.BytesIO(rr.content)).convert("RGB"), dtype=np.float64) / 255.0
        mse = float(np.mean((A - B) ** 2))
        ps = 99.0 if mse <= 1e-12 else 10 * np.log10(1 / mse)
        check("PSNR 独立复算一致", abs(ps - g["psnr"]) < 0.05,
              f"复算 {ps:.2f} vs 上报 {g['psnr']}")

        run_dir = os.path.join(a.runs_dir, g["run_id"])
        check("runs/<id>/ 三个文件都落盘",
              all(os.path.isfile(os.path.join(run_dir, f))
                  for f in ("orig.png", "recon.png", "key.bin")),
              run_dir)
        kb_path = os.path.join(run_dir, "key.bin")
        if os.path.isfile(kb_path):
            check("key.bin 大小 == key_bytes",
                  os.path.getsize(kb_path) == g["key_bytes"],
                  f"{os.path.getsize(kb_path)} vs {g['key_bytes']}")
            # 把落盘的钥匙直接丢给 B 机，应当还原出同一张图
            k2 = b64.b64encode(open(kb_path, "rb").read()).decode()
            r2 = c.post(f"{a.b_url}/decode", json={"key_b64": k2})
            check("落盘 key.bin 让 B 机还原出完全相同的 PNG",
                  r2.status_code == 200 and r2.content == rr.content)

    # ---------------- 6. 前端 & 历史
    print("\n== 6. 前端页面 / 历史记录 ==")
    r = c.get(f"{a.gw_url}/")
    check("GET / -> 200", r.status_code == 200, f"HTTP {r.status_code}")
    check("HTML 含页面标题", "SNN 图像压缩" in r.text and "A/B 机分离" in r.text)
    check("HTML 是单文件（无外部 js/css 依赖）",
          "<script src=" not in r.text and "<link rel=\"stylesheet\"" not in r.text)
    check("HTML 含关键控件 id", all(s in r.text for s in
                                    ('id="drop"', 'id="kb"', 'id="go"', 'id="hist"')))

    r = c.get(f"{a.gw_url}/api/history")
    check("GET /api/history -> 200", r.status_code == 200)
    hist = r.json()
    check("history 是非空列表", isinstance(hist, list) and len(hist) > 0, f"{len(hist)} 条")
    if hist:
        print("  最近一条:", json.dumps(hist[0], ensure_ascii=False)[:400])
        check("history 字段齐全",
              all(k in hist[0] for k in ("run_id", "ts", "key_bytes", "bpp", "psnr",
                                         "encode_ms", "decode_ms")))
        # 排序口径：Gateway 是「按行读 index.jsonl -> 取最后 50 条 -> reverse()」，
        # 也就是**文件插入顺序的倒序**。这里直接照着 index.jsonl 复算一遍来比对，
        # 比拿 ts 字符串比较靠谱 —— 宿主机和容器可能在不同时区，同一秒内也会跑多次。
        idx = os.path.join(a.runs_dir, "index.jsonl")
        if os.path.isfile(idx):
            with open(idx, encoding="utf-8") as f:
                lines = [ln for ln in (l.strip() for l in f) if ln]
            expect = list(reversed(lines[-50:]))
            got = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in hist]
            check("history == index.jsonl 末尾 50 条的倒序",
                  got == [json.dumps(json.loads(l), ensure_ascii=False, sort_keys=True)
                          for l in expect],
                  f"history {len(hist)} 条 / 文件 {len(lines)} 行")
        check("history 时间戳带时区（跨时区也能排）",
              all(("+" in r["ts"] or r["ts"].endswith("Z")) for r in hist),
              str([r["ts"] for r in hist[:2]]))

    # ---------------- 汇总
    print("\n" + "=" * 70)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  -", f)
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
