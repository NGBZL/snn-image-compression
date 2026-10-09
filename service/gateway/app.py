# -*- coding: utf-8 -*-
"""Gateway —— 唯一对外入口

职责
    1. POST /api/run：收原图 -> 调 A 机 /encode -> 调 B 机 /decode -> 在**这里**算 PSNR/SSIM
       （只有 Gateway 同时握有原图和重建图，A 机看不到重建、B 机看不到原图）。
    2. 每次运行落盘：runs/<run_id>/{orig.png, recon.png, key.bin, meta.json}
       + 追加一行到 runs/index.jsonl。
    3. GET / 返回单文件前端 static/index.html。

环境变量
    A_URL  默认 http://a-encoder:8000
    B_URL  默认 http://b-decoder:8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("gateway")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.environ.get("STATIC_DIR", os.path.join(HERE, "static"))
RUNS_DIR = os.path.abspath(os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(HERE), "runs")))
A_URL = os.environ.get("A_URL", "http://a-encoder:8000").rstrip("/")
B_URL = os.environ.get("B_URL", "http://b-decoder:8000").rstrip("/")
TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "120"))
IMAGE_SIZE = 256
HISTORY_LIMIT = 50

# Gateway 自己也做一遍同样的预处理：A 机拿到的就是这张 256x256 的图，
# 用它当 PSNR/SSIM 的参考才算"同一张图"。为了不把 torch 拖进 Gateway 镜像，
# 这里用 PIL 的 LANCZOS（≈ torchvision Resize 默认的 antialias 双线性）代替：
# 预处理细节的差异只影响 PSNR 小数点后第二位（重建误差 0.2 量级远大于它）。
_RESAMPLE = Image.LANCZOS

# PSNR/SSIM 用的 11x11 高斯窗（sigma=1.5），与 skimage 默认口径一致
_GAUSS_1D = np.exp(-((np.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
_GAUSS_1D /= _GAUSS_1D.sum()
_GAUSS_2D = np.outer(_GAUSS_1D, _GAUSS_1D)
C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

_client: Optional[httpx.Client] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    os.makedirs(RUNS_DIR, exist_ok=True)
    _client = httpx.Client(timeout=TIMEOUT)
    log.info("A_URL=%s  B_URL=%s  RUNS_DIR=%s", A_URL, B_URL, RUNS_DIR)
    try:
        yield
    finally:
        if _client is not None:
            _client.close()
            _client = None


app = FastAPI(title="SNN A/B — Gateway", version="1.0", lifespan=lifespan)


# ------------------------------------------------------------------ 质量指标
def _to_gray_255(arr: np.ndarray) -> np.ndarray:
    """[H,W,3] uint8 -> [H,W] float 灰度（0~255）。"""
    return (arr.astype(np.float64) @ np.array([0.299, 0.587, 0.114]))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """a/b: [H,W,3] uint8。走 RGB 三通道 MSE，峰值 255。"""
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """**简化版** SSIM：11x11 高斯窗（sigma=1.5），在灰度图上算，C1/C2 用标准常数。

    与 skimage.metrics.structural_similarity(gaussian_weights=True) 口径一致，
    但没做样本协方差的无偏修正（use_sample_covariance=True 那份 1/(N-1) 归一）。
    对这里的用途（横向比不同码率）足够，别拿它当论文指标。
    """
    ga, gb = _to_gray_255(a), _to_gray_255(b)
    pad = 5
    gaf = np.pad(ga, pad, mode="reflect")
    gbf = np.pad(gb, pad, mode="reflect")

    def blur(img):                                          # 11x11 高斯：两次一维卷积
        t = np.apply_along_axis(lambda m: np.convolve(m, _GAUSS_1D, mode="valid"), 0, img)
        return np.apply_along_axis(lambda m: np.convolve(m, _GAUSS_1D, mode="valid"), 1, t)

    mu_a, mu_b = blur(gaf), blur(gbf)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a2 = blur(gaf * gaf) - mu_a2
    sig_b2 = blur(gbf * gbf) - mu_b2
    sig_ab = blur(gaf * gbf) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


# ------------------------------------------------------------------ 存盘
def _save_run(run_id: str, orig_png: bytes, recon_png: bytes, key: bytes,
              record: dict) -> str:
    d = os.path.join(RUNS_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "orig.png"), "wb") as f:
        f.write(orig_png)
    with open(os.path.join(d, "recon.png"), "wb") as f:
        f.write(recon_png)
    with open(os.path.join(d, "key.bin"), "wb") as f:
        f.write(key)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RUNS_DIR, "index.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return d


def _preprocess(raw: bytes) -> Image.Image:
    """与 A 机等价的预处理：RGB -> Resize(256) -> CenterCrop(256)。"""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = IMAGE_SIZE / min(w, h)
    if (w, h) != (IMAGE_SIZE, IMAGE_SIZE):
        im = im.resize((max(IMAGE_SIZE, int(round(w * s))), max(IMAGE_SIZE, int(round(h * s)))),
                       _RESAMPLE)
    w, h = im.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    return im.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ 路由
@app.get("/health")
def health():
    out = {"ok": True, "runs_dir": RUNS_DIR, "a_url": A_URL, "b_url": B_URL,
           "a": None, "b": None}
    for name, url in (("a", A_URL), ("b", B_URL)):
        try:
            r = _client.get(f"{url}/health", timeout=10.0) if _client else None
            out[name] = r.json() if r is not None and r.status_code == 200 else \
                {"ok": False, "status": None if r is None else r.status_code}
        except Exception as e:                               # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out["ok"] = bool(out["a"] and out["a"].get("ok") and out["b"] and out["b"].get("ok"))
    return out


@app.get("/api/config")
def config():
    return {"a_url": A_URL, "b_url": B_URL, "image_size": IMAGE_SIZE}


@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(p):
        raise HTTPException(status_code=500, detail=f"前端文件缺失: {p}")
    with open(p, encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.post("/api/run")
async def run(image: UploadFile = File(...),
              target_kb: Optional[float] = Form(default=None)):
    if _client is None:
        raise HTTPException(status_code=503, detail="Gateway 尚未就绪")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传的图片为空")
    try:
        _preprocess(raw)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"图片无法解析: {e}") from e

    # 复用 A 机的图片字节（原样转发），A 机自己会做 256 预处理
    try:
        a_resp = await asyncio.to_thread(
            _client.post, f"{A_URL}/encode", params=({"target_kb": target_kb}
                                                     if target_kb else None),
            files={"image": (image.filename or "upload.png", raw,
                             image.content_type or "application/octet-stream")})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"调用 A 机失败: {type(e).__name__}: {e}")
    if a_resp.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"A 机返回 {a_resp.status_code}: {a_resp.text[:500]}")
    a = a_resp.json()
    key = base64.b64decode(a["key_b64"])

    try:
        b_resp = await asyncio.to_thread(
            _client.post, f"{B_URL}/decode", json={"key_b64": a["key_b64"]})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"调用 B 机失败: {type(e).__name__}: {e}")
    if b_resp.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"B 机返回 {b_resp.status_code}: {b_resp.text[:500]}")
    recon_png = b_resp.content

    model_hash = a.get("model_hash") or b_resp.headers.get("X-Model-Hash") or ""

    # ---- 指标：Gateway 侧同一张 256x256 前处理图当参考
    orig = _preprocess(raw)
    orig_png = _png_bytes(orig)
    o = np.asarray(orig, dtype=np.uint8)
    recon = Image.open(io.BytesIO(recon_png)).convert("RGB")
    if recon.size != orig.size:
        recon = recon.resize(orig.size, Image.BILINEAR)
    r = np.asarray(recon, dtype=np.uint8)

    p_val, s_val = psnr(o, r), ssim(o, r)
    n_pix = int(o.shape[0]) * int(o.shape[1])
    decode_ms = float(b_resp.headers.get("X-Decode-Ms", 0.0) or 0.0)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    # 带时区的 ISO 时间戳：宿主机（本地时区）和容器（UTC）写进同一个
    # index.jsonl 时，不带时区的话时间戳会互相矛盾、没法排序。
    now = datetime.now().astimezone()
    record = {
        "run_id": run_id,
        "ts": now.isoformat(timespec="seconds"),
        "ts_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "filename": image.filename or "upload.png",
        "target_kb": target_kb,
        "key_bytes": int(a["key_bytes"]),
        "bpp": float(a["bpp"]),
        "steps_used": int(a["steps_used"]),
        "channels_used": int(a["channels_used"]),
        "model_hash": model_hash,
        "encode_ms": float(a["encode_ms"]),
        "decode_ms": round(decode_ms, 1),
        "psnr": round(p_val, 2),
        "ssim": round(s_val, 4),
        "image_size": int(o.shape[0]),
        "orig_url": f"/runs/{run_id}/orig.png",
        "recon_url": f"/runs/{run_id}/recon.png",
        "probes": a.get("probes", []),
        "metric_note": "PSNR 走 RGB uint8、峰值 255；SSIM 为 11x11 高斯窗简化实现（灰度），非训练口径",
    }
    await asyncio.to_thread(_save_run, run_id, orig_png, recon_png, key, record)
    log.info("run %s: %d B  %.1f dB  ssim=%.4f  enc=%.0fms dec=%.0fms",
             run_id, record["key_bytes"], p_val, s_val, record["encode_ms"], decode_ms)
    return JSONResponse(record)


@app.get("/runs/{run_id}/{name}")
def get_run_file(run_id: str, name: str):
    if name not in ("orig.png", "recon.png", "key.bin", "meta.json"):
        raise HTTPException(status_code=404, detail="不允许的文件名")
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="非法 run_id")
    p = os.path.join(RUNS_DIR, run_id, name)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail=f"不存在: {run_id}/{name}")
    media = {"orig.png": "image/png", "recon.png": "image/png",
             "key.bin": "application/octet-stream",
             "meta.json": "application/json"}[name]
    return FileResponse(p, media_type=media)


@app.get("/api/history")
def history(limit: int = HISTORY_LIMIT):
    """最近 N 条记录，从 runs/index.jsonl 读（新的在前）。"""
    p = os.path.join(RUNS_DIR, "index.jsonl")
    if not os.path.isfile(p):
        return []
    rows: List[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = rows[-max(1, min(int(limit), 500)):]
    rows.reverse()
    return rows


@app.get("/api/history/{run_id}")
def history_one(run_id: str):
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="非法 run_id")
    p = os.path.join(RUNS_DIR, run_id, "meta.json")
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail=f"没有这次运行: {run_id}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)
