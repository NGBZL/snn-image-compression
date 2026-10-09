# -*- coding: utf-8 -*-
"""A 机服务（编码器半区 / 边缘端）

职责
    只持有 **编码器半区**（models/a/encoder.pth），接收一张图，产出「钥匙」字节串。
    A 机没有解码器，所以它无论如何也重建不出图像 —— 这是本系统的核心隔离点。

接口（已冻结）
    GET  /health
    POST /encode?target_kb=<float, 可选>   multipart/form-data 字段名 image
         -> {"key_b64", "key_bytes", "bpp", "steps_used", "channels_used",
             "model_hash", "encode_ms", "image_size"}

关键设计
    * model_hash 不从 encoder.pth 算（那只是半区，和原始 ckpt 的指纹不同），
      而是从 manifest.json 里读，启动时读一次缓存。容器里 /models 只挂了 models/a，
      没有 manifest.json，所以优先用环境变量 ENC_MODEL_HASH 注入。
    * target_kb -> (steps, channels)：候选 steps∈{10,5,3,2,1} × channels∈{32,24,16,8}，
      为控延迟最多真跑 MAX_PROBES 组，并带一个 (target_kb -> 结果) 的缓存。
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

import codec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("a-encoder")

# ------------------------------------------------------------------ 配置
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
ENCODER_CKPT = os.environ.get("ENC_CKPT", os.path.join(MODELS_DIR, "a", "encoder.pth"))
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", os.path.join(MODELS_DIR, "manifest.json"))
DEVICE = os.environ.get("DEVICE", "cpu")
IMAGE_SIZE = 256

# 候选推理旋钮：steps ∈ {10,5,3,2,1} × channels ∈ {32,24,16,8}
STEPS_CANDIDATES = (10, 5, 3, 2, 1)
CHANNELS_CANDIDATES = (32, 24, 16, 8)
MAX_PROBES = 4          # 为控延迟，单次请求最多真跑几组
DEFAULT_STEPS, DEFAULT_CHANNELS = 10, 32

# 「探测阶梯」：20 个候选格子按钥匙平均字节数从小到大排好（steps, channels）。
# 表由 service/_calib_knobs.py 实测得到，anchor_v5 + Flowers102 三张图上平均（kB）：
#   (1,8)1.20 (1,16)1.62 (2,8)2.03 (1,24)2.04 (1,32)2.50 (2,16)2.81 (3,8)3.03
#   (2,24)3.57 (3,16)4.20 (2,32)4.44 (5,8)4.88 (3,24)5.34 (3,32)6.64 (5,16)6.80
#   (5,24)8.70 (10,8)9.30 (5,32)10.84 (10,16)13.06 (10,24)16.81 (10,32)21.04
# 注意阶梯**不是**按 steps*channels 单调的（(3,32)=6.64 kB < (5,16)=6.80 kB），
# 所以别用乘法公式去猜，直接用实测表。换模型/换数据分布后重跑 _calib_knobs.py 更新即可。
LADDER: Tuple[Tuple[int, int], ...] = (
    (1, 8), (1, 16), (2, 8), (1, 24), (1, 32),
    (2, 16), (3, 8), (2, 24), (3, 16), (2, 32),
    (5, 8), (3, 24), (3, 32), (5, 16), (5, 24),
    (10, 8), (5, 32), (10, 16), (10, 24), (10, 32),
)
# 每个格子的实测平均大小（kB），只用于"挑哪 4 个格子去探"，不参与最终选优
LADDER_KB: Tuple[float, ...] = (
    1.20, 1.62, 2.03, 2.04, 2.50,
    2.81, 3.03, 3.57, 4.20, 4.44,
    4.88, 5.34, 6.64, 6.80, 8.70,
    9.30, 10.84, 13.06, 16.81, 21.04,
)
assert len(LADDER) == len(LADDER_KB)

_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

# 全局状态：启动时加载一次，避免每个请求都 torch.load
STATE: Dict[str, object] = {
    "model": None,
    "meta": None,
    "model_hash_hex": None,
    "load_error": None,
}
# 探测结果缓存：target_kb(量化到 0.1) -> (steps, channels, key_bytes)
_probe_cache: Dict[float, Tuple[int, int, int]] = {}
_probe_lock = threading.Lock()
# 模型前向不是线程安全的（snnTorch 的膜电位是模块内状态），串行化
_infer_lock = threading.Lock()


def _resolve_model_hash() -> Tuple[Optional[str], str]:
    """model_hash 优先环境变量，其次 manifest.json；返回 (hex, 来源说明)。"""
    env = (os.environ.get("ENC_MODEL_HASH") or "").strip()
    if env:
        return env.lower(), "env:ENC_MODEL_HASH"
    try:
        man = codec.manifest_of(os.path.dirname(MANIFEST_PATH))
        if man.get("model_hash"):
            return str(man["model_hash"]).lower(), f"manifest:{MANIFEST_PATH}"
    except Exception as e:                                  # noqa: BLE001
        return None, f"manifest 读取失败: {e}"
    return None, f"未找到 manifest（{MANIFEST_PATH}）"


def load_model():
    """加载编码器半区。失败时只记录错误，不抛 —— 让 /health 能报出来，容器不重启风暴。"""
    t0 = time.time()
    try:
        model, meta = codec.load_side(ENCODER_CKPT, "a", DEVICE)
        STATE["model"], STATE["meta"] = model, meta
        STATE["load_error"] = None
        log.info("编码器半区已加载 %s  meta=%s  用时 %.1f ms",
                 ENCODER_CKPT, meta, (time.time() - t0) * 1000)
    except Exception as e:                                   # noqa: BLE001
        STATE["model"], STATE["meta"] = None, None
        STATE["load_error"] = f"{type(e).__name__}: {e}"
        log.exception("编码器半区加载失败：%s", ENCODER_CKPT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    h, src = _resolve_model_hash()
    STATE["model_hash_hex"] = h
    log.info("model_hash=%s（来源 %s）", h, src)
    load_model()
    yield


app = FastAPI(title="SNN A/B — A 机编码器", version="1.0", lifespan=lifespan)


# ------------------------------------------------------------------ 工具
def _read_image(raw: bytes) -> torch.Tensor:
    """与训练/评测完全一致的预处理：RGB -> Resize(256) -> CenterCrop(256) -> ToTensor。"""
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"图片无法解析: {e}") from e
    return _tf(img).unsqueeze(0)


def _encode_once(model, x: torch.Tensor, steps: int, channels: int,
                 model_hash: bytes) -> Tuple[bytes, float]:
    t0 = time.perf_counter()
    with _infer_lock:
        key = codec.encode_key(model, x, steps=steps, channels=channels,
                               model_hash=model_hash, codec_id=codec.CODEC_RD_ANCHOR)
    return key, (time.perf_counter() - t0) * 1000.0


def _probe_set(target_bytes: float) -> List[Tuple[int, int]]:
    """挑最多 MAX_PROBES 个阶梯点去试探。

    取"实测平均大小"离 target 最近的 MAX_PROBES 个点（按平均大小就是单调阶梯，
    所以等价于取 target 在阶梯上的最近窗口），然后**从大到小**返回：
    真跑第一个 <= target 的点就可以停 —— 在那个窗口里它就是最接近 target 的。
    """
    tkb = target_bytes / 1024.0
    idx = int(np.searchsorted(np.asarray(LADDER_KB), tkb))
    lo = max(0, min(idx - MAX_PROBES // 2, len(LADDER) - MAX_PROBES))
    window = list(LADDER[lo:lo + MAX_PROBES])
    return sorted(window, key=lambda sc: LADDER_KB[LADDER.index(sc)], reverse=True)


def _pick_knobs(model, x: torch.Tensor, target_kb: Optional[float],
                model_hash: bytes) -> Tuple[bytes, int, int, float, List[dict]]:
    """返回 (key, steps, channels, encode_ms, 探测记录)。

    不传 target_kb -> 直接用 steps=10, channels=32（满配）。
    传了 -> 在阶梯上挑最多 MAX_PROBES 个点真跑，选 len(key) 最接近 target_kb*1024 的那组。
            同一 target_kb（量化到 0.1 kB）直接复用上次结果，不再真跑。
    """
    if target_kb is None or target_kb <= 0:
        key, ms = _encode_once(model, x, DEFAULT_STEPS, DEFAULT_CHANNELS, model_hash)
        return key, DEFAULT_STEPS, DEFAULT_CHANNELS, ms, [
            {"steps": DEFAULT_STEPS, "channels": DEFAULT_CHANNELS, "bytes": len(key)}]

    target_bytes = float(target_kb) * 1024.0
    ck = round(float(target_kb), 1)
    with _probe_lock:
        hit = _probe_cache.get(ck)
    if hit is not None:
        steps, channels, _ = hit
        key, ms = _encode_once(model, x, steps, channels, model_hash)
        log.info("target_kb=%.2f 命中缓存 -> steps=%d channels=%d", ck, steps, channels)
        return key, steps, channels, ms, [
            {"steps": steps, "channels": channels, "bytes": len(key), "cached": True}]

    probes: List[dict] = []
    best = None
    for steps, channels in _probe_set(target_bytes):
        key, ms = _encode_once(model, x, steps, channels, model_hash)
        probes.append({"steps": steps, "channels": channels, "bytes": len(key),
                       "encode_ms": round(ms, 1),
                       "ladder_kb": LADDER_KB[LADDER.index((steps, channels))]})
        diff = abs(len(key) - target_bytes)
        if best is None or diff < best[0]:
            best = (diff, steps, channels, key, ms)
        if len(key) <= target_bytes:
            # 这一档已经 <= 目标了；同一窗口里更小的档只会离目标更远
            break
    _, steps, channels, key, ms = best
    with _probe_lock:
        if len(_probe_cache) > 64:
            _probe_cache.clear()
        _probe_cache[ck] = (steps, channels, len(key))
    log.info("target_kb=%.2f -> steps=%d channels=%d bytes=%d  探测 %s",
             ck, steps, channels, len(key), probes)
    return key, steps, channels, ms, probes


# ------------------------------------------------------------------ 路由
@app.get("/health")
def health():
    return {
        "ok": STATE["model"] is not None and STATE["load_error"] is None,
        "side": "a",
        "model_hash": STATE["model_hash_hex"],
        "loaded": STATE["model"] is not None,
        "ckpt": ENCODER_CKPT,
        "load_error": STATE["load_error"],
    }


@app.post("/encode")
async def encode(image: UploadFile = File(...),
                 target_kb: Optional[float] = Query(default=None)):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503,
                            detail=f"A 机模型未加载: {STATE['load_error']}")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传的图片为空")

    x = _read_image(raw)

    mh_hex = STATE["model_hash_hex"]
    mh_bytes = bytes.fromhex(mh_hex) if mh_hex else b""
    if mh_bytes and len(mh_bytes) != 8:
        raise HTTPException(status_code=500,
                            detail=f"model_hash 必须是 8 字节的十六进制串，当前 {mh_hex!r}")

    key, steps, channels, enc_ms, probes = _pick_knobs(model, x, target_kb, mh_bytes)
    n_pix = int(x.shape[2]) * int(x.shape[3])
    return JSONResponse({
        "key_b64": base64.b64encode(key).decode("ascii"),
        "key_bytes": len(key),
        "bpp": round(len(key) * 8.0 / n_pix, 4),
        "steps_used": int(steps),
        "channels_used": int(channels),
        "model_hash": mh_hex or "",
        "encode_ms": round(enc_ms, 1),
        "image_size": int(x.shape[2]),
        "target_kb": target_kb,
        "probes": probes,
        # 便于核对：钥匙头部的 model_hash 与 manifest 是否一致
        "key_hash": codec.unpack_header(key)["model_hash"].hex(),
        "src_sha256_8": hashlib.sha256(raw).hexdigest()[:8],
    })
