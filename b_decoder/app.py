# -*- coding: utf-8 -*-
"""B 机服务（解码器半区 / 云侧）

职责
    只持有 **解码器半区**（models/b/decoder.pth），收到钥匙字节串，还原成 PNG。
    B 机没有编码器、也没有 hyperprior 的 h_a，所以它无法自己造钥匙。

接口（已冻结）
    GET  /health
    POST /decode   application/json {"key_b64": "..."}
         -> 200 image/png，附响应头 X-Model-Hash / X-Decode-Ms / X-Key-Bytes /
            X-Steps-Used / X-Channels-Used
         -> 400 {"detail": "..."}（钥匙坏了 / 模型指纹不匹配）

注意
    verify_hash 用的是 manifest 里的 model_hash（原始 ckpt 指纹）。
    钥匙头的指纹是 A 机写进去的，两边不一致时 decode_key 抛 ValueError，
    我们原样把错误信息带出去 —— 宁可明确报错，也不要静默还原出一张乱码。
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

import codec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("b-decoder")

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
DECODER_CKPT = os.environ.get("DEC_CKPT", os.path.join(MODELS_DIR, "b", "decoder.pth"))
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", os.path.join(MODELS_DIR, "manifest.json"))
DEVICE = os.environ.get("DEVICE", "cpu")

STATE: Dict[str, object] = {
    "model": None,
    "meta": None,
    "model_hash_hex": None,
    "load_error": None,
}
_infer_lock = threading.Lock()


def _resolve_model_hash() -> Optional[str]:
    env = (os.environ.get("DEC_MODEL_HASH") or os.environ.get("ENC_MODEL_HASH") or "").strip()
    if env:
        return env.lower()
    try:
        man = codec.manifest_of(os.path.dirname(MANIFEST_PATH))
        if man.get("model_hash"):
            return str(man["model_hash"]).lower()
    except Exception:                                        # noqa: BLE001
        pass
    return None


def load_model():
    t0 = time.time()
    try:
        model, meta = codec.load_side(DECODER_CKPT, "b", DEVICE)
        STATE["model"], STATE["meta"] = model, meta
        STATE["load_error"] = None
        log.info("解码器半区已加载 %s  meta=%s  用时 %.1f ms",
                 DECODER_CKPT, meta, (time.time() - t0) * 1000)
    except Exception as e:                                   # noqa: BLE001
        STATE["model"], STATE["meta"] = None, None
        STATE["load_error"] = f"{type(e).__name__}: {e}"
        log.exception("解码器半区加载失败：%s", DECODER_CKPT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model_hash_hex"] = _resolve_model_hash()
    log.info("verify_hash=%s", STATE["model_hash_hex"])
    load_model()
    yield


app = FastAPI(title="SNN A/B — B 机解码器", version="1.0", lifespan=lifespan)


def _tensor_to_png(img: torch.Tensor) -> bytes:
    """img: [1,3,H,W] 取值 0~1 -> PNG 字节。"""
    arr = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    return {
        "ok": STATE["model"] is not None and STATE["load_error"] is None,
        "side": "b",
        "model_hash": STATE["model_hash_hex"],
        "loaded": STATE["model"] is not None,
        "ckpt": DECODER_CKPT,
        "load_error": STATE["load_error"],
    }


@app.post("/decode")
async def decode(request: Request):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503,
                            detail=f"B 机模型未加载: {STATE['load_error']}")
    try:
        body = await request.json()
    except Exception as e:                                   # noqa: BLE001
        return JSONResponse(status_code=400, content={"detail": f"请求体不是合法 JSON: {e}"})

    key_b64 = (body or {}).get("key_b64")
    if not key_b64:
        return JSONResponse(status_code=400, content={"detail": "缺少字段 key_b64"})
    try:
        key = base64.b64decode(key_b64, validate=False)
    except (binascii.Error, ValueError) as e:
        return JSONResponse(status_code=400, content={"detail": f"key_b64 不是合法 base64: {e}"})
    if not key:
        return JSONResponse(status_code=400, content={"detail": "钥匙为空"})

    mh_hex = STATE["model_hash_hex"]
    verify = bytes.fromhex(mh_hex) if mh_hex else None

    # 先读文件头，把 steps/channels 拿出来给响应头用；头坏了直接 400
    try:
        hdr = codec.unpack_header(key)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    t0 = time.perf_counter()
    try:
        with _infer_lock:
            img = codec.decode_key(model, key, verify_hash=verify)
    except ValueError as e:                                  # 指纹不匹配 / 钥匙损坏
        log.warning("解码被拒绝: %s", e)
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:                                   # noqa: BLE001
        log.exception("解码失败")
        return JSONResponse(status_code=400,
                            content={"detail": f"{type(e).__name__}: {e}"})
    dec_ms = (time.perf_counter() - t0) * 1000.0

    png = _tensor_to_png(img)
    return Response(content=png, media_type="image/png", headers={
        "X-Model-Hash": hdr["model_hash"].hex(),
        "X-Decode-Ms": f"{dec_ms:.1f}",
        "X-Key-Bytes": str(len(key)),
        "X-Steps-Used": str(hdr["steps_used"]),
        "X-Channels-Used": str(hdr["channels_used"]),
        "X-Image-Size": str(img.shape[-1]),
        "Cache-Control": "no-store",
    })
