# -*- coding: utf-8 -*-
"""rd_codec.py — 把 RD / 锚图模型做成可序列化的编解码器（钥匙格式 v3）

设计要点
    1. **单个算术码流**：先编 z（边信息），再编 y（主体）。解码端按同样顺序：
       解 z -> 算 sigma = h_s(z_hat) -> 用 sigma 解 y。
       不需要在钥匙里传任何概率表 —— B 机用自己的 decoder 半区重算。
    2. **推理期旋钮写进文件头**：steps_used / channels_used，
       B 机不需要知道目标码率，照着头部解就行。
    3. **model_hash 防配错**：钥匙里带模型指纹，B 机比对不上就明确报错，
       而不是静默还原出一张乱码（这个坑我们踩过）。

钥匙布局
    0   4   magic "SNNK"
    4   1   version = 3
    5   1   flags
    6   1   codec_id    1=定长脉冲v2  2=RD高斯  3=RD高斯+锚图输入端拼接
    7   1   (对齐保留)
    8   2   image_size
    10  2   latent_channels
    12  2   num_steps（模型原始 T）
    14  2   steps_used
    16  2   channels_used
    18  8   model_hash 前 8 字节
    26  4   payload 字节数
    30  N   constriction 码流（uint32 字数组的原始字节）
"""
from __future__ import annotations

import math
import struct

import constriction
import numpy as np
import torch

from sae_rd import (_SIGMA_GRID, SIGMA_MIN, SIGMA_MAX, _bin_probs, _bucket_ids,
                    _categorical)

KEY_MAGIC = b"SNNK"
KEY_VERSION = 3
CODEC_RD = 2
CODEC_RD_ANCHOR = 3
HDR_FMT = "<4sBBBBHHHHHH8sII"
HDR_SIZE = struct.calcsize(HDR_FMT)          # 36
DEF_KMAX = 128      # y 字母表下限。实际用的 K 由符号真实范围决定，并写进头部
                    # （原来硬编码 ±32，而潜层能到 ±920 —— 直接截断成垃圾）
DEF_Z_KMAX = 128    # z 字母表。编解码两端必须用同一个（z 的 K 没有写进头部）


def _pick_kmax(yq, floor: int = DEF_KMAX, cap: int = 4096) -> int:
    """按符号的实际最大绝对值选字母表，向上取到 2 的幂。

    这样不管模型潜层尺度是多少，编码器都能无损装下 ——
    不需要去改损失函数"把潜层压小"，那属于拆东墙补西墙。
    K 写进头部，解码端按同一个值解。
    """
    m = int(yq.abs().max().item()) if yq.numel() else 0
    k = max(floor, 1)
    while k < min(m, cap):
        k *= 2
    return int(min(k, cap))


def pack_header(codec_id, image_size, channels, num_steps, steps_used,
                channels_used, model_hash: bytes, z_len: int, y_len: int,
                kmax: int = DEF_KMAX) -> bytes:
    return struct.pack(HDR_FMT, KEY_MAGIC, KEY_VERSION, 0, codec_id, 0,
                       image_size, channels, num_steps, steps_used, channels_used,
                       int(kmax),
                       (model_hash or b"")[:8].ljust(8, b"\0"), z_len, y_len)


def unpack_header(buf: bytes) -> dict:
    if len(buf) < HDR_SIZE:
        raise ValueError("钥匙文件头不完整")
    (magic, ver, flags, codec_id, _rsv, img, ch, T, steps, ch_used, kmax,
     mh, zlen, ylen) = struct.unpack(HDR_FMT, buf[:HDR_SIZE])
    if magic != KEY_MAGIC:
        raise ValueError("不是 SNNK 钥匙文件")
    if ver != KEY_VERSION:
        raise ValueError(f"钥匙版本 {ver} 不受支持（当前 {KEY_VERSION}）")
    return dict(codec_id=codec_id, image_size=img, channels=ch, num_steps=T,
                steps_used=steps, channels_used=ch_used, kmax=int(kmax),
                model_hash=mh.rstrip(b"\0"), z_len=zlen, y_len=ylen)


# ------------------------------------------------------------------------------
def _enc_z(enc, zq, z_prior):
    """把 z 的**整数符号** [T, Cz, hz, wz] 写进码流（逐通道，一个模型）。

    入参故意收整数符号而不是浮点 z_hat：编码端必须按"解码端将要拿到的那串符号"
    来算后面的 sigma，所以取整/夹字母表这一步要在调用方做完再传进来。
    """
    zq = zq.long().cpu()
    zc = zq.shape[1]
    sig = z_prior.sigma().detach().cpu()
    mu = z_prior.mu.detach().cpu()
    for c in range(zc):
        p = _bin_probs(float(sig.flatten()[c]), float(mu.flatten()[c]), DEF_Z_KMAX)
        enc.encode((zq[:, c].reshape(-1) + DEF_Z_KMAX).numpy().astype(np.int32),
                   _categorical(p))


def _dec_z(dec, z_prior, T, zc, hz, wz):
    sig = z_prior.sigma().detach().cpu()
    mu = z_prior.mu.detach().cpu()
    out = np.zeros((T, zc, hz, wz), dtype=np.int64)
    for c in range(zc):
        p = _bin_probs(float(sig.flatten()[c]), float(mu.flatten()[c]), DEF_Z_KMAX)
        got = dec.decode(_categorical(p), T * hz * wz)
        out[:, c] = (got - DEF_Z_KMAX).reshape(T, hz, wz)
    return out


@torch.no_grad()
def encode_key(model, x: torch.Tensor, steps: int | None = None,
               channels: int | None = None, model_hash: bytes = b"",
               codec_id: int = CODEC_RD_ANCHOR,
               ref: torch.Tensor | None = None) -> bytes:
    """x: [1,3,H,W] -> 钥匙字节串。steps/channels 是推理期码率旋钮。

    `ref`：**硬量化的参考潜层** y_ref_hat [1,T,C,h,w]，即
    `SAE_Anchor.encode_ref_latent(anchor_x)`（或从锚图钥匙解码出来的那串潜层）。
    传 None = 锚图 / 独立编码（和旧行为完全一致）；传了就是残差钥匙。
    codec 本身不需要知道谁是谁 —— 减法和加法都只发生在 `anchor.py` 的
    `code_residual` / `reconstruct` 里，这里只是把 latent 交给同一个编码器。

    **顺序约束（组码流必须满足）**：组里第一个钥匙必须是锚图（`ref=None`），
    残差钥匙要用对应锚图解码出来的 y_ref_hat 当 ref。也就是说码流顺序必须是
    "先锚图、后残差"，解码端也是这个顺序。钥匙头里没有"我是不是残差"的标记，
    所以发送端不能把残差钥匙排在它的锚图前面 —— 这条约束以前就存在
    （锚图必须先解出来），这里只是把它写下来。

    关于 `steps`/`channels` 截断：残差的参考端必须和锚图用**同一组**
    steps/channels，否则 `reconstruct` 的形状/语义都对不上。
    """
    T, C = model.num_steps, model.latent_channels
    steps = int(steps or T)
    channels = int(channels or C)
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(f"只支持正方形输入，收到 {tuple(x.shape)}")
    img_size = int(x.shape[-1])          # 真实输入边长 -> 写进头部，解码端据此定形状
    dev = next(model.parameters()).device

    utils_reset(model)
    # ref=None -> 锚图（coded = quant_hard(y)）；ref=y_ref_hat -> 残差
    # （coded = quant_hard(y_i) - y_ref_hat）。两者都走 SAE_Anchor.forward。
    y_hat, rate, _rec = model(x, ref=ref, checkpoints={model.num_steps})
    if ref is not None and tuple(ref.shape[1:]) != (T, C, y_hat.shape[-2], y_hat.shape[-1]):
        raise ValueError(
            f"参考潜层形状 {tuple(ref.shape)} 与编码潜层 {tuple(y_hat.shape)} 不匹配"
            f"（steps/channels 必须和锚图用同一组）")
    y_use = y_hat[:, :steps, :channels]

    # ---- z 流：先定格成整数符号，再用**同一串符号**算 sigma ----
    # 这一步是编码/解码能对齐的关键：解码端只会拿到夹过字母表的整数 z，
    # 如果编码端拿未夹的 z_hat 去算 sigma，只要有一个 |z| > DEF_Z_KMAX，
    # 后面的 y 流就会用两套不同的概率表 -> 解出来是乱码。
    z_hat = rate.get("z_hat")
    enc_z_ = constriction.stream.queue.RangeEncoder()
    extra = None
    if z_hat is not None:
        zq = torch.round(z_hat[:steps].detach()).clamp(-DEF_Z_KMAX, DEF_Z_KMAX).long()
        n_clip_z = int((torch.round(z_hat[:steps].detach()).abs() > DEF_Z_KMAX).sum())
        if n_clip_z:
            print(f"[codec warn] 有 {n_clip_z} 个 z 符号超出 ±{DEF_Z_KMAX} 被截断")
        _enc_z(enc_z_, zq, model.hyperprior.z_prior)
        extra = model.hyperprior.sigma_from_z(zq.float().to(dev))  # [steps, C, H, W]
    z_bytes = enc_z_.get_compressed().astype(np.uint32).tobytes()

    # ---- y 流：sigma 用和 ScaleHyperprior.forward / 解码端**同一个** prior.sigma() ----
    enc_y = constriction.stream.queue.RangeEncoder()
    sig_all = model.prior.sigma(extra).detach().cpu()             # [steps, C, H, W]
    mu = model.prior.mu.detach().cpu()
    yq = torch.round(y_use.detach()).long().cpu()
    kmax = _pick_kmax(yq)                     # 自适应字母表：按真实范围选
    yq = yq.clamp(-kmax, kmax).cpu()
    n_clip = int((torch.round(y_use.detach()).long().cpu().abs() > kmax).sum())
    if n_clip:
        print(f"[codec warn] 有 {n_clip} 个符号超出 ±{kmax} 被截断")
    for c in range(channels):
        syms = (yq[:, :, c].reshape(-1) + kmax).numpy().astype(np.int32)
        s = sig_all[:, c].reshape(-1).numpy().astype(np.float64)
        if s.size == 1:
            s = np.full(syms.size, float(s[0]))
        bid = _bucket_ids(s)
        for b in np.unique(bid):
            m = bid == b
            p = _bin_probs(float(_SIGMA_GRID[b]), float(mu.flatten()[c]), kmax)
            enc_y.encode(syms[m], _categorical(p))
    y_bytes = enc_y.get_compressed().astype(np.uint32).tobytes()

    hdr = pack_header(codec_id, img_size, C, T, steps, channels,
                      model_hash, len(z_bytes), len(y_bytes), kmax)
    return hdr + z_bytes + y_bytes


@torch.no_grad()
def _decode_latents(model, key: bytes, verify_hash: bytes | None = None,
                    ref: torch.Tensor | None = None):
    """钥匙 -> (y_hat 潜层 [1,steps,C,h,w], header)。decode_key 的公共前半段。

    `ref`：**硬量化的参考潜层** y_ref_hat（锚图解码/编码得到的那串值）。传 None
    就是独立钥匙，行为不变；传了就在返回前做一次精确加法
    `reconstruct(ref, 解码出来的残差)` —— 和编码端的 `code_residual` 是同一对函数。
    """
    h = unpack_header(key)
    if verify_hash is not None and h["model_hash"] and \
            h["model_hash"] != (verify_hash or b"")[:8]:
        raise ValueError(
            f"钥匙的模型指纹 {h['model_hash'].hex()} 与当前模型 "
            f"{(verify_hash or b'')[:8].hex()} 不一致 —— 这把钥匙不是这个模型压出来的，"
            f"继续解码只会得到乱码。")
    T, C = h["num_steps"], h["channels"]
    steps, ch_used = h["steps_used"], h["channels_used"]
    kmax = int(h["kmax"])            # 编码端按符号真实范围选的字母表
    z_bytes = key[HDR_SIZE:HDR_SIZE + h["z_len"]]
    y_off = HDR_SIZE + h["z_len"]
    y_bytes = key[y_off:y_off + h["y_len"]]

    # 空间尺寸必须由**头部的真实输入边长**推出来，而不是 model.latent_size：
    # 后者取自 checkpoint 的 image_size，一旦编码时喂的图不是那个尺寸
    # （历史上 eval 就用 128 的图喂 256 的模型），z/y 的形状全错。
    latent_hw = max(1, int(h["image_size"]) // 16)
    hz = wz = max(1, latent_hw // 4)
    dev = next(model.parameters()).device
    z_prior = model.hyperprior.z_prior
    zc = model.hyperprior.h_a[0].out_channels
    if len(z_bytes) >= 4:
        zdec = constriction.stream.queue.RangeDecoder(np.frombuffer(z_bytes, dtype=np.uint32))
        z_hat = _dec_z(zdec, z_prior, steps, zc, hz, wz)      # [steps, Cz, hz, wz]
        z_t = torch.from_numpy(z_hat).float().to(dev)
        # 和编码端逐字一致：同一个 sigma_from_z（clamp ±14 -> exp -> clamp[1e-4,1e6]）
        sigma = model.hyperprior.sigma_from_z(z_t)
    else:
        sigma = torch.ones(steps, C, latent_hw, latent_hw, device=dev)

    ydec = constriction.stream.queue.RangeDecoder(np.frombuffer(y_bytes, dtype=np.uint32))
    mu = model.prior.mu.detach().cpu()
    # 和编码端逐字一致：same prior.sigma(extra) -> (s*extra).clamp(SIGMA_MIN, SIGMA_MAX)
    sig_all = model.prior.sigma(sigma).detach().cpu()          # [steps, C, H, W]
    yq = np.zeros((1, steps, C, latent_hw, latent_hw), dtype=np.int64)
    for c in range(ch_used):
        s = sig_all[:, c].reshape(-1).numpy().astype(np.float64)
        if s.size == 1:
            s = np.full(steps * latent_hw ** 2, float(s[0]))
        bid = _bucket_ids(s)
        out = np.zeros(s.size, dtype=np.int64)
        for b in np.unique(bid):
            m = bid == b
            p = _bin_probs(float(_SIGMA_GRID[b]), float(mu.flatten()[c]), kmax)
            out[m] = ydec.decode(_categorical(p), int(m.sum())) - kmax
        yq[0, :, c] = out.reshape(steps, latent_hw, latent_hw)

    step = float(model.quant.step)
    y_hat = torch.from_numpy(yq).float().to(dev) * step
    # 残差钥匙：解码端把参考加回来。加法只有 model.reconstruct 一处定义。
    y_hat = model.reconstruct(y_hat, ref) if ref is not None else y_hat
    return y_hat, h


@torch.no_grad()
def decode_key(model, key: bytes, verify_hash: bytes | None = None,
               ref: torch.Tensor | None = None) -> torch.Tensor:
    """钥匙字节串 -> [1,3,H,W] 图像张量（取值 0~1）。

    `ref`：残差钥匙必须传**硬量化参考潜层** y_ref_hat；锚图传 None（行为不变）。
    """
    y_hat, h = _decode_latents(model, key, verify_hash, ref=ref)
    steps = h["steps_used"]
    return model.decode(y_hat, checkpoints={steps})[steps].clamp(0, 1)


def utils_reset(model):
    from snntorch import utils
    utils.reset(model)
