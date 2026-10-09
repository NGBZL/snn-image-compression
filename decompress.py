# -*- coding: utf-8 -*-
"""
decompress.py —— B 机（还原端）：拿"钥匙"文件还原图片

做的事
    key.pt -> 解包成脉冲序列 [1,T,C,h,w] -> 只跑训练好的【解码器】
           -> NxNx3 重建图（默认 256x256）-> recon.png

只加载解码器
    用 build_model_from_checkpoint(..., prefixes=("decoder.",)) 只把解码器的权重装进模型，
    编码器的参数保持随机初始化 —— B 机上没有可用的编码器。
    B 机不需要原图、不需要联网，只要有 key.pt + sae_cifar.pth。

用法
    python decompress.py --key key.pt --output recon.png
    python decompress.py --key key.pt --output recon.png --reference 原图.jpg
        # 给了原图就会算 PSNR / SSIM，并同时给出"同字节数的 JPEG"的 PSNR / SSIM
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows 控制台默认编码是 GBK，先关掉"打印字符编不出来就崩溃"的问题
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from sae_model import (
    IMG_CH, SSIM, bpp, build_model_from_checkpoint, human_bytes, jpeg_size_and_quality,
    load_key, ms_psnr, pretty_bits, channel_rate_entropy,
)
from snntorch import utils


def parse_args():
    p = argparse.ArgumentParser(
        description="B 机：用脉冲'钥匙'还原图片（只用解码器）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--key", default="key.pt", help="输入钥匙文件路径")
    p.add_argument("--output", default="recon.png", help="输出还原图路径")
    p.add_argument("--weights", default="sae_cifar.pth", help="训练好的权重文件（与 A 机同一份）")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # 一般不用手填，钥匙文件头和权重文件里都带了这些值
    p.add_argument("--image-size", type=int, default=None, help="覆盖 image_size")
    p.add_argument("--latent-channels", type=int, default=None, help="覆盖 latent_channels")
    p.add_argument("--num-steps", type=int, default=None, help="覆盖 num_steps")
    p.add_argument("--preview-scale", type=int, default=1,
                   help="额外存一份放大 N 倍的预览图；0 或 1 = 不存（256x256 本来就能看）")
    p.add_argument("--compare", action="store_true",
                   help="把钥匙的 0/1 比特可视化成一张图 key_bits.png")
    p.add_argument("--reference", default=None,
                   help="可选：原图路径。给了就算 PSNR/SSIM，并和同字节数的 JPEG 对比")
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    """B 机没有 GPU 时自动回退 CPU。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_image_hwc(tensor_chw: torch.Tensor, path: str, scale: int = 1):
    """把 [3,H,W]（取值 0~1）存成 PNG。

    ToTensor 得到的是 [C,H,W]，必须 permute(1,2,0) 变成 [H,W,3] 才是正确的彩色排列，
    否则会把通道当成宽高，出来的图颜色和形状都是错的。
    """
    arr = tensor_chw.detach().cpu().clamp(0, 1)
    arr = arr.permute(1, 2, 0).numpy()                     # [3,H,W] -> [H,W,3]
    arr = (arr * 255.0).round().astype(np.uint8)           # [0,1] 浮点 -> [0,255]
    img = Image.fromarray(arr, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img.save(path)
    return img.size


def bits_to_image(spk: torch.Tensor, path: str, scale: int = 4):
    """把 [1,T,C,h,w] 的脉冲压成一张 [T*h, C*w] 的黑白小图：白=脉冲，黑=静默。"""
    s = spk.detach().to(torch.uint8)[0]                    # [T,C,h,w]
    # 每个通道横着排，每个时间步竖着排，一眼能看出各通道的放电节奏
    grid = s.permute(0, 2, 1, 3).reshape(s.shape[0] * s.shape[2], s.shape[1] * s.shape[3])
    arr = (grid.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img.save(path)
    return img.size


def load_reference(path: str, size: int, device) -> torch.Tensor:
    """按和 A 机完全相同的方式读原图，用来和重建图逐像素比较。"""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                             transforms.ToTensor()])
    return tf(img).unsqueeze(0).to(device)


def main():
    args = parse_args()
    device = pick_device(args.device)

    if not os.path.isfile(args.key):
        raise SystemExit(f"找不到钥匙文件：{args.key}")

    # ---------------------------------------------------------------- 1. 读钥匙
    spk = load_key(args.key, device=device)          # [1,T,C,h,w]
    T, C, H, W = (int(v) for v in spk.shape[1:])
    key_bytes = os.path.getsize(args.key)

    # ---------------------------------------------------------------- 2. 只加载解码器
    model, cfg = build_model_from_checkpoint(
        args.weights, device=device,
        image_size=args.image_size, latent_channels=args.latent_channels,
        num_steps=args.num_steps, strict=False, prefixes=("decoder.",))
    model.eval()

    print("=" * 78)
    print("B 机 · 还原")
    print("=" * 78)
    print(f"设备            : {device}")
    print(f"权重            : {args.weights}  (image_size={model.image_size}, "
          f"C={model.latent_channels}, T={model.num_steps})")
    print(f"钥匙            : {args.key}  ({key_bytes} 字节 / {human_bytes(key_bytes)})")
    print(f"钥匙形状        : {tuple(spk.shape)}  = T{T} x C{C} x {H}x{W}  =  {T * C * H * W} bit")
    print(f"码率            : {bpp(key_bytes, model.image_size):.3f} bit/像素")
    print(f"钥匙内容        : {pretty_bits(spk)}")
    print(f"通道率熵        : {channel_rate_entropy(spk):.1f}/{C} bit  "
          f"(脉冲发放率 {float(spk.mean()):.3f})")

    # ---- 一致性检查：钥匙形状必须和模型对得上 ----
    expect_hw = model.latent_size
    if C != model.latent_channels:
        raise SystemExit(
            f"钥匙的 latent_channels={C} 与权重里的 {model.latent_channels} 不一致。\n"
            f"请确认这把钥匙是用同一份 {os.path.basename(args.weights)} 对应的编码器压出来的。")
    if (H, W) != (expect_hw, expect_hw):
        raise SystemExit(
            f"钥匙的潜层空间 {H}x{W} 与权重里的 {expect_hw}x{expect_hw} 不一致"
            f"（对应 image_size={model.image_size}）。\n"
            f"请确认 A 机和 B 机的 --image-size 一致。")
    if T != model.num_steps:
        print(f"[warn] 钥匙的 T={T} 与权重里的 T={model.num_steps} 不同，将按钥匙里的 {T} 步解码。")
        model.num_steps = T

    # ---------------------------------------------------------------- 3. 只跑解码器
    with torch.no_grad():
        # decode() 内部会先 utils.reset(decoder) 清膜电位，再做 T 步时间平均
        recon = model.decode(spk).clamp(0, 1)      # [1,3,N,N]

    # ---------------------------------------------------------------- 4. 保存图片
    size = save_image_hwc(recon[0], args.output)
    print("-" * 78)
    print(f"[完成] 还原完成：{args.output}   ({size[0]}x{size[1]})")

    if args.preview_scale > 1:
        base, ext = os.path.splitext(args.output)
        prev = f"{base}_x{args.preview_scale}{ext or '.png'}"
        psize = save_image_hwc(recon[0], prev, scale=args.preview_scale)
        print(f"   放大预览     : {prev}   ({psize[0]}x{psize[1]})")

    if args.compare:
        csize = bits_to_image(spk, "key_bits.png", scale=4)
        print(f"   钥匙可视化   : key_bits.png   ({csize[0]}x{csize[1]}，"
              f"横=C 通道，竖=T 时间步；白=脉冲1，黑=静默0)")

    print(f"   还原图像素范围: [{float(recon.min()):.3f}, {float(recon.max()):.3f}]，"
          f"均值 {float(recon.mean()):.3f}")

    # ---------------------------------------------------------------- 5. 有原图就量化评价
    if args.reference:
        if not os.path.isfile(args.reference):
            raise SystemExit(f"找不到参考原图：{args.reference}")
        ref = load_reference(args.reference, model.image_size, device)
        ssim_fn = SSIM().to(device)
        with torch.no_grad():
            p_snn = ms_psnr(recon, ref).item()
            s_snn = ssim_fn(recon, ref).item()
            # 同字节数 JPEG：二分搜索质量因子，让 JPEG 体积贴近钥匙大小
            jbytes, q, jrec, matched = jpeg_size_and_quality(ref[0], key_bytes)
            jrec = jrec.unsqueeze(0).to(device)
            p_jpg = ms_psnr(jrec, ref).item()
            s_jpg = ssim_fn(jrec, ref).item()
        print("-" * 78)
        print(f"量化评价（对比 {args.reference}，{model.image_size}x{model.image_size} 原图）")
        print(f"{'方案':<16}{'字节':>9}{'bit/像素':>11}{'PSNR':>10}{'SSIM':>9}")
        print(f"{'SNN 钥匙':<16}{key_bytes:>9}{bpp(key_bytes, model.image_size):>11.3f}"
              f"{p_snn:>10.2f}{s_snn:>9.4f}")
        print(f"{'JPEG(同码率)':<16}{jbytes:>9}{bpp(jbytes, model.image_size):>11.3f}"
              f"{p_jpg:>10.2f}{s_jpg:>9.4f}")
        print(f"{'差值(SNN-JPEG)':<16}{'':>9}{'':>11}{p_snn - p_jpg:>+10.2f}{s_snn - s_jpg:>+9.4f}")
        print("→ 差值为正才说明 SNN 方案在同等字节数下真的更好。")
        if not matched:
            print(f"[注意] JPEG 即使压到最低质量(q={q})也有 {jbytes} 字节 > 钥匙的 {key_bytes} 字节，"
                  f"两边码率对不上，这个对比不成立。")

    total = key_bytes + os.path.getsize(args.weights)
    print(f"\n[算一笔账] 这把钥匙 {key_bytes} 字节；模型权重 "
          f"{human_bytes(os.path.getsize(args.weights))}（两台机器共用，只需传一次）")


if __name__ == "__main__":
    main()
