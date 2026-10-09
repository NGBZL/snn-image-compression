# -*- coding: utf-8 -*-
"""
compress.py —— A 机（压缩端）：把任意图片压成"钥匙"文件

做的事
    图片 -> 缩放到 NxN（默认 256）-> ToTensor -> 只跑训练好的【编码器】
         -> 脉冲序列 [1,T,C,h,w] -> key.pt

只加载编码器
    用 build_model_from_checkpoint(..., prefixes=("encoder.",)) 只把编码器的权重装进模型，
    解码器的参数保持随机初始化 —— A 机上"物理上"没有可用的解码器。

用法
    python compress.py --image cat.jpg --output key.pt
    python compress.py --image cat.jpg --output key.pt --format float   # 对比不打包时的体积
    python compress.py --image cat.jpg --output key.pt --save-input32    # 存缩放后的输入图

钥匙大小 = T * C * (N/16)^2 比特，和图片内容无关（定长比特域）。
默认 256x256 / C=32 / T=10 -> 81920 bit = 10240 字节 + 14 字节头。
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

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from sae_model import (
    IMG_CH, build_model_from_checkpoint, bpp, human_bytes, load_key, pretty_bits,
    save_key, channel_rate_entropy,
)
from snntorch import utils


def parse_args():
    p = argparse.ArgumentParser(
        description="A 机：把图片压缩成脉冲'钥匙'文件（只用编码器）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image", required=True, help="输入图片路径（jpg/png/bmp/... 任意格式）")
    p.add_argument("--output", default="key.pt", help="输出钥匙文件路径")
    p.add_argument("--weights", default="sae_cifar.pth", help="训练好的权重文件")
    p.add_argument("--format", choices=["bits", "float"], default="bits",
                   help="bits=按位打包（最省空间）；float=float32 明文（用来对比体积）")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # 一般不用手填，权重文件里已经带了这些值；只有在权重是"纯 state_dict"时才需要
    p.add_argument("--image-size", type=int, default=None, help="覆盖权重里的 image_size")
    p.add_argument("--latent-channels", type=int, default=None, help="覆盖 latent_channels")
    p.add_argument("--num-steps", type=int, default=None, help="覆盖 num_steps")
    p.add_argument("--save-input32", action="store_true",
                   help="顺便把缩放后的输入图存成 input_resized.png，方便和还原图对比")
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    """没有 GPU 时自动回退 CPU。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_as_tensor(path: str, size: int, device: torch.device) -> torch.Tensor:
    """读图 -> 转正 -> 缩放到 NxN -> ToTensor -> [1,3,N,N]，取值 [0,1]。

    训练时用的是 Resize + CenterCrop + ToTensor（不做 Normalize），这里必须对齐，
    否则编码器看到的分布和训练时不一样，压出来的钥匙就是错的。
    """
    if not os.path.isfile(path):
        raise SystemExit(f"找不到输入图片：{path}")
    img = Image.open(path)
    # 手机拍的照片常常靠 EXIF 记录旋转方向，PIL 默认不会自动转正。
    # 不处理的话竖拍照片会被当成横的压进去，还原出来方向就是错的。
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")            # 灰度图 / CMYK / 带透明通道的图统一成三通道
    tf = transforms.Compose([
        transforms.Resize(size),                   # 短边缩放到 size
        transforms.CenterCrop(size),               # 再中心裁剪成 size x size
        transforms.ToTensor(),                     # [0,255] -> [0,1]，HWC -> CHW
    ])
    return tf(img).unsqueeze(0).to(device)


def main():
    args = parse_args()
    device = pick_device(args.device)

    # ---------------------------------------------------------------- 1. 只加载编码器
    # build_model_from_checkpoint 先从权重文件里读出 image_size / latent_channels / num_steps，
    # 所以 A 机和 B 机不需要人工对齐这些参数。
    model, cfg = build_model_from_checkpoint(
        args.weights, device=device,
        image_size=args.image_size, latent_channels=args.latent_channels,
        num_steps=args.num_steps, strict=False, prefixes=("encoder.",))
    model.eval()                        # 关掉 BatchNorm 的批统计，保证同一张图每次编出同一把钥匙

    print("=" * 78)
    print("A 机 · 压缩")
    print("=" * 78)
    print(f"设备            : {device}")
    print(f"权重            : {args.weights}")
    print(f"图像尺寸        : {model.image_size}x{model.image_size}")
    print(f"潜层            : C={model.latent_channels}  T={model.num_steps}  "
          f"空间 {model.latent_size}x{model.latent_size}")

    # ---------------------------------------------------------------- 2. 读入并预处理
    x = load_image_as_tensor(args.image, model.image_size, device)
    orig_bytes = os.path.getsize(args.image)
    print(f"输入图片        : {args.image}  ({human_bytes(orig_bytes)})")
    print(f"预处理后        : {tuple(x.shape)}  取值 [{x.min():.3f}, {x.max():.3f}]")

    if args.save_input32:
        from torchvision.utils import save_image
        save_image(x.cpu(), "input_resized.png")
        print("                 -> 已另存缩放后的输入图 input_resized.png")

    # ---------------------------------------------------------------- 3. 只跑编码器
    with torch.no_grad():
        # encode() 内部会先 utils.reset(encoder) 清膜电位，再跑完整的 num_steps 步
        spk_rec = model.encode(x)       # [1,T,C,h,w]，每个元素 0 或 1

    T, C, H, W = spk_rec.shape[1:]

    # ---------------------------------------------------------------- 4. 存成钥匙文件
    bitpack = (args.format == "bits")
    n_bytes = save_key(args.output, spk_rec, bitpack=bitpack)

    # ---------------------------------------------------------------- 5. 报告体积
    n_bits = int(spk_rec.numel())
    raw_f32 = n_bits * 4
    raw_img = model.image_size * model.image_size * IMG_CH
    entropy = channel_rate_entropy(spk_rec)
    rate = float(spk_rec.mean())

    print("-" * 78)
    print(f"脉冲序列形状    : {tuple(spk_rec.shape)}   ([batch, T, C, h, w])")
    print(f"脉冲发放率      : {rate:.3f}   (0=全沉默, 1=全放电)")
    print(f"通道率熵        : {entropy:.1f}/{C} bit  (码字多样性；钥匙总容量 {n_bits} bit)")
    print(f"钥匙内容        : {pretty_bits(spk_rec)}")
    print("-" * 78)
    print(f"[完成] 压缩完成：{args.output}")
    print(f"   钥匙实际大小  : {n_bytes} 字节 ({human_bytes(n_bytes)})"
          f"   [14 字节头 + {'按位打包' if bitpack else 'float32 明文'}]")
    print(f"   码率          : {bpp(n_bytes, model.image_size):.3f} bit/像素")
    print(f"   同一串脉冲若用 float32 存 : {raw_f32} 字节 ({human_bytes(raw_f32)})")
    print(f"   缩放后的原图像素          : {raw_img} 字节 ({human_bytes(raw_img)})")
    print(f"   相对原始 RGB 压缩         : {100 * (1 - n_bytes / raw_img):.1f}%")
    if orig_bytes > 0:
        print(f"   相对原图文件压缩          : {orig_bytes / max(1, n_bytes):,.0f} : 1")
    if not bitpack:
        print("   [提示] 用 --format bits 可以再省 4~8 倍空间")

    # 码率太低的提醒：钥匙比特数直接决定重建质量上限
    bp = bpp(n_bytes, model.image_size)
    if bp < 0.1:
        print(f"   [警告] 当前码率只有 {bp:.3f} bit/像素，太低，还原图大概率只能看出颜色块。")
        print(f"          想看清内容请加大 --latent-channels 或 --num-steps 后重新训练。")

    # 读回来验证一遍，确保 B 机能正确解包
    chk = load_key(args.output, device="cpu")
    assert torch.equal(chk.to(torch.uint8), spk_rec.cpu().to(torch.uint8)), "钥匙文件自检失败！"
    print(f"   [自检] 回读 {args.output} 成功，形状 {tuple(chk.shape)}，内容与写入一致")

    print(f"\n把 {args.output} 拷到 B 机，然后运行：")
    print(f"  python decompress.py --key {args.output} --output recon.png "
          f"--weights {os.path.basename(args.weights)}")


if __name__ == "__main__":
    main()
