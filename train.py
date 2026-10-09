# -*- coding: utf-8 -*-
"""
train.py —— 训练脉冲自编码器（Spiking Autoencoder），256x256 版本

流程
    1. 加载数据集（默认 Flowers102，可换 stl10 / pets / cifar10 / fake）
    2. 用复合损失 L = w_ssim*(1-SSIM) + w_mae*MAE + w_mse*MSE 训练 SAE
    3. 每 N 个 epoch 把重建样例存到 recon_cifar/（上排原图、下排重建图）
    4. 按验证集 SSIM 早停，保存 sae_cifar.pth
    5. 训练结束跑一次"同码率 JPEG 对比"——只报"相对原始 RGB 压缩 95%"是没意义的，
       256x256 的 JPEG 本身也就 10 KB 左右，必须和它比才说明问题

为什么用复合损失
    纯 MSE 在数学上倾向于输出"条件均值"，天生过平滑，重建图会糊成一团。
    加 SSIM 项管结构、MAE 项管锐度，这是重建质量提升最直接的一招。

常用命令
    # 正式训练（默认 256x256，钥匙约 10 KB）
    python train.py --dataset flowers --epochs 200 --batch-size 16

    # 显存不够就把 batch 调小；想先快速看效果就用小分辨率
    python train.py --dataset flowers --image-size 128 --epochs 60 --batch-size 32

    # 不联网冒烟测试（几秒钟跑通全流程）
    python train.py --dataset fake --subset 64 --epochs 3 --image-size 64 \
                    --latent-channels 8 --num-steps 4 --batch-size 8 --workers 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Windows 控制台默认编码是 GBK，遇到特殊字符会直接抛 UnicodeEncodeError，
# 让训练在打印日志时崩掉。这里改成"编不出来的字符用 ? 代替"，保证日志永不中断。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from sae_model import (
    DEFAULT_IMAGE_SIZE, DEFAULT_LATENT_CHANNELS, DEFAULT_NUM_STEPS,
    IMG_CH, SAE, SSIM, bpp, human_bytes, jpeg_size_and_quality, ms_psnr,
    channel_rate_entropy,
)
from snntorch import utils


# ======================================================================================
# 0. 命令行参数
# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="训练 snnTorch 脉冲自编码器（256x256 有损图像压缩）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- 压缩率相关：钥匙比特数 = num_steps * latent_channels * (image_size/16)^2 ----
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                   help="输入/输出边长，必须是 16 的倍数")
    p.add_argument("--latent-channels", type=int, default=DEFAULT_LATENT_CHANNELS,
                   help="潜层通道数 C，是调压缩率的主要旋钮")
    p.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
                   help="SNN 时间步数 T")
    p.add_argument("--beta", type=float, default=0.4, help="LIF 膜电位衰减系数")
    p.add_argument("--threshold", type=float, default=0.75, help="LIF 发放阈值")

    # ---- 损失权重：L = w_ssim*(1-SSIM) + w_mae*MAE + w_mse*MSE ----
    p.add_argument("--w-ssim", type=float, default=1.0, help="SSIM 项权重（管结构）")
    p.add_argument("--w-mae", type=float, default=1.0, help="MAE/L1 项权重（管锐度）")
    p.add_argument("--w-mse", type=float, default=1.0, help="MSE 项权重（管整体误差）")

    # ---- 训练超参 ----
    p.add_argument("--batch-size", type=int, default=16,
                   help="256x256 + T 步的显存开销不小，显存不够就调小")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--time-budget-min", type=float, default=0.0,
                   help="墙钟时间预算（分钟）。到点后不再开新 epoch，保证按时结束；0=不限时")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="梯度裁剪阈值；SNN 沿时间反传容易梯度爆炸，0 表示不裁剪")
    p.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument("--l2", type=float, default=0.0,
                   help="对卷积/转置卷积权重额外加的 L2 惩罚（论文里用了，默认关）")

    # ---- 脉冲发放率正则 ----
    # 注意：我们的钥匙是"定长比特域"（T*C*h*w 比特），文件大小和实际发了多少脉冲无关，
    # 所以稀疏并不能省字节，反而浪费容量。target-rate 设 0.5（二元熵最大）最划算。
    p.add_argument("--spike-reg", type=float, default=1e-3,
                   help="发放率正则权重，防止潜层退化成全 0 或全 1；0 = 关掉")
    p.add_argument("--target-rate", type=float, default=0.5,
                   help="期望发放率；定长比特域下 0.5 的信息量最大")

    # ---- 早停 ----
    p.add_argument("--early-stop-patience", type=int, default=30,
                   help="验证 SSIM 连续多少个 epoch 没提升就停；0 = 不早停")
    p.add_argument("--early-stop-delta", type=float, default=1e-4)

    # ---- 数据 ----
    p.add_argument("--dataset", choices=["flowers", "stl10", "pets", "cifar10", "fake"],
                   default="flowers",
                   help="flowers=Flowers102(330MB,1024类花) / stl10 / pets / cifar10(32x32,太小) / fake=离线合成")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--augment", choices=["none", "flip", "crop"], default="flip",
                   help="训练集增强；重建任务里增强后的图就是重建目标")
    p.add_argument("--subset", type=int, default=0, help="只用前 N 张训练图（0=全部），调试用")
    p.add_argument("--eval-limit", type=int, default=512,
                   help="每个 epoch 评估用多少张测试图（0=全部）；256x256 下全量评估很慢")
    p.add_argument("--max-batches", type=int, default=0,
                   help="每个 epoch 最多跑多少个 batch（0=全部），冒烟测试用")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-download", action="store_true", help="禁止联网下载数据集")

    # ---- 输出 ----
    p.add_argument("--out", type=str, default="sae_cifar.pth", help="权重保存路径")
    p.add_argument("--best-out", type=str, default="sae_cifar_best.pth")
    p.add_argument("--recon-dir", type=str, default="recon_cifar")
    p.add_argument("--recon-interval", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=8, help="样例网格里的图片数量")
    p.add_argument("--compare-jpeg", type=int, default=64,
                   help="训练后用多少张测试图做同码率 JPEG 对比（0=跳过）")
    p.add_argument("--plot", action="store_true", help="额外画 loss / 指标曲线")

    # ---- 设备 ----
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    """按需选设备；没有 GPU 时自动回退 CPU。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# 1. 数据
# ======================================================================================
class FakeImageDataset(Dataset):
    """离线冒烟测试用的合成数据：随机背景 + 若干随机色块/椭圆。

    有明显的内容和布局，哪怕只训练几个 epoch，也能看出重建图和原图是不是"像"。
    """

    def __init__(self, n: int = 256, size: int = 64, seed: int = 0):
        super().__init__()
        import random

        from PIL import Image, ImageDraw
        rng = random.Random(seed)
        self.size = size
        self.imgs = []
        for _ in range(n):
            bg = tuple(rng.randint(30, 220) for _ in range(3))
            im = Image.new("RGB", (size, size), bg)
            d = ImageDraw.Draw(im)
            for _ in range(rng.randint(2, 4)):
                x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
                w, h = rng.randint(size // 6, size // 2), rng.randint(size // 6, size // 2)
                col = tuple(rng.randint(0, 255) for _ in range(3))
                if rng.random() < 0.5:
                    d.rectangle([x0, y0, min(size - 1, x0 + w), min(size - 1, y0 + h)], fill=col)
                else:
                    d.ellipse([x0, y0, min(size - 1, x0 + w), min(size - 1, y0 + h)], fill=col)
            self.imgs.append(im)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        from torchvision.transforms.functional import to_tensor
        return to_tensor(self.imgs[idx]), 0


def build_transforms(args, train: bool):
    """只做 Resize + CenterCrop + ToTensor，不做 Normalize。

    因为解码器最后用 sigmoid 输出 [0,1]，两边必须保持一致。
    """
    ops = [transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size)]
    if train and args.augment == "flip":
        ops.append(transforms.RandomHorizontalFlip())
    elif train and args.augment == "crop":
        ops = [transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),
               transforms.RandomHorizontalFlip()]
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_datasets(args):
    """返回 (train_set, test_set, 名字)。"""
    dl = not args.no_download
    name = args.dataset

    if name == "fake":
        return (FakeImageDataset(512, args.image_size, seed=args.seed),
                FakeImageDataset(128, args.image_size, seed=args.seed + 999),
                "Fake 合成图")

    if name == "flowers":
        # Flowers102：train=1020, val=1020, test=6149
        # 训练用 train+val，测试用 test（严格不相交，报出来的指标是诚实的）
        tr = ConcatDataset([
            datasets.Flowers102(args.data_dir, split="train", download=dl,
                                transform=build_transforms(args, True)),
            datasets.Flowers102(args.data_dir, split="val", download=dl,
                                transform=build_transforms(args, True)),
        ])
        te = datasets.Flowers102(args.data_dir, split="test", download=dl,
                                 transform=build_transforms(args, False))
        return tr, te, "Flowers102 (~500x700 彩色照片, 102 类花)"

    if name == "stl10":
        # STL-10：96x96 真实照片，train=5000, test=8000
        tr = datasets.STL10(args.data_dir, split="train", download=dl,
                            transform=build_transforms(args, True))
        te = datasets.STL10(args.data_dir, split="test", download=dl,
                            transform=build_transforms(args, False))
        return tr, te, "STL-10 (96x96 真实照片)"

    if name == "pets":
        # Oxford-IIIT Pet：~224x224 猫狗照片，trainval=3680, test=3669
        tr = datasets.OxfordIIITPet(args.data_dir, split="trainval", download=dl,
                                    transform=build_transforms(args, True))
        te = datasets.OxfordIIITPet(args.data_dir, split="test", download=dl,
                                    transform=build_transforms(args, False))
        return tr, te, "Oxford-IIIT Pet (猫狗照片)"

    if name == "cifar10":
        # CIFAR-10 原生只有 32x32，放大到 256 也不会有更多细节，只适合调试
        tr = datasets.CIFAR10(args.data_dir, train=True, download=dl,
                              transform=build_transforms(args, True))
        te = datasets.CIFAR10(args.data_dir, train=False, download=dl,
                              transform=build_transforms(args, False))
        return tr, te, "CIFAR-10 (原生 32x32，放大后无细节，仅调试用)"

    raise ValueError(f"未知数据集 {name}")


def build_loaders(args):
    train_set, test_set, name = build_datasets(args)
    if args.subset > 0:
        train_set = Subset(train_set, range(min(args.subset, len(train_set))))
    if args.eval_limit > 0 and len(test_set) > args.eval_limit:
        test_set = Subset(test_set, range(args.eval_limit))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin, drop_last=False)
    return train_loader, test_loader, name


# ======================================================================================
# 2. 损失
# ======================================================================================
class CompositeLoss(nn.Module):
    """L = w_ssim*(1-SSIM) + w_mae*MAE + w_mse*MSE

    拆开打印每一项，方便你判断是哪一项在主导 —— 调权重的时候这个非常有用。
    """

    def __init__(self, w_ssim=1.0, w_mae=1.0, w_mse=1.0):
        super().__init__()
        self.w_ssim, self.w_mae, self.w_mse = w_ssim, w_mae, w_mse
        self.ssim = SSIM()

    def forward(self, pred, target):
        parts = {}
        total = pred.new_zeros(())
        if self.w_ssim > 0:
            parts["ssim_loss"] = 1.0 - self.ssim(pred, target)
            total = total + self.w_ssim * parts["ssim_loss"]
        if self.w_mae > 0:
            parts["mae"] = F.l1_loss(pred, target)
            total = total + self.w_mae * parts["mae"]
        if self.w_mse > 0:
            parts["mse"] = F.mse_loss(pred, target)
            total = total + self.w_mse * parts["mse"]
        return total, parts


def l2_penalty(model, prefixes=("encoder.", "decoder.")):
    """对卷积 / 转置卷积权重加 L2（论文里用的，默认关闭）。"""
    s = 0.0
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)) and name.startswith(prefixes):
            s = s + mod.weight.pow(2).sum()
    return s


# ======================================================================================
# 3. 训练 / 评估一个 epoch
# ======================================================================================
def run_epoch(model, loader, optimizer, device, criterion, args, train: bool):
    """返回一个 dict：loss 各分量、MSE、MS-PSNR、SSIM、发放率、通道熵。"""
    model.train(train)
    acc = {"loss": 0.0, "mse": 0.0, "ms_psnr": 0.0, "ssim_metric": 0.0,
           "rate": 0.0, "entropy": 0.0, "n": 0}
    ssim_fn = criterion.ssim
    max_batches = args.max_batches if args.max_batches > 0 else len(loader)

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)

        # ---- 关键：每个 batch 前清零所有 LIF 的膜电位 ----
        # 漏掉这一步会得到完全错误的结果（膜电位会跨 batch 累积）
        utils.reset(model)

        spk = model.encode(x)                  # [B,T,C,h,w] 0/1
        recon = model.decode(spk)              # [B,3,H,W] in [0,1]

        loss, parts = criterion(recon, x)

        # ---- 可选的发放率正则：防止潜层退化成全 0 / 全 1 ----
        if args.spike_reg > 0:
            loss = loss + args.spike_reg * (spk.mean() - args.target_rate) ** 2
        if args.l2 > 0:
            loss = loss + args.l2 * l2_penalty(model)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # SNN 要沿时间维反传，梯度容易爆炸，裁剪一下更稳
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        b = x.shape[0]
        with torch.no_grad():
            acc["loss"] += loss.item() * b
            acc["mse"] += F.mse_loss(recon, x).item() * b
            acc["ms_psnr"] += ms_psnr(recon, x).item() * b
            acc["ssim_metric"] += ssim_fn(recon, x).item() * b
            acc["rate"] += spk.detach().float().mean().item() * b
            acc["entropy"] += channel_rate_entropy(spk.detach()) * b
            acc["n"] += b

    n = max(1, acc.pop("n"))
    return {k: v / n for k, v in acc.items()}


# ======================================================================================
# 4. 重建样例 / 权重保存
# ======================================================================================
@torch.no_grad()
def build_recon_grid(model, batch, device, upscale: int = 1):
    """[2n,3,H,W] 网格：上排原图、下排重建图。"""
    x = batch.to(device)
    utils.reset(model)
    recon = model.decode(model.encode(x)).clamp(0, 1)
    n = x.shape[0]
    grid_in = torch.cat([x.cpu(), recon.cpu()], dim=0)
    if upscale > 1:
        grid_in = F.interpolate(grid_in, scale_factor=upscale, mode="nearest")
    return make_grid(grid_in, nrow=n, padding=2, pad_value=1.0)


@torch.no_grad()
def save_recon_grid(model, batch, device, paths, upscale: int = 1):
    """把对比网格图存到 paths（字符串或列表）。只跑一次前向。

    注意：重建样例只是给人看的可视化，**写不出来也绝不能打断训练**。
    （踩过的坑：目标文件带只读属性 -> PermissionError -> 整个 200 epoch 的训练在第 10 轮崩掉）
    """
    if isinstance(paths, str):
        paths = [paths]
    grid = build_recon_grid(model, batch, device, upscale)
    for path in paths:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            save_image(grid, path)
        except Exception as e:
            print(f"[warn] 重建样例写入失败（不影响训练）：{path} -> "
                  f"{type(e).__name__}: {e}")


def save_checkpoint(path, model, args, epoch, metrics):
    """保存完整权重 + 配置。配置跟着权重走，B 机不用手动交代 image_size / latent_channels。

    同样不因为写不进去就崩掉整个训练：只报错继续，避免跑了几小时却什么都没有。
    """
    try:
        torch.save({
            "state_dict": model.state_dict(),
            "image_size": model.image_size,
            "latent_channels": model.latent_channels,
            "num_steps": model.num_steps,
            "beta": model.beta,
            "threshold": model.threshold,
            "epoch": epoch,
            "metrics": metrics,
            "format_version": 2,
        }, path)
        return os.path.getsize(path)
    except Exception as e:
        print(f"[错误] 权重保存失败！{path} -> {type(e).__name__}: {e}")
        print("       检查该文件是否被占用或带只读属性；否则这次训练的成果会丢失。")
        return 0


# ======================================================================================
# 5. 同码率 JPEG 对比
# ======================================================================================
@torch.no_grad()
def compare_with_jpeg(model, loader, device, n_images: int, ssim_fn):
    """在测试集上做"同码率"对比：我们的钥匙 vs 相同字节数的 JPEG。

    这一步很重要：只报"相对原始 RGB 压缩了 95%"没有意义，
    因为 256x256 的 JPEG（质量 85）本身也就 10-15 KB。
    真正要回答的问题是：同样的字节数，谁的图更像原图。
    """
    if n_images <= 0:
        return None
    key_b = model.key_bytes
    rows_sae, rows_jpeg = [], []
    got = 0
    for x, _ in loader:
        for k in range(x.shape[0]):
            if got >= n_images:
                break
            img = x[k].to(device)
            # 我们的方案
            utils.reset(model)
            recon = model.decode(model.encode(img.unsqueeze(0))).clamp(0, 1)[0]
            rows_sae.append((img.cpu(), recon.cpu()))
            # JPEG：二分搜索质量因子，让体积贴近钥匙大小
            jbytes, q, jrecon, matched = jpeg_size_and_quality(img, key_b)
            rows_jpeg.append((img.cpu(), jrecon, jbytes, q, matched))
            got += 1
        if got >= n_images:
            break

    def stats(pairs):
        xs = torch.stack([a for a, _ in pairs])
        ys = torch.stack([b for _, b in pairs])
        return (ms_psnr(ys, xs).item(), ssim_fn(ys, xs).item(), xs.shape[0])

    sae_psnr, sae_ssim, n = stats(rows_sae)
    j_bytes = sum(r[2] for r in rows_jpeg) / len(rows_jpeg)
    n_matched = sum(1 for r in rows_jpeg if r[4])
    j_psnr, j_ssim, _ = stats([(a, b) for a, b, _, _, _ in rows_jpeg])
    return {
        "n": n,
        "sae_bytes": key_b, "sae_bpp": bpp(key_b, model.image_size),
        "sae_psnr": sae_psnr, "sae_ssim": sae_ssim,
        "jpeg_bytes": j_bytes, "jpeg_bpp": bpp(j_bytes, model.image_size),
        "jpeg_psnr": j_psnr, "jpeg_ssim": j_ssim,
        "n_matched": n_matched,
    }


# ======================================================================================
# 6. 主流程
# ======================================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)

    print("=" * 78)
    print("脉冲自编码器训练 SAE (spiking autoencoder)  —  256x256 有损图像压缩")
    print("=" * 78)
    print(f"设备            : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    # 最常见的"明明有显卡却跑在 CPU 上"：装的是 CPU-only 版 PyTorch。
    # 它不报错、版本号看着也正常（比如 2.14.0+cpu），只是编译目标里根本没有 CUDA。
    if device.type == "cpu" and torch.version.cuda is None:
        print("[提示] 当前 PyTorch 是 CPU-only 版（torch.version.cuda 为 None），"
              "即使机器上有 NVIDIA 显卡也只会用 CPU。")
        print("       确认命令：python -c \"import torch;print(torch.__version__, torch.version.cuda)\"")
        print("       换 GPU 版（先按你的 Python 版本选 CUDA 标签，再强制重装）：")
        print("         python -m pip uninstall -y torch torchvision")
        print("         python -m pip install --index-url "
              "https://download.pytorch.org/whl/cu130 torch torchvision")
    print(f"图像尺寸        : {args.image_size}x{args.image_size}x{IMG_CH}")
    print(f"潜层            : C={args.latent_channels}, T={args.num_steps}, "
          f"空间 {args.image_size // 16}x{args.image_size // 16}")
    print(f"损失权重        : ssim={args.w_ssim}  mae={args.w_mae}  mse={args.w_mse}")
    print(f"batch / lr      : {args.batch_size} / {args.lr}     epochs: {args.epochs}")

    train_loader, test_loader, ds_name = build_loaders(args)
    print(f"数据集          : {ds_name}")
    print(f"训练 / 测试     : {len(train_loader.dataset)} / {len(test_loader.dataset)} 张")

    model = SAE(latent_channels=args.latent_channels, num_steps=args.num_steps,
                image_size=args.image_size, beta=args.beta, threshold=args.threshold).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    enc_param = sum(p.numel() for p in model.encoder.parameters())

    raw_bytes = args.image_size * args.image_size * IMG_CH
    print("-" * 78)
    print(f"钥匙大小        : {model.key_bits} bit = {model.key_bytes} 字节 "
          f"({human_bytes(model.key_bytes)})")
    print(f"                = {bpp(model.key_bytes, args.image_size):.3f} bit/像素")
    print(f"相对原始 RGB    : {100 * (1 - model.key_bytes / raw_bytes):.1f}% 压缩 "
          f"({human_bytes(raw_bytes)} -> {human_bytes(model.key_bytes)})")
    print(f"                [注意] 256x256 的 JPEG(q85) 本身也只有 10-15 KB，"
          f"所以这个百分比不能说明任何问题，要看最后的同码率对比")
    print(f"模型参数        : 全部 {n_param:,}  其中编码器(要放边缘端) {enc_param:,} "
          f"= {human_bytes(enc_param * 4)} (fp32)")
    print(f"输出目录        : {args.out} , {args.recon_dir}/")
    print("-" * 78)

    criterion = CompositeLoss(args.w_ssim, args.w_mae, args.w_mse).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.scheduler == "cosine" else None)

    fixed_batch, _ = next(iter(test_loader))
    fixed_batch = fixed_batch[: args.n_samples]
    os.makedirs(args.recon_dir, exist_ok=True)

    best_ssim, last_metrics, history = -1.0, {}, []
    stale, epoch = 0, 0
    last_epoch_sec = None      # 上一轮实测耗时，用来预估还能不能塞下下一轮
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- 墙钟时间预算保护 ----
        # 开新一轮之前先判断：按上一轮的实测耗时，这一轮还跑得完吗？
        # 跑不完就不开，直接收尾。这样"X 小时内训练完"是被强制保证的，而不是估计出来的。
        if args.time_budget_min > 0:
            elapsed_min = (time.time() - t_start) / 60.0
            # 首轮还没有实测数据，用 1.5 分钟保守估计（要算上 CUDA 初始化）
            est_min = (last_epoch_sec / 60.0) if last_epoch_sec else 1.5
            reserve_min = 1.5      # 留给最后的 JPEG 同码率对比 / 画曲线 / 存权重
            if elapsed_min + est_min + reserve_min > args.time_budget_min:
                print(f"\n[时间预算] 已用 {elapsed_min:.1f} 分钟，再跑一轮约需 {est_min:.1f} 分钟，"
                      f"加收尾 {reserve_min:.1f} 分钟会超出 {args.time_budget_min:.0f} 分钟预算。")
                print(f"           在 epoch {epoch - 1} 处提前结束（已跑 {epoch - 1}/{args.epochs} 轮）。")
                epoch = epoch - 1
                break

        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, criterion, args, train=True)
        with torch.no_grad():
            te = run_epoch(model, test_loader, None, device, criterion, args, train=False)
        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        last_epoch_sec = dt
        lr_now = optimizer.param_groups[0]["lr"]
        history.append((epoch, tr, te))
        last_metrics = te

        print(f"epoch {epoch:3d}/{args.epochs} | train {tr['loss']:.4f} "
              f"(mse {tr['mse']:.4f}) | test {te['loss']:.4f} (mse {te['mse']:.4f}) | "
              f"PSNR {te['ms_psnr']:5.2f}dB | SSIM {te['ssim_metric']:.4f} | "
              f"脉冲率 {tr['rate']:.2f} | 熵 {tr['entropy']:.1f}/{args.latent_channels} | "
              f"lr {lr_now:.1e} | {dt:.1f}s")

        # ---- 重建样例 ----
        if args.recon_interval > 0 and (epoch % args.recon_interval == 0 or epoch == 1
                                        or epoch == args.epochs):
            save_recon_grid(model, fixed_batch, device,
                            [os.path.join(args.recon_dir, f"epoch_{epoch:03d}.png"),
                             os.path.join(args.recon_dir, "latest.png")])

        # ---- 存权重：每 5 个 epoch 存一次 + 最后一定存 ----
        if epoch % 5 == 0 or epoch == args.epochs:
            size = save_checkpoint(args.out, model, args, epoch, te)
            if size:
                print(f"          -> 已保存 {args.out}  ({human_bytes(size)})")

        # ---- 按验证集 SSIM 判断是否进步，并决定早停 ----
        # 用 delta 做"显著提升"的阈值，避免 SSIM 在 1e-5 量级抖动就不断刷新最优
        if te["ssim_metric"] > best_ssim + args.early_stop_delta:
            best_ssim = te["ssim_metric"]
            stale = 0
            if args.best_out:
                save_checkpoint(args.best_out, model, args, epoch, te)
        else:
            stale += 1
        if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
            print(f"\n早停：验证 SSIM 连续 {stale} 个 epoch 没有明显提升（最佳 {best_ssim:.4f}）")
            break

    size = save_checkpoint(args.out, model, args, epoch, last_metrics)
    total = (time.time() - t_start) / 60
    print("-" * 78)
    if size:
        print(f"训练完成，用时 {total:.1f} 分钟。权重：{args.out}（{human_bytes(size)}）")
    else:
        print(f"训练结束（用时 {total:.1f} 分钟），但权重没有保存成功，请看上面的 [错误] 行。")
    print(f"最后一个 epoch：PSNR {last_metrics.get('ms_psnr', float('nan')):.2f} dB，"
          f"SSIM {last_metrics.get('ssim_metric', float('nan')):.4f}，"
          f"MSE {last_metrics.get('mse', float('nan')):.5f}")
    print(f"重建样例：{args.recon_dir}/latest.png（上排原图、下排重建图）")

    # ---- 同码率 JPEG 对比 ----
    if args.compare_jpeg > 0:
        print("-" * 78)
        print(f"同码率对比（测试集前 {args.compare_jpeg} 张，JPEG 质量因子二分搜索到相同体积）...")
        res = compare_with_jpeg(model, test_loader, device, args.compare_jpeg, criterion.ssim)
        if res:
            print(f"{'方案':<14}{'字节':>10}{'bit/像素':>11}{'MS-PSNR':>11}{'SSIM':>9}")
            print(f"{'SNN 钥匙':<14}{res['sae_bytes']:>10}{res['sae_bpp']:>11.3f}"
                  f"{res['sae_psnr']:>11.2f}{res['sae_ssim']:>9.4f}")
            print(f"{'JPEG(同码率)':<14}{res['jpeg_bytes']:>10.0f}{res['jpeg_bpp']:>11.3f}"
                  f"{res['jpeg_psnr']:>11.2f}{res['jpeg_ssim']:>9.4f}")
            d_psnr = res['sae_psnr'] - res['jpeg_psnr']
            d_ssim = res['sae_ssim'] - res['jpeg_ssim']
            print(f"{'差值(SNN-JPEG)':<14}{'':>10}{'':>11}{d_psnr:>+11.2f}{d_ssim:>+9.4f}")
            print("→ 差值为正才说明 SNN 方案在同等字节数下真的更好；为负说明还不如直接传 JPEG。")
            if res["n_matched"] < res["n"]:
                print(f"[注意] {res['n'] - res['n_matched']}/{res['n']} 张图：JPEG 即使压到最低质量"
                      f"(q=5)也比钥匙大，两边码率对不上，这个对比不成立。")
                print(f"       说明当前压缩率过于激进。请加大 --latent-channels 或 --num-steps，"
                      f"让钥匙大到 JPEG 也能达到，对比才有意义。")

    # ---- 曲线 ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ep = [h[0] for h in history]
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].plot(ep, [h[1]["mse"] for h in history], label="train")
            ax[0].plot(ep, [h[2]["mse"] for h in history], label="test")
            ax[0].set_title("MSE"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[1].plot(ep, [h[2]["ms_psnr"] for h in history], color="tab:green")
            ax[1].set_title("test MS-PSNR (dB)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
            ax[2].plot(ep, [h[2]["ssim_metric"] for h in history], color="tab:red")
            ax[2].set_title("test SSIM"); ax[2].set_xlabel("epoch"); ax[2].grid(alpha=.3)
            plt.tight_layout(); plt.savefig("training_curves.png", dpi=120)
            print("训练曲线：training_curves.png")
        except Exception as e:
            print(f"[warn] 画曲线失败：{e}")

    print("\n下一步：")
    print(f"  A 机：python compress.py   --image 你的图.jpg --output key.pt --weights {args.out}")
    print(f"  B 机：python decompress.py --key key.pt --output recon.png --weights {args.out}")


if __name__ == "__main__":
    main()
