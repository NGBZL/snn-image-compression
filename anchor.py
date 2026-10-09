# -*- coding: utf-8 -*-
"""anchor.py — 改进5：同组图片的语义关联压缩（锚图 + 残差）

思路
    一批相似的图（同一类花、连续帧、同一片区域的卫星图）里：
      · 第 1 张当**锚图**，按普通方式编码，传输它的潜层 y0
      · 后面每张只编码 **相对锚图的潜层残差** r_i
    接收端：先解锚图，再用 y0 + r_i 还原后续图。

    关键在于**让编码器看得见锚图**：给编码器一路参考输入 ref，
    输出 h = GDN(conv4(f)) + ref_proj(ref)。
    图与锚图越像，网络越容易学到 h≈0，残差就越小、越省比特。

    残差是显式的：解码端潜层 = ref + r_hat，所以解码器本体完全不用改。

传输格式
    [锚图钥匙][残差钥匙 #1][残差钥匙 #2]...

训练
    用 GroupedBatchSampler 保证一个 batch 内的图属于同一类（相似度高），
    第 0 张当锚图，其余用锚图潜层当参考。总损失 = 全部图的失真 + λ*总码率。

用法
    # 默认 = **独立编码（无锚图）**，数据 = Flowers102 + DIV2K/Flickr2K
    python anchor.py train --epochs 30 --batch-size 16 --lam 0.01
    # 复现已被证伪的锚图+残差机制（默认关闭）
    python anchor.py train --epochs 30 --use-anchor --group-size 4 --lam 0.01
    # 对比"独立编码" vs "锚图+残差"的总字节数
    python anchor.py eval --weights anchor_rd.pth --group-size 4

数据
    Flowers102 在 data/flowers-102/；DIV2K+Flickr2K 解压到 data/div2k_extracted/
    （zip 留在 data/div2k/ 原地）。DIV2K/Flickr2K **没有类别标签** ——
    这正是"独立编码"通路不需要标签的原因。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from torchvision import datasets, transforms

from sae_model import human_bytes
from sae_rd import SAE_RD, IMG_CH, encode_latent_bytes, encode_z_bytes
from snntorch import utils


# ======================================================================================
class SAE_Anchor(SAE_RD):
    """在 SAE_RD 上加一路"参考潜层"，两种残差语义。

    residual_mode == "sub"（默认，设计说明的方案 a）
        **硬结构残差**。编码器永远看不到参考：锚图和目标图走的是**完全同一条**
        3 通道通路（就是 `SAE_RD.encode_raw`），所以
            y_i  = encoder(x_i)          # 普通潜层，和"独立编码"角色的定义逐字相同
            y_r  = encoder(x_ref)        # 锚图潜层
        被编码的量不是自由变量，而是**差**：
            coded = quant_hard(y_i) - quant_hard(y_r)      (= quant_hard(y_r) 为参考端)
        解码端做**精确加法** y_full = y_ref_hat + coded_hat。y_ref_hat 是锚图自己
        那串已经量化的潜层，解码端本来就有，所以**不产生任何额外边信息**。
        残差不再需要"学"——它是代数恒等式，所以无用的参考最多退化到
        coded == y_i（代价 = 独立编码 + 先验失配），而不会像旧方案那样把潜层撑大。

    residual_mode == "learned"（旧行为，保留做 A/B）
        coded = encode_raw(x, ref_hint)，参考只作为编码器的输入提示
        （`ref_wiring="inp"` 拼通道 / `"out"` 加偏置），网络自己去学抵消。
        复现旧结果时必须用这个模式，并且 `ref_hint` 也应该是**量化后**的参考潜层
        （旧代码在 evaluate / anchor_bytes 里就是这么传的）。
    """

    def __init__(self, *a, ref_wiring: str = "out", residual_mode: str = "sub", **kw):
        super().__init__(*a, **kw)
        from sae_rd import IMG_CH
        C = self.latent_channels
        if residual_mode not in ("learned", "sub"):
            raise ValueError(f"residual_mode 只能是 learned / sub，收到 {residual_mode!r}")
        self.residual_mode = residual_mode
        self.ref_wiring = ref_wiring
        if ref_wiring == "inp":
            # 输入端拼接：第一层直接吃 3+C 通道（原图 + 上采样后的参考潜层）。
            # 编码器从 (x_i, y_j) **联合**算特征，可以直接学"两张图差在哪"，
            # 而不必在输出端做精确抵消。
            self.conv1 = nn.Conv2d(IMG_CH + C, 32, 3, 2, 1, bias=False)
            self.ref_proj = None
        else:
            # 输出端加偏置（旧方案）。初始化为 0 —— 单位阵会让 r = a + y_j，
            # 同一张图时 r = 2a，起点比不用参考还差。
            self.ref_proj = nn.Conv2d(C, C, 1, bias=False)
            with torch.no_grad():
                self.ref_proj.weight.zero_()

    def _prepare_input(self, x, ref_hint):
        """inp 接线时把参考**提示**拼到输入通道上。

        · learned：ref_hint 是参考潜层（量化前后都行），给编码器当"看得到锚图"的提示。
        · sub    ：`encode_raw` 永远传 None 进来，所以参考通道恒为 0 —— 编码器看到的
          只是一串常量零，不含参考图像的任何信息。这不是"看不见参考"的妥协，而是
          和"锚图自己的编码 路径"保持逐字一致所必需的（见 `encode_raw` 的说明）。
        """
        if self.ref_wiring != "inp":
            return x
        B, _, H, W = x.shape
        if ref_hint is None:
            r = x.new_zeros(B, self.latent_channels, H, W)      # 参考通道置 0
        else:
            r = ref_hint.mean(dim=1)                            # [B,T,C,h,w] -> 对时间取均值
            r = F.interpolate(r, size=(H, W), mode="bilinear", align_corners=False)
        return torch.cat([x, r], dim=1)                        # [B, 3+C, H, W]

    # ------------------------------------------------------------------ 编码器
    def encode_raw(self, x, ref_hint=None):
        """跑 T 步，返回编码器的**原始输出** [B,T,C,h,w]。

        ref_hint 只是给编码器的参考提示，**不是**被编码的量：
          · learned：沿用旧语义，提示进 `_prepare_input` / `ref_proj`；编码器输出
            就是"学出来的残差"，由 `code_residual` 原样透传。
          · sub    ：编码器**看不到参考** —— `_prepare_input` 里的参考通道恒为 0
            （`ref_hint` 被完全忽略），所以输出就是普通潜层 y_i，和独立编码角色
            的定义逐字相同（见 `code_residual` 里的 sub 分支）。

        注意 `_prepare_input(x, None)` 在 `inp` 接线下是"3 通道图 + C 个零通道"，
        这正是**锚图走的那条路**（train 里 ref=None 的锚图、以及改动前的
        `forward(x, ref=None)`），所以 sub 模式把那串常量零当成普通输入通道，
        和"锚图自己的编码器"逐字一致。对 `inp` 的旧 checkpoint（如 v5，conv1 是
        35 通道）这是必须的：它学到的 BatchNorm 统计量就是按 35 通道标定的，
        只喂前 3 通道会让潜层尺度漂 1.6 倍（实测 std 71 -> 115）。对**从头训练**的
        sub 模型，直接构造 `ref_wiring="out"` 就是纯 3 通道 conv1，没有任何零通道。
        两种情况下列零通道都是**常量**，不含参考图像的任何信息。

        真正的减法是 `code_residual`，加法是 `reconstruct`，各只有一处。
        """
        from sae_rd import _lif_fire
        # sub：ref_hint 丢弃 -> 参考通道恒 0，编码器不可能看到参考信息
        xin = self._prepare_input(x, ref_hint if self.residual_mode == "learned" else None)
        utils.reset(self)
        outs = []
        for t in range(self.num_steps):
            h = _lif_fire(self.lif1, self.bn1(self.conv1(xin)))
            h = _lif_fire(self.lif2, self.bn2(self.conv2(h)))
            h = _lif_fire(self.lif3, self.bn3(self.conv3(h)))
            h = self.gdn(self.conv4(h))                        # [B,C,h,w]
            if self.residual_mode == "learned" and self.ref_wiring == "out" \
                    and ref_hint is not None:
                h = h + self.ref_proj(ref_hint[:, t])
            outs.append(h)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def encode_ref_latent(self, x):
        """把 x 编成**硬量化**的参考潜层 y_ref_hat（解码端能精确拿到的那串值）。

        这是 sub 模式下参考端的唯一来源：调用方必须把它的返回值当 `ref` 传回去，
        而不是传 `forward` 里的 `coded_hat` 或任何量化前的量。
        """
        return self.quant_hard(self.encode_raw(x, None))

    # ------------------------------------------------------- sub 模式的减法 / 加法
    @staticmethod
    def quant_hard(z, step: float = 1.0):
        """硬量化（前向 = round，反向 = STE），**全仓库唯一的硬量化定义**。

        为什么必须是 STE：`y_ref_hat` 要参与 `coded = quant_hard(y_i) - y_ref_hat`，
        如果这里 `detach()` 掉，锚图编码器就收不到任何来自残差通路的梯度；
        用 `round(...) + (z - z.detach())` 则前向精确等于量化值、反向梯度为 1。
        """
        return torch.round(z / step) * step + (z - z.detach())

    def code_residual(self, y, y_ref_hat=None):
        """被编码的量：`quant_hard(y_i) - y_ref_hat`。**唯一的减法定义**。

        y_ref_hat 必须已经是硬量化值（`quant_hard` 的输出）——解码端只可能拿到
        精确的量化值，训练时用别的量就是 train/deploy 失配。锚图（y_ref_hat=None）
        返回 quant_hard(y_i)，和独立编码路径逐字相同。
        """
        if self.latent_mode == "bernoulli":
            return y if y_ref_hat is None else (y - y_ref_hat)
        r = self.quant_hard(y) if self.residual_mode == "sub" else y
        return r if y_ref_hat is None else (r - y_ref_hat)

    def reconstruct(self, coded_hat, y_ref_hat=None):
        """解码端潜层：`y_ref_hat + coded_hat`。**唯一的加法定义**。"""
        return coded_hat if y_ref_hat is None else (y_ref_hat + coded_hat)

    def _code_and_rate(self, coded, ref_used):
        """coded -> (coded_hat, rate)。超先验 / 先验 / z 流全部作用在 coded 上。"""
        bits_z = torch.zeros((), device=coded.device)
        extra, z_hat, z_step = None, None, None
        if self.latent_mode == "bernoulli":
            coded_hat = coded
            bits_y = self.prior.bits(coded_hat)
        else:
            if self.hyperprior is not None:
                extra, bits_z, z_hat, z_step = self.hyperprior(coded, self.training)
            coded_hat = self.quant(coded)
            bits_y = self.prior.bits(coded_hat, self.quant.step, extra)
        bpp = (bits_y.sum() + bits_z) / (coded.shape[0] * self.n_pix)
        rate = {"bits_y": bits_y, "bits_z": bits_z, "bpp": bpp,
                "extra_sigma": extra, "z_hat": z_hat, "z_step": z_step,
                "ref_used": ref_used}
        return coded_hat, rate

    def forward(self, x, ref=None, checkpoints=None):
        """ref: **硬量化**参考潜层 y_ref_hat [B,T,C,h,w]；锚图传 None。

        被编码的量 = `code_residual(encode_raw(x, hint), ref)`；
        解码端潜层 = `reconstruct(coded_hat, ref)`。两处都只有这一个定义。
        """
        y = self.encode_raw(x, ref)
        coded = self.code_residual(y, ref)
        coded_hat, rate = self._code_and_rate(coded, ref)
        y_full = self.reconstruct(coded_hat, ref)
        return coded_hat, rate, self.decode(y_full, checkpoints)

    @torch.no_grad()
    def decode_with_ref(self, coded_hat, ref=None, checkpoints=None):
        """解码端接口：只拿到被编码的残差和**硬量化**参考潜层，重建图像。"""
        return self.decode(self.reconstruct(coded_hat, ref), checkpoints)


# ======================================================================================
# 同类别分组采样
# ======================================================================================
class GroupedBatchSampler(Sampler):
    """每个 batch 取同一类别的 group_size 张图，构造"相似图一组"的场景。"""

    def __init__(self, labels, group_size: int, seed: int = 0, drop_last: bool = True):
        self.labels = np.asarray(labels)
        self.g = max(2, int(group_size))
        self.rng = np.random.RandomState(seed)
        by_cls = {}
        for i, y in enumerate(self.labels):
            by_cls.setdefault(int(y), []).append(i)
        # 只保留样本数够组成一组的类
        self.groups = [np.array(v) for k, v in sorted(by_cls.items()) if len(v) >= self.g]
        # 每个类每轮产出 floor(n/g) 个互不重叠的组 —— 一个 epoch 覆盖整个数据集。
        # 之前每个类只产出 1 组（102 类 x 16 = 1632 张/轮），大部分数据每轮都看不到，
        # 反复重采样同样的图正是过拟合的主因；加数据却不改这里等于白加。
        self.per_cls = [max(1, len(idx) // self.g) for idx in self.groups]
        self.drop_last = drop_last

    def __iter__(self):
        order = self.rng.permutation(len(self.groups))
        for gi in order:
            idx = self.groups[gi]
            k = self.per_cls[gi]
            if k <= 1:
                yield self.rng.choice(idx, size=self.g, replace=False).tolist()
                continue
            perm = self.rng.permutation(len(idx))          # 每轮打乱 -> 组内容每轮不同
            for j in range(k):
                yield idx[perm[j * self.g:(j + 1) * self.g]].tolist()

    def __len__(self):
        return int(sum(self.per_cls))


def _labels_of(ds):
    for attr in ("_labels", "labels", "targets", "y"):
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            return np.asarray(v if not torch.is_tensor(v) else v.numpy())
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):     # Subset
        return _labels_of(ds.dataset)[np.asarray(ds.indices)]
    if hasattr(ds, "datasets"):                                # ConcatDataset
        return np.concatenate([_labels_of(d) for d in ds.datasets])
    raise ValueError("拿不到数据集标签，无法做分组采样")


# ======================================================================================
# 无标签图片文件夹（DIV2K / Flickr2K）
# ======================================================================================
IMG_EXTS = (".png", ".jpg", ".jpeg")
DIV2K_SUBDIR = "div2k_extracted"


def _list_images(root, limit: int = 0, seed: int = 0):
    """递归 glob 收集 root 下所有 .png/.jpg/.jpeg 路径，排序后返回。"""
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()                       # 固定顺序 -> 训练集/留出集索引对齐
    if limit and limit > 0 and len(paths) > limit:
        rng = np.random.RandomState(seed)
        sel = np.sort(rng.choice(len(paths), size=int(limit), replace=False))
        paths = [paths[int(i)] for i in sel]
    return paths


class ImageFolderFlat(Dataset):
    """递归图片文件夹数据集，**不需要类别标签**（DIV2K/Flickr2K 就没有标签）。

    `_labels` 是**合成**的 shard id（每 16 张连续文件算一组），只为了让仍然要求
    分组的旧锚图通路（`--use-anchor`）不至于因为没标签而崩掉；它不携带任何语义
    类别信息，"独立编码"（默认通路）完全不读它。
    """

    SHARD = 16

    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform
        self._labels = np.asarray([i // self.SHARD for i in range(len(self.paths))],
                                  dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")     # DIV2K 有灰度/带 alpha 的，统一成 3 通道
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self._labels[i])


# ======================================================================================
def build_data(args):
    """返回 (训练集, 留出集)。

    `--data`：
      · flowers —— Flowers102 的 train+val+test **全量合并（8189 张）后重新划分**。
        之前只用 train+val = 2040 张，等于把 75% 的本地数据闲置着；而实测过拟合
        峰值出现在第 20~100 轮，数据量正是瓶颈。
      · div2k   —— 递归 glob 解压后的 DIV2K+Flickr2K（**无类别标签**，这正是重点）。
      · both    —— 两者拼接（默认）。

    留出集用同一份 `perm` 索引，所以两个数据集的**顺序必须逐字对齐**：
    flowers 两次构造的文件顺序相同，div2k 两次复用同一个已排序的 `d2_paths`。
    留出集始终走 `post`（无增强）通路。
    """
    aug = []
    if args.augment == "crop":
        # 原图约 500x700（DIV2K 是 2K），缩到 128/256 有大量冗余，随机裁剪+翻转等效扩充
        aug = [transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1.0)),
               transforms.RandomHorizontalFlip()]
    elif args.augment == "flip":
        aug = [transforms.RandomHorizontalFlip()]
    post = [transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size)]

    def _flowers(tf):
        return torch.utils.data.ConcatDataset([
            datasets.Flowers102(args.data_dir, split=sp, download=False, transform=tf)
            for sp in ("train", "val", "test")])

    # 老脚本（anchor_bytes.py / _verify_fix*.py / recalibrate_prior.py / _residual_*.py）
    # 自己造 Namespace 调 build_data，里面没有 `data` 字段 —— 那时 build_data 只认
    # Flowers102，所以缺省必须回落成 "flowers" 才能逐字复现它们的结果。
    mode = getattr(args, "data", "flowers")
    want_fl = mode in ("flowers", "both")
    want_d2 = mode in ("div2k", "both")
    d2_paths = []
    if want_d2:
        root = os.path.join(args.data_dir, DIV2K_SUBDIR)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"找不到解压后的 DIV2K/Flickr2K 目录 {root}；"
                f"把 data/div2k/*.zip 解压到该目录即可（不要重新下载）")
        d2_paths = _list_images(root, getattr(args, "div2k_limit", 0), args.seed)
        if not d2_paths:
            raise FileNotFoundError(f"{root} 下没有 .png/.jpg/.jpeg")
        print(f"DIV2K/Flickr2K: 扫描到 {len(d2_paths)} 张（--div2k-limit="
              f"{getattr(args, 'div2k_limit', 0)}）")

    tr_parts, ev_parts = [], []
    if want_fl:
        tr_parts.append(_flowers(transforms.Compose(aug + post + [transforms.ToTensor()])))
        ev_parts.append(_flowers(transforms.Compose(post + [transforms.ToTensor()])))
    if d2_paths:
        tr_parts.append(ImageFolderFlat(
            d2_paths, transforms.Compose(aug + post + [transforms.ToTensor()])))
        ev_parts.append(ImageFolderFlat(
            d2_paths, transforms.Compose(post + [transforms.ToTensor()])))

    all_tr = torch.utils.data.ConcatDataset(tr_parts)
    all_ev = torch.utils.data.ConcatDataset(ev_parts)
    n = len(all_tr)
    n_fl = len(tr_parts[0]) if want_fl else 0
    n_d2 = len(d2_paths)
    n_hold = max(200, int(n * args.holdout_frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    hold, tr = perm[:n_hold], perm[n_hold:]
    if args.eval_limit > 0 and len(hold) > args.eval_limit:
        hold = hold[:args.eval_limit]
    print(f"数据[{mode}]: 全量 {n} 张 (flowers {n_fl} + div2k/flickr2k {n_d2}) "
          f"-> 训练 {len(tr)} / 留出 {len(hold)} (增强={args.augment})")
    return torch.utils.data.Subset(all_tr, tr), torch.utils.data.Subset(all_ev, hold)


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 两个免费的 GPU 加速：输入形状固定，cudnn.benchmark 能挑到最快的卷积算法；
    # TF32 在 Ampere+ 上对 matmul/卷积有明显提速。都不影响数值结果的可复现性要求。
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU {torch.cuda.get_device_name(0)}  cudnn.benchmark=True  TF32=high  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    tr, te = build_data(args)
    # ------------------------------------------------------------------------------
    # 采样器：锚图通路 (--use-anchor) 需要同组同类图，所以按标签分组；
    # 关闭锚图（**默认**）时每张图都是独立的，标签没有意义（DIV2K 根本没有标签），
    # 用最普通的 RandomSampler，batch 大小由 --batch-size 决定（--group-size 不再影响）。
    # ------------------------------------------------------------------------------
    use_anchor = bool(getattr(args, "use_anchor", False))
    if use_anchor:
        labels = _labels_of(tr)
        sampler = GroupedBatchSampler(labels, args.group_size, seed=args.seed)
        loader = DataLoader(tr, batch_sampler=sampler, num_workers=args.workers,
                            pin_memory=(dev.type == "cuda"),
                            # 复用 worker 进程：默认每个 epoch 都会销毁重建 8 个进程，
                            # 每轮白扔 0.5~2 秒，440 轮累计 4~15 分钟
                            persistent_workers=(args.workers > 0))
        print(f"设备 {dev}  训练图 {len(tr)}  类别组 {len(sampler)}  "
              f"group_size={args.group_size}（锚图通路，每 batch {args.group_size} 张）")
    else:
        sampler = RandomSampler(tr, generator=torch.Generator().manual_seed(args.seed))
        loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                            drop_last=(len(tr) >= args.batch_size),
                            num_workers=args.workers, pin_memory=(dev.type == "cuda"),
                            persistent_workers=(args.workers > 0))
        print(f"设备 {dev}  训练图 {len(tr)}  独立编码（无锚图）  "
              f"batch_size={args.batch_size}  每轮 {len(loader)} 个 batch  "
              f"每轮图数={len(loader) * args.batch_size}")
    if not args.residual_mode:
        args.residual_mode = "sub"
    if use_anchor:
        print(f"残差语义 residual_mode={args.residual_mode}"
              + ("（硬结构残差：coded = quant_hard(y_i) - quant_hard(y_ref)，"
                 "编码器看不到参考）" if args.residual_mode == "sub" else
                 "（旧行为：参考只当编码器输入提示）"))

    model = SAE_Anchor(latent_channels=args.latent_channels, num_steps=args.num_steps,
                       image_size=args.image_size, latent_mode="gaussian",
                       use_hyperprior=args.use_hyperprior, z_channels=args.z_channels,
                       learn_step=False, norm_type=args.norm,
                       ref_wiring=args.ref_wiring,
                       residual_mode=args.residual_mode).to(dev)
    if args.hot_start and os.path.isfile(args.hot_start):
        nk, nt = model.load_encoder_front(args.hot_start)
        print(f"热启动 {args.hot_start}: {nk} 个参数")
    s0 = model.warmup_sigma(loader, dev, n_batches=3)
    print(f"sigma 预热: 潜层 std {s0:.4f}")
    if use_anchor and args.residual_mode == "sub":
        # mode (a)：被编码的残差是**两个潜层之差**，尺度和"潜层本身"通常不是一个量级。
        # warmup_sigma 拟合的是潜层 std，先验可能系统性偏大；训练日志里的
        # y_std（现在是 coded 的 std）能直接看出失配程度。
        print("    [sub] 被编码量 = quant_hard(y_i) - quant_hard(y_ref)，"
              "编码器/解码器都不看参考的通道")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    anneal = max(1, int(args.epochs * 0.8))
    g = args.group_size        # 只有 --use-anchor 通路才有意义
    g_div = g if use_anchor else max(1, args.batch_size)   # 日志归一化分母

    def _save(path, ep):
        # 存盘前先检查权重是否健康，绝不把 NaN 权重写成 checkpoint
        bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            print(f"          !! 检测到 {len(bad)} 个参数是 NaN/inf，拒绝存盘"
                  f"（例如 {bad[:3]}）。保留上一个健康 checkpoint。")
            return False
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "image_size": model.image_size,
                    "latent_channels": model.latent_channels, "num_steps": model.num_steps,
                    "z_channels": args.z_channels, "group_size": g,
                    "latent_mode": "gaussian", "use_hyperprior": model.use_hyperprior,
                    "ref_wiring": model.ref_wiring, "ref_mode": args.ref_mode,
                    "residual_mode": model.residual_mode,
                    "use_anchor": use_anchor, "batch_size": args.batch_size,
                    "data": getattr(args, "data", ""),
                    "norm_type": getattr(model, "norm_type", "gn")}, path)
        return True

    start_ep = 1
    t_start = time.time()
    last_sec = 30.0        # 首轮耗时估计（秒）
    bad_steps = 0          # 被 NaN 保护跳过的步数

    # 固定的验证 batch（来自测试集，训练从没见过）。用于 eval 模式的真实验证。
    val_batches = []
    if args.val_batches > 0 and len(te) > 0:
        _vl = DataLoader(te, batch_size=8, shuffle=False, num_workers=0)
        for _i, (_vb, _) in enumerate(_vl):
            if _i >= args.val_batches:
                break
            val_batches.append(_vb.to(dev))
    # 按真实验证集记录最优；只存最优，避免最后几轮退化把前面白跑
    best_psnr = -1e9
    best_path = os.path.splitext(args.out)[0] + "_best.pth"

    if args.resume and os.path.isfile(args.resume):
        ck0 = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck0["state_dict"])
        if "opt" in ck0:
            try:
                opt.load_state_dict(ck0["opt"])
            except Exception:
                pass
        start_ep = int(ck0.get("epoch", 0)) + 1
        for _ in range(start_ep - 1):
            sched.step()
        print(f"[断点续训] 从 {args.resume} 恢复到 epoch {start_ep}")

    for ep in range(start_ep, args.epochs + 1):
        # 墙钟时间预算：开新一轮前先估算，跑不完就不开，保证按时结束且权重完整
        if args.time_budget_min > 0 and ep > start_ep:
            el = (time.time() - t_start) / 60.0
            if el + last_sec / 60.0 + 1.0 > args.time_budget_min:
                print(f"\n[时间预算] 已用 {el:.1f} 分钟，再跑一轮约需 {last_sec/60:.1f} 分钟，"
                      f"超出 {args.time_budget_min:.0f} 分钟预算 -> 在 epoch {ep-1} 提前结束")
                _save(args.out, ep - 1)
                break
        ns = max(args.noise_floor, 1.0 - (ep - 1) / anneal)
        model.quant.noise_scale = ns
        if model.hyperprior is not None:
            model.hyperprior.z_quant.noise_scale = ns
        model.train(True)
        agg = np.zeros(4)
        t0 = time.time()
        nb = args.max_batches if args.max_batches > 0 else len(loader)
        for bi, (x, _) in enumerate(loader):
            if bi >= nb:
                break
            x = x.to(dev)                       # 有锚图: [g,3,H,W] 全同类；无锚图: [B,3,H,W]
            g_eff = x.shape[0]
            if not use_anchor:
                # ---- 独立编码（默认）：每张图各自 forward，ref=None ----
                # 没有锚图、没有残差、没有分组；`--anchor-frac` / `--group-size` /
                # `--ref-mode` 在这个通路里全都不起作用。
                r, rate, rec = model(x, ref=None, checkpoints={model.num_steps})
                rec_full = rec[model.num_steps]
                bits = rate["bits_y"].sum() + rate["bits_z"]
            else:
                # ---- 1) 选锚图 + 给残差图分配参考 ----
                # anchor_frac 控制"多少张当锚图"。原实现固定只有第 0 张是锚图，
                # 于是 7/8 的训练样本走残差通路，纯锚图编码成了分布外输入 ——
                # 而评估里的"独立编码"基线恰恰全是锚图，导致 train/eval 差 6.1 倍。
                n_anchor = max(1, min(g_eff, int(round(g_eff * args.anchor_frac))))
                anchor_idx = sorted(np.random.choice(g_eff, size=n_anchor, replace=False).tolist())
                parent = np.full(g_eff, -1, dtype=int)
                if args.ref_mode != "star":
                    for i in range(g_eff):
                        if i in anchor_idx:
                            continue
                        # 只参考"真锚图"：这样参考端一定是锚图潜层，
                        # 与 MST 解码时"参考必须先被独立解码出来"保持一致
                        parent[i] = int(np.random.choice(anchor_idx))
                # ---- 2) 硬量化锚图潜层 y_ref_hat，作为参考 ----
                # 参考端必须是**解码端真正拿到的东西**：量化误差在解码端不可逆，
                # 用未量化的潜层当参考就是 train/deploy 失配。
                with torch.no_grad():
                    ra = model.encode_ref_latent(x)                # [g,T,C,h,w]
                ref_batch = ra[parent.clip(min=0)].clone()
                ref_batch[parent < 0] = 0.0                        # 锚图位置：参考置零（不用）
                # ---- 3) 分组前向：锚图 ref=None；残差图 ref=y_ref_hat ----
                # 必须分两次算：锚图的参考语义是 None（不是全零张量），而残差图的参考
                # 是一串**已经量化好的** y_ref_hat。anchor_idx 通常只有 1~4 张，
                # 两次前向的 batch 仍是 g，GPU 利用率和原来单次前向没有区别。
                # `r` 只用于日志，指向最后算的那一批（残差或锚图），不影响损失。
                bidx = torch.as_tensor(anchor_idx, dtype=torch.long, device=dev)
                rec_full = torch.zeros(g_eff, x.shape[1], *x.shape[2:], device=dev)
                bits = torch.zeros((), device=dev)
                if len(anchor_idx) > 0:
                    xa = x[bidx]
                    ra_hat, rate_a, rec_a = model(xa, ref=None,
                                                  checkpoints={model.num_steps})
                    rec_full[bidx] = rec_a[model.num_steps]
                    bits = bits + rate_a["bits_y"].sum() + rate_a["bits_z"]
                    r = ra_hat
                nidx = [i for i in range(g_eff) if parent[i] >= 0]
                if nidx:
                    nidx_t = torch.as_tensor(nidx, dtype=torch.long, device=dev)
                    xn = x[nidx_t]
                    # 参考端在用之前必须是对应父锚图的 y_ref_hat，且按 nidx 重排
                    ref_n = ref_batch[nidx_t]
                    rn_hat, rate_n, rec_n = model(xn, ref=ref_n,
                                                  checkpoints={model.num_steps})
                    rec_full[nidx_t] = rec_n[model.num_steps]
                    bits = bits + rate_n["bits_y"].sum() + rate_n["bits_z"]
                    r = rn_hat
            D = F.mse_loss(rec_full, x)                        # 按图平均
            L = D + args.lam * (bits / (g_eff * model.n_pix))
            # NaN 保护：单个坏 batch 一旦反向传播，权重会永久变成 NaN
            # （实测第 27 轮突然全 NaN，之后 80 多轮全在空转）。
            # 这里检测到非有限损失就跳过这一步，绝不让它污染权重。
            if not torch.isfinite(L):
                bad_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            L.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 梯度里若出现 NaN/inf，同样跳过这一步
            bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in model.parameters())
            if bad_grad:
                bad_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            agg += [L.item(), D.item() / g_div, bits.item() / (g_div * model.n_pix), 1]
        sched.step()
        n = max(1, agg[3])
        last_sec = time.time() - t0

        # ---- 真实验证：eval 模式 + 硬量化 + ref=None ----
        # 训练里的 D 是 train 模式（BN 用 batch 统计量）+ 带噪量化的口径，
        # 实测能比部署时的真实表现乐观 25 dB。不做这一步就等于闭眼跑 9 小时。
        eval_psnr = float("nan")
        if args.val_every > 0 and val_batches and (ep % args.val_every == 0 or ep == 1):
            model.eval()
            for _m in model.modules():
                if hasattr(_m, "hard"):
                    _m.hard = True
            with torch.no_grad():
                ps = []
                for vb in val_batches:
                    utils.reset(model)
                    _r, _rate, _rec = model(vb, ref=None, checkpoints={model.num_steps})
                    recon = _rec[model.num_steps].clamp(0, 1)
                    ps.append(10 * torch.log10(
                        1.0 / F.mse_loss(recon, vb).clamp_min(1e-12)).item())
                eval_psnr = float(sum(ps) / max(1, len(ps)))
            model.train(True)
            for _m in model.modules():
                if hasattr(_m, "hard"):
                    _m.hard = False
            # 按"真实验证"存最优：实测 v5 峰值在 ep85/360、v7 在 ep20/1300，
            # 只存最后一轮等于把训练时间全浪费在退化的模型上。
            if eval_psnr == eval_psnr and eval_psnr > best_psnr:
                if _save(best_path, ep):
                    best_psnr = eval_psnr
                    print(f"          -> ★新的最优 ({eval_psnr:.2f}dB @ ep{ep}) 已存 {best_path}")

        print(f"ep {ep:3d}/{args.epochs} | L {agg[0]/n:.4f} | D {agg[1]/n:.4f} | "
              f"{'组均' if use_anchor else '独立'}码率 {agg[2]/n:.4f} bpp | "
              f"sigma {float(model.prior.sigma().detach().mean()):.3f} | "
              f"噪声 {ns:.2f} | {time.time()-t0:.1f}s"
              + f" | y_std {float(r.detach().std()):.2f} y_max {float(r.detach().abs().max()):.0f}"
              + (f" | 跳过{bad_steps}步" if bad_steps else "")
              + (f" | ★真实验证 {eval_psnr:5.2f}dB" if eval_psnr == eval_psnr else ""))
            # 周期性存盘：绝不能再出现"跑了半天没存下来"
        if args.save_every > 0 and ep % args.save_every == 0:
            _save(args.out, ep)
            print(f"          -> checkpoint 已保存 (epoch {ep})"
                  + (f" 当前最优 {best_psnr:.2f}dB" if best_psnr > -1e8 else ""))

    ck = args.out
    _save(ck, args.epochs)
    print(f"已保存 {ck} ({human_bytes(os.path.getsize(ck))})")


def _norm_of(obj):
    """推断 checkpoint 的解码器归一化类型。

    老 checkpoint（如 v5）没有 norm_type 字段，但它们的 state_dict 里带
    BatchNorm 的 running_mean；据此自动判断，否则 v5 这类权重根本加载不进来。
    """
    nt = obj.get("norm_type")
    if nt:
        return nt
    keys = obj.get("state_dict", {})
    return "bn" if any(k.startswith("bn_") and k.endswith("running_mean") for k in keys) else "gn"


def build_model(args, obj, dev):
    """按 checkpoint 里的结构参数建模型并加载权重（evaluate / eval_allpairs 共用）。

    `residual_mode`：checkpoint 里有就用它（v9 之后的 ckpt 都写进去了）；
    没有的（v5~v8）默认 `learned` —— 那是它们训练时真正的语义，默认成 `sub`
    会静默改变这些 checkpoint 的含义。命令行给了 `--residual-mode` 时以命令行为准
    （用来对**同一个** checkpoint 做 learned / sub 的 A/B）。
    """
    rm = getattr(args, "residual_mode", "") or obj.get("residual_mode", "learned")
    model = SAE_Anchor(latent_channels=obj["latent_channels"], num_steps=obj["num_steps"],
                       image_size=obj["image_size"], latent_mode="gaussian",
                       use_hyperprior=obj.get("use_hyperprior", True),
                       z_channels=obj.get("z_channels", 48),
                       norm_type=_norm_of(obj),
                       ref_wiring=obj.get("ref_wiring", "out"),
                       residual_mode=rm).to(dev)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    for m in model.modules():
        if hasattr(m, "hard"):
            m.hard = True
    return model


# ======================================================================================
def container_bits(model, y_hat, rate, kmax: int | None = None, z_kmax: int | None = None):
    """真实容器口径的码流比特数：(总比特, y 流比特, z 流比特)。

    这里就是 `pair_bits_matrix` 一直在用的口径，抽出来共用：
      · y 流 **必须** 带 hyperprior 的逐像素 sigma（extra_sigma），
        否则编码器用逐通道粗模型、解码器用逐像素细模型，真实字节会大一大截；
      · z 流（边信息）**必须**单独算 —— 它对 v5 是整张图的 ~31%；
      · 字母表按符号真实范围自适应（和 `codec/rd_codec.py::_pick_kmax` 同一个函数）。
        写死 ±32 时 |y|≈900 的符号全被压进边界桶，会系统性低估 y 流。
    以前 `evaluate` 前两条都没做，还把 `encode_latent_bytes` 返回的**比特数**
    当成"总字节"打印，于是所有下游数字都有 8 倍单位错、还少算三成码流。
    """
    from codec.rd_codec import DEF_KMAX, _pick_kmax
    if kmax is None:
        kmax = _pick_kmax(torch.round(y_hat.detach()).long())
    if z_kmax is None:
        z_kmax = DEF_KMAX
    yb, _ns, _p, _y = encode_latent_bytes(y_hat, model.quant.step, model.prior,
                                          extra_sigma=rate.get("extra_sigma"), k_max=kmax)
    zb = 0
    if rate.get("z_hat") is not None and model.hyperprior is not None:
        zb, _ = encode_z_bytes(rate["z_hat"], rate["z_step"],
                               model.hyperprior.z_prior, k_max=z_kmax)
    return int(yb) + int(zb), int(yb), int(zb)


# ======================================================================================
def evaluate(args):
    """对比：每张图独立编码 vs 锚图+残差，一组的总字节数。

    单位说明：`container_bits` 返回**比特**，这里一律 /8 换算成**字节**再打印，
    标签里写清楚 B（字节）还是 bit（比特）。bpp = 总比特 / (张数 * image_size^2)。
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(args.weights, map_location="cpu", weights_only=False)
    # 数据尺寸必须跟 checkpoint 走：模型是 256 训的、却用 128 的数据评估，
    # 会让 model.n_pix(65536) 和真实像素数(16384) 差 4 倍，
    # 而且 model.latent_size(16) 和真实潜层边长(8) 对不上（历史上就是这么评的）。
    ck_size = int(obj.get("image_size", args.image_size))
    if ck_size != args.image_size:
        print(f"[eval] 数据尺寸跟随 checkpoint: image_size={ck_size}"
              f"（命令行给的是 {args.image_size}）")
        args.image_size = ck_size
    _, te = build_data(args)
    model = build_model(args, obj, dev)
    labels = _labels_of(te)
    g = args.group_size

    by_cls = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    groups = [np.array(v) for v in by_cls.values() if len(v) >= g][: args.n_groups]
    n_img = len(groups) * g
    n_pix = model.image_size * model.image_size
    print(f"评估 {len(groups)} 组，每组 {g} 张同类图（image_size={model.image_size}）")
    print("口径：y 流带逐像素 hyperprior sigma + z 流（= 真实容器）；单位已换算成字节\n")

    tot_ind_bits = tot_anc_bits = tot_anchor_bits = tot_res_bits = 0
    tot_ind_z = tot_anc_z = 0
    pers = []
    with torch.no_grad():
        for gi, idx in enumerate(groups):
            xs = torch.stack([te[int(i)][0] for i in idx[:g]]).to(dev)
            # --- 独立编码：每张都当锚图 ---
            ind_psnr, anc_psnr = [], []
            ib = iz = 0
            for i in range(g):
                utils.reset(model)
                r, rate, rec = model(xs[i:i + 1], ref=None, checkpoints={model.num_steps})
                b, _yb, zb = container_bits(model, r, rate)
                ib += b
                iz += zb
                ind_psnr.append(10 * math.log10(1.0 / max(1e-12,
                                 F.mse_loss(rec[model.num_steps], xs[i:i + 1]).item())))
            # --- 锚图 + 残差 ---
            utils.reset(model)
            y0 = model.encode_raw(xs[0:1], None)               # 锚图原始潜层（只有这一处会用到）
            ref = model.code_residual(y0, None) if model.residual_mode == "sub" \
                else _quantize(model, y0)                      # = y_ref_hat，解码端能拿到的值
            coded0, rate0 = model._code_and_rate(ref, None)
            rec0 = model.decode(model.reconstruct(coded0, None),
                                checkpoints={model.num_steps})
            ab, _y0, z0 = container_bits(model, coded0, rate0)
            anchor_b = ab
            anchor_z = z0
            anc_psnr.append(10 * math.log10(1.0 / max(1e-12,
                             F.mse_loss(rec0[model.num_steps], xs[0:1]).item())))
            res_b = 0
            res_z = 0
            for i in range(1, g):
                utils.reset(model)
                ri, ratei, reci = model(xs[i:i + 1], ref=ref, checkpoints={model.num_steps})
                b, _yb, zb = container_bits(model, ri, ratei)
                res_b += b
                res_z += zb
                anc_psnr.append(10 * math.log10(1.0 / max(1e-12,
                                 F.mse_loss(reci[model.num_steps], xs[i:i + 1]).item())))
            tot_ind_bits += ib
            tot_anc_bits += anchor_b + res_b
            tot_anchor_bits += anchor_b
            tot_res_bits += res_b
            tot_ind_z += iz
            tot_anc_z += anchor_z + res_z
            pers.append((ib, anchor_b + res_b, np.mean(ind_psnr), np.mean(anc_psnr)))
            if gi < 5:
                print(f"  组{gi+1}: 独立 {ib/8:8.1f} B -> 锚图+残差 {(anchor_b+res_b)/8:8.1f} B "
                      f"(锚 {anchor_b/8:.1f} + 残差 {res_b/8:.1f})   "
                      f"PSNR {np.mean(ind_psnr):.2f} -> {np.mean(anc_psnr):.2f} dB")

    print(f"\n{'-'*76}")
    print(f"总字节（{len(groups)} 组 × {g} 张 = {n_img} 张图）")
    print(f"  独立编码      : {tot_ind_bits/8:10.1f} B   平均 {tot_ind_bits/8/n_img:8.1f} B/张"
          f"   ({tot_ind_bits/n_img/n_pix:.3f} bpp)")
    print(f"  锚图+残差     : {tot_anc_bits/8:10.1f} B   平均 {tot_anc_bits/8/n_img:8.1f} B/张"
          f"   ({tot_anc_bits/n_img/n_pix:.3f} bpp)")
    print(f"  其中 锚图 {tot_anchor_bits/8:.1f} B + 残差 {tot_res_bits/8:.1f} B")
    print(f"  z 流（边信息）: 独立 {tot_ind_z/8:.1f} B "
          f"({100*tot_ind_z/max(1,tot_ind_bits):.1f}% 的独立码流) -> "
          f"锚+残差 {tot_anc_z/8:.1f} B ({100*tot_anc_z/max(1,tot_anc_bits):.1f}%)")
    print(f"  => 省了 {100*(1-tot_anc_bits/max(1,tot_ind_bits)):.1f}%   "
          f"({tot_ind_bits/max(1,tot_anc_bits):.2f}x)")
    print(f"  残差/锚图 体积比 = {tot_res_bits/max(1,tot_anchor_bits)/(g-1):.3f} "
          f"(远小于 1 说明确实在编码差异而不是重复编码)")
    print(f"  平均 PSNR: 独立 {np.mean([p[2] for p in pers]):.2f} dB -> "
          f"锚图+残差 {np.mean([p[3] for p in pers]):.2f} dB")
    out = dict(n_groups=len(groups), group_size=g, image_size=model.image_size,
               unit="bytes（容器口径：y 流带 extra_sigma + z 流；不含 36 B 头部）",
               independent_bytes=tot_ind_bits / 8, anchor_bytes=tot_anc_bits / 8,
               anchor_only=tot_anchor_bits / 8, residual_total=tot_res_bits / 8,
               independent_bits=tot_ind_bits, anchor_bits=tot_anc_bits,
               independent_bpp=tot_ind_bits / n_img / n_pix,
               anchor_bpp=tot_anc_bits / n_img / n_pix,
               independent_z_bytes=tot_ind_z / 8, anchor_z_bytes=tot_anc_z / 8,
               saving_pct=100 * (1 - tot_anc_bits / max(1, tot_ind_bits)))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.out_json}（字段 *_bytes 是真字节，*_bits 是比特）")


# ======================================================================================
# 全连接锚图：任意两张图都可以互为参考
# ======================================================================================
@torch.no_grad()
def _quantize(model, r):
    """把残差/潜层按推理时的硬量化取整，保证"参考端"是解码端真能拿到的东西。"""
    if model.latent_mode == "bernoulli":
        return r
    s = model.quant.step
    return torch.round(r / s) * s


@torch.no_grad()
def pair_bits_matrix(model, xs):
    """xs: [N,3,H,W] -> (anchor_bits[N], pair_bits[N,N])

    anchor_bits[i] = 把 i 当锚图单独编码的比特数
    pair_bits[i][j] = 以 j 的潜层为参考、编码 i 的残差所需比特数（i==j 时等于锚图）
    把 N*N 全部算出来，就是"全连接"：每张图都能看到其他任意一张。
    """
    N = xs.shape[0]
    lats, anchors = [], []
    for i in range(N):
        utils.reset(model)
        y = model.encode_raw(xs[i:i + 1], None)          # 锚图原始潜层
        r = model.code_residual(y, None) if model.residual_mode == "sub" \
            else _quantize(model, y)                     # 锚图被编码的量 = y_ref_hat
        coded, rate = model._code_and_rate(r, None)
        lats.append(coded)                               # 参考端：解码端能精确拿到的那串值
        b, _yb, _zb = container_bits(model, coded, rate)
        anchors.append(b)
    pair = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                pair[i][j] = anchors[i]
                continue
            utils.reset(model)
            r, rate, _rec = model(xs[i:i + 1], ref=lats[j], checkpoints={1})
            b, _yb, _zb = container_bits(model, r, rate)
            pair[i][j] = b
    return np.asarray(anchors, dtype=np.float64), pair


def mst_plan(anchor_bits, pair_bits):
    """在"全连接"图上找最优解码树（Prim，从一个虚拟根出发）。

    虚拟根 -> i 的边权 = 把 i 当锚图的成本
    i <-> j 的边权     = min(编码 i 参考 j, 编码 j 参考 i)
    返回 (解码顺序, 每张图的参考, 总比特)。树保证每个节点被加入时其参考已在树里，
    所以这个顺序一定是可解码的。
    """
    N = len(anchor_bits)
    INF = float("inf")
    in_tree = np.zeros(N, dtype=bool)
    best = anchor_bits.astype(float).copy()
    parent = np.full(N, -1, dtype=int)          # -1 表示当锚图
    order, total = [], 0.0
    for _ in range(N):
        cand = np.where(in_tree, INF, best)
        k = int(np.argmin(cand))
        total += float(best[k])
        in_tree[k] = True
        order.append(k)
        for m in range(N):
            if not in_tree[m]:
                w = min(pair_bits[m][k], pair_bits[k][m])
                if w < best[m]:
                    best[m] = w
                    parent[m] = k
    return order, parent, total


def eval_allpairs(args):
    """对比三种方案的总字节：独立 / 单一锚图(star) / 全连接最优树(MST)。"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(args.weights, map_location="cpu", weights_only=False)
    # 和 evaluate 一样：数据尺寸必须跟 checkpoint 走
    ck_size = int(obj.get("image_size", args.image_size))
    if ck_size != args.image_size:
        print(f"[allpairs] 数据尺寸跟随 checkpoint: image_size={ck_size}")
        args.image_size = ck_size
    _, te = build_data(args)
    model = build_model(args, obj, dev)

    labels = _labels_of(te)
    g = args.group_size
    by_cls = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    groups = [np.array(v) for v in by_cls.values() if len(v) >= g][: args.n_groups]
    print(f"全连接评估：{len(groups)} 组 × {g} 张同类图（image_size={model.image_size}）")
    print(f"{'组':>3} {'独立':>9} {'star':>9} {'MST全连接':>10} {'MST省':>8}   最优结构")
    tot_i = tot_s = tot_m = 0.0
    edges_used = []
    psnr_ind, psnr_anc = [], []
    for gi, idx in enumerate(groups):
        xs = torch.stack([te[int(i)][0] for i in idx[:g]]).to(dev)
        anchors, pair = pair_bits_matrix(model, xs)
        order, parent, mst_bits = mst_plan(anchors, pair)
        # 顺带量一下重建质量（独立编码路径），确认省字节不是靠牺牲质量换的
        with torch.no_grad():
            for i in range(g):
                utils.reset(model)
                _r, _ra, rec = model(xs[i:i + 1], ref=None, checkpoints={model.num_steps})
                psnr_ind.append(10 * math.log10(1.0 / max(
                    1e-12, F.mse_loss(rec[model.num_steps], xs[i:i + 1]).item())))
        ind = float(anchors.sum())
        star = float(anchors[0] + pair[1:, 0].sum())
        tot_i += ind; tot_s += star; tot_m += mst_bits
        # 记录用到的参考关系（看它到底有没有利用"非锚图"参考）
        rel = [f"{i}<-{parent[i]}" if parent[i] >= 0 else f"{i}=锚" for i in order]
        edges_used.append(rel)
        print(f"{gi+1:>3} {ind/8:>8.0f}B {star/8:>8.0f}B {mst_bits/8:>9.0f}B "
              f"{100*(1-mst_bits/max(1,ind)):>7.1f}%   {' '.join(rel)}")
        # 残差 vs 锚图 的对比（诊断参考通路是否有效）
        off = pair[~np.eye(g, dtype=bool)]
        print(f"      诊断: 锚图均值 {anchors.mean()/8:.0f}B, "
              f"任意两图残差均值 {off.mean()/8:.0f}B, 比值 {off.mean()/anchors.mean():.3f}")
    n = max(1, len(groups))
    print("-" * 76)
    print(f"总字节（{len(groups)*g} 张）")
    print(f"  独立编码        : {tot_i/8:9.0f} B   平均 {tot_i/8/(len(groups)*g):7.1f} B/张")
    print(f"  单一锚图 star   : {tot_s/8:9.0f} B   ({100*(1-tot_s/tot_i):+.1f}%)")
    print(f"  全连接 MST      : {tot_m/8:9.0f} B   ({100*(1-tot_m/tot_i):+.1f}%)  "
          f"相对 star 再省 {100*(1-tot_m/max(1e-9,tot_s)):+.1f}%")
    if psnr_ind:
        print(f"  重建质量(独立路径) : 平均 PSNR {sum(psnr_ind)/len(psnr_ind):.2f} dB "
              f"（{len(psnr_ind)} 张）")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(dict(n_groups=len(groups), group_size=g,
                       independent_bytes=tot_i / 8, star_bytes=tot_s / 8,
                       mst_bytes=tot_m / 8, edges=edges_used), f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.out_json}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "allpairs"):
        q = sub.add_parser(name)
        q.add_argument("--image-size", type=int, default=128)
        q.add_argument("--latent-channels", type=int, default=24)
        q.add_argument("--num-steps", type=int, default=6)
        q.add_argument("--z-channels", type=int, default=48)
        q.add_argument("--use-hyperprior", action="store_true", default=True)
        q.add_argument("--no-hyperprior", dest="use_hyperprior", action="store_false")
        q.add_argument("--data-dir", default="./data")
        q.add_argument("--data", choices=["flowers", "div2k", "both"], default="both",
                       help="flowers=Flowers102(train+val+test 合并，有类别标签)；"
                            "div2k=递归 glob 解压后的 DIV2K+Flickr2K（**无类别标签**）；"
                            "both=两者拼接（默认）")
        q.add_argument("--div2k-limit", type=int, default=6000,
                       help="div2k/both 时最多用多少张 DIV2K/Flickr2K 图（0=全部）；"
                            "本机解压后一共 3450 张，所以默认值等于全用")
        q.add_argument("--augment", choices=["none", "flip", "crop"], default="crop",
                       help="训练集增强；留出集始终不增强")
        q.add_argument("--norm", choices=["gn", "bn", "none"], default="bn",
                       help="解码器归一化：bn=BatchNorm(默认，受控 A/B 17.19dB)"
                            " gn=GroupNorm(15.54dB) none=不用(崩)")
        q.add_argument("--holdout-frac", type=float, default=0.15,
                       help="从全量数据里留出的验证比例")
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--eval-limit", type=int, default=300)
        q.add_argument("--group-size", type=int, default=4,
                       help="仅 --use-anchor 通路有效（每组同类图张数）；"
                            "无锚图通路用 --batch-size，本参数不参与")
        q.add_argument("--residual-mode", choices=["learned", "sub"], default="",
                       help="残差语义：sub=硬结构残差 coded=quant_hard(y_i)-y_ref_hat，"
                            "编码器看不到参考（设计说明方案a）；learned=旧行为，"
                            "参考只当编码器输入提示、残差靠网络学。"
                            "留空=用 checkpoint 里记录的（eval/allpairs）；train 时留空按 sub")
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--weights", default="anchor_rd.pth")
        q.add_argument("--out", default="anchor_rd.pth")
        q.add_argument("--out-json", default="anchor_eval.json")
        q.add_argument("--n-groups", type=int, default=12)
    t = sub.choices["train"]
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lam", type=float, default=0.01)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--use-anchor", action="store_true", default=False,
                   help="打开锚图+残差通路。**默认关闭**：锚图/跨图残差机制已被严格"
                        "证伪（同类别图像对潜层相关只有 0.12~0.33，潜在空间里没有可抵消的"
                        "共享分量，连 oracle 逐通道直方图都更费字节），保留仅为复现。"
                        "关闭时 = 独立编码：RandomSampler + 每张图 ref=None")
    t.add_argument("--batch-size", type=int, default=16,
                   help="独立编码（默认通路）的 batch size；--use-anchor 时改由 "
                        "--group-size 决定每 batch 张数，本参数只进日志/checkpoint")
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--max-batches", type=int, default=0)
    t.add_argument("--hot-start", default="")
    t.add_argument("--ref-wiring", choices=["out", "inp"], default="inp",
                   help="out=参考加在编码器输出端(旧)；inp=参考拼到输入端(推荐)")
    t.add_argument("--ref-mode", choices=["star", "pairs"], default="pairs",
                   help="star=只有第0张当锚图；pairs=组内随机两两配对")
    t.add_argument("--save-every", type=int, default=5, help="每多少 epoch 存一次盘")
    t.add_argument("--resume", default="", help="从这里断点续训")
    t.add_argument("--anchor-frac", type=float, default=0.5,
                   help="每组里多少比例当锚图（1/g 会让纯锚图编码成为分布外，默认 0.5）")
    t.add_argument("--noise-floor", type=float, default=0.2,
                   help="噪声退火的下限。退到 0 会让熵模型与硬量化失配，实测码率跳升 75%%")
    t.add_argument("--val-every", type=int, default=10,
                   help="每多少轮做一次 eval 模式真实验证（0=关闭）")
    t.add_argument("--val-batches", type=int, default=4,
                   help="验证用几个 batch（每个 8 张，取自测试集）")
    t.add_argument("--time-budget-min", type=float, default=0.0,
                   help="墙钟时间预算（分钟），到点不再开新 epoch；0=不限")
    a = p.parse_args()
    {"train": train, "eval": evaluate, "allpairs": eval_allpairs}[a.cmd](a)


if __name__ == "__main__":
    main()
