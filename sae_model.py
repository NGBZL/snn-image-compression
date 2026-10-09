# -*- coding: utf-8 -*-
"""
sae_model.py —— 基于 snnTorch 的脉冲自编码器（Spiking Autoencoder, SAE）

A/B 机分离
    A 机（压缩端）：只跑 encoder，把 HxWx3 的图片压成一串 0/1 脉冲（"钥匙"）。
    B 机（还原端）：只跑 decoder，拿这串脉冲把图片还原出来。
    两台机器必须共用同一份 sae_model.py 和同一份 sae_cifar.pth。

网络结构（默认 256x256，可通过 image_size 改）
    编码器 SpikingEncoder（四层 stride=2 下采样，每层 3x3 卷积 + BN + LIF）：
        [B,3,256,256]
          -> Conv(3->32,   s2) + BN + LIF   -> [B,32,128,128]
          -> Conv(32->64,  s2) + BN + LIF   -> [B,64,64,64]
          -> Conv(64->128, s2) + BN + LIF   -> [B,128,32,32]
          -> Conv(128->C,  s2) + BN + LIF   -> [B,C,16,16]   <-- 潜层脉冲（钥匙）
    潜层是一个"空间特征图"而不是一个打平的向量：
    图像信息本来就是空间局部的，保留 16x16 的空间结构，解码器不用从零把空间关系重新发明出来，
    同样的比特数能换到明显更好的重建质量。

    解码器 SpikingDecoder（先 1x1 空间上的 3x3 卷积"解读"潜层，再四次上采样）：
        [B,C,16,16]
          -> Conv(C->128, s1) + BN + LIF    -> [B,128,16,16]
          -> ConvT(128->64, s2) + BN + LIF  -> [B,64,32,32]
          -> ConvT(64->32, s2)  + BN + LIF  -> [B,32,64,64]
          -> ConvT(32->16, s2)  + BN + LIF  -> [B,16,128,128]
          -> ConvT(16->3, s2)   + LIF(阈值=1e3) -> [B,3,256,256]  只取膜电位
          -> sigmoid + 时间维平均            -> [B,3,256,256] 取值 [0,1]

时间维（num_steps）与钥匙大小
    静态图片用"直接编码"：同一张图在每个时间步重复送入编码器，跑 T 步，收集每步的潜层脉冲。
        spk: [B, T, C, h, w]，取值 0/1
    钥匙信息量 = T * C * h * w 比特。默认 T=10, C=32, h=w=16 -> 81920 bit = 10240 字节。
    这就是"压缩率"的全部来源：256*256*3 = 196608 字节 -> 10240 字节 = 94.8%。

⚠ 关于能不能"看出图是什么"
    钥匙比特数决定了重建质量的上限，这是信息论决定的下限，不是调参能绕过的：
        T*C*h*w = 81920 bit   -> 约 10 KB，0.42 bit/像素，大致能看出内容和布局
        T*C*h*w = 8192  bit   -> 约 1 KB，只能表达很粗的颜色块
        太小（几百比特）      -> 只能重建出训练集的"平均图"
    想要更清楚就加大 latent_channels 或 num_steps；想更省就反过来。
    注意：256x256 的 JPEG（质量 85）本身也只有 ~10-15 KB，
    所以"相对原始 RGB 压缩 95%"这个说法很好听，但要证明价值必须和同码率的 JPEG 比。

会用到的 snnTorch 关键点
    1. 每个 LIF 用 init_hidden=True，自己维护膜电位 mem。
    2. 每次前向传播前必须调用 snntorch.utils.reset(model) 清零膜电位。
    3. LIF 的 forward 返回值有两种约定，见下面 _lif_fire / _lif_mem 的说明。
"""

from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import utils

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------
IMG_CH = 3
DEFAULT_IMAGE_SIZE = 256         # 输入/输出边长；必须是 16 的倍数（4 层 stride=2 下采样）
DEFAULT_LATENT_CHANNELS = 32     # 潜层通道数 C
DEFAULT_NUM_STEPS = 10           # SNN 时间步数 T
DOWN = 4                         # 下采样层数（同时也是上采样层数）

# 钥匙文件格式（自定义二进制头，比 torch.save 省掉几百字节的 zip 头）
#   偏移  长度  含义
#   0     4     magic  b"SNNK"
#   4     1     version
#   5     1     flags  bit0=1 表示数据是按位打包的 0/1；bit0=0 表示 float32 明文
#   6     2     num_steps   uint16 小端
#   8     2     channels    uint16 小端
#   10    2     height      uint16 小端
#   12    2     width       uint16 小端
KEY_MAGIC = b"SNNK"
KEY_VERSION = 2                  # v1 是旧的 32x32 打平版本，已不兼容
KEY_FLAG_BITPACK = 0b0000_0001
KEY_HEADER_FMT = "<4sBBHHHH"
KEY_HEADER_SIZE = struct.calcsize(KEY_HEADER_FMT)   # = 14 字节


# --------------------------------------------------------------------------------------
# 小工具：安全地从 LIF 神经元里取出"脉冲"和"膜电位"
#
# ⚠ 这里有个非常容易踩的坑。snnTorch 对 LIF.forward 的返回值有三种约定：
#       output=True                     -> 返回 (spk, mem)
#       init_hidden=True, output=False  -> 只返回 spk      <-- 1.0 版本的默认情况
#       其它                            -> 返回 (spk, mem)
#   所以如果无脑写 `spk = lif(x)[0]`，在 init_hidden=True 时会变成"取 batch 维的第 0 张图"，
#   网络照样能跑、不报错，但结果是错的。下面两个函数把这两种情况统一掉。
# --------------------------------------------------------------------------------------
def _lif_fire(lif, x: torch.Tensor) -> torch.Tensor:
    """调用 LIF，只取脉冲（0/1），兼容返回单值和返回二元组两种写法。"""
    out = lif(x)
    return out[0] if isinstance(out, tuple) else out


def _lif_mem(lif, x: torch.Tensor) -> torch.Tensor:
    """调用 LIF，只取膜电位。需要神经元用 output=True 构造才会返回二元组。"""
    out = lif(x)
    if isinstance(out, tuple):
        return out[1]
    return lif.mem          # 兜底：直接读神经元内部状态


# ======================================================================================
# 1. 编码器
# ======================================================================================
class SpikingEncoder(nn.Module):
    """脉冲编码器：HxWx3 -> [C, h, w] 的 0/1 脉冲特征图（每次 forward 处理一个时间步）。

    h = H/16, w = W/16。256x256 输入时潜层是 16x16。
    """

    def __init__(self, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 beta: float = 0.4, threshold: float = 0.75,
                 latent_bias: float = 1.0):
        super().__init__()
        self.latent_channels = latent_channels
        self.beta = beta
        self.threshold = threshold

        # bias=False：后面紧跟 BatchNorm，卷积自己的偏置会被 BN 减掉，加了纯属浪费
        self.conv1 = nn.Conv2d(IMG_CH, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

        # 最后一层把特征压到 latent_channels，输出就是"钥匙"
        self.conv4 = nn.Conv2d(128, latent_channels, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(latent_channels)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

        # 小技巧：把潜层 BN 的偏置初始化成正数，避免潜层神经元一开始就"沉默"（死脉冲）。
        # 全零的钥匙携带 0 比特信息，解码器就只能输出训练集的平均图。
        #
        # 数值要算一下：LIF 在固定输入下膜电位的稳态是  bias / (1 - beta)。
        # 默认 beta=0.4, threshold=0.75，偏置必须明显大于 0.75*(1-0.4)=0.45，
        # 否则膜电位只会无限逼近阈值却永远跨不过去 ——
        # 表现出来就是"钥匙永远全 0、怎么训都不动"，很容易误判成学习率问题。
        # 这里给 1.0，稳态约 1.67，一开始就有一半时间在放电，之后由发放率正则拉到目标附近。
        nn.init.constant_(self.bn4.bias, latent_bias)
        nn.init.constant_(self.bn4.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] -> spk: [B,C,h,w] 的 0/1 脉冲

        注意：init_hidden=True 时 LIF 直接返回脉冲张量，不要再写 [0]（那会切掉 batch 维）。
        """
        x = _lif_fire(self.lif1, self.bn1(self.conv1(x)))
        x = _lif_fire(self.lif2, self.bn2(self.conv2(x)))
        x = _lif_fire(self.lif3, self.bn3(self.conv3(x)))
        return _lif_fire(self.lif4, self.bn4(self.conv4(x)))


# ======================================================================================
# 2. 解码器
# ======================================================================================
class SpikingDecoder(nn.Module):
    """脉冲解码器：[C,h,w] 脉冲 -> HxWx3 的膜电位。

    最后一层 LIF 的阈值设成极大值（默认 1e3），永远达不到，
    所以它只输出膜电位、不发放脉冲，相当于一个"带泄漏累加器"，
    让梯度能平滑地流回前面的脉冲层。
    """

    def __init__(self, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 beta: float = 0.4, out_threshold: float = 1e3):
        super().__init__()
        self.latent_channels = latent_channels
        self.beta = beta

        # 先在一个空间尺寸不变的卷积里"解读"潜层
        self.stem = nn.Conv2d(latent_channels, 128, 3, stride=1, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(128)
        self.lif_stem = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)

        # 四次上采样，每次边长 x2（k=4, s=2, p=1 时 out = 2*in）
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)

        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)

        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.lif3 = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)

        # 最后一层保留 bias，方便网络学出图片的整体颜色均值
        self.deconv4 = nn.ConvTranspose2d(16, IMG_CH, 4, stride=2, padding=1, bias=True)
        # output=True 很关键：只有它才会让 forward 返回 (spk, mem)，我们才能拿到膜电位。
        # 如果只写 init_hidden=True，snnTorch 1.0 只返回脉冲，就拿不到膜电位了。
        self.lif_out = snn.Leaky(beta=beta, threshold=out_threshold,
                                 reset_mechanism="none", init_hidden=True, output=True)

    def forward(self, spk: torch.Tensor) -> torch.Tensor:
        """spk: [B,C,h,w] 脉冲 -> mem: [B,3,H,W] 膜电位（未归一化）"""
        x = _lif_fire(self.lif_stem, self.bn_stem(self.stem(spk)))
        x = _lif_fire(self.lif1, self.bn1(self.deconv1(x)))
        x = _lif_fire(self.lif2, self.bn2(self.deconv2(x)))
        x = _lif_fire(self.lif3, self.bn3(self.deconv3(x)))
        # 最后一层只要膜电位，不要脉冲（阈值 1e3 也永远不会发放）
        return _lif_mem(self.lif_out, self.deconv4(x))


# ======================================================================================
# 3. 完整模型
# ======================================================================================
class SAE(nn.Module):
    """Spiking AutoEncoder。

    参数
        latent_channels : 潜层通道数 C，决定钥匙的"宽度"
        num_steps       : SNN 时间步数 T，决定钥匙的"长度"
        image_size      : 输入/输出边长，必须是 16 的倍数
        beta / threshold: LIF 膜电位衰减系数 / 发放阈值
    钥匙比特数 = num_steps * latent_channels * (image_size/16)^2
    """

    def __init__(self, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 num_steps: int = DEFAULT_NUM_STEPS,
                 image_size: int = DEFAULT_IMAGE_SIZE,
                 beta: float = 0.4, threshold: float = 0.75):
        super().__init__()
        if image_size % (2 ** DOWN) != 0:
            raise ValueError(f"image_size 必须是 {2 ** DOWN} 的倍数，收到 {image_size}")
        self.latent_channels = int(latent_channels)
        self.num_steps = int(num_steps)
        self.image_size = int(image_size)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.latent_size = self.image_size // (2 ** DOWN)      # 256 -> 16
        self.latent_hw = (self.latent_size, self.latent_size)

        self.encoder = SpikingEncoder(self.latent_channels, beta=beta, threshold=threshold)
        self.decoder = SpikingDecoder(self.latent_channels, beta=beta)

    # ---------------------------------------------------------------- 钥匙大小
    @property
    def key_bits(self) -> int:
        """钥匙的比特数 = T * C * h * w"""
        return self.num_steps * self.latent_channels * self.latent_size * self.latent_size

    @property
    def key_bytes(self) -> int:
        """钥匙的字节数（含文件头，按位打包）"""
        return KEY_HEADER_SIZE + (self.key_bits + 7) // 8

    # ---------------------------------------------------------------- 编码（A 机）
    def encode(self, x: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """跑完整的时间维，返回脉冲序列。

        x    : [B,3,H,W]，已用 ToTensor 归一化到 [0,1]
        返回 : [B,T,C,h,w]，取值 {0,1}

        注意：这里内部自己循环 T 步，外部不要重复调用再 stack。
        """
        T = int(num_steps or self.num_steps)
        # 关键：清零膜电位，否则上一张图的状态会残留下来
        utils.reset(self.encoder)
        # 直接编码：同一个静态输入在每个时间步都送一遍
        spk_rec = [self.encoder(x) for _ in range(T)]
        return torch.stack(spk_rec, dim=1)          # [T,B,C,h,w] -> [B,T,C,h,w]

    def encode_step(self, x: torch.Tensor) -> torch.Tensor:
        """只跑一个时间步，返回 [B,C,h,w]。想手动写时间循环时用这个。"""
        return self.encoder(x)

    # ---------------------------------------------------------------- 解码（B 机）
    def decode(self, spk: torch.Tensor) -> torch.Tensor:
        """根据脉冲序列还原图片。

        spk  : [B,T,C,h,w]，取值 {0,1}；也接受 [T,C,h,w]
        返回 : [B,3,H,W]，取值 [0,1]

        做法：每一步解码出一张图，再把 T 步的结果做时间平均。
        实测时间平均比"只取最后一步"好很多（MSE 0.0165 vs 0.0278，差 68%）：
        脉冲是随机的，平均能压掉噪声，而且给梯度提供了 T 条通路。
        """
        if spk.dim() == 4:                     # 允许传 [T,C,h,w]，自动补 batch 维
            spk = spk.unsqueeze(0)
        spk = spk.float()

        T = spk.shape[1]
        # 关键：清零解码器膜电位
        utils.reset(self.decoder)

        acc = 0.0
        for t in range(T):
            mem = self.decoder(spk[:, t])      # [B,3,H,W] 膜电位
            acc = acc + torch.sigmoid(mem)     # sigmoid 压到 [0,1]，当作像素强度
        return acc / T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """训练时用：编码 + 解码一条龙。"""
        return self.decode(self.encode(x))


# ======================================================================================
# 4. 评价指标：PSNR / SSIM（自包含实现，不需要额外装 pytorch_msssim / skimage）
# ======================================================================================
def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """一维高斯核，后面外积成二维。"""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


class SSIM(nn.Module):
    """可微的 SSIM（结构相似度），既能当损失函数也能当评价指标。

    实现的是标准 SSIM：
        SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                    / ((mu_x^2+mu_y^2+C1)(sigma_x^2+sigma_y^2+C2))
    用 11x11 高斯窗做局部统计，最后对所有位置和通道取平均。
    图像取值是 [0,1]，所以 data_range=1，C1=(0.01)^2, C2=(0.03)^2。
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2
        g = _gaussian_kernel(window_size, sigma)
        self.register_buffer("_w2d", (g[:, None] * g[None, :]).contiguous())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: [B,C,H,W]，取值 [0,1] -> 标量 SSIM（越大越好，1 表示完全相同）"""
        c = x.shape[1]
        w = self._w2d.expand(c, 1, self.window_size, self.window_size).contiguous()
        w = w.to(dtype=x.dtype, device=x.device)

        mu_x = F.conv2d(x, w, groups=c)
        mu_y = F.conv2d(y, w, groups=c)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        # 用 E[x^2]-E[x]^2 算方差；理论上非负，浮点上可能出现极小负数，clamp 一下更稳
        sig_x2 = (F.conv2d(x * x, w, groups=c) - mu_x2).clamp_min(0)
        sig_y2 = (F.conv2d(y * y, w, groups=c) - mu_y2).clamp_min(0)
        sig_xy = F.conv2d(x * y, w, groups=c) - mu_xy

        ssim_map = ((2 * mu_xy + self.c1) * (2 * sig_xy + self.c2)) / \
                   ((mu_x2 + mu_y2 + self.c1) * (sig_x2 + sig_y2 + self.c2))
        return ssim_map.mean()


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """逐张算 PSNR，返回 [B]。x, y: [B,C,H,W]，取值 [0,1]。"""
    mse = (x - y).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10((data_range ** 2) / mse.clamp_min(1e-12))


def ms_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """MS-PSNR = 逐张 PSNR 再平均（论文里用的就是这个，和"整体 MSE 再换算"不是一回事）。"""
    return psnr_per_image(x, y, data_range).mean()


def jpeg_size_and_quality(img_chw: torch.Tensor, target_bytes: int,
                          quality_lo: int = 5, quality_hi: int = 95):
    """把一个 [3,H,W] 的 [0,1] 图像编成 JPEG，二分搜索出接近 target_bytes 的质量因子。

    返回 (JPEG 字节数, 质量因子, 解码回来的 [3,H,W] 张量, matched)

    matched=False 表示"连最低质量都比目标大"——也就是这个码率 JPEG 根本到不了。
    这种情况下返回的是它能做到的**最小**体积（而不是最大），
    好让调用方如实说明"两边码率对不上，这个对比不成立"，而不是拿一个虚高的数字去比。
    """
    import io
    from PIL import Image

    arr = (img_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")

    def encode(q):
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q, subsampling=0)
        return buf.getvalue()

    def decode(data):
        back = Image.open(io.BytesIO(data)).convert("RGB")
        return torch.from_numpy(np.asarray(back).astype(np.float32) / 255.0).permute(2, 0, 1)

    # 先试最低质量：如果它都超标，说明目标码率 JPEG 达不到
    lo_data = encode(quality_lo)
    if len(lo_data) > target_bytes:
        return len(lo_data), quality_lo, decode(lo_data), False

    lo, hi = quality_lo, quality_hi
    best_data, best_q = lo_data, quality_lo
    # 二分：JPEG 体积关于 quality 单调递增，找"不超过目标体积的最大质量"
    while lo < hi:
        mid = (lo + hi + 1) // 2
        data = encode(mid)
        if len(data) <= target_bytes:
            lo, best_data, best_q = mid, data, mid
        else:
            hi = mid - 1
    return len(best_data), best_q, decode(best_data), True


# ======================================================================================
# 5. 权重加载工具（A/B 机各取所需）
# ======================================================================================
def torch_load_compat(path, map_location="cpu"):
    """兼容不同 PyTorch 版本的 torch.load（2.6 之后 weights_only 默认为 True）。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)      # 老版本没有 weights_only
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def split_checkpoint(obj):
    """把 sae_cifar.pth 拆成 (state_dict, config)。

    既支持 train.py 存的 {"state_dict":..., "latent_channels":...} 字典，
    也支持直接 torch.save(model.state_dict()) 存出来的纯权重。
    """
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return obj, {}                                          # 纯 state_dict
    raise ValueError("无法识别的权重文件格式，请用 train.py 生成的 pth")


def load_partial(model: nn.Module, state_dict: dict, prefixes) -> tuple[list, list]:
    """只加载名字以 prefixes 开头的权重。

    A 机用 prefixes=("encoder.",)              -> 只加载编码器
    B 机用 prefixes=("decoder.",)              -> 只加载解码器
    返回 (已加载的 key 列表, 缺失的 key 列表)
    """
    prefixes = tuple(prefixes)
    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k.startswith(prefixes) and k in own and own[k].shape == v.shape}
    missing = [k for k in own if k.startswith(prefixes) and k not in matched]
    model.load_state_dict(matched, strict=False)                # strict=False：只关心挑出来的部分
    return sorted(matched), missing


def read_checkpoint_config(path, latent_channels=None, num_steps=None,
                           image_size=None, beta=None):
    """读权重文件，返回 (state_dict, config)。命令行显式给了参数就优先用命令行的。"""
    obj = torch_load_compat(path, map_location="cpu")
    state_dict, cfg = split_checkpoint(obj)
    return state_dict, {
        "latent_channels": int(latent_channels if latent_channels is not None
                               else cfg.get("latent_channels", DEFAULT_LATENT_CHANNELS)),
        "num_steps": int(num_steps if num_steps is not None
                         else cfg.get("num_steps", DEFAULT_NUM_STEPS)),
        "image_size": int(image_size if image_size is not None
                          else cfg.get("image_size", DEFAULT_IMAGE_SIZE)),
        "beta": float(beta if beta is not None else cfg.get("beta", 0.4)),
    }


def build_model_from_checkpoint(path, device="cpu", latent_channels=None, num_steps=None,
                                image_size=None, beta=None, strict=True, prefixes=None):
    """读权重文件 -> 还原配置 -> 构造模型并加载权重。

    prefixes=None        加载全部权重（单机测试用）
    prefixes=("encoder.",)              A 机压缩用：只加载编码器，解码器保持随机初始化
    prefixes=("decoder.",)              B 机还原用：只加载解码器

    返回 (model, config_dict)
    """
    state_dict, cfg = read_checkpoint_config(path, latent_channels, num_steps, image_size, beta)

    model = SAE(latent_channels=cfg["latent_channels"], num_steps=cfg["num_steps"],
                image_size=cfg["image_size"], beta=cfg["beta"])

    if prefixes is None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if strict and missing:
            raise RuntimeError(
                f"权重文件 {path} 与当前模型结构不匹配，缺少 {len(missing)} 个参数：{missing[:5]} ...\n"
                f"请确认 A 机和 B 机用的是同一份 sae_model.py 和同一个 pth 文件。")
    else:
        prefixes = tuple(prefixes)
        expected = [k for k in model.state_dict() if k.startswith(prefixes)]
        loaded, missing = load_partial(model, state_dict, prefixes=prefixes)
        if not loaded:
            raise RuntimeError(
                f"权重文件 {path} 里找不到任何以 {prefixes} 开头的参数。\n"
                f"请确认这个 pth 是 train.py 训练出来的完整权重。")
        # 只匹配上一部分是很危险的：模型照样能跑，但输出全是垃圾，而且不报错。
        # 所以要求至少匹配上 90%，否则直接报错，避免"静默产生乱码图"。
        if len(loaded) < 0.9 * len(expected):
            raise RuntimeError(
                f"权重文件 {path} 与当前模型结构不匹配：{prefixes} 下有 {len(expected)} 个参数，"
                f"只匹配上 {len(loaded)} 个。\n"
                f"最常见的原因：A 机和 B 机用了不同版本的 sae_model.py，"
                f"或者改了 image_size / latent_channels / 网络层数之后忘了重新训练。\n"
                f"请确保两台机器用的是同一份 sae_model.py 和同一个 pth 文件。")
        unexpected = []

    model.to(device)
    model.eval()
    cfg = dict(cfg)
    cfg["missing"] = missing
    cfg["unexpected"] = unexpected
    return model, cfg


# ======================================================================================
# 6. "钥匙"文件的打包 / 解包
# ======================================================================================
def pack_spikes(spk: torch.Tensor, bitpack: bool = True) -> bytes:
    """把 [1,T,C,h,w] 的 0/1 脉冲压成字节流。

    bitpack=True  : 每 8 个脉冲塞进 1 个字节（numpy.packbits）
    bitpack=False : 直接按 float32 存（体积大 32 倍，用来对比压缩收益）
    """
    if spk.dim() != 5 or spk.shape[0] != 1:
        raise ValueError(f"只支持单张图的 [1,T,C,h,w] 脉冲，收到 {tuple(spk.shape)}")
    arr = spk.detach().to(torch.uint8).reshape(-1).cpu().numpy()
    if not bitpack:
        return arr.astype("<f4").tobytes()
    return np.packbits(arr, bitorder="big").tobytes()


def save_key(path, spk: torch.Tensor, bitpack: bool = True) -> int:
    """把脉冲序列写成钥匙文件，返回文件字节数。

    文件 = 14 字节头 + 数据。头部自带 T/C/h/w，
    所以 B 机只拿到钥匙也能知道该按什么形状解包。
    """
    T, C, H, W = (int(v) for v in spk.shape[1:])
    payload = pack_spikes(spk, bitpack=bitpack)
    flags = KEY_FLAG_BITPACK if bitpack else 0
    header = struct.pack(KEY_HEADER_FMT, KEY_MAGIC, KEY_VERSION, flags, T, C, H, W)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return len(header) + len(payload)


def load_key(path, device="cpu") -> torch.Tensor:
    """读钥匙文件，返回 [1,T,C,h,w] 的浮点脉冲张量。

    兼容三种格式：本模块 save_key() 写的紧凑二进制 / torch.save(spk) / torch.save({"spk":...})
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] == KEY_MAGIC:
        magic, version, flags, T, C, H, W = struct.unpack(
            KEY_HEADER_FMT, raw[:KEY_HEADER_SIZE])
        if version != KEY_VERSION:
            raise ValueError(
                f"钥匙文件版本 {version} 不受支持（当前 {KEY_VERSION}）。\n"
                f"v1 是旧的 32x32 打平版本，架构已经变了，请用新的 compress.py 重新生成钥匙。")
        body = raw[KEY_HEADER_SIZE:]
        n_bits = T * C * H * W
        if flags & KEY_FLAG_BITPACK:
            n_bytes = (n_bits + 7) // 8
            packed = np.frombuffer(body[:n_bytes], dtype=np.uint8)
            bits = np.unpackbits(packed, bitorder="big")[:n_bits]
            spk = torch.from_numpy(bits.astype(np.float32))
        else:
            arr = np.frombuffer(body[: n_bits * 4], dtype="<f4")
            if arr.size != n_bits:
                raise ValueError("钥匙文件数据区长度不足，文件可能已损坏")
            spk = torch.from_numpy(arr.astype(np.float32))
        return spk.view(1, T, C, H, W).to(device)

    # ---- 兼容旧的 torch.save 格式 ----
    obj = torch_load_compat(path, map_location="cpu")
    if isinstance(obj, dict) and "spk" in obj:
        obj = obj["spk"]
    if not torch.is_tensor(obj):
        raise ValueError(f"无法解析的钥匙文件：{path}")
    spk = obj.float()
    if spk.dim() == 2:                       # [T, L]
        spk = spk.view(1, spk.shape[0], spk.shape[1], 1, 1)
    elif spk.dim() == 4:                     # [T,C,h,w]
        spk = spk.unsqueeze(0)
    if spk.dim() != 5:
        raise ValueError(f"钥匙张量形状异常：{tuple(spk.shape)}")
    return spk.to(device)


def pretty_bits(spk: torch.Tensor, max_bits: int = 160) -> str:
    """把脉冲序列打印成 0101 的样子，方便肉眼确认钥匙里确实有内容。"""
    flat = spk.detach().to(torch.uint8).reshape(-1).cpu().tolist()
    s = "".join(str(b) for b in flat[:max_bits])
    if len(flat) > max_bits:
        s += f" ...(共 {len(flat)} bit)"
    return s


def channel_rate_entropy(spk: torch.Tensor) -> float:
    """每个潜层通道的发放率有多分散（二元熵之和），上限是 C。

    怎么读这个数：
      - 接近 C  -> 每个通道都在 0/1 之间摇摆，码字是"活"的
      - 接近 0  -> 所有通道恒定放电或恒定沉默，钥匙退化了，
                   这时候解码器只能输出训练集的平均图
    它并不是钥匙的真实信息量（真实容量是 T*C*h*w 比特），只是用来快速判断编码器有没有"躺平"。
    """
    s = spk.detach().float()
    p = s.permute(0, 1, 3, 4, 2).reshape(-1, s.shape[2]).mean(dim=0).clamp(0, 1)
    p = p[(p > 0) & (p < 1)]                 # 熵为 0 的通道直接丢掉
    if p.numel() == 0:
        return 0.0
    h = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
    return float(h.sum())


def human_bytes(n: int) -> str:
    """把字节数变成好看的字符串。"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n / 1024 / 1024:.2f} MB"


def bpp(key_bytes: int, image_size: int) -> float:
    """bits per pixel：钥匙字节数换算成"每像素多少比特"，跨分辨率比较时用这个。"""
    return key_bytes * 8.0 / (image_size * image_size * IMG_CH)


__all__ = [
    "SAE", "SpikingEncoder", "SpikingDecoder", "SSIM",
    "save_key", "load_key", "pack_spikes", "torch_load_compat", "split_checkpoint",
    "load_partial", "build_model_from_checkpoint", "read_checkpoint_config",
    "psnr_per_image", "ms_psnr", "jpeg_size_and_quality",
    "pretty_bits", "channel_rate_entropy", "human_bytes", "bpp",
    "IMG_CH", "DEFAULT_IMAGE_SIZE", "DEFAULT_LATENT_CHANNELS", "DEFAULT_NUM_STEPS",
    "KEY_HEADER_SIZE",
]
