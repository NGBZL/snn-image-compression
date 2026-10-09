# -*- coding: utf-8 -*-
"""sae_rd.py — 率失真版脉冲自编码器（改进 1/2/3/4）

改进1 率项 + 真实熵模型
      UniformQuantizer: 加性均匀噪声模拟量化 + 噪声退火 + 可选可学步长
      FactorizedGaussianPrior: 每通道独立 (mu, sigma)，按量化区间概率质量算码率
      BernoulliPrior: 二值模式用，率项会把发放率推离 0.5
改进2 编码器最后一层 BN -> GDN（保留分布结构，不做 whitening）
改进3 Scale Hyperprior（z -> 逐像素 sigma，空间自适应比特分配）
改进4 时间维渐进式（解码器逐步累加，可截断）

两种潜层模式
  gaussian : 连续潜层 -> 量化 -> 高斯熵模型（默认，码率可控性最好）
  bernoulli: 二值脉冲 -> 伯努利熵模型（保留"码就是脉冲"的叙事）

关键设计决定：默认 step 固定为 1.0，码率完全由 sigma 控制。
  原因：L 只依赖 step/sigma 的比值，两者存在近似尺度退化，有效自由度只有 1 个。
  与其让两个旋钮互掐，不如按 Balle 2018 的标准形式把 step 固定掉。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import utils

IMG_CH = 3
_SQRT2 = math.sqrt(2.0)


def _lif_fire(lif, x):
    """取脉冲。snnTorch 1.0 里 init_hidden=True 时 LIF 只返回 spk，不要再写 [0]。"""
    out = lif(x)
    return out[0] if isinstance(out, tuple) else out


def _lif_mem(lif, x):
    """取膜电位（需要神经元用 output=True 构造）。"""
    out = lif(x)
    return out[1] if isinstance(out, tuple) else lif.mem


def _phi(x):
    """标准正态 CDF。**注意：float32 下 |x| > ~3.9 就饱和成 1.0**，
    两个 CDF 相减在两侧尾部都会灾难性抵消，因此熵模型不再直接用它。"""
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _log_bin_prob(t, h):
    """log(Phi(t+h) - Phi(t-h))，h = 0.5/sigma > 0；两尾都不饱和、不抵消。

    为什么要换掉 `_phi(b) - _phi(a)`：float32 的 erf 在 |x| > ~3.9 处**恰好**变成 1.0，
    于是
      · 概率被 `clamp_min(1e-9)` 兜住 -> 任何偏离 ~5.5 sigma 的符号恒定 29.90 bit；
      · 更糟的是 d(rate)/d(sigma) 在那一整片区域**恰好为 0**（两个 CDF 都是常数 1），
        sigma 拿不到任何梯度。实测 v7 训 1300 轮 sigma 只从 0.434 漂到 0.374，
        而潜层 y_std 从 0.68 涨到 50.4 —— 熵模型根本没在管事。
    这里改在 log 域做差，恒等式：
        log(Phi(b) - Phi(a)) = L(b) + log(1 - exp(L(a) - L(b))),  L = log_ndtr, a < b
    两个尾部处理：
      · 区间整段落在正半轴时镜像到负半轴（Phi(b)-Phi(a) = Phi(-a)-Phi(-b)）。
        正半轴的 log_ndtr 会趋近 -0，两个 -0 相减把差值整个丢掉；负半轴则保留
        完整相对精度，尾部渐近 log Phi(x) ~ -x^2/2 - log|x| 是精确的。
      · 两个边界在 float32 下无法区分（h 相对 |t| 太小，例如 sigma 到 1e6）时，
        退化为"密度近似" log(2h * phi(t))。
    `1 - exp(d)` 用 `-expm1(d)` 算，d -> 0 时不掉精度。返回 nats，可微。
    """
    a = t - h
    b = t + h
    upper = a >= 0.0                      # 整段在正半轴 -> 镜像
    a2 = torch.where(upper, -b, a)
    b2 = torch.where(upper, -a, b)
    La = torch.special.log_ndtr(a2)
    Lb = torch.special.log_ndtr(b2)
    # a2 < b2 保证 d < 0；clamp 只是防止 float 舍入让 d 变成 0 后 log(0) = -inf
    d = (La - Lb).clamp_max(-1e-30)
    exact = Lb + torch.log(-torch.expm1(d))
    approx = math.log(2.0) + torch.log(h) - 0.5 * t * t - 0.5 * math.log(2.0 * math.pi)
    return torch.where(a2 != b2, exact, approx)


# ======================================================================================
# 改进2：GDN / IGDN
# ======================================================================================
class GDN(nn.Module):
    """广义分裂归一化（Balle 2016）。

        forward: y_i = x_i / (beta_i + sum_j gamma_ij |x_j|^alpha)^nu
        inverse: y_i = x_i * (beta_i + sum_j gamma_ij |x_j|^alpha)^nu

    替换 BN 的理由：BN 把每通道均值/方差强行归一化（whitening），
    恰好抹掉熵模型最需要的统计结构 —— 实测有 BN 时潜层接近均匀分布、
    空间上下文只能压掉 3%，熵编码收益仅 4%。
    GDN 是除法归一化，保留并塑造分布形状。

    inverse=True 是解码器用的**独立学习层**（IGDN），非数学逆（Balle 标准做法）。
    需要验证可逆性时用 exact_inverse_diag()。
    """

    def __init__(self, channels: int, inverse: bool = False, alpha: float = 2.0,
                 nu: float = 0.5, beta_init: float = 0.1, gamma_init: float = 0.1,
                 beta_min: float = 1e-6, gamma_min: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.inverse = inverse
        self.alpha, self.nu = alpha, nu
        self.beta_min, self.gamma_min = beta_min, gamma_min
        self.beta = nn.Parameter(torch.full((channels,), float(beta_init)))
        self.gamma = nn.Parameter(torch.eye(channels) * float(gamma_init))

    def _denom(self, x):
        a = x.abs().pow(self.alpha)
        g = self.gamma.clamp(min=self.gamma_min).view(self.channels, self.channels, 1, 1)
        b = self.beta.clamp(min=self.beta_min).view(1, self.channels, 1, 1)
        return (b + F.conv2d(a, g)).clamp_min(1e-8)

    def forward(self, x):
        d = self._denom(x)
        return x * d.pow(self.nu) if self.inverse else x * d.pow(-self.nu)

    @torch.no_grad()
    def exact_inverse_diag(self, y, eps: float = 1e-6):
        """gamma 为对角阵时的解析逆：y = x/sqrt(beta+gamma x^2) => x = y*sqrt(beta/(1-gamma y^2))"""
        gd = torch.diagonal(self.gamma).clamp(min=self.gamma_min).view(1, -1, 1, 1)
        b = self.beta.clamp(min=self.beta_min).view(1, -1, 1, 1)
        return y * (b / (1.0 - gd * y.pow(2)).clamp_min(eps)).sqrt()


# ======================================================================================
# 改进1：量化器
# ======================================================================================
class UniformQuantizer(nn.Module):
    """均匀标量量化 + 训练期加性均匀噪声 + 噪声退火 +（可选）可学步长。

    训练：早期 noise_scale=1.0 -> y + U(-s/2, s/2)；后期 ->0 逼近硬量化。
    推理：round(y/s)*s

    STE 写法的坑：`y + (round(y/s)*s - y).detach()` 会让 s 的梯度被一起 detach 掉
    （实测 grad = 0，可学步长永远学不动）。必须写成 k*s + (y - y.detach())：
        前向 = k*s 正好是量化值；对 y 梯度 = 1（STE）；对 s 梯度 = k（真实且精确）。
    """

    def __init__(self, step_init: float = 1.0, learnable: bool = False,
                 step_min: float = 1e-3, step_max: float = 1e3):
        super().__init__()
        self.log_step = nn.Parameter(torch.tensor(math.log(step_init)),
                                     requires_grad=learnable)
        self.step_min, self.step_max = step_min, step_max
        self.noise_scale = 1.0
        self.hard = False

    @property
    def step(self):
        return self.log_step.exp().clamp(self.step_min, self.step_max)

    def forward(self, y):
        s = self.step
        if self.training and not self.hard:
            if self.noise_scale > 1e-3:
                return y + (torch.rand_like(y) - 0.5) * (s * self.noise_scale)
            k = torch.round(y / s).detach()
            return k * s + (y - y.detach())
        return torch.round(y / s) * s


# ======================================================================================
# 改进1：熵模型
# ======================================================================================
class FactorizedGaussianPrior(nn.Module):
    """每通道独立可学高斯，按**量化区间概率质量**算真实离散码率：
        p(y_hat) = Phi((yn-mu+0.5)/sigma) - Phi((yn-mu-0.5)/sigma),  yn = y_hat/step
    """

    def __init__(self, channels: int, sigma_init: float = 1.0):
        super().__init__()
        self.channels = channels
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_sigma = nn.Parameter(torch.full((1, channels, 1, 1), math.log(sigma_init)))

    def sigma(self, extra=None):
        # 上限从 1e3 放到 1e6：撞到上限后"再大的符号也不会更贵"，
        # 率项的梯度压力会被完全废掉（实测潜层尺度因此飙到 std 71）。
        # 放宽后大符号真的贵，模型才会主动把潜层压小。
        s = self.log_sigma.exp().clamp(1e-4, 1e6)
        return s if extra is None else (s * extra).clamp(1e-4, 1e6)

    def bits(self, y_hat, step, extra_sigma=None):
        """每个符号的码率（比特）。用 `_log_bin_prob` 在 log 域算，两尾不饱和。

        与旧写法的区别只在尾部：旧的 Phi 相减在 |t| > ~5.5 处变成常数 1e-9
        （29.90 bit、零梯度），新的给出真实码率并且 d/d(sigma) 非零、有限。
        """
        yn = y_hat / step
        sig = self.sigma(extra_sigma)
        t = (yn - self.mu) / sig
        h = 0.5 / sig
        return -_log_bin_prob(t, h) / math.log(2.0)


class BernoulliPrior(nn.Module):
    """每通道独立可学伯努利概率。R = -log2(p 或 1-p)。

    H2(p) 在 p=0.5 处最大 => 最小化 R 会把发放率推离 0.5。
    注意 p=0.5 时 dR/dspk = log2((1-p)/p) = 0，率项对编码器**零梯度**，
    所以存在"先让先验学到实际发放率、再推动编码器"的两阶段冷启动。
    """

    def __init__(self, channels: int, p_init: float = 0.5):
        super().__init__()
        p = min(max(p_init, 1e-4), 1 - 1e-4)
        self.logit_p = nn.Parameter(torch.full((1, channels, 1, 1), math.log(p / (1 - p))))

    def prob(self):
        return torch.sigmoid(self.logit_p).clamp(1e-6, 1 - 1e-6)

    def bits(self, spk):
        p = self.prob()
        return -(spk * torch.log2(p) + (1 - spk) * torch.log2(1 - p))


# ======================================================================================
# 改进3：Scale Hyperprior
# ======================================================================================
class ScaleHyperprior(nn.Module):
    """z = h_a(y) 量化后传输；解码端 sigma = exp(h_s(z_hat)) 逐像素预测。"""

    def __init__(self, channels: int, z_channels: int = 48):
        super().__init__()
        self.h_a = nn.Sequential(
            nn.Conv2d(channels, z_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(z_channels, z_channels, 3, stride=2, padding=1))
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(z_channels, z_channels, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(z_channels, channels, 3, stride=2, padding=1, output_padding=1))
        self.z_quant = UniformQuantizer(step_init=1.0)
        self.z_prior = FactorizedGaussianPrior(z_channels, sigma_init=0.5)

    def analyze_z(self, y):
        """y [B,T,C,h,w] -> 原始（未量化）z，和 forward 里走的是同一条 h_a 路径。

        单独抽出来是为了让 sigma 预热 / 重标定能拿到**和训练完全一致**的 z，
        而不是在别处复制一遍 reshape 逻辑（复制过的代码迟早会漂移）。
        """
        shape = y.shape
        return self.h_a(y.reshape(-1, *shape[2:]))

    def sigma_from_z(self, z_hat):
        """z_hat -> 逐像素 sigma。**编解码两端唯一的 clamp/exp 实现**。

        以前编码端用 clamp(±14) -> exp -> clamp(1e-4,1e6)，解码端另写了一份
        clamp(±8) -> exp -> clamp(1e-3,1e3)：只要 h_s 的输出绝对值超过 8
        （v5 的 sigma p99 是 27 万，非常常见），两端就会落到**不同的 sigma 桶**，
        概率表不一致 -> 解出来直接是垃圾。现在两边都只调这一个函数。
        """
        return self.h_s(z_hat).clamp(-LOG_SIGMA_CLAMP, LOG_SIGMA_CLAMP).exp() \
            .clamp(SIGMA_MIN, SIGMA_MAX)

    def forward(self, y, training: bool = True):
        shape = y.shape
        z = self.analyze_z(y)
        z_hat = self.z_quant(z)
        zb = self.z_prior.bits(z_hat, self.z_quant.step).sum()
        # 关键：**先 clamp 再 exp**，防止 exp 溢出成 inf（inf 进 STE 会得到 NaN）。
        sigma = self.sigma_from_z(z_hat).reshape(shape[0], shape[1], *shape[2:])
        return sigma, zb, z_hat, self.z_quant.step


# ======================================================================================
# 主干
# ======================================================================================
class SAE_RD(nn.Module):
    """率失真版脉冲自编码器。编码器前三层是脉冲卷积（边缘端事件驱动稀疏性保留），
    最后一层无 BN、走 GDN；潜层经量化 + 熵模型；解码器支持渐进式截断。"""

    def __init__(self, latent_channels: int = 32, num_steps: int = 10,
                 image_size: int = 256, latent_mode: str = "gaussian",
                 use_hyperprior: bool = True, z_channels: int = 48,
                 beta: float = 0.4, threshold: float = 0.75,
                 learn_step: bool = False, norm_type: str = "gn"):
        super().__init__()
        assert image_size % 16 == 0
        assert latent_mode in ("gaussian", "bernoulli")
        assert norm_type in ("gn", "bn", "none")
        self.norm_type = norm_type
        self.latent_channels = int(latent_channels)
        self.num_steps = int(num_steps)
        self.image_size = int(image_size)
        self.latent_mode = latent_mode
        self.use_hyperprior = bool(use_hyperprior and latent_mode == "gaussian")
        self.beta, self.threshold = float(beta), float(threshold)
        self.latent_size = image_size // 16
        self.learn_step = learn_step

        # ---- 编码器 ----
        self.conv1 = nn.Conv2d(IMG_CH, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)
        # 最后一层：无 BN，改 GDN（改进2）
        self.conv4 = nn.Conv2d(128, self.latent_channels, 3, 2, 1, bias=True)
        self.gdn = GDN(self.latent_channels)
        if latent_mode == "bernoulli":
            # 稳态 = bias/(1-beta)；取 threshold*(1-beta) 让初始发放率约 50%，
            # 之后交给率项把它推下去。
            self.latent_bias = nn.Parameter(
                torch.full((1, self.latent_channels, 1, 1), threshold * (1.0 - beta)))
            self.lif4 = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)

        # ---- 量化 + 熵模型 ----
        self.quant = UniformQuantizer(step_init=1.0, learnable=learn_step)
        self.prior = (BernoulliPrior(self.latent_channels, 0.5) if latent_mode == "bernoulli"
                      else FactorizedGaussianPrior(self.latent_channels, 1.0))
        self.hyperprior = ScaleHyperprior(self.latent_channels, z_channels) \
            if self.use_hyperprior else None

        # ---- 解码器 ----
        if latent_mode == "gaussian":
            self.igdgn = GDN(self.latent_channels, inverse=True)
        # 解码器归一化：gn / bn / none 三选一。
        # 之前一律用 GroupNorm，理由是"潜层呈双峰（锚图满量级 / 残差近 0），
        # BatchNorm 的 running 统计量会把两个分布平均掉"。这个推理只对了一半：
        # 真正的坑是 **GroupNorm 逐样本去掉均值/方差 = 把潜层的"量级"信息抹掉了**，
        # 而压缩解码器恰恰靠量级判断内容强弱；SNN 脉冲层（阈值 1.0）又进一步放大损失。
        # Ballé 2018 的合成网络干脆不用归一化，也是这个原因。
        # 实测：BatchNorm 的 v5 = 21.44 dB，GroupNorm 的 v7 = 14.07 dB。
        def _norm(c):
            if norm_type == "bn":
                return nn.BatchNorm2d(c)
            if norm_type == "none":
                return nn.Identity()
            return nn.GroupNorm(8, c)

        self.stem = nn.Conv2d(self.latent_channels, 128, 3, 1, 1, bias=False)
        self.bn_stem = _norm(128)
        self.lif_stem = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn_d1 = _norm(64)
        self.lif_d1 = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.bn_d2 = _norm(32)
        self.lif_d2 = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False)
        self.bn_d3 = _norm(16)
        self.lif_d3 = snn.Leaky(beta=beta, threshold=1.0, init_hidden=True)
        self.deconv4 = nn.ConvTranspose2d(16, IMG_CH, 4, 2, 1, bias=True)
        self.lif_out = snn.Leaky(beta=beta, threshold=1e3, reset_mechanism="none",
                                 init_hidden=True, output=True)
        self.n_pix = image_size * image_size

    # ---------------------------------------------------------------- 编码
    def encode_raw(self, x):
        """跑完 T 步，返回量化**前**的潜层 [B,T,C,h,w]。"""
        utils.reset(self)
        outs = []
        for _ in range(self.num_steps):
            h = _lif_fire(self.lif1, self.bn1(self.conv1(x)))
            h = _lif_fire(self.lif2, self.bn2(self.conv2(h)))
            h = _lif_fire(self.lif3, self.bn3(self.conv3(h)))
            h = self.gdn(self.conv4(h))
            if self.latent_mode == "bernoulli":
                h = _lif_fire(self.lif4, h + self.latent_bias)
            outs.append(h)
        return torch.stack(outs, dim=1)

    def forward(self, x, checkpoints=None):
        """返回 (y_hat, rate_dict, recon_dict)。

        rate_dict 里除了码率，还带上算术编码需要的量：
            extra_sigma : hyperprior 给的逐像素 sigma（编码器必须用同一个，否则真实字节会偏大）
            z_hat/z_step: 边信息 z 的量化值与量化步长（z 也要真的编码传输）
            bits_y 可对时间维累加，得到"只收到前 k 步"的码率
        """
        y = self.encode_raw(x)
        bits_z = torch.zeros((), device=x.device)
        extra, z_hat, z_step = None, None, None
        if self.latent_mode == "bernoulli":
            y_hat = y
            bits_y = self.prior.bits(y_hat)
        else:
            if self.hyperprior is not None:
                extra, bits_z, z_hat, z_step = self.hyperprior(y, self.training)
            y_hat = self.quant(y)
            bits_y = self.prior.bits(y_hat, self.quant.step, extra)
        bpp = (bits_y.sum() + bits_z) / (x.shape[0] * self.n_pix)
        rate = {"bits_y": bits_y, "bits_z": bits_z, "bpp": bpp,
                "extra_sigma": extra, "z_hat": z_hat, "z_step": z_step}
        return y_hat, rate, self.decode(y_hat, checkpoints)

    # ---------------------------------------------------------------- 解码（渐进）
    def decode(self, y_hat, checkpoints=None):
        """y_hat: [B,T,C,h,w] -> {k: recon_k}，recon_k 是只用前 k 步、对时间维取平均的输出。"""
        T = y_hat.shape[1]
        ck = {T} if checkpoints is None else ({int(k) for k in checkpoints if 1 <= int(k) <= T} | {T})
        utils.reset(self)       # 编码已结束，一起清膜电位没关系
        acc, outs = 0.0, {}
        for t in range(T):
            yt = y_hat[:, t]
            if self.latent_mode == "gaussian":
                yt = self.igdgn(yt)
            h = _lif_fire(self.lif_stem, self.bn_stem(self.stem(yt)))
            h = _lif_fire(self.lif_d1, self.bn_d1(self.deconv1(h)))
            h = _lif_fire(self.lif_d2, self.bn_d2(self.deconv2(h)))
            h = _lif_fire(self.lif_d3, self.bn_d3(self.deconv3(h)))
            acc = acc + torch.sigmoid(_lif_mem(self.lif_out, self.deconv4(h)))
            if (t + 1) in ck:
                outs[t + 1] = acc / (t + 1)
        return outs

    # ---------------------------------------------------------------- 工具
    def key_bits(self):
        return self.num_steps * self.latent_channels * self.latent_size * self.latent_size

    @torch.no_grad()
    def warmup_sigma(self, loader, device, n_batches: int = 4, fit_y: bool = True,
                     fit_z: bool = True, verbose: bool = True,
                     train_mode: bool = True) -> float:
        """把 sigma 初始化成潜层的实际标准差 —— **y 先验和 z 先验都要做**。

        必须做：sigma 写死 1.0 而潜层实际 std 是 0.4 时，熵模型一开始就严重失配。
        必须在 **train 模式**下测 —— BN 在 eval 模式走 running stats（初始 0/1），
        分布和训练时用 batch stats 差 3 个数量级（实测 0.001 vs 0.408）。

        z 先验以前从来没人碰过，而它是**边信息码流**的全部概率模型：实测 anchor_v5
        `z_prior.sigma()` = 0.0566，而逐通道拟合出来的 z std 是 0.35（pooled 1.26）——
        差 5~20 倍。z 流占 v5 整个文件的 31%，所以这一项单独就值 ~20% 的体积。
        这里用和 y 先验**完全相同**的口径：mu=0，log_sigma = log(逐通道 std)。

        顺带修一个单位错：原来 `acc` 累积的是**方差**，却直接 `log()` 当 sigma 用，
        于是 sigma 被初始化成 var 而不是 std（v7 日志里 sigma 0.434 而 y_std 0.68，
        0.68^2 = 0.46 —— 正好对上；函数自己的 docstring 写的是"标准差"）。
        现在两处都开方，sigma 才真的是 std。实测 z 流码长：var 口径 28879 bits/图，
        std 口径 12848 bits/图（同一批图，见 recalibrate_prior.py）。

        `train_mode`：训练前预热用 True（BN 走 batch 统计量，和训练时一致）；
        给**已经训练好的** checkpoint 做重标定时用 False（走 running 统计量 + 硬量化，
        和部署时一致）。
        """
        if self.latent_mode != "gaussian":
            return float("nan")
        self.train(train_mode)
        acc = torch.zeros(self.latent_channels, device=device)
        use_z = fit_z and self.hyperprior is not None
        zacc = torch.zeros(self.hyperprior.z_prior.channels, device=device) if use_z else None
        old_zsig = (float(self.hyperprior.z_prior.sigma().mean()) if use_z else float("nan"))
        cnt = 0
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            y = self.encode_raw(x.to(device))
            yf = y.permute(0, 1, 3, 4, 2).reshape(-1, self.latent_channels)
            acc = acc + yf.var(dim=0, unbiased=False)
            if use_z:
                z = self.hyperprior.analyze_z(y)                  # [B*T, Cz, hz, wz]
                zf = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
                zacc = zacc + zf.var(dim=0, unbiased=False)
            cnt += 1
        # 方差 -> 标准差（下限 1e-8 开方后 = 1e-4 = SIGMA_MIN，和 sigma() 的夹取一致）
        std = (acc / max(1, cnt)).clamp_min(1e-8).sqrt()
        with torch.no_grad():
            if fit_y:
                self.prior.log_sigma.copy_(std.log().view(1, -1, 1, 1))
                self.prior.mu.copy_(torch.zeros_like(self.prior.mu))
            if use_z:
                zstd = (zacc / max(1, cnt)).clamp_min(1e-8).sqrt()
                self.hyperprior.z_prior.log_sigma.copy_(zstd.log().view(1, -1, 1, 1))
                self.hyperprior.z_prior.mu.copy_(
                    torch.zeros_like(self.hyperprior.z_prior.mu))
                if verbose:
                    print(f"z 先验预热: z std 均值 {float(zstd.mean()):.4f} "
                          f"[{float(zstd.min()):.4f}, {float(zstd.max()):.4f}] "
                          f"(旧 sigma 均值 {old_zsig:.4f})")
        return float(std.mean())

    @torch.no_grad()
    def load_encoder_front(self, ckpt_path: str):
        """从旧模型热启动编码器前三层（conv1-3 / bn1-3 形状完全一致）。"""
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj)
        own = self.state_dict()
        keep = {k: v for k, v in sd.items()
                if k.startswith(("conv1.", "bn1.", "conv2.", "bn2.", "conv3.", "bn3."))
                and k in own and own[k].shape == v.shape}
        self.load_state_dict(keep, strict=False)
        return len(keep), len(sd)


# ======================================================================================
# 真实算术编码（constriction）—— 实际字节数
# ======================================================================================
def _categorical(p):
    import constriction
    return constriction.stream.model.Categorical(p, perfect=False)


# 有效 sigma 的硬边界。**全仓库唯一的定义**：先验 sigma、hyperprior 的 log_sigma、
# 算术编码的网格、编解码两端的分桶，全部从这里取，避免各处各写一个常量而对不上。
SIGMA_MIN = 1e-4
SIGMA_MAX = 1e6
# hyperprior 的 h_s 输出 clamp 到 ±14 -> exp 之后 sigma ∈ [8.3e-7, 1.2e6]，
# 再被上面的 [1e-4, 1e6] 夹住。
LOG_SIGMA_CLAMP = 14.0

# 固定的对数 sigma 网格。constriction 0.5 的 Categorical **只接受 1-D 概率数组**，
# 没法一次喂 [N,K] 的逐符号分布。所以把逐像素 sigma 量化到这张固定网格上，
# 同一桶的符号共用一个 1-D 模型。编解码两端都能从 sigma 独立算出桶号，不需要传边信息。
#
# 旧网格是 exp(linspace(log 0.01, log 100, 48)) —— 上界 100，而有效 sigma 的实际
# 上界是 1e6，实测 v5 有 **9.4%** 的像素（normab 5.0%）被强行按 sigma=100 编码，
# 与模型自己预测的 sigma（p99 27 万、max 38 万）差 3~4 个数量级：编码器用的概率
# 模型根本不是模型的概率模型。新网格上下界取 SIGMA_MIN/SIGMA_MAX（= clamp 边界，
# 所以"落在网格外"的像素**恒为 0**），160 点 -> 每格 0.0755 decade（19%），
# 比旧网格的 0.085 decade（21.6%）还细。桶号从 sigma 推导，不进码流，多几桶不花钱。
SIGMA_GRID_N = 160
_SIGMA_GRID = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), SIGMA_GRID_N))
# 端点写死成常量本身，免得 exp(log(1e-4)) 的舍入让 sigma == SIGMA_MIN 的像素
# 落到"网格下界之外"（实测会多出 0.7% 的假越界）。
_SIGMA_GRID[0] = SIGMA_MIN
_SIGMA_GRID[-1] = SIGMA_MAX


def _bucket_ids(s: np.ndarray) -> np.ndarray:
    """逐元素的 sigma -> 对数网格上的桶号（两端饱和到首尾桶）。"""
    ls = np.log(np.clip(s.astype(np.float64), SIGMA_MIN, SIGMA_MAX))
    lg = np.log(_SIGMA_GRID)
    return np.abs(ls[:, None] - lg[None, :]).argmin(axis=1)


def _bin_probs(sigma_val: float, mu_val: float, k_max: int) -> np.ndarray:
    """给定标量 sigma/mu，算出整数的量化区间概率质量 [K]（float64，已归一化）。

    和 `FactorizedGaussianPrior.bits` 用同一个 `_log_bin_prob`，所以
    "训练时用的解析码率"和"算术编码真正用的概率表"是同一个模型（旧版两边
    各写一遍 `_phi` 相减，尾部都是 0 -> 1e-9 的地板）。
    float64 在这里是免费的（每个桶只算一次），顺带避开 float32 的抵消。
    保底 1e-12 只为让 constriction 永远拿到正概率，正常区间用不到。
    """
    s = max(float(sigma_val), 1e-12)
    ks = torch.arange(-k_max, k_max + 1, dtype=torch.float64)
    t = (ks - float(mu_val)) / s
    h = torch.full_like(t, 0.5 / s)
    p = _log_bin_prob(t, h).exp().clamp_min(1e-12)
    return (p / p.sum()).numpy().astype(np.float64)


def encode_latent_bytes(y_hat, step, prior, extra_sigma=None,
                        k_max: int = 32, verify: bool = False):
    """按先验做算术编码，返回 (实际比特数, 符号总数, 每通道概率表, 量化符号)。

    三个关键点：
    1) **整张图只开一个 RangeEncoder**，逐通道/逐桶切换模型。
       早前按通道各开一个编码器，每个都有约 4 字节 flush 开销 ——
       16 通道 = 64 字节固定底噪，在 43 字节的工作点上直接把体积翻倍。
    2) **必须把 hyperprior 的逐像素 sigma 传进来**（extra_sigma）。
       不传的话编码器用逐通道粗模型、而码率估计用逐像素细模型，
       两者不一致会让真实字节远大于估计（实测 λ=0.001 时 +42.9%）。
    3) 逐像素 sigma 经固定对数网格分桶（constriction 不支持 batch 模型）。
       桶号由 sigma 唯一决定，解码端能独立复现，**不产生额外边信息**。

    统计量与 FactorizedGaussianPrior.bits 完全一致：
        p(sym=k) = Phi((k+0.5-mu)/sig) - Phi((k-0.5-mu)/sig),  sig = log_sigma.exp()*extra_sigma
    """
    import constriction
    yn = (y_hat.detach() / float(step)).round().clamp(-k_max, k_max).long().cpu()
    mu = prior.mu.detach().cpu()
    # 注意：extra_sigma 要先和 GPU 上的 log_sigma 相乘再搬到 CPU，否则设备不匹配
    sig = prior.sigma(extra_sigma).detach().cpu()        # 5D 或 4D
    C = yn.shape[2]
    probs_per_c = []

    enc = constriction.stream.queue.RangeEncoder()
    sym_store = []
    for c in range(C):
        syms = (yn[:, :, c].reshape(-1) + k_max).numpy().astype(np.int32)   # [N]
        s = (sig[:, :, c] if sig.dim() == 5 else sig[:, c]).reshape(-1).numpy().astype(np.float64)
        m = float(mu[:, c].reshape(-1)[0])
        if s.size == 1:
            s = np.full(syms.size, float(s[0]))
        sym_store.append(syms)
        if extra_sigma is None:
            # 单桶：所有符号共用一个模型
            p = _bin_probs(float(s[0]), m, k_max)
            probs_per_c.append(p)
            enc.encode(syms, _categorical(p))
        else:
            bid = _bucket_ids(s)
            per_bucket = {}
            for b in np.unique(bid):
                mask = bid == b
                p = _bin_probs(float(_SIGMA_GRID[b]), m, k_max)
                per_bucket[int(b)] = p
                enc.encode(syms[mask], _categorical(p))
            probs_per_c.append(per_bucket)

    n_sym = int(yn[:, :, 0].numel()) * C
    if verify:
        dec = enc.get_decoder()
        for c in range(C):
            syms = sym_store[c]
            s = (sig[:, :, c] if sig.dim() == 5 else sig[:, c]).reshape(-1).numpy().astype(np.float64)
            m = float(mu[:, c].reshape(-1)[0])
            if s.size == 1:
                s = np.full(syms.size, float(s[0]))
            out = np.zeros(syms.size, dtype=np.int32)
            if extra_sigma is None:
                out = dec.decode(_categorical(probs_per_c[c]), syms.size)
            else:
                bid = _bucket_ids(s)
                for b in np.unique(bid):
                    mask = bid == b
                    out[mask] = dec.decode(_categorical(probs_per_c[c][int(b)]), int(mask.sum()))
            if not bool((out == syms).all()):
                raise RuntimeError(f"算术编码回读校验失败 (通道 {c})")
    return int(enc.num_bits()), n_sym, probs_per_c, yn


def encode_z_bytes(z_hat, z_step, z_prior, k_max: int = 32, verify: bool = False):
    """对 hyperprior 的边信息 z 做算术编码。

    之前 z 的比特只算进"估计"，从没真正编码过 —— 真实字节数会系统性偏小。
    z 只有 [T, Cz, h/4, w/4] 个符号（256² 时约 7680 个），但低码率下不可忽略。
    """
    import constriction
    yn = (z_hat.detach() / float(z_step)).round().clamp(-k_max, k_max).long().cpu()
    mu = z_prior.mu.detach().cpu().cpu() if hasattr(z_prior.mu, "cpu") else z_prior.mu
    sig = z_prior.sigma().detach().cpu()
    # z 的形状是 [B*T, Cz, h, w]，通道维在 dim=1
    C = yn.shape[1]
    enc = constriction.stream.queue.RangeEncoder()
    probs_per_c = []
    for c in range(C):
        s, m = float(sig.flatten()[c]), float(mu.flatten()[c])
        p = _bin_probs(s, m, k_max)
        probs_per_c.append(p)
        syms = (yn[:, c].reshape(-1) + k_max).numpy().astype(np.int32)
        enc.encode(syms, _categorical(p))
    n_sym = int(yn[:, 0].numel()) * C
    if verify:
        dec = enc.get_decoder()
        for c in range(C):
            syms = (yn[:, c].reshape(-1) + k_max).numpy().astype(np.int32)
            got = dec.decode(_categorical(probs_per_c[c]), syms.size)
            if not bool((got == syms).all()):
                raise RuntimeError(f"z 算术编码回读校验失败 (通道 {c})")
    return int(enc.num_bits()), n_sym


def encode_binary_bytes(spk, prior, verify: bool = False):
    """按伯努利先验做算术编码（binary 模式），同样只用单一码流。"""
    import constriction
    p1 = prior.prob().detach().cpu().flatten()
    bits = spk.detach().round().long().cpu()
    C = bits.shape[2]
    probs = [np.array([1 - float(p1[c]), float(p1[c])], dtype=np.float64) for c in range(C)]

    enc = constriction.stream.queue.RangeEncoder()
    for c in range(C):
        enc.encode(bits[:, :, c].reshape(-1).numpy().astype(np.int32), _categorical(probs[c]))
    n_sym = int(bits[:, :, 0].numel()) * C
    if verify:
        dec = enc.get_decoder()
        for c in range(C):
            got = dec.decode(_categorical(probs[c]), int(bits[:, :, c].numel()))
            if not bool((got == bits[:, :, c].reshape(-1).numpy()).all()):
                raise RuntimeError("算术编码回读校验失败")
    return int(enc.num_bits()), n_sym, probs, bits
