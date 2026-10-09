[English](README.md) | **简体中文**

# 强制编码器 / 解码器分离的 SNN 图像压缩

一个基于**脉冲神经网络（spiking neural networks，`snnTorch`）**加上**算术编码
（arithmetic coding，`constriction`）**构建的图像压缩系统。训练好的 SNN **编码器**
把图像变成极小的算术编码密钥；训练好的 SNN **解码器**把密钥还原成图像。

这两半被设计为运行在**两台不同的机器**上：

```
  Machine A — "edge"                             Machine B — "ground station"
  ┌──────────────────────────────┐               ┌──────────────────────────────┐
  │ image (any size)             │               │ key.bin (a few kB)           │
  │   ↓ Resize + CenterCrop      │               │   ↓                          │
  │ 256×256×3                    │   key.bin     │ range-decode →  ŷ, ẑ          │
  │   ↓ SNN encoder (LIF, T=10)  │ ────────────► │   ↓ SNN decoder (LIF + IGDN) │
  │   ↓ GDN → y                  │  ~1–21 kB     │ 256×256×3 reconstruction      │
  │   ↓ … arithmetic-coded key   │               │                              │
  └──────────────────────────────┘               └──────────────────────────────┘
    has: encoder half of the weights               has: decoder half of the weights
    lacks: decoder,  h_a                           lacks: encoder, h_a, the original
```

图中展示的是 **256 px、C=32、T=10 参考配置**。CLI 自身的默认值更小（128 px、C=24、
T=6）——参见 §3.3 中的注意事项。

A 无法重建（它没有解码器）。B 无法编码（它没有编码器，也没有 `h_a`，即超先验分析
变换）。这不是一种约定——它由 Docker 强制执行：两个镜像、一个内部网络、各自一个
只读模型挂载，以及一个携带**模型哈希**的密钥头，使得不匹配的一对会大声失败，而
不是输出垃圾数据。见 §6.4 和 [`service/README.md`](service/README.md)。

**一句话状态：** 该流水线是正确的，并已端到端验证（比特精确的往返；十分钟运行的
五个 epoch 内达到 22.50 dB），但*压缩*性能没有竞争力——在这些码率下 JPEG 以很大
优势胜出，而且本项目最重要的研究想法已被严格证伪并默认关闭（§7）。这两件事都是
刻意记录在此的。

---

## 1. 哪些能用 / 哪些不能用

**能用**

* 端到端：图像 → SNN 编码器 → 算术编码密钥 → SNN 解码器 → PNG。
* **比特精确**的编码/解码往返（最大绝对差 `0.0`），在 256 px 下、在把 128 px 输入
  256 训练的模型时、以及对于*部分*密钥（时间步 / 通道数少于模型训练时所用的）均
  成立。
* A/B 分离是真实的且经过验证：每个容器只能看到自己那一半权重，A 和 B 无法触及
  网络，网关是唯一对外发布的端口。
* 一套真实验证协议（`eval` 模式 + 硬量化 + BN 运行统计量），它不会像训练损失那样
  对你说谎，以及基于它的检查点选择。
* 一轮审计发现并修复了六个真实缺陷——包括一个存在**零梯度死区**的熵模型，以及
  一个**编码器/解码器 sigma 不匹配、损坏了 9.34 % 解码像素**的问题。见 §5。

**不能用 / 尚不存在**

* **质量。** 这里最诚实的最佳数字约为 22.5 dB——在 256 px 参考几何下、5 个 epoch
  的 10 分钟运行后为 22.50 dB（`models/anchor_best.pth`），而此前最佳模型在 2.845 bpp
  容器计量下为 21.13 dB。在可比的字节数下 JPEG 仍然遥遥领先。这不是一个有竞争力
  的编解码器。
* **没有作为结果的率–失真扫描。** `--lam` 是一个旋钮；没有人据此发表过正规的
  BD-rate 曲线。
* **锚点 / 跨图像的想法已死。** 在每一种测试过的诚实熵模型下，把一组相似图像相对
  于一张参考图像来编码都会花费*更多*字节。它保留在代码中、位于 `--use-anchor`
  之后以便复现，并且**默认关闭**。
* **早期的“胜利”是假象。** `anchor_v7` 表面上的低码率来自一个无法学习 sigma 的
  熵模型（§5，第 2 项）；相对于 oracle 测量时，它在锚点路径上比独立编码*差* 58 %。
* **没有针对超先验的学习式熵模型。** 先验是在预热时通过矩拟合校准的，而非联合
  训练，到离散 MLE 之间的剩余差距有文档记录但未实现。

---

## 2. 架构

`SAE_RD`（位于 `sae_rd.py`），一个率–失真 SNN 自编码器：

```
encoder                                        decoder (mirror)
 x [3,256,256]                                  ŷ [C,16,16]
   Conv(3→32,  s2) + BN + LIF      → [32,128,128]     IGDN → stem Conv(C→128, s1) + BN + LIF
   Conv(32→64, s2) + BN + LIF      → [64, 64, 64]     ConvT(128→64, s2) + BN + LIF
   Conv(64→128,s2) + BN + LIF      → [128,32, 32]     ConvT(64 →32, s2) + BN + LIF
   Conv(128→C, s2) + GDN           → [C,  16, 16]     ConvT(32 →16, s2) + BN + LIF
                                                        ConvT(16 →3,  s2) + LIF(mem only)
   y ──► ScaleHyperprior             ẑ, μ, log σ        sigmoid → average over T → [3,256,256]
        (h_a → ẑ → h_s → σ)        (factorized Gaussian)
   y ──► + uniform noise (annealed) ──► y_hat         ← quantizer: noise-annealing
         1.0 (train) / round (eval)                     + straight-through estimator
```

* **带有 `T` 个时间步的 LIF 神经元**（参考配置中默认 `T = 10`）。编码器在每个时间
  步发放脉冲；解码器累积 `sigmoid(membrane)` 并沿时间求平均。在一次早期消融中，
  时间平均把测试 MSE 从 0.0278（仅用最后一步）降到 0.0165——只用最后一步的变体
  差 68 %——因为平均抑制了脉冲噪声，并给梯度提供了 `T` 条路径。
* **GDN / IGDN**（广义除法归一化，generalized divisive normalization）位于潜层，
  来自尺度超先验文献——它在瓶颈处取代了白化 BatchNorm，并保持潜层的分布结构。
* **尺度超先验（Scale Hyperprior，Ballé 2018）**：`h_a` 把 `y` 下采样为 `ẑ`；`ẑ`
  本身也被算术编码（即 *z 流*）；`h_s` 把 `ẑ` 映射回逐像素的 `σ`，用作 `y` 上因子化
  高斯熵模型的尺度。交付的字节中大约 30 % 是 z 流，这就是它必须被计入的原因（§4）。
* **带噪声退火 + STE 的均匀量化器**，量化步长固定为 `1.0`：损失只依赖于 `step/σ`，
  所以让两者都可浮动只会留下一个有效自由度，它们会互相打架。
* **解码器归一化默认使用 BatchNorm**，基于实测证据：在一次受控 A/B 中（相同数据、
  采样器和种子），BatchNorm 达到 **17.19 dB**，GroupNorm 为 **15.54 dB**，而不做
  归一化则**完全崩塌**——潜层死亡（`bpp → 0.0000`、`y_max = 0`、10.78 dB）。
  日志：`normab_bn.log`、`normab_gn.log`、`normab_none.log`。

量化后的潜层用 `constriction` 的区间编码器做熵编码：先 z 流，再使用由解码出的 `ẑ`
导出的 σ 编码 y 流。**不传输任何概率表**——B 从它自己的解码器那一半重新计算。
容器头（`codec/rd_codec.py`，密钥格式 v3，36 字节）携带 magic `SNNK`、版本与
codec id、`image_size`、模型原生的 `latent_channels`/`num_steps`、用于该特定密钥的
`steps_used`/`channels_used` 旋钮、8 字节**模型哈希**、字母表大小 `K` 以及载荷长度。

---

## 3. 快速开始

### 3.1 安装

```bash
python -m pip install torch torchvision snntorch constriction pillow numpy matplotlib
```

检查你的 PyTorch 是否是仅 CPU 的构建——这是一个静默且非常常见的失败：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.14.0+cpu None False   -> CPU-only wheel: the GPU will not be used
# 2.14.0+cu130 13.0 True  -> OK
```

在 Python 3.14 上，Blackwell（RTX 50 系列）显卡的可用组合是
`cu130 / torch 2.14.0 / torchvision 0.29.0`；`cu128` 的 torch 更旧，而 `cu129` 没有
`cp314` wheel。安装时使用 `--force-reinstall`（或先卸载），否则 pip 会认为
`2.14.0+cpu` 已满足要求而什么都不做：

```bash
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

`get_arch_list()` 必须包含 `sm_120`，否则你会得到
`CUDA error: no kernel image is available for execution on the device`。

一切也都能在 CPU 上运行（代码会自动回退），只是更慢。

### 3.2 数据

| 数据集 | 图像数 | 位置 | 备注 |
|---|---|---|---|
| Flowers102 | 8189 | `data/flowers-102/` | 三个划分（`train`+`val`+`test`）全部**合并**，然后重新划分 |
| DIV2K + Flickr2K | 3450 | `data/div2k_extracted/` | 2K 图像，**没有类别标签**；zip 文件保留在 `data/div2k/` |

合计：**11639** 张图像。15 % 留出（`--holdout-frac 0.15` → `max(200, 0.15·n)`）：
**9894 训练 / 1745 留出**；每个 epoch 评估留出集中前 `--eval-limit` 张（默认 300）。
数据增强（`--augment crop` = `RandomResizedCrop(0.5–1.0)` + 水平翻转）**仅应用于
训练划分**；留出集始终只做普通的 resize + center crop。

```powershell
# DIV2K download helper (PowerShell), then unzip into data/div2k_extracted/
.\download_div2k.ps1
```

两个数据集都未提交——`data/` 和所有 `*.pth` 都在 `.gitignore` 中。请另行获取数据集，
并注意仓库中不附带任何检查点；**训练好的检查点单独提供**（release 资源）。没有它
你仍然可以从零开始训练，`service/_smoke_codec.py` 会告诉你缺什么。

### 3.3 训练

```bash
python anchor.py train --epochs 30          # the default command (anchor OFF)
```

那一条命令就是推荐的起点。它的 argparse 默认值是
`--data both --batch-size 16 --norm bn --augment crop --image-size 128
--latent-channels 24 --num-steps 6 --lam 0.01 --lr 1e-3 --holdout-frac 0.15`，不使用
锚点，并写出 `anchor_rd.pth`（按真实验证最优的会保存为 `anchor_rd_best.pth`）。

> **注意事项。** 上面的*架构*一节描述的是 256 px / C=32 / T=10 参考配置，它**不是**
> argparse 默认值。要得到它，你必须显式地要求——这是 `run_v8.ps1` 使用的配置
> （`$Epochs = 330`、`$BudgetMin = 570`、自动续跑、`--group-size 16 --anchor-frac 0.5`
> 用于那条现已默认关闭的锚点路径）：
>
> ```powershell
> python anchor.py train --image-size 256 --latent-channels 32 --num-steps 10 `
>     --epochs 330 --lam 0.01 --lr 3e-4 --noise-floor 0.2 --workers 8 `
>     --save-every 5 --val-every 5 --val-batches 10 --eval-limit 1228 --out anchor_v8.pth
> ```

在 RTX 5070 Ti Laptop、9894 张训练图像、618 个 batch × 16 = 9888 图像/epoch 下测得
的开销：

| 配置 | epoch 时间 | 来源 |
|---|---|---|
| 128 px、C=24、T=6（默认） | **51.0 s** | `anchor_v9.log` |
| 256 px、C=32、T=10（+ 真实验证） | **~102 s** | `anchor_v8.log` |

### 3.4 评估

```bash
# independent coding vs anchor+residual, per-group byte totals
python anchor.py eval --weights anchor_v5.pth --group-size 4 --n-groups 12

# independent vs star vs minimum-spanning-tree over all pairs in a group
python anchor.py allpairs --weights anchor_v5.pth --group-size 4 --n-groups 12
```

`eval` 会以无歧义的标签同时打印 bits 和 bytes，包含 z 流和超先验的逐像素 sigma，
并且**先加载检查点，从而按模型原生分辨率进行评估**（过去 256 训练的模型会被静默
地在 128 px 数据上评估）。`--out-json`（默认 `anchor_eval.json`）会得到同样的数字。

### 3.5 拆分模型并部署 A/B

```bash
python -m codec.model_io --ckpt anchor_v5.pth --out ./models
# → models/a/encoder.pth  models/b/decoder.pth  models/manifest.json
```

```
模型指纹 1a5f6470c5592f19
A 机 encoder.pth : 36 个参数张量
B 机 decoder.pth : 36 个参数张量
A 独有 28 个（B 拿不到）...   B 独有 28 个（A 拿不到）...
```

`h_s` 有意出现在**两半**中——编码器需要 σ 来*编码*，解码器需要 σ 来*解码*，而且它
很小，不会泄露任何关于解码器的信息。拆分器还会计算成为密钥头 `model_hash` 的
SHA-256 前缀。

然后，在 Docker 已运行的情况下：

```powershell
docker compose -f service\docker-compose.yml build     # first build ~5–10 min (torch CPU wheel)
docker compose -f service\docker-compose.yml up -d
docker compose -f service\docker-compose.yml ps        # all three healthy
# open http://localhost:8080
```

仅主机（host-only）的替代方案（三个终端，不用 Docker）：`python service\_run.py a --port 8001`、
`python service\_run.py b --port 8002`，以及
`python service\_run.py gateway --port 8000 --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002`。

完整细节——API 参考、20 级码率控制阶梯、如何从 shell 验证隔离，以及 Docker 的坑
——都在 [`service/README.md`](service/README.md) 中。

---

## 4. 结果，以及如何诚实地读一个码率

### 4.0 这里的码率是怎么计的（先读这一节）

这个仓库里有三种不同的 “bpp”。静默地比较它们是最容易自欺的方式，所以：

* **训练日志中的 `bpp`**（`anchor_*.log` 里的 `独立码率 … bpp`）是一个*解析式*
  估计，由熵模型在量化仍以退火噪声模拟时算出。它不是编码器实际写出的东西。
* **容器 `bpp`** 才是真实的东西：`codec/rd_codec.py::encode_key` 实际输出的内容
  ——算术编码的 z 流**加上** y 流**再加上** 36 字节头，字母表按实际符号数量设定。
  这才是部署的一对 A/B 实际传输的东西。
* **仓库的两半对像素的计法也不同。** RD 路径（`anchor.py`、`anchor_bytes.py`、
  容器）使用 `bits / (H·W)`；遗留路径的 `sae_model.bpp()` 使用 `bits / (H·W·3)`，
  即按*子像素*。同一个 10 254 字节的密钥在 `compress.py` 中是 **0.417 bpp**，而按
  RD 约定是 **1.25 bpp**。下面每一张表都会说明它用的是哪种约定；永远不要跨约定
  比较。

对于 256 px 下的 `anchor_v5`，两者相差约 13 %（训练日志中解析式 3.21 bpp，对比容器
2.845 bpp）。本仓库中的历史报告有时会把一种协议的 PSNR 与另一种协议的码率并列
引用；下面的表始终会标明协议。

### 4.1 短运行合理性基线（不是一个结果）

`anchor_v9.log` 是在 **256 px 参考几何**下运行默认（无锚点）命令、由 10 分钟墙钟
预算在 5 个 epoch 后停止的一次运行：

```
python anchor.py train --image-size 256 --latent-channels 32 --num-steps 10 \
    --epochs 30 --time-budget-min 10 --val-every 1 --save-every 1 --out anchor_v9.pth
```

每个 epoch 都会做真实验证（eval 模式、硬量化、模型从未见过的留出图像）：

| epoch | 真实验证 PSNR | 解析码率 | sigma | 时间 |
|---|---|---|---|---|
| 1 | 20.75 dB | 1.2685 bpp | 0.612 | 107.6 s |
| 2 | 21.75 dB | 1.2540 bpp | 0.606 | 106.2 s |
| 3 | 22.06 dB | 1.1679 bpp | 0.598 | 105.9 s |
| 4 | 22.16 dB | 1.1264 bpp | 0.595 | 106.2 s |
| 5 | **22.50 dB** ← 最佳 | 1.0670 bpp | 0.590 | 105.9 s |

```
ep   5/30 | L 0.0174 | D 0.0004 | 独立码率 1.0670 bpp | sigma 0.590 | 噪声 0.83 | 105.9s
         | y_std 0.65 y_max 5 | ★真实验证 22.50dB
```

把它读作*“损失接线正确，五个 epoch 就已经超过了此前最佳”*——一个冒烟测试基线，
**不是最终结果**。该检查点已提交在 `models/anchor_best.pth`（从 `anchor_v9_best.pth`
复制而来）。

让这个数字有意义的是**潜层尺度**：`y_std 0.65`、`y_max 5`。在熵模型修复之前，
一次可比的运行漂移到 `y_std ~130`、`y_max ~970`，而学到的先验一直冻结——模型在
针对一个它实际上无法影响的码率项做优化。

两个注意事项：

- **码率列是来自熵模型的解析式估计**，不是部署的一对实际传输的容器码率。它不能
  与 §4.2 中的容器数字直接比较。
- 同一条命令更早的一次启动省略了显式几何参数，静默地使用了 argparse 默认值
  （`--image-size 128 --latent-channels 24 --num-steps 6`），到第 4 个 epoch 时报告为
  0.7559 bpp 下的 19.77 dB。那些数字来自**不同的几何配置**，不得与上表比较。始终
  显式传入几何参数。

**PSNR 列是可信的，并且与 §4.2 可比**，因为测量方式相同（eval 模式、硬量化、未见过
的图像）：**22.50 dB 已经超过了此前最佳模型的 21.13 dB**——而且码率低得多。

### 4.2 参考点：此前最佳的模型

诚实的容器计量，在模型原生的 256 px 下、42 张同类留出图像
（`anchor_bytes_report.md`、`anchor_bytes_summary.txt`）：

| 检查点 | 角色 | y 流 B/图 | z 流 B/图 | 合计 B/图 | PSNR | 容器 bpp |
|---|---|---|---|---|---|---|
| `anchor_v5` (BN, C=32, T=10) | 独立 | 15988.0 | 7281.5 | **23305.5** | **21.13 dB** | **2.845** |
| `anchor_v5` | 锚点 | 16420.7 | 7602.7 | 24059.3 | 20.76 dB | 2.937 |
| `anchor_v5` | 残差 | 17875.2 | 8828.2 | 26739.4 | 20.73 dB | 3.264 |
| `normab_gn_best` (GN) | 独立 | 11472.7 | 953.9 | 12462.6 | 16.83 dB | 1.521 |

所以：**21.13 dB @ 2.845 bpp。** 注意 z 流约占这些字节的 31 %——任何省略它的计量
都是错的，而这正是 §5 第 1 项修复的 bug。

> 关于早期笔记中流传的一个数字，*“21.13 dB @ 0.948 bpp”*：这两个数字来自不同的
> 模型和不同的协议。21.13 dB 是 `anchor_v5` 的容器 PSNR；≈0.95 bpp 是 `anchor_v7`
> （以及 GroupNorm 消融）训练日志中的*解析式*码率，其熵模型后来被证明是坏的。
> 不要把两者配对。

### 4.3 参照

256×256、质量 85 的 JPEG 大约是 10–15 kB。因此来自*遗留*路径（见 §6.3）的 10.25 kB
固定码率密钥在码率上与之相近，而在相近的字节数下 JPEG 比这里测得的任何东西都好
**10–15 dB**。“比原始 RGB 小 94.8 %” 是一个稻草人论证，本 README 中任何地方都没有
把它当作结果引用。

---

## 5. 正确性：六个被测量到的 bug，已发现并修复

一轮专门的审计（`fix_round_report.md`，附逐项 `_verify_fixN.py` 脚本）发现了六个
缺陷。四个在静默地消耗码率或正确性，一个是部署破坏性的，一个是用测量取代的设计
约定。实测的前/后对比：

| # | 缺陷 | 实测影响 |
|---|---|---|
| 1 | `evaluate` 把 **bits 当作 bytes 打印**（8×），省略了 z 流和 `extra_sigma`，并在 128 px 数据上评估 256 训练的模型 | 码率被高估 **9.96×**；232 296 “B” → 真实的 **23 320 B**（2.847 bpp）；z 流 = 字节的 31.6 % |
| 2 | 熵模型有一个**零梯度死区**：float32 的 `erf` 在约 3.9σ 附近饱和，随后 `clamp_min(1e-9)` 把超过约 5.5σ 的每个符号钉在恰好 **29.8974 bits、对 σ 的梯度恰好为 0.0**——σ 完全无法学习 | 在 `t=y/σ=10` 时：码率 29.8974 / 梯度 0.0 → **69.6909 / −131.6**；与 float64 参考值吻合到 **7.7e-12 bits**；`anchor_v7` 的解析式 y 码率 387 034.8 → 20 249 102 bits/图，即旧模型一直*隐藏*了其真实代价的 98 % |
| 3 | `_SIGMA_GRID` 只覆盖 [0.01, 100]，而有效 σ 可达 1e5–1e6 | 4 个数量级上 48 个点（0.085 decade 步长）→ **[1e-4, 1e6] 上 160 个对数间隔点**（0.063 步长，比以前更细）；落在网格外的像素 **9.374 % → 0.000 %**（v5）、23.601 % → 0 %（v7）、5.467 % → 0 %（normab_gn，其 y 流下降 −24.71 %） |
| 4 | `z_prior` 从未被校准，且一个单位 bug 取了 `log(variance)` 却把它当作 `log σ` | z 流 **97 301.9 → 15 413.4 bits/图（−84.2 %）**；由于 z 流约占交付字节的 31 %，这相当于**整个文件少约 26 %**，而且它在解码时是**免费的**（没有旁信息）。旧 σ 为 0.0566，而实测 z std 为 1.2594 |
| 5 | 编码器与解码器在 σ 截断上不一致（`±14`/`[1e-4,1e6]` 对比 `±8`/`[1e-3,1e3]`），并且 σ 的组装方式不同 | **9.34 % 的像素解码到了错误的 σ 桶**；现在使用单一共享路径，并在 256 px、在把 128 px 输入 256 模型、以及部分密钥下都实现**比特精确的往返**（最大绝对差 `0.0`） |
| 6 | 解码器归一化按惯例是 GroupNorm | 受控 A/B：**BatchNorm 17.19 dB / GroupNorm 15.54 dB / 无归一化崩塌**（`bpp → 0.0000`、`y_max 0`、10.78 dB）→ 现在默认使用 BatchNorm |

#5 的部署层面后果：`python service\_smoke_codec.py` 现在加载拆分后的
`models/a/encoder.pth` + `models/b/decoder.pth`，在三个码率点
（24 292 / 7 496 / 3 164 B）编码，解码到 16.26 dB，并正确地**拒绝**错误哈希的密钥
和被截断的密钥。

报告中已说明的、*未*完全修复的残留注意事项：桶 id 仍由 float32 的 σ 导出，所以
处在桶边界 1 ulp 之内的像素原则上可能在不同技术栈间被不同地编码和解码（概率约
每像素 1e-7，约 1 % 的 256 px 图像会出现一个翻转像素）；并且 z 字母表默认值
（`sae_rd.encode_z_bytes` 中的 `32`）仍与容器的（`128`）不同。

---

## 6. CLI 参考

### 6.1 `anchor.py` —— 当前的 RD 路径

```
python anchor.py train    [flags]     # train the RD SNN autoencoder
python anchor.py eval     [flags]     # independent vs anchor+residual byte totals
python anchor.py allpairs [flags]     # independent vs star vs MST optimum
```

共享标志（三个子命令通用）：

| 标志 | 默认值 | 含义 |
|---|---|---|
| `--data` | `both` | `flowers`（8189，有标签）/ `div2k`（3450，无标签）/ `both` |
| `--norm` | `bn` | 解码器归一化：`bn`（BatchNorm，实测最优）/ `gn` / `none` |
| `--batch-size` | `16` | 无锚点（默认）路径的批大小；使用 `--use-anchor` 时组大小优先 |
| `--image-size` | `128` | 输入/输出边长；**必须是 16 的倍数**；参考配置使用 `256` |
| `--latent-channels` | `24` | C。主要的码率旋钮：密钥符号数 ∝ C |
| `--num-steps` | `6` | T。SNN 时间步；符号数 ∝ T，计算量 ∝ T |
| `--lam` | `0.01` | （仅 `train`）率–失真权衡：`L = D + lam · bits/pixel` |
| `--use-anchor` | 关闭 | 启用**已被证伪**的锚点+残差路径（默认关闭；§7） |
| `--augment` | `crop` | 仅对训练划分做 `none` / `flip` / `crop`（RandomResizedCrop + 水平翻转） |
| `--holdout-frac` | `0.15` | 从 11 639 张图像中留出的比例（`max(200, ·)`） |
| `--residual-mode` | `train` 时为 `sub`，`eval` 时为检查点中的值 | `sub` = 硬结构残差 `coded = quant_hard(y_i) − quant_hard(y_ref)`；`learned` = 旧的 learned-residual 行为，保留用于 A/B 复现 |

其他值得了解的标志：`--z-channels 48`、`--use-hyperprior`（开启；`--no-hyperprior`
将其关闭）、`--div2k-limit 6000`、`--workers 4`、`--eval-limit 300`、
`--group-size 4`（仅锚点路径）、`--seed 42`、`--weights anchor_rd.pth`、
`--out anchor_rd.pth`、`--out-json anchor_eval.json`、`--n-groups 12`。

仅 `train`：`--epochs 30`、`--lr 1e-3`、`--grad-clip 1.0`、`--max-batches 0`、
`--hot-start ""`、`--ref-wiring inp|out`、`--ref-mode star|pairs`、`--save-every 5`、
`--resume ""`、`--anchor-frac 0.5`、`--noise-floor 0.2`（退火下限；把它降到 0 会让
熵模型与硬量化不一致，码率跳升约 75 %）、`--val-every 10`、`--val-batches 4`、
`--time-budget-min 0`。

输出：`--out` 检查点加上 `<out>_best.pth`（按**真实验证**最优，不是按训练损失），
以及每个 epoch 一行日志，其 `★真实验证 … dB` 字段才是可信的数字。

### 6.2 `codec.model_io` —— 模型拆分器

```bash
python -m codec.model_io --ckpt anchor_v5.pth --out ./models [--codec-id 3]
```

写出 `models/a/encoder.pth`（仅编码器那一半）、`models/b/decoder.pth`（仅解码器那
一半）和 `models/manifest.json`（源名称、`model_hash`、几何、张量计数）。
`codec.load_side(ckpt, "a"|"b")` 以 `strict=False` 加载一半，报告它拿到多少张量、
缺多少，并把量化器切换为硬取整。`manifest_of()` 读回清单。同一文件还暴露
`model_hash_of()`。

### 6.3 `compress.py` / `decompress.py` —— 遗留的固定码率路径

这些是仓库中**更旧的、算术编码之前**的一半，构建在 `sae_model.py` 和
`sae_cifar.pth` 检查点之上。它们*仅*按名称前缀加载一半权重
（`build_model_from_checkpoint(..., prefixes=("encoder.",))` /
`prefixes=("decoder.",)`），这是 A/B 分离的原始——也弱得多——的形式。用它们复现
旧的固定码率行为，不要用于新工作。

```bash
python compress.py   --image cat.jpg --output key.pt [--format bits|float] [--save-input32]
python decompress.py --key key.pt --output recon.png [--reference cat.jpg] [--compare]
```

密钥大小是**固定的、与内容无关的**：`T × C × (N/16)²` bits，加上 14 字节头
（magic `SNNK`、版本 2、flags、`T`、`C`、`h`、`w`）。下表的 `bpp` 列使用遗留的
`sae_model.bpp()` 约定——`bits / (H·W·3)`，即按子像素。除以 3 可换算为
§4.1–§4.2 使用的 RD 约定。

| C | T | 密钥 bits | 密钥文件 | bpp |
|---|---|---|---|---|
| 8 | 10 | 20 480 | 2.57 kB | 0.104 |
| 16 | 10 | 40 960 | 5.14 kB | 0.208 |
| **32** | **10** | **81 920** | **10.25 kB** | **0.417** |
| 64 | 10 | 163 840 | 20.5 kB | 0.833 |
| 64 | 20 | 327 680 | 41.0 kB | 1.667 |

这个固定码率设计正是为什么脉冲*发放率*在这里不能省字节——无论其中 1 % 还是 50 %
发放，文件都是 `T·C·h·w` bits——也是为什么后来的路径用真正的熵编码器取代了它。
`--reference` 还会打印同字节数的 JPEG 对比，并在两个码率无法匹配时拒绝比较。

### 6.4 `service/` —— Docker A/B 部署

```powershell
python -m codec.model_io --ckpt anchor_v5.pth --out ./models   # once
docker compose -f service\docker-compose.yml build
docker compose -f service\docker-compose.yml up -d             # → http://localhost:8080
python service\_e2e_test.py --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002 --gw-url http://127.0.0.1:8000
powershell -ExecutionPolicy Bypass -File service\_docker_verify.ps1
```

| 镜像 | 包含 | 模型挂载 | 端口 |
|---|---|---|---|
| `snn-ab-a`（`Dockerfile.a`） | `codec/`、`sae_rd.py`、`sae_model.py`、`anchor.py`、`a_encoder/`——**没有** `b_decoder/` | `models/a:/models:ro` | 无 |
| `snn-ab-b`（`Dockerfile.b`） | 同样的代码加上 `b_decoder/`——**没有** `a_encoder/` | `models/b:/models:ro` | 无 |
| `snn-ab-gateway`（`Dockerfile.gateway`） | 仅 `gateway/`，**没有 torch**（numpy + pillow） | `runs/` | `8080:8000` |

网络：`frontend`（仅网关）和启用 `internal: true` 的 `backend`（A、B、网关），因此
A 和 B 无法访问互联网，也无法从主机访问。端点：`A POST /encode`、
`B POST /decode`（返回带 `X-Model-Hash` 等的 PNG）、网关 `POST /api/run`、
`GET /api/history`、`GET /runs/<id>/{orig,recon}.png`、`GET /health`。PSNR/SSIM
**只在**网关计算，它是唯一同时持有原图和重建的组件。

隔离是被验证的，而不是被断言的——显示 `/models` 每一侧恰好只含一个文件的精确
`docker compose exec` 命令，以及从主机无法访问 8001/8002，都在
[`service/README.md`](service/README.md) §7，记录输出在
`service/evidence/docker_e2e_test.txt` 和 `host_e2e_test.txt`。

---

## 7. 负面结果：跨图像“语义相关性”在这个潜空间中不存在

这是本项目最重要的科学产出，而且它是一个**证伪**。

**想法。** 语义相似的图像（同一花种、同一场景）如果编码其中一张、并把其余的作为
相对于它的残差发送，应该压缩得更好。最初的实现正是这么做的：把相似图像分组，
正常编码第 0 张图像，并给编码器一个参考潜层，使它能够学会对其余图像输出 `h ≈ 0`。

**测试。** 六组、每组八张同类留出图像。对每一对，在五种熵模型配置下，对*同一张
图像*比较 `bits(residual) / bits(independent)`——检查点自身的先验（仅 y，以及交付
时的 y+z）、按实测符号拟合的逐通道高斯，以及两个 **oracle** 模型（逐图经验直方图
和合并经验直方图），它们被给予了真实符号分布和免费的旁信息。oracle 是无法被击败
的，所以如果该方案输给它，错的是方案，而不是熵模型。

**结果——该方案处处落败：**

| 熵模型 | `learned` 中位比值 | `sub`（硬结构残差）中位比值 |
|---|---|---|
| 模型先验，仅 y | 1.023 | 1.492 |
| 模型先验，y+z（**实际交付的**） | 1.075 | 1.474 |
| 拟合的逐通道高斯 | 1.064 | 1.244 |
| **ORACLE** 逐图直方图 | 1.072 | **1.262** |
| **ORACLE** 合并直方图 | 1.081 | 1.263 |

（`anchor_v5` @ 256 px。比值 > 1 意味着锚点方案花费*更多*字节。）

* 在 oracle 计量下，learned 参数化多花 **+4.7 % 到 +58.5 %** 的字节，取决于检查点
  和输入尺寸；硬结构残差在方案层面仍然多花 **+22 % 到 +24 %**（v5 @256 为
  `+22.99 %`，v7 @256 为 `+23.78 %`，v5 @128 为 `+22.25 %`）。
* **42 对同类图像中 100 % 在残差下都更差**（四分位数 1.246 / 1.283；最小 1.213，
  最大 1.360）——每组之间的离散度是 ±5 %，比损失小一个数量级。
* `learned` 参数化还把编码后的潜层膨胀到最多 **10.1×**（v7）。把它换成精确、非学习
  的减法 `coded = quant_hard(y_i) − quant_hard(y_ref)` 修复了该病态（比值降到
  1.30–1.36×），但没有让该方案获胜。

**根因，已测量**（`_diag_residual_corr.py`）：同类图像对的**像素**相关性为 **0.74**，
但它们的**编码器潜层**相关性只有 **0.12–0.33**，而且把一个潜层最佳全局投影到另一个
只能解释 **R² = 0.063**（六个组之间为 0.022–0.112）。一个类外对照得到
R² ≈ 0.01–0.06——几乎没差别。在这个潜空间中，**根本没有可抵消的跨图像共享成分**。
对于去相关的潜层，`std(y_i − y_ref) ≈ √2·std(y) = 1.41×`，而结构残差正好落在
1.30–1.36×，恰好处于该区间。逐通道均值平移编码也省不下任何东西（1.001–1.025×）。

因此编码器产生的是逐图像的纹理编码，而不是语义编码。失败发生在**参数化的上游**：
没有任何残差方案能恢复编码器从未创造出来的相关性。

**交付了什么：** 锚点路径保留在 `--use-anchor`（以及
`--residual-mode learned|sub`）之后，纯粹为了可复现性，并且**默认关闭**。默认路径
是使用 `RandomSampler`、不带参考的普通独立编码——这也意味着训练流水线完全不再
需要类别标签。

完整证据：[`anchor_bytes_report.md`](anchor_bytes_report.md)、
[`design_note_anchor_residual.md`](design_note_anchor_residual.md)、
[`residual_sub_report.md`](residual_sub_report.md)。

---

## 8. 局限

* **率–失真很差。** 最诚实的最佳结果：在 256 px 参考几何下、5 个 epoch 的 10 分钟
  运行后为 **22.50 dB 真实验证**（`models/anchor_best.pth`）。此前最佳模型在 2.845 bpp
  容器计量下达到 21.13 dB。在可比的字节数下 JPEG 仍然领先很多，而且针对它的完整
  BD-rate 扫描至今仍未发表。
  JPEG 在相近字节数的结果上领先 10–15 dB；这里的任何东西都不构成 state-of-the-art
  （最先进）声明。
* **小规模评估。** 字节层面的结论建立在 6–8 组共 42–48 张图像上，而短运行基线是
  在 128 px 下的一次十分钟运行（记录到六个 epoch，最佳 21.01 dB）。证伪是稳健的
  （oracle 模型、逐图配对比较、离散度极小），但这些*正面*性能数字不是一个基准。
* **只有一个码率点是真实的。** `--lam` 旋钮和服务中的 20 点（steps × channels）阶梯
  确实存在，但没有已发表的扫描，没有 BD-rate。
* **瓶颈是量化器的噪声模型**，而不是 SNN 本身：在高码率下重建改善缓慢而码率持续
  攀升（这就是 `anchor_v5.log` 的形状，它跑了 360 个 epoch，最终停在解析式
  3.21 bpp——容器 21.13 dB / 2.845 bpp）。
* **熵模型只被部分学习。** y 先验在预热时做矩拟合，z 先验单独校准
  （`recalibrate_prior.py`）；逐通道离散 MLE 仍能把 z 流从 15 413 降到
  8 971 bits/图（约 42 %），而学习式超先验尚未实现。
* **float32 σ 桶边界的脆弱性**（§5，1 ulp）源于从 σ 导出编码桶而不是传输它。
* **遗留代码仍在。** `compress.py`/`decompress.py`/`sae_model.py` 描述了一个已被
  取代的固定码率设计，保留用于可复现性。服务代码存在两份拷贝（`service/` 是源头，
  根目录的 `a_encoder/`、`b_decoder/`、`gateway/` 是由 `python service\sync.py`
  同步的部署拷贝）。
* **没有许可证文件、没有 CI、没有 tests 目录**——验证脚本
  （`_verify_fix*.py`、`_e2e_test.py`、`_docker_verify.ps1`）就是测试套件。

---

## 9. 目录布局

```
anchor.py            train / eval / allpairs — the current RD path (SAE_Anchor ⊂ SAE_RD)
sae_rd.py            SAE_RD, LIF wiring, GDN/IGDN, quantizer, Gaussian prior, hyperprior,
                     log-domain bin probabilities, sigma grid
codec/model_io.py    split a checkpoint into A (encoder) and B (decoder) halves + manifest
codec/rd_codec.py    the v3 key container: single range-coded stream (z then y), header
compress.py          legacy A-side: fixed-rate key from the encoder half only
decompress.py        legacy B-side: reconstruct from a fixed-rate key, PSNR/SSIM vs JPEG
sae_model.py         legacy model + key v2 format + JPEG comparison helpers
service/             Docker A/B deployment (see service/README.md)
models/              split halves + manifest (produced by codec.model_io)
data/                Flowers102 / DIV2K / Flickr2K (not committed)
anchor_*.log         training logs; anchor_v9.log is the current short run
anchor_bytes*.{py,md,json,txt}   byte-level accounting + the falsification evidence
fix_round_report.md  the six-bug audit, with reproduction commands
_verify_fix*.py      one script per fixed bug
docs/evidence-index.md   every headline number → the file and line it came from
run_v8.ps1           the 256 px reference training launcher (auto-resume)
```

本 README 中每一个数字的来源都在
[`docs/evidence-index.md`](docs/evidence-index.md) 中有索引。

## 10. 复现验证

```powershell
$env:CUDA_VISIBLE_DEVICES = ""      # all of the below are CPU-only and fast
python _verify_fix1.py               # bit accounting, z stream, native input size
python _verify_fix2.py               # dead zone: before/after gradients vs float64
python _verify_fix3.py               # sigma grid coverage on three checkpoints
python _verify_fix4.py               # training-path smoke test + y-prior units
python recalibrate_prior.py --weights anchor_v5.pth --n-images 8 --mle
python _verify_fix5.py               # bit-exact encode/decode round trip
python service\_smoke_codec.py       # the deployed A/B split, end to end
```
