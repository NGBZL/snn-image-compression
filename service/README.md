# SNN 图像压缩 · A/B 机分离服务

把 `snn_ab` 里的 SNN 图像压缩模型拆成两个**互不信任**的半区，用 HTTP 串成一条链路：

```
                    ┌──────────────────────────────────────────┐
   浏览器  ────────▶ │  gateway（唯一对外入口，不持有模型）        │
  :8080              │  /  /api/run  /api/history  /health       │
                    └───────┬───────────────────────┬──────────┘
                            │ POST /encode          │ POST /decode
                            ▼                       ▼
                 ┌────────────────────┐   ┌────────────────────┐
                 │ a-encoder（边缘端） │   │ b-decoder（云侧）   │
                 │ 只有 encoder 半区   │   │ 只有 decoder 半区   │
                 │ 2 核 / 无 GPU / 只读│   │ 只读 / 可加 GPU     │
                 └────────────────────┘   └────────────────────┘
                      返回「钥匙」字节串  ──────────▶  还原成 PNG
```

- **A 机**拿不到解码器，所以它无论如何重建不出图像；
- **B 机**拿不到编码器、也没有 hyperprior 的 `h_a`，所以它无法自己造钥匙；
- 钥匙里带 **model_hash 指纹**，两边模型配不上时 B 机直接报 400，而不是静默吐一张乱码；
- `PSNR / SSIM` 只能在 Gateway 算 —— 只有它同时握有原图和重建图。

---

## 1. 目录结构

```
snn_ab/
├── codec/                      # 已有：编解码核心（未改动）
├── sae_rd.py  anchor.py        # 已有：模型定义（未改动）
├── models/                     # 切分产物：a/encoder.pth  b/decoder.pth  manifest.json
├── runs/                       # 运行记录：<run_id>/{orig,recon}.png key.bin meta.json + index.jsonl
├── a_encoder/app.py            # A 机服务（部署副本）
├── b_decoder/app.py            # B 机服务（部署副本）
├── gateway/app.py              # Gateway（部署副本）
├── gateway/static/index.html   # 单文件前端（零构建）
├── .dockerignore
└── service/                    # ★ 源码目录 + Docker + 测试脚本
    ├── a_encoder/app.py        #   A 机服务源码
    ├── b_decoder/app.py        #   B 机服务源码
    ├── gateway/app.py          #   Gateway 源码
    ├── gateway/static/index.html
    ├── Dockerfile.a  Dockerfile.b  Dockerfile.gateway
    ├── docker-compose.yml
    ├── sync.py                 #   源码 -> 根目录部署副本
    ├── _run.py                 #   宿主机直测启动器（把 snn_ab 塞进 sys.path）
    ├── _smoke_codec.py         #   不经 HTTP，直接验 codec 链路
    ├── _calib_knobs.py         #   量 (steps, channels) 阶梯的钥匙大小
    ├── _e2e_test.py            #   端到端验收（宿主机 / Docker 同一份）
    ├── _docker_verify.ps1      #   Docker 验收（临时容器进 backend 网络）
    ├── _show_history.py        #   拉一次 /api/history 打成表
    ├── _probe_registries.py    #   探当前网络能连哪些 Docker 镜像源
    ├── _probe_mirror.py        #   验某个镜像源能不能真的拉到基础镜像
    └── evidence/               #   实测输出留档（host_e2e_test.txt / docker_e2e_test.txt）
```

> **为什么服务代码有两份？** Docker 构建上下文必须是 `snn_ab` 根目录（要 `COPY codec/ sae_rd.py anchor.py`），
> 而 `COPY` 只能看到上下文内的路径；宿主机直测也统一从根目录启动最稳。
> 改完 `service/` 下的源码后跑一次 `python service\sync.py` 即可同步。

---

## 2. 快速开始

### 2.1 准备模型半区（只需一次）

```powershell
cd C:\Users\chengyu\order-tracking\snn_ab
python -m codec.model_io --ckpt anchor_v5.pth --out ./models
```

输出：

```
模型指纹 1a5f6470c5592f19
A 机 encoder.pth : 36 个参数张量
B 机 decoder.pth : 36 个参数张量
A 独有 28 个（B 拿不到），例如 ['bn1.bias', 'bn1.num_batches_tracked', 'bn1.running_mean']
B 独有 28 个（A 拿不到），例如 ['bn_d1.bias', 'bn_d1.num_batches_tracked', 'bn_d1.running_mean']
已写入 ./models/manifest.json
```

> PowerShell 里如果看到 `[exit code: 1]`，那是 `python -m` 往 stderr 打的
> `RuntimeWarning: 'codec.model_io' found in sys.modules ...` 被 PowerShell 当成了错误，
> 不影响结果 —— 看 `models/` 里的文件是否生成即可。

### 2.2 方式一：Docker（推荐）

```powershell
cd C:\Users\chengyu\order-tracking\snn_ab
docker compose -f service\docker-compose.yml build      # 首次约 5~10 分钟（拉 torch CPU 轮子）
docker compose -f service\docker-compose.yml up -d
docker compose -f service\docker-compose.yml ps
```

打开 **http://localhost:8080** 。

> **本机注意事项（实测遇到的，换台机器可能不需要）**
>
> 1. **Docker Desktop 要先手动启动。** 只装了 CLI 不会自动起引擎：
>    `Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"`
>    然后等 `docker info` 能返回 ServerVersion。
> 2. **`docker.io` 拉不动。** 本机 `registry-1.docker.io` TCP 443 直接超时
>    （`service\_probe_registries.py` 有探测输出），但 PyPI 和 download.pytorch.org 都通。
>    所以基础镜像要走代理拉，再 local tag 成 Dockerfile 里写的名字：
>    ```powershell
>    docker pull docker.m.daocloud.io/library/python:3.11-slim
>    docker tag  docker.m.daocloud.io/library/python:3.11-slim python:3.11-slim
>    ```
>    镜像 tag 出来之后 `docker compose build` 就不再需要访问 Docker Hub 了
>    （torch 等依赖是从 download.pytorch.org / pypi.org 直接下的，本来就通）。
>    如果换了能直连 Hub 的网络，这步可以跳过。

### 2.3 方式二：宿主机直跑（不装 Docker 也能验）

三个终端（都在 `snn_ab` 根目录）：

```powershell
# 终端 1：A 机
python service\_run.py a       --port 8001

# 终端 2：B 机
python service\_run.py b       --port 8002

# 终端 3：Gateway
python service\_run.py gateway --port 8000 --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002
```

> `_run.py` 做了一件关键的事：把 `C:\Users\chengyu\order-tracking\snn_ab` 插到 `sys.path` 最前面。
> 直接 `uvicorn app:app` 会因为 import 不到 `codec` / `sae_rd` / `anchor` 而失败。

然后打开 **http://127.0.0.1:8000** 。

---

## 3. 接口

### A 机 `POST /encode?target_kb=<float,可选>`

`multipart/form-data`，字段名 `image`。返回：

```json
{
  "key_b64": "...", "key_bytes": 5276, "bpp": 0.644,
  "steps_used": 3, "channels_used": 24,
  "model_hash": "1a5f6470c5592f19", "encode_ms": 70.7, "image_size": 256
}
```

实现要点：

| 项 | 做法 |
|---|---|
| 加载 | `codec.load_side(ENC_CKPT, "a", device)`，启动时加载一次（lifespan） |
| 读图 | `Image.open(...).convert("RGB")` → `Resize(256)` → `CenterCrop(256)` → `ToTensor()` → `unsqueeze(0)` |
| `model_hash` | 从 `manifest.json` 读（容器里 `/models` 只有 `encoder.pth`，所以 manifest 单文件挂进来），启动时缓存 |
| `target_kb` 映射 | 见下方「码率控制」 |
| 不传 `target_kb` | 直接 `steps=10, channels=32`（满配） |

### B 机 `POST /decode`

`application/json`：`{"key_b64": "..."}` → `200 image/png`，附响应头
`X-Model-Hash` / `X-Decode-Ms` / `X-Key-Bytes` / `X-Steps-Used` / `X-Channels-Used`（另有 `X-Image-Size`）。
错误一律 `400` + `{"detail": "..."}`。

### Gateway

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单文件前端 `static/index.html` |
| POST | `/api/run` | `multipart(image, target_kb)`，内部先 A 后 B，返回 `{key_bytes, bpp, steps_used, channels_used, model_hash, encode_ms, decode_ms, psnr, ssim, run_id, recon_url, orig_url, ...}` |
| GET | `/runs/<run_id>/orig.png` | Gateway 侧 256×256 原图 |
| GET | `/runs/<run_id>/recon.png` | B 机重建图 |
| GET | `/runs/<run_id>/key.bin` | 那把钥匙（附加，方便离线复现） |
| GET | `/api/history?limit=50` | 最近记录，从 `runs/index.jsonl` 读，新的在前 |
| GET | `/health` | Gateway + A + B 的健康状态 |

---

## 4. 码率控制（A 机内部）

候选格子：`steps ∈ {10,5,3,2,1}` × `channels ∈ {32,24,16,8}`，共 20 组。
实测（`anchor_v5` + Flowers102 三张图，见 `service/_calib_knobs.py`）各格子钥匙大小：

| kB | 1.20 | 1.62 | 2.03 | 2.04 | 2.50 | 2.81 | 3.03 | 3.57 | 4.20 | 4.44 |
|---|---|---|---|---|---|---|---|---|---|---|
| (steps,ch) | 1,8 | 1,16 | 2,8 | 1,24 | 1,32 | 2,16 | 3,8 | 2,24 | 3,16 | 2,32 |

| kB | 4.88 | 5.34 | 6.64 | 6.80 | 8.70 | 9.30 | 10.84 | 13.06 | 16.81 | 21.04 |
|---|---|---|---|---|---|---|---|---|---|---|
| (steps,ch) | 5,8 | 3,24 | 3,32 | 5,16 | 5,24 | 10,8 | 5,32 | 10,16 | 10,24 | 10,32 |

**注意阶梯不是按 `steps × channels` 单调的**（`(3,32)=6.64 kB` < `(5,16)=6.80 kB`），
所以代码里用的是一张实测表（`LADDER` / `LADDER_KB`），不是乘法公式。

算法：取实测大小离 `target_kb` 最近的 4 个格子（**最多 4 次试探**），
从大到小真跑，第一个 `len(key) <= target` 就停 —— 在这个窗口里它就是最接近的。
同一 `target_kb`（量化到 0.1 kB）再请求会直接复用上次选定的旋钮，不再试探。
换模型或换数据分布后，重跑 `python service\_calib_knobs.py` 更新那张表即可。

实测误差：目标 2 kB → 1.94 kB（2.9%）；5 kB → 5.15 kB（3.0%）；10 kB → 10.53 kB（5.3%）。

---

## 5. 质量指标口径

只有 Gateway 同时有原图和重建图，所以 PSNR/SSIM 在这里算（纯 numpy，无 scipy/skimage）：

- **PSNR**：RGB 三通道 MSE，`uint8`，峰值 255。`MSE<=1e-12` 时返回 99.0。
- **SSIM**：**简化实现** —— 灰度图 + 11×11 高斯窗（σ=1.5）+ 标准 `C1=(0.01·255)²`、`C2=(0.03·255)²`，
  用 `np.convolve` 做两次一维卷积。与 `skimage` 的 `gaussian_weights=True` 口径接近，
  但**没有**做样本协方差的无偏修正（`use_sample_covariance` 那份 `1/(N-1)` 归一）。
  横向比较不同码率够用，**不要当论文指标**。
- Gateway 侧参考图用 PIL 预处理（`RGB → Resize(256) → CenterCrop(256)`，LANCZOS），
  与 A 机的 torchvision 预处理在数值上差一点点，对 PSNR 的影响在小数点后第二位
  （实测同一张图：PIL 参考 13.46 dB / 20.86 kB 时 14.20 dB，torchvision 参考 13.81 / 14.61 dB）。
  这样做是为了**不让 Gateway 镜像拖进 2 GB 的 torch**。

> `anchor_v5` 本身质量不佳，满配也就 **13~15 dB** —— 这是模型的问题，不是链路的 bug。

---

## 6. Docker 部署细节

### 6.1 镜像

| 服务 | Dockerfile | 内容 |
|---|---|---|
| `snn-ab-a` | `service/Dockerfile.a` | `codec/` + `sae_rd.py` + `anchor.py` + `a_encoder/`（**没有** `b_decoder/`） |
| `snn-ab-b` | `service/Dockerfile.b` | `codec/` + `sae_rd.py` + `anchor.py` + `b_decoder/`（**没有** `a_encoder/`） |
| `snn-ab-gateway` | `service/Dockerfile.gateway` | 只有 `gateway/`，**不装 torch**（numpy+pillow 足够） |

三个都基于 `python:3.11-slim`；A/B 用
`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`
装 CPU 版，其余（`numpy snntorch constriction pillow fastapi uvicorn python-multipart httpx`）走 PyPI。

### 6.2 隔离与资源

| | a-encoder | b-decoder | gateway |
|---|---|---|---|
| 挂载 | `../models/a:/models:ro`（`manifest.json` 构建时 COPY 进 `/app/manifest.json`） | `../models/b:/models:ro`（同上） | `../runs:/app/runs` |
| CPU | `cpus: "2"`，`mem_limit: 2g` | 不限 | 不限 |
| GPU | 无（模拟边缘端） | 无（本地无 GPU 可分配；见下方注释） | 无 |
| 端口 | 不对外 | 不对外 | `8080:8000` |
| `restart` | `unless-stopped` | `unless-stopped` | `unless-stopped` |

B 机真实部署时上 GPU：把 `DEVICE` 改成 `cuda`、换成 CUDA 版 torch 镜像，并在 compose 里加

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

（本地模拟不加，因为没有真 GPU 可分；这段注释也写在 `docker-compose.yml` 里。）

### 6.3 网络

```yaml
networks:
  frontend: { driver: bridge }
  backend:  { driver: bridge, internal: true }
```

- `frontend`：只有 `gateway` 接入，宿主机只能通过 `8080` 摸到 Gateway。
- `backend`（`internal: true`）：`a-encoder` / `b-decoder` / `gateway` 三个都在里面。
  internal 网络没有对外网关，A、B 不能主动访问外网，也不会被宿主机直接访问到。

---

## 7. 怎么验证隔离

> 前提：`docker compose -f service\docker-compose.yml up -d` 已经起来，三个容器都 healthy。
> `$dc` 只是为了让下面几行短一点。

```powershell
cd C:\Users\chengyu\order-tracking\snn_ab
$dc = "docker compose -f service\docker-compose.yml"

# 1) A 机只能看到 encoder.pth
Invoke-Expression "$dc exec -T a-encoder ls /models"
#   -> encoder.pth          （**没有** decoder.pth，也没有 manifest.json）

# 2) B 机只能看到 decoder.pth
Invoke-Expression "$dc exec -T b-decoder ls /models"
#   -> decoder.pth

# 3) A 机拿不到 decoder.pth
Invoke-Expression "$dc exec -T a-encoder python -c `"import os;print(os.path.exists('/models/decoder.pth'))`""
#   -> False

# 4) B 机拿不到 encoder.pth
Invoke-Expression "$dc exec -T b-decoder python -c `"import os;print(os.path.exists('/models/encoder.pth'))`""
#   -> False

# 5) A 机里没有解码器代码 / B 机里没有编码器代码
Invoke-Expression "$dc exec -T a-encoder python -c `"import os;print(os.path.exists('/app/b_decoder'))`""
#   -> False
Invoke-Expression "$dc exec -T b-decoder python -c `"import os;print(os.path.exists('/app/a_encoder'))`""
#   -> False

# 6) A/B 都没有发布端口：从宿主机直接访问它们的 8000 应该连不上
curl.exe -s -o NUL -w "%{http_code}`n" http://localhost:8001/health   # 连不上
#   只有 gateway 的 8080 通
```

一条命令跑完全部功能验收：

```powershell
# 宿主机直跑（A/B 在 8001/8002，Gateway 在 8000）
python service\_e2e_test.py --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002 --gw-url http://127.0.0.1:8000

# Docker：A/B **没有**发布端口，宿主机打不到它们，所以要跑在 backend 网络内部
powershell -ExecutionPolicy Bypass -File service\_docker_verify.ps1
```

`_docker_verify.ps1` 做的事：

1. 用 `snn-ab-gateway:local` 镜像起一个**一次性容器**，接进 `snn-ab_backend` 网络，
   在里面用容器名 `a-encoder:8000` / `b-decoder:8000` / `gateway:8000` 跑同一份
   `_e2e_test.py`（这样既跑到了 A/B 的裸接口，又不需要给它们发布端口）；
2. 从宿主机打 `http://localhost:8080/`，确认能拿到前端 HTML；
3. 从宿主机打 8001/8002，确认**连不上**（隔离性的一部分）。

> 实测输出留档在 `service/evidence/docker_e2e_test.txt`。
>
> 另有一个 `service/docker-compose.test.yml` 想把 A/B 的端口临时映射出来，
> 但在本机 Docker Desktop 上 `up` 之后端口代理没真的绑上
> （`docker inspect` 里 `HostConfig.PortBindings` 有、`NetworkSettings.Ports` 是空的）。
> 所以正式的验证路径是上面的"临时容器 + backend 网络"，那个不依赖端口映射。
> 那个 override 文件保留着，在正常环境里应该可用。

---

## 8. 换模型

1. 把新的 checkpoint 放到 `snn_ab\` 下；
2. 重新切分：`python -m codec.model_io --ckpt 新权重.pth --out ./models`；
3. **重跑标定**：`python service\_calib_knobs.py`，把输出的那张表更新到
   `service/a_encoder/app.py` 的 `LADDER` / `LADDER_KB`，并 `python service\sync.py`；
4. 重建镜像：`docker compose -f service\docker-compose.yml build`；
5. 重启：`docker compose -f service\docker-compose.yml up -d`。

`manifest.json` 里的 `model_hash` 会自动变成新权重的前 8 字节 SHA-256，
A 机写进钥匙头、B 机拿它做校验，**两边不一致时 B 机会明确 400**，不会静默出乱码。
所以换模型只需要保证 `models/a`、`models/b`、`manifest.json` 是**同一次切分**的产物。

---

## 9. 实测结果（本机跑出来的真实数字）

模型 `anchor_v5`，测试图 `data/flowers-102/jpg/image_00042.jpg`（58288 B）。
两个环境用同一份 `_e2e_test.py`，**99 项检查全过**。

| 指标 | 宿主机直跑 | Docker 容器 |
|---|---|---|
| 满配 (10×32) 钥匙 | 20864 B（bpp 2.5469） | 20864 B（bpp 2.5469） |
| 满配 PSNR / SSIM | 14.20 dB / 0.2685 | 14.20 dB / 0.2685 |
| 5 kB 目标 → 实得 | 5276 B（3×24，误差 3.0%） | 5276 B（3×24，误差 3.0%） |
| 5 kB 的 PSNR / SSIM | 13.46 dB / 0.2278 | 13.46 dB / 0.2278 |
| 2 kB 目标 → 实得 | 1988 B（1×24，误差 2.9%） | 1988 B（1×24，误差 2.9%） |
| 10 kB 目标 → 实得 | 10780 B（5×32，误差 5.3%） | 10780 B（5×32，误差 5.3%） |
| 编码 / 解码耗时 | 75~89 ms / 30~55 ms | 91~121 ms / 16~46 ms |
| `/api/run` 往返 | 159~197 ms | 176~207 ms |

**两边数字完全一致**（同一个 torch 版本、同一份权重、同样的前处理），
说明容器化没有引入任何数值偏差。

`anchor_v5` 满配也只有 14.2 dB —— 模型本身质量就这样，**不是链路的 bug**。

## 10. 踩过的坑（都已修好，记下来免得再踩）

| 现象 | 根因 | 处理 |
|---|---|---|
| A/B 容器起不来，`ModuleNotFoundError: No module named 'codec'` | uvicorn 的 cwd 是 `/app/a_encoder`，`/app` 不在 `sys.path` 里 | Dockerfile 里加 `PYTHONPATH=/app` |
| `No module named 'sae_model'` | `anchor.py` 顶部 `from sae_model import human_bytes`，只拷 `anchor.py` 不够 | `COPY sae_rd.py sae_model.py anchor.py ./` |
| `error mounting .../manifest.json ... read-only file system` | `/models` 是只读挂载点，Docker 没法在里面创建单文件挂载点 | 改成构建时 `COPY models/manifest.json ./manifest.json`（只是元数据，不含权重） |
| `failed to calculate checksum ... /models/manifest.json: not found` | `.dockerignore` 里的 `models/` 把 manifest 也挡了 | 精确写成 `models/a/` 和 `models/b/`，不要写 `models/` |
| 容器 "healthy" 但模型根本没加载 | healthcheck 只查 HTTP 200，而服务是懒报错的（加载失败也起 HTTP） | healthcheck 改成断言 `/health` 里的 `loaded==true` |
| `docker compose build` 卡在 `resolve image config for docker/dockerfile:1` | 本机连不上 Docker Hub，而 `# syntax=` 指令要额外拉一个 Hub 镜像 | 删掉三个 Dockerfile 的 `# syntax=docker/dockerfile:1`；基础镜像走代理拉再 local tag |
| 同一份 `runs/index.jsonl` 里时间戳互相矛盾 | 宿主机写本地时间、容器写 UTC，不带时区 | 时间戳改成带时区的 ISO 格式，另存一个 `ts_utc` |
| 前端的 PSNR 和宿主机直测差 0.3~0.4 dB | Gateway 用 PIL 预处理、A 机用 torchvision，细节略有差异 | 可接受（重建误差 0.2 量级远大于这点差异），已在 `/api/run` 返回的 `metric_note` 里标注口径 |

## 11. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'codec'` | 宿主机直跑没走 `service\_run.py`。用 `_run.py`，或自己 `sys.path.insert(0, r"...\snn_ab")`。容器里则是少了 `PYTHONPATH=/app` |
| A 机 `/health` 里 `model_hash` 是 `null` | 没读到 manifest。容器里它由 Dockerfile `COPY` 成 `/app/manifest.json`，检查 `MANIFEST_PATH` 和构建时 `models/manifest.json` 是否存在 |
| B 机返回 400「钥匙的模型指纹 … 与当前模型 … 不一致」 | 钥匙和 B 机模型不是同一次切分的产物，重新切分并把两边都换成新产物 |
| PSNR 只有 13~15 dB | `anchor_v5` 本身质量就差，属正常；换成训练更好的权重即可 |
| 满配钥匙 21 kB，比预期大 | z（边信息）也要真的编码，小 steps 时占比很高。看第 4 节的实测表 |
| `docker compose build` 拉 torch 很慢 | torch CPU 轮子约 200 MB，首次构建 5~10 分钟；可配镜像加速或先 `docker pull python:3.11-slim` |
| PowerShell 里 `docker compose exec ... python -c "..."` 引号被吃 | 用 `Invoke-Expression "..."` 包一层，或写成 `.py` 文件再 exec（`_docker_verify.ps1` 就是这么干的） |
| Windows PowerShell 5.1 报「字符串缺少终止符」之类语法错，但文件看着没问题 | `.ps1` 里写了中文却没存成**带 BOM 的 UTF-8**，PS5.1 会按 ANSI 读，中文变乱码后把引号吃掉了。存文件时加 BOM（`service\_docker_verify.ps1` 已加） |
| A/B 发布了端口但宿主机连不上 | 先确认端口没被别的进程占着；本机 Docker Desktop 出现过 `HostConfig.PortBindings` 有值但 `NetworkSettings.Ports` 为空的情况，用 `_docker_verify.ps1` 那条路绕开即可（正式部署 A/B 本来就不发布端口） |
