# SNN image compression with an enforced encoder / decoder split

An image compression system built on **spiking neural networks** (`snnTorch`) plus
**arithmetic coding** (`constriction`). A trained SNN **encoder** turns an image into a
tiny arithmetic-coded key; a trained SNN **decoder** turns the key back into the image.

The two halves are meant to live on **two different machines**:

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

Shown at the **256 px, C=32, T=10 reference configuration**. The CLI's own defaults are
smaller (128 px, C=24, T=6) — see the gotcha in §3.3.

A cannot reconstruct (it has no decoder). B cannot encode (it has no encoder and no
`h_a`, the hyperprior analysis transform). This is not a convention — it is enforced by
Docker: two images, one internal network, one read-only model mount each, and a key
header carrying a **model hash** so a mismatched pair fails loudly instead of emitting
garbage. See §6.4 and [`service/README.md`](service/README.md).

**Status in one line:** the pipeline is correct and verified end to end (bit-exact round
trips; ≈21 dB within four epochs of a ten-minute default run), but the *compression* is
not competitive — JPEG wins by a wide margin at these rates, and the project's headline
research idea was rigorously falsified and is off by default (§7). Both facts are
documented here deliberately.

---

## 1. What works / what does not

**Works**

* End-to-end: image → SNN encoder → arithmetic-coded key → SNN decoder → PNG.
* **Bit-exact** encode/decode round trip (max abs difference `0.0`), at 256 px, at
  128 px fed into a 256-trained model, and for *partial* keys (fewer temporal steps /
  channels than the model was trained with).
* The A/B split is real and verified: each container sees only its own half of the
  weights, A and B cannot reach the network, the gateway is the only published port.
* A real-validation protocol (`eval` mode + hard quantization + BN running stats) that
  does not lie to you the way the training loss does, plus checkpoint selection by it.
* A round of auditing found and fixed six genuine bugs — including an entropy model with
  a **zero-gradient dead zone** and an **encoder/decoder sigma mismatch that corrupted
  9.34 % of decoded pixels**. See §5.

**Does not work / does not exist yet**

* **Quality.** The best honest numbers here are ≈21 dB — 21.13 dB at 2.845 bpp container
  accounting (256 px, C=32, T=10), and ≈21 dB at 0.63 bpp analytic on the small default
  geometry (128 px, C=24, T=6). JPEG is 10–15 dB better at a comparable byte count. This
  is not a competitive codec.
* **No rate–distortion sweep as a result.** `--lam` is a knob; nobody has published a
   proper BD-rate curve from it.
* **The anchor / cross-image idea is dead.** Coding a group of similar images relative to
  a reference image costs *more* bytes under every honest entropy model tested. It is
  kept in the code behind `--use-anchor` for reproducibility and is **off by default**.
* **Earlier "wins" were artefacts.** `anchor_v7`'s apparently low bitrate came from an
  entropy model that could not learn sigma (§5, item 2); measured against an oracle it is
  58 % *worse* than independent coding on the anchor path.
* **No learned entropy model over the hyperprior.** The prior is calibrated by moment
  fitting at warm-up, not trained jointly, and the remaining gap to a discrete MLE is
  documented but unimplemented.

---

## 2. Architecture

`SAE_RD` (in `sae_rd.py`), a rate–distortion SNN autoencoder:

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

* **LIF neurons with `T` temporal steps** (default `T = 10` in the reference config).
  The encoder fires spikes at every step; the decoder accumulates `sigmoid(membrane)` and
  averages over time. In an early ablation, time-averaging cut test MSE from 0.0278 (last
  step only) to 0.0165 — the last-step variant is 68 % worse — because averaging
  suppresses spike noise and gives the gradient `T` paths.
* **GDN / IGDN** (generalized divisive normalization) at the latent, from the
  scale-hyperprior literature — it replaces a whitening BatchNorm at the bottleneck and
  keeps the latent's distributional structure.
* **Scale hyperprior (Ballé 2018)**: `h_a` downsamples `y` to `ẑ`; `ẑ` is itself
  arithmetic-coded (the *z stream*); `h_s` maps `ẑ` back to a per-pixel `σ` used as the
  scale of a factorized Gaussian entropy model on `y`. Roughly 30 % of the shipped bytes
  are the z stream, which is why it has to be accounted for (§4).
* **Uniform quantizer with noise annealing + STE**, quantization step fixed at `1.0`:
  the loss depends only on `step/σ`, so letting both float leaves one effective degree of
  freedom and they fight each other.
* **Decoder normalization is BatchNorm by default**, on measured evidence: in a
  controlled A/B (same data, sampler and seed) BatchNorm reached **17.19 dB**, GroupNorm
  **15.54 dB**, and no normalization **collapsed entirely** — the latent died
  (`bpp → 0.0000`, `y_max = 0`, 10.78 dB). Logs: `normab_bn.log`, `normab_gn.log`,
  `normab_none.log`.

The quantized latent is entropy-coded with `constriction`'s range coder: the z stream
first, then the y stream using the σ derived from the decoded `ẑ`. **No probability table
is transmitted** — B recomputes it from its own decoder half. The container header
(`codec/rd_codec.py`, key format v3, 36 bytes) carries the magic `SNNK`, the version and
codec id, `image_size`, the model's native `latent_channels`/`num_steps`, the
`steps_used`/`channels_used` knobs for this particular key, an 8-byte **model hash**, the
alphabet size `K`, and the payload length.

---

## 3. Quick start

### 3.1 Install

```bash
python -m pip install torch torchvision snntorch constriction pillow numpy matplotlib
```

Check whether your PyTorch is a CPU-only build — a silent and very common failure:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.14.0+cpu None False   -> CPU-only wheel: the GPU will not be used
# 2.14.0+cu130 13.0 True  -> OK
```

For a Blackwell (RTX 50-series) card on Python 3.14 the working combination was
`cu130 / torch 2.14.0 / torchvision 0.29.0`; `cu128` has an older torch and `cu129` has
no `cp314` wheel. Install with `--force-reinstall` (or uninstall first), otherwise pip
considers `2.14.0+cpu` already satisfied and does nothing:

```bash
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

`get_arch_list()` must contain `sm_120`, otherwise you will get
`CUDA error: no kernel image is available for execution on the device`.

Everything runs on CPU too (the code falls back automatically), just slower.

### 3.2 Data

| dataset | images | where | notes |
|---|---|---|---|
| Flowers102 | 8189 | `data/flowers-102/` | all three splits (`train`+`val`+`test`) **merged**, then re-split |
| DIV2K + Flickr2K | 3450 | `data/div2k_extracted/` | 2K images, **no class labels**; zips stay in `data/div2k/` |

Combined: **11639** images. 15 % is held out
(`--holdout-frac 0.15` → `max(200, 0.15·n)`): **9894 train / 1745 holdout**; each epoch
evaluates the first `--eval-limit` (default 300) of the holdout. Augmentation
(`--augment crop` = `RandomResizedCrop(0.5–1.0)` + horizontal flip) is applied to the
**training split only**; the holdout always gets plain resize + center crop.

```powershell
# DIV2K download helper (PowerShell), then unzip into data/div2k_extracted/
.\download_div2k.ps1
```

Neither dataset is committed — `data/` and every `*.pth` are in `.gitignore`. Fetch the
datasets separately, and note that no checkpoint ships in the repo; **a trained
checkpoint is provided separately** (release asset). Without one you can still train from
scratch, and `service/_smoke_codec.py` will tell you what is missing.

### 3.3 Train

```bash
python anchor.py train --epochs 30          # the default command (anchor OFF)
```

That single command is the recommended starting point. Its argparse defaults are
`--data both --batch-size 16 --norm bn --augment crop --image-size 128
--latent-channels 24 --num-steps 6 --lam 0.01 --lr 1e-3 --holdout-frac 0.15`, no anchor,
and it writes `anchor_rd.pth` (best-by-real-validation goes to `anchor_rd_best.pth`).

> **Gotcha.** The *architecture* section above describes the 256 px / C=32 / T=10
> reference configuration, which is **not** the argparse default. To get it you must ask
> for it explicitly — this is the configuration used by `run_v8.ps1` (`$Epochs = 330`,
> `$BudgetMin = 570`, auto-resume, `--group-size 16 --anchor-frac 0.5` for the anchor
> path that is now off by default):
>
> ```powershell
> python anchor.py train --image-size 256 --latent-channels 32 --num-steps 10 `
>     --epochs 330 --lam 0.01 --lr 3e-4 --noise-floor 0.2 --workers 8 `
>     --save-every 5 --val-every 5 --val-batches 10 --eval-limit 1228 --out anchor_v8.pth
> ```

Measured cost on an RTX 5070 Ti Laptop, 9894 training images, 618 batches × 16 = 9888
images/epoch:

| config | epoch time | source |
|---|---|---|
| 128 px, C=24, T=6 (defaults) | **51.0 s** | `anchor_v9.log` |
| 256 px, C=32, T=10 (+ real validation) | **~102 s** | `anchor_v8.log` |

### 3.4 Evaluate

```bash
# independent coding vs anchor+residual, per-group byte totals
python anchor.py eval --weights anchor_v5.pth --group-size 4 --n-groups 12

# independent vs star vs minimum-spanning-tree over all pairs in a group
python anchor.py allpairs --weights anchor_v5.pth --group-size 4 --n-groups 12
```

`eval` prints both bits and bytes with unambiguous labels, includes the z stream and the
hyperprior per-pixel sigma, and **loads the checkpoint first so it evaluates at the
model's native resolution** (a 256-trained model used to be silently evaluated on 128 px
data). `--out-json` (default `anchor_eval.json`) gets the same numbers.

### 3.5 Split the model and deploy A/B

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

`h_s` deliberately appears in **both** halves — the encoder needs σ to *encode* and the
decoder needs σ to *decode*, and it is tiny and reveals nothing about the decoder. The
splitter also computes the SHA-256 prefix that becomes the key header's `model_hash`.

Then, with Docker running:

```powershell
docker compose -f service\docker-compose.yml build     # first build ~5–10 min (torch CPU wheel)
docker compose -f service\docker-compose.yml up -d
docker compose -f service\docker-compose.yml ps        # all three healthy
# open http://localhost:8080
```

Host-only alternative (three terminals, no Docker): `python service\_run.py a --port 8001`,
`python service\_run.py b --port 8002`, and
`python service\_run.py gateway --port 8000 --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002`.

Full detail — API reference, the 20-point rate-control ladder, how to verify the
isolation from the shell, and the Docker gotchas — is in [`service/README.md`](service/README.md).

---

## 4. Results, and how to read a bitrate honestly

### 4.0 How a bitrate is counted here (read this first)

There are three different "bpp"s in this repo. Comparing them silently is the easiest way
to fool yourself, so:

* **Training-log `bpp`** (`独立码率 … bpp` in `anchor_*.log`) is an *analytic* estimate
  computed from the entropy model while quantization is still simulated by annealed noise.
  It is not what the coder writes.
* **Container `bpp`** is the real thing: what `codec/rd_codec.py::encode_key` actually
  emits — arithmetic-coded z stream **plus** y stream **plus** the 36-byte header, with an
  alphabet sized to the actual symbols. This is what a deployed A/B pair transmits.
* **The two halves of the repo also count pixels differently.** The RD path
  (`anchor.py`, `anchor_bytes.py`, the container) uses `bits / (H·W)`; the legacy path's
  `sae_model.bpp()` uses `bits / (H·W·3)`, i.e. per *subpixel*. The very same 10 254-byte
  key is **0.417 bpp** in `compress.py` and **1.25 bpp** by the RD convention. Every table
  below states which convention it uses; never compare across them.

For `anchor_v5` at 256 px the two differ by ~13 % (3.21 bpp analytic in the training log
vs 2.845 bpp container). Historical reports in this repo sometimes quoted a PSNR from one
protocol next to a bitrate from the other; the tables below always name the protocol.

### 4.1 Short-run sanity baseline (not a result)

`anchor_v9.log` is a live run of the default command with a 10-minute wall-clock budget
(`--time-budget-min 10 --val-every 1`). The table is a **snapshot of the log, not a final
model** — each epoch is validated for real (eval mode, hard quantization, BN running
statistics, holdout images the model has never seen):

| epoch | real-validation PSNR | analytic rate | sigma | time |
|---|---|---|---|---|
| 1 | 19.77 dB | 0.7559 bpp | 0.661 | 51.0 s |
| 2 | 20.42 dB | 0.7159 bpp | 0.641 | 50.4 s |
| 3 | 20.68 dB | 0.6741 bpp | 0.634 | 48.6 s |
| 4 | **21.01 dB** ← best so far | 0.6283 bpp | 0.632 | 48.1 s |
| 5 | 20.97 dB | 0.6357 bpp | 0.624 | 49.1 s |
| 6 | 20.53 dB | 0.6129 bpp | 0.621 | 49.8 s |

```
ep   4/30 | L 0.0154 | D 0.0006 | 独立码率 0.6283 bpp | sigma 0.632 | 噪声 0.88 | 48.1s
         | y_std 0.86 y_max 9 | ★真实验证 21.01dB
```

Read this as *"the loss is wired up correctly and a ten-minute run already lands near
21 dB"* — a smoke-test baseline. **It is not a final result:** a handful of epochs of the
smallest default configuration, at 128 px rather than the 256 px reference geometry, and
the rate is the analytic estimate from the entropy model, not the container rate that a
deployed pair actually transmits. An earlier launch of the same command reported 19.99 dB
at 0.7671 bpp for epoch 1.

Do **not** line this up against the 21.13 dB in the next table: different resolution,
different latent geometry, different rate protocol.

### 4.2 Reference point: the previous best model

Honest container accounting, 42 same-class held-out images at the model's native 256 px
(`anchor_bytes_report.md`, `anchor_bytes_summary.txt`):

| checkpoint | role | y stream B/img | z stream B/img | total B/img | PSNR | container bpp |
|---|---|---|---|---|---|---|
| `anchor_v5` (BN, C=32, T=10) | independent | 15988.0 | 7281.5 | **23305.5** | **21.13 dB** | **2.845** |
| `anchor_v5` | anchor | 16420.7 | 7602.7 | 24059.3 | 20.76 dB | 2.937 |
| `anchor_v5` | residual | 17875.2 | 8828.2 | 26739.4 | 20.73 dB | 3.264 |
| `normab_gn_best` (GN) | independent | 11472.7 | 953.9 | 12462.6 | 16.83 dB | 1.521 |

So: **21.13 dB at 2.845 bpp.** Note the z stream is ~31 % of those bytes — any accounting
that omits it is wrong, which is precisely the bug fixed in §5, item 1.

> A note on a number that circulated in earlier notes, *"21.13 dB @ 0.948 bpp"*: those two
> figures come from different models and different protocols. 21.13 dB is `anchor_v5`'s
> container PSNR; ≈0.95 bpp is the training-log *analytic* rate of `anchor_v7` (and of the
> GroupNorm ablation), whose entropy model later turned out to be broken. Do not pair them.

### 4.3 Perspective

256×256 JPEG at quality 85 is roughly 10–15 kB. A 10.25 kB fixed-rate key from the
*legacy* path (see §6.3) is therefore a similar bitrate, and at similar byte counts JPEG
is **10–15 dB better** than anything measured here. "94.8 % smaller than raw RGB" is a
strawman and is not quoted as a result anywhere in this README.

---

## 5. Correctness: six measured bugs, found and fixed

A dedicated audit round (`fix_round_report.md`, with per-item `_verify_fixN.py` scripts)
found six defects. Four silently cost rate or correctness, one was deployment-breaking, and
one was a design convention replaced by measurement. Measured before/after:

| # | defect | measured impact |
|---|---|---|
| 1 | `evaluate` printed **bits as bytes** (8×), omitted the z stream and `extra_sigma`, and evaluated 256-trained models on 128 px data | rate overstated **9.96×**; 232 296 "B" → the real **23 320 B** (2.847 bpp); z stream = 31.6 % of bytes |
| 2 | entropy model had a **zero-gradient dead zone**: float32 `erf` saturates around 3.9σ, and `clamp_min(1e-9)` then pinned every symbol past ~5.5σ at exactly **29.8974 bits with gradient exactly 0.0** w.r.t. σ — σ could not learn at all | at `t=y/σ=10`: rate 29.8974 / grad 0.0 → **69.6909 / −131.6**; matches a float64 reference to **7.7e-12 bits**; `anchor_v7`'s analytic y-rate 387 034.8 → 20 249 102 bits/img, i.e. the old model had been *hiding* 98 % of its true cost |
| 3 | `_SIGMA_GRID` covered only [0.01, 100] while effective σ reaches 1e5–1e6 | 48 points over 4 decades (0.085 decade step) → **160 log-spaced points over [1e-4, 1e6]** (0.063 step, finer than before); pixels outside the grid **9.374 % → 0.000 %** (v5), 23.601 % → 0 % (v7), 5.467 % → 0 % (normab_gn, whose y stream dropped −24.71 %) |
| 4 | `z_prior` was never calibrated, and a unit bug took `log(variance)` and treated it as `log σ` | z stream **97 301.9 → 15 413.4 bits/img (−84.2 %)**; since the z stream is ~31 % of the shipped bytes that is **≈26 % off the whole file**, and it is **free at decode time** (no side information). Old σ was 0.0566 against a measured z std of 1.2594 |
| 5 | encoder and decoder disagreed on σ clamping (`±14`/`[1e-4,1e6]` vs `±8`/`[1e-3,1e3]`) and assembled σ differently | **9.34 % of pixels decoded into the wrong σ bucket**; now one shared path and a **bit-exact round trip** (max abs diff `0.0`) at 256 px, at 128 px into a 256 model, and for partial keys |
| 6 | decoder normalization was GroupNorm by convention | controlled A/B: **BatchNorm 17.19 dB / GroupNorm 15.54 dB / none collapses** (`bpp → 0.0000`, `y_max 0`, 10.78 dB) → BatchNorm is now the default |

Deployment-level consequence of #5: `python service\_smoke_codec.py` now loads the split
`models/a/encoder.pth` + `models/b/decoder.pth`, encodes at three rate points
(24 292 / 7 496 / 3 164 B), decodes to 16.26 dB, and correctly **rejects** both a
wrong-hash key and a truncated key.

Residual caveats that are *not* fully fixed, and are stated in the report: bucket ids are
still derived from a float32 σ, so a pixel sitting within 1 ulp of a bucket boundary could
in principle be encoded and decoded differently across stacks (probability ≈1e-7 per
pixel, ≈1 % of 256 px images get one flipped pixel); and the z alphabet default (`32` in
`sae_rd.encode_z_bytes`) still differs from the container's (`128`).

---

## 6. CLI reference

### 6.1 `anchor.py` — the current RD path

```
python anchor.py train    [flags]     # train the RD SNN autoencoder
python anchor.py eval     [flags]     # independent vs anchor+residual byte totals
python anchor.py allpairs [flags]     # independent vs star vs MST optimum
```

Shared flags (all three subcommands):

| flag | default | meaning |
|---|---|---|
| `--data` | `both` | `flowers` (8189, labeled) / `div2k` (3450, unlabeled) / `both` |
| `--norm` | `bn` | decoder normalization: `bn` (BatchNorm, measured best) / `gn` / `none` |
| `--batch-size` | `16` | batch size for the no-anchor (default) path; with `--use-anchor` the group size wins |
| `--image-size` | `128` | input/output edge; **must be a multiple of 16**; the reference config uses `256` |
| `--latent-channels` | `24` | C. The main rate knob: key symbols ∝ C |
| `--num-steps` | `6` | T. SNN time steps; symbols ∝ T and compute ∝ T |
| `--lam` | `0.01` | (`train` only) rate–distortion tradeoff: `L = D + lam · bits/pixel` |
| `--use-anchor` | off | enable the **falsified** anchor+residual path (off by default; §7) |
| `--augment` | `crop` | `none` / `flip` / `crop` (RandomResizedCrop + horizontal flip) on the train split only |
| `--holdout-frac` | `0.15` | fraction of the 11 639 images held out (`max(200, ·)`) |
| `--residual-mode` | `sub` for train, checkpoint value for eval | `sub` = hard structural residual `coded = quant_hard(y_i) − quant_hard(y_ref)`; `learned` = the old learned-residual behaviour, kept for A/B reproduction |

Other flags worth knowing: `--z-channels 48`, `--use-hyperprior` (on; `--no-hyperprior`
turns it off), `--div2k-limit 6000`, `--workers 4`, `--eval-limit 300`,
`--group-size 4` (anchor path only), `--seed 42`, `--weights anchor_rd.pth`,
`--out anchor_rd.pth`, `--out-json anchor_eval.json`, `--n-groups 12`.

`train` only: `--epochs 30`, `--lr 1e-3`, `--grad-clip 1.0`, `--max-batches 0`,
`--hot-start ""`, `--ref-wiring inp|out`, `--ref-mode star|pairs`, `--save-every 5`,
`--resume ""`, `--anchor-frac 0.5`, `--noise-floor 0.2` (annealing floor; dropping it to
0 makes the entropy model disagree with hard quantization and the rate jumps ~75 %),
`--val-every 10`, `--val-batches 4`, `--time-budget-min 0`.

Outputs: `--out` checkpoint plus `<out>_best.pth` (best by **real validation**, not by
training loss), and a per-epoch log line whose `★真实验证 … dB` field is the number to
trust.

### 6.2 `codec.model_io` — the model splitter

```bash
python -m codec.model_io --ckpt anchor_v5.pth --out ./models [--codec-id 3]
```

Writes `models/a/encoder.pth` (encoder half only), `models/b/decoder.pth` (decoder half
only) and `models/manifest.json` (source name, `model_hash`, geometry, tensor counts).
`codec.load_side(ckpt, "a"|"b")` loads one half with `strict=False`, reports how many
tensors it got and how many are missing, and switches the quantizer to hard rounding.
`manifest_of()` reads back the manifest. Same file also exposes `model_hash_of()`.

### 6.3 `compress.py` / `decompress.py` — the legacy fixed-rate path

These are the **older, pre-arithmetic-coding** half of the repo, built on `sae_model.py`
and the `sae_cifar.pth` checkpoint. They load *only* one half of the weights by name
prefix (`build_model_from_checkpoint(..., prefixes=("encoder.",))` /
`prefixes=("decoder.",)`), which is the original — and much weaker — form of the A/B
separation. Use them to reproduce the old fixed-rate behaviour, not for new work.

```bash
python compress.py   --image cat.jpg --output key.pt [--format bits|float] [--save-input32]
python decompress.py --key key.pt --output recon.png [--reference cat.jpg] [--compare]
```

Key size is **fixed and content-independent**: `T × C × (N/16)²` bits, plus a 14-byte
header (magic `SNNK`, version 2, flags, `T`, `C`, `h`, `w`). The `bpp` column below uses
the legacy `sae_model.bpp()` convention — `bits / (H·W·3)`, per subpixel. Divide by 3 to
convert to the RD convention used in §4.1–§4.2.

| C | T | key bits | key file | bpp |
|---|---|---|---|---|
| 8 | 10 | 20 480 | 2.57 kB | 0.104 |
| 16 | 10 | 40 960 | 5.14 kB | 0.208 |
| **32** | **10** | **81 920** | **10.25 kB** | **0.417** |
| 64 | 10 | 163 840 | 20.5 kB | 0.833 |
| 64 | 20 | 327 680 | 41.0 kB | 1.667 |

This fixed-rate design is why spike *rate* does not save bytes here — the file is
`T·C·h·w` bits whether 1 % or 50 % of them fire — and why the later path replaced it with
a real entropy coder. `--reference` also prints a same-byte-count JPEG comparison, and
refuses to compare when the two rates cannot be matched.

### 6.4 `service/` — the Docker A/B deployment

```powershell
python -m codec.model_io --ckpt anchor_v5.pth --out ./models   # once
docker compose -f service\docker-compose.yml build
docker compose -f service\docker-compose.yml up -d             # → http://localhost:8080
python service\_e2e_test.py --a-url http://127.0.0.1:8001 --b-url http://127.0.0.1:8002 --gw-url http://127.0.0.1:8000
powershell -ExecutionPolicy Bypass -File service\_docker_verify.ps1
```

| image | contains | model mount | ports |
|---|---|---|---|
| `snn-ab-a` (`Dockerfile.a`) | `codec/`, `sae_rd.py`, `sae_model.py`, `anchor.py`, `a_encoder/` — **no** `b_decoder/` | `models/a:/models:ro` | none |
| `snn-ab-b` (`Dockerfile.b`) | the same code plus `b_decoder/` — **no** `a_encoder/` | `models/b:/models:ro` | none |
| `snn-ab-gateway` (`Dockerfile.gateway`) | `gateway/` only, **no torch** (numpy + pillow) | `runs/` | `8080:8000` |

Networks: `frontend` (gateway only) and `backend` with `internal: true` (A, B, gateway),
so A and B cannot reach the internet and cannot be reached from the host. Endpoints:
`A POST /encode`, `B POST /decode` (returns PNG with `X-Model-Hash` etc.), gateway
`POST /api/run`, `GET /api/history`, `GET /runs/<id>/{orig,recon}.png`, `GET /health`.
PSNR/SSIM are computed **only** in the gateway, the single component that holds both the
original and the reconstruction.

Isolation is verified, not asserted — the exact `docker compose exec` commands that show
`/models` containing exactly one file per side, and 8001/8002 being unreachable from the
host, are in [`service/README.md`](service/README.md) §7 with the recorded
output in `service/evidence/docker_e2e_test.txt` and `host_e2e_test.txt`.

---

## 7. The negative result: cross-image "semantic correlation" does not exist in this latent space

This is the project's most substantial scientific output, and it is a **falsification**.

**The idea.** Images that are semantically similar (same flower species, same scene) should
compress better if you code one of them and send the others as residuals against it. The
original implementation did exactly that: group similar images, code image 0 normally, and
give the encoder a reference latent so it could learn to emit `h ≈ 0` for the rest.

**The test.** Six groups of eight same-class held-out images. For every pair, compare
`bits(residual) / bits(independent)` for *the same image*, under five entropy-model
configurations — the checkpoint's own prior (y only, and y+z as shipped), a per-channel
Gaussian fitted to the measured symbols, and two **oracle** models (per-image and pooled
empirical histograms) that are given the true symbol distribution and free side
information. An oracle cannot be beaten, so if the scheme loses to it, the scheme is wrong,
not the entropy model.

**Result — the scheme loses, everywhere:**

| entropy model | `learned` median ratio | `sub` (hard structural residual) median ratio |
|---|---|---|
| model prior, y only | 1.023 | 1.492 |
| model prior, y+z (**what ships**) | 1.075 | 1.474 |
| fitted per-channel Gaussian | 1.064 | 1.244 |
| **ORACLE** per-image histogram | 1.072 | **1.262** |
| **ORACLE** pooled histogram | 1.081 | 1.263 |

(`anchor_v5` @ 256 px. Ratios > 1 mean the anchor scheme costs *more* bytes.)

* Under oracle accounting the learned parameterisation costs **+4.7 % to +58.5 %** more
  bytes depending on checkpoint and input size; the hard structural residual still costs
  **+22 % to +24 %** at the scheme level (`+22.99 %` for v5 @256, `+23.78 %` for v7 @256,
  `+22.25 %` for v5 @128).
* **100 % of the 42 same-class pairs are worse** under the residual (quartiles
  1.246 / 1.283; min 1.213, max 1.360) — the per-group spread is ±5 %, an order of
  magnitude smaller than the loss.
* The `learned` parameterisation also inflated the coded latent by up to **10.1×** (v7).
  Replacing it with an exact, un-learned subtraction `coded = quant_hard(y_i) −
  quant_hard(y_ref)` fixed that pathology (ratio down to 1.30–1.36×) but did not make the
  scheme win.

**Root cause, measured** (`_diag_residual_corr.py`): same-class image pairs have **pixel**
correlation **0.74**, but their **encoder-latent** correlation is only **0.12–0.33**, and
the best global projection of one latent onto the other explains **R² = 0.063**
(0.022–0.112 across six groups). An out-of-class control gets R² ≈ 0.01–0.06 — barely
different. There is simply **no shared inter-image component in this latent space to
cancel**. For decorrelated latents, `std(y_i − y_ref) ≈ √2·std(y) = 1.41×`, and the
structural residual lands at 1.30–1.36×, exactly in that regime. A per-channel mean-shift
code saves nothing either (1.001–1.025×).

The encoder therefore produces a per-image texture code, not a semantic code. The failure
is **upstream of the parameterisation**: no residual scheme can recover correlation that
the encoder never created.

**What shipped:** the anchor path is retained behind `--use-anchor` (and
`--residual-mode learned|sub`) purely for reproducibility, and is **off by default**. The
default path is plain independent coding with a `RandomSampler` and no reference — which
also means the training pipeline no longer needs class labels at all.

Full evidence: [`anchor_bytes_report.md`](anchor_bytes_report.md),
[`design_note_anchor_residual.md`](design_note_anchor_residual.md),
[`residual_sub_report.md`](residual_sub_report.md).

---

## 8. Limitations

* **Rate–distortion is poor.** Best honest results: **21.13 dB @ 2.845 bpp** container
  accounting at 256 px, and ≈21 dB at 0.63 bpp (analytic) on the small default geometry.
  JPEG beats comparable-byte-count results by 10–15 dB; nothing here is a state-of-the-art
  claim.
* **Small-scale evaluation.** Byte-level conclusions rest on 42–48 images in 6–8 groups,
  and the short-run baseline is a ten-minute run at 128 px (six epochs logged, best
  21.01 dB). The falsification is
  robust (oracle models, paired per-image comparison, tiny spread) but the *positive*
  performance numbers are not a benchmark.
* **Only one rate point is real.** The `--lam` knob and the service's 20-point
  (steps × channels) ladder exist, but no published sweep, no BD-rate.
* **The bottleneck is the quantizer's noise model**, not the SNN per se: at high rates the
  reconstruction improves slowly while the rate keeps climbing (that is the shape of
  `anchor_v5.log`, which runs 360 epochs and ends at 3.21 bpp analytic — 21.13 dB /
  2.845 bpp container).
* **Entropy models are only partly learned.** The y prior is moment-fitted at warm-up and
  the z prior is calibrated separately (`recalibrate_prior.py`); a per-channel discrete MLE
  would still take the z stream from 15 413 to 8 971 bits/img (~42 %), and a learned
  hyperprior is not implemented.
* **Float32 σ bucket boundary fragility** (§5, 1 ulp) is inherent to deriving the coding
  bucket from σ instead of transmitting it.
* **Legacy code remains.** `compress.py`/`decompress.py`/`sae_model.py` describe a
  superseded fixed-rate design and are kept for reproducibility. Two service-code copies
  exist (`service/` is the source, the root `a_encoder/`, `b_decoder/`, `gateway/` are
  deployment copies synced by `python service\sync.py`).
* **No licence file, no CI, no tests directory** — the verification scripts
  (`_verify_fix*.py`, `_e2e_test.py`, `_docker_verify.ps1`) are the test suite.

---

## 9. Layout

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

Everything a number in this README came from is indexed in
[`docs/evidence-index.md`](docs/evidence-index.md).

## 10. Reproducing the verification

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
