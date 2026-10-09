# mode (a) implemented: hard structural residual (`--residual-mode sub`) — and falsified

**Status: implemented, round-trip verified, falsification test run. No training was done.**
The structural residual is now algebra instead of a learned behaviour, and it does fix the
train/deploy mismatch and the "reference inflates the latent 10.1x" pathology — but
**it does not make the residual cheaper**. The design note's own falsification criterion is
met: the ORACLE ratio is still `> 1.0` with a margin far larger than the per-group spread.
The recommendation therefore falls through to option (c) for these checkpoints.

Everything below is measured on the held-out split of `anchor.build_data`, 6 groups x 8
same-class images, on the existing checkpoints. Commands at the end.

---

## 1. What changed

| file | function | change |
|---|---|---|
| `anchor.py` | `SAE_Anchor.__init__` | new `residual_mode` in `{"sub","learned"}`; validated; stored on the model and written into checkpoints |
| `anchor.py` | `SAE_Anchor._prepare_input` | now takes a `ref_hint`; in `sub` the caller always passes `None`, so the reference channels are a constant zero block |
| `anchor.py` | `SAE_Anchor.encode_raw` | in `sub` the reference hint is **discarded** (`_prepare_input(x, None)`), the encoder never sees the anchor; in `learned` the old hint path is unchanged |
| `anchor.py` | `SAE_Anchor.code_residual` | **the single subtraction**: `quant_hard(y_i) - y_ref_hat`; `None` -> `quant_hard(y_i)` |
| `anchor.py` | `SAE_Anchor.reconstruct` | **the single addition**: `y_ref_hat + coded_hat`; `None` -> `coded_hat` |
| `anchor.py` | `SAE_Anchor.quant_hard` | the single hard-quantisation definition, `round(z/s)*s + (z - z.detach())` (straight-through) |
| `anchor.py` | `SAE_Anchor._code_and_rate` | hyperprior / prior / z stream factored out and applied to `coded` |
| `anchor.py` | `SAE_Anchor.forward` | `coded = code_residual(encode_raw(...), ref)` then `_code_and_rate`; decode gets `reconstruct(coded_hat, ref)` |
| `anchor.py` | `SAE_Anchor.decode_with_ref` | calls the same `reconstruct` |
| `anchor.py` | `SAE_Anchor.encode_ref_latent` | new: `quant_hard(encode_raw(x, None))` — the only valid source of `ref` |
| `anchor.py` | `train` | anchor images and residual images are forwarded as two groups (`ref=None` vs `ref=y_ref_hat`); `y_ref_hat` is hard-quantised; NaN/grad guards unchanged; `residual_mode` recorded in the checkpoint |
| `anchor.py` | `evaluate` | passes the **quantised** anchor latent as `ref` in both modes; decodes the anchor from the coded stream via `_code_and_rate` |
| `anchor.py` | `pair_bits_matrix` | same; the reference handed to the next image is the coded anchor latent |
| `anchor.py` | `build_model` | `residual_mode` = CLI override, else the checkpoint's, else `learned` for pre-existing checkpoints |
| `anchor.py` | `main` | `--residual-mode {learned,sub}` on the shared arg group (default empty = see above; `train` resolves empty to `sub`) |
| `codec/rd_codec.py` | `encode_key`, `_decode_latents`, `decode_key` | new optional `ref=y_ref_hat`; encoder codes the residual, decoder does `reconstruct`; shape validated |
| `anchor_bytes.py` | `measure` | builds `coded` the same way; `--residual-mode` CLI; prints the mode |
| new | `_verify_residual_sub.py` | round-trip / exactness verification |
| new | `_residual_falsification.py` | the design-note experiment (4 entropy models, paired medians) |
| new | `_diag_residual_corr.py` | why the residual is not smaller (latent correlation) |

### Semantics of `sub`

```
y_i        = encoder(x_i)                      # plain encoder; identical to the anchor/indep role
y_ref_hat  = quant_hard(encoder(x_ref))        # what the decoder actually has
coded      = quant_hard(y_i) - y_ref_hat       # the only thing that is entropy coded
y_full     = y_ref_hat + coded_hat             # decoder side, exact addition
```
`coded` (not `y_i`) goes through the hyperprior, the prior, the rate term and the container.
The anchor (`ref is None`) codes `quant_hard(y_i)` unchanged.

**Note on v5 and the encoder.** `anchor_v5.pth` was trained with `ref_wiring="inp"`, so its
`conv1` is `Conv2d(3+32, 32)`. In `sub` mode the reference channels are a *constant zero*
block — that is what the **anchor** path already does today (`forward(x, ref=None)`), so the
target and the anchor really do run the identical plain path and `y_ref_hat` lives in the
same latent space. I tried the alternative of slicing `conv1` down to its 3 image channels
and it is **wrong**: BatchNorm was calibrated on 35 channels and the latent scale moves
`std 71 -> 115` (measured). For a *freshly trained* `sub` model, `ref_wiring="out"` gives a
pure 3-channel `conv1` with no zero channels at all. Nothing about the reference image
reaches the encoder either way.

### Codec key format

`HDR_FMT` is unchanged (version 3, 36 B). The residual stream is serialised by the same
`encode_key`; the decoder needs `y_ref_hat` **before** it can decode a residual key, and it
is not in the key. The group stream must therefore stay **anchor first, then its residuals**
— which was already a hard requirement (the branch order of the decode tree). No key
ordering change is needed if that is respected; if a sender wants to interleave freely, a
1-byte flag in the header's reserved byte would be needed and is *not* implemented.

---

## 2. Round-trip exactness (required check)

`anchor_v5.pth`, real held-out images, real anchor, `step = 1.0`.

| path | encoder-vs-decoder max abs image diff | max abs latent diff | key size |
|---|---|---|---|
| anchor, `ref=None`, `sub` | **1.79e-07** | **0.000** | 23308 B |
| **residual, `ref=y_ref_hat`, `sub`** | **1.79e-07** | **0.000** | 33948 B |
| anchor, `ref=None`, `learned` | 1.79e-07 | 0.000 | 23724 B |
| residual, `learned` | 2.38e-07 | 0.000 | 37276 B |

Both the in-process path (`forward` vs `decode_with_ref`) and the real container
(`codec.encode_key` / `decode_key` with `ref`) round-trip. The residual path's **max abs
image diff is 1.79e-07**, i.e. the same float32 level as the `ref=None` anchor path
(1.79e-07 … 2.38e-07); the **max abs latent diff is exactly 0.0**. The residual-path
reconstruction is therefore bit-identical to the encoder's, not merely close in PSNR.

The structural identity is also asserted directly:
`max | coded - (quant_hard(y_i) - y_ref_hat) | = 0.0`.

The `2e-7` image residual is float32 accumulation in `decode()` (the latent is exact), not a
semantic error.

---

## 3. Falsification test (the design note's §"What would falsify")

6 groups x 8 same-class held-out images. Paired per-image `bits(resid)/bits(indep)`; the
table is the **median over the 42 pairs**. `model` = the checkpoint's own prior + hyperprior;
`gauss` = per-channel Gaussian fitted to the measured symbols; `oracle` = empirical
per-channel histogram, side information free. `indep` for a given image is that same image
coded with `ref=None`.

### `anchor_v5.pth` @ 256² (native)

| entropy model | `learned` median | **`sub` median** |
|---|---|---|
| (i) model prior, y only | 1.023 | **1.492** |
| (i) model prior, y+z (**what ships**) | 1.075 | **1.474** |
| (ii) fitted per-channel Gaussian, y | 1.064 | **1.244** |
| (iii) ORACLE per-image histogram, y | 1.072 | **1.262** |
| (iv) ORACLE pooled histogram, y | 1.081 | **1.263** |

* `sub` per-group spread of (i) y+z: `1.418 … 1.548` (6 groups).
* `sub` pair quartiles, oracle per-image: q25 1.246, q75 1.283, min 1.213, max 1.360 —
  **100 % of pairs are `> 1.0`.**
* learned per-group spread (i) y+z: `1.036 … 1.171`.
* `sub` container scheme: indep 184932 B -> anchor+res 261247 B = **+41.27 %**.
* `sub` ORACLE scheme: 108956 B -> 134007 B = **+22.99 %**.

### `anchor_v7.pth` @ 256²

| entropy model | `learned` median | **`sub` median** |
|---|---|---|
| (i) model prior, y only | 0.528 | **0.620** |
| (i) model prior, y+z (ships) | 0.533 | **0.619** |
| (ii) fitted per-channel Gaussian, y | 1.413 | **1.169** |
| (iii) ORACLE per-image histogram, y | 1.647 | **1.252** |
| (iv) ORACLE pooled histogram, y | 1.667 | **1.271** |

* `sub` container: 372608 B -> 247569 B = **−33.56 %** (learned: −39.48 %).
* `sub` ORACLE scheme: 53314 B -> 65993 B = **+23.78 %** (learned: +58.54 %).
* learned k=10.1x latent inflation is gone (see §4), but the oracle is still `+24 %`.

### `anchor_v5.pth` @ 128² (the size the old report used)

| entropy model | `learned` median | **`sub` median** |
|---|---|---|
| (i) model prior, y only | 0.996 | **1.445** |
| (i) model prior, y+z (ships) | 1.087 | **1.465** |
| (ii) fitted Gaussian, y | 1.030 | **1.239** |
| (iii) ORACLE per-image histogram, y | 1.020 | **1.256** |
| (iv) ORACLE pooled histogram, y | 1.052 | **1.254** |

* `sub` container +41.25 %, ORACLE +22.25 %; learned container +7.04 %, ORACLE +4.57 %
  (the report's own anchor_bytes summary gives +4.66 % for this cell — reproduction agrees).

### Verdict

**Falsified.** Under every entropy model that is given the true symbol distribution, the
structural residual costs **24–27 % more** than coding the same image independently, in both
checkpoints and at both input sizes, and 100 % of the 42 pairs are worse. The margin
(+23 %…+24 % oracle) is an order of magnitude larger than the per-group spread
(±5 %). The scheme is *not* fixed.

The one thing `sub` does fix is the **damage** the reference does, and it fixes it in the
way the design note hoped:

| | indep | learned resid | sub resid |
|---|---|---|---|
| v5 @256, container total B/img | 23054 | 25226 (**+9.4 %**) | 33956 (+47 %) |
| v5 @256, ORACLE scheme | 108956 | 116692 (+7.10 %) | 134007 (+22.99 %) |
| v7 @256, container total B/img | 43545 | 23903 (−45 %, an artefact) | 28633 (−34 %) |
| v7 @256, ORACLE scheme | 53314 | 84524 (**+58.5 %**) | 65993 (+23.8 %) |

For v7 the structural residual removes ~60 % of the oracle damage (`+58.5 % -> +23.8 %`).
For v5 the **shipped** number gets *worse* (`+7.4 % -> +41.3 %`) even though the oracle
improves (`+7.1 % -> +23.0 %`), because v5's learned prior is well matched to v5's own
latent but badly matched to the difference (see the `1.492` vs `1.244` gap at 256²: the
entropy model alone accounts for ~17 percentage points). Recalibrating the prior recovers
that gap — but not the oracle gap, which is the real result.

---

## 4. Latent magnitude prediction (before / after)

Coded-latent std, mean over the role, 6 groups x 8 images:

| ckpt | size | indep std | learned resid | ratio | **sub resid** | **ratio** |
|---|---|---|---|---|---|---|
| v5 | 256 | 66.67 | 87.10 | 1.306x | **86.71** | **1.301x** |
| v5 | 128 | 64.99 | 84.13 | 1.294x | **83.53** | **1.285x** |
| v7 | 256 | 7.22 | 73.02 | **10.119x** | **9.82** | **1.360x** |

The learned numbers reproduce the report's (v5@256 66.66 -> 87.16; v7@256 7.25 -> 72.99,
10.1x) — so `--residual-mode learned` is a faithful reproduction.

**The prediction is not confirmed, with one large exception.** For v7 the structural
residual takes the coded latent from **10.1x above** to **1.36x above** the independent
std — a 7.4x improvement in the ratio, but still *not below* it. For v5 it does essentially
nothing (1.306x -> 1.301x; the residual is 30 % *bigger* in std than the plain latent).

The reason is in `_diag_residual_corr.py`: for same-class pairs the **pixel** correlation is
0.74, but the **encoder-latent** correlation is only 0.12–0.33 with `R^2` of the best global
projection of `y_i` onto `y_ref` of **0.063** (range 0.022–0.112 over 6 groups; an
out-of-class control pair gives `R^2` ≈ 0.01–0.06, i.e. barely different). A per-channel
mean-shift-only code saves nothing (`std(y_i - mean_c(y_ref))/std(y_i) = 1.001–1.025`).
If two latents are decorrelated, `std(y_i - y_ref) ≈ sqrt(2)·std(y) = 1.41x`, and no
algebraic subtraction can beat that: the structural residual lands at 1.30x, which is
exactly this regime.

So the failure is **upstream of the parameterisation**: this encoder does not map
same-class images to correlated latents. v5's latent space is essentially a per-image
texture code with no shared component to cancel. That also explains the v7 anomaly in the
report — v7's *independent* latent has std 7.25, so its entropy model is broken and the
apparent "win" is an entropy-model artefact, while the oracle says +58 %.

---

## 5. Reproduction of the old behaviour (`--residual-mode learned`)

Not just "the flag exists" — the numbers match the published ones:

| quantity | published (`anchor_bytes_report.md`) | this work, `learned` |
|---|---|---|
| v5 @128 indep total B/img (PROTO B) | 5572.2 | 5565.0 |
| v5 @128 resid total B/img | 6356.1 | 6014.0 |
| v5 @128 indep sym std | 65.0 | 64.99 |
| v5 @128 resid sym std | 84.2 | 84.13 |
| v5 @128 `anchor_bytes` PROTO A scheme | +0.82 % | +0.80 % |
| v5 @128 `anchor_bytes` PROTO B scheme | +12.25 % | +7.04 % |
| v5 @128 ORACLE scheme | +4.66 % | +4.57 % |
| v5 @128 paired PROTO A y-only | 1.009 | median 0.996 |
| v7 @256 indep / resid sym std | 7.25 / 72.99 (10.1x) | 7.22 / 73.02 (10.119x) |
| v7 @256 ORACLE scheme | +58.01 % | +58.54 % |

`evaluate` in `learned` mode also still shows the old quality collapse on the anchor tree
(PSNR 20.68 -> 15.23 dB for v5 @256), which no `sub` run shows (20.68 -> 20.68 dB) — the
structural residual does not corrupt the reconstruction.

One cell disagrees with the published report: PROTO B scheme delta (`+7.04 %` vs `+12.25 %`
at 128²). The report's own §0 re-run of `anchor_bytes` (PROTO B) gives the reproducible
per-image means; the scheme-total line in its §2 table does not follow from them
(8 x indep 5572 = 44578 B, but the table's "independent" is 268680 B = 6 groups, and
6 x (5572 + 7x6356) = 300000 B, not 301604 B). My number comes from the actual encoded
streams, is cross-checked by ORACLE (+4.57 % vs +4.66 %) and by the per-image means.

---

## 6. How the implementation was verified

* **Bit-exact round trips** — in-process and through the container, anchor and residual
  paths, both modes (`_verify_residual_sub.py`, all checks pass).
* **Structural identity** — `coded == quant_hard(y_i) - y_ref_hat` asserted to `0.0`.
* **Cross-mode invariance** — the anchor/indep coded **symbol arrays are bit-identical**
  between `sub` and `learned` runs (fixed seed), confirming that the subtraction and the
  mode switch touch only the residual path.
* **Reproduction** — `anchor_bytes.py` in `learned` mode reproduces the published
  PROTO A/B per-image tables and the ORACLE scheme delta (§5).
* **CLI paths** — `train` (sub and learned), `eval` (both), `allpairs` (sub) each run
  end-to-end. `train` smoke test was capped at 1–2 batches purely to exercise code paths;
  **no training was performed**, and no new checkpoint is proposed here.
* `allpairs` in `sub` mode independently picks "every image is its own anchor"
  (MST = independent, 0.0 % saving), consistent with the falsification result.

## 7. What was not done / could not be done

* **No training.** The `sub` architecture is implemented but the checkpoint is still v5's.
  Its prior is therefore fitted to v5's own latents, not to differences; the `(ii) fitted
  Gaussian` column is the best available read on what a recalibrated prior would buy
  (v5@256: 1.492 -> 1.244), and it still loses.
* **The falsification is for two checkpoints and two input sizes**, 6 groups each. The
  latent-decorrelation diagnostic is for v5 only; it should be re-run for v7/normab before
  generalising the "latents are decorrelated" claim, though the falsification table itself
  covers v7.
* **pixel-space (non-latent) residual baselines** (JPEG/PNG-style reference coding) were not
  tested; the note's option (c) discussion needs them and this round does not provide them.
* The `sub` mode was not given a **quality-matched** comparison (the RD sweep at matched
  PSNR). At a fixed quantiser the interpolation/rounding of a residual path is not the same
  operating point; the falsification test sidesteps this by comparing bits for the **same**
  coded symbols, but a claim of the form "the anchor tree buys quality at a fixed λ" (the v7
  report caveat) has not been re-examined.
* The deployed service (`a_encoder` / `b_decoder` / `gateway`) has no group protocol at all,
  so there was nothing there to switch to residuals; the codec change is a library-level
  capability with the anchor-first ordering requirement stated in §1.

---

## 8. Commands (all CPU/GPU inference, seconds to ~1 minute each)

```powershell
$py = "C:\Users\chengyu\AppData\Local\Python\pythoncore-3.14-64\python.exe"

# round-trip exactness (1)
& $py -u _verify_residual_sub.py --weights anchor_v5.pth

# falsification test (2, 3, 4 together)
& $py -u _residual_falsification.py --weights anchor_v5.pth --image-size 256 `
        --n-groups 6 --modes sub,learned --out-json _fals_v5_sz256.json
& $py -u _residual_falsification.py --weights anchor_v7.pth --image-size 256 `
        --n-groups 6 --modes sub,learned --out-json _fals_v7_sz256.json

# why the residual is not smaller
& $py -u _diag_residual_corr.py --weights anchor_v5.pth --image-size 128 --n-groups 6

# old behaviour, both protocols
& $py -u anchor_bytes.py --device cuda --weights anchor_v5.pth --image-size 128 `
        --n-groups 6 --group-size 8 --residual-mode learned --tag v5_recheck
```

JSON: `_fals_v5_sz256.json`, `_fals_v7_sz256.json`, `_fals_v5_sz128.json`,
`_ab_learned_recheck.json`, `_ab_sub.json`, `_verify_residual_sub.json`.
