# Fix round: 6 measured bugs in `snn_ab` — what changed and the evidence

All measurements below were run **CPU-only** (`CUDA_VISIBLE_DEVICES=""`) with
`C:\Users\chengyu\AppData\Local\Python\pythoncore-3.14-64\python.exe`. No training was launched and
no process was killed. Reproduce each item with the listed `_verify_fixN.py` script.

| # | file : function | status |
|---|---|---|
| 1 | `anchor.py` : `evaluate` (+ new `container_bits`, `build_model`; `pair_bits_matrix`, `eval_allpairs`) | fixed, verified |
| 2 | `sae_rd.py` : `_log_bin_prob` (new), `FactorizedGaussianPrior.bits`, `_bin_probs`, `encode_z_bytes` | fixed, verified |
| 3 | `sae_rd.py` : `SIGMA_MIN/SIGMA_MAX/_SIGMA_GRID` | fixed, verified |
| 4 | `sae_rd.py` : `warmup_sigma`, new `ScaleHyperprior.analyze_z`; new `recalibrate_prior.py` | fixed, verified |
| 5 | `codec/rd_codec.py` : `encode_key` / `_decode_latents` / `decode_key`, new `ScaleHyperprior.sigma_from_z` | fixed, verified (bit-exact) |
| 6 | `design_note_anchor_residual.md` | written (no code change, as instructed) |

---

## 1. `evaluate` reported bits as bytes, dropped the z stream, and used the wrong input size

**Changed** (`anchor.py`)
* new `container_bits(model, y_hat, rate, …)` — the single bit-accounting path: y stream **with**
  `extra_sigma`, plus the z stream, with the alphabet chosen by `codec.rd_codec._pick_kmax`
  (the container's own rule; the old ±32 clipped |y|≈900 symbols into the edge bin).
* `evaluate` now loads the checkpoint **first** and sets `args.image_size = obj["image_size"]`,
  so a 256-trained model is never evaluated on 128 px data (`model.n_pix` and `model.latent_size`
  now match the tensors). Same reordering in `eval_allpairs`. Shared `build_model()`.
* all labels/JSON keys are unambiguous: `*_bytes` are bytes, `*_bits` are bits, plus `bpp`
  and the z-stream share.

**Evidence** (`python anchor.py eval --weights anchor_v5.pth --n-groups 2 --group-size 4 --image-size 128`,
which now auto-switches to 256; `_verify_fix1.py`)

| | old code | new code |
|---|---|---|
| input size used | 128 (model is 256) | 256 (follows the checkpoint) |
| unit printed | `encode_latent_bytes` **bits** as "总字节" | real bytes (`bits/8`), z included |
| entropy model | per-channel only, **no `extra_sigma`** | per-pixel hyperprior sigma |
| per image | 232 296 "B" (really bits) | **23 320 B** (2.847 bpp) |
| z stream | absent | 31.6 % of the stream |

* Overstatement of the old number: **9.96×** (8× unit + a wrong entropy model).
* Cross-checks: `independent_bits / independent_bytes == 8.000` exactly; `independent_bpp ==
  bits/(n·H·W)`; `container_bits` equals `pair_bits_matrix`'s diagonal **exactly**; the new
  per-image figure (23 320 B) reproduces `anchor_bytes_report.md`'s PROTO-B container number
  (23 305.5 B for v5 @256) to 0.06 % — the two must now agree because they are the same accounting.

## 2. Zero-gradient dead zone in the entropy model (sigma could not learn)

**Changed** (`sae_rd.py`): `FactorizedGaussianPrior.bits` no longer computes
`Phi(b) - Phi(a)` in float32 with `clamp_min(1e-9)`. New `_log_bin_prob(t, h)` computes
`log(Phi(t+h) - Phi(t-h))` in the log domain:
`L(b) + log(1 - exp(L(a)-L(b)))` with `L = log_ndtr`, the "both boundaries positive" case mirrored
onto the negative half-line (where `log_ndtr` keeps relative precision), `1-exp(d)` via `-expm1`,
and a density fallback (`log 2h·phi(t)`) when the two boundaries are indistinguishable in float32.
`_bin_probs` (the table the arithmetic coder really uses) now calls the same helper, so the
training-time rate and the coder's probability model are one model.

**Evidence** (`_verify_fix2.py`) — required test, |standardized residual| > 5:

| t = y/σ | old rate | old d(rate)/d logσ | new rate | new d(rate)/d logσ |
|---|---|---|---|---|
| 6 | 29.8974 | **0.0** | 25.6533 | −44.96 |
| 10 | 29.8974 | **0.0** | 69.6909 | −131.6 |
| 50 | 29.8974 | **0.0** | 1 774.44 | −3 536.3 |
| 900 | 29.8974 | **0.0** | 583 653.6 | −1.165e6 |
| 5 000 | 29.8974 | **0.0** | 18 030 092 | −8.578e6 |

* Accuracy vs an independent float64 `math.erfc` reference (mirrored to avoid cancellation):
  max |Δ| = **7.7e-12 bits** over t ∈ [−20, 20].
* Real checkpoints (@256, 4 images): v5 analytic y-rate 165 098.6 → 165 098.8 bits/img (+0.00 %,
  only 0.043 % of its symbols had |t| > 5.5 — the fix is inert for it);
  **v7 387 034.8 → 20 249 102 bits/img (+5132 %)**, because 13.8 % of its symbols sat exactly on
  the old 1e-9 floor and 35.9 % had |t| > 5.5 (max |t| = 1241). That is the dead zone quantified:
  the old model charged a badly-predicted symbol 29.9 bits *and* gave σ no gradient at all.
* Gradients on real data reach `prior.log_sigma` (6.8e6) and `h_s`'s last conv (2.1e6), finite.

## 3. `_SIGMA_GRID` was too narrow

**Changed** (`sae_rd.py`): grid is now built from new module constants
`SIGMA_MIN=1e-4`, `SIGMA_MAX=1e6`, `LOG_SIGMA_CLAMP=14` (the same bounds the priors clamp to),
endpoints pinned to the constants, `SIGMA_GRID_N=160`. `_bucket_ids` clips into the same range.

| | points | range | step |
|---|---|---|---|
| old | 48 | [0.01, 100] | 0.0851 decade |
| new | 160 | [1e-4, 1e6] | 0.0629 decade (finer than before) |

**Evidence** (`_verify_fix3.py`), effective σ field on 4 real 256 px images:

| ckpt | old: outside grid | new: outside grid | mean \|log2(σ/grid)\| old → new | y stream old → new |
|---|---|---|---|---|
| anchor_v5 | 9.374 % | **0.000 %** | 0.802 → 0.053 | 130 528 → 127 664 bits/img (−2.19 %) |
| anchor_v7 | 23.601 % (mostly *below*) | **0.000 %** | 0.380 → 0.052 | unchanged (its model is broken for other reasons) |
| normab_gn_best | 5.467 % | **0.000 %** | 0.139 → 0.052 | 91 960 → 69 240 bits/img (−24.71 %) |

## 4. `z_prior` was never calibrated

**Changed** (`sae_rd.py`, plus new `recalibrate_prior.py`)
* `warmup_sigma` now fits **both** priors on the same pass, via the new
  `ScaleHyperprior.analyze_z` (one shared `h_a` path): per-channel, `mu = 0`,
  `log_sigma = log(std)`. New args `fit_y/fit_z/verbose/train_mode`.
* **Unit bug fixed**: the old code accumulated the **variance** and used `log(variance)` as
  `log_sigma` (its own docstring says "standard deviation"; v7's log shows σ = 0.434 with
  y_std = 0.68 ≈ 0.68² — the two priors are now both in std units).
* `recalibrate_prior.py` re-fits an **existing** checkpoint on real images and reports the old/new
  z-stream cost; `--mle` also reports the per-channel discrete-MLE bound.

**Evidence** (`recalibrate_prior.py --weights anchor_v5.pth --n-images 8 --mle`, @256)

* `z_prior.sigma()` old: mean **0.0566** [0.0514, 0.0628]; measured per-channel z std printed by
  the tool (pooled **1.2594**, per-channel mean 0.4133; the array is in its output).
* z-stream cost per image: **97 301.9 bits → 15 413.4 bits (−84.2 %, 6.3×)**, which lands right on
  the report's independently estimated 58 252 → 16 440 bits.
* diagnostics: variance-convention fit would give 18 294.5; per-channel discrete MLE 8 970.9;
  per-channel empirical histogram (oracle) 5 808.0 bits/img — so the moment fit captures most of
  the available gain and the remaining gap is a known, documented opportunity.
* the constriction range coder really produces 13 792 bits/img for the first image with the new
  prior (not just the analytic estimate).
* training path smoke-tested with `train_mode=True` (BN batch stats used, z prior fitted, no NaN).
* side effect measured: the std convention also improves the **y** prior initialisation,
  325 952 → 167 752 bits/img for the same checkpoint (2×).

## 5. `encode_key` and `decode_key` disagreed on sigma (deployment-breaking)

**Changed** (`codec/rd_codec.py`, `sae_rd.py`)
* one shared `ScaleHyperprior.sigma_from_z()` = `clamp(±14) → exp → clamp(1e-4, 1e6)`; `forward`
  and the decoder both call it (decode used to re-implement `clamp(±8) → clamp(1e-3, 1e3)`).
* one shared effective-sigma path: `model.prior.sigma(extra)` on both sides
  (`(s·extra).clamp(SIGMA_MIN, SIGMA_MAX)`), instead of decode's hand-written
  `(prior.sigma() * sigma).clamp(1e-3, 1e3)`.
* the encoder now derives its per-pixel sigma from the **integer z symbols the decoder will get**
  (round + clamp to the z alphabet) instead of the raw `z_hat` — otherwise one clipped z symbol
  desynchronises the two probability tables.
* `encode_key` writes the **actual** input edge into the header; `_decode_latents` derives
  `latent_hw = image_size//16`, `hz = wz = latent_hw//4` from it instead of `model.latent_size//4`.
* `DEF_Z_KMAX = 128` named explicitly; `_enc_z` takes integer symbols; `decode_key` split into
  `_decode_latents` + reconstruction (so the latents can be compared in tests).

**Evidence** (`_verify_fix5.py`, `service/_smoke_codec.py`) — required bit-exact round trip:

| case | max abs latent diff | max abs image diff |
|---|---|---|
| 256 px, real checkpoint | **0.000e+00** | **0.000e+00** |
| 128 px input with the 256 model (the case that used to break) | **0.000e+00** | **0.000e+00** |
| partial key (`steps=6, channels=20`) | **0.000e+00** (coded sub-tensor) | — |

* Severity of the old bug, measured the same run: the old decode-side formula puts **9.34 %** of
  pixels (9.06 % of pixels have |h_s| > 8) into a **different sigma bucket** → garbage.
* Deployment path: `python service\_smoke_codec.py` loads `models/a/encoder.pth` and
  `models/b/decoder.pth`, encodes at three rate points (24 292 / 7 496 / 3 164 B), decodes to
  16.26 dB, and correctly rejects both a wrong-hash key and a truncated key.

## 6. Anchor/residual parameterisation — design note only

`design_note_anchor_residual.md` (~1 page): compares (a) coding the explicit difference
`y_i − y_ref` as the coded quantity, (b) a learned path with an identity/skip start,
(c) dropping the anchor mechanism. Recommends (a) first, (b) as an ablation, (c) as the fallback,
and states the falsifying experiment (oracle-entropy ratio on ≥6 same-class groups, paired
per-image median, plus the coded-latent-std prediction). No architecture change was made.

---

## Reproduce

```
set CUDA_VISIBLE_DEVICES=
python _verify_fix1.py            # unit/z-stream/input-size + evaluate round trip
python _verify_fix2.py            # dead zone: before/after gradients + accuracy + real ckpts
python _verify_fix3.py            # grid endpoints + pixels outside the grid, 3 checkpoints
python _verify_fix4.py            # training-path smoke test + y-prior convention
python recalibrate_prior.py --weights anchor_v5.pth --n-images 8 --mle
python _verify_fix5.py            # bit-exact encode/decode round trip
python service\_smoke_codec.py    # deployed A/B split end to end
```

## Things I could not verify / caveats

* `anchor_bytes_normab_gn_best_sz128.json` was regenerated by `anchor_bytes.py --selftest` **before
  I noticed that selftest writes to its default output path** (the old content was untracked, so it
  is not recoverable from git). The file is now a complete, consistent 6-group run of the fixed
  coder, so its numbers differ from the tables in `anchor_bytes_report.md`. The report itself is
  untouched. Re-running the other `anchor_bytes_*.json` files would shift them the same way.
* The z prior is fitted on the **raw** z while the reported per-channel std is on the rounded
  symbols; they differ by <1 %.
* Bucket ids are still derived from float32 sigma, so an encoder on CUDA and a decoder on a
  different CPU/CUDA stack could in principle disagree on a pixel whose sigma sits within 1 ulp of a
  bucket boundary (probability ≈1e-7 per pixel, ≈1 % of 256 px images get one flipped pixel).
  That is inherent to the "no side information, derive the bucket from sigma" design; the fix
  removes the *systematic* 9.3 % disagreement, not this last-ulp fragility.
* `k_max` for the coders was not changed to be shared: `sae_rd.encode_z_bytes` still defaults to
  32 while `rd_codec` uses 128 (report item 3). Not part of this batch; `evaluate` now uses the
  container's rule.
