# `anchor_bytes.py` — byte-level breakdown of the anchor+residual scheme

Analysis only. **Nothing was run on the GPU** (the norm A/B job was still running; the marker
`normab_done.txt` does not exist yet — `run_norm_ab.ps1` runs `gn` → `bn` → `none` sequentially and
only writes it after all three). Everything below was produced on CPU, which turned out to be fast
(7 s per 6-group @128 run). No training, no `anchor.py train`, no python process touched.

```
python anchor_bytes.py --device cpu --weights <ckpt> --image-size {128,256} \
                       --n-groups 6 --group-size 8 --out-json anchor_bytes_<tag>_sz<sz>.json
python anchor_bytes_summary.py
```

Logs: `anchor_bytes_<ckpt>_sz<sz>.txt`; JSON: `anchor_bytes_<ckpt>_sz<sz>.json`;
consolidated: `anchor_bytes_summary.txt`.

## 0. First: the reported numbers are reproducible, but mislabelled

Re-run of the exact `evaluate` call sequence (PROTO A below) reproduces `_eval_v5.json` /
`_eval_v7.json` almost exactly, so the reported eval used the **default `--image-size 128`**
(not the checkpoint's native 256):

| quantity | reported json | my PROTO A | 
|---|---|---|
| v5 independent | 2788928 / 48 = **58103.2** | **58107** bits/img |
| v5 anchor only | 348960 / 6 = **58160** | **58160** |
| v5 residual | 2462400 / 42 = **58628.6** | **58649** |
| v7 independent | 1744000 / 48 = **36333.3** | **36374** |
| v7 anchor only | 215328 / 6 = **35888** | **35888** |
| v7 residual | 2195456 / 42 = **52272.8** | **52297** |

Two consequences:

1. `anchor.py::evaluate` prints `encode_latent_bytes(...)`'s return value — `enc.num_bits()` —
   under the label **"总字节" (total bytes)**. Those numbers are **bits**. The real cost of the
   v5 independent path is 58107 **bits = 7263 bytes**/image, not 58103 bytes.
2. The premise's "19.00 dB @ 2.363 bpp" / "14.07 dB @ 1.478 bpp" do not correspond to any bpp I can
   derive. Bits/pixel at 128² for the reproduced PROTO A numbers are **3.547 bpp (v5)** and
   **2.220 bpp (v7)**; the two quoted values are exactly 2/3 of those (and their ratio 1.5988 matches
   3.547/2.220 = 1.5977). They are not the training-log bpp either (v5 final 3.206, v7 final 0.736).
   Correct bpp values are given in §2.

## 1. Two protocols

**PROTO A** = `anchor.py::evaluate` (what produced the reported table):
`encode_latent_bytes(r_hat, quant.step, prior)` — per-channel factorized Gaussian only, **`k_max=32`**,
**`extra_sigma` not passed**, **no z stream**, no header. `enc.num_bits()` reported as bytes.

**PROTO B** = `codec/rd_codec.py::encode_key`, i.e. what the deployed service actually writes:
z stream coded first (adaptive alphabet, floor 128), then the y stream with the hyperprior per-pixel
sigma bucketed on `_SIGMA_GRID`, adaptive alphabet, plus the **36-byte** container header
(`HDR_FMT = "<4sBBBBHHHHHH8sII"`) and the range-coder flush.

They disagree qualitatively, so both are reported. `PROTO A` omits 27–33 % of v5's real bytes (the
whole z stream) and uses an entropy model whose alphabet is clipped at ±32 while the latents reach
|y| ≈ 900.

## 2. The byte-breakdown tables

Bytes per image. `n` = number of images in that role (6 groups × 8 images: role `anchor` is image 0 of
each group; role `indep` is images 1–7 encoded with `ref=None`; role `resid` is those same 7 images
encoded with `ref=<image 0's quantised latent>`).

### 256×256 — native training resolution

| ckpt | role | y-stream B | z-stream B | hdr/other B | total B | PSNR | bpp | sym std | sym max |
|---|---|---|---|---|---|---|---|---|---|
| v5 (BN) | indep | 15988.0 | 7281.5 | 35.5 | **23305.5** | 21.13 | 2.845 | 66.7 | 879 |
| v5 | anchor | 16420.7 | 7602.7 | 35.3 | **24059.3** | 20.76 | 2.937 | 69.4 | 894 |
| v5 | resid | 17875.2 | 8828.2 | 34.4 | **26739.4** | 20.73 | 3.264 | 87.2 | 947 |
| v7 (GN) | indep | 42479.8 | 1045.5 | 35.3 | **43561.3** | 15.15 | 5.318 | 7.2 | 444 |
| v7 | anchor | 42250.0 | 1171.3 | 35.3 | **43457.3** | 14.46 | 5.305 | 6.5 | 414 |
| v7 | resid | 24636.5 | 612.1 | 34.6 | **25284.6** | 20.11 | 3.087 | 73.0 | 818 |
| normab_gn_best | indep | 11472.7 | 953.9 | 35.6 | **12462.6** | 16.83 | 1.521 | 106.0 | 902 |
| normab_gn_best | anchor | 11426.7 | 957.3 | 35.0 | **12420.0** | 15.58 | 1.516 | 103.8 | 885 |
| normab_gn_best | resid | 13540.1 | 1004.6 | 34.7 | **14580.7** | 15.72 | 1.780 | 117.5 | 938 |

### 128×128 — the size the reported eval actually used

| ckpt | role | y-stream B | z-stream B | hdr/other B | total B | PSNR | sym std | sym max |
|---|---|---|---|---|---|---|---|---|
| v5 | indep | 4008.3 | 1527.9 | 35.2 | **5572.2** | 19.05 | 65.0 | 831 |
| v5 | anchor | 4083.3 | 1655.3 | 34.7 | **5774.7** | 18.69 | 67.8 | 820 |
| v5 | resid | 4390.0 | 1930.1 | 35.1 | **6356.1** | 18.65 | 84.2 | 899 |
| v7 | indep | 10318.6 | 224.7 | 35.2 | **10579.2** | 14.14 | 5.8 | 285 |
| v7 | anchor | 10332.7 | 242.0 | 35.7 | **10610.7** | 13.51 | 5.9 | 330 |
| v7 | resid | 5756.8 | 147.6 | 35.4 | **5940.4** | 18.07 | 71.3 | 774 |
| normab_gn_best | indep | 2877.2 | 244.8 | 35.0 | **3158.0** | 15.74 | 103.1 | 869 |
| normab_gn_best | anchor | 2896.7 | 246.0 | 35.7 | **3178.7** | 14.66 | 101.0 | 842 |
| normab_gn_best | resid | 3363.2 | 261.6 | 34.9 | **3660.9** | 14.92 | 114.1 | 924 |

### Scheme totals per group of 8 (same 48 images, 6 groups)

| ckpt | size | PROTO A indep | PROTO A anch+res | Δ | PROTO B indep | PROTO B anch+res | Δ | ORACLE Δ |
|---|---|---|---|---|---|---|---|---|
| v5 | 128 | 348684 B | 351528 B | **+0.82 %** | 268680 B | 301604 B | **+12.25 %** | +4.66 % |
| v5 | 256 | 1392752 B | 1413260 B | +1.47 % | 1123188 B | 1267412 B | **+12.84 %** | +7.39 % |
| v7 | 128 | 217880 B | 301476 B | **+38.37 %** | 507992 B | 313160 B | **−38.35 %** | +58.09 % |
| v7 | 256 | 867428 B | 1212580 B | +39.79 % | 2090320 B | 1322696 B | **−36.72 %** | +58.01 % |
| normab | 128 | 225304 B | 230976 B | +2.52 % | 151708 B | 172828 B | **+13.92 %** | +12.71 % |
| normab | 256 | 900904 B | 920740 B | +2.20 % | 597948 B | 686908 B | **+14.88 %** | +12.34 % |

*PROTO A* reproduces the reported "+0.8 %" (v5) and "+38.2 %" (v7). *PROTO B* (the real container)
says **+12–15 %** for v5 and normab_gn_best, and −37 % for v7.
*ORACLE* = each channel's symbols coded with their own empirical histogram, side information free —
a lower bound that no real entropy model can beat. Under it the anchor scheme loses in **all six**
configurations (+4.7 % … +58 %).

## 3. Answers to the four questions

### Q1 — Is the extra cost from the hyperprior z stream or from the y stream?

**Both, but the split depends on the checkpoint.** Note first that in PROTO A the z stream does not
exist at all, so the reported comparison could not have answered this.

`resid − indep` per image, PROTO B, 256×256:

| ckpt | Δ total | Δ y | Δ z | z share of the delta |
|---|---|---|---|---|
| v5 | **+3434 B** | +1887 B | +1547 B | y 55 % / **z 45 %** |
| v7 | −18277 B | −17843 B | −433 B | y 98 % / z 2 % |
| normab_gn_best | +2118 B | +2067 B | +51 B | y 97 % / z 3 % |

But the more important fact is how big the z stream is in absolute terms:

| ckpt | z as % of the whole stream (indep) |
|---|---|
| v5 @128 / @256 | **27.6 % / 31.3 %** |
| v7 @128 / @256 | 2.1 % / 2.4 % |
| normab @128 / @256 | 7.8 % / 7.7 % |

For **anchor_v5 the z stream is a third of the file and is itself mis-coded** (see Q4 / §5): it costs
58252 bits/img at 256 where a per-channel Gaussian fitted to the same z symbols needs 16440 bits and
the empirical histogram 12517 bits. So for v5 the answer is "the y stream slightly more than the z
stream, and the z stream is separately broken".

### Q2 — Does the residual image produce a smaller y stream than coding it independently?

**No — the residual latent is larger, not smaller, in every checkpoint and at both input sizes.**

Latent magnitude (mean over the role):

| ckpt | size | indep raw std | resid raw std | ratio | indep max | resid max |
|---|---|---|---|---|---|---|
| v5 | 256 | 66.66 | **87.16** | 1.31× | 879 | 947 |
| v5 | 128 | 64.98 | **84.17** | 1.30× | 831 | 899 |
| v7 | 256 | 7.25 | **72.99** | **10.1×** | 444 | 818 |
| v7 | 128 | 5.79 | **71.32** | **12.3×** | 285 | 774 |
| normab | 256 | 106.02 | **117.45** | 1.11× | 902 | 938 |
| normab | 128 | 103.13 | **114.05** | 1.11× | 869 | 924 |

True symbol entropy (empirical histogram, pooled over the role, bits/img):

| ckpt | size | indep | resid | Δ |
|---|---|---|---|---|
| v5 | 256 | 108668 | 117551 | +8.2 % |
| v7 | 256 | 53508 | 89210 | **+66.7 %** |
| normab | 256 | 55221 | 63098 | +14.3 % |

Paired per-image ratio (resid vs the *same* image coded independently), 42 pairs:

| ckpt | size | PROTO B total ratio | PROTO A y-only ratio | images worse |
|---|---|---|---|---|
| v5 | 128 | **1.145** | 1.009 | 90 % |
| v5 | 256 | **1.151** | 1.017 | 95 % |
| v7 | 128 | 0.560 | 1.450 | 0 % |
| v7 | 256 | 0.579 | 1.466 | 0 % |
| normab | 128 | **1.160** | 1.029 | 100 % |
| normab | 256 | **1.171** | 1.025 | 100 % |

Subtlety: `ref_wiring="inp"` means the encoder sees the anchor latent (bilinearly upsampled 16×) as
32 extra input channels; nothing in the architecture forces `r ≈ y_i − y_ref`. The encoder simply
emits a *larger* latent when the reference is present. Note the paradox for v7: its independent
latent has std 7.25 while its residual latent has std 72.99, yet PROTO B codes the residual *cheaper*
— purely because v7's entropy model is far more badly calibrated on the independent path (Q4).

### Q3 — Header / key overhead

The RD container header is `HDR_FMT = "<4sBBBBHHHHHH8sII"` → **36 bytes** (`codec/rd_codec.py:44`).
Measured "hdr/other" column (36 B header + range-coder flush/byte-rounding on both streams) is
**34.4–35.7 B/image**, i.e.

| ckpt | size | hdr+flush as % of a resid image's total |
|---|---|---|
| v5 | 128 / 256 | 0.57 % / 0.14 % |
| v7 | 128 / 256 | 0.61 % / 0.14 % |
| normab | 128 / 256 | 0.98 % / 0.25 % |

**At this scale it is negligible** — under 1 % everywhere, under 0.3 % at 256². It does not explain
the anchor scheme's rate loss. (The old pulse-key path uses a 14-byte header, `sae_model.KEY_HEADER_SIZE`;
the RD path uses 36 B. Both are irrelevant here.)

### Q4 — Latent magnitude indep vs resid, and the frozen sigma

Latent magnitude: see the Q2 table. For the sigma question:

`prior.sigma()` is per-channel and essentially frozen (v5: log shows 0.369–0.434; checkpoint mean
0.3686; v7: 0.3739; normab: 0.4261), while the latent std is 65/7.25/106. **But `prior.sigma()` is
multiplied by the hyperprior's per-pixel `extra`, so the effective sigma is not frozen.** What I
measured is that the effective sigma field is degenerate, not frozen:

| ckpt (indep, 256) | p1 | p10 | p50 | p90 | p99 | max | symbol std | % pixels > grid top (100) |
|---|---|---|---|---|---|---|---|---|
| v5 | 0.137 | 0.188 | 0.266 | 0.77 | 275413 | 384339 | 66.66 | **9.4 %** |
| v7 | 0.0001 | 0.0050 | 0.0205 | 0.07 | 56 | 653 | 7.25 | 0.28 % |
| normab | 0.0499 | 0.0649 | 0.0945 | 0.17 | 442 | 1490 | 106.02 | 5.0 % |

Three things stand out:

* For **v7** the model predicts sigma ≈ 0.02 over 90 % of pixels while the symbols have std 7.25 —
  the model's own distribution is ~350× too narrow, so ~24 % of symbols (the non-zeros) all sit on
  the clamped probability floor and cost exactly 29.9 bits each.
* For **v5** the sigma field is bimodal across 6 orders of magnitude, and `eff_sigma_max` is
  **exactly 384338.5625 for all 42 images and at both input sizes** (verified per-image in the JSON:
  `unique == 1`) — a content-independent escape value from the hyperprior, not a data-driven prediction.
* `_SIGMA_GRID = exp(linspace(log 0.01, log 100, 48))` tops out at **100**, while
  `ScaleHyperprior` allows `exp(±14)` = 1.2e6 and `FactorizedGaussianPrior.sigma` clamps at 1e6.
  So 9.4 % of v5's pixels (and 5 % of normab's) are coded with sigma = 100 whatever the model
  predicted. The coder's probability model is not the model's probability model there.

## 4. The single most likely cause of the rate loss

**The reference is not being used to reduce the latent.** With `ref_wiring="inp"` the anchor's
upsampled latent is concatenated as 32 extra input channels and the encoder emits a *larger,
higher-entropy* latent when it is present instead of a difference:

* v5 @256: residual latent std **87.16** vs independent **66.66** (+31 %), |max| 947 vs 879, and the
  empirical symbol entropy is **117551 vs 108668 bits** (+8.2 %); the z stream grows from 58252 to
  70626 bits (+21 %). Net: **+3434 B/img (+12.8 %)** for a group, and PSNR gets *worse*
  (21.13 → 20.73 dB).
* normab_gn_best @256: std 117.45 vs 106.02, entropy 63098 vs 55221 (+14 %), **+2118 B/img (+14.9 %)**,
  PSNR also worse (16.83 → 15.72 dB).
* v7 @256: std **72.99 vs 7.25 (10.1×)**, entropy 89210 vs 53508 (+67 %).

Because the scheme pays *more* bytes for a latent that carries *no less* information, the loss is not
a coding artefact — it survives with an oracle entropy model (ORACLE column: +4.7 % … +58 % in all
six configurations). The reported "+0.8 %" massively understates it because PROTO A drops the z
stream and clips the alphabet at ±32, which pushes both roles onto the same 1e-9 probability floor
(≈29.9 bits/symbol) and hides the difference between a std-67 and a std-87 latent
(paired ratio 1.009 in PROTO A vs 1.145 in PROTO B for v5).

Caveat, stated plainly: for **v7** the residual path also buys a large quality gain
(15.15 → 20.11 dB @256, 14.14 → 18.07 dB @128). That is not a pure rate loss — it is a different
operating point and cannot be judged without an RD sweep at matched λ. What can be said is that the
v7 anchor scheme's *apparent* 37 % rate **win** in the real container (PROTO B) is an artefact of the
independent path being mis-coded by 5–6×, not evidence that the reference helps: with an oracle
entropy model v7's anchor scheme costs **+58 %**.

## 5. Things that look like bugs (all measured, plus two read from source)

1. **`anchor.py::evaluate` labels bits as bytes.** `encode_latent_bytes` returns `enc.num_bits()`
   (see its own docstring, "实际比特数"); `evaluate` prints it as `"总字节"` and writes it to
   `independent_bytes` / `anchor_bytes` / `residual_total` in `_eval_v5.json` / `_eval_v7.json`.
   Everything downstream that consumed those JSONs has an 8× unit error.
2. **`evaluate` omits the z stream and the hyperprior sigma.** It calls
   `encode_latent_bytes(r, model.quant.step, model.prior)` with no `extra_sigma` and never calls
   `encode_z_bytes`. For anchor_v5 the missing z stream is **27.6 % of the independent image's real
   bytes (31.3 % at 256²)**. `pair_bits_matrix` (line 543–547) does it correctly — `evaluate` does not.
3. **`encode_latent_bytes`'s default `k_max=32` vs latents of |y| ≈ 900.** Used by `evaluate`, all
   symbols beyond ±32 collapse to the edge bin. `codec/rd_codec.py` fixed this with `_pick_kmax`
   (floor 128, cap 4096) and `sae_rd.encode_z_bytes` still defaults to `k_max=32` while `rd_codec`
   uses 128 — the two z coders in the repo disagree.
4. **`clamp_min(1e-9)` + float32 `erf` saturation create a zero-gradient dead zone in the entropy
   model.** `FactorizedGaussianPrior.bits` and `_bin_probs` clamp at 1e-9; `_phi` saturates to 1.0
   once |x| ≳ 5.3 in float32, so any symbol with |y| > ~3σ costs exactly 29.9 bits and contributes
   **no gradient w.r.t. sigma**. This is consistent with the training logs: v7's `sigma` drifts
   0.434 → 0.374 over 1300 epochs while `y_std` grows 0.68 → 50.4, and `z_prior.sigma` collapses.
5. **`z_prior` is badly miscalibrated for v5 and v7** (it is never touched by `warmup_sigma`, which
   only recalibrates the y prior — `sae_rd.py:366-390`). Measured at 256², independent role:

   | ckpt | `z_prior.sigma()` | pooled z std | z real bits | z fitted-Gaussian bits | z empirical entropy |
   |---|---|---|---|---|---|
   | v5 | **0.0566** | 1.285 | **58252** | 16440 | 12517 |
   | v7 | **0.0440** | 0.256 | **8364** | 3111 | 2772 |
   | normab | 0.4396 | 1.847 | 7631 | 7736 | 2202 |

   For v5 the z stream costs **3.5× a per-channel Gaussian fit and 4.7× the empirical entropy**
   (6.4 bits per z-symbol, of which ~32 % are non-zero and each pays the 29.9-bit floor).
   v5's z stream is ~31 % of the file, so **recalibrating `z_prior.sigma` alone is worth ≈ 22 % of
   the whole file** with zero side information. For normab the learned sigma is already at the
   Gaussian optimum.
6. **Rate left on the table overall** (independent role, 256², bytes/img; "fixed" = y coded with a
   per-channel Gaussian fitted to the actual symbols and z likewise — an oracle read, side info free):

   | ckpt | current total | y current → fitted | z current → fitted | fixed total | saving |
   |---|---|---|---|---|---|
   | v5 | 23306 | 15988 → 13631 | 7282 → 2055 | 15722 | **−32.5 %** |
   | v7 | 43561 | 42480 → 7855 | 1046 → 389 | 8280 | **−81.0 %** |
   | normab | 12463 | 11473 → 6966 | 954 → 967 | 7969 | **−36.1 %** |

   These are lower bounds (the fitted priors are oracle side information), but the direction is
   unambiguous: for v7 the model's *own* analytic estimate is 411345 bits/img where the symbols'
   empirical entropy is 53508 bits/img — a 7.7× gap.
7. **`_SIGMA_GRID` range vs the hyperprior's range** (measured above): coder tops out at sigma = 100,
   model reaches 1e5–1e6; 9.4 % (v5) and 5 % (normab) of pixels are affected.
8. **`codec/rd_codec.py` encode/decode disagree on sigma** (read from source, not measured):
   encode clamps `log_sigma` to ±14 and `sigma` to [1e-4, 1e6] (`ScaleHyperprior.forward`,
   `FactorizedGaussianPrior.sigma`), decode clamps to ±8 and [1e-3, 1e3] (`rd_codec.py:187`);
   encode assembles sigma as `prior.sigma(es_full)[:, :, c]`, decode as
   `(prior.sigma() * sigma)[:, c]`, and the two use different `${clamp}` bounds → for any pixel where
   `h_s` output falls outside ±8 (very common, see the p99 column) encoder and decoder pick
   **different sigma buckets** and the decoded symbols are garbage. Also `decode_key` uses
   `hz = wz = model.latent_size // 4` (16//4 = 4) but the real z spatial size is `input_size/16/4`,
   which is 2 at the 128 input size the eval uses.
9. **`build_data` and the model use different `image_size`s.** `evaluate` builds the data with
   `args.image_size` (default **128**) but constructs the model from `obj["image_size"]` (**256**).
   `model.n_pix` (used for every bpp the model reports) stays 65536, so all bpp values from the
   model are 4× off when it is driven at 128, and `model.latent_size` (16) no longer matches the
   actual latent size (8). The reported eval ran in exactly this state.

## 6. Caveats on my own numbers

* "fitted Gaussian" and "oracle" columns are **lower bounds with free side information**
  (per-channel and, for y, per-image moment fits). They are diagnostics of miscalibration, not
  achievable rates. The `z` fits are pooled over the role (per-image z fits are overfit: only 40
  z-samples per channel per image).
* PROTO B reproduces `rd_codec.encode_key`'s stream construction exactly, but I did not round-trip a
  real `encode_key`/`decode_key` pair (item 8 predicts it would fail).
* All numbers are from the 48 held-out images in groups 1–6 of `build_data` — the same images as the
  reported eval. Per-group spread is large (v5 @128 group deltas range +5.6 % to +19.3 %), so
  single-group comparisons are not reliable.
* `normab_gn_best.pth` is the ep-12 checkpoint (val 15.92 dB) of the still-running A/B; the `gn` leg
  finished at 11:29:54 and this file is final, but the parent may prefer to re-run against the
  eventual `none` winner.
