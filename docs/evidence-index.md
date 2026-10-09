# Evidence index

Where every headline number in the [README](../README.md) comes from. All paths are
relative to the repository root. Numbers that could not be traced to a file in this repo
are listed at the bottom as **unverified** rather than quietly repeated.

## Data and training cost

| claim | source |
|---|---|
| Flowers102 8189 + DIV2K/Flickr2K 3450 = **11639** total | `anchor_v9.log` line 3; `anchor.py::build_data` |
| 15 % holdout → **9894 train / 1745 holdout** | `anchor.py::build_data` (`n_hold = max(200, int(n * holdout_frac))`); `anchor_v9.log` line 3 |
| each epoch evaluates the first `--eval-limit` (300) of the holdout | `anchor_v9.log` line 3 ("留出 300") |
| 618 batches × 16 = **9888 images/epoch** | `anchor_v9.log` line 4 |
| augmentation = `RandomResizedCrop(0.5–1.0)` + horizontal flip, train split only; holdout = resize + center crop | `anchor.py::build_data` lines 350–357 |
| **51.0 s/epoch** at 128 px / C=24 / T=6 | `anchor_v9.log` line 8 |
| **~102 s/epoch** at 256 px / C=32 / T=10 incl. real validation | `anchor_v8.log` line 7 (102.1 s) |
| argparse defaults (`--image-size 128 --latent-channels 24 --num-steps 6 --data both --batch-size 16 --norm bn --augment crop --lam 0.01 --lr 1e-3 --holdout-frac 0.15`) | `anchor.py::main` lines 990–1058 |

## Short-run baseline (§4.1)

| claim | source |
|---|---|
| epoch 1: 19.77 dB real validation, 0.7559 bpp analytic, 51.0 s | `anchor_v9.log` line 8 |
| epoch 2: 20.42 dB @ 0.7159 bpp, 50.4 s | `anchor_v9.log` line 11 |
| epoch 3: 20.68 dB @ 0.6741 bpp, 48.6 s | `anchor_v9.log` line 14 |
| epoch 4 (best so far): **21.01 dB @ 0.6283 bpp**, 48.1 s | `anchor_v9.log` line 17 |
| epoch 5: 20.97 dB @ 0.6357 bpp, 49.1 s | `anchor_v9.log` line 20 |
| epoch 6: 20.53 dB @ 0.6129 bpp, 49.8 s | `anchor_v9.log` line 23 |
| same command, an earlier launch: 19.99 dB @ 0.7671 bpp at epoch 1 | reported during the run; **not present in any log file in this repo** (see *unverified*) |
| the run's actual geometry is 128 px / C=24 / T=6, anchor off, BN | `anchor_v9.pth` metadata (`image_size: 128, latent_channels: 24, num_steps: 6, use_anchor: False, norm_type: 'bn'`), read with `torch.load(..., weights_only=False)`; the launch command line was `python -u anchor.py train --epochs 30 --time-budget-min 10 --val-every 1 --val-batches 8 --save-every 1 --out anchor_v9.pth` |
| `z 先验预热: z std 均值 0.1126 [0.0990, 0.1352] (旧 sigma 均值 0.5000)`, `sigma 预热: 潜层 std 0.6537` | `anchor_v9.log` lines 5–6 |
| the log's `bpp` field is analytic (`bits_y + bits_z` over `n_pix`), not container bytes | `anchor.py` line 622 and 656–657 |

## Reference quality (§4.2)

| claim | source |
|---|---|
| `anchor_v5` @256 independent: 15988.0 B y + 7281.5 B z + 35.5 B other = **23305.5 B/img**, **21.13 dB**, bpp 2.845 | `anchor_bytes_report.md` §2 table; `anchor_bytes_anchor_v5_sz256.txt`; `anchor_bytes_summary.txt` line 30 |
| `anchor_v5` training log ends at **3.2064 bpp** analytic after 360 epochs | `anchor_v5.log` last line |
| `normab_gn_best` @256: 12462.6 B/img, 16.83 dB, bpp 1.521 | `anchor_bytes_normab_gn_best_sz256.txt` |
| z stream ≈31 % of shipped bytes | `anchor_bytes_summary.txt` (31.3 %); `fix_round_report.md` §1 (31.6 %) |
| JPEG q85 at 256 px is ~10–15 kB; JPEG at ~10.2 kB scores 34.53 dB | `flowers_22epoch_recovered.log` lines 51–53 |

## The six-bug audit (§5)

| claim | source |
|---|---|
| old `evaluate` printed bits as bytes; 232 296 "B" was really 23 320 B (2.847 bpp); overstatement 9.96× | `fix_round_report.md` §1 |
| dead zone: 29.8974 bits with `d(rate)/dlogσ = 0.0` for `t ≥ ~5.5`; at `t=10` now 69.6909 with −131.6; float64 agreement 7.7e-12 bits | `fix_round_report.md` §2; `_verify_fix2.py` |
| `anchor_v7` analytic y-rate 387 034.8 → 20 249 102 bits/img (+5132 %) | `fix_round_report.md` §2 |
| grid: 48 pts [0.01, 100] → 160 pts [1e-4, 1e6]; outside-grid 9.374 %/23.601 %/5.467 % → 0.000 %; normab_gn y stream −24.71 % | `fix_round_report.md` §3; `_verify_fix3.py` |
| z prior: 97 301.9 → 15 413.4 bits/img (−84.2 %); σ 0.0566 vs z std 1.2594; y prior 325 952 → 167 752 bits/img | `fix_round_report.md` §4; `recalibrate_prior.py` |
| σ mismatch put **9.34 %** of pixels in the wrong bucket (9.06 % have `|h_s| > 8`); now max abs diff **0.0** at 256 px, at 128 px into a 256 model, and for partial keys | `fix_round_report.md` §5; `_verify_fix5.py` |
| deployed smoke test: 24 292 / 7 496 / 3 164 B keys → 16.26 dB; rejects wrong-hash and truncated keys | `fix_round_report.md` §5; `service/_smoke_codec.py` |
| BatchNorm 17.19 dB vs GroupNorm 15.54 dB vs none collapsed (bpp 0.0000, y_max 0, 10.78 dB) | `normab_bn.log`, `normab_gn.log`, `normab_none.log`; `normab_done.txt` |
| `--noise-floor 0` raises the rate ~75 % | `anchor.py` `--noise-floor` help text |
| time-averaging 0.0165 vs last-step 0.0278 MSE | pre-`sae_rd.py` ablation recorded in the previous README revision; the two numbers are not reproducible from a script in this repo |

## The falsification (§7)

| claim | source |
|---|---|
| ratio tables (model prior / fitted Gaussian / oracle per-image / oracle pooled) for `learned` and `sub`, `anchor_v5` @256 and @128, `anchor_v7` @256 | `residual_sub_report.md` §3 |
| **100 % of 42 pairs** worse; quartiles 1.246 / 1.283, min 1.213, max 1.360 | `residual_sub_report.md` §3 |
| scheme-level `sub` oracle +22.99 % (v5@256), +23.78 % (v7@256), +22.25 % (v5@128); scheme-level `learned` oracle +4.57 % … +58.54 % | `residual_sub_report.md` §3, §5 |
| `learned` coded-latent inflation 10.119× (v7), reduced to 1.360× by `sub`; v5 1.306× → 1.301× | `residual_sub_report.md` §4 |
| pixel correlation 0.74 vs latent correlation 0.12–0.33; best global projection R² = **0.063** (0.022–0.112); out-of-class control 0.01–0.06; mean-shift-only saves 1.001–1.025× | `residual_sub_report.md` §4; `_diag_residual_corr.py` |
| `std(y_i − y_ref) ≈ √2·std(y)` regime | `residual_sub_report.md` §4 |
| raw byte tables that the above are derived from | `anchor_bytes_report.md`; `_fals_v5_256.txt`, `_fals_v5_128.txt`, `_fals_v7_256.txt` |

## The A/B split (§3.5, §6)

| claim | source |
|---|---|
| 36 encoder + 36 decoder tensors, 28 unique to each side, `model_hash 1a5f6470c5592f19` | `models/manifest.json`; `service/README.md` §2.1 |
| `h_s` is in both halves; `h_a` is encoder-only | `codec/model_io.py` `ENCODER_PREFIXES` / `DECODER_PREFIXES` |
| container header is 36 bytes, key format v3, one stream (z then y) | `codec/rd_codec.py` `HDR_FMT`, module docstring |
| legacy fixed-rate key: `T·C·(N/16)²` bits + 14-byte v2 header | `sae_rd.py::key_bits`; `sae_model.py` `KEY_HEADER_FMT`; §1 table of the previous README revision |
| fixed-rate key sizes 2.57 / 5.14 / 10.25 / 20.5 / 41.0 kB | previous README revision (derived from the size formula, not a fresh measurement) |
| container isolation verified by exec/ls and by 8001/8002 being unreachable | `service/README.md` §7; `service/evidence/docker_e2e_test.txt`, `service/evidence/host_e2e_test.txt` |

## Unverified / inconsistent, stated rather than smoothed over

1. **`21.13 dB @ 0.948 bpp`.** This pairing appears in earlier notes but does not exist in
   any file here. 21.13 dB is `anchor_v5`'s container PSNR (2.845 bpp container); the
   ≈0.95 bpp figure matches the *analytic* rate in `anchor_v7.log`
   (ep 550: 0.9473 bpp, real validation 21.08 dB) and in `normab_gn.log` (ep 14:
   0.9473 bpp, 13.90 dB — different model). The README therefore quotes
   **21.13 dB @ 2.845 bpp** and warns against mixing protocols.
2. **`19.99 dB @ 0.7671 bpp`** for the short run: the first epoch present in the log is
   `19.77 dB @ 0.7559 bpp` (`anchor_v9.log` line 8), and the log ran on to epoch 6
   (best 21.01 dB @ 0.6283 bpp at epoch 4). The README reports the logged progression and
   mentions the 19.99/0.7671 figure as a separate launch of the same command.
3. **The short run's resolution.** The run in `anchor_v9.log` uses the argparse defaults
   (**128 px / C=24 / T=6**, confirmed from `anchor_v9.pth` metadata), not 256 px. Its
   numbers must not be compared against the 256 px reference in §4.2 without saying so.
4. **`anchor_v9.log` is a live file.** It is being appended by a running
   `anchor.py train --epochs 30 --time-budget-min 10 …` process; more epochs will appear,
   and the README's "latest epoch" is a snapshot, not a final value.
5. **Time-averaging ablation (0.0165 vs 0.0278 MSE)** comes from the previous README
   revision; no script in the current tree reproduces it. The legacy fixed-rate key-size
   table is *consistent* with `sae_rd.py::key_bits` and with `sae_model.bpp() =
   bytes·8/(H·W·3)`, but was not re-measured by running `compress.py`.
6. **`anchor_bytes_normab_gn_best_sz128.json`** was overwritten by an
   `anchor_bytes.py --selftest` run (see the caveats at the end of `fix_round_report.md`),
   so its numbers do not match the tables in `anchor_bytes_report.md`.
