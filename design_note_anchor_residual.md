# Design note: what to do about the anchor + residual parameterisation

**Status: analysis + proposal only. No architecture change was made.** All numbers below are
measured (see `anchor_bytes_report.md`, plus the post-fix re-measurements noted inline).

> **Outcome update (see `residual_sub_report.md`).** Option (a) has since been implemented as
> `anchor.py --residual-mode sub` and the experiment below was run. **It is falsified**: the
> ORACLE ratio is still `> 1.0` in every configuration measured (+23 % … +24 % at 256², 100 %
> of 42 same-class pairs worse, spread ±5 %), so by the note's own criterion the correct
> move is the fallback **option (c)** for these checkpoints. The structural subtraction does
> remove the *parameterisation* pathology — v7's coded-latent std inflation drops from
> **10.1x to 1.36x**, and v7's oracle damage from **+58 % to +24 %** — but it cannot remove
> the underlying one: for same-class pairs the encoder latents are essentially uncorrelated
> (`R^2` of the best global projection of `y_i` onto `y_ref` is **0.063**, while the pixel
> correlation is 0.74). If `y_i` and `y_ref` are decorrelated then
> `std(y_i - y_ref) ≈ sqrt(2)·std(y_i)`, so no algebraic subtraction can win; measured 1.30x.


## The problem in one paragraph

With `ref_wiring="inp"` the reference is *only* concatenated to the encoder input as 32 extra
channels (`anchor.py::SAE_Anchor._prepare_input`). Nothing in the architecture or the loss forces
the coded quantity to be `y_i - y_ref`, so the encoder keeps emitting its own latent — and emits a
*larger, higher-entropy* one when a reference is present: measured coded-latent std rises
66.7 → 87.2 (v5 @256), 7.25 → 72.99 (v7 @256, **10.1×**) and 106.0 → 117.5 (normab), while PSNR
gets *worse* (21.13 → 20.73 dB for v5). The rate loss survives an **oracle** entropy model
(per-channel empirical histogram, free side information) in **all six** tested configurations,
`+4.7 % … +58 %`. So it is not an entropy-model bug: the scheme really does pay more bits for a
latent that carries no less information. The apparent wins reported earlier (e.g. v7 −37 % in the
real container) are artefacts of the independent path being mis-coded — exactly the bugs fixed in
items 1–5 of this round (`anchor.py::evaluate` counted bits as bytes and dropped the z stream;
`z_prior` was never calibrated; the sigma grid stopped at 100 while the hyperprior reaches 1e6).

The two facts that decide the design are:

1. **A useless reference must not cost anything.** Today it does.
2. **The decoder already knows how to add a reference back** (`SAE_Anchor.decode_with_ref` does
   `y_full = ref + r_hat`), but the *encoder* is under no obligation to make `r_hat` small.

## Options

### (a) Make the difference structural: code `r = y_i - y_ref` explicitly

Produce the anchor latent `y_ref` as usual, then define the coded quantity as
`r = quantise(y_i) - quantise(y_ref)` and reconstruct with `y = quantise(y_ref) + r_hat`
(the architecture keeps the reference as an *input hint* for the entropy model / encoder, but the
subtraction and addition happen outside the learned path, exactly like the existing
`ref + r_hat` in `decode_with_ref`).

* **Failure mode addressed:** "the network never learned the cancellation". Cancellation is no
  longer learned — it is algebra. If the reference is useless, `r` degenerates to `y_i` and the
  scheme costs at most what independent coding costs (plus the entropy-model mismatch, which is now
  measurable). The decoder never needs to trust the encoder for correctness, only for rate.
* **Trade-off:** `r` is no longer distributed like a latent the encoder "wants", so the
  per-channel prior must be fitted to `r` (with the `warmup_sigma` / `recalibrate_prior.py` path
  from this round this is a one-liner, but it must be done or the tail symbols pay the floor). A
  bad reference also injects its own quantisation error into the reconstruction; with
  `step = 1.0` and v5 latent std ≈ 67 that error is small, but it must be measured at matched
  quality, not at matched λ.
* **Cost:** ~20 lines. Changes no weights; the codec (item 5) codes whatever latent it is handed,
  and its round-trip is now bit-exact (max abs diff 0.0 in-process, including the 128px-input case
  that used to break).

### (b) Keep the learned path, add a skip/identity connection

Feed the reference in as now, but wire the final conv so that the output starts at
`y_i - y_ref` (`h = GDN(conv4(f)) + ref_proj(ref)` with `ref_proj` initialised to identity rather
than zero, or an explicit `- ref` residual branch).

* **Failure mode addressed:** the *initialisation* pathology (the current `ref_proj` is zeroed, so
  training starts from "no reference" and the encoder has no gradient pressure to cancel).
* **Trade-off:** this only biases the solution; the encoder can still spend rate on a component
  that the reference already represents, and the loss has no term that punishes double-coding.
  It is strictly weaker than (a) and harder to falsify, because "the network could learn it" is
  always true in principle. Worth keeping as an ablation, not as the fix.
* **Cost:** ~5 lines, but it invalidates existing checkpoints (needs retraining).

### (c) Drop the anchor mechanism entirely

Delete `ref_*`, `GroupedBatchSampler`, `pair_bits_matrix`, `mst_plan`, `eval_allpairs` and the
multi-key container, and spend the complexity budget on the single-image path (which is where the
measured wins are: fix 4 alone removes ~84 % of v5's z stream — 97302 → 15413 bits/img — with zero
side information, and fix 3 removes another 2–25 % of the y stream).

* **Failure mode addressed:** "the whole idea does not pay for itself". This is the right answer if
  and only if (a) still loses under an oracle entropy model.
* **Trade-off:** throws away the only mechanism that exploits inter-image redundancy, and the
  measured ORACLE losses are *losses relative to coding `y_i` with a per-image oracle*, i.e. they
  already include the reference's information. Note also that the anchor path buys real quality at
  a fixed λ for v7 (15.15 → 20.11 dB @256); that is a different operating point, not a rate win,
  and has never been measured at matched PSNR.
* Deployment cost matters too: the group protocol needs the anchor decoded first (the MST order),
  which is a real product constraint the single-image path does not have.

## Recommendation

**Do (a), keep (b) as an ablation, keep (c) as the fallback.** The reason (a) wins is not that it
is likely to give a large win — it is that it is the only option whose *worst case is bounded and
measurable*: with an explicit difference, a useless reference can cost nothing, whereas today an
unhelpful reference actively inflates the latent (10.1× on v7). Making the failure mode impossible
is worth more than a speculative gain, and it is the prerequisite for the experiment below to mean
anything.

Order of work: land fixes 1–5 first (done). They are not cosmetic — the old numbers could not
distinguish "the anchor helps" from "the independent path is broken".

## What would falsify the recommendation

One experiment, one dataset split, no training of new architectures beyond the two variants:

1. Take the existing group sampler (8 same-class images/group, `build_data` hold-out), one
   checkpoint family, one λ.
2. Build variant (a): the same weights, but code `r = quantise(y_i) - quantise(y_ref)` with
   `ref = argmin` chosen by the *same* `pair_bits_matrix` cost.
3. Report, per image: `bits(a) / bits(independent)` under (i) the learned entropy model, (ii) a
   per-channel Gaussian fitted to `r` (`recalibrate_prior.py`), and (iii) the **oracle** empirical
   histogram of `r`. Also report PSNR for both.
4. **Falsified if** the oracle ratio at (iii) is still `> 1.0` with a margin larger than the
   per-group spread (the report measures ±5.6…19.3 % spread on 8-image groups, so use ≥ 6 groups
   and quote the paired per-image median, not the group total). In that case the reference carries
   no information the latent does not already carry at this resolution and λ — go to (c).
5. **Supported if** the oracle ratio is `≤ 1.0` while the learned-model ratio is `> 1.0`: the
   difference is real, only the entropy model is behind — then invest in the prior (per-image
   fitting, `r`-specific calibration), not in the wiring.

**Result: step 4 happened.** `anchor_v5` @256², 6 groups x 8: paired per-image medians
`bits(sub residual)/bits(independent)` = 1.492 (model prior), 1.244 (fitted Gaussian),
1.262 (oracle per-image), 1.263 (oracle pooled), 100 % of pairs `> 1`; scheme-level
container +41.3 %, oracle +23.0 %. `anchor_v7` @256²: oracle 1.271 median, oracle scheme
+23.8 %, container −33.6 % (the container "win" is still the independent path's miscalibrated
prior, exactly as in the report). Full tables in `residual_sub_report.md`.

Secondary prediction to check in the same run: with (a) the coded-latent std for the residual role
should drop **below** the independent role's std (today it is 1.11–10.1× *above*). If the std falls
but the oracle bits do not, the residual is becoming low-variance but high-entropy (dense small
values), which means the fix should be a GDN/whitening change, not a wiring change.

**Result: the std falls, but not below, and the oracle bits do not fall at all.** v5@256
66.67 -> 86.71 (**1.301x**, was 1.306x), v7@256 7.22 -> 9.82 (**1.360x**, was **10.119x**).
So v7 matches "the std falls but the oracle bits do not" — but the oracle bits do not fall for
the stronger reason that the two latents are decorrelated (`R^2` 0.063), not because the
residual became dense-and-small: the coded residual's zero fraction *rises* (70.7 % -> 74.8 %)
while its std rises 1.30x, i.e. it is exactly what the difference of two independent latents
looks like. A whitening/GDN change cannot create information that the pair does not share.
