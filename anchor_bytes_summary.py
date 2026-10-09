# -*- coding: utf-8 -*-
"""anchor_bytes_summary.py -- read the anchor_bytes_*.json outputs and print one
consolidated table.  No model, no GPU, no data.

Usage: python anchor_bytes_summary.py [glob]
"""
from __future__ import annotations

import glob
import json
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

HDR = 36


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def line(d, role):
    r = d["per_role"][role]
    return r


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "anchor_bytes_*_sz*.json"
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no files match {pat}")
    print("# anchor_bytes summary\n")
    print("Units: bytes per image for PROTO B (deployment container); 'A bits' = the")
    print("anchor.py::evaluate protocol, which counts BITS and omits z.\n")
    for p in files:
        d = load(p)
        A = d["per_role"]
        P = d["pooled"]
        g = d["group_size"]
        print("=" * 118)
        print(f"## {d['tag']}  (weights={d['weights']}, input {d['image_size']}x{d['image_size']}, "
              f"{d['n_groups']} groups x {g})")
        print(f"{'role':<7}{'n':>4}{'y B':>10}{'z B':>9}{'hdr':>6}{'total B':>10}{'PSNR':>7}"
              f"{'y bits':>9}{'z bits':>9}{'z%':>6}{'sym std':>9}{'sym max':>9}"
              f"{'eff sig':>9}{'eff max':>10}")
        for role in ("indep", "anchor", "resid"):
            r = A[role]
            tot = r["B_y_nbytes"] + r["B_z_nbytes"] + HDR
            print(f"{role:<7}{r['n']:>4}{r['B_y_nbytes']:>10.1f}{r['B_z_nbytes']:>9.1f}"
                  f"{HDR:>6}{tot:>10.1f}{r['psnr']:>7.2f}{r['B_y_bits']:>9.0f}"
                  f"{r['B_z_bits']:>9.0f}"
                  f"{100 * r['B_z_bits'] / max(1e-9, r['B_y_bits'] + r['B_z_bits']):>5.1f}%"
                  f"{r['sym_std']:>9.2f}{r['sym_absmax']:>9.0f}"
                  f"{r['eff_sigma_med']:>9.4f}{r['eff_sigma_max']:>10.1f}")
        print(f"{'':7}{'oracle (pooled hist) bits/img':>34}{'fitted Gauss bits/img':>24}"
              f"{'model analytic bits/img':>26}")
        for role in ("indep", "anchor", "resid"):
            r = A[role]
            print(f"{role:<7}{P[role]['pooled_oracle_bits_per_img']:>34.0f}"
                  f"{r['refit_bits']:>24.0f}{r['ana_y_bits']:>26.0f}")
        print(f"{'z:':<7}{'real':>10}{'model ana_z':>14}{'fitted(pooled)':>17}"
              f"{'oracle(pooled)':>17}{'z_prior.sigma':>16}{'z std pooled':>15}")
        for role in ("indep", "anchor", "resid"):
            r = A[role]
            print(f"{role:<7}{r['B_z_bits']:>10.0f}{r['ana_z_bits']:>14.0f}"
                  f"{P[role]['z_refit_pooled_bits_per_img']:>17.0f}"
                  f"{P[role]['z_oracle_pooled_bits_per_img']:>17.0f}"
                  f"{'':>16}{P[role]['z_pooled_std']:>15.3f}")
        s = d["scheme"]
        g = d["group_size"]
        di, dr = A["indep"], A["resid"]
        dy = dr["B_y_nbytes"] - di["B_y_nbytes"]
        dz = dr["B_z_nbytes"] - di["B_z_nbytes"]
        tot = abs(dy) + abs(dz)
        print(f"  resid - indep = {dy + dz:+.0f} B/img   (y {dy:+.0f} B = "
              f"{100 * abs(dy) / max(1e-9, tot):.0f}%, z {dz:+.0f} B = "
              f"{100 * abs(dz) / max(1e-9, tot):.0f}%)")
        print(f"  sigma field (per-pixel, prior.sigma()*hyperprior): p1 {di['eff_sigma_p1']:.4f} "
              f"p10 {di['eff_sigma_p10']:.4f} p50 {di['eff_sigma_med']:.4f} "
              f"p90 {di['eff_sigma_p90']:.2f} p99 {di['eff_sigma_p99']:.0f} "
              f"max {di['eff_sigma_max']:.0f}   symbol std {di['sym_std']:.2f}   "
              f"{100 * di['eff_over_gridmax']:.1f}% of pixels above _SIGMA_GRID top (100)")
        print(f"  SCHEME PROTO B  independent {s['protoB_independent_B']:>9.0f} B   "
              f"anchor+resid {s['protoB_anchor_resid_B']:>9.0f} B   "
              f"delta {100 * (s['protoB_anchor_resid_B'] / s['protoB_independent_B'] - 1):+.2f}%")
        print(f"  SCHEME PROTO A  independent {s['protoA_independent_B']:>9.0f} B   "
              f"anchor+resid {s['protoA_anchor_resid_B']:>9.0f} B   "
              f"delta {100 * (s['protoA_anchor_resid_B'] / s['protoA_independent_B'] - 1):+.2f}%")
        # oracle-calibrated scheme: what a per-channel model fitted to the actual
        # symbols would cost (lower bound on the rate the anchor idea must beat)
        oi = g * P["indep"]["pooled_oracle_bits_per_img"]
        oa = P["anchor"]["pooled_oracle_bits_per_img"] + (g - 1) * P["resid"]["pooled_oracle_bits_per_img"]
        print(f"  SCHEME ORACLE   independent {oi / 8:>9.0f} B   anchor+resid {oa / 8:>9.0f} B   "
              f"delta {100 * (oa / oi - 1):+.2f}%   (empirical symbol entropy, side info free)")
        print(f"  SIMPLE FIX (recalibrate the priors to the measured symbol scale, no side info):")
        fy = P["indep"]["y_refit_pooled_bits_per_img"] if "y_refit_pooled_bits_per_img" in P["indep"] \
            else A["indep"]["refit_bits"]
        y_now = A["indep"]["B_y_nbytes"]
        z_now = A["indep"]["B_z_nbytes"]
        z_fix = P["indep"]["z_refit_pooled_bits_per_img"] / 8
        y_fix = A["indep"]["refit_bits"] / 8
        print(f"      indep total {y_now + z_now + HDR:>8.0f} B  ->  y {y_fix:>8.0f} + z {z_fix:>7.0f} + {HDR} "
              f"= {y_fix + z_fix + HDR:>8.0f} B   ({100 * ((y_fix + z_fix + HDR) / (y_now + z_now + HDR) - 1):+.1f}%)")
    print()


if __name__ == "__main__":
    main()
