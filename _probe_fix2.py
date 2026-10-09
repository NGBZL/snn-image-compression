# -*- coding: utf-8 -*-
"""Probe: erf saturation + log_ndtr backward behaviour (CPU only)."""
import math
import torch

print("torch", torch.__version__)
SQRT2 = math.sqrt(2.0)


def phi32(x):
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


print("\n--- float32 erf/Phi saturation ---")
for t in [2.0, 3.0, 3.5, 3.9, 4.0, 4.5, 5.0, 5.3, 6.0, 10.0]:
    a = phi32(torch.tensor(-t + 0.5))
    b = phi32(torch.tensor(-t - 0.5))
    print(f"  t={t:5.1f}  Phi(t+0.5)={float(a):.10e}  Phi(t-0.5)={float(b):.10e}  diff={float(a-b):.3e}")

print("\n--- float64 erf/Phi saturation ---")
for t in [6.0, 8.0, 9.0, 10.0, 20.0, 37.0, 38.0, 40.0]:
    x1 = torch.tensor(t + 0.5, dtype=torch.float64)
    x2 = torch.tensor(t - 0.5, dtype=torch.float64)
    p = 0.5 * (1.0 + torch.erf(x1 / SQRT2)) - 0.5 * (1.0 + torch.erf(x2 / SQRT2))
    print(f"  t={t:5.1f}  diff={float(p):.6e}  bits={float(-torch.log2(p)):.4f}")

print("\n--- log_ndtr forward/backward at large |x| ---")
for t in [5.0, 10.0, 50.0, 1000.0, 1e4, 1e6]:
    x = torch.tensor([t], dtype=torch.float32, requires_grad=True)
    y = torch.special.log_ndtr(x)
    g, = torch.autograd.grad(y, x)
    print(f"  x={t:9.1f}  log_ndtr={float(y):.6e}  dlog_ndtr/dx={float(g):.6e}  (Mills~{t:.1f})")

print("\n--- current code: rate grad wrt log_sigma for |t|>5 ---")


class OldPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.log_sigma = torch.nn.Parameter(torch.tensor([math.log(0.1)]))

    def sigma(self, extra=None):
        s = self.log_sigma.exp().clamp(1e-4, 1e6)
        return s if extra is None else (s * extra).clamp(1e-4, 1e6)

    def bits(self, y_hat, step, extra_sigma=None):
        yn = y_hat / step
        sig = self.sigma(extra_sigma)
        p = (phi32((yn - self.mu + 0.5) / sig) - phi32((yn - self.mu - 0.5) / sig)).clamp_min(1e-9)
        return -torch.log2(p)


for yv in [0.6, 1.0, 5.0, 10.0, 100.0]:
    pr = OldPrior()
    y = torch.tensor([[[[[yv]]]]], dtype=torch.float32)
    r = pr.bits(y, 1.0).sum()
    g, = torch.autograd.grad(r, pr.log_sigma)
    print(f"  y={yv:7.1f}  t={yv/0.1:8.1f}  rate={float(r):10.4f} bits  d rate/d log_sigma={float(g):.6e}")
