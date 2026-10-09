# -*- coding: utf-8 -*-
"""progress.py — 训练进度看板（在一个独立窗口里实时刷新）

读取训练日志 + nvidia-smi，画出进度条、关键指标、ETA 和 GPU 状态。
单独跑，不干扰训练。

    python progress.py --log anchor_v4.log --epochs 440
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EP = re.compile(
    r"ep\s+(\d+)/(\d+)\s*\|\s*L\s*([\d.]+)\s*\|\s*D\s*([\d.]+)\s*\|\s*"
    r"组均码率\s*([\d.]+)\s*bpp\s*\|\s*sigma\s*([\d.]+)\s*\|\s*噪声\s*([\d.]+)\s*\|\s*([\d.]+)s")
CK = re.compile(r"checkpoint 已保存 \(epoch (\d+)\)")
DONE = re.compile(r"已保存 .*\.pth")
BUDGET = re.compile(r"\[时间预算\]")


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
        u, mu, mt, pw, tp = [x.strip() for x in out.split(",")]
        return dict(util=float(u), mem=float(mu), memtot=float(mt), power=float(pw), temp=float(tp))
    except Exception:
        return None


def bar(frac, width=46):
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def hm(sec):
    sec = max(0, int(sec))
    return f"{sec//3600:d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def read_log(path):
    """读日志。PowerShell 的 *> 重定向默认写 UTF-16LE（带 BOM），
    直接按 UTF-8 读会全是乱码、正则一条都匹配不上。这里按 BOM 判断。"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return ""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig", errors="replace")
    return raw.decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="anchor_v4.log")
    p.add_argument("--epochs", type=int, default=440)
    p.add_argument("--interval", type=float, default=4.0)
    a = p.parse_args()

    t_start = time.time()
    while True:
        epochs, losses, rates, sigmas, secs = [], [], [], [], []
        state = {}
        text = read_log(a.log)
        for ln in text.splitlines():
            m = EP.search(ln)
            if m:
                e = int(m.group(1))
                epochs.append(e)
                losses.append(float(m.group(4)))
                rates.append(float(m.group(5)))
                sigmas.append(float(m.group(6)))
                secs.append(float(m.group(8)))
                state = dict(ep=e, total=int(m.group(2)), L=float(m.group(3)),
                             D=float(m.group(4)), rate=float(m.group(5)),
                             sigma=float(m.group(6)), noise=float(m.group(7)),
                             sec=float(m.group(8)))
            elif CK.search(ln):
                state["ck"] = int(CK.search(ln).group(1))
            elif BUDGET.search(ln):
                state["budget"] = True
            elif DONE.search(ln):
                state["done"] = True

        cols = shutil.get_terminal_size((100, 30)).columns
        sys.stdout.write("\x1b[2J\x1b[H")
        print("=" * min(cols, 96))
        print("   SNN 锚图压缩 · 训练进度看板".center(min(cols, 96) - 2))
        print("=" * min(cols, 96))

        if not state:
            print(f"\n  等待训练日志写入…  ({a.log})")
            print(f"  已运行 {hm(time.time()-t_start)}")
            time.sleep(a.interval)
            continue

        ep, total = state.get("ep", 0), state.get("total", a.epochs)
        frac = ep / max(1, total)
        # 用最近 20 轮的平均耗时估 ETA（比全程平均更贴近当前速度）
        avg = (sum(secs[-20:]) / len(secs[-20:])) if secs else 26.0
        eta = (total - ep) * avg
        el = time.time() - t_start

        print(f"\n  [{bar(frac)}]  {frac*100:5.1f}%")
        print(f"   epoch {ep:4d} / {total}     已用 {hm(el)}     预计还需 {hm(eta)}"
              f"     预计总计 {hm(el+eta)}")

        if state.get("done"):
            print("\n  ✅ 训练已完成")
        elif state.get("budget"):
            print("\n  ⏱ 时间预算触发，已提前收尾")
        elif state.get("ck"):
            print(f"\n  最近存盘: epoch {state['ck']}")

        print("\n  ── 损失 / 码率 ──────────────────────────────────────────")
        print(f"   总损失 L      {state.get('L', 0):.4f}")
        print(f"   失真 D        {state.get('D', 0):.4f}      "
              f"(等效 PSNR ≈ {10 * __import__('math').log10(1/max(1e-9, state.get('D', 1))):.2f} dB, 训练集)")
        lam_r = state.get("L", 0) - state.get("D", 0)
        print(f"   率项 λ·R      {lam_r:.4f}")
        print(f"   组均码率      {state.get('rate', 0):.4f} bpp")
        print(f"   sigma         {state.get('sigma', 0):.4f}")
        print(f"   噪声退火      {state.get('noise', 0):.3f}")
        print(f"   单轮耗时      {state.get('sec', 0):.1f} s")

        if len(losses) >= 6:
            d = losses[-1] - losses[-6]
            arrow = "↓ 下降中" if d < 0 else "→ 平台/波动"
            print(f"\n   近 5 轮 D 趋势 {losses[-5]:.4f} -> {losses[-1]:.4f}   {arrow}")

        g = gpu()
        print("\n  ── GPU ────────────────────────────────────────────────")
        if g:
            filled = int(round(g["util"] / 5))
            print(f"   利用率  [{bar(g['util']/100, 24)}] {g['util']:3.0f}%")
            print(f"   显存    {g['mem']:.0f} / {g['memtot']:.0f} MB"
                  f"      功耗 {g['power']:.0f} W      温度 {g['temp']:.0f} °C")
            if g["util"] < 60:
                print("   提示: 利用率偏低，可能是 GPU 被别的程序占用或数据加载成为瓶颈")
        else:
            print("   读取失败")

        print("\n  (Ctrl+C 退出看板，不影响训练)")
        sys.stdout.flush()
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n看板已退出，训练继续运行。")
