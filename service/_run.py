# -*- coding: utf-8 -*-
"""_run.py —— 宿主机本地直测用的启动器（不是服务的一部分）

用途
    `codec` / `sae_rd` / `anchor` 都在 snn_ab 根目录下，必须先把根目录塞进 sys.path
    才能 import。Docker 镜像里我们把根目录当成 WORKDIR，不需要这个脚本；
    但在 Windows 宿主机上直接跑 uvicorn 时，用这个包一层最省事。

用法
    python service\\_run.py  a        --port 8001
    python service\\_run.py  b        --port 8002
    python service\\_run.py  gateway  --port 8000
    可选：--models ./models --runs ./runs --a-url ... --b-url ...
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "a_encoder"), os.path.join(ROOT, "b_decoder")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description="宿主机直测启动器")
    ap.add_argument("target", choices=["a", "b", "gateway"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--a-url", default=os.environ.get("A_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--b-url", default=os.environ.get("B_URL", "http://127.0.0.1:8002"))
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()

    os.environ["MODELS_DIR"] = os.path.abspath(a.models)
    os.environ["RUNS_DIR"] = os.path.abspath(a.runs)
    os.environ["A_URL"] = a.a_url
    os.environ["B_URL"] = a.b_url
    # 宿主机上没有 Docker 的多阶段 COPY，直接指定绝对路径最稳
    os.environ.setdefault("ENC_CKPT",
                          os.path.join(os.path.abspath(a.models), "a", "encoder.pth"))
    os.environ.setdefault("DEC_CKPT",
                          os.path.join(os.path.abspath(a.models), "b", "decoder.pth"))

    import uvicorn

    port = a.port or {"a": 8001, "b": 8002, "gateway": 8000}[a.target]
    app = {"a": "app:app", "b": "app:app", "gateway": "app:app"}[a.target]
    cwd = {"a": os.path.join(ROOT, "a_encoder"),
           "b": os.path.join(ROOT, "b_decoder"),
           "gateway": os.path.join(ROOT, "gateway")}[a.target]
    os.chdir(cwd)
    sys.path.insert(0, cwd)
    print(f"[_run] target={a.target} cwd={cwd} MODELS_DIR={os.environ['MODELS_DIR']} "
          f"A_URL={os.environ['A_URL']} B_URL={os.environ['B_URL']}", flush=True)
    uvicorn.run(app, host=a.host, port=port, reload=a.reload, log_level="info")


if __name__ == "__main__":
    main()
