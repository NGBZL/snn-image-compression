# -*- coding: utf-8 -*-
"""sync.py —— 把 service/ 下的源码同步到 snn_ab 根目录的 a_encoder/ b_decoder/ gateway/

为什么要两份：
    Docker 构建上下文必须是 snn_ab 根目录（要 COPY codec/ sae_rd.py anchor.py），
    所以 Dockerfile 只能 COPY 根目录下存在的路径；宿主机直测也从根目录启动最稳。
    service/ 是便于阅读的源码目录，根目录下的是部署/运行副本。

用法：python service\\sync.py
"""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PAIRS = [
    ("a_encoder/app.py", "a_encoder/app.py"),
    ("b_decoder/app.py", "b_decoder/app.py"),
    ("gateway/app.py", "gateway/app.py"),
    ("gateway/static/index.html", "gateway/static/index.html"),
]


def main():
    for src_rel, dst_rel in PAIRS:
        src = os.path.join(HERE, src_rel.replace("/", os.sep))
        dst = os.path.join(ROOT, dst_rel.replace("/", os.sep))
        if not os.path.isfile(src):
            print(f"[skip] 源不存在: {src}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[ok] {src_rel} -> {dst_rel}  ({os.path.getsize(dst)} B)")


if __name__ == "__main__":
    main()
