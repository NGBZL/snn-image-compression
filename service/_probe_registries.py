# -*- coding: utf-8 -*-
"""_probe_registries.py —— 探一下当前网络能不能连上常见 Docker 镜像源。

用于说明 Docker 构建到底卡在哪一步（拉不到基础镜像 / 拉不到 torch）。
"""
from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request

CANDS = [
    ("registry-1.docker.io", "https://registry-1.docker.io/v2/"),
    ("auth.docker.io", "https://auth.docker.io/token"),
    ("docker.m.daocloud.io", "https://docker.m.daocloud.io/v2/"),
    ("dockerproxy.com", "https://dockerproxy.com/v2/"),
    ("docker.1panel.live", "https://docker.1panel.live/v2/"),
    ("hub-mirror.c.163.com", "https://hub-mirror.c.163.com/v2/"),
    ("mirror.ccs.tencentyun.com", "https://mirror.ccs.tencentyun.com/v2/"),
    ("quay.io", "https://quay.io/v2/"),
    ("ghcr.io", "https://ghcr.io/v2/"),
    ("registry.cn-hangzhou.aliyuncs.com", "https://registry.cn-hangzhou.aliyuncs.com/v2/"),
    ("pypi.org", "https://pypi.org/simple/"),
    ("download.pytorch.org", "https://download.pytorch.org/whl/cpu/"),
]

print(f"{'host':38s} {'DNS':10s} {'TCP443':8s} HTTP")
for host, url in CANDS:
    try:
        ip = socket.gethostbyname(host)
        dns = ip
    except Exception as e:                                   # noqa: BLE001
        dns = "FAIL"
        ip = None
    tcp = "-"
    if ip:
        s = socket.socket()
        s.settimeout(5)
        t0 = time.time()
        try:
            s.connect((ip, 443))
            tcp = "ok"
        except Exception:                                    # noqa: BLE001
            tcp = "timeout"
        finally:
            s.close()
    http = "-"
    if tcp == "ok":
        try:
            r = urllib.request.urlopen(url, timeout=8)
            http = str(r.status)
        except urllib.error.HTTPError as e:
            # /v2/ 返回 401 是正常的（要 token），说明这个源是活的
            auth = e.headers.get("Www-Authenticate", "")[:50]
            http = f"HTTP {e.code} {auth}"
        except Exception as e:                               # noqa: BLE001
            http = f"FAIL {type(e).__name__}"
    print(f"{host:38s} {dns:10s} {tcp:8s} {http}")
