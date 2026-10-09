# -*- coding: utf-8 -*-
"""_probe_mirror.py —— 验证某个 Docker Hub 镜像代理能否真的拉到 python:3.11-slim。

Docker Hub 直连不通时，用这个脚本判断能不能通过镜像源构建。

    python service\\_probe_mirror.py docker.m.daocloud.io python 3.11-slim
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "docker.m.daocloud.io"
REPO = sys.argv[2] if len(sys.argv) > 2 else "library/python"
TAG = sys.argv[3] if len(sys.argv) > 3 else "3.11-slim"

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])


def get(url, token=None, accept=ACCEPT):
    req = urllib.request.Request(url)
    req.add_header("Accept", accept)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=15)


def main():
    print(f"探测 {HOST} / {REPO}:{TAG}")
    try:
        r = get(f"https://{HOST}/v2/")
        print("  /v2/ ->", r.status)
    except urllib.error.HTTPError as e:
        print("  /v2/ ->", e.code, e.headers.get("Www-Authenticate", "")[:120])
        realm = None
        hdr = e.headers.get("Www-Authenticate", "")
        for part in hdr.replace("Bearer ", "").split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() == "realm":
                    realm = v.strip().strip('"')
        if not realm:
            print("  拿不到 token realm，放弃")
            return 1
        tok_url = f"{realm}?service={HOST}&scope=repository:{REPO}:pull"
        try:
            tr = get(tok_url, accept="application/json")
            tok = json.loads(tr.read()).get("token")
            print(f"  匿名 token: {'拿到' if tok else '没有'}")
        except Exception as e2:                              # noqa: BLE001
            print("  取 token 失败:", type(e2).__name__, str(e2)[:120])
            return 1
        try:
            mr = get(f"https://{HOST}/v2/{REPO}/manifests/{TAG}", token=tok)
            data = json.loads(mr.read())
            print(f"  manifest -> {mr.status}  mediaType={data.get('mediaType')}  "
                  f"平台数={len(data.get('manifests', []))}")
            if data.get("manifests"):
                for m in data["manifests"][:6]:
                    p = m.get("platform", {})
                    print(f"     - {p.get('os')}/{p.get('architecture')} {m.get('mediaType','')[:40]}")
            print("  >>> 这个源可用")
            return 0
        except urllib.error.HTTPError as e3:
            print(f"  拉 manifest 失败: HTTP {e3.code} {e3.read()[:200]!r}")
            return 1
    except Exception as e:                                   # noqa: BLE001
        print("  失败:", type(e).__name__, str(e)[:120])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
