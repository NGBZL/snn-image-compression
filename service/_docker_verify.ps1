# -*- coding: utf-8 -*-
"""_docker_verify.ps1 —— Docker 部署验证（用临时容器从 backend 网络内部打 A/B/Gateway）

为什么不能像宿主机那样直接打 127.0.0.1:8001/8002：
    A/B 按设计**不发布端口**，宿主机根本访问不到它们，这是隔离性的一部分。
    所以验收脚本要跑在 backend 网络内部，用容器名访问 a-encoder:8000 / b-decoder:8000。
    临时容器用 gateway 镜像（里面有 httpx + pillow + numpy，够跑 _e2e_test.py）。

用法（在 snn_ab 根目录）：
    powershell -ExecutionPolicy Bypass -File service\\_docker_verify.ps1
"""
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$net  = 'snn-ab_backend'
$img  = 'snn-ab-gateway:local'
$name = 'snn-ab-verify'

Write-Output '=== 0. 清理旧临时容器 ==='
docker rm -f $name 2>$null | Out-Null

Write-Output '=== 1. 从 backend 网络内部跑端到端验收 ==='
# -w /py 是临时容器的挂载点；路径全部用容器内路径
docker run --rm --name $name `
  --network $net `
  -v "${root}\service:/py/service:ro" `
  -v "${root}\data:/py/data:ro" `
  -v "${root}\runs:/py/runs" `
  -w /py `
  $img `
  python -u /py/service/_e2e_test.py `
    --a-url http://a-encoder:8000 `
    --b-url http://b-decoder:8000 `
    --gw-url http://gateway:8000 `
    --runs-dir /py/runs
$rc = $LASTEXITCODE

Write-Output ''
Write-Output '=== 2. 从宿主机只打 Gateway（应该成功） ==='
try {
  $r = Invoke-WebRequest 'http://localhost:8080/' -UseBasicParsing -TimeoutSec 10
  Write-Output "GET http://localhost:8080/ -> $($r.StatusCode), 含标题: $($r.Content -match 'SNN 图像压缩')"
} catch { Write-Output "FAIL: $($_.Exception.Message)" }

Write-Output ''
Write-Output '=== 3. 从宿主机打 A/B 端口（应该失败，它们没发布端口） ==='
foreach ($p in 8001, 8002) {
  try {
    Invoke-WebRequest "http://localhost:$p/health" -UseBasicParsing -TimeoutSec 5 | Out-Null
    Write-Output "端口 $p 竟然可达 —— 与「不发布端口」的设计不符"
  } catch { Write-Output "端口 $p 不可达（符合预期）" }
}

Write-Output ''
Write-Output "端到端脚本退出码: $rc"
exit $rc
