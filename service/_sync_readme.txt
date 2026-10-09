# -*- coding: utf-8 -*-
这个目录是“源码目录”。为兼容两种运行方式，服务代码在 snn_ab 根目录下也有一份副本：

  snn_ab/a_encoder/app.py        <- service/a_encoder/app.py
  snn_ab/b_decoder/app.py        <- service/b_decoder/app.py
  snn_ab/gateway/app.py          <- service/gateway/app.py
  snn_ab/gateway/static/index.html <- service/gateway/static/index.html

原因：
  * Docker 构建上下文必须是 snn_ab 根目录（要 COPY codec/ sae_rd.py anchor.py），
    Dockerfile 里的 `COPY a_encoder/ ./a_encoder/` 只能看到根目录下的文件；
  * 宿主机直测时统一从根目录启动，import codec 也最稳。

两边内容完全一致，改完用下面这条命令同步（PowerShell）：

  Copy-Item service\a_encoder\app.py            a_encoder\app.py -Force
  Copy-Item service\b_decoder\app.py            b_decoder\app.py -Force
  Copy-Item service\gateway\app.py             gateway\app.py -Force
  Copy-Item service\gateway\static\index.html  gateway\static\index.html -Force

或者直接跑：python service\sync.py
