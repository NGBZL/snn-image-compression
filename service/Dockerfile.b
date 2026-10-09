# =============================================================================
# B 机：云侧解码器（只有解码器半区）
#   构建上下文同样是 snn_ab 根目录：
#     docker build -f service/Dockerfile.b -t snn-ab-b .
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install --no-cache-dir \
        "numpy<2.3" snntorch constriction pillow fastapi "uvicorn[standard]" \
        python-multipart httpx

COPY codec/ ./codec/
COPY sae_rd.py sae_model.py anchor.py ./
# 注意：只拷 b_decoder，**不拷 a_encoder** —— B 机拿不到编码器代码
COPY b_decoder/ ./b_decoder/
# B 机需要 manifest 的 model_hash 做钥匙校验；同样只拷元数据，不带权重
COPY models/manifest.json ./manifest.json

ENV MODELS_DIR=/models \
    DEC_CKPT=/models/decoder.pth \
    MANIFEST_PATH=/app/manifest.json \
    DEVICE=cpu \
    PYTHONPATH=/app

WORKDIR /app/b_decoder
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
