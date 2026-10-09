# -*- coding: utf-8 -*-
"""codec —— A/B 分离系统共用的编解码核心

    from codec import load_side, encode_key, decode_key, model_hash_of
"""
from .rd_codec import (CODEC_RD, CODEC_RD_ANCHOR, DEF_KMAX, HDR_SIZE, KEY_MAGIC, KEY_VERSION,
                       decode_key, encode_key, pack_header, unpack_header)
from .model_io import load_side, manifest_of, model_hash_of

__all__ = ["CODEC_RD", "CODEC_RD_ANCHOR", "DEF_KMAX", "HDR_SIZE", "KEY_MAGIC", "KEY_VERSION",
           "encode_key", "decode_key", "pack_header", "unpack_header",
           "load_side", "manifest_of", "model_hash_of"]
