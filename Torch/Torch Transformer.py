# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:19:06 2025

@author: USER
"""

import torch
import torch.nn as nn

# d_model 必須是 nhead 的整數倍，這樣才能保證每個頭的大小是均等的。
nn.TransformerEncoderLayer(d_model=256, nhead=8)

nn.TransformerDecoderLayer(d_model=256, nhead=8)

# =============================================================================
# TransformerDecoderLayer(
#   (self_attn): MultiheadAttention(
#     (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
#   )
#   (multihead_attn): MultiheadAttention(
#     (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
#   )
#   (linear1): Linear(in_features=256, out_features=2048, bias=True)
#   (dropout): Dropout(p=0.1, inplace=False)
#   (linear2): Linear(in_features=2048, out_features=256, bias=True)
#   (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#   (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#   (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#   (dropout1): Dropout(p=0.1, inplace=False)
#   (dropout2): Dropout(p=0.1, inplace=False)
#   (dropout3): Dropout(p=0.1, inplace=False)
# )
# =============================================================================

# =============================================================================
# self_attn（Masked Multihead Attention）：
# 這是解碼器層中的自注意力（self-attention）機制。該層在計算注意力時會使用遮罩來確保模型無法看到未來的位置。這是通過傳遞 tgt_mask 來實現的。
# multihead_attn（Cross-Attention）：
# 這層是交叉注意力，通常使用來利用來自編碼器的 memory。它不是 "masked" 的，而是基於編碼器的輸出來進行注意力計算。
# =============================================================================
