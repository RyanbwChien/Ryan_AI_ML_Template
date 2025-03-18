# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 23:17:12 2025

@author: user
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

# ValueError: You cannot pass both `shape` and `batch_shape` at the same time.
Input(shape=(10,),batch_size=20, batch_shape = (20,10))
Input(batch_shape = (20,10))

# Dense 沒有batch_size or batch_shape 參數
Dense(units=10,input_dim=10)
Dense(units=10,input_shape=(10,)) 


# 這表示每個輸入樣本的形狀是 (10,)，即每個輸入樣本是長度為 10 的一維向量。
Dense(units=10,input_shape=(10,))

#input_shape=(10, 10)，表示每個輸入樣本是 10x10 的二維矩陣。這意味著每一個樣本有 10 行和 10 列，總共有 100 個元素。
Dense(units=10,input_shape=(10,10))


# =============================================================================
# Input shape
# 
# N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
# 
# Output shape
# 
# N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
# 
# =============================================================================
