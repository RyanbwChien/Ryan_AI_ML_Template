# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 02:24:22 2025

@author: user
"""

import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


model = Sequential([
    Dense(10),
    Dense(5),    
    Dense(2),   
    ])

# 在tf keras Sequential model 在還沒輸入Input前，model參數是空的，除非先給初始Input 讓模型才知道第一層參數量多少，可以初始化參數

X = tf.random.normal((100,5))
model(X)


# 整個模型各層的參數
model.weights
model.weights[0].trainable #各層參數都是一個物件，有trainable屬性
# 整個模型各層的可訓練的參數
model.trainable_variables


model.layers
