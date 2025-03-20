# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:15:51 2025

@author: user
"""

import tensorflow as tf
import numpy as np
import pandas as pd

#  1. tf.GradientTape() 如何追蹤變數？
# tf.GradientTape() 會自動記錄所有與 losses 計算相關的 TensorFlow 運算，並且追蹤 可微分變數（trainable variables）。

tf.cast(np.ones((3,3)),dtype='float32')

np.array([1,2,3]*5).reshape(3,5)

X = tf.random.normal([10,20], stddev=0.1)
b = tf.ones([1, 20])
Y = X+b

class TFmodel:
    def __init__(self, input_dim:int, neuron_size:list):
        self.parameter = {}
        self.size = len(neuron_size)
        self.grads = None
        for i in range(len(neuron_size)):
            self.parameter[f'b{i+1}'] = tf.Variable(tf.random.normal(shape=(1,neuron_size[i])),name=f'b{i+1}')
            if i==0:                
                self.parameter[f'W{i+1}'] = tf.Variable(tf.random.normal(shape=(input_dim,neuron_size[i])),name=f'W{i+1}')
            else:
                self.parameter[f'W{i+1}'] = tf.Variable(tf.random.normal(shape=(neuron_size[i-1],neuron_size[i])),name=f'W{i+1}')
            
            
    def forward(self,x):
        X_tf = tf.cast(x,dtype=tf.float32)
        iter_ = X_tf 
        for i in range(self.size):
            iter_ = tf.matmul(iter_, self.parameter[f'W{i+1}'])+self.parameter[f'b{i+1}']
            if i < self.size-1:
                iter_ = tf.nn.relu(iter_)
                
        return iter_
    
    def backword(self,x,y):
        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            predict = self.forward(x)
            losses = loss(predict,y)
# =============================================================================
#         在 TensorFlow 中，計算圖是動態創建的，每一次進行反向傳播時（每一次步驟），
#         GradientTape 都會記錄一次運算圖，並且只在當前步驟內有效。這意味著，在一次訓練步驟完成後，GradientTape 會自動釋放計算圖，也就不再記錄梯度了。
# =============================================================================
            
        grads = tape.gradient(losses, list(self.parameter.values())) # tf.GradientTape 依賴計算圖來回溯梯度，而不是 params 的順序。
# =============================================================================
#         tf.GradientTape 不會永久存儲梯度。它記錄運算來構建計算圖，並在呼叫 tape.gradient() 時計算並返回梯度。
# =============================================================================

        
        self.grads = grads
        optimizer.apply_gradients(zip(grads, self.parameter.values()))

# 首刻batch_size split    
def batch(data, batch_size):
    batch_data = []
    num_full_size = len(data) // batch_size
    remain_size = len(data) % batch_size
    for i in range(num_full_size):
        batch_data.append(data[i*batch_size:i*batch_size+batch_size])
    if remain_size != 0:
        batch_data.append(data[num_full_size*batch_size:num_full_size*batch_size+remain_size])
    return batch_data    
 
model = TFmodel(10,[20,5,2])

X = tf.random.normal([50,10])
# y = tf.random.normal([50,2])
    
column_sum = tf.reshape(tf.reduce_sum(X, axis=1),(-1,1))
column_sum_2 = 2*column_sum+1
                 
y = tf.concat((column_sum,column_sum_2), axis=1)

model.forward(tf.random.normal([50,10], stddev=0.1))
model.backword(X,y)


# =============================================================================
# tape.gradient() 的工作原理
# 前向傳播（Forward Pass）：
# 
# GradientTape 會記錄所有參與運算的 tf.Variable 及其操作（OP）。
# 這會建立一個「動態計算圖」，每個變數和操作都有對應關係。
# 反向傳播（Backward Pass）：
# 
# 當 tape.gradient(loss, params) 被呼叫時，TensorFlow 會根據計算圖 自動回溯，而不是根據 params 的順序來計算梯度。
# 例如，如果 params[1] 其實對應於模型中的第一層，而 params[0] 對應於第二層，TensorFlow 仍然會根據計算圖的拓撲結構來回傳正確的梯度。
# =============================================================================
        

# =============================================================================
# 1️計算圖如何記錄 tf.Variable？
# 在 TensorFlow 的計算圖中，每個 tf.Variable 會被綁定到：
# 
# 運算節點（Operation Node）：每個變數參與的計算都會被記錄，例如矩陣乘法 (MatMul)、加法 (Add)、ReLU (Relu) 等。
# 梯度節點（Gradient Node）：當 tape.gradient(loss, params) 被呼叫時，TensorFlow 會根據**反向傳播規則（Chain Rule）**來動態生成對應的梯度計算節點。
# 因此，即使 params 的順序被調換，TensorFlow 仍然能夠透過計算圖來回溯正確的梯度，因為變數與運算節點的對應關係是由計算圖內部管理的，而不是由 params 列表的順序決定的。
# 
# 
# =============================================================================

batch_data = batch(data=X, batch_size=5)
len(batch_data)

epoch = 10


for i in range(epoch):
    for sub_batch_data in batch_data:
        model.backword(X,y)

model.forward(X)

y
