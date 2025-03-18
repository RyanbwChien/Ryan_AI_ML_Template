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
            if i==0:                
                self.parameter[f'W{i+1}'] = tf.Variable(tf.random.normal(shape=(input_dim,neuron_size[i])),name=f'W{i+1}')
            else:
                self.parameter[f'W{i+1}'] = tf.Variable(tf.random.normal(shape=(neuron_size[i-1],neuron_size[i])),name=f'W{i+1}')
            self.parameter[f'b{i+1}'] = tf.Variable(tf.random.normal(shape=(1,neuron_size[i])),name=f'b{i+1}')
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
        grads = tape.gradient(losses, list(self.parameter.values()))
        self.grads = grads
        optimizer.apply_gradients(zip(grads, self.parameter.values()))
    
model = TFmodel(10,[20,5,2])

X = tf.random.normal([50,10])
# y = tf.random.normal([50,2])
    
column_sum = tf.reshape(tf.reduce_sum(X, axis=1),(-1,1))
column_sum_2 = 2*column_sum+1
                 
y = tf.concat((column_sum,column_sum_2), axis=1)

model.forward(tf.random.normal([50,10], stddev=0.1))
model.backword(X,y)


def batch(data, batch_size):
    batch_data = []
    num_full_size = len(data) // batch_size
    remain_size = len(data) % batch_size
    for i in range(num_full_size):
        batch_data.append(data[i*batch_size:i*batch_size+batch_size])
    if remain_size != 0:
        batch_data.append(data[num_full_size*batch_size:num_full_size*batch_size+remain_size])
    return batch_data
        
batch_data = batch(data=X, batch_size=5)
len(batch_data)

epoch = 100


for i in range(epoch):
    for sub_batch_data in batch_data:
        model.backword(X,y)

model.forward(X)

y
