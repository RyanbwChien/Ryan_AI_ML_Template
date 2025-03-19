# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:06:31 2025

@author: user
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from Ryan_dataset_batch_split import batch_data

# tf.keras.Model 與 tf.keras.models.Model 本質上是一樣，因通常會import tf.keras.models.Sequential，看是不是要統一寫法

def batch(data, batch_size):
    batch_data = []
    num_full_size = len(data) // batch_size
    remain_size = len(data) % batch_size
    for i in range(num_full_size):
        batch_data.append(data[i*batch_size:i*batch_size+batch_size])
    if remain_size != 0:
        batch_data.append(data[num_full_size*batch_size:num_full_size*batch_size+remain_size])
    return batch_data    


class TF_Model_Subclassing:
    def __init__(self,input_dim:int, hidden_layer_neuron:list, out_dim:int):
        super(TF_Model_Subclassing,self).__init__()
        self.layer = {}
        self.num_hidden_layer=len(hidden_layer_neuron)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        # self.model_input = Input(shape=(input_dim,))
        for i in range(self.num_hidden_layer):
            self.layer[f"layer_{i+1}"] = Dense(hidden_layer_neuron[i], activation='relu')
        
        
        self.layer["layer_output"] = Dense(out_dim, activation='relu')

        self.dummy_call(tf.random.normal([1, input_dim]))
        variables = []
        for layer in self.layer.values():
            variables.extend(layer.trainable_variables)
        self.trainable_variables = variables
    
    
    def dummy_call(self,x):
        _ = self.call(x)
        return(_)
    
    def call(self, x):
        iter_ = x
        
        for i in range(self.num_hidden_layer):

            iter_ = self.layer[f"layer_{i+1}"](iter_)
            
        output = self.layer["layer_output"](iter_)
        return(output)
    
    def backward(self,x,y):
        with tf.GradientTape() as tape:
            predict = self.call(x)
            loss = self.loss(y,predict)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

# =============================================================================
#     @property
#     def trainable_variables(self):
#         """手動收集所有 Dense 層的 trainable_variables"""
#         variables = []
#         for layer in self.layer.values():
#             variables.extend(layer.trainable_variables)
#         return variables
# =============================================================================


# 沒有給input dense layer 是不會初始化參數，因為不知道INPUT是幾維
dense_layer = Dense(10)
dense_layer.weights


# 給input dense layer 才知道INPUT是幾維，才會初始化參數
inputs = Input((10,))
dense_layer = Dense(10)
outputs = dense_layer(inputs) 
dense_layer.weights

# 創建一個TF_Model實體
model = TF_Model_Subclassing(10,[20,10,5],2) # input_dim=10等於告訴模型INPUT NEURON數量，會初始化參數
model.trainable_variables

#定義資料集
X = tf.random.normal((100,10),dtype=tf.float32)
y = tf.random.normal((100,2),dtype=tf.float32)

# 定義模型超參數
epochs = 100
batch_size = 25
batch_data(X,batch_size)
#訓練模型

for epoch in range(len(epochs)):
    for 
    
    




