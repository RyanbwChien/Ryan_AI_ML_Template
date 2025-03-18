# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:06:31 2025

@author: user
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

tf.keras.Model
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

    @property
    def trainable_variables(self):
        """手動收集所有 Dense 層的 trainable_variables"""
        variables = []
        for layer in self.layer.values():
            variables.extend(layer.trainable_variables)
        return variables

inputs = Input((10,))
dense_layer = Dense(10)
outputs = dense_layer(inputs) 
dense_layer.weights
        
model = TF_Model_Subclassing(10,[20,10,5],2)

X = tf.random.normal((100,10),dtype=tf.float32)


y = tf.random.normal((100,2),dtype=tf.float32)

model.call(X)
model.trainable_variables
model.backward(X, y)
