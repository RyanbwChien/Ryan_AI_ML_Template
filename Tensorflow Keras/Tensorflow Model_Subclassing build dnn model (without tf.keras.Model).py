# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:06:31 2025

@author: user
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from Ryan_dataset_batch_split import batch_data
from sklearn.model_selection import train_test_split


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

        self.dummy_call(tf.random.normal([1, input_dim])) # 在初始化時故意給一個input 讓模型可以初始化參數
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
        return loss, predict
    
    def mse(self, y_true, y_pred):
        error = y_true - y_pred
        squared_error = tf.square(error)
        return tf.reduce_mean(squared_error)
    
# =============================================================================
#     @property # 使用 @property 方法
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


#定義資料集
X = tf.random.normal((100,10),dtype=tf.float32)
y = tf.random.normal((100,2),dtype=tf.float32)

#拆分訓練與測試集
X_train, X_test, Y_train, Y_test = train_test_split(X.numpy() , y.numpy(), test_size=0.2)

# 定義模型超參數
epochs = 100
batch_size = 25
input_dim = 10
hidden_layer_neuron = [20,10,5]
output_dim = 2

# 創建一個TF_Model實體
model = TF_Model_Subclassing(10,[20,10,5],2) # input_dim=10等於告訴模型INPUT NEURON數量，會初始化參數
model.trainable_variables

#訓練模型
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0
    for batch_x, batch_y in zip(batch_data(X_train,batch_size), batch_data(Y_train,batch_size)):
        loss, predict = model.backward(batch_x, batch_y)
    
        
        # 更新每個 epoch 的損失和準確度
        epoch_loss += loss
        num_batches += 1
        
    mse = model.mse(batch_y, predict)    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}, Mse: {mse:.4f}")
    
    

# 訓練結束後評估模型 理論上策是集因該就不用在分batch了
test_loss = 0
test_acc = 0

pred = model.call(X_test)
test_loss = model.loss(Y_test, pred)
test_mse = model.mse(Y_test, pred)

# 顯示測試集的損失和準確度
print(f"Test Loss: {test_loss:.4f}, Test mse: {test_acc:.4f}")


