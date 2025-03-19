# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 02:24:22 2025

@author: user
"""

import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# 建構模型
model = Sequential([
    Dense(10, activation='relu'),    
    Dense(5, activation='relu'),    
    Dense(2),   
    ])
model.summary()

model.compile(loss='mse', optimizer='adam')


# 在tf keras Sequential model 在還沒輸入Input前，model參數是空的，除非先給初始Input 讓模型才知道第一層參數量多少，可以初始化參數
X = tf.random.normal((100,5))
model(X)
# 整個模型各層的參數
model.weights
model.weights[0].trainable #各層參數都是一個物件，有trainable屬性
# 整個模型各層的可訓練的參數
model.trainable_variables

# build dataset
X = tf.random.normal((100,5))
y = tf.random.normal((100,2))

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2)

# Trian model
history = model.fit(X_train, Y_train, epochs =200, batch_size=32,validation_split=0.2 ) #, validation_data=(X_test, Y_test)


# 繪製 Loss 曲線
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# [Evaluate] testing set performance 

ytest_pred = model(X_test).numpy()

MSE = mean_squared_error(Y_test, ytest_pred)
MAE = mean_absolute_error(Y_test, ytest_pred)
MAPE = mean_absolute_percentage_error(Y_test, ytest_pred)

print(f"MSE:{MSE:.2f}; MAE:{MAE:.2f}; MAPE:{MAPE:.2f}" )

# "%s%.2f" %("A",2.333)
