
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, InputLayer, TimeDistributed, RepeatVector, Dense, Flatten
from tensorflow.keras.models import Sequential


X = np.random.normal(0,1,(200,15))
Y = np.random.normal(0,1,(200,6))

p = 5
q = 3
X_new = []
Y_new = []
for i in range(len(X)-p-q):
    X_new.append(X[i:i+p])
    Y_new.append(Y[i+p:i+p+q])

X_new = np.array(X_new)
X_new.shape
Y_new = np.array(Y_new)
Y_new.shape

batch_size = 12
batch_number = int(len(X_new) / batch_size)

X_new_batch = []
Y_new_batch = []
strat = 0  
for i in range( batch_number):
    X_new_batch.append(X_new[strat  : strat + batch_size])
    Y_new_batch.append(Y_new[strat  : strat + batch_size])
    strat = strat  + batch_size

X_new_batch = np.array(X_new_batch)
Y_new_batch = np.array(Y_new_batch)
X_new_batch.shape



# 模拟输入数据：批量大小为32，时间步长为15，特征维度为64
batch_size = 12
time_steps = 5
feature_dim = 15

inputs = tf.random.normal((batch_size, time_steps, feature_dim))
inputs.shape
# =============================================================================
# # 定义 LSTM 层
# lstm_layer = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
# 
# # 前向传播
# outputs, state_h, state_c = lstm_layer(inputs)
# =============================================================================

model = Sequential()


model.add(InputLayer(batch_input_shape = (12,5,15)))

# =============================================================================
# Batch size 和 Batch count
# 兩個參數名稱有點相似，他們的差異是：
# Batch size 調整 一次計算過程中，需要同時產出幾張圖片 （一次計算可以視為一個 Batch）
# Batch count 調整 按下算圖後，總共要跑幾次計算 （幾次 Batch）
# =============================================================================

# TimeDistributed 用途
# =============================================================================
# model.add( LSTM(32, return_sequences = True)) 
# model.add(TimeDistributed(Dense(1))) #TimeDistributed改變特徵數?
# =============================================================================

# 第一層有32個 LSTM Neuron 每一個 TIMESTEP要跑出32 , return_sequences = True

model.add( LSTM(32, return_sequences = True, stateful=True) ) # return_sequences = False 回傳 將 n*p矩陣 只關心最後一次timestep output



model.add(Flatten())  # 將維度轉換為 (batch_size, timesteps * features)
# Input 與 Output timestep長度不同
model.add(RepeatVector(3)) # 將 n*p 矩陣 轉換成 n*Tout*h1 矩陣
model.summary()
model.add( LSTM(12, return_sequences = True,stateful=True)) # 將 n*Tout*h1 矩陣 轉換成 output n*Tout*q 矩陣
# model.add( LSTM(6, return_sequences = True)) # 將 n*Tout*h1 矩陣 轉換成 output n*Tout*q 矩陣
model.add(TimeDistributed(Dense(6)))
# TimeDistributed 層（TimeDistributed(Dense(6))）：
# TimeDistributed 會將某個層（如 Dense）應用到每個時間步的輸入數據，對每個時間步的數據進行獨立的處理。假設您將 TimeDistributed(Dense(6)) 添加到 LSTM 層後，這會將每個時間步的輸出（即 (batch_size, timesteps, LSTM_units)）傳遞到 Dense(6) 層，每個時間步的輸出都會進行 Dense 層的處理，並且每個時間步的輸出是 6 維。


model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])

history = model.fit(X_new, Y_new, epochs = 100,batch_size =12)
import matplotlib.pyplot as plt        
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


