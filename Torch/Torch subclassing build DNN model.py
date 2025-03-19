# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 03:34:01 2025

@author: user
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# =============================================================================
# nn.Linear 初始化權重時更高效
# nn.LazyLinear 只有在第一次 forward() 時才初始化權重，這意味著：
# 可能會導致 首次 forward() 時有額外的開銷，降低效率。
# 如果你在多層結構中使用 LazyLinear，每一層都要等到 forward() 時才初始化，可能會讓調試變得更困難。
# nn.Linear 一開始就初始化好所有權重，這樣在訓練過程中更高效，因為不會在 forward() 時執行額外的初始化步驟。
# =============================================================================

class torch_dnn_model(nn.Module):
    def __init__(self, input_dim:int , layer_neuron:list[int]):
        super(torch_dnn_model,self).__init__()
        self.layers = nn.ModuleList()  # 使用 nn.ModuleList 儲存所有層
        for i in range(len(layer_neuron)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, layer_neuron[0]))
            else:
                self.layers.append(nn.Linear(layer_neuron[i-1], layer_neuron[i]))
                
    def forward(self,X): # torch nn.Module 要自訂forward function 向前傳播
        model_input = X
        for i, layer in enumerate(self.layers):
            if i==0:
                iter_ = layer(model_input)
            else:
                iter_ = layer(iter_)
        return(iter_)
                

class Torch_dataset(Dataset):
    def __init__(self,X,y, transform = None):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,ind):
        return(self.X[ind], self.y[ind])


# DataLoader 本身是為了從 Dataset 類別中載入資料設計的，它需要接受一個繼承自 torch.utils.data.Dataset 的物件，這樣才能正確處理數據。
# 注意如果DataLoder 前面加iter 會讓之後訓練跑迴圈第一次跑完 全部就變空的，會有異常
# for x,y in torch_dataset:
#     print(x,y)


# Defined model hyper parameter

batch_size = 32
input_dim = 5
layer_neuron = [10,20,5,3,2]
epochs = 100
lr=1*0.001

# Loader dataset
X = torch.normal(0,1,(100,5))
y = torch.normal(0,1,(100,1))

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# Dataset
train_dataset = Torch_dataset(X_train,Y_train)
test_dataset = Torch_dataset(X_train,Y_train)

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size = batch_size)

# Defined model                
torch_model = torch_dnn_model(input_dim, layer_neuron)
for param in torch_model.parameters():
    print(param)
    


# Adjust model layer parameter trainable
# list(torch_model.parameters())[0].requires_grad = False
# list(torch_model.parameters())


#training
criterion = nn.MSELoss()
optimizer = Adam(params = torch_model.parameters(), lr=lr)

for epoch in range(epochs):
    for x_train_batch, y_train_batch in train_dataset:
        
        torch_model.zero_grad() # pytorch 每一次Iteration 要清空模型中每一層屬性參數的當前梯度，避免後續梯度累加，與Tensorflow 不一樣，TF不用是因為梯度部會記錄在模型中每一層屬性裡面
        y_pred = torch_model(x_train_batch)
        loss = criterion(y_pred, y_train_batch) #因為y_pred 是從頭INPUT到模型OUTPUT包含模型各層參數
        loss.backward()#因為y_pred是包含模型各層參數，所以反向傳播在計算各層梯度時，就會將梯度記錄在 torch_model.parameters()
        optimizer.step() #因為一開始 torch_model.parameters() 就放在optimize裡面所以直接利用梯度更新當前參數
        
    print(f'Epoch [{epoch+1}/epochs], Loss: {loss.item()}')
