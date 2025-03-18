# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 03:34:01 2025

@author: user
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


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
    def forward(self,X):
        model_input = X
        for i, layer in enumerate(self.layers):
            if i==0:
                iter_ = layer(model_input)
            else:
                iter_ = layer(iter_)
        return(iter_)
                
                
torch_model = torch_dnn_model(5,[10,20,5,3,2])
for param in torch_model.parameters():
    print(param)


X = torch.normal(0,1,(100,5))
torch_model(X)