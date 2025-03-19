# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 18:08:54 2025

@author: user
"""

import torch
import torch.nn as nn


# class CBR(nn.Sequential) 這種寫法可以將它想像成
# 繼承了nn.Sequential 初始化函數，所以可以直接在super().__init__(裡面寫nn.Sequential 自己的初始化參數並帶入nn 相關模組)
# 因為是可變動位置參數(位置引數（Positional Arguments） 傳遞值時，不需要指定參數名稱，直接按照順序賦值。


# 🔹 為什麼 CBR 可以這樣寫？
# 因為 nn.Sequential 本身的 __init__ 定義大概是這樣：
# class Sequential(nn.Module):
#     def __init__(self, *args):
#         super(Sequential, self).__init__()
#         # 依序將傳入的 layers 加入
#         for idx, module in enumerate(args):
#             self.add_module(str(idx), module)

class CBR(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size):
        super(CBR,self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()         
            )


CBR_object = CBR(3,10,3)

nn.Sequential()


class L(nn.Sequential):
    def __init__(self):
        super(L,self).__init__(
            nn.Linear(10,20),
            nn.ReLU(),
            nn.Linear(20,2)
            )
# 這種寫法我可以理解為 繼承 nn.Sequential，
# 在nn.Sequential原生寫法 初始化參數裡面可以餵給他多個 nn模型
# 那新建立的L物件 一開始會先執行 將super裡面的多個模型引數帶到父類別裡面，
# 此時若執行 子類別()建立實體物件時，一開始會先執行 將super裡面的多個模型引數帶到父類別nn.Sequential裡面，
# 並呼叫 繼承父類別的CALL函數將 繼承nn.Sequential的初始化參數內容執行出來
L()
