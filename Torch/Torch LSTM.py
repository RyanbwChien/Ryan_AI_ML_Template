# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:19:20 2025

@author: user
"""

import torch
import torch.nn as nn


model = nn.Sequential(nn.LSTM(10,5,))

x = torch.tensor(range(10),dtype=torch.float32)
x.shape

model(x)
