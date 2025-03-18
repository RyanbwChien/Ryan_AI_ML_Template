# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:24:52 2025

@author: user
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np

from sklearn.datasets import load_iris
load_iris().get('data')



def build_model(data):
    model_input = Input(batch_shape=(10,10))
    
   