# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 00:02:04 2025

@author: user
"""

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_gen = ImageDataGenerator()
img_gen.flow_from_directory()
