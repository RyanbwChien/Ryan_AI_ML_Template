# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 00:03:58 2025

@author: user
"""
import numpy as np
from sklearn.impute import SimpleImputer


imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]) # 取得訓練資料各欄平均值

X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))
