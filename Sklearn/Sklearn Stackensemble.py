# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:46:11 2025

@author: user
"""

# =============================================================================
# 策略	Base Models	Meta-Model	適用場景
# 策略 1	RandomForest, XGBoost	MLP (NN)	結構化數據
# 策略 2	Neural Network	XGBoost, LightGBM	非結構化數據 (影像/文本)
# 策略 3	RF + XGB + NN	XGBoost	混合數據
# 這三種方法可以根據數據特性選擇，讓 Stacking 更靈活！ 🚀
# =============================================================================
