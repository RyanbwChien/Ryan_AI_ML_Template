# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:40:08 2025

@author: user
"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

# =============================================================================
# StratifiedKFold 是 分層抽樣交叉驗證
# 適用於類別不平衡的數據集，確保每折類別比例一致
# =============================================================================

iris = load_iris()

iris.data
iris.feature_names
iris.target
iris.target_names



# 如果是 DNN Neuron network hyperparameter 'epochs' 'batch_size' 是要丟到超參數裡面
# 但treebased model 樹狀模型只有幾棵樹


# =============================================================================
# 我認為是要先做train_test_split 後，用TRAINING SET 做 StratifiedKFold 去看模型中哪組超參數比較好，
# 之後用全部TRAINING SET 放到最好的模型超參數下訓練模型，再用測試集去驗證模型
# 
# =============================================================================
parameter = { 'n_estimators':[50,100,200],
             'learning_rate':[0.001,0.01,0.1,0.2],
             'max_depth':[2,3,4,5,6]          
    }



# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(
#     XX, yy, 
#     stratify=(y["OPEN901ppm"] > 5000) | (y["SHORT902ppm"] > 5000), 
#     test_size=0.2,
#     shuffle=True
# )
# StratifiedKFold 主要用於分類問題，保證每一折中類別的比例與原始資料集一致。它通常只接受目標變數 y 來進行分層抽樣，但並不像 train_test_split 那樣可以直接接受複雜的條件來創建分層標籤。
# 
# 然而，你可以用類似的方式來達到這個效果：在使用 StratifiedKFold 前，先根據條件 (y["OPEN901ppm"] > 5000) | (y["SHORT902ppm"] > 5000) 創建一個新的分層標籤，然後將這個標籤傳遞給 StratifiedKFold 來進行分層抽樣。
# 
# 具體做法如下：
# 
# 根據條件創建一個布林型的標籤。
# 
# 使用 StratifiedKFold 時，將這個標籤作為 y 傳入。
# =============================================================================

skf = StratifiedKFold(n_splits=5,shuffle=True)
kfold = KFold(n_splits=5,shuffle=True)

list(kfold.split(iris.data)) # 會拆成 K等分 每一等分依序當Y 剩下的當X


GBDT = GradientBoostingClassifier()

GSCV = GridSearchCV(GBDT,parameter,cv = skf)
GSCV.fit(iris.data, iris.target)

GSCV.best_params_
