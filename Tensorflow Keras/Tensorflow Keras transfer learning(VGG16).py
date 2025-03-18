# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 04:02:30 2025

@author: user
"""
# =============================================================================
# 使用 Sequential 模型構建
# 雖然 VGG16 的基礎模型通常是使用函數式 API（Model）來構建，但你也可以使用 Sequential 來將新層添加到模型中。
# =============================================================================

# =============================================================================
# 遷移學習
# 使用 VGG16 訓練好的參數來初始化模型，並且 只訓練新的分類層（而不是重新訓練整個模型），
# VGG16 的捲積層已經學到了圖像的高層次特徵（例如邊緣、形狀、顏色模式等），這些特徵對許多圖像分類任務都是有用的。
# =============================================================================

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 1️⃣ 載入 VGG16 去掉分類層
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 2️⃣ 定義 Sequential 模型
model = Sequential()

# 3️⃣ 添加 VGG16 作為基礎模型，並凍結層
model.add(base_model)  # Add VGG16 as base model (without top)
base_model.trainable = False

# 4️⃣ 添加新的分類層
model.add(Flatten())  # 攤平為一維
model.add(Dense(128, activation="relu"))  # 隱藏層
model.add(Dense(10, activation="softmax"))  # 10 類別輸出層

# 5️⃣ 編譯模型
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 顯示模型架構
model.summary()

# =============================================================================
# 使用 Sequential 讓模型的層一個接一個地疊加，語法更簡潔。
# 不過，當你需要更多靈活性（例如共享層，或多輸入/輸出的情況），Functional API（如前一例）是更好的選擇。
# =============================================================================


# =============================================================================
# ✅ 2. 使用 Keras 層的自定義功能
# 如果你需要更複雜的層設計，甚至想將這些層放在一個自定義類中，這樣可以讓你對層進行更高層次的控制。
# =============================================================================


from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, base_model):
        super(MyModel, self).__init__()
        self.base_model = base_model
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# 載入 VGG16 基礎模型
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 建立自定義模型
model = MyModel(base_model)

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 顯示模型架構
model.summary()

# 使用自定義 tf.keras.Model 類別，你可以在 call 方法中定義自己的層邏輯，這對於某些複雜的結構非常有用。
# 3. 使用函數式 API（另一種語法）

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

# 載入 VGG16 去掉分類層
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 凍結 VGG16 層
for layer in base_model.layers:
    layer.trainable = False

# 這裡的寫法可以將層定義成變數，然後再拼接
x = base_model.output
x = layers.Flatten()(x)  # 攤平
x = layers.Dense(128, activation='relu')(x)  # 隱藏層
x = layers.Dense(10, activation='softmax')(x)  # 輸出層

# 建立最終模型
model = Model(inputs=base_model.input, outputs=x)

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔥 顯示模型架構
model.summary()
# 在這個例子中，我們還是使用了函數式 API，但語法會稍微不同，層的定義和連接方式都可以清楚地控制。 

# =============================================================================
# 總結
# Sequential 模型：當你想簡單地將層按順序堆疊時使用。
# 自定義模型（tf.keras.Model）：對模型有更複雜的需求時，可以自定義層的邏輯。
# 函數式 API：更靈活，可以讓你定義複雜的網絡結構。
# 無論是哪一種方式，都能幫助你輕鬆地將 VGG16 與新的分類層結合並訓練模型！
# =============================================================================
