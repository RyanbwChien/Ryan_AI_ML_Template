# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:36:47 2025

@author: user
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchvision.transforms.v2 as v2

# 目的是訓練 CNN 模型，PIL 和 torchvision.io.read_image 是最常見的選擇。
from torchvision.io import read_image
from PIL import Image
# v2.Compose(v2.Resize())
import kagglehub
import cv2
import matplotlib.pyplot as plt

# =============================================================================
# jpg jepg
# 8 位元二進制：
# 每個字節實際上是由 8 位元（bit）組成的。每個位元只能是 0 或 1，因此 8 位元可以有 256 種不同的組合，對應的數字範圍就是 0 到 255。
# =============================================================================

# =============================================================================
# 解決方案：使用 numpy 與 cv2.imdecode()
# 為了解決這個問題，我們可以避開 cv2.imread() 的限制，透過以下步驟來解決：
# 
# 使用 numpy.fromfile() 來讀取圖片的二進位資料。這個方法能夠處理包含中文字符的路徑。
# 使用 cv2.imdecode() 將二進位資料解碼為圖片格式。
# 這個方法可以繞過路徑編碼問題，成功讀取圖片。以下是完整的程式碼範例：
# =============================================================================
# cv2_image = cv2.imread(r"D:\Linebot_photo\橫幅-詐騙零容忍.jpg")
# =============================================================================
# OpenCV 無法讀取路徑含有中文的圖片
# OpenCV 使用的是 C++ 標準庫來進行檔案讀取操作，而這些函式在不同作業系統上對於非 ASCII 字符（例如中文、日文、韓文等）的支援並不一致。
# 特別是在 Windows 系統中，檔案路徑的默認編碼並非 UTF-8，這導致了當 cv2.imread() 嘗試讀取含有中文的路徑時，會因為編碼錯誤使程式回傳 None ，
# 導致圖片無法正常被加載。
# =============================================================================

np.fromfile(r"D:\Linebot_photo\橫幅-詐騙零容忍.jpg", dtype=np.uint8).shape #output (449637,)

cv2_image = cv2.imdecode(np.fromfile(r"D:\Linebot_photo\橫幅-詐騙零容忍.jpg", dtype=np.uint8), -1) # 成功讀取到圖片路徑有中文的照片

type(cv2_image)
cv2_image
cv2_image_RGB = cv2_image[:,:,::-1]
plt.imshow(cv2_image_RGB)


plt_image = plt.imread(r"D:\Linebot_photo\橫幅-詐騙零容忍.jpg")
plt_image.shape

torchvision_image = read_image(path = r"D:\Linebot_photo\橫幅-詐騙零容忍.jpg")
torchvision_image.shape
type(torchvision_image)

torchvision_image.permute(1,2,0).shape
np.transpose(torchvision_image,(1,2,0)).shape

PIL_image = Image.open(fp = r"D:\Linebot_photo\橫幅-詐騙零容忍.jpg")
type(PIL_image)
np.array(PIL_image).shape

v2.ToTensor()(PIL_image).shape



class Torch_CNN(nn.Module):
    def __init__(self):
        super(Torch_CNN,self).__init__()
        # stride > 1 時，PyTorch 不能使用 padding='same'
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(32,32), padding='same', stride = 1)
        self.Maxpool1 = nn.MaxPool2d(3, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(16,16), padding='same', stride = 1)
        self.Maxpool2 = nn.MaxPool2d(3, stride=2)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(16,16), padding='same', stride = 1)
        self.Maxpool3 = nn.MaxPool2d(3, stride=2)
        self.Flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(10)
        self.fc2 = nn.LazyLinear(2)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Maxpool1(x)
        x = self.Conv2(x)
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        x = self.Maxpool3(x)
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Torch_CNN()

dummy_input = torch.randn(1, 3, 128, 128)
output = model(dummy_input)
print("Output shape:", output.shape)     

# =============================================================================
# # ***transforms 內建到 Dataset 類別***
# 如果你有一個 自定義的 Dataset 類別，你也可以在 __getitem__ 方法內應用 transforms：
# =============================================================================

# =============================================================================
# torchvision.io.read_image() 讀進來的圖片是 torch.Tensor，形狀為 [C, H, W]，所以不能直接用 transforms.Resize()，因為 Resize 預設適用於 PIL 圖像或 [H, W, C] 的 Tensor。
# 
# 要 對 read_image() 讀進來的 Tensor 進行 Resize，你可以改用 transforms.v2.Resize()，它支援 torch.Tensor，或者用 F.interpolate() 來手動縮放。
# =============================================================================

# 在外面先定義 transforms 模組，之後再套用到自訂義好的Dataset初始化參數transform裡面
transforms =  v2.Compose([v2.Resize((128,128)), 
                          v2.ToTensor(),
                          v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dummy_input = torch.randn(1, 3, 128, 128)
transforms(dummy_input)

# =============================================================================
# padding 參數	說明	適用情境
# padding=16	所有邊填充 16	讓輸入和輸出尺寸一致
# padding=(16, 8)	高度填充 16，寬度填充 8	需要不同方向的填充
# padding="same"	自動計算填充，使輸入和輸出尺寸相同	讓 PyTorch 自動處理
# padding_mode="reflect"	使用鏡像填充	需要不同的邊界填充策略
# padding=(kernel_size//2, kernel_size//2)	手動計算適當填充	確保輸出尺寸與輸入相同
# 一般來說，如果你要讓輸入和輸出尺寸相同，最簡單的方法是 padding="same" 或 padding=kernel_size//2！        
# =============================================================================
        
# Download CNN latest version
path = kagglehub.dataset_download("puneet6060/intel-image-classification")        
print("Path to dataset files:", path)

train_path = r'C:\Users\user\.cache\kagglehub\datasets\puneet6060\intel-image-classification\versions\2\seg_train\seg_train'
test_path = r'C:\Users\user\.cache\kagglehub\datasets\puneet6060\intel-image-classification\versions\2\seg_test\seg_test'
classes = os.listdir(train_path)

train_photo = []
train_label = []
test_photo = []
test_label = []

for split_set in [train_path, test_path]:
    for class_ in classes:
        photo_path = os.listdir(os.path.join(split_set,class_))
        if split_set == train_path:
            train_label.append([class_]*len(photo_path))
        else:
            test_label.append([class_]*len(photo_path))

    if split_set == train_path:
        train_photo = [ os.path.join(train_path,class_,i) for i in photo_path]
    else:
        test_photo = [ os.path.join(test_path, class_, i) for i in photo_path]

        

'''transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化'''
# =============================================================================
# 這些 mean 和 std 值 (mean=[0.485, 0.456, 0.406] 和 std=[0.229, 0.224, 0.225]) 是基於 ImageNet 數據集的統計數據。ImageNet 是一個包含超過 1400 萬張標註圖片的大型圖像數據集，它被廣泛用於圖像分類任務，並且許多模型（如 ResNet、VGG、Inception 等）都是在這個數據集上進行預訓練的。
# 
# 當你使用預訓練模型（如 ResNet、VGG 等）時，它們通常會期望輸入數據的分佈與它們在訓練時所使用的數據分佈一致。這些 mean 和 std 值就是這些模型在 ImageNet 數據集上訓練過程中計算得到的。
# 
# 為什麼這些值是這樣的？
# 這些值是基於 ImageNet 數據集的各個圖像通道（紅色、綠色和藍色通道）的均值和標準差計算出來的。具體來說：
# 
# Mean：每個通道的像素值的平均值。對於 ImageNet，這三個通道的平均值分別為 0.485, 0.456, 和 0.406。
# Std (標準差)：每個通道的像素值的標準差。對於 ImageNet，這三個通道的標準差分別為 0.229, 0.224, 和 0.225。
# 這些均值和標準差用來讓模型的輸入圖像在訓練時和推理時保持一致，這樣有助於提高訓練的穩定性並使推理結果更準確。
# 
# 如何得到這些值？
# 通常，這些統計數值是通過計算 ImageNet 數據集中的每一張圖片的每個通道的均值和標準差來得到的。
# 這些值是對整個數據集的圖像通道進行統計計算的結果。
# 預訓練模型和標準化：
# 當你使用基於 ImageNet 預訓練的模型時，你希望輸入圖片經過與訓練過程相同的標準化處理，這樣模型就能有效地理解並處理輸入圖像。
# 
# 總結：
# mean=[0.485, 0.456, 0.406] 和 std=[0.229, 0.224, 0.225] 是 ImageNet 數據集的統計結果，反映了每個顏色通道（RGB）的均值和標準差。
# 這些值被用來將新圖像標準化，使其與訓練預訓練模型時使用的數據一致，從而提高模型的預測準確性和穩定性。
# =============================================================================

class Torch_dataset(Dataset):
    def __init__(self,X,y, transforms=None):
        self.X = [np.array(Image.open(path)) for path in X]
        self.y = y
        
    def __len__(self):
        return(len(self.X ))
    
    def __getitem__(self,ind):
        if transforms:
            return(transforms(self.X[ind],self.y))
        else:            
             return(self.X[ind],self.y)      
                   
                   
puneet6060_train_dataset = Torch_dataset(train_photo, train_label)        
        
        