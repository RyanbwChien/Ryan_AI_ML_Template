import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 假設我們有一個 DataFrame 叫做 data_frame
# 這裡生成一個隨機數據框作為示例
path = r"C:\Users\u048\0002 G1_A1.xlsx"
df = pd.read_excel(path)
data_frame = df.iloc[:,[1,3]]


# 標準化數據
scaler = StandardScaler()
data = scaler.fit_transform(data_frame.values)
data = torch.tensor(data, dtype=torch.float32)


# =============================================================================
# 擴散模型的核心思想是通過加噪聲和去噪過程來進行生成。訓練過程中，模型學會了如何從加了噪聲的數據中預測出噪聲，
# 而生成過程則是反向執行去噪過程，從隨機噪聲中生成新的數據。這種方法與傳統的生
# =============================================================================


# 定義U-Net結構（簡化版）
class SimpleUNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# 定義擴散過程
def q_sample(data, t, noise_level):
    """將數據加入噪聲"""
    return data * (1 - noise_level) + noise_level * torch.randn_like(data) #输入与传入参数size相同的满足标准正态分布的随机数字tensor

def noise_prediction_loss(model, x_noisy, t, clean_data, noise_level):
    """計算損失：模型預測噪聲"""
    pred_noise = model(x_noisy)
    true_noise = (x_noisy - clean_data) / noise_level
    return nn.MSELoss()(pred_noise, true_noise)

# 訓練與生成函數
def train_diffusion_model(data, num_steps=1000, noise_level=0.1, epochs=100, lr=1e-4):
    input_dim = data.shape[1]
    model = SimpleUNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 隨機選擇時間步數 t
        t = torch.randint(1, num_steps, (data.shape[0],)).float().to(device) / num_steps
        noisy_data = q_sample(data, t, noise_level)

        # 計算損失並反向傳播
        loss = noise_prediction_loss(model, noisy_data, t, data, noise_level)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    return model

def generate_samples(model, num_samples, input_dim, num_steps=1000, noise_level=0.1):
    model.eval()
    with torch.no_grad():
        # 初始化隨機噪聲
        samples = torch.randn(num_samples, input_dim).to(device)

        # 逐步去噪過程
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((num_samples,), t / num_steps).float().to(device)
            pred_noise = model(samples)
            samples = samples - noise_level * pred_noise

        return samples.cpu().numpy()

# 設置參數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 2000
input_dim = 2  # 一維數據


# 標準化數據
scaler = MinMaxScaler()
data = scaler.fit_transform(data_frame.values)
data = torch.tensor(data, dtype=torch.float32)


# # 生成訓練數據（例如：高斯分布）
# true_data = np.random.normal(loc=0, scale=1, size=(num_samples, input_dim))
# train_data = torch.tensor(true_data, dtype=torch.float32).to(device)

# 訓練模型
print("Training model...")
model = train_diffusion_model(data, epochs=10000)

# 生成新樣本
print("Generating samples...")
generated_samples_trans = generate_samples(model, num_samples=500, input_dim = input_dim)
generated_samples = scaler.inverse_transform(generated_samples_trans) 

plt.scatter(generated_samples[:,0],generated_samples[:,1])

# # 可視化生成結果
# plt.hist(generated_samples, bins=50, alpha=0.6, label="Generated Data")
# plt.hist(true_data, bins=50, alpha=0.6, label="True Data")
# plt.legend()
# plt.show()
