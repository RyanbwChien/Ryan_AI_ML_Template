import torch
import torch.nn as nn

# 定義 Seq2Seq 模型
class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
        super(Seq2SeqModel, self).__init__()

        # LSTM 層：第一層 LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # RepeatVector 等效：設定時間步長
        self.num_timesteps = num_timesteps
        
        # LSTM 層：第二層 LSTM
        self.lstm2 = nn.LSTM(hidden_dim, 12, batch_first=True)

        # TimeDistributed 等效：每個時間步應用全連接層
        self.fc = nn.Linear(12, output_dim)

    def forward(self, x):
        # 輸入資料到第一層 LSTM
        x, (hn, cn) = self.lstm1(x)

        # 使用 `unsqueeze()` 和 `expand()` 或 `view()`，這樣可以保證時間步長為 num_timesteps
        batch_size, seq_len, hidden_dim = x.size()
        
        # 假設原本的時間步數是 5，要變成 3
        x = x[:, -1:, :]  # 取最後一個時間步的輸出，形狀 (batch_size, 1, hidden_dim)

        # 複製到指定時間步長 (num_timesteps)
        x = x.expand(-1, self.num_timesteps, -1)  # 形狀變為 (batch_size, num_timesteps, hidden_dim)

        # 輸入到第二層 LSTM
        x, (hn, cn) = self.lstm2(x)

        # 應用 TimeDistributed 操作，對每個時間步使用全連接層
        x = self.fc(x)

        return x

# 假設的輸入數據 (batch_size, timesteps, features)
input_data = torch.randn(192, 5, 15)  # 192 是 batch_size，5 是 timesteps，15 是 features

# 初始化模型
input_dim = 15
hidden_dim = 32
output_dim = 6
num_timesteps = 3

model = Seq2SeqModel(input_dim, hidden_dim, output_dim, num_timesteps)

# 模型前向傳播
output = model(input_data)

# 顯示輸出形狀
print("Output shape:", output.shape)
