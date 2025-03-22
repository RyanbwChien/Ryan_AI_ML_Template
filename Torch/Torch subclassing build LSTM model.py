import torch
import torch.nn as nn

# =============================================================================
# 當 nn.LSTM 的輸出傳遞到下一層 nn.LSTM，它會包含所有時間步的 hidden states。
# 這等同於 Keras LSTM return_sequences=True 的行為。
# 如果只想取最後一個時間步的 hidden state，使用 output[:, -1, :]。
# =============================================================================

# 定義 Seq2Seq 模型
class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim,input_data_time_step, hidden_dim, output_dim, num_timesteps):
        super(Seq2SeqModel, self).__init__()

        # LSTM 層：第一層 LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.flatten = torch.nn.Flatten()
        # RepeatVector 等效：設定時間步長
        self.num_timesteps = num_timesteps
        
        # LSTM 層：第二層 LSTM
        self.lstm2 = nn.LSTM(hidden_dim*input_data_time_step, 12, batch_first=True)

        # TimeDistributed 等效：每個時間步應用全連接層
        self.fc = nn.Linear(12, output_dim)

    def forward(self, x):
        # 輸入資料到第一層 LSTM
        x, (hn, cn) = self.lstm1(x)

        # 使用 `unsqueeze()` 和 `expand()` 或 `view()`，這樣可以保證時間步長為 num_timesteps
        batch_size, seq_len, hidden_dim = x.size()
        x = self.flatten(x)
# =============================================================================
#         # 假設原本的時間步數是 5，要變成 3
#         x = x[:, -1:, :]  # 取最後一個時間步的輸出，形狀 (batch_size, 1, hidden_dim)
# =============================================================================

# =============================================================================
#         # 複製到指定時間步長 (num_timesteps)
#         x = x.expand(-1, self.num_timesteps, -1)  # 形狀變為 (batch_size, num_timesteps, hidden_dim)
# 
# =============================================================================
        # 複製到指定時間步長 (num_timesteps)
        x = x.unsqueeze(1).repeat(1, num_timesteps, 1)  # 形狀變為 (batch_size, num_timesteps, hidden_dim)
        # 這表示 (batch_size, features) → (batch_size, 1, features)，相當於在 dim=1 增加了一個新的 time_step=1 維度。


# =============================================================================
#         🔹 expand() vs repeat()
#         方法	作用	記憶體	適用場景
#         expand(-1, new_timesteps, -1)	擴展張量視圖，不複製數據	✅ 節省記憶體	僅讀取資料，不修改內容
#         repeat(1, new_timesteps, 1)	直接重複數據	🚫 消耗更多記憶體	需要修改數據時
#         ✅ 結論
#         如果你只是想 擴展時間維度，但 不修改數據內容，使用 expand() 更省記憶體。
#         如果你需要對數據做變換（例如 LSTM/Transformer 訓練時），使用 repeat() 確保不共享記憶體。
# =============================================================================


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
input_data_time_step= 5

model = Seq2SeqModel(input_dim, input_data_time_step, hidden_dim, output_dim, num_timesteps)

# 模型前向傳播
output = model(input_data)

# 顯示輸出形狀
print("Output shape:", output.shape)
