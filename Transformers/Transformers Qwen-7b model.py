# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:20:14 2025

@author: USER
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入模型和分詞器
model_name = "Qwen/Qwen-7B"  # 替換為真實的模型名稱，根據 Hugging Face 上的模型名稱
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# 確認是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 準備輸入文本
input_text = "你好"

# 將輸入文本轉換為模型需要的格式
inputs = tokenizer(input_text, return_tensors="pt")

# 使用模型生成回應
outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)

# 解碼生成的文本
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model Response:", response)