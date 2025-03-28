from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 下載並加載 GPT-2 模型與 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 測試文本
input_text = "你好"

# 編碼文本
inputs = tokenizer(input_text, return_tensors='pt')

# 生成文本
outputs = model.generate(
    inputs['input_ids'], 
    max_length=50, 
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,  # top-k 取樣
    top_p=0.95,  # top-p (nucleus) 取樣
    no_repeat_ngram_size=2  # 防止重複
)

# 解碼生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
