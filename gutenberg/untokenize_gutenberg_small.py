import json
import numpy as np
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open('selected_dataset_mixed.json', 'r') as f:
    data = json.load(f)
train_dataset = np.array(data['train_data_np'])  
print(train_dataset.shape)

decoded_texts = tokenizer.batch_decode(train_dataset, skip_special_tokens=True)
print(f"Decoded {len(decoded_texts)} sequences.")
print(decoded_texts[0])  
print(decoded_texts[1])
with open('selected_dataset_mixed_decoded.json', 'w') as f:
    json.dump(decoded_texts, f)