## Dataset "Selected_dataset_mixed"
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import json
import numpy as np
import pickle
model_path = os.path.join(os.path.dirname(__file__), '../out/gpt2-scratch-mixed')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval()
print(model.config)
print(model.num_parameters())

data_path = os.path.join(os.path.dirname(__file__), 'selected_dataset_mixed.json')
with open(data_path, 'r') as f:
    data = json.load(f)

print(len(data))
print(data.keys())
train_dataset = data['train_data_np']
train_authors = data['train_data_authors']
train_titles = data['train_data_titles']
eval_dataset = np.array(data['eval_data_np'])
eval_authors = data['eval_data_authors']
eval_titles = data['eval_data_titles']


authors = list(set(eval_authors))
print(authors)


for author in authors:
    model = GPT2LMHeadModel.from_pretrained(model_path).eval().to(device)
    print("Author: ", author)
    indices = np.where(np.array(eval_authors) == author)[0]
    print(len(indices))
    print(indices)
    
    author_samples = eval_dataset[indices]
    print("Number of samples: ", len(indices))
    random_index = np.random.choice(indices)
    print("Random index: ", random_index)
    text = tokenizer.decode(eval_dataset[random_index], skip_special_tokens=True)
    texts = text.split('.')
    if len(texts) > 1:
        text = texts[1]
    else:
        text = texts[0]
    print(text)
    prompt = text
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(input_ids.shape)
    output_ids = model.generate(input_ids, max_length=128, num_return_sequences=1, 
                                do_sample=True, temperature=1, top_k=50, top_p=0.95)
    
    print("Generated sentence: ")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    prompt = text
    sentence = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    print("Prompt: ", prompt)
    print("Sentence: ", sentence)
    sample = {
        'prompt': prompt,
        'sentence': sentence,
        'author': author,
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "../data/samples_gutenberg"), exist_ok=True)
    torch.save(sample, os.path.join(os.path.dirname(__file__), "../data/samples_gutenberg", f"{author}.pt"))
    print("--------------------------------\n")