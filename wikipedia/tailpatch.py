import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import random
import gc
import math

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cpu")

from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
def collate_fn(batch):
    input_ids = [sample['input_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()  # Labels same as input_ids for LM
    }
import random
import numpy as np
import torch
def tp_debug_log(samples,output_ids, prompt_length=1, model_path='wiki_model'):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = GPT2LMHeadModel.from_pretrained(f'out/{model_path}')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    with torch.no_grad():
        model_pass = model(output_ids, labels=output_ids)

    logits = model_pass.logits
    logits = logits[:, prompt_length-1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = output_ids[:, prompt_length:]
    token_probs = log_probs.gather(2, target_ids.unsqueeze(-1))
    token_probs = token_probs.squeeze(-1)
    original_probability = 0
    for i in range(token_probs.shape[1]):
        original_probability += token_probs[0,i]
    original_probability = math.exp(original_probability.item())
    original_probs = token_probs.clone()

    batch_size = 1
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.train()
    optimizer.zero_grad()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / len(dataloader)
        loss.backward()
        del outputs, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        model_pass = model(output_ids, labels=output_ids)

    logits = model_pass.logits
    logits = logits[:, prompt_length-1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = output_ids[:, prompt_length:]
    token_probs = log_probs.gather(2, target_ids.unsqueeze(-1))
    token_probs = token_probs.squeeze(-1)
    probability = 1
    for i in range(token_probs.shape[1]):
        probability += token_probs[0,i]
    probability = math.exp(probability.item())   
    return probability, original_probability, token_probs, original_probs



from datasets import load_from_disk
train_dataset = load_from_disk("data/training_data/tokenized_wit_dataset/train")
from rank_bm25 import BM25Okapi
import torch
corpus = torch.load('data/training_data/untokenized_wiki.pt', map_location='cpu')
print(f"Loaded {len(corpus)} documents.")
print("Sample:", corpus[0][:300])
corpus = np.array(corpus)
print(len(corpus))
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


examples = [
    'ancient_rome',
    'impressionism',
    'ww2',
    'dna',
    'solar_system',
    'philosophy_mind',
    'jazz',
    'thermodynamics',
    'internet',
    'feminism',
    'everest',
    'iss',
    'ww1',
    'ancient_egypt',
    'ancient_greece',
    'art_deco',
    'big_bang',
    'buddhism',
    'democracy',
    'ecology',
    'genetics',
    'gothic_architecture',
    'probability',
    'renaissance',
    'shakespeare'
]

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dfs = []
for i, example in enumerate(examples):
    print(example)
    
    ckpt = torch.load(f'out/wiki_models_finetuned/fisher_regularized_models/{example}_finetuned_fisher_10.pt', map_location='cpu')
    output_ids = ckpt['output_ids']
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    splits = output_text.split('.')
    prompt = splits[0]+'.'
    print(prompt)
    prompt_length = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_length)
    print(prompt_length)
    losses_af = np.load(f'data/losses/wiki/finetuned/fisher_regularized/{example}_finetuned.npy')
    losses_bf = np.load(f'data/losses/wiki/losses_bf.npy')
    
    print('----------------------------------------')
    losses_unlearned = np.load(f'data/losses/wiki/unlearned/fisher_regularized/{example}_unlearned.npy')
    losses_diff_unlearned = np.abs(losses_af - losses_unlearned)
    sorted_indices_unlearned = np.argsort(losses_diff_unlearned)[::-1]
    samples_un_ft = []
    for i in range(100):
        
        sample_un_ft = {
            'input_ids': torch.tensor(train_dataset[int(sorted_indices_unlearned[i])]['input_ids']),
            'attention_mask': torch.ones_like(torch.tensor(train_dataset[int(sorted_indices_unlearned[i])]['input_ids'])),
            'labels': torch.tensor(train_dataset[int(sorted_indices_unlearned[i])]['input_ids'])
        }
        samples_un_ft.append(sample_un_ft)
    p_1_un, original_un, _, _ = tp_debug_log(samples_un_ft[:1],output_ids=output_ids, prompt_length=prompt_length)
    p_3_un, original_un, _, _ = tp_debug_log(samples_un_ft[:3],output_ids=output_ids, prompt_length=prompt_length)
    p_5_un, original_un, _, _ = tp_debug_log(samples_un_ft[:5],output_ids=output_ids, prompt_length=prompt_length)
    p_7_un, original_un, _, _ = tp_debug_log(samples_un_ft[:7],output_ids=output_ids, prompt_length=prompt_length)
    p_10_un, original_un, _, _ = tp_debug_log(samples_un_ft[:10],output_ids=output_ids, prompt_length=prompt_length)
    p_15_un, original_un, _, _ = tp_debug_log(samples_un_ft[:15],output_ids=output_ids, prompt_length=prompt_length)
    p_20_un, original_un, _, _ = tp_debug_log(samples_un_ft[:20],output_ids=output_ids, prompt_length=prompt_length)
    dfs.append({
        'method': 'FT & UN',
        'name': example,
        'num_steps': 10,
        '1': p_1_un,
        '3': p_3_un,
        '5': p_5_un,
        '7': p_7_un,
        '10': p_10_un,
        '15': p_15_un,
        '20': p_20_un,
    })
    dfs.append({
        'method': 'Original',
        'name': example,
        'num_steps': 10,
        '1': original_un,
        '3': original_un,
        '5': original_un,
        '7': original_un,
        '10': original_un,
        '15': original_un,
        '20': original_un,      
    })

    print('----------------------------------------')
    query = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    print(query)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:100]
    samples_bm25 = []
    for rank, (idx, score) in enumerate(top_n, 1):
        
        sample_bm25 = {
            'input_ids': torch.tensor(train_dataset[idx]['input_ids']),
            'attention_mask': torch.ones_like(torch.tensor(train_dataset[idx]['input_ids'])),
            'labels': torch.tensor(train_dataset[idx]['input_ids'])
        }
        samples_bm25.append(sample_bm25)
    p_1, original, _, _ = tp_debug_log(samples_bm25[:1],output_ids=output_ids, prompt_length=prompt_length)
    p_3, original, _, _ = tp_debug_log(samples_bm25[:3],output_ids=output_ids, prompt_length=prompt_length)
    p_5, original, _, _ = tp_debug_log(samples_bm25[:5],output_ids=output_ids, prompt_length=prompt_length)
    p_7, original, _, _ = tp_debug_log(samples_bm25[:7],output_ids=output_ids, prompt_length=prompt_length)
    p_10, original, _, _ = tp_debug_log(samples_bm25[:10],output_ids=output_ids, prompt_length=prompt_length)
    p_15, original, _, _ = tp_debug_log(samples_bm25[:15],output_ids=output_ids, prompt_length=prompt_length)
    p_20, original, _, _ = tp_debug_log(samples_bm25[:20],output_ids=output_ids, prompt_length=prompt_length)
    dfs.append({
        'name': example,
        'method': 'BM25',
        'num_steps': 10,
        '1': p_1,
        '3': p_3,
        '5': p_5,
        '7': p_7,
        '10': p_10,
        '15': p_15,
        '20': p_20,
    })
    

    trackstar_influence = np.load(f'data/trackstar/wiki/testing/gradients/influence_list_{example}.npy')
    print(trackstar_influence.shape)
    trackstar_influence = trackstar_influence[:len(train_dataset)] ## When running in chunks, last couple items are empty
    sorted_indices = np.argsort(trackstar_influence)[::-1]
    samples_ft = []
    for i in range(100):
        sample_ft = {
            'input_ids': torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids']),
            'attention_mask': torch.ones_like(torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids'])),
            'labels': torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids'])
        }
        samples_ft.append(sample_ft)
    p_1, original, _, _ = tp_debug_log(samples_ft[:1],output_ids=output_ids, prompt_length=prompt_length)
    p_3, original, _, _ = tp_debug_log(samples_ft[:3],output_ids=output_ids, prompt_length=prompt_length)
    p_5, original, _, _ = tp_debug_log(samples_ft[:5],output_ids=output_ids, prompt_length=prompt_length)
    p_7, original, _, _ = tp_debug_log(samples_ft[:7],output_ids=output_ids, prompt_length=prompt_length)
    p_10, original, _, _ = tp_debug_log(samples_ft[:10],output_ids=output_ids, prompt_length=prompt_length)
    p_15, original, _, _ = tp_debug_log(samples_ft[:15],output_ids=output_ids, prompt_length=prompt_length)
    p_20, original, _, _ = tp_debug_log(samples_ft[:20],output_ids=output_ids, prompt_length=prompt_length)
    dfs.append({
        'method': 'TrackStar',
        'name': example,
        'num_steps': 10,
        '1': p_1,
        '3': p_3,
        '5': p_5,
        '7': p_7,
        '10': p_10,
        '15': p_15,
        '20': p_20,
    })
        
    
    random_samples = np.random.choice(len(train_dataset), size=100)
    samples_random = []
    for sample in random_samples:
        samples_random.append({
            'input_ids': torch.tensor(train_dataset[int(sample)]['input_ids']),
            'attention_mask': torch.ones_like(torch.tensor(train_dataset[int(sample)]['input_ids'])),
            'labels': torch.tensor(train_dataset[int(sample)]['input_ids'])
        })
    p_1, original, _, _ = tp_debug_log(samples_random[:1],output_ids=output_ids, prompt_length=prompt_length)
    p_3, original, _, _ = tp_debug_log(samples_random[:3],output_ids=output_ids, prompt_length=prompt_length)
    p_5, original, _, _ = tp_debug_log(samples_random[:5],output_ids=output_ids, prompt_length=prompt_length)
    p_7, original, _, _ = tp_debug_log(samples_random[:7],output_ids=output_ids, prompt_length=prompt_length)
    p_10, original, _, _ = tp_debug_log(samples_random[:10],output_ids=output_ids, prompt_length=prompt_length)
    p_15, original, _, _ = tp_debug_log(samples_random[:15],output_ids=output_ids, prompt_length=prompt_length)
    p_20, original, _, _ = tp_debug_log(samples_random[:20],output_ids=output_ids, prompt_length=prompt_length)

    dfs.append({
        'method': 'Random',
        'name': example,
        'num_steps': 10,
        '1': p_1,
        '3': p_3,
        '5': p_5,
        '7': p_7,
        '10': p_10,
        '15': p_15,
        '20': p_20,
    })
    total_df = pd.DataFrame(dfs)
    os.makedirs('data/results', exist_ok=True)
    total_df.to_csv(f'data/results/{args.save_path}.csv', index=False)
    print('\n\n\n\n')
total_df = pd.DataFrame(dfs)
os.makedirs('data/results', exist_ok=True)
total_df.to_csv(f'data/results/{args.save_path}.csv', index=False)

