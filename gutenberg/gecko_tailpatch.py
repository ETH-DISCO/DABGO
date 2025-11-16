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
def tp_debug_log(samples,output_ids, prompt_length=1, model_path='gpt2-scratch-mixed'):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If using multi-GPU

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
    probability = 0
    for i in range(token_probs.shape[1]):
        probability += token_probs[0,i]
    probability = math.exp(probability.item())   
    return probability, original_probability, token_probs, original_probs



from datasets import load_from_disk
import json
with open('gutenberg/selected_dataset_mixed.json', 'r') as f:
    train_dataset = json.load(f)

train_dataset = np.array(train_dataset['train_data_np'])


examples = [
    """Albert Frederick Calvert_2.pt""",
    """Alice Ilgenfritz Jones_2.pt""",
    """Arthur Christopher Benson_2.pt""",
    """Augusta Huiell Seaman.pt""",
"""Augusta Huiell Seaman_2.pt""",
"""Carolyn Wells_2.pt""",
"""Charles Dickens_2.pt""",
"""Edgar Allan Poe_2.pt""",
"""Edith Wharton_2.pt""",
"""Edward Payson Roe_2.pt""",
"""Ernest Hemingway_2.pt""",
"""F. Scott Fitzgerald.pt""",
"""F. Scott Fitzgerald_2.pt""",
"""Henry Rowe Schoolcraft.pt""",
"""Henry Rowe Schoolcraft_2.pt""",
"""Herman Melville_2.pt""",
"""Howard Pyle_2.pt""",
"""J. Berg Esenwein_2.pt""",
"""Jane Austen_2.pt""",
"""Joseph H. Adams.pt""",
"""Joseph H. Adams_2.pt""",
"""Mark Twain.pt""",
"""Mark Twain_2.pt""",
"""Mary McNeil Fenollosa_2.pt""",
"""Oscar Wilde_2.pt""",
"""Ottwell Binns_2.pt""",
"""Randall Parrish_2.pt""",
"""Robert Moore Williams_2.pt""",
"""Rudyard Kipling.pt""",
"""Rudyard Kipling_2.pt""",
"""Stephen Marlowe.pt""",
"""Stephen Marlowe_2.pt""",
"""Virginia Woolf_2.pt""",
"""Wilbur B. Stover_2.pt""",
"""William Shakespeare_2.pt""",
]


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dfs = []

# Define label priority
label_priority = {
    "Highly Relevant": 2,
    "Somewhat Relevant": 1,
    "Not Relevant": 0,
}

def sort_key(item):
    idx, sim, nll, label, prob = item
    
    nll = float(nll)
    prob = float(prob)
    priority = label_priority.get(label, -1)
    score = (1.0 / nll if nll > 0 else float("inf")) + (1.0 / prob if prob > 0 else float("inf"))
    return (priority, score)
for num_steps in [10]:
    for i,example in enumerate(examples):
        print(example)
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), f'../out/gutenberg/gutenberg_experiments/finetuned_models/{example}_finetuned_fisher_10.pt'), map_location='cpu')
        output_ids = ckpt['output_ids']
        prompt_length = ckpt['prompt_length']
        print(tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True))
        gecko_results = np.load(f'data/gecko/gutenberg/{example}.npy')
        # Sort descending (best first)
        gecko_results_sorted = sorted(gecko_results, key=sort_key, reverse=True)
        print("\nSorted results:")
        samples_ft = []
        for rank, (i, sim, nll, label, p) in enumerate(gecko_results_sorted, 1):
            nll = float(nll)
            p = float(p)
            sample_ft = {
                'input_ids': torch.tensor(train_dataset[int(i)]),
                'attention_mask': torch.ones_like(torch.tensor(train_dataset[int(i)])),
                'labels': torch.tensor(train_dataset[int(i)])
            }
            samples_ft.append(sample_ft)
        print(len(samples_ft))
        p_1, original, _, _ = tp_debug_log(samples_ft[:1],output_ids=output_ids, prompt_length=prompt_length)
        p_3, original, _, _ = tp_debug_log(samples_ft[:3],output_ids=output_ids, prompt_length=prompt_length)
        p_5, original, _, _ = tp_debug_log(samples_ft[:5],output_ids=output_ids, prompt_length=prompt_length)
        p_7, original, _, _ = tp_debug_log(samples_ft[:7],output_ids=output_ids, prompt_length=prompt_length)
        p_10, original, _, _ = tp_debug_log(samples_ft[:10],output_ids=output_ids, prompt_length=prompt_length)
        p_15, original, _, _ = tp_debug_log(samples_ft[:15],output_ids=output_ids, prompt_length=prompt_length)
        p_20, original, _, _ = tp_debug_log(samples_ft[:20],output_ids=output_ids, prompt_length=prompt_length)
        dfs.append({
            'method': 'Gecko',
            'num_steps': num_steps,
            'name': example,
            '1': p_1,
            '3': p_3,
            '5': p_5,
            '7': p_7,
            '10': p_10,
            '15': p_15,
            '20': p_20,
        })
            
        
        dfs.append({
            'method': 'Original',
            'num_steps': num_steps,
            'name': example,
            '1': original,
            '3': original,
            '5': original,
            '7': original,
            '10': original,
            '15': original,
            '20': original,
        })
        print('----------------------------------------')
                
        total_df = pd.DataFrame(dfs)
        os.makedirs('data/results', exist_ok=True)
        total_df.to_csv(f'data/results/results_aggregated_gecko_gutenberg_test.csv', index=False)
        print('\n\n\n\n')
total_df = pd.DataFrame(dfs)
os.makedirs('data/results', exist_ok=True)
total_df.to_csv('data/results/results_aggregated_gecko_gutenberg_test.csv', index=False)

