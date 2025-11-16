import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from torch.nn import functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np

def compute_fisher_diagonal(model, dataset, tokenizer, device='cuda', batch_size=8, 
                                       max_batches=None, token_count=0):
    model.eval()
    model.to(device)
    print("Starting fresh Fisher diagonal computation")
    fisher_diag = {}
    for name, param in model.named_parameters():
        fisher_diag[name] = torch.zeros_like(param, device='cpu')
    total_tokens = token_count
    current_batch_tokens = 0
    for _ in range(2): 
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Fisher Diagonal (Incremental)"):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_diag[name] += (param.grad.detach().cpu() ** 2)

            batch_tokens = input_ids.numel()
            current_batch_tokens += batch_tokens
            total_tokens += batch_tokens
            model.zero_grad()
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} batches, total tokens: {total_tokens}")
                fisher_normalized = {}
                for name in fisher_diag:
                    fisher_normalized[name] = fisher_diag[name] / total_tokens
                
                #torch.save(fisher_normalized, os.path.join(os.path.dirname(__file__), "fisher_diag_intermediate.pt"))

        fisher_normalized = {}
        for name in fisher_diag:
            fisher_normalized[name] = fisher_diag[name] / total_tokens
        torch.save(fisher_normalized, os.path.join(os.path.dirname(__file__), "fisher_diag_wiki.pt"))
    print(f"Final Fisher diagonal computed with {total_tokens} total tokens")
    return fisher_normalized, total_tokens


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model_path = os.path.join(os.path.dirname(__file__), '../out/wiki_model')
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print('model loaded')

data_dir = os.path.join(os.path.dirname(__file__), '../data')
train_dataset = load_from_disk(os.path.join(data_dir, 'training_data/tokenized_wit_dataset'))
train_dataset = train_dataset['train']
train_dataset = train_dataset.remove_columns(['page_title', 'section_title'])
train_dataset.set_format('torch', columns=['input_ids'])

print('Training dataset loaded')
print("Calculating Fisher diagonal...")
fisher_diag, total_tokens = compute_fisher_diagonal(
    model, 
    train_dataset, 
    tokenizer, 
    device=device, 
    batch_size=30,
    token_count=0 
)
print('Finished computing Fisher diagonal')