import os
import argparse
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
import numpy as np
import math
import gc
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
from copy import deepcopy
import pandas as pd
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  


pad_token_id = 50256
def collate_fn(batch):

    input_ids = [sample['input_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()  
    }

def tailpatch(samples,output_ids, prompt_length, model,lr=1e-5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_ids = output_ids.to(device)

    model.to(device)
    model.eval()
    optimizer = AdamW(model.parameters(), lr=lr)
    with torch.no_grad():
        model_pass = model(output_ids, labels=output_ids)

    logits = model_pass.logits
    logits = logits[:, prompt_length-1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = output_ids[:, prompt_length:]
    token_probs = log_probs.gather(2, target_ids.unsqueeze(-1))
    token_probs = token_probs.squeeze(-1)
    original_log_probability_sum = torch.sum(token_probs)

    batch_size = 1
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.train()
    optimizer.zero_grad()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss 
        loss.backward()
        del outputs, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    log_probability_sum = torch.sum(token_probs)
    del optimizer
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()
    p = math.exp(log_probability_sum.item())
    original = math.exp(original_log_probability_sum.item())
    return p, original




def main(num_samples, lr, save_dir, method):
    base_dir = os.path.join(os.path.dirname(__file__), "../")
    print(base_dir)
    with open(os.path.join(os.path.dirname(__file__), "selected_dataset_mixed.json"), "r") as f:
        data = json.load(f)
    print(data.keys())
    train_data = data['train_data_np']
    train_data_authors = data['train_data_authors']

    train_data = torch.tensor(train_data)
    print(train_data.shape)
    samples = os.listdir(os.path.join(os.path.dirname(__file__), "gecko/sample_scores"))
    samples = [name.replace('.npy', '') for name in samples]
    print(len(samples))
    print(samples)
    model = GPT2LMHeadModel.from_pretrained(os.path.join(base_dir, "out/gpt2-scratch-mixed"))
    base_model_dict = deepcopy(model.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded model")
    print(model.config)
    results = []
    average_score = 0
    for s, sample in enumerate(samples):
        model.load_state_dict(base_model_dict)
        sample_scores = [i for i in range(num_samples)]
        print(f"Sample: {sample}")
        
        if method == "gecko":
            sample_scores = np.load(os.path.join(os.path.dirname(__file__), "gecko/sample_scores", f"{sample}.npy"))
            sample_scores = np.array(sample_scores)
            sample_scores = sample_scores.argsort()[::-1]
            sample_scores = sample_scores[:num_samples]
        if method == "random":
            sample_scores = random.sample(range(len(train_data)), num_samples)
            print(sample_scores[:num_samples])
        if method == "bm25":
            sample_scores = np.load(os.path.join(os.path.dirname(__file__), "bm25", "sample_scores", f"{sample}.npy"))
            sample_scores = sample_scores.argsort()[::-1]
            sample_scores = sample_scores[:num_samples]
        if method == "trackstar":
            trackstar_influence = np.load(os.path.join(base_dir, f"data/trackstar/gutenberg/sample_scores/influence_list_{sample}.npy"))
            sample_scores = trackstar_influence[:len(train_data)]
            sample_scores = sample_scores.argsort()[::-1]
            sample_scores = sample_scores[:num_samples]
        if method == "dabgo":
            ascent_losses = np.load(os.path.join(os.path.dirname(__file__), "../data/losses/gutenberg", f"{sample}", "ascent", f"losses_{sample}_ascent.npy"))
            descent_losses = np.load(os.path.join(os.path.dirname(__file__), "../data/losses/gutenberg", f"{sample}", "descent", f"losses_{sample}_descent.npy"))
            sample_scores = np.abs(ascent_losses - descent_losses)
            sample_scores = sample_scores.argsort()[::-1]
            sample_scores = sample_scores[:num_samples]
        sample_sentence = torch.load(os.path.join(base_dir, "data/samples_gutenberg", f"{sample}.pt"), map_location='cpu', weights_only=False)
        sentence = sample_sentence['sentence']
        prompt = sample_sentence['prompt']
        full_sentence = prompt + sentence
        
        output_ids = tokenizer.encode(full_sentence, add_special_tokens=False, return_tensors='pt')
        if output_ids.shape[1] > model.config.n_positions:
            output_ids = output_ids[:, :model.config.n_positions]
            print(f"Output ids shape: {output_ids.shape}")
            
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        attributed_samples = []
        for i in range(num_samples):
            print(train_data_authors[sample_scores[i]])
            attributed_samples.append({
                'input_ids': train_data[sample_scores[i]],
                'attention_mask': torch.ones_like(train_data[sample_scores[i]]),
                'labels': train_data[sample_scores[i]]
            })
        
        probability, original_probability = tailpatch(attributed_samples, output_ids, prompt_length, model, lr)
        print(f"p: {probability}")
        print(f"original: {original_probability}")
        print(f"absolute probability difference: {np.abs(probability - original_probability)}")
        print(f"\n\npercentage difference: {np.abs(probability - original_probability) / original_probability * 100}%\n\n")
        average_score += (np.abs(probability - original_probability) / original_probability* 100)
        results.append({
            'num_samples': num_samples,
            'title': sample,
            'method': method,
            'source': 'gutenberg',
            'lr': lr,
            'steps': 10,
            'updated_probability': probability,
            'original_probability': original_probability,
            'absolute_probability_difference': np.abs(probability - original_probability),
            'text_tokens': 128,
            'prompt_length': prompt_length,
            'full_sentence': full_sentence,
            'prompt': prompt
        })
    df = pd.DataFrame(results)
    print(f"\n\nAverage score: {average_score / len(samples)}\n\n")
    os.makedirs(os.path.join(base_dir, "data/results_dir", save_dir), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/results_dir", f"{save_dir}",f"{method}", 'gutenberg', f'{num_samples}'), exist_ok=True)
    df.to_csv(os.path.join(base_dir, "data/results_dir", f"{save_dir}",f"{method}", 'gutenberg', f'{num_samples}', f'results.csv'), index=False)
    print(f"Results saved to {os.path.join(base_dir, "data/results_dir", f"{save_dir}",f"{method}", 'gutenberg', f'{num_samples}', f'results.csv')}")
    
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default='results')
    parser.add_argument("--method", type=str, default='random')
    args = parser.parse_args()
    main(args.num_samples, args.lr, args.save_dir, args.method)
    