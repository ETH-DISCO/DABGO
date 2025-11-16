### Datasets
### Attributed Samples in dictionary
### Retraining script like in attribution.py
### DataLoader and perplexity computation



import torch
import numpy as np
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import accelerate  
from transformers import EarlyStoppingCallback, TrainerCallback
import random
import wandb
import math as m
from collections import deque
from rank_bm25 import BM25Okapi



## Dataset Functions and loss collection functions
class GutenbergDatasetAuthorText(Dataset):
    def __init__(self, training_data,tokenizer ):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        # Generate the (N, B) matrix
        self.data = np.array(training_data)
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return as torch tensors
        input_ids = torch.tensor(self.data[idx], dtype=torch.long)
        target_ids = torch.tensor(self.data[idx], dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {'input_ids': input_ids, 'labels': target_ids, 'attention_mask': attention_mask}

from transformers import TrainerCallback
import torch
import wandb

class EvalLoggingSimple(TrainerCallback):
    def __init__(self, model_inputs, prompt_length, device):
        self.model_inputs = model_inputs.to(device)
        self.prompt_length = prompt_length
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        model.to(self.device)

        # Slice to desired length (optional)
        inputs = self.model_inputs.clone()
        labels = inputs.clone()
        labels[:, :self.prompt_length] = -100  # mask prompt
        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss.item()

            logits = outputs.logits  # (B, T, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Shift logits and targets for log-likelihood
            shifted_log_probs = log_probs[:, prompt_length:-1, :]
            shifted_labels = inputs[:, prompt_length+1:]

            # Gather log-probs for the true labels (no need for masking now)
            selected_log_probs = shifted_log_probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)
            log_likelihood = selected_log_probs.sum().item()

        print(f"[Epoch {state.epoch}] Masked Loss: {loss:.4f}")
        print(f"[Epoch {state.epoch}] Log-Likelihood: {log_likelihood:.4f}")

        wandb.log({
            "Masked Loss": loss,
            "NLL": -log_likelihood,
            'Probability': m.exp(log_likelihood),
        })

        return control




def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt2-scratch-mixed")
    parser.add_argument("--authors", nargs='+', required=True, help="List of authors")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="selected_dataset_mixed.json")
    parser.add_argument("--project_name", type=str, default="Retraining Gutenberg FTUN")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_base_path = os.path.join(os.path.dirname(__file__))
    data_path = os.path.join(data_base_path, args.data_path)
    

    with open(os.path.join(data_base_path, 'selected_dataset_mixed_decoded.json'), 'r') as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} documents.")
    print("Sample:", corpus[0][:300])
    corpus = np.array(corpus)

    print(len(corpus))
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    print('bm25 initialized')
    with open(data_path, 'r') as f:
        data = json.load(f)
    train_data = data['train_data_np']
    train_data_authors = data['train_data_authors']
    eval_data_authors = data['eval_data_authors']
    eval_data = data['eval_data_np']
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_base_path = os.path.join(os.path.dirname(__file__), '../out')
    model_path = os.path.join(model_base_path, args.model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(model.config)
    n_embd = model.config.n_embd
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    block_size = 128
    seeds = [42]
    del model
    torch.cuda.empty_cache()
    print(args.authors)
    for author in args.authors:
        print(f'author: {author}')
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), f'../out/gutenberg/gutenberg_experiments/finetuned_models/{author}.pt'))
        output_ids = ckpt['output_ids']
        prompt_length = ckpt['prompt_length']
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        print(output_ids)
        print(type(output_ids))
        print(output_ids.shape)
        print('Output ids:')
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        query = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.num_samples]
        
        num_samples = args.num_samples
        samples_indices = []
        print(f'top {num_samples} samples:')
        for i in range(num_samples):
            samples_indices.append(top_n[i][0])
            print(f'sample {i}: {corpus[top_n[i][0]]}')

        train_data_np = np.array(train_data)
        train_data_new = np.delete(train_data_np, np.array(samples_indices), axis=0)
        train_data_authors_new = np.delete(train_data_authors, np.array(samples_indices), axis=0)
        ## Train model on new training data 
        training_dataset = GutenbergDatasetAuthorText(train_data_new, tokenizer)
        eval_dataset = GutenbergDatasetAuthorText(eval_data, tokenizer)
        
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f'seed: {seed}')
            os.makedirs(os.path.join(model_base_path, f'retrained_models/{author}/bm25_samples_{num_samples}_seed_{seed}_final'), exist_ok=True)
            
            print('scratch training')
            config = GPT2Config(
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                block_size=block_size,
                vocab_size=50257,
                n_ctx=block_size,
                n_positions=block_size,
                pad_token_id=tokenizer.pad_token_id,
            )
            model = GPT2LMHeadModel(config)
            model.to(device)
            model.train()
            num_epochs = 20
            batch_size = 32
            gradient_accumulation_steps = 2
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
            print(model.config)
            training_args = TrainingArguments(
                output_dir=os.path.join(model_base_path, f'retrained_models/{author}/bm25_samples_{num_samples}_seed_{seed}'),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_epochs,
                save_strategy="epoch",
                learning_rate=1e-4,
                warmup_steps=100,
                weight_decay=0.01,
                save_total_limit=2,
                logging_dir='./logs',
                eval_strategy="epoch",
                report_to="wandb",
                max_grad_norm=1.0,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                seed=seed
            )
            print('training...')
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=eval_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.02), 
                        EvalLoggingSimple(output_ids, prompt_length, device)]
            )
            # inside loop
            wandb.init(
                project=args.project_name,
                name=f"bm25-samples_seed-{seed}_num_samples-{num_samples}",
                config={"mode": "bm25", "seed": seed, "num_samples": num_samples, "author": author},
                reinit=True 
            )
            wandb.watch(model, log=None)
            trainer.train()
            model.eval()
            model.to('cuda')
            model.to('cpu')
            os.makedirs(os.path.join(model_base_path, f'retrained_models/{author}/bm25_samples_{num_samples}_seed_{seed}_final'), exist_ok=True)
            trainer.save_model(f'out/retrained_models/{author}/bm25_samples_{num_samples}_seed_{seed}_final')
            trainer.save_state()
            del model
            torch.cuda.empty_cache()
            wandb.finish()

        