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
from collections import deque


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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt2-scratch-mixed")
    parser.add_argument("--data_path", type=str, default="selected_dataset_mixed.json")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_base_path = os.path.join(os.path.dirname(__file__), 'out')
    data_base_path = os.path.join(os.path.dirname(__file__))
    data_path = os.path.join(data_base_path, args.data_path)

    with open(data_path, 'r') as f:
        data = json.load(f)
    train_data = data['train_data_np']
    train_data_authors = data['train_data_authors']
    eval_data_authors = data['eval_data_authors']
    eval_data = data['eval_data_np']
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = GPT2Config.from_pretrained('gpt2-medium')
    config.n_layer=16
    config.pad_token_id = tokenizer.pad_token_id
    config.n_ctx = 128
    config.n_positions = 128
    model = GPT2LMHeadModel(config)
    model.to(device)
    model.train()
    print(model.config)
    training_dataset = GutenbergDatasetAuthorText(train_data, tokenizer)
    eval_dataset = GutenbergDatasetAuthorText(eval_data, tokenizer)

    num_epochs = 20
    batch_size = 32
    gradient_accumulation_steps = 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    print(model.config)
    training_args = TrainingArguments(
        output_dir=os.path.join(model_base_path, f'{args.model_path}'),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        learning_rate=1e-4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        logging_steps=10,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
    print('training...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.02)]
    )

    trainer.train()
    model.eval()
    model.to('cpu')
    os.makedirs(os.path.join(os.path.dirname(__file__), f'{args.model_path}'), exist_ok=True)
    trainer.save_model(f'out/{args.model_path}')
    trainer.save_state()
    

            