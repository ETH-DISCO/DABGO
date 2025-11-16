## Scratch training
import pandas as pd
import torch
from torch.utils.data import Dataset
import argparse
import random
import numpy as np
from transformers import EarlyStoppingCallback
import wandb
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import accelerate  

device_count = torch.cuda.device_count()
print(f"Using {device_count} GPUs")
print("Current device:", torch.cuda.current_device())
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="wiki_model")
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--n_layer', type=int, default=16)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_embd', type=int, default=768)
args = parser.parse_args()



from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  
train_dataset = load_from_disk("../data/training_data/tokenized_wit_dataset/train")
eval_dataset = load_from_disk("../data/training_data/tokenized_wit_dataset/test")

print(train_dataset)
print(eval_dataset)
train_dataset.set_format(type="torch", columns=["input_ids"])
eval_dataset.set_format(type="torch", columns=["input_ids"])

print('split done')
block_size = 256
print(len(train_dataset))
print(len(eval_dataset))
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

from transformers import GPT2Config
config = GPT2Config.from_pretrained("gpt2-medium")
config.n_positions = 256
config.n_ctx = 256
config.n_embd = 768
config.n_head = 12
config.n_layer = 16
model = GPT2LMHeadModel(config)
model.gradient_checkpointing_enable()

num_train_epochs = args.num_epochs
batch_size = 32
gradient_accumulation_steps = 512//batch_size
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
wandb.init(
    project="wiki-model",
    name=args.model_name,
    config={
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "seed": SEED,
        "model_parameters": model.num_parameters(),
    }
)
fp16=torch.cuda.is_available()
total_steps = len(train_dataset) / batch_size / gradient_accumulation_steps * num_train_epochs
warmup_steps = int(total_steps * 0.03)
print(model.num_parameters())
print(model.config)
training_args = TrainingArguments(
    output_dir=f"../out/{args.model_name}",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=1e-4,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    save_total_limit=4,
    logging_dir='./logs',
    eval_strategy="steps",
    max_grad_norm=1.0,
    report_to="wandb", 
    logging_steps=1000,
    fp16=fp16,
    half_precision_backend="amp",
    lr_scheduler_type="cosine", 
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)
print('training...')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.02)]
)

trainer.train()
trainer.save_model(f'../out/{args.model_name}')
trainer.save_state()
wandb.finish()