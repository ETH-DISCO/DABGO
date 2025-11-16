from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from datasets import load_dataset
import pandas as pd

# Load the WIT dataset (train split)
ds = load_dataset("google/wit", split="train", trust_remote_code=True, num_proc=4, download_mode="force_redownload",
)

# Filter for English-language entries
ds_en = ds.filter(lambda x: x["language"] == "en")

## Will only be using the context_page_description but keeping the rest for now. 
columns_to_keep = [
    "page_title",
    "section_title",
    "hierarchical_section_title",
    "context_page_description",
    "context_section_description",
]
ds_filtered = ds_en.remove_columns([col for col in ds_en.column_names if col not in columns_to_keep])
df = ds_filtered.to_pandas()
os.makedirs("../data", exist_ok=True)
df.to_csv("../data/wikipedia_intro_data.csv", index=False)

df = pd.read_csv("../data/wikipedia_intro_data.csv") ## From WIT dataset, comment this out if everything is already downloaded. 
print(df.head())
print(df.shape)
texts = df["context_page_description"].dropna().tolist()
print(len(texts))
dataset = Dataset.from_dict({"text": texts})

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

def tokenize_and_chunk(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        stride=64,
        return_overflowing_tokens=True,
        return_attention_mask=True,
    )


tokenized_dataset = dataset.map(
    tokenize_and_chunk,
    batched=True,
    remove_columns=["text"],
    batch_size=50,  
    num_proc=8      
)

train_dataset, eval_dataset = train_test_split(tokenized_dataset, test_size=0.1, random_state=42)

train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) > 32)
eval_dataset = eval_dataset.filter(lambda x: len(x['input_ids']) > 32)
train_dataset = train_dataset.map(
    lambda example, idx: {"new_idx": idx},
    with_indices=True
)

from datasets import DatasetDict


dataset_dict = DatasetDict({
    "train": train_dataset,
    "eval": eval_dataset,
})
os.makedirs("../data/training_data", exist_ok=True)
dataset_dict.save_to_disk("../data/training_data/tokenized_wit_dataset")


## BM25 

from datasets import load_from_disk
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import torch

batch_size = 10000
texts = []
train_dataset = load_from_disk("../data/training_data/tokenized_wit_dataset/train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print('starting tokenization')
for i in tqdm(range(0, len(train_dataset), batch_size)):
    batch_ids = train_dataset["input_ids"][i : i + batch_size]

    texts.extend(
        tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
    )
torch.save(texts, f"../data/training_data/untokenized_wiki.pt")


print('untokenized')
