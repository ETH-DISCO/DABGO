# extract_embeddings.py
# pip install transformers torch datasets

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import GPT2Model, AutoTokenizer
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load dataset ---
train_dataset = load_from_disk("data/training_data/tokenized_wit_dataset/train")

# --- Load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained("out/wiki_model").to(device).eval()

@torch.no_grad()
def meanpool_last_hidden(input_ids, attention_mask):
    """
    input_ids: [B, T]
    attention_mask: [B, T]
    returns: [B, H] embeddings (L2 normalized)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    h = out.last_hidden_state                          # [B, T, H]
    mask = attention_mask.unsqueeze(-1).type_as(h)     # [B, T, 1]
    summed = (h * mask).sum(dim=1)                     # [B, H]
    denom = mask.sum(dim=1).clamp(min=1.0)             # [B, 1]
    pooled = summed / denom                            # [B, H]
    emb = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return emb.cpu().numpy().astype("float32")

# --- DataLoader over tokenized dataset ---
def make_loader(ds, batch_size=64):
    def collate(batch):
        ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        att = [torch.ones_like(x["input_ids"], dtype=torch.long) for x in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        att = pad_sequence(att, batch_first=True, padding_value=0)
        return {"input_ids": ids, "attention_mask": att}
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

# --- Encode full dataset ---
loader = make_loader(train_dataset, batch_size=64)
N = len(train_dataset)
H = model.config.n_embd
all_embs = np.zeros((N, H), dtype="float32")

idx = 0
for batch in tqdm(loader):
    ids = batch["input_ids"].to(device)
    att = batch["attention_mask"].to(device)
    embs = meanpool_last_hidden(ids, att)   # [B, H]
    bsz = embs.shape[0]
    all_embs[idx:idx+bsz] = embs
    idx += bsz
    if idx % 100 == 0:
        print(f"Processed {idx}/{N}")
        np.save("wikipedia/train_embeddings.npy", all_embs)
# Save to disk for later use
np.save("wikipedia/train_embeddings.npy", all_embs)
print("Done! Saved embeddings to train_embeddings.npy")
