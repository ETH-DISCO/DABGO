import torch, math, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAMA_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

llama_tok = AutoTokenizer.from_pretrained(LLAMA_ID, use_fast=True)
if llama_tok.pad_token is None:
    llama_tok.pad_token = llama_tok.eos_token
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

LABELS = ["Not Relevant", "Somewhat Relevant", "Highly Relevant"]

def build_messages(query: str, candidate: str):
    instruction = (
        "You are a careful relevance judge. "
        "Given a query and a candidate sentence, rate relevance as one of: "
        + ", ".join(LABELS) + ". Answer with exactly one label."
    )
    user_content = f"Query: {query}\nCandidate: {candidate}\nRelevance:"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]

def truncate_to_tokens(text: str, max_tokens: int = 128) -> str:
    ids = llama_tok(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return llama_tok.decode(ids, skip_special_tokens=True)

@torch.no_grad()
def label_logprobs_llama_chat_efficient(messages, labels):
    """
    Efficient label scoring:
    1) Build chat prompt for assistant reply.
    2) One forward on prompt to get past_key_values.
    3) Score each label token-by-token reusing KV cache.
    """
    enc = llama_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    input_ids = enc["input_ids"].to(llama_model.device)
    attention_mask = enc["attention_mask"].to(llama_model.device) if "attention_mask" in enc else None

    # One forward to get past KV; we only keep the last token as input for next step
    out = llama_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    last_token = input_ids[:, -1:]  # shape [1,1]

    logps = []
    for lab in labels:
        lab_ids = llama_tok(lab, add_special_tokens=False, return_tensors="pt").input_ids.to(llama_model.device)
        total = 0.0
        cur_input = last_token
        cur_past = past

        # Feed label tokens one by one using cached KV (tiny memory)
        for tok_id in lab_ids[0]:
            logits = llama_model(input_ids=cur_input, past_key_values=cur_past, use_cache=True).logits
            cur_past = llama_model._past_key_values  if hasattr(llama_model, "_past_key_values") else None  # HF handles internally
            # safer way: get past from the model call return if available
            # but many builds don't return it unless return_dict=True and output_hidden_states=False:
            # out = llama_model(..., use_cache=True, return_dict=True)
            # logits, cur_past = out.logits, out.past_key_values

            # compute log prob of target tok_id
            lsm = torch.log_softmax(logits[:, -1, :], dim=-1)  # [1, V]
            total += float(lsm[0, tok_id].item())

            # Next step feeds the just-scored token as input
            cur_input = lab_ids[:, :1]  # set to current label token
            lab_ids = lab_ids[:, 1:]    # advance label sequence
            if lab_ids.shape[1] == 0:
                pass  # will finish after this iteration
        logps.append(total)

    return logps

def softmax_from_logps(logps):
    m = max(logps)
    probs = np.exp(np.array(logps) - m)
    probs = probs / probs.sum()
    return probs



## Query Likelihood
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LM with head (for log-likelihoods)
tok = GPT2Tokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lm = GPT2LMHeadModel.from_pretrained("out/wiki_model").to(device).eval()

@torch.no_grad()
def query_nll_given_prefix(prefix_ids, query_text, max_len=256):
    """
    prefix_ids: list[int] (tokenized training sample)
    query_text: str (your generated query sentence)
    returns: average NLL per query token (float)
    """
    q_ids = tok.encode(query_text, add_special_tokens=False)
    # Truncate prefix if too long
    keep = max_len - len(q_ids)
    if keep <= 0:
        prefix_ids = []
    else:
        prefix_ids = prefix_ids[-keep:]
    prefix_ids = list(prefix_ids)
    ids = prefix_ids + q_ids
    labels = [-100] * len(prefix_ids) + q_ids  # mask prefix
    att = [1] * len(ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([att], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)

    out = lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # out.loss is mean NLL across query tokens
    return float(out.loss.item())

import numpy as np

train_embeddings = np.load("train_embeddings.npy")
train_embeddings.shape


import numpy as np
import torch
from transformers import GPT2Model, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = GPT2Model.from_pretrained("out/wiki_model").to(device).eval()

@torch.no_grad()
def encode_query(text: str):
    x = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    out = model(**x)
    h = out.last_hidden_state             # [1, T, H]
    mask = x.attention_mask.unsqueeze(-1).type_as(h)
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype("float32")[0]

def top_k_neighbors(query: str, embs: np.ndarray, k=5):
    q = encode_query(query)               # [H]
    sims = embs @ q                       # cosine, since all normalized
    idx = np.argsort(-sims)[:k]           # top-k indices
    return [(int(i), float(sims[i])) for i in idx]



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
from datasets import load_from_disk
train_data = load_from_disk('data/training_data/tokenized_wit_dataset/train')
import os
for example in examples:
    
    ckpt = torch.load(f'out/wiki_models_finetuned/fisher_regularized_models/{example}_finetuned_fisher_10.pt', map_location='cpu')
    output_ids = ckpt['output_ids']
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    reranked = []
    topk = top_k_neighbors(output_text, train_embeddings, k=100)
    query = output_text
    for rank, (i, sim) in enumerate(topk, 1):
        prefix_ids = train_data[i]['input_ids']
        nll = query_nll_given_prefix(prefix_ids, query)  
        # decode candidate text with GPT-2 tokenizer you already loaded as `tokenizer`
        candidate_text = tokenizer.decode(prefix_ids)
        # decode candidate (you already have `tokenizer` for GPT-2)
        candidate_text = tokenizer.decode(prefix_ids)
        candidate_text = truncate_to_tokens(candidate_text, max_tokens=128)

        msgs = build_messages(query, candidate_text)
        lps = label_logprobs_llama_chat_efficient(msgs, LABELS)
        probs = softmax_from_logps(lps)
        best_j = int(np.argmax(probs))
        best_label = LABELS[best_j]
        best_prob = float(probs[best_j])
        
        reranked.append((i, sim, nll, best_label, best_prob))
            
    
    print("\nReranked top-10 (with relevance label & prob):")
    for rank, (i, sim, nll, label, p) in enumerate(reranked[:10], 1):
        snippet = tokenizer.decode(train_data[i]['input_ids'])[:80].replace("\n"," ")
        print(f"{rank:>2}. idx={i} | cos={sim:.4f} | NLL={nll:.4f} | {label} ({p:.2f}) | {snippet}...")
    reranked = np.array(reranked)
    os.makedirs(f'data/gecko', exist_ok=True)
    np.save(f'data/gecko/{example}.npy', reranked)