import torch
import os
from collections import defaultdict
import torch.nn as nn
import json
import re
import math
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from datasets import Dataset
import numpy as np
from torch.utils.data import TensorDataset

class TrackstarUnbatched:
    def __init__(self, model, device='cuda', eps=1e-8):
        
        self.model = model.to(device).eval()
        self.device = device
        self.eps = eps
        self.second_moment = None
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.projector = None
        self.counter = 0
        self.grouped_grads = []
        self.number_of_samples = None

    def compute_grouped_gradients_batch(self, sample, group_size=2, source="original", sample_name="test"):
        
        self.model.zero_grad()

        input_ids = sample["input_ids"].to(self.device)         
        mask      = sample["attention_mask"].to(self.device)    

        labels = sample.get("labels", input_ids).to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        block_grads = defaultdict(list)
        for name, p in self.model.named_parameters():
            if p.grad is None or 'wte' in name: continue
            g = p.grad.detach().flatten()
            # assign to block
            
            if name.startswith('transformer.h.'):
                L = int(name.split('.')[2])
                b = L // group_size
                t = 'attn' if 'attn' in name else 'mlp'
                key = f'group{b}_{t}'
            elif name.startswith('transformer.ln_f'):
                key = 'final_ln'
            elif name.startswith('transformer.lm_head'):
                key = 'lm_head'
            else:
                continue
            block_grads[key].append(g)
        
        
        if source != "original":
            block_grads = {k: torch.cat(v, dim=0).detach() for k, v in block_grads.items()}
            block_grads = {key: g/(self.second_moment[key].to(self.device) + self.eps) for key, g in block_grads.items()}
            block_grads = self.projector.project_per_block(block_grads)

            os.makedirs(os.path.join(os.path.dirname(__file__), f'../../data/trackstar/gutenberg/sample_gradients/{args.source}'), exist_ok=True)
            torch.save([block_grads], os.path.join(os.path.dirname(__file__), f'../../data/trackstar/gutenberg/sample_gradients/{args.source}/{sample_name}.pt'))
            print(f"Saved grouped gradients to {os.path.join(os.path.dirname(__file__), f'../../data/trackstar/gutenberg/sample_gradients/{args.source}/{sample_name}.pt')}")
            return
        block_grads = {k: torch.cat(v, dim=0) for k, v in block_grads.items()}
        block_grads = {key: g/(self.second_moment[key].to(self.device) + self.eps) for key, g in block_grads.items()}

        block_grads = self.projector.project_per_block(block_grads)
        self.grouped_grads.append(block_grads)
        self.counter += 1
        if self.counter == self.number_of_samples-1 or self.counter % 10000 == 0:
            if source == "original":

                os.makedirs(os.path.join(os.path.dirname(__file__), f'../../data/trackstar/gutenberg/gradients'), exist_ok=True)
                torch.save(self.grouped_grads, os.path.join(os.path.dirname(__file__), f'../../data/trackstar/gutenberg/gradients/grads_{self.counter}.pt'))
            
                
            print('saved', len(self.grouped_grads), 'grouped gradients')
            self.grouped_grads = []
            torch.cuda.empty_cache()
            print('empty cache')
        return self.grouped_grads  

    @staticmethod
    def compute_R_inv_sqrt(block_projs, eps=1e-12):
        R_inv_sqrt = {}
        for k, Phi in block_projs.items():
            chunks = []
            for filename in Phi:
                chunks.append(torch.load(filename, weights_only=True))
            Phi = torch.cat(chunks, dim=0)
            print(Phi.shape)
            
            R = Phi.T @ Phi  # [d,d]
            w, V = torch.linalg.eigh(R)
            w_inv_sqrt = torch.clamp(w, min=eps).rsqrt()
            R_inv_sqrt[k] = (V * w_inv_sqrt.unsqueeze(0)) @ V.T
        return R_inv_sqrt


def compute_block_shapes(grouped_second_moment, embedding_dim):
    block_shapes = {}
    for key, vec in grouped_second_moment.items():
        total_dim = vec.numel()
        n = embedding_dim
        if total_dim % n != 0:
            raise ValueError(f"Dimension mismatch in {key}: total {total_dim} not divisible by embedding_dim {n}")
        m = total_dim // n
        block_shapes[key] = (m, n)
    return block_shapes

class BlockProjector:
    def __init__(self, block_shapes, d=4096, device='cuda'):
        self.d = d
        self.sqrt_d = int(math.sqrt(d))
        self.device = device
        self.proj_matrices = {}
        torch.manual_seed(0)
        for key, (m, n) in block_shapes.items():
            P0 = torch.randn(self.sqrt_d, m, device=device) / math.sqrt(self.sqrt_d)
            P1 = torch.randn(self.sqrt_d, n, device=device) / math.sqrt(self.sqrt_d)
            self.proj_matrices[key] = (P0, P1)
            
        
    def project_per_block(self, block_grads):
        out = {}
        for key, vec in block_grads.items():
            P0, P1 = self.proj_matrices[key]
            m, n = P0.shape[1], P1.shape[1]
            W = vec.view(m, n)
            out[key] = (P0 @ W @ P1.T).flatten()  
        return out


def get_block_shapes_from_model(model, group_size=2):
    block_dims = defaultdict(int)
    for name, p in model.named_parameters():
        if 'wte' in name: continue
        
        if name.startswith('transformer.h.'):
            L = int(name.split('.')[2])
            b = L // group_size
            t = 'attn' if 'attn' in name else 'mlp'
            key = f'group{b}_{t}'
        elif name.startswith('transformer.ln_f'):
            key = 'final_ln'
        elif name.startswith('transformer.lm_head'):
            key = 'lm_head'
        else: continue
        
        block_dims[key] += p.numel()

    n = model.config.n_embd
    shapes = {}
    for key, total_dim in block_dims.items():
        shapes[key] = (total_dim // n, n)
    return shapes

def compute_second_moment_blockwise(opt_path, n_layers=16, eps=1e-8, group_size=2):
    ckpt   = torch.load(opt_path, map_location="cpu")
    state  = ckpt["state"]
    pg     = ckpt["param_groups"]
    beta2  = pg[0]["betas"][1]

    second_moment = defaultdict(list)

    
    for idx in range(2, 2 + 4 * n_layers):
        rel = idx - 2
        layer, pos = divmod(rel, 4)
        entry = state[idx]
        # bias‐correct and sqrt:
        v = entry["exp_avg_sq"]
        v = v.flatten()

        # pos 0 or 1 → attn; pos 2 or 3 → mlp
        b = layer // group_size
        key = f"group{b}_attn" if pos < 2 else f"group{b}_mlp"
        
        
        second_moment[key].append(v)
    base = 66
    for idx in range(base, base + 8 * n_layers):
        rel = idx - base
        layer, pos = divmod(rel, 8)
        entry = state[idx]
        v = entry["exp_avg_sq"]
        v = v.flatten()
        b = layer // group_size
        key = f"group{b}_attn" if pos < 4 and pos>=2 else f"group{b}_mlp"
        second_moment[key].append(v)
    for idx in (194, 195):
        entry = state[idx]
        v = entry["exp_avg_sq"]
        v = v.div(1 - beta2 ** entry["step"]).sqrt().add_(eps).flatten()
        second_moment["final_ln"].append(v)

    return { k: torch.cat(vs, dim=0) for k, vs in second_moment.items() }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10)
    parser.add_argument("--source", type=str, default="original")
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Second Moment Computation and grouping. 
    ## Compute block shapes from grouped second moments
    ## Use these block shapes to initialize a BlockProjector which essentially initializes projection matrices for each block


    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), '../../out/gpt2-scratch-mixed/checkpoint-7980'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = torch.load(os.path.join(os.path.dirname(__file__), '../../out/gpt2-scratch-mixed/checkpoint-7980/optimizer.pt'), map_location=device)
    print(optimizer.keys())

    opt_path = os.path.join(os.path.dirname(__file__), '../../out/gpt2-scratch-mixed/checkpoint-7980/optimizer.pt')
    print("Computing second moment...")
    second_moment  = compute_second_moment_blockwise(opt_path, n_layers=model.config.n_layer)
    for k, vec in second_moment.items():
        print(f"{k:12s} → {tuple(vec.shape)}")
    os.makedirs(os.path.join(os.path.dirname(__file__), '../../data/trackstar/gutenberg/second_moment'), exist_ok=True)
    torch.save(second_moment, os.path.join(os.path.dirname(__file__), '../../data/trackstar/gutenberg/second_moment/second_moment.pt'))
    second_moment = torch.load(os.path.join(os.path.dirname(__file__), '../../data/trackstar/gutenberg/second_moment/second_moment.pt'), map_location=device)
    print("Second Moment Loaded...")


    
    ## Last checkpoint for second moment approximation
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), '../../out/gpt2-scratch-mixed'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded Model and Tokenizer...")
    
    print("Computing block shapes from model...")
    block_shapes = get_block_shapes_from_model(model, group_size=2)
    for k, v in block_shapes.items():
        print(k, v)
    
    print("Initializing BlockProjector...")
    proj = BlockProjector(block_shapes, d=4096, device=device)
    print("BlockProjector initialized...")
    ## Compute projected gradients
    ## Go through the dataset in batches 
    ## For each batch, compute the gradient of each sample in the batch and group the gradients by blocks defined before
    ## For each gradient in the batch, normalize it with its corresponding second moment block
    ## Each gradient is essentially a dictionary of grouped block names, which will be passed to the projector
    ## Projector is initialized before and projects each gradient dictionary and appends it to a list of gradients. 
    ## Once the list reaches a certain cutoff, store it in a file. 
    print("Initializing TrackstarUnbatched...")
    trackstar = TrackstarUnbatched(model, device=device)
    trackstar.second_moment = second_moment
    trackstar.projector = proj
    print("TrackstarUnbatched initialized...")
    print("Loading train dataset...")
    names = None
    
    with open(os.path.join(os.path.dirname(__file__), '../selected_dataset_mixed.json'), 'r') as f:
        train_dataset = json.load(f)
    train_dataset = np.array(train_dataset['train_data_np'])
    trackstar.number_of_samples = len(train_dataset)
    train_dataset = train_dataset[args.start_idx:args.end_idx]
    
    print('number of samples', trackstar.number_of_samples)
    trackstar.counter = args.start_idx
    print('counter', trackstar.counter)
    # Convert to tensor
    input_ids_tensor = torch.tensor(train_dataset, dtype=torch.long)

    pad_token_id = 50256
    attention_mask_tensor = (input_ids_tensor != pad_token_id).long()

    dataset = TensorDataset(input_ids_tensor, attention_mask_tensor)

    # No need for a collator
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # Loop and compute gradients
    print('starting to compute gradients')
    for i, (input_ids, attention_mask) in tqdm(enumerate(loader), total=len(loader)):
        batch = {
            "input_ids": input_ids.to(trackstar.device),
            "attention_mask": attention_mask.to(trackstar.device),
            "labels": input_ids.to(trackstar.device),  # labels same as input_ids for LM
        }
        trackstar.compute_grouped_gradients_batch(batch, group_size=2)
            
        