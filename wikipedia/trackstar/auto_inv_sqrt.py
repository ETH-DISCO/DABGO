import torch
import gc
import os
import re
auto_corr_files = [
    'autocorr_matrices_300000',
    'autocorr_matrices_500000',
    'autocorr_matrices_800000',
    'autocorr_matrices_1000000',
    'autocorr_matrices_1400000',
    'autocorr_matrices_1500000',
    'autocorr_matrices_1600000',
    'autocorr_matrices_1700000',
    'autocorr_matrices_1800000',
    'autocorr_matrices_1858975'
]
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blocks = ['group0_mlp', 'group0_attn', 'group1_mlp', 'group1_attn', 'group2_mlp', 'group2_attn', 'group3_mlp', 'group3_attn', 'group4_mlp', 'group4_attn', 'group5_mlp', 'group5_attn', 'group6_mlp', 'group6_attn', 'group7_mlp', 'group7_attn', 'final_ln']
sum_outer = {block: torch.zeros(4096, 4096, device=device) for block in blocks}  

for auto_corr_file in auto_corr_files:
    ## Adapt the filepath to wherever the individual chunks of autocorrelation matrices are stored on disk. We stored them in chunks and processed all chunks seperately to now add them together.
    ## One can also loop through all gradients and buffer all autocorrelation matrices in memory and then compute the inverse sqrt.
    auto_corr = torch.load(f"data/trackstar/wiki/gradients/{auto_corr_file}.pt", map_location='cpu')
    for block in blocks:
        sum_outer[block] += auto_corr[block]
    del auto_corr
    gc.collect()
print(sum_outer)
eps = 1e-6
block_autocorr_inv_sqrt = {}
for block, R in sum_outer.items():
    print(f"Computing inverse sqrt for block {block}...")
    R_cpu = R.cpu()
    w, V = torch.linalg.eigh(R_cpu)
    w = torch.clamp(w, min=eps)
    R_inv_sqrt = (V * w.rsqrt().unsqueeze(0)) @ V.T
    block_autocorr_inv_sqrt[block] = R_inv_sqrt.to(device)

torch.save(block_autocorr_inv_sqrt, os.path.join(os.path.dirname(__file__), f"autocorr_matrices_inv_sqrt.pt"))
print("Saved inverse sqrt matrices.")
