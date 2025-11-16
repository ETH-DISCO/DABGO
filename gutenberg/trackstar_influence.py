## Compute Trackstar Influence

## Load Query vector
## Load All Sample vectors
## Sort by name grads_x.pt
## Compute Influence and append to one long list
## Save list to file also during checkpoints


import torch
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample_gradients', type=str, nargs='+', required=True,
                    help='One or more sample gradient names')
parser.add_argument('--dataset', type=str, default='gutenberg',
                    help='Dataset to use')

args = parser.parse_args()
gradient_dir = os.path.join(os.path.dirname(__file__), f"../data/trackstar/{args.dataset}/gradients")
gradient_files = [os.path.join(gradient_dir, f) for f in os.listdir(gradient_dir) if f.endswith(".pt") and f.startswith("normed_grads")]
print(len(gradient_files))
import re

from tqdm import tqdm



def compute_influence(sample_gradients, gradient_files, sample_grad_name, sample_gradient_dir):
        influence_list = torch.zeros(len(gradient_files)*20000)
        j = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Computing influence for ', sample_grad_name)
        for gradient_file in tqdm(gradient_files, total=len(gradient_files), desc="Computing influence"):
            gradient = torch.load(gradient_file, map_location=device)
            for i in range(len(gradient)):
                grad = gradient[i]
                for key, value in grad.items():
                    curr_infl = torch.dot(sample_gradients[key], value)
                    
                    influence_list[j] += curr_infl
                if j % 20000 == 0:
                    np.save(os.path.join(sample_gradient_dir, f"influence_list_{sample_grad_name}.npy"), influence_list)
                    print(f"Saved influence list to {os.path.join(sample_gradient_dir, f'influence_list_{sample_grad_name}.npy')}")
                j += 1
        np.save(os.path.join(sample_gradient_dir, f"influence_list_{sample_grad_name}.npy"), influence_list)
        print(f"Saved influence list to {os.path.join(sample_gradient_dir, f'influence_list_{sample_grad_name}.npy')}")
        return influence_list

sample_gradient_dir = os.path.join(os.path.dirname(__file__), f"../data/trackstar/{args.dataset}/testing/gradients")
gradient_files.sort(key=lambda f: int(re.search(r'normed_grads_(\d+)\.pt', os.path.basename(f)).group(1)))
print(gradient_files[:10])
sample_gradients = args.sample_gradients
print(sample_gradients)
print('Sample gradients: ', sample_gradients)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_autocorr_inv_sqrt = torch.load(os.path.join(gradient_dir, "autocorr_matrices_inv_sqrt.pt"), map_location=device)
for sample_grad_name in sample_gradients:
    sample_gradient = torch.load(os.path.join(sample_gradient_dir, f"{sample_grad_name}_gradient.pt"), map_location=device)
    print(sample_gradient.keys())
    influence_list = compute_influence(sample_gradient, gradient_files, sample_grad_name, sample_gradient_dir)
    np.save(os.path.join(sample_gradient_dir, f"influence_list_{sample_grad_name}.npy"), influence_list)
    print(f"Saved influence list to {os.path.join(sample_gradient_dir, f'influence_list_{sample_grad_name}.npy')}")