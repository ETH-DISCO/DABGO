import torch
import os
from tqdm import tqdm
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=0)
args = parser.parse_args()


def compute_blockwise_autocorrelation(gradient_files, eps=1e-12, device='cpu', start_index=0):

    from collections import defaultdict
    blocks = ['group0_mlp', 'group0_attn', 'group1_mlp', 'group1_attn', 'group2_mlp', 'group2_attn', 'group3_mlp', 'group3_attn', 'group4_mlp', 'group4_attn', 'group5_mlp', 'group5_attn', 'group6_mlp', 'group6_attn', 'group7_mlp', 'group7_attn', 'final_ln']
    sum_outer = {block: torch.zeros(4096, 4096, device=device) for block in blocks}    
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for file_path in tqdm(gradient_files, total=len(gradient_files), desc="Accumulating autocorr"):
        gradients_list = torch.load(file_path, map_location=device)  # list of dicts
        print(file_path)
        
        if type(gradients_list[0]) == list:
            gradients_list = [gradients_list[i][j] for i in range(len(gradients_list)) for j in range(len(gradients_list[i]))]
        block_vectors = defaultdict(list)
        for grad in gradients_list:
            
            for block, vec in grad.items():
                block_vectors[block].append(vec.view(1, -1))  # [1, 4096]
                
            count += 1

            if count % 10000 == 0:  # Save memory regularly
                print(f"Processing batch of {count}")
                with torch.no_grad():
                    for block, vec_list in block_vectors.items():
                        if vec_list:  
                            stacked = torch.cat(vec_list, dim=0).to(device)  # [B, 4096]
                            sum_outer[block] += stacked.T @ stacked
                    block_vectors.clear()  
                    torch.cuda.empty_cache()
            if count % 100000 == 0:
                torch.save(sum_outer, os.path.join(gradient_dir, f"autocorr_matrices_{count+start_index}.pt"))
                print('saved', count)
        del gradients_list
        torch.cuda.empty_cache()
        print(count)
    torch.save(sum_outer, os.path.join(gradient_dir, f"autocorr_matrices_{count+start_index}.pt"))
    
    
    return sum_outer


end_index = args.end_index
start_index = args.start_index
gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/wiki/gradients")
gradient_files = [os.path.join(gradient_dir, f) for f in os.listdir(gradient_dir) if f.endswith(".pt") and f.startswith("grads")]

gradient_files = sorted(
    [f for f in gradient_files if int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1)) > start_index and int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1)) < end_index],
    key=lambda f: int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1))
)
print(len(gradient_files))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('starting to compute autocorrelation matrices')
autocorr_matrices = compute_blockwise_autocorrelation(gradient_files, device=device, start_index=start_index)



## Run this part once all autocorrelation matrices are computed in auto_inv_sqrt.py
## Save this periodically due to very high memory usage. be aware disk space will be very high up to 1TB or more for all gradients.
@torch.no_grad()
def compute_normed_vectors_batched(gradient_files, autocorr_matrix, batch_size=10000, start_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pbar = tqdm(total=len(gradient_files), desc="Normalizing gradients")

    for file_path in gradient_files:
        pbar.update(1)
        print(f"Processing: {file_path}")

        gradients_list = torch.load(file_path, map_location='cpu')

        if isinstance(gradients_list[0], list):
            gradients_list = [g for sub in gradients_list for g in sub]

        num_grads = len(gradients_list)
        for batch_start in range(0, num_grads, batch_size):
            batch_end = min(batch_start + batch_size, num_grads)
            batch = gradients_list[batch_start:batch_end]
            print(batch_start, batch_end)

            block_vectors = {block: [] for block in autocorr_matrix}
            block_indices = {block: [] for block in autocorr_matrix}

            for idx, grad in enumerate(batch):
                for block, vec in grad.items():
                    block_vectors[block].append(vec.view(1, -1))
                    block_indices[block].append(batch_start + idx)

            for block, vectors in block_vectors.items():
                if not vectors:
                    continue
                R = autocorr_matrix[block].to(device)
                stacked = torch.cat(vectors, dim=0).to(device)
                transformed = (R @ stacked.T).T
                norms = torch.norm(transformed, dim=1, keepdim=True)
                normalized = torch.where(norms > 0, transformed / norms, transformed)
                for i, idx in enumerate(block_indices[block]):
                    gradients_list[idx][block] = normalized[i]

            del block_vectors, block_indices, batch
            torch.cuda.empty_cache()
        
        file_path = os.path.normpath(file_path)

        dir_path, filename = os.path.split(file_path)
        new_filename = f"normed_{filename}"
        new_file_path = os.path.join(dir_path, new_filename)
        print(f"Saving: {new_file_path}")
        torch.save(gradients_list, new_file_path)
        print(f"Saved: {new_file_path}")
        del gradients_list
        torch.cuda.empty_cache()

    pbar.close()

end_index = args.end_index
start_index = args.start_index
gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/wiki/gradients")
gradient_files = [os.path.join(gradient_dir, f) for f in os.listdir(gradient_dir) if f.endswith(".pt") and f.startswith("grads")]
gradient_files = sorted(
    [f for f in gradient_files if int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1)) > start_index and int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1)) < end_index],
    key=lambda f: int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1))
)
print('number of gradient files', len(gradient_files))
autocorr_matrix = torch.load(os.path.join(gradient_dir, 'autocorr_matrices_inv_sqrt.pt'))
print('starting to compute normed vectors')
compute_normed_vectors_batched(gradient_files, autocorr_matrix, batch_size=10000, start_index=start_index)
