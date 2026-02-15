## Make an index of training samples with gecko embeddings and store the index as a list for tailpatch.py

import numpy as np
import os
import argparse
from tqdm import tqdm
def main(source):
    
    
    base_dir = os.path.dirname(__file__)
    gutenberg_embeddings = np.load(os.path.join(os.path.dirname(__file__), "data_embeddings", "gutenberg_embeddings.npy"))
    
    
    # Normalize in-place for cosine similarity
    norms = np.linalg.norm(gutenberg_embeddings, axis=1, keepdims=True)
    gutenberg_embeddings /= np.clip(norms, 1e-8, None)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'sample_scores'), exist_ok=True)
    sample_embeddings = os.listdir(os.path.join(base_dir, 'sample_embeddings'))
    print(len(sample_embeddings))
    sample_embeddings = [name.replace('.npy', '') for name in sample_embeddings]
    print(len(sample_embeddings))
    print("Normalized train embeddings")
    for name in tqdm(sample_embeddings):
        print(name)
        print(f"Processing {name}")
        
        embeddings_sample = np.load(os.path.join(base_dir,  'sample_embeddings', f"{name}.npy"))
        print(embeddings_sample.shape)
        norms = np.linalg.norm(embeddings_sample, axis=0, keepdims=True)
        embeddings_sample /= np.clip(norms, 1e-8, None)
        print(embeddings_sample.shape)
        print(f"Normalized embeddings sample")
        scores_total = gutenberg_embeddings @ embeddings_sample
        print(f"Cosine similarity: {scores_total.shape}")
        os.makedirs(os.path.join(base_dir, 'sample_scores'), exist_ok=True)
        
        np.save(os.path.join(base_dir, 'sample_scores', f"{name}.npy"), scores_total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="gutenberg")
    args = parser.parse_args()
    main(args.source)