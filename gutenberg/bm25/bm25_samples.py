## BM25 getting all samples for later use in tailpatch.py
import os
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import load_from_disk
import json
import argparse
import torch
from transformers import GPT2Tokenizer

def main(sample_name, source):
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../../")
    with open(os.path.join(os.path.dirname(__file__), "../selected_dataset_mixed_decoded.json"), "r") as f:
        data = json.load(f)
    print(len(data))
    print(data[0])
    corpus = data
    print("Loaded docs:", len(corpus))
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    file_dir = os.path.dirname(__file__)
    
    
    sample_names = os.listdir(os.path.join(BASE_DIR, "data", "samples_gutenberg"))
    sample_names = [name.replace(".pt", "") for name in sample_names]
    print(sample_names)
    for name in sample_names:
        sample = torch.load(os.path.join(BASE_DIR, "data", "samples_gutenberg", f"{name}.pt"), map_location='cpu', weights_only=False)
        sentence = sample['sentence']
        prompt = sample['prompt']
        query = sentence[len(prompt):]
        print(f"Query: {query}")
        tokenized_query = query.split(" ")
        scores = bm25.get_scores(tokenized_query)
        print(f"BM25 scores for {name}: {scores[:5]}")
        os.makedirs(os.path.join(file_dir, "sample_scores"), exist_ok=True)
        scores = np.array(scores)
        np.save(os.path.join(file_dir, "sample_scores", f"{name}.npy"), scores)
        print(f"BM25 scores saved to file_dir/bm25/sample_scores/{name}.npy")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_name", type=str, default="All")
    parser.add_argument("--source", type=str, default="gutenberg")
    args = parser.parse_args()
    main(args.sample_name, args.source)