## Load Gecko samples get embeddings and order them by similarity to embeddings of training datasamples
import os
import torch
from transformers import GPT2Tokenizer
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel
import argparse
import numpy as np
from tqdm import tqdm
import json
import torch

def main(project_name, location, source, batch_size, start_index, end_index):
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vertex_init(project=project_name, location=location)
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    names = os.listdir(os.path.join(os.path.dirname(__file__), "../../data/samples_gutenberg"))
    names = [name.replace('.pt', '') for name in names]
    print(len(names))

    os.makedirs(os.path.join(os.path.dirname(__file__), 'sample_embeddings'), exist_ok=True)
    for name in tqdm(names):
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), "../../data/samples_gutenberg", f"{name}.pt"))
        samples = ckpt['sentence']
        prompt = ckpt['prompt']
        
        print(prompt)
        print(len(prompt))
        print(len(samples))
        samples = samples[len(prompt):]
        
        
        print(samples)
        if len(samples) == 0:
            continue
        resp = model.get_embeddings([samples])
        sample_embeddings = resp[0].values
        sample_embeddings = np.array(sample_embeddings)
        print(sample_embeddings.shape)
        np.save(os.path.join(os.path.dirname(__file__), 'sample_embeddings', f"{name}.npy"), sample_embeddings)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="gecko-479220")
    parser.add_argument("--location", type=str, default="europe-west4")
    parser.add_argument("--source", type=str, default="wikipedia")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    args = parser.parse_args()
    main(args.project_name, args.location, args.source, args.batch_size, args.start_index, args.end_index)