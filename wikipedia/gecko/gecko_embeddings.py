## Load Gecko samples get embeddings and order them by similarity to embeddings of training datasamples
import os
import torch
from transformers import GPT2Tokenizer
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel
import argparse
import numpy as np
from tqdm import tqdm
def main(project_name, location, source, batch_size, start_index, end_index):
    
    training_data = torch.load(os.path.join(os.path.dirname(__file__), f"../../data/untokenized_wiki.pt"), weights_only=False)
    print(training_data.shape)
    training_data = training_data[start_index:end_index]
    print(training_data.shape)
    print(training_data[0])
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    vertex_init(project=project_name, location=location)
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    os.makedirs(os.path.join(os.path.dirname(__file__), "data_embeddings"), exist_ok=True)
    embeddings = []
    for i in tqdm(range(0, len(training_data), batch_size)):
        samples = training_data[i:i+batch_size]
        samples = samples.tolist()
        resp = model.get_embeddings(samples)
        embeddings.extend([r.values for r in resp])
        
        if len(embeddings) % 100000 == 0:
            print(len(embeddings))
            embeddings = np.array(embeddings)
            print(embeddings.shape)
            
            np.save(os.path.join(os.path.dirname(__file__), "data_embeddings", f"wikipedia_embeddings_{start_index}_{end_index}_{i}.npy"), embeddings)
            embeddings = []
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        print(embeddings.shape)
        np.save(os.path.join(os.path.dirname(__file__), "data_embeddings", f"wikipedia_embeddings_{start_index}_{end_index}_end.npy"), embeddings)
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