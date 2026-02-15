from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import argparse
from copy import deepcopy
from fisher_computation import compute_fisher_diagonal
import json
import numpy as np
from datasets import Dataset
def optimization(output_ids, prompt_length, model, fisher_diag, num_steps, sign=1, learning_rate=1e-4, dataset_size=17019):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    real_ids = output_ids.clone()
    B, L = real_ids.shape
    labels = real_ids.clone()
    labels[:, :prompt_length] = -100 ## Mask out prompt tokens for loss
    labels = labels.to(device)
    real_ids = real_ids.to(device)
    attention_mask = torch.ones(B, L, dtype=torch.long).to(device)
    attention_mask = attention_mask.to(device)
    original_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    for i in range(num_steps):
        model.zero_grad()
        outputs = model(input_ids=real_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  
        print(f"Step {i}: Loss: {loss.item()}")
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    
                    if name in fisher_diag:
                        fisher = fisher_diag[name]
                        if fisher.device != param.device:
                            fisher = fisher.to(param.device)                        
                        epsilon = 1e-6 
                        
                        inverse_fisher = (1.0 / (fisher + epsilon)) * param.grad
                        param.add_(sign * learning_rate / dataset_size * inverse_fisher) 

    return model

def main(args, tokenizer,base_dir, model, device, dataset_size):
    samples = os.listdir(os.path.join(base_dir, "data/samples_gutenberg"))
    samples = [name.replace('.pt', '') for name in samples]
    print(len(samples))
    print(samples)
    base_model_dict = deepcopy(model.state_dict())
    fisher_diag = torch.load(os.path.join(os.path.dirname(__file__), "fisher_diag/fisher_diag_normalized.pt"), map_location=device, weights_only=False)
    for sample in samples:
        sample_sentence = torch.load(os.path.join(base_dir, "data/samples_gutenberg", f"{sample}.pt"), map_location='cpu', weights_only=False)
        sentence = sample_sentence['sentence']
        prompt = sample_sentence['prompt']
        full_sentence = prompt + sentence
        print(f"Sample: {sample}")
        print(f"Prompt: {prompt}")
        print(f"Sentence: {sentence}")
        print("--------------------------------")
        
        output_ids = tokenizer.encode(full_sentence, add_special_tokens=False, return_tensors='pt')
        if output_ids.shape[1] > model.config.n_positions:
            output_ids = output_ids[:, :model.config.n_positions]
            print(f"Output ids shape: {output_ids.shape}")
        
        prompt_length = len(tokenizer.encode(prompt))
        print("Running descent...")
        descent_model = GPT2LMHeadModel(model.config)
        descent_model.load_state_dict(base_model_dict)
        descent_model = optimization(output_ids, prompt_length, descent_model, fisher_diag, args.descent_steps, sign=-1, learning_rate=args.learning_rate, dataset_size=dataset_size)
        print("Running ascent...")
        ascent_model = GPT2LMHeadModel(model.config)
        ascent_model.load_state_dict(base_model_dict)
        ascent_model = optimization(output_ids, prompt_length, ascent_model, fisher_diag, args.ascent_steps, sign=1, learning_rate=args.learning_rate, dataset_size=dataset_size)
        print("Saving models...")
        ckpt = {
            "sentence": sentence,
            "prompt": prompt,
            "descent_model": descent_model.state_dict(),
            "ascent_model": ascent_model.state_dict(),
            "ascent_steps": args.ascent_steps,
            "descent_steps": args.descent_steps,
            "learning_rate": args.learning_rate,
        }
        os.makedirs(os.path.join(base_dir, "out", "optimized_models"), exist_ok=True)
        torch.save(ckpt, os.path.join(base_dir, "out", "optimized_models", f"{sample}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descent_steps", type=int, default=10)
    parser.add_argument("--ascent_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--compute_fisher", action="store_true")
    args = parser.parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), "../")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(os.path.join(base_dir, "out/gpt2-scratch-mixed"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with open(os.path.join(base_dir, "gutenberg/selected_dataset_mixed.json"), "r") as f:
        data = json.load(f)
    print("loaded data")
    train_data = data['train_data_np']
    dataset_size = len(train_data)
    num_tokens = len(train_data[0]) * len(train_data)
    if args.compute_fisher:
        
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset = Dataset.from_dict({"input_ids": train_data})

        print(len(train_data))
        fisher_ckpt, num_grads =compute_fisher_diagonal(model, train_dataset, tokenizer, end_index=len(train_data), batch_size=1, device=device)
        fisher_diag = fisher_ckpt['fisher_diag']
        fisher_normalized = {name: fisher_diag[name].detach().clone() for name in fisher_diag}
        fisher_normalized = {name: fisher_normalized[name] / num_grads for name in fisher_normalized}
        os.makedirs(os.path.join(os.path.dirname(__file__), "fisher_diag"), exist_ok=True)
        torch.save(fisher_normalized, os.path.join(os.path.dirname(__file__), "fisher_diag/fisher_diag_normalized.pt"))

    main(args, tokenizer, base_dir, model, device, dataset_size)