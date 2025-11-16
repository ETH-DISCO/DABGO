## Dataset "Selected_dataset_mixed"
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch 
import os
model_path = os.path.join(os.path.dirname(__file__), 'out/gpt2-scratch-mixed')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval()
print(model.config)
print(model.num_parameters())
import json
import numpy as np
import pickle
import torch
data_path = os.path.join(os.path.dirname(__file__), 'selected_dataset_mixed.json')
with open(data_path, 'r') as f:
    data = json.load(f)

print(len(data))
print(data.keys())
train_dataset = data['train_data_np']
train_authors = data['train_data_authors']
train_titles = data['train_data_titles']
eval_dataset = data['eval_data_np']
eval_authors = data['eval_data_authors']
eval_titles = data['eval_data_titles']

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

def collect_per_sample_losses(model, input_id_matrix, device, batch_size=4, pad_token_id=50256, cutoff=20000):
    model.eval()
    model.to(device)
    input_id_matrix = np.array(input_id_matrix)
    if isinstance(input_id_matrix, np.ndarray):
        input_id_matrix = torch.tensor(input_id_matrix, dtype=torch.long)

    attention_masks = (input_id_matrix != pad_token_id).long()

    dataset = TensorDataset(input_id_matrix, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_sample_losses = []
    processed_samples = 0

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating Losses"):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :]
            shift_labels = input_ids[..., 1:]
            shift_attention = attention_mask[..., 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token_loss = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            masked_loss = per_token_loss * shift_attention.float()

            valid_tokens = shift_attention.sum(dim=1).clamp(min=1)
            sample_losses = masked_loss.sum(dim=1) / valid_tokens

            all_sample_losses.extend(sample_losses.cpu().tolist())
            processed_samples += len(sample_losses)
            del input_ids, attention_mask, outputs, logits, per_token_loss, masked_loss, log_probs
            torch.cuda.empty_cache()

            if processed_samples >= cutoff:
                break

    return np.array(all_sample_losses, dtype=np.float32)
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

def compute_fisher_diagonal(model, input_id_matrix, pad_token_id=50256,
                                                    device='cuda', batch_size=8, 
                                                    max_epochs=2,
                                                    save_dir="fisher_approximation"):

    model.eval()
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    pad_token_id = 50256  
    if isinstance(input_id_matrix, np.ndarray):
        input_id_matrix = torch.tensor(input_id_matrix, dtype=torch.long)
    os.makedirs(os.path.join(os.path.dirname(__file__), save_dir), exist_ok=True)
    save_dir = os.path.join(os.path.dirname(__file__), save_dir)
    attention_masks = (input_id_matrix != pad_token_id).long()
    dataset = TensorDataset(input_id_matrix, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Starting fresh Fisher diagonal computation")
    fisher_diag = {name: torch.zeros_like(p, device='cpu') for name, p in model.named_parameters()}
    total_tokens = 0

    batches_processed = 0
    for epoch in range(max_epochs):
        print(f"\nStarting epoch {epoch + 1} / {max_epochs}")
        for batch_idx, (input_ids, attention_mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Fisher Diagonal"):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_diag[name] += (param.grad.detach().cpu() ** 2)

            total_tokens += input_ids.numel()
            model.zero_grad()
            batches_processed += 1

            if batches_processed % 1000 == 0:
                print(f"Checkpoint at {batches_processed} batches, {total_tokens} tokens")
                fisher_normalized = {k: v / total_tokens for k, v in fisher_diag.items()}
                #torch.save(fisher_normalized, os.path.join(save_dir, "fisher_diag_intermediate_gutenberg_mixed.pt"))
            del input_ids, attention_mask, labels, outputs, loss
            torch.cuda.empty_cache()

    # Final normalization
    fisher_normalized = {k: v / total_tokens for k, v in fisher_diag.items()}
    torch.save(fisher_normalized, os.path.join(save_dir, "fisher_diag_gutenberg_mixed.pt"))
    print(f"\nFinal Fisher diagonal computed over {total_tokens} tokens, {batches_processed} batches")
    return fisher_normalized, total_tokens


def finetuning_fisher(output_ids, prompt_length, fisher_diag, device, batch_size=1, 
                      finetuning_steps=10, ewc_weight=1, learning_rate=1e-4, unlearning_parameter=1):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.train()

    params_old = {name: param.detach().clone() for name, param in model.named_parameters()}
    input_ids = output_ids.clone().to(device)
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100  
    attention_mask = torch.ones_like(input_ids).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(finetuning_steps):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in fisher_diag and name in params_old:
                fisher = fisher_diag[name].to(device)
                param_old = params_old[name].to(device)
                ewc_loss += (fisher * (param - param_old).pow(2)).sum()

        total_loss =unlearning_parameter*( loss + (ewc_weight / 2) * ewc_loss)
        print(f"Epoch {step+1}, Loss: {total_loss.item()}, EWC Loss: {ewc_loss.item()}")

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del outputs, loss, total_loss, ewc_loss
        torch.cuda.empty_cache()

    return model 
print('Computing Fisher diagonal')
fisher_diag, total_tokens = compute_fisher_diagonal(
    model=model,
    input_id_matrix=train_dataset,  
    device=device,
    batch_size=20,
    max_epochs=2,
)
print('Fisher diagonal computed')
fisher_diag_path = os.path.join(os.path.dirname(__file__), 'fisher_diag_gutenberg_mixed.pt')
fisher_diag = torch.load(fisher_diag_path)
authors = list(set(eval_authors))
print(len(authors))
print(authors)

import numpy as np
np.random.seed(42)
losses_dir = os.path.join(os.path.dirname(__file__), 'data/losses/gutenberg/gutenberg_losses')
output_dir = os.path.join(os.path.dirname(__file__), 'out/gutenberg/gutenberg_experiments')
for author in authors:
    model = GPT2LMHeadModel.from_pretrained(model_path).eval().to(device)
    print(author)
    indices = np.where(np.array(eval_authors) == author)[0]
    print(len(indices))
    random_index = np.random.choice(indices)
    print(random_index)
    text = tokenizer.decode(eval_dataset[random_index], skip_special_tokens=True)
    texts = text.split('.')
    if len(texts) > 1:
        text = texts[1]
    else:
        text = texts[0]
    print(text)
    prompt = text
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(input_ids.shape)
    output_ids = model.generate(input_ids, max_length=128, num_return_sequences=1, 
                                do_sample=True, temperature=1, top_k=50, top_p=0.95)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    finetuned_model = finetuning_fisher(output_ids, prompt_length=input_ids.shape[1], fisher_diag=fisher_diag, device=device, batch_size=1, finetuning_steps=10,
                      ewc_weight=1, learning_rate=1e-4, unlearning_parameter=1)
    losses_ft = collect_per_sample_losses(finetuned_model, train_dataset, device, batch_size=30)
    losses_ft = np.array(losses_ft)
    print(losses_ft.mean())
    np.save(os.path.join(losses_dir, f'experiments_finetuned/losses_ft_{author}_2.npy'), losses_ft)
    ckpt = {
        'model': finetuned_model.state_dict(),
        'output_ids': output_ids,
        'prompt_length': input_ids.shape[1],
    }
    torch.save(ckpt, os.path.join(output_dir, f'finetuned_models/{author}_2.pt'))
    finetuned_model.cpu()
    del finetuned_model
    torch.cuda.empty_cache()
    model = GPT2LMHeadModel.from_pretrained(model_path).eval().to(device)
    unlearned_model = finetuning_fisher(output_ids, prompt_length=input_ids.shape[1], fisher_diag=fisher_diag, device=device, batch_size=1, finetuning_steps=10,
                      ewc_weight=1, learning_rate=1e-4, unlearning_parameter=-1)
    losses_unlearned = collect_per_sample_losses(unlearned_model, train_dataset, device, batch_size=30)
    losses_unlearned = np.array(losses_unlearned)
    print(losses_unlearned.mean())
    np.save(os.path.join(losses_dir, f'experiments_unlearned/losses_unlearned_{author}_2.npy'), losses_unlearned)
    ckpt = {
        'model': unlearned_model.state_dict(),
        'output_ids': output_ids,
        'prompt_length': input_ids.shape[1],
    }
    torch.save(ckpt, os.path.join(output_dir, f'unlearned_models/{author}_2.pt'))
    unlearned_model.cpu()
    del unlearned_model
    torch.cuda.empty_cache()
    print('--------------------------------')