### Compute the loss of each training sample given a model and a dataset
### Used for Wikipedia Model


import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
import random
from torch.cuda.amp import autocast
from datasets import load_from_disk
import gc
parser = argparse.ArgumentParser()
parser.add_argument('--finetuned_model_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  
tokenized_dataset_path = os.path.join(os.path.dirname(__file__), '../data/training_data/tokenized_wit_dataset')
tokenized_dataset = load_from_disk(tokenized_dataset_path)
train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataset = train_dataset.remove_columns(['page_title', 'section_title'])
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
)
print('dataset loaded')




## Initialize optimized models, saved as .pt files along with the generated sequence for later use
base_model_path = os.path.join(os.path.dirname(__file__), '../out/wiki_model')
model = GPT2LMHeadModel.from_pretrained(base_model_path)
model_path = args.finetuned_model_path
print('Model path: ', model_path)
print('Mode: ', args.mode)
model_dir = os.path.join(os.path.dirname(__file__), f'../out/wiki_models_{args.mode}/fisher_regularized_models')
checkpoint = torch.load(os.path.join(model_dir, f'{args.finetuned_model_path}_{args.mode}_fisher.pt'), map_location=device)
print(os.path.join(model_dir, f'{args.finetuned_model_path}_{args.mode}_fisher.pt'))
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

torch.cuda.empty_cache()




def collect_per_sample_losses(model, dataset, device, batch_size=4, data_collator=None, vocab_size=50257, cutoff=1000):
    model.eval()
    dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False,
    collate_fn=data_collator, pin_memory=True
)
    num_samples = len(dataset)
    print(f'Number of samples: {num_samples}')

    ## Make a set of all unique tokens which have been generated and compute the average token loss for each token in training samples
    ## To get vocabulary level losses
    token_loss_dict = {i: [0., 0.] for i in range(vocab_size)}  # [sum(loss), count] ## Comment out if not needed for faster computation



    all_sample_losses = []
    processed_samples = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader), desc="Calculating Losses"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
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
            
            ## To get vocabulary level losses
            ## Comment out if not needed for faster computation
            token_ids = shift_labels 
            token_losses = masked_loss 

            for b in range(token_ids.size(0)):
                for t in range(token_ids.size(1)):
                    token_id = token_ids[b, t].item()
                    if shift_attention[b, t]:  # only count if attended (i.e., not padding)
                        token_loss_dict[token_id][0] += token_losses[b, t].item()
                        token_loss_dict[token_id][1] += 1

            
            del input_ids, attention_mask, outputs, logits, per_token_loss, sample_losses, log_probs
            torch.cuda.empty_cache()
            if (batch_idx + 1) % 10000 == 0:
                gc.collect()
                print(f"Processed {processed_samples} samples")                
                os.makedirs(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized'), exist_ok=True)
                np.save(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized/{args.save_path}.npy'), all_sample_losses)
                os.makedirs(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized/token_losses'), exist_ok=True)
                with open(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized/token_losses/{args.save_path}_{args.mode}_token.pkl'), 'wb') as f:
                    pickle.dump(token_loss_dict, f)
            if processed_samples > cutoff:
                break
                
    return np.array(all_sample_losses, dtype=np.float32), token_loss_dict


losses_af, token_loss_dict = collect_per_sample_losses(
    model=model,
    dataset=train_dataset,
    device=device,
    batch_size=50,
    data_collator=data_collator,
    cutoff=len(train_dataset)
)
print("Mean loss over dataset:", losses_af.mean().item())
print('Collected post-finetuning training losses')


os.makedirs(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized'), exist_ok=True)
np.save(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized/{args.save_path}.npy'), losses_af)
os.makedirs(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized/token_losses'), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), f'../data/losses/wiki/{args.mode}/fisher_regularized/token_losses/{args.save_path}_{args.mode}_token.pkl'), 'wb') as f:
    pickle.dump(token_loss_dict, f)