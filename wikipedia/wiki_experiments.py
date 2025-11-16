
## Sample Prompt: FC Barcelona. 
## Output: 
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
## Base Model
model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print('model loaded')
## Training Data:
from datasets import load_from_disk
ds = load_from_disk('data/training_data/tokenized_wit_dataset')
train_dataset = ds['train']
print('Training Data Size:', len(train_dataset))

## Get subjects
subjects = [
    "Ancient Rome", "Impressionism", "World War II", "DNA", "Solar System", "Philosophy of Mind", "Jazz", "Thermodynamics", "Internet", "Feminism","Mount Everest","William Shakespeare",
    "Democracy","Buddhism","Genetics","The Renaissance","Ancient Egypt","Probability Theory","Ecology","Gothic Architecture","Big Bang","Art Deco","Ancient Greece"
]
names = [
    'ancient_rome','impressionism','ww2','dna','solar_system','philosophy_mind','jazz','thermodynamics','internet','feminism','everest','shakespeare',
    'democracy','buddhism','genetics','renaissance','ancient_egypt','probability','ecology','gothic_architecture','big_bang','art_deco','ancient_greece'
]
def generate_output_punkt(model, tokenizer, input_text, device='cuda', max_length=64, num_beams=5, do_sample=True, temperature=1, top_k=50, top_p=0.90, repetition_penalty=1.5):
    input_text = input_text + '.'
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=num_beams, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    return output_ids, len(input_ids[0])
## Load fisher information 

import torch
fisher_info = torch.load('wikipedia/fisher_diag_wiki.pt')
print(fisher_info.keys())
def finetuning_fisher(model, output_ids,fisher_info, prompt_length=1, finetuning_steps=10, unlearning_parameter=1, ewc_lambda=1000, learning_rate=1e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = True
        
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    real_ids = output_ids.clone()
    
    B, L = real_ids.shape
    labels = real_ids.clone()
    labels[:, :prompt_length] = -100 ## Mask out prompt tokens for loss
    labels = labels.to(device)
    real_ids = real_ids.to(device)
    attention_mask = torch.ones(B, L, dtype=torch.long).to(device)
    attention_mask = attention_mask.to(device)
    original_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(finetuning_steps):
        optimizer.zero_grad()

        outputs = model(input_ids=real_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in fisher_info:
                fisher = fisher_info[name]
                
                # Ensure fisher info is on correct device
                if fisher.device != param.device:
                    fisher = fisher.to(param.device)
                param_diff = param - original_params[name]
                penalty = fisher * param_diff.pow(2)
                
                ewc_loss += penalty.sum()
        
        ewc_loss = ewc_lambda * ewc_loss
        total_loss = unlearning_parameter * (loss + ewc_loss)
        print(f"Step {i}: Loss: {total_loss}")
        print('--------------------------------')
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    return model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer.pad_token = tokenizer.eos_token

print('Starting Finetuning')
for k in range(len(subjects)):
    subject = subjects[k]
    name = names[k]
    model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
    model.to(device)
    model.eval()
    prompt = subject 
    output_ids, prompt_length = generate_output_punkt(model, tokenizer, prompt, device)
    print(output_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    final_model = finetuning_fisher(model, output_ids=output_ids, fisher_info=fisher_info, prompt_length=prompt_length,finetuning_steps=10, unlearning_parameter=1, learning_rate=1e-4)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print(prompt_length)

    ckpt = {
        'output_ids': output_ids,
        'prompt_length': prompt_length,
        'model': final_model.state_dict()
    }
    os.makedirs(f'out/wiki_models_finetuned/fisher_regularized_models', exist_ok=True)
    torch.save(ckpt, f'out/wiki_models_finetuned/fisher_regularized_models/{name}_finetuned_fisher.pt')
    final_model.cpu()
    model.cpu()
    del final_model
    del model
    torch.cuda.empty_cache()
    model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
    model.to(device)
    model.eval()
    final_model = finetuning_fisher(model, output_ids=output_ids, fisher_info=fisher_info, prompt_length=prompt_length,finetuning_steps=10, unlearning_parameter=-1, learning_rate=1e-4)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print(prompt_length)
    ckpt = {
        'output_ids': output_ids,
        'prompt_length': prompt_length,
        'model': final_model.state_dict()
    }
    torch.save(ckpt, f'out/wiki_models_unlearned/fisher_regularized_models/{name}_unlearned_fisher.pt')
    final_model.cpu()
    model.cpu()
    del final_model
    del model
    torch.cuda.empty_cache()    
    
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
model.to(device)
model.eval()
print('loaded model')
subject = 'World War I'  
print('Starting World War I Example, Descent')
input_text = subject + '.'  
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
output_ids = model.generate(input_ids, max_length=100, num_beams=10, early_stopping=True, do_sample=True, 
                            temperature=1, top_k=50, top_p=0.9, num_return_sequences=1, repetition_penalty=1.5)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

prompt = output_text.split('.')[0]
output_text = output_text.split('.')[:-1]
output_text = '.'.join(output_text)
sample_ids = tokenizer.encode(output_text, return_tensors='pt').to(device)
prompt_length = len(tokenizer.encode('World War I.'))
print('Prompt length: ', prompt_length)
print('Output text: ', tokenizer.decode(sample_ids[0], skip_special_tokens=True))
final_model = finetuning_fisher(model, output_ids=sample_ids, fisher_info=fisher_info, prompt_length=prompt_length,finetuning_steps=10, unlearning_parameter=1, learning_rate=1e-4)
ckpt = {
    'output_ids': sample_ids,
    'prompt_length': prompt_length,
    'model': final_model.state_dict()
}
torch.save(ckpt, f'out/wiki_models_finetuned/fisher_regularized_models/ww1_finetuned_fisher.pt')
print('Running Ascent')
model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
model.to(device)
model.eval()

final_model = finetuning_fisher(model, output_ids=sample_ids, fisher_info=fisher_info, prompt_length=prompt_length,finetuning_steps=10, unlearning_parameter=-1, learning_rate=1e-4)
ckpt = {
    'output_ids': sample_ids,
    'prompt_length': prompt_length,
    'model': final_model.state_dict()
}
torch.save(ckpt, f'out/wiki_models_unlearned/fisher_regularized_models/ww1_unlearned_fisher.pt')


print('World War I Example done')
print('Starting International Space Station Example, Descent')
model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
model.to(device)
model.eval()
subject = 'International Space Station'   
input_text = subject + '.'  
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
prompt_length = len(input_ids[0])
output_ids = model.generate(input_ids, max_length=100, num_beams=10, early_stopping=True, do_sample=True, 
                            temperature=1, top_k=50, top_p=0.9, num_return_sequences=1, repetition_penalty=1.5)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
output_text = output_text.split('.')[:-1]
output_text = '.'.join(output_text)
sample_ids = tokenizer.encode(output_text, return_tensors='pt').to(device)
print('Output text: ', tokenizer.decode(sample_ids[0], skip_special_tokens=True))
print('Prompt length: ', prompt_length)
final_model = finetuning_fisher(model, output_ids=sample_ids, fisher_info=fisher_info, prompt_length=prompt_length,finetuning_steps=10, unlearning_parameter=1, learning_rate=1e-4)
ckpt = {
    'output_ids': sample_ids,
    'prompt_length': prompt_length,
    'model': final_model.state_dict()
}
torch.save(ckpt, f'out/wiki_models_finetuned/fisher_regularized_models/iss_finetuned_fisher.pt')
print('Running Ascent')
model = GPT2LMHeadModel.from_pretrained('out/wiki_model')
model.to(device)
model.eval()
final_model = finetuning_fisher(model, output_ids=sample_ids, fisher_info=fisher_info, prompt_length=prompt_length,finetuning_steps=10, unlearning_parameter=-1, learning_rate=1e-4)


ckpt = {
    'output_ids': sample_ids,
    'prompt_length': prompt_length,
    'model': final_model.state_dict()
}
torch.save(ckpt, f'out/wiki_models_unlearned/fisher_regularized_models/iss_unlearned_fisher.pt')
