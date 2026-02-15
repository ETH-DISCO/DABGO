
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import os
import random
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default=None)
parser.add_argument("--sample_subject", action="store_true")
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
## Base Model
model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), "../out/wiki_model"))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print('model loaded')
## Training Data:
from datasets import load_from_disk
ds = load_from_disk(os.path.join(os.path.dirname(__file__), "../data/training_data/tokenized_wit_dataset"))
train_dataset = ds['train']
print('Training Data Size:', len(train_dataset))
print(train_dataset)
## Subjects used
subjects = [
    "Ancient Rome", "Impressionism", "World War II", "DNA", "Solar System", "Philosophy of Mind", "Jazz", "Thermodynamics", "Internet", "Feminism","Mount Everest","William Shakespeare",
    "Democracy","Buddhism","Genetics","The Renaissance","Ancient Egypt","Probability Theory","Ecology","Gothic Architecture","Big Bang","Art Deco","Ancient Greece"
]

def generate_output_punkt(model, tokenizer, input_text, device='cuda', max_length=64, num_beams=5, do_sample=True, temperature=1, top_k=50, top_p=0.90, repetition_penalty=1.5):
    input_text = input_text + '.'
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=num_beams, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    return output_ids, len(input_ids[0])

if args.subject is not None:
    subjects = [args.subject]

if args.sample_subject:
    page_titles = train_dataset['page_title']
    subjects = [random.choice(page_titles)]
    print("Sampled subject (page title from train data):", subjects[0])
    
for k in range(len(subjects)):
    subject = subjects[k]
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), "../out/wiki_model"))
    model.to(device)
    model.eval()
    prompt = subject
    output_ids, prompt_length = generate_output_punkt(model, tokenizer, prompt, device)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

    ckpt = {
        "sentence": output_text,
        "prompt": prompt,
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "../data/samples_wikipedia"), exist_ok=True)
    torch.save(ckpt, os.path.join(os.path.dirname(__file__), "../data/samples_wikipedia", f"{subject}.pt"))
    print(f"Sample saved to {os.path.join(os.path.dirname(__file__), "../data/samples_wikipedia", f"{subject}.pt")}")
    print("--------------------------------")