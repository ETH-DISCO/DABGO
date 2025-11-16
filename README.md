# Data Attribution via Bidirectional Gradient Optimization

This repository contains experiments for studying attribution in language models via bidirectional gradient optimization on two main datasets: Wikipedia facts and Project Gutenberg books. It is the code for the Data Attribution in Large Language Models via Bidirectional Gradient Optimization work ([Paper](https://tik-db.ee.ethz.ch/file/3811d93c1807bd4f700bbd00697a8cd8/))

## Overview

The repository is organized into two main experiment folders:
- **`wikipedia/`**: Experiments on Wikipedia factual knowledge
- **`gutenberg/`**: Experiments on literary text from Project Gutenberg


## Usage 

All required dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```


## Wikipedia Experiments (`wikipedia/`)

Study attribution in the context of factual statements.

To first get the data, model and Fisher Information run the following: 

- **`wiki_tokenization.py`**: Get and preprocess the data of wikipedia abstracts
- **`wiki_model.py`**: Main training script for Wikipedia language model
  - Trains GPT-2 architecture from scratch on Wikipedia data

```bash
cd wikipedia
python wiki_tokenization.py
python wiki_model.py 
python fisher_diag.py
```
Afterwards generate samples and record loss by performing both gradient ascent and descent.

```bash
python wikipedia/wiki_experiments.py
```

```bash
## The model name are specified in wiki_experiments during the saving process (These are the optimized models by either gradient ascent or descent corresponding to a certain generated text sample). 
## In wiki_experiments.py for each generated sample we save an gradient ascent and descent optimized model. (Command template for loss computation):
python wikipedia/loss_computation.py --finetuned_model_path <model_name> --save_path <results> --mode <finetuned/unlearned>

## Need to run once with --mode unlearned and finetuned for each generated sample: 
## Example to get influential training samples for the Ancient Rome case:
python wikipedia/loss_computation.py --finetuned_model_path "ancient_rome" --save_path "ancient_rome" --mode "finetuned"
python wikipedia/loss_computation.py --finetuned_model_path "ancient_rome" --save_path "ancient_rome" --mode "unlearned"

## To get the losses of the base model run with following arguments:
python wikipedia/loss_computation.py --finetuned_model_path "wiki_model" --save_path "losses_bf" --mode "base"
```
#### Evaluation

```bash
python wikipedia/tailpatch.py --save_path=results_aggregated
```


## Gutenberg Experiments (`gutenberg/`)

Run these scripts to extract and preprocess for our Gutenberg Dataset:
- **`concurrent_scraping.py`**: Parallel data collection from Project Gutenberg. Use following link to get access to the books: https://www.gutenberg.org/cache/epub/feeds/ the pgmarc.xml file (if you want the current gutenberg book collection)
- **`gutenberg_book_preprocessing.ipynb`**: Preprocess data for Gutenberg. Preprocessed booktexts along with metadata used are already provided in `selected_dataset_mixed.csv`. Run the last cell of `gutenberg_book_preprocessing.ipynb`to obtain tokenized dataset for training. 
- **`untokenize_gutenberg_small.py`**: Text reconstruction for BM25 retrieval

Training can be done with following script:
```bash
# Gutenberg model
cd gutenberg
python gutenberg_training.py --model_path gpt2-scratch-mixed --data_path selected_dataset_mixed.json
```
Sample Generation and loss computation: 
```bash
# Generate samples and compute losses for author-specific attribution
cd gutenberg
python samples_gutenberg.py
```


To evaluate with the tailpatch score on our samples use the `gutenberg/tailpatch.ipynb` notebook. If you want to test on your own samples take the tailpatch function given in the first cell. 

To evaluate via retraining run one of the following: `gutenberg/retraining_gutenberg_bm25/ftun/gecko/trackstar.py`. Each one of these handles how we stored attributed samples according to their method. Example for ours:
```bash
cd gutenberg
python retraining_gutenberg_ftun --authors "William Shakespeare" --num_samples <k>
```
$k$ in this case corresponds to the leave-k-out method. Which we refer to as ground-truth in the context of data attribution. We will be removing the top-$k$ influential samples identified by the corresponding method (in our case **DABGO**) for a generated samples $x$ and retraining without those training-data-samples. At end, one can evaluate influence by measuring the loss on $x$ in the newly trained model compared to the base model (This analysis is in `tailpatch.ipynb`). In our analysis we used $k=20,50,100$
  
  
## Comparison Baselines

To compute the comparison baselines, refer to their corresponding papers, **BM25** (Robertson & Walker, 1994), **GECKO embeddings** (Lee et al., 2024), and **Scalable Influence and Fact Tracing for Large Language Model Pretraining** (Chang et al., 2024), as well as their implementations in the `wikipedia` and `gutenberg` folders, which are adapted to our experiments.
