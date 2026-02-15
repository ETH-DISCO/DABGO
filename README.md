<div align="center">

# **DABGO: Data Attribution via Bidirectional Gradient Optimization**  

**Frédéric Berdoz · Luca A. Lanzendörfer · Kaan Bayraktar · Roger Wattenhofer**

<!--[![arXiv](https://img.shields.io/badge/arXiv-2511.09844-b31b1b.svg)](https://arxiv.org/abs/2511.09844)-->
[Paper](https://tik-db.ee.ethz.ch/file/3811d93c1807bd4f700bbd00697a8cd8/)

Accepted at AIGOV @ AAAI 2026

</div>



## Overview

This repository contains experiments for studying attribution in language models via bidirectional gradient optimization on two main datasets: Wikipedia facts and Project Gutenberg books. It is organized into two main experiment folders:
- **`wikipedia/`**: Experiments on Wikipedia factual knowledge
- **`gutenberg/`**: Experiments on literary text from Project Gutenberg

## Updates:

* Cleaned up the repo. All different methods are in their own folder.
* Fixed Natural Gradient Descent and Gecko implementations in gradient_steps.py and gecko folder in both gutenberg and wikipedia folders.

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
python wikipedia/samples_wikipedia.py --subject "Ancient Rome"
python wikipedia/gradient_steps.py --sample_name "Ancient Rome"
python wikipedia/loss_computation.py --mode "descent" --both --batch_size=30 --model_name "Ancient Rome" #Adjust Batch size when running into memory issues or speedup
```

#### Evaluation

```bash
python wikipedia/tailpatch.py --method "dabgo"
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
python gutenberg/samples_gutenberg.py
python gutenberg/gradient_steps.py --compute_fisher
python gutenberg/loss_computation.py --model_name <author_to_be_attributed> --mode "ascent"
```
Evaluation:
```bash
python gutenberg/tailpatch.py --method "dabgo"
```

To evaluate via retraining run one of the following: `gutenberg/retraining_gutenberg_bm25/dabgo/gecko/trackstar.py`. Each one of these handles how we stored attributed samples according to their method. Example for ours:
```bash
cd gutenberg
python retraining_gutenberg_dabgo.py --authors "William Shakespeare" --num_samples <k>
```
$k$ in this case corresponds to the leave-k-out method. Which we refer to as ground-truth in the context of data attribution. We will be removing the top-$k$ influential samples identified by the corresponding method (in our case **DABGO**) for a generated samples $x$ and retraining without those training-data-samples. At end, one can evaluate influence by measuring the loss on $x$ in the newly trained model compared to the base model (This analysis is in `tailpatch.ipynb`). In our analysis we used $k=20,50,100$
  
  
## Comparison Baselines

To compute the comparison baselines, refer to their corresponding papers, **BM25** (Robertson & Walker, 1994), **GECKO embeddings** (Lee et al., 2024), and **Scalable Influence and Fact Tracing for Large Language Model Pretraining** (Chang et al., 2024), as well as their implementations in the `wikipedia` and `gutenberg` folders, which are adapted to our experiments.
