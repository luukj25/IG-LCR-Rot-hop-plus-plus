# IG-LCR-Rot-hop++: Attribution-Based Debiasing for Aspect-Based Sentiment Classification

This repository contains the implementation of **IG-LCR-Rot-hop++**, proposed in the bachelor's thesis:

> *An Attribution-Based Approach for Debiasing Aspect-Based Sentiment Classification*  
> Luuk Jille, Erasmus University Rotterdam, 2026  
> Supervised by Dr. Flavius Frăsincar

IG-LCR-Rot-hop++ extends [LCR-Rot-hop++](https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus) by using Integrated Gradients (IG) to identify and mask Low-Significance Frequent Tokens (LSFTs) — tokens that are both frequent and spuriously associated with sentiment classes — before retraining the model on the debiased data.

## Method Overview

The pipeline consists of three phases:

1. **Phase 1** — Train a standard LCR-Rot-hop++ model on the original training data
2. **Phase 2** — Compute per-sentence IG attribution scores using the Phase 1 model, identify LSFTs using joint criteria ($A_i < \tau$ and $f(w) > \omega$), and build sentence-level bias masks
3. **Phase 3** — Retrain LCR-Rot-hop++ on the masked training data; at test time, fresh IG scores are computed per sentence using the Phase 1 model to apply local masks before prediction

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list of dependencies.

## Data

This repository uses the SemEval 2015 and 2016 restaurant datasets. Place the data files in `data/raw/` following the structure of the original LCR-Rot-hop++ repository.

## Reproducing Results

### Step 1 — Preprocess embeddings

```bash
python main_preprocess.py --year 2015 --phase Train
python main_preprocess.py --year 2015 --phase Test
python main_preprocess.py --year 2016 --phase Train
python main_preprocess.py --year 2016 --phase Test

python main_preprocess_ig.py --year 2015 --phase Train
python main_preprocess_ig.py --year 2015 --phase Test
python main_preprocess_ig.py --year 2016 --phase Train
python main_preprocess_ig.py --year 2016 --phase Test
```

### Step 2 — Train baseline LCR-Rot-hop++

```bash
# SemEval 2015
python main_train.py --year 2015 --hops 2 --lr 0.09 --dropout 0.4 \
  --momentum 0.9 --weight-decay 0.001

# SemEval 2016
python main_train.py --year 2016 --hops 4 --lr 0.01 --dropout 0.4 \
  --momentum 0.99 --weight-decay 0.00001
```

### Step 3 — Compute local IG attribution scores

```bash
# SemEval 2015
python compute_ig_scores.py --year 2015 --hops 2 \
  --model data/models/2015_LCR_hops2_*.pt \
  --variant entropy --phase Train

# SemEval 2016
python compute_ig_scores.py --year 2016 --hops 4 \
  --model data/models/2016_LCR_hops4_*.pt \
  --variant entropy --phase Train
```

### Step 4 — Hyperparameter tuning (optional)

```bash
python main_hyperparam_ig_lcr.py --year 2015 --variant entropy
python main_hyperparam_ig_lcr.py --year 2016 --variant entropy
```

Or use the optimal hyperparameters reported in the thesis directly (Step 5).

### Step 5 — Train IG-LCR-Rot-hop++

```bash
# SemEval 2015
python main_train_ig_lcr.py --year 2015 --variant entropy \
  --hops 4 --lr 0.02 --dropout 0.6 --momentum 0.95 --weight-decay 0.0001 \
  --tau 0.0155 --omega 0.002

# SemEval 2016
python main_train_ig_lcr.py --year 2016 --variant entropy \
  --hops 5 --lr 0.07 --dropout 0.4 --momentum 0.95 --weight-decay 0.00001 \
  --tau 0.0041 --omega 0.005 \
  --scores data/models/2016_ig_entropy_train_scores.json
```

### Step 6 — Validate

```bash
# Baseline
python main_validate.py --year 2015 --hops 2 \
  --model data/models/2015_LCR_hops2_*.pt

python main_validate.py --year 2016 --hops 4 \
  --model data/models/2016_LCR_hops4_*.pt

# IG-LCR-Rot-hop++
python main_validate_ig_lcr.py --year 2015 --variant entropy \
  --phase1-hops 2 --phase2-hops 4 \
  --phase1-model data/models/2015_LCR_hops2_*.pt \
  --phase2-model data/models/2015_ig_entropy_phase2_hops4_*.pt \
  --tau 0.0155 --omega 0.002

python main_validate_ig_lcr.py --year 2016 --variant entropy \
  --phase1-hops 4 --phase2-hops 5 \
  --phase1-model data/models/2016_LCR_hops4_*.pt \
  --phase2-model data/models/2016_ig_entropy_phase2_hops5_*.pt \
  --tau 0.0041 --omega 0.005 \
  --scores data/models/2016_ig_entropy_train_scores.json
```

## Repository Structure

```
├── main_preprocess_ig.py       # Preprocessing with token strings
├── main_train.py               # Baseline LCR-Rot-hop++ training
├── main_train_ig_lcr.py         # IG-LCR-Rot-hop++ training (Phase 2)
├── main_hyperparam.py          # Baseline hyperparameter tuning
├── main_hyperparam_ig_lcr.py    # IG-LCR-Rot-hop++ hyperparameter tuning
├── main_validate_ig_lcr.py      # IG-LCR-Rot-hop++ validation
├── ig_attribution.py           # Integrated Gradients (entropy variant)
├── compute_ig_scores.py     # Per-sentence IG score computation
├── utils/
│   └── embeddings_dataset_ig.py  # Dataset with token strings
└── data/
    ├── raw/                    # SemEval XML files (not included)
    ├── embeddings/             # Preprocessed embeddings (generated)
    ├── embeddings_ig/          # Preprocessed embeddings with tokens (generated)
    └── models/                 # Saved models and scores (generated)
```

## Citation

If you use this code, please cite:

```
@thesis{jille2026attribution,
  author = {Jille, Luuk},
  title  = {An Attribution-Based Approach for Debiasing Aspect-Based Sentiment Classification},
  school = {Erasmus University Rotterdam},
  year   = {2026}
}
```

## Acknowledgements

This work builds on the LCR-Rot-hop++ implementation by Charlotte Visser. The SemEval datasets are used under their respective licenses.
