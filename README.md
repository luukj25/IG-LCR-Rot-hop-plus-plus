# IG-LCR-Rot-hop++: Attribution-Based Debiasing for Aspect-Based Sentiment Classification

This repository contains the implementation of **IG-LCR-Rot-hop++**, proposed in the bachelor's thesis:

> *An Attribution-Based Approach for Debiasing Aspect-Based Sentiment Classification*  
> Luuk Jille, Erasmus University Rotterdam, 2026  
> Supervised by Dr. Flavius Frăsincar
>
Code from https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus has been used.
## Data

This repository uses the SemEval 2015 and 2016 restaurant datasets. Place the data files in `data/raw/` following the structure of the original LCR-Rot-hop++ repository.

## Reproducing Results

### Step 1 — Preprocess embeddings

Run:
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
Run:
# SemEval 2015
python main_train.py --year 2015 --hops 2 --lr 0.09 --dropout 0.4 \
  --momentum 0.9 --weight-decay 0.001

# SemEval 2016
python main_train.py --year 2016 --hops 4 --lr 0.01 --dropout 0.4 \
  --momentum 0.99 --weight-decay 0.00001

### Step 3 — Compute IG attribution scores
Run:
# SemEval 2015
python compute_ig_scores.py --year 2015 --hops 2 \
  --model data/models/2015_LCR_hops2_*.pt \
  --variant entropy --phase Train

# SemEval 2016
python compute_ig_scores.py --year 2016 --hops 4 \
  --model data/models/2016_LCR_hops4_*.pt \
  --variant entropy --phase Train

### Step 4 — Hyperparameter tuning (optional)
Run:
python main_hyperparam_ig_lcr.py --year 2015 --variant entropy
python main_hyperparam_ig_lcr.py --year 2016 --variant entropy
```

Or use the optimal hyperparameters reported in the thesis directly (Step 5).

### Step 5 — Train IG-LCR-Rot-hop++
Run:
# SemEval 2015
python main_train_ig_lcr.py --year 2015 --variant entropy \
  --hops 4 --lr 0.02 --dropout 0.6 --momentum 0.95 --weight-decay 0.0001 \
  --tau 0.0155 --omega 0.002

# SemEval 2016
python main_train_ig_lcr.py --year 2016 --variant entropy \
  --hops 5 --lr 0.07 --dropout 0.4 --momentum 0.95 --weight-decay 0.00001 \
  --tau 0.0041 --omega 0.005 \
  --scores data/models/2016_ig_entropy_train_scores.json
### Step 6 — Validate
Run:
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
  --scores data/models/2016_ig_entropy_train_scores.js
