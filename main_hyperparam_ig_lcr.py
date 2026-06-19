"""
TPE Hyperparameter Optimisation for IG-LCR-Rot-hop++

Tunes: lr, dropout, momentum, weight_decay, hops, tau, omega
Uses pre-computed IG attribution scores from compute_ig_scores.py.

Usage:
    python main_hyperparam_ig_lcr.py --year 2015 --variant entropy
    python main_hyperparam_ig_lcr.py --year 2016 --variant entropy
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDatasetIG, train_validation_split_ig

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stringify_float(value: float):
    return str(value).replace('.', '-')


def mask_tokens(left, target, right, tokens, V_b):
    n_left = left.shape[0]
    n_target = target.shape[0]
    left_m = left.clone()
    target_m = target.clone()
    right_m = right.clone()
    for i, token in enumerate(tokens):
        if token in V_b:
            if i < n_left:
                left_m[i] = 0.0
            elif i < n_left + n_target:
                target_m[i - n_left] = 0.0
            else:
                right_m[i - n_left - n_target] = 0.0
    return left_m, target_m, right_m


def build_sentence_masks(scores_data: dict, tau: float, omega: float) -> dict:
    frequencies = scores_data['frequencies']
    per_sentence = scores_data['per_sentence_scores']
    sentence_masks = {}
    total_masked = 0
    for idx_str, sent in per_sentence.items():
        V_b = set()
        for token, score in zip(sent['tokens'], sent['scores']):
            if score < tau and frequencies.get(token, 0) > omega:
                V_b.add(token)
        sentence_masks[int(idx_str)] = V_b
        total_masked += sum(1 for t in sent['tokens'] if t in V_b)
    print(f"  Masks built: {total_masked} token occurrences masked "
          f"(tau={tau:.4f}, omega={omega})")
    return sentence_masks


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return original_idx, self.dataset[original_idx]


def train_phase2(
    train_dataset,
    sentence_masks,
    device,
    hyperparams,
    n_epochs=20,
    batch_size=32,
):
    set_seed(SEED)

    train_idx, val_idx = train_validation_split_ig(train_dataset, seed=SEED)

    indexed_train = IndexedDataset(train_dataset, train_idx)
    indexed_val = IndexedDataset(train_dataset, val_idx)

    training_loader = DataLoader(
        indexed_train, batch_size=batch_size,
        collate_fn=lambda b: b, shuffle=True)
    validation_loader = DataLoader(
        indexed_val, collate_fn=lambda b: b)

    model = LCRRotHopPlusPlus(
        hops=hyperparams['lcr_hops'],
        dropout_prob=hyperparams['dropout_rate'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        momentum=hyperparams['momentum'],
        weight_decay=hyperparams['weight_decay'],
    )

    best_accuracy = None

    for epoch in range(n_epochs):
        model.train()
        for batch in training_loader:
            torch.set_default_device(device)
            outputs = []
            labels = []
            for original_idx, ((left, target, right), label, hops, tokens) \
                    in batch:
                V_b = sentence_masks.get(original_idx, set())
                left_m, target_m, right_m = mask_tokens(
                    left, target, right, tokens, V_b)
                outputs.append(model(left_m, target_m, right_m, hops))
                labels.append(label.item())

            batch_outputs = torch.stack(outputs, dim=0)
            batch_labels = torch.tensor(labels)
            loss = criterion(batch_outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.set_default_device('cpu')

        # Validation
        model.eval()
        val_n, val_correct = 0, 0
        for data in validation_loader:
            torch.set_default_device(device)
            with torch.inference_mode():
                original_idx, ((left, target, right), label, hops, tokens) \
                    = data[0]
                output = model(left, target, right, hops)
                val_correct += (output.argmax(0) == label).int().item()
                val_n += 1
            torch.set_default_device('cpu')

        val_acc = val_correct / val_n
        if best_accuracy is None or val_acc > best_accuracy:
            best_accuracy = val_acc

    return best_accuracy


class HyperOptManagerLocal:
    def __init__(self, year: int, scores_path: str, variant: str = "target_class"):
        self.year = year
        self.variant = variant
        self.n_epochs = 20
        self.eval_num = 0
        self.best_loss = None
        self.best_hyperparams = None
        self.trials = Trials()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        with open(scores_path) as f:
            self.scores_data = json.load(f)

        # Pre-compute all scores as sorted list for percentile-based tau
        all_scores = sorted([
            s for sent in self.scores_data['per_sentence_scores'].values()
            for s in sent['scores']
        ])
        n = len(all_scores)
        self.p10 = all_scores[int(0.10 * n)]
        self.p25 = all_scores[int(0.25 * n)]
        self.p50 = all_scores[int(0.50 * n)]

        print(f"Score percentiles: p10={self.p10:.4f}, "
              f"p25={self.p25:.4f}, p50={self.p50:.4f}")

        self.train_dataset = EmbeddingsDatasetIG(
            year=year, device=self.device, phase="Train")

        self.__checkpoint_dir = \
            f"data/checkpoints/{year}_ig_{variant}_epochs{self.n_epochs}_seed{SEED}"

        if os.path.isdir(self.__checkpoint_dir):
            try:
                with open(f"{self.__checkpoint_dir}/hyperparams.json") as f:
                    self.best_hyperparams = json.load(f)
                with open(f"{self.__checkpoint_dir}/trials.pkl", "rb") as f:
                    self.trials = pickle.load(f)
                    self.eval_num = len(self.trials)
                with open(f"{self.__checkpoint_dir}/loss.txt") as f:
                    self.best_loss = float(f.read())
                print(f"Resuming from checkpoint ({self.eval_num} evals done)")
            except IOError:
                raise ValueError(
                    f"Checkpoint incomplete, remove {self.__checkpoint_dir}")
        else:
            print("Starting from scratch")

    def run(self):
        space = [
            hp.choice('learning_rate',
                      [0.005, 0.001, 0.01, 0.02, 0.05,
                       0.06, 0.07, 0.08, 0.09, 0.1]),
            hp.quniform('dropout_rate', 0.25, 0.75, 0.1),
            hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
            hp.choice('weight_decay',
                      [0.00001, 0.0001, 0.001, 0.01, 0.1]),
            hp.choice('lcr_hops', [2, 3, 4, 5]),
            hp.choice('tau', [self.p25, self.p50]),
            hp.choice('omega', [0.0005, 0.001, 0.002, 0.005]),
        ]
        fmin(self.objective, space=space, algo=tpe.suggest,
             trials=self.trials, max_evals=50,
             show_progressbar=False,
             rstate=np.random.default_rng(SEED))

    def objective(self, hyperparams):
        set_seed(SEED)
        self.eval_num += 1
        lr, dropout, momentum, wd, hops, tau, omega = hyperparams

        print(f"\nEval {self.eval_num}: lr={lr}, dropout={dropout:.1f}, "
              f"momentum={momentum}, wd={wd}, hops={hops}, "
              f"tau={tau:.4f}, omega={omega}")

        sentence_masks = build_sentence_masks(
            self.scores_data, tau=tau, omega=omega)

        hp_dict = {
            'learning_rate': lr,
            'dropout_rate': dropout,
            'momentum': momentum,
            'weight_decay': wd,
            'lcr_hops': hops,
        }

        best_accuracy = train_phase2(
            train_dataset=self.train_dataset,
            sentence_masks=sentence_masks,
            device=self.device,
            hyperparams=hp_dict,
            n_epochs=self.n_epochs,
        )

        objective_loss = -best_accuracy
        self.check_best(objective_loss, hyperparams)

        return {'loss': objective_loss,
                'status': STATUS_OK,
                'space': hyperparams}

    def check_best(self, loss, hyperparams):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = hyperparams
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
            with open(f"{self.__checkpoint_dir}/hyperparams.json", "w") as f:
                json.dump(hyperparams, f)
            with open(f"{self.__checkpoint_dir}/loss.txt", "w") as f:
                f.write(str(self.best_loss))
            print(f"  New best: acc={-loss:.4f}")
        with open(f"{self.__checkpoint_dir}/trials.pkl", "wb") as f:
            pickle.dump(self.trials, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int)
    parser.add_argument("--scores", type=str, default=None)
    parser.add_argument("--variant", choices=["target_class", "entropy"],
                        default="target_class")
    args = parser.parse_args()

    set_seed(SEED)

    scores_path = args.scores or \
        f"data/models/{args.year}_ig_{args.variant}_train_scores.json"

    opt = HyperOptManagerLocal(year=args.year, scores_path=scores_path,
                               variant=args.variant)
    opt.run()

    print(f"\nBest hyperparameters:")
    print(json.dumps(opt.best_hyperparams, indent=2))
    print(f"Best val accuracy: {-opt.best_loss:.4f}")


if __name__ == "__main__":
    main()
