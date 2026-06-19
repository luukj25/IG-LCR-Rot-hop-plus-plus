"""
IG-LCR-Rot-hop++ Training Script — Phase 2

Uses pre-computed per-sentence IG attribution scores (from compute_ig_scores.py)
to mask biased tokens (LSFTs) and retrain LCR-Rot-hop++ from scratch.

Masking criterion (per sentence):
    token w is an LSFT if A_i < tau AND f(w) > omega

where A_i is the per-sentence IG attribution score,
and f(w) is the global token frequency from training data.

Usage:
    python main_train_ig_lcr.py --year 2015 \
        --scores data/models/2015_ig_entropy_train_scores.json \
        --hops 4 --lr 0.02 --dropout 0.6 --momentum 0.95 --weight-decay 0.0001 \
        --tau 0.0155 --omega 0.002

    python main_train_ig_lcr.py --year 2016 \
        --scores data/models/2016_ig_entropy_train_scores.json \
        --hops 5 --lr 0.07 --dropout 0.4 --momentum 0.95 --weight-decay 0.00001 \
        --tau 0.0041 --omega 0.005
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDatasetIG, train_validation_split_ig
from pytorchtools import EarlyStopping

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
    """Zero out embeddings of biased tokens."""
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
    """
    Build per-sentence V_b from local scores.
    Returns dict: int(idx) -> set of biased token strings for that sentence.
    """
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

    unique_biased = set(t for v in sentence_masks.values() for t in v)
    print(f"Local masks built: {total_masked} token occurrences masked, "
          f"{len(unique_biased)} unique biased token types "
          f"(tau={tau}, omega={omega})")
    return sentence_masks


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps EmbeddingsDatasetIG to return (original_idx, item)."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return original_idx, self.dataset[original_idx]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int)
    parser.add_argument("--hops", default=3, type=int)
    parser.add_argument("--tau", default=0.020, type=float)
    parser.add_argument("--omega", default=0.001, type=float)
    parser.add_argument("--lr", default=0.09, type=float)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.00001, type=float)
    parser.add_argument("--scores", type=str, default=None)
    parser.add_argument("--variant", choices=["target_class", "entropy"],
                        default="target_class")
    parser.add_argument("--run", default=1, type=int)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-computed local scores
    scores_path = args.scores or \
        f"data/models/{args.year}_ig_{args.variant}_train_scores.json"
    with open(scores_path) as f:
        scores_data = json.load(f)

    # Build per-sentence masks
    sentence_masks = build_sentence_masks(
        scores_data, tau=args.tau, omega=args.omega)

    # Load dataset
    train_dataset = EmbeddingsDatasetIG(
        year=args.year, device=device, phase="Train")
    print(f"Training dataset: {len(train_dataset)} samples")

    # Split
    train_idx, val_idx = train_validation_split_ig(train_dataset, seed=SEED)

    val_indices_path = f"data/models/{args.year}_ig_val_indices.json"
    os.makedirs("data/models", exist_ok=True)
    with open(val_indices_path, 'w') as f:
        json.dump({'train_idx': list(train_idx),
                   'validation_idx': list(val_idx)}, f)

    # Use IndexedDataset so we know original idx per sample
    indexed_train = IndexedDataset(train_dataset, train_idx)
    indexed_val = IndexedDataset(train_dataset, val_idx)

    training_loader = DataLoader(
        indexed_train, batch_size=32,
        collate_fn=lambda b: b, shuffle=True)
    validation_loader = DataLoader(
        indexed_val, collate_fn=lambda b: b)

    # Phase 2 model — fresh initialisation
    model = LCRRotHopPlusPlus(
        hops=args.hops, dropout_prob=args.dropout).to(device)

    save_path = os.path.join(
        "data", "models",
        f"{args.year}_ig_{args.variant}_phase2"
        f"_hops{args.hops}"
        f"_dropout{stringify_float(args.dropout)}"
        f"_tau{stringify_float(args.tau)}"
        f"_omega{stringify_float(args.omega)}"
        f"_run{args.run}.pt"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=30, verbose=True, path=save_path)

    best_accuracy = None
    best_state_dict = None
    train_losses = []
    valid_losses = []

    epochs_progress = tqdm(range(100), unit='epoch')

    try:
        for epoch in epochs_progress:
            model.train()
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            train_n, train_correct, train_steps, train_loss_sum = 0, 0, 0, 0.0

            for batch in epoch_progress:
                torch.set_default_device(device)

                outputs = []
                labels = []

                for original_idx, ((left, target, right), label, hops, tokens) \
                        in batch:
                    # Look up this sentence's local mask by original dataset index
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

                train_loss_sum += loss.item()
                train_steps += 1
                train_correct += (batch_outputs.argmax(1) ==
                                  batch_labels).sum().item()
                train_n += len(batch)
                train_losses.append(loss.item())

                epoch_progress.set_description(
                    f"Loss: {train_loss_sum/train_steps:.3f} "
                    f"Acc: {train_correct/train_n:.3f}")
                torch.set_default_device('cpu')

            # Validation — no masking (clean embeddings)
            model.eval()
            val_n, val_correct = 0, 0
            for data in tqdm(validation_loader, unit='obs', leave=False):
                torch.set_default_device(device)
                with torch.inference_mode():
                    original_idx, ((left, target, right), label, hops, tokens) \
                        = data[0]
                    output = model(left, target, right, hops)
                    val_correct += (output.argmax(0) == label).int().item()
                    val_n += 1
                    loss = criterion(output, label)
                    valid_losses.append(loss.item())
                torch.set_default_device('cpu')

            val_acc = val_correct / val_n
            train_loss_avg = np.average(train_losses)
            valid_loss_avg = np.average(valid_losses)
            train_losses = []
            valid_losses = []

            print(f'train_loss: {train_loss_avg:.5f}  '
                  f'valid_loss: {valid_loss_avg:.5f}')

            early_stopping(valid_loss_avg, model)

            if best_accuracy is None or val_acc > best_accuracy:
                epochs_progress.set_description(
                    f"Best Val Acc.: {val_acc:.3f}")
                best_accuracy = val_acc
                best_state_dict = model.state_dict()

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    except KeyboardInterrupt:
        print("Interrupted, saving best model...")

    if best_state_dict is not None:
        os.makedirs("data/models", exist_ok=True)
        torch.save(best_state_dict, save_path)
        print(f"Saved to {save_path}")

    print(f"\nPhase 2 complete. Best val accuracy: {best_accuracy:.4f}")
    print(f"\nTo validate on test set:")
    print(f"  python main_validate_local.py --year {args.year} "
          f"--hops {args.hops} "
          f"--phase1-model <path_to_phase1> "
          f"--phase2-model {save_path} "
          f"--tau {args.tau} --omega {args.omega}")


if __name__ == "__main__":
    main()
