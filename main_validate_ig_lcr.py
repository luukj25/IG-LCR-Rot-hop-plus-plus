"""
IG-LCR-Rot-hop++ Validation Script

At test time, for each sentence:
1. Compute per-sentence IG attribution scores using Phase 1 model
2. Build bias mask: token w is an LSFT if A_i < tau AND f(w) > omega
3. Mask biased tokens (zero embedding)
4. Forward pass through Phase 2 model -> prediction

f(w) is always computed from the training set (loaded from scores JSON).

Usage:
    # Test set (2015):
    python main_validate_ig_lcr.py --year 2015 --variant entropy \
        --phase1-hops 2 --phase2-hops 4 \
        --phase1-model data/models/2015_LCR_hops2_lr0-09_dropout0-4_acc0-921875.pt \
        --phase2-model data/models/2015_ig_entropy_phase2_hops4_....pt \
        --tau 0.0155 --omega 0.002

    # Test set (2016):
    python main_validate_ig_lcr.py --year 2016 --variant entropy \
        --phase1-hops 4 --phase2-hops 5 \
        --phase1-model data/models/2016_LCR_hops4_lr0-01_dropout0-4_acc0-8962765957446809.pt \
        --phase2-model data/models/2016_ig_entropy_phase2_hops5_....pt \
        --tau 0.0041 --omega 0.005 \
        --scores data/models/2016_ig_entropy_train_scores.json
"""

import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDatasetIG

DEFAULT_T = 50
CLASS_NAMES = ['negative', 'neutral', 'positive']


def compute_ig_true_class(model, left, target, right, hops, true_label, device, T=DEFAULT_T):
    """Compute IG w.r.t. true class, return per-token normalized scores."""
    model.train()

    n_left = left.shape[0]
    n_target = target.shape[0]

    E = torch.cat([left, target, right], dim=0)
    N, d = E.shape

    ig_sum = torch.zeros(N, d, device=device)

    for t in range(1, T + 1):
        alpha = t / T
        E_interp = (alpha * E).detach().requires_grad_(True)

        left_interp = E_interp[:n_left]
        target_interp = E_interp[n_left:n_left + n_target]
        right_interp = E_interp[n_left + n_target:]

        torch.set_default_device(device)
        output = model(left_interp, target_interp, right_interp, hops)
        torch.set_default_device('cpu')

        if E_interp.grad is not None:
            E_interp.grad.zero_()
        output[true_label].backward()
        ig_sum += E_interp.grad.detach()

    ig = (E.detach() / T) * ig_sum  # [N, d]
    model.eval()

    # Normalize: alpha_i = |IG_i|_1 / sum_k |IG_k|_1
    G = ig.abs().sum(dim=1)  # [N]
    G_sum = G.sum()
    if G_sum == 0:
        return torch.zeros(N, device=device)
    return G / G_sum


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


def validate(phase1_model, phase2_model, dataset, frequencies,
             tau, omega, device, T=DEFAULT_T, val_indices=None):
    if val_indices is not None:
        eval_dataset = Subset(dataset, val_indices)
        print(f"Evaluating on validation set ({len(eval_dataset)} obs)")
    else:
        eval_dataset = dataset
        print(f"Evaluating on test set ({len(eval_dataset)} obs)")

    loader = DataLoader(eval_dataset, collate_fn=lambda b: b)

    n_classes = 3
    n_correct = [0] * n_classes
    n_label = [0] * n_classes
    n_predicted = [0] * n_classes
    brier_score = 0.0
    masked_count = 0
    total_tokens = 0

    for i, data in enumerate(tqdm(loader, unit='obs')):
        torch.set_default_device(device)

        (left, target, right), label, hops, tokens = data[0]
        true_label = label.item()

        # Step 1: compute local IG using Phase 1 model
        alpha = compute_ig_true_class(
            phase1_model, left, target, right, hops, true_label, device, T)

        # Step 2: build local mask
        V_b = set()
        for token, score in zip(tokens, alpha.tolist()):
            if score < tau and frequencies.get(token, 0) > omega:
                V_b.add(token)

        masked_count += sum(1 for t in tokens if t in V_b)
        total_tokens += len(tokens)

        # Step 3: mask and run Phase 2
        left_m, target_m, right_m = mask_tokens(
            left, target, right, tokens, V_b)

        with torch.inference_mode():
            torch.set_default_device(device)
            output = phase2_model(left_m, target_m, right_m, hops)
            torch.set_default_device('cpu')
            pred = output.argmax(0).item()

        n_label[true_label] += 1
        n_predicted[pred] += 1
        if pred == true_label:
            n_correct[true_label] += 1

        for j in range(n_classes):
            brier_check = 1 if j == true_label else 0
            brier_score += (output[j].item() - brier_check) ** 2

        torch.set_default_device('cpu')

    total = sum(n_label)
    precision = sum(
        n_correct[i] / n_predicted[i] if n_predicted[i] > 0 else 0
        for i in range(n_classes)) / n_classes
    recall = sum(n_correct[i] / n_label[i] for i in range(n_classes)) / n_classes
    f1 = (2 * precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0.0

    print(f"\nResults:")
    print(f"  Accuracy:   {sum(n_correct)/total*100:.2f}%")
    print(f"  Precision:  {precision*100:.2f}%")
    print(f"  Recall:     {recall*100:.2f}%")
    print(f"  F1:         {f1*100:.2f}%")
    print(f"  Brier:      {brier_score/total:.4f}")
    print(f"  Avg tokens masked per sentence: "
          f"{masked_count/total:.2f} / {total_tokens/total:.2f}")

    return {
        'accuracy': sum(n_correct) / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier_score / total,
        'avg_masked': masked_count / total,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int)
    parser.add_argument("--phase1-hops", default=2, type=int,
                        help="Hops for Phase 1 model (used for IG computation)")
    parser.add_argument("--phase2-hops", default=3, type=int,
                        help="Hops for Phase 2 model (used for final prediction)")
    parser.add_argument("--phase1-model", type=str, required=True)
    parser.add_argument("--phase2-model", type=str, required=True)
    parser.add_argument("--scores", type=str, default=None,
                        help="Path to local scores JSON (for frequencies). ")
    parser.add_argument("--variant", choices=["target_class", "entropy"],
                        default="target_class")
    parser.add_argument("--tau", default=0.020, type=float)
    parser.add_argument("--omega", default=0.001, type=float)
    parser.add_argument("--T", default=DEFAULT_T, type=int)
    parser.add_argument("--val", action="store_true",
                        help="Evaluate on validation set instead of test set")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load frequencies from training scores
    scores_path = args.scores or \
        f"data/models/{args.year}_ig_{args.variant}_train_scores.json"
    with open(scores_path) as f:
        scores_data = json.load(f)
    frequencies = scores_data['frequencies']

    # Load Phase 1 model
    phase1_model = LCRRotHopPlusPlus(
        hops=args.phase1_hops, dropout_prob=0.0).to(device)
    phase1_model.load_state_dict(
        torch.load(args.phase1_model, map_location=device))
    phase1_model.eval()
    print(f"Loaded Phase 1 model: {args.phase1_model} (hops={args.phase1_hops})")

    # Load Phase 2 model
    phase2_model = LCRRotHopPlusPlus(
        hops=args.phase2_hops, dropout_prob=0.0).to(device)
    phase2_model.load_state_dict(
        torch.load(args.phase2_model, map_location=device))
    phase2_model.eval()
    print(f"Loaded Phase 2 model: {args.phase2_model} (hops={args.phase2_hops})")

    if args.val:
        dataset = EmbeddingsDatasetIG(
            year=args.year, device=device, phase="Train")
        val_indices_path = f"data/models/{args.year}_ig_val_indices.json"
        with open(val_indices_path) as f:
            indices = json.load(f)
        val_indices = indices['validation_idx']
    else:
        dataset = EmbeddingsDatasetIG(
            year=args.year, device=device, phase="Test")
        val_indices = None

    validate(phase1_model, phase2_model, dataset, frequencies,
             tau=args.tau, omega=args.omega,
             device=device, T=args.T,
             val_indices=val_indices)


if __name__ == "__main__":
    main()
