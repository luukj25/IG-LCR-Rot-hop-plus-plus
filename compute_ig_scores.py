"""
Compute per-sentence IG attribution scores using Phase 1 model (LCR-Rot-hop++).

Implements the entropy-based attribution variant:
    A_i = log2(3) - H_i
where H_i is the entropy of normalized |IG| attributions across sentiment classes.

Tokens with low attribution scores (A_i < tau) and high corpus frequency (f(w) > omega)
are identified as Low-Significance Frequent Tokens (LSFTs) during Phase 2 training.

Saves per-sentence scores to JSON keyed by sentence index, with token-aligned score lists.
Also saves global token frequencies for use at test time.

Usage:
    python compute_ig_scores.py --year 2015 --hops 2 \
        --model data/models/2015_LCR_hops2_lr0-09_dropout0-4_acc0-921875.pt \
        --variant entropy --phase Train

    python compute_ig_scores.py --year 2016 --hops 4 \
        --model data/models/2016_LCR_hops4_lr0-01_dropout0-4_acc0-8962765957446809.pt \
        --variant entropy --phase Train
"""

import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDatasetIG

DEFAULT_T = 50


def compute_integrated_gradients_all_classes(
    model, left, target, right, hops, device, T=DEFAULT_T
):
    """
    Compute IG for all 3 classes. Returns ig: [N, d, 3]
    """
    model.train()

    n_left = left.shape[0]
    n_target = target.shape[0]

    E = torch.cat([left, target, right], dim=0)  # [N, d]
    N, d = E.shape
    n_classes = 3

    ig_sum = torch.zeros(N, d, n_classes, device=device)

    for t in range(1, T + 1):
        alpha = t / T
        E_interp = (alpha * E).detach().requires_grad_(True)

        left_interp = E_interp[:n_left]
        target_interp = E_interp[n_left:n_left + n_target]
        right_interp = E_interp[n_left + n_target:]

        torch.set_default_device(device)
        output = model(left_interp, target_interp, right_interp, hops)
        torch.set_default_device('cpu')

        for c in range(n_classes):
            if E_interp.grad is not None:
                E_interp.grad.zero_()
            output[c].backward(retain_graph=(c < n_classes - 1))
            ig_sum[:, :, c] += E_interp.grad.detach()

    ig = (E.detach() / T).unsqueeze(2) * ig_sum  # [N, d, 3]

    model.eval()
    return ig


def score_target_class(ig: torch.Tensor, true_label: int) -> torch.Tensor:
    """
    A_i = sum_j |IG^(c*)_ij|, normalized within sentence -> alpha_i in [0,1]
    """
    ig_true = ig[:, :, true_label]          # [N, d]
    G = ig_true.abs().sum(dim=1)            # [N], L1 norm with abs
    G_sum = G.sum()
    if G_sum == 0:
        return torch.zeros_like(G)
    return G / G_sum


def score_entropy(ig: torch.Tensor) -> torch.Tensor:
    """
    A_i = log2(3) - H_i

    Steps:
    1. ig_token[c] = sum_j |IG^(c)_ij|  (abs over dims, per class)
    2. p^(c)_i = ig_token[c] / sum_c' ig_token[c']  (normalize over classes)
    3. H_i = -sum_c p^(c)_i * log2(p^(c)_i)
    4. A_i = log2(3) - H_i
    """
    N, d, n_classes = ig.shape

    ig_token = ig.abs().sum(dim=1)  # [N, 3], abs over dims

    ig_sum = ig_token.sum(dim=1, keepdim=True)  # [N, 1]
    zero_mask = (ig_sum.squeeze(1) == 0)

    ig_sum_safe = ig_sum.clone()
    ig_sum_safe[zero_mask] = 1.0

    p = ig_token / ig_sum_safe  # [N, 3]

    log_p = torch.zeros_like(p)
    nonzero = p > 0
    log_p[nonzero] = torch.log2(p[nonzero])
    H = -(p * log_p).sum(dim=1)  # [N]

    log2_3 = math.log2(3)
    A = log2_3 - H  # [N]
    A[zero_mask] = 0.0

    return A


def compute_token_frequencies(dataset) -> dict[str, float]:
    loader = DataLoader(dataset, collate_fn=lambda batch: batch)
    token_counts = {}
    total = 0
    for data in loader:
        _, _, _, tokens = data[0]
        for token_str in tokens:
            token_counts[token_str] = token_counts.get(token_str, 0) + 1
            total += 1
    return {token: count / total for token, count in token_counts.items()}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int)
    parser.add_argument("--hops", default=3, type=int)
    parser.add_argument("--model", type=str, required=True,
                        help="Path to Phase 1 model")
    parser.add_argument("--variant", choices=["target_class", "entropy"], required=True)
    parser.add_argument("--phase", default="Train", type=str)
    parser.add_argument("--T", default=DEFAULT_T, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LCRRotHopPlusPlus(hops=args.hops, dropout_prob=0.0).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    dataset = EmbeddingsDatasetIG(year=args.year, device=device, phase=args.phase)
    print(f"Dataset: {len(dataset)} {args.phase} samples")
    print(f"Variant: {args.variant}")

    per_sentence_scores = {}

    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f"  Processing sample {idx}/{len(dataset)}...")

        (left, target, right), label, hops, tokens = dataset[idx]
        true_label = label.item()

        torch.set_default_device(device)
        ig = compute_integrated_gradients_all_classes(
            model, left, target, right, hops, device, T=args.T)
        torch.set_default_device('cpu')

        if args.variant == "target_class":
            scores = score_target_class(ig, true_label)
        else:
            scores = score_entropy(ig)

        per_sentence_scores[str(idx)] = {
            "tokens": tokens,
            "scores": scores.cpu().tolist(),
            "true_label": true_label,
        }

    # Also compute token frequencies (always from training set)
    if args.phase == "Train":
        frequencies = compute_token_frequencies(dataset)
    else:
        # For test phase, still load training frequencies separately if needed
        frequencies = None

    save_path = os.path.join(
        "data", "models",
        f"{args.year}_ig_{args.variant}_{args.phase.lower()}_scores.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    output = {"per_sentence_scores": per_sentence_scores}
    if frequencies is not None:
        output["frequencies"] = frequencies

    with open(save_path, "w") as f:
        json.dump(output, f)

    print(f"\nSaved to {save_path}")

    # Quick distribution summary
    all_scores = []
    for s in per_sentence_scores.values():
        all_scores.extend(s["scores"])

    import statistics
    print(f"\nScore distribution ({args.variant}, {args.phase}):")
    print(f"  N tokens: {len(all_scores)}")
    print(f"  Min: {min(all_scores):.6f}")
    print(f"  Max: {max(all_scores):.6f}")
    print(f"  Mean: {sum(all_scores)/len(all_scores):.6f}")
    print(f"  Median: {statistics.median(all_scores):.6f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        idx = int(p / 100 * len(all_scores))
        print(f"  {p:3d}th percentile: {sorted(all_scores)[idx]:.6f}")


if __name__ == "__main__":
    main()
