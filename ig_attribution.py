"""
Integrated Gradients Attribution Module for IG-LCR-Rot-hop++

Implements:
- Integrated Gradients (IG) per token per class (Eq. 4.14-4.15)
- Token-level IG by summing over dimensions (Eq. 4.16)
- Entropy-based attribution score A_i (Eq. 4.17-4.19)
- Global attribution score A^g_w (Eq. 4.20)
- Bias classification (Eq. 4.21-4.23)
"""

import math
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset

# Number of interpolation steps for Riemann sum approximation of IG integral (Eq. 4.15)
# Higher T = more accurate approximation but slower computation
DEFAULT_T = 50


def compute_integrated_gradients(
    model: LCRRotHopPlusPlus,
    left: torch.Tensor,
    target: torch.Tensor,
    right: torch.Tensor,
    hops: Optional[torch.Tensor],
    device: torch.device,
    T: int = DEFAULT_T,
) -> torch.Tensor:
    """
    Compute Integrated Gradients for all tokens in a sentence.

    The model takes left, target, right as separate tensors. We concatenate them
    into a full embedding matrix E = [left; target; right] and compute gradients
    w.r.t. the full matrix, then split back. The baseline is the zero matrix E' = 0.

    Since E' = 0, Eq. 4.15 simplifies to:
        IG^(c)_ij ≈ (e_ij / T) * sum_{t=1}^{T} dF_c(t/T * E) / de_ij

    Returns:
        ig: tensor of shape [N, d, 3] where
            N = total number of tokens
            d = embedding dimension (768 for BERT base)
            3 = number of classes (negative, neutral, positive)
    """
    model.eval()

    # Concatenate full embedding matrix E = [left; target; right]
    n_left = left.shape[0]
    n_target = target.shape[0]

    E = torch.cat([left, target, right], dim=0)  # [N, d]
    N, d = E.shape
    n_classes = 3

    # Accumulate gradients over T interpolation steps
    # ig_sum[i, j, c] = sum_{t=1}^{T} dF_c(t/T * E) / de_ij
    ig_sum = torch.zeros(N, d, n_classes, device=device)

    for t in range(1, T + 1):
        # Interpolated input: alpha * E where alpha = t/T
        alpha = t / T
        E_interp = (alpha * E).detach().requires_grad_(True)  # [N, d]

        # Split back into left, target, right
        left_interp = E_interp[:n_left]
        target_interp = E_interp[n_left:n_left + n_target]
        right_interp = E_interp[n_left + n_target:]

        # Forward pass — get output probabilities for all 3 classes
        torch.set_default_device(device)
        output = model(left_interp, target_interp, right_interp, hops)  # [3]
        torch.set_default_device('cpu')

        # Compute gradient of each class output w.r.t. each embedding dimension
        for c in range(n_classes):
            # Zero out any existing gradients
            if E_interp.grad is not None:
                E_interp.grad.zero_()

            # Backpropagate for class c
            output[c].backward(retain_graph=(c < n_classes - 1))

            # Accumulate gradient: dF_c / de_ij for all i, j
            ig_sum[:, :, c] += E_interp.grad.detach()  # [N, d]

    # Apply IG formula (Eq. 4.15): IG^(c)_ij ≈ (e_ij / T) * sum_t grad
    # E / T multiplied by accumulated gradients
    ig = (E.detach() / T).unsqueeze(2) * ig_sum  # [N, d, 3]

    return ig


def compute_attribution_score(ig: torch.Tensor) -> torch.Tensor:
    """
    Compute token-level attribution score A_i from IG values.

    Steps:
    1. Sum over embedding dimensions to get IG^(c)_i (Eq. 4.16)
    2. Normalise to get probability distribution p^(c)_i (Eq. 4.17)
    3. Compute Shannon entropy H_i (Eq. 4.18)
    4. Compute attribution score A_i = log2(3) - H_i (Eq. 4.19)

    Args:
        ig: [N, d, 3] tensor of IG values

    Returns:
        A: [N] tensor of attribution scores, one per token
    """
    N, d, n_classes = ig.shape

    # Step 1: Sum over dimensions -> IG^(c)_i, shape [N, 3]
    ig_token = ig.sum(dim=1)  # [N, 3]

    # Step 2: Normalise over classes to get probability distribution (Eq. 4.17)
    # Handle edge case where all IG values are 0 for a token
    ig_abs_sum = ig_token.abs().sum(dim=1, keepdim=True)  # [N, 1]
    zero_mask = (ig_abs_sum.squeeze(1) == 0)  # [N] bool mask

    # Avoid division by zero
    ig_abs_sum_safe = ig_abs_sum.clone()
    ig_abs_sum_safe[zero_mask] = 1.0

    p = ig_token.abs() / ig_abs_sum_safe  # [N, 3], normalised

    # Step 3: Shannon entropy H_i (Eq. 4.18)
    # H_i = -sum_c p^(c)_i * log2(p^(c)_i), with 0*log(0) = 0
    log_p = torch.zeros_like(p)
    nonzero = p > 0
    log_p[nonzero] = torch.log2(p[nonzero])
    H = -(p * log_p).sum(dim=1)  # [N]

    # Step 4: Attribution score A_i = log2(3) - H_i (Eq. 4.19)
    log2_3 = math.log2(3)
    A = log2_3 - H  # [N], ranges from 0 to log2(3)

    # Edge case: tokens with zero IG get attribution score 0
    A[zero_mask] = 0.0

    return A


def compute_global_attribution_scores(
    model: LCRRotHopPlusPlus,
    dataset: EmbeddingsDataset,
    device: torch.device,
    T: int = DEFAULT_T,
) -> dict[str, float]:
    """
    Compute global attribution scores A^g_w for all tokens in the dataset (Eq. 4.20).

    For each sentence, computes IG and attribution scores, then averages
    attribution scores per token type across all occurrences in the dataset.

    Args:
        model: trained LCRRotHopPlusPlus model
        dataset: EmbeddingsDataset with token strings stored
        device: torch device
        T: number of interpolation steps

    Returns:
        global_scores: dict mapping token string -> global attribution score A^g_w
    """
    model.eval()
    loader = DataLoader(dataset, collate_fn=lambda batch: batch)

    # Accumulate attribution scores and counts per token
    token_scores = defaultdict(float)   # token -> sum of A_i across occurrences
    token_counts = defaultdict(int)     # token -> number of occurrences

    print(f"Computing IG attribution scores over {len(dataset)} samples...")

    for i, data in enumerate(loader):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{len(dataset)}...")

        (left, target, right), label, hops, tokens = data[0]

        # Compute IG for this sentence
        ig = compute_integrated_gradients(model, left, target, right, hops, device, T)

        # Compute attribution scores A_i for each token
        A = compute_attribution_score(ig)  # [N]

        # Accumulate scores per token string
        for token_idx, token_str in enumerate(tokens):
            token_scores[token_str] += A[token_idx].item()
            token_counts[token_str] += 1

    # Compute global average score A^g_w (Eq. 4.20)
    global_scores = {
        token: token_scores[token] / token_counts[token]
        for token in token_scores
    }

    return global_scores


def compute_token_frequencies(dataset: EmbeddingsDataset) -> dict[str, float]:
    """
    Compute dataset-level token frequencies f(w) (Eq. 4.21).

    f(w) = N_w / sum_v N_v

    Args:
        dataset: EmbeddingsDataset with token strings stored

    Returns:
        frequencies: dict mapping token string -> frequency f(w)
    """
    loader = DataLoader(dataset, collate_fn=lambda batch: batch)

    token_counts = defaultdict(int)
    total_tokens = 0

    for data in loader:
        _, _, _, tokens = data[0]
        for token_str in tokens:
            token_counts[token_str] += 1
            total_tokens += 1

    frequencies = {
        token: count / total_tokens
        for token, count in token_counts.items()
    }

    return frequencies


def build_bias_dictionary(
    global_scores: dict[str, float],
    frequencies: dict[str, float],
    tau: float,
    omega: float,
) -> tuple[set, set]:
    """
    Build the bias dictionary V_u (unbiased) and V_b (biased) (Eq. 4.22-4.23).

    Token w is biased if: A^g_w < tau AND f(w) > omega

    Args:
        global_scores: dict mapping token -> A^g_w
        frequencies: dict mapping token -> f(w)
        tau: attribution score threshold
        omega: frequency threshold

    Returns:
        V_u: set of unbiased tokens
        V_b: set of biased tokens
    """
    V_b = set()
    V_u = set()

    for token in global_scores:
        score = global_scores[token]
        freq = frequencies.get(token, 0.0)

        if score < tau and freq > omega:
            V_b.add(token)
        else:
            V_u.add(token)

    print(f"Bias dictionary built: {len(V_b)} biased tokens, {len(V_u)} unbiased tokens")
    print(f"Biased tokens: {sorted(V_b)[:20]}{'...' if len(V_b) > 20 else ''}")

    return V_u, V_b


def mask_biased_embeddings(
    left: torch.Tensor,
    target: torch.Tensor,
    right: torch.Tensor,
    tokens: list[str],
    V_b: set,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mask embeddings of biased tokens by setting them to zero (Eq. 4.24).

    e_tilde_i = 0 if token w in V_b, else e_i

    Args:
        left: [n_left, d] left context embeddings
        target: [n_target, d] target embeddings
        right: [n_right, d] right context embeddings
        tokens: list of token strings for all N tokens
        V_b: set of biased tokens

    Returns:
        left, target, right with biased token embeddings zeroed out
    """
    n_left = left.shape[0]
    n_target = target.shape[0]

    left_masked = left.clone()
    target_masked = target.clone()
    right_masked = right.clone()

    for i, token_str in enumerate(tokens):
        if token_str in V_b:
            if i < n_left:
                left_masked[i] = 0.0
            elif i < n_left + n_target:
                target_masked[i - n_left] = 0.0
            else:
                right_masked[i - n_left - n_target] = 0.0

    return left_masked, target_masked, right_masked
