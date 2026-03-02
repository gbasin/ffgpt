from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_goodness(activations: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Compute per-example mean goodness over non-PAD positions."""
    if activations.ndim != 3:
        raise ValueError(f"Expected activations of shape [B, T, H], got {tuple(activations.shape)}")
    if pad_mask.ndim != 2:
        raise ValueError(f"Expected pad_mask of shape [B, T], got {tuple(pad_mask.shape)}")

    token_goodness = activations.mean(dim=-1)
    mask = pad_mask.to(dtype=token_goodness.dtype)
    masked_sum = (token_goodness * mask).sum(dim=-1)
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return masked_sum / denom


def ff_loss_bce(g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    threshold_tensor = (
        threshold
        if isinstance(threshold, torch.Tensor)
        else torch.tensor(float(threshold), device=g_pos.device, dtype=g_pos.dtype)
    )

    logits = torch.cat([g_pos - threshold_tensor, g_neg - threshold_tensor], dim=0)
    labels = torch.cat([torch.ones_like(g_pos), torch.zeros_like(g_neg)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def update_threshold_ema(
    g_pos: torch.Tensor,
    g_neg: torch.Tensor,
    current_threshold: torch.Tensor | None,
    momentum: float = 0.9,
) -> torch.Tensor:
    target = ((g_pos.mean() + g_neg.mean()) * 0.5).detach()
    if current_threshold is None:
        return target
    return momentum * current_threshold + (1.0 - momentum) * target


def candidate_set_ce_loss(
    candidate_logits: torch.Tensor,
    targets: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    logits = candidate_logits / temperature
    if targets is None:
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)
