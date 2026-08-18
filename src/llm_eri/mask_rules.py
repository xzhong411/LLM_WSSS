"""Mask utilities, including clearly labeled non-paper fallback rules."""

from __future__ import annotations


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mask_rules requires PyTorch") from exc
    return torch


def resolve_conflicts_max_score(class_scores, threshold: float = 0.5, background_index: int = 0):
    """Resolve overlapping class-wise masks by max score.

    This is a reproduction fallback, not a paper-specified rule. The paper
    gives M_q = sigmoid(MLP(Concat(A_1, ..., A_K))) for a query mask, but it
    does not specify how multi-label classes are merged at pixel level.

    Args:
        class_scores: Tensor shaped [C, H, W], foreground scores after sigmoid.
        threshold: pixels below this max foreground score become background.
        background_index: label used for background in the returned map.
    """

    torch = _require_torch()
    if class_scores.ndim != 3:
        raise ValueError("class_scores must have shape [C, H, W]")
    max_scores, labels = class_scores.max(dim=0)
    labels = labels + 1
    bg = torch.full_like(labels, int(background_index))
    return torch.where(max_scores >= threshold, labels, bg)

