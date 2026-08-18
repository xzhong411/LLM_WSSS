

from __future__ import annotations

from typing import Tuple


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("paper_ops requires PyTorch") from exc
    return torch


def select_top1_confident_part(attention_maps, theta: float, eps: float = 1e-6) -> Tuple[int, object]:
    """Implement Eq. (2): argmax_k ||A_pk||_2 / Area(A_pk > theta).

    Args:
        attention_maps: Tensor shaped [K, H, W]
        theta: trainable confidence threshold
        eps: numerical guard for empty areas

    Returns:
        The selected component index and the score tensor shaped [K].
    """

    torch = _require_torch()
    if attention_maps.ndim != 3:
        raise ValueError("attention_maps must have shape [K, H, W]")
    l2 = torch.linalg.vector_norm(attention_maps.flatten(1), ord=2, dim=1)
    area = (attention_maps > theta).flatten(1).sum(dim=1).to(attention_maps.dtype)
    scores = l2 / (area + eps)
    return int(torch.argmax(scores).item()), scores


def support_feature_map(features, anchor_attention, tau: float):
    """Implement Eq. (3): keep F_kl only where A_p*(k,l) > tau."""

    torch = _require_torch()
    if features.ndim != 3:
        raise ValueError("features must have shape [C, H, W]")
    if anchor_attention.shape != features.shape[-2:]:
        raise ValueError("anchor_attention must have shape [H, W]")
    mask = (anchor_attention > tau).to(features.dtype).unsqueeze(0)
    return torch.where(mask.bool(), features, torch.zeros_like(features))


def hypercorrelation(features, support, eps: float = 1e-6):
    """Implement Eq. (4): ReLU cosine similarity T(i,j,k,l)."""

    torch = _require_torch()
    if features.shape != support.shape or features.ndim != 3:
        raise ValueError("features and support must both have shape [C, H, W]")
    features_n = features / (torch.linalg.vector_norm(features, dim=0, keepdim=True) + eps)
    support_n = support / (torch.linalg.vector_norm(support, dim=0, keepdim=True) + eps)
    corr = torch.einsum("chw,ckl->hwkl", features_n, support_n)
    return torch.relu(corr)

