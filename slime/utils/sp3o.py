"""Sparse critic supervision utilities for SP3O.

SP3O changes only the positions used by the critic value loss. Policy loss
masks and value targets are left unchanged.
"""

from argparse import Namespace
import math

import torch


def add_sp3o_arguments(parser):
    """Register the public CLI surface used by SP3O."""
    parser.add_argument(
        "--critic-token-loss",
        action="store_true",
        help="Apply critic value loss only at configured response anchors.",
    )
    parser.add_argument(
        "--critic-token-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Response-relative critic anchors, for example: 0.3 0.6 0.9.",
    )
    parser.add_argument(
        "--critic-extra-tail-ratio",
        type=float,
        default=None,
        help="Optional extra critic anchor for sufficiently long responses.",
    )
    parser.add_argument(
        "--critic-extra-tail-min-response-len",
        type=int,
        default=0,
        help="Minimum valid response length required for the extra tail anchor.",
    )
    parser.add_argument(
        "--dense-critic-only-warmup",
        action="store_true",
        help=(
            "Use dense critic loss while rollout_id is smaller than "
            "--num-critic-only-steps, then switch to configured sparse anchors."
        ),
    )
    return parser


def _validated_ratios(value, *, name: str) -> list[float]:
    if value is None:
        return []
    values = [float(value)] if isinstance(value, (int, float)) else [float(item) for item in value]
    for ratio in values:
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} values must be finite and in [0, 1], got {ratio}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be unique, got {values}")
    if values != sorted(values):
        raise ValueError(f"{name} values must be sorted, got {values}")
    return values


def validate_sp3o_args(args: Namespace) -> None:
    """Validate SP3O options after the upstream argument parser runs."""
    ratios = _validated_ratios(getattr(args, "critic_token_ratios", None), name="critic_token_ratios")
    tail = _validated_ratios(getattr(args, "critic_extra_tail_ratio", None), name="critic_extra_tail_ratio")
    min_len = int(getattr(args, "critic_extra_tail_min_response_len", 0) or 0)
    if min_len < 0:
        raise ValueError("critic_extra_tail_min_response_len must be non-negative")
    if getattr(args, "critic_token_loss", False) and not ratios and not tail:
        raise ValueError("--critic-token-loss requires --critic-token-ratios or --critic-extra-tail-ratio")
    if tail and min_len <= 0:
        raise ValueError("--critic-extra-tail-ratio requires a positive --critic-extra-tail-min-response-len")
    if getattr(args, "dense_critic_only_warmup", False) and int(getattr(args, "num_critic_only_steps", 0) or 0) <= 0:
        raise ValueError("--dense-critic-only-warmup requires --num-critic-only-steps > 0")


def build_ratio_anchor_mask(
    loss_mask: torch.Tensor,
    ratios: list[float],
    *,
    avoid_terminal_for_nonterminal_ratio: bool = True,
) -> torch.Tensor:
    """Map response-relative ratios onto positions enabled by ``loss_mask``."""
    anchor_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
    valid_positions = torch.nonzero(loss_mask.to(dtype=torch.bool), as_tuple=False).squeeze(-1)
    if valid_positions.numel() == 0:
        return anchor_mask

    last_valid_offset = valid_positions.numel() - 1
    for ratio in ratios:
        offset = int(round(last_valid_offset * ratio))
        offset = max(0, min(last_valid_offset, offset))
        if avoid_terminal_for_nonterminal_ratio and ratio < 1.0 and last_valid_offset > 0:
            offset = min(offset, last_valid_offset - 1)
        anchor_mask[valid_positions[offset]] = True
    return anchor_mask


def build_critic_value_loss_masks(
    args: Namespace,
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Return critic-only masks without mutating the input policy masks."""
    if not getattr(args, "critic_token_loss", False):
        return loss_masks

    ratios = _validated_ratios(getattr(args, "critic_token_ratios", None), name="critic_token_ratios")
    tail_ratios = _validated_ratios(
        getattr(args, "critic_extra_tail_ratio", None), name="critic_extra_tail_ratio"
    )
    tail_min_len = int(getattr(args, "critic_extra_tail_min_response_len", 0) or 0)

    result = []
    for loss_mask in loss_masks:
        sparse = build_ratio_anchor_mask(loss_mask, ratios)
        valid_count = int(loss_mask.to(dtype=torch.bool).sum().item())
        if tail_ratios and valid_count >= tail_min_len:
            sparse |= build_ratio_anchor_mask(
                loss_mask,
                tail_ratios,
                avoid_terminal_for_nonterminal_ratio=False,
            )
        result.append(loss_mask * sparse.to(dtype=loss_mask.dtype))
    return result


def in_dense_critic_only_warmup(args: Namespace, batch) -> bool:
    if not getattr(args, "dense_critic_only_warmup", False):
        return False
    rollout_id = batch.get("rollout_id")
    if rollout_id is None:
        return False
    return int(rollout_id) < int(getattr(args, "num_critic_only_steps", 0) or 0)
