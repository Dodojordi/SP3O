from argparse import Namespace

import pytest
import torch

from slime.utils.sp3o import (
    build_critic_value_loss_masks,
    build_ratio_anchor_mask,
    in_dense_critic_only_warmup,
    validate_sp3o_args,
)


def _args(**overrides):
    values = {
        "critic_token_loss": True,
        "critic_token_ratios": [0.3, 0.6, 0.9],
        "critic_extra_tail_ratio": None,
        "critic_extra_tail_min_response_len": 0,
        "dense_critic_only_warmup": False,
        "num_critic_only_steps": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_main_ratios_select_three_valid_response_positions():
    loss_mask = torch.ones(11)

    result = build_critic_value_loss_masks(_args(), [loss_mask])[0]

    assert result.tolist() == [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]


def test_ratios_count_only_positions_enabled_by_original_mask():
    loss_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

    result = build_critic_value_loss_masks(_args(), [loss_mask])[0]

    assert result.tolist() == [0, 0, 1, 1, 0, 1, 0]


def test_short_response_deduplicates_colliding_anchors():
    loss_mask = torch.ones(2)

    result = build_critic_value_loss_masks(_args(), [loss_mask])[0]

    assert result.tolist() == [1, 0]


def test_empty_response_stays_empty():
    result = build_critic_value_loss_masks(_args(), [torch.zeros(5)])[0]

    assert torch.count_nonzero(result).item() == 0


def test_extra_tail_anchor_obeys_valid_token_threshold():
    args = _args(critic_extra_tail_ratio=0.95, critic_extra_tail_min_response_len=10)

    short = build_critic_value_loss_masks(args, [torch.ones(9)])[0]
    long = build_critic_value_loss_masks(args, [torch.ones(10)])[0]

    assert int(short.sum().item()) == 3
    assert int(long.sum().item()) == 4
    assert long[9].item() == 1.0


def test_input_policy_mask_is_not_mutated():
    loss_mask = torch.ones(8)
    original = loss_mask.clone()

    result = build_critic_value_loss_masks(_args(), [loss_mask])[0]

    assert torch.equal(loss_mask, original)
    assert result.data_ptr() != loss_mask.data_ptr()


def test_dense_mode_returns_original_masks():
    loss_mask = torch.ones(8)
    result = build_critic_value_loss_masks(_args(critic_token_loss=False), [loss_mask])

    assert result[0] is loss_mask


def test_ratio_one_can_select_terminal_position():
    mask = build_ratio_anchor_mask(torch.ones(5), [1.0])

    assert mask.tolist() == [False, False, False, False, True]


@pytest.mark.parametrize("ratios", [[-0.1], [1.1], [float("inf")], [float("nan")]])
def test_invalid_ratios_are_rejected(ratios):
    with pytest.raises(ValueError, match="finite and in"):
        build_critic_value_loss_masks(_args(critic_token_ratios=ratios), [torch.ones(4)])


@pytest.mark.parametrize("ratios", [[0.6, 0.3], [0.3, 0.3]])
def test_ambiguous_ratio_lists_are_rejected(ratios):
    with pytest.raises(ValueError):
        validate_sp3o_args(_args(critic_token_ratios=ratios))


def test_tail_requires_a_positive_threshold():
    with pytest.raises(ValueError, match="positive"):
        validate_sp3o_args(_args(critic_extra_tail_ratio=0.95))


def test_sparse_mode_requires_at_least_one_anchor_selector():
    with pytest.raises(ValueError, match="requires"):
        validate_sp3o_args(_args(critic_token_ratios=None))


def test_dense_warmup_switches_at_num_critic_only_steps():
    args = _args(dense_critic_only_warmup=True, num_critic_only_steps=20)

    assert in_dense_critic_only_warmup(args, {"rollout_id": 19})
    assert not in_dense_critic_only_warmup(args, {"rollout_id": 20})


def test_dense_warmup_requires_positive_duration():
    with pytest.raises(ValueError, match="num-critic-only-steps"):
        validate_sp3o_args(_args(dense_critic_only_warmup=True))
