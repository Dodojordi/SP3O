import asyncio
from argparse import Namespace

import pytest

from slime.rollout.rm_hub import async_rm
from slime.rollout.rm_hub.sp3o_math import compute_score
from slime.utils.types import Sample


def test_single_boxed_answer():
    result = compute_score("reasoning... \\boxed{42}", ["42"])

    assert result["score"] == 1.0
    assert result["acc"] is True


def test_only_visible_answer_after_think_is_scored():
    result = compute_score("\\boxed{1}</think>final: \\boxed{2}", ["2"])

    assert result["score"] == 1.0
    assert result["extracted_pred"] == ["2"]


def test_equivalent_fraction_is_accepted():
    result = compute_score("\\boxed{1/2}", ["\\boxed{0.5}"])

    assert result["score"] == 1.0


def test_multiple_answers_and_points():
    result = compute_score("\\boxed{2} and \\boxed{4}", ["2", "5"], points=[3.0, 7.0])

    assert result["score"] == 0.5
    assert result["point"] == 3.0
    assert result["scored_by"] == ["rule", "wrong"]


def test_missing_box_is_incorrect():
    result = compute_score("the answer is 42", ["42"])

    assert result["score"] == 0.0


def test_empty_labels_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        compute_score("\\boxed{1}", [])


def test_reward_hub_dispatches_to_local_sp3o_scorer():
    args = Namespace(custom_rm_path=None, rm_type="sp3o_math")
    sample = Sample(response="final: \\boxed{42}", label=["42"])

    result = asyncio.run(async_rm(args, sample))

    assert result["score"] == 1.0
