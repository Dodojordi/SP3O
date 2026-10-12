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
    assert result["scored_by"] == ["math_verify", "wrong"]


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


@pytest.mark.parametrize(
    "prediction,target",
    [("2.001", "2"), ("2.9", "3"), ("0.333", "1/3"), ("3.14", "\\pi"), ("YES", "TRUE")],
)
def test_p1_reward_compatibility(prediction, target):
    # Retain even P1's permissive rounding: tightening it changes rewards.
    assert compute_score("\\boxed{" + prediction + "}", [target])["score"] == 1.0


def test_repeated_think_markers_match_p1():
    response = "<think>reasoning</think>\\boxed{2}</think>\\boxed{3}"
    assert compute_score(response, ["3"])["score"] == 0.0


def test_reward_dispatch_runs_off_event_loop(monkeypatch):
    import threading
    from slime.rollout.rm_hub import sp3o_math

    event_loop_thread = threading.get_ident()
    scorer_threads = []

    def score(*args, **kwargs):
        scorer_threads.append(threading.get_ident())
        return {"score": 1.0}

    monkeypatch.setattr(sp3o_math, "compute_score", score)
    sample = Sample(response="\\boxed{42}", label=["42"])
    args = Namespace(custom_rm_path=None, rm_type="sp3o_math")
    assert asyncio.run(async_rm(args, sample))["score"] == 1.0
    assert scorer_threads and scorer_threads[0] != event_loop_thread
