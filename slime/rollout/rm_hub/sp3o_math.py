"""Local rule-based math reward used by the SP3O experiments.

The scorer intentionally has no network fallback. It extracts the final boxed
answer(s) and reuses slime's MathD and SymPy equivalence checks.
"""

from slime.rollout.rm_hub.math_utils import (
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)


def _last_boxed_answers(text: str, count: int) -> list[str]:
    remaining = str(text)
    answers = []
    while remaining and len(answers) < count:
        end = len(remaining)
        boxed = None
        while end > 0:
            candidate = remaining[:end]
            boxed = extract_boxed_answer(candidate)
            if boxed is not None:
                marker = max(candidate.rfind("\\boxed"), candidate.rfind("\\fbox"))
                remaining = candidate[:marker]
                break
            marker = max(candidate.rfind("\\boxed"), candidate.rfind("\\fbox"))
            if marker < 0:
                end = 0
            else:
                end = marker
        if boxed is None:
            break
        answers.append(boxed)
    answers.reverse()
    return [""] * (count - len(answers)) + answers


def _ground_truth(value) -> str:
    text = str(value)
    extracted = extract_boxed_answer(text)
    return extracted if extracted is not None else text


def _equivalent(prediction: str, target: str) -> bool:
    candidates = [(prediction, target)]
    if "=" in prediction or "=" in target:
        candidates.append((prediction.split("=")[-1], target.split("=")[-1]))
    for pred, gold in candidates:
        if grade_answer_mathd(pred, gold) or grade_answer_sympy(pred, gold):
            return True
    return False


def compute_score(response: str, label, points=None) -> dict:
    """Score one response against one or more expected boxed answers."""
    labels = list(label) if isinstance(label, (list, tuple)) else [label]
    if not labels:
        raise ValueError("SP3O math reward requires at least one label")

    visible_response = str(response).split("</think>", maxsplit=1)[-1]
    predictions = _last_boxed_answers(visible_response, len(labels))
    targets = [_ground_truth(item) for item in labels]
    correct = [_equivalent(pred, gold) for pred, gold in zip(predictions, targets, strict=True)]
    score = sum(correct) / len(correct)

    if points is not None and len(points) == len(correct):
        point = sum(float(weight) * ok for weight, ok in zip(points, correct, strict=True))
    else:
        point = score

    return {
        "score": float(score),
        "point": float(point),
        "acc": score == 1.0,
        "extracted_pred": predictions,
        "extracted_gt": targets,
        "scored_by": ["rule" if ok else "wrong" for ok in correct],
    }
