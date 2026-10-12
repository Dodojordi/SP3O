"""Offline reward matching slime_B's compute_score_p1(use_xverify=False).

The P1 numerical tolerances and extraction behavior are intentional: changing
them changes training rewards. No external judge or Math Verify is invoked.
"""

from .p1_math_rules import grade_answer_mathd, grade_answer_sympy, solution2answer


def compute_score(response: str, label, points=None) -> dict:
    labels = list(label) if isinstance(label, (list, tuple)) else [label]
    if not labels:
        raise ValueError("SP3O math reward requires at least one label")

    # Match P1 exactly, including its behavior for repeated </think> markers.
    visible_response = response.split("</think>")[1] if "</think>" in response else response
    predictions = solution2answer(str(visible_response), num_answers=len(labels))
    targets = [solution2answer(str(item), return_origin=True)[0] for item in labels]
    correct = []
    for prediction, target in zip(predictions, targets, strict=True):
        prediction = "" if prediction is None else prediction
        target = "" if target is None else target
        split_prediction = prediction.split("=")[-1]
        split_target = target.split("=")[-1]
        correct.append(
            grade_answer_mathd(prediction, target)[0]
            or grade_answer_sympy(prediction, target)[0]
            or grade_answer_mathd(split_prediction, split_target)[0]
            or grade_answer_sympy(split_prediction, split_target)[0]
        )

    score = sum(correct) / len(labels)
    point = sum(ok * weight for ok, weight in zip(correct, points)) if points and len(points) == len(labels) else score
    return {
        "score": score,
        "point": point,
        "acc": abs(score - 1.0) < 1e-5,
        "extracted_pred": predictions,
        "extracted_gt": targets,
        # P1's historical name is retained for log compatibility. This does
        # not mean that the optional Math Verify package was called.
        "scored_by": ["math_verify" if ok else "wrong" for ok in correct],
    }
