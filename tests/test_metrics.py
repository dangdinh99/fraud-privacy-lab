"""Hand-computed tests for fraudpriv.metrics."""

import numpy as np
import pytest

from fraudpriv.metrics import (
    budget_metrics,
    evaluate,
    expected_calibration_error,
    flag_top_budget,
    n_flagged,
    pr_auc,
    recall_ceiling,
)

# --- number flagged -------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "budget", "expected"),
    [
        (100, 0.01, 1),
        (150, 0.01, 2),  # 1.5 rounds up
        (100, 0.07, 7),  # float 0.07 * 100 = 7.000000000000001 must not become 8
        (1000, 0.005, 5),
        (3, 1.0, 3),
    ],
)
def test_n_flagged(n, budget, expected):
    assert n_flagged(n, budget) == expected


@pytest.mark.parametrize("budget", [0.0, -0.01, 1.5])
def test_n_flagged_rejects_bad_budget(budget):
    with pytest.raises(ValueError):
        n_flagged(100, budget)


# --- ties at the cutoff ---------------------------------------------------------


def test_ties_broken_by_ascending_transaction_id():
    # Budget 0.6 of 5 -> 3 flagged. id 10 (0.9) is first; three ids tie at 0.5 for
    # the remaining two slots, so the two lowest ids (2, 3) win and id 4 does not.
    ids = np.array([10, 4, 2, 3, 1])
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    mask = flag_top_budget(scores, ids, 0.6)
    assert set(ids[mask]) == {10, 2, 3}


def test_ties_independent_of_input_order():
    rng = np.random.default_rng(0)
    ids = np.arange(100)
    scores = rng.choice([0.1, 0.5, 0.9], size=100)
    base = set(ids[flag_top_budget(scores, ids, 0.17)])
    perm = rng.permutation(100)
    assert set(ids[perm][flag_top_budget(scores[perm], ids[perm], 0.17)]) == base
    assert len(base) == 17


def test_flag_rejects_nan_and_duplicate_ids():
    with pytest.raises(ValueError):
        flag_top_budget([0.1, np.nan], [1, 2], 0.5)
    with pytest.raises(ValueError):
        flag_top_budget([0.1, 0.2], [1, 1], 0.5)


# --- recall ceiling -------------------------------------------------------------


def test_recall_ceiling_hand_computed():
    assert recall_ceiling(n=200, n_fraud=10, budget=0.01) == pytest.approx(0.2)  # 2 / 10
    assert recall_ceiling(n=200, n_fraud=1, budget=0.01) == 1.0  # min(1, 2 / 1)
    assert recall_ceiling(n=150, n_fraud=4, budget=0.01) == pytest.approx(0.5)  # ceil(1.5)=2


def test_recall_ceiling_requires_fraud():
    with pytest.raises(ValueError):
        recall_ceiling(n=100, n_fraud=0, budget=0.01)


def test_perfect_model_reaches_ceiling_and_never_exceeds_it():
    # n=150, budget 1% -> ceil(1.5) = 2 flagged, 2 frauds. A perfect model catches both:
    # recall 1.0. With k*n in the numerator the ceiling would be 0.75 < recall.
    n = 150
    y = np.zeros(n, dtype=int)
    y[[5, 80]] = 1
    m = budget_metrics(y, y.astype(float), np.arange(n), np.ones(n), 0.01)
    assert m["recall"] == 1.0
    assert m["recall_ceiling"] == 1.0
    assert m["recall_share_of_ceiling"] == 1.0


# --- full hand-computed example: recall, precision, dollar loss ------------------


def _example():
    # 10 transactions, budget 0.2 -> 2 flagged: ids 1 (0.9) and 2 (0.8).
    # Frauds: ids 1 ($100), 3 ($250), 7 ($40). Caught: id 1 only.
    ids = np.arange(1, 11)
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.15, 0.1, 0.7, 0.05, 0.04, 0.03])
    amounts = np.array([100, 50, 250, 10, 10, 10, 40, 10, 10, 10], dtype=float)
    return y, scores, ids, amounts


def test_budget_metrics_hand_computed():
    m = budget_metrics(*_example(), budget=0.2)
    assert m["n_flagged"] == 2
    assert m["n_fraud"] == 3
    assert m["frauds_flagged"] == 1
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["recall_ceiling"] == pytest.approx(2 / 3)
    assert m["recall_share_of_ceiling"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["dollar_loss"] == pytest.approx(290.0)  # 250 + 40 missed
    assert m["fraud_dollars_total"] == pytest.approx(390.0)


def test_dollar_loss_zero_when_all_fraud_flagged():
    y, scores, ids, amounts = _example()
    m = budget_metrics(y, scores, ids, amounts, budget=0.5)  # 5 flagged: ids 1,2,7,3,4
    assert m["frauds_flagged"] == 3
    assert m["dollar_loss"] == 0.0
    assert m["recall_ceiling"] == 1.0


# --- secondary metrics ------------------------------------------------------------


def test_ece_hand_computed():
    # 10 bins. Bin 0: mean p 0.05, rate 0.5 -> 0.45 * 2/4. Bin 9: 0.95 vs 1.0 -> 0.05 * 2/4.
    y = np.array([0, 1, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.25)


def test_ece_includes_one_in_last_bin_and_rejects_non_probabilities():
    assert expected_calibration_error([1], [1.0], n_bins=10) == 0.0
    with pytest.raises(ValueError):
        expected_calibration_error([1, 0], [1.5, 0.2], n_bins=10)


def test_pr_auc_perfect():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_evaluate_structure_and_no_accuracy():
    out = evaluate(*_example(), budgets=(0.2, 0.1), ece_bins=10)
    assert {"pr_auc", "ece", "ece_bins", "budget_0.2", "budget_0.1"} <= out.keys()
    assert "recall_share_of_ceiling" in out["budget_0.2"]
    assert not any("acc" in k for k in out)
    assert "ece" not in evaluate(*_example(), budgets=(0.2,), ece_bins=None)


# --- segments (seen vs new cards) ---------------------------------------------------


def test_segment_recall_hand_computed():
    # Flagged: ids 1, 2. Segment a = ids 1-5 (frauds 1, 3 -> 1 caught); b = ids 6-10
    # (fraud 7 -> 0 caught); c = ids 9-10 (no fraud -> recall None).
    ids = np.arange(1, 11)
    segs = {"a": ids <= 5, "b": ids > 5, "c": ids >= 9}
    m = budget_metrics(*_example(), budget=0.2, segments=segs)
    assert m["segments"]["a"] == {"n": 5, "n_fraud": 2, "frauds_flagged": 1, "recall": 0.5}
    assert m["segments"]["b"] == {"n": 5, "n_fraud": 1, "frauds_flagged": 0, "recall": 0.0}
    assert m["segments"]["c"]["recall"] is None


def test_segments_do_not_change_flags():
    ids = np.arange(1, 11)
    plain = budget_metrics(*_example(), budget=0.2)
    seg = budget_metrics(*_example(), budget=0.2, segments={"a": ids <= 5})
    seg.pop("segments")
    assert seg == plain


def test_segment_wrong_length_raises():
    with pytest.raises(ValueError):
        budget_metrics(*_example(), budget=0.2, segments={"a": np.ones(3, dtype=bool)})


def test_evaluate_ece_uses_probs_and_ranking_uses_scores():
    y, scores, ids, amounts = _example()
    raw = np.log(scores / (1 - scores))  # logits: same ranking, not probabilities
    out = evaluate(y, raw, ids, amounts, budgets=(0.2,), ece_bins=10, probs=scores)
    ref = evaluate(y, scores, ids, amounts, budgets=(0.2,), ece_bins=10)
    assert out["ece"] == pytest.approx(ref["ece"])
    assert out["budget_0.2"] == ref["budget_0.2"]
    assert out["pr_auc"] == pytest.approx(ref["pr_auc"])


# --- aggregation over windows (replay weeks) ---------------------------------------


def test_aggregate_windows_hand_computed():
    from fraudpriv.metrics import aggregate_windows

    # Window 1: 2 flagged, 3 frauds, 1 caught. Window 2: 4 flagged, 1 fraud, 1 caught.
    w1 = {
        "n": 100,
        "n_fraud": 3,
        "n_flagged": 2,
        "frauds_flagged": 1,
        "dollar_loss": 50.0,
        "fraud_dollars_total": 80.0,
    }
    w2 = {
        "n": 200,
        "n_fraud": 1,
        "n_flagged": 4,
        "frauds_flagged": 1,
        "dollar_loss": 0.0,
        "fraud_dollars_total": 20.0,
    }
    a = aggregate_windows([w1, w2])
    assert a["n_fraud"] == 4 and a["frauds_flagged"] == 2 and a["n_flagged"] == 6
    assert a["recall"] == pytest.approx(0.5)
    assert a["recall_ceiling"] == pytest.approx(3 / 4)  # min(2, 3) + min(4, 1) = 3
    assert a["recall_share_of_ceiling"] == pytest.approx(2 / 3)
    assert a["precision"] == pytest.approx(2 / 6)
    assert a["dollar_loss"] == 50.0 and a["fraud_dollars_total"] == 100.0


def test_aggregate_windows_matches_single_window():
    from fraudpriv.metrics import aggregate_windows

    m = budget_metrics(*_example(), budget=0.2)
    a = aggregate_windows([m])
    for k in ("recall", "recall_ceiling", "recall_share_of_ceiling", "precision", "dollar_loss"):
        assert a[k] == pytest.approx(m[k])


def test_aggregate_windows_rejects_no_fraud():
    from fraudpriv.metrics import aggregate_windows

    w = {
        "n": 10,
        "n_fraud": 0,
        "n_flagged": 1,
        "frauds_flagged": 0,
        "dollar_loss": 0.0,
        "fraud_dollars_total": 0.0,
    }
    with pytest.raises(ValueError):
        aggregate_windows([w])
