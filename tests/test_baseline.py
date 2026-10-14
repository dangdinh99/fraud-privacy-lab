"""Tests for fraudpriv.baseline: calibration, reproducibility, and Opacus compatibility."""

import numpy as np
import pytest
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from fraudpriv.baseline import (
    MLP,
    PlattCalibrator,
    fit_mlp,
    make_mlp_training,
    poisson_loader,
    score_lightgbm,
    score_mlp,
    train_lightgbm,
    train_mlp,
)
from fraudpriv.metrics import budget_metrics

MLP_PARAMS = {"hidden": [16, 8], "batch_size": 64, "lr": 0.01, "epochs": 2}
LGB_PARAMS = {
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 5,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_threads": 1,
    "max_rounds": 30,
    "early_stopping_rounds": 5,
}


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6)).astype(np.float32)
    p = 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1] - 2)))
    y = (rng.random(800) < p).astype(int)
    return X, y


def test_platt_preserves_ranking_and_budget_metrics(data):
    X, y = data
    scores = 2 * X[:, 0] - X[:, 1] + 0.01 * X[:, 2]
    probs = PlattCalibrator().fit(scores, y).transform(scores)
    assert np.all(np.diff(probs[np.argsort(scores)]) >= 0)
    ids, amounts = np.arange(len(y)), np.ones(len(y))
    raw = budget_metrics(y, scores, ids, amounts, 0.05)
    cal = budget_metrics(y, probs, ids, amounts, 0.05)
    assert raw == cal


def test_platt_rejects_inverted_scores(data):
    X, y = data
    with pytest.raises(ValueError):
        PlattCalibrator().fit(-(2 * X[:, 0] - X[:, 1]), y)


def test_mlp_is_opacus_compatible():
    assert ModuleValidator.validate(MLP(10, [128, 64]), strict=False) == []


def test_mlp_same_seed_same_scores(data):
    X, y = data
    a, info = train_mlp(X, y, MLP_PARAMS, seed=0)
    b, _ = train_mlp(X, y, MLP_PARAMS, seed=0)
    c, _ = train_mlp(X, y, MLP_PARAMS, seed=1)
    np.testing.assert_array_equal(score_mlp(a, X), score_mlp(b, X))
    assert not np.array_equal(score_mlp(a, X), score_mlp(c, X))
    assert info["steps"] == MLP_PARAMS["epochs"] * len(poisson_loader(X, y, 64, 0))


def test_lightgbm_same_seed_same_scores(data):
    X, y = data
    a, info = train_lightgbm(X[:600], y[:600], X[600:], y[600:], LGB_PARAMS, seed=0)
    b, _ = train_lightgbm(X[:600], y[:600], X[600:], y[600:], LGB_PARAMS, seed=0)
    np.testing.assert_array_equal(score_lightgbm(a, X), score_lightgbm(b, X))
    assert info["best_iteration"] >= 1


def test_cosine_schedule_reaches_zero_nonprivate(data):
    X, y = data
    loader = poisson_loader(X, y, 64, seed=0)
    model, opt, total = make_mlp_training(X.shape[1], MLP_PARAMS, loader)
    info = fit_mlp(model, opt, loader, total)
    assert info["steps"] == total
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_cosine_schedule_works_through_opacus(data):
    # The DP model in stage 2 wraps model, optimizer, and loader; the schedule must still
    # drive the learning rate the wrapped optimizer actually uses.
    X, y = data
    torch.manual_seed(0)
    loader = poisson_loader(X, y, 64, seed=0)
    model, opt, total = make_mlp_training(X.shape[1], MLP_PARAMS, loader)
    engine = PrivacyEngine(accountant="rdp")
    model, dp_opt, dp_loader = engine.make_private(
        module=model, optimizer=opt, data_loader=loader, noise_multiplier=1.0, max_grad_norm=1.0
    )
    info = fit_mlp(model, dp_opt, dp_loader, total)
    assert info["steps"] == total
    assert dp_opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)
    assert dp_opt.original_optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)
    assert engine.get_epsilon(1e-6) > 0
