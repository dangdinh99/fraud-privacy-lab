"""Tests for fraudpriv.dp_train: privacy accounting, logging, and Opacus behavior."""

import numpy as np
import pytest
import torch
from opacus import PrivacyEngine

from fraudpriv.baseline import base_loader, fit_mlp, make_mlp_training, score_mlp
from fraudpriv.dp_train import check_delta, train_dp_mlp

PARAMS = {
    "hidden": [8],
    "batch_size": 100,
    "lr": 0.01,
    "epochs": 2,
    "target_epsilon": 2.0,
    "delta": 1e-6,
    "max_grad_norm": 0.5,
}
REQUIRED = {
    "private",
    "accountant",
    "target_epsilon",
    "epsilon_spent",
    "delta",
    "n_train",
    "noise_multiplier",
    "max_grad_norm",
    "sample_rate",
    "expected_batch_size",
    "epochs",
    "steps",
    "poisson_sampling",
    "secure_mode",
}


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(2 * X[:, 0] - 2)))).astype(int)
    return X, y


def test_check_delta():
    check_delta(1e-6, 354_324)
    with pytest.raises(ValueError):
        check_delta(1e-6, 1_000_000)  # delta == 1/n is not allowed
    with pytest.raises(ValueError):
        check_delta(1e-3, 2000)


@pytest.mark.parametrize("hidden", [[8], []])
def test_privacy_is_logged_and_within_target(data, hidden):
    X, y = data
    _, info = train_dp_mlp(X, y, {**PARAMS, "hidden": hidden}, seed=0)
    priv = info["privacy"]
    assert REQUIRED <= priv.keys()
    assert priv["accountant"] == "rdp"
    assert 0 < priv["epsilon_spent"] <= PARAMS["target_epsilon"]
    assert priv["epsilon_spent"] > 0.9 * PARAMS["target_epsilon"]  # calibrated, not loose
    assert priv["poisson_sampling"] is True
    assert priv["sample_rate"] == pytest.approx(1 / 20)  # 2000 rows / batch 100
    assert priv["steps"] == info["steps"] == PARAMS["epochs"] * 20
    assert priv["noise_multiplier"] > 0


def test_smaller_epsilon_means_more_noise(data):
    X, y = data
    _, tight = train_dp_mlp(X, y, {**PARAMS, "target_epsilon": 0.5}, seed=0)
    _, loose = train_dp_mlp(X, y, {**PARAMS, "target_epsilon": 8.0}, seed=0)
    assert tight["privacy"]["noise_multiplier"] > loose["privacy"]["noise_multiplier"]


def test_same_seed_same_model(data):
    X, y = data
    a, _ = train_dp_mlp(X, y, PARAMS, seed=0)
    b, _ = train_dp_mlp(X, y, PARAMS, seed=0)
    c, _ = train_dp_mlp(X, y, PARAMS, seed=1)
    np.testing.assert_array_equal(score_mlp(a, X), score_mlp(b, X))
    assert not np.array_equal(score_mlp(a, X), score_mlp(c, X))


def test_delta_checked_against_training_size(data):
    X, y = data
    with pytest.raises(ValueError):
        train_dp_mlp(X, y, {**PARAMS, "delta": 1e-3}, seed=0)


def test_empty_batch_still_takes_a_counted_noisy_step(data):
    X, y = data
    torch.manual_seed(0)
    loader = base_loader(X, y, 100)
    model, opt, _ = make_mlp_training(X.shape[1], PARAMS, loader)
    engine = PrivacyEngine(accountant="rdp")
    model, opt, _ = engine.make_private(
        module=model, optimizer=opt, data_loader=loader, noise_multiplier=1.0, max_grad_norm=1.0
    )
    before = [p.detach().clone() for p in model.parameters()]
    empty = (torch.zeros(0, X.shape[1]), torch.zeros(0))
    fit_mlp(model, opt, [empty], total_steps=1)
    assert sum(steps for _, _, steps in engine.accountant.history) == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_private_runs_are_marked_private(data):
    X, y = data
    _, info = train_dp_mlp(X, y, PARAMS, seed=0)
    assert info["privacy"]["private"] is True


def test_clipping_only_diagnostic_has_no_privacy_claim(data):
    X, y = data
    params = {k: v for k, v in PARAMS.items() if k != "target_epsilon"}
    _, info = train_dp_mlp(X, y, {**params, "noise_multiplier": 0}, seed=0)
    priv = info["privacy"]
    assert priv["private"] is False
    assert priv["epsilon_spent"] is None and priv["target_epsilon"] is None
    assert priv["noise_multiplier"] == 0.0 and priv["max_grad_norm"] == PARAMS["max_grad_norm"]


def test_noise_settings_are_exclusive_and_zero_only(data):
    X, y = data
    with pytest.raises(ValueError):
        train_dp_mlp(X, y, {**PARAMS, "noise_multiplier": 0}, seed=0)  # both set
    params = {k: v for k, v in PARAMS.items() if k != "target_epsilon"}
    with pytest.raises(ValueError):
        train_dp_mlp(X, y, params, seed=0)  # neither set
    with pytest.raises(ValueError):
        train_dp_mlp(X, y, {**params, "noise_multiplier": 0.5}, seed=0)  # nonzero fixed noise


# --- planned composition (stage 3) ------------------------------------------------------


def test_composed_epsilon_matches_opacus_accountant():
    from opacus.accountants import RDPAccountant

    from fraudpriv.dp_train import composed_epsilon

    runs = [(1.1, 0.003, 3000), (0.9, 0.0025, 4000), (1.3, 0.002, 5000)]
    acc = RDPAccountant()
    acc.history = list(runs)
    assert composed_epsilon(runs, 1e-6) == pytest.approx(acc.get_epsilon(1e-6), rel=1e-9)


def test_calibrated_noise_meets_total_budget_and_grows_with_runs():
    from fraudpriv.dp_train import calibrate_composed_noise, composed_epsilon

    one = [(0.003, 3000)]
    four = [(0.003, 3000), (0.0028, 3200), (0.0026, 3400), (0.0024, 3600)]
    s1 = calibrate_composed_noise(one, 2.0, 1e-6)
    s4 = calibrate_composed_noise(four, 2.0, 1e-6)
    e4 = composed_epsilon([(s4, q, k) for q, k in four], 1e-6)
    assert 0.97 * 2.0 < e4 <= 2.0
    assert s4 > s1


def test_calibration_single_run_close_to_opacus():
    from opacus.accountants.utils import get_noise_multiplier

    from fraudpriv.dp_train import calibrate_composed_noise

    ours = calibrate_composed_noise([(0.003, 3000)], 1.0, 1e-6)
    theirs = get_noise_multiplier(
        target_epsilon=1.0, target_delta=1e-6, sample_rate=0.003, steps=3000, accountant="rdp"
    )
    assert ours == pytest.approx(theirs, rel=0.02)


def test_fixed_noise_requires_declared_composition(data):
    X, y = data
    params = {k: v for k, v in PARAMS.items() if k != "target_epsilon"}
    _, info = train_dp_mlp(
        X, y, {**params, "noise_multiplier": 1.5, "composed_epsilon_total": 4.0}, seed=0
    )
    priv = info["privacy"]
    assert priv["private"] is True and priv["accounting"] == "composed"
    assert priv["composed_epsilon_total"] == 4.0 and priv["noise_multiplier"] == 1.5
    assert priv["epsilon_spent"] > 0  # this run alone


def test_warm_start_loads_previous_weights(data):
    from fraudpriv.baseline import base_loader, make_mlp_training

    X, y = data
    first, _ = train_dp_mlp(X, y, PARAMS, seed=0)
    state = {k: v.detach().cpu().clone() for k, v in first.state_dict().items()}
    model, _, _ = make_mlp_training(X.shape[1], PARAMS, base_loader(X, y, 100), "cpu", state)
    for k, v in model.state_dict().items():
        assert torch.equal(v, state[k])
    tuned, info = train_dp_mlp(X, y, PARAMS, seed=1, init_state=state)
    assert info["privacy"]["private"] is True
    assert not torch.equal(tuned.state_dict()["net.0.weight"], state["net.0.weight"])
