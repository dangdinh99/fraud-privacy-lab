"""GPU training path: sampling, reproducibility, and privacy accounting (skipped without CUDA)."""

import numpy as np
import pytest
import torch

from fraudpriv.baseline import DevicePoissonBatches, score_mlp, train_mlp
from fraudpriv.dp_train import train_dp_mlp
from fraudpriv.utils import configure_device

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PARAMS = {
    "hidden": [8],
    "batch_size": 100,
    "lr": 0.01,
    "epochs": 2,
    "target_epsilon": 2.0,
    "delta": 1e-6,
    "max_grad_norm": 0.5,
}


@pytest.fixture(autouse=True)
def deterministic_cuda():
    configure_device("cuda")
    yield
    torch.use_deterministic_algorithms(False)  # do not leak into CPU tests


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(2 * X[:, 0] - 2)))).astype(int)
    return X, y


def test_device_batches_have_poisson_sizes_and_repeat_by_seed():
    X = torch.zeros(10_000, 2, device="cuda")
    y = torch.zeros(10_000, device="cuda")
    sizes = [len(xb) for xb, _ in DevicePoissonBatches(X, y, 0.01, 2000, seed=0)]
    assert np.mean(sizes) == pytest.approx(100, rel=0.02)  # q * n
    assert np.var(sizes) == pytest.approx(100 * 0.99, rel=0.1)  # binomial variance
    again = [len(xb) for xb, _ in DevicePoissonBatches(X, y, 0.01, 2000, seed=0)]
    assert sizes == again


def test_nonprivate_gpu_same_seed_same_scores(data):
    X, y = data
    params = {k: PARAMS[k] for k in ("hidden", "batch_size", "lr", "epochs")}
    a, info = train_mlp(X, y, params, seed=0, device="cuda")
    b, _ = train_mlp(X, y, params, seed=0, device="cuda")
    assert next(a.parameters()).is_cuda
    np.testing.assert_array_equal(score_mlp(a, X), score_mlp(b, X))
    assert info["steps"] == 2 * 20


def test_dp_gpu_reproducible_and_accounted(data):
    X, y = data
    a, info = train_dp_mlp(X, y, PARAMS, seed=0, device="cuda")
    b, _ = train_dp_mlp(X, y, PARAMS, seed=0, device="cuda")
    c, _ = train_dp_mlp(X, y, PARAMS, seed=1, device="cuda")
    np.testing.assert_array_equal(score_mlp(a, X), score_mlp(b, X))
    assert not np.array_equal(score_mlp(a, X), score_mlp(c, X))
    priv = info["privacy"]
    assert priv["device"] == "cuda"
    assert priv["sample_rate"] == pytest.approx(1 / 20)
    assert priv["steps"] == PARAMS["epochs"] * 20
    assert 0.9 * PARAMS["target_epsilon"] < priv["epsilon_spent"] <= PARAMS["target_epsilon"]


def test_dp_gpu_matches_cpu_privacy_settings(data):
    X, y = data
    _, gpu = train_dp_mlp(X, y, PARAMS, seed=0, device="cuda")
    torch.use_deterministic_algorithms(False)
    _, cpu = train_dp_mlp(X, y, PARAMS, seed=0, device="cpu")
    for key in ("noise_multiplier", "sample_rate", "steps", "epsilon_spent"):
        assert gpu["privacy"][key] == pytest.approx(cpu["privacy"][key])
