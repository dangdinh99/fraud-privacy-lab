"""Non-private baselines: LightGBM, the MLP shared with the DP model, and Platt calibration.

The MLP and its training loop are reused by the DP-SGD model in stage 2, so the two
differ only in gradient clipping and noise (CLAUDE.md rule 5):

- Same architecture (hidden ReLU layers, no BatchNorm, so Opacus can use it).
- Same Poisson sampling of batches, which DP-SGD requires: Opacus ``DPDataLoader`` on CPU,
  or :class:`DevicePoissonBatches` on a GPU (the same sampling at the same rate, drawn on
  the device to avoid per-row Python batching).
- Same optimizer (Adam), unweighted binary cross-entropy, and a fixed number of epochs
  with cosine decay of the learning rate to zero. No early stopping: under DP the
  privacy cost depends on the planned number of steps.

Scores returned by the ``score_*`` functions are raw logits (ranking); probabilities come
from :class:`PlattCalibrator`, fit on validation.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- calibration --------------------------------------------------------------------


class PlattCalibrator:
    """Platt scaling: ``p = sigmoid(a * score + b)`` with ``a > 0``.

    Strictly increasing, so the ranking of scores (and every alert-budget metric) is
    unchanged. Fit on validation scores only.
    """

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)  # unregularized
        lr.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if self.a <= 0:
            raise ValueError(f"Platt slope {self.a:.4g} <= 0: scores are not positively ranked")
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-z))


# --- LightGBM -------------------------------------------------------------------------


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train with early stopping on validation average precision.

    ``params`` holds LightGBM parameters plus ``max_rounds`` and ``early_stopping_rounds``.
    Deterministic mode is always on, so a seed reproduces the same model.
    """
    params = dict(params)
    max_rounds = params.pop("max_rounds")
    patience = params.pop("early_stopping_rounds")
    lgb_params = {
        "objective": "binary",
        "metric": "average_precision",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        **params,
    }
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)
    booster = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return booster, {"best_iteration": booster.best_iteration}


def score_lightgbm(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Raw margin scores (log-odds) at the best iteration."""
    return booster.predict(X, raw_score=True, num_iteration=booster.best_iteration)


# --- MLP ------------------------------------------------------------------------------


class MLP(nn.Module):
    """Feed-forward network: ``n_inputs -> hidden[0] -> ... -> 1`` with ReLU, no BatchNorm."""

    def __init__(self, n_inputs: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def base_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Plain loader over (X, y); Opacus turns it into a Poisson loader for DP training."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size)


def poisson_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    """Batches by Poisson sampling with expected size ``batch_size`` (as DP-SGD requires)."""
    generator = torch.Generator().manual_seed(seed)
    return DPDataLoader.from_data_loader(base_loader(X, y, batch_size), generator=generator)


class DevicePoissonBatches:
    """Poisson sampling on the data's device: each row is included independently with
    probability ``sample_rate`` at every step; ``len()`` steps make one epoch.

    Matches Opacus's ``DPDataLoader`` (same sampling, same rate when ``sample_rate`` and
    ``steps_per_epoch`` come from it), so the privacy accounting is unchanged. Batches can
    be empty; :func:`fit_mlp` handles that.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_rate: float,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        self.X, self.y = X, y
        self.sample_rate = sample_rate
        self.steps_per_epoch = steps_per_epoch
        self.generator = torch.Generator(device=X.device).manual_seed(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            u = torch.rand(len(self.X), device=self.X.device, generator=self.generator)
            mask = u < self.sample_rate
            yield self.X[mask], self.y[mask]


def device_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int, seed: int, device: torch.device
) -> DevicePoissonBatches:
    """:class:`DevicePoissonBatches` at Opacus's rate for ``batch_size`` (1 / number of batches)."""
    steps = len(base_loader(X, y, batch_size))
    Xd = torch.from_numpy(X).to(device)
    yd = torch.from_numpy(y.astype(np.float32)).to(device)
    return DevicePoissonBatches(Xd, yd, 1.0 / steps, steps, seed)


def make_mlp_training(
    n_inputs: int,
    params: dict[str, Any],
    loader: Any,
    device: str | torch.device = "cpu",
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[MLP, torch.optim.Optimizer, int]:
    """Model (on ``device``), Adam optimizer, and total step count for a fixed-epoch run.

    ``init_state`` warm-starts the model from earlier weights (stage 3 fine-tuning).
    """
    model = MLP(n_inputs, params["hidden"])
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 0.0)
    )
    total_steps = params["epochs"] * len(loader)
    return model, optimizer, total_steps


def fit_mlp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    total_steps: int,
) -> dict[str, Any]:
    """Train for exactly ``total_steps`` steps with cosine decay of the learning rate to 0.

    Shared by the non-private and DP models; the DP model passes an Opacus-wrapped model,
    optimizer, and loader. Returns training info (steps and mean loss of the last epoch).
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    step, losses = 0, []
    while step < total_steps:
        losses = []
        for xb, yb in loader:
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            if len(xb) == 0:
                # Poisson sampling can produce an empty batch (probability (1 - q)^n, negligible
                # at our sizes). Under DP the step must still happen: Opacus adds noise and the
                # accountant counts it, so run a zero-loss backward to create per-sample grads.
                if isinstance(optimizer, DPOptimizer):
                    model(xb).sum().mul(0.0).backward()
                optimizer.step()
                scheduler.step()
                step += 1
                continue
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
    return {"steps": step, "last_epoch_loss": float(np.mean(losses)) if losses else None}


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: str = "cpu",
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[MLP, dict[str, Any]]:
    """Non-private MLP: the shared training loop without clipping or noise."""
    torch.manual_seed(seed)
    if device == "cpu":
        loader = poisson_loader(X_train, y_train, params["batch_size"], seed)
    else:
        loader = device_batches(X_train, y_train, params["batch_size"], seed, torch.device(device))
    model, optimizer, total_steps = make_mlp_training(
        X_train.shape[1], params, loader, device, init_state
    )
    info = fit_mlp(model, optimizer, loader, total_steps)
    return model, info


@torch.no_grad()
def score_mlp(model: nn.Module, X: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    """Logit scores in evaluation mode, computed on the model's device."""
    model.eval()
    device = next(model.parameters()).device
    out = [
        model(torch.from_numpy(X[i : i + batch_size]).to(device)).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(out).numpy().astype(np.float64)
