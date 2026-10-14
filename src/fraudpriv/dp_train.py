"""DP-SGD training (Opacus) for the MLP and logistic regression.

Reuses the stage 1 model and training loop (``baseline.MLP``, ``baseline.fit_mlp``), so the
DP and non-private models differ only in per-example gradient clipping and noise
(CLAUDE.md rule 5). Logistic regression is the same code with no hidden layers.

Privacy accounting:

- Privacy unit: one transaction. delta must be below 1 / n_train (asserted).
- Poisson sampling, so the accountant's guarantee holds: Opacus's loader on CPU, or
  ``baseline.DevicePoissonBatches`` on a GPU at exactly the rate Opacus computed.
- Noise multiplier calibrated by Opacus to reach ``target_epsilon`` over the planned
  number of steps, with the RDP accountant.
- The reported epsilon is what the accountant spent after training, never the target
  alone. A run fails if it exceeds the target.
- Secure RNG is off (standard for research; it is much slower).
- Diagnostic only: ``noise_multiplier: 0`` (instead of ``target_epsilon``) trains with
  clipping but no noise, to split the privacy cost into clipping and noise. Such runs are
  logged with ``private: False`` and no epsilon: they carry no privacy guarantee.
- Planned composition (stage 3): when several runs reuse the same transactions, their
  privacy losses add up. :func:`calibrate_composed_noise` finds one noise multiplier for
  all planned runs so that :func:`composed_epsilon` of the whole plan stays within a
  total budget; each run then passes ``noise_multiplier`` together with
  ``composed_epsilon_total``. The caller reports the composed total, not the per-run value.
- Preprocessing statistics and hyperparameter tuning are not charged to the privacy
  budget; the guarantee covers model training only.
"""

from typing import Any

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis import rdp as rdp_analysis
from opacus.validators import ModuleValidator

from fraudpriv.baseline import (
    MLP,
    DevicePoissonBatches,
    base_loader,
    fit_mlp,
    make_mlp_training,
)

ACCOUNTANT = "rdp"


def check_delta(delta: float, n_train: int) -> None:
    """Require delta < 1 / n_train (privacy unit: one transaction)."""
    if not delta < 1.0 / n_train:
        raise ValueError(f"delta={delta} must be < 1/n_train = {1.0 / n_train:.3g}")


def composed_epsilon(runs: list[tuple[float, float, int]], delta: float) -> float:
    """Epsilon (RDP) of runs over the same data, each ``(noise_multiplier, sample_rate, steps)``.

    RDP adds across runs; this is the worst case for a transaction used in every run.
    """
    orders = RDPAccountant.DEFAULT_ALPHAS
    total = sum(
        np.asarray(rdp_analysis.compute_rdp(q=q, noise_multiplier=s, steps=k, orders=orders))
        for s, q, k in runs
    )
    eps, _ = rdp_analysis.get_privacy_spent(orders=orders, rdp=total, delta=delta)
    return float(eps)


def calibrate_composed_noise(
    plan: list[tuple[float, int]], epsilon_total: float, delta: float, tolerance: float = 0.005
) -> float:
    """Smallest common noise multiplier (within ``tolerance``) such that all planned runs,
    given as ``(sample_rate, steps)``, compose to at most ``epsilon_total``."""

    def eps(sigma: float) -> float:
        return composed_epsilon([(sigma, q, k) for q, k in plan], delta)

    lo, hi = 0.0, 1.0
    while eps(hi) > epsilon_total:
        lo, hi = hi, hi * 2
    while (hi - lo) / hi > tolerance:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if eps(mid) > epsilon_total else (lo, mid)
    return hi


def train_dp_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: str = "cpu",
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[MLP, dict[str, Any]]:
    """Train the MLP (or logistic regression, ``hidden: []``) with DP-SGD.

    ``params`` adds ``target_epsilon``, ``delta``, and ``max_grad_norm`` to the stage 1 MLP
    parameters (``hidden``, ``batch_size``, ``lr``, ``epochs``). Returns the trained module
    and info including every privacy setting and the epsilon actually spent.

    For the clipping-only diagnostic, pass ``noise_multiplier: 0`` instead of
    ``target_epsilon``; the run is then marked ``private: False`` with no epsilon. For
    planned composition, pass a positive ``noise_multiplier`` from
    :func:`calibrate_composed_noise` together with ``composed_epsilon_total``.
    ``init_state`` warm-starts from earlier weights (a DP model's weights can be reused
    without extra privacy cost).
    """
    fixed = "noise_multiplier" in params
    if fixed == ("target_epsilon" in params):
        raise ValueError("set exactly one of target_epsilon or noise_multiplier")
    sigma = params.get("noise_multiplier")
    if fixed and sigma != 0 and "composed_epsilon_total" not in params:
        raise ValueError(
            "a nonzero noise_multiplier is only for planned composition "
            "(set composed_epsilon_total); use 0 for the clipping-only diagnostic"
        )
    clip_only = fixed and sigma == 0
    n = len(X_train)
    delta = params["delta"]
    check_delta(delta, n)
    torch.manual_seed(seed)  # also drives Opacus's sampling and noise (secure RNG off)

    loader = base_loader(X_train, y_train, params["batch_size"])
    model, optimizer, _ = make_mlp_training(X_train.shape[1], params, loader, device, init_state)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise ValueError(f"model is not Opacus-compatible: {errors}")

    engine = PrivacyEngine(accountant=ACCOUNTANT, secure_mode=False)
    if fixed:
        model, optimizer, loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=float(sigma),
            max_grad_norm=params["max_grad_norm"],
            poisson_sampling=True,
        )
    else:
        model, optimizer, loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=params["target_epsilon"],
            target_delta=delta,
            epochs=params["epochs"],
            max_grad_norm=params["max_grad_norm"],
            poisson_sampling=True,
        )
    sample_rate = float(loader.sample_rate)
    if device != "cpu":  # same Poisson sampling at Opacus's rate, drawn on the device
        loader = DevicePoissonBatches(
            torch.from_numpy(X_train).to(device),
            torch.from_numpy(y_train.astype(np.float32)).to(device),
            sample_rate,
            len(loader),
            seed,
        )
    total_steps = params["epochs"] * len(loader)
    info = fit_mlp(model, optimizer, loader, total_steps)

    spent = None  # no noise: no privacy guarantee, so no epsilon
    if not clip_only:
        spent = float(engine.get_epsilon(delta))  # this run alone
    if not fixed:
        if spent > params["target_epsilon"] + 1e-6:
            raise RuntimeError(
                f"spent epsilon {spent:.4f} exceeds target {params['target_epsilon']}"
            )
    info["privacy"] = {
        "private": not clip_only,
        "accounting": "composed" if fixed and not clip_only else "single_run",
        "composed_epsilon_total": params.get("composed_epsilon_total"),
        "accountant": ACCOUNTANT,
        "target_epsilon": params.get("target_epsilon"),
        "epsilon_spent": spent,
        "delta": delta,
        "n_train": n,
        "noise_multiplier": float(optimizer.noise_multiplier),
        "max_grad_norm": params["max_grad_norm"],
        "sample_rate": sample_rate,
        "expected_batch_size": sample_rate * n,
        "epochs": params["epochs"],
        "steps": info["steps"],
        "poisson_sampling": True,
        "secure_mode": False,
        "device": device,
    }
    return model._module, info
