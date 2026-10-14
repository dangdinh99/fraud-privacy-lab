"""Run an experiment from a YAML config.

    python -m fraudpriv.run --config configs/baseline_lgbm.yaml
    python -m fraudpriv.run --config configs/<final>.yaml --final   # test set

Each run writes ``results/<stage>/<run_id>/`` with the resolved config, environment info,
the fitted preprocessor, and one ``metrics.json`` per grid point (per seed, plus mean and
sample standard deviation across seeds). ``summary.json`` ranks grid points by
``select_score``: per seed, the average share of ceiling over the co-primary budgets in
``select_budgets`` (1% and 2%), then the mean over seeds. Points with a
``target_epsilon`` are ranked within each epsilon.

Grid points come from ``grid`` (Cartesian product over ``model.params``) or from an
explicit ``points`` list. ``--only key=value`` restricts a run to matching points, so
one config can be split across parallel processes (for example one per epsilon).

The test set is used only when the config sets ``eval_split: test`` and ``--final`` is
passed (CLAUDE.md rule 4). Models are always trained on train; early stopping and
calibration always use validation.
"""

import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fraudpriv import baseline, dp_train
from fraudpriv.data import (
    AMOUNT_COL,
    ID_COL,
    TARGET,
    card_segments,
    load_dev_frame,
    load_test_frame,
)
from fraudpriv.features import Preprocessor
from fraudpriv.metrics import evaluate
from fraudpriv.utils import (
    configure_device,
    env_info,
    load_config,
    save_json,
    save_yaml,
    set_seed,
)

MODELS = ("lightgbm", "mlp", "logreg", "dp_mlp", "dp_logreg")
LINEAR = ("logreg", "dp_logreg")  # logistic regression: the MLP code with no hidden layers


def expand_grid(base: dict[str, Any], grid: dict[str, list] | None) -> list[tuple[str, dict]]:
    """Cartesian product of ``grid`` over ``base`` params, as ``(name, params)`` pairs."""
    if not grid:
        return [("default", dict(base))]
    keys = list(grid)
    points = []
    for values in itertools.product(*(grid[k] for k in keys)):
        name = "__".join(f"{k}={v}" for k, v in zip(keys, values, strict=True))
        points.append((name, {**base, **dict(zip(keys, values, strict=True))}))
    return points


def _point_name(point: dict[str, Any]) -> str:
    return "__".join(f"{k}={v}" for k, v in point.items())


def expand_points(
    base: dict[str, Any], grid: dict[str, list] | None, points: list[dict] | None
) -> list[tuple[str, dict]]:
    """Grid points from either ``grid`` or an explicit ``points`` list (not both)."""
    if grid and points:
        raise ValueError("config sets both grid and points")
    if points:
        return [(_point_name(pt), {**base, **pt}) for pt in points]
    return expand_grid(base, grid)


def parse_only(only: list[str] | None) -> dict[str, Any]:
    """``["target_epsilon=1", ...]`` -> ``{"target_epsilon": 1}`` (values parsed as YAML)."""
    import yaml

    out = {}
    for item in only or []:
        key, _, value = item.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def summarize_seeds(per_seed: dict[int, dict]) -> dict[str, dict[str, float | None]]:
    """Mean and sample standard deviation (ddof=1) of every numeric metric across seeds.

    With a single seed the standard deviation is ``None`` (undefined), not NaN.
    """
    flat = pd.json_normalize(list(per_seed.values()), sep=".")
    numeric = flat.select_dtypes(include="number").drop(columns=["seed"], errors="ignore")
    multi = len(numeric) > 1
    return {
        col: {
            "mean": float(numeric[col].mean()),
            "std": float(numeric[col].std(ddof=1)) if multi else None,
        }
        for col in numeric.columns
    }


def _train_and_score(
    model_type: str,
    params: dict[str, Any],
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_ev: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validation scores, evaluation scores, and training info."""
    if model_type == "lightgbm":
        booster, info = baseline.train_lightgbm(X_tr, y_tr, X_va, y_va, params, seed)
        return baseline.score_lightgbm(booster, X_va), baseline.score_lightgbm(booster, X_ev), info
    if model_type in LINEAR and params["hidden"]:
        raise ValueError(f"{model_type} requires hidden: []")
    train = dp_train.train_dp_mlp if model_type.startswith("dp_") else baseline.train_mlp
    model, info = train(X_tr, y_tr, params, seed, device)
    return baseline.score_mlp(model, X_va), baseline.score_mlp(model, X_ev), info


def run(
    config_path: str | Path,
    final: bool = False,
    results_root: str | Path = "results",
    only: dict[str, Any] | None = None,
) -> Path:
    """Run every grid point (matching ``only``) and seed in a config; returns the run directory."""
    cfg = load_config(config_path)
    eval_split = cfg.get("eval_split", "val")
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
    if eval_split == "test" and not final:
        raise ValueError("eval_split: test requires --final (CLAUDE.md rule 4)")
    if final and eval_split != "test":
        raise ValueError("--final requires eval_split: test in the config")
    model_type = cfg["model"]["type"]
    if model_type not in MODELS:
        raise ValueError(f"unknown model type {model_type!r}")
    points = expand_points(cfg["model"]["params"], cfg.get("grid"), cfg.get("points"))
    only = only or {}
    points = [(n, p) for n, p in points if all(p.get(k) == v for k, v in only.items())]
    if not points:
        raise ValueError(f"no grid points match {only}")
    if "torch_threads" in cfg:
        torch.set_num_threads(cfg["torch_threads"])
    device = cfg.get("device", "cpu")  # LightGBM always runs on CPU
    if model_type != "lightgbm":
        configure_device(device)

    features_cfg = load_config(cfg["features_config"])
    dev = load_dev_frame(cfg["data_dir"], cfg["split_dir"])
    train = dev[dev["split"] == "train"].reset_index(drop=True)
    val = dev[dev["split"] == "val"].reset_index(drop=True)
    ev = load_test_frame(cfg["data_dir"], cfg["split_dir"]) if final else val

    pre = Preprocessor(features_cfg).fit(train)
    X_tr, X_va = pre.transform(train), pre.transform(val)
    X_ev = pre.transform(ev) if final else X_va
    y_tr, y_va = train[TARGET].to_numpy(), val[TARGET].to_numpy()
    y_ev, ids_ev, amt_ev = ev[TARGET].to_numpy(), ev[ID_COL].to_numpy(), ev[AMOUNT_COL].to_numpy()
    segments = card_segments(ev, train)

    tag = "".join(f"_{k}={v}" for k, v in only.items())
    run_id = f"{cfg['name']}{tag}_{datetime.now():%Y%m%d-%H%M%S}" + ("_final" if final else "")
    run_dir = Path(results_root) / cfg["stage"] / run_id
    save_yaml(
        {"config": cfg, "features": features_cfg, "final": final, "only": only},
        run_dir / "config.yaml",
    )
    save_json(env_info(), run_dir / "env.json")
    pre.save(run_dir / "preprocessor.json")

    budgets = tuple(cfg["budgets"])
    select_budgets = tuple(cfg["select_budgets"])
    if not set(select_budgets) <= set(budgets):
        raise ValueError("select_budgets must be a subset of budgets")
    ranking = []
    for name, params in points:
        per_seed = {}
        for seed in cfg["seeds"]:
            set_seed(seed)
            s_va, s_ev, info = _train_and_score(
                model_type, params, seed, X_tr, y_tr, X_va, y_va, X_ev, device
            )
            if device != "cpu":
                torch.cuda.empty_cache()  # Poisson batch sizes vary and fragment GPU memory
            calibrator = baseline.PlattCalibrator().fit(s_va, y_va)
            metrics = evaluate(
                y_ev,
                s_ev,
                ids_ev,
                amt_ev,
                budgets,
                cfg["ece_bins"],
                segments,
                probs=calibrator.transform(s_ev),
            )
            shares = [metrics[f"budget_{b}"]["recall_share_of_ceiling"] for b in select_budgets]
            metrics.update(
                seed=seed,
                select_score=float(np.mean(shares)),
                train_info=info,
                platt={"a": calibrator.a, "b": calibrator.b},
            )
            per_seed[seed] = metrics
            print(f"[{name}] seed {seed}: select_score = {metrics['select_score']:.4f}")
        summary = summarize_seeds(per_seed)
        save_json(
            {
                "eval_split": eval_split,
                "final": final,
                "ece_in_sample": eval_split == "val",  # Platt is fit on validation
                "n_inputs": len(pre.feature_names),
                "device": "cpu" if model_type == "lightgbm" else device,
                "params": params,
                "summary": summary,
                "per_seed": {str(k): v for k, v in per_seed.items()},
            },
            run_dir / name / "metrics.json",
        )
        ranking.append(
            {
                "grid_point": name,
                "target_epsilon": params.get("target_epsilon"),
                **summary["select_score"],
            }
        )

    # Rank within each epsilon (non-DP points have target_epsilon None).
    ranking.sort(key=lambda r: (str(r["target_epsilon"]), -r["mean"]))
    save_json(
        {
            "select_metric": "mean share of ceiling over select_budgets",
            "select_budgets": list(select_budgets),
            "eval_split": eval_split,
            "ranking": ranking,
        },
        run_dir / "summary.json",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--final", action="store_true", help="evaluate on the test set")
    parser.add_argument(
        "--only", action="append", metavar="KEY=VALUE", help="run only matching grid points"
    )
    args = parser.parse_args()
    print(f"results: {run(args.config, final=args.final, only=parse_only(args.only))}")


if __name__ == "__main__":
    main()
