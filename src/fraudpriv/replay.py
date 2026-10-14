"""Stage 3: time-ordered replay with retraining schedules under a fixed total privacy budget.

    python -m fraudpriv.replay --config configs/replay_dev.yaml            # validation
    python -m fraudpriv.replay --config configs/replay_final.yaml --final  # test, once
    python -m fraudpriv.replay --config <same config> [--final] --resume <run dir>

``--resume`` continues an interrupted run in its own folder: the config and ``--final``
must match the saved ones, finished experiment/schedule pairs are skipped, and the summary
is rebuilt from every saved result. Seeds are deterministic, so resumed results are the
same as an uninterrupted run would give.

The replay period is split into weeks. Each week is scored by the most recent model, and
models are retrained on a schedule (never, or every 3, 2, or 1 weeks). A retrain at time t
only uses labels of transactions older than ``t - label_delay`` (fraud labels arrive late).

Two strategies for spending a total privacy budget ``epsilon_total``:

- ``expanding``: every run trains from scratch on all labeled data so far. Runs reuse the
  same transactions, so their privacy losses compose: one noise multiplier is calibrated
  for the whole plan so that the composed epsilon stays within ``epsilon_total``.
- ``disjoint``: each retrain fine-tunes the previous model (warm start, which costs no
  privacy) on only the data labeled since the previous run. Every transaction is used in
  exactly one run, so each run gets the full ``epsilon_total`` (parallel composition).
  Warm-start runs train for ``finetune_epochs_fraction`` of the setting's epochs (at
  least 1); the first run trains in full. The fraction is chosen on validation.

Each run uses the stage 2 setting selected for the epsilon closest (log scale) to that
run's share of the budget: ``epsilon_total / number of runs`` for ``expanding``,
``epsilon_total`` for ``disjoint``. Preprocessing is the stage 1 fit on the original
training period, fixed for every run. Only ranking metrics are reported (no calibration).
Drift (PSI) is descriptive and does not trigger retraining.
"""

import argparse
import gc
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fraudpriv import baseline, dp_train
from fraudpriv.data import AMOUNT_COL, ID_COL, TARGET, TIME_COL, load_dev_frame, load_test_frame
from fraudpriv.drift import psi
from fraudpriv.features import Preprocessor
from fraudpriv.metrics import aggregate_windows, budget_metrics
from fraudpriv.run import summarize_seeds
from fraudpriv.utils import configure_device, env_info, load_config, save_json, save_yaml

SECONDS_PER_DAY = 86400
PRIVATE = ("dp_mlp", "dp_logreg")
MODELS = PRIVATE + ("mlp", "logreg")
STRATEGIES = ("expanding", "disjoint")


def week_windows(t_start: int, t_end: int, days: int) -> list[tuple[int, int]]:
    """Consecutive ``[start, end)`` windows of ``days`` days; the last one may be shorter."""
    width = days * SECONDS_PER_DAY
    starts = range(int(t_start), int(t_end), width)
    return [(s, min(s + width, int(t_end))) for s in starts]


def retrain_weeks(n_weeks: int, every: int | None) -> list[int]:
    """Weeks at whose start a model is trained: week 0, then every ``every`` weeks."""
    return [0] if every is None else list(range(0, n_weeks, every))


def training_windows(
    run_times: list[int], delay: int, strategy: str
) -> list[tuple[int | None, int]]:
    """Label window ``[lo, hi)`` of transaction times for each run (``lo=None``: no lower bound).

    A run at time t uses only transactions older than ``t - delay``. ``expanding`` uses all
    of them; ``disjoint`` uses only those labeled since the previous run.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    out: list[tuple[int | None, int]] = []
    for i, t in enumerate(run_times):
        lo = None if strategy == "expanding" or i == 0 else run_times[i - 1] - delay
        out.append((lo, t - delay))
    return out


def window_mask(times: np.ndarray, lo: int | None, hi: int) -> np.ndarray:
    """Rows with ``lo <= time < hi``."""
    mask = times < hi
    return mask if lo is None else mask & (times >= lo)


def nearest_setting(settings: dict[float, dict], epsilon: float) -> tuple[float, dict]:
    """Stage 2 setting whose epsilon is closest to ``epsilon`` on a log scale."""
    key = min(settings, key=lambda e: abs(math.log(float(e)) - math.log(epsilon)))
    return float(key), dict(settings[key])


def _plan_entry(n: int, batch_size: int, epochs: int) -> tuple[float, int]:
    """Opacus's sample rate and step count for a run on ``n`` rows."""
    batches = math.ceil(n / batch_size)
    return 1.0 / batches, epochs * batches


def replay_one(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    weeks: list[np.ndarray],
    week_bounds: list[tuple[int, int]],
    ids: np.ndarray,
    amounts: np.ndarray,
    exp: dict[str, Any],
    settings: dict[str, Any],
    strategy: str,
    every: int | None,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """One replay: train on the schedule, score every week, return per-week and pooled metrics."""
    model_type, device = exp["model"], cfg["device"]
    delay = cfg["label_delay_days"] * SECONDS_PER_DAY
    starts = retrain_weeks(len(weeks), every)
    run_times = [week_bounds[w][0] for w in starts]
    windows = training_windows(run_times, delay, strategy)
    masks = [window_mask(times, lo, hi) for lo, hi in windows]
    n_runs = len(starts)

    private = model_type in PRIVATE
    # Diagnostic: same settings as the DP experiment with this epsilon_total, but zero noise
    # (clipping only). No privacy guarantee; logged as not private.
    clip_only = private and exp.get("clip_only", False)
    base = {"hidden": cfg["hidden"][model_type], "weight_decay": 0.0}
    if private:
        eps_total = float(exp["epsilon_total"])
        share = eps_total / n_runs if strategy == "expanding" else eps_total
        setting_eps, setting = nearest_setting(settings[model_type], share)
        params = {**base, **setting, "delta": cfg["delta"]}
        if clip_only:
            params.update(noise_multiplier=0)
        elif strategy == "expanding":
            plan = [
                _plan_entry(int(m.sum()), params["batch_size"], params["epochs"]) for m in masks
            ]
            sigma = dp_train.calibrate_composed_noise(plan, eps_total, cfg["delta"])
            params.update(noise_multiplier=sigma, composed_epsilon_total=eps_total)
        else:
            params.update(target_epsilon=eps_total)
    else:
        setting_eps, params = None, {**base, **settings[model_type]}

    scores = [None] * len(weeks)
    runs, state = [], None
    for i, (mask, (lo, hi)) in enumerate(zip(masks, windows, strict=True)):
        init = state if strategy == "disjoint" else None
        run_params = params
        if init is not None:  # gentler warm-start fine-tuning on the new window only
            frac = cfg.get("finetune_epochs_fraction", 1.0)
            run_params = {**params, "epochs": max(1, round(params["epochs"] * frac))}
        train = dp_train.train_dp_mlp if private else baseline.train_mlp
        model, info = train(X[mask], y[mask], run_params, seed * 100 + i, device, init)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        end = starts[i + 1] if i + 1 < n_runs else len(weeks)
        for w in range(starts[i], end):
            scores[w] = baseline.score_mlp(model, X[weeks[w]])
        if every is None and i == 0:  # score drift for the never-retrained model
            ref_rows = np.flatnonzero(mask)
            ref_rows = np.random.default_rng(seed).choice(
                ref_rows, size=min(len(ref_rows), cfg["drift_sample"]), replace=False
            )
            ref_scores = baseline.score_mlp(model, X[ref_rows])
            score_psi = [psi(ref_scores, scores[w], cfg["psi_bins"]) for w in range(len(weeks))]
        runs.append(
            {
                "week": starts[i],
                "label_window": [lo, hi],
                "n_train": int(mask.sum()),
                "max_train_time": int(times[mask].max()),
                "warm_start": init is not None,
                "epochs": run_params["epochs"],
                "train_info": info,
            }
        )
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "model": model_type,
        "strategy": strategy if every is not None else "none",
        "every_weeks": every,
        "n_runs": n_runs,
        "setting_epsilon": setting_eps,
        "params": {k: v for k, v in params.items() if k != "composed_epsilon_total"},
        "runs": runs,
    }
    if clip_only:
        result["privacy"] = {"private": False, "note": "clipping only, no noise: no guarantee"}
    elif private:
        priv = [r["train_info"]["privacy"] for r in runs]
        if strategy == "expanding":
            spent = dp_train.composed_epsilon(
                [(p["noise_multiplier"], p["sample_rate"], p["steps"]) for p in priv], cfg["delta"]
            )
        else:
            spent = max(p["epsilon_spent"] for p in priv)  # parallel composition
        if spent > eps_total + 1e-6:
            raise RuntimeError(f"total epsilon {spent:.4f} exceeds budget {eps_total}")
        result["privacy"] = {
            "epsilon_total": eps_total,
            "epsilon_spent_total": spent,
            "composition": "sequential (RDP)" if strategy == "expanding" else "parallel",
            "delta": cfg["delta"],
        }
    if every is None:
        result["score_psi_by_week"] = score_psi

    per_week = []
    for w, rows in enumerate(weeks):
        per_week.append(
            {
                b: budget_metrics(y[rows], scores[w], ids[rows], amounts[rows], b)
                for b in cfg["budgets"]
            }
        )
    result["per_week"] = [{f"budget_{b}": m[b] for b in cfg["budgets"]} for m in per_week]
    pooled = {f"budget_{b}": aggregate_windows([m[b] for m in per_week]) for b in cfg["budgets"]}
    result["pooled"] = pooled
    result["select_score"] = float(
        np.mean([pooled[f"budget_{b}"]["recall_share_of_ceiling"] for b in cfg["budgets"]])
    )
    return result


def _load(cfg: dict[str, Any], final: bool) -> tuple[pd.DataFrame, str]:
    """Frame with train and later rows in time order, and the name of the replay split."""
    period = cfg["replay_split"]
    if period not in ("val", "test"):
        raise ValueError("replay_split must be 'val' or 'test'")
    if period == "test" and not final:
        raise ValueError("replay_split: test requires --final (CLAUDE.md rule 4)")
    if final and period != "test":
        raise ValueError("--final requires replay_split: test")
    frame = load_dev_frame(cfg["data_dir"], cfg["split_dir"])
    if final:
        test = load_test_frame(cfg["data_dir"], cfg["split_dir"]).assign(split="test")
        frame = pd.concat([frame, test], ignore_index=True)
    frame = frame.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)
    return frame, period


def drift_report(
    X: np.ndarray, names: list[str], ref_rows: np.ndarray, weeks: list[np.ndarray], cfg: dict
) -> dict[str, Any]:
    """PSI per week of each non-binary input against the initial training data."""
    rng = np.random.default_rng(0)
    ref = rng.choice(ref_rows, size=min(len(ref_rows), cfg["drift_sample"]), replace=False)
    cols = [j for j in range(X.shape[1]) if len(np.unique(X[ref[:5000], j])) > 2]
    out = []
    for rows in weeks:
        vals = {names[j]: psi(X[ref, j], X[rows, j], cfg["psi_bins"]) for j in cols}
        top = sorted(vals.items(), key=lambda kv: -kv[1])[:5]
        out.append(
            {
                "mean_feature_psi": float(np.mean(list(vals.values()))),
                "max_feature_psi": float(max(vals.values())),
                "top_features": [{"feature": k, "psi": v} for k, v in top],
                "share_features_psi_above_0.25": float(np.mean([v > 0.25 for v in vals.values()])),
            }
        )
    return {"n_features_checked": len(cols), "by_week": out}


def replay(
    config_path: str | Path,
    final: bool = False,
    results_root: str | Path = "results",
    resume: str | Path | None = None,
) -> Path:
    """Run every experiment, strategy, schedule, and seed in a replay config.

    With ``resume``, continue the interrupted run in that folder (see module docstring).
    """
    cfg = load_config(config_path)
    frame, period = _load(cfg, final)
    configure_device(cfg["device"])

    pre = Preprocessor(load_config(cfg["features_config"])).fit(frame[frame["split"] == "train"])
    X = pre.transform(frame)
    y = frame[TARGET].to_numpy()
    times = frame[TIME_COL].to_numpy()
    ids, amounts = frame[ID_COL].to_numpy(), frame[AMOUNT_COL].to_numpy()
    in_period = (frame["split"] == period).to_numpy()
    t0, t_end = int(times[in_period].min()), int(times[in_period].max()) + 1
    bounds = week_windows(t0, t_end, cfg["week_days"])
    weeks = [np.flatnonzero(in_period & (times >= a) & (times < b)) for a, b in bounds]
    del frame  # only the arrays above are needed from here on
    gc.collect()

    if resume is None:
        run_id = f"{cfg['name']}_{datetime.now():%Y%m%d-%H%M%S}" + ("_final" if final else "")
        run_dir = Path(results_root) / cfg["stage"] / run_id
        save_yaml({"config": cfg, "final": final}, run_dir / "config.yaml")
        save_json(env_info(), run_dir / "env.json")
        pre.save(run_dir / "preprocessor.json")
    else:
        run_dir = Path(resume)
        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        if saved != {"config": cfg, "final": final}:
            raise ValueError("resume: config or --final differs from the interrupted run")
        save_json(env_info(), run_dir / f"env_resume_{datetime.now():%Y%m%d-%H%M%S}.json")

    delay = cfg["label_delay_days"] * SECONDS_PER_DAY
    initial_rows = np.flatnonzero(times < t0 - delay)
    if not (run_dir / "drift.json").exists():
        save_json(
            {
                "replay_split": period,
                "weeks": [
                    {
                        "start": a,
                        "end": b,
                        "n": len(r),
                        "n_fraud": int(y[r].sum()),
                        "fraud_rate": float(y[r].mean()),
                    }
                    for (a, b), r in zip(bounds, weeks, strict=True)
                ],
                **drift_report(X, pre.feature_names, initial_rows, weeks, cfg),
            },
            run_dir / "drift.json",
        )

    schedules = [("none", None)] if cfg.get("include_never", True) else []
    schedules += [(s, e) for s in cfg["strategies"] for e in cfg["retrain_every_weeks"]]
    summary = []
    for exp in cfg["experiments"]:
        if exp["model"] not in MODELS:
            raise ValueError(f"unknown model {exp['model']!r}")
        for strategy, every in schedules:
            name = f"{strategy}" + ("" if every is None else f"_every{every}")
            path = run_dir / exp["name"] / name / "metrics.json"
            if path.exists():  # finished before an interruption
                stats = json.loads(path.read_text())["summary"]
                summary.append(
                    {"experiment": exp["name"], "schedule": name, **stats["select_score"]}
                )
                print(f"[{exp['name']} {name}] already saved, skipped", flush=True)
                continue
            per_seed = {}
            for seed in exp.get("seeds", cfg["seeds"]):
                res = replay_one(
                    X,
                    y,
                    times,
                    weeks,
                    bounds,
                    ids,
                    amounts,
                    exp,
                    cfg["settings"],
                    "expanding" if strategy == "none" else strategy,
                    every,
                    seed,
                    cfg,
                )
                per_seed[seed] = res
                print(
                    f"[{exp['name']} {strategy} every={every}] seed {seed}: "
                    f"score {res['select_score']:.4f}",
                    flush=True,
                )
            flat = {
                s: {
                    "select_score": r["select_score"],
                    "pooled": r["pooled"],
                    "epsilon_spent_total": r.get("privacy", {}).get("epsilon_spent_total"),
                }
                for s, r in per_seed.items()
            }
            stats = summarize_seeds(flat)
            save_json(
                {
                    "experiment": exp,
                    "summary": stats,
                    "per_seed": {str(k): v for k, v in per_seed.items()},
                },
                path,
            )
            summary.append({"experiment": exp["name"], "schedule": name, **stats["select_score"]})
    save_json({"replay_split": period, "final": final, "rows": summary}, run_dir / "summary.json")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--final", action="store_true", help="replay the test period")
    parser.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run")
    args = parser.parse_args()
    print(f"results: {replay(args.config, final=args.final, resume=args.resume)}")


if __name__ == "__main__":
    main()
