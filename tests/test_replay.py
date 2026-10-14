"""Tests for fraudpriv.replay: windows, label delay, strategies, privacy totals, end to end."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fraudpriv.data import compute_split, save_split
from fraudpriv.replay import (
    nearest_setting,
    replay,
    retrain_weeks,
    training_windows,
    week_windows,
    window_mask,
)

DAY = 86400


def test_week_windows_with_partial_last_week():
    assert week_windows(0, 17 * DAY, 7) == [(0, 7 * DAY), (7 * DAY, 14 * DAY), (14 * DAY, 17 * DAY)]
    assert week_windows(100, 100 + 7 * DAY, 7) == [(100, 100 + 7 * DAY)]


def test_retrain_weeks():
    assert retrain_weeks(6, None) == [0]
    assert retrain_weeks(6, 1) == [0, 1, 2, 3, 4, 5]
    assert retrain_weeks(6, 2) == [0, 2, 4]
    assert retrain_weeks(6, 3) == [0, 3]


def test_training_windows_respect_delay_and_strategy():
    times = [100, 200, 300]
    assert training_windows(times, 10, "expanding") == [(None, 90), (None, 190), (None, 290)]
    assert training_windows(times, 10, "disjoint") == [(None, 90), (90, 190), (190, 290)]
    with pytest.raises(ValueError):
        training_windows(times, 10, "other")


def test_disjoint_masks_never_overlap_and_never_see_future_labels():
    rng = np.random.default_rng(0)
    times = np.sort(rng.integers(0, 1000, size=5000))
    run_times, delay = [400, 550, 700, 850], 60
    masks = [
        window_mask(times, lo, hi) for lo, hi in training_windows(run_times, delay, "disjoint")
    ]
    for i, m in enumerate(masks):
        assert times[m].max() < run_times[i] - delay
        for other in masks[i + 1 :]:
            assert not (m & other).any()
    assert np.array_equal(np.logical_or.reduce(masks), times < run_times[-1] - delay)


def test_nearest_setting_on_log_scale():
    settings = {0.5: {"a": 1}, 1: {"a": 2}, 2: {"a": 3}, 4: {"a": 4}, 8: {"a": 5}}
    assert nearest_setting(settings, 0.33) == (0.5, {"a": 1})
    assert nearest_setting(settings, 1.4) == (1.0, {"a": 2})  # |ln 1.4| < |ln(2/1.4)|
    assert nearest_setting(settings, 3.0) == (4.0, {"a": 4})  # ln(4/3) < ln(3/2)
    assert nearest_setting(settings, 100) == (8.0, {"a": 5})


# --- end to end on synthetic data ----------------------------------------------------------

FEATURES = {
    "exclude": ["TransactionID", "TransactionDT", "isFraud", "split"],
    "max_missing_rate": 0.9,
    "prune_prefix": "V",
    "prune_corr": 0.95,
    "log1p_columns": ["TransactionAmt"],
    "clip_sd": 5.0,
    "categorical": ["cat", "card1", "addr1"],
    "onehot_min_freq": 0.01,
    "frequency_columns": ["card1"],
    "hour_of_day": True,
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("replay")
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    trans = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(DAY, 61 * DAY, size=n)),
            "isFraud": (rng.random(n) < 1 / (1 + np.exp(-(3 * x - 2.5)))).astype(int),
            "TransactionAmt": rng.lognormal(4, 1, size=n),
            "V1": x + rng.normal(scale=0.5, size=n),
            "D1": rng.integers(0, 100, size=n).astype(float),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "card1": rng.integers(1000, 1100, size=n),
            "addr1": rng.choice([100.0, 200.0, np.nan], size=n),
        }
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    trans.to_csv(raw / "train_transaction.csv", index=False)
    trans[["TransactionID"]].iloc[::4].assign(id_01=0.0).to_csv(
        raw / "train_identity.csv", index=False
    )
    save_split(compute_split(trans, 0.6, 0.2), tmp_path / "splits", meta={})
    (tmp_path / "features.yaml").write_text(yaml.safe_dump(FEATURES))
    return tmp_path


def write_config(ws, replay_split="val"):
    small = {"batch_size": 64, "lr": 0.01, "epochs": 2}
    cfg = {
        "name": "t_replay",
        "stage": "stage3",
        "data_dir": str(ws / "raw"),
        "split_dir": str(ws / "splits"),
        "features_config": str(ws / "features.yaml"),
        "replay_split": replay_split,
        "device": "cpu",
        "seeds": [10, 11],
        "budgets": [0.05, 0.1],
        "label_delay_days": 2,
        "week_days": 4,
        "delta": 1.0e-6,
        "drift_sample": 1000,
        "psi_bins": 10,
        "strategies": ["expanding", "disjoint"],
        "retrain_every_weeks": [1],
        "hidden": {"dp_logreg": [], "logreg": [], "dp_mlp": [8], "mlp": [8]},
        "settings": {
            "dp_logreg": {1: {**small, "max_grad_norm": 1.0}, 4: {**small, "max_grad_norm": 5.0}},
            "logreg": small,
        },
        "experiments": [
            {"name": "dp_logreg_eps2", "model": "dp_logreg", "epsilon_total": 2},
            {"name": "logreg", "model": "logreg", "seeds": [10]},
        ],
    }
    path = ws / f"replay_{replay_split}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="module")
def run_dir(workspace):
    return replay(write_config(workspace), results_root=workspace / "results")


def metrics(run_dir, exp, sched):
    return json.loads((run_dir / exp / sched / "metrics.json").read_text())


def test_replay_writes_outputs(run_dir):
    for f in ("config.yaml", "env.json", "preprocessor.json", "drift.json", "summary.json"):
        assert (run_dir / f).exists()
    drift = json.loads((run_dir / "drift.json").read_text())
    assert len(drift["weeks"]) == 3 and len(drift["by_week"]) == 3
    rows = json.loads((run_dir / "summary.json").read_text())["rows"]
    assert {(r["experiment"], r["schedule"]) for r in rows} == {
        (e, s)
        for e in ("dp_logreg_eps2", "logreg")
        for s in ("none", "expanding_every1", "disjoint_every1")
    }


def test_expanding_composes_all_runs_within_budget(run_dir):
    m = metrics(run_dir, "dp_logreg_eps2", "expanding_every1")
    for res in m["per_seed"].values():
        assert res["n_runs"] == 3
        assert 0.97 * 2 < res["privacy"]["epsilon_spent_total"] <= 2
        assert res["privacy"]["composition"] == "sequential (RDP)"
        assert res["setting_epsilon"] == 1.0  # 2 / 3 runs -> nearest 1 on a log scale
        for run in res["runs"]:
            lo, hi = run["label_window"]
            assert lo is None and run["max_train_time"] < hi
            assert run["train_info"]["privacy"]["accounting"] == "composed"
        assert [r["n_train"] for r in res["runs"]] == sorted(r["n_train"] for r in res["runs"])


def test_disjoint_runs_get_full_budget_on_new_data_only(run_dir):
    res = metrics(run_dir, "dp_logreg_eps2", "disjoint_every1")["per_seed"]["10"]
    assert res["privacy"]["composition"] == "parallel"
    assert res["privacy"]["epsilon_spent_total"] <= 2
    windows = [r["label_window"] for r in res["runs"]]
    for prev, cur in zip(windows, windows[1:], strict=False):
        assert cur[0] == prev[1]  # each window starts where the previous one ended
    assert [r["warm_start"] for r in res["runs"]] == [False, True, True]
    assert [r["epochs"] for r in res["runs"]] == [2, 2, 2]  # default fraction 1.0
    for r in res["runs"]:
        assert r["train_info"]["privacy"]["epsilon_spent"] <= 2


def test_never_retrain_has_one_run_and_score_drift(run_dir):
    res = metrics(run_dir, "dp_logreg_eps2", "none")["per_seed"]["10"]
    assert res["n_runs"] == 1 and len(res["per_week"]) == 3
    assert len(res["score_psi_by_week"]) == 3
    pooled = res["pooled"]["budget_0.05"]
    assert pooled["n_windows"] == 3 and pooled["recall"] <= pooled["recall_ceiling"]


def test_single_seed_experiment_has_no_std(run_dir):
    m = metrics(run_dir, "logreg", "none")
    assert m["summary"]["select_score"]["std"] is None
    assert "privacy" not in m["per_seed"]["10"]


def test_test_period_requires_final(workspace):
    with pytest.raises(ValueError, match="--final"):
        replay(write_config(workspace, "test"), results_root=workspace / "r")
    with pytest.raises(ValueError, match="replay_split: test"):
        replay(write_config(workspace, "val"), final=True, results_root=workspace / "r")


def test_finetune_fraction_shortens_warm_starts_only(workspace):
    cfg = yaml.safe_load(write_config(workspace).read_text())
    cfg.update(
        name="t_frac",
        finetune_epochs_fraction=0.5,
        include_never=False,
        seeds=[10],
        settings={**cfg["settings"], "logreg": {**cfg["settings"]["logreg"], "epochs": 4}},
        experiments=[{"name": "logreg", "model": "logreg"}],
    )
    path = workspace / "frac.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_dir = replay(path, results_root=workspace / "results_frac")
    assert not (run_dir / "logreg" / "none").exists()
    disjoint = metrics(run_dir, "logreg", "disjoint_every1")["per_seed"]["10"]
    assert [r["epochs"] for r in disjoint["runs"]] == [4, 2, 2]
    expanding = metrics(run_dir, "logreg", "expanding_every1")["per_seed"]["10"]
    assert [r["epochs"] for r in expanding["runs"]] == [4, 4, 4]


def test_resume_skips_finished_work_and_reproduces_the_rest(workspace):
    import shutil

    cfg = yaml.safe_load(write_config(workspace).read_text())
    cfg.update(
        name="t_resume",
        seeds=[10],
        strategies=["expanding"],
        experiments=[{"name": "logreg", "model": "logreg"}],
    )
    path = workspace / "resume.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_dir = replay(path, results_root=workspace / "results_resume")
    done = run_dir / "logreg" / "none" / "metrics.json"
    lost = run_dir / "logreg" / "expanding_every1"
    original = json.loads((lost / "metrics.json").read_text())
    done_mtime = done.stat().st_mtime_ns
    shutil.rmtree(lost)  # as if the run stopped before this schedule finished
    (run_dir / "summary.json").unlink()

    assert replay(path, resume=run_dir) == run_dir
    assert done.stat().st_mtime_ns == done_mtime  # finished work was not redone
    assert json.loads((lost / "metrics.json").read_text())["per_seed"] == original["per_seed"]
    rows = json.loads((run_dir / "summary.json").read_text())["rows"]
    assert {r["schedule"] for r in rows} == {"none", "expanding_every1"}
    assert len(list(run_dir.glob("env_resume_*.json"))) == 1

    cfg["seeds"] = [11]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="resume"):
        replay(path, resume=run_dir)


def test_clip_only_diagnostic_uses_dp_settings_without_noise(workspace):
    cfg = yaml.safe_load(write_config(workspace).read_text())
    cfg.update(
        name="t_clip",
        seeds=[10],
        strategies=["expanding"],
        experiments=[{"name": "clip", "model": "dp_logreg", "epsilon_total": 2, "clip_only": True}],
    )
    path = workspace / "clip.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_dir = replay(path, results_root=workspace / "results_clip")
    res = metrics(run_dir, "clip", "expanding_every1")["per_seed"]["10"]
    assert res["privacy"]["private"] is False
    assert res["setting_epsilon"] == 1.0  # chosen exactly as the DP experiment would
    for run in res["runs"]:
        priv = run["train_info"]["privacy"]
        assert priv["private"] is False and priv["noise_multiplier"] == 0.0
        assert priv["max_grad_norm"] == 1.0 and priv["epsilon_spent"] is None
