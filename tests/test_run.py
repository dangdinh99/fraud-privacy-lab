"""End-to-end tests for fraudpriv.run on tiny synthetic raw files."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fraudpriv.data import compute_split, save_split
from fraudpriv.run import expand_grid, expand_points, parse_only, run, summarize_seeds

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


@pytest.fixture
def workspace(tmp_path):
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    fraud = (rng.random(n) < 1 / (1 + np.exp(-(3 * x - 3)))).astype(int)
    trans = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(86400, 60 * 86400, size=n)),
            "isFraud": fraud,
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


def write_config(ws, model_type, eval_split="val"):
    params = (
        {
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "num_threads": 1,
            "max_rounds": 20,
            "early_stopping_rounds": 5,
        }
        if model_type == "lightgbm"
        else {"hidden": [8], "batch_size": 64, "epochs": 1}
    )
    grid = {"num_leaves": [7]} if model_type == "lightgbm" else {"lr": [0.01]}
    cfg = {
        "name": f"t_{model_type}",
        "stage": "stage1",
        "data_dir": str(ws / "raw"),
        "split_dir": str(ws / "splits"),
        "features_config": str(ws / "features.yaml"),
        "eval_split": eval_split,
        "seeds": [0, 1],
        "budgets": [0.01, 0.05],
        "select_budgets": [0.01, 0.05],
        "ece_bins": 10,
        "model": {"type": model_type, "params": params},
        "grid": grid,
    }
    path = ws / f"{model_type}_{eval_split}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_expand_grid():
    points = expand_grid({"a": 1}, {"lr": [0.1, 0.2], "epochs": [5]})
    assert [name for name, _ in points] == ["lr=0.1__epochs=5", "lr=0.2__epochs=5"]
    assert points[1][1] == {"a": 1, "lr": 0.2, "epochs": 5}
    assert expand_grid({"a": 1}, None) == [("default", {"a": 1})]


@pytest.mark.parametrize("model_type", ["lightgbm", "mlp"])
def test_run_writes_all_outputs(workspace, model_type):
    run_dir = run(write_config(workspace, model_type), results_root=workspace / "results")
    for f in ("config.yaml", "env.json", "preprocessor.json", "summary.json"):
        assert (run_dir / f).exists()
    point = next(p for p in run_dir.iterdir() if p.is_dir())
    m = json.loads((point / "metrics.json").read_text())
    assert m["eval_split"] == "val" and m["final"] is False and m["ece_in_sample"] is True
    for key in ("recall", "recall_ceiling", "recall_share_of_ceiling", "dollar_loss"):
        stats = m["summary"][f"budget_0.01.{key}"]
        assert set(stats) == {"mean", "std"}
    seed0 = m["per_seed"]["0"]
    assert set(seed0["budget_0.01"]["segments"]) == {"seen_card", "new_card", "no_key"}
    assert "ece" in seed0 and "accuracy" not in json.dumps(m)
    env = json.loads((run_dir / "env.json").read_text())
    assert {"git_commit", "git_dirty", "packages"} <= env.keys()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["select_budgets"] == [0.01, 0.05]
    expected = np.mean([seed0[f"budget_{b}"]["recall_share_of_ceiling"] for b in (0.01, 0.05)])
    assert seed0["select_score"] == pytest.approx(expected)
    assert summary["ranking"][0]["mean"] == pytest.approx(m["summary"]["select_score"]["mean"])


def test_test_set_requires_final(workspace):
    with pytest.raises(ValueError, match="--final"):
        run(write_config(workspace, "lightgbm", "test"), results_root=workspace / "results")
    with pytest.raises(ValueError, match="eval_split: test"):
        run(write_config(workspace, "lightgbm", "val"), final=True, results_root=workspace / "r")


def test_final_run_evaluates_test(workspace):
    cfg = write_config(workspace, "lightgbm", "test")
    run_dir = run(cfg, final=True, results_root=workspace / "results")
    assert run_dir.name.endswith("_final")
    m = json.loads((next(p for p in run_dir.iterdir() if p.is_dir()) / "metrics.json").read_text())
    assert m["eval_split"] == "test" and m["final"] is True and m["ece_in_sample"] is False
    assert m["per_seed"]["0"]["budget_0.01"]["n"] == 300  # 20% of 1500


def test_expand_points_and_parse_only():
    pts = expand_points({"a": 1}, None, [{"lr": 0.1, "epochs": 5}, {"lr": 0.2, "epochs": 5}])
    assert pts == [
        ("lr=0.1__epochs=5", {"a": 1, "lr": 0.1, "epochs": 5}),
        ("lr=0.2__epochs=5", {"a": 1, "lr": 0.2, "epochs": 5}),
    ]
    with pytest.raises(ValueError):
        expand_points({}, {"lr": [1]}, [{"lr": 1}])
    assert parse_only(["target_epsilon=1", "hidden=[]"]) == {"target_epsilon": 1, "hidden": []}


def test_dp_logreg_run_ranks_within_epsilon_and_logs_privacy(workspace):
    path = write_config(workspace, "mlp")
    cfg = yaml.safe_load(path.read_text())
    cfg["model"] = {
        "type": "dp_logreg",
        "params": {"hidden": [], "batch_size": 64, "epochs": 1, "lr": 0.01, "delta": 1e-6},
    }
    cfg["grid"] = {"target_epsilon": [1, 8], "max_grad_norm": [0.2, 1.0]}
    cfg["torch_threads"] = 1
    path.write_text(yaml.safe_dump(cfg))
    run_dir = run(path, results_root=workspace / "results", only={"target_epsilon": 8})
    assert "target_epsilon=8" in run_dir.name
    summary = json.loads((run_dir / "summary.json").read_text())
    assert {r["target_epsilon"] for r in summary["ranking"]} == {8}
    assert len(summary["ranking"]) == 2
    m = json.loads((run_dir / summary["ranking"][0]["grid_point"] / "metrics.json").read_text())
    priv = m["per_seed"]["0"]["train_info"]["privacy"]
    assert priv["epsilon_spent"] <= 8 and priv["accountant"] == "rdp"
    assert "train_info.privacy.epsilon_spent" in m["summary"]


def test_logreg_requires_no_hidden_layers(workspace):
    path = write_config(workspace, "mlp")
    cfg = yaml.safe_load(path.read_text())
    cfg["model"]["type"] = "logreg"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="hidden"):
        run(path, results_root=workspace / "results")


def test_summarize_seeds_single_seed_has_no_std():
    one = summarize_seeds({0: {"a": 1.0}})
    assert one == {"a": {"mean": 1.0, "std": None}}
    two = summarize_seeds({0: {"a": 1.0}, 1: {"a": 3.0}})
    assert two["a"]["mean"] == 2.0 and two["a"]["std"] == pytest.approx(2**0.5)
