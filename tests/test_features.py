"""Tests for fraudpriv.features on a small synthetic frame."""

import copy

import numpy as np
import pandas as pd
import pytest

from fraudpriv.features import MISSING, OTHER, Preprocessor

CONFIG = {
    "exclude": ["TransactionID", "TransactionDT", "isFraud", "split"],
    "max_missing_rate": 0.9,
    "prune_prefix": "V",
    "prune_corr": 0.95,
    "log1p_columns": ["TransactionAmt"],
    "clip_sd": 5.0,
    "categorical": ["cat", "card1"],
    "onehot_min_freq": 0.01,
    "frequency_columns": ["card1"],
    "hour_of_day": True,
}


def make_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v_missing = rng.random(n) < 0.3
    v1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": rng.integers(0, 10 * 86400, size=n),
            "isFraud": rng.integers(0, 2, size=n),
            "TransactionAmt": rng.lognormal(4, 1, size=n),
            "V1": np.where(v_missing, np.nan, v1),
            "V2": np.where(v_missing, np.nan, 2 * v1 + 1),  # exact copy of V1 up to scale
            "V3": np.where(v_missing, np.nan, rng.normal(size=n)),  # same pattern, independent
            "D1": np.where(rng.random(n) < 0.2, np.nan, rng.integers(0, 500, size=n)),
            "mostly_missing": np.where(rng.random(n) < 0.95, np.nan, 1.0),
            "const": 7.0,
            "cat": rng.choice(["a", "b"], size=n, p=[0.6, 0.4]),
            "card1": rng.integers(1000, 1050, size=n),
        }
    )
    df.loc[0, "cat"] = "rare"  # 0.1% of rows, below the 1% floor
    df.loc[1:20, "cat"] = None
    return df


@pytest.fixture
def train():
    return make_frame()


@pytest.fixture
def pre(train):
    return Preprocessor(CONFIG).fit(train)


def block(pre, X, prefix):
    idx = [i for i, name in enumerate(pre.feature_names) if name.startswith(prefix)]
    return X[:, idx]


def test_column_selection(pre):
    names = pre.feature_names
    for kept in ("TransactionAmt", "V1", "V3", "D1", "missing__V1", "missing__D1"):
        assert kept in names
    for dropped in ("V2", "mostly_missing", "const", "missing__V2", "missing__V3"):
        assert dropped not in names
    for forbidden in CONFIG["exclude"]:
        assert not any(forbidden in name for name in names)
    assert not any("card_key" in name for name in names)


def test_feature_names_deterministic(train):
    a = Preprocessor(CONFIG).fit(train).feature_names
    b = Preprocessor(CONFIG).fit(train.sample(frac=1, random_state=3)).feature_names
    assert a == b


def test_output_finite_and_clipped(pre, train):
    X = pre.transform(train)
    assert X.dtype == np.float32
    assert X.shape == (len(train), len(pre.feature_names))
    assert np.isfinite(X).all()
    assert np.abs(X).max() <= CONFIG["clip_sd"]


def test_onehot_rare_unseen_and_missing(pre):
    later = make_frame(seed=1).iloc[:4].copy()
    later["cat"] = ["a", "rare", "never_seen", None]
    X = block(pre, pre.transform(later), "cat==")
    names = [n for n in pre.feature_names if n.startswith("cat==")]
    assert names == ["cat==a", "cat==b", f"cat=={OTHER}", f"cat=={MISSING}"]
    np.testing.assert_array_equal(X, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def test_numeric_imputation_and_clipping(pre, train):
    stats = pre.state["num_stats"]["D1"]
    later = make_frame(seed=1).iloc[:2].copy()
    later["D1"] = [np.nan, 1e9]
    col = pre.feature_names.index("D1")
    X = pre.transform(later)
    assert X[0, col] == pytest.approx((stats["median"] - stats["mean"]) / stats["std"], rel=1e-5)
    assert X[1, col] == CONFIG["clip_sd"]
    assert stats["median"] == pytest.approx(train["D1"].median())


def test_log1p_applied_before_standardizing(pre, train):
    stats = pre.state["num_stats"]["TransactionAmt"]
    assert stats["mean"] == pytest.approx(np.log1p(train["TransactionAmt"]).mean())


def test_frequency_encoding_unseen_is_zero_count(pre):
    later = make_frame(seed=1).iloc[:1].copy()
    later["card1"] = 999_999
    p = pre.state["freq"]["card1"]
    X = pre.transform(later)
    expected = max(-CONFIG["clip_sd"], (0 - p["mean"]) / p["std"])
    assert X[0, pre.feature_names.index("freq__card1")] == pytest.approx(expected, rel=1e-5)


def test_hour_of_day(pre):
    later = make_frame(seed=1).iloc[:2].copy()
    later["TransactionDT"] = [86400 * 5, 86400 * 5 + 6 * 3600]  # midnight, 06:00
    X = pre.transform(later)
    s, c = pre.feature_names.index("hour_sin"), pre.feature_names.index("hour_cos")
    np.testing.assert_allclose(X[:, [s, c]], [[0, 1], [1, 0]], atol=1e-6)


def test_no_leakage_transform_never_changes_state(pre):
    before = copy.deepcopy(pre.state)
    later = make_frame(seed=2)
    later["D1"] = 1e6
    later["cat"] = "new"
    original = later.copy()
    pre.transform(later)
    assert pre.state == before
    pd.testing.assert_frame_equal(later, original)


def test_fit_rejects_non_train_rows(train):
    mixed = train.assign(split=["train"] * (len(train) - 1) + ["val"])
    with pytest.raises(ValueError):
        Preprocessor(CONFIG).fit(mixed)


def test_fit_rejects_undeclared_strings_and_missing_columns(train):
    with pytest.raises(ValueError):
        Preprocessor(CONFIG).fit(train.assign(extra="x"))
    with pytest.raises(KeyError):
        Preprocessor(CONFIG).fit(train.drop(columns="card1"))


def test_save_load_roundtrip(pre, train, tmp_path):
    pre.save(tmp_path / "pre.json")
    loaded = Preprocessor.load(tmp_path / "pre.json")
    assert loaded.feature_names == pre.feature_names
    np.testing.assert_array_equal(loaded.transform(train), pre.transform(train))
