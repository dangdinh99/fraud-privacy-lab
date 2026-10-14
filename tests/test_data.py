"""Tests for the frozen time split in fraudpriv.data."""

import numpy as np
import pandas as pd
import pytest

from fraudpriv.data import (
    apply_split,
    card_key,
    card_segments,
    compute_split,
    load_dev_frame,
    load_split,
    save_split,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "TransactionID": rng.permutation(np.arange(1000, 1000 + n)),
            "TransactionDT": rng.integers(0, 50, size=n),  # many duplicate times
            "isFraud": rng.integers(0, 2, size=n),
        }
    )


def test_split_sizes_disjoint_and_complete(frame):
    split = compute_split(frame, 0.6, 0.2)
    assert [len(split[k]) for k in ("train", "val", "test")] == [60, 20, 20]
    all_ids = np.concatenate(list(split.values()))
    assert len(set(all_ids)) == len(frame)
    assert set(all_ids) == set(frame["TransactionID"])


def test_split_is_chronological(frame):
    train, val, test = apply_split(frame, compute_split(frame, 0.6, 0.2))
    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert val["TransactionDT"].max() <= test["TransactionDT"].min()


def test_split_independent_of_row_order(frame):
    a = compute_split(frame, 0.6, 0.2)
    b = compute_split(frame.sample(frac=1, random_state=1), 0.6, 0.2)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_save_load_roundtrip_and_frozen(frame, tmp_path):
    split = compute_split(frame, 0.6, 0.2)
    save_split(split, tmp_path, meta={"x": 1})
    loaded = load_split(tmp_path)
    for k in split:
        np.testing.assert_array_equal(split[k], loaded[k])
    with pytest.raises(FileExistsError):
        save_split(split, tmp_path, meta={})
    save_split(split, tmp_path, meta={}, force=True)


def test_apply_split_rejects_mismatched_data(frame):
    split = compute_split(frame, 0.6, 0.2)
    with pytest.raises(ValueError):
        apply_split(frame.iloc[:-1], split)


@pytest.mark.parametrize(("tr", "va"), [(0.0, 0.2), (0.8, 0.2), (0.6, 0.0)])
def test_compute_split_rejects_bad_fractions(frame, tr, va):
    with pytest.raises(ValueError):
        compute_split(frame, tr, va)


def test_load_dev_frame_excludes_test(frame, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frame.assign(TransactionAmt=1.0).to_csv(raw / "train_transaction.csv", index=False)
    frame[["TransactionID"]].iloc[::3].assign(id_01=0.0).to_csv(
        raw / "train_identity.csv", index=False
    )
    split = compute_split(frame, 0.6, 0.2)
    save_split(split, tmp_path / "splits", meta={})
    dev = load_dev_frame(raw, tmp_path / "splits")
    assert len(dev) == 80
    assert not dev["TransactionID"].isin(split["test"]).any()
    assert set(dev.loc[dev["split"] == "val", "TransactionID"]) == set(split["val"])


def _cards():
    # day = TransactionDT // 86400; key = card1|addr1|(day - D1)
    return pd.DataFrame(
        {
            "card1": [100, 100, 200, 300, np.nan],
            "addr1": [10.0, 10.0, 20.0, np.nan, 30.0],
            "D1": [0.0, 1.0, 5.0, 0.0, 0.0],
            "TransactionDT": [86400 * 3, 86400 * 4, 86400 * 10, 86400, 86400],
        }
    )


def test_card_key_hand_computed():
    key = card_key(_cards())
    assert key.iloc[0] == "100|10|3"
    assert key.iloc[1] == "100|10|3"  # same card one day later, D1 grew by one
    assert key.iloc[2] == "200|20|5"
    assert key.iloc[3:].isna().all()  # missing addr1, missing card1


def test_card_segments():
    df = _cards()
    segs = card_segments(df.iloc[1:], train_df=df.iloc[:1])
    np.testing.assert_array_equal(segs["seen_card"], [True, False, False, False])
    np.testing.assert_array_equal(segs["new_card"], [False, True, False, False])
    np.testing.assert_array_equal(segs["no_key"], [False, False, True, True])
