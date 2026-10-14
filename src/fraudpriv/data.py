"""Load IEEE-CIS and build, save, and apply the frozen time-based split.

The split is built once with ``python -m fraudpriv.data --build-split`` and saved as
``TransactionID`` lists in ``data/splits/``. Every experiment loads those saved IDs;
nothing re-splits the data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudpriv.metrics import recall_ceiling
from fraudpriv.utils import load_config, save_json

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"  # seconds from an unknown reference point, not a timestamp
AMOUNT_COL = "TransactionAmt"
RAW_FILES = ("train_transaction.csv", "train_identity.csv")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FILE = "split.csv.gz"
SPLIT_META_FILE = "split_meta.json"


def missing_raw_files(data_dir: str | Path) -> list[str]:
    """Names of required raw IEEE-CIS files not present in ``data_dir``."""
    return [f for f in RAW_FILES if not (Path(data_dir) / f).is_file()]


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes; the merged frame is about 590k rows by 430 columns."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_ieee(data_dir: str | Path) -> pd.DataFrame:
    """Load train_transaction and left-join train_identity on ``TransactionID``.

    Many transactions have no identity row; ``has_identity`` records which do.
    Rows are sorted by time, then ``TransactionID``.
    """
    d = Path(data_dir)
    trans = pd.read_csv(d / RAW_FILES[0])
    ident = pd.read_csv(d / RAW_FILES[1])
    df = trans.merge(ident, on=ID_COL, how="left").copy()  # copy consolidates the wide frame
    df["has_identity"] = df[ID_COL].isin(ident[ID_COL]).astype("int8")
    df = _downcast(df)
    return df.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)


def compute_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> dict[str, np.ndarray]:
    """Chronological split into train, val, and test ``TransactionID`` arrays.

    Rows are ordered by ``TransactionDT`` then ``TransactionID``; the first
    ``train_frac`` of rows is train, the next ``val_frac`` is val, the rest is test.
    No shuffling.
    """
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError("need 0 < train_frac, 0 < val_frac, and train_frac + val_frac < 1")
    ordered = df.sort_values([TIME_COL, ID_COL])[ID_COL].to_numpy()
    n = len(ordered)
    i_val = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))
    return {
        "train": ordered[:i_val],
        "val": ordered[i_val:i_test],
        "test": ordered[i_test:],
    }


def save_split(
    split: dict[str, np.ndarray], split_dir: str | Path, meta: dict, force: bool = False
) -> Path:
    """Write the split IDs and metadata. Refuses to overwrite unless ``force``."""
    split_dir = Path(split_dir)
    path = split_dir / SPLIT_FILE
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; the split is frozen. Pass force=True to rebuild.")
    split_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [pd.DataFrame({ID_COL: split[name], "split": name}) for name in SPLIT_NAMES],
        ignore_index=True,
    )
    frame.to_csv(path, index=False)
    save_json(meta, split_dir / SPLIT_META_FILE)
    return path


def load_split(split_dir: str | Path) -> dict[str, np.ndarray]:
    """Load the frozen split as ``{"train": ids, "val": ids, "test": ids}``."""
    path = Path(split_dir) / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python -m fraudpriv.data --build-split")
    frame = pd.read_csv(path)
    return {name: frame.loc[frame["split"] == name, ID_COL].to_numpy() for name in SPLIT_NAMES}


def apply_split(
    df: pd.DataFrame, split: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) frames for the saved split, each in time order.

    Raises if the split IDs overlap or do not exactly cover ``df``.
    """
    all_ids = np.concatenate([split[name] for name in SPLIT_NAMES])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("split IDs overlap")
    if len(all_ids) != len(df) or not np.isin(df[ID_COL].to_numpy(), all_ids).all():
        raise ValueError("split IDs do not match the data; was the split built on other data?")
    ordered = df.sort_values([TIME_COL, ID_COL])
    parts = tuple(
        ordered[ordered[ID_COL].isin(split[name])].reset_index(drop=True) for name in SPLIT_NAMES
    )
    return parts  # type: ignore[return-value]


def load_dev_frame(data_dir: str | Path, split_dir: str | Path) -> pd.DataFrame:
    """Train and validation rows only, with a ``split`` column; test rows are dropped.

    For EDA and model development, so the test period cannot be inspected by accident.
    """
    df = load_ieee(data_dir)
    split = load_split(split_dir)
    apply_split(df, split)  # checks the saved split matches this data
    df = df[~df[ID_COL].isin(split["test"])].copy()
    df["split"] = np.where(df[ID_COL].isin(split["train"]), "train", "val")
    return df.reset_index(drop=True)


def load_test_frame(data_dir: str | Path, split_dir: str | Path) -> pd.DataFrame:
    """Test rows only. For final evaluation through ``run.py --final``; nothing else."""
    df = load_ieee(data_dir)
    split = load_split(split_dir)
    apply_split(df, split)  # checks the saved split matches this data
    return df[df[ID_COL].isin(split["test"])].reset_index(drop=True)


def card_key(df: pd.DataFrame) -> pd.Series:
    """Approximate card identifier: ``card1|addr1|(transaction day - D1)``.

    EVALUATION ONLY. This is never a model input: using it as a feature lets a model
    memorize cards (see docs/decisions.md). It is used to split evaluation results into
    seen and new cards. Rows missing ``card1``, ``addr1``, or ``D1`` get no key (NA).
    """
    day = df[TIME_COL] // 86400
    start = (day - df["D1"]).round()
    parts = [df["card1"], df["addr1"], start]
    key = (
        parts[0].astype("Int64").astype("string")
        + "|"
        + parts[1].astype("Int64").astype("string")
        + "|"
        + parts[2].astype("Int64").astype("string")
    )
    return key.where(pd.concat(parts, axis=1).notna().all(axis=1))


def card_segments(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over ``eval_df``: ``seen_card``, ``new_card``, and ``no_key``.

    A card is seen if its :func:`card_key` appears in ``train_df``. EVALUATION ONLY.
    """
    train_keys = set(card_key(train_df).dropna())
    key = card_key(eval_df)
    has_key = key.notna().to_numpy()
    seen = key.isin(train_keys).to_numpy() & has_key
    return {"seen_card": seen, "new_card": has_key & ~seen, "no_key": ~has_key}


def split_summary(df: pd.DataFrame, split: dict[str, np.ndarray], budget: float) -> dict:
    """Per-split row counts, fraud rates, time ranges, and recall ceiling at ``budget``."""
    out = {}
    for name, part in zip(SPLIT_NAMES, apply_split(df, split), strict=True):
        n, n_fraud = len(part), int(part[TARGET].sum())
        out[name] = {
            "n": n,
            "n_fraud": n_fraud,
            "fraud_rate": n_fraud / n,
            "dt_min": int(part[TIME_COL].min()),
            "dt_max": int(part[TIME_COL].max()),
            "days": (part[TIME_COL].max() - part[TIME_COL].min()) / 86400,
            f"recall_ceiling_at_{budget}": recall_ceiling(n, n_fraud, budget),
        }
    return out


def build_split(config_path: str | Path, force: bool = False) -> dict:
    """Build and save the frozen split from a config; returns the summary."""
    cfg = load_config(config_path)
    path = Path(cfg["data_dir"]) / RAW_FILES[0]
    df = pd.read_csv(path, usecols=[ID_COL, TIME_COL, TARGET])
    split = compute_split(df, cfg["train_frac"], cfg["val_frac"])
    summary = split_summary(df, split, cfg["summary_budget"])
    meta = {
        "source": str(path),
        "train_frac": cfg["train_frac"],
        "val_frac": cfg["val_frac"],
        "order": [TIME_COL, ID_COL],
        "summary": summary,
    }
    save_split(split, cfg["split_dir"], meta, force=force)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-split", action="store_true", help="build the frozen split")
    parser.add_argument("--config", default="configs/split.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()
    if not args.build_split:
        parser.error("nothing to do; pass --build-split")
    print(json.dumps(build_split(args.config, force=args.force), indent=2))


if __name__ == "__main__":
    main()
