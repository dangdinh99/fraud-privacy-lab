"""Shared preprocessing: one transform, fit on train only, used by every model.

Every model (LightGBM, the MLPs, logistic regression) receives the same matrix
(CLAUDE.md rule 2). All statistics are learned in :meth:`Preprocessor.fit` from training
rows only, then applied unchanged to later periods. These statistics are computed
without differential privacy and are not charged to the privacy budget.

Steps, all configured in ``configs/features.yaml``:

1. Exclude ID, time, target, and split columns, and numeric columns whose train
   missing rate exceeds ``max_missing_rate``.
2. Prune columns starting with ``prune_prefix``: within each missingness pattern, a
   column is dropped if ``|r| > prune_corr`` with a column already kept (column order).
3. Missing-value flags: one per missingness pattern among kept numeric columns; the
   flag is whether the pattern's first column is missing.
4. Numeric: ``log1p`` on ``log1p_columns`` (negatives clipped to 0), fill with the train
   median, standardize with train mean and standard deviation, clip to ``+/- clip_sd``.
   Columns constant on train are dropped.
5. Categorical: one-hot for values with train share ``>= onehot_min_freq``, plus an
   ``other`` column (rare or unseen values) and a ``missing`` column.
6. Frequency encoding for ``frequency_columns``: ``log(1 + train count of the value)``,
   standardized and clipped; unseen values count as 0.
7. Hour of day: sine and cosine of ``TransactionDT mod 86400``.

Card identity is deliberately not a feature (see docs/decisions.md).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudpriv.data import TIME_COL

SECONDS_PER_DAY = 86400
OTHER = "__other__"
MISSING = "__missing__"


def _as_category_strings(s: pd.Series) -> pd.Series:
    """Values as strings, so categories compare the same across periods and dtypes.

    Integral numbers are written without a decimal part (13926.0 and 13926 both give
    "13926"). Missing values stay NA.
    """
    if not pd.api.types.is_numeric_dtype(s):
        return s.astype("string")
    f = s.astype("float64")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    present = f.notna()
    integral = present & (f == np.floor(f))
    out[integral] = f[integral].astype("int64").astype("string")
    out[present & ~integral] = f[present & ~integral].astype("string")
    return out


def _log1p(x: np.ndarray) -> np.ndarray:
    """``log1p`` with negatives clipped to 0; NaN stays NaN."""
    return np.log1p(np.clip(x, 0, None))


def _missing_patterns(df: pd.DataFrame, cols: list[str]) -> list[list[str]]:
    """Group columns whose missing-value mask is identical, in first-seen order."""
    groups: dict[bytes, list[str]] = {}
    for c in cols:
        key = hashlib.sha1(np.packbits(df[c].isna().to_numpy()).tobytes()).digest()
        groups.setdefault(key, []).append(c)
    return list(groups.values())


def _prune_correlated(
    df: pd.DataFrame, cols: list[str], prefix: str, threshold: float
) -> list[str]:
    """Drop ``prefix`` columns with ``|r| > threshold`` to a kept column in the same pattern."""
    targets = [c for c in cols if c.startswith(prefix)]
    drop: set[str] = set()
    for group in _missing_patterns(df, targets):
        if len(group) < 2:
            continue
        sub = df.loc[df[group[0]].notna(), group].to_numpy(dtype=np.float64)
        sd = sub.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(sub, rowvar=False)
        kept: list[int] = []
        for i, c in enumerate(group):
            if sd[i] == 0:
                continue  # constant columns are dropped later
            if kept and (np.abs(r[i, kept]) > threshold).any():
                drop.add(c)
            else:
                kept.append(i)
    return [c for c in cols if c not in drop]


class Preprocessor:
    """Fit on train with :meth:`fit`; apply to any period with :meth:`transform`."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.state: dict[str, Any] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Output column names, in order."""
        return list(self._fitted()["feature_names"])

    def _fitted(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Preprocessor is not fitted")
        return self.state

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        """Learn every statistic from ``train``. Raises if non-train rows are present."""
        cfg = self.config
        if "split" in train.columns and (train["split"] != "train").any():
            raise ValueError("fit must receive training rows only")
        excluded = set(cfg["exclude"])
        cats = list(cfg["categorical"])
        missing_cats = [c for c in cats + cfg["frequency_columns"] if c not in train.columns]
        if missing_cats:
            raise KeyError(f"configured columns not in data: {missing_cats}")

        candidates = [c for c in train.columns if c not in excluded and c not in cats]
        non_numeric = [c for c in candidates if not pd.api.types.is_numeric_dtype(train[c])]
        if non_numeric:
            raise ValueError(f"non-numeric columns not declared categorical: {non_numeric}")

        miss = train[candidates].isna().mean()
        numeric = [c for c in candidates if miss[c] <= cfg["max_missing_rate"]]
        numeric = _prune_correlated(train, numeric, cfg["prune_prefix"], cfg["prune_corr"])

        log_cols = set(cfg["log1p_columns"])
        num_stats: dict[str, dict[str, float]] = {}
        for c in numeric:
            x = train[c].to_numpy(dtype=np.float64)
            if c in log_cols:
                x = _log1p(x)
            median = float(np.nanmedian(x))
            x = np.where(np.isnan(x), median, x)
            sd = float(x.std())
            if sd > 0:
                num_stats[c] = {"median": median, "mean": float(x.mean()), "std": sd}
        numeric = [c for c in numeric if c in num_stats]

        flag_groups = _missing_patterns(train, [c for c in numeric if miss[c] > 0])

        onehot: dict[str, list[str]] = {}
        for c in cats:
            share = _as_category_strings(train[c]).value_counts() / len(train)
            onehot[c] = sorted(share[share >= cfg["onehot_min_freq"]].index)

        freq: dict[str, dict[str, Any]] = {}
        for c in cfg["frequency_columns"]:
            s = _as_category_strings(train[c]).fillna(MISSING)
            counts = s.value_counts()
            enc = np.log1p(s.map(counts).to_numpy(dtype=np.float64))
            freq[c] = {
                "counts": {str(k): int(v) for k, v in counts.items()},
                "mean": float(enc.mean()),
                "std": float(enc.std()) or 1.0,
            }

        names = list(numeric)
        names += [f"missing__{g[0]}" for g in flag_groups]
        for c, kept in onehot.items():
            names += [f"{c}=={v}" for v in kept] + [f"{c}=={OTHER}", f"{c}=={MISSING}"]
        names += [f"freq__{c}" for c in freq]
        if cfg["hour_of_day"]:
            names += ["hour_sin", "hour_cos"]

        self.state = {
            "numeric": numeric,
            "num_stats": num_stats,
            "flag_groups": flag_groups,
            "onehot": onehot,
            "freq": freq,
            "feature_names": names,
            "n_train": len(train),
        }
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transform. Returns float32 with no NaN or inf."""
        st, cfg = self._fitted(), self.config
        n, clip = len(df), cfg["clip_sd"]
        X = np.empty((n, len(st["feature_names"])), dtype=np.float32)
        log_cols = set(cfg["log1p_columns"])
        j = 0

        for c in st["numeric"]:
            p = st["num_stats"][c]
            x = df[c].to_numpy(dtype=np.float64)
            if c in log_cols:
                x = _log1p(x)
            x = np.where(np.isnan(x), p["median"], x)
            X[:, j] = np.clip((x - p["mean"]) / p["std"], -clip, clip)
            j += 1

        for g in st["flag_groups"]:
            X[:, j] = df[g[0]].isna().to_numpy()
            j += 1

        rows = np.arange(n)
        for c, kept in st["onehot"].items():
            s = _as_category_strings(df[c])
            isna = s.isna().to_numpy()
            codes = pd.Index(kept, dtype="string").get_indexer(s)  # -1: rare, unseen, or NA
            known = (codes >= 0) & ~isna
            block = np.zeros((n, len(kept) + 2), dtype=np.float32)
            block[rows[known], codes[known]] = 1
            block[~known & ~isna, len(kept)] = 1
            block[isna, len(kept) + 1] = 1
            X[:, j : j + block.shape[1]] = block
            j += block.shape[1]

        for c, p in st["freq"].items():
            s = _as_category_strings(df[c]).fillna(MISSING)
            count = s.map(p["counts"]).fillna(0).to_numpy(dtype=np.float64)
            X[:, j] = np.clip((np.log1p(count) - p["mean"]) / p["std"], -clip, clip)
            j += 1

        if cfg["hour_of_day"]:
            angle = 2 * np.pi * (df[TIME_COL].to_numpy() % SECONDS_PER_DAY) / SECONDS_PER_DAY
            X[:, j], X[:, j + 1] = np.sin(angle), np.cos(angle)
            j += 2

        assert j == X.shape[1]
        if not np.isfinite(X).all():
            raise ValueError("transform produced NaN or inf")
        return X

    def save(self, path: str | Path) -> None:
        """Write config and fitted state as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"config": self.config, "state": self._fitted()}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Load a preprocessor written by :meth:`save`."""
        with open(path) as f:
            saved = json.load(f)
        pre = cls(saved["config"])
        pre.state = saved["state"]
        return pre
