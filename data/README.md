# Data

Raw data is not committed. Only IEEE-CIS is in scope.

| Path | Contents | Committed |
|---|---|---|
| `data/raw/ieee-cis/` | `train_transaction.csv`, `train_identity.csv` from the Kaggle competition `ieee-fraud-detection` | no |
| `data/splits/` | frozen `TransactionID` lists for train, validation, and test, written by `python -m fraudpriv.data --build-split` | summary only: `split_meta.json` is published; the ID list `split.csv.gz` is not (regenerate it; the build is deterministic and should match the counts in `split_meta.json`) |
| `data/processed/` | local caches | no |

Only the Kaggle `train_*` files are used: the competition `test_*` files have no labels.
Run `python scripts/download_data.py` to check the files are in place; it prints the manual download steps if they are not.
