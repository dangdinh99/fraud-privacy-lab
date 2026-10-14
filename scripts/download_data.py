"""Check that the IEEE-CIS raw files are in place; print manual download steps if not."""

import sys

from fraudpriv.data import missing_raw_files

DATA_DIR = "data/raw/ieee-cis"

missing = missing_raw_files(DATA_DIR)
if not missing:
    print(f"All raw files present in {DATA_DIR}/")
    sys.exit(0)

print(f"Missing from {DATA_DIR}/: {', '.join(missing)}")
print(
    "\nTo download:\n"
    "  1. Accept the rules at https://www.kaggle.com/c/ieee-fraud-detection\n"
    "  2. Download the competition data (Data tab, 'Download All')\n"
    f"  3. Unzip train_transaction.csv and train_identity.csv into {DATA_DIR}/\n"
)
sys.exit(1)
