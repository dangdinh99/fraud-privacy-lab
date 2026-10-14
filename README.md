# Private Fraud Models Under Realistic Operations

**What does privacy cost a fraud model, and how does that cost change when fraud drifts and fraudsters adapt?**

This project is an applied empirical study of fraud detection with differential privacy (DP). It measures how much fraud a model catches at a fixed review capacity (the alert budget) as privacy tightens, then tests those private models under time-ordered replay, drift, an adaptive adversary, and a compact federated setting where institutions cannot pool raw data.

> Status: core project complete (stages 1 to 3, including the one-time test-set evaluation), with the full write-up in [`docs/writeup.md`](docs/writeup.md). Stages 4 and 5 are stretch goals.

---

## Research questions

1. **Privacy cost.** At a fixed alert budget, how much fraud recall does a model lose as the DP budget (epsilon) tightens?
2. **Operations.** How do private models degrade over time under drift, and what is the best retraining cadence when every retrain spends privacy budget? Under DP, retraining on data the model has already seen is not free: the privacy loss of every retrain that touches a transaction adds up. Retraining more often fights drift but exhausts a fixed total budget faster, leaving more noise in each model.
3. **Adversaries.** When fraudsters adapt to a deployed model, do private models break faster, and what does it cost in dollars?
4. **Collaboration.** When banks cannot share data, does federated training help the smallest institution?

## Why the framing matters

Most fraud projects report accuracy or ROC-AUC on a random split. Real fraud teams work differently:

- **Alert budget:** analysts can only review a small fraction of transactions, so the metric that matters is recall within that budget.
- **Time:** fraud patterns drift, so models are trained on the past and evaluated on the future.
- **Money:** missing a $5,000 fraud is worse than missing a $5 one, so losses are dollar-weighted.
- **Privacy:** regulation limits how transaction data can be used and shared.

This project evaluates every model under those conditions.

## Setup

- **Data:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle). About 590k transactions, about 3.5% fraud, with heavy missingness and anonymized features.
- **Split:** time-based. Train on the earliest period, validate on the middle, test on the latest. No shuffling.
- **Alert budget:** the fraction of all transactions in the evaluation period that the model flags for review (the top k% by score). This is not the same as a false-positive rate.
- **Primary metrics:** recall at 1% and 2% alert budgets (co-primary). Sensitivity check at 0.5%. Both budgets are reported because the top 1% fills with easy frauds: in stage 1, LightGBM and the MLP tie at 1% (share of ceiling 0.921) but differ clearly at 2% (0.842 vs 0.775). A single 1% budget could hide the cost of privacy.
- **Recall ceiling:** a budget smaller than the fraud rate caps achievable recall (at 1% with about 3.5% fraud, even a perfect model catches at most about 29%). Every recall figure is reported alongside its ceiling and as a share of that ceiling.
- **Secondary metrics:** precision at the budget, PR-AUC, dollar loss (total amount of fraud not flagged within the budget), calibration error.
- **Seen vs new cards:** recall is also broken down by whether a transaction's card (approximated from `card1`, `addr1`, and `D1`) appears in training. Labels in IEEE-CIS cluster by card, so this shows how much performance comes from familiar cards. The card identifier is used for evaluation only, never as a model input.
- **Privacy unit:** transaction-level DP via DP-SGD. All epsilon values use delta = 1e-6, below 1/n for the training set.
- **Cumulative privacy:** when a transaction is used by more than one training run (retraining in stage 3, rounds in stage 5), the reported epsilon is the cumulative value composed across all of those runs, not the per-run value.
- **Seeds:** at least 3 per configuration. Results report mean and spread.

## Stages

| # | Stage | What it answers |
|---|---|---|
| 1 | Baselines | LightGBM with calibration, plus a non-private MLP with the same architecture as the DP model. The non-private references. |
| 2 | DP sweep | DP-SGD MLP (Opacus) and DP logistic regression across epsilon of 0.5, 1, 2, 4, 8, and no DP. |
| 3 | Replay and drift | Score the test period in time order and monitor drift. Compare retraining cadences under a fixed total privacy budget: each retrain's privacy loss composes with every earlier run on the same transactions, so cadence trades recovery from drift against noise per model. |
| 4 | Adaptive adversary (stretch) | Fraudsters shift behavior after deployment. Measure decay and recovery for private vs non-private models. |
| 5 | Federated (stretch) | 3 simulated banks with different fraud rates, formed by a fixed partition rule recorded in `docs/decisions.md`. Local-only vs federated (Flower) vs pooled, at 2 to 3 epsilon values. |
| 6 | Write-up | Report, figures, and conclusions. |

## Results

**The short version.** What differential privacy takes away from a fraud model is mostly its memory of recent, specific patterns, such as cards it has already seen. That memory goes stale within weeks, so a model trained once pays little for privacy after a while. Retraining is what keeps it fresh, and that is where privacy costs the most. **Privacy's real cost is losing the ability to keep a model fresh, not the drop you measure on day one.**

![Privacy erases the benefit of retraining](docs/figures/05_privacy_erases_retraining_gain.png)

Key findings (test period, mean of 3 seeds never used for tuning):

- **Privacy takes away most of the benefit of retraining.** Weekly retraining improves the non-private neural network by +0.026 (share-of-ceiling score) but the DP one by only +0.006 to +0.009, so the private/non-private gap grows from 0.004 to 0.024.
- **DP erases memory of known cards, and that memory fades anyway.** Right after training, DP at epsilon 1 removes 77% of catches on cards seen in training; about 40 days later even the non-private model catches only 6 of 488 such frauds.
- **Under strict privacy, the simple model wins.** At epsilon 1 or less, logistic regression matched or beat the neural network in every evaluation.
- **LightGBM is the strongest model** (0.775 vs about 0.70) and the only one that still recognizes known cards weeks later.
- **Metrics and monitoring can mislead.** A single 1% alert budget hid a real gap between models, and input-drift monitoring stayed flat while performance swung week to week.

![DP erases memory of known cards, but that memory fades](docs/figures/04_card_memory_fades.png)

Full tables, all six figures, the retraining experiment, and caveats (including why DP sometimes scores slightly above non-private training) are in the write-up: [`docs/writeup.md`](docs/writeup.md). Every decision, with its date and reason, is in [`docs/decisions.md`](docs/decisions.md).

## Repository layout

```
fraud-privacy-lab/
  configs/            experiment configs (one YAML per run family)
  data/               raw and processed data (gitignored)
  docs/               scope, decisions log, write-up drafts
  notebooks/          EDA and results figures only
  results/            generated tables, figures, run logs
  scripts/            entry points (download data, run sweeps)
  src/fraudpriv/
    data.py           loading, cleaning, time split
    features.py       shared preprocessing used by every model
    metrics.py        recall at alert budget, dollar loss, PR-AUC
    run.py            config-driven experiment runner
    baseline.py       LightGBM + calibration, non-private MLP
    dp_train.py       DP-SGD (Opacus) and DP logistic regression
    replay.py         time-ordered scoring of the test period
    drift.py          drift monitoring and retraining schedules under a total privacy budget
    adversary.py      adaptive adversary simulation (stretch)
    federated.py      Flower simulation (stretch)
    serve/api.py      FastAPI scoring endpoint (stretch)
  tests/              unit tests (metrics first)
```

## Quickstart

Commands for later stages become available as those stages are completed.

```bash
# 1. Environment (Python 3.11)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
pip install -e .

# 2. Data: checks data/raw/ieee-cis/ and prints manual download steps if files are missing
python scripts/download_data.py

# 3. Freeze the time split
python -m fraudpriv.data --build-split

# 4. Run the baselines (validation only; add --final with eval_split: test for the test set)
python -m fraudpriv.run --config configs/baseline_lgbm.yaml
python -m fraudpriv.run --config configs/baseline_mlp.yaml

# 5. DP sweep, step 1 (validation only; configs use device: cuda, see requirements.txt)
python -m fraudpriv.run --config configs/dp_mlp.yaml
python -m fraudpriv.run --config configs/dp_mlp.yaml --only target_epsilon=1   # one epsilon

# 6. Replay and retraining (stage 3): development replay on validation, then the final one
python -m fraudpriv.replay --config configs/replay_dev.yaml
python -m fraudpriv.replay --config configs/replay_final.yaml --final

# 7. Tests
pytest
```

## Related work

DP and federated learning for fraud detection are active research areas, and this project does not claim a new method. Its contribution is a careful operational evaluation: alert-budget and dollar metrics, time-based splits, multiple seeds, and stress tests under drift and adaptive adversaries. Its most specific question is how retraining cadence should be chosen when retrains compose against a fixed privacy budget. The related-work section and the full report are in [`docs/writeup.md`](docs/writeup.md).

## Limitations

- IEEE-CIS features are anonymized, which limits domain-driven feature engineering.
- The data is a static snapshot, so drift and adversary experiments are simulations built on replay.
- Federated results use simulated institutions, not real banks.
- DP is at the transaction level. A card or customer with many transactions is protected more weakly than a single transaction; card-level DP is out of scope.
- The privacy guarantee covers model training only. Preprocessing statistics (encodings, imputation values) are computed without DP, and hyperparameter tuning is not charged to the privacy budget.
- Platt calibration is fit on validation labels without DP. It is strictly increasing, so recall, precision, and dollar loss do not depend on it; only calibration error does.

## Author

Dang Dinh, MS Data Science, Boston University. [GitHub](https://github.com/dangdinh99) · [Portfolio](https://dangdinh99.github.io/Portfolio/)
