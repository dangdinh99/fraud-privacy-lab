# Private Fraud Models Under Realistic Operations

**What does privacy cost a fraud model, and how does that cost change as fraud drifts and models are retrained?**

An applied study of fraud detection with differential privacy (DP), evaluated the way fraud teams work: a fixed review capacity, time-ordered data, dollars at stake, and periodic retraining.

> Status: core project complete (stages 1 to 3, test set used once). Full report: [`docs/writeup.md`](docs/writeup.md). Stages 4 and 5 are stretch goals.

## Research questions

1. **Privacy cost.** At a fixed alert budget, how much fraud recall does a model lose as the privacy budget (epsilon) tightens?
2. **Operations.** How do private models degrade under drift, and how often should they be retrained when every retrain that reuses a transaction spends more of a fixed privacy budget?
3. **Adversaries** (stretch). When fraudsters adapt to a deployed model, do private models break faster?
4. **Collaboration** (stretch). When banks cannot share data, does federated training help the smallest institution?

## Why the framing matters

Most fraud projects report accuracy or ROC-AUC on a random split. Real fraud teams work differently:

- **Alert budget:** analysts review only a small fraction of transactions, so what matters is recall within that budget.
- **Time:** fraud patterns drift, so models are trained on the past, evaluated on the future, and retrained.
- **Money:** missing a $5,000 fraud is worse than missing a $5 one.
- **Privacy:** regulation limits how transaction data can be used.

## Setup

- **Data:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection), about 590k transactions with about 3.5% fraud and anonymized features.
- **Split:** frozen and time-based (train, then validation, then test); no shuffling. Tuning uses validation only; the test set is used once.
- **Metrics:** recall when flagging the top 1% and 2% of transactions (co-primary), always shown with its ceiling (the most any model could catch with that many alerts), plus PR-AUC, dollar loss, calibration error, and recall on cards seen vs not seen in training.
- **Privacy:** transaction-level DP-SGD (Opacus), delta = 1e-6, RDP accountant; the epsilon actually spent is reported, composed across every run that reuses a transaction.
- **Rigor:** 3 random seeds never used for tuning per reported result; every decision logged with date and reason in [`docs/decisions.md`](docs/decisions.md).

## Stages

| # | Stage | Status |
|---|---|---|
| 1 | Baselines: LightGBM and a non-private MLP with the same architecture as the DP model | done |
| 2 | DP sweep: DP-SGD MLP and logistic regression at epsilon 0.5, 1, 2, 4, 8, and none | done |
| 3 | Replay and drift: weekly replay of the test period, retraining schedules under a fixed total privacy budget | done |
| 4 | Adaptive adversary | stretch |
| 5 | Federated learning across simulated banks | stretch |
| 6 | Write-up: [`docs/writeup.md`](docs/writeup.md) | done |

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
src/fraudpriv/   library: data, features, metrics, models (baseline, dp_train), run, replay, drift
configs/         one YAML per experiment
results/         metrics and run records for every reported experiment
notebooks/       EDA and the figures notebook
docs/            write-up, decision log, figures
tests/           unit and end-to-end tests
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt && pip install -e .  # GPU build of torch: see requirements.txt

python scripts/download_data.py                      # checks data/raw/ieee-cis/, prints download steps
python -m fraudpriv.data --build-split               # frozen time split

python -m fraudpriv.run --config configs/baseline_lgbm.yaml           # any config in configs/
python -m fraudpriv.replay --config configs/replay_dev.yaml           # stage 3 replay on validation
python -m fraudpriv.run --config configs/final_mlp.yaml --final       # test set: final configs only

pytest
```

## Limitations

Transaction-level (not card-level) privacy, one dataset with anonymized features, simulated time on a static snapshot, and small effect sizes. The privacy guarantee covers model training only. Details and related work: [`docs/writeup.md`](docs/writeup.md).

## Author

Dang Dinh, MS Data Science, Boston University. [GitHub](https://github.com/dangdinh99) · [Portfolio](https://dangdinh99.github.io/Portfolio/)
