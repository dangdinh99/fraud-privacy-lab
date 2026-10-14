# CLAUDE.md

Guidance for Claude Code working in this repository. Read this before making changes. `README.md` is the source of truth for research questions, metrics, and stages; if this file and the README ever disagree, stop and ask.

## Project in one paragraph

An applied empirical study of fraud detection under differential privacy. Every model is evaluated on a frozen time-based split at fixed alert budgets (the top 1% and 2% of transactions in the evaluation period, ranked by score; both are co-primary), with recall reported alongside its ceiling. The core project is stages 1 to 3: baselines, DP sweep, replay and drift. Stages 4 and 5 (adaptive adversary, federated) are stretch goals. The quality and fairness of the comparison matter more than model sophistication.

## Current stage

Core project and write-up complete (stages 1 to 3 and the one-time test-set evaluation, 2026-10-13; write-up in `docs/writeup.md`, 2026-10-14). Stretch stages 4 and 5 not started; do not start them without the user's go-ahead. Update this line when a stage is finished.

Stages (do not reorder or start a stretch stage without asking):
1. Baselines: LightGBM with calibration, plus a non-private MLP with the same architecture as the DP model
2. DP sweep: DP-SGD MLP (Opacus) and DP logistic regression, epsilon in {0.5, 1, 2, 4, 8, none}
3. Replay and drift: time-ordered scoring of the test period, drift monitoring, retraining cadences compared under a fixed total privacy budget (retrains compose)
4. Adaptive adversary (stretch)
5. Federated (stretch): 3 simulated banks, 2 to 3 epsilon values, partition rule recorded in `docs/decisions.md`
6. Write-up

## Commands

```bash
python -m venv .venv
source .venv/bin/activate              # macOS / Linux
# .\.venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt        # pinned versions only
pip install -e .                       # installs src/fraudpriv as a package

python scripts/download_data.py        # IEEE-CIS into data/raw/
python -m fraudpriv.data --build-split # writes frozen split indices to data/splits/
python -m fraudpriv.run --config configs/<name>.yaml
pytest                                 # run before every commit
ruff check . && ruff format .          # lint and format
```

Commands for later stages are added as those stages are built.

## Metric definitions (implement exactly these, in `metrics.py` only)

- **Alert budget:** the fraction of all transactions in the evaluation period flagged for review, i.e. the top k% by score. It is not a false-positive rate. Co-primary k = 1% and 2%; sensitivity check at 0.5%. Model selection ranks by the mean over seeds of the average share of ceiling at 1% and 2% (on validation).
- **Ties at the cutoff:** flag exactly `ceil(k * n)` transactions; break score ties deterministically (stable sort by score, then by `TransactionID`). Document this in the docstring.
- **Recall at budget:** frauds flagged within the budget divided by all frauds in the evaluation period.
- **Recall ceiling:** `min(1, ceil(k * n) / n_fraud)` (the number actually flagged, so recall can never exceed it), computed from the actual evaluation period, never from the overall 3.5% rate. Always report recall together with its ceiling and as a share of the ceiling.
- **Precision at budget:** frauds flagged divided by transactions flagged.
- **Dollar loss:** total `TransactionAmt` of fraud transactions not flagged within the budget.
- **Secondary:** PR-AUC and calibration error (state the method, e.g. ECE with fixed bins, in the docstring).
- **Seen vs new cards:** report recall within `seen_card`, `new_card`, and `no_key` segments (from `data.card_segments`). Flags come from the whole-period budget; segments get no ceiling of their own. `card_key` is for evaluation only and must never become a model input.
- Never report accuracy.

## Non-negotiable rules for evaluation

Breaking any of these invalidates results.

1. **Frozen time split.** Use the saved indices in `data/splits/`. Never reshuffle, resplit, or create a random split.
2. **No leakage.** Preprocessing (imputation, encoding, scaling) is fit on train only, inside `features.py`, and is identical across all models. Watch for features computed from future transactions.
3. **One metrics module.** All metrics come from `src/fraudpriv/metrics.py`. Do not reimplement them elsewhere. Any change to `metrics.py` requires updated tests in `tests/test_metrics.py`, including hand-computed cases for the ceiling, ties, and dollar loss.
4. **Validation for selection, test for final only.** Model selection, hyperparameters, and calibration are chosen on validation. The test set is evaluated only for final configs.
5. **Fair DP comparison.** The non-private MLP and the DP-SGD MLP share architecture, preprocessing, and feature set. Only the training procedure (clipping and noise) differs. LightGBM is a reference, not the privacy-cost comparison.
6. **Seeds.** Every reported result uses at least 3 seeds, set in config and logged, with mean and standard deviation reported.
7. **Honest privacy accounting.**
   - Privacy unit is the transaction (transaction-level DP).
   - delta = 1e-6 for all runs. Assert at runtime that delta < 1 / n_train.
   - Report the epsilon actually spent according to the Opacus accountant, never the target alone.
   - Log clipping norm, noise multiplier, sample rate or batch size, epochs, accountant type, and final epsilon for every DP run.
   - In federated runs, epsilon accumulates across rounds; report the cumulative value.
   - In retraining runs (stage 3), epsilon accumulates across every retrain that uses a transaction; report the cumulative value against the fixed total budget. Compose all retrains on overlapping data. Parallel composition (no accumulation) applies only to training windows that are truly disjoint.
   - Preprocessing statistics and hyperparameter tuning are not charged to the privacy budget. Do not claim otherwise in code comments, logs, or docs.

## Code conventions

- Python 3.11, type hints on public functions, docstrings on modules and public functions.
- Library code lives in `src/fraudpriv/`. Scripts and CLIs only call library functions.
- Notebooks are for EDA and figures only. Anything run more than once or trusted for results belongs in `src/`.
- Experiments are driven by YAML in `configs/`. No hardcoded hyperparameters in library code.
- Each run writes to `results/<stage>/<run_id>/` with the resolved config, metrics JSON (including recall ceiling and share of ceiling), and environment info (package versions, git commit).
- Keep dependencies minimal: pandas, numpy, scikit-learn, lightgbm, torch, opacus, pyyaml, pytest, ruff. flwr and fastapi only when their stretch stages begin. Ask before adding anything else.
- Pin versions in `requirements.txt`. Opacus, torch, and Flower are version-sensitive; do not upgrade them casually.

## Scope guardrails

Out of scope unless the user explicitly asks:
- Card-level or user-level DP
- Private preprocessing or private hyperparameter tuning
- Graph methods or GNNs
- Synthetic data generation
- Membership inference attacks
- New languages, new frameworks, dashboards, or cloud deployment
- Datasets beyond IEEE-CIS (PaySim is a possible later stretch)

Do not start stages 4 or 5, or build the FastAPI endpoint, until the user says stages 1 to 3 are done. If a task seems to need something out of scope, stop and ask.

## How to work

- Before a non-trivial change, state a short plan (files to touch, what changes, how it is tested) and wait for approval.
- Make small, focused changes. One concern per commit, with a clear message.
- Write tests alongside code, starting with `metrics.py`, `data.py`, and `features.py`.
- When results look surprising (DP beating non-private, recall above its ceiling, sudden jumps between seeds), suspect a bug first: check leakage, split misuse, threshold or calibration fit on test, and accountant settings.
- Never delete files in `data/` or `results/` without asking. Never commit data, credentials, or Kaggle API keys.
- Record every choice that affects results in `docs/decisions.md` with the date and reason: preprocessing, MLP architecture, hyperparameter grids, epsilon grid, drift windows, retraining cadences, the total privacy budget for stage 3 and how it is allocated across retrains, and the federated partition rule.
- If you change anything that affects the README (metrics, stages, commands, results), update the README in the same change.

## Known pitfalls

- IEEE-CIS has two files (transaction and identity) joined on `TransactionID`; many transactions have no identity row. `TransactionDT` is a time offset in seconds, not a timestamp.
- Fraud is about 3.5% overall but varies by period, so the recall ceiling must be computed per evaluation period.
- Heavy missingness: keep imputation simple, fit on train, and apply identically everywhere.
- DP-SGD on tabular data is sensitive to clipping norm, learning rate, and batch size. Tune within the documented grid before concluding that private performance is weak; weak performance is still a valid finding.
- Opacus requires compatible layers (no BatchNorm; use GroupNorm or LayerNorm if normalization is needed). Use Opacus's validator before training.
- Opacus uses Poisson sampling by default; keep it enabled so the accountant's guarantee holds.
