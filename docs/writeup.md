# What Privacy Costs a Fraud Model: An Operational Evaluation of Differential Privacy Under Drift and Retraining

Dang Dinh, October 2026

## Summary

Fraud detection models are trained on sensitive transaction data. Differential privacy (DP) can guarantee that a trained model reveals little about any single transaction, at the price of noise during training. This project measures that price the way a fraud team would experience it: how many frauds are caught when analysts can review only 1% or 2% of transactions, how many dollars slip through, and how this changes as fraud patterns drift and models are retrained.

On the IEEE-CIS fraud dataset, with a frozen time-based split, three findings stand out:

1. **What DP removes is memory of specific, recent patterns.** Right after training, a non-private neural network catches 70 of 763 frauds on cards it saw during training; with DP at epsilon 1 it catches 16 (-77%), while easy frauds are caught just as often.
2. **That memory goes stale within weeks.** About 40 days after training, even the non-private model catches only 6 of 488 such frauds. A model that is trained once therefore pays little for privacy after a few weeks.
3. **Retraining is where privacy really costs.** Without privacy, retraining the neural network weekly improves it by +0.026 (on a share-of-ceiling score); with DP at a total epsilon of 8 the gain is about +0.006 to +0.009. The gap between private and non-private models grows from 0.004 when neither is retrained to 0.024 when both are retrained weekly.

In one line: **privacy's real cost is losing the ability to keep a model fresh, not the drop measured on day one.**

The contribution is a careful operational evaluation, not a new method. Every result uses fresh random seeds that were never used for tuning, the test period was used only for final configurations, and every decision is logged with its date and reason in [`decisions.md`](decisions.md).

## 1. Why the framing matters

Most fraud projects report accuracy or ROC-AUC on a random split. Real fraud teams work under different constraints:

- **Review capacity.** Analysts can review only a small share of transactions, so the metric that matters is how many frauds are in the reviewed set.
- **Time.** Fraud patterns drift. Models are trained on the past and used on the future, and they are retrained periodically.
- **Money.** Missing a $5,000 fraud is worse than missing a $5 one.
- **Privacy.** Regulation and customer expectations limit how transaction data can be used.

This project evaluates every model under all four constraints.

## 2. Data and evaluation

**Data.** IEEE-CIS Fraud Detection (Kaggle, provided by Vesta): about 590,000 card-not-present transactions with about 3.5% fraud, 432 anonymized features, and heavy missingness. Only the labeled training files are used.

**Split.** Transactions are ordered by time and split 60/20/20 by row count: train (354,324 rows, 100 days), validation (118,108 rows, 40 days, starting the day after training ends), and test (118,108 rows, 42 days, starting about 40 days after training ends). The split is saved as transaction IDs and never changed.

**Alert budget.** A model flags the top k% of transactions in the evaluation period by score. Exactly ceil(k x n) transactions are flagged, with score ties broken by transaction ID. Recall at 1% and 2% are co-primary metrics; 0.5% is a sensitivity check. The 2% budget was promoted to co-primary in stage 1, before any private model existed, after LightGBM and the neural network tied at 1% but clearly differed at 2%.

**Recall ceiling.** When the budget is smaller than the number of frauds, even a perfect model cannot catch them all. The ceiling is min(1, flagged / frauds), computed per evaluation period (0.256 at 1% on validation, 0.291 on test). Every recall is reported with its share of the ceiling. When the ceiling is below 1, the share of ceiling equals precision.

**Other metrics.** PR-AUC, dollar loss (total amount of unflagged fraud), expected calibration error with 10 equal-width bins, and recall by card segment. A rough card identifier (card1, address, and the card's apparent start day) is used **for evaluation only** to split frauds into cards seen in training, new cards, and transactions without a usable key (mostly missing address).

**Selection score.** Model settings are chosen on validation by the average share of ceiling at 1% and 2%, averaged over seeds.

**Seeds.** Searches use seed 0 or seeds 0 to 2; every reported number uses fresh seeds 10 to 12, never used for selection, with mean and standard deviation.

## 3. Models and training

**Shared preprocessing**, fit on the training period only and identical for every model: numeric columns more than 90% missing are dropped; near-duplicate V columns are pruned (|r| > 0.95 within a missingness pattern); median imputation with one missing-value flag per missingness pattern; log1p on amounts and count columns; standardization clipped at 5 standard deviations; one-hot categories with a 1% frequency floor; frequency encoding of four high-cardinality columns; hour of day. The result is 632 inputs. Card identity, past fraud labels, and target encoding are excluded as model inputs.

**Models.**

- LightGBM (127 leaves, early stopping on validation): the strong non-private reference.
- MLP (632 -> 128 -> 64 -> 1, ReLU) trained with Adam, Poisson-sampled batches of expected size 1,024, a fixed number of epochs, and cosine learning-rate decay. The same architecture, sampling, optimizer, and schedule are used with and without DP, so the two differ only in gradient clipping and noise.
- Logistic regression: the same code with no hidden layers.

**DP training.** DP-SGD via Opacus with transaction-level privacy, delta = 1e-6 (below 1/n_train), Poisson sampling, and the RDP accountant. Noise is calibrated to each target epsilon in {0.5, 1, 2, 4, 8}; the epsilon actually spent is verified after training and never exceeded the target. Secure random number generation is off, which is standard for research. Preprocessing and hyperparameter tuning are not charged to the privacy budget.

**Tuning effort.** Each model and epsilon got its own search over clipping norm, learning rate, epochs, and batch size, extended past the grid edge whenever the best setting sat on it, and stopped when the gain fell below seed noise (about 0.004). In total: 166 DP MLP settings, 186 DP logistic regression settings, and matching extensions for the non-private models. The non-private MLP grid was also extended, so that an under-tuned baseline would not understate the cost of privacy.

**Calibration.** Platt scaling fit on validation. It is strictly increasing, so it changes no ranking metric; ranking metrics use raw scores and calibration error uses the Platt probabilities.

**Compute.** One RTX 4060 Ti GPU. For DP training on GPU, batches are sampled on the device with the same Poisson rate Opacus uses, which made a DP epoch about 10 times faster than on CPU without changing the privacy accounting.

## 4. Results

### 4.1 Baselines: LightGBM's edge is recognizing known cards

LightGBM is the strongest model in both periods (selection score 0.878 on validation and 0.775 on test, against 0.848 and 0.697 for the MLP). A shuffled-label check gave a precision of 0.060 at the 1% budget, close to the 3.9% base rate, so its strong validation numbers are not leakage.

The card segments show where its lead comes from. On new cards the two models are level in both periods (recall about 0.22 on validation, 0.20 on test). On cards seen during training, LightGBM catches far more (0.27 vs 0.09 on validation, 0.16 vs 0.01 on test). Trees can recombine the card, address, and day columns to recognize cards even though no card identifier is a model input. LightGBM is the only model that still recognizes known cards weeks later.

![LightGBM still recognizes known cards weeks later](figures/02_lightgbm_lead_is_card_memory.png)

A metric lesson came out of this stage. Right after training, the MLP tied LightGBM at the 1% budget (share of ceiling 0.92 vs 0.92), because the top 1% fills with easy frauds, mostly transactions without an address. Only the 2% budget separated them. On the later test period the 1% budget separates them too (0.78 vs 0.88). A single small budget can hide real differences between models.

![The 1% budget can hide differences](figures/01_budget_hides_differences.png)

### 4.2 The static cost of privacy

On validation, DP costs the MLP 0.019 (epsilon 8) to 0.059 (epsilon 0.5) on the selection score, and logistic regression 0.004 to 0.021. Under strict privacy (epsilon 1 or less) DP logistic regression beats the DP MLP; the MLP pays off only from epsilon 4. Two diagnostics clarify where the cost comes from:

- **Clipping alone costs nothing on validation.** Training with DP's gradient clipping but no noise matches non-private training (MLP 0.847 vs 0.848; logistic regression 0.825 vs 0.825). The cost is the noise.
- **A narrower network does not help.** A 64 -> 32 MLP at epsilon 1 scores 0.803 vs 0.807 for 128 -> 64.

The most telling diagnostic is where the lost catches are. At the 2% budget, DP at epsilon 1 leaves catches without a card key unchanged (1,331 to 1,339), cuts new-card catches by 22% (431 to 336), and removes 77% of seen-card catches (70 to 16). With clipping alone and the same training, the model catches 174 seen-card frauds. The noise specifically removes catches that rely on remembering cards, which is what DP is designed to prevent.

On the later test period the picture shifts:

| Model | Epsilon | Share of ceiling, 1% | Share of ceiling, 2% | PR-AUC |
|---|---|---|---|---|
| LightGBM | none | 0.875 | 0.674 | 0.508 |
| MLP | none | 0.780 | 0.613 | 0.445 |
| MLP + DP | 8 | 0.812 | 0.595 | 0.425 |
| MLP + DP | 1 | 0.813 | 0.587 | 0.412 |
| MLP + DP | 0.5 | 0.810 | 0.579 | 0.401 |
| Logistic regression | none | 0.801 | 0.593 | 0.433 |
| Logistic regression + DP | 8 | 0.818 | 0.600 | 0.438 |
| Logistic regression + DP | 0.5 | 0.803 | 0.583 | 0.418 |

Privacy still costs the MLP recall at the 2% budget and in PR-AUC, increasingly as epsilon shrinks, and under DP the simple model does as well as the MLP at every epsilon. But the non-private MLP has lost its edge over logistic regression (selection score 0.697 vs 0.697), and the memorization that DP removed on validation has mostly faded: the non-private MLP catches only 6 of 488 seen-card frauds on test.

![Privacy cost vs epsilon](figures/03_privacy_cost_vs_epsilon.png)

![DP erases memory of known cards, but that memory fades](figures/04_card_memory_fades.png)

### 4.3 Retraining under a fixed privacy budget

The test period was replayed week by week. Each week is scored by the most recent model. Models are retrained never, or every 3, 2, or 1 weeks, and a retrain at time t uses only labels older than 14 days (fraud labels arrive late through chargebacks). Two strategies spend a fixed total privacy budget:

- **Expanding window:** retrain from scratch on all labeled data so far. Retrains reuse earlier transactions, so their privacy losses add up. One noise level is calibrated for the whole plan so that the composed epsilon (RDP) stays within the total, and the total is recomputed from what actually ran.
- **Disjoint windows:** fine-tune the previous model on only the newly labeled data. No transaction is used twice, so every retrain gets the full budget; reusing the previous model costs no privacy.

The retraining design and a fine-tuning setting (warm-start runs use a quarter of the epochs) were fixed on a validation replay before the single test replay. All privacy totals stayed within budget.

| Schedule | Logistic regression | DP LR, total eps 8 | DP LR, total eps 2 | MLP | DP MLP, total eps 8 |
|---|---|---|---|---|---|
| Never retrain | 0.702 | 0.707 | 0.704 | 0.712 | 0.708 |
| Expanding, every 3 weeks | 0.708 | 0.715 | 0.709 | 0.732 | 0.714 |
| Expanding, every 2 weeks | 0.709 | 0.715 | 0.708 | 0.735 | 0.717 |
| Expanding, weekly | 0.712 | 0.717 | 0.707 | 0.738 | 0.714 |
| Disjoint, every 3 weeks | 0.694 | 0.710 | 0.701 | 0.720 | 0.710 |
| Disjoint, every 2 weeks | 0.691 | 0.713 | 0.705 | 0.725 | 0.709 |
| Disjoint, weekly | 0.685 | 0.712 | 0.702 | 0.724 | 0.708 |

Scores are the selection score pooled over weeks, mean of 3 seeds; seed spread is about 0.005.

- **Privacy takes away most of the benefit of retraining the neural network.** Without privacy, weekly retraining gains +0.026; with DP at a total epsilon of 8 the best schedule gains +0.009 and weekly gains +0.006.
- **The private-vs-non-private gap grows with retraining**, from 0.004 when neither model is retrained to 0.024 when both are retrained weekly. Measuring privacy on a model trained once understates its operational cost.
- **The simple model keeps its retraining benefit at a loose budget** (+0.011 at total epsilon 8, as without privacy) but loses half of it at total epsilon 2, where retraining less often scores marginally higher (within noise).
- **Retrain on everything, not only new data.** The expanding strategy beats the disjoint one for every model. A full budget per retrain does not make up for training on only one or two weeks of data.

![Privacy erases the benefit of retraining](figures/05_privacy_erases_retraining_gain.png)

**Drift monitoring.** The population stability index (PSI) of every non-binary input, against the training data, stayed tiny every week (mean 0.004 to 0.010; almost no feature above the common 0.25 alarm level), and so did the PSI of model scores. Meanwhile the weekly share of ceiling swung between about 0.50 and 0.75. Input drift monitoring would not have told a team when its model was struggling; the relationship between features and fraud changed, not the features themselves.

![Drift monitoring missed the performance swings](figures/06_drift_monitoring_missed_it.png)

### 4.4 A caveat: DP sometimes scores above non-private training

On the test period, DP logistic regression scored slightly above non-private logistic regression (for example 0.707 vs 0.702 never retrained in the replay). Following the rule to suspect a bug first, two post-hoc checks were run with their reading rules stated in advance:

1. Matching the non-private learning rate to the DP setting did not close the gap (DP still ahead on all 7 schedules).
2. Training with DP's clipping but zero noise scored above both: clipping alone added +0.017 over non-private training, and the noise then cost about 0.008, on all 7 schedules.

DP's small edge on test therefore comes from gradient clipping, which limits the influence of individual outlier transactions and helps under the test period's drift. It is not a benefit of privacy, and it did not appear on validation, where clipping alone changed nothing. Post-hoc analysis of the test set stopped there.

## 5. What this means for a fraud team

- **Expect privacy to cost more than a one-off evaluation suggests.** Measure it with the retraining schedule you will actually run.
- **Under strict privacy, prefer simple models.** At epsilon 1 or less, logistic regression matched or beat a neural network in every evaluation here.
- **Retrain on all labeled data, not only on fresh windows**, even though fresh windows would allow a larger privacy budget per retrain.
- **Report more than one alert budget.** A single small budget can make different models look identical.
- **Do not rely on input drift alone to decide when to retrain.** Performance moved while inputs did not.
- **Know where a strong model's edge comes from.** Here the best model's lead came largely from recognizing cards it had already seen.

## 6. Limitations

- **One dataset with anonymized features.** Results may not transfer to other institutions or fraud types, and anonymization limits domain-driven features.
- **Transaction-level privacy.** A card with many transactions is protected more weakly than a single transaction; card-level privacy is out of scope.
- **What the guarantee covers.** Only model training. Preprocessing statistics, hyperparameter tuning, and Platt calibration (fit on validation labels) are not private. Calibration does not affect any ranking metric.
- **Simulated time.** The replay is a simulation on a static snapshot with an assumed 14-day label delay, not a live deployment. The test period is only 42 days, which limits how many retrains can be compared.
- **Small effects.** Many differences are 0.005 to 0.03 on the score. Differences within about 0.005 (one seed standard deviation) are reported as ties or as suggestive, not as findings.
- **The card key is approximate.** Seen and new card segments rely on a reconstructed identifier; transactions without an address cannot be assigned.
- **Two periods.** Validation follows training immediately and test starts about 40 days later, so the comparison between them mixes time since training with differences between the periods themselves.

## 7. Related work

DP-SGD was introduced by Abadi et al. (2016, "Deep Learning with Differential Privacy"), and Rényi differential privacy (Mironov, 2017) underlies the accountant used here; the implementation is Opacus (Yousefpour et al., 2021). Tramèr and Boneh (2021, "Differentially Private Learning Needs Better Features (or Much More Data)") showed that simple models on good features often beat deep networks under DP, consistent with the logistic regression results here. Dwork and Roth (2014) give the standard treatment of differential privacy, including composition. Privacy-preserving and federated learning for fraud detection have been studied before; this project does not propose a new method. Its contribution is an operational evaluation: alert-budget and dollar metrics, a time-based split, fresh-seed reporting, and an explicit treatment of how privacy budgets compose across retraining.

## 8. Reproducing the results

All runs are driven by YAML configs in `configs/` and write the resolved config, environment, fitted preprocessor, and per-seed metrics to `results/`. The commands are in the README quickstart; tests run with `pytest`. The full decision record, with dates, reasons, and corrections, is [`decisions.md`](decisions.md). Figures are produced by `notebooks/02_figures.ipynb` from saved results only.

## 9. Future work

- **Federated training across simulated banks** (planned stage 5): does collaboration help the smallest institution under privacy, and how does privacy compose over federated rounds?
- **Card-level privacy**, which protects customers rather than single transactions.
- **Clipping as robustness** for non-private fraud models under drift, suggested by the post-hoc result in section 4.4.
- **Retraining schedules that adapt to performance** rather than fixed calendars, given that input drift did not track performance.
