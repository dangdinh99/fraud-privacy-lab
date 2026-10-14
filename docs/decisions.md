# Decisions log

Every choice that affects results, with the date and the reason. Newest at the bottom.

## 2026-10-12: Environment

- Python 3.11.9. Versions pinned in `requirements.txt` (torch 2.14.1, opacus 1.6.0). Smoke test passed: Opacus validator, Poisson sampling, and the RDP accountant work with this torch version.
- matplotlib and jupyter are kept for figures and EDA notebooks only. No library code imports them.
- The `kaggle` package is not a dependency. `scripts/download_data.py` checks for the raw files and prints manual download steps.

## 2026-10-12: Stage 3 framing

- Stage 3 compares retraining cadences under a fixed total privacy budget. A retrain that reuses transactions composes with every earlier run on them, so the reported epsilon is cumulative. Parallel composition is used only for truly disjoint training windows.
- To decide when stage 3 starts: the total budget, how it is allocated across retrains, the training windows, and the accountant used for composition.

## 2026-10-12: Scope trimmed to one study

- Removed the earlier `projects/` layout, including the Elliptic graph project (graph methods and datasets beyond IEEE-CIS are out of scope).

## 2026-10-12: Metric implementation details

- Recall ceiling uses `ceil(k * n)` in the numerator, not `k * n`, because `ceil(k * n)` transactions are flagged. With `k * n`, a perfect model could exceed its ceiling on small periods (n=150, k=1%, 2 frauds: recall 1.0 vs ceiling 0.75). At IEEE-CIS test sizes the difference is under 0.1%.
- `ceil(k * n)` is computed exactly from the decimal form of k, avoiding float errors such as `ceil(0.07 * 100) = 8`.
- Ties at the cutoff: rank by score descending, then by `TransactionID` ascending.
- Calibration error: ECE with equal-width bins on [0, 1]; the bin count is passed from config. Default to use: 10 bins.

## 2026-10-12: Frozen time split

- 60 / 20 / 20 by row count, ordered by `TransactionDT` then `TransactionID`. Only the Kaggle `train_*` files are used (the competition test set has no labels). Config: `configs/split.yaml`; output: `data/splits/split.csv.gz` and `split_meta.json` (committed).
- Reason: row-count fractions give equal-size validation and test periods; calendar-month boundaries would be guesses, since `TransactionDT` is an offset with an unknown start.
- Result: train 354,324 rows (100.2 days, fraud 3.38%), val 118,108 (39.9 days, 3.90%), test 118,108 (41.9 days, 3.44%). Recall ceiling at 1%: 0.296 / 0.256 / 0.291.
- Note for stage 3: the test period is about 6 weeks, which limits how many retrains a cadence comparison can include.

## 2026-10-12: Card-level features

- Not allowed as model inputs: a rebuilt card ID (`card1` + `addr1` + day - `D1`), the "day - D1" column itself, features built from past fraud labels, and target encoding.
- Reason: 52.5% of repeat cards with any train fraud are fraud on every transaction. This is largely the dataset's labeling rule (later transactions linked to a fraudulent account are also labeled fraud). A model that memorizes cards would look better than it is, and the effect would confound the DP comparison (transaction-level DP does not stop card-level memorization in the same way). Past-label features would also assume fraud labels arrive instantly; in practice they arrive weeks later through chargebacks.
- `card1` enters only as a one-hot with a 1% frequency floor and as a frequency encoding (see the features entry).
- Deferred: card behavior aggregates computed from past transactions only (counts and amounts over windows, no labels). A possible later ablation.
- Evaluation: recall is reported for `seen_card`, `new_card`, and `no_key` segments, with the card key used for evaluation only.
- Validation breakdown (corrected 2026-10-12): seen cards are 40.4% of validation rows but only 16.5% of validation frauds (fraud rate 1.6%); new cards 46.4% of rows, 40.9% of frauds (3.4%); no key (mostly missing `addr1`) 13.2% of rows, 42.5% of frauds (12.5%). An earlier EDA figure (54% of rows, 59% of frauds seen) was a bug: the notebook's key was NaN for rows missing `addr1` or `D1`, and `isin` matched those NaN keys to each other, counting every keyless row as seen. The decision stands on the labeling rule and the DP confound, but the room for card memorization is smaller than first estimated.

## 2026-10-12: Shared preprocessing (`features.py`, `configs/features.yaml`)

- Same matrix for every model, LightGBM included (CLAUDE.md rule 2). Fit on train only.
- Numeric columns with more than 90% missing on train are dropped; columns constant on train are dropped.
- `V` columns pruned within each missingness pattern at |r| > 0.95 (339 to 279). 0.95 removes only near-duplicates; 0.9 (228 kept) would cut about 50 more inputs but reduce DP noise only slightly. Possible check later: compare both on validation with LightGBM.
- Missing values: train median; one missing-value flag per missingness pattern among kept numeric columns (30 flags).
- `log1p` on `TransactionAmt` and `C1` to `C14`, then standardize and clip to +/- 5 train SD (limits extreme inputs, which matters for DP-SGD clipping).
- Categoricals (per the Kaggle data description): one-hot for values with at least 1% of train rows, plus `other` (rare or unseen) and `missing` columns. The floor is a share of all train rows. No target encoding (leakage risk, and it would compute label statistics outside DP).
- Frequency encoding (log of 1 + train count) for `card1`, `card2`, `addr1`, `DeviceInfo`.
- Hour of day as sine and cosine of `TransactionDT mod 86400`; `TransactionDT` itself is excluded.
- Result on real data: 632 inputs (319 numeric, 30 flags, 277 one-hot, 4 frequency, 2 hour). Fit takes 5 s; transforming train and validation takes 8 s; the train matrix is 0.9 GB float32. Largest `other` shares: `card1` 86% (as intended: card identity stays out), `card2` 36 to 38%, `id_31` 7% train vs 11% val (browser versions drift).

## 2026-10-12: Stage 1 baselines

- Calibration: Platt scaling (`sigmoid(a * score + b)`, a > 0) fit on validation. Isotonic was rejected because its step function creates many ties at the budget cutoff. Platt is strictly increasing, so recall, precision, and dollar loss are unchanged by calibration. Ranking metrics use raw scores; ECE uses the Platt probabilities. ECE on validation is in-sample and labeled so; the honest ECE comes from the final test run.
- Test-set guard: `eval_split: test` runs only with `--final`. Models train on train; early stopping and calibration use validation, also in final runs.
- MLP, shared with the stage 2 DP-SGD model: 632 -> 128 -> 64 -> 1, ReLU, no BatchNorm or other normalization (inputs are standardized). About 89k parameters, 91% in the first layer. Chosen as a middle ground: DP-SGD noise grows with parameter count, while width matters little for a non-private MLP on tabular data. Planned stage 2 sensitivity check: 64 -> 32 at one epsilon.
- MLP training, identical for both MLPs: Adam, unweighted binary cross-entropy, Poisson sampling with expected batch size 1,024 (Opacus `DPDataLoader`, also for the non-private MLP), a fixed number of epochs with cosine decay of the learning rate to zero, no early stopping (under DP, privacy cost depends on the planned steps). The schedule costs no privacy and adds no hyperparameters. `ReduceLROnPlateau` rejected: it reads validation during training.
- LightGBM: same 632-column matrix; early stopping on validation average precision (patience 100, at most 3,000 rounds); deterministic mode, 8 threads. Fixed: learning rate 0.05, `min_child_samples` 100, feature fraction 0.5, bagging 0.8.
- Grids: LightGBM `num_leaves` in {31, 127}; MLP learning rate in {1e-3, 3e-4} x epochs in {5, 10}. Selection: mean share of ceiling at the 1% budget on validation over seeds 0, 1, 2. Standard deviations are sample standard deviations (ddof = 1).

## 2026-10-12: Co-primary budgets (1% and 2%)

- Recall at 1% and 2% are both primary metrics; 0.5% stays a sensitivity check. Decided before any DP results exist, so the metric is not chosen after seeing which one shows a larger privacy cost.
- Reason: in the first stage 1 runs (validation, 3 seeds), LightGBM and the non-private MLP tied at 1% (share of ceiling 0.921 +/- 0.003 each) but differed at 2% (0.842 vs 0.775) and in PR-AUC (0.629 vs 0.562). The top 1% fills with easy frauds (mostly transactions without an address), so 1% alone could make DP models look free of cost.
- Note: when the ceiling is below 1, share of ceiling equals precision at that budget (frauds caught / transactions flagged). True at all three budgets here, so the two are not separate evidence.
- Selection rule: per seed, the average of share of ceiling at 1% and 2%; grid points ranked by the mean over seeds, on validation.

## 2026-10-12: Stage 1 results (validation) and selected configs

- Runs: `results/stage1/baseline_lgbm_20261012-145701`, `results/stage1/baseline_mlp_20261012-150714` (code at 8909546; `git_dirty` is true only because of uncommitted notebook outputs). A repeat run reproduced every metric exactly.
- Selected: LightGBM `num_leaves=127` (select score 0.8815 +/- 0.0010); MLP `lr=3e-4, epochs=10` (0.8482 +/- 0.0030). These carry into stage 2 and the final test runs.
- Validation, mean of 3 seeds: share of ceiling at 1% is 0.921 for both; at 2%, LightGBM 0.842 vs MLP 0.775; PR-AUC 0.629 vs 0.562.
- LightGBM's lead is from seen cards: recall at 2% on seen cards 0.301 vs 0.098, while new cards (0.219 vs 0.223) and keyless rows (0.686 vs 0.681) are level. Trees can recombine `card1`, `addr1`, and `D1` to recognize cards without an explicit card ID. Report LightGBM's edge as card recognition, not better general detection.
- Sanity check: LightGBM trained on shuffled labels gives precision 0.060 at 1% (base rate 0.039), so the 0.92 is real signal, not leakage.
- Earlier runs (`baseline_lgbm_20261012-143450`, `baseline_mlp_20261012-144431`) used the 1%-only selection rule; kept locally, not committed.

## 2026-10-12: Extended MLP grid; stage 1 complete

- The first MLP winner (lr 3e-4, 10 epochs) sat at the corner of its grid, so the grid was extended to lr {3e-4, 1e-4} x epochs {10, 20} (`configs/baseline_mlp_ext.yaml`, `results/stage1/baseline_mlp_ext_20261012-152249`). The repeated point reproduced the first grid exactly.
- Result: performance plateaus. lr 3e-4 / 10 epochs stays best (select score 0.8482 +/- 0.0030); lr 1e-4 / 20 epochs ties (0.8475 +/- 0.0046); lr 3e-4 / 20 epochs is worse (0.8436): lower training loss (0.056 vs 0.072) but lower PR-AUC (0.549 vs 0.562), i.e. overfitting. The best point is now inside the explored region. Selected MLP: lr 3e-4, 10 epochs.
- Longer training raises seen-card recall at 2% (0.098 to 0.181 at lr 3e-4) while overall quality drops: the extra fit goes into recognizing cards from train.
- For stage 2: the DP models should get a comparable tuning effort, or the privacy comparison tilts the other way.
- Stage 1 (baselines) is complete on validation. Test-set evaluation of final configs is deferred to the end of the project (rule 4).

## 2026-10-12: Stage 2 design (DP sweep)

- Privacy unit: one transaction. delta = 1e-6, asserted < 1 / n_train (1 / 354,324 = 2.8e-6).
- Accountant: RDP. Chosen over PRV (slightly tighter) because it is conservative and its privacy losses add simply across runs, which stage 3 needs for retrains. At every grid setting the optimal RDP order lies inside Opacus's default range (2.9 to 39 within 1.1 to 63), so the bound is as tight as RDP allows. Opacus's "optimal order is the largest alpha" warnings come from intermediate steps of its noise search, not from the final values.
- Noise: `make_private_with_epsilon` calibrates the noise multiplier to the target epsilon over the planned steps; runs fail if the spent epsilon exceeds the target. Reported epsilon is the accountant's spent value. Poisson sampling on; secure RNG off (research setting, listed as a limitation). Every run logs accountant, target and spent epsilon, delta, n_train, noise multiplier, clipping norm, sample rate, expected batch size, epochs, and steps.
- Fairness (rule 5): architecture, features, and preprocessing are identical; each model gets its own tuned learning rate, epochs, and batch size, plus clipping norm for DP. Freezing the non-private hyperparameters for DP would handicap DP-SGD.
- Step 0 measurements (non-private training, per-example gradient norms on 4,096 train rows): MLP median 1.36 at start, 0.18 after 1 epoch, 0.14 after 10 (99th percentile 26); logistic regression 6.79, 1.15, 0.19 (99th percentile 15). Clipping grid {0.05, 0.2, 1.0} is about 1/4x, 1x, 5x the trained median; 1.0 also covers logistic regression's larger early gradients. One DP-MLP epoch takes about 46 to 74 s on CPU (fastest with 4 threads per process).
- Step 1 grid (seed 0, per epsilon in {0.5, 1, 2, 4, 8}): clipping {0.05, 0.2, 1.0} x learning rate {1e-3, 3e-3} x epochs {5, 10} x expected batch size {1,024, 4,096}; 24 points per epsilon, for the DP MLP and DP logistic regression. Non-private logistic regression gets the same grid without clipping or noise. The non-private MLP also gets batch 4,096 (lr {3e-4, 1e-3}, 10 epochs, seeds 0 to 2) so DP is not the only model allowed to tune batch size.
- Step 2: the best point per epsilon and model (same selection score) is re-run on fresh seeds {10, 11, 12}, never used in selection, and so are the selected non-private models. Only step 2 numbers are reported.
- Planned diagnostics: clipping with zero noise at the selected settings (splits the cost into clipping and noise); 64 -> 32 architecture at epsilon = 1.
- Tuning and preprocessing are not charged to the privacy budget.

## 2026-10-12: GPU training for stage 2

- Hardware: NVIDIA RTX 4060 Ti (8 GB). torch switched from the CPU build to `2.14.1+cu130` (same version, CUDA 13.0 build, installed with `--no-deps`; `pip check` clean; Opacus unchanged at 1.6.0).
- Speed (one DP epoch, train set): MLP about 50 s on CPU vs 4.3 s (batch 1,024) and 4.6 s (batch 4,096) on GPU; logistic regression 0.3 to 1.5 s on GPU. Step 1 drops from about 5 to 6 hours to about 1.5 to 2.
- Data path on GPU: training data lives on the GPU and each step includes every row independently with probability q (`baseline.DevicePoissonBatches`). This is the same Poisson sampling as Opacus's `DPDataLoader` at exactly Opacus's rate (q = 1 / number of batches), so accounting is unchanged; Opacus still clips, adds noise, and accounts. Opacus's own loader assembles batches row by row on the CPU and was slower on GPU (92 s per epoch) than CPU-only training.
- The non-private MLP and logistic regression use the same device sampler on GPU, so DP and non-private still differ only in clipping and noise.
- Reproducibility: `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; tests confirm identical scores for the same seed on GPU and identical privacy settings (noise multiplier, sample rate, steps, spent epsilon) on GPU and CPU.
- Memory: variable Poisson batch sizes fragment the GPU allocator (reserved memory above 8 GB even though peak use is about 3.8 GB), so the cache is cleared after every run, and only one GPU process runs at a time.
- Device consistency: every reported MLP and logistic regression number comes from GPU runs (step 2 re-runs all of them on fresh seeds). Stage 1's CPU MLP runs were used only for selection. LightGBM stays on CPU. The CPU path is unchanged and stage 1 remains reproducible on it.

## 2026-10-12: Stage 2 search closed; selected settings

- Search (validation, seed 0, GPU): step 1 grid (120 points per DP model), one extension beyond the step 1 corner (`*_ext` configs), and a final round for logistic regression only (`*_ext2`). Totals: DP MLP 166 points, DP logistic regression 186, non-private logistic regression 14. No run spent more than its target epsilon. Repeated points reproduced exactly on GPU.
- Stopping rule: extend only while the best selection score per epsilon improves by more than about 0.004 (seed spread seen in stage 1), at most one round past the first extension. DP MLP stopped after the first extension (gains +0.000 to +0.004). Logistic regression gained up to +0.013 in the first extension and exactly 0 in the final round, so its search is closed.
- Remaining edge: the DP MLP's best settings use 20 epochs (the largest tried) for epsilon >= 1, but the gain over 10 epochs was within noise (at most +0.004), so the search was not extended further. Noted as a caveat.
- Selected (selection score on validation, seed 0):

| Model | epsilon | Clipping | lr | Epochs | Batch | Score |
|---|---|---|---|---|---|---|
| DP MLP | 0.5 | 0.2 | 3e-3 | 10 | 1,024 | 0.7908 |
| DP MLP | 1 | 0.05 | 3e-3 | 20 | 1,024 | 0.8034 |
| DP MLP | 2 | 0.2 | 3e-3 | 20 | 1,024 | 0.8153 |
| DP MLP | 4 | 0.05 | 3e-3 | 20 | 1,024 | 0.8227 |
| DP MLP | 8 | 1.0 | 3e-3 | 20 | 1,024 | 0.8265 |
| DP logistic regression | 0.5, 1, 2, 8 | 5.0 | 3e-3 | 20 | 1,024 | 0.8111, 0.8155, 0.8161, 0.8187 |
| DP logistic regression | 4 | 1.0 | 1e-2 | 10 | 1,024 | 0.8176 |
| Logistic regression | none | n/a | 1e-2 | 20 | 1,024 | 0.8244 |
| MLP (stage 1) | none | n/a | 3e-4 | 10 | 1,024 | 0.8482 (3 seeds, CPU) |

- Batch 4,096 never won, for DP or non-private, at a fixed number of epochs (4x fewer steps).
- Early observation, to be confirmed in step 2: DP logistic regression beats the DP MLP at epsilon 0.5 and 1; the DP MLP leads only from epsilon 2 to 8.

## 2026-10-12: Stage 2 step 2 (reported runs and diagnostics)

- Reported runs (`configs/report_*.yaml`): the selected settings above, plus the non-private MLP (lr 3e-4, 10 epochs, batch 1,024, re-run on GPU so devices match) and non-private logistic regression (lr 1e-2, 20 epochs), all on fresh seeds {10, 11, 12} that were never used in selection. LightGBM (127 leaves, CPU) is also re-run on these seeds, so every reference line uses unseen seeds.
- Clipping-only diagnostic (`configs/diag_clip_only_*.yaml`): the selected epsilon = 1 and 8 settings with zero noise. `dp_train` accepts `noise_multiplier: 0` only for this purpose, as the alternative to `target_epsilon`; such runs are logged with `private: False` and no epsilon, and must never be reported as private results.
- Width diagnostic (`configs/diag_width_dp_mlp.yaml`): a 64 -> 32 MLP at the selected epsilon = 1 setting, compared with the 128 -> 64 result.

## 2026-10-12: Stage 2 results (validation, fresh seeds 10 to 12, mean +/- std)

Selection score = average share of ceiling at 1% and 2%. All MLP and logistic regression runs on GPU; LightGBM on CPU. Spent epsilon never exceeded the target (0.492 to 7.995).

| Model | epsilon | Score | Share @1% | Share @2% | PR-AUC | $ missed @2% |
|---|---|---|---|---|---|---|
| LightGBM | none | 0.878 +/- 0.001 | 0.918 | 0.837 | 0.625 | 543k |
| MLP | none | 0.848 +/- 0.004 | 0.920 | 0.775 | 0.562 | 581k |
| DP MLP | 8 | 0.829 +/- 0.002 | 0.915 | 0.744 | 0.519 | 622k |
| DP MLP | 4 | 0.822 +/- 0.002 | 0.910 | 0.735 | 0.512 | 629k |
| DP MLP | 2 | 0.816 +/- 0.005 | 0.907 | 0.725 | 0.504 | 634k |
| DP MLP | 1 | 0.807 +/- 0.003 | 0.899 | 0.716 | 0.490 | 646k |
| DP MLP | 0.5 | 0.789 +/- 0.002 | 0.889 | 0.690 | 0.472 | 653k |
| Logistic regression | none | 0.825 +/- 0.003 | 0.914 | 0.735 | 0.521 | 618k |
| DP logistic regression | 8 | 0.821 +/- 0.001 | 0.906 | 0.736 | 0.520 | 618k |
| DP logistic regression | 4 | 0.815 +/- 0.003 | 0.901 | 0.729 | 0.503 | 628k |
| DP logistic regression | 2 | 0.819 +/- 0.001 | 0.902 | 0.735 | 0.516 | 618k |
| DP logistic regression | 1 | 0.816 +/- 0.001 | 0.900 | 0.732 | 0.511 | 621k |
| DP logistic regression | 0.5 | 0.804 +/- 0.001 | 0.893 | 0.715 | 0.500 | 629k |

- Privacy cost (score, own non-private model minus DP): MLP 0.019 at epsilon 8 to 0.059 at 0.5; logistic regression 0.004 at 8 to 0.021 at 0.5.
- Crossover: DP logistic regression beats the DP MLP at epsilon 0.5 and 1 (and ties at 2); the DP MLP leads at 4 and 8.
- DP logistic regression at epsilon 4 dips below epsilon 2. Its epsilon 4 setting (clipping 1.0, lr 1e-2, 10 epochs) won a seed-0 tie (0.8176 vs 0.8176) against the setting used at the other epsilons and does worse on fresh seeds: selection noise, not a privacy effect. Reported as is; re-selecting on the reporting seeds would reuse them for selection.
- Clipping vs noise: clipping alone costs nothing measurable. MLP clipping-only at the epsilon 1 setting scores 0.847 (non-private 0.848); logistic regression clipping-only 0.825 (non-private 0.825). The privacy cost is the noise. At the epsilon 8 MLP setting (clipping 1.0), clipping-only is 0.836 +/- 0.012, too variable to split precisely.
- Where the cost lands (frauds caught at the 2% budget, mean of 3 seeds; validation has 763 seen-card, 1,887 new-card, 1,961 keyless frauds): non-private MLP 1,832 (seen 70, new 431, keyless 1,331); DP MLP epsilon 1: 1,691 (seen 16, new 336, keyless 1,339). Keyless frauds, the easy ones, are untouched; DP removes almost all seen-card catches (-77%) and a fifth of new-card catches. Clipping-only training with the same budget catches 174 seen-card frauds, so the noise specifically erases card-level memorization, which is what DP is designed to do.
- Width: the 64 -> 32 MLP at epsilon 1 scores 0.803 +/- 0.007 vs 0.807 for 128 -> 64; a narrower network does not reduce the privacy cost.
- LightGBM on fresh seeds: 0.878 vs 0.882 on the selection seeds (selection optimism of about 0.004).

## 2026-10-13: Stage 3 design (replay and retraining under a fixed privacy budget)

- Replay: the period is split into 7-day windows from its first transaction (validation: 6 windows, the last partial; test: 6). Each week is scored by the latest model. Schedules: never retrain, or retrain every 3, 2, or 1 weeks.
- Label delay: 14 days. A run at time t trains only on transactions older than t - 14 days (fraud labels arrive late through chargebacks). Applies to the initial model too, so all schedules share the same starting point.
- Strategy A, expanding: each run trains from scratch on all labeled data so far. Runs reuse transactions, so privacy losses compose (RDP); one noise multiplier is calibrated for the whole plan (`dp_train.calibrate_composed_noise`), and after training the composed epsilon is recomputed from the runs' actual noise, sample rates, and steps and must not exceed the budget. Worst case assumed: a transaction used in every run.
- Strategy B, disjoint: each retrain fine-tunes the previous model (warm start; reusing a DP model's output costs no privacy) on only the transactions labeled since the previous run. Each transaction is used in exactly one run, so each run gets the full budget (parallel composition); the reported total is the largest single-run epsilon.
- Settings per run: the stage 2 selection at the epsilon closest on a log scale to epsilon_total / number of runs (expanding) or epsilon_total (disjoint). Non-private models use their stage 2 settings.
- Experiments: DP logistic regression at total epsilon 2 and 8; DP MLP at total epsilon 8 only (below 8 it already lost to logistic regression in stage 2, and composition pushes per-run epsilon lower); non-private logistic regression and MLP on the same schedules and strategies as references. Seeds 10 to 12 (the development replay runs the DP MLP on seed 10 only).
- Preprocessing: the stage 1 fit on the original training period, fixed for every run, so all runs share one feature space.
- Metrics: per week, recall and share of ceiling at 1% and 2% of that week's transactions (each week's own ceiling); pooled over weeks with `metrics.aggregate_windows`; selection-style score = mean pooled share over both budgets. Ranking metrics only, so no calibration in the replay.
- Drift (descriptive only): per week, PSI of each non-binary input against the initial training data (sample of 50,000 rows, 10 quantile bins), and PSI of the never-retrained model's scores against its training scores; weekly fraud rates.
- Test set (rule 4): the development replay on validation (history = train) fixes every detail and selects nothing; the final replay on test (history = train + validation) runs once with `--final`.

## 2026-10-13: Stage 3 development replay and warm-start fine-tuning

- Development replay (validation, `results/stage3/replay_dev_20261013-082146`): all privacy totals within budget (1.982 to 1.999 for epsilon 2; 7.940 to 7.999 for epsilon 8). Preview of the pattern the final replay will test: without privacy, weekly expanding retraining gains about +0.011 over never retraining (logistic regression 0.803 -> 0.814, MLP 0.827 -> 0.839); with privacy the gain shrinks to noise at total epsilon 2 and to about +0.004 (logistic regression) and +0.006 (MLP, one seed) at epsilon 8. Input drift stays small (mean feature PSI 0.007 to 0.014 per week) while every model's weekly share of ceiling falls late in the period, so feature PSI alone would not have flagged the decline. Disjoint strategies give the lowest dollar loss.
- Design issue found: warm-start fine-tunes used the full from-scratch epochs on one or two weeks of data, so non-private disjoint logistic regression degraded with frequent fine-tuning (0.803 never -> 0.780 weekly).
- Check (validation, `results/stage3/replay_dev_finetune_20261013-105740`): warm-start runs at 1/4 of the setting's epochs vs full epochs, disjoint schedules, all five experiments. Rule stated before running: one fraction for all models, chosen by the higher mean selection score over the 15 schedule-experiment pairs. Result: 0.8097 (1/4) vs 0.8079 (full). Chosen: `finetune_epochs_fraction: 0.25`, frozen in `configs/replay_final.yaml`. Non-private logistic regression no longer collapses (weekly 0.780 -> 0.792; every 2 weeks 0.798 -> 0.809); DP models improve slightly or are unchanged; the non-private MLP is slightly worse (every 3 weeks 0.835 -> 0.829).
- The final replay on test runs once with these settings; schedules and strategies are reported as they come out, not selected.

## 2026-10-13: Stage 3 final replay (test period, run once)

- Run: `results/stage3/replay_final_20261013-115756_final`. Interrupted once by low system memory after 17 of 35 experiment/schedule pairs; resumed with `--resume` (same frozen config, deterministic seeds; `env_resume_*.json` records the resumed session). All privacy totals within budget (max 2.000 and 7.999).
- Selection score (mean pooled share of ceiling at 1% and 2%), mean of seeds 10 to 12:

| Schedule | Logistic regression | DP LR, eps 8 | DP LR, eps 2 | MLP | DP MLP, eps 8 |
|---|---|---|---|---|---|
| Never retrain | 0.702 | 0.707 | 0.704 | 0.712 | 0.708 |
| Expanding, every 3 weeks | 0.708 | 0.715 | 0.709 | 0.732 | 0.714 |
| Expanding, every 2 weeks | 0.709 | 0.715 | 0.708 | 0.735 | 0.717 |
| Expanding, weekly | 0.712 | 0.717 | 0.707 | 0.738 | 0.714 |
| Disjoint, every 3 weeks | 0.694 | 0.710 | 0.701 | 0.720 | 0.710 |
| Disjoint, every 2 weeks | 0.691 | 0.713 | 0.705 | 0.725 | 0.709 |
| Disjoint, weekly | 0.685 | 0.712 | 0.702 | 0.724 | 0.708 |

- Gain of the best schedule over never retraining: MLP +0.026 without privacy vs +0.009 with DP at epsilon 8; logistic regression +0.010 without privacy, +0.011 at epsilon 8, +0.005 at epsilon 2.
- Privacy cost depends on whether the model is retrained: never retrained, the DP MLP is 0.004 below the non-private MLP; with weekly expanding retraining the gap is 0.024.
- At total epsilon 2, less frequent retraining scores slightly higher (every 3 weeks 0.709, every 2 weeks 0.708, weekly 0.707); at epsilon 8 weekly is best. Differences are within seed noise (about 0.005).
- Expanding beats disjoint for every model; without privacy, disjoint fine-tuning still hurts logistic regression (0.685 weekly vs 0.702 never) even at 1/4 epochs. The validation pattern of disjoint giving the lowest dollar loss did not hold on test.
- Red flag, investigated as possible bug: DP logistic regression scores above non-private logistic regression on test (never retrain 0.707 vs 0.702; weekly expanding 0.717 vs 0.712), by about 1 to 1.5 seed standard deviations, consistently across schedules. Not seen on validation (0.802 vs 0.803) or in stage 2. Spent epsilon is within budget and both use the same data path. Most likely explanation: the two use different selected settings (DP: lr 3e-3, clipping 5.0; non-private: lr 1e-2), and the lower learning rate plus clipping generalizes slightly better under the test period's shift. Not reported as "DP improves the model".
- Drift: mean feature PSI per week 0.004 to 0.010, with almost no feature above 0.25 (only D11 in the last week, 0.27), and score PSI of the never-retrained model at most 0.009, while weekly share of ceiling at 2% ranges from about 0.50 to 0.75. Input drift monitoring did not track performance in either period.
- Test scores are about 0.08 to 0.09 below validation for all models, driven by the first four test weeks (lower fraud rate, 2.8% to 3.5%).

## 2026-10-13: Post-hoc diagnostic of the DP vs non-private logistic regression gap (test)

- Run: `results/stage3/replay_final_diag_logreg_lr_20261013-223538_final` (`configs/replay_final_diag_logreg_lr.yaml`); post-hoc, selects nothing, not a reported configuration. Rule stated before running: if non-private logistic regression with the DP setting's learning rate (3e-3) matches or beats DP logistic regression at epsilon 8, the learning-rate explanation is confirmed.
- Result: rejected. With lr 3e-3 and 20 epochs (the DP setting, minus clipping and noise), non-private logistic regression is still below DP at epsilon 8 on all 7 schedules (mean -0.009; never retrain 0.702 vs 0.707; weekly expanding 0.711 vs 0.717; largest gap for weekly disjoint fine-tuning, 0.694 vs 0.712).
- What is ruled out: the learning rate, a data or accounting bug (same rows and code path, privacy totals verified, identical scoring).
- Remaining differences: gradient clipping (max norm 5.0) and DP noise. Leading hypothesis, unconfirmed: the noise acts as regularization that suppresses learning from rare features, which generalizes slightly better under the test period's shift and prevents overfitting during warm-start fine-tuning. Consistent with stage 2 (DP removed most seen-card catches) but not tested here. Reported as an open observation, not as "DP improves the model". Not seen on validation or in stage 2.

## 2026-10-13: Post-hoc diagnostic 2 (last): clipping vs noise on test; investigation closed

- Run: `results/stage3/replay_final_diag_logreg_clip_20261013-224615_final` (`configs/replay_final_diag_logreg_clip.yaml`): DP logistic regression at epsilon 8 with its exact per-schedule settings but zero noise (clipping only, not private). Post-hoc, selects nothing. Reading rule stated before running: clipping-only close to non-private means the noise explains the gap; close to DP means clipping does; in between means both.
- Result: neither. Clipping-only scores above both, on all 7 schedules: clipping-only > DP > non-private (lr 3e-3). Mean differences: clipping-only minus non-private +0.017; DP minus clipping-only -0.008; DP minus non-private +0.009. Example, weekly expanding: 0.711 / 0.725 / 0.717.
- Conclusion: DP logistic regression's edge on test comes from per-example gradient clipping (max norm 5.0), which limits the influence of individual (outlier) transactions and helps under the test period's shift. The privacy noise costs about 0.008, as expected. The earlier hypothesis that noise acts as helpful regularization is rejected. Not seen on validation, where clipping alone made no difference (stage 2: 0.825 vs 0.825), so the effect appears under the test period's drift only.
- Reporting: DP's small advantage on test is attributed to clipping, not to privacy; privacy itself has a cost. Clipping as a robustness technique for non-private fraud models is noted as future work, not claimed. Post-hoc analysis of the test set stops here.

## 2026-10-13: Final test-set evaluation of the stage 1 and 2 models (run once)

- Runs: `results/final/final_*_2026101323*` (`configs/final_*.yaml`; identical to the stage 2 report configs except `eval_split: test`). Models trained on train only; early stopping and Platt calibration on validation. Seeds 10 to 12. All spent epsilons within target (max 7.995). Test period: 118,108 transactions, 4,064 frauds (488 seen-card, 1,949 new-card, 1,627 keyless); ceilings 0.291 at 1% and 0.581 at 2%.
- Selection score on test (validation in brackets): LightGBM 0.775 (0.878); MLP 0.697 (0.848); logistic regression 0.697 (0.825); DP MLP 0.703 / 0.704 / 0.706 / 0.700 / 0.695 at epsilon 8 / 4 / 2 / 1 / 0.5; DP logistic regression 0.709 / 0.709 / 0.705 / 0.701 / 0.693.
- Share of ceiling, test: at 1%, LightGBM 0.875, MLP 0.780, logistic regression 0.801, DP models 0.803 to 0.820. At 2%, LightGBM 0.674, MLP 0.613, logistic regression 0.593, DP MLP 0.595 to 0.579, DP logistic regression 0.600 to 0.583. PR-AUC: LightGBM 0.508, MLP 0.445, logistic regression 0.433, DP MLP 0.425 (epsilon 8) to 0.401 (0.5), DP logistic regression 0.438 to 0.418. Out-of-sample ECE 0.004 to 0.006 for all models.
- What replicates on test: LightGBM is best by a clear margin; the DP MLP's cost grows as epsilon shrinks at the 2% budget and in PR-AUC; calibration is good out of sample.
- What does not replicate:
  - The 1% tie: on test the 1% budget separates LightGBM from the MLP (0.875 vs 0.780).
  - LightGBM's lead coming only from seen cards: at 2% it catches 145 more frauds than the MLP, 71 from seen cards and 66 from keyless frauds.
  - The epsilon crossover: on test the DP MLP and DP logistic regression are level at every epsilon, and the non-private MLP and logistic regression are level too (0.697 vs 0.697); the MLP's advantage is gone with or without privacy.
  - The seen-card memorization effect cannot be confirmed: 40 days after training even the non-private MLP catches only 6 of 488 seen-card frauds (validation: 70 of 763), so the catches DP removes on validation have little lasting value.
- DP above non-private on test (selection score): for the MLP this is driven by the 1% budget (DP 0.81 to 0.82 vs 0.780); at 2% and in PR-AUC the DP MLP is below the non-private MLP. DP logistic regression at epsilon 2 to 8 is slightly above non-private logistic regression at 1%, at 2%, and in PR-AUC (e.g. 0.600 vs 0.593 at 2% and PR-AUC 0.438 vs 0.433 at epsilon 8). Consistent with the stage 3 clipping diagnostic (clipping limits outlier influence and helps under drift). Recorded as is; no further post-hoc test analysis. (Corrected 2026-10-14: the first version of this entry said DP was below non-private at 2% and in PR-AUC for all models, which is true only for the MLP.)
- Interpretation: validation immediately follows training, while the test period starts about 40 days later, and the test period is harder (stage 3 showed the same drop for retrained models). Findings that depend on recency (card memorization, the 1% tie, the MLP's edge over logistic regression) fade with time.

## 2026-10-14: Public repository

- The public GitHub repository is a single snapshot commit of this project, made with a no-reply author email. The full development history (dated commits referenced by `git_commit` in each run's `env.json` and in this log) is kept by the author and is not published.
- Not published, to stay clear of redistributing Kaggle competition data: `data/splits/split.csv.gz` (transaction IDs per split) and each run's `preprocessor.json` (fitted category values and counts). Both are regenerated by the documented commands: `python -m fraudpriv.data --build-split` rebuilds the identical split (deterministic; check against the committed `data/splits/split_meta.json`), and any run refits the preprocessor on the training period.
