# Policy-Aware Portfolio Allocation with Black–Litterman and Safe RL

This  project studies portfolio allocation across **EUR/USD, GBP/USD, gold, and crude oil**. It combines a Black–Litterman (BL) optimizer with Lag-Llama forecasts, FAISS retrieval of historical market episodes, LLM-generated investment views, and a reinforcement-learning adjustment layer.

The main question is whether these additional information and learning layers improve allocation relative to a no-views BL baseline and an equal-weight benchmark.

**Main finding:** the saved test-period results show progressively higher annualized returns and Sharpe ratios as forecasting, LLM/memory, and RL layers are added. However, the RL-adjusted portfolio has **more CVaR violations and a worse maximum drawdown than the LLM/memory portfolio**. These artifacts support a performance improvement in this experiment, but do not establish improved out-of-sample safety.

This is a notebook-based research prototype, with saved datasets, model artifacts, and backtests. It is not a packaged trading application. The limitations below are part of interpreting its results.

## Project structure

```text
Base BL Model and Data/
  Project8_data_Setup.ipynb            Market/macro collection and v2 dataset assembly
  Project08_data.ipynb                Earlier dataset assembly workflow
  BL_base_model.ipynb                 Original no-views BL baseline
  project8_dataset_v2.csv             Combined market, macro, and policy data

FAISS_episodic_memory_Modified/
  Step2_FAISS_Episodic_Memory_v2.ipynb  Historical episode construction and retrieval
  episodic_memory_v2.index            FAISS vector index
  episodes_v2.pkl                     Episode features and subsequent outcomes
  scaler_v2.pkl                       Embedding scaler

World_model_Modified/
  Step3_WorldModel_LagLlama.ipynb      Fine-tuning and rolling forecast generation
  lag_llama_views.csv                 Dated forecasts and confidence scores

gpt_view_generation/
  gpt-oss-120b_ViewGeneration_Modified.ipynb
  final_views_checkpoint.csv         Resumable view-generation output
  final_views.csv                    Per-asset views, confidence, and reasoning
  final_views_for_step5.json          Optimizer input, keyed by date

BL_model_updated (after views)/
  BL_base_model.ipynb                 Integrated four-strategy backtest
  bl_*_weights.csv                   Allocation histories
  all_strategies_daily_returns.csv
  all_strategies_backtest_metrics.csv

Safe RL/
  Step6_OfflineDT_New/                Synthetic trajectories, Decision Transformer,
                                     and PPO behavior-cloning warm start
  Step6_OnlinePPO_New/                PPO fine-tuning, Lagrangian safety training,
                                     inference module, and final evaluation
```

Several folders contain copies of upstream inputs so their notebooks can run locally. All seven non-checkpoint copies of `project8_dataset_v2.csv` were byte-identical when this README was prepared. Hidden notebook checkpoints and `Safe RL.zip` are additional snapshots; the workflow here refers to the named notebooks in the main folders.

## Data and experimental setup

- **Assets:** `EURUSD`, `GBPUSD`, `GOLD`, and `OIL`, in that order throughout the allocation and RL interfaces. The collection notebook uses Yahoo Finance symbols `EURUSD=X`, `GBPUSD=X`, `GC=F`, and `CL=F`.
- **Macro variables:** federal funds rate, inflation, USD index, VIX, and US 10-year yield. The data setup notebook uses FRED; the memory and forecasting stages select VIX, USD index, and US 10-year yield as macro features.
- **Policy events:** 59 records in `policy_events.csv`, including dates, types, severity, descriptions, and source fields. The combined dataset also includes policy flags and asset metadata.
- **Saved v2 dataset:** 2,211 rows, covering **2018-01-02 through 2026-06-29**; 1,826 rows labeled `TRAIN` and 385 labeled `TEST`.
- **Portfolio/RL split:** training through 2024-12-31; testing from 2025-01-02. Dropping rows without all four close prices leaves 2,132 price observations, 1,758 training return observations, and **373 test return observations**.
- **World-model split:** 2018–2022 for initial training, 2023–2024 for validation, and 2025–2026 for walk-forward forecasting. Its filling rules differ from the portfolio notebooks.
- **Return treatment:** the portfolio code clips daily asset returns to ±20%, including extreme oil observations around the negative-price episode. These are backtests on transformed returns.
- **Allocation schedule:** the integrated outputs contain **19 allocation dates**, including the first test day and monthly dates. Forecast and LLM-view exports each contain **76 rows**, one per asset per date.

The integrated portfolio loop records the day's return using the existing weights, then updates the allocation for subsequent observations. It uses a trailing 252-observation covariance window, long-only weights, and an equal-weight-implied equilibrium prior.

## How the pipeline works

### 1. Black–Litterman baseline

The baseline uses PyPortfolioOpt for Ledoit–Wolf covariance shrinkage, Black–Litterman posterior estimation, and maximum quadratic utility optimization with risk aversion `2.5`. With no market-cap weighting for this asset universe, the prior is constructed as:

```text
prior returns = risk aversion × annualized covariance × equal weights
```

`run_bl(views, confidences)` returns posterior returns, posterior covariance, and portfolio weights. The view-enabled path uses Idzorek confidence handling. Weights are bounded between zero and one.

### 2. FAISS episodic memory

The memory notebook constructs market episodes and represents each with a **24-dimensional vector**: five return statistics for each of four assets, three macro features, and one policy-event flag. Embeddings are standardized and indexed with FAISS `IndexFlatL2`.

Retrieval supplies similar episodes and their subsequent outcomes to the LLM. An embargoed retrieval function filters for older episodes. The current memory notebook uses a 30-observation history and 30-observation outcome window; its consumer still uses a 20-observation query history. This mismatch and the embargo limitations are documented below.

### 3. Lag-Llama world model

The current forecasting code initializes from the pretrained Lag-Llama checkpoint, fine-tunes on the project data, and refreshes the model every three rebalance dates using an expanding historical window. It specifies a 64-step context, a 30-step forecast horizon, 30 initial training epochs, and 15 epochs for each refresh.

Forecast-derived return views compare the mean predicted price across the horizon with the last observed price. Expected returns are clipped using historical monthly-return variability. Confidence comes from validation directional hit rates, with dampening. `lag_llama_views.csv` contains the dated views and diagnostics consumed downstream.

### 4. LLM views with historical context

The view-generation notebook configures `openai/gpt-oss-120b` through the Groq client. It combines the dated forecast, the most recent policy event, and five retrieved historical episodes.

For each date, it attempts three sampled responses and requires at least two valid responses. It averages expected returns and penalizes confidence when sampled returns disagree. Results are checkpointed and exported to CSV and a date-keyed JSON structure:

```json
{
  "2025-01-02": {
    "views": {"EURUSD": 0.01, "GBPUSD": 0.01, "GOLD": 0.02, "OIL": -0.01},
    "confidences": [0.5, 0.5, 0.6, 0.4]
  }
}
```

The numbers above illustrate the schema, not an actual saved forecast. Expected returns are decimal fractions; confidence order must match view order. Some notebook comments retain older “Llama-3” naming, but the configured model is GPT-OSS-120B.

### 5. Integrated allocation backtest

The updated BL notebook compares equal weights, BL without views, BL with world-model views, and BL with LLM/memory views in the same portfolio loop. It saves daily returns, comparison metrics, and allocation histories. `bl_llm_memory_weights.csv` is the input anchor for final RL evaluation.

### 6. Offline pretraining and online PPO

The offline stage creates six synthetic allocation trajectories around the training-period BL baseline and attaches return-to-go targets. A small Decision Transformer learns action prediction from these trajectories. Its predictions then provide behavior-cloning targets for PPO.

The online stage first fine-tunes PPO for reward, then adds a Lagrangian penalty for constraint breaches. “Online” here means interaction with a historical-data environment, not live broker execution.

- **State:** 12 values: four trailing five-day mean returns, four trailing five-day volatilities, and four candidate portfolio weights.
- **Action:** a per-asset adjustment bounded to `[-0.15, 0.15]`, followed by clipping and weight normalization.
- **Safety measurements:** historical 95% CVaR over 60 observations with a 2.3% budget, and a 19% drawdown budget.
- **Training anchor:** the no-views BL portfolio on 2018–2024 training data.
- **Evaluation anchor:** the LLM/memory portfolio on 2025–2026 test data, adjusted at allocation dates.

The packaged module exposes `safe_rl_adjust(bl_weights_today, state)` and loads the neighboring `ppo_lagrangian_trained.zip` at import time.

## Findings from the saved experiments

### Portfolio performance

The following figures cover the **373-observation test period, 2025-01-02 through 2026-06-29**. Values were independently recalculated from the saved daily-return CSVs and matched the exported metrics to four decimal places.

- **Equal weight:** annualized return **12.91%**, annualized volatility **13.42%**, Sharpe **0.9615**, maximum drawdown **−13.75%**.
- **BL baseline:** annualized return **12.78%**, annualized volatility **13.00%**, Sharpe **0.9831**, maximum drawdown **−13.26%**.
- **BL + world model:** annualized return **14.26%**, annualized volatility **13.92%**, Sharpe **1.0242**, maximum drawdown **−13.55%**.
- **BL + LLM + memory:** annualized return **17.59%**, annualized volatility **14.78%**, Sharpe **1.1901**, maximum drawdown **−12.60%**.
- **BL + LLM + memory + Safe RL:** annualized return **23.49%**, annualized volatility **15.53%**, Sharpe **1.5125**, maximum drawdown **−13.25%**.

The world-model strategy improves return and Sharpe over baseline, with higher volatility and a slightly worse drawdown. The LLM/memory strategy improves return, Sharpe, and drawdown relative to baseline. Adding RL raises annualized return by **5.90 percentage points** and Sharpe by **0.3224** over the LLM/memory anchor, but increases volatility and worsens maximum drawdown by **0.65 percentage points**.

Sources: [integrated metrics](<BL_model_updated (after views)/all_strategies_backtest_metrics.csv>), [integrated daily returns](<BL_model_updated (after views)/all_strategies_daily_returns.csv>), [final comparison](<Safe RL/Step6_OnlinePPO_New/step7_final_comparison_table.csv>), and [final daily returns](<Safe RL/Step6_OnlinePPO_New/step7_daily_returns.csv>).

### Safety training improves training constraints, but does not establish test safety

The saved [Lagrangian training history](<Safe RL/Step6_OnlinePPO_New/lagrangian_training_history.csv>) records:

- Average constraint cost falling from **0.007986 to 0.000483**, approximately **94%**.
- CVaR violations falling from **669 to 83**, and drawdown violations from **126 to 2**.
- Episode reward falling from **0.6183 to 0.4561** as the policy trades reward for lower constraint cost. These are accumulated training rewards, not annualized or compounded portfolio returns.

The test-period violation calculation gives a different result:

- **LLM/memory:** **99 / 312** CVaR violations (**31.73%**), zero drawdown-budget violations.
- **LLM/memory + RL:** **146 / 312** CVaR violations (**46.79%**), zero drawdown-budget violations.

These counts were independently reproduced from the saved weights and dataset using the evaluation notebook's calculation. The RL layer adds **47 CVaR breaches**, increasing the violation rate by **15.06 percentage points**. Its adjustment log reports application on all **19 / 19** allocation dates.

The final notebook's passing assertion checks only that RL does not increase drawdown-violation counts. It does **not** require fewer CVaR breaches. “Safe RL” therefore describes the training approach, not a demonstrated guarantee for the evaluated strategy.

Source: [walk-forward evaluation notebook](<Safe RL/Step6_OnlinePPO_New/Step6B_WalkForwardEvaluation.ipynb>) and [adjustment log](<Safe RL/Step6_OnlinePPO_New/saferl_rl_applied_log.csv>).

### Metric definitions

The notebooks use 252 periods per year:

```text
annualized return     = product(1 + daily returns) ** (252 / number of days) - 1
annualized volatility = sample standard deviation(daily returns) × sqrt(252)
reported Sharpe       = annualized return / annualized volatility
maximum drawdown      = min(cumulative wealth / running peak wealth - 1)
```

The reported Sharpe uses geometric annualized return and no risk-free-rate subtraction. CVaR averages losses at or above the historical 95th percentile, applying the current allocation to a trailing asset-return window; it is not the same as CVaR of the strategy's realized changing-weight return series.

## Running the project

### Inspect or reproduce saved results first

The saved CSVs can be inspected without model downloads or API calls. To reproduce the main portfolio comparisons, start with `BL_model_updated (after views)/BL_base_model.ipynb`, which already has its dataset and view inputs beside it. To evaluate the saved RL checkpoint, use `Safe RL/Step6_OnlinePPO_New/Step6B_WalkForwardEvaluation.ipynb` with its existing module, checkpoint, dataset, and weight files.

There is no dependency lockfile or unified runner. A starting environment for these two evaluation notebooks is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install jupyterlab numpy pandas matplotlib pyportfolioopt stable-baselines3
python -m jupyter lab
```

This is an unpinned starting setup, not a verified reproduction environment. Run each notebook with its own folder as the working directory. Review embedded installation cells before execution: some use Colab-specific installation flags, and the world-model notebook includes a NumPy reinstall and kernel restart.

### Regenerate all stages

Resolve the relevant consistency issues in the next section before treating a full regeneration as a clean experiment.

1. **Data:** use `Base BL Model and Data/Project8_data_Setup.ipynb`, or retain the supplied v2 dataset. Fresh collection additionally needs `yfinance`, `fredapi`, network access, and a FRED credential. The existing policy-event CSV is an input.
2. **Memory:** run `FAISS_episodic_memory_Modified/Step2_FAISS_Episodic_Memory_v2.ipynb`; this requires `faiss-cpu` and `scikit-learn`. Copy the resulting `.index`, episode pickle, and scaler pickle together into `gpt_view_generation/`.
3. **Forecasts:** run `World_model_Modified/Step3_WorldModel_LagLlama.ipynb` in a compatible PyTorch/GluonTS environment. Its setup installs Lag-Llama from GitHub and downloads a Hugging Face checkpoint; a CUDA GPU is useful for fine-tuning. The notebook currently reads a remote project CSV: change that input to the local v2 dataset if reproducing this snapshot. Copy the generated `lag_llama_views.csv` to both the LLM and updated BL folders.
4. **LLM views:** run `gpt_view_generation/gpt-oss-120b_ViewGeneration_Modified.ipynb` with `groq`, the memory dependencies, and a Groq credential. Remove the hardcoded credential assignment and read `GROQ_API_KEY` from the environment. Existing checkpoint rows are skipped: use a fresh checkpoint filename for a new experiment. Copy `final_views_for_step5.json` into the updated BL folder.
5. **Allocation:** run `BL_model_updated (after views)/BL_base_model.ipynb`; copy the new `bl_llm_memory_weights.csv` into `Safe RL/Step6_OnlinePPO_New/`.
6. **Offline RL:** in `Safe RL/Step6_OfflineDT_New/`, run `Step6A_OfflineDT_DataPrep.ipynb`, `Step6A_DecisionTransformer_Train.ipynb`, then `Step6A_WarmStart_PPO.ipynb`. Dependencies include PyTorch, Gymnasium, and Stable-Baselines3. The supplied `bl_baseline_train_weights.csv` is an input to this stage. Copy the resulting `offline_trajectories.pkl` and `ppo_warmstarted.zip`, along with the matching training anchor and dataset, to the online folder.
7. **Online RL and evaluation:** run `Step6B_PlainPPO_SanityCheck.ipynb`, `Step6B_LagrangianSafetyLoop.ipynb`, `Step6B_PackagePolicy.ipynb`, and `Step6B_WalkForwardEvaluation.ipynb`, in that order.

Training and API generation were not rerun for this documentation. Package compatibility, model-service availability, and exact stochastic reproducibility remain unverified.

### Using the saved policy

From `Safe RL/Step6_OnlinePPO_New/`, with its dependencies installed:

```python
from safe_rl_policy import safe_rl_adjust

# Both arrays must follow the asset order EURUSD, GBPUSD, GOLD, OIL.
# state: float32 vector [4 trailing means, 4 trailing standard deviations,
#                       4 candidate weights], as built in the evaluation notebook.
final_weights = safe_rl_adjust(bl_weights_today, state)
```

Use real trailing observations and the matching candidate weights to build the state. The module normalizes clipped allocations; it does not independently enforce a hard CVaR or drawdown limit at inference.

## Limitations and code-review findings

These findings describe the checked-in code; this README does not modify the implementation.

1. **Memory preprocessing can use future information.** The FAISS scaler is fitted to embeddings across the full dataset. Filtering retrieved dates afterward does not remove future information from the scaling. Fit preprocessing on training data or an expanding history available at each decision date.

2. **The retrieval embargo does not ensure outcome availability.** Outcomes span 30 dataset observations, while the filter uses a 30-calendar-day gap. An eligible episode can still contain outcomes after the query date. Filter by the actual outcome-window end date.

3. **Memory feature windows disagree.** Episode construction uses `WINDOW_SIZE = 30`, while LLM query construction uses `WINDOW_SIZE = 20`. Both generate 24-dimensional statistical embeddings, so a dimension assertion does not detect the semantic mismatch. Align the windows and regenerate the index, scaler, episodes, and dependent views together.

4. **View-return horizons are not aligned with the BL prior.** Forecast/LLM views describe roughly 30-step returns, whereas BL covariance and implied prior returns are annualized. The integration passes these views directly into BL without an explicit horizon conversion. Standardize units before interpreting view strength or allocation changes.

5. **Forecast-error diagnostics compare different time intervals.** The world-model metric cell predicts beyond each supplied split but compares the forecast with the first observations of that split. Its MAE/RMSE/MAPE values should not be treated as valid forecast accuracy measurements until forecast and target timestamps are aligned. Memory outcomes and forecast views also use average prices/returns over a horizon rather than a simple endpoint return.

6. **Constraint evaluation differs from the performance loop.** Violation checks skip the first 60 test observations and the last observation, reset wealth at the start of that shorter window, and apply a rebalance's new weights before measuring its same-day return. The main performance loop updates weights after recording that day's return. Consequently, zero drawdown-budget breaches in the constraint loop do not validate the full-period return path. Use a shared execution timeline and wealth history.

7. **The portfolio simulation is simplified.** It holds numerical weights fixed between scheduled updates instead of letting holdings drift with prices, implicitly maintaining those weights daily. The equal-weight benchmark behaves similarly; it is not a drifting buy-and-hold portfolio. Transaction costs, slippage, FX financing, and futures roll/execution mechanics are not modeled in the reported returns.

8. **Credentials are embedded in notebooks.** The data setup and GPT view-generation notebooks contain literal API credentials, despite comments describing environment-variable usage. Remove those literals, rotate the exposed credentials, and use environment variables. Credential values are intentionally not reproduced here.

9. **Reproducibility is incomplete.** Dependency versions are not locked, some inputs are remote, outputs are copied manually between folders, and notebook comments sometimes describe older configurations. The final RL comparison also hardcodes the published baseline/world-model/equal-weight metrics rather than recalculating every strategy. Those values match the current upstream CSVs, but must be refreshed after reruns. Some notebooks reference `SAFE_RL_GUIDE.md` or `.pdf`, neither of which is present as a standalone file in this tree.

10. **Evidence is limited to one saved experiment.** No multi-seed confidence intervals or attribution experiments separating the LLM contribution from memory retrieval are supplied. Training adjusts weights daily, while final evaluation applies RL at monthly allocation dates and changes the anchor strategy. This distribution change helps explain why training safety improvements cannot be assumed to transfer. Point-in-time macro release availability and historical LLM knowledge also need auditing before claiming the full pipeline is leakage-free.

## Validation performed for this README

The review inspected the main notebooks, policy module, dataset copies, saved allocation histories, and exported results. Portfolio metrics were recalculated from both daily-return exports; the two final strategies' CVaR and drawdown violation counts were reproduced from their weights and the local dataset. These checks validate the reported artifact arithmetic, not model training, causal attribution, or the absence of information leakage.
