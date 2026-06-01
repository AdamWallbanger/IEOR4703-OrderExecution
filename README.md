# Limit-Order Placement via Conditional Empirical PDF

**IEOR 4703 Term Project: Order Execution**

This project studies how to convert a stream of target trades produced by an AI execution agent into limit-order placement decisions on seven futures contracts. For each trade signal the system estimates a state-conditional empirical distribution of short-horizon price ranges, selects an order offset (in ticks) consistent with a user-specified minimum fill probability, and evaluates realized execution quality against the agent's reference price.

## 1. Problem Setup

An upstream AI agent emits a sequence of desired inventory changes for each instrument. Each desired trade carries a timestamp and a reference price `price`, which we treat as the execution benchmark. The execution problem is to decide, for every signal, how aggressively the limit order should be placed relative to the prevailing market. An order resting close to the touch fills with high probability but captures little price improvement, while an order resting deeper inside the spread offers larger savings at the cost of a higher chance of going unfilled. The project quantifies this trade-off as a joint function of the holding window `τ` (in minutes) and the minimum fill probability `P` required at order entry.

## 2. Data

The repository contains 1-minute OHLCV bars for seven futures contracts together with one signal file per instrument produced by an AI agent. The mapping between instrument symbols, descriptive names, and the tick sizes used by the execution layer (defined in `main.py`) is summarized below.

| Symbol | Instrument                | Tick size |
| :----- | :------------------------ | --------: |
| VG     | EuroStoxx 50 Futures      |      0.50 |
| BP     | British Pound Futures     |      0.01 |
| RX     | German Bund Futures       |      0.01 |
| GC     | Gold Futures              |      0.10 |
| HO     | Heating Oil Futures       |      0.01 |
| JY     | Japanese Yen Futures      |     0.005 |
| NQ     | Nasdaq 100 Futures        |      0.25 |

After backtesting, that is after dropping signals that lack a sufficient pre-history for ePDF training or that lie beyond the available OHLCV window, the dataset comprises 5,766 trade signals distributed across the seven instruments, with Nasdaq and Gold contributing the largest sample sizes (2,164 and 1,276 respectively) and EuroStoxx the smallest (212).

## 3. Methods

### 3.1 Pipeline overview

Execution proceeds in three stages, orchestrated by `main.py` and parallelized across signals with Python `multiprocessing`. Signal generation (`Signal_gen.py`) extracts discrete trades from the AI agent's inventory series. The execution module (`execution.py`) reads each signal, trains a conditional empirical distribution on data strictly preceding the signal timestamp, classifies the current market state using the same preprocessing pipeline as the training step, and computes a recommended limit price. The backtest module (`backtest.py`) replays each placement against the realized OHLC path inside the τ-minute window and records the realized fill price and slippage relative to the benchmark.

### 3.2 Signal generation

The signal file for each instrument is derived from the AI agent's inventory time series via a first difference. Rows with zero change are discarded, so each remaining row encodes a directional trade of magnitude `|signal|` at the recorded timestamp. Positive signals are interpreted as buy orders and negative signals as sell orders.

### 3.3 Conditional empirical PDF

The probabilistic core of the method is contained in `epdf/` and exposes a single class `ePDFCalculator`. Given an instrument, a holding window τ, and three discretization parameters (M, N, K), the calculator estimates the probability distribution of the τ-minute range conditioned on the prevailing state of the market.

**Preprocessing.** The data processor loads the 1-minute OHLCV file, filters out trading days whose minute count falls below 90 % of the historical median, and resamples to τ-minute bars by aggregating only those windows that retain at least 80 % of their expected minutes. Day boundaries are inferred from time gaps of four hours or more. For every aggregated bar the processor computes three tick-normalized range quantities,

$$R = \mathrm{round}\!\left(\frac{H-L}{\varepsilon}\right), \qquad R_{\text{up}} = \mathrm{round}\!\left(\frac{H-O}{\varepsilon}\right), \qquad R_{\text{dn}} = \mathrm{round}\!\left(\frac{O-L}{\varepsilon}\right),$$

where ε denotes the contract tick size.

**State construction.** Three features describe the prevailing market state: an exponentially weighted moving average of volume, of range (taken as a volatility proxy), and of the one-step change in the open price. The EWMA recursion is implemented with the strict lag convention η<sub>j-1</sub> at step j so that the state at bar j contains no information from bar j itself. After EWMA features are computed, the calculator selects the burn-in index

$$J_s = \max\bigl(100,\ 3 \cdot \text{halflife},\ 50 \cdot \max(M, N, K)\bigr),$$

and uses the empirical quantiles of the three EWMA series on the slice `[J_s :]` to define M, N, and K state bins. Every bar is assigned a discrete state triple `(m, n, k)`, yielding a state space of size M × N × K = 18 under the parameters used throughout the experiments.

**Distribution estimation.** For each observed state, the estimator builds three conditional empirical mass functions over `R`, `R_up`, and `R_dn` from the corresponding histograms of post-`J_s` bars. The smoothed estimator applies Laplace smoothing with parameter α = 0.5, yielding

$$\hat{p}(R = \ell \mid s) = \frac{n_{\ell,s} + \alpha}{n_s + \alpha\,(R_{\max,s} + 1)},$$

after which the upper-tail CDF is obtained by reverse cumulative summation,

$$F(\ell \mid s) = \Pr(R \ge \ell \mid s) = \sum_{r \ge \ell} \hat{p}(R = r \mid s).$$

The class also exposes a validation routine that verifies normalization of each PDF and monotonicity of each CDF, both of which hold throughout the experiments reported here.

### 3.4 Limit-order placement

For each signal, the execution module retrains the calculator on data strictly earlier than the signal timestamp. It then reapplies the same `process_pipeline` and `compute_all_ewma_features` routines as the training step on the full data file (without a training cutoff) so that the query-time features are guaranteed to live in the same coordinate system as the bin boundaries learned at training. The last τ-minute bar whose timestamp is strictly less than the signal timestamp provides the values of `v_ewma`, `sigma_ewma`, and `delta_x_ewma` that classify the market into one of the eighteen states. Order direction is determined by the sign of the signal, with a buy order placed below the prevailing close (`range_dn`) and a sell order placed above it (`range_up`). The placement offset ℓ (in ticks) is then chosen as

$$\ell^\star \;=\; \max\bigl\{\,\ell \in \{0, 1, \dots, 9\} \;:\; F(\ell \mid s) \ge P\,\bigr\},$$

so that the order is placed at the deepest tick offset whose conditional fill probability still meets the minimum threshold `P`. The recommended limit price is

$$p^\star = \text{close}_{\text{prev}} \pm \ell^\star \cdot \varepsilon,$$

with the sign chosen by direction.

### 3.5 Backtest

For each placement, the backtest module isolates the OHLC path on the window `[t, t + τ]` and declares a fill at the recommended price `p★` whenever some bar in that window satisfies `low ≤ p★ ≤ high`. If no bar reaches the recommended price, the order is closed at the close price of the last bar in the window, or at the open of the first bar after the window if the window contains no bars. Slippage is signed so that positive values correspond to execution that improves on the benchmark,

$$\text{slippage} = \bigl(\text{benchmark amount} - \text{realized amount}\bigr) \cdot \operatorname{sign}(\text{signal}),$$

with benchmark amount equal to the AI agent's reference price multiplied by `|signal|`.

### 3.6 Experimental parameters

All experiments use M = 3, N = 3, K = 2, EWMA half-life 10 bars, smoothed estimation with α = 0.5, and a maximum search depth of 10 ticks. Two design axes are swept: the holding window takes the values τ ∈ {5, 10, 15, 30, 60} minutes, and the minimum fill probability takes the values P ∈ {0.3, 0.6, 0.9}, giving fifteen `(P, τ)` configurations per instrument and 105 configurations in total. The results presented in this README focus on the conservative regime P = 0.9. The corresponding panels for P = 0.6 and P = 0.3 are produced by the same code path and can be reproduced by re-running `main.py` with `risk_percentage` set to the desired value.

## 4. Results

### 4.1 Evaluation metrics

Two metrics summarize execution quality per instrument. The **hit rate** is the fraction of signals for which the realized fill price equals the recommended limit price, with the complementary fraction being closed at the fallback price. The **notional-weighted slippage**, expressed in basis points, is

$$\text{wtd slippage (bps)} = \frac{\sum_i \text{slippage}_i}{\sum_i |\text{benchmark amount}_i|} \times 10{,}000,$$

so that a positive value indicates execution at a price strictly better than the AI agent's reference price, while a negative value indicates worse execution. All aggregation is performed within instruments only. Cross-instrument totals are not reported because the contracts differ in tick size, notional, and liquidity to an extent that renders pooled averages economically meaningless.

### 4.2 Slippage by instrument and τ at P = 0.9

The notional-weighted slippage in basis points for each `(instrument, τ)` cell is reported in Table 1. The cell achieving the highest value in each row is shown in **bold**.

**Table 1.** Notional-weighted slippage (bps), P = 0.9.

| Instrument  |     τ = 5 |    τ = 10 |    τ = 15 |    τ = 30 |    τ = 60 |
| :---------- | --------: | --------: | --------: | --------: | --------: |
| Bunds       |    +0.237 |    +0.312 |    +0.635 |    +0.780 | **+1.058** |
| EuroStoxx   |    +1.065 |    +2.192 |    +2.000 | **+3.261** |    +2.950 |
| GBPUSD      |    −0.245 |    −0.135 | **+0.006** |    −0.231 |    +0.004 |
| Gold        |    −0.108 |    +0.087 |    +0.384 |    +0.898 | **+1.517** |
| HeatingOil  | **+8.989** |    +8.366 |    +8.565 |    +3.227 |    −0.487 |
| JPY         |    +0.403 |    +0.677 |    +1.000 |    +0.702 | **+1.461** |
| Nasdaq      |    +0.430 |    +0.597 |    +0.401 |    +0.052 | **+1.027** |

Collecting the best cell from each row identifies, within the P = 0.9 panel, the most favourable holding window for each instrument. The result is summarized in Table 2.

**Table 2.** Best `τ` per instrument at P = 0.9.

| Instrument  | Optimal τ | Weighted slippage (bps) | Hit rate at optimum |
| :---------- | --------: | ----------------------: | ------------------: |
| Bunds       |        60 |                  +1.058 |              94.5 % |
| EuroStoxx   |        30 |                  +3.261 |              94.3 % |
| GBPUSD      |        15 |                  +0.006 |              98.0 % |
| Gold        |        60 |                  +1.517 |              94.3 % |
| HeatingOil  |         5 |                  +8.989 |              72.8 % |
| JPY         |        60 |                  +1.461 |              94.2 % |
| Nasdaq      |        60 |                  +1.027 |              97.3 % |

### 4.3 Hit rate by instrument and τ at P = 0.9

Table 3 reports the hit rate at P = 0.9. With the exception of Heating Oil, every instrument achieves a hit rate of at least 93 % across all five holding windows, indicating that the chosen probability threshold is largely binding and that fallback fills are uncommon.

**Table 3.** Hit rate, P = 0.9.

| Instrument  |  τ = 5 | τ = 10 | τ = 15 | τ = 30 | τ = 60 |
| :---------- | -----: | -----: | -----: | -----: | -----: |
| Bunds       | 99.3 % | 99.2 % | 98.4 % | 96.4 % | 94.5 % |
| EuroStoxx   | 96.7 % | 95.8 % | 97.2 % | 94.3 % | 93.4 % |
| GBPUSD      | 97.3 % | 98.2 % | 98.0 % | 96.9 % | 97.1 % |
| Gold        | 97.9 % | 96.9 % | 96.9 % | 95.5 % | 94.3 % |
| HeatingOil  | 72.8 % | 76.2 % | 76.9 % | 81.4 % | 84.5 % |
| JPY         | 97.4 % | 96.9 % | 97.8 % | 95.3 % | 94.2 % |
| Nasdaq      | 96.2 % | 96.2 % | 96.1 % | 97.0 % | 97.3 % |

### 4.4 Per-configuration visualizations

The bar charts below report the per-instrument average slippage in basis points for each `τ` at P = 0.9, with green bars denoting positive slippage relative to the AI-agent benchmark and red bars denoting negative slippage. The hit rate corresponding to each bar is reported in the legend.

| τ = 5 minutes | τ = 10 minutes |
| :-----------: | :------------: |
| ![P=0.9, τ=5](<Result_9new/slippage_tau5.png>) | ![P=0.9, τ=10](<Result_9new/slippage_tau10.png>) |

| τ = 15 minutes | τ = 30 minutes |
| :------------: | :------------: |
| ![P=0.9, τ=15](<Result_9new/slippage_tau15.png>) | ![P=0.9, τ=30](<Result_9new/slippage_tau30.png>) |

| τ = 60 minutes |
| :------------: |
| ![P=0.9, τ=60](<Result_9new/slippage_tau60.png>) |

### 4.5 Key findings

**Finding 1: Under the current dataset and decision rule, every instrument admits a configuration that improves on the AI-agent benchmark.** The optimum reported in Table 2 is positive for all seven contracts, ranging from a marginal +0.006 bps on GBPUSD to a substantial +8.989 bps on Heating Oil. Six of the seven instruments deliver an optimum of at least +1 bps, with EuroStoxx in particular reaching +3.26 bps at τ = 30. This indicates that, given a sufficiently conservative probability threshold and an instrument-specific choice of holding window, the conditional ePDF placement consistently extracts price improvement relative to the agent's reference price.

**Finding 2: The optimal holding window is instrument-specific and biased toward longer windows.** Within the P = 0.9 panel the best `τ` takes four distinct values across the seven instruments. Four contracts (Bunds, Gold, JPY, Nasdaq) attain their optimum at τ = 60, EuroStoxx at τ = 30, GBPUSD at τ = 15, and Heating Oil at τ = 5. The bias toward longer windows is consistent with the structure of the placement rule: a longer τ admits a larger tick offset at the same fill probability, which translates into greater price improvement provided the price actually reaches the deeper limit.

**Finding 3: Heating Oil is qualitatively different from the other instruments.** The slippage on Heating Oil is by far the largest in magnitude at short windows (+8.99 bps at τ = 5) but decays monotonically with τ and turns negative at τ = 60 (−0.49 bps). Its hit rate is also distinctly lower than the rest of the cross-section, sitting between 73 % and 85 % at P = 0.9 while every other contract clears 93 %. Both anomalies point to a conditional return distribution with a heavier short-horizon tail in the favourable direction and a much wider dispersion at long horizons, which the chosen state grid and threshold combine to exploit at short τ but mismatch at long τ.

**Finding 4: Several financial futures exhibit a monotone improvement with τ.** Bunds, Gold, and JPY all show slippage that increases monotonically (or near-monotonically) from τ = 5 to τ = 60. Gold provides the cleanest example, moving from −0.108 bps at τ = 5 to +1.517 bps at τ = 60. Nasdaq does not share this monotonicity: its slippage declines from +0.430 bps at τ = 5 to +0.052 bps at τ = 30 before rising sharply to its maximum of +1.027 bps at τ = 60. All four instruments nonetheless retain hit rates above 94 % across every τ and attain their optimum at τ = 60, so the long-window benefit is not bought at the cost of substantially more unfilled orders. This pattern is consistent with these contracts exhibiting enough short-horizon mean reversion or oscillation for limit orders to be reached when given sufficient time.

**Finding 5: GBPUSD is the marginal case.** Among the seven contracts, GBPUSD is the only one whose optimum is essentially flat (+0.006 bps at τ = 15). Its slippage hovers near zero across the entire τ grid and never becomes meaningfully positive. The hit rate stays above 96 % everywhere, so the absence of price improvement is not due to unfilled orders but rather to the placed limits being barely better than the reference price. Within the present setup this instrument behaves close to the no-improvement boundary.

**Finding 6: Hit rates are high and stable at P = 0.9 for all financial futures.** The six financial futures (Bunds, EuroStoxx, GBPUSD, Gold, JPY, Nasdaq) all achieve hit rates in the 93 %–99 % band across every holding window. Heating Oil is the sole exception, with a hit rate that climbs from 73 % at τ = 5 to 85 % at τ = 60 but never reaches the level of the other contracts at the same `P`. This combination, where Heating Oil is the only contract that simultaneously delivers the largest price improvement and the lowest fill probability at the same threshold, suggests that the empirical CDF for this instrument has a substantively different shape than for the others, possibly because the ratio of typical range to tick size differs from the cross-section.

## 5. Repository Structure

```
.
├── main.py                       # Multi-process entry point linking execution and backtest
├── Signal_gen.py                 # Builds per-instrument signal files from AI-agent inventories
├── execution.py                  # Computes optimal limit price for each signal
├── backtest.py                   # Replays each order against the realized OHLC path
├── plot.py                       # Generates the per-configuration slippage bar charts
├── ui.py                         # Streamlit front-end for interactive optimal-price queries
├── get_ticksize.py               # Auxiliary script for inferring tick sizes from OHLC data
├── example_usage.py              # Standalone training and query example for the ePDF model
├── requirements.txt              # Minimal Python dependencies
├── epdf/                         # ePDF model package used by execution.py
│   ├── calculator.py             # ePDFCalculator class, train and query interface
│   ├── data_processor.py         # Preprocessing, resampling, range computation
│   ├── state_classifier.py       # EWMA features and quantile-based state binning
│   ├── probability_estimator.py  # Conditional PDF and CDF estimation
│   └── instrument_config.py      # Tick sizes and futures contract symbol parsing
├── Data/                         # Raw 1-minute OHLCV and AI-agent signal files (per instrument)
├── Signal/                       # Generated signal files consumed by execution.py
├── Result_9new/Result_{τ}min/    # Backtested outputs for P = 0.9 (results reported in Section 4)
├── Result_{τ}min/                # Latest main.py run, overwritten by every new invocation
└── TermProject2_OrderExecution.pdf  # Project specification
```

Each backtested CSV in the `Result_*/Result_{τ}min/` folders contains, in addition to the original signal columns, the recommended limit price `opti_price`, the realized fill price `filled_price`, the benchmark and realized notional amounts, and the signed slippage in price units.

## 6. How to Reproduce

The runtime depends on `numpy`, `pandas`, `tqdm`, and `matplotlib`. They can be installed via

```bash
pip install -r requirements.txt
pip install tqdm matplotlib
```

Reproducing the results consists of three steps. First, generate the per-instrument signal files from the AI-agent inventory series by uncommenting the `signal_gen(...)` call in `main.py` or by calling `Signal_gen.signal_gen("Data/", "Signal/")` in a Python session. Second, run `python main.py`. The script iterates over the holding-window list defined at the top of `main.py` and writes the backtested CSVs into folders named `Result_{τ}min/`. The minimum fill probability is controlled by the `risk_percentage` parameter inside the same script; sweeping across the three values reported in this README requires running the script three times with `risk_percentage` set to 0.3, 0.6, and 0.9. Third, run `python plot.py` to regenerate the per-configuration slippage bar charts.

## 7. Interactive UI

A Streamlit front-end is provided in `ui.py` as a single-query alternative to the batched execution pipeline in `main.py`. It exposes the same ePDF training and placement logic exercised by the backtest but operates on one trading request at a time, so that the dependence of the recommended limit price on the holding window, the probability threshold, and the conditioning state can be inspected interactively.

### 7.1 Installation and launch

The UI requires Streamlit in addition to the dependencies listed in Section 6. After installing it via

```bash
pip install streamlit
```

the application is launched from the repository root with

```bash
streamlit run ui.py
```

The repository root is required because `ui.py` references the `Data/` folder by relative path. Streamlit opens the application in the default browser at `http://localhost:8501`. The process can be terminated with `Ctrl-C` in the terminal that owns it.

### 7.2 Inputs

The sidebar collects the parameters of a single trading request. Their roles are summarized in the table below.

| Field             | Type        | Meaning                                                                 |
| :---------------- | :---------- | :---------------------------------------------------------------------- |
| Future contract   | dropdown    | Instrument code (one of VG, BP, RX, GC, HO, JY, NQ)                     |
| Trade side        | dropdown    | Direction of the order (Buy or Sell)                                    |
| Trade date        | text        | Trade date in `YYYY/MM/DD` format                                       |
| Trade time        | text        | Trade time at minute granularity in `HH:MM` format                      |
| tau               | integer     | Holding window in minutes used for resampling and ePDF construction     |
| risk_percentage   | float       | Minimum fill probability `P`, strictly between 0 and 1                  |

An expandable `Advanced Parameters` panel exposes the ePDF hyperparameters: the number of volume, volatility, and price-change states `M`, `N`, `K`; the EWMA half-life `ewma_halflife`; the estimation method (`smoothed` for Laplace smoothing or `raw` for unsmoothed empirical frequencies); and the Laplace parameter `smoothing_alpha`. These default to the values used throughout Section 4 (`M = 3`, `N = 3`, `K = 2`, `ewma_halflife = 10`, smoothed estimation with `alpha = 0.5`).

### 7.3 Outputs

Pressing the `Generate Optimal Price` button executes the placement logic in six steps that are rendered inline. The first step locates the instrument folder under `Data/`. The second step iterates over the available contract files for that instrument and selects, among those whose data covers the requested timestamp, the contract whose most recent volume is largest, treating it as the active main contract. The third step trains an `ePDFCalculator` on data up to and including the trade-timestamp minute, relying on the strict-lag EWMA convention to prevent the bar at the trade timestamp from leaking into its own state. The query path mirrors the batched pipeline in `execution.py`: the OHLC file is reprocessed through `process_pipeline` and `compute_all_ewma_features` so that features are computed on τ-minute bars, and the last τ-minute bar whose timestamp is strictly less than the trade timestamp supplies the state. The upper-tail CDF is then queried for successive tick offsets `ℓ = 0, 1, …`, stopping at the first level at which `F(ℓ | state) < P` (and in any case at `ℓ = 9`); each queried level is reported with its CDF value and a flag indicating whether the threshold is met. The fourth step extracts the largest passing `ℓ` and computes the recommended limit price as `close_prev ± ℓ · tick`, with the sign chosen by trade direction. The fifth step performs a single-trade backtest of that recommendation: it reads the raw 1-minute series of the main contract, isolates the placement window `[t, t + τ)`, and reports the order as filled at the recommended price whenever some bar in the window satisfies `low ≤ p★ ≤ high`. If no bar reaches the recommended price the order is closed at the last close of the placement window, or at the open of the first bar after the window when the placement window itself contains no bars. The step displays the placement-window OHLC table together with the fill status, the filled price, and a textual reason. The sixth step assembles a single-row summary containing the instrument, side, trade timestamp, main contract, holding window, probability threshold, direction, the conditioning state, the selected level and its CDF value, the previous τ-bar timestamp and close, the tick size, the recommended optimal price, the placement-window endpoints, the fill status, the filled price, and the fill reason; the row can be downloaded as a CSV via the button beneath the table.

The tick sizes used by the UI are defined in `tick_dict` at the top of `ui.py` and do not coincide with the values used by the batched pipeline for every instrument. Users querying contracts other than `NQ`, `GC`, and `RX` should consult `ui.py` directly for the active values; the recommended limit price returned by the UI will be computed on the price grid implied by that table.

### 7.4 A sample query

The following inputs produce a complete, reproducible run on Gold data shipped with the repository: instrument `GC`, side `Buy`, date `2024/01/15`, time `10:30`, `tau = 15`, `risk_percentage = 0.60`, advanced parameters at their defaults. The UI selects `GCG24.csv` as the main contract. The previous τ-minute bar, indexed at `2024-01-15 10:15`, has close 2055.80, and the tick size is 0.10. The conditioning state is `(2, 1, 1)`. The CDF table reported in Step 3 is monotonically decreasing in `ℓ` and crosses each of the three reference thresholds in turn, as summarized below.

| ℓ | F(ℓ \| state) | ≥ 0.30 ? | ≥ 0.60 ? | ≥ 0.90 ? | ≥ 0.95 ? |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 1.000 | ✓ | ✓ | ✓ | ✓ |
| 1 | 0.924 | ✓ | ✓ | ✓ |   |
| 2 | 0.837 | ✓ | ✓ |   |   |
| 3 | 0.769 | ✓ | ✓ |   |   |
| 4 | 0.712 | ✓ | ✓ |   |   |
| 5 | 0.630 | ✓ | ✓ |   |   |
| 6 | 0.570 | ✓ |   |   |   |
| 7 | 0.525 | ✓ |   |   |   |
| 8 | 0.493 | ✓ |   |   |   |
| 9 | 0.445 | ✓ |   |   |   |

At the default threshold `P = 0.60`, the largest passing level is `ℓ★ = 5` and the recommended limit price is `2055.80 − 5 × 0.10 = 2055.30`. Step 5 then evaluates this recommendation against the realized 1-minute path on `[10:30, 10:45)`: the placement window oscillates between a low of 2055.50 and a high of 2056.70, so the limit at 2055.30 is never touched and the order is closed at the window's last close of 2055.60.

Holding every other field fixed and varying only `risk_percentage` traces the trade-off between aggressiveness and fill probability. At `P = 0.30` all ten queried levels pass and `ℓ★ = 9`, giving a recommended price of `2054.90` that the placement window again fails to reach, with fallback fill at 2055.60. At `P = 0.90` the search stops at `ℓ★ = 1`, the recommended price rises to `2055.70`, and the placement window does reach it, producing an actual fill at 2055.70. At `P = 0.95` only `ℓ★ = 0` passes and the order is placed at the previous τ-bar close of `2055.80`, where it is also filled. The example therefore illustrates both ends of the trade-off in a single chart of inputs: the two aggressive thresholds yield deeper price improvement but are not reached inside the holding window, whereas the two conservative thresholds yield shallower offsets that the market does cross.

## 8. References

The project specification is included in the repository as `TermProject2_OrderExecution.pdf`. Tick sizes used by the execution layer are inferred empirically from the OHLC quoting convention observed in the dataset (see `get_ticksize.py`) and are hard-coded into the `tick_dict` defined in `main.py`. They reflect the price grid of the specific data files used in this project and may differ from the official exchange tick sizes documented in the `epdf/instrument_config.py` comments. Instrument symbols follow the standard futures convention of two-letter root with month-code and year suffix.
