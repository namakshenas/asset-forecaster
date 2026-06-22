# Daily (one-month-ahead) Price Forecasting

Daily automated Bitcoin, ETH, GOLD, SILVER, and BRENT price forecasts using deep neural networks plus a non-neural Polymarket-Implied forecast and Google's TimesFM foundation model. Updates daily at midnight UTC automatically.

Each chart shows **seven forecasters**: the five neural models (TSMixer, NBEATS, NHITS, MLP, TiDE), a **Polymarket-Implied** forecast (dashed line), and **Google TimesFM** (dotted line) — a pretrained zero-shot foundation model — both explained below.

## Which model should I trust?

Short answer: none of them unconditionally, and none of them is dramatically better than the others.

The chart below shows the results of a 52-week directional backtest (June 2025 to June 2026) across all five assets. Each week, every model made a single call: will this asset close higher or lower next Friday? 260 predictions per model in total.

![Backtest results](https://github.com/namakshenas/neural-asset-forecaster/blob/main/backtests/backtest_visual.png)

The five neural models all clustered between 50.4% and 54.2% overall accuracy — barely above a coin flip, and within a few correct calls of each other. No single model dominated across all assets. NHITS led in Gold (63.5%), MLP led in Bitcoin (63.5%), TSMixer led in Ethereum and Brent. Pick one based on the asset you care about most, but do not expect the ranking to hold permanently.

Google's TimesFM, a zero-shot foundation model pretrained on massive generic time series data, finished last at 45.0% overall and is the only model that fell below the random baseline. Its zero-shot nature is a liability here: financial markets are noisy enough that asset-specific training, even on a small window of history, outperforms broad generalization.

The practical takeaway is to watch the **consensus** across all five neural models rather than any single line. When several models agree on a direction, the signal is more meaningful than when they diverge.

## How the neural forecasters work

The five neural models come from the [neuralforecast](https://github.com/Nixtla/neuralforecast) library. They are **trained from scratch on each run**: every model learns from roughly the last two years of that asset's own daily closing prices (a 504-day window, 252 for MLP) and predicts the next 30 days. Because each architecture sees the same history but models it differently, comparing their forecasts gives a sense of how much the outlook depends on the modelling assumptions rather than the data alone.

- **TSMixer** — an all-MLP architecture that alternately "mixes" information along the time axis and across features. It captures temporal patterns without recurrence or attention, making it fast and surprisingly strong on structured series.
- **NBEATS** — a deep stack of fully-connected blocks linked by backward/forward residuals. Here it's configured with interpretable **trend** and **seasonality** basis functions, so it decomposes the series into a smooth trend plus repeating cycles.
- **NHITS** — an evolution of NBEATS that adds multi-rate pooling and hierarchical interpolation. By processing the series at several resolutions it handles long horizons efficiently and resists overfitting to short-term noise.
- **MLP** — a plain multilayer perceptron that maps a window of past prices straight to the forecast. It's the simplest baseline here and a useful reference point for whether the fancier models are actually adding value.
- **TiDE** — a Time-series Dense Encoder: an MLP-based encoder–decoder that compresses the input window into a dense representation before decoding the forecast. It pairs the speed of MLPs with the structure of an encoder–decoder.

## How the Polymarket forecast works

Polymarket runs live betting markets on asset prices — e.g. *"Will Bitcoin be above $70,000 on June 20?"* or *"What will Gold hit this month?"*. The price of each bet **is a probability** (a bet trading at 0.30 means the market thinks there's a 30% chance). We turn those probabilities into a 30-day price forecast:

1. **Find the markets.** For each asset we automatically pull its active price markets from the Polymarket API and keep the ones about that asset's price.
2. **Rebuild the implied distribution.** Each asset has many bets at different price levels, all resolving on the same date. Read together, they describe the market's probability distribution for the price on that date. We average that distribution to get the market's **expected price** for each resolution date.
   - "above $X" and price-bucket markets give the distribution directly.
   - "what will it hit" (high/low touch) markets are converted to an end-of-period distribution using the **reflection principle** (a standard random-walk identity: the chance of *touching* a level is about twice the chance of *ending* beyond it).
3. **Connect the dots.** This yields a few future points (e.g. expected price on June 19, on June 30, …). We start from today's actual price and draw a smooth line through those implied expected prices across the next 30 days. If the markets only reach part of the horizon, the line is extended flat.

Everything is expressed as a percentage move from today's spot price, so it lines up with the real price even when the market settles on a different feed (e.g. the Brent forecast is driven by WTI oil markets).

## How the TimesFM forecast works

[TimesFM](https://github.com/google-research/timesfm) is Google's pretrained **foundation model** for time-series forecasting. Unlike the five neural models — which train from scratch on each asset's own history every run — TimesFM is **zero-shot**: a single large model, pretrained on a huge corpus of time series, that forecasts a new series without any asset-specific training. We feed it the asset's recent daily closing prices as context and read off its 30-day point forecast (dotted line), smoothed to match the other models.

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/btc.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/eth.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/gold.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/brent.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/silver.png)
