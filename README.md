# Daily (one-month-ahead) Price Forecasting

Daily automated Bitcoin, ETH, GOLD, SILVER, and BRENT price forecasts using deep neural networks plus a non-neural Polymarket-Implied forecast. Updates daily at midnight UTC automatically.

Each chart shows **six forecasters**: the five neural models (TSMixer, NBEATS, NHITS, MLP, TiDE) plus a non-neural **Polymarket-Implied** forecast (dashed line), explained below.

## How the Polymarket forecast works

Polymarket runs live betting markets on asset prices — e.g. *"Will Bitcoin be above $70,000 on June 20?"* or *"What will Gold hit this month?"*. The price of each bet **is a probability** (a bet trading at 0.30 means the market thinks there's a 30% chance). We turn those probabilities into a 30-day price forecast:

1. **Find the markets.** For each asset we automatically pull its active price markets from the Polymarket API and keep the ones about that asset's price.
2. **Rebuild the implied distribution.** Each asset has many bets at different price levels, all resolving on the same date. Read together, they describe the market's probability distribution for the price on that date. We average that distribution to get the market's **expected price** for each resolution date.
   - "above $X" and price-bucket markets give the distribution directly.
   - "what will it hit" (high/low touch) markets are converted to an end-of-period distribution using the **reflection principle** (a standard random-walk identity: the chance of *touching* a level is about twice the chance of *ending* beyond it).
3. **Connect the dots.** This yields a few future points (e.g. expected price on June 19, on June 30, …). We start from today's actual price and draw a smooth line through those implied expected prices across the next 30 days. If the markets only reach part of the horizon, the line is extended flat.

Everything is expressed as a percentage move from today's spot price, so it lines up with the real price even when the market settles on a different feed (e.g. the Brent forecast is driven by WTI oil markets).

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/btc.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/eth.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/gold.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/brent.png)

![image](https://github.com/namakshenas/neural-asset-forecaster/blob/main/predictions/silver.png)
