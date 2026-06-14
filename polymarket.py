"""Market-Implied Forecast from Polymarket prediction markets (no neural nets).

Polymarket's price-bracket / threshold contracts are a live probability distribution over
an asset's future price. For each asset we reconstruct the market-implied *expected* price
at every resolution date and connect those points - anchored on today's spot - into a
smooth 30-day path. See the README for a plain-English walkthrough.
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

GAMMA = "https://gamma-api.polymarket.com"
POINT_COL = "PolymarketImplied"
_HEADERS = {"User-Agent": "neural-asset-forecaster/1.0"}
_NUM = re.compile(r"\d[\d,]*\.?\d*")  # positive prices; no '-' so "1,200-1,300" splits cleanly
# titles that match an asset keyword but are not a price-distribution market
_BLOCK = ("volatility", "index", "outperform", " vs", "vs.", "etf", "reserve",
          "treasury", "dominance", "market cap", "flippening")


def _get_json(url):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def discover_events(query, match):
    """Active Polymarket events whose title matches the asset (and isn't off-topic)."""
    url = f"{GAMMA}/public-search?" + urllib.parse.urlencode(
        {"q": query, "limit_per_type": 25, "events_status": "active"})
    events = _get_json(url).get("events") or []
    return [e for e in events
            if (t := (e.get("title") or "").lower())
            and any(k in t for k in match) and not any(b in t for b in _BLOCK)]


def _nums(s):
    return [float(x) for x in _NUM.findall(s.replace(",", ""))]


def _yes(market):
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        prices = json.loads(prices)
    return float(prices[0]) if prices else None


def _mean_from_survival(points):
    """Expected value of the distribution described by survival points S(K)=P(price>=K)."""
    pts = sorted({k: s for k, s in points}.items())
    ks = np.array([k for k, _ in pts], float)
    surv = np.minimum.accumulate(np.clip([s for _, s in pts], 0, 1))  # non-increasing in K
    cdf = 1 - surv
    xs = np.concatenate([[ks[0]], (ks[:-1] + ks[1:]) / 2, [ks[-1]]])   # bin midpoints + tails
    w = np.clip(np.concatenate([[cdf[0]], np.diff(cdf), [surv[-1]]]), 0, None)  # mass per bin
    return float((xs * w).sum() / w.sum()) if w.sum() > 0 else None


def _event_mean(event, spot):
    """(resolution_date, implied expected price) for one event, or None if unusable.

    All three Polymarket market shapes are reduced to survival points S(K)=P(price>=K):
      * brackets   "1,200-1,300" / "<1,200" / ">2,100"   -> terminal distribution
      * thresholds "Bitcoin above 52,000 on <date>"      -> P(price>=K) directly
      * touch      "↑ $5,200" / "↓ $20"                  -> reflection: P(max>=K) ~= 2*P(end>=K)
    """
    title = (event.get("title") or "").lower()
    touch, buckets, thresh = [], [], []
    for m in event.get("markets") or []:
        if m.get("closed"):
            continue
        git = (m.get("groupItemTitle") or "").strip()
        nums = _nums(git)
        p = _yes(m)
        if p is None or not nums:
            continue
        k = nums[0]
        if "↑" in git:
            touch.append((k, min(0.5, p / 2)))
        elif "↓" in git:
            touch.append((k, max(0.5, 1 - p / 2)))
        elif len(nums) == 2:
            buckets.append(((nums[0], nums[1]), p))
        elif git.startswith("<"):
            buckets.append(((None, k), p))
        elif git.startswith(">"):
            buckets.append(((k, None), p))
        elif "above" in title:
            thresh.append((k, p))

    if touch:
        points = touch + [(spot, 0.5)]            # anchor the reflection at the spot
    elif buckets:
        total = sum(p for _, p in buckets) or 1.0  # normalize away the market overround
        points = [(e, 1 - sum(p for (lo, hi), p in buckets if hi and hi <= e) / total)
                  for e in sorted({x for (lo, hi), _ in buckets for x in (lo, hi) if x})]
    else:
        points = thresh

    date = _parse_date(event.get("endDate"))
    if date is None or len(points) < 3:
        return None
    mean = _mean_from_survival(points)
    return (date, mean) if mean and mean > 0 else None


def _parse_date(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def forecast_path(spot_df, events, horizon, sigma=2):
    """30-day forecast DataFrame [ds, PolymarketImplied], anchored on the latest spot.

    Connects today's spot to the market-implied expected price at each resolution date and
    interpolates across the horizon (stays flat if no usable markets are found).
    """
    last = pd.Timestamp(spot_df["ds"].iloc[-1])
    spot = float(spot_df["y"].iloc[-1])
    future = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")

    by_day = {0.0: [spot]}  # day offset -> implied means (day 0 = today's spot)
    for e in events:
        got = _event_mean(e, spot)
        if not got:
            continue
        date, mean = got
        d = float((date - last.tz_localize("UTC")).days)
        if 0 < d <= horizon + 21 and 0.5 * spot <= mean <= 2 * spot:  # near-term & plausible
            by_day.setdefault(d, []).append(mean)

    days = sorted(by_day)
    means = [float(np.mean(by_day[d])) for d in days]
    path = np.interp(np.arange(1, horizon + 1), days, means)
    if sigma:
        path = gaussian_filter1d(path, sigma=sigma)
    return pd.DataFrame({"ds": future, POINT_COL: path})


if __name__ == "__main__":  # quick smoke test against live data
    import yfinance as yf

    for name, symbol, query, match in [
        ("BTC", "BTC-USD", "bitcoin", ["bitcoin", "btc"]),
        ("ETH", "ETH-USD", "ethereum price", ["ethereum", "eth"]),
        ("GOLD", "GC=F", "gold price", ["gold", "xauusd"]),
        ("SILVER", "SI=F", "silver price", ["silver", "xagusd"]),
        ("BRENT", "BZ=F", "oil price", ["wti", "crude", "oil"]),
    ]:
        df = yf.download(symbol, period="6mo", progress=False)["Close"].reset_index()
        df.columns = ["ds", "y"]
        fc = forecast_path(df, discover_events(query, match), 30)
        print(f"{name:6} spot={df['y'].iloc[-1]:>10,.1f}  "
              f"day1={fc[POINT_COL].iloc[0]:>10,.1f}  day30={fc[POINT_COL].iloc[-1]:>10,.1f}")
