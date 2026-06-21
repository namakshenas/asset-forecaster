"""Weekly directional walk-forward backtest (run manually — NOT part of CI).

Evaluates how well the production forecasters call next week's *direction* (long/short)
for each asset. The six models scored are the five neuralforecast models
(TSMixer, NBEATS, NHITS, MLP, TiDE) plus Google's TimesFM zero-shot foundation model.
The Polymarket column used in production is intentionally excluded here.

Methodology (train once, roll weekly):
  * For each asset, train the neural ensemble ONCE on 2018-01-01 -> one year ago.
  * Then walk forward one week at a time across the last year. At each weekly cutoff we
    feed the trained models the price history up to that cutoff (no retraining, via
    NeuralForecast.predict(df=...)) and TimesFM the same context, read off the ~1-week
    forecast, and label it "long" if it sits above the cutoff close else "short".
  * The realized label is "long" if the actual close ~1 week later is above the cutoff
    close else "short". A prediction is correct when the two labels match.
  * Set RETRAIN_WEEKLY=True to instead refit the ensemble on the expanding window at every
    cutoff (a truer walk-forward, but ~52x the training time).

Outputs (written to backtests/ with an execution timestamp down to the second):
  * backtest_<ts>.md  — per-asset table of the 6 models sorted by accuracy, with the
    long-call / short-call success split, plus a pooled leaderboard.
  * backtest_<ts>.csv — every per-week predicted-vs-actual row, for auditing.

Run it yourself:  python backtest.py
"""

import os
import logging
import importlib
from datetime import datetime

import pandas as pd
import yfinance as yf
from neuralforecast import NeuralForecast

import timesfm_model as tfm  # zero-shot foundation model (Polymarket deliberately not imported)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

ASSETS = ["GOLD", "SILVER", "BTC", "ETH", "BRENT"]
HORIZON = 7          # forecast one week ahead
START_DATE = "2018-01-01"
TEST_DAYS = 365      # backtest the last year
STEP_DAYS = 7        # one prediction per week
OUT_DIR = "backtests"

# False: train each asset's ensemble once (2018 -> one year ago), then roll forecasts off the
#        frozen weights — fast (~5 fits total).
# True:  refit the whole ensemble on the expanding window at EVERY weekly cutoff — a truer
#        walk-forward, but ~52x the training cost (5 assets x ~52 weeks of full fits).
RETRAIN_WEEKLY = True

NEURAL_MODELS = ["TSMixer", "NBEATS", "NHITS", "MLP", "TiDE"]
ALL_MODELS = NEURAL_MODELS + [tfm.POINT_COL]  # 5 neural + "TimesFM"


def load_close(symbol):
    """Full-precision daily close as a Series indexed by Timestamp (same source as update_pred.py)."""
    close = yf.download(symbol, start=START_DATE, progress=False)["Close"]
    if isinstance(close, pd.DataFrame):           # single-ticker download can come back 1-col
        close = close.iloc[:, 0]
    close = close.astype(float)
    close.index = pd.to_datetime(close.index)
    return close.dropna()


def to_model_df(close):
    """Integer-truncated long frame [unique_id, ds, y] — exactly what production feeds the models."""
    df = close.astype(int).rename("y").reset_index()
    df.columns = ["ds", "y"]
    df.insert(0, "unique_id", "1.0")
    return df


def direction(value, base):
    """'long' if value sits above the cutoff close, else 'short'."""
    return "long" if value > base else "short"


def weekly_cutoffs():
    """Weekly cutoff dates from one year ago up to the last week with a realized outcome."""
    today = pd.Timestamp.today().normalize()
    cutoffs, c = [], today - pd.Timedelta(days=TEST_DAYS)
    while c <= today - pd.Timedelta(days=STEP_DAYS):
        cutoffs.append(c)
        c += pd.Timedelta(days=STEP_DAYS)
    return cutoffs


def fit_ensemble(module, train_df):
    """Fit a fresh 5-model ensemble on train_df (progress bars + per-fit file logs off)."""
    models = module.get_models(HORIZON)
    for m in models:
        m.trainer_kwargs["enable_progress_bar"] = False  # keep the console readable
        m.trainer_kwargs["logger"] = False               # don't spew a lightning_logs dir per fit
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=train_df, val_size=HORIZON)
    return nf


def backtest_asset(asset, cutoffs):
    """Score every weekly direction call for one asset. Returns a list of row dicts.

    RETRAIN_WEEKLY=False -> train once on 2018 -> one year ago, then roll forecasts off the
    frozen weights. RETRAIN_WEEKLY=True -> refit the whole ensemble on the expanding window at
    every cutoff (truer walk-forward, ~52x slower). TimesFM is zero-shot in both modes.
    """
    module = importlib.import_module(f"assets.{asset}")
    close = load_close(module.get_symbol())
    df = to_model_df(close)

    nf = None
    if not RETRAIN_WEEKLY:
        train_df = df[df["ds"] <= cutoffs[0]]       # one year ago
        print(f"[{asset}] training once on {len(train_df)} rows up to {cutoffs[0].date()}...")
        nf = fit_ensemble(module, train_df)

    rows = []
    mode = "retrain each week" if RETRAIN_WEEKLY else "roll, no retrain"
    print(f"[{asset}] {len(cutoffs)} weekly predictions [{mode}]...")
    for i, cutoff in enumerate(cutoffs, 1):
        prior = close.index[close.index <= cutoff]
        future = close[(close.index > cutoff) & (close.index <= cutoff + pd.Timedelta(days=STEP_DAYS))]
        if len(prior) == 0 or future.empty:
            continue                                # need an anchor and a realized outcome
        p0 = float(close.loc[prior[-1]])            # real close at the cutoff (the anchor)
        realized = float(future.iloc[-1])           # last real close within the next week
        actual = direction(realized, p0)

        context = df[df["ds"] <= cutoff]
        if RETRAIN_WEEKLY:
            print(f"[{asset}]   week {i}/{len(cutoffs)} {cutoff.date()} — refitting on {len(context)} rows...")
            nf = fit_ensemble(module, context)      # expanding-window retrain through the cutoff
            yhat = nf.predict()                     # forecast the next week after the cutoff
        else:
            yhat = nf.predict(df=context)           # reuse frozen weights on this week's context

        preds = {}
        for model in NEURAL_MODELS:
            if model in yhat.columns:
                preds[model] = float(yhat[model].iloc[-1])
        try:
            ts = tfm.forecast_path(context[["ds", "y"]], HORIZON)
            preds[tfm.POINT_COL] = float(ts[tfm.POINT_COL].iloc[-1])
        except Exception as exc:                    # keep the run alive if TimesFM hiccups
            print(f"[{asset}] TimesFM failed at {cutoff.date()}: {exc}")

        for model, p_pred in preds.items():
            pred_dir = direction(p_pred, p0)
            rows.append({
                "asset": asset, "model": model, "cutoff_date": cutoff.date().isoformat(),
                "P0": round(p0, 4), "P_pred": round(p_pred, 4), "predicted_dir": pred_dir,
                "realized_close": round(realized, 4), "actual_dir": actual,
                "correct": int(pred_dir == actual),
            })
    return rows


def summarize(rows_df):
    """Per (asset, model): totals, accuracy, and the long-call / short-call success split."""
    records = []
    for (asset, model), g in rows_df.groupby(["asset", "model"]):
        longs = g[g["predicted_dir"] == "long"]
        shorts = g[g["predicted_dir"] == "short"]
        records.append({
            "asset": asset, "model": model,
            "total": len(g), "correct": int(g["correct"].sum()),
            "accuracy": 100.0 * g["correct"].sum() / len(g),
            "long_made": len(longs), "long_correct": int(longs["correct"].sum()),
            "short_made": len(shorts), "short_correct": int(shorts["correct"].sum()),
        })
    return pd.DataFrame(records)


def _pct(correct, made):
    return f"{correct}/{made}" + (f" ({100.0 * correct / made:.0f}%)" if made else " (n/a)")


def write_report(stats, md_path, timestamp):
    lines = [
        f"# Weekly directional backtest — {timestamp}",
        "",
        "- Models: 5 neural (TSMixer, NBEATS, NHITS, MLP, TiDE) + TimesFM (Polymarket excluded).",
        f"- Train start: {START_DATE}. Test window: the last {TEST_DAYS} days, one prediction per week.",
        f"- Training mode: {'retrain each week (expanding window)' if RETRAIN_WEEKLY else 'train once, then roll (no retraining)'}.",
        "- 'long' = model expects next-week close above the cutoff close; 'short' = below.",
        "- Long/short columns are call precision: of the calls of that direction, how many were right.",
        "",
        "## Overall leaderboard (pooled across all assets)",
        "",
        "| Rank | Model | Accuracy | Correct/Total |",
        "|------|-------|----------|---------------|",
    ]
    pooled = []
    for model, g in stats.groupby("model"):
        c, t = int(g["correct"].sum()), int(g["total"].sum())
        pooled.append((model, 100.0 * c / t if t else 0.0, c, t))
    for rank, (model, acc, c, t) in enumerate(sorted(pooled, key=lambda x: x[1], reverse=True), 1):
        lines.append(f"| {rank} | {model} | {acc:.1f}% | {c}/{t} |")
    lines.append("")

    for asset in ASSETS:
        a = stats[stats["asset"] == asset].sort_values("accuracy", ascending=False)
        if a.empty:
            continue
        lines += [
            f"## {asset}",
            "",
            "| Model | Accuracy | Correct/Total | Long calls | Short calls |",
            "|-------|----------|---------------|------------|-------------|",
        ]
        for _, r in a.iterrows():
            lines.append(
                f"| {r['model']} | {r['accuracy']:.1f}% | {int(r['correct'])}/{int(r['total'])} "
                f"| {_pct(int(r['long_correct']), int(r['long_made']))} "
                f"| {_pct(int(r['short_correct']), int(r['short_made']))} |"
            )
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main():
    cutoffs = weekly_cutoffs()
    print(f"Backtesting {len(cutoffs)} weeks from {cutoffs[0].date()} to {cutoffs[-1].date()}.\n")

    all_rows = []
    for asset in ASSETS:
        all_rows.extend(backtest_asset(asset, cutoffs))

    rows_df = pd.DataFrame(all_rows)
    if rows_df.empty:
        print("No scored weeks produced — nothing to write.")
        return

    stats = summarize(rows_df)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    md_path = os.path.join(OUT_DIR, f"backtest_{ts}.md")
    csv_path = os.path.join(OUT_DIR, f"backtest_{ts}.csv")
    rows_df.to_csv(csv_path, index=False)
    write_report(stats, md_path, ts)

    print(f"\nSaved report   -> {md_path}")
    print(f"Saved raw rows -> {csv_path}")


if __name__ == "__main__":
    main()
