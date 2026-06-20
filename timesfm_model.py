"""TimesFM zero-shot foundation-model forecast (no per-asset training).

TimesFM (https://github.com/google-research/timesfm) is Google's pretrained decoder-only
foundation model for time-series forecasting. Unlike the neuralforecast ensemble, it is *not*
trained on each asset - we load the pretrained checkpoint once, feed it the asset's recent
closing-price history as context, and read off a 30-day point forecast. This mirrors how
polymarket.py contributes an extra (non-neural) column to the ensemble.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

POINT_COL = "TimesFM"

# 200M PyTorch checkpoint - small/fast enough for the CPU-only daily CI runner.
_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
_MAX_CONTEXT = 1024  # number of trailing daily points fed to the model as context

_model = None  # module-level cache: load the checkpoint once, reuse across all assets in a run


def _load(horizon):
    """Lazily load and compile the TimesFM model (cached for the whole run)."""
    global _model
    if _model is None:
        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(_CHECKPOINT)
        model.compile(
            timesfm.ForecastConfig(
                max_context=_MAX_CONTEXT,
                max_horizon=max(horizon, 64),
                normalize_inputs=True,
            )
        )
        _model = model
    return _model


def forecast_path(hist_df, horizon, sigma=2):
    """30-day forecast DataFrame [ds, TimesFM] continuing the asset's price history.

    Feeds the trailing `_MAX_CONTEXT` closing prices to TimesFM and emits a `horizon`-step
    point forecast over the next `horizon` calendar days, smoothed to match the neural ensemble.
    """
    model = _load(horizon)

    last = pd.Timestamp(hist_df["ds"].iloc[-1])
    context = hist_df["y"].to_numpy(dtype=np.float32)[-_MAX_CONTEXT:]

    point_forecast, _ = model.forecast(horizon=horizon, inputs=[context])
    path = np.asarray(point_forecast)[0][:horizon].astype(float)
    if sigma:
        path = gaussian_filter1d(path, sigma=sigma)

    future = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"ds": future, POINT_COL: path})


if __name__ == "__main__":  # quick smoke test against live data
    import yfinance as yf

    df = yf.download("BTC-USD", start="2018-01-01")["Close"].astype(int).reset_index()
    df.columns = ["ds", "y"]
    print(forecast_path(df[["ds", "y"]], 30).head())
